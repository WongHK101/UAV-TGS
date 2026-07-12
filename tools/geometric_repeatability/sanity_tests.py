from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from evaluator import (
    DEFAULT_THRESHOLDS_RATIO,
    build_shared_roi_from_points,
    brute_force_nn_distances,
    compute_bidirectional_prf,
    deterministic_voxel_downsample,
    evaluate_scene_bundle,
    kd_tree_nn_distances,
)


def _assert_close(name: str, got: np.ndarray, want: np.ndarray, atol: float = 1e-12) -> None:
    if not np.allclose(got, want, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch:\nGOT={got}\nWANT={want}")


def _test_identical_clouds() -> None:
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rows = compute_bidirectional_prf(pts, pts.copy(), thresholds=[1e-9, 0.1])
    for row in rows:
        if abs(row["precision"] - 1.0) > 1e-12 or abs(row["recall"] - 1.0) > 1e-12 or abs(row["fscore"] - 1.0) > 1e-12:
            raise AssertionError(f"Identical cloud test failed: {row}")


def _test_translated_clouds() -> None:
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pred = ref + np.array([0.2, 0.0, 0.0], dtype=np.float64)
    rows = compute_bidirectional_prf(pred, ref, thresholds=[0.1, 0.25])
    if rows[0]["fscore"] != 0.0:
        raise AssertionError(f"Translated cloud test expected F=0 below threshold, got {rows[0]}")
    if rows[1]["fscore"] <= 0.99:
        raise AssertionError(f"Translated cloud test expected near-perfect F above threshold, got {rows[1]}")


def _test_outlier_behavior() -> None:
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    pred = np.concatenate([ref.copy(), np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]], dtype=np.float64)], axis=0)
    row = compute_bidirectional_prf(pred, ref, thresholds=[0.01])[0]
    if not (row["precision"] < row["recall"] and row["recall"] > 0.99):
        raise AssertionError(f"Outlier behavior should reduce precision more than recall, got {row}")


def _test_voxel_determinism() -> None:
    pts = np.array(
        [
            [0.01, 0.01, 0.01],
            [0.02, 0.02, 0.02],
            [1.01, 1.01, 1.01],
            [1.02, 1.02, 1.02],
        ],
        dtype=np.float64,
    )
    out1 = deterministic_voxel_downsample(pts, voxel_size=0.5, origin=np.zeros(3, dtype=np.float64))
    out2 = deterministic_voxel_downsample(pts.copy(), voxel_size=0.5, origin=np.zeros(3, dtype=np.float64))
    _assert_close("voxel_determinism", out1, out2)
    if out1.shape[0] != 2:
        raise AssertionError(f"Expected 2 voxels after downsampling, got {out1.shape[0]}")


def _test_kdtree_matches_bruteforce() -> None:
    query = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.5, 0.0],
            [2.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    brute = brute_force_nn_distances(query, ref)
    kdtree = kd_tree_nn_distances(query, ref)
    _assert_close("kdtree_vs_bruteforce", kdtree, brute)


def _write_demo_view(path: Path, depth_value: float, opacity_value: float) -> None:
    depth = np.full((2, 2), depth_value, dtype=np.float64)
    opacity = np.full((2, 2), opacity_value, dtype=np.float64)
    np.savez_compressed(path, depth=depth, opacity=opacity)


def _test_end_to_end_bundle(depth_semantics: str, stored_depth_value: float) -> None:
    with tempfile.TemporaryDirectory(prefix="geom_repeatability_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        roi_points = np.array(
            [
                [-1.0, -1.0, 1.0],
                [1.0, 1.0, 3.0],
                [0.5, -0.5, 2.0],
                [-0.5, 0.5, 2.5],
            ],
            dtype=np.float64,
        )
        roi = build_shared_roi_from_points(roi_points, scene_name="ToyScene")
        roi_path = tmp_dir / "roi.json"
        import json
        roi_path.write_text(json.dumps(roi, indent=2), encoding="utf-8")

        odd_dir = tmp_dir / "odd"
        even_dir = tmp_dir / "even"
        odd_dir.mkdir()
        even_dir.mkdir()
        _write_demo_view(odd_dir / "00000.npz", depth_value=stored_depth_value, opacity_value=1.0)
        _write_demo_view(even_dir / "00000.npz", depth_value=stored_depth_value, opacity_value=1.0)
        manifest = {
            "protocol_name": "pose-controlled-cross-subset-geometric-repeatability-v1",
            "scene_name": "ToyScene",
            "roi_path": "roi.json",
            "depth_semantics": depth_semantics,
            "distance_domain": "after_roi_crop_and_after_voxel_downsampling",
            "validity_rule": {
                "mode": "opacity_threshold",
                "opacity_threshold": 0.5,
                "depth_min": 1e-6,
            },
            "views": [
                {
                    "view_id": "00000",
                    "width": 2,
                    "height": 2,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "camera_to_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "odd_file": "odd/00000.npz",
                    "even_file": "even/00000.npz",
                }
            ],
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out_dir = tmp_dir / "out"
        metrics = evaluate_scene_bundle(manifest_path=manifest_path, out_dir=out_dir)
        row = metrics["metrics"][1]
        if row["threshold_ratio"] if "threshold_ratio" in row else None:
            pass
        if row["fscore"] < 0.999999:
            raise AssertionError(f"End-to-end identical toy bundle should reach F~1, got {row}")
        expected_files = [
            out_dir / "manifest_snapshot.json",
            out_dir / "roi_snapshot.json",
            out_dir / "odd_points_after_roi.npz",
            out_dir / "odd_points_after_voxel.npz",
            out_dir / "even_points_after_roi.npz",
            out_dir / "even_points_after_voxel.npz",
            out_dir / "metrics.json",
            out_dir / "metrics.csv",
        ]
        missing = [str(p) for p in expected_files if not p.exists()]
        if missing:
            raise AssertionError(f"Evaluator did not save expected audit artifacts: {missing}")


def _test_end_to_end_bundle_absolute_threshold_override() -> None:
    with tempfile.TemporaryDirectory(prefix="geom_repeatability_abs_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        roi_points = np.array(
            [
                [-1.0, -1.0, 1.0],
                [1.0, 1.0, 3.0],
                [0.5, -0.5, 2.0],
                [-0.5, 0.5, 2.5],
            ],
            dtype=np.float64,
        )
        roi = build_shared_roi_from_points(roi_points, scene_name="ToySceneAbs")
        roi_path = tmp_dir / "roi.json"
        import json

        roi_path.write_text(json.dumps(roi, indent=2), encoding="utf-8")

        odd_dir = tmp_dir / "odd"
        even_dir = tmp_dir / "even"
        odd_dir.mkdir()
        even_dir.mkdir()
        _write_demo_view(odd_dir / "00000.npz", depth_value=2.0, opacity_value=1.0)
        _write_demo_view(even_dir / "00000.npz", depth_value=2.0, opacity_value=1.0)
        manifest = {
            "protocol_name": "pose-controlled-cross-subset-geometric-repeatability-v1",
            "scene_name": "ToySceneAbs",
            "roi_path": "roi.json",
            "depth_semantics": "metric_camera_z_from_renderer",
            "distance_domain": "after_roi_crop_and_after_voxel_downsampling",
            "validity_rule": {
                "mode": "opacity_threshold",
                "opacity_threshold": 0.5,
                "depth_min": 1e-6,
            },
            "views": [
                {
                    "view_id": "00000",
                    "width": 2,
                    "height": 2,
                    "fx": 1.0,
                    "fy": 1.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "camera_to_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "odd_file": "odd/00000.npz",
                    "even_file": "even/00000.npz",
                }
            ],
        }
        manifest_path = tmp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        out_dir = tmp_dir / "out_abs"
        metrics = evaluate_scene_bundle(
            manifest_path=manifest_path,
            out_dir=out_dir,
            threshold_abs_override=[0.25, 0.5],
            voxel_size_override=0.05,
        )
        if metrics["threshold_mode"] != "absolute_meter":
            raise AssertionError(f"Expected absolute threshold mode, got {metrics['threshold_mode']!r}")
        if metrics["voxel_mode"] != "absolute_meter":
            raise AssertionError(f"Expected absolute voxel mode, got {metrics['voxel_mode']!r}")
        if abs(metrics["voxel_size"] - 0.05) > 1e-12:
            raise AssertionError(f"Expected overridden voxel size 0.05, got {metrics['voxel_size']!r}")
        got_thresholds = [float(row["threshold_abs"]) for row in metrics["metrics"]]
        _assert_close("abs_threshold_override", np.asarray(got_thresholds), np.asarray([0.25, 0.5]))
        if not (out_dir / "effective_eval_config.json").exists():
            raise AssertionError("Expected effective_eval_config.json to be saved for absolute override runs")


def main() -> None:
    _test_identical_clouds()
    _test_translated_clouds()
    _test_outlier_behavior()
    _test_voxel_determinism()
    _test_kdtree_matches_bruteforce()
    _test_end_to_end_bundle(depth_semantics="metric_camera_z_from_renderer", stored_depth_value=2.0)
    _test_end_to_end_bundle(depth_semantics="inverse_camera_z_from_renderer", stored_depth_value=0.5)
    _test_end_to_end_bundle_absolute_threshold_override()
    print("GEOMETRIC_REPEATABILITY_SANITY_OK")


if __name__ == "__main__":
    main()
