from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_points3D_binary, read_points3D_text


PROTOCOL_NAME = "pose-controlled-cross-subset-geometric-repeatability-v1"
DEFAULT_THRESHOLDS_RATIO = (0.005, 0.01, 0.02)
DEFAULT_LOWER_QUANTILE = 0.01
DEFAULT_UPPER_QUANTILE = 0.99
DEFAULT_PADDING_RATIO = 0.02
DEFAULT_VOXEL_RATIO = 0.001
DEFAULT_OPACITY_THRESHOLD = 0.5
DEFAULT_DEPTH_MIN = 1e-6
SUPPORTED_DEPTH_SEMANTICS = (
    "metric_camera_z_from_renderer",
    "inverse_camera_z_from_renderer",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, default=_json_default)
        f.write("\n")


def _save_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _resolve_path(base_path: Path, rel_or_abs: str) -> Path:
    candidate = Path(rel_or_abs)
    if candidate.is_absolute():
        return candidate
    return (base_path.parent / candidate).resolve()


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        ply = PlyData.read(str(path))
        vertex = ply["vertex"]
        points = np.stack(
            [
                np.asarray(vertex["x"], dtype=np.float64),
                np.asarray(vertex["y"], dtype=np.float64),
                np.asarray(vertex["z"], dtype=np.float64),
            ],
            axis=1,
        )
        return points
    if suffix == ".npy":
        points = np.load(path)
        return np.asarray(points, dtype=np.float64)
    if suffix == ".npz":
        with np.load(path) as data:
            key = "points" if "points" in data.files else data.files[0]
            return np.asarray(data[key], dtype=np.float64)
    if suffix == ".bin":
        points3d = read_points3D_binary(str(path))
        return np.asarray([point.xyz for point in points3d.values()], dtype=np.float64)
    if suffix == ".txt":
        points3d = read_points3D_text(str(path))
        return np.asarray([point.xyz for point in points3d.values()], dtype=np.float64)
    raise ValueError(f"Unsupported point-cloud format: {path}")


def _filter_finite_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got shape {points.shape}")
    finite_mask = np.all(np.isfinite(points), axis=1)
    return points[finite_mask]


def build_shared_roi_from_points(
    points: np.ndarray,
    scene_name: str,
    lower_quantile: float = DEFAULT_LOWER_QUANTILE,
    upper_quantile: float = DEFAULT_UPPER_QUANTILE,
    padding_ratio: float = DEFAULT_PADDING_RATIO,
    voxel_ratio: float = DEFAULT_VOXEL_RATIO,
    threshold_ratios: Sequence[float] = DEFAULT_THRESHOLDS_RATIO,
) -> Dict[str, Any]:
    points = _filter_finite_points(points)
    if points.shape[0] == 0:
        raise ValueError("Cannot build ROI from an empty point cloud")
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")
    robust_min = np.quantile(points, lower_quantile, axis=0)
    robust_max = np.quantile(points, upper_quantile, axis=0)
    robust_extent = robust_max - robust_min
    robust_diagonal = float(np.linalg.norm(robust_extent))
    if robust_diagonal <= 0.0:
        raise ValueError("Robust ROI diagonal is non-positive")
    padding = float(padding_ratio) * robust_diagonal
    bbox_min = robust_min - padding
    bbox_max = robust_max + padding
    scene_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    voxel_size = float(max(float(voxel_ratio) * scene_diagonal, 1e-12))
    thresholds = [float(r) * scene_diagonal for r in threshold_ratios]
    return {
        "protocol_name": PROTOCOL_NAME,
        "scene_name": scene_name,
        "roi_rule": {
            "type": "training_sparse_quantile_aabb",
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
            "padding_ratio_of_robust_diagonal": float(padding_ratio),
        },
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "scene_diagonal": scene_diagonal,
        "voxel_ratio_of_scene_diagonal": float(voxel_ratio),
        "voxel_size": voxel_size,
        "threshold_ratios": [float(v) for v in threshold_ratios],
        "thresholds": thresholds,
        "distance_domain": "after_roi_crop_and_after_voxel_downsampling",
    }


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    required = ("protocol_name", "scene_name", "roi_path", "depth_semantics", "distance_domain", "validity_rule", "views")
    for key in required:
        if key not in manifest:
            raise KeyError(f"Manifest missing required key: {key}")
    if manifest["protocol_name"] != PROTOCOL_NAME:
        raise ValueError(
            f"Manifest protocol_name mismatch: {manifest['protocol_name']!r} != {PROTOCOL_NAME!r}"
        )
    if manifest["depth_semantics"] not in SUPPORTED_DEPTH_SEMANTICS:
        raise ValueError(
            "This evaluator expects manifest['depth_semantics'] to be one of "
            f"{SUPPORTED_DEPTH_SEMANTICS!r}, got {manifest['depth_semantics']!r}"
        )
    if manifest["distance_domain"] != "after_roi_crop_and_after_voxel_downsampling":
        raise ValueError("This evaluator expects the v1 distance domain")
    validity_rule = manifest["validity_rule"]
    if validity_rule.get("mode") != "opacity_threshold":
        raise ValueError("This evaluator only supports validity_rule.mode == 'opacity_threshold'")
    for view in manifest["views"]:
        for key in ("view_id", "width", "height", "fx", "fy", "cx", "cy", "camera_to_world", "odd_file", "even_file"):
            if key not in view:
                raise KeyError(f"View manifest missing required key {key!r} for view {view!r}")


def _load_view_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        if "depth" not in data.files:
            raise KeyError(f"{path} is missing 'depth'")
        if "opacity" not in data.files:
            raise KeyError(f"{path} is missing 'opacity'")
        return {
            "depth": np.asarray(data["depth"], dtype=np.float64),
            "opacity": np.asarray(data["opacity"], dtype=np.float64),
        }


def _make_valid_mask(depth: np.ndarray, opacity: np.ndarray, depth_min: float, opacity_threshold: float) -> np.ndarray:
    return (
        np.isfinite(depth)
        & np.isfinite(opacity)
        & (depth > float(depth_min))
        & (opacity >= float(opacity_threshold))
    )


def _raw_depth_to_metric_camera_z(raw_depth: np.ndarray, depth_semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if depth_semantics == "metric_camera_z_from_renderer":
        return raw_depth
    if depth_semantics == "inverse_camera_z_from_renderer":
        metric_depth = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        metric_depth[positive] = 1.0 / raw_depth[positive]
        return metric_depth
    raise ValueError(f"Unsupported depth semantics: {depth_semantics!r}")


def _backproject_depth_to_world(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    camera_to_world: np.ndarray,
) -> np.ndarray:
    if depth.shape != valid_mask.shape:
        raise ValueError(f"Depth/mask shape mismatch: {depth.shape} vs {valid_mask.shape}")
    y_idx, x_idx = np.nonzero(valid_mask)
    if y_idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[y_idx, x_idx].astype(np.float64, copy=False)
    x = ((x_idx.astype(np.float64) - float(cx)) / float(fx)) * z
    y = ((y_idx.astype(np.float64) - float(cy)) / float(fy)) * z
    points_cam = np.stack([x, y, z], axis=1)
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 4x4, got {c2w.shape}")
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    return points_cam @ rotation.T + translation[None, :]


def crop_points_to_roi(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.reshape(0, 3).astype(np.float64)
    inside = np.all((points >= bbox_min[None, :]) & (points <= bbox_max[None, :]), axis=1)
    return points[inside]


def deterministic_voxel_downsample(points: np.ndarray, voxel_size: float, origin: np.ndarray) -> np.ndarray:
    points = _filter_finite_points(points)
    if points.shape[0] == 0:
        return points
    if voxel_size <= 0.0:
        raise ValueError("voxel_size must be positive")
    voxel_index = np.floor((points - origin[None, :]) / float(voxel_size)).astype(np.int64)
    order = np.lexsort((np.arange(points.shape[0], dtype=np.int64), voxel_index[:, 2], voxel_index[:, 1], voxel_index[:, 0]))
    voxel_sorted = voxel_index[order]
    points_sorted = points[order]
    keep = np.ones(points_sorted.shape[0], dtype=bool)
    keep[1:] = np.any(voxel_sorted[1:] != voxel_sorted[:-1], axis=1)
    return points_sorted[keep]


def _resolve_eval_config(
    roi: Dict[str, Any],
    threshold_abs_override: Sequence[float] | None = None,
    voxel_size_override: float | None = None,
) -> Dict[str, Any]:
    scene_diagonal = float(roi["scene_diagonal"])
    if scene_diagonal <= 0.0:
        raise ValueError("scene_diagonal must be positive")

    if threshold_abs_override is None:
        thresholds_abs = [float(v) for v in roi["thresholds"]]
        threshold_mode = "ratio_of_scene_diagonal"
    else:
        thresholds_abs = [float(v) for v in threshold_abs_override]
        if not thresholds_abs:
            raise ValueError("threshold_abs_override must contain at least one threshold")
        if any((not math.isfinite(v)) or (v <= 0.0) for v in thresholds_abs):
            raise ValueError("All absolute thresholds must be finite and > 0")
        threshold_mode = "absolute_meter"
    threshold_ratios = [float(v) / scene_diagonal for v in thresholds_abs]

    if voxel_size_override is None:
        voxel_size = float(roi["voxel_size"])
        voxel_mode = "ratio_of_scene_diagonal"
    else:
        voxel_size = float(voxel_size_override)
        if (not math.isfinite(voxel_size)) or (voxel_size <= 0.0):
            raise ValueError("voxel_size_override must be finite and > 0")
        voxel_mode = "absolute_meter"
    voxel_ratio = voxel_size / scene_diagonal

    return {
        "scene_diagonal": scene_diagonal,
        "threshold_mode": threshold_mode,
        "thresholds_abs": thresholds_abs,
        "threshold_ratios": threshold_ratios,
        "voxel_mode": voxel_mode,
        "voxel_size": voxel_size,
        "voxel_ratio": voxel_ratio,
        "distance_unit": "meter",
    }


def brute_force_nn_distances(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    query = _filter_finite_points(query)
    ref = _filter_finite_points(ref)
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if ref.shape[0] == 0:
        return np.full((query.shape[0],), np.inf, dtype=np.float64)
    diff = query[:, None, :] - ref[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return np.sqrt(np.min(d2, axis=1))


def kd_tree_nn_distances(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    query = _filter_finite_points(query)
    ref = _filter_finite_points(ref)
    if query.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if ref.shape[0] == 0:
        return np.full((query.shape[0],), np.inf, dtype=np.float64)
    tree = cKDTree(ref)
    distances, _ = tree.query(query, k=1, workers=1)
    return np.asarray(distances, dtype=np.float64)


def compute_bidirectional_prf(
    pred_points: np.ndarray,
    ref_points: np.ndarray,
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    pred_points = _filter_finite_points(pred_points)
    ref_points = _filter_finite_points(ref_points)
    pred_to_ref = kd_tree_nn_distances(pred_points, ref_points)
    ref_to_pred = kd_tree_nn_distances(ref_points, pred_points)
    rows: List[Dict[str, Any]] = []
    pred_count = int(pred_points.shape[0])
    ref_count = int(ref_points.shape[0])
    for threshold in thresholds:
        t = float(threshold)
        precision = float(np.mean(pred_to_ref <= t)) if pred_count > 0 else 0.0
        recall = float(np.mean(ref_to_pred <= t)) if ref_count > 0 else 0.0
        denom = precision + recall
        fscore = (2.0 * precision * recall / denom) if denom > 0.0 else 0.0
        rows.append(
            {
                "threshold": t,
                "precision": precision,
                "recall": recall,
                "fscore": fscore,
                "pred_count": pred_count,
                "ref_count": ref_count,
            }
        )
    return rows


def _save_points_npz(path: Path, points: np.ndarray, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=np.asarray(points, dtype=np.float64), metadata_json=json.dumps(metadata, ensure_ascii=True))


def _build_split_point_cloud(
    split_name: str,
    manifest_path: Path,
    manifest: Dict[str, Any],
    roi: Dict[str, Any],
    voxel_size: float | None = None,
) -> Dict[str, Any]:
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)
    voxel_size = float(roi["voxel_size"] if voxel_size is None else voxel_size)
    validity_rule = manifest["validity_rule"]
    depth_semantics = str(manifest["depth_semantics"])
    depth_min = float(validity_rule.get("depth_min", DEFAULT_DEPTH_MIN))
    opacity_threshold = float(validity_rule.get("opacity_threshold", DEFAULT_OPACITY_THRESHOLD))
    all_points: List[np.ndarray] = []
    per_view_stats: List[Dict[str, Any]] = []
    views = sorted(manifest["views"], key=lambda v: str(v["view_id"]))
    for view in views:
        file_key = f"{split_name}_file"
        view_path = _resolve_path(manifest_path, view[file_key])
        arrays = _load_view_arrays(view_path)
        raw_depth = arrays["depth"]
        opacity = arrays["opacity"]
        expected_shape = (int(view["height"]), int(view["width"]))
        if raw_depth.shape != expected_shape:
            raise ValueError(f"{view_path} depth shape {raw_depth.shape} != expected {expected_shape}")
        if opacity.shape != expected_shape:
            raise ValueError(f"{view_path} opacity shape {opacity.shape} != expected {expected_shape}")
        metric_depth = _raw_depth_to_metric_camera_z(raw_depth, depth_semantics=depth_semantics)
        valid_mask = _make_valid_mask(metric_depth, opacity, depth_min=depth_min, opacity_threshold=opacity_threshold)
        world_points = _backproject_depth_to_world(
            depth=metric_depth,
            valid_mask=valid_mask,
            fx=float(view["fx"]),
            fy=float(view["fy"]),
            cx=float(view["cx"]),
            cy=float(view["cy"]),
            camera_to_world=np.asarray(view["camera_to_world"], dtype=np.float64),
        )
        all_points.append(world_points)
        per_view_stats.append(
            {
                "view_id": str(view["view_id"]),
                "file": str(view_path),
                "valid_pixel_count": int(valid_mask.sum()),
                "world_point_count": int(world_points.shape[0]),
                "raw_depth_min": float(np.nanmin(raw_depth)) if raw_depth.size else float("nan"),
                "raw_depth_max": float(np.nanmax(raw_depth)) if raw_depth.size else float("nan"),
                "metric_depth_min": float(np.nanmin(metric_depth)) if metric_depth.size else float("nan"),
                "metric_depth_max": float(np.nanmax(metric_depth)) if metric_depth.size else float("nan"),
            }
        )
    if all_points:
        points_world = np.concatenate(all_points, axis=0)
    else:
        points_world = np.zeros((0, 3), dtype=np.float64)
    points_roi = crop_points_to_roi(points_world, bbox_min=bbox_min, bbox_max=bbox_max)
    points_voxel = deterministic_voxel_downsample(points_roi, voxel_size=voxel_size, origin=bbox_min)
    return {
        "split_name": split_name,
        "points_world_count": int(points_world.shape[0]),
        "points_after_roi_count": int(points_roi.shape[0]),
        "points_after_voxel_count": int(points_voxel.shape[0]),
        "points_roi": points_roi,
        "points_voxel": points_voxel,
        "per_view_stats": per_view_stats,
    }


def evaluate_scene_bundle(
    manifest_path: Path,
    out_dir: Path,
    threshold_abs_override: Sequence[float] | None = None,
    voxel_size_override: float | None = None,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    _validate_manifest(manifest)
    roi_path = _resolve_path(manifest_path, manifest["roi_path"])
    roi = _load_json(roi_path)
    if roi.get("protocol_name") != PROTOCOL_NAME:
        raise ValueError("ROI file protocol_name mismatch")
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_json(out_dir / "manifest_snapshot.json", manifest)
    _save_json(out_dir / "roi_snapshot.json", roi)
    eval_config = _resolve_eval_config(
        roi=roi,
        threshold_abs_override=threshold_abs_override,
        voxel_size_override=voxel_size_override,
    )
    _save_json(out_dir / "effective_eval_config.json", eval_config)

    odd = _build_split_point_cloud(
        "odd",
        manifest_path=manifest_path,
        manifest=manifest,
        roi=roi,
        voxel_size=eval_config["voxel_size"],
    )
    even = _build_split_point_cloud(
        "even",
        manifest_path=manifest_path,
        manifest=manifest,
        roi=roi,
        voxel_size=eval_config["voxel_size"],
    )

    _save_points_npz(
        out_dir / "odd_points_after_roi.npz",
        odd["points_roi"],
        {
            "scene_name": manifest["scene_name"],
            "split_name": "odd",
            "stage": "after_roi",
            "point_count": odd["points_after_roi_count"],
        },
    )
    _save_points_npz(
        out_dir / "even_points_after_roi.npz",
        even["points_roi"],
        {
            "scene_name": manifest["scene_name"],
            "split_name": "even",
            "stage": "after_roi",
            "point_count": even["points_after_roi_count"],
        },
    )
    _save_points_npz(
        out_dir / "odd_points_after_voxel.npz",
        odd["points_voxel"],
        {
            "scene_name": manifest["scene_name"],
            "split_name": "odd",
            "stage": "after_voxel",
            "point_count": odd["points_after_voxel_count"],
            "voxel_size": eval_config["voxel_size"],
        },
    )
    _save_points_npz(
        out_dir / "even_points_after_voxel.npz",
        even["points_voxel"],
        {
            "scene_name": manifest["scene_name"],
            "split_name": "even",
            "stage": "after_voxel",
            "point_count": even["points_after_voxel_count"],
            "voxel_size": eval_config["voxel_size"],
        },
    )

    thresholds = [float(v) for v in eval_config["thresholds_abs"]]
    threshold_ratios = [float(v) for v in eval_config["threshold_ratios"]]
    metrics_rows = compute_bidirectional_prf(odd["points_voxel"], even["points_voxel"], thresholds=thresholds)
    rows_for_csv: List[Dict[str, Any]] = []
    for ratio, row in zip(threshold_ratios, metrics_rows):
        enriched = {
            "scene_name": manifest["scene_name"],
            "threshold_mode": eval_config["threshold_mode"],
            "threshold_ratio": ratio,
            "threshold_abs": row["threshold"],
            "voxel_mode": eval_config["voxel_mode"],
            "voxel_size": eval_config["voxel_size"],
            "precision": row["precision"],
            "recall": row["recall"],
            "fscore": row["fscore"],
            "pred_count": row["pred_count"],
            "ref_count": row["ref_count"],
        }
        rows_for_csv.append(enriched)

    _save_csv(
        out_dir / "metrics.csv",
        rows_for_csv,
        fieldnames=(
            "scene_name",
            "threshold_mode",
            "threshold_ratio",
            "threshold_abs",
            "voxel_mode",
            "voxel_size",
            "precision",
            "recall",
            "fscore",
            "pred_count",
            "ref_count",
        ),
    )

    metrics_payload = {
        "protocol_name": PROTOCOL_NAME,
        "scene_name": manifest["scene_name"],
        "depth_semantics": manifest["depth_semantics"],
        "validity_rule": manifest["validity_rule"],
        "distance_domain": manifest["distance_domain"],
        "scene_diagonal": eval_config["scene_diagonal"],
        "threshold_mode": eval_config["threshold_mode"],
        "threshold_ratios": threshold_ratios,
        "thresholds": thresholds,
        "voxel_mode": eval_config["voxel_mode"],
        "voxel_size": eval_config["voxel_size"],
        "voxel_ratio": eval_config["voxel_ratio"],
        "odd": {
            "points_world_count": odd["points_world_count"],
            "points_after_roi_count": odd["points_after_roi_count"],
            "points_after_voxel_count": odd["points_after_voxel_count"],
            "per_view_stats": odd["per_view_stats"],
        },
        "even": {
            "points_world_count": even["points_world_count"],
            "points_after_roi_count": even["points_after_roi_count"],
            "points_after_voxel_count": even["points_after_voxel_count"],
            "per_view_stats": even["per_view_stats"],
        },
        "metrics": rows_for_csv,
    }
    _save_json(out_dir / "metrics.json", metrics_payload)
    return metrics_payload


def _parse_threshold_ratios(text: str) -> Tuple[float, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one threshold ratio is required")
    return tuple(values)


def _parse_positive_float_list(text: str, field_name: str) -> Tuple[float, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if (not math.isfinite(value)) or (value <= 0.0):
            raise ValueError(f"{field_name} must contain only finite values > 0, got {value!r}")
        values.append(value)
    if not values:
        raise ValueError(f"At least one {field_name} value is required")
    return tuple(values)


def _cmd_build_roi(args: argparse.Namespace) -> None:
    points_path = Path(args.points_path).resolve()
    points = _load_points(points_path)
    roi = build_shared_roi_from_points(
        points=points,
        scene_name=args.scene_name,
        lower_quantile=float(args.lower_quantile),
        upper_quantile=float(args.upper_quantile),
        padding_ratio=float(args.padding_ratio),
        voxel_ratio=float(args.voxel_ratio),
        threshold_ratios=_parse_threshold_ratios(args.threshold_ratios),
    )
    roi["source_points_path"] = str(points_path)
    out_path = Path(args.out).resolve()
    _save_json(out_path, roi)
    print(f"ROI_SAVED {out_path}")
    print(f"SCENE_DIAGONAL {roi['scene_diagonal']:.12f}")
    print(f"VOXEL_SIZE {roi['voxel_size']:.12f}")


def _cmd_evaluate_scene(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    threshold_abs_override = None
    if args.threshold_abs_m:
        threshold_abs_override = _parse_positive_float_list(args.threshold_abs_m, field_name="threshold_abs_m")
    voxel_size_override = None
    if args.voxel_size_m is not None:
        voxel_size_override = float(args.voxel_size_m)
        if (not math.isfinite(voxel_size_override)) or (voxel_size_override <= 0.0):
            raise ValueError(f"voxel_size_m must be finite and > 0, got {voxel_size_override!r}")
    metrics = evaluate_scene_bundle(
        manifest_path=manifest_path,
        out_dir=out_dir,
        threshold_abs_override=threshold_abs_override,
        voxel_size_override=voxel_size_override,
    )
    print(f"EVAL_DONE {out_dir}")
    print(
        "THRESHOLD_MODE={} VOXEL_MODE={} VOXEL_SIZE={:.6f}".format(
            metrics["threshold_mode"],
            metrics["voxel_mode"],
            metrics["voxel_size"],
        )
    )
    for row in metrics["metrics"]:
        print(
            "THR={:.6f}m (ratio={:.8f}) P={:.6f} R={:.6f} F={:.6f}".format(
                row["threshold_abs"], row["threshold_ratio"], row["precision"], row["recall"], row["fscore"]
            )
        )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_roi = sub.add_parser("build-roi")
    ap_roi.add_argument("--scene_name", required=True)
    ap_roi.add_argument("--points_path", required=True)
    ap_roi.add_argument("--out", required=True)
    ap_roi.add_argument("--lower_quantile", type=float, default=DEFAULT_LOWER_QUANTILE)
    ap_roi.add_argument("--upper_quantile", type=float, default=DEFAULT_UPPER_QUANTILE)
    ap_roi.add_argument("--padding_ratio", type=float, default=DEFAULT_PADDING_RATIO)
    ap_roi.add_argument("--voxel_ratio", type=float, default=DEFAULT_VOXEL_RATIO)
    ap_roi.add_argument("--threshold_ratios", default="0.005,0.01,0.02")
    ap_roi.set_defaults(func=_cmd_build_roi)

    ap_eval = sub.add_parser("evaluate-scene")
    ap_eval.add_argument("--manifest", required=True)
    ap_eval.add_argument("--out_dir", required=True)
    ap_eval.add_argument(
        "--threshold_abs_m",
        default="",
        help="Optional comma-separated absolute distance thresholds in meters. Default: use ROI ratio thresholds.",
    )
    ap_eval.add_argument(
        "--voxel_size_m",
        type=float,
        default=None,
        help="Optional absolute voxel size in meters. Default: use ROI voxel ratio.",
    )
    ap_eval.set_defaults(func=_cmd_evaluate_scene)
    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
