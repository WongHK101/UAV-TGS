from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.geometric_repeatability.build_conditional_resplat_decision import (
    SelectionView,
    _candidate_block_counts,
    _selection_views,
    collect_selection_signals,
    compute_fixed_decision,
    run,
)
from tools.geometric_repeatability.evaluate_depth_definitions import DIAGNOSTIC_DEPTH_SEMANTICS


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_state(count: int) -> dict[str, object]:
    mass = np.full((count,), 40.0 / float(count - 100), dtype=np.float64)
    mass[:10] = 3.0
    mass[10:100] = 30.0 / 90.0
    view_count = np.zeros((count,), dtype=np.int64)
    view_count[:100] = 3
    return {
        "front_mass": mass,
        "raw_front_mass": 100.0,
        "assigned_front_mass": 100.0,
        "view_mass": {f"v{index}": 20.0 for index in range(5)},
        "block_mass": {"b0": 50.0, "b1": 50.0},
        "front_assigned_view_count": view_count,
    }


class ConditionalResplatDecisionTests(unittest.TestCase):
    def test_fixed_and_contract_can_trigger_on_stable_concentrated_support_anomalies(self) -> None:
        count = 10_000
        states = {"train": _passing_state(count), "guard": _passing_state(count)}
        decision = compute_fixed_decision(
            states,
            gaussian_count=count,
            scsp_indices=frozenset(range(12)),
            finite_support_indices=frozenset(range(100)),
            candidate_block_counts={index: 2 for index in range(100)},
        )
        self.assertTrue(decision["triggered"])
        self.assertEqual(decision["decision"], "execute_one_deterministic_resplat")
        self.assertEqual(decision["failed_conditions"], [])
        self.assertGreaterEqual(
            decision["split_evidence"]["train"]["top_0.1pct"]["mass_share"],
            0.20,
        )
        self.assertGreaterEqual(
            decision["split_evidence"]["train"]["top_1pct"]["mass_share"],
            0.50,
        )

    def test_diffuse_mass_fails_without_relaxing_contract(self) -> None:
        count = 10_000
        state = _passing_state(count)
        state["front_mass"] = np.ones((count,), dtype=np.float64)
        state["raw_front_mass"] = float(count)
        state["assigned_front_mass"] = float(count)
        decision = compute_fixed_decision(
            {"train": state, "guard": dict(state)},
            gaussian_count=count,
            scsp_indices=frozenset(range(12)),
            finite_support_indices=frozenset(range(count)),
            candidate_block_counts={index: 2 for index in range(count)},
        )
        self.assertFalse(decision["triggered"])
        self.assertEqual(decision["decision"], "skip_resplat")
        self.assertIn("train_top_1pct_mass_share", decision["failed_conditions"])
        self.assertIn("guard_top_0.1pct_mass_share", decision["failed_conditions"])

    def test_collect_signals_skips_test_npz_and_accumulates_top_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            views: list[SelectionView] = []
            for split in ("train", "guard"):
                reference = root / f"{split}_reference.npz"
                model = root / f"{split}_model.npz"
                np.savez_compressed(
                    reference,
                    depth=np.full((2, 2), 3.0, dtype=np.float32),
                    valid_mask=np.ones((2, 2), dtype=np.bool_),
                )
                np.savez_compressed(
                    model,
                    depth_expected_alpha_normalized=np.full((2, 2), 2.0, dtype=np.float32),
                    depth_transmittance_median=np.full((2, 2), 2.0, dtype=np.float32),
                    depth_max_contribution=np.full((2, 2), 2.0, dtype=np.float32),
                    top_contributor_index=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
                    top_contributor_weight=np.full((2, 2), 0.6, dtype=np.float32),
                    accumulated_opacity=np.full((2, 2), 0.8, dtype=np.float32),
                )
                views.append(SelectionView(split, split, "shared_physical_block", reference, model, (2, 2)))
            views.append(
                SelectionView(
                    "test",
                    "test",
                    "b1",
                    root / "missing_test_reference.npz",
                    root / "missing_test_model.npz",
                    (2, 2),
                )
            )
            states = collect_selection_signals(
                views,
                gaussian_count=2,
                appearance_modality="none",
                temperature_contract=None,
            )
            self.assertEqual(states["train"]["view_count"], 1)
            self.assertAlmostEqual(float(states["train"]["front_mass"].sum()), 3.0)
            np.testing.assert_allclose(
                states["train"]["top1_opacity_contribution"],
                np.asarray([1.2, 1.2]),
            )
            np.testing.assert_array_equal(
                states["train"]["top1_projected_footprint_pixels"],
                np.asarray([2, 2]),
            )
            # The same physical block appearing in train and guard is one block.
            self.assertEqual(_candidate_block_counts(views, np.asarray([0]))[0], 1)

    def test_temperature_observation_supports_signed_celsius(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.npz"
            model = root / "model.npz"
            np.savez_compressed(
                reference,
                depth=np.full((1, 2), 3.0, dtype=np.float32),
                valid_mask=np.ones((1, 2), dtype=np.bool_),
            )
            np.savez_compressed(
                model,
                depth_expected_alpha_normalized=np.full((1, 2), 2.0, dtype=np.float32),
                depth_transmittance_median=np.full((1, 2), 2.0, dtype=np.float32),
                depth_max_contribution=np.full((1, 2), 2.0, dtype=np.float32),
                top_contributor_index=np.zeros((1, 2), dtype=np.int32),
                top_contributor_weight=np.full((1, 2), 0.5, dtype=np.float32),
                accumulated_opacity=np.full((1, 2), 0.8, dtype=np.float32),
                render_temperature_c=np.zeros((1, 2), dtype=np.float32),
                target_temperature_c=np.asarray([[-5.0, 5.0]], dtype=np.float32),
                temperature_valid_mask=np.ones((1, 2), dtype=np.bool_),
            )
            views = [SelectionView("train", "train", "b0", reference, model, (1, 2))]
            states = collect_selection_signals(
                views,
                gaussian_count=1,
                appearance_modality="none",
                temperature_contract={
                    "keys": {
                        "rendered": "render_temperature_c",
                        "target": "target_temperature_c",
                        "valid_mask": "temperature_valid_mask",
                    }
                },
            )
            from tools.geometric_repeatability.build_conditional_resplat_decision import _finalize_temperature

            _finalize_temperature(states["train"])
            self.assertAlmostEqual(float(states["train"]["temperature_abs_mass"][0]), 10.0)
            self.assertAlmostEqual(float(states["train"]["temperature_observation_mean_c"][0]), 0.0)
            self.assertAlmostEqual(float(states["train"]["temperature_observation_variance_c2"][0]), 25.0)

    def test_selection_manifest_resolution_never_opens_test_npz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal.json"
            records = [
                {"pair_id": "train", "split": "train", "block_id": "b0"},
                {"pair_id": "guard", "split": "guard", "block_id": "b1"},
                {"pair_id": "test", "split": "test", "block_id": "b2"},
            ]
            _write_json(formal, {"scene_name": "Scene", "records": records})
            split_sha = _sha(formal)
            views = []
            for name, split in (("train", "train"), ("guard", "guard"), ("test", "test")):
                file_name = f"{name}.npz"
                if split != "test":
                    (root / file_name).write_bytes(name.encode("ascii"))
                    file_sha = _sha(root / file_name)
                    file_size = (root / file_name).stat().st_size
                else:
                    file_sha = "f" * 64
                    file_size = 1
                views.append(
                    {
                        "image_name": name,
                        "bound_split": split,
                        "width": 2,
                        "height": 2,
                        "npz_file": file_name,
                        "npz_sha256": file_sha,
                        "npz_size_bytes": file_size,
                    }
                )
            common = {
                "scene_name": "Scene",
                "formal_split_manifest_identity": {"sha256": split_sha},
                "camera_set_sha256": "c" * 64,
                "views": views,
            }
            reference = root / "reference.json"
            model = root / "model.json"
            _write_json(
                reference,
                {
                    **common,
                    "depth_semantics": "metric_camera_z_reference_mesh",
                    "all_split_reference_binding": {
                        "bound_split_labels": ["train", "guard", "test"],
                        "formal_split_manifest_sha256": split_sha,
                    },
                },
            )
            _write_json(
                model,
                {
                    **common,
                    "depth_diagnostics": {"enabled": True, **DIAGNOSTIC_DEPTH_SEMANTICS},
                },
            )
            resolved, *_ = _selection_views(reference, model, formal)
            by_split = {view.split: view for view in resolved}
            self.assertTrue(by_split["train"].model_path.is_file())
            self.assertTrue(by_split["guard"].reference_path.is_file())
            self.assertEqual(str(by_split["test"].model_path), "__formal_test_not_opened__")
            self.assertFalse((root / "test.npz").exists())

    def test_full_sidecar_publishes_receipt_without_test_npz(self) -> None:
        from plyfile import PlyData, PlyElement

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal.json"
            records = [
                {"pair_id": "train", "split": "train", "block_id": "b0"},
                {"pair_id": "guard", "split": "guard", "block_id": "b1"},
                {"pair_id": "test", "split": "test", "block_id": "b2"},
            ]
            _write_json(formal, {"scene_name": "Scene", "records": records})
            split_sha = _sha(formal)

            model_dir = root / "model_bundle"
            reference_dir = root / "reference_bundle"
            model_dir.mkdir()
            reference_dir.mkdir()
            reference_views = []
            model_views = []
            for split in ("train", "guard", "test"):
                reference_name = f"{split}.npz"
                model_name = f"{split}.npz"
                if split != "test":
                    reference_path = reference_dir / reference_name
                    model_path = model_dir / model_name
                    np.savez_compressed(
                        reference_path,
                        depth=np.full((2, 2), 3.0, dtype=np.float32),
                        valid_mask=np.ones((2, 2), dtype=np.bool_),
                    )
                    np.savez_compressed(
                        model_path,
                        depth_expected_alpha_normalized=np.full((2, 2), 2.0, dtype=np.float32),
                        depth_transmittance_median=np.full((2, 2), 2.0, dtype=np.float32),
                        depth_max_contribution=np.full((2, 2), 2.0, dtype=np.float32),
                        top_contributor_index=np.asarray([[0, 0], [1, 1]], dtype=np.int32),
                        top_contributor_weight=np.full((2, 2), 0.6, dtype=np.float32),
                        accumulated_opacity=np.full((2, 2), 0.8, dtype=np.float32),
                    )
                    reference_sha, reference_size = _sha(reference_path), reference_path.stat().st_size
                    model_sha, model_size = _sha(model_path), model_path.stat().st_size
                else:
                    reference_sha = model_sha = "f" * 64
                    reference_size = model_size = 1
                common_view = {
                    "image_name": split,
                    "bound_split": split,
                    "width": 2,
                    "height": 2,
                }
                reference_views.append(
                    {
                        **common_view,
                        "npz_file": reference_name,
                        "npz_sha256": reference_sha,
                        "npz_size_bytes": reference_size,
                    }
                )
                model_views.append(
                    {
                        **common_view,
                        "npz_file": model_name,
                        "npz_sha256": model_sha,
                        "npz_size_bytes": model_size,
                    }
                )

            dtype = [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("opacity", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
            ]
            vertices = np.zeros((4,), dtype=dtype)
            vertices["x"] = np.arange(4, dtype=np.float32)
            ply_path = root / "model.ply"
            PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(ply_path))
            ply_sha = _sha(ply_path)

            reference_manifest = reference_dir / "manifest.json"
            model_manifest = model_dir / "manifest.json"
            _write_json(
                reference_manifest,
                {
                    "scene_name": "Scene",
                    "depth_semantics": "metric_camera_z_reference_mesh",
                    "formal_split_manifest_identity": {"sha256": split_sha},
                    "camera_set_sha256": "c" * 64,
                    "all_split_reference_binding": {
                        "bound_split_labels": ["train", "guard", "test"],
                        "formal_split_manifest_sha256": split_sha,
                    },
                    "views": reference_views,
                },
            )
            _write_json(
                model_manifest,
                {
                    "scene_name": "Scene",
                    "appearance_modality": "none",
                    "gaussian_count": 4,
                    "formal_split_manifest_identity": {"sha256": split_sha},
                    "camera_set_sha256": "c" * 64,
                    "depth_diagnostics": {"enabled": True, **DIAGNOSTIC_DEPTH_SEMANTICS},
                    "gaussian_index_anchor": {"sha256": ply_sha},
                    "gaussian_index_binding": {"status": "verified", "gaussian_count": 4},
                    "model_point_cloud": {
                        "path": str(ply_path),
                        "sha256": ply_sha,
                        "size_bytes": ply_path.stat().st_size,
                    },
                    "views": model_views,
                },
            )
            scsp = root / "scsp.json"
            _write_json(
                scsp,
                {
                    "status": "passed",
                    "method": "scsp",
                    "input": {"ply_sha256": ply_sha},
                    "modified_indices": [0],
                    "invariants": {"no_training": True},
                    "sparse_support": {
                        "voxel_size": 1.5,
                        "max_voxel_radius": 2,
                        "points3d_sha256": "a" * 64,
                    },
                },
            )

            def fake_support(xyz, indices, **_kwargs):
                self.assertEqual(tuple(xyz.shape), (4, 3))
                return {int(index): 1.0 for index in indices.tolist()}, {"points3d_sha256": "a" * 64}

            output = root / "decision"
            receipt = run(
                reference_manifest_path=reference_manifest,
                model_manifest_path=model_manifest,
                formal_split_manifest_path=formal,
                scsp_manifest_path=scsp,
                sparse_root=root,
                out_dir=output,
                support_query=fake_support,
            )
            self.assertEqual(receipt["selection_policy"]["formal_test_npz_open_count_before_receipt"], 0)
            self.assertEqual(receipt["decision_contract"]["constants"]["material_front_threshold_m"], 0.25)
            self.assertIn("D_ref-D_max_contribution-0.25m", receipt["decision_contract"]["front_mass_semantics"])
            self.assertFalse(receipt["decision"]["triggered"])
            self.assertTrue((output / "conditional_resplat_decision_receipt.json").is_file())
            self.assertTrue((output / "selection_signals_train_guard_only.npz").is_file())
            csv_header = (output / "candidate_diagnostics_train_guard_only.csv").read_text(
                encoding="utf-8"
            ).splitlines()[0]
            self.assertIn("train_temperature_view_mean_count", csv_header)
            self.assertIn("guard_temperature_view_mean_count", csv_header)
            self.assertFalse((model_dir / "test.npz").exists())
            self.assertFalse((reference_dir / "test.npz").exists())


if __name__ == "__main__":
    unittest.main()
