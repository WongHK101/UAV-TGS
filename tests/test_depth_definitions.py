from __future__ import annotations

import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from tools.thermal_radiometry.palette_lut import lut_sha256

from tools.geometric_repeatability.evaluate_depth_definitions import (
    DEFAULT_THRESHOLDS_M,
    DIAGNOSTIC_DEPTH_SEMANTICS,
    TEMPERATURE_RESPONSIBILITY_PROTOCOL,
    TEMPERATURE_SEMANTICS,
    _camera_set_sha256,
    _camera_sha256,
    compute_depth_metrics,
    evaluate,
    summarize_responsibility,
)
from tools.geometric_repeatability.extend_depth_reference_all_splits import _save_npz_deterministic
from tools.geometric_repeatability.build_all_split_probe_camera_manifest import (
    MODEL_CAMERA_MAX_POSITION_ERROR_M,
    _validate_model_camera,
)
from tools.geometric_repeatability.export_gaussian_probe_bundle import (
    _index_probe_images,
    _ply_xyz_sequence_identity,
    _resolve_manifest_image_path,
    _validate_gaussian_index_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _camera(image_name: str, x: float) -> dict:
    c2w = np.eye(4, dtype=np.float64)
    c2w[0, 3] = x
    return {
        "image_name": image_name,
        "width": 2,
        "height": 2,
        "fx": 2.0,
        "fy": 2.0,
        "cx": 1.0,
        "cy": 1.0,
        "camera_to_world": c2w.tolist(),
    }


def _write_xyz_ply(path: Path, xyz: np.ndarray, marker: float) -> None:
    vertices = np.empty(
        (xyz.shape[0],),
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("marker", "<f4")],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["marker"] = marker
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


class DepthMetricTests(unittest.TestCase):
    def test_model_camera_validation_accepts_serialization_noise_only(self) -> None:
        camera = SimpleNamespace(
            R=np.eye(3, dtype=np.float64),
            T=np.zeros(3, dtype=np.float64),
            width=100,
            height=80,
            FovX=np.pi / 2.0,
            FovY=np.pi / 2.0,
        )
        entry = {
            "width": 100,
            "height": 80,
            "fx": 50.0,
            "fy": 40.0,
            "position": [MODEL_CAMERA_MAX_POSITION_ERROR_M / 2.0, 0.0, 0.0],
            "rotation": np.eye(3).tolist(),
        }
        _validate_model_camera(entry, camera, "camera.png")
        entry["position"] = [MODEL_CAMERA_MAX_POSITION_ERROR_M * 2.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "position mismatch"):
            _validate_model_camera(entry, camera, "camera.png")

    def test_probe_image_resolution_prefers_exact_then_unique_stem(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.joinpath("0001.png").write_bytes(b"thermal")
            root.joinpath("0002.JPG").write_bytes(b"rgb")
            indexed = _index_probe_images(root)
            self.assertEqual(
                _resolve_manifest_image_path(
                    image_root=root,
                    image_name="0002.JPG",
                    indexed_images=indexed,
                ).name,
                "0002.JPG",
            )
            self.assertEqual(
                _resolve_manifest_image_path(
                    image_root=root,
                    image_name="0001.jpg",
                    indexed_images=indexed,
                ).name,
                "0001.png",
            )
            root.joinpath("0001.jpeg").write_bytes(b"ambiguous")
            indexed = _index_probe_images(root)
            with self.assertRaisesRegex(FileNotFoundError, "stem is not unique"):
                _resolve_manifest_image_path(
                    image_root=root,
                    image_name="0001.jpg",
                    indexed_images=indexed,
                )

    def test_front_agreement_missing_and_opacity(self) -> None:
        reference = np.full((2, 2), 10.0, dtype=np.float32)
        rendered = np.array([[8.0, 9.0], [10.0, 12.0]], dtype=np.float32)
        opacity = np.array([[0.8, 0.8], [0.8, 0.2]], dtype=np.float32)
        metrics = compute_depth_metrics(
            reference,
            rendered,
            accumulated_opacity=opacity,
            opacity_threshold=0.5,
            thresholds_m=DEFAULT_THRESHOLDS_M,
        )
        self.assertEqual(metrics["reference_count"], 4)
        self.assertEqual(metrics["valid_count"], 3)
        self.assertEqual(metrics["missing_count"], 1)
        self.assertEqual(metrics["front_counts"][1.0], 1)
        self.assertEqual(metrics["agreement_counts"][1.0], 2)

    def test_positive_mass_only_realized_k_and_set_mass(self) -> None:
        mass = np.zeros((1000,), dtype=np.float64)
        mass[7], mass[9], mass[11] = 70.0, 20.0, 10.0
        summary = summarize_responsibility(mass, scsp_indices={7, 12}, clamp20_indices={9})
        self.assertEqual(summary["top20"]["requested_count"], 20)
        self.assertEqual(summary["top20"]["realized_count"], 3)
        self.assertEqual(summary["top_0.1pct"]["requested_count"], 1)
        self.assertAlmostEqual(summary["top_0.1pct"]["mass_share"], 0.7)
        self.assertAlmostEqual(summary["scsp_set_mass"]["mass_share"], 0.7)
        self.assertAlmostEqual(summary["clamp20_set_mass"]["mass_share"], 0.2)
        self.assertEqual(len(summary["top20_entries"]), 3)

    def test_all_split_reference_npz_writer_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            arrays = {
                "depth": np.array([[1.0, np.nan]], np.float64),
                "valid_mask": np.array([[1, 0]], np.uint8),
            }
            first, second = root / "first.npz", root / "second.npz"
            _save_npz_deterministic(first, arrays)
            _save_npz_deterministic(second, arrays)
            self.assertEqual(_sha256(first), _sha256(second))

    def test_gaussian_index_binding_requires_order_proof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
            model, same_xyz, moved = root / "model.ply", root / "same_xyz.ply", root / "moved.ply"
            _write_xyz_ply(model, xyz, 1.0)
            _write_xyz_ply(same_xyz, xyz, 2.0)
            _write_xyz_ply(moved, xyz + np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], np.float32), 3.0)
            exact = _validate_gaussian_index_binding(
                model_point_cloud_path=model,
                index_anchor_path=same_xyz,
                gaussian_count=2,
                binding_manifest_path=None,
            )
            self.assertEqual(exact["proof"], "exact_ordered_xyz_sequence")
            with self.assertRaisesRegex(ValueError, "fixed-topology index binding receipt"):
                _validate_gaussian_index_binding(
                    model_point_cloud_path=model,
                    index_anchor_path=moved,
                    gaussian_count=2,
                    binding_manifest_path=None,
                )
            start_checkpoint = root / "start.pth"
            final_checkpoint = root / "final.pth"
            start_checkpoint.write_bytes(b"start-checkpoint")
            final_checkpoint.write_bytes(b"final-checkpoint")
            evidence = _write_json(
                root / "continuation.json",
                {
                    "schema": "uav-tgs-rgb-continuation-protocol-v1",
                    "topology_fixed": True,
                    "densification": False,
                    "pruning": False,
                    "opacity_reset": False,
                    "artifact_save_semantics": "aligned",
                    "anchor_iteration": 30000,
                    "final_iteration": 35000,
                    "start_checkpoint_sha256": _sha256(start_checkpoint),
                },
            )
            generator = (
                Path(__file__).resolve().parents[1]
                / "tools"
                / "geometric_repeatability"
                / "build_gaussian_index_binding_receipt.py"
            )

            def identity(path: Path) -> dict:
                return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}

            receipt = _write_json(
                root / "receipt.json",
                {
                    "protocol": "uav-tgs-gaussian-index-binding-v1",
                    "status": "verified",
                    "rendered_model_point_cloud_sha256": _sha256(model),
                    "gaussian_index_anchor_sha256": _sha256(moved),
                    "gaussian_count": 2,
                    "topology_fixed": True,
                    "index_order_preserved": True,
                    "evidence_manifest_identity": identity(evidence),
                    "anchor_checkpoint_identity": identity(start_checkpoint),
                    "final_checkpoint_identity": identity(final_checkpoint),
                    "anchor_iteration": 30000,
                    "final_iteration": 35000,
                    "ordered_xyz_proof": {
                        "anchor_ply": _ply_xyz_sequence_identity(moved),
                        "anchor_checkpoint": _ply_xyz_sequence_identity(moved),
                        "rendered_ply": _ply_xyz_sequence_identity(model),
                        "final_checkpoint": _ply_xyz_sequence_identity(model),
                        "anchor_ply_equals_start_checkpoint": True,
                        "rendered_ply_equals_final_checkpoint": True,
                        "continuation_has_no_topology_mutation": True,
                    },
                    "producer_identity": {"script": identity(generator)},
                },
            )
            bound = _validate_gaussian_index_binding(
                model_point_cloud_path=model,
                index_anchor_path=moved,
                gaussian_count=2,
                binding_manifest_path=receipt,
            )
            self.assertEqual(bound["proof"], "fixed_topology_invariant_audit_receipt")


class FormalEvaluatorTests(unittest.TestCase):
    def _bundle(self, root: Path, *, appearance_modality: str = "rgb", include_temperature: bool = True) -> dict:
        ref_root, model_root = root / "reference", root / "model"
        ref_root.mkdir()
        model_root.mkdir()
        specs = (
            ("train.png", "train", "b0", "nadir", np.array([[8.0, 9.0], [10.0, 12.0]], np.float32)),
            ("guard.png", "guard", "b1", "oblique", np.array([[10.0, 10.0], [0.0, 11.0]], np.float32)),
            ("test.png", "test", "b2", "oblique", np.array([[9.0, 10.0], [11.0, 10.0]], np.float32)),
        )
        formal_records, probe_views, ref_views, model_views = [], [], [], []
        for index, (name, split, block, orientation, depth) in enumerate(specs):
            camera = _camera(name, float(index))
            camera["camera_sha256"] = _camera_sha256(camera)
            probe_views.append({**camera, "bound_split": split, "split": split})
            formal_records.append(
                {
                    "pair_id": Path(name).stem,
                    "filename": name,
                    "scene": "InternalRoad",
                    "split": split,
                    "block_id": block,
                    "stratum": orientation,
                }
            )
            ref_path = ref_root / f"{index}.npz"
            model_path = model_root / f"{index}.npz"
            np.savez_compressed(
                ref_path,
                depth=np.full((2, 2), 10.0, np.float64),
                valid_mask=np.ones((2, 2), np.uint8),
            )
            opacity = np.array([[0.8, 0.4], [0.2, 0.0]], np.float32)
            expected_depth = depth.copy()
            expected_depth[1, 1] = 0.0
            present = expected_depth > 0.0
            opacity[~present] = 0.0
            median_depth = np.where(opacity >= 0.5, expected_depth, 0.0).astype(np.float32)
            top_index = np.array([[0, 1], [2, 3]], np.int32)
            top_weight = np.array([[0.6, 0.3], [0.1, 0.0]], np.float32)
            top_index[~present] = -1
            top_weight[~present] = 0.0
            arrays = {
                "depth_expected_alpha_normalized": expected_depth,
                "depth_transmittance_median": median_depth,
                "depth_max_contribution": expected_depth,
                "accumulated_opacity": opacity,
                "top_contributor_index": top_index,
                "top_contributor_weight": top_weight,
            }
            arrays["top_contributor_index"][1, 1] = -1
            arrays["top_contributor_weight"][1, 1] = 0.0
            if appearance_modality == "rgb":
                arrays.update(
                    render_rgb=np.zeros((3, 2, 2), np.float32),
                    target_rgb=np.ones((3, 2, 2), np.float32),
                )
            elif appearance_modality == "thermal_canonical":
                arrays.update(
                    render_thermal_canonical=np.zeros((3, 2, 2), np.float32),
                    target_thermal_canonical=np.ones((3, 2, 2), np.float32),
                )
            if include_temperature:
                arrays.update(
                    render_temperature_c=np.full((2, 2), 20.0, np.float32),
                    target_temperature_c=np.full((2, 2), 22.0, np.float32),
                    temperature_valid_mask=np.ones((2, 2), np.uint8),
                )
            np.savez_compressed(model_path, **arrays)
            common = {
                **camera,
                "bound_split": split,
                "split": split,
                "block_id": block,
                "orientation": orientation,
                "native_camera_to_world": camera["camera_to_world"],
                "render_camera_sha256": camera["camera_sha256"],
                "bound_native_camera_sha256": camera["camera_sha256"] if split != "guard" else "",
                "alignment_center_error_m": 0.0 if split != "guard" else None,
                "alignment_rotation_error_deg": 0.0 if split != "guard" else None,
                "render_camera_is_alignment_extrapolation": split == "guard",
            }
            ref_views.append(
                {
                    **common,
                    "npz_file": ref_path.name,
                    "npz_size_bytes": ref_path.stat().st_size,
                    "npz_sha256": _sha256(ref_path),
                }
            )
            model_views.append(
                {
                    **common,
                    "npz_file": model_path.name,
                    "npz_size_bytes": model_path.stat().st_size,
                    "npz_sha256": _sha256(model_path),
                }
            )

        native_cameras_path = _write_json(
            root / "cameras.json",
            [
                {
                    "img_name": view["image_name"],
                    "width": view["width"],
                    "height": view["height"],
                    "fx": view["fx"],
                    "fy": view["fy"],
                    "position": np.asarray(view["camera_to_world"], dtype=np.float64)[:3, 3].tolist(),
                    "rotation": np.asarray(view["camera_to_world"], dtype=np.float64)[:3, :3].tolist(),
                }
                for view in model_views
                if view["bound_split"] != "guard"
            ],
        )
        native_identity = {
            "path": str(native_cameras_path),
            "size_bytes": native_cameras_path.stat().st_size,
            "sha256": _sha256(native_cameras_path),
        }
        formal_path = _write_json(
            root / "formal.json",
            {"scene": "InternalRoad", "records": formal_records},
        )
        probe_path = _write_json(
            root / "probe.json",
            {
                "camera_manifest_type": "formal_all_split_probe_camera_manifest_v1",
                "scene_name": "InternalRoad",
                "bound_split_manifest_identity": {"sha256": _sha256(formal_path)},
                "camera_set_sha256": _camera_set_sha256({view["image_name"]: view for view in probe_views}),
                "model_cameras_json_identity": native_identity,
                "views": probe_views,
            },
        )
        identities = {
            "probe_camera_manifest_identity": {"sha256": _sha256(probe_path)},
            "formal_split_manifest_identity": {"sha256": _sha256(formal_path)},
        }
        base_reference_path = _write_json(
            root / "base_reference_manifest.json",
            {
                "scene_name": "InternalRoad",
                "reference_mesh_sha256": "a" * 64,
                "reference_mesh_backend": "openmvs_refine_mesh",
            },
        )
        base_reference_identity = {
            "path": str(base_reference_path),
            "size_bytes": base_reference_path.stat().st_size,
            "sha256": _sha256(base_reference_path),
        }
        reference_lock_path = _write_json(
            root / "formal_reference_lock.json",
            {
                "protocol": "uav-tgs-formal-reference-lock-v1",
                "status": "approved",
                "scene_name": "InternalRoad",
                "base_reference_manifest_sha256": base_reference_identity["sha256"],
                "reference_mesh_sha256": "a" * 64,
                "formal_split_manifest_sha256": _sha256(formal_path),
            },
        )
        reference_lock_identity = {
            "path": str(reference_lock_path),
            "size_bytes": reference_lock_path.stat().st_size,
            "sha256": _sha256(reference_lock_path),
        }
        ref_manifest = _write_json(
            ref_root / "manifest.json",
            {
                "scene_name": "InternalRoad",
                "depth_semantics": "metric_camera_z_reference_mesh",
                "reference_mesh_sha256": "a" * 64,
                "reference_mesh_backend": "openmvs_refine_mesh",
                "all_split_reference_binding": {
                    "extension_protocol": "fixed-openmvs-mesh-all-formal-splits-v1",
                    "base_reference_manifest_sha256": base_reference_identity["sha256"],
                    "reference_mesh_sha256": "a" * 64,
                    "reference_mesh_backend": "openmvs_refine_mesh",
                    "probe_camera_manifest_sha256": _sha256(probe_path),
                    "formal_split_manifest_sha256": _sha256(formal_path),
                    "bound_split_labels": ["train", "guard", "test"],
                    "mesh_or_backend_rebuilt": False,
                    "operator_pinned_base_reference_sha256": base_reference_identity["sha256"],
                    "operator_pinned_reference_mesh_sha256": "a" * 64,
                    "formal_reference_lock_sha256": _sha256(reference_lock_path),
                },
                "camera_set_sha256": _camera_set_sha256({view["image_name"]: view for view in ref_views}),
                "base_reference_manifest_identity": base_reference_identity,
                "formal_reference_lock_identity": reference_lock_identity,
                **identities,
                "views": ref_views,
            },
        )
        model_manifest = _write_json(
            model_root / "manifest.json",
            {
                "scene_name": "InternalRoad",
                "gaussian_count": 1000,
                "gaussian_index_anchor": {"sha256": "c" * 64},
                "model_point_cloud": {"sha256": "c" * 64},
                "gaussian_index_binding": {
                    "status": "verified",
                    "proof": "identical_ply_sha256",
                    "gaussian_count": 1000,
                    "rendered_model_point_cloud": {"sha256": "c" * 64},
                    "gaussian_index_anchor": {"sha256": "c" * 64},
                    "rendered_ordered_xyz": {"sequence_sha256": "f" * 64},
                    "anchor_ordered_xyz": {"sequence_sha256": "f" * 64},
                    "binding_receipt_identity": None,
                },
                "appearance_modality": appearance_modality,
                "depth_diagnostics": {"enabled": True, **DIAGNOSTIC_DEPTH_SEMANTICS},
                "camera_set_sha256": _camera_set_sha256({view["image_name"]: view for view in model_views}),
                "render_camera_set_sha256": _camera_set_sha256(
                    {view["image_name"]: view for view in model_views}
                ),
                "native_cameras_json": str(native_cameras_path),
                "native_cameras_json_identity": native_identity,
                "native_camera_coverage": {
                    "probe_bound_model_cameras_json_identity_verified": True,
                    "render_camera_set_sha256": _camera_set_sha256(
                        {view["image_name"]: view for view in model_views}
                    ),
                },
                "strict_to_native_alignment": {
                    "strict_to_native_transform": np.eye(4, dtype=np.float64).tolist(),
                    "revalidated_against_bound_native_cameras": {
                        "count": 2,
                        "translation_error_mean_m": 0.0,
                        "translation_error_max_m": 0.0,
                        "rotation_error_mean_deg": 0.0,
                        "rotation_error_max_deg": 0.0,
                        "maximum_allowed_translation_error_m": 1.0e-4,
                        "maximum_allowed_rotation_error_deg": 5.0e-2,
                        "status": "passed",
                    },
                },
                "temperature_responsibility_derivation": (
                    {
                        "protocol": TEMPERATURE_RESPONSIBILITY_PROTOCOL,
                        "source_model_manifest_sha256": "1" * 64,
                        "source_render_binding_manifest_sha256": "2" * 64,
                        "formal_split_manifest_sha256": _sha256(formal_path),
                        "tsdk_binding_manifest_sha256": "3" * 64,
                        "tsdk_protocol_sha256": "d" * 64,
                        "temperature_manifest_sha256": "4" * 64,
                        "valid_support_manifest_sha256": "5" * 64,
                        "range_manifest_sha256": "6" * 64,
                        "canonical_manifest_sha256": "7" * 64,
                        "lut_sha256_uint8_rgb": lut_sha256(),
                    }
                    if include_temperature
                    else None
                ),
                **identities,
                "views": model_views,
            },
        )
        temp_manifest = None
        if include_temperature:
            temp_manifest = _write_json(
                root / "temperature_responsibility.json",
                {
                    "protocol": TEMPERATURE_RESPONSIBILITY_PROTOCOL,
                    "scene_name": "InternalRoad",
                    "semantics": TEMPERATURE_SEMANTICS,
                    "units": "Celsius",
                    "dtype": "float32",
                    "tsdk_protocol_sha256": "d" * 64,
                    "source_model_manifest_sha256": "1" * 64,
                    "source_render_binding_manifest_sha256": "2" * 64,
                    "formal_split_manifest_sha256": _sha256(formal_path),
                    "tsdk_binding_manifest_sha256": "3" * 64,
                    "temperature_manifest_sha256": "4" * 64,
                    "valid_support_manifest_sha256": "5" * 64,
                    "range_manifest_sha256": "6" * 64,
                    "canonical_manifest_sha256": "7" * 64,
                    "lut_sha256_uint8_rgb": lut_sha256(),
                    "coverage": {
                        "all_formal_views_exactly_once": True,
                        "split_counts": {"train": 1, "guard": 1, "test": 1},
                    },
                    "target_provenance": {
                        "png_or_palette_inverse_used_for_target": False,
                        "source_training_target_forward_colorization_exact_on_valid_support": True,
                    },
                    "model_manifest_sha256": _sha256(model_manifest),
                    "keys": {
                        "rendered": "render_temperature_c",
                        "target": "target_temperature_c",
                        "valid_mask": "temperature_valid_mask",
                    },
                    "views": [
                        {
                            "image_name": view["image_name"],
                            "split": view["bound_split"],
                            "npz_sha256": view["npz_sha256"],
                            "source_npz_sha256": "8" * 64,
                            "temperature_target_sha256": "9" * 64,
                            "valid_support_sha256": "a" * 64,
                        }
                        for view in model_views
                    ],
                },
            )
        scsp = _write_json(root / "scsp.json", {"selected_indices": [0], "anchor_ply_sha256": "c" * 64})
        clamp20 = _write_json(root / "clamp20.json", {"modified_indices": [1], "anchor_ply_sha256": "c" * 64})
        return {
            "reference": ref_manifest,
            "model": model_manifest,
            "probe": probe_path,
            "formal": formal_path,
            "temperature": temp_manifest,
            "scsp": scsp,
            "clamp20": clamp20,
        }

    def test_formal_end_to_end_has_curves_but_only_max_depth_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root)
            out = root / "out"
            summary = evaluate(
                reference_manifest_path=bundle["reference"],
                model_manifest_path=bundle["model"],
                probe_camera_manifest_path=bundle["probe"],
                formal_split_manifest_path=bundle["formal"],
                temperature_responsibility_manifest_path=bundle["temperature"],
                scsp_indices_path=bundle["scsp"],
                clamp20_indices_path=bundle["clamp20"],
                out_dir=out,
            )
            self.assertEqual(set(summary["depth_definitions"]), {"expected", "median", "max_contribution"})
            for split in ("train", "guard", "test"):
                signals = summary["responsibility"]["splits"][split]
                self.assertIn("front_max_contribution", signals)
                self.assertNotIn("front_expected", signals)
                self.assertNotIn("front_median", signals)
                self.assertIn("rgb_abs_top1_occupancy_approx", signals)
                self.assertIn("temperature_abs_top1_occupancy_approx", signals)
                self.assertEqual(signals["front_max_contribution"]["view_coverage_fraction"], 1.0)
                assigned_fraction = signals["rgb_abs_top1_occupancy_approx"]["assignment_coverage"][
                    "assigned_positive_pixel_fraction"
                ]
                self.assertGreater(assigned_fraction, 0.0)
                self.assertLess(assigned_fraction, 1.0)
                self.assertGreater(
                    signals["temperature_abs_top1_occupancy_approx"]["assignment_coverage"]["unassigned_mass"],
                    0.0,
                )
            expected = summary["depth_definitions"]["expected"]
            self.assertIn("front_curve_auc", expected["groups"]["split"]["train"])
            self.assertIn("no_opacity_mask", expected["opacity_sensitivity"])
            self.assertIn("[0.25,0.5)", expected["opacity_sensitivity"]["opacity_bins"])
            self.assertTrue((out / expected["signed_residual_cdf"]["groups"]["split"]["train"]).is_file())

    def test_base_reference_identity_is_bound_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root, include_temperature=False)
            reference = json.loads(bundle["reference"].read_text(encoding="utf-8"))
            reference["base_reference_manifest_identity"]["size_bytes"] += 1
            _write_json(bundle["reference"], reference)
            with self.assertRaisesRegex(ValueError, "Base reference manifest file identity mismatch"):
                evaluate(
                    reference_manifest_path=bundle["reference"],
                    model_manifest_path=bundle["model"],
                    probe_camera_manifest_path=bundle["probe"],
                    formal_split_manifest_path=bundle["formal"],
                    out_dir=root / "out-size",
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root, include_temperature=False)
            alternate_base = _write_json(root / "alternate_base_reference_manifest.json", {"alternate": True})
            reference = json.loads(bundle["reference"].read_text(encoding="utf-8"))
            reference["base_reference_manifest_identity"] = {
                "path": str(alternate_base),
                "size_bytes": alternate_base.stat().st_size,
                "sha256": _sha256(alternate_base),
            }
            _write_json(bundle["reference"], reference)
            with self.assertRaisesRegex(
                ValueError, "Top-level base reference identity/all-split binding SHA-256 mismatch"
            ):
                evaluate(
                    reference_manifest_path=bundle["reference"],
                    model_manifest_path=bundle["model"],
                    probe_camera_manifest_path=bundle["probe"],
                    formal_split_manifest_path=bundle["formal"],
                    out_dir=root / "out-binding",
                )

    def test_temperature_arrays_without_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root)
            with self.assertRaisesRegex(ValueError, "temperature-responsibility manifest"):
                evaluate(
                    reference_manifest_path=bundle["reference"],
                    model_manifest_path=bundle["model"],
                    probe_camera_manifest_path=bundle["probe"],
                    formal_split_manifest_path=bundle["formal"],
                    out_dir=root / "out",
                )

    def test_thermal_palette_bundle_does_not_emit_rgb_responsibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root, appearance_modality="thermal_canonical", include_temperature=False)
            summary = evaluate(
                reference_manifest_path=bundle["reference"],
                model_manifest_path=bundle["model"],
                probe_camera_manifest_path=bundle["probe"],
                formal_split_manifest_path=bundle["formal"],
                out_dir=root / "out",
            )
            self.assertEqual(summary["responsibility"]["rgb_responsibility_status"], "not_applicable")
            self.assertNotIn("rgb_abs_top1_occupancy_approx", summary["responsibility"]["splits"]["test"])

    def test_formal_threshold_below_half_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = self._bundle(root, include_temperature=False)
            with self.assertRaisesRegex(ValueError, ">= 0.5"):
                evaluate(
                    reference_manifest_path=bundle["reference"],
                    model_manifest_path=bundle["model"],
                    probe_camera_manifest_path=bundle["probe"],
                    formal_split_manifest_path=bundle["formal"],
                    opacity_threshold=0.49,
                    out_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
