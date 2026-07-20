from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.hold8_expected_depth_evaluator import (
    EXPECTED_DEPTH_EPSILON,
    MAIN_THRESHOLDS_M,
    SUPPLEMENTAL_THRESHOLDS_M,
    compute_expected_depth_metrics,
    evaluate_manifests,
    expected_depth_from_arrays,
    geometry_na_receipt,
    _validate_authoritative_split,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class Hold8ExpectedDepthMetricTests(unittest.TestCase):
    def test_fixed_contract_and_joint_valid_denominators(self) -> None:
        reference = np.array([10.0, 10.0, 10.0, 10.0, 50.0])
        ref_valid = np.array([True, True, True, True, False])
        expected = np.array([8.0, 10.5, 12.0, np.nan, 1.0])
        weight = np.array([0.9, 1.0, 1.0, 1.0, 1.0])
        result = compute_expected_depth_metrics(reference, ref_valid, expected, weight)

        self.assertEqual(MAIN_THRESHOLDS_M, (1.0, 2.0, 5.0))
        self.assertEqual(
            SUPPLEMENTAL_THRESHOLDS_M,
            (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0),
        )
        self.assertEqual(result["counts"]["reference_valid_pixels"], 4)
        self.assertEqual(result["counts"]["joint_valid_pixels"], 3)
        self.assertEqual(result["counts"]["missing_pixels"], 1)
        self.assertEqual(result["missing_rate"], 0.25)
        self.assertEqual(result["median_absolute_depth_error_m"], 2.0)
        tau1 = next(
            row for row in result["threshold_metrics"] if row["threshold_m"] == 1.0
        )
        self.assertEqual(tau1["front_rate"], 1.0 / 3.0)
        self.assertEqual(tau1["agreement_rate"], 1.0 / 3.0)
        self.assertTrue(tau1["is_main_table_threshold"])
        tau025 = result["threshold_metrics"][0]
        self.assertFalse(tau025["is_main_table_threshold"])

    def test_weight_epsilon_is_strict_and_not_tunable(self) -> None:
        result = compute_expected_depth_metrics(
            np.array([2.0, 2.0]),
            np.array([True, True]),
            np.array([2.0, 2.0]),
            np.array([EXPECTED_DEPTH_EPSILON, np.nextafter(EXPECTED_DEPTH_EPSILON, np.inf)]),
        )
        self.assertEqual(result["counts"]["joint_valid_pixels"], 1)
        self.assertEqual(result["missing_rate"], 0.5)

    def test_all_missing_uses_null_not_nan_for_joint_metrics(self) -> None:
        result = compute_expected_depth_metrics(
            np.array([2.0]), np.array([True]), np.array([2.0]), np.array([0.0])
        )
        self.assertEqual(result["missing_rate"], 1.0)
        self.assertIsNone(result["median_absolute_depth_error_m"])
        self.assertIsNone(result["threshold_metrics"][0]["front_rate"])
        self.assertIsNone(result["threshold_metrics"][0]["agreement_rate"])

    def test_existing_renderer_arrays_and_numerator_are_supported(self) -> None:
        expected, weight, positive, representation = expected_depth_from_arrays(
            {
                "depth_expected_alpha_normalized": np.array([2.0, 4.0]),
                "accumulated_opacity": np.array([0.5, 0.75]),
                "has_finite_positive_depth_sample": np.array([True, True]),
            }
        )
        np.testing.assert_array_equal(expected, [2.0, 4.0])
        np.testing.assert_array_equal(weight, [0.5, 0.75])
        np.testing.assert_array_equal(positive, [True, True])
        self.assertEqual(representation, "normalized_expected_depth")

        expected, weight, positive, representation = expected_depth_from_arrays(
            {
                "weighted_depth_sum": np.array([1.0, 3.0]),
                "weight_sum": np.array([0.5, 0.75]),
                "has_finite_positive_depth_sample": np.array([True, True]),
            }
        )
        np.testing.assert_allclose(expected, [2.0, 4.0])
        self.assertEqual(representation, "weighted_depth_sum")

    def test_ambiguous_or_conflicting_arrays_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            expected_depth_from_arrays(
                {
                    "expected_depth_camera_z": np.array([2.0]),
                    "weighted_depth_sum": np.array([1.0]),
                    "weight_sum": np.array([0.5]),
                    "has_finite_positive_depth_sample": np.array([True]),
                }
            )
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            expected_depth_from_arrays(
                {
                    "expected_depth_camera_z": np.array([2.0]),
                    "depth_expected_alpha_normalized": np.array([3.0]),
                    "weight_sum": np.array([0.5]),
                    "has_finite_positive_depth_sample": np.array([True]),
                }
            )

    def test_geometry_na_has_no_placeholder_metrics(self) -> None:
        receipt = geometry_na_receipt(
            scene_name="Building",
            method_name="external",
            technical_reason="Official renderer does not expose equivalent weights.",
        )
        self.assertEqual(receipt["status"], "not_applicable")
        self.assertEqual(receipt["geometry_status"], "not_available")
        self.assertIsNone(receipt["metrics"])


class Hold8ExpectedDepthManifestTests(unittest.TestCase):
    @staticmethod
    def _evaluate(reference: Path, model: Path, collection: Path, split: Path) -> dict:
        return evaluate_manifests(
            reference,
            model,
            collection,
            split,
            expected_collection_manifest_sha256=_sha256(collection),
            expected_scene_split_manifest_sha256=_sha256(split),
        )

    def _bundle(self, root: Path) -> tuple[Path, Path, Path, Path]:
        reference_npz = root / "reference.npz"
        model_npz = root / "model.npz"
        np.savez(
            reference_npz,
            depth=np.array([[10.0, 10.0]], dtype=np.float32),
            valid_mask=np.array([[1, 1]], dtype=np.uint8),
        )
        np.savez(
            model_npz,
            depth_expected_alpha_normalized=np.array([[8.0, 10.0]], dtype=np.float32),
            accumulated_opacity=np.array([[1.0, 1.0]], dtype=np.float32),
            has_finite_positive_depth_sample=np.array([[1, 1]], dtype=np.uint8),
        )
        test_list_sha = hashlib.sha256(b"000\n").hexdigest()
        split_manifest = root / "Building.split.json"
        split_payload = {
            "protocol_id": "uav-tgs-aaai27-hold8-v2",
            "scene": "Building",
            "collection_hash": "1" * 64,
            "split_hash": "3" * 64,
            "hashes": {"test_list_sha256": test_list_sha},
            "records": [
                {
                    "pair_id": "000",
                    "zero_based_sorted_index": 0,
                    "split": "test",
                }
            ],
        }
        _write_json(split_manifest, split_payload)
        collection_manifest = root / "collection_manifest.json"
        _write_json(
            collection_manifest,
            {
                "protocol_id": "uav-tgs-aaai27-hold8-v2",
                "collection_hash": "1" * 64,
                "collection_split_hash": "2" * 64,
                "scenes": [
                    {
                        "scene": "Building",
                        "split_hash": "3" * 64,
                        "manifest_sha256": _sha256(split_manifest),
                    }
                ],
            },
        )
        split_binding = {
            "collection_hash": "1" * 64,
            "collection_split_hash": "2" * 64,
            "scene_split_hash": "3" * 64,
            "test_list_sha256": test_list_sha,
            "scene_split_manifest_sha256": _sha256(split_manifest),
            "collection_manifest_sha256": _sha256(collection_manifest),
        }
        reference_manifest = root / "reference.json"
        model_manifest = root / "model.json"
        _write_json(
            reference_manifest,
            {
                "scene_name": "Building",
                "split": "test",
                "depth_semantics": "metric_camera_z",
                **split_binding,
                "views": [
                    {
                        "pair_id": "000",
                        "image_name": "000.png",
                        "npz_file": reference_npz.name,
                        "npz_size_bytes": reference_npz.stat().st_size,
                        "npz_sha256": _sha256(reference_npz),
                    }
                ],
            },
        )
        _write_json(
            model_manifest,
            {
                "scene_name": "Building",
                "method_name": "Raw-F3",
                "split": "test",
                "depth_contract": {
                    "name": "alpha_volume_weighted_expected_camera_z",
                    "z_semantics": "metric_camera_z",
                    "weight_semantics": "official_renderer_alpha_or_volume_contribution",
                    "weight_epsilon": 1.0e-8,
                    "positive_sample_evidence": "has_finite_positive_depth_sample",
                },
                **split_binding,
                "views": [
                    {
                        "pair_id": "000",
                        "image_name": "000.png",
                        "npz_file": model_npz.name,
                        "npz_size_bytes": model_npz.stat().st_size,
                        "npz_sha256": _sha256(model_npz),
                    }
                ],
            },
        )
        return reference_manifest, model_manifest, collection_manifest, split_manifest

    def test_bound_split_validates_against_frozen_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Building.source.json"
            payload = {
                "protocol_id": "uav-tgs-aaai27-hold8-v2",
                "scene": "Building",
                "collection_hash": "1" * 64,
                "split_hash": "3" * 64,
                "hashes": {"test_list_sha256": hashlib.sha256(b"000\n").hexdigest()},
                "records": [
                    {"pair_id": "000", "zero_based_sorted_index": 0, "split": "test"}
                ],
            }
            _write_json(source, payload)
            collection = root / "collection.json"
            _write_json(
                collection,
                {
                    "protocol_id": "uav-tgs-aaai27-hold8-v2",
                    "collection_hash": "1" * 64,
                    "collection_split_hash": "2" * 64,
                    "scenes": [
                        {
                            "scene": "Building",
                            "split_hash": "3" * 64,
                            "manifest_sha256": _sha256(source),
                        }
                    ],
                },
            )
            bound = root / "Building.bound.json"
            _write_json(bound, {**payload, "hold8_source_manifest_sha256": _sha256(source)})

            binding, test_ids = _validate_authoritative_split(
                collection_manifest_path=collection,
                scene_split_manifest_path=bound,
                scene_name="Building",
            )
            self.assertEqual(test_ids, ["000"])
            self.assertEqual(binding["scene_split_manifest_sha256"], _sha256(bound))

    def test_end_to_end_is_expected_only_without_auc_or_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reference, model, collection, split = self._bundle(Path(tmp))
            result = self._evaluate(reference, model, collection, split)
        self.assertEqual(result["geometry_status"], "available")
        self.assertEqual(result["depth_definition"]["weight_epsilon"], 1.0e-8)
        self.assertEqual(result["metrics"]["counts"]["joint_valid_pixels"], 2)
        serialized = json.dumps(result)
        self.assertNotIn("behind", serialized.lower())
        self.assertNotIn("auc", serialized.lower())
        self.assertNotIn("guard", serialized.lower())
        self.assertNotIn("median_depth", serialized.lower())
        self.assertNotIn("max_contribution", serialized.lower())

    def test_view_set_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            payload = json.loads(model.read_text(encoding="utf-8"))
            payload["views"][0]["pair_id"] = "different"
            _write_json(model, payload)
            with self.assertRaisesRegex(ValueError, "view-set mismatch"):
                self._evaluate(reference, model, collection, split)

    def test_wrong_depth_semantics_and_missing_positive_sample_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            payload = json.loads(model.read_text(encoding="utf-8"))
            payload["depth_contract"]["z_semantics"] = "euclidean_ray_distance"
            _write_json(model, payload)
            with self.assertRaisesRegex(ValueError, "depth_contract"):
                self._evaluate(reference, model, collection, split)

            reference, model, collection, split = self._bundle(root)
            model_npz = root / "model.npz"
            np.savez(
                model_npz,
                depth_expected_alpha_normalized=np.array([[8.0, 10.0]], dtype=np.float32),
                accumulated_opacity=np.array([[1.0, 1.0]], dtype=np.float32),
            )
            payload = json.loads(model.read_text(encoding="utf-8"))
            payload["views"][0]["npz_size_bytes"] = model_npz.stat().st_size
            payload["views"][0]["npz_sha256"] = _sha256(model_npz)
            _write_json(model, payload)
            with self.assertRaisesRegex(ValueError, "finite positive camera-z sample"):
                self._evaluate(reference, model, collection, split)

    def test_equal_but_non_authoritative_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            for path in (reference, model):
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["test_list_sha256"] = "f" * 64
                _write_json(path, payload)
            with self.assertRaisesRegex(ValueError, "not authoritative"):
                self._evaluate(reference, model, collection, split)

    def test_split_binding_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            payload = json.loads(model.read_text(encoding="utf-8"))
            payload["test_list_sha256"] = "f" * 64
            _write_json(model, payload)
            with self.assertRaisesRegex(ValueError, "test_list_sha256 mismatch"):
                self._evaluate(reference, model, collection, split)

    def test_frozen_manifest_hash_and_npz_hash_are_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            with self.assertRaisesRegex(ValueError, "frozen protocol"):
                evaluate_manifests(
                    reference,
                    model,
                    collection,
                    split,
                    expected_collection_manifest_sha256="f" * 64,
                    expected_scene_split_manifest_sha256=_sha256(split),
                )

            payload = json.loads(model.read_text(encoding="utf-8"))
            payload["views"][0].pop("npz_sha256")
            _write_json(model, payload)
            with self.assertRaisesRegex(ValueError, "NPZ SHA-256"):
                self._evaluate(reference, model, collection, split)

    def test_cli_writes_metrics_curve_and_na_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, model, collection, split = self._bundle(root)
            out_dir = root / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "hold8_expected_depth_evaluator.py"),
                    "evaluate",
                    "--reference-manifest",
                    str(reference),
                    "--model-manifest",
                    str(model),
                    "--collection-manifest",
                    str(collection),
                    "--scene-split-manifest",
                    str(split),
                    "--expected-collection-manifest-sha256",
                    _sha256(collection),
                    "--expected-scene-split-manifest-sha256",
                    _sha256(split),
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((out_dir / "geometry_metrics.json").is_file())
            self.assertTrue((out_dir / "front_agreement_curve.csv").is_file())

            na_path = root / "geometry_na.json"
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "hold8_expected_depth_evaluator.py"),
                    "geometry-na",
                    "--scene",
                    "Building",
                    "--method",
                    "External",
                    "--technical-reason",
                    "Official renderer cannot fairly export equivalent weights.",
                    "--output",
                    str(na_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(na_path.read_text(encoding="utf-8"))["status"],
                "not_applicable",
            )


if __name__ == "__main__":
    unittest.main()
