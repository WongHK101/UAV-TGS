from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from oct_gs.formal import sha256_file, sha256_json
from tools.evaluate_formal_baseline_hotspots import (
    EVALUATION_SUPPORT_POLICY,
    _hotspot_source_receipt,
    _resolve_under,
    _support_index,
    evaluate_formal_baseline_hotspots,
    write_atomic_report,
)
from tools.thermal_radiometry.palette_lut import (
    PALETTE_NAME,
    lut_sha256,
    temperature_to_rgb,
    rgb_to_temperature,
)
from tools.thermal_radiometry.build_formal_evaluation_binding import (
    build_formal_evaluation_binding,
    expected_split_labels,
)
from oct_gs.radiance import METHOD_SEMANTICS, TARGET_SEMANTICS


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.scene = "InternalRoad"
        self.temperature_root = root / "temperature"
        self.render_root = root / "renders"
        self.optimization_support_root = root / "optimization_support"
        self.evaluation_support_root = root / "evaluation_support"
        for path in (
            self.temperature_root / self.scene,
            self.render_root,
            self.optimization_support_root,
            self.evaluation_support_root / "bool",
        ):
            path.mkdir(parents=True, exist_ok=True)

        self.records = (
            ("train1", "train", 0),
            ("guard1", "guard", 0),
            ("test1", "test", 1),
            ("test2", "test", 2),
        )
        target_values = {
            "train1": np.asarray([[5, 25, 60], [75, 90, 40]], dtype=np.float32),
            "guard1": np.asarray([[5, 10, 15], [20, 25, 30]], dtype=np.float32),
            "test1": np.asarray([[10, 55, 80], [20, 45, 90]], dtype=np.float32),
            "test2": np.asarray([[49, 50, 51], [0, 100, 75]], dtype=np.float32),
        }
        prediction_values = {
            "test1": np.asarray([[10, 60, 70], [30, 40, 90]], dtype=np.float32),
            "test2": np.asarray([[40, 60, 55], [0, 90, 60]], dtype=np.float32),
        }
        self.target_paths: dict[str, Path] = {}
        for pair_id, values in target_values.items():
            target = self.temperature_root / self.scene / f"{pair_id}.npy"
            np.save(target, values, allow_pickle=False)
            self.target_paths[pair_id] = target
        for pair_id, values in prediction_values.items():
            rgb, _ = temperature_to_rgb(values, 0.0, 100.0)
            Image.fromarray(rgb, mode="RGB").save(self.render_root / f"{pair_id}.png")

        decode_rows = []
        protocol_rows = []
        protocol_hash = "a" * 64
        decode_parameters = {
            "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
            "distance_m": {"value": 5.0, "source": "lrf"},
            "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
            "humidity_percent": {
                "value": 70.0,
                "source": "benchmark_assumption",
            },
            "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
        }
        for pair_id, _, _ in self.records:
            digest = sha256_file(self.target_paths[pair_id])
            decode_rows.append(
                {
                    "pair_id": pair_id,
                    "scene": self.scene,
                    "success": True,
                    "dtype": "float32",
                    "output_sha256": digest,
                }
            )
            protocol_rows.append(
                {
                    "pair_id": pair_id,
                    "scene": self.scene,
                    "schema_version": "uav-tgs.radiometry-protocol.v1",
                    "protocol_hash": protocol_hash,
                    "decode_parameters": decode_parameters,
                }
            )
        self.decode_manifest = root / "decode.jsonl"
        self.decode_protocol = root / "protocol.jsonl"
        self.decode_manifest.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in decode_rows),
            encoding="utf-8",
        )
        self.decode_protocol.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in protocol_rows),
            encoding="utf-8",
        )

        self.bound_split = root / "split.json"
        split_hash = "b" * 64
        _write_json(
            self.bound_split,
            {
                "scene": self.scene,
                "split_hash": split_hash,
                "counts": {"total": 4, "train": 1, "guard": 1, "test": 2},
                "decode_binding": {
                    "adapter_backend": "official-dji-irp",
                    "decode_manifest_sha256": sha256_file(self.decode_manifest),
                    "decode_protocol_sha256": sha256_file(self.decode_protocol),
                    "protocol_hash": protocol_hash,
                    "verified_decode_requests": True,
                    "verified_raw_rjpeg_hashes": True,
                    "verified_temperature_file_hashes": True,
                },
                "records": [
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "thermal_camera_name": f"{pair_id}.png",
                        "stratum": "nadir:p-090",
                        "strip_id": "tg-0000",
                        "block_index": block_index,
                    }
                    for pair_id, split, block_index in self.records
                ],
            },
        )

        self.range_manifest = root / "range.json"
        range_basis = {
            "scene": self.scene,
            "split_hash": split_hash,
            "configuration": {
                "guard_role": "not_read",
                "test_role": "qa_only_not_used_for_estimation",
            },
            "Tmin": 0.0,
            "Tmax": 100.0,
        }
        range_hash = sha256_json(range_basis)
        _write_json(
            self.range_manifest,
            {
                **range_basis,
                "source_split_manifest_sha256": sha256_file(self.bound_split),
                "range_hash": range_hash,
                "per_frame_quantiles": [
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "minimum": float(target_values[pair_id].min()),
                        "maximum": float(target_values[pair_id].max()),
                    }
                    for pair_id, split, _ in self.records
                ],
            },
        )

        canonical_rows = []
        support_rows = []
        for pair_id, _, _ in self.records:
            digest = sha256_file(self.target_paths[pair_id])
            canonical_rows.append(
                {
                    "pair_id": pair_id,
                    "relative_input": f"{self.scene}/{pair_id}.npy",
                    "input_sha256": digest,
                    "temperature_dtype": "float32",
                    "relative_output": f"{self.scene}/{pair_id}.png",
                    "output_sha256": "d" * 64,
                }
            )
            support_path = self.optimization_support_root / f"{pair_id}.npy"
            np.save(support_path, np.ones((2, 3), dtype=np.bool_), allow_pickle=False)
            support_rows.append(
                {
                    "pair_id": pair_id,
                    "input_temperature": {"dtype": "float32", "sha256": digest},
                    "output_temperature": {"dtype": "float32", "sha256": digest},
                    "valid_support": {
                        "dtype": "bool",
                        "relative_path": support_path.name,
                        "sha256": sha256_file(support_path),
                    },
                }
            )
        self.canonical_manifest = root / "canonical.json"
        _write_json(
            self.canonical_manifest,
            {
                "schema": "uav-tgs-canonical-hot-iron-v1",
                "status": "complete",
                "palette": {"name": PALETTE_NAME, "sha256_uint8_rgb": lut_sha256()},
                "image_encoding": {
                    "format": "PNG",
                    "mode": "RGB",
                    "lossless": True,
                    "gamma": 1.0,
                },
                "temperature_range": {
                    "tmin_c": 0.0,
                    "tmax_c": 100.0,
                    "source": {"sha256": sha256_file(self.range_manifest)},
                },
                "files": canonical_rows,
            },
        )
        self.optimization_support_manifest = root / "optimization_support.json"
        _write_json(
            self.optimization_support_manifest,
            {
                "schema": "uav-tgs-undistorted-temperature-v1",
                "status": "complete",
                "files": support_rows,
            },
        )

        evaluation_rows = []
        for pair_id in ("test1", "test2"):
            support = np.ones((2, 3), dtype=np.bool_)
            if pair_id == "test2":
                support[1, 1] = False
            support_path = self.evaluation_support_root / "bool" / f"{pair_id}.npy"
            np.save(support_path, support, allow_pickle=False)
            evaluation_rows.append(
                {
                    "pair_id": pair_id,
                    "outputs": {
                        "bool": {
                            "dtype": "bool",
                            "relative_path": f"bool/{pair_id}.npy",
                            "sha256": sha256_file(support_path),
                        }
                    },
                }
            )
        self.evaluation_support_manifest = root / "evaluation_support.json"
        _write_json(
            self.evaluation_support_manifest,
            {
                "schema_name": "uav-tgs-formal-temperature-support",
                "schema_version": 1,
                "split": "test",
                "expected_test_count": 2,
                "policy": EVALUATION_SUPPORT_POLICY,
                "source_manifests": {
                    "split": {"sha256": sha256_file(self.bound_split)},
                    "valid_support": {
                        "sha256": sha256_file(self.optimization_support_manifest)
                    },
                },
                "records": evaluation_rows,
            },
        )

        support_index = _support_index(
            [pair_id for pair_id, _, _ in self.records],
            {row["pair_id"]: row for row in support_rows},
        )
        evaluation_index = [
            {
                "pair_id": row["pair_id"],
                "sha256": row["outputs"]["bool"]["sha256"],
                "encoding": "bool-npy",
            }
            for row in evaluation_rows
        ]
        parameter_index = [
            {
                "pair_id": pair_id,
                "output_sha256": sha256_file(self.target_paths[pair_id]),
                "parameters_sha256": sha256_json(
                    {
                        key: {
                            "value": float(value["value"]),
                            "source": value["source"],
                        }
                        for key, value in sorted(decode_parameters.items())
                    }
                ),
            }
            for pair_id, _, _ in self.records
        ]
        canonical_by_pair = {row["pair_id"]: row for row in canonical_rows}
        target_index = [
            {
                "pair_id": pair_id,
                "split": split,
                "camera_name": f"{pair_id}.png",
                "temperature_sha256": sha256_file(self.target_paths[pair_id]),
                "canonical_sha256": canonical_by_pair[pair_id]["output_sha256"],
                "shape_hw": [2, 3],
            }
            for pair_id, split, _ in self.records
        ]
        formal_binding = {
            "schema": "uav-tgs-oct-formal-binding-v1",
            "scene_name": self.scene,
            "bound_split": {
                "sha256": sha256_file(self.bound_split),
                "split_hash": split_hash,
                "counts": {"total": 4, "train": 1, "guard": 1, "test": 2},
            },
            "temperature_range": {
                "sha256": sha256_file(self.range_manifest),
                "range_hash": range_hash,
                "tmin_c": 0.0,
                "tmax_c": 100.0,
            },
            "tsdk_target": {
                "decode_manifest_sha256": sha256_file(self.decode_manifest),
                "decode_protocol_sha256": sha256_file(self.decode_protocol),
                "protocol_hash": protocol_hash,
                "pair_parameter_index_sha256": sha256_json(parameter_index),
                "target_semantics": TARGET_SEMANTICS,
                "method_semantics": METHOD_SEMANTICS,
                "adapter_backend": "official-dji-irp",
                "absolute_thermometry_claimed": False,
            },
            "canonical_target": {
                "manifest_sha256": sha256_file(self.canonical_manifest),
                "lut_sha256": lut_sha256(),
                "target_index_sha256": sha256_json(target_index),
            },
            "support": {
                "optimization": {
                    "manifest_sha256": sha256_file(self.optimization_support_manifest),
                    "support_index_sha256": sha256_json(support_index),
                },
                "evaluation": {
                    "manifest_sha256": sha256_file(self.evaluation_support_manifest),
                    "support_index_sha256": sha256_json(evaluation_index),
                    "policy": EVALUATION_SUPPORT_POLICY,
                    "split": "test",
                },
            },
        }
        formal_binding["formal_protocol_sha256"] = sha256_json(formal_binding)
        source_files = [{"path": "fixture.py", "sha256": "e" * 64, "bytes": 1}]
        source_provenance = {
            "schema": "uav-tgs-oct-training-source-v1",
            "git_commit": "f" * 40,
            "git_clean": True,
            "git_status_porcelain_sha256": hashlib.sha256(b"").hexdigest(),
            "files": source_files,
            "files_sha256": sha256_json(source_files),
        }
        self.formal_protocol_manifest = root / "formal_protocol.json"
        formal_protocol = {
            "schema": "uav-tgs-oct-formal-protocol-v3",
            "git_commit": "f" * 40,
            "source_provenance": source_provenance,
            "formal_binding": formal_binding,
        }
        formal_protocol["manifest_sha256"] = sha256_json(formal_protocol)
        _write_json(self.formal_protocol_manifest, formal_protocol)

        source_receipt = _hotspot_source_receipt(
            scene_name=self.scene,
            split_sha256=sha256_file(self.bound_split),
            split_hash=split_hash,
            decode_manifest_sha256=sha256_file(self.decode_manifest),
            decode_protocol_sha256=sha256_file(self.decode_protocol),
            range_sha256=sha256_file(self.range_manifest),
            range_hash=range_hash,
            support_sha256=sha256_file(self.optimization_support_manifest),
            support_index_sha256=sha256_json(support_index),
            train_camera_names=["train1.png"],
        )
        self.threshold_manifest = root / "threshold.json"
        threshold = {
            "schema": "uav-tgs-oct-train-only-hotspot-threshold-v1",
            "scene_name": self.scene,
            "source_receipt": source_receipt,
            "source_split": "train",
            "test_statistics_used": False,
            "quantile": 0.95,
            "histogram_bins": 65536,
            "valid_train_pixels": 6,
            "threshold_c": 50.0,
            "range_c": [0.0, 100.0],
            "train_view_ids_sha256": sha256_json(["train1.png"]),
        }
        threshold["threshold_sha256"] = sha256_json(threshold)
        _write_json(self.threshold_manifest, threshold)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            method_name="Raw-F3",
            scene_name=self.scene,
            formal_protocol_manifest=self.formal_protocol_manifest,
            bound_split=self.bound_split,
            decode_manifest=self.decode_manifest,
            decode_protocol=self.decode_protocol,
            range_manifest=self.range_manifest,
            canonical_manifest=self.canonical_manifest,
            optimization_support_manifest=self.optimization_support_manifest,
            evaluation_support_manifest=self.evaluation_support_manifest,
            hotspot_threshold_manifest=self.threshold_manifest,
            temperature_root=self.temperature_root,
            evaluation_support_root=self.evaluation_support_root,
            render_root=self.render_root,
            chunk_pixels=8,
            output=self.root / "report.json",
        )


class FormalBaselineHotspotEvaluatorTests(unittest.TestCase):
    def test_split_labels_follow_frozen_protocol(self) -> None:
        self.assertEqual(
            expected_split_labels({"protocol_id": "uav-tgs-aaai27-hold8-v2"}),
            {"train", "test"},
        )
        self.assertEqual(expected_split_labels({}), {"train", "guard", "test"})

    def test_happy_path_has_comparable_global_view_block_and_hash_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            report = evaluate_formal_baseline_hotspots(fixture.args())
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["split"], "test")
            self.assertEqual(report["metrics"]["valid_pixels"], 11)
            self.assertEqual(len(report["per_view"]), 2)
            self.assertEqual(len(report["per_block"]), 2)
            self.assertIsNotNone(report["metrics"]["hotspot_auprc_histogram_4096"])
            self.assertEqual(report["metrics"]["hotspot_iou"], 1.0)
            self.assertEqual(report["metrics"]["hotspot_precision"], 1.0)
            self.assertEqual(report["metrics"]["hotspot_recall"], 1.0)
            self.assertAlmostEqual(
                report["metrics"]["hotspot_auprc_histogram_4096"], 1.0, places=14
            )
            self.assertTrue(
                all(row["metrics"]["hotspot_iou"] == 1.0 for row in report["per_block"])
            )
            absolute = []
            for pair_id in ("test1", "test2"):
                with Image.open(fixture.render_root / f"{pair_id}.png") as image:
                    prediction, off_lut, _ = rgb_to_temperature(
                        np.asarray(image.convert("RGB")), 0.0, 100.0
                    )
                target = np.load(fixture.target_paths[pair_id], allow_pickle=False)
                support = np.load(
                    fixture.evaluation_support_root / "bool" / f"{pair_id}.npy",
                    allow_pickle=False,
                )
                absolute.append(np.abs(prediction[support] - target[support]))
                self.assertTrue(np.all(off_lut[support] == 0.0))
            exact_p95 = float(
                np.quantile(np.concatenate(absolute), 0.95, method="higher")
            )
            temperature_error = report["metrics"]["temperature_error"]
            self.assertGreaterEqual(temperature_error["p95_abs_error_c"], exact_p95)
            self.assertLessEqual(
                temperature_error["p95_abs_error_c"] - exact_p95,
                temperature_error["p95_histogram_bin_width_c"] + 1e-6,
            )
            self.assertEqual(report["metrics"]["off_lut_distance_rgb"]["p95"], 0.0)
            for row in [*report["per_view"], *report["per_block"]]:
                self.assertIn("p95_abs_error_c", row["metrics"]["temperature_error"])
                self.assertIn("p95", row["metrics"]["off_lut_distance_rgb"])
            self.assertTrue(report["display_semantics"]["comparable_to_oct_evaluator_v2"])
            self.assertFalse(
                report["selection_boundary"]["test_statistics_used_for_threshold"]
            )
            self.assertIn("hot_mae_c", report["metrics"]["temperature_mae_by_target_hotspot"])
            digest = report["report_payload_sha256"]
            basis = dict(report)
            basis.pop("report_payload_sha256")
            self.assertEqual(digest, sha256_json(basis))
            self.assertEqual(
                report["inputs"]["selected_test_inputs_sha256"],
                sha256_json(report["inputs"]["selected_test_inputs"]),
            )

    def test_train_only_gate_precedes_evaluation_support_and_all_test_io(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            threshold = json.loads(fixture.threshold_manifest.read_text(encoding="utf-8"))
            threshold["test_statistics_used"] = True
            threshold.pop("threshold_sha256")
            threshold["threshold_sha256"] = sha256_json(threshold)
            _write_json(fixture.threshold_manifest, threshold)
            args = fixture.args()
            args.temperature_root = Path(temporary) / "does-not-exist"
            original_read_text = Path.read_text
            original_is_file = Path.is_file
            evaluation_manifest = fixture.evaluation_support_manifest.resolve()

            def guarded_read_text(path: Path, *call_args, **call_kwargs):
                if path.resolve() == evaluation_manifest:
                    raise AssertionError("evaluation-support manifest was read before threshold gate")
                return original_read_text(path, *call_args, **call_kwargs)

            def guarded_is_file(path: Path):
                if path.resolve() == evaluation_manifest:
                    raise AssertionError("evaluation-support manifest was stat'ed before threshold gate")
                return original_is_file(path)

            module = "tools.evaluate_formal_baseline_hotspots"
            with (
                patch.object(Path, "read_text", guarded_read_text),
                patch.object(Path, "is_file", guarded_is_file),
                patch(f"{module}._index_pngs", side_effect=AssertionError("render I/O")),
                patch(f"{module}._load_float_temperature", side_effect=AssertionError("target I/O")),
                patch(f"{module}._load_bool_support", side_effect=AssertionError("support I/O")),
                self.assertRaisesRegex(ValueError, "not train-only"),
            ):
                evaluate_formal_baseline_hotspots(args)

    def test_threshold_self_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            threshold = json.loads(fixture.threshold_manifest.read_text(encoding="utf-8"))
            threshold["threshold_c"] = 60.0
            _write_json(fixture.threshold_manifest, threshold)
            with self.assertRaisesRegex(ValueError, "manifest/hash mismatch"):
                evaluate_formal_baseline_hotspots(fixture.args())

    def test_render_shape_mismatch_is_rejected_without_resize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            Image.new("RGB", (1, 1), (0, 0, 0)).save(fixture.render_root / "test1.png")
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                evaluate_formal_baseline_hotspots(fixture.args())

    def test_atomic_report_refuses_overwrite_and_preserves_logical_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            report = evaluate_formal_baseline_hotspots(fixture.args())
            output = write_atomic_report(fixture.args().output, report)
            on_disk = json.loads(output.read_text(encoding="utf-8"))
            digest = on_disk.pop("report_payload_sha256")
            self.assertEqual(digest, sha256_json(on_disk))
            with self.assertRaises(FileExistsError):
                write_atomic_report(output, report)

    def test_threshold_q_bins_and_train_membership_tamper_are_rejected(self) -> None:
        cases = {
            "quantile": 0.90,
            "histogram_bins": 4096,
            "train_view_ids_sha256": "0" * 64,
        }
        for key, value in cases.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                threshold = json.loads(fixture.threshold_manifest.read_text(encoding="utf-8"))
                threshold[key] = value
                threshold.pop("threshold_sha256")
                threshold["threshold_sha256"] = sha256_json(threshold)
                _write_json(fixture.threshold_manifest, threshold)
                with self.assertRaises(ValueError):
                    evaluate_formal_baseline_hotspots(fixture.args())

    def test_manifest_binding_tamper_is_rejected(self) -> None:
        def mutate_formal(fixture: _Fixture) -> None:
            payload = json.loads(fixture.formal_protocol_manifest.read_text())
            payload["formal_binding"]["scene_name"] = "BadScene"
            _write_json(fixture.formal_protocol_manifest, payload)

        def mutate_decode(fixture: _Fixture) -> None:
            rows = fixture.decode_manifest.read_text().splitlines()
            row = json.loads(rows[0])
            row["output_sha256"] = "0" * 64
            rows[0] = json.dumps(row, sort_keys=True)
            fixture.decode_manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")

        def mutate_json(fixture: _Fixture, attribute: str, *keys: str) -> None:
            path = getattr(fixture, attribute)
            payload = json.loads(path.read_text())
            cursor = payload
            for key in keys[:-1]:
                cursor = cursor[key]
            cursor[keys[-1]] = "0" * 64
            _write_json(path, payload)

        cases = {
            "formal": mutate_formal,
            "decode": mutate_decode,
            "range": lambda fixture: mutate_json(fixture, "range_manifest", "range_hash"),
            "canonical": lambda fixture: mutate_json(
                fixture, "canonical_manifest", "palette", "sha256_uint8_rgb"
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                mutate(fixture)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    evaluate_formal_baseline_hotspots(fixture.args())
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            payload = json.loads(fixture.optimization_support_manifest.read_text())
            payload["files"][0]["valid_support"]["sha256"] = "0" * 64
            _write_json(fixture.optimization_support_manifest, payload)
            with self.assertRaises(ValueError):
                evaluate_formal_baseline_hotspots(fixture.args())

    def test_write_rehashes_target_support_render_and_manifest_inputs(self) -> None:
        mutators = {
            "target": lambda fixture: np.save(
                fixture.target_paths["test1"],
                np.zeros((2, 3), dtype=np.float32),
                allow_pickle=False,
            ),
            "support": lambda fixture: np.save(
                fixture.evaluation_support_root / "bool" / "test1.npy",
                np.zeros((2, 3), dtype=np.bool_),
                allow_pickle=False,
            ),
            "render": lambda fixture: Image.new("RGB", (3, 2), (0, 0, 0)).save(
                fixture.render_root / "test1.png"
            ),
            "canonical_manifest": lambda fixture: fixture.canonical_manifest.write_bytes(
                fixture.canonical_manifest.read_bytes() + b" "
            ),
        }
        for label, mutate in mutators.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                fixture = _Fixture(Path(temporary))
                report = evaluate_formal_baseline_hotspots(fixture.args())
                mutate(fixture)
                with self.assertRaisesRegex(ValueError, "changed before write"):
                    write_atomic_report(fixture.args().output, report)

    def test_traversal_and_bad_report_selfhash_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "unsafe relative"):
                _resolve_under(fixture.temperature_root, "../outside.npy", "test1")
            with self.assertRaisesRegex(ValueError, "unsafe relative"):
                _resolve_under(fixture.temperature_root, str(fixture.target_paths["test1"]), "test1")
            report = evaluate_formal_baseline_hotspots(fixture.args())
            report["method_name"] = "tampered"
            with self.assertRaisesRegex(ValueError, "self-hash mismatch"):
                write_atomic_report(fixture.args().output, report)

    def test_generic_radiometry_binding_matches_legacy_oct_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            legacy = evaluate_formal_baseline_hotspots(fixture.args())
            binding = build_formal_evaluation_binding(
                scene_name=fixture.scene,
                bound_split_path=fixture.bound_split,
                decode_manifest_path=fixture.decode_manifest,
                decode_protocol_path=fixture.decode_protocol,
                range_manifest_path=fixture.range_manifest,
                canonical_manifest_path=fixture.canonical_manifest,
                optimization_support_manifest_path=fixture.optimization_support_manifest,
                evaluation_support_manifest_path=fixture.evaluation_support_manifest,
                temperature_root=fixture.temperature_root,
            )
            binding_path = fixture.root / "formal_radiometry_evaluation_binding.json"
            _write_json(binding_path, binding)
            args = fixture.args()
            args.formal_protocol_manifest = None
            args.formal_radiometry_binding_manifest = binding_path
            generic = evaluate_formal_baseline_hotspots(args)
            self.assertEqual(generic["metrics"], legacy["metrics"])
            self.assertEqual(generic["per_view"], legacy["per_view"])
            self.assertEqual(generic["per_block"], legacy["per_block"])
            self.assertEqual(generic["display_semantics"], legacy["display_semantics"])
            self.assertEqual(
                generic["inputs"]["formal_binding_kind"],
                "generic_radiometry_evaluation",
            )

    def test_generic_radiometry_binding_self_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            protocol = json.loads(fixture.formal_protocol_manifest.read_text())
            core = dict(protocol["formal_binding"])
            core["schema"] = "uav-tgs-formal-radiometry-evaluation-core-v1"
            core.pop("formal_protocol_sha256")
            core["formal_protocol_sha256"] = sha256_json(core)
            binding = {
                "schema": "uav-tgs-formal-radiometry-evaluation-binding-v1",
                "schema_version": 1,
                "status": "complete",
                "purpose": "method-independent formal radiometry evaluation only",
                "formal_binding": core,
            }
            binding["binding_manifest_sha256"] = sha256_json(binding)
            binding["purpose"] = "tampered"
            binding_path = fixture.root / "tampered_binding.json"
            _write_json(binding_path, binding)
            args = fixture.args()
            args.formal_protocol_manifest = None
            args.formal_radiometry_binding_manifest = binding_path
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                evaluate_formal_baseline_hotspots(args)


if __name__ == "__main__":
    unittest.main()
