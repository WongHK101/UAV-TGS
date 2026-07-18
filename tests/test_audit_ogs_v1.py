from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools import audit_ogs_v1


class OgsV1AuditSidecarTests(unittest.TestCase):
    def test_camera_audit_payload_is_defined_and_hash_bound(self) -> None:
        camera = SimpleNamespace(
            image_name="0001.jpg",
            uid=7,
            colmap_id=11,
            R=np.eye(3, dtype=np.float64),
            T=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            FoVx=0.8,
            FoVy=0.6,
            image_width=1259,
            image_height=1007,
        )
        manifest = audit_ogs_v1.build_ordered_camera_manifest(
            "InternalRoad", [camera.image_name], [camera]
        )
        payload = manifest["cameras"]
        self.assertEqual(payload[0]["image_name"], "0001.jpg")
        self.assertEqual(payload[0]["image_width"], 1259)
        self.assertEqual(
            manifest["camera_parameters_sha256"],
            audit_ogs_v1.canonical_json_sha256(payload),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ordered_camera_names.json"
            audit_ogs_v1._write_json(path, manifest)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), manifest)

    def test_parse_established_clamp_sidecar_and_fail_closed_checks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clamp.json"
            payload = {
                "gaussian_count": 10,
                "selected_count": 2,
                "records": [
                    {"gaussian_index": 7},
                    {"gaussian_index": 2},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            indices, loaded = audit_ogs_v1.parse_clamp_indices(
                path, gaussian_count=10, expected_count=2
            )
            np.testing.assert_array_equal(indices, np.asarray([2, 7]))
            self.assertEqual(loaded, payload)

            payload["records"][1]["gaussian_index"] = 7
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                audit_ogs_v1.OgsAuditError, "duplicate"
            ):
                audit_ogs_v1.parse_clamp_indices(
                    path, gaussian_count=10, expected_count=2
                )

    def test_scale_matching_is_deterministic_and_does_not_reuse_when_sufficient(self) -> None:
        raw = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [0.1, 1.1, 2.1],
                [0.2, 1.2, 2.2],
                [0.3, 1.3, 2.3],
                [0.4, 1.4, 2.4],
                [0.5, 1.5, 2.5],
                [0.6, 1.6, 2.6],
                [0.7, 1.7, 2.7],
            ],
            dtype=np.float64,
        )
        eligible = np.ones(8, dtype=bool)
        clamp = np.asarray([0, 7], dtype=np.int64)
        first = audit_ogs_v1.match_scale_controls(
            raw, eligible, clamp, controls_per_target=3
        )
        second = audit_ogs_v1.match_scale_controls(
            raw, eligible, clamp, controls_per_target=3
        )
        self.assertEqual(first, second)
        self.assertFalse(first["reuse_required"])
        self.assertEqual(first["reuse_count"], 0)
        selected = [row["control_index"] for row in first["rows"]]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected).isdisjoint(set(clamp.tolist())))

    def test_scale_matching_records_reuse_only_when_pool_is_insufficient(self) -> None:
        raw = np.arange(15, dtype=np.float64).reshape(5, 3)
        eligible = np.ones(5, dtype=bool)
        result = audit_ogs_v1.match_scale_controls(
            raw,
            eligible,
            np.asarray([0, 1], dtype=np.int64),
            controls_per_target=2,
        )
        self.assertTrue(result["reuse_required"])
        self.assertGreater(result["reuse_count"], 0)
        self.assertEqual(len(result["rows"]), 4)

    def test_complete_non_support_requires_all_three_conditions(self) -> None:
        passed = audit_ogs_v1.evaluate_complete_non_support(
            eligible_active_clamp_count=4,
            clamp_risk_percentiles=np.asarray([10.0, 40.0]),
            clamp_risk=np.asarray([1.0, 2.0]),
            control_risk=np.asarray([2.0, 3.0]),
        )
        self.assertTrue(passed["complete_non_support"])
        for key in passed["conditions"]:
            self.assertTrue(passed["conditions"][key])

        not_passed = audit_ogs_v1.evaluate_complete_non_support(
            eligible_active_clamp_count=5,
            clamp_risk_percentiles=np.asarray([10.0, 40.0]),
            clamp_risk=np.asarray([1.0, 2.0]),
            control_risk=np.asarray([2.0, 3.0]),
        )
        self.assertFalse(not_passed["complete_non_support"])
        self.assertFalse(
            not_passed["conditions"]["eligible_and_active_clamp_count_lt_5"]
        )

    def test_binding_manifest_must_match_scene_and_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "binding.json"
            payload = {
                "status": "passed",
                "scene": "InternalRoad",
                "binding_hash": "a" * 64,
                "scene_split_hash": "b" * 64,
                "counts": {"train": 2},
                "files": [
                    {"camera_name": "0001.jpg", "split": "train"},
                    {"camera_name": "0002.jpg", "split": "train"},
                    {"camera_name": "0003.jpg", "split": "test"},
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, hashes = audit_ogs_v1.load_binding_manifest(
                path,
                scene_name="InternalRoad",
                train_names=["0002.jpg", "0001.jpg"],
            )
            self.assertEqual(loaded, payload)
            self.assertEqual(
                hashes,
                {"binding_hash": "a" * 64, "scene_split_hash": "b" * 64},
            )
            with self.assertRaisesRegex(
                audit_ogs_v1.OgsAuditError, "binding/train-list mismatch"
            ):
                audit_ogs_v1.load_binding_manifest(
                    path,
                    scene_name="InternalRoad",
                    train_names=["0001.jpg"],
                )

    def test_risk_percentiles_and_ordered_name_hash_are_stable(self) -> None:
        from utils.camera_sequence import ordered_camera_hash

        percentiles = audit_ogs_v1.risk_percentiles(
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.5, 3.0]),
        )
        np.testing.assert_allclose(percentiles, [25.0, 50.0, 100.0])
        a = audit_ogs_v1.ordered_names_sha256(["a.jpg", "b.jpg"])
        b = audit_ogs_v1.ordered_names_sha256(["b.jpg", "a.jpg"])
        self.assertNotEqual(a, b)
        self.assertEqual(a, audit_ogs_v1.ordered_names_sha256(["a.jpg", "b.jpg"]))
        self.assertEqual(a, ordered_camera_hash(["a.jpg", "b.jpg"]))


if __name__ == "__main__":
    unittest.main()
