from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tools.thermal_radiometry import bind_formal_scene as binding


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class FormalSceneBindingTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Any]:
        scene = "Fixture"
        split_hash = _token("split")
        rule_hash = _token("rule")
        collection_hash = _token("collection")
        collection_split_hash = _token("collection-split")
        formal_rule_hash = _token("formal-rule")
        adapter_sha = _token("dji-irp-executable")
        raw_root = root / "raw_thermal"
        temperature_root = root / "temperature_c"
        request_root = root / "manifests" / "decode_requests"
        (temperature_root / scene).mkdir(parents=True)
        raw_root.mkdir(parents=True)

        split_rows: list[dict[str, Any]] = []
        decode_rows: list[dict[str, Any]] = []
        protocol_rows: list[dict[str, Any]] = []
        labels = ("train", "test", "guard")
        for index, split in enumerate(labels, 1):
            pair_id = f"{index:04d}"
            source_record_hash = _token(f"source-record-{pair_id}")
            raw_path = raw_root / f"{pair_id}.JPG"
            raw_path.write_bytes((f"fixture-rjpeg-{pair_id}" * 11).encode("ascii"))
            npy_path = temperature_root / scene / f"{pair_id}.npy"
            np.save(npy_path, np.full((1024, 1280), index, dtype=np.float32))

            parameters = {
                "distance_m": {"value": float(10 + index), "source": "strip_valid_lrf_robust_median"},
                "humidity_percent": {"value": 70.0, "source": "benchmark_assumption"},
                "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
                "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
                "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
            }
            metadata = {
                "radiometry_protocol": {
                    "strip_id": "strip-0",
                    "used_distance_m": float(10 + index),
                    "used_distance_source": "strip_valid_lrf_robust_median",
                }
            }
            protocol = {
                "schema_version": binding.FORMAL_PROTOCOL_SCHEMA,
                "scene": scene,
                "frame_id": pair_id,
                "pair_id": pair_id,
                "source_path": str(raw_path.resolve()),
                "strip_id": "strip-0",
                "decode_parameters": parameters,
                "raw_lrf_distance_m": float(10 + index),
                "raw_lrf_status": "Normal",
                "raw_lrf_valid": True,
                "used_distance_m": float(10 + index),
                "used_distance_source": "strip_valid_lrf_robust_median",
                "distance_fallback_reason": "none",
                "source_audit_record_hash": source_record_hash,
                "source_manifest": str(root / "audit.jsonl"),
                "source_manifest_sha256": _token("audit-manifest"),
                "metadata": metadata,
            }
            protocol["protocol_record_hash"] = _json_hash(protocol)
            protocol_rows.append(protocol)

            output_path = str(npy_path.resolve())
            request = {
                "schema_version": binding.FORMAL_DECODE_SCHEMA,
                "scene": scene,
                "frame_id": pair_id,
                "pair_id": pair_id,
                "source_path": str(raw_path.resolve()),
                "output_path": output_path,
                "temperature_npy": output_path,
                "tsdk_root": str(root / "tsdk"),
                "adapter": binding.FORMAL_ADAPTER,
                "parameters": parameters,
                "strip_id": "strip-0",
                "metadata": metadata,
            }
            request_path = request_root / f"{scene}--{pair_id}.json"
            _write_json(request_path, request)
            decode = dict(request)
            decode.update(
                {
                    "request_path": str(request_path.resolve()),
                    "success": True,
                    "dtype": "float32",
                    "shape_hw": [1024, 1280],
                    "source_size_bytes": raw_path.stat().st_size,
                    "source_sha256": _sha(raw_path),
                    "output_sha256": _sha(npy_path),
                    "adapter_diagnostics": {
                        "backend": binding.FORMAL_ADAPTER_BACKEND,
                        "executable_sha256": adapter_sha,
                        "parameters_applied": {
                            name: float(item["value"]) for name, item in parameters.items()
                        },
                        "resolution": {"width": 1280, "height": 1024},
                        "dirp_api_version": "0x14",
                        "rjpeg_version": "0x300",
                    },
                }
            )
            decode_rows.append(decode)
            split_rows.append(
                {
                    "scene": scene,
                    "pair_id": pair_id,
                    "split": split,
                    "strip_id": "strip-0",
                    "source_record_hash": source_record_hash,
                }
            )

        protocol_basis = [
            {field: row[field] for field in binding.FORMAL_PROTOCOL_BASIS_FIELDS}
            for row in protocol_rows
        ]
        protocol_hash = _json_hash(protocol_basis)
        for row in protocol_rows:
            row["protocol_hash"] = protocol_hash

        scene_payload = {
            "scene": scene,
            "schema_name": "fixture",
            "schema_version": 1,
            "split_hash": split_hash,
            "rule_hash": rule_hash,
            "counts": {"total": 3, "train": 1, "test": 1, "guard": 1},
            "validation": {"status": "passed"},
            "records": split_rows,
        }
        scene_path = root / "scene.json"
        _write_json(scene_path, scene_payload)
        collection_path = root / "collection.json"
        _write_json(
            collection_path,
            {
                "collection_hash": collection_hash,
                "collection_split_hash": collection_split_hash,
                "formal_rule_hash": formal_rule_hash,
                "validation": {"status": "passed"},
                "scenes": [
                    {
                        "scene": scene,
                        "manifest_sha256": _sha(scene_path),
                        "split_hash": split_hash,
                        "rule_hash": rule_hash,
                        "counts": {"total": 3, "train": 1, "test": 1, "guard": 1},
                    }
                ],
            },
        )
        decode_path = root / "manifests" / "decode_manifest.jsonl"
        protocol_path = root / "manifests" / "decode_protocol_used_v1.jsonl"
        _write_jsonl(decode_path, decode_rows)
        _write_jsonl(protocol_path, protocol_rows)
        return {
            "scene_path": scene_path,
            "collection_path": collection_path,
            "decode_path": decode_path,
            "protocol_path": protocol_path,
            "temperature_root": temperature_root,
            "raw_root": raw_root,
            "adapter_sha": adapter_sha,
            "protocol_hash": protocol_hash,
            "expected": {
                "expected_collection_manifest_sha256": _sha(collection_path),
                "expected_collection_hash": collection_hash,
                "expected_collection_split_hash": collection_split_hash,
                "expected_formal_rule_hash": formal_rule_hash,
                "expected_scene_manifest_sha256": _sha(scene_path),
                "expected_scene_split_hash": split_hash,
                "expected_scene_rule_hash": rule_hash,
                "expected_decode_protocol_hash": protocol_hash,
                "expected_adapter_executable_sha256": adapter_sha,
            },
        }

    def _upgrade_fixture_to_robust_lrf(
        self, fixture: dict[str, Any]
    ) -> list[dict[str, Any]]:
        protocol_rows = [
            json.loads(line)
            for line in fixture["protocol_path"].read_text(encoding="utf-8").splitlines()
        ]
        decode_rows = [
            json.loads(line)
            for line in fixture["decode_path"].read_text(encoding="utf-8").splitlines()
        ]
        for index, (protocol, decode) in enumerate(zip(protocol_rows, decode_rows, strict=True)):
            robust_inlier = index != 1
            robust_outlier = not robust_inlier
            protocol["raw_lrf_robust_inlier"] = robust_inlier
            protocol["raw_lrf_robust_outlier"] = robust_outlier
            protocol_metadata = protocol["metadata"]["radiometry_protocol"]
            protocol_metadata["raw_lrf_robust_inlier"] = robust_inlier
            protocol_metadata["raw_lrf_robust_outlier"] = robust_outlier
            decode["metadata"] = protocol["metadata"]
            request_path = (
                fixture["decode_path"].parent
                / "decode_requests"
                / f"Fixture--{protocol['pair_id']}.json"
            )
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["metadata"] = protocol["metadata"]
            _write_json(request_path, request)

            protocol.pop("protocol_hash", None)
            protocol.pop("protocol_record_hash", None)
            protocol["protocol_record_hash"] = _json_hash(protocol)

        protocol_basis = [
            {field: row[field] for field in binding.ROBUST_LRF_PROTOCOL_BASIS_FIELDS}
            for row in protocol_rows
        ]
        protocol_hash = _json_hash(protocol_basis)
        for row in protocol_rows:
            row["protocol_hash"] = protocol_hash
        _write_jsonl(fixture["protocol_path"], protocol_rows)
        _write_jsonl(fixture["decode_path"], decode_rows)
        fixture["protocol_hash"] = protocol_hash
        fixture["expected"]["expected_decode_protocol_hash"] = protocol_hash
        return protocol_rows

    @staticmethod
    def _bind(
        fixture: dict[str, Any],
        *,
        decode_protocol_basis: str = binding.LEGACY_PROTOCOL_BASIS,
    ):
        return binding.bind_formal_scene(
            scene_manifest_path=fixture["scene_path"],
            collection_manifest_path=fixture["collection_path"],
            decode_manifest_path=fixture["decode_path"],
            decode_protocol_path=fixture["protocol_path"],
            temperature_root=fixture["temperature_root"],
            raw_thermal_root=fixture["raw_root"],
            scene="Fixture",
            sfm_image_scope="shared_sfm_all_images",
            decode_protocol_basis=decode_protocol_basis,
            **fixture["expected"],
        )

    def test_binds_exact_split_and_full_decode_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            bound, lists, manifest = self._bind(fixture)
            self.assertEqual(
                lists,
                {
                    "train": ["0001.JPG"],
                    "test": ["0002.JPG"],
                    "guard": ["0003.JPG"],
                    "thermal_train": ["0001.png"],
                    "thermal_test": ["0002.png"],
                    "thermal_guard": ["0003.png"],
                },
            )
            self.assertEqual(bound["records"][0]["temperature_npy"], "Fixture/0001.npy")
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["decode_protocol_hash"], fixture["protocol_hash"])
            self.assertEqual(
                manifest["decode_protocol_basis"], binding.LEGACY_PROTOCOL_BASIS
            )
            self.assertEqual(
                bound["decode_binding"]["protocol_basis_variant"],
                binding.LEGACY_PROTOCOL_BASIS,
            )
            self.assertEqual(manifest["adapter_executable_sha256"], fixture["adapter_sha"])
            self.assertEqual(
                manifest["counts"], {"total": 3, "train": 1, "test": 1, "guard": 1}
            )
            self.assertEqual(manifest["files"][0]["raw_thermal"]["raw_relative_path"], "0001.JPG")

    def test_tampered_scene_manifest_fails_external_sha_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            payload = json.loads(fixture["scene_path"].read_text(encoding="utf-8"))
            payload["records"][0]["split"] = "guard"
            _write_json(fixture["scene_path"], payload)
            with self.assertRaisesRegex(ValueError, "scene manifest SHA-256 mismatch"):
                self._bind(fixture)

    def test_later_regenerated_protocol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in fixture["protocol_path"].read_text(encoding="utf-8").splitlines()]
            rows[0]["raw_lrf_robust_inlier"] = True
            rows[0]["raw_lrf_robust_outlier"] = False
            _write_jsonl(fixture["protocol_path"], rows)
            with self.assertRaisesRegex(ValueError, "not the frozen decode_protocol_used_v1 schema"):
                self._bind(fixture)

    def test_explicit_robust_lrf_protocol_binds_full_decode_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            self._upgrade_fixture_to_robust_lrf(fixture)
            bound, _, manifest = self._bind(
                fixture, decode_protocol_basis=binding.ROBUST_LRF_PROTOCOL_BASIS
            )
            self.assertEqual(
                manifest["decode_protocol_basis"], binding.ROBUST_LRF_PROTOCOL_BASIS
            )
            self.assertEqual(
                bound["decode_binding"]["protocol_basis_variant"],
                binding.ROBUST_LRF_PROTOCOL_BASIS,
            )

    def test_explicit_robust_lrf_protocol_rejects_mixed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = self._upgrade_fixture_to_robust_lrf(fixture)
            rows[-1].pop("raw_lrf_robust_inlier")
            _write_jsonl(fixture["protocol_path"], rows)
            with self.assertRaisesRegex(ValueError, "must contain both"):
                self._bind(
                    fixture, decode_protocol_basis=binding.ROBUST_LRF_PROTOCOL_BASIS
                )

    def test_explicit_robust_lrf_protocol_rejects_inconsistent_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = self._upgrade_fixture_to_robust_lrf(fixture)
            rows[0]["raw_lrf_robust_outlier"] = True
            _write_jsonl(fixture["protocol_path"], rows)
            with self.assertRaisesRegex(ValueError, "not complementary"):
                self._bind(
                    fixture, decode_protocol_basis=binding.ROBUST_LRF_PROTOCOL_BASIS
                )

    def test_explicit_robust_lrf_protocol_rejects_nested_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = self._upgrade_fixture_to_robust_lrf(fixture)
            rows[0]["metadata"]["radiometry_protocol"]["raw_lrf_robust_inlier"] = False
            _write_jsonl(fixture["protocol_path"], rows)
            with self.assertRaisesRegex(ValueError, "nested raw_lrf_robust_inlier mismatch"):
                self._bind(
                    fixture, decode_protocol_basis=binding.ROBUST_LRF_PROTOCOL_BASIS
                )

    def test_protocol_record_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in fixture["protocol_path"].read_text(encoding="utf-8").splitlines()]
            rows[0]["metadata"]["radiometry_protocol"]["used_distance_m"] = 999.0
            _write_jsonl(fixture["protocol_path"], rows)
            with self.assertRaisesRegex(ValueError, "protocol_record_hash mismatch"):
                self._bind(fixture)

    def test_decode_parameter_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in fixture["decode_path"].read_text(encoding="utf-8").splitlines()]
            rows[0]["parameters"]["distance_m"]["value"] = 999.0
            _write_jsonl(fixture["decode_path"], rows)
            with self.assertRaisesRegex(ValueError, "decode parameters"):
                self._bind(fixture)

    def test_adapter_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            rows = [json.loads(line) for line in fixture["decode_path"].read_text(encoding="utf-8").splitlines()]
            rows[0]["adapter"] = "custom:unverified"
            _write_jsonl(fixture["decode_path"], rows)
            with self.assertRaisesRegex(ValueError, "decode adapter"):
                self._bind(fixture)

    def test_raw_rjpeg_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            (fixture["raw_root"] / "0001.JPG").write_bytes(b"tampered-rjpeg")
            with self.assertRaisesRegex(ValueError, "raw R-JPEG SHA-256"):
                self._bind(fixture)

    def test_temperature_npy_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture["temperature_root"] / "Fixture" / "0001.npy"
            values = np.load(path, allow_pickle=False)
            values[0, 0] += 1.0
            np.save(path, values, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "temperature file SHA mismatch"):
                self._bind(fixture)


if __name__ == "__main__":
    unittest.main()
