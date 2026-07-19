from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.thermal_radiometry import combine_formal_support


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_hash(name: str) -> str:
    return hashlib.sha256(f"split:{name}".encode("ascii")).hexdigest()


class FormalSupportTests(unittest.TestCase):
    def _fixture(self, root: Path, *, proxy_shape=(2, 3)) -> dict:
        names = ["0001", "0002", "0003"]
        valid_root = root / "valid_support"
        opacity_root = root / "opacity_proxy"
        valid_root.mkdir(parents=True)
        opacity_root.mkdir(parents=True)

        split = {
            "schema_name": "fixture-split",
            "split_hash": hashlib.sha256(b"fixture-split").hexdigest(),
            "counts": {"total": 3, "train": 0, "test": 3, "guard": 0},
            "records": [
                {
                    "pair_id": name,
                    "split": "test",
                    "hash": _record_hash(name),
                }
                for name in names
            ],
        }
        split_path = root / "split.json"
        split_path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")

        valid_records = []
        opacity_entries = []
        for name in names:
            valid = np.array(
                [[True, True, False], [True, False, True]],
                dtype=np.bool_,
            )
            opacity = np.array(
                [[0.0, 0.01, 1.0], [0.011, 0.5, 0.0]],
                dtype=np.float32,
            )
            if proxy_shape != (2, 3):
                opacity = np.ones(proxy_shape, dtype=np.float32)
            valid_path = valid_root / f"{name}.npy"
            opacity_path = opacity_root / f"{name}.npy"
            np.save(valid_path, valid, allow_pickle=False)
            np.save(opacity_path, opacity, allow_pickle=False)
            valid_records.append(
                {
                    "image_name": f"{name}.JPG",
                    "valid_support": {
                        "relative_path": f"valid_support/{name}.npy",
                        "sha256": _sha256(valid_path),
                        "dtype": "bool",
                        "shape": list(valid.shape),
                    },
                }
            )
            opacity_entries.append(
                {
                    "split": "test",
                    "iteration": 30000,
                    "source_image_name": f"{name}.JPG",
                    "output": {
                        "render": f"renders/{name}.png",
                        "ground_truth": f"gt/{name}.png",
                        "renderer_support": None,
                        "opacity_proxy": f"opacity_proxy/{name}.npy",
                    },
                    "output_sha256": {
                        "render": "0" * 64,
                        "ground_truth": "1" * 64,
                        "renderer_support": None,
                        "opacity_proxy": _sha256(opacity_path),
                    },
                    "opacity_proxy_semantics": (
                        combine_formal_support.OPACITY_PROXY_SEMANTICS
                    ),
                    "support_threshold_applied": False,
                    "support_threshold": None,
                }
            )

        valid_manifest = {
            "schema": "fixture-undistorted-temperature",
            "files": valid_records,
        }
        valid_manifest_path = root / "valid_manifest.json"
        valid_manifest_path.write_text(
            json.dumps(valid_manifest, sort_keys=True),
            encoding="utf-8",
        )
        opacity_manifest = {
            "schema_name": "uav-tgs-render-output-mapping",
            "schema_version": 1,
            "split": "test",
            "iteration": 30000,
            "opacity_proxy_saved": True,
            "opacity_proxy_semantics": combine_formal_support.OPACITY_PROXY_SEMANTICS,
            "support_threshold_applied": False,
            "support_threshold": None,
            "entries": opacity_entries,
        }
        opacity_manifest_path = root / "opacity_manifest.json"
        opacity_manifest_path.write_text(
            json.dumps(opacity_manifest, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "names": names,
            "split_manifest": split_path,
            "valid_support_root": valid_root,
            "valid_support_manifest": valid_manifest_path,
            "opacity_proxy_root": opacity_root,
            "opacity_proxy_manifest": opacity_manifest_path,
        }

    def _combine(self, fixture: dict, output_root: Path) -> dict:
        return combine_formal_support.combine_formal_support(
            split_manifest=fixture["split_manifest"],
            valid_support_root=fixture["valid_support_root"],
            valid_support_manifest=fixture["valid_support_manifest"],
            opacity_proxy_root=fixture["opacity_proxy_root"],
            opacity_proxy_manifest=fixture["opacity_proxy_manifest"],
            output_root=output_root,
            opacity_threshold=0.01,
            expected_test_count=3,
        )

    def test_intersection_outputs_and_portable_manifest_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root / "sources")
            first_root = root / "first"
            second_root = root / "second"
            first = self._combine(fixture, first_root)
            second = self._combine(fixture, second_root)

            first_manifest_bytes = (first_root / "manifest.json").read_bytes()
            second_manifest_bytes = (second_root / "manifest.json").read_bytes()
            self.assertEqual(first_manifest_bytes, second_manifest_bytes)
            self.assertEqual(first["portable_content_sha256"], second["portable_content_sha256"])
            self.assertNotIn(str(root).encode("utf-8"), first_manifest_bytes)

            combined_bool = np.load(first_root / "bool" / "0001.npy", allow_pickle=False)
            combined_float = np.load(first_root / "float" / "0001.npy", allow_pickle=False)
            expected = np.array(
                [[False, False, False], [True, False, False]],
                dtype=np.bool_,
            )
            np.testing.assert_array_equal(combined_bool, expected)
            np.testing.assert_array_equal(combined_float, expected.astype(np.float32))
            self.assertEqual(combined_bool.dtype, np.bool_)
            self.assertEqual(combined_float.dtype, np.float32)

            manifest = json.loads(first_manifest_bytes)
            self.assertEqual(manifest["summary"]["file_count"], 3)
            self.assertEqual(manifest["policy"]["opacity_threshold"], 0.01)
            self.assertEqual(
                manifest["policy"]["expression"],
                "valid_support AND (opacity_proxy > opacity_threshold)",
            )
            manifest_sha = _sha256(first_root / "manifest.json")
            self.assertEqual(first["manifest_sha256"], manifest_sha)
            self.assertEqual(
                (first_root / "manifest.sha256").read_text(encoding="ascii"),
                f"{manifest_sha}  manifest.json\n",
            )

    def test_source_hash_mismatch_fails_before_output_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root / "sources")
            tampered = fixture["opacity_proxy_root"] / "0002.npy"
            np.save(tampered, np.full((2, 3), 0.5, dtype=np.float32), allow_pickle=False)
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                self._combine(fixture, output)
            self.assertFalse(output.exists())

    def test_hold8_source_record_sha_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root / "sources")
            split = json.loads(fixture["split_manifest"].read_text(encoding="utf-8"))
            for record in split["records"]:
                record["source_record_sha256"] = record.pop("hash")
            fixture["split_manifest"].write_text(
                json.dumps(split, sort_keys=True), encoding="utf-8"
            )
            self._combine(fixture, root / "output")
            manifest = json.loads(
                (root / "output" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["summary"]["file_count"], 3)

    def test_exact_test_names_and_shapes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._fixture(root / "name_sources")
            np.save(
                fixture["opacity_proxy_root"] / "extra.npy",
                np.ones((2, 3), dtype=np.float32),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "opacity-proxy files differ"):
                self._combine(fixture, root / "name_output")

            shape_fixture = self._fixture(root / "shape_sources", proxy_shape=(2, 2))
            with self.assertRaisesRegex(ValueError, "support shape mismatch"):
                self._combine(shape_fixture, root / "shape_output")


if __name__ == "__main__":
    unittest.main()
