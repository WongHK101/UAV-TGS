from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.aaai27_hold8.bind_expected_depth_bundle import bind_bundle
from tools.hold8_expected_depth_evaluator import evaluate_manifests


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Hold8ExpectedDepthBundleBindingTests(unittest.TestCase):
    def test_reference_and_model_bind_to_authoritative_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "scene.json"
            records = []
            for index in range(9):
                pair = f"p{index:02d}"
                records.append(
                    {
                        "pair_id": pair,
                        "zero_based_sorted_index": index,
                        "split": "test" if index % 8 == 0 else "train",
                        "camera_name": f"{pair}.JPG",
                        "thermal_camera_name": f"{pair}.png",
                    }
                )
            test_ids = ["p00", "p08"]
            test_sha = hashlib.sha256("p00\np08\n".encode()).hexdigest()
            split_payload = {
                "protocol_id": "uav-tgs-aaai27-hold8-v2",
                "scene": "Tiny",
                "collection_hash": "1" * 64,
                "split_hash": "2" * 64,
                "hashes": {"test_list_sha256": test_sha},
                "records": records,
            }
            _write(split, split_payload)
            collection = root / "collection.json"
            collection_payload = {
                "protocol_id": "uav-tgs-aaai27-hold8-v2",
                "collection_hash": "1" * 64,
                "collection_split_hash": "3" * 64,
                "scenes": [
                    {
                        "scene": "Tiny",
                        "manifest_sha256": _sha(split),
                        "split_hash": "2" * 64,
                    }
                ],
            }
            _write(collection, collection_payload)

            reference_source = root / "reference_source"
            model_source = root / "model_source"
            reference_source.mkdir(); model_source.mkdir()
            reference_views = []; model_views = []
            for index, pair in enumerate(test_ids):
                ref = reference_source / f"{index}.npz"
                model = model_source / f"{index}.npz"
                depth = np.full((2, 3), float(index + 2), dtype=np.float32)
                np.savez_compressed(ref, depth=depth, valid_mask=np.ones_like(depth, dtype=np.uint8))
                np.savez_compressed(
                    model,
                    depth_expected_alpha_normalized=depth + 0.25,
                    accumulated_opacity=np.ones_like(depth, dtype=np.float32),
                )
                reference_views.append(
                    {"image_name": f"{pair}.JPG", "npz_file": ref.name,
                     "npz_size_bytes": ref.stat().st_size, "npz_sha256": _sha(ref)}
                )
                model_views.append(
                    {"image_name": f"{pair}.png", "npz_file": model.name,
                     "npz_size_bytes": model.stat().st_size, "npz_sha256": _sha(model)}
                )
            ref_source_manifest = reference_source / "manifest.json"
            model_source_manifest = model_source / "manifest.json"
            _write(ref_source_manifest, {"views": reference_views})
            _write(model_source_manifest, {"views": model_views})
            ref_bound = bind_bundle(
                kind="reference", source_manifest=ref_source_manifest,
                collection_manifest=collection, scene_split_manifest=split,
                output_root=root / "reference_bound",
            )
            model_bound = bind_bundle(
                kind="model", source_manifest=model_source_manifest,
                collection_manifest=collection, scene_split_manifest=split,
                output_root=root / "model_bound", method_name="raw_f3",
            )
            result = evaluate_manifests(
                ref_bound, model_bound, collection, split,
                expected_collection_manifest_sha256=_sha(collection),
                expected_scene_split_manifest_sha256=_sha(split),
            )
            self.assertEqual(result["scene_name"], "Tiny")
            self.assertEqual(result["method_name"], "raw_f3")
            self.assertEqual(result["metrics"]["missing_rate"], 0.0)
            self.assertAlmostEqual(
                result["metrics"]["median_absolute_depth_error_m"], 0.25
            )


if __name__ == "__main__":
    unittest.main()
