import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analyze_shared_anchor_blocks import analyze


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _npz(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "npz_file": str(path.name),
        "npz_size_bytes": path.stat().st_size,
        "npz_sha256": _sha(path),
    }


def _make_depth_endpoint(
    root: Path,
    group: str,
    names: list[str],
    reference_manifest: Path,
    model_depth: float,
) -> Path:
    endpoint = root / group
    model_root = endpoint / "probe_bundle"
    views = []
    for index, name in enumerate(names):
        path = model_root / "views" / f"{index:05d}.npz"
        identity = _npz(
            path,
            depth=np.full((2, 2), model_depth, dtype=np.float32),
            opacity=np.ones((2, 2), dtype=np.float32),
        )
        identity["npz_file"] = f"views/{index:05d}.npz"
        views.append({"image_name": name, **identity})
    model_manifest = _write(model_root / "split_manifest.json", {"views": views})
    adapter = _write(
        model_root / "depth_adapter_manifest.json",
        {
            "depth_semantics": "metric_camera_z_from_renderer",
            "validity_rule": {"depth_min": 1e-6, "opacity_threshold": 0.5},
        },
    )
    _write(
        endpoint / "metrics" / "metrics_summary.json",
        {
            "reference_manifest": str(reference_manifest),
            "reference_manifest_sha256": _sha(reference_manifest),
            "model_manifest": str(model_manifest),
            "model_manifest_sha256": _sha(model_manifest),
            "adapter_manifest": str(adapter),
            "adapter_manifest_sha256": _sha(adapter),
        },
    )
    return endpoint


class SharedAnchorBlockAnalysisTests(unittest.TestCase):
    def test_one_complete_block(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = [f"{index:04d}.png" for index in range(1, 17)]
            test_list = root / "test_list.txt"
            test_list.write_text("\n".join(names) + "\n", encoding="utf-8")
            records = [
                {
                    "split": "test",
                    "thermal_camera_name": name,
                    "strip_id": "strip0",
                    "block_index": 0,
                    "block_offset": index,
                    "position_in_strip": index,
                    "pair_id": index,
                    "stratum": "nadir",
                    "hash": str(index),
                }
                for index, name in enumerate(names)
            ]
            bound_split = _write(
                root / "bound_split.json",
                {
                    "records": records,
                    "selected_test_blocks_hash": "a" * 64,
                },
            )
            reference_root = root / "reference"
            reference_views = []
            for index, name in enumerate(names):
                path = reference_root / "views" / f"{index:05d}.npz"
                identity = _npz(
                    path,
                    depth=np.full((2, 2), 10.0, dtype=np.float32),
                    valid_mask=np.ones((2, 2), dtype=np.uint8),
                )
                identity["npz_file"] = f"views/{index:05d}.npz"
                reference_views.append({"image_name": name, **identity})
            reference_manifest = _write(
                reference_root / "reference_depth_manifest.json",
                {"views": reference_views},
            )
            anchor_endpoint = _make_depth_endpoint(
                root, "Anchor", names, reference_manifest, 8.0
            )
            shared_endpoint = _make_depth_endpoint(
                root, "S", names, reference_manifest, 9.5
            )
            appearance = {
                "ours_30000": {
                    "PSNR": {name: 25.0 for name in names},
                    "SSIM": {name: 0.8 for name in names},
                    "LPIPS": {name: 0.2 for name in names},
                }
            }
            anchor_appearance = _write(root / "anchor_per_view.json", appearance)
            shared_appearance = _write(root / "shared_per_view.json", appearance)
            payload = analyze(
                argparse.Namespace(
                    scene="Synthetic",
                    anchor_iteration=30000,
                    test_list=str(test_list),
                    bound_split=str(bound_split),
                    appearance=[
                        f"Anchor={anchor_appearance}",
                        f"S={shared_appearance}",
                    ],
                    depth_endpoint=[
                        f"Anchor={anchor_endpoint}",
                        f"S={shared_endpoint}",
                    ],
                )
            )
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["blocks"]), 1)
            block = payload["blocks"][0]
            self.assertEqual(block["block"]["size"], 16)
            self.assertGreater(
                block["groups"]["Anchor"]["depth_pixel_micro"]["thresholds"]["1"]["front"],
                block["groups"]["S"]["depth_pixel_micro"]["thresholds"]["1"]["front"],
            )
            self.assertLess(
                block["groups"]["S"]["depth_pixel_micro"]["abs_depth_error_mean"],
                block["groups"]["Anchor"]["depth_pixel_micro"]["abs_depth_error_mean"],
            )
            json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
