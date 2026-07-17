import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn

from tools.build_shared_clamped_anchor import (
    SharedAnchorError,
    build_shared_anchor,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_simple_knn_stub():
    package = types.ModuleType("simple_knn")
    extension = types.ModuleType("simple_knn._C")
    extension.distCUDA2 = lambda value: torch.zeros(
        value.shape[0], device=value.device, dtype=value.dtype
    )
    package._C = extension
    sys.modules.setdefault("simple_knn", package)
    sys.modules.setdefault("simple_knn._C", extension)


def _write_ply(path: Path, params):
    xyz = params[1].detach().numpy()
    normal = np.zeros_like(xyz)
    dc = (
        params[2]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .numpy()
    )
    rest = (
        params[3]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .numpy()
    )
    opacity = params[6].detach().numpy()
    scaling = params[4].detach().numpy()
    rotation = params[5].detach().numpy()
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(dc.shape[1])]
    names += [f"f_rest_{index}" for index in range(rest.shape[1])]
    names += ["opacity"]
    names += [f"scale_{index}" for index in range(3)]
    names += [f"rot_{index}" for index in range(4)]
    values = np.concatenate(
        [xyz, normal, dc, rest, opacity, scaling, rotation], axis=1
    )
    dtype = [(name, "f4") for name in names]
    records = np.empty(xyz.shape[0], dtype=dtype)
    records[:] = list(map(tuple, values))
    path.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(records, "vertex")]).write(path)


def _synthetic_model(root: Path) -> tuple[Path, str, str]:
    model = root / "input"
    checkpoint = model / "chkpnt30000.pth"
    ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    model.mkdir(parents=True)
    count = 4
    scaling = torch.tensor(
        [
            [math.log(9.0), math.log(2.0), math.log(1.0)],
            [math.log(10.0), math.log(1.0), math.log(1.0)],
            [math.log(12.0), math.log(4.0), math.log(2.0)],
            [math.log(3.0), math.log(15.0), math.log(11.0)],
        ],
        dtype=torch.float32,
    )
    params = (
        3,
        nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)),
        nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3)),
        nn.Parameter(
            torch.arange(count * 15 * 3, dtype=torch.float32).reshape(count, 15, 3)
        ),
        nn.Parameter(scaling),
        nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count, dtype=torch.float32)
        ),
        nn.Parameter(torch.zeros((count, 1), dtype=torch.float32)),
        torch.zeros(count, dtype=torch.float32),
        torch.zeros((count, 1), dtype=torch.float32),
        torch.zeros((count, 1), dtype=torch.float32),
        {"state": {}, "param_groups": []},
        1.0,
    )
    torch.save((params, 30000), checkpoint)
    _write_ply(ply, params)
    (model / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
    (model / "cameras.json").write_text("[]\n", encoding="utf-8")
    return model, _sha256(checkpoint), _sha256(ply)


class SharedClampedAnchorTests(unittest.TestCase):
    def setUp(self):
        _install_simple_knn_stub()

    def _args(self, model, output, checkpoint_sha, ply_sha, expected=2):
        return argparse.Namespace(
            scene_name="Synthetic",
            input_model_dir=str(model),
            output_model_dir=str(output),
            anchor_iteration=30000,
            max_scale=10.0,
            expected_clamped_count=expected,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_ply_sha256=ply_sha,
            code_commit="0" * 40,
        )

    def test_build_changes_only_expected_scaling_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, checkpoint_sha, ply_sha = _synthetic_model(root)
            output = root / "output"
            manifest = build_shared_anchor(
                self._args(model, output, checkpoint_sha, ply_sha)
            )
            self.assertEqual(manifest["clamped_indices"], [2, 3])
            self.assertEqual(manifest["counts"]["actual_clamped_gaussians"], 2)
            self.assertEqual(manifest["counts"]["changed_scaling_rows"], 2)
            self.assertEqual(manifest["counts"]["changed_scaling_elements"], 3)
            self.assertTrue(manifest["invariants"]["only_scaling_changed"])
            self.assertTrue(manifest["invariants"]["input_anchor_unchanged"])
            self.assertEqual(
                _sha256(model / "chkpnt30000.pth"), checkpoint_sha
            )
            self.assertEqual(
                _sha256(
                    model
                    / "point_cloud"
                    / "iteration_30000"
                    / "point_cloud.ply"
                ),
                ply_sha,
            )
            payload = json.loads(
                (output / "shared_clamp_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["clamped_indices_sha256"], manifest["clamped_indices_sha256"])
            self.assertEqual(
                (output / "SHARED_ANCHOR_STATUS").read_text(encoding="ascii"),
                "passed\n",
            )
            output_checkpoint = torch.load(
                output / "chkpnt30000.pth", weights_only=False
            )[0]
            activated = torch.exp(output_checkpoint[4].detach())
            self.assertLessEqual(float(activated.max()), 10.0 + 1e-6)
            self.assertTrue(torch.equal(output_checkpoint[1], torch.load(
                model / "chkpnt30000.pth", weights_only=False
            )[0][1]))

    def test_expected_count_mismatch_fails_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, checkpoint_sha, ply_sha = _synthetic_model(root)
            output = root / "output"
            with self.assertRaisesRegex(SharedAnchorError, "row count mismatch"):
                build_shared_anchor(
                    self._args(
                        model,
                        output,
                        checkpoint_sha,
                        ply_sha,
                        expected=3,
                    )
                )
            self.assertFalse(output.exists())
            self.assertEqual(_sha256(model / "chkpnt30000.pth"), checkpoint_sha)

    def test_output_must_not_exist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, checkpoint_sha, ply_sha = _synthetic_model(root)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(SharedAnchorError, "refusing to overwrite"):
                build_shared_anchor(
                    self._args(model, output, checkpoint_sha, ply_sha)
                )


if __name__ == "__main__":
    unittest.main()
