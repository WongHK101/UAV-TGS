import argparse
import hashlib
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

from tools.build_adaptive_scale_anchor import (
    build_adaptive_anchor,
    project_largest_scale_axis,
    scene_tail_thresholds,
    scsp_thresholds,
)
from utils.sparse_support import VoxelHashNN


def _install_simple_knn_stub():
    package = types.ModuleType("simple_knn")
    extension = types.ModuleType("simple_knn._C")
    extension.distCUDA2 = lambda value: torch.zeros(
        value.shape[0], device=value.device, dtype=value.dtype
    )
    package._C = extension
    sys.modules.setdefault("simple_knn", package)
    sys.modules.setdefault("simple_knn._C", extension)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ply(path: Path, params) -> None:
    xyz = params[1].detach().numpy()
    normal = np.zeros_like(xyz)
    dc = params[2].detach().transpose(1, 2).flatten(start_dim=1).numpy()
    rest = params[3].detach().transpose(1, 2).flatten(start_dim=1).numpy()
    values = np.concatenate(
        [
            xyz,
            normal,
            dc,
            rest,
            params[6].detach().numpy(),
            params[4].detach().numpy(),
            params[5].detach().numpy(),
        ],
        axis=1,
    )
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(dc.shape[1])]
    names += [f"f_rest_{index}" for index in range(rest.shape[1])]
    names += ["opacity", "scale_0", "scale_1", "scale_2"]
    names += ["rot_0", "rot_1", "rot_2", "rot_3"]
    records = np.empty(xyz.shape[0], dtype=[(name, "f4") for name in names])
    records[:] = list(map(tuple, values))
    path.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(records, "vertex")]).write(path)


def _synthetic_model(root: Path) -> tuple[Path, str, str]:
    model = root / "input"
    model.mkdir()
    count = 9
    activated = torch.tensor(
        [
            [1.0, 0.8, 0.7],
            [1.1, 0.9, 0.8],
            [1.2, 1.0, 0.9],
            [1.3, 1.1, 1.0],
            [1.4, 1.2, 1.0],
            [1.5, 1.2, 1.1],
            [1.6, 1.3, 1.1],
            [1.7, 1.4, 1.2],
            [30.0, 2.0, 1.0],
        ],
        dtype=torch.float32,
    )
    params = (
        3,
        nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 3)),
        nn.Parameter(torch.zeros((count, 1, 3), dtype=torch.float32)),
        nn.Parameter(torch.zeros((count, 15, 3), dtype=torch.float32)),
        nn.Parameter(torch.log(activated)),
        nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count)),
        nn.Parameter(torch.zeros((count, 1))),
        torch.zeros(count),
        torch.zeros((count, 1)),
        torch.zeros((count, 1)),
        {"state": {}, "param_groups": []},
        1.0,
    )
    checkpoint = model / "chkpnt30000.pth"
    ply = model / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    torch.save((params, 30000), checkpoint)
    _write_ply(ply, params)
    (model / "cfg_args").write_text("Namespace()\n", encoding="utf-8")
    return model, _sha256(checkpoint), _sha256(ply)


class AdaptiveScaleTests(unittest.TestCase):
    def setUp(self):
        _install_simple_knn_stub()

    def test_scene_tail_projects_only_outlier_max_axis(self):
        activated = torch.tensor(
            [[1.0, 0.8, 0.7], [1.1, 1.0, 0.8], [1.2, 1.0, 0.9],
             [1.3, 1.1, 1.0], [1.4, 1.2, 1.0], [1.5, 1.2, 1.1],
             [1.6, 1.3, 1.1], [1.7, 1.4, 1.2], [30.0, 2.0, 1.0]]
        )
        thresholds, stats = scene_tail_thresholds(activated)
        output, selected, axis = project_largest_scale_axis(torch.log(activated), thresholds)
        self.assertEqual(torch.nonzero(selected).flatten().tolist(), [8])
        self.assertEqual(int(axis[8]), 0)
        self.assertAlmostEqual(float(torch.exp(output[8, 1])), 2.0, places=6)
        self.assertAlmostEqual(float(torch.exp(output[8, 2])), 1.0, places=6)
        self.assertAlmostEqual(float(torch.exp(output[8, 0])), stats["tau_scene"], places=5)

    def test_scsp_uses_local_threshold_and_scene_fallback(self):
        activated = torch.tensor([[2.0, 1.0, 0.5]] * 8 + [[20.0, 1.0, 0.5]])
        scene = torch.full((9,), 10.0)
        local = torch.tensor([1.0] * 8 + [float("inf")])
        threshold, stats = scsp_thresholds(activated, local, scene)
        self.assertEqual(stats["fallback_count"], 1)
        self.assertAlmostEqual(float(threshold[-1]), 10.0)
        self.assertTrue(bool((threshold[:-1] <= scene[:-1]).all()))

    def test_voxel_hash_top2_is_sorted_and_missing_is_inf(self):
        support = torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [4.1, 0.1, 0.1]])
        index = VoxelHashNN(support, voxel_size=1.0)
        distances = index.query_topk_torch(
            torch.tensor([[0.0, 0.0, 0.0], [20.0, 20.0, 20.0]]),
            k=2,
            max_voxel_radius=2,
        )
        self.assertLessEqual(float(distances[0, 0]), float(distances[0, 1]))
        self.assertTrue(bool(torch.isinf(distances[1]).all()))

    def test_full_scene_tail_builder_preserves_non_scale_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model, checkpoint_sha, ply_sha = _synthetic_model(root)
            output = root / "output"
            args = argparse.Namespace(
                scene_name="Synthetic",
                method="scene_tail",
                input_model_dir=str(model),
                output_model_dir=str(output),
                anchor_iteration=30000,
                sparse_root=None,
                support_voxel_size=1.5,
                support_max_voxel_radius=2,
                query_chunk_size=32,
                absolute_clamp_manifest=None,
                expected_checkpoint_sha256=checkpoint_sha,
                expected_ply_sha256=ply_sha,
                code_commit="0" * 40,
            )
            manifest = build_adaptive_anchor(args)
            self.assertEqual(manifest["counts"]["modified_gaussians"], 1)
            self.assertTrue(manifest["invariants"]["only_scaling_changed"])
            self.assertTrue(manifest["invariants"]["schema_index_count_exact"])
            self.assertEqual(manifest["modified_indices"], [8])
            self.assertEqual(
                (output / "ADAPTIVE_SCALE_STATUS").read_text(encoding="ascii"),
                "passed\n",
            )
            self.assertEqual(_sha256(model / "chkpnt30000.pth"), checkpoint_sha)


if __name__ == "__main__":
    unittest.main()
