from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train.py"


def _load_save_helper():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_save_iteration_artifacts"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"os": os, "torch": torch}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace["_save_iteration_artifacts"]


def _optimizer_step(gaussians) -> int:
    if not gaussians.optimizer.state:
        return 0
    state = next(iter(gaussians.optimizer.state.values()))
    step = state.get("step", 0)
    return int(step.item()) if torch.is_tensor(step) else int(step)


class _FakeGaussians:
    def __init__(self, value: float):
        self.value = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.optimizer = torch.optim.Adam([self.value], lr=0.05)
        self.zeroed_caps = []

    def zero_sh_above_degree_(self, cap):
        self.zeroed_caps.append(int(cap))

    def capture(self):
        return {
            "value": self.value.detach().clone(),
            "optimizer": self.optimizer.state_dict(),
        }


class _FakeScene:
    def __init__(self, model_path: str, gaussians: _FakeGaussians):
        self.model_path = model_path
        self.gaussians = gaussians
        self.saved = []

    def save(self, iteration):
        self.saved.append(
            {
                "iteration": int(iteration),
                "value": self.gaussians.value.detach().clone(),
                "optimizer_step": _optimizer_step(self.gaussians),
            }
        )


def _take_step(gaussians: _FakeGaussians) -> None:
    gaussians.value.square().sum().backward()
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)


class Stage2AlignedSaveSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_artifacts = staticmethod(_load_save_helper())

    def _save(self, root: str, gaussians: _FakeGaussians, iteration: int):
        scene = _FakeScene(root, gaussians)
        self.save_artifacts(
            scene,
            gaussians,
            iteration,
            save_gaussians=True,
            save_checkpoint=True,
            thermal_max_sh_degree=3,
        )
        checkpoint = Path(root) / f"chkpnt{iteration}.pth"
        payload, saved_iteration = torch.load(checkpoint, weights_only=False)
        return scene.saved[0], payload, saved_iteration

    def test_aaai_ply_and_checkpoint_share_one_post_step_endpoint(self):
        gaussians = _FakeGaussians(1.0)
        _take_step(gaussians)
        self.assertEqual(_optimizer_step(gaussians), 1)

        with tempfile.TemporaryDirectory() as temporary:
            ply_endpoint, checkpoint, iteration = self._save(
                temporary, gaussians, 40000
            )

        self.assertEqual(iteration, 40000)
        self.assertEqual(ply_endpoint["optimizer_step"], 1)
        self.assertTrue(torch.equal(ply_endpoint["value"], checkpoint["value"]))
        checkpoint_state = next(iter(checkpoint["optimizer"]["state"].values()))
        self.assertEqual(int(checkpoint_state["step"].item()), 1)
        self.assertEqual(gaussians.zeroed_caps, [3])

    def test_aligned_checkpoint_resumes_adam_and_saves_next_endpoint(self):
        source = _FakeGaussians(1.0)
        _take_step(source)

        with tempfile.TemporaryDirectory() as temporary:
            _, checkpoint, _ = self._save(temporary, source, 40000)

            resumed = _FakeGaussians(float(checkpoint["value"].item()))
            resumed.optimizer.load_state_dict(checkpoint["optimizer"])
            self.assertEqual(_optimizer_step(resumed), 1)

            _take_step(resumed)
            self.assertEqual(_optimizer_step(resumed), 2)
            ply_endpoint, next_checkpoint, iteration = self._save(
                temporary, resumed, 40001
            )

        self.assertEqual(iteration, 40001)
        self.assertEqual(ply_endpoint["optimizer_step"], 2)
        self.assertTrue(
            torch.equal(ply_endpoint["value"], next_checkpoint["value"])
        )
        next_state = next(iter(next_checkpoint["optimizer"]["state"].values()))
        self.assertEqual(int(next_state["step"].item()), 2)

    def test_training_keeps_legacy_order_and_moves_only_aaai_pair_post_step(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        training_start = source.index("def training(")
        training_end = source.index("\ndef prepare_output_and_logger", training_start)
        training_source = source[training_start:training_end]

        legacy_ply = training_source.index(
            "# Legacy compatibility: PLY is written before the optimizer step."
        )
        optimizer_step = training_source.index("# Optimizer step", legacy_ply)
        aaai_pair = training_source.index(
            "# AAAI recipe: both artifacts are written after the same optimizer",
            optimizer_step,
        )
        self.assertLess(legacy_ply, optimizer_step)
        self.assertLess(optimizer_step, aaai_pair)
        self.assertIn(
            'str(getattr(args, "thermal_recipe", "legacy")) == "aaai_strict"',
            training_source,
        )
        self.assertIn(
            "if (not aligned_artifact_saves) and (iteration in saving_iterations):",
            training_source,
        )
        self.assertIn("elif iteration in checkpoint_iterations:", training_source)


if __name__ == "__main__":
    unittest.main()
