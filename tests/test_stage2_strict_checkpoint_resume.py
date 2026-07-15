from __future__ import annotations

import ast
import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_MODEL_PATH = REPO_ROOT / "scene" / "gaussian_model.py"


class _CpuTorch:
    optim = torch.optim

    @staticmethod
    def zeros(shape, device=None):
        del device
        return torch.zeros(shape)


def _load_contract_class():
    tree = ast.parse(GAUSSIAN_MODEL_PATH.read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    selected = {"capture", "restore", "training_setup_appearance_only"}
    methods = [
        node for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    if {node.name for node in methods} != selected:
        raise RuntimeError("Failed to locate strict Stage-2 checkpoint methods")
    contract_class = ast.ClassDef(
        name="StrictCheckpointContractUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[contract_class], type_ignores=[]))
    namespace = {"copy": copy, "torch": _CpuTorch()}
    exec(compile(module, str(GAUSSIAN_MODEL_PATH), "exec"), namespace)
    return namespace["StrictCheckpointContractUnderTest"]


class Stage2StrictCheckpointResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract_class()

    @staticmethod
    def _args():
        return SimpleNamespace(percent_dense=0.01, feature_lr=0.0025)

    def _model(self):
        model = self.contract()
        model.max_sh_degree = 3
        model.optimizer_type = "default"
        model.active_sh_degree = 0
        model.spatial_lr_scale = 1.0
        model.pretrained_exposures = None
        model._xyz = torch.nn.Parameter(torch.zeros(4, 3))
        model._features_dc = torch.nn.Parameter(torch.ones(4, 1, 3))
        model._features_rest = torch.nn.Parameter(torch.ones(4, 15, 3))
        model._scaling = torch.nn.Parameter(torch.zeros(4, 3))
        model._rotation = torch.nn.Parameter(torch.zeros(4, 4))
        model._opacity = torch.nn.Parameter(torch.zeros(4, 1))
        model._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4))
        model.get_xyz = model._xyz
        model.max_radii2D = torch.zeros(4)
        model.xyz_gradient_accum = torch.zeros(4, 1)
        model.denom = torch.zeros(4, 1)
        return model

    @staticmethod
    def _take_step(model, cap):
        loss = model._features_dc.square().sum()
        if cap > 0:
            loss = loss + model._features_rest.square().sum()
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

    def test_strict_sh0_and_sh1_checkpoint_restore_and_continue_adam(self):
        for cap, expected_groups in ((0, ["f_dc"]), (1, ["f_dc", "f_rest"])):
            with self.subTest(cap=cap):
                source = self._model()
                source.training_setup_appearance_only(self._args(), sh_degree_cap=cap)
                self._take_step(source, cap)

                checkpoint = io.BytesIO()
                torch.save((source.capture(), 30100), checkpoint)
                checkpoint.seek(0)
                model_args, iteration = torch.load(checkpoint, weights_only=False)

                resumed = self._model()
                resumed.training_setup = lambda _args: self.fail(
                    "strict checkpoint must not be restored into the legacy six-group optimizer"
                )
                resumed.restore(model_args, self._args(), optimizer_restore_mode="restore")

                self.assertEqual(iteration, 30100)
                self.assertEqual(
                    [group["name"] for group in resumed.optimizer.param_groups],
                    expected_groups,
                )
                self.assertEqual(len(resumed.optimizer.state), len(expected_groups))
                self.assertTrue(all(not p.requires_grad for p in (
                    resumed._xyz,
                    resumed._scaling,
                    resumed._rotation,
                    resumed._opacity,
                    resumed._exposure,
                )))
                for state in resumed.optimizer.state.values():
                    self.assertEqual(int(state["step"].item()), 1)

                self._take_step(resumed, cap)
                for state in resumed.optimizer.state.values():
                    self.assertEqual(int(state["step"].item()), 2)


if __name__ == "__main__":
    unittest.main()
