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
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    selected = {
        "capture",
        "restore",
        "training_setup_appearance_only",
        "training_setup_geometry_frozen_opacity_adaptive",
    }
    methods = [
        node for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    if {node.name for node in methods} != selected:
        raise RuntimeError("Failed to locate opacity-adaptive Stage-2 model methods")
    contract_class = ast.ClassDef(
        name="OpacityAdaptiveStage2ContractUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[contract_class], type_ignores=[]))
    namespace = {"copy": copy, "torch": _CpuTorch()}
    exec(compile(module, str(GAUSSIAN_MODEL_PATH), "exec"), namespace)
    return namespace["OpacityAdaptiveStage2ContractUnderTest"]


class Stage2OpacityAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract_class()

    @staticmethod
    def _args():
        return SimpleNamespace(
            percent_dense=0.01,
            feature_lr=0.0025,
            opacity_lr=2e-4,
        )

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
        model._opacity = torch.nn.Parameter(torch.ones(4, 1))
        model._exposure = torch.nn.Parameter(torch.zeros(1, 3, 4))
        model.get_xyz = model._xyz
        model.max_radii2D = torch.zeros(4)
        model.xyz_gradient_accum = torch.zeros(4, 1)
        model.denom = torch.zeros(4, 1)
        model.optimizer = None
        model.exposure_optimizer = None
        return model

    @staticmethod
    def _take_step(model):
        loss = (
            model._features_dc.square().sum()
            + model._features_rest.square().sum()
            + model._opacity.square().sum()
        )
        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def _steps(model):
        return {
            group["name"]: int(model.optimizer.state[group["params"][0]]["step"].item())
            for group in model.optimizer.param_groups
        }

    def test_exact_optimizer_groups_and_requires_grad_contract(self):
        model = self._model()
        model.training_setup_geometry_frozen_opacity_adaptive(
            self._args(), sh_degree_cap=3
        )

        self.assertEqual(
            [group["name"] for group in model.optimizer.param_groups],
            ["f_dc", "f_rest", "opacity"],
        )
        self.assertEqual(
            [group["lr"] for group in model.optimizer.param_groups],
            [0.0025, 0.0025 / 20.0, 2e-4],
        )
        self.assertTrue(model._features_dc.requires_grad)
        self.assertTrue(model._features_rest.requires_grad)
        self.assertTrue(model._opacity.requires_grad)
        for parameter in (
            model._xyz,
            model._scaling,
            model._rotation,
            model._exposure,
        ):
            self.assertFalse(parameter.requires_grad)
        self.assertIsNone(model.exposure_optimizer)
        self.assertEqual(len(model.optimizer.state), 0)

    def test_recipe_rejects_non_sh3_cap(self):
        model = self._model()
        with self.assertRaisesRegex(ValueError, "requires the full SH degree 3"):
            model.training_setup_geometry_frozen_opacity_adaptive(
                self._args(), sh_degree_cap=1
            )

    def test_fresh_rebuild_discards_all_adam_state(self):
        model = self._model()
        model.training_setup_geometry_frozen_opacity_adaptive(self._args())
        self._take_step(model)
        self.assertEqual(self._steps(model), {"f_dc": 1, "f_rest": 1, "opacity": 1})

        model.training_setup_geometry_frozen_opacity_adaptive(
            self._args(), preserve_optimizer_state=False
        )
        self.assertEqual(len(model.optimizer.state), 0)

    def test_preserve_rebuild_keeps_only_three_group_steps(self):
        model = self._model()
        model.training_setup_geometry_frozen_opacity_adaptive(self._args())
        self._take_step(model)

        model.training_setup_geometry_frozen_opacity_adaptive(
            self._args(), preserve_optimizer_state=True
        )
        self.assertEqual(self._steps(model), {"f_dc": 1, "f_rest": 1, "opacity": 1})
        self._take_step(model)
        self.assertEqual(self._steps(model), {"f_dc": 2, "f_rest": 2, "opacity": 2})

    def test_checkpoint_restore_rebuilds_a3_layout_and_continues_steps(self):
        source = self._model()
        source.training_setup_geometry_frozen_opacity_adaptive(self._args())
        self._take_step(source)

        checkpoint = io.BytesIO()
        torch.save((source.capture(), 30100), checkpoint)
        checkpoint.seek(0)
        model_args, iteration = torch.load(checkpoint, weights_only=False)

        resumed = self._model()
        resumed.training_setup = lambda _args: self.fail(
            "A3 checkpoint must not be restored into the legacy six-group optimizer"
        )
        resumed.restore(model_args, self._args(), optimizer_restore_mode="restore")

        self.assertEqual(iteration, 30100)
        self.assertEqual(
            [group["name"] for group in resumed.optimizer.param_groups],
            ["f_dc", "f_rest", "opacity"],
        )
        self.assertEqual(self._steps(resumed), {"f_dc": 1, "f_rest": 1, "opacity": 1})
        self.assertTrue(resumed._opacity.requires_grad)
        self.assertTrue(resumed._features_dc.requires_grad)
        self.assertTrue(resumed._features_rest.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in (
            resumed._xyz,
            resumed._scaling,
            resumed._rotation,
            resumed._exposure,
        )))
        self.assertIsNone(resumed.exposure_optimizer)

        self._take_step(resumed)
        self.assertEqual(self._steps(resumed), {"f_dc": 2, "f_rest": 2, "opacity": 2})


if __name__ == "__main__":
    unittest.main()
