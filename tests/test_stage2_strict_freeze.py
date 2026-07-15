from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_MODEL_PATH = REPO_ROOT / "scene" / "gaussian_model.py"


class _FakeParameter:
    def __init__(self):
        self.requires_grad = True

    def requires_grad_(self, enabled):
        self.requires_grad = bool(enabled)
        return self


class _FakeAdam:
    def __init__(self, groups, lr, eps):
        self.param_groups = groups
        self.defaults = {"lr": lr, "eps": eps}
        self.state = {}


class _FakeTorch:
    class optim:
        Adam = _FakeAdam

    @staticmethod
    def zeros(shape, device=None):
        return SimpleNamespace(shape=shape, device=device)


def _load_contract_class():
    tree = ast.parse(GAUSSIAN_MODEL_PATH.read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    selected = {"training_setup_appearance_only", "update_learning_rate"}
    methods = [
        node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    if {node.name for node in methods} != selected:
        raise RuntimeError("Failed to locate strict Stage-2 model methods")
    contract_class = ast.ClassDef(
        name="StrictStage2ContractUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[contract_class], type_ignores=[]))
    namespace = {"copy": copy, "torch": _FakeTorch()}
    exec(compile(module, str(GAUSSIAN_MODEL_PATH), "exec"), namespace)
    return namespace["StrictStage2ContractUnderTest"]


class Stage2StrictFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_contract_class()

    def _model(self):
        model = self.contract()
        model.max_sh_degree = 3
        model.optimizer_type = "default"
        model.get_xyz = SimpleNamespace(shape=(11, 3))
        model.pretrained_exposures = None
        model._xyz = _FakeParameter()
        model._features_dc = _FakeParameter()
        model._features_rest = _FakeParameter()
        model._scaling = _FakeParameter()
        model._rotation = _FakeParameter()
        model._opacity = _FakeParameter()
        model._exposure = _FakeParameter()
        return model

    @staticmethod
    def _args():
        return SimpleNamespace(percent_dense=0.01, feature_lr=0.0025)

    def test_sh0_optimizer_contains_only_dc_and_freezes_everything_else(self):
        model = self._model()
        model.training_setup_appearance_only(self._args(), sh_degree_cap=0)

        self.assertEqual([group["name"] for group in model.optimizer.param_groups], ["f_dc"])
        self.assertTrue(model._features_dc.requires_grad)
        self.assertFalse(model._features_rest.requires_grad)
        self.assertIsNone(model.exposure_optimizer)
        for parameter in (
            model._xyz,
            model._scaling,
            model._rotation,
            model._opacity,
            model._exposure,
        ):
            self.assertFalse(parameter.requires_grad)
        self.assertEqual(len(model.optimizer.state), 0)

    def test_sh1_and_sh3_optimizer_groups_are_appearance_only(self):
        for cap in (1, 3):
            with self.subTest(cap=cap):
                model = self._model()
                model.training_setup_appearance_only(self._args(), sh_degree_cap=cap)
                self.assertEqual(
                    [group["name"] for group in model.optimizer.param_groups],
                    ["f_dc", "f_rest"],
                )
                self.assertTrue(model._features_dc.requires_grad)
                self.assertTrue(model._features_rest.requires_grad)
                self.assertIsNone(model.exposure_optimizer)
                frozen = (
                    model._xyz,
                    model._scaling,
                    model._rotation,
                    model._opacity,
                    model._exposure,
                )
                self.assertTrue(all(not parameter.requires_grad for parameter in frozen))

    def test_default_cap_keeps_full_appearance_optimizer(self):
        model = self._model()
        model.training_setup_appearance_only(self._args())
        self.assertEqual(
            [group["name"] for group in model.optimizer.param_groups],
            ["f_dc", "f_rest"],
        )

    def test_invalid_cap_is_rejected(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model.training_setup_appearance_only(self._args(), sh_degree_cap=4)

    def test_learning_rate_update_accepts_disabled_exposure_optimizer(self):
        model = self._model()
        model.training_setup_appearance_only(self._args(), sh_degree_cap=1)
        self.assertIsNone(model.update_learning_rate(1))

    def test_optional_feature_state_transfer_keeps_only_appearance_state(self):
        model = self._model()
        old_optimizer = _FakeAdam(
            [
                {"params": [model._features_dc], "name": "f_dc"},
                {"params": [model._features_rest], "name": "f_rest"},
                {"params": [model._xyz], "name": "xyz"},
            ],
            lr=0.0,
            eps=1e-15,
        )
        old_optimizer.state[model._features_dc] = {"step": 30000}
        old_optimizer.state[model._features_rest] = {"step": 30000}
        old_optimizer.state[model._xyz] = {"step": 30000}
        model.optimizer = old_optimizer

        model.training_setup_appearance_only(
            self._args(), sh_degree_cap=1, preserve_feature_state=True
        )
        self.assertEqual(len(model.optimizer.state), 2)
        self.assertEqual(model.optimizer.state[model._features_dc]["step"], 30000)
        self.assertEqual(model.optimizer.state[model._features_rest]["step"], 30000)
        self.assertNotIn(model._xyz, model.optimizer.state)

    def test_legacy_training_setup_contract_is_unchanged(self):
        source = GAUSSIAN_MODEL_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        source_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
        )
        legacy = next(
            node for node in source_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "training_setup"
        )
        legacy_source = ast.get_source_segment(source, legacy)
        self.assertIsNotNone(legacy_source)
        for name in ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"):
            self.assertIn(f'"name": "{name}"', legacy_source.replace("'", '"'))
        self.assertIn("self.exposure_optimizer = torch.optim.Adam", legacy_source)


if __name__ == "__main__":
    unittest.main()
