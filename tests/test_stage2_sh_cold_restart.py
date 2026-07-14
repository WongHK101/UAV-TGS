from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_MODEL_PATH = REPO_ROOT / "scene" / "gaussian_model.py"
TRAIN_PATH = REPO_ROOT / "train.py"
PIPELINE_PATH = REPO_ROOT / "run_uavfgs_pipeline.py"


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _TensorView:
    def __init__(self, array: np.ndarray):
        self.array = array

    def zero_(self):
        self.array[...] = 0.0
        return self


class _FakeTensor:
    def __init__(self, array: np.ndarray):
        self.array = array

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, item):
        return _TensorView(self.array[item])


class _FakeOptimizer:
    def __init__(self):
        self.state = {}
        self.loaded = False

    def load_state_dict(self, state_dict):
        self.loaded = True
        self.state = dict(state_dict["state"])


def _load_gaussian_contract_class():
    tree = ast.parse(GAUSSIAN_MODEL_PATH.read_text(encoding="utf-8"))
    source_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel"
    )
    selected = {
        "restore",
        "zero_sh_above_degree_",
        "configure_sh_degree_cap_",
        "oneupSHdegree",
    }
    methods = [
        node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in selected
    ]
    if {node.name for node in methods} != selected:
        raise RuntimeError("Failed to locate the Stage-2 GaussianModel contract methods")
    contract_class = ast.ClassDef(
        name="GaussianContractUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[contract_class], type_ignores=[]))
    namespace = {"torch": _FakeTorch()}
    exec(compile(module, str(GAUSSIAN_MODEL_PATH), "exec"), namespace)
    return namespace["GaussianContractUnderTest"]


def _model_args(optimizer_state):
    return (
        3,  # active_sh_degree
        "xyz",
        "f_dc",
        "f_rest",
        "scaling",
        "rotation",
        "opacity",
        "max_radii",
        "xyz_accum",
        "denom",
        {"state": optimizer_state},
        1.0,
    )


class Stage2ShColdRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load_gaussian_contract_class()

    def _restore_model(self):
        model = self.contract()

        def training_setup(_args):
            model.optimizer = _FakeOptimizer()

        model.training_setup = training_setup
        return model

    def test_restore_remains_the_default_and_loads_adam_step(self):
        model = self._restore_model()
        model.restore(_model_args({"rgb-param": {"step": 30000}}), object())
        self.assertTrue(model.optimizer.loaded)
        self.assertEqual(model.optimizer.state["rgb-param"]["step"], 30000)

    def test_fresh_restore_has_no_inherited_state_or_step(self):
        model = self._restore_model()
        model.restore(
            _model_args({"rgb-param": {"step": 30000}}),
            object(),
            optimizer_restore_mode="fresh",
        )
        self.assertFalse(model.optimizer.loaded)
        self.assertEqual(len(model.optimizer.state), 0)
        self.assertFalse(any("step" in state for state in model.optimizer.state.values()))

    def test_cold_restart_caps_oneup_and_preserves_sh3_schema(self):
        for cap in (0, 1, 3):
            with self.subTest(cap=cap):
                model = self.contract()
                model.max_sh_degree = 3
                model.active_sh_degree = 3
                values = np.ones((2, 15, 3), dtype=np.float32)
                model._features_rest = _FakeTensor(values)

                model.configure_sh_degree_cap_(cap, cold_restart=True)
                self.assertEqual(model.active_sh_degree, 0)
                self.assertEqual(values.shape, (2, 15, 3))
                keep_rest = (cap + 1) ** 2 - 1
                np.testing.assert_array_equal(values[:, keep_rest:, :], 0.0)

                for _ in range(10):
                    model.oneupSHdegree(max_degree=cap)
                self.assertEqual(model.active_sh_degree, cap)

    def test_uncapped_oneup_keeps_legacy_behavior(self):
        model = self.contract()
        model.max_sh_degree = 3
        model.active_sh_degree = 0
        for expected in (1, 2, 3, 3):
            model.oneupSHdegree()
            self.assertEqual(model.active_sh_degree, expected)

    def test_pipeline_defaults_and_explicit_forwarding_are_orthogonal(self):
        pipeline = PIPELINE_PATH.read_text(encoding="utf-8")
        train = TRAIN_PATH.read_text(encoding="utf-8")

        self.assertIn(
            '"--thermal_optimizer_state", choices=["restore", "fresh"], default="restore"',
            pipeline,
        )
        self.assertIn('"--thermal_max_sh_degree", type=int, choices=[0, 1, 3], default=None', pipeline)
        self.assertIn('if args.thermal_max_sh_degree is not None:', pipeline)
        self.assertIn('["--thermal_max_sh_degree", str(args.thermal_max_sh_degree)]', pipeline)
        self.assertIn('if args.thermal_optimizer_state != "restore":', pipeline)
        self.assertIn('["--thermal_optimizer_state", str(args.thermal_optimizer_state)]', pipeline)

        self.assertIn(
            '"--thermal_optimizer_state", choices=["restore", "fresh"], default="restore"',
            train,
        )
        self.assertIn('"--thermal_max_sh_degree", type=int, choices=[0, 1, 3], default=None', train)


if __name__ == "__main__":
    unittest.main()
