from __future__ import annotations

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from utils.camera_sequence import (
    build_sequence_manifest,
    load_sequence_manifest,
    save_sequence_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train.py"


def _load_train_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    names = {
        "_optimizer_state_step",
        "_validate_rgb_continuation_checkpoint",
        "_should_optimizer_step",
        "_validate_rgb_continuation_schedule",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if "_RGB_CONTINUATION_GROUPS" in targets:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    namespace = {"torch": torch}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace


class FixedCameraSequenceTests(unittest.TestCase):
    def test_private_rng_random_pop_is_reproducible_and_global_rng_untouched(self):
        names = ["a", "b", "c"]
        random.seed(91)
        before = random.getstate()
        first = build_sequence_manifest(names, steps=5000, seed=17)
        after = random.getstate()
        second = build_sequence_manifest(names, steps=5000, seed=17)
        self.assertEqual(before, after)
        self.assertEqual(first["sequence"], second["sequence"])
        self.assertEqual(first["sequence_sha256"], second["sequence_sha256"])
        self.assertEqual(len(first["sequence"]), 5000)
        for start in range(0, 4998, 3):
            self.assertEqual(set(first["sequence"][start : start + 3]), set(names))

    def test_hash_and_order_mismatches_fail_closed(self):
        manifest = build_sequence_manifest(["a", "b", "c"], steps=5000, seed=2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.json"
            save_sequence_manifest(path, manifest)
            loaded = load_sequence_manifest(
                path, camera_names=["a", "b", "c"], expected_steps=5000
            )
            self.assertEqual(loaded["sequence_sha256"], manifest["sequence_sha256"])
            with self.assertRaisesRegex(ValueError, "Ordered training-camera"):
                load_sequence_manifest(
                    path, camera_names=["b", "a", "c"], expected_steps=5000
                )

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["sequence"][0] = "b" if tampered["sequence"][0] != "b" else "a"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence SHA-256"):
                load_sequence_manifest(
                    path, camera_names=["a", "b", "c"], expected_steps=5000
                )


class RGBContinuationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_train_helpers()

    def test_final_step_removes_5000_update_off_by_one(self):
        should_step = self.helpers["_should_optimizer_step"]
        with_final = sum(should_step(i, 35000, True) for i in range(30001, 35001))
        legacy = sum(should_step(i, 35000, False) for i in range(30001, 35001))
        self.assertEqual(with_final, 5000)
        self.assertEqual(legacy, 4999)

    def test_schedule_validation_covers_smoke_and_formal_endpoints(self):
        validate = self.helpers["_validate_rgb_continuation_schedule"]
        self.assertEqual(validate(30000, 30000, 200, 30200, True), 30200)
        self.assertEqual(validate(30000, 30000, 5000, 35000, True), 35000)
        with self.assertRaisesRegex(ValueError, "final optimizer"):
            validate(30000, 30000, 5000, 35000, False)
        with self.assertRaisesRegex(ValueError, "final iteration"):
            validate(30000, 30000, 5000, 34999, True)
        with self.assertRaisesRegex(ValueError, "Scheduler"):
            validate(30000, 35000, 5000, 35000, True)

    @staticmethod
    def _checkpoint(step=29971):
        group_names = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
        groups = []
        states = {}
        for index, name in enumerate(group_names):
            groups.append({"name": name, "params": [index], "lr": 1e-3})
            states[index] = {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros((2, 3)),
                "exp_avg_sq": torch.zeros((2, 3)),
            }
        model_params = [None] * 12
        model_params[10] = {"param_groups": groups, "state": states}
        return tuple(model_params)

    def test_six_group_adam_is_required_and_actual_steps_are_reported(self):
        validate = self.helpers["_validate_rgb_continuation_checkpoint"]
        summary = validate(self._checkpoint(), 30000, 30000)
        self.assertEqual(summary["uniform_adam_step"], 29971)
        self.assertEqual(len(summary["groups"]), 6)

        malformed = list(self._checkpoint())
        malformed[10] = dict(malformed[10])
        malformed[10]["param_groups"] = malformed[10]["param_groups"][:-1]
        with self.assertRaisesRegex(RuntimeError, "six-group"):
            validate(tuple(malformed), 30000, 30000)

    def test_new_protocol_is_opt_in_and_legacy_defaults_remain(self):
        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
        defaults = {}
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
            ):
                try:
                    option = ast.literal_eval(node.args[0])
                except Exception:
                    continue
                if option in {
                    "--rgb_continuation_recipe",
                    "--optimizer_step_at_final_iteration",
                }:
                    defaults[option] = {
                        item.arg: ast.literal_eval(item.value)
                        for item in node.keywords
                        if item.arg == "default"
                    }["default"]
        self.assertEqual(defaults["--rgb_continuation_recipe"], "legacy")
        self.assertFalse(defaults["--optimizer_step_at_final_iteration"])

        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("if fixed_camera_sequence is not None:", source)
        self.assertIn("rand_idx = randint(0, len(viewpoint_indices) - 1)", source)
        self.assertIn('args.artifact_save_semantics = "aligned"', source)
        self.assertIn("and not rgb_continuation", source)


if __name__ == "__main__":
    unittest.main()
