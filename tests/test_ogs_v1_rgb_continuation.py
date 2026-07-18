from __future__ import annotations

import ast
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from utils.camera_sequence import (
    build_sequence_manifest,
    camera_parameters_hash,
    load_sequence_manifest,
    save_sequence_manifest,
    sequence_hash,
    sha256_json,
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
        "_normalized_sha256",
        "_verified_sha256_binding",
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
    @staticmethod
    def _camera(name="a", translation=(0.0, 0.0, 0.0)):
        return SimpleNamespace(
            image_name=name,
            uid=1,
            colmap_id=7,
            R=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            T=translation,
            FoVx=0.8,
            FoVy=0.7,
            image_width=1280,
            image_height=1024,
        )

    def test_camera_parameter_hash_binds_pose_and_order(self):
        baseline = camera_parameters_hash(
            [self._camera("a"), self._camera("b", (1.0, 0.0, 0.0))]
        )
        self.assertEqual(len(baseline), 64)
        self.assertNotEqual(
            baseline,
            camera_parameters_hash(
                [self._camera("a"), self._camera("b", (1.001, 0.0, 0.0))]
            ),
        )
        self.assertNotEqual(
            baseline,
            camera_parameters_hash(
                [self._camera("b", (1.0, 0.0, 0.0)), self._camera("a")]
            ),
        )

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

    def test_self_consistent_sequence_reordering_is_rejected(self):
        manifest = build_sequence_manifest(["a", "b", "c"], steps=5000, seed=2)
        manifest["sequence"][0], manifest["sequence"][1] = (
            manifest["sequence"][1],
            manifest["sequence"][0],
        )
        manifest["sequence_sha256"] = sequence_hash(manifest["sequence"])
        manifest.pop("manifest_sha256")
        manifest["manifest_sha256"] = sha256_json(manifest)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sequence.json"
            save_sequence_manifest(path, manifest)
            with self.assertRaisesRegex(ValueError, "deterministic sequence"):
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

    def test_external_hash_binding_helper_fails_on_every_mismatch(self):
        verify = self.helpers["_verified_sha256_binding"]
        expected = "a" * 64
        self.assertEqual(
            verify("OGS-v1 camera-parameters", expected, expected),
            {
                "expected_sha256": expected,
                "actual_sha256": expected,
                "verified": True,
            },
        )
        for label in (
            "OGS-v1 anchor",
            "OGS-v1 ordered-camera",
            "OGS-v1 camera-parameters",
            "OGS-v1 fixed-sequence",
            "OGS-v1 fixed-sequence manifest",
            "OGS-v1 cache file",
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    verify(label, expected, "b" * 64)
        with self.assertRaisesRegex(RuntimeError, "64-character SHA-256"):
            verify("OGS-v1 formula", "not-a-sha", expected)

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
                    "--lambda_ogs",
                    "--ogs_v1_lambda_protocol",
                }:
                    keywords = {
                        item.arg: ast.literal_eval(item.value)
                        for item in node.keywords
                        if item.arg in {"default", "choices"}
                    }
                    defaults[option] = keywords
        self.assertEqual(
            defaults["--rgb_continuation_recipe"]["default"], "legacy"
        )
        self.assertFalse(
            defaults["--optimizer_step_at_final_iteration"]["default"]
        )
        self.assertEqual(defaults["--lambda_ogs"]["default"], 1e-3)
        self.assertEqual(
            defaults["--ogs_v1_lambda_protocol"]["default"], "initial_1e-3"
        )
        self.assertIn(
            "train_only_gradient_probe_recalibrated_1_1",
            defaults["--ogs_v1_lambda_protocol"]["choices"],
        )

        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn("if fixed_camera_sequence is not None:", source)
        self.assertIn("rand_idx = randint(0, len(viewpoint_indices) - 1)", source)
        self.assertIn('args.artifact_save_semantics = "aligned"', source)
        self.assertIn("and not rgb_continuation", source)

    def test_recalibrated_mode_is_one_shot_and_manifest_binds_all_hash_classes(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn(
            'args.lambda_ogs != 1.1',
            source,
        )
        for option in (
            "--expected_ogs_cache_sha256",
            "--expected_ogs_anchor_sha256",
            "--expected_ogs_camera_sha256",
            "--expected_ogs_camera_parameters_sha256",
            "--expected_ogs_sequence_sha256",
            "--expected_ogs_sequence_manifest_sha256",
            "--expected_ogs_formula_sha256",
        ):
            self.assertIn(option, source)
        for manifest_key in (
            '"anchor_checkpoint"',
            '"ordered_camera_set"',
            '"fixed_5000_camera_sequence"',
            '"cache_file"',
            '"cache_semantic"',
            '"formula"',
            '"ogs_lambda_calibration"',
            '"ogs_v1_verified_bindings"',
        ):
            self.assertIn(manifest_key, source)


if __name__ == "__main__":
    unittest.main()
