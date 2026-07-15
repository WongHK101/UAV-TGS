import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "run_uavfgs_pipeline.py"

sys.path.insert(0, str(REPO_ROOT))
import run_uavfgs_pipeline as pipeline  # noqa: E402


class PipelineThermalRecipeTests(unittest.TestCase):
    def _dry_run(self, *extra_args):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            out_root = root / "out"
            data_root.mkdir()
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(PIPELINE),
                    "--data_root",
                    str(data_root),
                    "--out_root",
                    str(out_root),
                    "--from_step",
                    "10",
                    "--to_step",
                    "10",
                    "--dry_run",
                    "--debug_dump",
                    *extra_args,
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                self.fail(
                    f"pipeline dry-run failed ({completed.returncode})\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            payload = json.loads((out_root / "pipeline_debug.json").read_text(encoding="utf-8"))
            return payload, payload["cmds"]["train2_cmd"].split()

    @staticmethod
    def _value(tokens, flag):
        return tokens[tokens.index(flag) + 1]

    @staticmethod
    def _values(tokens, flag):
        start = tokens.index(flag) + 1
        values = []
        for token in tokens[start:]:
            if token.startswith("--"):
                break
            values.append(token)
        return values

    def test_legacy_default_train_tokens_remain_exact(self):
        payload, tokens = self._dry_run()
        suffix = tokens[tokens.index("--iterations") :]
        self.assertEqual(
            suffix,
            [
                "--iterations", "60000",
                "--checkpoint_iterations", "60000",
                "--position_lr_init", "0", "--position_lr_final", "0",
                "--scaling_lr", "0", "--rotation_lr", "0",
                "--opacity_lr", "0.0002", "--feature_lr", "0.001",
                "--densify_from_iter", "999999",
                "--densify_until_iter", "0",
                "--densification_interval", "999999",
                "--opacity_reset_interval", "999999",
                "--lambda_dssim", "0.05", "--eval",
                "--clamp_scale_max", "10.0",
                "--thermal_reset_features",
                "--t_struct_grad_w", "0.006",
                "--t_struct_grad_norm", "true",
            ],
        )
        protocol = payload["protocol"]["thermal_stage2"]
        self.assertEqual(protocol["thermal_recipe"], "legacy")
        self.assertIsNone(protocol["thermal_checkpoint_offsets"])
        self.assertEqual(protocol["thermal_checkpoint_offsets_configured"], [10000, 20000, 30000])
        self.assertEqual(protocol["thermal_checkpoint_iterations"], [60000])
        self.assertFalse(protocol["thermal_checkpoint_offsets_applied"])

    def test_aaai_strict_resolves_and_forwards_complete_protocol(self):
        payload, tokens = self._dry_run("--thermal_recipe", "aaai_strict")
        protocol = payload["protocol"]["thermal_stage2"]

        self.assertEqual(protocol["thermal_recipe"], "aaai_strict")
        self.assertEqual(protocol["thermal_freeze_mode"], "strict")
        self.assertEqual(protocol["thermal_scale_clamp"], "off")
        self.assertEqual(protocol["thermal_max_sh_degree"], 1)
        self.assertEqual(protocol["thermal_optimizer_state"], "fresh")
        self.assertEqual(protocol["thermal_checkpoint_offsets"], [10000, 20000, 30000])
        self.assertEqual(protocol["thermal_checkpoint_iterations"], [40000, 50000, 60000])
        self.assertTrue(protocol["topology_fixed"])

        self.assertEqual(self._values(tokens, "--checkpoint_iterations"), ["40000", "50000", "60000"])
        self.assertEqual(self._values(tokens, "--save_iterations"), ["40000", "50000", "60000"])
        self.assertEqual(self._value(tokens, "--thermal_recipe"), "aaai_strict")
        self.assertEqual(self._value(tokens, "--thermal_freeze_mode"), "strict")
        self.assertEqual(self._value(tokens, "--thermal_scale_clamp"), "off")
        self.assertEqual(self._value(tokens, "--thermal_max_sh_degree"), "1")
        self.assertEqual(self._value(tokens, "--thermal_optimizer_state"), "fresh")
        self.assertEqual(self._value(tokens, "--opacity_lr"), "0")
        self.assertNotIn("--clamp_scale_max", tokens)

    def test_continuous_unfrozen_has_nonzero_lrs_and_fixed_topology(self):
        payload, tokens = self._dry_run(
            "--thermal_freeze_mode", "continuous_unfrozen"
        )
        protocol = payload["protocol"]["thermal_stage2"]
        self.assertEqual(protocol["thermal_recipe"], "legacy")
        self.assertEqual(protocol["thermal_freeze_mode"], "continuous_unfrozen")
        self.assertEqual(self._value(tokens, "--thermal_freeze_mode"), "continuous_unfrozen")
        self.assertEqual(protocol["t_unfrozen_position_lr"], 1.6e-6)
        self.assertEqual(protocol["t_unfrozen_scaling_lr"], 0.005)
        self.assertEqual(protocol["t_unfrozen_rotation_lr"], 0.001)
        self.assertEqual(payload["flags"]["t_unfrozen_position_lr"], 1.6e-6)
        self.assertEqual(payload["flags"]["t_unfrozen_scaling_lr"], 0.005)
        self.assertEqual(payload["flags"]["t_unfrozen_rotation_lr"], 0.001)
        self.assertEqual(float(self._value(tokens, "--position_lr_init")), 1.6e-6)
        self.assertEqual(float(self._value(tokens, "--position_lr_final")), 1.6e-6)
        self.assertEqual(float(self._value(tokens, "--scaling_lr")), 0.005)
        self.assertEqual(float(self._value(tokens, "--rotation_lr")), 0.001)

        for flag in (
            "--position_lr_init",
            "--position_lr_final",
            "--scaling_lr",
            "--rotation_lr",
            "--opacity_lr",
            "--feature_lr",
        ):
            with self.subTest(flag=flag):
                self.assertGreater(float(self._value(tokens, flag)), 0.0)
        self.assertEqual(self._value(tokens, "--densify_from_iter"), "999999")
        self.assertEqual(self._value(tokens, "--densify_until_iter"), "0")
        self.assertEqual(self._value(tokens, "--densification_interval"), "999999")
        self.assertEqual(self._value(tokens, "--opacity_reset_interval"), "999999")

    def test_checkpoint_offsets_beyond_endpoint_are_omitted(self):
        self.assertEqual(
            pipeline._resolve_thermal_checkpoint_iterations(
                rgb_iter=30000,
                t_iter=45000,
                offsets=[10000, 20000, 30000],
                use_offsets=True,
            ),
            [40000, 45000],
        )

    def test_strict_recipe_rejects_incompatible_explicit_controls(self):
        cases = (
            ("thermal_optimizer_state", "--thermal_optimizer_state", "restore"),
            ("thermal_freeze_mode", "--thermal_freeze_mode", "continuous_unfrozen"),
            ("thermal_scale_clamp", "--thermal_scale_clamp", "legacy"),
        )
        for attribute, option, value in cases:
            with self.subTest(option=option):
                args = argparse.Namespace(
                    thermal_recipe="aaai_strict",
                    thermal_max_sh_degree=None,
                    thermal_optimizer_state="restore",
                    thermal_freeze_mode="legacy",
                    thermal_scale_clamp="legacy",
                )
                setattr(args, attribute, value)
                with self.assertRaisesRegex(ValueError, "aaai_strict requires"):
                    pipeline._apply_thermal_recipe_defaults(args, [option, value])


if __name__ == "__main__":
    unittest.main()
