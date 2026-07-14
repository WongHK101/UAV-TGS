import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "aaai27_protocol_v1.json"


def _literal_cli_defaults(source_path: Path):
    """Return literal defaults from add_argument calls without importing runtime code."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            try:
                defaults[flag] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
    return defaults


class AaaiProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_protocol_fixture_is_versioned_and_legacy_default(self):
        self.assertEqual(self.protocol["schema"], "uav-tgs-aaai27-protocol-v1")
        self.assertEqual(self.protocol["aaai_stage2_defaults"]["thermal_recipe"], "legacy")
        self.assertIn("gpu-training", self.protocol["deferred_until_pilot_review"])

    def test_locked_terminology_and_claim_boundaries(self):
        terms = self.protocol["terminology"]
        self.assertEqual(terms["cfr"], "Cross-sensor FoV/Resolution Canonicalization")
        self.assertEqual(terms["stage2"], "Geometry-Frozen Radiometric Thermal Transfer")
        self.assertEqual(
            terms["temperature_metric"],
            "TSDK-referenced apparent-temperature consistency",
        )
        self.assertIn(
            "absolute thermometry",
            self.protocol["claim_boundaries"]["temperature_metric_is_not"],
        )

    def test_existing_pipeline_legacy_defaults_are_pinned(self):
        defaults = _literal_cli_defaults(REPO_ROOT / "run_uavfgs_pipeline.py")
        expected = self.protocol["legacy_pipeline_defaults"]
        for name in (
            "rgb_iter",
            "t_iter",
            "t_feature_lr",
            "t_opacity_lr",
            "clamp_scale_max_t",
            "thermal_reset_features",
            "sgf_disable",
        ):
            with self.subTest(name=name):
                self.assertIn("--" + name, defaults)
                self.assertEqual(defaults["--" + name], expected[name])

    def test_radiometry_split_contract_is_explicit(self):
        radiometry = self.protocol["radiometry"]
        self.assertEqual(radiometry["block_frames"], 16)
        self.assertEqual(radiometry["test_block_period"], 8)
        self.assertEqual(radiometry["guard_frames"], 2)
        self.assertEqual(radiometry["range_source"], "train-only")


if __name__ == "__main__":
    unittest.main()
