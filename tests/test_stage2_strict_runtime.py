from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train.py"


def _load_audit_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    selected_names = {
        "_tensor_sha256",
        "_snapshot_strict_freeze_fields",
        "_write_strict_freeze_audit",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "_STRICT_FREEZE_FIELDS" in names:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected_names:
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"hashlib": hashlib, "json": json, "os": os, "torch": torch}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace


class Stage2StrictRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_audit_helpers()

    @staticmethod
    def _gaussians():
        return SimpleNamespace(
            _xyz=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            _scaling=torch.zeros((2, 3), dtype=torch.float32),
            _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
            _opacity=torch.zeros((2, 1), dtype=torch.float32),
        )

    def test_audit_is_atomic_and_passes_only_for_exactly_unchanged_fields(self):
        model = self._gaussians()
        before = self.helpers["_snapshot_strict_freeze_fields"](model)
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.helpers["_write_strict_freeze_audit"](
                temporary, model, before, 30000, 60000
            )
            report_path = Path(temporary) / "strict_freeze_audit.json"
            self.assertTrue(report_path.is_file())
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["status"],
                "passed",
            )
            self.assertEqual(list(Path(temporary).glob("*.tmp-*")), [])

    def test_audit_writes_failed_report_before_raising(self):
        model = self._gaussians()
        before = self.helpers["_snapshot_strict_freeze_fields"](model)
        model._opacity[0, 0] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "_opacity"):
                self.helpers["_write_strict_freeze_audit"](
                    temporary, model, before, 30000, 60000
                )
            report = json.loads(
                (Path(temporary) / "strict_freeze_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertGreater(report["fields"]["_opacity"]["max_abs_diff"], 0.0)

    def test_train_source_enforces_strict_topology_and_exposure_freeze(self):
        source = TRAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('thermal_freeze_mode in ("strict", "continuous_unfrozen")', source)
        self.assertIn("if (not topology_frozen) and iteration < opt.densify_until_iter:", source)
        self.assertIn("if gaussians.exposure_optimizer is not None:", source)
        self.assertIn("Strict thermal freeze invariant failed", source)
        self.assertIn('"--thermal_recipe", choices=["legacy", "aaai_strict"]', source)
        self.assertIn('"--thermal_scale_clamp", choices=["legacy", "off"]', source)


if __name__ == "__main__":
    unittest.main()
