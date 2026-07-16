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


def _load_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    selected_names = {
        "_tensor_sha256",
        "_snapshot_opacity_adaptive_freeze_fields",
        "_write_opacity_adaptive_freeze_audit",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "_OPACITY_ADAPTIVE_FREEZE_FIELDS" in names:
                nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in selected_names:
            nodes.append(node)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {"hashlib": hashlib, "json": json, "os": os, "torch": torch}
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace


class OpacityAdaptiveRuntimeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.helpers = _load_helpers()

    @staticmethod
    def _model(opacity=None):
        return SimpleNamespace(
            _xyz=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            _scaling=torch.zeros((2, 3), dtype=torch.float32),
            _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
            get_opacity=(
                torch.tensor([[0.2], [0.8]], dtype=torch.float32)
                if opacity is None
                else opacity
            ),
        )

    def test_opacity_may_change_while_structural_geometry_stays_exact(self):
        model = self._model()
        before = self.helpers["_snapshot_opacity_adaptive_freeze_fields"](model)
        model.get_opacity = torch.tensor([[0.01], [0.99]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temporary:
            payload = self.helpers["_write_opacity_adaptive_freeze_audit"](
                temporary, model, before, 30000, 60000
            )
            self.assertEqual(payload["status"], "passed")
            self.assertEqual(set(payload["frozen_fields"]), {"_xyz", "_scaling", "_rotation"})
            self.assertTrue(payload["activated_opacity"]["finite"])
            self.assertEqual(
                json.loads(
                    (Path(temporary) / "opacity_adaptive_freeze_audit.json").read_text(
                        encoding="utf-8"
                    )
                )["status"],
                "passed",
            )

    def test_structural_change_fails_closed(self):
        model = self._model()
        before = self.helpers["_snapshot_opacity_adaptive_freeze_fields"](model)
        model._scaling[0, 0] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "_scaling"):
                self.helpers["_write_opacity_adaptive_freeze_audit"](
                    temporary, model, before, 30000, 60000
                )
            report = json.loads(
                (Path(temporary) / "opacity_adaptive_freeze_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(report["status"], "failed")

    def test_nonfinite_activated_opacity_fails_closed(self):
        model = self._model(torch.tensor([[float("nan")], [0.5]], dtype=torch.float32))
        before = self.helpers["_snapshot_opacity_adaptive_freeze_fields"](model)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "activated_opacity_nonfinite"):
                self.helpers["_write_opacity_adaptive_freeze_audit"](
                    temporary, model, before, 30000, 60000
                )


if __name__ == "__main__":
    unittest.main()
