from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train.py"


def _load_freeze_helpers():
    tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
    wanted = {"_tensor_sha256", "_snapshot_strict_freeze_fields", "_write_strict_freeze_audit"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "_STRICT_FREEZE_FIELDS": ("_xyz", "_scaling", "_rotation", "_opacity"),
        "hashlib": __import__("hashlib"),
        "json": json,
        "os": __import__("os"),
        "torch": torch,
    }
    exec(compile(module, str(TRAIN_PATH), "exec"), namespace)
    return namespace


class RGBAppearanceRefitFreezeAuditTests(unittest.TestCase):
    @staticmethod
    def _gaussians():
        return SimpleNamespace(
            _xyz=torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
            _scaling=torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            _rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            _opacity=torch.tensor([[0.5]], dtype=torch.float32),
        )

    def test_custom_rgb_refit_audit_passes_and_records_exact_hashes(self):
        helpers = _load_freeze_helpers()
        gaussians = self._gaussians()
        before = helpers["_snapshot_strict_freeze_fields"](gaussians)
        with tempfile.TemporaryDirectory() as directory:
            payload = helpers["_write_strict_freeze_audit"](
                directory,
                gaussians,
                before,
                30000,
                35000,
                schema="uav-tgs-rgb-appearance-refit-freeze-audit-v1",
                filename="rgb_appearance_refit_freeze_audit.json",
                label="RGBAppearanceRefitFreezeAudit",
            )
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(all(item["unchanged"] for item in payload["fields"].values()))
            saved = json.loads(
                (Path(directory) / "rgb_appearance_refit_freeze_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                saved["schema"], "uav-tgs-rgb-appearance-refit-freeze-audit-v1"
            )

    def test_custom_rgb_refit_audit_fails_closed_on_geometry_change(self):
        helpers = _load_freeze_helpers()
        gaussians = self._gaussians()
        before = helpers["_snapshot_strict_freeze_fields"](gaussians)
        gaussians._scaling[0, 0] += 0.01
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "_scaling"):
                helpers["_write_strict_freeze_audit"](
                    directory,
                    gaussians,
                    before,
                    30000,
                    35000,
                    schema="uav-tgs-rgb-appearance-refit-freeze-audit-v1",
                    filename="rgb_appearance_refit_freeze_audit.json",
                    label="RGBAppearanceRefitFreezeAudit",
                )


if __name__ == "__main__":
    unittest.main()
