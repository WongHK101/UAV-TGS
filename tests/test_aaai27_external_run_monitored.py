from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "aaai27_external" / "run_monitored.py"
SPEC = importlib.util.spec_from_file_location("run_monitored", MODULE_PATH)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_process_receipt_success(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt = mod.run(
        command=[sys.executable, "-c", "print('finite')"],
        cwd=tmp_path,
        receipt_path=receipt_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        poll_seconds=0.01,
    )
    assert receipt["status"] == "SUCCEEDED"
    assert receipt["return_code"] == 0
    assert receipt["wall_time_s"] > 0
    assert (tmp_path / "stdout.log").read_text().strip() == "finite"
    assert json.loads(receipt_path.read_text())["stdout"]["sha256"] == receipt["stdout"]["sha256"]

