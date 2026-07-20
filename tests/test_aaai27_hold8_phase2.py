from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aaai27_hold8.collect_phase2_main_results import resolve_cost, summarize


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_phase2_modified_cost_includes_projection(tmp_path: Path) -> None:
    method = tmp_path / "method"
    raw = tmp_path / "raw"
    device = {"peak_torch_reserved_bytes": 10}
    _write(method / "efficiency/scsp_projection.json", {"wall_time_s": 2})
    _write(method / "efficiency/rgb_refit.json", {"wall_time_s": 3, "device": device})
    _write(method / "efficiency/f3_train.json", {"wall_time_s": 5, "device": {"peak_torch_reserved_bytes": 20}})
    result = resolve_cost(method, raw, alias=False)
    assert result["reported_method_wall_time_s"] == 10
    assert result["peak_vram_bytes"] == 20


def test_phase2_noop_alias_inherits_raw_cost(tmp_path: Path) -> None:
    method = tmp_path / "method"
    raw = tmp_path / "raw"
    _write(method / "efficiency/scsp_projection.json", {"wall_time_s": 2})
    _write(raw / "efficiency/train.json", {"wall_time_s": 5, "device": {"peak_torch_reserved_bytes": 20}})
    result = resolve_cost(method, raw, alias=True)
    assert result["batch_execution_wall_time_s"] == 7
    assert result["reported_method_wall_time_s"] == 5


def test_phase2_summary_requires_all_scenes() -> None:
    with pytest.raises(ValueError, match="completeness"):
        summarize([])
