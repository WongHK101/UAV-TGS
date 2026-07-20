from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aaai27_hold8.collect_phase2_main_results import resolve_cost, summarize, validate_scene_record


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


def test_portable_scene_record_keeps_endpoint_hash_without_endpoint_file() -> None:
    row = {
        "scene": "Plaza",
        "method": "scsp_refit_f3",
        "alias_raw_f3": False,
        "scsp_modified_count": 2,
        "endpoint_sha256": "a" * 64,
        "scsp_manifest_sha256": "b" * 64,
        "reported_method_wall_time_s": 10,
        "batch_execution_wall_time_s": 10,
        "peak_vram_bytes": 20,
        "gaussian_count": 30,
        "model_size_bytes": 40,
        "render_fps": 50,
    }
    from tools.aaai27_hold8.collect_phase2_main_results import METRIC_DIRECTIONS

    row.update({name: 1.0 for name in METRIC_DIRECTIONS})
    assert validate_scene_record(row, "Plaza")["endpoint_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_scene_record(row, "Urban50K")
