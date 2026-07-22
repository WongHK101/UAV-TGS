from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.aaai27_final.aggregate_final_tables import (
    BENCHMARK_TO_RESULT,
    REPRESENTATIVE_SCENES,
    _load_benchmarks,
)
from tools.aaai27_final.benchmark_render_only import _order_views


class _View:
    def __init__(self, image_name: str) -> None:
        self.image_name = image_name


def test_formal_camera_order_is_explicit() -> None:
    values = [_View("b.png"), _View("a.JPG")]
    ordered = _order_views(values, ["A.png", "B.png"])
    assert [item.image_name for item in ordered] == ["a.JPG", "b.png"]


def test_benchmark_macro_averages_latency_then_inverts(tmp_path: Path) -> None:
    expected_latency: dict[str, list[float]] = {}
    for method_index, method in enumerate(BENCHMARK_TO_RESULT):
        expected_latency[method] = []
        for scene_index, scene in enumerate(REPRESENTATIVE_SCENES):
            latency = float(1 + method_index * 10 + scene_index)
            expected_latency[method].append(latency)
            target = tmp_path / scene / f"{method}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "method": method,
                        "scene": scene,
                        "scene_result": {
                            "median_ms_per_view": latency,
                            "views_per_s": 1000.0 / latency,
                        },
                        "peak_cuda_allocated_bytes": 1024,
                        "peak_cuda_reserved_bytes": 2048,
                        "view_count": 3,
                        "output_resolution_wh": [4, 3],
                        "inference_dtype": "torch.float32",
                        "source_repository": {"commit": "a" * 40},
                        "passes": [
                            {"pass": index, "elapsed_ns": index * 1000, "ms_per_view": latency}
                            for index in (1, 2, 3)
                        ],
                    }
                ),
                encoding="utf-8",
            )

    raw, macros = _load_benchmarks(tmp_path)
    assert len(raw) == 48
    for benchmark_method, result_method in BENCHMARK_TO_RESULT.items():
        expected = sum(expected_latency[benchmark_method]) / len(REPRESENTATIVE_SCENES)
        assert macros[result_method]["render_latency_ms_per_view"] == pytest.approx(expected)
        assert macros[result_method]["render_fps"] == pytest.approx(1000.0 / expected)


def test_benchmark_matrix_rejects_missing_receipt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected 48"):
        _load_benchmarks(tmp_path)
