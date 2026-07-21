from __future__ import annotations

from tools.aaai27_external.aggregate_phase_b import (
    METHODS,
    SCENES,
    aggregate,
    flatten_metrics,
)


def row(scene: str, method: str, status: str, metrics: dict[str, float]):
    return {
        "scene": scene,
        "method": method,
        "display_name": method,
        "status": status,
        "source": "fixture",
        "metrics": metrics,
    }


def test_failures_are_not_zero_filled_and_rgb_na_does_not_shrink_thermal_macro():
    rows = []
    for method_index, method in enumerate(METHODS):
        for scene_index, scene in enumerate(SCENES):
            status = "OOM" if method == "PhysIR-Splat-SH" and scene == "Orchard" else "SUCCEEDED"
            metrics = {} if status != "SUCCEEDED" else {
                "thermal_psnr": float(method_index + scene_index),
                "temperature_mae_c": float(method_index + 1),
            }
            if method != "Thermal3D-GS" and status == "SUCCEEDED":
                metrics["rgb_psnr"] = float(method_index + scene_index)
            rows.append(row(scene, method, status, metrics))

    summary = aggregate(rows)
    thermal = summary["metric_macros"]["thermal_psnr"]
    assert thermal["common_scenes"] == list(SCENES[:-1])
    assert thermal["available_scene_macro_secondary"]["PhysIR-Splat-SH"]["n_valid"] == 5
    rgb = summary["metric_macros"]["rgb_psnr"]
    assert "Thermal3D-GS" not in rgb["supporting_methods"]
    assert summary["coverage"]["PhysIR-Splat-SH"]["completed_scene_count"] == 5


def test_missing_endpoint_is_explicit():
    summary = aggregate([])
    assert len(summary["completion_matrix"]) == len(METHODS) * len(SCENES)
    assert {item["status"] for item in summary["completion_matrix"]} == {"MISSING"}
    assert not summary["all_30_endpoints_complete"]


def test_efficiency_uses_native_benchmark_or_official_render_receipt():
    native = flatten_metrics(
        {
            "render_benchmark": {
                "render_test_views_per_s": 1.5,
                "render_wall_time_s": 20.0,
            },
            "render": {
                "render_test_views_per_s": 0.5,
                "wall_time_s": 60.0,
                "peak_vram_mib": 1234.0,
            },
        }
    )
    assert native["render_test_views_per_s"] == 1.5
    assert native["render_wall_time_s"] == 20.0
    assert native["render_peak_vram_mib"] == 1234.0

    official = flatten_metrics(
        {
            "render": {
                "render_test_views_per_s": 0.5,
                "wall_time_s": 60.0,
            }
        }
    )
    assert official["render_test_views_per_s"] == 0.5
    assert official["render_wall_time_s"] == 60.0
