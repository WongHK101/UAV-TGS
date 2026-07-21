#!/usr/bin/env python3
"""Merge Building qualification and Phase-B host summaries into the final 5x6 table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any


SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)
METHODS = (
    "ThermalGaussian-OMMG",
    "MMOne",
    "Thermal3D-GS",
    "ThermoNeRF",
    "PhysIR-Splat-SH",
)
DISPLAY_NAMES = {
    "PhysIR-Splat-SH": "PhysIR-Splat-SH†",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def canonical_method(value: str) -> str:
    if value in {
        "PhysIR-Splat",
        "PhysIR-Splat-default-thermal-SH-no-opacity-reset",
        "PhysIR-Splat-SH†",
    }:
        return "PhysIR-Splat-SH"
    if value not in METHODS:
        raise ValueError(f"unknown method: {value}")
    return value


def finite(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def flatten_metrics(row: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}

    def add(name: str, value: Any) -> None:
        parsed = finite(value)
        if parsed is not None:
            result[name] = parsed

    appearance = row.get("appearance") or {}
    for modality in ("rgb", "thermal"):
        source = appearance.get(modality) or row.get(modality) or {}
        for metric in ("psnr", "ssim", "lpips"):
            add(f"{modality}_{metric}", source.get(metric))
    temperature = row.get("temperature") or {}
    for source, target in (
        ("mae_c", "temperature_mae_c"),
        ("rmse_c", "temperature_rmse_c"),
        ("p95_abs_error_c", "temperature_p95_abs_error_c"),
        ("signed_bias_c", "temperature_signed_bias_c"),
    ):
        add(target, temperature.get(source))
    hotspot = row.get("hotspot") or {}
    add("hotspot_auprc", hotspot.get("auprc", row.get("hotspot_auprc")))

    geometry = row.get("geometry") or row.get("expected_depth") or {}
    add(
        "depth_median_absolute_error_m",
        geometry.get("median_absolute_depth_error_m", geometry.get("median_absolute_error_m")),
    )
    add("depth_missing_rate", geometry.get("missing_rate"))
    thresholds = geometry.get("thresholds_m") or {}
    if thresholds:
        for threshold in ("1.0", "2.0", "5.0"):
            entry = thresholds.get(threshold) or thresholds.get(str(int(float(threshold)))) or {}
            add(f"depth_front_{threshold}m", entry.get("front"))
            add(f"depth_agreement_{threshold}m", entry.get("agreement"))
    else:
        for metric, values in (
            ("front", geometry.get("front_1_2_5m")),
            ("agreement", geometry.get("agreement_1_2_5m")),
        ):
            if values:
                for threshold, value in zip(("1.0", "2.0", "5.0"), values, strict=True):
                    add(f"depth_{metric}_{threshold}m", value)

    training = row.get("training") or {}
    add(
        "train_wall_time_s",
        training.get(
            "wall_time_s",
            row.get("train_wall_time_s", row.get("train_and_official_eval_wall_time_s")),
        ),
    )
    add(
        "train_peak_vram_mib",
        training.get(
            "peak_vram_mib",
            row.get("train_peak_vram_mib", row.get("train_and_official_eval_peak_vram_mib")),
        ),
    )
    benchmark = row.get("render_benchmark") or {}
    render = row.get("render") or {}
    add(
        "render_test_views_per_s",
        benchmark.get(
            "render_test_views_per_s",
            render.get(
                "render_test_views_per_s",
                row.get("render_test_views_per_s", row.get("render_only_test_views_per_s")),
            ),
        ),
    )
    add(
        "render_wall_time_s",
        benchmark.get(
            "render_wall_time_s",
            render.get(
                "wall_time_s",
                row.get("render_wall_time_s", row.get("render_only_wall_time_s")),
            ),
        ),
    )
    add(
        "render_peak_vram_mib",
        render.get(
            "peak_vram_mib",
            row.get("render_peak_vram_mib", row.get("render_only_peak_vram_mib")),
        ),
    )
    model = row.get("model") or {}
    add("gaussian_count", model.get("gaussian_count", row.get("gaussian_count")))
    add(
        "model_size_bytes",
        model.get("artifact_bytes_excluding_render_outputs", row.get("model_size_bytes")),
    )
    return result


def building_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for source in payload["methods"]:
        method = canonical_method(str(source["method"]))
        rows.append(
            {
                "scene": "Building",
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "status": "SUCCEEDED"
                if str(source.get("status", "")).startswith("PASSED_")
                else str(source.get("status", "FAILED")),
                "source": "phase_a_locked_building",
                "metrics": flatten_metrics(source),
            }
        )
    return rows


def phase_b_rows(payload: dict[str, Any], host: str) -> list[dict[str, Any]]:
    rows = []
    for source in payload["endpoints"]:
        method = canonical_method(str(source["method"]))
        training_status = (source.get("training") or {}).get("status")
        rows.append(
            {
                "scene": str(source["scene"]),
                "method": method,
                "display_name": DISPLAY_NAMES.get(method, method),
                "status": str(source.get("status", "MISSING")),
                "evaluation_status": source.get("evaluation_status"),
                "training_status": training_status,
                "source": host,
                "metrics": flatten_metrics(source)
                if source.get("status") == "SUCCEEDED"
                and source.get("evaluation_status") == "SUCCEEDED"
                else {},
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["scene"], row["method"])
        if key in index:
            raise ValueError(f"duplicate endpoint: {key}")
        index[key] = row
    matrix = []
    for method in METHODS:
        for scene in SCENES:
            row = index.get((scene, method))
            if row is None:
                row = {
                    "scene": scene,
                    "method": method,
                    "display_name": DISPLAY_NAMES.get(method, method),
                    "status": "MISSING",
                    "source": None,
                    "metrics": {},
                }
            matrix.append(row)

    coverage: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in matrix if row["method"] == method]
        completed = [row["scene"] for row in selected if row["status"] == "SUCCEEDED"]
        coverage[method] = {
            "completed_scene_count": len(completed),
            "formal_scene_count": len(SCENES),
            "completed_scenes": completed,
            "incomplete": [
                {"scene": row["scene"], "status": row["status"]}
                for row in selected
                if row["status"] != "SUCCEEDED"
            ],
        }

    metric_names = sorted(
        {metric for row in matrix for metric in row.get("metrics", {})}
    )
    metric_macros: dict[str, Any] = {}
    for metric in metric_names:
        supporting = [
            method
            for method in METHODS
            if any(
                metric in row["metrics"]
                for row in matrix
                if row["method"] == method
            )
        ]
        common_scenes = [
            scene
            for scene in SCENES
            if all(
                metric in index.get((scene, method), {"metrics": {}})["metrics"]
                for method in supporting
            )
        ] if supporting else []
        available: dict[str, Any] = {}
        common: dict[str, float | None] = {}
        for method in METHODS:
            values = [
                (row["scene"], row["metrics"][metric])
                for row in matrix
                if row["method"] == method and metric in row["metrics"]
            ]
            available[method] = {
                "n_valid": len(values),
                "scenes": [scene for scene, _ in values],
                "scene_equal_macro": statistics.fmean(value for _, value in values)
                if values else None,
            }
            common_values = (
                [index[(scene, method)]["metrics"][metric] for scene in common_scenes]
                if method in supporting
                else []
            )
            common[method] = statistics.fmean(common_values) if common_values else None
        metric_macros[metric] = {
            "supporting_methods": supporting,
            "common_scene_count": len(common_scenes),
            "common_scenes": common_scenes,
            "common_scene_equal_macro": common,
            "available_scene_macro_secondary": available,
        }

    return {
        "schema": "uav-tgs-aaai27-external-phase-b-final-summary-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": "aaai27_hold8_v2",
        "representative_scenes": list(SCENES),
        "methods": list(METHODS),
        "display_names": DISPLAY_NAMES,
        "geometry_claim_boundary": "OpenMVS-referenced held-out expected-depth consistency",
        "physir_claim_boundary": (
            "default thermal SH with opacity reset disabled; not the full physical renderer "
            "and not PhysIR-Splat+VGGT-IR"
        ),
        "completion_matrix": matrix,
        "coverage": coverage,
        "metric_macros": metric_macros,
        "all_30_endpoints_complete": all(
            row["status"] == "SUCCEEDED" for row in matrix
        ),
        "next_state": "WAITING_GPT",
        "prohibited_next_task": "final_paper_aggregation",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-summary", type=Path, required=True)
    parser.add_argument("--host-summary", type=Path, action="append", required=True)
    parser.add_argument("--host-label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.host_summary) != len(args.host_label):
        parser.error("--host-summary and --host-label counts must match")
    rows = building_rows(load_json(args.building_summary))
    for path, label in zip(args.host_summary, args.host_label, strict=True):
        rows.extend(phase_b_rows(load_json(path), label))
    result = aggregate(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
