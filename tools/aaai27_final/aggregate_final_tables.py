#!/usr/bin/env python3
"""Merge frozen AAAI27 results and emit the three formal result tables."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Sequence


SCHEMA = "uav-tgs-aaai27-final-aggregation-v1"
REPRESENTATIVE_SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)
ALL_SCENES = REPRESENTATIVE_SCENES + (
    "Garden",
    "Plaza",
    "Road",
    "Urban50K",
    "Urban100K",
)
INTERNAL_METHODS = (
    "raw_f3",
    "scsp_refit_f3",
    "adaptive_opacity_scale_clamp",
)
EXTERNAL_METHODS = (
    "ThermalGaussian-OMMG",
    "MMOne",
    "Thermal3D-GS",
    "ThermoNeRF",
    "PhysIR-Splat-SH",
)
DISPLAY_NAMES = {
    "raw_f3": "Raw-F3",
    "scsp_refit_f3": "SCSP-Refit+F3",
    "adaptive_opacity_scale_clamp": "Adaptive Opacity+Scale-Clamp",
    "ours_full": "Ours-Full",
    "ours_adapt": "Ours-Adapt",
    "ThermalGaussian-OMMG": "ThermalGaussian-OMMG",
    "MMOne": "MMOne",
    "Thermal3D-GS": "Thermal3D-GS",
    "ThermoNeRF": "ThermoNeRF",
    "PhysIR-Splat-SH": "PhysIR-Splat-SH†",
}
BENCHMARK_TO_RESULT = {
    "raw_f3": "raw_f3",
    "scsp_refit_f3": "scsp_refit_f3",
    "adaptive_opacity_scale_clamp": "adaptive_opacity_scale_clamp",
    "thermalgaussian_ommg": "ThermalGaussian-OMMG",
    "mmone": "MMOne",
    "thermal3dgs": "Thermal3D-GS",
    "thermonerf": "ThermoNeRF",
    "physir_splat_sh": "PhysIR-Splat-SH",
}

PRIMARY_METRICS = (
    "rgb_psnr",
    "rgb_ssim",
    "rgb_lpips",
    "thermal_psnr",
    "thermal_ssim",
    "thermal_lpips",
    "temperature_mae_c",
    "temperature_rmse_c",
    "hotspot_auprc",
    "depth_front_1m",
    "depth_front_2m",
    "depth_front_5m",
    "depth_agreement_1m",
    "depth_agreement_2m",
    "depth_agreement_5m",
    "depth_median_abs_error_m",
    "depth_missing_rate",
)
SUPPLEMENTARY_METRICS = (
    "temperature_bias_c",
    "temperature_p95_c",
    "hotspot_iou",
    "off_lut_mean_rgb",
    "off_lut_p95_rgb",
    "depth_front_0p25m",
    "depth_front_0p5m",
    "depth_front_10m",
    "depth_front_15m",
    "depth_front_20m",
    "depth_agreement_0p25m",
    "depth_agreement_0p5m",
    "depth_agreement_10m",
    "depth_agreement_15m",
    "depth_agreement_20m",
)
EFFICIENCY_FIELDS = (
    "reported_method_wall_time_s",
    "train_peak_vram_bytes",
    "model_size_bytes",
    "gaussian_count",
)

LABELS = {
    "scene": "Scene",
    "display_name": "Method",
    "rgb_psnr": "RGB PSNR↑",
    "rgb_ssim": "RGB SSIM↑",
    "rgb_lpips": "RGB LPIPS↓",
    "thermal_psnr": "T PSNR↑",
    "thermal_ssim": "T SSIM↑",
    "thermal_lpips": "T LPIPS↓",
    "temperature_mae_c": "Temp MAE (°C)↓",
    "temperature_rmse_c": "Temp RMSE (°C)↓",
    "hotspot_auprc": "Hotspot AUPRC↑",
    "depth_front_1m": "Front@1m↓",
    "depth_front_2m": "Front@2m↓",
    "depth_front_5m": "Front@5m↓",
    "depth_agreement_1m": "Agree@1m↑",
    "depth_agreement_2m": "Agree@2m↑",
    "depth_agreement_5m": "Agree@5m↑",
    "depth_median_abs_error_m": "Depth MedAE (m)↓",
    "depth_missing_rate": "Missing↓",
    "temperature_bias_c": "Temp bias (°C)",
    "temperature_p95_c": "Temp P95 (°C)↓",
    "hotspot_iou": "Hotspot IoU↑",
    "off_lut_mean_rgb": "Off-LUT mean↓",
    "off_lut_p95_rgb": "Off-LUT P95↓",
    "reported_method_wall_time_s": "Training time (s)↓",
    "train_peak_vram_gib": "Train VRAM (GiB)↓",
    "model_size_mib": "Model (MiB)↓",
    "gaussian_count_m": "Gaussians (M)↓",
    "render_latency_ms_per_view": "Render latency (ms/view)↓",
    "render_fps": "FPS↑",
    "inference_peak_allocated_gib": "Inference VRAM (GiB)↓",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for value in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(value)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    if value is None or value == "" or value == "N/A":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(valid) if valid else None


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _canonical_internal(
    row: dict[str, str], cost: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scene": row["scene"],
        "method": row["method"],
        "display_name": DISPLAY_NAMES[row["method"]],
        "status": "SUCCEEDED",
        "source": "phase1_hold8_v2",
    }
    for key in PRIMARY_METRICS + SUPPLEMENTARY_METRICS:
        value[key] = _float(row.get(key))
    record = cost[(row["scene"], row["method"])]
    value.update(
        {
            "reported_method_wall_time_s": _float(record.get("reported_method_wall_time_s")),
            "train_peak_vram_bytes": _float(record.get("peak_vram_bytes")),
            "model_size_bytes": _float(record.get("model_size_bytes")),
            "gaussian_count": _float(record.get("gaussian_count")),
            "alias": str(row.get("alias", "False")).casefold() == "true",
        }
    )
    return value


def _canonical_phase2(row: dict[str, str]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "scene": row["scene"],
        "method": "scsp_refit_f3",
        "display_name": DISPLAY_NAMES["scsp_refit_f3"],
        "status": "SUCCEEDED",
        "source": "phase2_hold8_v2",
    }
    for key in PRIMARY_METRICS + SUPPLEMENTARY_METRICS:
        value[key] = _float(row.get(key))
    value.update(
        {
            "reported_method_wall_time_s": _float(row.get("reported_method_wall_time_s")),
            "train_peak_vram_bytes": _float(row.get("peak_vram_bytes")),
            "model_size_bytes": _float(row.get("model_size_bytes")),
            "gaussian_count": _float(row.get("gaussian_count")),
            "alias": str(row.get("alias_raw_f3", "False")).casefold() == "true",
            "endpoint_sha256": row.get("endpoint_sha256"),
            "scsp_modified_count": _float(row.get("scsp_modified_count")),
        }
    )
    return value


def _canonical_external(row: dict[str, str]) -> dict[str, Any]:
    mapping = {
        "depth_front_1m": "depth_front_1.0m",
        "depth_front_2m": "depth_front_2.0m",
        "depth_front_5m": "depth_front_5.0m",
        "depth_agreement_1m": "depth_agreement_1.0m",
        "depth_agreement_2m": "depth_agreement_2.0m",
        "depth_agreement_5m": "depth_agreement_5.0m",
        "depth_median_abs_error_m": "depth_median_absolute_error_m",
        "temperature_bias_c": "temperature_signed_bias_c",
        "temperature_p95_c": "temperature_p95_abs_error_c",
        "reported_method_wall_time_s": "train_wall_time_s",
    }
    value: dict[str, Any] = {
        "scene": row["scene"],
        "method": row["method"],
        "display_name": DISPLAY_NAMES[row["method"]],
        "status": row["status"],
        "source": row["source"],
    }
    for key in PRIMARY_METRICS + SUPPLEMENTARY_METRICS:
        value[key] = _float(row.get(mapping.get(key, key)))
    value.update(
        {
            "reported_method_wall_time_s": _float(row.get("train_wall_time_s")),
            "train_peak_vram_bytes": (
                _float(row.get("train_peak_vram_mib")) * 1024**2
                if _float(row.get("train_peak_vram_mib")) is not None
                else None
            ),
            "model_size_bytes": _float(row.get("model_size_bytes")),
            "gaussian_count": _float(row.get("gaussian_count")),
            "alias": False,
        }
    )
    return value


def _load_benchmarks(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    receipts = sorted(root.glob("*/*.json"))
    if len(receipts) != 8 * len(REPRESENTATIVE_SCENES):
        raise ValueError(f"expected 48 benchmark receipts, found {len(receipts)} under {root}")
    raw: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str]] = set()
    for path in receipts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"incomplete benchmark receipt: {path}")
        method = str(payload["method"])
        scene = str(payload["scene"])
        key = (method, scene)
        if key in seen:
            raise ValueError(f"duplicate benchmark receipt: {key}")
        seen.add(key)
        if scene not in REPRESENTATIVE_SCENES or method not in BENCHMARK_TO_RESULT:
            raise ValueError(f"unexpected benchmark key: {key}")
        item = {
            "benchmark_method": method,
            "result_method": BENCHMARK_TO_RESULT[method],
            "scene": scene,
            "median_ms_per_view": float(payload["scene_result"]["median_ms_per_view"]),
            "views_per_s": float(payload["scene_result"]["views_per_s"]),
            "peak_cuda_allocated_bytes": float(payload["peak_cuda_allocated_bytes"]),
            "peak_cuda_reserved_bytes": float(payload["peak_cuda_reserved_bytes"]),
            "view_count": int(payload["view_count"]),
            "output_resolution_wh": payload["output_resolution_wh"],
            "inference_dtype": payload["inference_dtype"],
            "source_commit": payload["source_repository"]["commit"],
            "gpu_name": payload["gpu"]["name"],
            "gpu_total_memory_bytes": int(payload["gpu"]["total_memory_bytes"]),
            "torch_version": payload["gpu"]["torch_version"],
            "cuda_version": payload["gpu"]["cuda_version"],
            "wrapper_sha256": payload["benchmark_wrapper"]["sha256"],
            "receipt": str(path.resolve()),
            "receipt_sha256": _sha256(path),
        }
        for pass_value in payload["passes"]:
            item[f"pass_{pass_value['pass']}_elapsed_ns"] = int(pass_value["elapsed_ns"])
            item[f"pass_{pass_value['pass']}_ms_per_view"] = float(pass_value["ms_per_view"])
        raw.append(item)
        by_method.setdefault(item["result_method"], []).append(item)
    expected = {(method, scene) for method in BENCHMARK_TO_RESULT for scene in REPRESENTATIVE_SCENES}
    if seen != expected:
        raise ValueError(f"benchmark matrix mismatch: missing={sorted(expected-seen)} extra={sorted(seen-expected)}")
    macros: dict[str, dict[str, float]] = {}
    for method, values in by_method.items():
        latency = statistics.fmean(value["median_ms_per_view"] for value in values)
        macros[method] = {
            "render_latency_ms_per_view": latency,
            "render_fps": 1000.0 / latency,
            "inference_peak_allocated_bytes_scene_mean": statistics.fmean(
                value["peak_cuda_allocated_bytes"] for value in values
            ),
            "inference_peak_allocated_bytes_max": max(
                value["peak_cuda_allocated_bytes"] for value in values
            ),
            "inference_peak_reserved_bytes_scene_mean": statistics.fmean(
                value["peak_cuda_reserved_bytes"] for value in values
            ),
        }
    return raw, macros


def _macro(rows: Sequence[dict[str, Any]], method: str, display: str) -> dict[str, Any]:
    if len(rows) != len(REPRESENTATIVE_SCENES) or {row["scene"] for row in rows} != set(REPRESENTATIVE_SCENES):
        raise ValueError(f"six-scene macro input is incomplete for {method}")
    value: dict[str, Any] = {"method": method, "display_name": display, "scene_count": len(rows)}
    for field in PRIMARY_METRICS + SUPPLEMENTARY_METRICS + EFFICIENCY_FIELDS:
        value[field] = _mean(row.get(field) for row in rows)
    return value


def _enrich_units(value: dict[str, Any], benchmark: dict[str, float] | None = None) -> dict[str, Any]:
    result = dict(value)
    vram = result.get("train_peak_vram_bytes")
    size = result.get("model_size_bytes")
    count = result.get("gaussian_count")
    result["train_peak_vram_gib"] = vram / 1024**3 if vram is not None else None
    result["model_size_mib"] = size / 1024**2 if size is not None else None
    result["gaussian_count_m"] = count / 1_000_000 if count is not None else None
    per_scene_inference = result.get("inference_peak_allocated_bytes")
    if per_scene_inference is not None:
        result["inference_peak_allocated_gib"] = per_scene_inference / 1024**3
    if benchmark is not None:
        result.update(benchmark)
        result["inference_peak_allocated_gib"] = (
            benchmark["inference_peak_allocated_bytes_scene_mean"] / 1024**3
        )
    return result


def _format(value: Any, field: str) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    if field in {"scene_count"}:
        return str(int(value))
    if field in {"rgb_psnr", "thermal_psnr", "render_latency_ms_per_view", "render_fps"}:
        return f"{value:.2f}"
    if field in {"reported_method_wall_time_s"}:
        return f"{value:.1f}"
    if field in {"gaussian_count_m", "model_size_mib", "train_peak_vram_gib", "inference_peak_allocated_gib"}:
        return f"{value:.3f}"
    if field.startswith("temperature_") or field == "depth_median_abs_error_m":
        return f"{value:.3f}"
    return f"{value:.4f}"


def _csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    temporary.replace(path)


def _markdown(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    headers = [LABELS.get(field, field) for field in fields]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format(row.get(field), field) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "†": r"$^{\dagger}$",
        "↑": r"$\uparrow$",
        "↓": r"$\downarrow$",
        "°": r"$^{\circ}$",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    align = "l" + "r" * (len(fields) - 1)
    headers = " & ".join(_latex_escape(LABELS.get(field, field)) for field in fields)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [_latex_escape(_format(row.get(field), field)) for field in fields]
        lines.append(" & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    return "\n".join(lines)


def _write_table(root: Path, name: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    _csv(root / f"{name}.csv", rows, fields)
    _atomic_text(root / f"{name}.md", _markdown(rows, fields))
    _atomic_text(root / f"{name}.tex", _latex(rows, fields))


def aggregate(args: argparse.Namespace) -> Path:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"refusing non-empty aggregation directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    phase1_raw = _read_csv(args.phase1_metrics)
    phase1_cost_list = json.loads(args.phase1_cost.read_text(encoding="utf-8"))
    phase1_cost = {(item["scene"], item["method"]): item for item in phase1_cost_list}
    phase1 = [_canonical_internal(row, phase1_cost) for row in phase1_raw]
    if len(phase1) != 18:
        raise ValueError(f"Phase 1 must contain 18 rows, found {len(phase1)}")

    phase2 = [_canonical_phase2(row) for row in _read_csv(args.phase2_metrics)]
    if len(phase2) != 11 or {row["scene"] for row in phase2} != set(ALL_SCENES):
        raise ValueError("Phase 2 11-scene SCSP table is incomplete")
    # Preserve Phase-1 supplementary fields for the six reused endpoints.
    phase1_scsp = {(row["scene"], row["method"]): row for row in phase1}
    for row in phase2:
        prior = phase1_scsp.get((row["scene"], "scsp_refit_f3"))
        if prior is not None:
            for field in SUPPLEMENTARY_METRICS:
                if row.get(field) is None:
                    row[field] = prior.get(field)

    external = [_canonical_external(row) for row in _read_csv(args.external_metrics)]
    if len(external) != 30 or any(row["status"] != "SUCCEEDED" for row in external):
        raise ValueError("External Phase B must contain 30 successful rows")

    benchmark_raw, benchmark_macros = _load_benchmarks(args.benchmark_root)
    benchmark_by_key = {
        (row["result_method"], row["scene"]): row for row in benchmark_raw
    }
    for row in phase1 + phase2 + external:
        benchmark = benchmark_by_key.get((row["method"], row["scene"]))
        if benchmark is not None:
            row["render_latency_ms_per_view"] = benchmark["median_ms_per_view"]
            row["render_fps"] = benchmark["views_per_s"]
            row["inference_peak_allocated_bytes"] = benchmark["peak_cuda_allocated_bytes"]

    # Phase 2 is the authoritative 11-scene SCSP source.  Do not duplicate its
    # six representative-scene rows from Phase 1 in the merged evidence table.
    merged = phase2 + [row for row in phase1 if row["method"] != "scsp_refit_f3"] + external
    merged_keys = [(row["method"], row["scene"]) for row in merged]
    if len(merged_keys) != len(set(merged_keys)):
        raise ValueError("merged per-scene evidence contains duplicate method/scene rows")
    merged_fields = (
        "scene",
        "method",
        "display_name",
        "status",
        "source",
        *PRIMARY_METRICS,
        *SUPPLEMENTARY_METRICS,
        *EFFICIENCY_FIELDS,
        "render_latency_ms_per_view",
        "render_fps",
        "inference_peak_allocated_bytes",
        "alias",
    )
    _csv(args.output / "merged_per_scene_results.csv", merged, merged_fields)
    _atomic_json(
        args.output / "merged_per_scene_results.json",
        {"schema": SCHEMA, "created_at": _now(), "rows": merged},
    )

    table1_rows = [_enrich_units(row) for row in phase2]
    table1_macro: dict[str, Any] = {"scene": "Scene-equal macro", "display_name": "SCSP-Refit+F3"}
    for field in PRIMARY_METRICS + EFFICIENCY_FIELDS:
        table1_macro[field] = _mean(row.get(field) for row in phase2)
    table1_rows.append(_enrich_units(table1_macro))

    table2_sources: list[tuple[str, str, list[dict[str, Any]]]] = [
        ("ours_full", "Ours-Full", [row for row in phase1 if row["method"] == "scsp_refit_f3"]),
        ("ours_adapt", "Ours-Adapt", [row for row in phase1 if row["method"] == "adaptive_opacity_scale_clamp"]),
    ]
    table2_sources += [
        (method, DISPLAY_NAMES[method], [row for row in external if row["method"] == method])
        for method in EXTERNAL_METHODS
    ]
    table2_rows: list[dict[str, Any]] = []
    for table_method, display, rows in table2_sources:
        source_method = (
            "scsp_refit_f3" if table_method == "ours_full"
            else "adaptive_opacity_scale_clamp" if table_method == "ours_adapt"
            else table_method
        )
        table2_rows.append(_enrich_units(_macro(rows, table_method, display), benchmark_macros[source_method]))

    table3_rows: list[dict[str, Any]] = []
    for method in INTERNAL_METHODS:
        rows = [row for row in phase1 if row["method"] == method]
        table3_rows.append(
            _enrich_units(_macro(rows, method, DISPLAY_NAMES[method]), benchmark_macros[method])
        )

    table1_fields = (
        "scene",
        *PRIMARY_METRICS,
        "reported_method_wall_time_s",
        "train_peak_vram_gib",
        "gaussian_count_m",
    )
    comparison_fields = (
        "display_name",
        *PRIMARY_METRICS,
        "reported_method_wall_time_s",
        "train_peak_vram_gib",
        "model_size_mib",
        "gaussian_count_m",
        "render_latency_ms_per_view",
        "render_fps",
        "inference_peak_allocated_gib",
    )
    _write_table(args.output, "table1_scsp_11scene", table1_rows, table1_fields)
    _write_table(args.output, "table2_external_sixscene_macro", table2_rows, comparison_fields)
    _write_table(args.output, "table3_internal_ablation_sixscene_macro", table3_rows, comparison_fields)

    full_macro_rows: list[dict[str, Any]] = []
    for method in INTERNAL_METHODS:
        rows = [row for row in phase1 if row["method"] == method]
        full_macro_rows.append(_enrich_units(_macro(rows, method, DISPLAY_NAMES[method]), benchmark_macros[method]))
    for method in EXTERNAL_METHODS:
        rows = [row for row in external if row["method"] == method]
        full_macro_rows.append(_enrich_units(_macro(rows, method, DISPLAY_NAMES[method]), benchmark_macros[method]))
    supplementary_fields = comparison_fields + SUPPLEMENTARY_METRICS
    _write_table(
        args.output,
        "supplementary_sixscene_full_metric_macro",
        full_macro_rows,
        supplementary_fields,
    )
    supplementary_per_scene_rows = [
        _enrich_units(row) for row in phase1 + external
    ]
    supplementary_per_scene_fields = (
        "scene",
        "display_name",
        *PRIMARY_METRICS,
        *SUPPLEMENTARY_METRICS,
        "reported_method_wall_time_s",
        "train_peak_vram_gib",
        "model_size_mib",
        "gaussian_count_m",
        "render_latency_ms_per_view",
        "render_fps",
        "inference_peak_allocated_gib",
    )
    _write_table(
        args.output,
        "supplementary_sixscene_full_metric_per_scene",
        supplementary_per_scene_rows,
        supplementary_per_scene_fields,
    )

    benchmark_fields = (
        "result_method",
        "scene",
        "median_ms_per_view",
        "views_per_s",
        "pass_1_elapsed_ns",
        "pass_1_ms_per_view",
        "pass_2_elapsed_ns",
        "pass_2_ms_per_view",
        "pass_3_elapsed_ns",
        "pass_3_ms_per_view",
        "peak_cuda_allocated_bytes",
        "peak_cuda_reserved_bytes",
        "view_count",
        "output_resolution_wh",
        "inference_dtype",
        "source_commit",
        "gpu_name",
        "gpu_total_memory_bytes",
        "torch_version",
        "cuda_version",
        "wrapper_sha256",
        "receipt_sha256",
    )
    _csv(args.output / "render_benchmark_per_scene_raw.csv", benchmark_raw, benchmark_fields)
    benchmark_macro_rows = []
    for method, value in benchmark_macros.items():
        benchmark_macro_rows.append(
            {
                "method": method,
                "display_name": DISPLAY_NAMES[method],
                **value,
                "inference_peak_allocated_gib": value[
                    "inference_peak_allocated_bytes_scene_mean"
                ] / 1024**3,
            }
        )
    benchmark_macro_fields = (
        "display_name",
        "render_latency_ms_per_view",
        "render_fps",
        "inference_peak_allocated_gib",
        "inference_peak_allocated_bytes_max",
        "inference_peak_reserved_bytes_scene_mean",
    )
    _write_table(args.output, "render_benchmark_scene_equal_macro", benchmark_macro_rows, benchmark_macro_fields)

    gpu_names = {row["gpu_name"] for row in benchmark_raw}
    gpu_memory = {row["gpu_total_memory_bytes"] for row in benchmark_raw}
    wrappers = {row["wrapper_sha256"] for row in benchmark_raw}
    if len(gpu_names) != 1 or len(gpu_memory) != 1 or len(wrappers) != 1:
        raise ValueError(
            f"unified benchmark provenance differs: gpu={gpu_names} memory={gpu_memory} wrappers={wrappers}"
        )
    runtime_rows = []
    for method in BENCHMARK_TO_RESULT.values():
        values = [row for row in benchmark_raw if row["result_method"] == method]
        runtime_rows.append(
            {
                "method": DISPLAY_NAMES[method],
                "source_commit": values[0]["source_commit"],
                "torch_versions": ", ".join(sorted({row["torch_version"] for row in values})),
                "cuda_versions": ", ".join(sorted({str(row["cuda_version"]) for row in values})),
                "inference_dtypes": ", ".join(sorted({row["inference_dtype"] for row in values})),
                "output_resolutions_wh": ", ".join(
                    sorted({"x".join(map(str, row["output_resolution_wh"])) for row in values})
                ),
            }
        )
    scope_payload = {
        "schema": "uav-tgs-aaai27-render-only-scope-hardware-v1",
        "host": "AutoDL 900",
        "gpu_name": next(iter(gpu_names)),
        "gpu_total_memory_bytes": next(iter(gpu_memory)),
        "clean_process_per_method_scene": True,
        "batch_size": 1,
        "warmup_full_test_passes": 1,
        "timed_full_test_passes": 3,
        "scene_result": "median of three complete-pass ms/view values",
        "macro_latency": "arithmetic mean of six scene median ms/view values",
        "macro_fps": "1000 / macro latency; per-scene FPS is not averaged",
        "wrapper_sha256": next(iter(wrappers)),
        "runtime_rows": runtime_rows,
    }
    _atomic_json(args.output / "benchmark_scope_hardware.json", scope_payload)
    scope_lines = [
        "# Unified render-only benchmark scope and hardware",
        "",
        f"- Host: AutoDL 900",
        f"- GPU: {scope_payload['gpu_name']} ({scope_payload['gpu_total_memory_bytes'] / 1024**3:.2f} GiB)",
        "- Scope: thermal render-only end-to-end in-memory latency",
        "- One clean process per method × scene; batch size 1",
        "- One full-test-list warm-up, followed by three synchronized full passes",
        "- Scene value: median ms/view; macro latency: arithmetic scene mean; macro FPS: reciprocal of macro latency",
        "- Excluded: model/data loading, GT reads, file encoding/saving, GPU-to-CPU copy and all metric/depth postprocessing",
        "",
        _markdown(
            runtime_rows,
            ("method", "source_commit", "torch_versions", "cuda_versions", "inference_dtypes", "output_resolutions_wh"),
        ).rstrip(),
        "",
    ]
    _atomic_text(args.output / "BENCHMARK_SCOPE_AND_HARDWARE.md", "\n".join(scope_lines))

    input_files = [args.phase1_metrics, args.phase1_cost, args.phase2_metrics, args.external_metrics]
    provenance = {
        "schema": SCHEMA,
        "created_at": _now(),
        "geometry_claim_boundary": "OpenMVS-referenced held-out expected-depth consistency",
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in input_files
        ],
        "aggregation_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "counts": {
            "merged_per_scene_rows": len(merged),
            "table1_rows_including_macro": len(table1_rows),
            "table2_methods": len(table2_rows),
            "table3_methods": len(table3_rows),
            "benchmark_receipts": len(benchmark_raw),
            "supplementary_per_scene_rows": len(supplementary_per_scene_rows),
        },
        "adaptive_rgb_policy": (
            "RGB PSNR/SSIM/LPIPS are the frozen formal RGB Stage-1 anchor metrics; "
            "thermal, temperature, hotspot and geometry are from the Adaptive Stage-2 endpoint."
        ),
        "benchmark_macro_policy": (
            "arithmetic mean of six scene median ms/view; FPS = 1000 / macro latency; "
            "per-scene FPS is never averaged"
        ),
    }
    _atomic_json(args.output / "completion_provenance_summary.json", provenance)

    notes = """# Final table notes

- **Geometry boundary:** all geometry columns report *OpenMVS-referenced held-out expected-depth consistency*; they are not true-depth accuracy.
- **PhysIR-Splat-SH†:** frozen default thermal-SH endpoint without the complete physical renderer or VGGT-IR branch; it must not be described as PhysIR-Splat+VGGT-IR.
- **Internal roles:** SCSP-Refit+F3 is the RGB/geometry-stable main method; Adaptive Opacity+Scale-Clamp is the thermal-fidelity operating point.
- **Adaptive RGB:** formal RGB Stage-1 anchor metrics are used directly, without rerendering or a special table marker.
- **Efficiency:** render latency is the unified in-memory render-only benchmark on one RTX PRO 6000. Scene macro latency is the arithmetic mean of six scene medians and macro FPS is its reciprocal.
- **Interpretation:** OMMG/MMOne can be stronger on temperature/hotspot metrics; no claim is made that one method wins every metric.
"""
    _atomic_text(args.output / "TABLE_NOTES.md", notes)

    claim_matrix = [
        {
            "claim": "SCSP-Refit+F3 is the RGB/geometry-stable main operating point",
            "evidence": "Table 1 and Tables 2–3",
            "allowed": True,
        },
        {
            "claim": "Adaptive Opacity+Scale-Clamp is a thermal-fidelity operating point",
            "evidence": "Tables 2–3",
            "allowed": True,
        },
        {
            "claim": "OMMG/MMOne may outperform ours on temperature or hotspot metrics",
            "evidence": "Table 2",
            "allowed": True,
        },
        {
            "claim": "One UAV-TGS configuration is best on all metrics",
            "evidence": "Not supported by the frozen matrix",
            "allowed": False,
        },
        {
            "claim": "Geometry columns measure true depth accuracy",
            "evidence": "Reference is an OpenMVS diagnostic backend",
            "allowed": False,
        },
    ]
    _csv(args.output / "claim_matrix.csv", claim_matrix, ("claim", "evidence", "allowed"))
    _atomic_text(args.output / "claim_matrix.md", _markdown(claim_matrix, ("claim", "evidence", "allowed")))

    final_report = "\n".join(
        [
            "# UAV-TGS AAAI27 final frozen-result aggregation",
            "",
            "Status: **completed**. No training, tuning, or endpoint modification is part of this aggregation.",
            "",
            "## Formal outputs",
            "",
            "- Table 1: SCSP-Refit+F3 on all 11 scenes plus a scene-equal macro.",
            "- Table 2: seven-method six-scene external comparison.",
            "- Table 3: Raw-F3 / SCSP-Refit+F3 / Adaptive internal ablation.",
            "- Supplement: eight-method six-scene full metrics, both per scene and scene-equal macro.",
            "",
            "## Unified render-only benchmark",
            "",
            f"- Hardware: AutoDL 900, {scope_payload['gpu_name']}.",
            "- Each method × scene uses a clean process, one full test-list warm-up and three synchronized full passes.",
            "- The scene result is median ms/view; macro latency is the arithmetic mean of six scene latencies; FPS is its reciprocal.",
            "- Model/data loading, encoding, saving, CPU copies and metric/depth postprocessing are excluded.",
            "",
            _markdown(
                benchmark_macro_rows,
                ("display_name", "render_latency_ms_per_view", "render_fps", "inference_peak_allocated_gib"),
            ).rstrip(),
            "",
            "## Interpretation boundary",
            "",
            "- Geometry means OpenMVS-referenced held-out expected-depth consistency, not true-depth accuracy.",
            "- SCSP-Refit+F3 is the RGB/geometry-stable main method; Adaptive is the thermal-fidelity operating point.",
            "- PhysIR-Splat-SH† is the frozen thermal-SH configuration, not the complete physical renderer or VGGT-IR path.",
            "- No claim is made that one configuration wins every metric.",
            "",
        ]
    )
    _atomic_text(args.output / "FINAL_RESULTS_REPORT.md", final_report)

    report = {
        "schema": SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "representative_scenes": list(REPRESENTATIVE_SCENES),
        "all_scenes": list(ALL_SCENES),
        "table1_macro": table1_rows[-1],
        "table2": table2_rows,
        "table3": table3_rows,
        "benchmark_macro": benchmark_macro_rows,
    }
    target = args.output / "final_aggregation.json"
    _atomic_json(target, report)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-metrics", type=Path, required=True)
    parser.add_argument("--phase1-cost", type=Path, required=True)
    parser.add_argument("--phase2-metrics", type=Path, required=True)
    parser.add_argument("--external-metrics", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    target = aggregate(build_parser().parse_args())
    print(json.dumps({"status": "completed", "summary": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
