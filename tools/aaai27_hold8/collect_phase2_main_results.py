#!/usr/bin/env python3
"""Collect the frozen SCSP-Refit+strict-F3 result over all eleven scenes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCENES = (
    "Building", "InternalRoad", "PVpanel", "TransmissionTower", "Urban20K", "Orchard",
    "Garden", "Plaza", "Road", "Urban50K", "Urban100K",
)
METHOD = "scsp_refit_f3"
METRIC_DIRECTIONS = {
    "rgb_psnr": "max", "rgb_ssim": "max", "rgb_lpips": "min",
    "thermal_psnr": "max", "thermal_ssim": "max", "thermal_lpips": "min",
    "temperature_mae_c": "min", "temperature_rmse_c": "min",
    "hotspot_auprc": "max", "depth_front_1m": "min", "depth_front_2m": "min",
    "depth_front_5m": "min", "depth_agreement_1m": "max",
    "depth_agreement_2m": "max", "depth_agreement_5m": "max",
    "depth_median_abs_error_m": "min", "depth_missing_rate": "min",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def single_result(path: Path) -> dict[str, float]:
    value = load(path)
    if len(value) != 1:
        raise ValueError(f"expected one result entry: {path}")
    row = next(iter(value.values()))
    return {name: float(row[name]) for name in ("PSNR", "SSIM", "LPIPS")}


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value}")
    return number


def resolve_cost(method_root: Path, raw_root: Path, *, alias: bool) -> dict[str, Any]:
    projection_path = method_root / "efficiency/scsp_projection.json"
    projection = finite(load(projection_path)["wall_time_s"], "projection time") if projection_path.is_file() else None
    if alias:
        raw = load(raw_root / "efficiency/train.json")
        thermal = finite(raw["wall_time_s"], "raw F3 time")
        return {
            "projection_wall_time_s": projection,
            "projection_cost_status": "measured" if projection is not None else "not_recorded_phase1",
            "rgb_refit_wall_time_s": 0.0,
            "strict_f3_wall_time_s": thermal,
            "batch_execution_wall_time_s": thermal + (projection or 0.0),
            "reported_method_wall_time_s": thermal,
            "reported_cost_policy": "SCSP no-op aliases Raw-F3 and inherits Raw-F3 method cost",
            "peak_vram_bytes": finite(raw["device"]["peak_torch_reserved_bytes"], "peak VRAM"),
        }
    refit = load(method_root / "efficiency/rgb_refit.json")
    f3 = load(method_root / "efficiency/f3_train.json")
    refit_time = finite(refit["wall_time_s"], "RGB refit time")
    f3_time = finite(f3["wall_time_s"], "F3 time")
    total = (projection or 0.0) + refit_time + f3_time
    return {
        "projection_wall_time_s": projection,
        "projection_cost_status": "measured" if projection is not None else "not_recorded_phase1",
        "rgb_refit_wall_time_s": refit_time,
        "strict_f3_wall_time_s": f3_time,
        "batch_execution_wall_time_s": total,
        "reported_method_wall_time_s": total,
        "reported_cost_policy": "projection + RGB SH-only refit + strict F3",
        "peak_vram_bytes": max(
            finite(refit["device"]["peak_torch_reserved_bytes"], "refit peak VRAM"),
            finite(f3["device"]["peak_torch_reserved_bytes"], "F3 peak VRAM"),
        ),
    }


def collect_scene(root: Path, scene: str) -> dict[str, Any]:
    scene_root = root / f"experiments/aaai27_hold8_v2/{scene}"
    method_root = scene_root / f"methods/{METHOD}"
    raw_root = scene_root / "methods/raw_f3"
    manifest_path = method_root / "scsp_anchor/adaptive_scale_manifest.json"
    manifest = load(manifest_path)
    modified = int(manifest["counts"]["modified_gaussians"])
    alias = (method_root / "alias_to_raw_f3.json").is_file()
    if alias != (modified == 0):
        raise RuntimeError(f"SCSP alias/modified-count mismatch: {scene}")
    source_method = "raw_f3" if alias else METHOD
    source_root = scene_root / f"methods/{source_method}"
    evaluation = scene_root / f"evaluation/{source_method}"
    iteration = 60000 if alias else 65000
    model = source_root / "model"
    endpoint = model / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    if not endpoint.is_file():
        raise FileNotFoundError(endpoint)
    rgb = single_result(evaluation / "appearance/rgb_results.json")
    thermal = single_result(evaluation / "appearance/thermal_results.json")
    temperature = load(evaluation / "temperature/test.json")
    supported = temperature["summary"]["pixel_micro_aggregate"]["supported_pixels"]
    hotspot = load(evaluation / "hotspot/test.json")
    geometry = load(evaluation / "geometry/metrics/geometry_metrics.json")["metrics"]
    metrics = {
        "rgb_psnr": finite(rgb["PSNR"], "RGB PSNR"),
        "rgb_ssim": finite(rgb["SSIM"], "RGB SSIM"),
        "rgb_lpips": finite(rgb["LPIPS"], "RGB LPIPS"),
        "thermal_psnr": finite(thermal["PSNR"], "T PSNR"),
        "thermal_ssim": finite(thermal["SSIM"], "T SSIM"),
        "thermal_lpips": finite(thermal["LPIPS"], "T LPIPS"),
        "temperature_mae_c": finite(supported["mae_c"], "temperature MAE"),
        "temperature_rmse_c": finite(supported["rmse_c"], "temperature RMSE"),
        "hotspot_auprc": finite(hotspot["metrics"]["hotspot_auprc_histogram_4096"], "hotspot AUPRC"),
        "depth_median_abs_error_m": finite(geometry["median_absolute_depth_error_m"], "depth median error"),
        "depth_missing_rate": finite(geometry["missing_rate"], "missing rate"),
    }
    by_threshold = {float(row["threshold_m"]): row for row in geometry["threshold_metrics"]}
    for threshold in (1.0, 2.0, 5.0):
        row = by_threshold[threshold]
        token = int(threshold)
        metrics[f"depth_front_{token}m"] = finite(row["front_rate"], "front rate")
        metrics[f"depth_agreement_{token}m"] = finite(row["agreement_rate"], "agreement rate")
    render = load(source_root / "efficiency/render_efficiency.json")
    cost = resolve_cost(method_root, raw_root, alias=alias)
    return {
        "scene": scene,
        "method": METHOD,
        "alias_raw_f3": alias,
        "independent_endpoint_run": not alias,
        "scsp_modified_count": modified,
        "scsp_manifest_sha256": sha(manifest_path),
        "endpoint_sha256": sha(endpoint),
        "gaussian_count": int(render["gaussian_count"]),
        "model_size_bytes": endpoint.stat().st_size,
        "render_fps": finite(render["benchmark"]["cuda_event_fps"], "render FPS"),
        **cost,
        **metrics,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if [row["scene"] for row in rows] != list(SCENES):
        raise ValueError("eleven-scene row order/completeness mismatch")
    macro = {metric: sum(float(row[metric]) for row in rows) / len(rows) for metric in METRIC_DIRECTIONS}
    worst = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        function = min if direction == "max" else max
        row = function(rows, key=lambda item: float(item[metric]))
        worst[metric] = {"scene": row["scene"], "value": float(row[metric]), "desired": direction}
    modified_scenes = [row["scene"] for row in rows if not row["alias_raw_f3"]]
    noop_scenes = [row["scene"] for row in rows if row["alias_raw_f3"]]
    return {
        "schema": "uav-tgs-aaai27-hold8-phase2-eleven-scene-summary-v1",
        "status": "complete",
        "scene_count": len(rows),
        "method": METHOD,
        "macro": macro,
        "worst_scene": worst,
        "scsp": {
            "modified_scene_count": len(modified_scenes),
            "noop_scene_count": len(noop_scenes),
            "modified_scene_rate": len(modified_scenes) / len(rows),
            "noop_scene_rate": len(noop_scenes) / len(rows),
            "modified_scenes": modified_scenes,
            "noop_scenes": noop_scenes,
        },
        "cost": {
            "reported_method_wall_time_sum_s": sum(float(row["reported_method_wall_time_s"]) for row in rows),
            "batch_execution_wall_time_sum_s": sum(float(row["batch_execution_wall_time_s"]) for row in rows),
            "projection_measured_scene_count": sum(row["projection_cost_status"] == "measured" for row in rows),
        },
    }


def validate_scene_record(row: Mapping[str, Any], scene: str) -> dict[str, Any]:
    """Validate a portable, already-evaluated scene row before aggregation.

    This permits a scratch host to retain its large endpoint while the
    authoritative host receives only the metrics, receipts and endpoint hash.
    """
    if row.get("scene") != scene or row.get("method") != METHOD:
        raise ValueError(f"portable scene record identity mismatch: {scene}")
    if set(METRIC_DIRECTIONS).difference(row):
        missing = sorted(set(METRIC_DIRECTIONS).difference(row))
        raise ValueError(f"portable scene record missing metrics for {scene}: {missing}")
    for metric in METRIC_DIRECTIONS:
        finite(row[metric], f"{scene}/{metric}")
    for field in (
        "reported_method_wall_time_s", "batch_execution_wall_time_s", "peak_vram_bytes",
        "gaussian_count", "model_size_bytes", "render_fps", "scsp_modified_count",
    ):
        finite(row[field], f"{scene}/{field}")
    endpoint_sha = str(row.get("endpoint_sha256", ""))
    manifest_sha = str(row.get("scsp_manifest_sha256", ""))
    if len(endpoint_sha) != 64 or len(manifest_sha) != 64:
        raise ValueError(f"portable scene record hash invalid: {scene}")
    alias = bool(row.get("alias_raw_f3"))
    if alias != (int(row["scsp_modified_count"]) == 0):
        raise ValueError(f"portable scene record alias mismatch: {scene}")
    return dict(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-record-dir", type=Path)
    parser.add_argument("--export-scene", choices=SCENES)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if args.export_scene:
        row = collect_scene(args.root.resolve(), args.export_scene)
        target = output / f"{args.export_scene}.json"
        target.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "complete", "scene": args.export_scene, "output": str(target)}, sort_keys=True))
        return
    record_dir = args.scene_record_dir.resolve() if args.scene_record_dir else None
    rows = []
    for scene in SCENES:
        record_path = record_dir / f"{scene}.json" if record_dir else None
        if record_path is not None and record_path.is_file():
            rows.append(validate_scene_record(load(record_path), scene))
        else:
            rows.append(collect_scene(args.root.resolve(), scene))
    summary = summarize(rows)
    (output / "eleven_scene_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key in row}, key=lambda key: (key not in {"scene", "method"}, key))
    with (output / "eleven_scene_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "scene_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
