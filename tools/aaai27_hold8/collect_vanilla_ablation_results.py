#!/usr/bin/env python3
"""Collect the frozen three-method vanilla ablation ladder without large assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)
METHODS = ("t_only_sfm_3dgs", "rgb_sfm_t_3dgs", "naive_two_pass_3dgs")
METRICS = (
    "thermal_psnr",
    "thermal_ssim",
    "thermal_lpips",
    "temperature_mae_c",
    "temperature_rmse_c",
    "temperature_p95_c",
    "hotspot_auprc",
    "depth_front_1m",
    "depth_agreement_1m",
    "depth_median_abs_error_m",
    "depth_missing_rate",
    "training_wall_time_s",
    "peak_training_vram_gib",
    "gaussian_count",
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def optional_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def extract_appearance(path: Path) -> dict[str, float]:
    payload = load(path)
    if len(payload) != 1:
        raise ValueError(f"appearance result must contain exactly one method: {path}")
    values = next(iter(payload.values()))
    return {
        "thermal_psnr": finite(values["PSNR"], "PSNR"),
        "thermal_ssim": finite(values["SSIM"], "SSIM"),
        "thermal_lpips": finite(values["LPIPS"], "LPIPS"),
    }


def extract_temperature(path: Path) -> dict[str, float]:
    summary = load(path)["summary"]["temperature_error_supported_pixels"]
    return {
        "temperature_mae_c": finite(summary["mae_c"], "temperature MAE"),
        "temperature_rmse_c": finite(summary["rmse_c"], "temperature RMSE"),
        "temperature_p95_c": finite(summary["p95_abs_error_c"], "temperature P95"),
    }


def extract_hotspot(path: Path) -> dict[str, float]:
    return {
        "hotspot_auprc": finite(
            load(path)["metrics"]["hotspot_auprc_histogram_4096"], "hotspot AUPRC"
        )
    }


def extract_geometry(path: Path) -> dict[str, float]:
    values = load(path)["metrics"]
    thresholds = {
        float(item["threshold_m"]): item for item in values["threshold_metrics"]
    }
    one = thresholds[1.0]
    return {
        "depth_front_1m": finite(one["front_rate"], "front@1m"),
        "depth_agreement_1m": finite(one["agreement_rate"], "agreement@1m"),
        "depth_median_abs_error_m": finite(
            values["median_absolute_depth_error_m"], "depth MedAE"
        ),
        "depth_missing_rate": finite(values["missing_rate"], "depth missing"),
    }


def extract_efficiency(path: Path) -> dict[str, float | int]:
    values = load(path)
    result = values["result"]
    return {
        "training_wall_time_s": finite(values["wall_time_s"], "training wall time"),
        "peak_training_vram_gib": finite(
            values["device"]["peak_torch_reserved_bytes"], "peak training memory"
        )
        / (1024.0**3),
        "gaussian_count": int(result["gaussian_count"]),
        "optimizer_updates_executed": int(result["optimizer_updates_executed"]),
    }


def collect_row(root: Path, scene: str, method: str) -> dict[str, Any]:
    experiment = root / "experiments/aaai27_hold8_v2_native_vanilla_ablation" / scene
    method_root = experiment / method
    evaluation = experiment / "evaluation" / method
    row: dict[str, Any] = {
        "scene": scene,
        "method": method,
        "training_status": optional_status(method_root / "STATUS") or "missing",
        "evaluation_status": optional_status(evaluation / "STATUS") or "missing",
    }
    if row["training_status"] != "passed":
        failure = method_root / "failure.json"
        if failure.is_file():
            row["failure"] = load(failure)
        return row
    row.update(extract_efficiency(method_root / "train_efficiency.json"))
    if row["evaluation_status"] != "passed":
        return row
    row.update(extract_appearance(evaluation / "appearance/thermal_results.json"))
    row.update(extract_temperature(evaluation / "temperature/test.json"))
    hotspot = evaluation / "hotspot/test.json"
    geometry = evaluation / "geometry/metrics/geometry_metrics.json"
    if hotspot.is_file():
        row.update(extract_hotspot(hotspot))
    if geometry.is_file():
        row.update(extract_geometry(geometry))
    return row


def payload_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/UAV-TGS"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [collect_row(args.root, scene, method) for method in METHODS for scene in SCENES]
    macros: dict[str, Any] = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        method_macro: dict[str, Any] = {
            "completed_training": sum(row["training_status"] == "passed" for row in selected),
            "completed_evaluation": sum(row["evaluation_status"] == "passed" for row in selected),
        }
        for metric in METRICS:
            values = [finite(row[metric], metric) for row in selected if row.get(metric) is not None]
            method_macro[metric] = None if not values else fmean(values)
            method_macro[f"{metric}_n"] = len(values)
        macros[method] = method_macro
    payload = {
        "schema": "uav-tgs-aaai27-vanilla-ablation-results-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_order": list(SCENES),
        "method_order": list(METHODS),
        "rows": rows,
        "scene_equal_macros": macros,
    }
    payload["payload_sha256"] = payload_sha256(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(args.output_dir / "results.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = ["scene", "method", "training_status", "evaluation_status"] + list(METRICS)
    csv_path = args.output_dir / "results.csv"
    temporary = csv_path.with_name(csv_path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, csv_path)
    lines = ["# AAAI27 vanilla thermal ablation", "", f"Payload SHA-256: `{payload['payload_sha256']}`", ""]
    lines.append("| Method | Completed | T-PSNR | Temp MAE | AUPRC | Front@1m |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method in METHODS:
        values = macros[method]
        def fmt(key: str) -> str:
            value = values[key]
            return "N/A" if value is None else f"{value:.6f}"
        lines.append(
            f"| {method} | {values['completed_evaluation']}/6 | {fmt('thermal_psnr')} | "
            f"{fmt('temperature_mae_c')} | {fmt('hotspot_auprc')} | {fmt('depth_front_1m')} |"
        )
    atomic_text(args.output_dir / "summary.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": "passed", "payload_sha256": payload["payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
