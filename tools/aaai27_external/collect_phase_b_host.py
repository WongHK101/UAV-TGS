#!/usr/bin/env python3
"""Collect compact Phase-B evidence from one execution host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aaai27_external.evaluate_phase_b_endpoint import method_layout


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def optional_json(path: Path) -> dict[str, Any] | None:
    return load_json(path) if path.is_file() else None


def appearance(path: Path) -> dict[str, float] | None:
    payload = optional_json(path)
    if payload is None or len(payload) != 1:
        return None
    value = next(iter(payload.values()))
    if not isinstance(value, dict):
        return None
    return {
        "psnr": float(value["PSNR"]),
        "ssim": float(value["SSIM"]),
        "lpips": float(value["LPIPS"]),
    }


def ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        for raw in stream:
            line = raw.decode("ascii", "strict").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"PLY lacks vertex count: {path}")


def model_artifact_bytes(model: Path) -> int | None:
    if not model.is_dir():
        return None
    total = 0
    count = 0
    for path in model.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(model)
        if relative.parts and relative.parts[0] in {"test", "train", "rgb", "thermal"}:
            continue
        total += path.stat().st_size
        count += 1
    return total if count else None


def collect_endpoint(root: Path, scene: str, method: str) -> dict[str, Any]:
    run_root, dataset, _ = method_layout(root, scene, method)
    completion = optional_json(run_root / "COMPLETION.json")
    status = str(completion.get("status")) if completion else "MISSING"
    result: dict[str, Any] = {
        "scene": scene,
        "method": method,
        "status": status,
        "run_root": str(run_root),
        "adapter_manifest": str(dataset / "adapter_manifest.json"),
    }
    train_name = "train_eval_receipt.json" if method == "ThermoNeRF" else "train_receipt.json"
    train = optional_json(run_root / train_name)
    if train:
        result["training"] = {
            "status": train.get("status"),
            "wall_time_s": train.get("wall_time_s"),
            "peak_vram_mib": train.get("peak_process_vram_mib"),
            "command": train.get("command"),
        }
    render = optional_json(run_root / "render_receipt.json")
    if render:
        render_wall_time_s = render.get("wall_time_s")
        binding = optional_json(run_root / "normalized_thermal/render_binding_manifest.json")
        binding_rows = binding.get("rows") if binding else None
        view_count = len(binding_rows) if isinstance(binding_rows, list) else None
        result["render"] = {
            "status": render.get("status"),
            "wall_time_s": render_wall_time_s,
            "peak_vram_mib": render.get("peak_process_vram_mib"),
            "view_count": view_count,
            "render_test_views_per_s": (
                view_count / float(render_wall_time_s)
                if view_count is not None
                and render_wall_time_s is not None
                and float(render_wall_time_s) > 0.0
                else None
            ),
            "timing_scope": (
                "frozen official test-render command wall time divided by the "
                "formal test-view count; includes process/model setup"
            ),
        }
    if status != "SUCCEEDED":
        return result

    evaluation = run_root / "evaluation"
    eval_completion = optional_json(evaluation / "COMPLETION.json")
    result["evaluation_status"] = (
        eval_completion.get("status") if eval_completion else "MISSING"
    )
    result["appearance"] = {
        "rgb": appearance(run_root / "normalized_rgb/results.json"),
        "thermal": appearance(run_root / "normalized_thermal/results.json"),
    }
    temperature = optional_json(evaluation / "temperature/test.json")
    if temperature:
        supported = temperature["summary"]["temperature_error_supported_pixels"]
        result["temperature"] = {
            "mae_c": supported["mae_c"],
            "rmse_c": supported["rmse_c"],
            "p95_abs_error_c": supported["p95_abs_error_c"],
            "signed_bias_c": supported["signed_bias_c"],
        }
    hotspot = optional_json(evaluation / "hotspot/test.json")
    if hotspot:
        result["hotspot"] = {
            "auprc": hotspot["metrics"]["hotspot_auprc_histogram_4096"],
            "iou": hotspot["metrics"]["hotspot_iou"],
            "precision": hotspot["metrics"]["hotspot_precision"],
            "recall": hotspot["metrics"]["hotspot_recall"],
        }
    geometry = optional_json(evaluation / "geometry/evaluation/geometry_metrics.json")
    if geometry:
        thresholds = {
            str(row["threshold_m"]): {
                "front": row["front_rate"],
                "agreement": row["agreement_rate"],
            }
            for row in geometry["metrics"]["threshold_metrics"]
            if row.get("is_main_table_threshold")
        }
        result["geometry"] = {
            "metric_name": "OpenMVS-referenced held-out expected-depth consistency",
            "median_absolute_depth_error_m": geometry["metrics"][
                "median_absolute_depth_error_m"
            ],
            "missing_rate": geometry["metrics"]["missing_rate"],
            "thresholds_m": thresholds,
        }

    model = run_root / "model"
    ply = model / "point_cloud/iteration_30000/point_cloud.ply"
    result["model"] = {
        "gaussian_count": ply_vertex_count(ply),
        "artifact_bytes_excluding_render_outputs": model_artifact_bytes(model),
    }
    benchmark = optional_json(evaluation / "efficiency/render_benchmark/benchmark.json")
    if benchmark:
        result["render_benchmark"] = {
            "view_count": benchmark.get("view_count"),
            "render_wall_time_s": benchmark.get("render_wall_time_s"),
            "render_test_views_per_s": benchmark.get("render_test_views_per_s"),
            "timing_scope": benchmark.get("timing_scope"),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--method", action="append", choices=(
        "ThermalGaussian-OMMG", "PhysIR-Splat", "MMOne", "Thermal3D-GS", "ThermoNeRF"
    ), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    payload = {
        "schema": "uav-tgs-aaai27-external-phase-b-host-summary-v1",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "project_root": str(root),
        "endpoints": [
            collect_endpoint(root, scene, method)
            for method in args.method
            for scene in args.scene
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
