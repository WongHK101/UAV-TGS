#!/usr/bin/env python3
"""Collect the six-scene internal Hold-8 matrix into minimal formal receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from tools.hold8_minimal_receipts import (
    FORMAL_SCENES,
    METRIC_VALID,
    SUCCEEDED,
    finalize_evaluation_receipt,
    finalize_training_receipt,
    make_noop_alias_receipt,
    summarize_six_scene_study,
)


METHODS = ("raw_f3", "scsp_refit_f3", "adaptive_opacity_scale_clamp")
METHOD_RECIPES = {
    "raw_f3": "strict F3: SH3 cold restart, fresh Adam, appearance-only, no clamp",
    "scsp_refit_f3": "SCSP projection + 5k RGB SH-only refit + strict F3",
    "adaptive_opacity_scale_clamp": "legacy thermal opacity adaptation + scale clamp 10",
}
SOURCE_HOST = {
    "Building": "900",
    "InternalRoad": "900",
    "Urban20K": "900",
    "PVpanel": "901",
    "TransmissionTower": "901",
    "Orchard": "901",
    "Garden": "901",
    "Plaza": "901",
    "Road": "900",
    "Urban50K": "901",
    "Urban100K": "900",
}
TRAINING_CODE = (
    "train.py",
    "run_uavfgs_pipeline.py",
    "gaussian_renderer/__init__.py",
    "scene/gaussian_model.py",
    "tools/aaai27_hold8/run_phase1_internal_scene.sh",
    "tools/build_adaptive_scale_anchor.py",
    "tools/stage2_endpoint_audit.py",
)
EVALUATOR_CODE = (
    "tools/aaai27_hold8/evaluate_internal_method.sh",
    "tools/aaai27_hold8/bind_expected_depth_bundle.py",
    "tools/hold8_expected_depth_evaluator.py",
    "tools/thermal_radiometry/evaluate_temperature.py",
    "tools/evaluate_formal_baseline_hotspots.py",
    "metrics.py",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _scoped_hash(code: Path, names: Sequence[str]) -> str:
    rows = []
    for name in names:
        path = code / name
        if not path.is_file():
            raise FileNotFoundError(path)
        # Git may materialize the same text blob with LF or CRLF depending on
        # host checkout settings.  Line endings do not change Python/shell
        # semantics and must not invalidate otherwise identical endpoints.
        payload = path.read_bytes().replace(b"\r\n", b"\n")
        rows.append(
            {
                "path": name,
                "sha256_lf_normalized": hashlib.sha256(payload).hexdigest(),
                "normalized_size_bytes": len(payload),
            }
        )
    return _canonical_sha(rows)


def _git_head(code: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
    ).strip()


def _single_result(path: Path) -> tuple[str, dict[str, float]]:
    payload = _load(path)
    if len(payload) != 1:
        raise ValueError(f"expected one result row: {path}")
    key, row = next(iter(payload.items()))
    return str(key), {name: float(row[name]) for name in ("PSNR", "SSIM", "LPIPS")}


def _metric(value: Any) -> dict[str, Any]:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite metric: {value}")
    return {"status": METRIC_VALID, "value": number}


def _training_times(method: str, efficiency: Path) -> tuple[float, float, dict[str, Any]]:
    if method in {"raw_f3", "adaptive_opacity_scale_clamp"}:
        stage = _load(efficiency / "train.json")
        stages = {"thermal": float(stage["wall_time_s"]), "projection": None}
        return stages["thermal"], float(stage["device"]["peak_torch_reserved_bytes"]), stages
    refit = _load(efficiency / "rgb_refit.json")
    f3 = _load(efficiency / "f3_train.json")
    projection_path = efficiency / "scsp_projection.json"
    projection = _load(projection_path) if projection_path.is_file() else None
    projection_time = float(projection["wall_time_s"]) if projection is not None else 0.0
    stages = {
        "scsp_projection": projection_time if projection is not None else None,
        "rgb_refit": float(refit["wall_time_s"]),
        "thermal": float(f3["wall_time_s"]),
        "projection_cost_status": "measured" if projection is not None else "not_recorded_phase1",
    }
    return (
        projection_time + stages["rgb_refit"] + stages["thermal"],
        max(
            float(refit["device"]["peak_torch_reserved_bytes"]),
            float(f3["device"]["peak_torch_reserved_bytes"]),
        ),
        stages,
    )


def _method_paths(root: Path, scene: str, method: str) -> dict[str, Path | int]:
    base = root / "experiments/aaai27_hold8_v2" / scene
    method_root = base / "methods" / method
    iteration = 65000 if method == "scsp_refit_f3" else 60000
    return {
        "scene_root": base,
        "method_root": method_root,
        "model": method_root / "model",
        "iteration": iteration,
        "ply": method_root / f"model/point_cloud/iteration_{iteration}/point_cloud.ply",
        "endpoint": method_root / "endpoint.json",
        "efficiency": method_root / "efficiency",
        "evaluation": base / "evaluation" / method,
    }


def _build_training(
    *, root: Path, code: Path, scene: str, method: str, current_commit: str,
    scoped_code_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _method_paths(root, scene, method)
    endpoint = _load(paths["endpoint"])  # type: ignore[arg-type]
    inputs = endpoint["inputs"]
    ply = Path(paths["ply"])
    if inputs["ply"]["sha256"] != _sha(ply):
        raise RuntimeError(f"endpoint PLY hash mismatch: {scene}/{method}")
    method_time, peak_vram, breakdown = _training_times(method, Path(paths["efficiency"]))
    render = _load(Path(paths["efficiency"]) / "render_efficiency.json")
    gaussian_count = int(render["gaussian_count"])
    derived = root / "derived/aaai27_hold8_v2" / scene
    split = root / f"derived/thermal_radiometry/aaai27_hold8_v2/{scene}/binding/bound_split.json"
    canonical = derived / "radiometry/canonical_manifest.json"
    canonical_payload = _load(canonical)
    camera_records = [
        {"name": name, "sha256": _sha(derived / f"workspace/sparse/0/{name}")}
        for name in ("cameras.bin", "images.bin")
    ]
    config_basis = {
        "method": method,
        "recipe": METHOD_RECIPES[method],
        "iteration": int(paths["iteration"]),
        "training_metadata": [
            _load(item)["metadata"]
            for item in (
                [Path(paths["efficiency"]) / "train.json"]
                if method != "scsp_refit_f3"
                else [Path(paths["efficiency"]) / "rgb_refit.json", Path(paths["efficiency"]) / "f3_train.json"]
            )
        ],
    }
    training = finalize_training_receipt(
        {
            "scene": scene,
            "method_id": method,
            "repository": "https://github.com/WongHK101/UAV-TGS",
            "repository_commit": endpoint["code_commit"],
            "source_training_commit": endpoint["code_commit"],
            "current_runner_commit": current_commit,
            "source_scoped_code_sha256": scoped_code_hash,
            "reused_endpoint": False,
            "reuse_reason": None,
            "runtime_patch_sha256": None,
            "recipe": METHOD_RECIPES[method],
            "config_sha256": _canonical_sha(config_basis),
            "seed": 0,
            "split_manifest_sha256": _sha(split),
            "data_sha256": _sha(canonical),
            "camera_sha256": _canonical_sha(camera_records),
            "range_sha256": _sha(derived / "radiometry/range_manifest.json"),
            "lut_sha256": canonical_payload["palette"]["sha256_uint8_rgb"],
            "host": SOURCE_HOST[scene],
            "gpu": _load(Path(paths["efficiency"]) / ("f3_train.json" if method == "scsp_refit_f3" else "train.json"))["device"]["device_name"],
            "command": f"tools/aaai27_hold8/run_phase1_internal_scene.sh {scene} method {method}",
            "endpoint_sha256": _sha(ply),
            "completion_status": SUCCEEDED,
            "failure_reason": None,
            "batch_execution_wall_time_s": method_time,
            "reported_method_wall_time_s": method_time,
            "peak_vram_bytes": peak_vram,
            "model_size_bytes": ply.stat().st_size,
            "gaussian_count": gaussian_count,
        }
    )
    return training, breakdown


def _build_evaluation(
    *, root: Path, code: Path, scene: str, method: str,
    training: Mapping[str, Any], evaluator_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _method_paths(root, scene, method)
    evaluation = Path(paths["evaluation"])
    rgb_key, rgb = _single_result(evaluation / "appearance/rgb_results.json")
    thermal_key, thermal = _single_result(evaluation / "appearance/thermal_results.json")
    temperature = _load(evaluation / "temperature/test.json")
    temp = temperature["summary"]["pixel_micro_aggregate"]["supported_pixels"]
    off_lut = temperature["summary"]["off_lut_distance_aggregates"]["supported_pixels"]
    hotspot = _load(evaluation / "hotspot/test.json")
    geometry = _load(evaluation / "geometry/metrics/geometry_metrics.json")
    geometry_metrics = geometry["metrics"]
    metrics: dict[str, dict[str, Any]] = {
        "rgb_psnr": _metric(rgb["PSNR"]), "rgb_ssim": _metric(rgb["SSIM"]),
        "rgb_lpips": _metric(rgb["LPIPS"]), "thermal_psnr": _metric(thermal["PSNR"]),
        "thermal_ssim": _metric(thermal["SSIM"]), "thermal_lpips": _metric(thermal["LPIPS"]),
        "temperature_mae_c": _metric(temp["mae_c"]),
        "temperature_rmse_c": _metric(temp["rmse_c"]),
        "temperature_bias_c": _metric(temp["signed_bias_c"]),
        "temperature_p95_c": _metric(temp["p95_abs_error_c"]),
        "off_lut_mean_rgb": _metric(off_lut["mean_rgb_distance"]),
        "off_lut_p95_rgb": _metric(off_lut["p95_rgb_distance"]),
        "hotspot_auprc": _metric(hotspot["metrics"]["hotspot_auprc_histogram_4096"]),
        "hotspot_iou": _metric(hotspot["metrics"]["hotspot_iou"]),
        "depth_median_abs_error_m": _metric(geometry_metrics["median_absolute_depth_error_m"]),
        "depth_missing_rate": _metric(geometry_metrics["missing_rate"]),
    }
    for row in geometry_metrics["threshold_metrics"]:
        threshold = float(row["threshold_m"])
        token = str(int(threshold)) if threshold.is_integer() else str(threshold).replace(".", "p")
        metrics[f"depth_front_{token}m"] = _metric(row["front_rate"])
        metrics[f"depth_agreement_{token}m"] = _metric(row["agreement_rate"])
    render = _load(Path(paths["efficiency"]) / "render_efficiency.json")
    reference = root / f"derived/aaai27_hold8_v2/{scene}/reference_openmvs_hold8_v2/bound_expected_depth/manifest.json"
    eval_config = {
        "rgb_key": rgb_key, "thermal_key": thermal_key,
        "geometry_definition": geometry["depth_definition"],
        "geometry_thresholds": geometry["thresholds"],
        "temperature_support_policy": temperature["support_policy"],
        "hotspot_threshold": hotspot["hotspot_threshold"],
    }
    receipt = finalize_evaluation_receipt(
        {
            "evaluator_code_sha256": evaluator_hash,
            "evaluator_config_sha256": _canonical_sha(eval_config),
            "reference_sha256": _sha(reference),
            "completion_status": SUCCEEDED,
            "failure_reason": None,
            "metrics": metrics,
            "render_fps": float(render["benchmark"]["cuda_event_fps"]),
        },
        training,
    )
    return receipt, eval_config


def collect(root: Path, code: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    current_commit = _git_head(code)
    training_hash = _scoped_hash(code, TRAINING_CODE)
    evaluator_hash = _scoped_hash(code, EVALUATOR_CODE)
    trainings: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    receipt_root = output / "receipts"

    for scene in FORMAL_SCENES:
        for method in METHODS:
            paths = _method_paths(root, scene, method)
            method_root = Path(paths["method_root"])
            alias_path = method_root / "alias_to_raw_f3.json"
            if alias_path.is_file():
                source_training = next(row for row in trainings if row["scene"] == scene and row["method_id"] == "raw_f3")
                source_evaluation = next(row for row in evaluations if row["scene"] == scene and row["method_id"] == "raw_f3")
                alias = make_noop_alias_receipt(
                    source_training, source_evaluation, alias_method_id=method,
                    proof_sha256=_sha(method_root / "scsp_anchor/adaptive_scale_manifest.json"),
                )
                target = receipt_root / scene / method
                target.mkdir(parents=True, exist_ok=True)
                (target / "alias.json").write_text(json.dumps(alias, indent=2, sort_keys=True)+"\n", encoding="utf-8")
                aliases.append(alias)
                cost_rows.append({"scene": scene, "method": method, "alias": True,
                                  "batch_execution_wall_time_s": 0.0,
                                  "reported_method_wall_time_s": alias["reported_method_wall_time_s"]})
                row = {"scene": scene, "method": method,
                       "render_fps": source_evaluation["render_fps"], "alias": True}
                row.update({name: value.get("value") for name, value in source_evaluation["metrics"].items()})
                table_rows.append(row)
                continue
            training, breakdown = _build_training(
                root=root, code=code, scene=scene, method=method,
                current_commit=current_commit, scoped_code_hash=training_hash,
            )
            evaluation, eval_config = _build_evaluation(
                root=root, code=code, scene=scene, method=method,
                training=training, evaluator_hash=evaluator_hash,
            )
            target = receipt_root / scene / method
            target.mkdir(parents=True, exist_ok=True)
            (target / "training.json").write_text(json.dumps(training, indent=2, sort_keys=True)+"\n", encoding="utf-8")
            (target / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True)+"\n", encoding="utf-8")
            trainings.append(training); evaluations.append(evaluation)
            cost_rows.append({"scene": scene, "method": method, "alias": False,
                              "batch_execution_wall_time_s": training["batch_execution_wall_time_s"],
                              "reported_method_wall_time_s": training["reported_method_wall_time_s"],
                              "peak_vram_bytes": training["peak_vram_bytes"],
                              "gaussian_count": training["gaussian_count"], **breakdown})
            row = {"scene": scene, "method": method, "render_fps": evaluation["render_fps"], "alias": False}
            row.update({name: value.get("value") for name, value in evaluation["metrics"].items()})
            table_rows.append(row)

    summary = summarize_six_scene_study(
        methods=METHODS, training_receipts=trainings,
        evaluation_receipts=evaluations, alias_receipts=aliases,
    )
    (output / "six_scene_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (output / "cost_breakdown.json").write_text(json.dumps(cost_rows, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    fields = sorted({key for row in table_rows for key in row}, key=lambda value: (value not in {"scene","method"}, value))
    with (output / "per_scene_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(table_rows)
    provenance = {
        "current_commit": current_commit,
        "training_scoped_code_sha256": training_hash,
        "evaluation_scoped_code_sha256": evaluator_hash,
        "training_receipt_count": len(trainings),
        "evaluation_receipt_count": len(evaluations),
        "alias_receipt_count": len(aliases),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return {"summary": summary, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.root.resolve(), args.code_root.resolve(), args.output_root.resolve())
    print(json.dumps(result["provenance"], sort_keys=True))


if __name__ == "__main__":
    main()
