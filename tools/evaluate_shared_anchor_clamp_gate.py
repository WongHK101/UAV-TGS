#!/usr/bin/env python3
"""Evaluate the preregistered shared-anchor clamp first-stage gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


class SharedClampGateError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedClampGateError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SharedClampGateError(f"JSON root must be an object: {path}")
    return value


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SharedClampGateError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SharedClampGateError(f"{label} is non-finite")
    return result


def _appearance(path: Path, iteration: int) -> dict[str, float]:
    payload = _load(path)
    key = f"ours_{iteration}"
    metrics = payload.get(key)
    if not isinstance(metrics, dict):
        raise SharedClampGateError(f"{path} is missing {key}")
    return {
        metric: _finite(metrics.get(metric), f"{path}:{key}:{metric}")
        for metric in ("PSNR", "SSIM", "LPIPS")
    }


def _depth(path: Path, expected_scene: str) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("scene_name") != expected_scene:
        raise SharedClampGateError(
            f"depth scene mismatch: expected={expected_scene} "
            f"actual={payload.get('scene_name')}"
        )
    secondary = payload.get("secondary_metrics")
    thresholds = payload.get("threshold_metrics")
    if not isinstance(secondary, dict) or not isinstance(thresholds, list):
        raise SharedClampGateError(f"invalid depth report: {path}")
    by_threshold: dict[float, dict[str, float]] = {}
    for row in thresholds:
        if not isinstance(row, dict):
            continue
        threshold = _finite(row.get("threshold_m"), f"{path}:threshold")
        by_threshold[threshold] = {
            "front": _finite(
                row.get("FrontIntrusionRate"), f"{path}:front@{threshold}"
            ),
            "behind": _finite(row.get("TooDeepRate"), f"{path}:behind@{threshold}"),
            "agreement": _finite(
                row.get("DepthAgreementRate"), f"{path}:agreement@{threshold}"
            ),
        }
    missing_thresholds = sorted({1.0, 2.0, 5.0} - set(by_threshold))
    if missing_thresholds:
        raise SharedClampGateError(
            f"depth report is missing thresholds: {missing_thresholds}"
        )
    return {
        "reference_manifest_sha256": payload.get("reference_manifest_sha256"),
        "missing": _finite(secondary.get("MissingRate"), f"{path}:missing"),
        "mean_error": _finite(
            secondary.get("AbsDepthError_Mean"), f"{path}:mean_error"
        ),
        "median_error": _finite(
            secondary.get("AbsDepthError_Median"), f"{path}:median_error"
        ),
        "thresholds": {str(int(key)): by_threshold[key] for key in (1.0, 2.0, 5.0)},
    }


def _recovery(anchor: float, shared: float, legacy: float) -> dict[str, Any]:
    denominator = anchor - legacy
    if denominator <= 0:
        return {
            "status": "not_applicable",
            "value": None,
            "numerator": anchor - shared,
            "denominator": denominator,
            "reason": "anchor_minus_legacy_denominator_not_positive",
        }
    return {
        "status": "applicable",
        "value": (anchor - shared) / denominator,
        "numerator": anchor - shared,
        "denominator": denominator,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    scene = str(args.scene)
    if scene not in {"InternalRoad", "Building"}:
        raise SharedClampGateError("scene must be InternalRoad or Building")
    iteration = int(args.anchor_iteration)
    paths = {
        "anchor_results": Path(args.anchor_results).resolve(),
        "shared_results": Path(args.shared_results).resolve(),
        "anchor_depth": Path(args.anchor_depth).resolve(),
        "shared_depth": Path(args.shared_depth).resolve(),
        "legacy_depth": Path(args.legacy_depth).resolve(),
        "shared_manifest": Path(args.shared_manifest).resolve(),
        "block_analysis": Path(args.block_analysis).resolve(),
        "qualitative_assessment": Path(args.qualitative_assessment).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise SharedClampGateError(f"required input is missing: {path}")

    anchor_appearance = _appearance(paths["anchor_results"], iteration)
    shared_appearance = _appearance(paths["shared_results"], iteration)
    anchor_depth = _depth(paths["anchor_depth"], scene)
    shared_depth = _depth(paths["shared_depth"], scene)
    legacy_depth = _depth(paths["legacy_depth"], scene)
    references = {
        anchor_depth["reference_manifest_sha256"],
        shared_depth["reference_manifest_sha256"],
        legacy_depth["reference_manifest_sha256"],
    }
    if None in references or len(references) != 1:
        raise SharedClampGateError(
            "anchor/shared/legacy depth reports do not use one reference manifest"
        )

    shared_manifest = _load(paths["shared_manifest"])
    if (
        shared_manifest.get("status") != "passed"
        or shared_manifest.get("scene") != scene
        or shared_manifest.get("anchor_iteration") != iteration
        or shared_manifest.get("operation", {}).get("training_updates") != 0
        or shared_manifest.get("operation", {}).get("max_activated_scale") != 10.0
        or shared_manifest.get("invariants", {}).get("only_scaling_changed") is not True
    ):
        raise SharedClampGateError("shared-anchor manifest failed its locked contract")
    expected_count = 20 if scene == "InternalRoad" else 35
    if shared_manifest.get("counts", {}).get("actual_clamped_gaussians") != expected_count:
        raise SharedClampGateError("shared-anchor clamped count does not match protocol")

    block_analysis = _load(paths["block_analysis"])
    if (
        block_analysis.get("status") != "complete"
        or block_analysis.get("scene") != scene
        or block_analysis.get("groups") != ["Anchor", "S"]
    ):
        raise SharedClampGateError("block analysis is incomplete or mismatched")

    qualitative = _load(paths["qualitative_assessment"])
    if (
        qualitative.get("status") != "complete"
        or qualitative.get("scene") != scene
        or qualitative.get("fixed_views_reviewed") is not True
        or qualitative.get("mechanism_render_reviewed") is not True
    ):
        raise SharedClampGateError("qualitative assessment is incomplete")

    appearance_delta = {
        "psnr_loss_db": anchor_appearance["PSNR"] - shared_appearance["PSNR"],
        "ssim_loss": anchor_appearance["SSIM"] - shared_appearance["SSIM"],
        "lpips_increase": shared_appearance["LPIPS"] - anchor_appearance["LPIPS"],
    }
    depth_delta = {
        "front_at_1m_increase_pp": 100.0
        * (
            shared_depth["thresholds"]["1"]["front"]
            - anchor_depth["thresholds"]["1"]["front"]
        ),
        "missing_increase_pp": 100.0
        * (shared_depth["missing"] - anchor_depth["missing"]),
        "mean_error_delta_m": shared_depth["mean_error"] - anchor_depth["mean_error"],
        "median_error_delta_m": (
            shared_depth["median_error"] - anchor_depth["median_error"]
        ),
    }
    recoveries = {
        "front_at_1m": _recovery(
            anchor_depth["thresholds"]["1"]["front"],
            shared_depth["thresholds"]["1"]["front"],
            legacy_depth["thresholds"]["1"]["front"],
        ),
        "mean_error": _recovery(
            anchor_depth["mean_error"],
            shared_depth["mean_error"],
            legacy_depth["mean_error"],
        ),
    }

    common_conditions = {
        "rgb_psnr_loss_lte_0_10_db": appearance_delta["psnr_loss_db"] <= 0.10,
        "rgb_ssim_loss_lte_0_003": appearance_delta["ssim_loss"] <= 0.003,
        "rgb_lpips_increase_lte_0_005": (
            appearance_delta["lpips_increase"] <= 0.005
        ),
        "missing_increase_lte_0_2_pp": depth_delta["missing_increase_pp"] <= 0.2,
    }
    if scene == "InternalRoad":
        recovery_pass = any(
            result["status"] == "applicable" and result["value"] >= 0.70
            for result in recoveries.values()
        )
        scene_conditions = {
            "front_or_mean_error_recovery_gte_0_70": recovery_pass,
            "no_obvious_thin_structure_collapse": (
                qualitative.get("obvious_thin_structure_collapse") is False
            ),
        }
    else:
        scene_conditions = {
            "front_at_1m_increase_lte_1_pp": (
                depth_delta["front_at_1m_increase_pp"] <= 1.0
            ),
            "no_visible_structural_failure": (
                qualitative.get("visible_structural_failure") is False
            ),
        }
    conditions = {**common_conditions, **scene_conditions}
    passed = all(conditions.values())
    return {
        "schema": "uav-tgs-shared-anchor-clamp-first-stage-gate-v1",
        "status": "passed" if passed else "failed",
        "scene": scene,
        "anchor_iteration": iteration,
        "first_stage_passed": passed,
        "decision": (
            "eligible_for_joint_two_scene_decision"
            if passed
            else "stop_shared_anchor_clamp_without_f3_or_threshold_tuning"
        ),
        "conditions": conditions,
        "appearance": {
            "anchor": anchor_appearance,
            "shared": shared_appearance,
            "delta": appearance_delta,
        },
        "depth": {
            "anchor": anchor_depth,
            "shared": shared_depth,
            "legacy_L": legacy_depth,
            "delta": depth_delta,
            "recovery": recoveries,
        },
        "qualitative": qualitative,
        "inputs": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
        "boundaries": {
            "clamp_threshold_changed": False,
            "parameter_search_used": False,
            "clamp20_used_for_selection": False,
            "ogs_used": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["InternalRoad", "Building"], required=True)
    parser.add_argument("--anchor-iteration", type=int, default=30000)
    parser.add_argument("--anchor-results", required=True)
    parser.add_argument("--shared-results", required=True)
    parser.add_argument("--anchor-depth", required=True)
    parser.add_argument("--shared-depth", required=True)
    parser.add_argument("--legacy-depth", required=True)
    parser.add_argument("--shared-manifest", required=True)
    parser.add_argument("--block-analysis", required=True)
    parser.add_argument("--qualitative-assessment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        payload = evaluate(args)
    except SharedClampGateError as error:
        parser.error(str(error))
    output = Path(args.output).resolve()
    if output.exists():
        parser.error(f"refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["first_stage_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
