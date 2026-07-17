#!/usr/bin/env python3
"""Collect one formal L-vs-A3 scene and apply the frozen confirmation gates.

The collector is intentionally read-only with respect to experiment inputs.
Opacity saturation is read from the fixed audit definition (at least 99% of
activated opacities at <=1e-4 or >=1-1e-4).  Off-LUT material degradation uses
the preregistered conservative rule that supported-pixel aggregate mean and P95
must not both strictly worsen; no magnitude threshold is selected after seeing
results.  A failed ordinary gate is recorded but does not stop the next formal
scene; only a detected catastrophic condition does.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "uav-tgs-a3-scene-l-vs-a3-summary-v1"
DETAIL_SCHEMA = "uav-tgs-formal-l-a3-scene-detail-v1"
MANUAL_ASSESSMENT_SCHEMA = "uav-tgs-stage2-scene-manual-assessment-v1"
ITERATIONS = (40000, 50000, 60000)
GROUPS = ("L", "A3")
DECISION_ITERATION = 60000

EXPECTED_TRAINABILITY = {
    "xyz": False,
    "f_dc": True,
    "f_rest": True,
    "opacity": True,
    "scaling": False,
    "rotation": False,
    "exposure": False,
}
EXPECTED_OPTIMIZER_GROUPS = ["f_dc", "f_rest", "opacity"]


class SceneCollectionError(RuntimeError):
    """Raised when a required formal artifact violates the input contract."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SceneCollectionError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise SceneCollectionError(f"JSON root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_gt(value: float, threshold: float) -> bool:
    """Mathematical strict greater-than without binary boundary accidents."""

    return value > threshold and not math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-12
    )


def _at_most(value: float, threshold: float) -> bool:
    return value < threshold or math.isclose(
        value, threshold, rel_tol=1e-12, abs_tol=1e-12
    )


class _Numbers:
    def __init__(self) -> None:
        self.nonfinite: list[str] = []

    def value(self, raw: Any, label: str) -> float | None:
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise SceneCollectionError(f"{label} is not numeric: {raw!r}") from exc
        if not math.isfinite(value):
            self.nonfinite.append(label)
            return None
        return value


def _required(mapping: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise SceneCollectionError(f"missing {label}/{key}")
    return mapping[key]


def _int(raw: Any, label: str) -> int:
    if isinstance(raw, bool):
        raise SceneCollectionError(f"{label} must be an integer")
    try:
        result = int(raw)
    except (TypeError, ValueError) as exc:
        raise SceneCollectionError(f"{label} must be an integer") from exc
    return result


def _selected_numbers(
    source: Mapping[str, Any],
    keys: Iterable[str],
    numbers: _Numbers,
    label: str,
) -> dict[str, float | None]:
    return {
        key: numbers.value(_required(source, key, label), f"{label}/{key}")
        for key in keys
    }


def _appearance(model_root: Path, numbers: _Numbers, group: str) -> dict[str, Any]:
    primary_path = model_root / "results.json"
    plus_path = model_root / "results_plus.json"
    primary = _load_json(primary_path)
    plus = _load_json(plus_path)
    output: dict[str, Any] = {}
    for iteration in ITERATIONS:
        method = f"ours_{iteration}"
        item = _required(primary, method, str(primary_path))
        plus_item = _required(plus, method, str(plus_path))
        if not isinstance(item, dict) or not isinstance(plus_item, dict):
            raise SceneCollectionError(f"invalid method object for {group}/{method}")
        output[str(iteration)] = {
            **_selected_numbers(
                item,
                ("PSNR", "SSIM", "LPIPS"),
                numbers,
                f"appearance/{group}/{iteration}",
            ),
            "EdgeHaloScore": numbers.value(
                _required(plus_item, "EdgeHaloScore", str(plus_path)),
                f"appearance/{group}/{iteration}/EdgeHaloScore",
            ),
        }
    return output


def _temperature(path: Path, numbers: _Numbers, group: str, iteration: int) -> dict[str, Any]:
    payload = _load_json(path)
    completion = {
        "status_complete": payload.get("status") == "complete",
        "completed_with_missing_false": payload.get("completed_with_missing") is False,
        "primary_metric_valid": payload.get("primary_metric_valid") is True,
        "explicit_support": payload.get("support_is_explicit") is True,
    }
    if not all(completion.values()):
        raise SceneCollectionError(
            f"incomplete formal temperature report for {group}/{iteration}: {completion}"
        )
    summary = _required(payload, "summary", str(path))
    if not isinstance(summary, dict):
        raise SceneCollectionError(f"temperature summary is not an object: {path}")
    primary = _required(summary, "temperature_error_supported_pixels", str(path))
    micro = _required(
        _required(summary, "pixel_micro_aggregate", str(path)),
        "supported_pixels",
        str(path),
    )
    macro = _required(
        _required(summary, "frame_macro_mean_std", str(path)),
        "supported_pixels",
        str(path),
    )
    off_lut = _required(
        _required(summary, "off_lut_distance_aggregates", str(path)),
        "supported_pixels",
        str(path),
    )
    clipping = _required(
        _required(summary, "clipping_aggregates", str(path)),
        "supported_pixels",
        str(path),
    )
    support = _required(summary, "support_coverage", str(path))
    for name, value in (
        ("primary", primary),
        ("pixel_micro", micro),
        ("frame_macro", macro),
        ("off_lut", off_lut),
        ("clipping", clipping),
        ("support", support),
    ):
        if not isinstance(value, dict):
            raise SceneCollectionError(f"temperature {name} is not an object: {path}")

    error_keys = (
        "mae_c",
        "rmse_c",
        "signed_bias_c",
        "p95_abs_error_c",
        "max_abs_error_c",
    )
    frame_macro: dict[str, Any] = {
        "frame_count": _int(_required(macro, "frame_count", str(path)), f"{path}/frame_count")
    }
    for metric in ("mae_c", "rmse_c", "signed_bias_c", "p95_abs_error_c"):
        item = _required(macro, metric, str(path))
        if not isinstance(item, dict):
            raise SceneCollectionError(f"frame macro {metric} is not an object: {path}")
        frame_macro[metric] = {
            "mean": numbers.value(
                _required(item, "mean", str(path)),
                f"temperature/{group}/{iteration}/frame_macro/{metric}/mean",
            ),
            "std": numbers.value(
                _required(item, "std", str(path)),
                f"temperature/{group}/{iteration}/frame_macro/{metric}/std",
            ),
            "frame_count": _int(
                _required(item, "frame_count", str(path)),
                f"{path}/{metric}/frame_count",
            ),
        }

    support_float_keys = (
        "supported_ratio",
        "unsupported_ratio",
        "missing_ratio",
        "unsupported_or_missing_ratio",
    )
    support_int_keys = (
        "expected_frames",
        "render_available_frames",
        "support_available_frames",
        "frames_without_supported_pixels",
        "missing_render_frames",
        "missing_mask_frames",
        "expected_pixels",
        "supported_pixels",
        "unsupported_pixels",
        "missing_pixels",
        "unsupported_or_missing_pixels",
    )
    output_support: dict[str, Any] = {
        key: _int(_required(support, key, str(path)), f"{path}/{key}")
        for key in support_int_keys
    }
    output_support.update(
        _selected_numbers(
            support,
            support_float_keys,
            numbers,
            f"temperature/{group}/{iteration}/support",
        )
    )
    return {
        "completion": completion,
        "evaluated_file_count": _int(
            _required(summary, "evaluated_file_count", str(path)),
            f"{path}/evaluated_file_count",
        ),
        "supported_pixel_primary": {
            **_selected_numbers(
                primary,
                error_keys,
                numbers,
                f"temperature/{group}/{iteration}/supported_primary",
            ),
            "pixels": _int(_required(primary, "pixels", str(path)), f"{path}/pixels"),
        },
        "pixel_micro": {
            **_selected_numbers(
                micro,
                error_keys,
                numbers,
                f"temperature/{group}/{iteration}/pixel_micro",
            ),
            "pixels": _int(_required(micro, "pixels", str(path)), f"{path}/micro_pixels"),
        },
        "frame_macro": frame_macro,
        "clipping": {
            **_selected_numbers(
                clipping,
                ("clipping_ratio",),
                numbers,
                f"temperature/{group}/{iteration}/clipping",
            ),
            **{
                key: _int(_required(clipping, key, str(path)), f"{path}/{key}")
                for key in (
                    "clipped_pixels",
                    "high_pixels",
                    "low_pixels",
                    "pixels",
                )
            },
        },
        "off_lut": {
            **_selected_numbers(
                off_lut,
                (
                    "mean_rgb_distance",
                    "p95_rgb_distance",
                    "max_rgb_distance",
                    "rms_rgb_distance",
                ),
                numbers,
                f"temperature/{group}/{iteration}/off_lut",
            ),
            "pixels": _int(_required(off_lut, "pixels", str(path)), f"{path}/off_lut_pixels"),
        },
        "support": output_support,
    }


def _depth(path: Path, numbers: _Numbers, group: str, iteration: int) -> dict[str, Any]:
    payload = _load_json(path)
    counts = _required(payload, "counts", str(path))
    secondary = _required(payload, "secondary_metrics", str(path))
    thresholds = _required(payload, "threshold_metrics", str(path))
    if not isinstance(counts, dict) or not isinstance(secondary, dict) or not isinstance(thresholds, list):
        raise SceneCollectionError(f"invalid depth report structure: {path}")
    by_threshold: dict[float, dict[str, Any]] = {}
    for item in thresholds:
        if not isinstance(item, dict):
            raise SceneCollectionError(f"invalid threshold record: {path}")
        threshold = numbers.value(
            _required(item, "threshold_m", str(path)),
            f"depth/{group}/{iteration}/threshold_m",
        )
        if threshold is not None:
            by_threshold[threshold] = item
    output_thresholds: dict[str, Any] = {}
    for threshold in (1.0, 2.0, 5.0):
        if threshold not in by_threshold:
            raise SceneCollectionError(f"missing depth threshold {threshold:g}m: {path}")
        item = by_threshold[threshold]
        output_thresholds[str(int(threshold))] = {
            "front_intrusion_rate": numbers.value(
                _required(item, "FrontIntrusionRate", str(path)),
                f"depth/{group}/{iteration}/front@{threshold:g}m",
            ),
            "behind_rate": numbers.value(
                _required(item, "TooDeepRate", str(path)),
                f"depth/{group}/{iteration}/behind@{threshold:g}m",
            ),
            "depth_agreement_rate": numbers.value(
                _required(item, "DepthAgreementRate", str(path)),
                f"depth/{group}/{iteration}/agreement@{threshold:g}m",
            ),
        }
    return {
        "protocol_name": payload.get("protocol_name"),
        "reference_valid_pixels": _int(
            _required(counts, "reference_valid_pixels", str(path)),
            f"{path}/reference_valid_pixels",
        ),
        "model_valid_on_reference_pixels": _int(
            _required(counts, "model_valid_on_reference_pixels", str(path)),
            f"{path}/model_valid_on_reference_pixels",
        ),
        "missing_pixels": _int(
            _required(counts, "missing_pixels", str(path)), f"{path}/missing_pixels"
        ),
        "missing_rate": numbers.value(
            _required(secondary, "MissingRate", str(path)),
            f"depth/{group}/{iteration}/missing_rate",
        ),
        "abs_depth_error_mean_m": numbers.value(
            _required(secondary, "AbsDepthError_Mean", str(path)),
            f"depth/{group}/{iteration}/mean_error",
        ),
        "abs_depth_error_median_m": numbers.value(
            _required(secondary, "AbsDepthError_Median", str(path)),
            f"depth/{group}/{iteration}/median_error",
        ),
        "signed_depth_bias_mean_m": numbers.value(
            _required(secondary, "SignedDepthBias_Mean", str(path)),
            f"depth/{group}/{iteration}/signed_bias",
        ),
        "thresholds_m": output_thresholds,
    }


def _audit_finite_tree(value: Any, numbers: _Numbers, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _audit_finite_tree(item, numbers, f"{label}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _audit_finite_tree(item, numbers, f"{label}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numbers.value(value, label)


def _efficiency(root: Path, group: str, numbers: _Numbers) -> tuple[dict[str, Any], list[Path]]:
    training_path = root / f"{group}_train.json"
    training = _load_json(training_path)
    if not (
        training.get("schema_name") == "uav-tgs-efficiency"
        and training.get("schema_version") == 1
        and training.get("kind") == "training_stage"
        and training.get("status") == "completed"
        and training.get("stage") == "thermal"
    ):
        raise SceneCollectionError(f"invalid thermal training efficiency report: {training_path}")
    _audit_finite_tree(training, numbers, f"efficiency/{group}/training")
    renders: dict[str, Any] = {}
    paths = [training_path]
    for iteration in ITERATIONS:
        path = root / f"{group}_render_{iteration}.json"
        report = _load_json(path)
        if not (
            report.get("schema_name") == "uav-tgs-efficiency"
            and report.get("schema_version") == 1
            and report.get("kind") == "render"
            and report.get("status") == "completed"
            and report.get("iteration") == iteration
        ):
            raise SceneCollectionError(f"invalid render efficiency report: {path}")
        _audit_finite_tree(report, numbers, f"efficiency/{group}/render/{iteration}")
        renders[str(iteration)] = report
        paths.append(path)
    return {"training": training, "render": renders}, paths


def _manual_entry(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = payload.get(key)
    if raw is None:
        return {"assessed": False, "detected": None, "basis": "not provided"}
    if not isinstance(raw, dict):
        raise SceneCollectionError(f"manual assessment {key} must be an object")
    assessed = raw.get("assessed") is True
    detected = raw.get("detected")
    if assessed and not isinstance(detected, bool):
        raise SceneCollectionError(
            f"manual assessment {key} requires boolean detected when assessed"
        )
    if not assessed:
        detected = None
    basis = raw.get("basis")
    if assessed and (not isinstance(basis, str) or not basis.strip()):
        raise SceneCollectionError(
            f"manual assessment {key} requires non-empty basis when assessed"
        )
    return {"assessed": assessed, "detected": detected, "basis": basis or "not provided"}


def _load_manual_assessment(path: Path | None, scene: str) -> dict[str, Any]:
    keys = ("pipeline_reference_irreparable_failure",)
    if path is None:
        return {key: _manual_entry({}, key) for key in keys}
    payload = _load_json(path)
    if payload.get("schema") != MANUAL_ASSESSMENT_SCHEMA:
        raise SceneCollectionError(f"unexpected manual-assessment schema: {path}")
    if payload.get("scene") != scene:
        raise SceneCollectionError(
            f"manual-assessment scene mismatch: expected {scene!r}, got {payload.get('scene')!r}"
        )
    return {key: _manual_entry(payload, key) for key in keys}


def _opacity_saturation(
    activated: Mapping[str, Any],
    numbers: _Numbers,
    label: str,
    source: Path,
) -> dict[str, Any]:
    saturation = activated.get("catastrophic_saturation")
    if not isinstance(saturation, dict):
        raise SceneCollectionError(
            f"opacity audit lacks catastrophic_saturation: {source}"
        )
    expected_thresholds = {
        "low_activated_opacity_lte": 1e-4,
        "high_activated_opacity_gte": 1.0 - 1e-4,
        "catastrophic_fraction_gte": 0.99,
    }
    if saturation.get("thresholds") != expected_thresholds:
        raise SceneCollectionError(
            f"unexpected catastrophic-saturation thresholds: {source}"
        )
    a3 = saturation.get("a3")
    if not isinstance(a3, dict):
        raise SceneCollectionError(f"catastrophic-saturation A3 record is missing: {source}")
    low_fraction = numbers.value(
        _required(a3, "low_fraction", str(source)),
        f"{label}/catastrophic_saturation/low_fraction",
    )
    high_fraction = numbers.value(
        _required(a3, "high_fraction", str(source)),
        f"{label}/catastrophic_saturation/high_fraction",
    )
    for name, value in (("low_fraction", low_fraction), ("high_fraction", high_fraction)):
        if value is not None and not 0.0 <= value <= 1.0:
            raise SceneCollectionError(
                f"catastrophic-saturation {name} is outside [0,1]: {source}"
            )
    detected = saturation.get("detected")
    endpoint_detected = a3.get("detected")
    if not isinstance(detected, bool) or not isinstance(endpoint_detected, bool):
        raise SceneCollectionError(
            f"catastrophic-saturation detected fields must be boolean: {source}"
        )
    if low_fraction is not None and high_fraction is not None:
        recomputed = low_fraction >= 0.99 or high_fraction >= 0.99
        if detected != recomputed or endpoint_detected != recomputed:
            raise SceneCollectionError(
                f"catastrophic-saturation fractions and decision disagree: {source}"
            )
    return {
        "definition": saturation.get("definition"),
        "thresholds": expected_thresholds,
        "low_fraction": low_fraction,
        "high_fraction": high_fraction,
        "detected": detected,
    }


def _opacity(
    audit_root: Path,
    a3_model_root: Path,
    numbers: _Numbers,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime_path = a3_model_root / "opacity_adaptive_protocol.json"
    freeze_path = a3_model_root / "opacity_adaptive_freeze_audit.json"
    endpoints_path = audit_root / "A3_endpoints.json"
    runtime = _load_json(runtime_path)
    freeze = _load_json(freeze_path)
    endpoints = _load_json(endpoints_path)

    runtime_checks = {
        "runtime_status_passed": runtime.get("status") == "passed",
        "recipe_exact": runtime.get("thermal_recipe") == "geometry_frozen_opacity_adaptive",
        "freeze_mode_exact": runtime.get("thermal_freeze_mode")
        == "geometry_frozen_opacity_adaptive",
        "sh3_cold_restart_cap": runtime.get("thermal_max_sh_degree") == 3,
        "fresh_adam": runtime.get("thermal_optimizer_state") == "fresh",
        "optimizer_groups_exact": runtime.get("optimizer_groups") == EXPECTED_OPTIMIZER_GROUPS,
        "opacity_lr_exact": runtime.get("optimizer_lrs", {}).get("opacity") == 2e-4,
        "trainability_exact": runtime.get("trainability") == EXPECTED_TRAINABILITY,
        "thermal_scale_clamp_off": runtime.get("thermal_scale_clamp") == "off",
        "topology_fixed": runtime.get("topology_frozen") is True,
        "no_densification": runtime.get("densification") is False,
        "no_pruning": runtime.get("pruning") is False,
        "no_opacity_reset": runtime.get("opacity_reset") is False,
        "aligned_artifact_semantics": runtime.get("artifact_save_semantics") == "aligned",
    }

    frozen_fields = freeze.get("frozen_fields", {})
    frozen_checks: dict[str, bool] = {
        "freeze_audit_passed": freeze.get("status") == "passed",
        "start_iteration_30000": freeze.get("start_iteration") == 30000,
        "final_iteration_60000": freeze.get("final_iteration") == 60000,
        "activated_opacity_finite": freeze.get("activated_opacity", {}).get("finite") is True,
    }
    for field in ("_xyz", "_scaling", "_rotation"):
        item = frozen_fields.get(field, {})
        frozen_checks[f"{field}_bit_exact"] = bool(
            item.get("unchanged") is True
            and item.get("max_abs_diff") == 0.0
            and item.get("before_sha256") == item.get("after_sha256")
        )

    endpoint_items = endpoints.get("endpoints")
    if not isinstance(endpoint_items, list):
        raise SceneCollectionError(f"endpoint audit has no endpoint list: {endpoints_path}")
    endpoint_by_iteration = {
        item.get("iteration"): item for item in endpoint_items if isinstance(item, dict)
    }
    endpoint_checks: dict[str, bool] = {
        "endpoint_audit_passed": endpoints.get("status") == "passed",
        "endpoint_set_exact": set(endpoint_by_iteration) == set(ITERATIONS),
        "rgb_anchor_count_available": isinstance(
            endpoints.get("rgb_reference", {}).get("gaussian_count"), int
        ),
    }
    reference_count = endpoints.get("rgb_reference", {}).get("gaussian_count")
    for iteration in ITERATIONS:
        item = endpoint_by_iteration.get(iteration, {})
        endpoint_checks[f"endpoint_{iteration}_passed"] = item.get("status") == "passed"
        endpoint_checks[f"endpoint_{iteration}_count_equal"] = bool(
            item.get("gaussian_count_equal_rgb") is True
            and item.get("gaussian_count") == reference_count
        )
        endpoint_checks[f"endpoint_{iteration}_schema_equal"] = (
            item.get("schema_equal_rgb") is True
        )

    opacity_output: dict[str, Any] = {}
    per_endpoint_checks: dict[str, bool] = {}
    for iteration in ITERATIONS:
        opacity_path = audit_root / f"A3_{iteration}_opacity.json"
        alignment_path = audit_root / f"A3_{iteration}_ply_checkpoint_exact.json"
        opacity_audit = _load_json(opacity_path)
        alignment = _load_json(alignment_path)
        ply = opacity_audit.get("ply_audit", {})
        activated = ply.get("activated_opacity", {})
        delta = activated.get("a3_minus_anchor", {})
        proxy = opacity_audit.get("opacity_proxy_audit", {})
        absolute = delta.get("absolute", {})
        fractions = delta.get("fractions_abs_delta_gt", {})
        pixel_micro = proxy.get("pixel_micro", {})
        frame_macro = proxy.get("frame_macro", {})
        label = f"opacity/{iteration}"
        saturation = _opacity_saturation(activated, numbers, label, opacity_path)
        opacity_output[str(iteration)] = {
            "gaussian_count": _int(
                _required(ply, "gaussian_count", str(opacity_path)),
                f"{opacity_path}/gaussian_count",
            ),
            "all_structural_fields_exact": ply.get("all_structural_fields_exact") is True,
            "activated_opacity_anchor": _selected_numbers(
                activated.get("anchor", {}),
                ("mean", "median", "p95", "p99", "max"),
                numbers,
                f"{label}/anchor",
            ),
            "activated_opacity_a3": _selected_numbers(
                activated.get("a3", {}),
                ("mean", "median", "p95", "p99", "max"),
                numbers,
                f"{label}/a3",
            ),
            "activated_opacity_abs_delta": _selected_numbers(
                absolute,
                ("mean", "median", "p95", "p99", "max"),
                numbers,
                f"{label}/abs_delta",
            ),
            "fractions_abs_delta_gt": _selected_numbers(
                fractions,
                ("0.01", "0.05", "0.10"),
                numbers,
                f"{label}/fractions",
            ),
            "catastrophic_saturation": saturation,
            "rendered_opacity_map_pixel_micro": _selected_numbers(
                pixel_micro,
                (
                    "abs_delta_mean",
                    "abs_delta_median",
                    "abs_delta_p95",
                    "abs_delta_p99",
                    "abs_delta_max",
                    "fraction_abs_delta_gt_0.01",
                    "fraction_abs_delta_gt_0.05",
                    "fraction_abs_delta_gt_0.10",
                ),
                numbers,
                f"{label}/proxy_micro",
            ),
            "rendered_opacity_map_frame_macro": _selected_numbers(
                frame_macro,
                ("mean_abs_delta_mean", "mean_rmse", "max_abs_delta"),
                numbers,
                f"{label}/proxy_macro",
            ),
        }
        per_endpoint_checks[f"opacity_audit_{iteration}_passed"] = (
            opacity_audit.get("status") == "passed"
        )
        per_endpoint_checks[f"structural_fields_{iteration}_exact"] = (
            ply.get("all_structural_fields_exact") is True
        )
        per_endpoint_checks[f"gaussian_count_{iteration}_unchanged"] = (
            ply.get("gaussian_count") == reference_count
        )
        alignment_groups = alignment.get("groups", {})
        per_endpoint_checks[f"ply_checkpoint_{iteration}_aligned"] = bool(
            alignment.get("status") == "passed"
            and alignment.get("iteration") == iteration
            and alignment.get("expected_iteration") == iteration
            and alignment.get("active_sh_degree") == 3
            and alignment.get("expected_active_sh_degree") == 3
            and alignment.get("gaussian_count") == reference_count
            and alignment.get("optimizer_group_names") == EXPECTED_OPTIMIZER_GROUPS
            and alignment.get("expected_optimizer_group_names")
            == EXPECTED_OPTIMIZER_GROUPS
            and alignment.get("all_parameter_groups_exact") is True
            and all(
                isinstance(alignment_groups.get(name), dict)
                and alignment_groups[name].get("finite") is True
                and alignment_groups[name].get("exact") is True
                and alignment_groups[name].get("max_abs_diff") == 0.0
                for name in ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
            )
        )

    checks = {**runtime_checks, **frozen_checks, **endpoint_checks, **per_endpoint_checks}
    opacity_nonfinite = [item for item in numbers.nonfinite if item.startswith("opacity/")]
    checks["all_reported_opacity_statistics_finite"] = not opacity_nonfinite
    checks["passed"] = all(checks.values())
    invariant_output = {
        "checks": checks,
        "passed": checks["passed"],
        "opacity_nonfinite_fields": opacity_nonfinite,
        "runtime_protocol": runtime,
        "freeze_audit": freeze,
    }
    return opacity_output, invariant_output


def _validate_block_analysis(
    path: Path,
    *,
    expected_inputs: Mapping[str, Path],
) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "complete" or payload.get("iteration") != 60000:
        raise SceneCollectionError("block analysis must be complete at 60k")
    protocol = payload.get("protocol")
    source_groups = protocol.get("groups") if isinstance(protocol, dict) else None
    if (
        not isinstance(source_groups, list)
        or not all(isinstance(group, str) for group in source_groups)
        or len(source_groups) != len(set(source_groups))
        or not {"L", "A3"}.issubset(source_groups)
    ):
        raise SceneCollectionError(
            "block-analysis groups must contain unique L and A3 entries"
        )
    block_count = protocol.get("block_count")
    test_views = protocol.get("test_views")
    views_per_block = protocol.get("views_per_block")
    if (
        isinstance(block_count, bool)
        or not isinstance(block_count, int)
        or block_count <= 0
        or isinstance(test_views, bool)
        or not isinstance(test_views, int)
        or test_views <= 0
        or isinstance(views_per_block, bool)
        or not isinstance(views_per_block, int)
        or views_per_block != 16
    ):
        raise SceneCollectionError(
            "block-analysis protocol requires positive integer counts and exactly 16 views per block"
        )
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != block_count:
        raise SceneCollectionError("block-analysis block count is inconsistent")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("block"), dict)
        or item.get("block", {}).get("size") != views_per_block
        or not isinstance(item.get("block", {}).get("views"), list)
        or len(item.get("block", {}).get("views", [])) != views_per_block
        or not isinstance(item.get("group_means"), dict)
        or not {"L", "A3"}.issubset(item.get("group_means", {}))
        or not isinstance(item.get("paired_comparisons"), dict)
        or "A3_minus_L" not in item.get("paired_comparisons", {})
        for item in blocks
    ):
        raise SceneCollectionError("block-analysis records are not complete paired L/A3 blocks")
    if test_views != len(blocks) * views_per_block:
        raise SceneCollectionError("block-analysis test-view count is inconsistent")

    required_input_keys = {
        "bound_split",
        "per_view:L",
        "per_view:A3",
        "temperature:L",
        "temperature:A3",
    }
    if set(expected_inputs) != required_input_keys:
        raise SceneCollectionError(
            "collector block-binding keys are inconsistent with the formal schema"
        )
    declared_inputs = payload.get("inputs")
    if not isinstance(declared_inputs, dict):
        raise SceneCollectionError("block analysis lacks its verified inputs object")
    missing_inputs = sorted(required_input_keys - set(declared_inputs))
    if missing_inputs:
        raise SceneCollectionError(
            f"block analysis lacks required input bindings: {missing_inputs}"
        )
    input_binding: dict[str, Any] = {}
    for label in sorted(required_input_keys):
        record = declared_inputs[label]
        if not isinstance(record, dict):
            raise SceneCollectionError(f"block input record is not an object: {label}")
        declared_path = record.get("path")
        declared_sha = record.get("sha256")
        if not isinstance(declared_path, str) or not declared_path:
            raise SceneCollectionError(f"block input path is missing: {label}")
        if (
            not isinstance(declared_sha, str)
            or len(declared_sha) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in declared_sha)
        ):
            raise SceneCollectionError(
                f"block input SHA-256 is invalid for {label}: {declared_sha!r}"
            )
        actual_path = Path(expected_inputs[label]).resolve()
        if not actual_path.is_file():
            raise FileNotFoundError(actual_path)
        actual_sha = _sha256_file(actual_path)
        if actual_sha != declared_sha.lower():
            raise SceneCollectionError(
                f"block input SHA-256 mismatch for {label}: "
                f"declared={declared_sha.lower()} actual={actual_sha} path={actual_path}"
            )
        input_binding[label] = {
            "block_declared_path": declared_path,
            "collector_actual_path": str(actual_path),
            "sha256": actual_sha,
        }

    # This collector is deliberately L-vs-A3 even when the source is the legacy
    # Building analysis containing C3/F3.  Preserve source provenance while
    # exposing only the comparison used by the confirmation gates.
    normalized = dict(payload)
    normalized["protocol"] = {
        **protocol,
        "groups": ["L", "A3"],
        "source_groups": list(source_groups),
    }
    normalized["collector_input_binding"] = input_binding
    normalized["blocks"] = [
        {
            **item,
            "group_means": {
                group: item["group_means"][group] for group in ("L", "A3")
            },
            "paired_comparisons": {
                "A3_minus_L": item["paired_comparisons"]["A3_minus_L"]
            },
        }
        for item in blocks
    ]
    classifications = payload.get("classification_counts")
    if isinstance(classifications, dict) and "A3_minus_L" in classifications:
        normalized["classification_counts"] = {
            "A3_minus_L": classifications["A3_minus_L"]
        }
    return normalized


def _bound_split_counts(path: Path, scene: str) -> dict[str, int]:
    payload = _load_json(path)
    if payload.get("scene") != scene:
        raise SceneCollectionError(
            f"bound-split scene mismatch: expected {scene!r}, got {payload.get('scene')!r}"
        )
    records = payload.get("records")
    declared = payload.get("counts")
    if not isinstance(records, list) or not isinstance(declared, dict):
        raise SceneCollectionError("bound split must contain records and counts")
    actual = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("split") not in {"train", "test", "guard"}:
            raise SceneCollectionError(f"invalid bound-split record at index {index}")
        actual[record["split"]] += 1
    counts = {
        split: _int(_required(declared, split, str(path)), f"{path}/counts/{split}")
        for split in ("train", "test", "guard")
    }
    counts["total"] = _int(
        _required(declared, "total", str(path)), f"{path}/counts/total"
    )
    if any(counts[split] != actual[split] for split in ("train", "test", "guard")):
        raise SceneCollectionError("bound-split declared counts disagree with records")
    if counts["total"] != len(records) or counts["total"] != sum(
        counts[split] for split in ("train", "test", "guard")
    ):
        raise SceneCollectionError("bound-split total count is inconsistent")
    return {
        "train_views": counts["train"],
        "test_views": counts["test"],
        "guard_views": counts["guard"],
        "total_views": counts["total"],
    }


def _difference(candidate: Any, baseline: Any) -> Any:
    if isinstance(candidate, dict) and isinstance(baseline, dict):
        return {
            key: _difference(candidate[key], baseline[key])
            for key in candidate.keys() & baseline.keys()
        }
    if (
        isinstance(candidate, (int, float))
        and not isinstance(candidate, bool)
        and isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
    ):
        return float(candidate) - float(baseline)
    return None


def _normal_gates(
    appearance: Mapping[str, Any],
    temperature: Mapping[str, Any],
    depth: Mapping[str, Any],
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    key = str(DECISION_ITERATION)
    l_app = appearance["L"][key]
    a3_app = appearance["A3"][key]
    l_temp = temperature["L"][key]["supported_pixel_primary"]
    a3_temp = temperature["A3"][key]["supported_pixel_primary"]
    l_depth = depth["L"][key]
    a3_depth = depth["A3"][key]

    def loss_check(
        name: str, baseline: float | None, candidate: float | None, limit: float, direction: str
    ) -> dict[str, Any]:
        if baseline is None or candidate is None:
            return {
                "passed": False,
                "evaluable": False,
                "baseline": baseline,
                "a3": candidate,
                "limit": limit,
            }
        delta = baseline - candidate if direction == "higher" else candidate - baseline
        return {
            "passed": _at_most(delta, limit),
            "evaluable": True,
            "baseline": baseline,
            "a3": candidate,
            "increase_or_loss": delta,
            "limit": limit,
            "operator": "<=",
            "metric": name,
        }

    checks: dict[str, Any] = {
        "psnr_loss_le_0_15_db": loss_check(
            "PSNR", l_app["PSNR"], a3_app["PSNR"], 0.15, "higher"
        ),
        "ssim_loss_le_0_006": loss_check(
            "SSIM", l_app["SSIM"], a3_app["SSIM"], 0.006, "higher"
        ),
        "lpips_increase_le_0_008": loss_check(
            "LPIPS", l_app["LPIPS"], a3_app["LPIPS"], 0.008, "lower"
        ),
        "front_1m_increase_le_2pp": loss_check(
            "front_intrusion@1m",
            l_depth["thresholds_m"]["1"]["front_intrusion_rate"],
            a3_depth["thresholds_m"]["1"]["front_intrusion_rate"],
            0.02,
            "lower",
        ),
        "missing_increase_le_0_2pp": loss_check(
            "missing_rate", l_depth["missing_rate"], a3_depth["missing_rate"], 0.002, "lower"
        ),
    }
    for metric in ("mae_c", "rmse_c"):
        baseline = l_temp[metric]
        candidate = a3_temp[metric]
        name = (
            "temperature_mae_relative_le_5pct_or_absolute_le_0_1c"
            if metric == "mae_c"
            else "temperature_rmse_relative_le_5pct_or_absolute_le_0_1c"
        )
        if baseline is None or candidate is None:
            checks[name] = {
                "passed": False,
                "evaluable": False,
                "baseline": baseline,
                "a3": candidate,
                "absolute_limit_c": 0.1,
                "relative_limit": 0.05,
            }
            continue
        increase = candidate - baseline
        relative = increase / baseline if baseline > 0 else None
        relative_pass = relative is not None and _at_most(relative, 0.05)
        absolute_pass = _at_most(increase, 0.1)
        checks[name] = {
            "passed": relative_pass or absolute_pass,
            "evaluable": True,
            "baseline": baseline,
            "a3": candidate,
            "absolute_increase_c": increase,
            "relative_increase": relative,
            "absolute_limit_c": 0.1,
            "relative_limit": 0.05,
            "logic": "relative OR absolute",
            "absolute_passed": absolute_pass,
            "relative_passed": relative_pass,
        }

    l_off = temperature["L"][key]["off_lut"]
    a3_off = temperature["A3"][key]["off_lut"]
    mean_delta = (
        a3_off["mean_rgb_distance"] - l_off["mean_rgb_distance"]
        if a3_off["mean_rgb_distance"] is not None
        and l_off["mean_rgb_distance"] is not None
        else None
    )
    p95_delta = (
        a3_off["p95_rgb_distance"] - l_off["p95_rgb_distance"]
        if a3_off["p95_rgb_distance"] is not None
        and l_off["p95_rgb_distance"] is not None
        else None
    )
    simultaneous_directional_worsening = bool(
        mean_delta is not None
        and p95_delta is not None
        and _strict_gt(mean_delta, 0.0)
        and _strict_gt(p95_delta, 0.0)
    )
    off_lut_evaluable = mean_delta is not None and p95_delta is not None
    checks["no_simultaneous_material_off_lut_mean_and_p95_degradation"] = {
        "passed": (
            not simultaneous_directional_worsening if off_lut_evaluable else None
        ),
        "resolved": off_lut_evaluable,
        "mean_a3_minus_l": mean_delta,
        "p95_a3_minus_l": p95_delta,
        "simultaneous_directional_worsening": simultaneous_directional_worsening,
        "material_degradation_rule": (
            "fail iff supported-pixel aggregate off-LUT mean and P95 are both "
            "strictly greater for A3 than L; zero magnitude tolerance"
        ),
        "resolution": "pre-registered conservative joint-direction rule",
        "per_block_source": "block_analysis.blocks[*].paired_comparisons.A3_minus_L",
    }

    failed = [name for name, item in checks.items() if item.get("passed") is False]
    unresolved = [name for name, item in checks.items() if item.get("passed") is None]
    status = "failed" if failed else "manual_review_required" if unresolved else "passed"
    return {
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "unresolved_checks": unresolved,
        "ordinary_failure_is_not_a_catastrophic_stop": True,
    }


def _catastrophic_gates(
    appearance: Mapping[str, Any],
    temperature: Mapping[str, Any],
    depth: Mapping[str, Any],
    invariants: Mapping[str, Any],
    opacity: Mapping[str, Any],
    nonfinite: Sequence[str],
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    key = str(DECISION_ITERATION)
    l_psnr = appearance["L"][key]["PSNR"]
    a3_psnr = appearance["A3"][key]["PSNR"]
    l_mae = temperature["L"][key]["supported_pixel_primary"]["mae_c"]
    a3_mae = temperature["A3"][key]["supported_pixel_primary"]["mae_c"]
    l_front = depth["L"][key]["thresholds_m"]["1"]["front_intrusion_rate"]
    a3_front = depth["A3"][key]["thresholds_m"]["1"]["front_intrusion_rate"]

    psnr_loss = l_psnr - a3_psnr if l_psnr is not None and a3_psnr is not None else None
    mae_increase = a3_mae - l_mae if l_mae is not None and a3_mae is not None else None
    mae_relative = (
        mae_increase / l_mae
        if mae_increase is not None and l_mae is not None and l_mae > 0
        else None
    )
    front_increase = (
        a3_front - l_front if l_front is not None and a3_front is not None else None
    )
    pipeline = assessment["pipeline_reference_irreparable_failure"]
    saturation_by_endpoint = {
        iteration: item["catastrophic_saturation"]["detected"]
        for iteration, item in opacity.items()
    }

    checks: dict[str, Any] = {
        "geometry_or_protocol_invariant_failure": {
            "detected": not invariants["passed"],
            "resolved": True,
            "invariants_passed": invariants["passed"],
        },
        "nan_or_inf": {
            "detected": bool(nonfinite),
            "resolved": True,
            "fields": list(nonfinite),
        },
        "opacity_saturation": {
            "detected": any(saturation_by_endpoint.values()),
            "resolved": True,
            "by_endpoint": saturation_by_endpoint,
            "definition": (
                "catastrophic iff at least 0.99 of activated opacities are <=1e-4 "
                "or >=1-1e-4 at any formal endpoint"
            ),
            "note": "Read from and cross-checked against each opacity audit.",
        },
        "psnr_loss_gt_0_30_db": {
            "detected": psnr_loss is not None and _strict_gt(psnr_loss, 0.30),
            "resolved": psnr_loss is not None,
            "loss_db": psnr_loss,
            "threshold_db": 0.30,
            "operator": ">",
        },
        "temperature_mae_increase_gt_0_20_c_and_gt_10_percent": {
            "detected": bool(
                mae_increase is not None
                and _strict_gt(mae_increase, 0.20)
                and (
                    (mae_relative is not None and _strict_gt(mae_relative, 0.10))
                    or (l_mae == 0 and _strict_gt(mae_increase, 0.0))
                )
            ),
            "resolved": mae_increase is not None,
            "absolute_increase_c": mae_increase,
            "relative_increase": mae_relative,
            "absolute_threshold_c": 0.20,
            "relative_threshold": 0.10,
            "logic": "absolute AND relative, both strict",
        },
        "front_1m_increase_gt_3pp": {
            "detected": front_increase is not None and _strict_gt(front_increase, 0.03),
            "resolved": front_increase is not None,
            "increase": front_increase,
            "threshold": 0.03,
            "operator": ">",
        },
        "pipeline_reference_failure_irreparable": {
            "detected": pipeline["detected"] if pipeline["assessed"] else None,
            "resolved": pipeline["assessed"],
            "assessment": pipeline,
        },
    }
    detected = [name for name, item in checks.items() if item.get("detected") is True]
    unresolved = [name for name, item in checks.items() if not item.get("resolved")]
    status = "detected" if detected else "manual_review_required" if unresolved else "not_detected"
    return {
        "status": status,
        "checks": checks,
        "detected_checks": detected,
        "unresolved_checks": unresolved,
        "stop_only_if_status_detected": True,
    }


def _input_record(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def collect_scene(
    *,
    scene: str,
    l_model_root: Path,
    a3_model_root: Path,
    evaluation_root: Path,
    a3_audit_root: Path,
    l_efficiency_root: Path,
    a3_efficiency_root: Path,
    bound_split: Path,
    rgb_anchor_depth: Path,
    block_analysis: Path,
    manual_assessment: Path | None = None,
    next_scene: str | None = None,
) -> dict[str, Any]:
    if not scene.strip():
        raise SceneCollectionError("scene must be non-empty")
    l_model_root = Path(l_model_root).resolve()
    a3_model_root = Path(a3_model_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    a3_audit_root = Path(a3_audit_root).resolve()
    l_efficiency_root = Path(l_efficiency_root).resolve()
    a3_efficiency_root = Path(a3_efficiency_root).resolve()
    bound_split = Path(bound_split).resolve()
    rgb_anchor_depth = Path(rgb_anchor_depth).resolve()
    block_analysis = Path(block_analysis).resolve()
    manual_path = Path(manual_assessment).resolve() if manual_assessment is not None else None

    numbers = _Numbers()
    appearance = {
        "L": _appearance(l_model_root, numbers, "L"),
        "A3": _appearance(a3_model_root, numbers, "A3"),
    }
    temperature: dict[str, Any] = {group: {} for group in GROUPS}
    depth: dict[str, Any] = {group: {} for group in GROUPS}
    per_view_paths = {
        "L": l_model_root / "per_view.json",
        "A3": a3_model_root / "per_view.json",
    }
    input_paths: list[Path] = [
        l_model_root / "results.json",
        l_model_root / "results_plus.json",
        per_view_paths["L"],
        a3_model_root / "results.json",
        a3_model_root / "results_plus.json",
        per_view_paths["A3"],
    ]
    for group in GROUPS:
        for iteration in ITERATIONS:
            temperature_path = evaluation_root / "temperature" / f"{group}_{iteration}.json"
            depth_path = (
                evaluation_root
                / "depth"
                / group
                / str(iteration)
                / "metrics"
                / "metrics_summary.json"
            )
            temperature[group][str(iteration)] = _temperature(
                temperature_path, numbers, group, iteration
            )
            depth[group][str(iteration)] = _depth(depth_path, numbers, group, iteration)
            input_paths.extend((temperature_path, depth_path))

    rgb_anchor = _depth(rgb_anchor_depth, numbers, "RGB_anchor", 30000)
    efficiency_l, efficiency_l_paths = _efficiency(l_efficiency_root, "L", numbers)
    efficiency_a3, efficiency_a3_paths = _efficiency(
        a3_efficiency_root, "A3", numbers
    )
    opacity, invariants = _opacity(a3_audit_root, a3_model_root, numbers)
    block = _validate_block_analysis(
        block_analysis,
        expected_inputs={
            "bound_split": bound_split,
            "per_view:L": per_view_paths["L"],
            "per_view:A3": per_view_paths["A3"],
            "temperature:L": evaluation_root / "temperature" / "L_60000.json",
            "temperature:A3": evaluation_root / "temperature" / "A3_60000.json",
        },
    )
    counts = _bound_split_counts(bound_split, scene)
    counts["test_blocks"] = block["protocol"]["block_count"]
    expected_test_views = block["protocol"]["test_views"]
    if counts["test_views"] != expected_test_views:
        raise SceneCollectionError(
            "bound-split test count does not match block analysis: "
            f"{counts['test_views']} != {expected_test_views}"
        )
    for group in GROUPS:
        for iteration in ITERATIONS:
            actual = temperature[group][str(iteration)]["evaluated_file_count"]
            if actual != expected_test_views:
                raise SceneCollectionError(
                    "temperature evaluated-file count does not match the formal test set: "
                    f"{group}/{iteration} has {actual}, expected {expected_test_views}"
                )
    assessment = _load_manual_assessment(manual_path, scene)
    input_paths.extend(
        [
            rgb_anchor_depth,
            block_analysis,
            bound_split,
            a3_model_root / "opacity_adaptive_protocol.json",
            a3_model_root / "opacity_adaptive_freeze_audit.json",
            a3_audit_root / "A3_endpoints.json",
            *efficiency_l_paths,
            *efficiency_a3_paths,
            *[
                a3_audit_root / f"A3_{iteration}_{suffix}.json"
                for iteration in ITERATIONS
                for suffix in ("opacity", "ply_checkpoint_exact")
            ],
        ]
    )
    if manual_path is not None:
        input_paths.append(manual_path)

    normal = _normal_gates(appearance, temperature, depth, assessment)
    catastrophic = _catastrophic_gates(
        appearance,
        temperature,
        depth,
        invariants,
        opacity,
        numbers.nonfinite,
        assessment,
    )
    if catastrophic["status"] == "detected":
        action = "stop"
        reason = "at least one preregistered catastrophic condition was detected"
    elif catastrophic["status"] == "manual_review_required" or normal["status"] == "manual_review_required":
        action = "manual_review_required"
        reason = "a qualitative/materiality gate is unresolved; no implicit threshold was applied"
    elif next_scene:
        action = "continue_to_next_scene"
        reason = (
            "normal gates passed"
            if normal["status"] == "passed"
            else "ordinary non-catastrophic gate failure is recorded and does not stop the next scene"
        )
    else:
        action = "scene_confirmation_complete"
        reason = (
            "normal gates passed"
            if normal["status"] == "passed"
            else "ordinary non-catastrophic gate failure recorded; no next scene was requested"
        )

    deltas = {
        str(iteration): {
            "appearance_a3_minus_l": _difference(
                appearance["A3"][str(iteration)], appearance["L"][str(iteration)]
            ),
            "temperature_a3_minus_l": _difference(
                temperature["A3"][str(iteration)], temperature["L"][str(iteration)]
            ),
            "depth_a3_minus_l": _difference(
                depth["A3"][str(iteration)], depth["L"][str(iteration)]
            ),
        }
        for iteration in ITERATIONS
    }

    def macro_metrics(group: str) -> dict[str, float | None]:
        key = str(DECISION_ITERATION)
        app = appearance[group][key]
        temp = temperature[group][key]
        primary = temp["supported_pixel_primary"]
        geometry = depth[group][key]
        return {
            "PSNR": app["PSNR"],
            "SSIM": app["SSIM"],
            "LPIPS": app["LPIPS"],
            "EdgeHaloScore": app["EdgeHaloScore"],
            "temperature_mae_c": primary["mae_c"],
            "temperature_rmse_c": primary["rmse_c"],
            "temperature_bias_c": primary["signed_bias_c"],
            "temperature_abs_bias_c": (
                abs(primary["signed_bias_c"])
                if primary["signed_bias_c"] is not None
                else None
            ),
            "temperature_p95_c": primary["p95_abs_error_c"],
            "temperature_clipping_ratio": temp["clipping"]["clipping_ratio"],
            "off_lut_mean_rgb_distance": temp["off_lut"]["mean_rgb_distance"],
            "off_lut_p95_rgb_distance": temp["off_lut"]["p95_rgb_distance"],
            **{
                f"front_intrusion_{threshold}m": geometry["thresholds_m"][threshold][
                    "front_intrusion_rate"
                ]
                for threshold in ("1", "2", "5")
            },
            **{
                f"depth_agreement_{threshold}m": geometry["thresholds_m"][threshold][
                    "depth_agreement_rate"
                ]
                for threshold in ("1", "2", "5")
            },
            **{
                f"behind_{threshold}m": geometry["thresholds_m"][threshold]["behind_rate"]
                for threshold in ("1", "2", "5")
            },
            "depth_mean_m": geometry["abs_depth_error_mean_m"],
            "depth_median_m": geometry["abs_depth_error_median_m"],
            "missing_rate": geometry["missing_rate"],
        }

    material_off_lut = normal["checks"][
        "no_simultaneous_material_off_lut_mean_and_p95_degradation"
    ]["simultaneous_directional_worsening"]
    saturation_check = catastrophic["checks"]["opacity_saturation"]
    pipeline_check = catastrophic["checks"]["pipeline_reference_failure_irreparable"]
    invariants["all_passed"] = invariants["passed"]
    payload = {
        "schema": SCHEMA,
        "detail_schema": DETAIL_SCHEMA,
        "status": "complete",
        "scene": scene,
        "counts": counts,
        "methods": {
            group: {"metrics": macro_metrics(group), "decision_iteration": DECISION_ITERATION}
            for group in GROUPS
        },
        "reviews": {
            "material_off_lut_degradation": material_off_lut,
            "opacity_saturation": saturation_check["detected"],
            "pipeline_reference_irreparable": pipeline_check["detected"],
        },
        "reviews_basis": {
            "material_off_lut_degradation": normal["checks"][
                "no_simultaneous_material_off_lut_mean_and_p95_degradation"
            ],
            "opacity_saturation": saturation_check,
            "pipeline_reference_irreparable": pipeline_check,
        },
        "decision_endpoint": DECISION_ITERATION,
        "protocol": {
            "groups": list(GROUPS),
            "iterations": list(ITERATIONS),
            "normal_gate_scope": "A3 versus formal legacy L at 60k",
            "catastrophic_gate_scope": "A3 versus formal legacy L at 60k",
            "claim_boundary": (
                "Reference-depth changes are rendered visibility/surface-consistency trade-offs; "
                "they are not Gaussian-center, true-geometry, or absolute-thermal-accuracy improvements."
            ),
        },
        "inputs": [_input_record(path) for path in dict.fromkeys(input_paths)],
        "appearance": appearance,
        "temperature": temperature,
        "depth": {**depth, "RGB_anchor": {"30000": rgb_anchor}},
        "a3_minus_l": deltas,
        "opacity": opacity,
        "efficiency": {"L": efficiency_l, "A3": efficiency_a3},
        "invariants": invariants,
        "block_analysis": block,
        "manual_assessment": assessment,
        "nonfinite_inputs": numbers.nonfinite,
        "normal_gates": normal,
        "catastrophic_gates": catastrophic,
        "continuation_decision": {
            "action": action,
            "target": next_scene,
            "reason": reason,
            "ordinary_failure_recorded": normal["status"] == "failed"
            and catastrophic["status"] != "detected",
        },
    }
    return payload


def _flatten_numeric(prefix: str, value: Any) -> Iterable[tuple[str, int | float]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten_numeric(f"{prefix}.{key}" if prefix else str(key), item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, value


def csv_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scene = payload["scene"]

    def add(
        category: str,
        metric: str,
        value: Any,
        *,
        group: str = "",
        iteration: Any = "",
        status: str = "",
        note: str = "",
    ) -> None:
        rows.append(
            {
                "scene": scene,
                "category": category,
                "group": group,
                "iteration": iteration,
                "metric": metric,
                "value": value,
                "status": status,
                "note": note,
            }
        )

    for group in GROUPS:
        for iteration in ITERATIONS:
            key = str(iteration)
            for metric, value in _flatten_numeric("", payload["appearance"][group][key]):
                add("appearance", metric, value, group=group, iteration=iteration)
            for metric, value in _flatten_numeric("", payload["temperature"][group][key]):
                add("temperature", metric, value, group=group, iteration=iteration)
            for metric, value in _flatten_numeric("", payload["depth"][group][key]):
                add("depth", metric, value, group=group, iteration=iteration)
    for iteration, item in payload["opacity"].items():
        for metric, value in _flatten_numeric("", item):
            add("opacity", metric, value, group="A3", iteration=iteration)
    for group in GROUPS:
        for metric, value in _flatten_numeric(
            "training", payload["efficiency"][group]["training"]
        ):
            add("efficiency", metric, value, group=group)
        for iteration, item in payload["efficiency"][group]["render"].items():
            for metric, value in _flatten_numeric("render", item):
                add("efficiency", metric, value, group=group, iteration=iteration)
    for name, item in payload["normal_gates"]["checks"].items():
        add(
            "normal_gate",
            name,
            item.get("increase_or_loss", item.get("absolute_increase_c", "")),
            iteration=DECISION_ITERATION,
            status=(
                "passed"
                if item.get("passed") is True
                else "failed"
                if item.get("passed") is False
                else "manual_review_required"
            ),
            note=str(item.get("resolution", item.get("logic", ""))),
        )
    for name, item in payload["catastrophic_gates"]["checks"].items():
        add(
            "catastrophic_gate",
            name,
            item.get("detected"),
            iteration=DECISION_ITERATION,
            status=(
                "detected"
                if item.get("detected") is True
                else "not_detected"
                if item.get("resolved")
                else "manual_review_required"
            ),
            note=str(item.get("note", item.get("logic", ""))),
        )
    for block in payload["block_analysis"]["blocks"]:
        block_id = f"{block['block']['strip_id']}:{block['block']['block_index']}"
        paired = block["paired_comparisons"]["A3_minus_L"]
        for metric, item in paired.get("metrics", {}).items():
            add(
                "block_delta",
                metric,
                item.get("raw_a3_minus_baseline"),
                group="A3_minus_L",
                iteration=DECISION_ITERATION,
                status=str(item.get("classification", "")),
                note=block_id,
            )
        for metric, item in paired.get("diagnostic_deltas", {}).items():
            add(
                "block_delta",
                metric,
                item.get("raw_a3_minus_baseline"),
                group="A3_minus_L",
                iteration=DECISION_ITERATION,
                status=str(item.get("classification", "diagnostic_only")),
                note=block_id,
            )
    return rows


def _write_outputs(payload: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = (
            "scene",
            "category",
            "group",
            "iteration",
            "metric",
            "value",
            "status",
            "note",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows(payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--l-model-root", required=True, type=Path)
    parser.add_argument("--a3-model-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--a3-audit-root", required=True, type=Path)
    parser.add_argument("--l-efficiency-root", required=True, type=Path)
    parser.add_argument("--a3-efficiency-root", required=True, type=Path)
    parser.add_argument("--bound-split", required=True, type=Path)
    parser.add_argument("--rgb-anchor-depth", required=True, type=Path)
    parser.add_argument("--block-analysis", required=True, type=Path)
    parser.add_argument("--manual-assessment", type=Path)
    parser.add_argument("--next-scene")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = collect_scene(
        scene=args.scene,
        l_model_root=args.l_model_root,
        a3_model_root=args.a3_model_root,
        evaluation_root=args.evaluation_root,
        a3_audit_root=args.a3_audit_root,
        l_efficiency_root=args.l_efficiency_root,
        a3_efficiency_root=args.a3_efficiency_root,
        bound_split=args.bound_split,
        rgb_anchor_depth=args.rgb_anchor_depth,
        block_analysis=args.block_analysis,
        manual_assessment=args.manual_assessment,
        next_scene=args.next_scene,
    )
    _write_outputs(payload, args.output_json, args.output_csv)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scene": payload["scene"],
                "normal_gates": payload["normal_gates"]["status"],
                "catastrophic_gates": payload["catastrophic_gates"]["status"],
                "action": payload["continuation_decision"]["action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
