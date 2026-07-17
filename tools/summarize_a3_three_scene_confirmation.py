#!/usr/bin/env python3
"""Summarize the frozen A3-versus-L confirmation across three formal scenes.

The input is one compact scene summary for each of ``Building``,
``InternalRoad``, and ``Urban20K``.  JSON inputs use this contract::

    {
      "schema": "uav-tgs-a3-scene-l-vs-a3-summary-v1",
      "status": "complete",
      "scene": "Building",
      "counts": {"train_views": 1, "test_views": 1, "guard_views": 1,
                 "test_blocks": 1},
      "methods": {
        "L":  {"metrics": {"PSNR": 20.0, "SSIM": 0.7, ...}},
        "A3": {"metrics": {"PSNR": 20.1, "SSIM": 0.7, ...}}
      },
      "invariants": {"all_passed": true,
                     "checks": {"spatial_fields_exact": true}},
      "reviews": {
        "material_off_lut_degradation": false,
        "opacity_saturation": false,
        "pipeline_reference_irreparable": false
      }
    }

CSV inputs contain exactly two rows (``method`` L and A3).  Metadata is
repeated on both rows.  Metrics use ``metric__<name>`` columns, counts use
``count__<name>``, and invariant details use ``invariant__<name>``.  Required
metadata columns are ``schema``, ``status``, ``scene``, ``method``,
``invariants_passed``, ``material_off_lut_degradation``,
``opacity_saturation``, and ``pipeline_reference_irreparable``.

The decision is deliberately scene-macro: every scene has weight one,
regardless of view or supported-pixel count.  Reference-depth visibility
changes are diagnostic only and never described as Gaussian-center,
covariance, or true-geometry improvement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA = "uav-tgs-a3-scene-l-vs-a3-summary-v1"
OUTPUT_SCHEMA = "uav-tgs-a3-three-scene-confirmation-v1"
EXPECTED_SCENES = ("Building", "InternalRoad", "Urban20K")
METHODS = ("L", "A3")
NUMERIC_EPS = 1e-12

# These thresholds are frozen before InternalRoad/Urban20K results are read.
# They implement the confirmation protocol; they are not tuned per scene.
NORMAL_THRESHOLDS = {
    "PSNR_loss_db_max": 0.15,
    "SSIM_loss_max": 0.006,
    "LPIPS_increase_max": 0.008,
    "temperature_mae_abs_increase_c_max": 0.10,
    "temperature_mae_relative_increase_max": 0.05,
    "temperature_rmse_abs_increase_c_max": 0.10,
    "temperature_rmse_relative_increase_max": 0.05,
    "front_intrusion_1m_increase_max": 0.02,
    "missing_rate_increase_max": 0.002,
}
CATASTROPHIC_THRESHOLDS = {
    "PSNR_loss_db_gt": 0.30,
    "temperature_mae_abs_increase_c_gt": 0.20,
    "temperature_mae_relative_increase_gt": 0.10,
    "front_intrusion_1m_increase_gt": 0.03,
}

# Exact ``methods[*].metrics`` contract emitted by the formal scene collector at
# the 60k decision endpoint.  Keep this fixed: accepting a subset or aggregating
# a cross-scene intersection would silently discard diagnostics from the macro.
METHOD_METRICS = (
    "PSNR",
    "SSIM",
    "LPIPS",
    "EdgeHaloScore",
    "temperature_mae_c",
    "temperature_rmse_c",
    "temperature_bias_c",
    "temperature_abs_bias_c",
    "temperature_p95_c",
    "temperature_clipping_ratio",
    "off_lut_mean_rgb_distance",
    "off_lut_p95_rgb_distance",
    "front_intrusion_1m",
    "front_intrusion_2m",
    "front_intrusion_5m",
    "depth_agreement_1m",
    "depth_agreement_2m",
    "depth_agreement_5m",
    "behind_1m",
    "behind_2m",
    "behind_5m",
    "depth_mean_m",
    "depth_median_m",
    "missing_rate",
)
# Retain the public name used by earlier callers, now bound to the full contract.
REQUIRED_METRICS = METHOD_METRICS

METRIC_DIRECTIONS = {
    "PSNR": "higher",
    "SSIM": "higher",
    "LPIPS": "lower",
    "EdgeHaloScore": "lower_auxiliary",
    "temperature_mae_c": "lower",
    "temperature_rmse_c": "lower",
    "temperature_abs_bias_c": "lower",
    "temperature_bias_c": "signed_diagnostic",
    "temperature_p95_c": "lower",
    "temperature_clipping_ratio": "lower_diagnostic",
    "off_lut_mean_rgb_distance": "lower_diagnostic",
    "off_lut_p95_rgb_distance": "lower_diagnostic",
    "front_intrusion_1m": "lower_reference_depth_diagnostic",
    "front_intrusion_2m": "lower_reference_depth_diagnostic",
    "front_intrusion_5m": "lower_reference_depth_diagnostic",
    "depth_agreement_1m": "higher_reference_depth_diagnostic",
    "depth_agreement_2m": "higher_reference_depth_diagnostic",
    "depth_agreement_5m": "higher_reference_depth_diagnostic",
    "behind_1m": "lower_reference_depth_diagnostic",
    "behind_2m": "lower_reference_depth_diagnostic",
    "behind_5m": "lower_reference_depth_diagnostic",
    "depth_mean_m": "lower_reference_depth_diagnostic",
    "depth_median_m": "lower_reference_depth_diagnostic",
    "missing_rate": "lower_reference_depth_diagnostic",
}

PROBABILITY_METRICS = {
    "SSIM",
    "temperature_clipping_ratio",
    "front_intrusion_1m",
    "front_intrusion_2m",
    "front_intrusion_5m",
    "depth_agreement_1m",
    "depth_agreement_2m",
    "depth_agreement_5m",
    "behind_1m",
    "behind_2m",
    "behind_5m",
    "missing_rate",
}


class ThreeSceneSummaryError(RuntimeError):
    """Raised when an input cannot support the frozen confirmation decision."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ThreeSceneSummaryError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ThreeSceneSummaryError(f"{label} is non-finite: {result!r}")
    return result


def _strict_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ThreeSceneSummaryError(f"{label} must be true or false, got {value!r}")


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ThreeSceneSummaryError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ThreeSceneSummaryError(f"{label} must be a non-negative integer") from exc
    if result < 0 or str(result) != str(value).strip():
        raise ThreeSceneSummaryError(f"{label} must be a non-negative integer")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThreeSceneSummaryError(f"cannot read JSON scene summary: {path}") from exc
    if not isinstance(value, dict):
        raise ThreeSceneSummaryError(f"JSON scene summary root must be an object: {path}")
    return value


def _consistent_csv_value(rows: Sequence[Mapping[str, str]], key: str, path: Path) -> str:
    values = {str(row.get(key, "")).strip() for row in rows}
    if len(values) != 1 or "" in values:
        raise ThreeSceneSummaryError(
            f"CSV metadata column {key!r} must be non-empty and identical on both rows: {path}"
        )
    return next(iter(values))


def _load_csv(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())
    except OSError as exc:
        raise ThreeSceneSummaryError(f"cannot read CSV scene summary: {path}") from exc
    if len(rows) != 2:
        raise ThreeSceneSummaryError(f"CSV scene summary must contain exactly two rows: {path}")
    required_columns = {
        "schema",
        "status",
        "scene",
        "method",
        "invariants_passed",
        "material_off_lut_degradation",
        "opacity_saturation",
        "pipeline_reference_irreparable",
    }
    missing_columns = sorted(required_columns - set(fieldnames))
    if missing_columns:
        raise ThreeSceneSummaryError(f"CSV scene summary lacks columns: {missing_columns}")
    by_method: dict[str, Mapping[str, str]] = {}
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method in by_method:
            raise ThreeSceneSummaryError(f"CSV scene summary repeats method {method!r}: {path}")
        by_method[method] = row
    if set(by_method) != set(METHODS):
        raise ThreeSceneSummaryError(
            f"CSV scene summary methods must be exactly {list(METHODS)}: {path}"
        )

    metric_columns = sorted(name for name in fieldnames if name.startswith("metric__"))
    count_columns = sorted(name for name in fieldnames if name.startswith("count__"))
    invariant_columns = sorted(name for name in fieldnames if name.startswith("invariant__"))
    if not metric_columns:
        raise ThreeSceneSummaryError(f"CSV scene summary has no metric__ columns: {path}")
    if not count_columns:
        raise ThreeSceneSummaryError(f"CSV scene summary has no count__ columns: {path}")

    methods: dict[str, Any] = {}
    for method in METHODS:
        row = by_method[method]
        methods[method] = {
            "metrics": {
                name.removeprefix("metric__"): row.get(name, "") for name in metric_columns
            }
        }
    counts = {
        name.removeprefix("count__"): _consistent_csv_value(rows, name, path)
        for name in count_columns
    }
    checks = {
        name.removeprefix("invariant__"): _strict_bool(
            _consistent_csv_value(rows, name, path), f"{path}/{name}"
        )
        for name in invariant_columns
    }
    # A compact CSV may only carry the aggregate invariant.  Preserve a named
    # check so the downstream contract remains explicit and non-empty.
    aggregate_invariant = _strict_bool(
        _consistent_csv_value(rows, "invariants_passed", path),
        f"{path}/invariants_passed",
    )
    if not checks:
        checks = {"csv_aggregate_invariant": aggregate_invariant}

    return {
        "schema": _consistent_csv_value(rows, "schema", path),
        "status": _consistent_csv_value(rows, "status", path),
        "scene": _consistent_csv_value(rows, "scene", path),
        "counts": counts,
        "methods": methods,
        "invariants": {"all_passed": aggregate_invariant, "checks": checks},
        "reviews": {
            "material_off_lut_degradation": _strict_bool(
                _consistent_csv_value(rows, "material_off_lut_degradation", path),
                f"{path}/material_off_lut_degradation",
            ),
            "opacity_saturation": _strict_bool(
                _consistent_csv_value(rows, "opacity_saturation", path),
                f"{path}/opacity_saturation",
            ),
            "pipeline_reference_irreparable": _strict_bool(
                _consistent_csv_value(rows, "pipeline_reference_irreparable", path),
                f"{path}/pipeline_reference_irreparable",
            ),
        },
    }


def _validate_scene_payload(payload: Mapping[str, Any], expected_scene: str) -> dict[str, Any]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise ThreeSceneSummaryError(
            f"{expected_scene} schema must be {INPUT_SCHEMA!r}, got {payload.get('schema')!r}"
        )
    if payload.get("status") != "complete":
        raise ThreeSceneSummaryError(f"{expected_scene} summary status must be 'complete'")
    if payload.get("scene") != expected_scene:
        raise ThreeSceneSummaryError(
            f"scene assignment mismatch: expected={expected_scene!r} embedded={payload.get('scene')!r}"
        )

    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        raise ThreeSceneSummaryError(f"{expected_scene} counts must be a non-empty object")
    counts = {
        str(key): _nonnegative_int(value, f"{expected_scene}/counts/{key}")
        for key, value in sorted(raw_counts.items())
    }
    for required in ("test_views", "test_blocks"):
        if required not in counts or counts[required] <= 0:
            raise ThreeSceneSummaryError(
                f"{expected_scene} counts must include positive {required!r}"
            )

    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, dict) or set(raw_methods) != set(METHODS):
        raise ThreeSceneSummaryError(
            f"{expected_scene} methods must be exactly {list(METHODS)}"
        )
    raw_metric_payloads: dict[str, Mapping[str, Any]] = {}
    metric_sets: dict[str, set[str]] = {}
    for method in METHODS:
        method_payload = raw_methods[method]
        if not isinstance(method_payload, dict) or not isinstance(
            method_payload.get("metrics"), dict
        ):
            raise ThreeSceneSummaryError(f"{expected_scene}/{method} lacks a metrics object")
        raw_metrics = method_payload["metrics"]
        if not all(isinstance(metric, str) for metric in raw_metrics):
            raise ThreeSceneSummaryError(
                f"{expected_scene}/{method} metric names must all be strings"
            )
        raw_metric_payloads[method] = raw_metrics
        metric_sets[method] = set(raw_metrics)
    if metric_sets["L"] != metric_sets["A3"]:
        raise ThreeSceneSummaryError(
            f"{expected_scene} L and A3 metric sets must match exactly; "
            f"L_only={sorted(metric_sets['L'] - metric_sets['A3'])} "
            f"A3_only={sorted(metric_sets['A3'] - metric_sets['L'])}"
        )
    expected_metrics = set(METHOD_METRICS)
    methods: dict[str, dict[str, float]] = {}
    for method in METHODS:
        actual_metrics = metric_sets[method]
        missing_metrics = sorted(expected_metrics - actual_metrics)
        extra_metrics = sorted(actual_metrics - expected_metrics)
        if missing_metrics or extra_metrics:
            raise ThreeSceneSummaryError(
                f"{expected_scene}/{method} metric set must exactly match the formal "
                f"60k collector contract; missing={missing_metrics} extra={extra_metrics}"
            )
        methods[method] = {
            metric: _finite_float(
                raw_metric_payloads[method][metric],
                f"{expected_scene}/{method}/metrics/{metric}",
            )
            for metric in METHOD_METRICS
        }
    for method in METHODS:
        for metric in PROBABILITY_METRICS & set(methods[method]):
            value = methods[method][metric]
            if not 0.0 <= value <= 1.0:
                raise ThreeSceneSummaryError(
                    f"{expected_scene}/{method}/{metric} must lie in [0, 1]"
                )
    for method in METHODS:
        if methods[method]["temperature_mae_c"] < 0:
            raise ThreeSceneSummaryError(
                f"{expected_scene}/{method}/temperature_mae_c must be non-negative"
            )

    raw_invariants = payload.get("invariants")
    if not isinstance(raw_invariants, dict):
        raise ThreeSceneSummaryError(f"{expected_scene} lacks invariants object")
    raw_checks = raw_invariants.get("checks")
    if not isinstance(raw_checks, dict) or not raw_checks:
        raise ThreeSceneSummaryError(f"{expected_scene} invariant checks must be non-empty")
    checks = {
        str(name): _strict_bool(value, f"{expected_scene}/invariants/{name}")
        for name, value in sorted(raw_checks.items())
    }
    all_passed = _strict_bool(
        raw_invariants.get("all_passed"), f"{expected_scene}/invariants/all_passed"
    )
    if all_passed != all(checks.values()):
        raise ThreeSceneSummaryError(
            f"{expected_scene} invariants.all_passed disagrees with invariant checks"
        )

    raw_reviews = payload.get("reviews")
    if not isinstance(raw_reviews, dict):
        raise ThreeSceneSummaryError(f"{expected_scene} lacks reviews object")
    review_names = (
        "material_off_lut_degradation",
        "opacity_saturation",
        "pipeline_reference_irreparable",
    )
    reviews = {
        name: _strict_bool(raw_reviews.get(name), f"{expected_scene}/reviews/{name}")
        for name in review_names
    }

    return {
        "scene": expected_scene,
        "counts": counts,
        "methods": methods,
        "invariants": {"all_passed": all_passed, "checks": checks},
        "reviews": reviews,
    }


def load_scene_summary(path: Path, expected_scene: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json(path)
    elif suffix == ".csv":
        payload = _load_csv(path)
    else:
        raise ThreeSceneSummaryError(
            f"scene summary must be JSON or CSV, got {path.suffix!r}: {path}"
        )
    result = _validate_scene_payload(payload, expected_scene)
    result["source"] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def _temperature_error_check(
    l_value: float, a3_value: float, *, metric: str
) -> dict[str, Any]:
    if metric not in ("mae", "rmse"):
        raise ValueError(f"unsupported temperature error metric: {metric}")
    delta = a3_value - l_value
    relative = None if delta > 0 and l_value == 0 else (0.0 if delta <= 0 else delta / l_value)
    absolute_limit = NORMAL_THRESHOLDS[f"temperature_{metric}_abs_increase_c_max"]
    relative_limit = NORMAL_THRESHOLDS[f"temperature_{metric}_relative_increase_max"]
    abs_pass = delta <= absolute_limit + NUMERIC_EPS
    relative_pass = bool(
        relative is not None
        and relative <= relative_limit + NUMERIC_EPS
    )
    return {
        "L": l_value,
        "A3": a3_value,
        "A3_minus_L": delta,
        "relative_increase": relative,
        "absolute_rule_passed": abs_pass,
        "relative_rule_passed": relative_pass,
        "passed": bool(abs_pass or relative_pass),
        "rule": f"temperature {metric.upper()}: A3-L <= 0.10 C OR positive relative increase <= 5%",
    }


def _temperature_mae_check(l_value: float, a3_value: float) -> dict[str, Any]:
    """Backward-compatible named helper for the catastrophic-primary MAE metric."""

    return _temperature_error_check(l_value, a3_value, metric="mae")


def _normal_checks(l_metrics: Mapping[str, float], a3_metrics: Mapping[str, float]) -> dict[str, Any]:
    deltas = {metric: a3_metrics[metric] - l_metrics[metric] for metric in l_metrics}
    checks: dict[str, Any] = {
        "PSNR": {
            "A3_minus_L": deltas["PSNR"],
            "passed": deltas["PSNR"]
            >= -NORMAL_THRESHOLDS["PSNR_loss_db_max"] - NUMERIC_EPS,
            "rule": "A3-L >= -0.15 dB",
        },
        "SSIM": {
            "A3_minus_L": deltas["SSIM"],
            "passed": deltas["SSIM"]
            >= -NORMAL_THRESHOLDS["SSIM_loss_max"] - NUMERIC_EPS,
            "rule": "A3-L >= -0.006",
        },
        "LPIPS": {
            "A3_minus_L": deltas["LPIPS"],
            "passed": deltas["LPIPS"]
            <= NORMAL_THRESHOLDS["LPIPS_increase_max"] + NUMERIC_EPS,
            "rule": "A3-L <= +0.008",
        },
        "temperature_mae_c": _temperature_mae_check(
            l_metrics["temperature_mae_c"], a3_metrics["temperature_mae_c"]
        ),
        "temperature_rmse_c": _temperature_error_check(
            l_metrics["temperature_rmse_c"],
            a3_metrics["temperature_rmse_c"],
            metric="rmse",
        ),
        "front_intrusion_1m": {
            "A3_minus_L": deltas["front_intrusion_1m"],
            "passed": deltas["front_intrusion_1m"]
            <= NORMAL_THRESHOLDS["front_intrusion_1m_increase_max"] + NUMERIC_EPS,
            "rule": "A3-L <= +0.02 (2 percentage points)",
        },
        "missing_rate": {
            "A3_minus_L": deltas["missing_rate"],
            "passed": deltas["missing_rate"]
            <= NORMAL_THRESHOLDS["missing_rate_increase_max"] + NUMERIC_EPS,
            "rule": "A3-L <= +0.002 (0.2 percentage points)",
        },
    }
    return checks


def _catastrophic_checks(
    l_metrics: Mapping[str, float],
    a3_metrics: Mapping[str, float],
    invariants_passed: bool,
    reviews: Mapping[str, bool],
) -> dict[str, bool]:
    psnr_loss = l_metrics["PSNR"] - a3_metrics["PSNR"]
    temp_delta = a3_metrics["temperature_mae_c"] - l_metrics["temperature_mae_c"]
    temp_relative = (
        math.inf
        if temp_delta > 0 and l_metrics["temperature_mae_c"] == 0
        else (temp_delta / l_metrics["temperature_mae_c"] if temp_delta > 0 else 0.0)
    )
    front_delta = a3_metrics["front_intrusion_1m"] - l_metrics["front_intrusion_1m"]
    return {
        "PSNR_loss_gt_0_30_db": psnr_loss
        > CATASTROPHIC_THRESHOLDS["PSNR_loss_db_gt"] + NUMERIC_EPS,
        "temperature_mae_increase_gt_0_20c_and_10pct": (
            temp_delta
            > CATASTROPHIC_THRESHOLDS["temperature_mae_abs_increase_c_gt"]
            + NUMERIC_EPS
            and temp_relative
            > CATASTROPHIC_THRESHOLDS["temperature_mae_relative_increase_gt"]
            + NUMERIC_EPS
        ),
        "front_intrusion_1m_increase_gt_0_03": front_delta
        > CATASTROPHIC_THRESHOLDS["front_intrusion_1m_increase_gt"] + NUMERIC_EPS,
        "invariant_failure": not invariants_passed,
        "opacity_saturation": reviews["opacity_saturation"],
        "pipeline_reference_irreparable": reviews["pipeline_reference_irreparable"],
    }


def _scene_result(scene: Mapping[str, Any]) -> dict[str, Any]:
    l_metrics = scene["methods"]["L"]
    a3_metrics = scene["methods"]["A3"]
    deltas = {metric: a3_metrics[metric] - l_metrics[metric] for metric in l_metrics}
    checks = _normal_checks(l_metrics, a3_metrics)
    appearance_passed = all(checks[name]["passed"] for name in ("PSNR", "SSIM", "LPIPS"))
    off_lut_review_passed = not scene["reviews"]["material_off_lut_degradation"]
    temperature_passed = bool(
        checks["temperature_mae_c"]["passed"]
        and checks["temperature_rmse_c"]["passed"]
        and off_lut_review_passed
    )
    catastrophic = _catastrophic_checks(
        l_metrics,
        a3_metrics,
        scene["invariants"]["all_passed"],
        scene["reviews"],
    )
    both_off_lut_worse_directionally = bool(
        deltas["off_lut_mean_rgb_distance"] > 0
        and deltas["off_lut_p95_rgb_distance"] > 0
    )
    return {
        "counts": scene["counts"],
        "source": scene["source"],
        "metrics": {"L": l_metrics, "A3": a3_metrics},
        "A3_minus_L": deltas,
        "normal_gate_checks": checks,
        "scene_classification": {
            "appearance_improved_or_tied_under_frozen_tolerance": appearance_passed,
            "temperature_improved_or_tied_under_frozen_tolerance": temperature_passed,
            "appearance_and_temperature_improved_or_tied": bool(
                appearance_passed and temperature_passed
            ),
            "invariants_passed": scene["invariants"]["all_passed"],
            "normal_visibility_gate_passed": bool(
                checks["front_intrusion_1m"]["passed"]
                and checks["missing_rate"]["passed"]
            ),
        },
        "invariants": scene["invariants"],
        "reviews": {
            **scene["reviews"],
            "both_off_lut_metrics_worse_directionally": both_off_lut_worse_directionally,
            "note": (
                "Directional off-LUT worsening is reported separately; the preregistered "
                "material-degradation review flag is not inferred from a post-hoc threshold."
            ),
        },
        "catastrophic_checks": {
            **catastrophic,
            "any": any(catastrophic.values()),
        },
    }


def _aggregate_counts(scene_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = sorted(set().union(*(result["counts"].keys() for result in scene_results.values())))
    totals: dict[str, int] = {}
    for key in keys:
        if all(key in result["counts"] for result in scene_results.values()):
            totals[key] = sum(result["counts"][key] for result in scene_results.values())
    return {
        "by_scene": {scene: scene_results[scene]["counts"] for scene in EXPECTED_SCENES},
        "descriptive_sum_for_common_count_fields": totals,
        "note": (
            "Counts are descriptive only. Scene-macro metrics give each scene weight one; "
            "the sums do not assert independent geographic domains."
        ),
    }


def summarize(scene_inputs: Mapping[str, Path]) -> dict[str, Any]:
    if set(scene_inputs) != set(EXPECTED_SCENES):
        raise ThreeSceneSummaryError(
            "scene inputs must be exactly Building, InternalRoad, and Urban20K; "
            f"missing={sorted(set(EXPECTED_SCENES) - set(scene_inputs))} "
            f"extra={sorted(set(scene_inputs) - set(EXPECTED_SCENES))}"
        )
    loaded = {
        scene: load_scene_summary(Path(scene_inputs[scene]), scene)
        for scene in EXPECTED_SCENES
    }
    scene_results = {scene: _scene_result(loaded[scene]) for scene in EXPECTED_SCENES}
    macro_methods: dict[str, dict[str, float]] = {method: {} for method in METHODS}
    for method in METHODS:
        for metric in METHOD_METRICS:
            macro_methods[method][metric] = sum(
                scene_results[scene]["metrics"][method][metric] for scene in EXPECTED_SCENES
            ) / len(EXPECTED_SCENES)
    macro_deltas = {
        metric: macro_methods["A3"][metric] - macro_methods["L"][metric]
        for metric in METHOD_METRICS
    }
    macro_checks = _normal_checks(macro_methods["L"], macro_methods["A3"])
    macro_non_inferior = all(check["passed"] for check in macro_checks.values())

    appearance_count = sum(
        result["scene_classification"][
            "appearance_improved_or_tied_under_frozen_tolerance"
        ]
        for result in scene_results.values()
    )
    temperature_count = sum(
        result["scene_classification"][
            "temperature_improved_or_tied_under_frozen_tolerance"
        ]
        for result in scene_results.values()
    )
    joint_count = sum(
        result["scene_classification"][
            "appearance_and_temperature_improved_or_tied"
        ]
        for result in scene_results.values()
    )
    invariant_count = sum(
        result["scene_classification"]["invariants_passed"]
        for result in scene_results.values()
    )
    catastrophic_visibility_scenes = [
        scene
        for scene, result in scene_results.items()
        if result["catastrophic_checks"]["front_intrusion_1m_increase_gt_0_03"]
    ]
    non_gate_catastrophic_conditions = (
        "PSNR_loss_gt_0_30_db",
        "temperature_mae_increase_gt_0_20c_and_10pct",
        "opacity_saturation",
        "pipeline_reference_irreparable",
    )
    non_gate_catastrophic_scenes = {
        condition: [
            scene
            for scene, result in scene_results.items()
            if result["catastrophic_checks"][condition]
        ]
        for condition in non_gate_catastrophic_conditions
    }
    required_gates = {
        "at_least_2_of_3_appearance_and_temperature_improved_or_tied": joint_count >= 2,
        "unweighted_scene_macro_non_inferior": macro_non_inferior,
        "all_3_scene_invariants_passed": invariant_count == 3,
        "no_catastrophic_reference_depth_visibility": not catastrophic_visibility_scenes,
    }
    freeze = all(required_gates.values())

    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "protocol": {
            "methods": list(METHODS),
            "scenes": list(EXPECTED_SCENES),
            "decision_endpoint": 60000,
            "scene_weighting": "unweighted arithmetic mean; one vote per scene",
            "normal_thresholds": NORMAL_THRESHOLDS,
            "catastrophic_thresholds": CATASTROPHIC_THRESHOLDS,
            "improve_or_tie_semantics": (
                "A pass includes values inside the frozen non-inferiority band; "
                "it does not assert exact equality or strict improvement."
            ),
            "catastrophic_condition_roles": (
                "PSNR, temperature-MAE, opacity-saturation, and irreparable-pipeline "
                "conditions are retained as scene-to-scene stop/continue diagnostics. "
                "The frozen final recipe rule has exactly four gates and adds no fifth "
                "gate; its catastrophic gate is reference-depth front intrusion only."
            ),
        },
        "scene_counts": _aggregate_counts(scene_results),
        "scenes": scene_results,
        "scene_macro": {
            "scene_count": len(EXPECTED_SCENES),
            "metrics_included": list(METHOD_METRICS),
            "methods": macro_methods,
            "A3_minus_L": macro_deltas,
            "normal_gate_checks": macro_checks,
            "non_inferior": macro_non_inferior,
        },
        "confirmation_counts": {
            "appearance_improved_or_tied_scenes": appearance_count,
            "temperature_improved_or_tied_scenes": temperature_count,
            "appearance_and_temperature_improved_or_tied_scenes": joint_count,
            "invariant_passed_scenes": invariant_count,
            "total_scenes": len(EXPECTED_SCENES),
        },
        "freeze_decision": {
            "required_gates": required_gates,
            "catastrophic_visibility_scenes": catastrophic_visibility_scenes,
            "non_gate_catastrophic_diagnostics": {
                "scenes_by_condition": non_gate_catastrophic_scenes,
                "role": (
                    "Reported for the preregistered scene-to-scene stop/continue audit; "
                    "not an additional final freeze gate."
                ),
            },
            "freeze_a3_recipe": freeze,
            "action": (
                "freeze A3 and stop Stage-2 recipe search"
                if freeze
                else "do not freeze A3; review failed preregistered gate(s)"
            ),
            "rule": (
                ">=2/3 scenes improve-or-tie on both appearance and temperature, "
                "unweighted scene-macro is non-inferior, 3/3 invariants pass, and "
                "no scene has catastrophic reference-depth front-intrusion degradation"
            ),
        },
        "claim_boundaries": {
            "geometry": (
                "Reference-depth and visibility deltas may arise from modality-specific "
                "opacity/visibility. They are reference-depth consistency diagnostics and "
                "do not establish Gaussian-center, covariance, or true-geometry improvement."
            ),
            "temperature": (
                "Temperature metrics are TSDK-referenced apparent-temperature consistency, "
                "not absolute thermometry or true surface-temperature accuracy."
            ),
            "scene_scope": (
                "The three scenes are confirmation cases; their counts and macro average do "
                "not imply three independent geographic domains."
            ),
        },
    }


def _parse_scene_assignments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        scene, separator, assigned = value.partition("=")
        if not separator or not scene or not assigned:
            raise ThreeSceneSummaryError(
                f"--scene-summary must use Scene=PATH, got {value!r}"
            )
        if scene in result:
            raise ThreeSceneSummaryError(f"duplicate scene assignment: {scene}")
        result[scene] = Path(assigned).resolve()
    return result


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    fieldnames = (
        "row_type",
        "scene",
        "metric",
        "direction",
        "L",
        "A3",
        "A3_minus_L",
        "appearance_and_temperature_improved_or_tied",
        "invariants_passed",
        "catastrophic_visibility",
    )
    rows: list[dict[str, Any]] = []
    for scene in EXPECTED_SCENES:
        result = payload["scenes"][scene]
        for metric, delta in result["A3_minus_L"].items():
            rows.append(
                {
                    "row_type": "scene",
                    "scene": scene,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS.get(metric, "reported_only"),
                    "L": result["metrics"]["L"][metric],
                    "A3": result["metrics"]["A3"][metric],
                    "A3_minus_L": delta,
                    "appearance_and_temperature_improved_or_tied": result[
                        "scene_classification"
                    ]["appearance_and_temperature_improved_or_tied"],
                    "invariants_passed": result["scene_classification"][
                        "invariants_passed"
                    ],
                    "catastrophic_visibility": result["catastrophic_checks"][
                        "front_intrusion_1m_increase_gt_0_03"
                    ],
                }
            )
    for metric, delta in payload["scene_macro"]["A3_minus_L"].items():
        rows.append(
            {
                "row_type": "scene_macro",
                "scene": "ALL_UNWEIGHTED",
                "metric": metric,
                "direction": METRIC_DIRECTIONS.get(metric, "reported_only"),
                "L": payload["scene_macro"]["methods"]["L"][metric],
                "A3": payload["scene_macro"]["methods"]["A3"][metric],
                "A3_minus_L": delta,
                "appearance_and_temperature_improved_or_tied": "",
                "invariants_passed": "",
                "catastrophic_visibility": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-summary",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="Repeat exactly for Building, InternalRoad, and Urban20K.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    parser.add_argument("--csv", type=Path, help="Optional long-form CSV output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene_inputs = _parse_scene_assignments(args.scene_summary)
    payload = summarize(scene_inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.csv is not None:
        _write_csv(args.csv, payload)
    print(json.dumps(payload["freeze_decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
