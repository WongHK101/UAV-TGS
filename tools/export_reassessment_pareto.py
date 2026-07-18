#!/usr/bin/env python3
"""Export the unconstrained UAV-TGS reassessment Pareto evidence package.

The input is a normalized, one-row-per-scene/method CSV.  Structural design and
``evidence_status`` are retained as metadata only and never gate a method.  A
point is excluded from a particular Pareto comparison only when an axis, its
source provenance, or render-comparability metadata is absent.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


SCHEMA = "uav-tgs-unconstrained-reassessment-pareto-export-v1"
FIGURE_STEM = "reassessment_pareto_4panel"

IDENTITY_COLUMNS = ("scene", "method_id", "display_name", "evidence_status")
REQUIRED_IDENTITY_COLUMNS = ("scene", "method_id", "display_name")
ENDPOINT_METADATA_COLUMNS = (
    "schema_version",
    "geometry_endpoint_id",
    "rgb_endpoint_id",
    "thermal_endpoint_id",
    "depth_source_id",
)
STRUCTURAL_METADATA_COLUMN = "structural_metadata_json"
HOTSPOT_DOMAIN_COLUMN = "hotspot_metric_domain"
HOTSPOT_THRESHOLD_RULE_COLUMN = "hotspot_threshold_rule"
HOTSPOT_THRESHOLD_BINS_COLUMN = "hotspot_threshold_histogram_bins"
HOTSPOT_THRESHOLD_SHA_COLUMN = "hotspot_threshold_sha256"

HOTSPOT_DISPLAY_DOMAIN = "display_temperature_c"
HOTSPOT_THRESHOLD_RULE = "frozen_train_q95"
HOTSPOT_THRESHOLD_BINS = 65536

FLOAT_COLUMNS = (
    "rgb_psnr_db",
    "rgb_ssim",
    "rgb_lpips",
    "thermal_psnr_db",
    "thermal_ssim",
    "thermal_lpips",
    "temp_mae_direct_c",
    "temp_rmse_direct_c",
    "temp_bias_direct_c",
    "temp_p95_direct_c",
    "temp_mae_display_c",
    "temp_rmse_display_c",
    "temp_bias_display_c",
    "temp_p95_display_c",
    "hotspot_auprc_display",
    "hotspot_iou_display",
    "front_auc_expected",
    "front_auc_median",
    "front_auc_max_contribution",
    "agreement_auc_expected",
    "agreement_auc_median",
    "agreement_auc_max_contribution",
    "train_wall_time_s",
    "render_ms_per_view",
)
INTEGER_COLUMNS = (
    "gaussian_count",
    "render_width",
    "render_height",
    HOTSPOT_THRESHOLD_BINS_COLUMN,
)
TRAIN_TIMING_COLUMN = "train_timing_scope"
RENDER_TIMING_COLUMN = "render_timing_scope"
LEGACY_TIMING_COLUMN = "timing_scope"

SOURCE_SHA_COLUMNS = (
    "formal_split_sha256",
    "reference_manifest_sha256",
    "appearance_source_sha256",
    "temperature_source_sha256",
    "hotspot_source_sha256",
    HOTSPOT_THRESHOLD_SHA_COLUMN,
    "depth_source_sha256",
    "efficiency_source_sha256",
    "alias_receipt_sha256",
)

CANONICAL_COLUMNS = (
    *IDENTITY_COLUMNS,
    *ENDPOINT_METADATA_COLUMNS,
    *FLOAT_COLUMNS,
    *INTEGER_COLUMNS,
    TRAIN_TIMING_COLUMN,
    RENDER_TIMING_COLUMN,
    HOTSPOT_DOMAIN_COLUMN,
    HOTSPOT_THRESHOLD_RULE_COLUMN,
    STRUCTURAL_METADATA_COLUMN,
    *SOURCE_SHA_COLUMNS,
)

# ``timing_scope`` was the original render-timing field.  It remains accepted
# at input only; every normalized output uses the two independent scope fields.
INPUT_ALIASES: dict[str, str] = {LEGACY_TIMING_COLUMN: RENDER_TIMING_COLUMN}
FORBIDDEN_GENERIC_HOTSPOT_COLUMNS = {"hotspot_auprc", "hotspot_iou"}
MISSING_TOKENS = {"", "null", "none"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN_DESIGN_MARKERS = ("feasible", "infeasible", "feasibility")
METHOD_COLORS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
NOOP_ALIAS_FIGURE_NOTE = (
    "Scene-level explicit no-op aliases share their declared source marker "
    "(one measured endpoint)."
)

METRIC_DIRECTIONS = {
    "rgb_psnr_db": "max",
    "rgb_ssim": "max",
    "rgb_lpips": "min",
    "thermal_psnr_db": "max",
    "thermal_ssim": "max",
    "thermal_lpips": "min",
    "temp_mae_direct_c": "min",
    "temp_rmse_direct_c": "min",
    "temp_p95_direct_c": "min",
    "temp_mae_display_c": "min",
    "temp_rmse_display_c": "min",
    "temp_p95_display_c": "min",
    "hotspot_auprc_display": "max",
    "hotspot_iou_display": "max",
    "front_auc_expected": "min",
    "front_auc_median": "min",
    "front_auc_max_contribution": "min",
    "agreement_auc_expected": "max",
    "agreement_auc_median": "max",
    "agreement_auc_max_contribution": "max",
    "gaussian_count": "min",
    "train_wall_time_s": "min",
    "render_ms_per_view": "min",
}


@dataclass(frozen=True)
class ParetoSpec:
    pareto_id: str
    depth_definition: str
    x_metric: str
    y_metric: str
    family: str

    @property
    def x_direction(self) -> str:
        return METRIC_DIRECTIONS[self.x_metric]

    @property
    def y_direction(self) -> str:
        return METRIC_DIRECTIONS[self.y_metric]


PARETO_SPECS = tuple(
    ParetoSpec(
        f"appearance_vs_front_{depth}",
        depth,
        f"front_auc_{depth}",
        "thermal_lpips",
        "appearance_vs_front",
    )
    for depth in ("expected", "median", "max_contribution")
) + tuple(
    ParetoSpec(
        f"temperature_vs_front_{depth}",
        depth,
        f"front_auc_{depth}",
        "temp_mae_display_c",
        "temperature_vs_front",
    )
    for depth in ("expected", "median", "max_contribution")
) + (
    ParetoSpec(
        "hotspot_vs_temperature",
        "",
        "temp_mae_display_c",
        "hotspot_auprc_display",
        "hotspot_vs_temperature",
    ),
    ParetoSpec(
        "training_cost",
        "",
        "gaussian_count",
        "train_wall_time_s",
        "training_cost",
    ),
    ParetoSpec(
        "render_cost",
        "",
        "gaussian_count",
        "render_ms_per_view",
        "render_cost",
    ),
)


APPEARANCE_METRICS = frozenset(
    {
        "rgb_psnr_db",
        "rgb_ssim",
        "rgb_lpips",
        "thermal_psnr_db",
        "thermal_ssim",
        "thermal_lpips",
    }
)
TEMPERATURE_METRICS = frozenset(
    {
        "temp_mae_direct_c",
        "temp_rmse_direct_c",
        "temp_bias_direct_c",
        "temp_p95_direct_c",
        "temp_mae_display_c",
        "temp_rmse_display_c",
        "temp_bias_display_c",
        "temp_p95_display_c",
    }
)
HOTSPOT_METRICS = frozenset({"hotspot_auprc_display", "hotspot_iou_display"})
DEPTH_METRICS = frozenset(
    {
        "front_auc_expected",
        "front_auc_median",
        "front_auc_max_contribution",
        "agreement_auc_expected",
        "agreement_auc_median",
        "agreement_auc_max_contribution",
    }
)
EFFICIENCY_METRICS = frozenset(
    {"gaussian_count", "train_wall_time_s", "render_ms_per_view"}
)

METRIC_PROVENANCE_FIELDS: dict[str, tuple[str, ...]] = {
    **{metric: ("appearance_source_sha256",) for metric in APPEARANCE_METRICS},
    **{metric: ("temperature_source_sha256",) for metric in TEMPERATURE_METRICS},
    **{
        metric: ("hotspot_source_sha256", HOTSPOT_THRESHOLD_SHA_COLUMN)
        for metric in HOTSPOT_METRICS
    },
    **{
        metric: (
            "formal_split_sha256",
            "reference_manifest_sha256",
            "depth_source_sha256",
        )
        for metric in DEPTH_METRICS
    },
    **{metric: ("efficiency_source_sha256",) for metric in EFFICIENCY_METRICS},
}


MEMBERSHIP_FIELDS = (
    "level",
    "scene",
    "method_id",
    "display_name",
    "evidence_status",
    "pareto_id",
    "depth_definition",
    "x_metric",
    "x_direction",
    "x_value",
    "y_metric",
    "y_direction",
    "y_value",
    "comparison_group",
    "render_width",
    "render_height",
    TRAIN_TIMING_COLUMN,
    RENDER_TIMING_COLUMN,
    "is_member",
    "structural_metadata_json",
)

EXCLUSION_FIELDS = (
    "level",
    "scene",
    "method_id",
    "display_name",
    "pareto_id",
    "depth_definition",
    "comparison_group",
    "exclusion_reason",
    "missing_fields",
    "details",
)


@dataclass
class LoadedRows:
    rows: list[dict[str, Any]]
    extra_columns: list[str]
    inputs: list[dict[str, Any]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_self_hash(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("manifest_sha256", None)
    return canonical_json_sha256(material)


def verify_manifest_self_hash(payload: Mapping[str, Any]) -> bool:
    value = str(payload.get("manifest_sha256", "")).lower()
    return bool(SHA256_RE.fullmatch(value)) and value == manifest_self_hash(payload)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in MISSING_TOKENS else text


def _ascii_figure_label(value: Any) -> str:
    """Return an ASCII-only display label for publication figures."""

    normalized = unicodedata.normalize("NFKD", str(value))
    label = normalized.encode("ascii", "ignore").decode("ascii").strip()
    return label or "unlabeled"


def _find_forbidden_design_marker(value: str) -> str | None:
    lowered = value.casefold()
    return next((marker for marker in FORBIDDEN_DESIGN_MARKERS if marker in lowered), None)


def _assert_no_design_markers(value: Any, context: str, path: str = "$") -> None:
    """Reject legacy design-gate markers in every metadata key or value.

    The walk intentionally includes unknown metadata and ``evidence_status``.
    This is stricter than checking a short deny-list of columns and prevents a
    nested JSON value from reintroducing a method-structure gate into outputs.
    """

    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            marker = _find_forbidden_design_marker(key)
            if marker:
                raise ValueError(
                    f"{context}: forbidden design marker {marker!r} in metadata key {path}.{key}"
                )
            _assert_no_design_markers(nested, context, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_design_markers(nested, context, f"{path}[{index}]")
    elif isinstance(value, str):
        marker = _find_forbidden_design_marker(value)
        if marker:
            raise ValueError(
                f"{context}: forbidden design marker {marker!r} in metadata value {path}"
            )


def _parse_optional_float(value: Any, column: str, context: str) -> float | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError as error:
        raise ValueError(f"{context}: invalid numeric {column}={text!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{context}: non-finite {column}={text!r}")
    if column in {
        "thermal_lpips",
        "rgb_lpips",
        "temp_mae_direct_c",
        "temp_rmse_direct_c",
        "temp_p95_direct_c",
        "temp_mae_display_c",
        "temp_rmse_display_c",
        "temp_p95_display_c",
        "train_wall_time_s",
        "render_ms_per_view",
    } and parsed < 0:
        raise ValueError(f"{context}: {column} must be non-negative")
    if column.startswith(("front_auc_", "agreement_auc_")) or column in {
        "hotspot_auprc_display",
        "hotspot_iou_display",
        "rgb_ssim",
        "thermal_ssim",
    }:
        if parsed < 0 or parsed > 1:
            raise ValueError(f"{context}: {column} must be in [0, 1]")
    return parsed


def _parse_optional_integer(value: Any, column: str, context: str) -> int | None:
    parsed = _parse_optional_float(value, column, context)
    if parsed is None:
        return None
    if not float(parsed).is_integer() or parsed <= 0:
        raise ValueError(f"{context}: {column} must be a positive integer")
    return int(parsed)


def _canonical_structural_metadata(value: Any, context: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}: structural_metadata_json is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: structural_metadata_json must encode an object")

    _assert_no_design_markers(payload, context, "$.structural_metadata_json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _forbidden_feasibility_columns(columns: Iterable[str]) -> list[str]:
    # Fail closed here: accepting spellings such as ``isFeasible`` would let a
    # legacy constrained table silently masquerade as the reassessment input.
    return [
        column
        for column in columns
        if "feasible" in column.strip().lower()
        or "feasibility" in column.strip().lower()
    ]


def _normalized_source(source: Mapping[str, Any], fields: set[str], context: str) -> dict[str, Any]:
    del context
    return {name: source.get(name, "") for name in fields}


def _scene_metric_provenance_missing(
    row: Mapping[str, Any], metric: str
) -> list[str]:
    missing = [
        field
        for field in METRIC_PROVENANCE_FIELDS.get(metric, ())
        if not SHA256_RE.fullmatch(_clean_text(row.get(field)).lower())
    ]
    if metric in HOTSPOT_METRICS:
        if _clean_text(row.get(HOTSPOT_DOMAIN_COLUMN)) != HOTSPOT_DISPLAY_DOMAIN:
            missing.append(HOTSPOT_DOMAIN_COLUMN)
        if _clean_text(row.get(HOTSPOT_THRESHOLD_RULE_COLUMN)) != HOTSPOT_THRESHOLD_RULE:
            missing.append(HOTSPOT_THRESHOLD_RULE_COLUMN)
        if row.get(HOTSPOT_THRESHOLD_BINS_COLUMN) != HOTSPOT_THRESHOLD_BINS:
            missing.append(HOTSPOT_THRESHOLD_BINS_COLUMN)
    return missing


def _validate_nonempty_metric_provenance(row: Mapping[str, Any], context: str) -> None:
    failures = {
        metric: missing
        for metric in METRIC_PROVENANCE_FIELDS
        if row.get(metric) is not None
        and (missing := _scene_metric_provenance_missing(row, metric))
    }
    if failures:
        raise ValueError(f"{context}: missing or invalid metric provenance: {failures}")


def load_normalized_csvs(paths: Sequence[Path]) -> LoadedRows:
    """Load normalized endpoint rows without feasibility or structural filtering."""

    if not paths:
        raise ValueError("at least one input CSV is required")
    rows: list[dict[str, Any]] = []
    extras: set[str] = set()
    inputs: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        input_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            raw_fields = reader.fieldnames or []
            fields = [str(name).strip() for name in raw_fields]
            if not fields:
                raise ValueError(f"{path.name}: missing CSV header")
            if len(set(fields)) != len(fields):
                raise ValueError(f"{path.name}: duplicate CSV column after trimming")
            forbidden = _forbidden_feasibility_columns(fields)
            if forbidden:
                raise ValueError(f"{path.name}: forbidden feasibility column(s): {forbidden}")
            generic_hotspot = sorted(
                field
                for field in fields
                if field.casefold() in FORBIDDEN_GENERIC_HOTSPOT_COLUMNS
            )
            if generic_hotspot:
                raise ValueError(
                    f"{path.name}: generic hotspot column(s) are forbidden; use explicit display-domain fields: "
                    f"{generic_hotspot}"
                )
            missing_identity = [name for name in REQUIRED_IDENTITY_COLUMNS if name not in fields]
            if missing_identity:
                raise ValueError(f"{path.name}: missing identity columns: {missing_identity}")
            field_map = dict(zip(raw_fields, fields))
            field_set = set(fields)
            extras.update(
                field_set
                - set(CANONICAL_COLUMNS)
                - set(INPUT_ALIASES)
            )
            for row_number, raw in enumerate(reader, start=2):
                source = {field_map[key]: value for key, value in raw.items() if key in field_map}
                if not any(_clean_text(value) for value in source.values()):
                    continue
                context = f"{path.name}:{row_number}"
                source = _normalized_source(source, field_set, context)
                row: dict[str, Any] = {}
                for column in IDENTITY_COLUMNS:
                    row[column] = _clean_text(source.get(column))
                    if column in REQUIRED_IDENTITY_COLUMNS and not row[column]:
                        raise ValueError(f"{context}: {column} is required")
                for column in ENDPOINT_METADATA_COLUMNS:
                    row[column] = _clean_text(source.get(column))
                for column in FLOAT_COLUMNS:
                    row[column] = _parse_optional_float(source.get(column), column, context)
                for column in INTEGER_COLUMNS:
                    row[column] = _parse_optional_integer(source.get(column), column, context)
                train_scope = _clean_text(source.get(TRAIN_TIMING_COLUMN))
                render_scope = _clean_text(source.get(RENDER_TIMING_COLUMN))
                legacy_render_scope = _clean_text(source.get(LEGACY_TIMING_COLUMN))
                if (
                    render_scope
                    and legacy_render_scope
                    and render_scope != legacy_render_scope
                ):
                    raise ValueError(
                        f"{context}: conflicting {RENDER_TIMING_COLUMN}="
                        f"{render_scope!r} and legacy {LEGACY_TIMING_COLUMN}="
                        f"{legacy_render_scope!r}"
                    )
                row[TRAIN_TIMING_COLUMN] = train_scope
                row[RENDER_TIMING_COLUMN] = render_scope or legacy_render_scope
                row[HOTSPOT_DOMAIN_COLUMN] = _clean_text(source.get(HOTSPOT_DOMAIN_COLUMN))
                row[HOTSPOT_THRESHOLD_RULE_COLUMN] = _clean_text(
                    source.get(HOTSPOT_THRESHOLD_RULE_COLUMN)
                )
                row[STRUCTURAL_METADATA_COLUMN] = _canonical_structural_metadata(
                    source.get(STRUCTURAL_METADATA_COLUMN), context
                )
                sha_columns = sorted(
                    set(SOURCE_SHA_COLUMNS)
                    | {name for name in field_set if name.lower().endswith("_sha256")}
                )
                for column in sha_columns:
                    value = _clean_text(source.get(column)).lower()
                    if value and not SHA256_RE.fullmatch(value):
                        raise ValueError(f"{context}: invalid SHA-256 in {column}")
                    row[column] = value
                for column in field_set - set(CANONICAL_COLUMNS) - set(INPUT_ALIASES):
                    if not column.lower().endswith("_sha256"):
                        row[column] = _clean_text(source.get(column))
                _assert_no_design_markers(row, context)
                _validate_nonempty_metric_provenance(row, context)
                row["source_input_name"] = path.name
                row["source_row_number"] = row_number
                rows.append(row)
                input_count += 1
        input_receipt = {
            "name": path.name,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "row_count": input_count,
        }
        _assert_no_design_markers(input_receipt, f"{path.name}: input receipt")
        inputs.append(input_receipt)
    if not rows:
        raise ValueError("no normalized endpoint rows were loaded")
    keys = [(row["scene"], row["method_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/method_id endpoint row")
    displays: dict[str, set[str]] = {}
    for row in rows:
        displays.setdefault(row["method_id"], set()).add(row["display_name"])
    inconsistent = {method: values for method, values in displays.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"inconsistent display_name by method_id: {inconsistent}")
    rows.sort(key=lambda row: (row["scene"], row["method_id"]))
    return LoadedRows(rows=rows, extra_columns=sorted(extras), inputs=inputs)


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any], spec: ParetoSpec) -> bool:
    comparisons = []
    strict = False
    for metric, direction in (
        (spec.x_metric, spec.x_direction),
        (spec.y_metric, spec.y_direction),
    ):
        a = float(left[metric])
        b = float(right[metric])
        weak = a <= b if direction == "min" else a >= b
        better = a < b if direction == "min" else a > b
        comparisons.append(weak)
        strict = strict or better
    return all(comparisons) and strict


def pareto_member_ids(rows: Sequence[Mapping[str, Any]], spec: ParetoSpec) -> set[str]:
    """Return nondominated row IDs with per-axis min/max directions respected."""

    members: set[str] = set()
    for candidate in rows:
        if not any(_dominates(other, candidate, spec) for other in rows if other is not candidate):
            members.add(str(candidate["_row_id"]))
    return members


def _render_context(row: Mapping[str, Any]) -> tuple[int, int, str] | None:
    width = row.get("render_width")
    height = row.get("render_height")
    scope = _clean_text(
        row.get(RENDER_TIMING_COLUMN) or row.get(LEGACY_TIMING_COLUMN)
    )
    if width is None or height is None or not scope:
        return None
    return int(width), int(height), scope


def _train_scope(row: Mapping[str, Any]) -> str | None:
    scope = _clean_text(row.get(TRAIN_TIMING_COLUMN))
    return scope or None


def _metric_provenance_missing(row: Mapping[str, Any], metric: str) -> list[str]:
    """Return missing evidence fields for one metric at scene or macro level."""

    if str(row.get("scene", "")) != "MACRO":
        return _scene_metric_provenance_missing(row, metric)

    coverage = [
        scene for scene in _clean_text(row.get("coverage_scenes")).split(";") if scene
    ]
    if len(coverage) != 2 or len(set(coverage)) != 2:
        return ["coverage_scenes"]
    try:
        sources = json.loads(_clean_text(row.get("source_sha256_by_scene_json")))
    except json.JSONDecodeError:
        return ["source_sha256_by_scene_json"]
    if not isinstance(sources, Mapping):
        return ["source_sha256_by_scene_json"]

    missing: list[str] = []
    for scene in coverage:
        scene_sources = sources.get(scene)
        if not isinstance(scene_sources, Mapping):
            missing.append(f"{scene}:source_sha256_by_scene_json")
            continue
        for field in METRIC_PROVENANCE_FIELDS.get(metric, ()):
            if not SHA256_RE.fullmatch(_clean_text(scene_sources.get(field)).lower()):
                missing.append(f"{scene}:{field}")
    if metric in HOTSPOT_METRICS:
        if row.get(HOTSPOT_DOMAIN_COLUMN) != HOTSPOT_DISPLAY_DOMAIN:
            missing.append(HOTSPOT_DOMAIN_COLUMN)
        if row.get(HOTSPOT_THRESHOLD_RULE_COLUMN) != HOTSPOT_THRESHOLD_RULE:
            missing.append(HOTSPOT_THRESHOLD_RULE_COLUMN)
        if row.get(HOTSPOT_THRESHOLD_BINS_COLUMN) != HOTSPOT_THRESHOLD_BINS:
            missing.append(HOTSPOT_THRESHOLD_BINS_COLUMN)
    return missing


def _axis_provenance_missing(row: Mapping[str, Any], spec: ParetoSpec) -> list[str]:
    missing: list[str] = []
    for metric in (spec.x_metric, spec.y_metric):
        missing.extend(
            f"{metric}:{field}" for field in _metric_provenance_missing(row, metric)
        )
    return sorted(set(missing))


def _membership_record(
    row: Mapping[str, Any],
    spec: ParetoSpec,
    *,
    level: str,
    scene: str,
    comparison_group: str,
    is_member: bool,
) -> dict[str, Any]:
    return {
        "level": level,
        "scene": scene,
        "method_id": row["method_id"],
        "display_name": row["display_name"],
        "evidence_status": row.get("evidence_status", ""),
        "pareto_id": spec.pareto_id,
        "depth_definition": spec.depth_definition,
        "x_metric": spec.x_metric,
        "x_direction": spec.x_direction,
        "x_value": row[spec.x_metric],
        "y_metric": spec.y_metric,
        "y_direction": spec.y_direction,
        "y_value": row[spec.y_metric],
        "comparison_group": comparison_group,
        "render_width": row.get("render_width"),
        "render_height": row.get("render_height"),
        TRAIN_TIMING_COLUMN: row.get(TRAIN_TIMING_COLUMN, ""),
        RENDER_TIMING_COLUMN: row.get(RENDER_TIMING_COLUMN, ""),
        "is_member": is_member,
        "structural_metadata_json": row.get(STRUCTURAL_METADATA_COLUMN, ""),
    }


def _exclusion_record(
    row: Mapping[str, Any],
    spec: ParetoSpec,
    *,
    level: str,
    scene: str,
    comparison_group: str,
    reason: str,
    missing_fields: Sequence[str] = (),
    details: str = "",
) -> dict[str, Any]:
    return {
        "level": level,
        "scene": scene,
        "method_id": row["method_id"],
        "display_name": row["display_name"],
        "pareto_id": spec.pareto_id,
        "depth_definition": spec.depth_definition,
        "comparison_group": comparison_group,
        "exclusion_reason": reason,
        "missing_fields": ";".join(missing_fields),
        "details": details,
    }


def compute_memberships(
    rows: Sequence[dict[str, Any]],
    *,
    level: str,
    scene_label: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute long-form memberships and explicit axis exclusions.

    Training membership groups include scene and training-timing scope. Render
    groups additionally include resolution. No dominance comparison is ever
    made across timing or resolution strata.
    """

    memberships: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for spec in PARETO_SPECS:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            scene = scene_label or str(row["scene"])
            if (
                spec.family == "training_cost"
                and row.get("train_context_consistent") is False
            ):
                exclusions.append(
                    _exclusion_record(
                        row,
                        spec,
                        level=level,
                        scene=scene,
                        comparison_group=scene,
                        reason="noncomparable_training_scope_across_scenes",
                        details=str(row.get("train_contexts_by_scene_json", "")),
                    )
                )
                continue
            if (
                spec.family == "render_cost"
                and row.get("render_context_consistent") is False
            ):
                exclusions.append(
                    _exclusion_record(
                        row,
                        spec,
                        level=level,
                        scene=scene,
                        comparison_group=scene,
                        reason="noncomparable_render_context_across_scenes",
                        details=str(row.get("render_contexts_by_scene_json", "")),
                    )
                )
                continue
            missing = [metric for metric in (spec.x_metric, spec.y_metric) if row.get(metric) is None]
            if missing:
                exclusions.append(
                    _exclusion_record(
                        row,
                        spec,
                        level=level,
                        scene=scene,
                        comparison_group=scene,
                        reason="missing_axis",
                        missing_fields=missing,
                    )
                )
                continue
            provenance_missing = _axis_provenance_missing(row, spec)
            if provenance_missing:
                exclusions.append(
                    _exclusion_record(
                        row,
                        spec,
                        level=level,
                        scene=scene,
                        comparison_group=scene,
                        reason="missing_axis_provenance",
                        missing_fields=provenance_missing,
                    )
                )
                continue
            if spec.family == "training_cost":
                scope = _train_scope(row)
                if scope is None:
                    exclusions.append(
                        _exclusion_record(
                            row,
                            spec,
                            level=level,
                            scene=scene,
                            comparison_group=scene,
                            reason="missing_training_comparability_metadata",
                            missing_fields=(TRAIN_TIMING_COLUMN,),
                        )
                    )
                    continue
                group = f"{scene}|{scope}"
            elif spec.family == "render_cost":
                context = _render_context(row)
                if context is None:
                    absent = [
                        name
                        for name in (
                            "render_width",
                            "render_height",
                            RENDER_TIMING_COLUMN,
                        )
                        if row.get(name) in (None, "")
                    ]
                    exclusions.append(
                        _exclusion_record(
                            row,
                            spec,
                            level=level,
                            scene=scene,
                            comparison_group=scene,
                            reason="missing_render_comparability_metadata",
                            missing_fields=absent,
                        )
                    )
                    continue
                width, height, scope = context
                group = f"{scene}|{width}x{height}|{scope}"
            else:
                group = scene
            candidate = dict(row)
            candidate["_row_id"] = f"{scene}|{row['method_id']}|{spec.pareto_id}"
            groups.setdefault(group, []).append(candidate)
        for group, candidates in sorted(groups.items()):
            members = pareto_member_ids(candidates, spec)
            for candidate in sorted(candidates, key=lambda item: item["method_id"]):
                memberships.append(
                    _membership_record(
                        candidate,
                        spec,
                        level=level,
                        scene=scene_label or str(candidate["scene"]),
                        comparison_group=group,
                        is_member=candidate["_row_id"] in members,
                    )
                )
    memberships.sort(key=lambda row: (row["level"], row["pareto_id"], row["comparison_group"], row["method_id"]))
    exclusions.sort(key=lambda row: (row["level"], row["pareto_id"], row["scene"], row["method_id"]))
    return memberships, exclusions


def resolve_macro_scenes(rows: Sequence[Mapping[str, Any]], requested: Sequence[str] | None = None) -> tuple[str, str] | tuple[()]:
    if requested:
        values = tuple(str(value).strip() for value in requested)
        if len(values) != 2 or not all(values) or values[0] == values[1]:
            raise ValueError("macro scenes must be two distinct non-empty labels")
        return values  # type: ignore[return-value]
    scenes = tuple(sorted({str(row["scene"]) for row in rows}))
    if len(scenes) == 2:
        return scenes  # type: ignore[return-value]
    return ()


def build_macro_rows(
    rows: Sequence[dict[str, Any]],
    macro_scenes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build unweighted two-scene macro rows only for complete method coverage."""

    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method_id"], {})[row["scene"]] = row
    macros: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    if len(macro_scenes) != 2:
        for method, scene_rows in sorted(by_method.items()):
            exemplar = next(iter(scene_rows.values()))
            for spec in PARETO_SPECS:
                exclusions.append(
                    _exclusion_record(
                        exemplar,
                        spec,
                        level="macro",
                        scene="MACRO",
                        comparison_group="MACRO",
                        reason="macro_requires_exactly_two_scenes",
                        details="global scene universe is not exactly two; pass --macro-scenes to pin two scenes",
                    )
                )
        return [], exclusions
    required = tuple(macro_scenes)
    for method, scene_rows in sorted(by_method.items()):
        exemplar = next(iter(scene_rows.values()))
        absent = [scene for scene in required if scene not in scene_rows]
        if absent:
            for spec in PARETO_SPECS:
                exclusions.append(
                    _exclusion_record(
                        exemplar,
                        spec,
                        level="macro",
                        scene="MACRO",
                        comparison_group="MACRO",
                        reason="incomplete_two_scene_coverage",
                        missing_fields=absent,
                        details=f"required scenes: {';'.join(required)}",
                    )
                )
            continue
        selected = [scene_rows[scene] for scene in required]
        macro: dict[str, Any] = {
            "scene": "MACRO",
            "method_id": method,
            "display_name": exemplar["display_name"],
            "evidence_status": "macro_metadata_only",
            "coverage_scenes": ";".join(required),
            "scene_count": 2,
            "structural_metadata_json": json.dumps(
                {scene: scene_rows[scene].get(STRUCTURAL_METADATA_COLUMN, "") for scene in required},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "evidence_status_by_scene_json": json.dumps(
                {scene: scene_rows[scene]["evidence_status"] for scene in required},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for column in ENDPOINT_METADATA_COLUMNS:
            values = [row.get(column, "") for row in selected]
            macro[column] = values[0] if values[0] == values[1] else ""
        for column in (
            HOTSPOT_DOMAIN_COLUMN,
            HOTSPOT_THRESHOLD_RULE_COLUMN,
            HOTSPOT_THRESHOLD_BINS_COLUMN,
        ):
            values = [row.get(column) for row in selected]
            macro[column] = values[0] if values[0] == values[1] else None
        for column in (*FLOAT_COLUMNS, "gaussian_count"):
            values = [row.get(column) for row in selected]
            macro[column] = (
                sum(float(value) for value in values) / 2.0
                if all(value is not None for value in values)
                and all(not _scene_metric_provenance_missing(row, column) for row in selected)
                else None
            )
        train_scopes = [_train_scope(row) for row in selected]
        macro["train_contexts_by_scene_json"] = json.dumps(
            {
                scene: ({TRAIN_TIMING_COLUMN: scope} if scope is not None else None)
                for scene, scope in zip(required, train_scopes)
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        train_comparable = (
            train_scopes[0] is not None and train_scopes[0] == train_scopes[1]
        )
        if train_comparable:
            macro[TRAIN_TIMING_COLUMN] = train_scopes[0]
        else:
            # A cross-scope mean is not a meaningful training-cost datum. Keep
            # the mismatch receipt and emit an explicit membership exclusion.
            macro["train_wall_time_s"] = None
            macro[TRAIN_TIMING_COLUMN] = ""
        macro["train_context_consistent"] = train_comparable
        contexts = [_render_context(row) for row in selected]
        macro["render_contexts_by_scene_json"] = json.dumps(
            {
                scene: (
                    {
                        "width": context[0],
                        "height": context[1],
                        RENDER_TIMING_COLUMN: context[2],
                    }
                    if context is not None
                    else None
                )
                for scene, context in zip(required, contexts)
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        render_comparable = contexts[0] is not None and contexts[0] == contexts[1]
        if render_comparable:
            (
                macro["render_width"],
                macro["render_height"],
                macro[RENDER_TIMING_COLUMN],
            ) = contexts[0]
        else:
            macro["render_ms_per_view"] = None
            macro["render_width"] = None
            macro["render_height"] = None
            macro[RENDER_TIMING_COLUMN] = ""
        macro["render_context_consistent"] = render_comparable
        sha_fields = sorted({key for row in selected for key in row if key.endswith("_sha256")})
        macro["source_sha256_by_scene_json"] = json.dumps(
            {
                scene: {field: scene_rows[scene].get(field, "") for field in sha_fields}
                for scene in required
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        macros.append(macro)
    macros.sort(key=lambda row: row["method_id"])
    return macros, exclusions


def build_worst_rows(
    rows: Sequence[dict[str, Any]],
    macro_scenes: Sequence[str],
) -> list[dict[str, Any]]:
    """Return long-form worst-scene evidence without cross-context render ranking."""

    if len(macro_scenes) != 2:
        return []
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(row["method_id"], {})[row["scene"]] = row
    output: list[dict[str, Any]] = []
    metrics = sorted(METRIC_DIRECTIONS)
    for method, scene_rows in sorted(by_method.items()):
        if any(scene not in scene_rows for scene in macro_scenes):
            continue
        selected = [scene_rows[scene] for scene in macro_scenes]
        for metric in metrics:
            values = [(row["scene"], row.get(metric)) for row in selected]
            if any(value is None for _, value in values) or any(
                _scene_metric_provenance_missing(row, metric) for row in selected
            ):
                continue
            if metric == "render_ms_per_view":
                contexts = [_render_context(row) for row in selected]
                if contexts[0] is None or contexts[0] != contexts[1]:
                    continue
                width, height, scope = contexts[0]
                group = f"{width}x{height}|{scope}"
            elif metric == "train_wall_time_s":
                scopes = [_train_scope(row) for row in selected]
                if scopes[0] is None or scopes[0] != scopes[1]:
                    continue
                group = f"training|{scopes[0]}"
            else:
                group = "two_scene"
            direction = METRIC_DIRECTIONS[metric]
            worst_scene, worst_value = (
                max(values, key=lambda pair: float(pair[1]))
                if direction == "min"
                else min(values, key=lambda pair: float(pair[1]))
            )
            output.append(
                {
                    "method_id": method,
                    "display_name": selected[0]["display_name"],
                    "metric": metric,
                    "direction": direction,
                    "worst_scene": worst_scene,
                    "worst_value": worst_value,
                    "comparison_group": group,
                    "coverage_scenes": ";".join(macro_scenes),
                }
            )
    return output


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("attempted to serialize non-finite CSV value")
        return format(value, ".17g")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    _assert_no_design_markers(list(fields), f"{path.name}: output columns")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for index, row in enumerate(rows, start=2):
            projected = {field: _csv_value(row.get(field)) for field in fields}
            _assert_no_design_markers(projected, f"{path.name}:{index}")
            writer.writerow(projected)


def _normalized_fields(rows: Sequence[Mapping[str, Any]], extra_columns: Sequence[str]) -> list[str]:
    dynamic_sha = sorted({key for row in rows for key in row if key.endswith("_sha256")})
    ordered = list(CANONICAL_COLUMNS)
    for name in (*dynamic_sha, *extra_columns, "source_input_name", "source_row_number"):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _macro_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    base = [
        "scene",
        "method_id",
        "display_name",
        "evidence_status",
        "coverage_scenes",
        "scene_count",
        *ENDPOINT_METADATA_COLUMNS,
        *FLOAT_COLUMNS,
        "gaussian_count",
        "render_width",
        "render_height",
        TRAIN_TIMING_COLUMN,
        RENDER_TIMING_COLUMN,
        "train_context_consistent",
        "train_contexts_by_scene_json",
        "render_context_consistent",
        "render_contexts_by_scene_json",
        HOTSPOT_DOMAIN_COLUMN,
        HOTSPOT_THRESHOLD_RULE_COLUMN,
        HOTSPOT_THRESHOLD_BINS_COLUMN,
        "structural_metadata_json",
        "evidence_status_by_scene_json",
        "source_sha256_by_scene_json",
    ]
    return [name for index, name in enumerate(base) if name not in base[:index]]


def _membership_lookup(memberships: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], bool]:
    return {
        (str(row["level"]), str(row["scene"]), str(row["method_id"]), str(row["pareto_id"])): bool(row["is_member"])
        for row in memberships
    }


def scene_visual_encodings(scene_labels: Iterable[str]) -> dict[str, str]:
    """Assign deterministic fill/edge encodings while reserving color for method."""

    labels = sorted({str(label) for label in scene_labels})
    styles = ("filled_method_color", "hollow_method_color_edge")
    return {label: styles[index % len(styles)] for index, label in enumerate(labels)}


def scene_rows_for_figure(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    """Consolidate explicit protocol no-op aliases with a present source row.

    The normalized tables and Pareto memberships retain every declared method.
    Only the scene-level visual layer is consolidated, so multiple aliases of
    one measured endpoint cannot be mistaken for independent measurements.
    Macro rows remain method-specific because their second-scene evidence can
    differ.
    """

    by_key = {
        (str(row.get("scene", "")), str(row.get("method_id", ""))): row
        for row in rows
    }
    equality_fields = (
        *FLOAT_COLUMNS,
        *INTEGER_COLUMNS,
        TRAIN_TIMING_COLUMN,
        RENDER_TIMING_COLUMN,
        HOTSPOT_DOMAIN_COLUMN,
        HOTSPOT_THRESHOLD_RULE_COLUMN,
        *tuple(field for field in SOURCE_SHA_COLUMNS if field != "alias_receipt_sha256"),
    )
    plotted: list[Mapping[str, Any]] = []
    consolidated: list[dict[str, str]] = []
    for row in rows:
        metadata_text = _clean_text(row.get(STRUCTURAL_METADATA_COLUMN))
        try:
            metadata = json.loads(metadata_text) if metadata_text else {}
        except json.JSONDecodeError:
            metadata = {}
        scene = str(row.get("scene", ""))
        source_method = str(metadata.get("alias_source_method_id", ""))
        if (
            metadata.get("alias_kind") == "explicit_noop"
            and source_method
            and (scene, source_method) in by_key
        ):
            source_row = by_key[(scene, source_method)]
            unequal = [
                field
                for field in equality_fields
                if row.get(field) != source_row.get(field)
            ]
            if unequal:
                raise ValueError(
                    f"{scene}/{row.get('method_id')}: explicit no-op figure alias "
                    f"differs from {source_method} in {unequal}"
                )
            consolidated.append(
                {
                    "scene": scene,
                    "alias_method_id": str(row.get("method_id", "")),
                    "source_method_id": source_method,
                }
            )
            continue
        plotted.append(row)
    return plotted, consolidated


def plot_four_panel(
    rows: Sequence[dict[str, Any]],
    macro_rows: Sequence[dict[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    output_dir: Path,
    stem: str = FIGURE_STEM,
) -> list[Path]:
    """Render the fixed Python/matplotlib four-panel publication figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    methods = sorted({row["method_id"] for row in (*rows, *macro_rows)})
    colors = {
        method: METHOD_COLORS[index % len(METHOD_COLORS)]
        for index, method in enumerate(methods)
    }
    displays = {
        row["method_id"]: row["display_name"]
        for row in (*rows, *macro_rows)
    }
    depth_markers = {"expected": "o", "median": "s", "max_contribution": "^"}
    lookup = _membership_lookup(memberships)
    scene_encodings = scene_visual_encodings(row["scene"] for row in rows)
    scene_rows, consolidated_noop_aliases = scene_rows_for_figure(rows)

    fig = plt.figure(figsize=(7.2, 5.4), facecolor="white")
    outer = fig.add_gridspec(2, 2, width_ratios=(1.12, 1.0), hspace=0.38, wspace=0.30)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    cost_grid = outer[1, 1].subgridspec(2, 1, hspace=0.48)
    ax_d_train = fig.add_subplot(cost_grid[0, 0])
    ax_d_render = fig.add_subplot(cost_grid[1, 0])

    def scatter_point(axis, x, y, row, marker, member, *, macro=False, alpha=1.0):
        size = 34 if macro else 15
        method_color = colors[row["method_id"]]
        scene_style = scene_encodings.get(str(row.get("scene", "")), "filled_method_color")
        if macro:
            facecolors: Any = [method_color]
            edgecolors: Any = "black"
        elif scene_style == "hollow_method_color_edge":
            facecolors = "none"
            edgecolors = [method_color]
        else:
            facecolors = [method_color]
            edgecolors = "white"
        axis.scatter(
            [x],
            [y],
            s=size,
            marker=marker,
            facecolors=facecolors,
            alpha=alpha,
            edgecolors=edgecolors,
            linewidths=0.7 if macro else 0.35,
            zorder=3 if macro else 2,
        )
        if member:
            axis.scatter(
                [x],
                [y],
                s=size + (25 if macro else 14),
                marker=marker,
                facecolors="none",
                edgecolors="#202020",
                linewidths=0.75,
                zorder=4,
            )

    def plot_depth_panel(axis, family, y_metric, y_label, letter, title):
        plotted = 0
        for level, source, macro in (("scene", scene_rows, False), ("macro", macro_rows, True)):
            for row in source:
                for depth, marker in depth_markers.items():
                    x_metric = f"front_auc_{depth}"
                    if row.get(x_metric) is None or row.get(y_metric) is None:
                        continue
                    pareto_id = f"{family}_{depth}"
                    scene = "MACRO" if macro else row["scene"]
                    member = lookup.get((level, scene, row["method_id"], pareto_id), False)
                    scatter_point(
                        axis,
                        100.0 * float(row[x_metric]),
                        float(row[y_metric]),
                        row,
                        marker,
                        member,
                        macro=macro,
                        alpha=1.0 if macro else 0.38,
                    )
                    plotted += 1
        if not plotted:
            axis.text(0.5, 0.5, "No comparable points", transform=axis.transAxes, ha="center", va="center", color="0.45")
        axis.set_xlabel("Front-curve AUC (%) (lower is better)")
        axis.set_ylabel(y_label)
        axis.set_title(title, loc="left", pad=3)
        axis.text(-0.16, 1.08, letter, transform=axis.transAxes, fontsize=9, fontweight="bold", va="top")
        axis.grid(True, color="#dddddd", linewidth=0.45, zorder=0)

    plot_depth_panel(
        ax_a,
        "temperature_vs_front",
        "temp_mae_display_c",
        "Display-temp MAE (deg C)\n(lower is better)",
        "a",
        "Thermometry-geometry trade-off",
    )
    plot_depth_panel(
        ax_b,
        "appearance_vs_front",
        "thermal_lpips",
        "Thermal LPIPS\n(lower is better)",
        "b",
        "Appearance-geometry trade-off",
    )

    plotted_c = 0
    for level, source, macro in (("scene", scene_rows, False), ("macro", macro_rows, True)):
        for row in source:
            if row.get("temp_mae_display_c") is None or row.get("hotspot_auprc_display") is None:
                continue
            scene = "MACRO" if macro else row["scene"]
            member = lookup.get((level, scene, row["method_id"], "hotspot_vs_temperature"), False)
            scatter_point(
                ax_c,
                float(row["temp_mae_display_c"]),
                float(row["hotspot_auprc_display"]),
                row,
                "D" if macro else "o",
                member,
                macro=macro,
                alpha=1.0 if macro else 0.38,
            )
            plotted_c += 1
    if not plotted_c:
        ax_c.text(0.5, 0.5, "No comparable points", transform=ax_c.transAxes, ha="center", va="center", color="0.45")
    ax_c.set_xlabel("Display-comparable temperature MAE (deg C) (lower is better)")
    ax_c.set_ylabel("Hotspot AUPRC (higher is better)")
    ax_c.set_title("Hotspot ranking-thermometry", loc="left", pad=3)
    ax_c.text(-0.16, 1.08, "c", transform=ax_c.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_c.grid(True, color="#dddddd", linewidth=0.45, zorder=0)

    group_markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
    train_scopes = sorted(
        {
            scope
            for row in (*rows, *macro_rows)
            if row.get("gaussian_count") is not None
            and row.get("train_wall_time_s") is not None
            and (scope := _train_scope(row)) is not None
        }
    )
    train_handles = []
    plotted_train = 0
    for scope_index, scope in enumerate(train_scopes):
        marker = group_markers[scope_index % len(group_markers)]
        train_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                color="0.3",
                markersize=4,
                label=_ascii_figure_label(
                    {
                        "oct_end_to_end_training_including_pretraining_setup": "OCT end-to-end (incl. setup)",
                        "training_call_boundary_including_model_setup_eval_and_save": "Train call (setup/eval/save)",
                    }.get(scope, scope.replace("_", " "))
                ),
            )
        )
        for level, source, macro in (("scene", scene_rows, False), ("macro", macro_rows, True)):
            for row in source:
                if (
                    _train_scope(row) != scope
                    or row.get("gaussian_count") is None
                    or row.get("train_wall_time_s") is None
                ):
                    continue
                scene = "MACRO" if macro else row["scene"]
                member = lookup.get(
                    (level, scene, row["method_id"], "training_cost"), False
                )
                scatter_point(
                    ax_d_train,
                    float(row["gaussian_count"]) / 1e6,
                    float(row["train_wall_time_s"]) / 3600.0,
                    row,
                    marker,
                    member,
                    macro=macro,
                    alpha=1.0 if macro else 0.38,
                )
                plotted_train += 1
    if not plotted_train:
        ax_d_train.text(0.5, 0.5, "No training-cost points", transform=ax_d_train.transAxes, ha="center", va="center", color="0.45")
    ax_d_train.set_ylabel("Train wall time (h)")
    ax_d_train.set_title("Measured cost (no weighted score)", loc="left", pad=2)
    ax_d_train.text(-0.17, 1.17, "d", transform=ax_d_train.transAxes, fontsize=9, fontweight="bold", va="top")
    ax_d_train.grid(True, color="#dddddd", linewidth=0.45, zorder=0)
    if train_handles:
        ax_d_train.legend(
            handles=train_handles,
            loc="center right",
            fontsize=4.8,
            title="Training scopes",
            title_fontsize=5.2,
        )

    render_groups = sorted(
        {
            _render_context(row)
            for row in (*rows, *macro_rows)
            if row.get("gaussian_count") is not None
            and row.get("render_ms_per_view") is not None
            and _render_context(row) is not None
        }
    )
    render_handles = []
    plotted_render = 0
    for group_index, context in enumerate(render_groups):
        marker = group_markers[group_index % len(group_markers)]
        width, height, scope = context
        render_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="none",
                color="0.3",
                markersize=4,
                label=_ascii_figure_label(
                    f"{width}x{height}, "
                    + {
                        "render_only_no_gt_no_cpu_transfer_no_io": "render-only (no GT/CPU/I/O)",
                        "pure_render_synchronized_cuda_event": "synced CUDA-event render",
                    }.get(scope, scope.replace("_", " "))
                ),
            )
        )
        for level, source, macro in (("scene", scene_rows, False), ("macro", macro_rows, True)):
            for row in source:
                if _render_context(row) != context or row.get("gaussian_count") is None or row.get("render_ms_per_view") is None:
                    continue
                scene = "MACRO" if macro else row["scene"]
                member = lookup.get((level, scene, row["method_id"], "render_cost"), False)
                scatter_point(
                    ax_d_render,
                    float(row["gaussian_count"]) / 1e6,
                    float(row["render_ms_per_view"]),
                    row,
                    marker,
                    member,
                    macro=macro,
                    alpha=1.0 if macro else 0.38,
                )
                plotted_render += 1
    if not plotted_render:
        ax_d_render.text(0.5, 0.5, "No comparable render points", transform=ax_d_render.transAxes, ha="center", va="center", color="0.45")
    ax_d_render.set_xlabel("Gaussian count (million) (lower is better)")
    ax_d_render.set_ylabel("Render (ms/view)")
    ax_d_render.grid(True, color="#dddddd", linewidth=0.45, zorder=0)
    if render_handles:
        ax_d_render.legend(handles=render_handles, loc="center", fontsize=4.7, title="Render strata", title_fontsize=5.1)

    method_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=colors[method], markeredgecolor="none", markersize=4.5, label=_ascii_figure_label(displays[method]))
        for method in methods
    ]
    depth_handles = [
        Line2D([0], [0], marker=marker, linestyle="none", color="0.35", markersize=4.5, label=depth.replace("_", " "))
        for depth, marker in depth_markers.items()
    ]
    encoding_handles = []
    for scene, style in scene_encodings.items():
        encoding_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor="none" if style == "hollow_method_color_edge" else "0.55",
                markeredgecolor="0.55" if style == "hollow_method_color_edge" else "white",
                alpha=0.55,
                markersize=4.5,
                label=_ascii_figure_label(scene),
            )
        )
    encoding_handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="0.55", markeredgecolor="black", markersize=5.5, label="two-scene macro (black edge)"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none", markeredgecolor="#202020", markersize=6, label="Pareto member"),
        ]
    )
    fig.legend(
        handles=method_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=min(6, max(1, len(method_handles))),
        fontsize=5.2,
        title="Method (structure is metadata only)",
        title_fontsize=5.6,
    )
    fig.legend(
        handles=depth_handles + encoding_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.072),
        ncol=len(depth_handles + encoding_handles),
        fontsize=4.8,
    )
    if consolidated_noop_aliases:
        fig.text(
            0.5,
            0.145,
            NOOP_ALIAS_FIGURE_NOTE,
            ha="center",
            va="center",
            fontsize=5.2,
            color="0.3",
        )
    fig.subplots_adjust(left=0.10, right=0.96, top=0.95, bottom=0.24)

    paths = [
        output_dir / f"{stem}.svg",
        output_dir / f"{stem}.pdf",
        output_dir / f"{stem}.tiff",
        output_dir / f"{stem}.png",
    ]
    fig.savefig(paths[0], format="svg", facecolor="white")
    fig.savefig(paths[1], format="pdf", facecolor="white")
    # Render the 600-dpi journal master into memory first. This avoids Windows
    # file-locking behavior when an RGBA TIFF is reopened and replaced.
    from io import BytesIO

    tiff_raster = BytesIO()
    fig.savefig(tiff_raster, format="png", dpi=600, facecolor="white")
    fig.savefig(paths[3], format="png", dpi=300, facecolor="white")
    plt.close(fig)
    # Save an uncompressed RGB TIFF while preserving the exact 600-dpi canvas.
    from PIL import Image

    tiff_raster.seek(0)
    with Image.open(tiff_raster) as tiff_image:
        rgb_tiff = tiff_image.convert("RGB")
    rgb_tiff.save(paths[2], format="TIFF", dpi=(600, 600), compression="raw")
    tiff_raster.close()
    return paths


def export_reassessment(
    input_paths: Sequence[Path],
    output_dir: Path,
    *,
    macro_scenes: Sequence[str] | None = None,
    figure_stem: str = FIGURE_STEM,
) -> dict[str, Any]:
    loaded = load_normalized_csvs(input_paths)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    scene_membership, scene_exclusions = compute_memberships(loaded.rows, level="scene")
    pinned_macro_scenes = resolve_macro_scenes(loaded.rows, macro_scenes)
    macro_rows, coverage_exclusions = build_macro_rows(loaded.rows, pinned_macro_scenes)
    macro_membership, macro_axis_exclusions = compute_memberships(
        macro_rows,
        level="macro",
        scene_label="MACRO",
    )
    memberships = scene_membership + macro_membership
    exclusions = scene_exclusions + coverage_exclusions + macro_axis_exclusions
    memberships.sort(key=lambda row: (row["level"], row["pareto_id"], row["comparison_group"], row["method_id"]))
    exclusions.sort(key=lambda row: (row["level"], row["pareto_id"], row["scene"], row["method_id"]))
    worst_rows = build_worst_rows(loaded.rows, pinned_macro_scenes)

    normalized_path = output / "normalized.csv"
    membership_path = output / "membership.csv"
    macro_path = output / "macro.csv"
    worst_path = output / "worst.csv"
    exclusions_path = output / "exclusions.csv"
    write_csv(normalized_path, loaded.rows, _normalized_fields(loaded.rows, loaded.extra_columns))
    write_csv(membership_path, memberships, MEMBERSHIP_FIELDS)
    write_csv(macro_path, macro_rows, _macro_fields(macro_rows))
    write_csv(
        worst_path,
        worst_rows,
        ("method_id", "display_name", "metric", "direction", "worst_scene", "worst_value", "comparison_group", "coverage_scenes"),
    )
    write_csv(exclusions_path, exclusions, EXCLUSION_FIELDS)
    figure_paths = plot_four_panel(loaded.rows, macro_rows, memberships, output, figure_stem)

    artifact_paths = [normalized_path, membership_path, macro_path, worst_path, exclusions_path, *figure_paths]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "size_bytes": Path(__file__).resolve().stat().st_size,
        },
        "inputs": loaded.inputs,
        "outputs": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifact_paths
        },
        "row_counts": {
            "normalized": len(loaded.rows),
            "membership": len(memberships),
            "macro": len(macro_rows),
            "worst": len(worst_rows),
            "exclusions": len(exclusions),
        },
        "metric_directions": METRIC_DIRECTIONS,
        "pareto_definitions": [
            {
                "pareto_id": spec.pareto_id,
                "depth_definition": spec.depth_definition or None,
                "x_metric": spec.x_metric,
                "x_direction": spec.x_direction,
                "y_metric": spec.y_metric,
                "y_direction": spec.y_direction,
                "family": spec.family,
            }
            for spec in PARETO_SPECS
        ],
        "policy": {
            "unconstrained": True,
            "evidence_status_role": "metadata_only",
            "structural_metadata_role": "metadata_only",
            "missing_values": "blank/null only; never zero or NaN imputation; per-Pareto exclusion recorded",
            "axis_provenance": "every populated axis requires its metric-specific source SHA; depth also requires split and reference SHAs",
            "scene_membership": "computed independently within each scene",
            "macro": "unweighted mean per metric; emitted only when both pinned scenes have that metric and its source evidence",
            "macro_scenes": list(pinned_macro_scenes),
            "training_membership": "grouped by scene and train_timing_scope; no cross-scope dominance; macro membership requires the same scope in both scenes",
            "render_membership": "grouped by scene, render_width, render_height, and render_timing_scope; no cross-stratum dominance",
            "test_depth_role": "report-only; exporter performs no selection or threshold tuning",
        },
        "hotspot_contract": {
            "metric": "hotspot_auprc_display",
            "domain": HOTSPOT_DISPLAY_DOMAIN,
            "threshold_rule": HOTSPOT_THRESHOLD_RULE,
            "histogram_bins": HOTSPOT_THRESHOLD_BINS,
            "threshold_hash_column": HOTSPOT_THRESHOLD_SHA_COLUMN,
            "source_hash_column": "hotspot_source_sha256",
        },
        "figure_contract": {
            "backend": "Python/matplotlib only",
            "size_inches": [7.2, 5.4],
            "method_palette": "Okabe-Ito colorblind-safe categorical palette",
            "protocol_noop_alias_rendering": {
                "scene_level": NOOP_ALIAS_FIGURE_NOTE,
                "macro_level": "method-specific because second-scene evidence can differ",
            },
            "visual_encodings": {
                "method": "color",
                "depth_definition": "marker shape",
                "scene": scene_visual_encodings(row["scene"] for row in loaded.rows),
                "macro": "larger marker with black edge",
                "training_scope": "marker shape within the training-cost subplot",
                "render_resolution_and_scope": "marker shape within the render-cost subplot",
            },
            "panels": {
                "a": "display-comparable temperature MAE vs three front-AUC definitions",
                "b": "thermal LPIPS vs three front-AUC definitions",
                "c": "hotspot AUPRC vs display-comparable temperature MAE",
                "d": "training and render cost; timing and resolution strata are not cross-ranked",
            },
            "exports": {"svg_text": "editable", "pdf_fonttype": 42, "tiff_dpi": 600, "png_dpi": 300},
        },
        "self_hash": {
            "algorithm": "sha256",
            "canonicalization": "UTF-8 JSON, sorted keys, compact separators, manifest_sha256 omitted",
        },
    }
    _assert_no_design_markers(manifest, "manifest.json")
    manifest["manifest_sha256"] = manifest_self_hash(manifest)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Normalized endpoint-summary CSV(s)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--macro-scenes", nargs=2, metavar=("SCENE_A", "SCENE_B"))
    parser.add_argument("--figure-stem", default=FIGURE_STEM)
    args = parser.parse_args(argv)
    try:
        export_reassessment(
            [Path(value) for value in args.input],
            Path(args.output_dir),
            macro_scenes=args.macro_scenes,
            figure_stem=args.figure_stem,
        )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"Reassessment Pareto export failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
