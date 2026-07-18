#!/usr/bin/env python3
"""Compile source-bound inputs for the unconstrained AAAI reassessment Pareto.

The compiler is intentionally manifest-driven: it never discovers methods from
directory names and never invents a missing endpoint.  Every file that it reads
must be declared as ``{"path": ..., "sha256": ...}`` in a self-hashed source
manifest.  The emitted ``endpoint_summary.csv`` is accepted by
``export_reassessment_pareto.py``; the other tables retain the test-only depth
curves, stratified efficiency evidence, and explicitly selected artifacts.

Source manifest (v1), abbreviated::

    {
      "schema": "uav-tgs-reassessment-pareto-source-manifest-v1",
      "scenes": [{
        "scene": "Building",
        "formal_split_sha256": "...",
        "reference_manifest_sha256": "...",
        "hotspot_threshold_sha256": "..."
      }],
      "endpoints": [{
        "scene": "Building", "method_id": "raw_f3",
        "display_name": "Raw F3",
        "geometry_endpoint_id": "...", "rgb_endpoint_id": "...",
        "thermal_endpoint_id": "...", "depth_source_id": "...",
        "appearance": [{
          "kind": "r1_aggregate", "modality": "thermal",
          "source": {"path": "appearance.json", "sha256": "..."},
          "selector": {"scene": "Building", "method": "Raw-F3"}
        }],
        "thermal": {
          "kind": "baseline_hotspot",
          "source": {"path": "hotspot.json", "sha256": "..."},
          "selector": {"scene": "Building", "method": "Raw-F3"}
        },
        "depth": {
          "kind": "phase_c_depth",
          "source": {"path": "metrics_summary.json", "sha256": "..."}
        },
        "efficiency": [{
          "kind": "uav_tgs_efficiency",
          "source": {"path": "train.json", "sha256": "..."}
        }]
      }],
      "manifest_sha256": "canonical JSON SHA after removing this field"
    }

Appearance kind ``results`` additionally requires ``endpoint`` and an explicit
``resolution``.  An OCT endpoint uses kind ``oct_evaluation_v2`` and an
explicit full-resolution width/height.  Exact geometry aliases certify depth
inheritance but do not copy metrics.  A whole endpoint is copied only when an
explicit ``noop`` alias entry names a source method and a verified receipt;
there is no implicit Building SCSP rule.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SOURCE_SCHEMA = "uav-tgs-reassessment-pareto-source-manifest-v1"
COMPILED_SOURCE_SCHEMA = "uav-tgs-reassessment-verified-source-manifest-v1"
OUTPUT_SCHEMA = "uav-tgs-reassessment-pareto-input-bundle-v1"
ENDPOINT_SCHEMA = "endpoint-summary-v1"

HOTSPOT_DOMAIN = "display_temperature_c"
HOTSPOT_RULE = "frozen_train_q95"
HOTSPOT_BINS = 65536
DEPTH_DEFINITIONS = ("expected", "median", "max_contribution")
DEPTH_THRESHOLDS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")

NOOP_ENDPOINT_COVERAGE_SCHEMA = "uav-tgs-protocol-noop-endpoint-coverage-v1"
NOOP_EVIDENCE_STATUS = "formal_protocol_noop_alias"
NOOP_ENDPOINT_EVIDENCE_STATUS = "formal_explicit_noop_alias"
NOOP_COPY_ROLES = "all"
NOOP_COPIED_METRIC_ROLES = (
    "rgb_appearance",
    "thermal_appearance",
    "temperature",
    "hotspot",
    "depth",
    "efficiency",
    "artifacts",
)
NOOP_CLAIM_BOUNDARY = (
    "protocol-level reuse of the declared source endpoint's complete compiled evidence "
    "for the declared alias endpoint only; no independent training or performance claim"
)

ENDPOINT_FIELDS = (
    "schema_version",
    "scene",
    "method_id",
    "display_name",
    "evidence_status",
    "geometry_endpoint_id",
    "rgb_endpoint_id",
    "thermal_endpoint_id",
    "depth_source_id",
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
    "gaussian_count",
    "train_wall_time_s",
    "render_ms_per_view",
    "render_width",
    "render_height",
    "train_timing_scope",
    "render_timing_scope",
    "timing_scope",
    "hotspot_metric_domain",
    "hotspot_threshold_rule",
    "hotspot_threshold_histogram_bins",
    "structural_metadata_json",
    "formal_split_sha256",
    "reference_manifest_sha256",
    "appearance_source_sha256",
    "rgb_appearance_source_sha256",
    "thermal_appearance_source_sha256",
    "temperature_source_sha256",
    "hotspot_source_sha256",
    "hotspot_threshold_sha256",
    "depth_source_sha256",
    "efficiency_source_sha256",
    "alias_receipt_sha256",
    "rgb_appearance_resolution_label",
    "rgb_appearance_width",
    "rgb_appearance_height",
    "thermal_appearance_resolution_label",
    "thermal_appearance_width",
    "thermal_appearance_height",
    "appearance_resolution_label",
    "appearance_width",
    "appearance_height",
    "temperature_metric_semantics_direct",
    "temperature_metric_semantics_display",
    "off_lut_mean_rgb",
    "off_lut_p95_rgb",
)

DEPTH_FIELDS = (
    "scene",
    "method_id",
    "depth_definition",
    "group_type",
    "group_label",
    "threshold_m",
    "front_rate",
    "agreement_rate",
    "missing_rate",
    "mean_abs_error_m",
    "median_abs_error_m",
    "signed_bias_m",
    "front_auc_normalized",
    "agreement_auc_normalized",
    "formal_split_sha256",
    "reference_manifest_sha256",
    "metrics_source_sha256",
    "alias_receipt_sha256",
)

EFFICIENCY_FIELDS = (
    "scene",
    "method_id",
    "kind",
    "stage",
    "iterations_or_views",
    "width",
    "height",
    "resolution_label",
    "wall_time_s",
    "end_to_end_wall_time_s",
    "ms_per_step",
    "ms_per_view",
    "fps",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "gaussian_count",
    "raster_passes",
    "device_name",
    "timing_scope",
    "comparison_group",
    "source_sha256",
    "alias_receipt_sha256",
)

ARTIFACT_FIELDS = (
    "scene",
    "method_id",
    "artifact_role",
    "split",
    "block_id",
    "view_id",
    "depth_definition",
    "path",
    "sha256",
    "source_manifest_sha256",
    "alias_receipt_sha256",
)

IDENTITY_FIELDS = (
    "scene",
    "method_id",
    "display_name",
    "geometry_endpoint_id",
    "rgb_endpoint_id",
    "thermal_endpoint_id",
    "depth_source_id",
)


class ContractError(ValueError):
    """Raised when a source or manifest violates the compiler contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ContractError(f"value is not canonical finite JSON: {error}") from error
    return hashlib.sha256(encoded).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    material = dict(value)
    material.pop(field, None)
    return canonical_json_sha256(material)


def _require_self_hash(value: Mapping[str, Any], field: str, context: str) -> str:
    supplied = _sha(value.get(field), f"{context}.{field}")
    expected = self_hash(value, field)
    if supplied != expected:
        raise ContractError(f"{context}: {field} mismatch; expected {expected}, got {supplied}")
    return supplied


def _sha(value: Any, context: str) -> str:
    text = str(value or "").strip().lower()
    if not SHA_RE.fullmatch(text):
        raise ContractError(f"{context}: expected lowercase SHA-256")
    return text


def _text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{context}: expected string")
    result = value.strip()
    if not result and not allow_empty:
        raise ContractError(f"{context}: must not be empty")
    return result


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context}: expected object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{context}: expected array")
    return value


def _finite(
    value: Any,
    context: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{context}: expected finite number, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{context}: expected finite number") from error
    if not math.isfinite(result):
        raise ContractError(f"{context}: expected finite number")
    if minimum is not None and result < minimum:
        raise ContractError(f"{context}: expected value >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{context}: expected value <= {maximum}")
    return result


def _positive_int(value: Any, context: str) -> int:
    number = _finite(value, context, minimum=1)
    if not number.is_integer():
        raise ContractError(f"{context}: expected positive integer")
    return int(number)


def _optional_number(value: Any, context: str) -> float | None:
    if value is None or value == "":
        return None
    return _finite(value, context)


def _nested(value: Mapping[str, Any], path: str, context: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ContractError(f"{context}: missing {path}")
        current = current[key]
    return current


def _assert_no_design_gate(value: Any, context: str) -> None:
    markers = ("feasible", "infeasible", "feasibility")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if any(marker in str(key).casefold() for marker in markers):
                raise ContractError(f"{context}: forbidden design-gate key {key!r}")
            _assert_no_design_gate(nested, context)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_design_gate(nested, context)
    elif isinstance(value, str) and any(marker in value.casefold() for marker in markers):
        raise ContractError(f"{context}: forbidden design-gate marker in value")


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"{context}: cannot read JSON {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ContractError(f"{context}: JSON root must be an object")
    return payload


def _canonical_source_digest(values: Mapping[str, str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values.values()))
    return canonical_json_sha256(dict(sorted(values.items())))


def _set_metric(row: MutableMapping[str, Any], name: str, value: Any, context: str) -> None:
    parsed = _finite(value, f"{context}.{name}")
    prior = row.get(name)
    if prior is not None and not math.isclose(float(prior), parsed, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(f"{context}: conflicting values for {name}: {prior} vs {parsed}")
    row[name] = parsed


def _resolution(spec: Mapping[str, Any], context: str) -> tuple[str, int, int]:
    item = _mapping(spec.get("resolution"), f"{context}.resolution")
    label = _text(item.get("label"), f"{context}.resolution.label")
    width = _positive_int(item.get("width"), f"{context}.resolution.width")
    height = _positive_int(item.get("height"), f"{context}.resolution.height")
    return label, width, height


@dataclass(frozen=True)
class SceneProtocol:
    scene: str
    formal_split_sha256: str
    reference_manifest_sha256: str
    hotspot_threshold_sha256: str


@dataclass
class CompiledEndpoint:
    row: dict[str, Any]
    depth_rows: list[dict[str, Any]]
    efficiency_rows: list[dict[str, Any]]
    artifact_rows: list[dict[str, Any]]


class SourceRegistry:
    def __init__(self, manifest_dir: Path) -> None:
        self.manifest_dir = manifest_dir
        self._records: dict[Path, dict[str, Any]] = {}

    def verify(self, identity: Any, role: str) -> tuple[Path, str]:
        item = _mapping(identity, f"{role}.source")
        raw = _text(item.get("path"), f"{role}.source.path")
        path = Path(raw)
        if not path.is_absolute():
            path = self.manifest_dir / path
        path = path.resolve()
        expected = _sha(item.get("sha256"), f"{role}.source.sha256")
        if not path.is_file():
            raise ContractError(f"{role}: declared input does not exist: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ContractError(
                f"{role}: SHA-256 mismatch for {path}; expected {expected}, got {actual}"
            )
        size = path.stat().st_size
        if item.get("size_bytes") is not None and _positive_int(
            item["size_bytes"], f"{role}.source.size_bytes"
        ) != size:
            raise ContractError(f"{role}: size_bytes mismatch for {path}")
        prior = self._records.get(path)
        if prior is not None and prior["sha256"] != actual:
            raise ContractError(f"{role}: conflicting identities for {path}")
        record = self._records.setdefault(
            path,
            {"path": str(path), "sha256": actual, "size_bytes": size, "roles": []},
        )
        if role not in record["roles"]:
            record["roles"].append(role)
        return path, actual

    def records(self) -> list[dict[str, Any]]:
        rows = [copy.deepcopy(item) for item in self._records.values()]
        for row in rows:
            row["roles"].sort()
        return sorted(rows, key=lambda row: row["path"])


class ParetoInputCompiler:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path.resolve()
        self.payload = _load_json(self.manifest_path, "source manifest")
        if self.payload.get("schema") != SOURCE_SCHEMA:
            raise ContractError(f"source manifest: schema must be {SOURCE_SCHEMA}")
        self.source_manifest_sha256 = _require_self_hash(
            self.payload, "manifest_sha256", "source manifest"
        )
        _assert_no_design_gate(self.payload, "source manifest")
        self.registry = SourceRegistry(self.manifest_path.parent)
        self.protocols = self._load_protocols()

    def _load_protocols(self) -> dict[str, SceneProtocol]:
        result: dict[str, SceneProtocol] = {}
        for index, raw in enumerate(_sequence(self.payload.get("scenes"), "source manifest.scenes")):
            context = f"source manifest.scenes[{index}]"
            item = _mapping(raw, context)
            scene = _text(item.get("scene"), f"{context}.scene")
            if scene in result:
                raise ContractError(f"{context}: duplicate scene {scene}")
            result[scene] = SceneProtocol(
                scene=scene,
                formal_split_sha256=_sha(
                    item.get("formal_split_sha256"), f"{context}.formal_split_sha256"
                ),
                reference_manifest_sha256=_sha(
                    item.get("reference_manifest_sha256"),
                    f"{context}.reference_manifest_sha256",
                ),
                hotspot_threshold_sha256=_sha(
                    item.get("hotspot_threshold_sha256"),
                    f"{context}.hotspot_threshold_sha256",
                ),
            )
        if not result:
            raise ContractError("source manifest.scenes must not be empty")
        return result

    def _protocol(self, scene: str) -> SceneProtocol:
        try:
            return self.protocols[scene]
        except KeyError as error:
            raise ContractError(f"endpoint references undeclared scene {scene!r}") from error

    def _check_protocol(
        self,
        scene: str,
        context: str,
        *,
        formal_split: Any | None = None,
        reference_manifest: Any | None = None,
        hotspot_threshold: Any | None = None,
    ) -> None:
        protocol = self._protocol(scene)
        checks = (
            (formal_split, protocol.formal_split_sha256, "formal split"),
            (reference_manifest, protocol.reference_manifest_sha256, "reference manifest"),
            (hotspot_threshold, protocol.hotspot_threshold_sha256, "hotspot threshold"),
        )
        for value, expected, label in checks:
            if value is None:
                continue
            actual = _sha(value, f"{context}.{label.replace(' ', '_')}_sha256")
            if actual != expected:
                raise ContractError(
                    f"{context}: {label} differs from scene protocol: {actual} != {expected}"
                )

    def compile(self) -> list[CompiledEndpoint]:
        raw_endpoints = _sequence(self.payload.get("endpoints"), "source manifest.endpoints")
        if not raw_endpoints:
            raise ContractError("source manifest.endpoints must not be empty")
        identities: set[tuple[str, str]] = set()
        direct: list[Mapping[str, Any]] = []
        noop: list[Mapping[str, Any]] = []
        for index, raw in enumerate(raw_endpoints):
            item = _mapping(raw, f"source manifest.endpoints[{index}]")
            scene = _text(item.get("scene"), f"endpoints[{index}].scene")
            method = _text(item.get("method_id"), f"endpoints[{index}].method_id")
            key = (scene, method)
            if key in identities:
                raise ContractError(f"duplicate endpoint {scene}/{method}")
            identities.add(key)
            alias = item.get("alias")
            if isinstance(alias, Mapping) and alias.get("kind") == "noop":
                noop.append(item)
            else:
                direct.append(item)

        compiled: dict[tuple[str, str], CompiledEndpoint] = {}
        for item in direct:
            endpoint = self._compile_direct(item)
            compiled[(endpoint.row["scene"], endpoint.row["method_id"])] = endpoint

        pending = list(noop)
        while pending:
            progress = False
            for item in list(pending):
                scene = str(item["scene"])
                alias = _mapping(item["alias"], f"{scene}/{item['method_id']}.alias")
                source_key = (
                    scene,
                    _text(alias.get("source_method_id"), f"{scene}/{item['method_id']}.alias.source_method_id"),
                )
                if source_key not in compiled:
                    continue
                endpoint = self._compile_noop(item, compiled[source_key])
                compiled[(scene, endpoint.row["method_id"])] = endpoint
                pending.remove(item)
                progress = True
            if not progress:
                labels = [f"{item.get('scene')}/{item.get('method_id')}" for item in pending]
                raise ContractError(f"unresolved or cyclic no-op aliases: {labels}")

        displays: dict[str, set[str]] = {}
        for endpoint in compiled.values():
            displays.setdefault(endpoint.row["method_id"], set()).add(endpoint.row["display_name"])
        inconsistent = {key: values for key, values in displays.items() if len(values) != 1}
        if inconsistent:
            raise ContractError(f"display_name differs across scenes: {inconsistent}")
        return [compiled[key] for key in sorted(compiled)]

    def _base_row(self, item: Mapping[str, Any], context: str) -> dict[str, Any]:
        values = {field: None for field in ENDPOINT_FIELDS}
        values["schema_version"] = ENDPOINT_SCHEMA
        for field in IDENTITY_FIELDS:
            values[field] = _text(item.get(field), f"{context}.{field}")
        values["evidence_status"] = _text(
            item.get("evidence_status", "formal"), f"{context}.evidence_status"
        )
        protocol = self._protocol(values["scene"])
        values["formal_split_sha256"] = protocol.formal_split_sha256
        values["reference_manifest_sha256"] = protocol.reference_manifest_sha256
        structural = item.get("structural_metadata", {})
        structural = _mapping(structural, f"{context}.structural_metadata")
        _assert_no_design_gate(structural, f"{context}.structural_metadata")
        values["structural_metadata_json"] = json.dumps(
            structural, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return values

    def _compile_direct(self, item: Mapping[str, Any]) -> CompiledEndpoint:
        context = f"{item.get('scene')}/{item.get('method_id')}"
        row = self._base_row(item, context)
        scene = row["scene"]
        appearance_sources: dict[str, str] = {}
        oct_source_sha: str | None = None
        oct_variant: str | None = None
        depth_rows: list[dict[str, Any]] = []
        efficiency_rows: list[dict[str, Any]] = []
        artifact_rows: list[dict[str, Any]] = []

        for index, raw in enumerate(item.get("appearance", [])):
            spec = _mapping(raw, f"{context}.appearance[{index}]")
            modality, source_sha = self._extract_appearance(scene, row, spec, context)
            if modality in appearance_sources:
                raise ContractError(f"{context}: duplicate {modality} appearance component")
            appearance_sources[modality] = source_sha

        thermal = item.get("thermal")
        if thermal is not None:
            source_sha = self._extract_baseline_hotspot(
                scene, row, _mapping(thermal, f"{context}.thermal"), context
            )
            row["temperature_source_sha256"] = source_sha
            row["hotspot_source_sha256"] = source_sha

        oct_spec = item.get("oct")
        if oct_spec is not None:
            if thermal is not None:
                raise ContractError(f"{context}: thermal and oct sources are mutually exclusive")
            parsed_oct_spec = _mapping(oct_spec, f"{context}.oct")
            oct_variant = _text(parsed_oct_spec.get("variant"), f"{context}.oct.variant")
            source_sha, oct_efficiency = self._extract_oct(scene, row, parsed_oct_spec, context)
            oct_source_sha = source_sha
            if "thermal" in appearance_sources:
                raise ContractError(f"{context}: OCT and a thermal appearance source conflict")
            appearance_sources["thermal"] = source_sha
            row["temperature_source_sha256"] = source_sha
            row["hotspot_source_sha256"] = source_sha
            efficiency_rows.extend(oct_efficiency)

        alias_sha = ""
        alias = item.get("alias")
        depth_source_sha: str | None = None
        depth = item.get("depth")
        if depth is not None:
            source_sha, depth_rows = self._extract_depth(
                scene,
                row["method_id"],
                _mapping(depth, f"{context}.depth"),
                context,
                "",
            )
            depth_source_sha = source_sha
            row["depth_source_sha256"] = source_sha
            for definition in DEPTH_DEFINITIONS:
                selected = [
                    value
                    for value in depth_rows
                    if value["depth_definition"] == definition
                ]
                row[f"front_auc_{definition}"] = selected[0]["front_auc_normalized"]
                row[f"agreement_auc_{definition}"] = selected[0]["agreement_auc_normalized"]

        if alias is not None:
            alias_sha = self._verify_exact_alias(
                scene,
                _mapping(alias, f"{context}.alias"),
                context,
                depth_source_sha=depth_source_sha,
                oct_source_sha=oct_source_sha,
                oct_variant=oct_variant,
            )
            row["alias_receipt_sha256"] = alias_sha
            for depth_row in depth_rows:
                depth_row["alias_receipt_sha256"] = alias_sha

        for index, raw in enumerate(item.get("efficiency", [])):
            spec = _mapping(raw, f"{context}.efficiency[{index}]")
            efficiency_rows.extend(self._extract_efficiency(scene, row["method_id"], spec, context))

        artifact_rows.extend(
            self._extract_artifacts(scene, row["method_id"], item.get("artifacts", []), context, alias_sha)
        )

        row["rgb_appearance_source_sha256"] = appearance_sources.get("rgb", "")
        row["thermal_appearance_source_sha256"] = appearance_sources.get("thermal", "")
        row["appearance_source_sha256"] = _canonical_source_digest(appearance_sources)
        self._bind_legacy_appearance_resolution(row, context)
        self._bind_headline_efficiency(row, efficiency_rows, context)
        self._final_validate_row(row, context)
        return CompiledEndpoint(row, depth_rows, efficiency_rows, artifact_rows)

    def _extract_appearance(
        self,
        scene: str,
        row: MutableMapping[str, Any],
        spec: Mapping[str, Any],
        context: str,
    ) -> tuple[str, str]:
        kind = _text(spec.get("kind"), f"{context}.appearance.kind")
        modality = _text(spec.get("modality"), f"{context}.appearance.modality")
        if modality not in {"rgb", "thermal"}:
            raise ContractError(f"{context}: appearance modality must be rgb or thermal")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.appearance.{modality}")
        payload = _load_json(path, f"{context}.appearance.{modality}")
        if kind == "r1_aggregate":
            if (
                payload.get("schema") != "uav-tgs-formal-baseline-appearance-r1-aggregate-v1"
                or payload.get("status") != "passed"
            ):
                raise ContractError(f"{context}: invalid r1 appearance aggregate")
            _require_self_hash(payload, "aggregate_sha256", f"{context}.appearance.r1_aggregate")
            selector = _mapping(spec.get("selector"), f"{context}.appearance.selector")
            selected_scene = _text(selector.get("scene"), f"{context}.appearance.selector.scene")
            selected_method = _text(selector.get("method"), f"{context}.appearance.selector.method")
            matches = [
                item
                for item in _sequence(payload.get("evaluations"), f"{context}.appearance.evaluations")
                if isinstance(item, Mapping)
                and item.get("scene") == selected_scene
                and item.get("method") == selected_method
            ]
            if selected_scene != scene or len(matches) != 1:
                raise ContractError(f"{context}: r1 appearance selector is not unique for this scene")
            selected = matches[0]
            if selected.get("split") != "test":
                raise ContractError(f"{context}: r1 appearance must be formal test")
            formal_split = _nested(selected, "formal_split.sha256", f"{context}.appearance")
            self._check_protocol(scene, context, formal_split=formal_split)
            resolution = _mapping(selected.get("resolution"), f"{context}.appearance.resolution")
            label = _text(resolution.get("label"), f"{context}.appearance.resolution.label")
            width = _positive_int(resolution.get("width"), f"{context}.appearance.resolution.width")
            height = _positive_int(resolution.get("height"), f"{context}.appearance.resolution.height")
            metrics = _mapping(selected.get("metrics"), f"{context}.appearance.metrics")
        elif kind == "results":
            endpoint = _text(spec.get("endpoint"), f"{context}.appearance.endpoint")
            if set(payload) == {endpoint}:
                metrics = _mapping(payload[endpoint], f"{context}.appearance.results[{endpoint}]")
            else:
                raise ContractError(
                    f"{context}: results source must contain exactly the declared endpoint {endpoint!r}"
                )
            label, width, height = _resolution(spec, f"{context}.appearance")
            if spec.get("split") != "test":
                raise ContractError(f"{context}: results appearance must explicitly declare split=test")
            self._check_protocol(
                scene,
                context,
                formal_split=spec.get("formal_split_sha256"),
            )
        else:
            raise ContractError(f"{context}: unsupported appearance kind {kind!r}")

        keys = {"PSNR", "SSIM", "LPIPS"}
        if set(metrics) != keys:
            raise ContractError(f"{context}: appearance metrics must be exactly {sorted(keys)}")
        for source_name, suffix in (("PSNR", "psnr_db"), ("SSIM", "ssim"), ("LPIPS", "lpips")):
            _set_metric(row, f"{modality}_{suffix}", metrics[source_name], context)
        prefix = f"{modality}_appearance"
        if row.get(f"{prefix}_width") is not None:
            raise ContractError(f"{context}: duplicate {modality} appearance resolution")
        row[f"{prefix}_resolution_label"] = label
        row[f"{prefix}_width"] = width
        row[f"{prefix}_height"] = height
        return modality, source_sha

    def _bind_legacy_appearance_resolution(
        self, row: MutableMapping[str, Any], context: str
    ) -> None:
        """Populate the legacy resolution triple only when it is unambiguous."""

        triples: list[tuple[str, int, int]] = []
        for modality in ("rgb", "thermal"):
            prefix = f"{modality}_appearance"
            values = (
                row.get(f"{prefix}_resolution_label"),
                row.get(f"{prefix}_width"),
                row.get(f"{prefix}_height"),
            )
            if any(value is not None for value in values):
                if any(value is None for value in values):
                    raise ContractError(f"{context}: incomplete {modality} appearance resolution")
                triples.append(values)  # type: ignore[arg-type]
        if len(triples) == 1 or (len(triples) == 2 and triples[0] == triples[1]):
            label, width, height = triples[0]
            row["appearance_resolution_label"] = label
            row["appearance_width"] = width
            row["appearance_height"] = height

    def _validate_threshold(
        self, scene: str, threshold: Any, context: str
    ) -> Mapping[str, Any]:
        item = _mapping(threshold, f"{context}.hotspot_threshold")
        if (
            item.get("schema") != "uav-tgs-oct-train-only-hotspot-threshold-v1"
            or item.get("scene_name") != scene
            or item.get("source_split") != "train"
            or item.get("test_statistics_used") is not False
            or _finite(item.get("quantile"), f"{context}.hotspot_threshold.quantile") != 0.95
            or _positive_int(item.get("histogram_bins"), f"{context}.hotspot_threshold.histogram_bins")
            != HOTSPOT_BINS
        ):
            raise ContractError(f"{context}: hotspot threshold is not frozen train-q95/65536")
        supplied = _sha(item.get("threshold_sha256"), f"{context}.hotspot_threshold.threshold_sha256")
        if self_hash(item, "threshold_sha256") != supplied:
            raise ContractError(f"{context}: hotspot threshold self-hash mismatch")
        bound_split = _nested(item, "source_receipt.bound_split_sha256", context)
        self._check_protocol(
            scene,
            context,
            formal_split=bound_split,
            hotspot_threshold=supplied,
        )
        _finite(item.get("threshold_c"), f"{context}.hotspot_threshold.threshold_c")
        return item

    def _extract_baseline_hotspot(
        self,
        scene: str,
        row: MutableMapping[str, Any],
        spec: Mapping[str, Any],
        context: str,
    ) -> str:
        if spec.get("kind") != "baseline_hotspot":
            raise ContractError(f"{context}: thermal.kind must be baseline_hotspot")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.thermal")
        payload = _load_json(path, f"{context}.thermal")
        selector = _mapping(spec.get("selector"), f"{context}.thermal.selector")
        selected_scene = _text(selector.get("scene"), f"{context}.thermal.selector.scene")
        selected_method = _text(selector.get("method"), f"{context}.thermal.selector.method")
        if (
            payload.get("schema") != "uav-tgs-formal-baseline-hotspot-evaluation-v1"
            or payload.get("status") != "complete"
            or payload.get("scene_name") != selected_scene
            or payload.get("method_name") != selected_method
            or selected_scene != scene
            or payload.get("split") != "test"
        ):
            raise ContractError(f"{context}: baseline hotspot selector/report mismatch")
        _require_self_hash(payload, "report_payload_sha256", f"{context}.thermal")
        inputs = _mapping(payload.get("inputs"), f"{context}.thermal.inputs")
        if canonical_json_sha256(inputs) != _sha(
            payload.get("inputs_sha256"), f"{context}.thermal.inputs_sha256"
        ):
            raise ContractError(f"{context}: baseline hotspot inputs_sha256 mismatch")
        if _nested(payload, "formal_binding_compatibility.status", context) != "passed":
            raise ContractError(f"{context}: baseline hotspot formal binding did not pass")
        if _nested(payload, "display_semantics.comparable_to_oct_evaluator_v2", context) is not True:
            raise ContractError(f"{context}: baseline hotspot is not display-comparable")
        threshold = self._validate_threshold(scene, payload.get("hotspot_threshold"), context)
        self._check_protocol(scene, context, formal_split=_nested(inputs, "bound_split.sha256", context))
        boundary = _mapping(payload.get("selection_boundary"), f"{context}.selection_boundary")
        if boundary != {
            "threshold_source_split": "train",
            "test_statistics_used_for_threshold": False,
            "quantile": 0.95,
            "threshold_histogram_bins": HOTSPOT_BINS,
            "threshold_c": threshold["threshold_c"],
            "test_role": "final_report_only",
        }:
            raise ContractError(f"{context}: baseline hotspot selection boundary differs")
        metrics = _mapping(payload.get("metrics"), f"{context}.thermal.metrics")
        temperature = _mapping(metrics.get("temperature_error"), f"{context}.thermal.temperature_error")
        for source_name, output_name in (
            ("mae_c", "temp_mae_display_c"),
            ("rmse_c", "temp_rmse_display_c"),
            ("signed_bias_c", "temp_bias_display_c"),
            ("p95_abs_error_c", "temp_p95_display_c"),
        ):
            _set_metric(row, output_name, temperature[source_name], context)
        _set_metric(row, "hotspot_auprc_display", metrics["hotspot_auprc_histogram_4096"], context)
        _set_metric(row, "hotspot_iou_display", metrics["hotspot_iou"], context)
        off_lut = _mapping(metrics.get("off_lut_distance_rgb"), f"{context}.thermal.off_lut")
        _set_metric(row, "off_lut_mean_rgb", off_lut["mean"], context)
        _set_metric(row, "off_lut_p95_rgb", off_lut["p95"], context)
        row["temperature_metric_semantics_display"] = (
            "palette-inverted display-comparable apparent temperature"
        )
        row["hotspot_metric_domain"] = HOTSPOT_DOMAIN
        row["hotspot_threshold_rule"] = HOTSPOT_RULE
        row["hotspot_threshold_histogram_bins"] = HOTSPOT_BINS
        row["hotspot_threshold_sha256"] = threshold["threshold_sha256"]
        return source_sha

    def _extract_oct(
        self,
        scene: str,
        row: MutableMapping[str, Any],
        spec: Mapping[str, Any],
        context: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        if spec.get("kind") != "oct_evaluation_v2":
            raise ContractError(f"{context}: oct.kind must be oct_evaluation_v2")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.oct")
        payload = _load_json(path, f"{context}.oct")
        variant = _text(spec.get("variant"), f"{context}.oct.variant")
        if (
            payload.get("schema") != "uav-tgs-oct-formal-evaluation-v2"
            or payload.get("scene_name") != scene
            or payload.get("variant") != variant
            or payload.get("split") != "test"
            or _nested(payload, "training_evaluation_compatibility.status", context) != "passed"
            or _nested(payload, "shared_occupancy_invariant.exact", context) is not True
        ):
            raise ContractError(f"{context}: OCT evaluation-v2 contract mismatch")
        label, width, height = _resolution(spec, f"{context}.oct")
        if payload.get("resolution") != label:
            raise ContractError(f"{context}: OCT resolution label differs")
        threshold = self._validate_threshold(scene, payload.get("hotspot_threshold"), context)
        metrics = _mapping(payload.get("metrics"), f"{context}.oct.metrics")
        for source_name, output_name in (
            ("formal_full_frame_psnr_db", "thermal_psnr_db"),
            ("formal_full_frame_ssim", "thermal_ssim"),
            ("formal_full_frame_lpips", "thermal_lpips"),
            ("temperature_mae_c", "temp_mae_direct_c"),
            ("temperature_rmse_c", "temp_rmse_direct_c"),
            ("temperature_bias_c", "temp_bias_direct_c"),
            ("temperature_p95_abs_c", "temp_p95_direct_c"),
            ("hotspot_auprc_histogram_4096", "hotspot_auprc_display"),
            ("hotspot_iou", "hotspot_iou_display"),
        ):
            _set_metric(row, output_name, metrics[source_name], context)
        comparable = _mapping(metrics.get("palette_inverted_comparable"), f"{context}.oct.display")
        for source_name, output_name in (
            ("mae_c", "temp_mae_display_c"),
            ("rmse_c", "temp_rmse_display_c"),
            ("signed_bias_c", "temp_bias_display_c"),
            ("p95_abs_error_c", "temp_p95_display_c"),
        ):
            _set_metric(row, output_name, comparable[source_name], context)
        off_lut = _mapping(metrics.get("off_lut_distance"), f"{context}.oct.off_lut")
        _set_metric(row, "off_lut_mean_rgb", off_lut["mean_rgb_distance"], context)
        _set_metric(row, "off_lut_p95_rgb", off_lut["p95_rgb_distance"], context)
        row["thermal_appearance_resolution_label"] = label
        row["thermal_appearance_width"] = width
        row["thermal_appearance_height"] = height
        row["temperature_metric_semantics_direct"] = _text(
            metrics.get("temperature_semantics"), f"{context}.oct.temperature_semantics"
        )
        row["temperature_metric_semantics_display"] = _text(
            metrics.get("hotspot_primary_semantics"), f"{context}.oct.hotspot_primary_semantics"
        )
        row["hotspot_metric_domain"] = HOTSPOT_DOMAIN
        row["hotspot_threshold_rule"] = HOTSPOT_RULE
        row["hotspot_threshold_histogram_bins"] = HOTSPOT_BINS
        row["hotspot_threshold_sha256"] = threshold["threshold_sha256"]
        row["gaussian_count"] = _positive_int(
            _nested(payload, "shared_occupancy_invariant.topology_count", context),
            f"{context}.oct.gaussian_count",
        )
        efficiency = self._oct_efficiency_rows(scene, row["method_id"], payload, source_sha, width, height)
        return source_sha, efficiency

    def _oct_efficiency_rows(
        self,
        scene: str,
        method_id: str,
        payload: Mapping[str, Any],
        source_sha: str,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        cost = _mapping(payload.get("cost"), f"{scene}/{method_id}.oct.cost")
        training = _mapping(cost.get("verified_training_endpoint"), f"{scene}/{method_id}.oct.training")
        device = _mapping(training.get("device"), f"{scene}/{method_id}.oct.training.device")
        train_row = self._empty_efficiency(scene, method_id, "training", source_sha)
        train_row.update(
            {
                "stage": "oct_thermal",
                "iterations_or_views": _positive_int(training.get("optimizer_steps"), "oct.optimizer_steps"),
                "wall_time_s": _finite(training.get("wall_time_s"), "oct.wall_time_s", minimum=0),
                "end_to_end_wall_time_s": _finite(
                    training.get("end_to_end_wall_time_s"), "oct.end_to_end_wall_time_s", minimum=0
                ),
                "ms_per_step": _finite(training.get("ms_per_step"), "oct.ms_per_step", minimum=0),
                "peak_allocated_bytes": _positive_int(
                    device.get("peak_torch_allocated_bytes"), "oct.peak_allocated"
                ),
                "peak_reserved_bytes": _positive_int(
                    device.get("peak_torch_reserved_bytes"), "oct.peak_reserved"
                ),
                "gaussian_count": _positive_int(
                    _nested(payload, "shared_occupancy_invariant.topology_count", "oct"),
                    "oct.gaussian_count",
                ),
                "raster_passes": _positive_int(training.get("raster_passes"), "oct.raster_passes"),
                "device_name": _text(device.get("device_name"), "oct.device_name"),
                "timing_scope": "oct_end_to_end_training_including_pretraining_setup",
            }
        )
        train_row["comparison_group"] = f"training:{train_row['timing_scope']}"

        pure = _mapping(cost.get("pure_render"), f"{scene}/{method_id}.oct.pure_render")
        if pure.get("synchronized_cuda_event") is not True:
            raise ContractError(f"{scene}/{method_id}: OCT pure render is not synchronized")
        render_row = self._empty_efficiency(scene, method_id, "render", source_sha)
        render_row.update(
            {
                "stage": "oct_thermal",
                "iterations_or_views": _positive_int(pure.get("views"), "oct.render.views"),
                "width": width,
                "height": height,
                "resolution_label": payload.get("resolution"),
                "ms_per_view": _finite(pure.get("mean_ms_per_view"), "oct.render.ms_per_view", minimum=0),
                "fps": _finite(pure.get("fps"), "oct.render.fps", minimum=0),
                "gaussian_count": train_row["gaussian_count"],
                "device_name": train_row["device_name"],
                "timing_scope": "pure_render_synchronized_cuda_event",
            }
        )
        render_row["comparison_group"] = (
            f"render:{width}x{height}:{render_row['timing_scope']}"
        )
        return [train_row, render_row]

    def _extract_depth(
        self,
        scene: str,
        method_id: str,
        spec: Mapping[str, Any],
        context: str,
        alias_sha: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        if spec.get("kind") != "phase_c_depth":
            raise ContractError(f"{context}: depth.kind must be phase_c_depth")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.depth")
        payload = _load_json(path, f"{context}.depth")
        if (
            payload.get("protocol") != "formal-multi-depth-definition-and-top1-responsibility-v2"
            or payload.get("scene_name") != scene
        ):
            raise ContractError(f"{context}: Phase-C depth protocol/scene mismatch")
        thresholds = tuple(_finite(value, f"{context}.depth.thresholds") for value in payload.get("thresholds_m", []))
        if thresholds != DEPTH_THRESHOLDS:
            raise ContractError(f"{context}: Phase-C thresholds differ: {thresholds}")
        formal_split = payload.get("formal_split_manifest_sha256")
        reference = payload.get("reference_manifest_sha256")
        binding = _mapping(payload.get("all_split_reference_binding"), f"{context}.depth.binding")
        if binding.get("mesh_or_backend_rebuilt") is not False:
            raise ContractError(f"{context}: Phase-C reference backend was rebuilt")
        if binding.get("formal_split_manifest_sha256") != formal_split:
            raise ContractError(f"{context}: Phase-C binding split differs")
        self._check_protocol(
            scene,
            context,
            formal_split=formal_split,
            reference_manifest=reference,
        )
        definitions = _mapping(payload.get("depth_definitions"), f"{context}.depth.depth_definitions")
        if not set(DEPTH_DEFINITIONS).issubset(definitions):
            raise ContractError(f"{context}: missing a required Phase-C depth definition")
        rows: list[dict[str, Any]] = []
        for definition in DEPTH_DEFINITIONS:
            value = _mapping(definitions[definition], f"{context}.depth.{definition}")
            test = _mapping(
                _nested(value, "groups.split.test", f"{context}.depth.{definition}"),
                f"{context}.depth.{definition}.groups.split.test",
            )
            # Intentionally never reads depth_definitions.<def>.overall.
            curve = _sequence(test.get("threshold_metrics"), f"{context}.depth.{definition}.test.curve")
            by_tau: dict[float, Mapping[str, Any]] = {}
            for curve_index, raw in enumerate(curve):
                point = _mapping(raw, f"{context}.depth.{definition}.curve[{curve_index}]")
                tau = _finite(point.get("threshold_m"), f"{context}.depth.{definition}.threshold_m")
                if tau in by_tau:
                    raise ContractError(f"{context}: duplicate depth threshold {tau}")
                by_tau[tau] = point
            if tuple(sorted(by_tau)) != DEPTH_THRESHOLDS:
                raise ContractError(f"{context}: test curve thresholds differ for {definition}")
            front_auc = _finite(
                _nested(test, "front_curve_auc.normalized", context),
                f"{context}.depth.{definition}.front_auc",
                minimum=0,
                maximum=1,
            )
            agreement_auc = _finite(
                _nested(test, "agreement_curve_auc.normalized", context),
                f"{context}.depth.{definition}.agreement_auc",
                minimum=0,
                maximum=1,
            )
            shared = {
                "scene": scene,
                "method_id": method_id,
                "depth_definition": definition,
                "group_type": "split",
                "group_label": "test",
                "missing_rate": _finite(
                    test.get("missing_rate"),
                    f"{context}.depth.missing_rate",
                    minimum=0,
                    maximum=1,
                ),
                "mean_abs_error_m": _finite(
                    test.get("mean_abs_error_m"),
                    f"{context}.depth.mean_abs_error_m",
                    minimum=0,
                ),
                "median_abs_error_m": _finite(
                    test.get("median_abs_error_m"),
                    f"{context}.depth.median_abs_error_m",
                    minimum=0,
                ),
                "signed_bias_m": _finite(
                    test.get("signed_bias_m"), f"{context}.depth.signed_bias_m"
                ),
                "front_auc_normalized": front_auc,
                "agreement_auc_normalized": agreement_auc,
                "formal_split_sha256": _sha(formal_split, f"{context}.depth.formal_split"),
                "reference_manifest_sha256": _sha(reference, f"{context}.depth.reference"),
                "metrics_source_sha256": source_sha,
                "alias_receipt_sha256": alias_sha,
            }
            for tau in DEPTH_THRESHOLDS:
                point = by_tau[tau]
                rows.append(
                    {
                        **shared,
                        "threshold_m": tau,
                        "front_rate": _finite(
                            point.get("front_rate"),
                            f"{context}.depth.front_rate",
                            minimum=0,
                            maximum=1,
                        ),
                        "agreement_rate": _finite(
                            point.get("agreement_rate"),
                            f"{context}.depth.agreement_rate",
                            minimum=0,
                            maximum=1,
                        ),
                    }
                )
        return source_sha, rows

    def _verify_exact_alias(
        self,
        scene: str,
        spec: Mapping[str, Any],
        context: str,
        *,
        depth_source_sha: str | None,
        oct_source_sha: str | None,
        oct_variant: str | None,
    ) -> str:
        if spec.get("kind") != "exact_geometry":
            raise ContractError(f"{context}: alias.kind must be exact_geometry or noop")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.alias")
        payload = _load_json(path, f"{context}.alias")
        schema = payload.get("schema")
        if payload.get("status") != "passed":
            raise ContractError(f"{context}: alias receipt did not pass")
        _require_self_hash(payload, "receipt_sha256", f"{context}.alias")
        if schema == "uav-tgs-exact-geometry-occupancy-alias-v2":
            if oct_source_sha is not None:
                raise ContractError(f"{context}: OCT endpoint requires an OCT-specific alias receipt")
            expected_pair = {
                "source_label": _text(spec.get("source_label"), f"{context}.alias.source_label"),
                "alias_label": _text(spec.get("alias_label"), f"{context}.alias.alias_label"),
                "source_ply_sha256": _sha(
                    spec.get("source_ply_sha256"), f"{context}.alias.source_ply_sha256"
                ),
                "alias_ply_sha256": _sha(
                    spec.get("alias_ply_sha256"), f"{context}.alias.alias_ply_sha256"
                ),
                "formal_probe_camera_manifest_sha256": _sha(
                    spec.get("formal_probe_camera_manifest_sha256"),
                    f"{context}.alias.formal_probe_camera_manifest_sha256",
                ),
            }
            observed_pair = {
                "source_label": _text(
                    payload.get("source_label"), f"{context}.alias.receipt.source_label"
                ),
                "alias_label": _text(
                    payload.get("alias_label"), f"{context}.alias.receipt.alias_label"
                ),
                "source_ply_sha256": _sha(
                    _nested(payload, "source.ply.sha256", context),
                    f"{context}.alias.receipt.source_ply_sha256",
                ),
                "alias_ply_sha256": _sha(
                    _nested(payload, "alias.ply.sha256", context),
                    f"{context}.alias.receipt.alias_ply_sha256",
                ),
                "formal_probe_camera_manifest_sha256": _sha(
                    _nested(payload, "formal_probe_camera_manifest.sha256", context),
                    f"{context}.alias.receipt.formal_probe_camera_manifest_sha256",
                ),
            }
            if observed_pair != expected_pair:
                raise ContractError(f"{context}: pair alias identity differs from manifest")
            if (
                payload.get("ordered_vertex_schema_exact") is not True
                or payload.get("ordered_xyz_and_topology_exact") is not True
            ):
                raise ContractError(f"{context}: pair alias is not exact")
            fields = _mapping(payload.get("geometry_and_occupancy_fields"), f"{context}.alias.fields")
            if len(fields) != 11 or any(
                not isinstance(value, Mapping)
                or value.get("exact") is not True
                or _finite(value.get("max_abs_diff"), f"{context}.alias.max_abs_diff") != 0.0
                for value in fields.values()
            ):
                raise ContractError(f"{context}: pair alias fields are not exact")
        elif schema == "uav-tgs-oct-exact-raw-anchor-alias-v2":
            if oct_source_sha is None:
                raise ContractError(f"{context}: OCT alias requires a declared OCT evaluation source")
            if depth_source_sha is None:
                raise ContractError(f"{context}: OCT alias requires a declared inherited depth source")
            expected_variant = _text(spec.get("variant"), f"{context}.alias.variant")
            expected_checkpoint = _sha(
                spec.get("endpoint_checkpoint_sha256"),
                f"{context}.alias.endpoint_checkpoint_sha256",
            )
            expected_probe = _sha(
                spec.get("formal_probe_camera_manifest_sha256"),
                f"{context}.alias.formal_probe_camera_manifest_sha256",
            )
            observed_variant = _text(
                payload.get("variant"), f"{context}.alias.receipt.variant"
            )
            observed_checkpoint = _sha(
                _nested(payload, "endpoint.checkpoint.sha256", context),
                f"{context}.alias.receipt.endpoint_checkpoint_sha256",
            )
            observed_probe = _sha(
                _nested(payload, "formal_probe_camera_manifest.sha256", context),
                f"{context}.alias.receipt.formal_probe_camera_manifest_sha256",
            )
            evaluation_sha = _sha(
                _nested(payload, "formal_evaluation_v2.sha256", context),
                f"{context}.alias.receipt.formal_evaluation_v2.sha256",
            )
            inherited = _sha(
                _nested(payload, "inherited_phase_c_depth.metrics_summary.sha256", context),
                f"{context}.alias.receipt.inherited_depth.sha256",
            )
            if (
                payload.get("scene") != scene
                or oct_variant is None
                or expected_variant != oct_variant
                or observed_variant != expected_variant
                or observed_checkpoint != expected_checkpoint
                or observed_probe != expected_probe
                or evaluation_sha != oct_source_sha
                or inherited != depth_source_sha
                or _nested(payload, "runtime_before_after.exact", context) is not True
            ):
                raise ContractError(f"{context}: OCT runtime alias is not exact for this scene")
            self._check_protocol(
                scene,
                context,
                formal_split=_nested(payload, "formal_camera_and_split_binding.bound_split.sha256", context),
            )
        else:
            raise ContractError(f"{context}: unsupported exact alias schema {schema!r}")
        return source_sha

    def _extract_efficiency(
        self,
        scene: str,
        method_id: str,
        spec: Mapping[str, Any],
        context: str,
    ) -> list[dict[str, Any]]:
        if spec.get("kind") != "uav_tgs_efficiency":
            raise ContractError(f"{context}: unsupported efficiency kind {spec.get('kind')!r}")
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.efficiency")
        payload = _load_json(path, f"{context}.efficiency")
        if (
            payload.get("schema_name") != "uav-tgs-efficiency"
            or payload.get("schema_version") != 1
            or payload.get("status") != "completed"
        ):
            raise ContractError(f"{context}: invalid uav-tgs efficiency report")
        kind = payload.get("kind")
        if kind == "training_stage":
            row = self._empty_efficiency(scene, method_id, "training", source_sha)
            device = _mapping(payload.get("device"), f"{context}.efficiency.device")
            result = _mapping(payload.get("result"), f"{context}.efficiency.result")
            iterations = result.get("optimizer_updates_executed", result.get("iterations_executed"))
            row.update(
                {
                    "stage": _text(payload.get("stage"), f"{context}.efficiency.stage"),
                    "iterations_or_views": _positive_int(iterations, f"{context}.efficiency.iterations"),
                    "wall_time_s": _finite(payload.get("wall_time_s"), f"{context}.efficiency.wall", minimum=0),
                    "end_to_end_wall_time_s": _finite(
                        payload.get("wall_time_s"), f"{context}.efficiency.wall", minimum=0
                    ),
                    "peak_allocated_bytes": _positive_int(
                        device.get("peak_torch_allocated_bytes"), f"{context}.efficiency.peak_allocated"
                    ),
                    "peak_reserved_bytes": _positive_int(
                        device.get("peak_torch_reserved_bytes"), f"{context}.efficiency.peak_reserved"
                    ),
                    "gaussian_count": _positive_int(
                        result.get("gaussian_count"), f"{context}.efficiency.gaussian_count"
                    ),
                    "device_name": _text(device.get("device_name"), f"{context}.efficiency.device_name"),
                    "timing_scope": _text(payload.get("timing_scope"), f"{context}.efficiency.timing_scope"),
                }
            )
            row["comparison_group"] = f"training:{row['timing_scope']}"
            return [row]
        if kind == "render":
            row = self._empty_efficiency(scene, method_id, "render", source_sha)
            benchmark = _mapping(payload.get("benchmark"), f"{context}.efficiency.benchmark")
            device = _mapping(benchmark.get("device"), f"{context}.efficiency.device")
            resolutions = _sequence(payload.get("resolutions"), f"{context}.efficiency.resolutions")
            if len(resolutions) == 1:
                resolution = _mapping(resolutions[0], f"{context}.efficiency.resolution")
                width = _positive_int(resolution.get("width"), f"{context}.efficiency.width")
                height = _positive_int(resolution.get("height"), f"{context}.efficiency.height")
                resolution_label = f"{width}x{height}"
            else:
                width = height = None
                normalized = [
                    [
                        _positive_int(_mapping(value, "resolution").get("width"), "resolution.width"),
                        _positive_int(_mapping(value, "resolution").get("height"), "resolution.height"),
                    ]
                    for value in resolutions
                ]
                resolution_label = "mixed:" + json.dumps(normalized, separators=(",", ":"))
            ms = benchmark.get("cuda_event_ms_per_view", benchmark.get("wall_ms_per_view"))
            fps = benchmark.get("cuda_event_fps", benchmark.get("wall_fps"))
            row.update(
                {
                    "stage": _text(spec.get("stage", "thermal"), f"{context}.efficiency.stage"),
                    "iterations_or_views": _positive_int(
                        benchmark.get("timed_views"), f"{context}.efficiency.timed_views"
                    ),
                    "width": width,
                    "height": height,
                    "resolution_label": resolution_label,
                    "wall_time_s": _finite(
                        benchmark.get("wall_total_s"), f"{context}.efficiency.wall_total", minimum=0
                    ),
                    "ms_per_view": _finite(ms, f"{context}.efficiency.ms_per_view", minimum=0),
                    "fps": _finite(fps, f"{context}.efficiency.fps", minimum=0),
                    "peak_allocated_bytes": _positive_int(
                        device.get("peak_torch_allocated_bytes"), f"{context}.efficiency.peak_allocated"
                    ),
                    "peak_reserved_bytes": _positive_int(
                        device.get("peak_torch_reserved_bytes"), f"{context}.efficiency.peak_reserved"
                    ),
                    "gaussian_count": _positive_int(
                        payload.get("gaussian_count"), f"{context}.efficiency.gaussian_count"
                    ),
                    "device_name": _text(device.get("device_name"), f"{context}.efficiency.device_name"),
                    "timing_scope": _text(
                        benchmark.get("timing_scope"), f"{context}.efficiency.timing_scope"
                    ),
                }
            )
            row["comparison_group"] = (
                f"render:{resolution_label}:{row['timing_scope']}"
            )
            return [row]
        raise ContractError(f"{context}: unsupported uav-tgs efficiency report kind {kind!r}")

    @staticmethod
    def _empty_efficiency(scene: str, method_id: str, kind: str, source_sha: str) -> dict[str, Any]:
        row = {field: None for field in EFFICIENCY_FIELDS}
        row.update(
            {
                "scene": scene,
                "method_id": method_id,
                "kind": kind,
                "source_sha256": source_sha,
                "alias_receipt_sha256": "",
            }
        )
        return row

    def _extract_artifacts(
        self,
        scene: str,
        method_id: str,
        raw_items: Any,
        context: str,
        alias_sha: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(_sequence(raw_items, f"{context}.artifacts")):
            item = _mapping(raw, f"{context}.artifacts[{index}]")
            path, artifact_sha = self.registry.verify(
                item.get("source"), f"{context}.artifact[{index}]"
            )
            definition = _text(
                item.get("depth_definition", ""),
                f"{context}.artifact[{index}].depth_definition",
                allow_empty=True,
            )
            if definition and definition not in DEPTH_DEFINITIONS:
                raise ContractError(f"{context}: invalid artifact depth definition {definition}")
            rows.append(
                {
                    "scene": scene,
                    "method_id": method_id,
                    "artifact_role": _text(
                        item.get("artifact_role"), f"{context}.artifact[{index}].artifact_role"
                    ),
                    "split": _text(
                        item.get("split", ""), f"{context}.artifact[{index}].split", allow_empty=True
                    ),
                    "block_id": _text(
                        item.get("block_id", ""), f"{context}.artifact[{index}].block_id", allow_empty=True
                    ),
                    "view_id": _text(
                        item.get("view_id", ""), f"{context}.artifact[{index}].view_id", allow_empty=True
                    ),
                    "depth_definition": definition,
                    "path": str(path),
                    "sha256": artifact_sha,
                    "source_manifest_sha256": self.source_manifest_sha256,
                    "alias_receipt_sha256": alias_sha,
                }
            )
        return rows

    def _bind_headline_efficiency(
        self, row: MutableMapping[str, Any], values: list[dict[str, Any]], context: str
    ) -> None:
        train = [item for item in values if item["kind"] == "training"]
        render = [item for item in values if item["kind"] == "render"]
        if len(train) > 1:
            raise ContractError(f"{context}: multiple training costs require separate endpoints")
        if len(render) > 1:
            raise ContractError(f"{context}: multiple render costs require separate endpoints")
        sources: dict[str, str] = {}
        if train:
            selected = train[0]
            row["train_wall_time_s"] = selected.get("end_to_end_wall_time_s") or selected.get("wall_time_s")
            row["train_timing_scope"] = selected.get("timing_scope")
            row["gaussian_count"] = selected.get("gaussian_count")
            sources["training"] = selected["source_sha256"]
        if render:
            selected = render[0]
            row["render_ms_per_view"] = selected.get("ms_per_view")
            row["render_width"] = selected.get("width")
            row["render_height"] = selected.get("height")
            row["render_timing_scope"] = selected.get("timing_scope")
            # Backward-compatible alias: the old generic field always means
            # render timing scope and is never populated from training.
            row["timing_scope"] = row["render_timing_scope"]
            if row.get("gaussian_count") is None:
                row["gaussian_count"] = selected.get("gaussian_count")
            elif selected.get("gaussian_count") != row["gaussian_count"]:
                raise ContractError(f"{context}: training/render Gaussian counts differ")
            sources["render"] = selected["source_sha256"]
        for item in values:
            item["alias_receipt_sha256"] = row.get("alias_receipt_sha256") or ""
        row["efficiency_source_sha256"] = _canonical_source_digest(sources)

    def _verify_noop_receipt(
        self, scene: str, source_method: str, alias_method: str, spec: Mapping[str, Any], context: str
    ) -> str:
        path, source_sha = self.registry.verify(spec.get("source"), f"{context}.noop")
        payload = _load_json(path, f"{context}.noop")
        schema = payload.get("schema")
        if schema != NOOP_ENDPOINT_COVERAGE_SCHEMA:
            raise ContractError(f"{context}: unsupported no-op receipt schema {schema!r}")
        _require_self_hash(payload, "receipt_sha256", f"{context}.noop")
        if "selector" in spec:
            raise ContractError(f"{context}: dedicated no-op coverage does not use a selector")
        expected_alias_method = _text(
            spec.get("alias_method_id"), f"{context}.noop.alias_method_id"
        )
        expected_copy_roles = _text(spec.get("copy_roles"), f"{context}.noop.copy_roles")
        expected_metric_roles = tuple(
            _text(value, f"{context}.noop.copied_metric_roles")
            for value in _sequence(
                spec.get("copied_metric_roles"), f"{context}.noop.copied_metric_roles"
            )
        )
        expected_claim = _text(
            spec.get("claim_boundary"), f"{context}.noop.claim_boundary"
        )
        observed_metric_roles = tuple(
            _text(value, f"{context}.noop.receipt.copied_metric_roles")
            for value in _sequence(
                payload.get("copied_metric_roles"),
                f"{context}.noop.receipt.copied_metric_roles",
            )
        )
        if (
            source_method != _text(
                spec.get("source_method_id"), f"{context}.noop.source_method_id"
            )
            or expected_alias_method != alias_method
            or expected_copy_roles != NOOP_COPY_ROLES
            or expected_metric_roles != NOOP_COPIED_METRIC_ROLES
            or expected_claim != NOOP_CLAIM_BOUNDARY
            or payload.get("status") != "passed"
            or payload.get("evidence_status") != NOOP_EVIDENCE_STATUS
            or payload.get("independent_endpoint_run") is not False
            or payload.get("independent_performance_claim") is not False
            or payload.get("scene") != scene
            or payload.get("source_method_id") != source_method
            or payload.get("alias_method_id") != alias_method
            or payload.get("copy_roles") != expected_copy_roles
            or observed_metric_roles != expected_metric_roles
            or payload.get("claim_boundary") != expected_claim
        ):
            raise ContractError(f"{context}: protocol no-op endpoint coverage differs")
        self._check_protocol(scene, context, formal_split=payload.get("formal_split_sha256"))
        return source_sha

    def _compile_noop(
        self, item: Mapping[str, Any], source: CompiledEndpoint
    ) -> CompiledEndpoint:
        context = f"{item.get('scene')}/{item.get('method_id')}"
        # No-op copying is intentionally all-or-nothing and explicit.  Mixing a
        # copy with new components would obscure which evidence supports a row.
        forbidden = [key for key in ("appearance", "thermal", "oct", "depth", "efficiency", "artifacts") if item.get(key)]
        if forbidden:
            raise ContractError(f"{context}: no-op endpoint cannot also declare components: {forbidden}")
        row = self._base_row(item, context)
        if row["evidence_status"] != NOOP_ENDPOINT_EVIDENCE_STATUS:
            raise ContractError(
                f"{context}: no-op endpoint evidence_status must be "
                f"{NOOP_ENDPOINT_EVIDENCE_STATUS}"
            )
        alias = _mapping(item.get("alias"), f"{context}.alias")
        if alias.get("copy_roles") != NOOP_COPY_ROLES:
            raise ContractError(f"{context}: no-op alias must explicitly declare copy_roles=all")
        source_method = _text(alias.get("source_method_id"), f"{context}.alias.source_method_id")
        receipt_sha = self._verify_noop_receipt(
            row["scene"], source_method, row["method_id"], alias, context
        )
        identity_values = {field: row[field] for field in IDENTITY_FIELDS}
        evidence_status = row["evidence_status"]
        structural = json.loads(row["structural_metadata_json"])
        if (
            structural.get("independent_training_run") is not False
            or structural.get("independent_performance_claim") is not False
        ):
            raise ContractError(
                f"{context}: no-op endpoint must explicitly deny independent "
                "training and performance claims"
            )
        row = copy.deepcopy(source.row)
        row.update(identity_values)
        row["evidence_status"] = evidence_status
        structural.update(
            {
                "alias_kind": "explicit_noop",
                "alias_source_method_id": source_method,
                "alias_receipt_schema": NOOP_ENDPOINT_COVERAGE_SCHEMA,
                "alias_claim_boundary": NOOP_CLAIM_BOUNDARY,
            }
        )
        row["structural_metadata_json"] = json.dumps(
            structural, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        row["alias_receipt_sha256"] = receipt_sha
        depth_rows = [
            {**copy.deepcopy(value), "method_id": row["method_id"], "alias_receipt_sha256": receipt_sha}
            for value in source.depth_rows
        ]
        efficiency_rows = [
            {**copy.deepcopy(value), "method_id": row["method_id"], "alias_receipt_sha256": receipt_sha}
            for value in source.efficiency_rows
        ]
        artifact_rows = [
            {**copy.deepcopy(value), "method_id": row["method_id"], "alias_receipt_sha256": receipt_sha}
            for value in source.artifact_rows
        ]
        self._final_validate_row(row, context)
        return CompiledEndpoint(row, depth_rows, efficiency_rows, artifact_rows)

    def _final_validate_row(self, row: Mapping[str, Any], context: str) -> None:
        for field in IDENTITY_FIELDS:
            _text(row.get(field), f"{context}.{field}")
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ContractError(f"{context}: non-finite output {key}")
        hotspot_present = row.get("hotspot_auprc_display") is not None or row.get("hotspot_iou_display") is not None
        if hotspot_present:
            expected = {
                "hotspot_metric_domain": HOTSPOT_DOMAIN,
                "hotspot_threshold_rule": HOTSPOT_RULE,
                "hotspot_threshold_histogram_bins": HOTSPOT_BINS,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ContractError(f"{context}: invalid hotspot contract field {key}")
            _sha(row.get("hotspot_source_sha256"), f"{context}.hotspot_source_sha256")
            _sha(row.get("hotspot_threshold_sha256"), f"{context}.hotspot_threshold_sha256")
        for category, fields in (
            ("appearance_source_sha256", ("rgb_psnr_db", "thermal_psnr_db")),
            ("temperature_source_sha256", ("temp_mae_direct_c", "temp_mae_display_c")),
            ("depth_source_sha256", ("front_auc_expected", "front_auc_median", "front_auc_max_contribution")),
            ("efficiency_source_sha256", ("train_wall_time_s", "render_ms_per_view", "gaussian_count")),
        ):
            if any(row.get(field) is not None for field in fields):
                _sha(row.get(category), f"{context}.{category}")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("attempted to emit a non-finite CSV value")
        return format(value, ".17g")
    return value


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def compile_reassessment_input(manifest_path: Path, output_dir: Path) -> Mapping[str, Any]:
    """Compile and atomically publish the normalized reassessment input bundle."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    compiler = ParetoInputCompiler(manifest_path)
    endpoints = compiler.compile()
    endpoint_rows = [item.row for item in endpoints]
    depth_rows = [row for item in endpoints for row in item.depth_rows]
    efficiency_rows = [row for item in endpoints for row in item.efficiency_rows]
    artifact_rows = [row for item in endpoints for row in item.artifact_rows]
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=parent))
    try:
        _write_csv(temporary / "endpoint_summary.csv", ENDPOINT_FIELDS, endpoint_rows)
        _write_csv(temporary / "depth_curve.csv", DEPTH_FIELDS, depth_rows)
        _write_csv(temporary / "efficiency.csv", EFFICIENCY_FIELDS, efficiency_rows)
        _write_csv(temporary / "artifact_manifest.csv", ARTIFACT_FIELDS, artifact_rows)

        verified_source = {
            "schema": COMPILED_SOURCE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_manifest": {
                "path": str(manifest_path),
                "file_sha256": sha256_file(manifest_path),
                "canonical_manifest_sha256": compiler.source_manifest_sha256,
                "size_bytes": manifest_path.stat().st_size,
            },
            "scene_protocols": [
                {
                    "scene": value.scene,
                    "formal_split_sha256": value.formal_split_sha256,
                    "reference_manifest_sha256": value.reference_manifest_sha256,
                    "hotspot_threshold_sha256": value.hotspot_threshold_sha256,
                }
                for value in sorted(compiler.protocols.values(), key=lambda item: item.scene)
            ],
            "verified_inputs": compiler.registry.records(),
            "compiled_endpoint_keys": [
                [row["scene"], row["method_id"]] for row in endpoint_rows
            ],
        }
        verified_source["manifest_sha256"] = self_hash(verified_source, "manifest_sha256")
        source_out = temporary / "source_manifest.json"
        source_out.write_text(
            json.dumps(verified_source, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Preserve the user's declared source manifest byte-for-byte.  The
        # generated source_manifest.json above is a verification summary, not
        # a substitute for the original declaration.
        declared_source_out = temporary / "declared_source_manifest.json"
        shutil.copyfile(manifest_path, declared_source_out)
        if sha256_file(declared_source_out) != sha256_file(manifest_path):
            raise ContractError("declared source manifest copy is not byte-identical")

        output_names = (
            "endpoint_summary.csv",
            "depth_curve.csv",
            "efficiency.csv",
            "artifact_manifest.csv",
            "source_manifest.json",
            "declared_source_manifest.json",
        )
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "compiler": _identity(Path(__file__)),
            "source_manifest_sha256": compiler.source_manifest_sha256,
            "row_counts": {
                "endpoint_summary": len(endpoint_rows),
                "depth_curve": len(depth_rows),
                "efficiency": len(efficiency_rows),
                "artifact_manifest": len(artifact_rows),
            },
            "policies": {
                "manifest_driven_no_discovery": True,
                "missing_values_emitted_as_empty": True,
                "depth_scope": "split/test only",
                "depth_definitions": list(DEPTH_DEFINITIONS),
                "depth_thresholds_m": list(DEPTH_THRESHOLDS),
                "hotspot_domain": HOTSPOT_DOMAIN,
                "hotspot_threshold_rule": HOTSPOT_RULE,
                "hotspot_threshold_histogram_bins": HOTSPOT_BINS,
                "oct_direct_and_display_temperature_separate": True,
                "implicit_noop_or_method_generation": False,
                "declared_source_manifest_copied_verbatim": True,
                "render_efficiency_stratified_by_resolution_and_timing_scope": True,
            },
            "outputs": {name: _identity(temporary / name) for name in output_names},
        }
        # Paths in receipts describe the final published directory, not the
        # disposable staging directory.
        for name, receipt in manifest["outputs"].items():
            receipt["path"] = str((output_dir / name).resolve())
        manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="self-hashed v1 source manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compile_reassessment_input(args.manifest, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "manifest_sha256": result["manifest_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
