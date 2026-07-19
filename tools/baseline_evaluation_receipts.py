#!/usr/bin/env python3
"""Validate and summarize scoped external-baseline evaluation receipts.

This module keeps three identities deliberately separate:

* an adapter/training receipt owns a reusable endpoint;
* an evaluator receipt owns metrics derived from that endpoint;
* an explicit alias receipt may report the exact endpoint under another method
  name without duplicating either measured performance or batch cost.

Relevant-file hashes, settings, inputs, and environment form a scoped
provenance digest.  Repository metadata and absolute paths are audit metadata
only, so relocating a run or changing an unrelated repository file cannot
invalidate a trained endpoint.  Evaluator provenance is independently scoped;
changing evaluator-only code requires reevaluation, never retraining.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence


ADAPTER_SCHEMA = "uav-tgs-baseline-adapter-receipt-v1"
EVALUATOR_SCHEMA = "uav-tgs-baseline-evaluator-receipt-v1"
ALIAS_SCHEMA = "uav-tgs-baseline-alias-receipt-v1"
STUDY_SCHEMA = "uav-tgs-baseline-study-manifest-v1"
SUMMARY_SCHEMA = "uav-tgs-baseline-study-summary-v1"
SCHEMA_BUNDLE = "uav_tgs_baseline_receipts_v1.schema.json"

SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
UNSUPPORTED = "UNSUPPORTED"
NOT_REQUIRED = "NOT_REQUIRED"
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, UNSUPPORTED})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
FORMAL_DEPTH_DEFINITIONS = (
    "transmittance_median",
    "alpha_weighted_expected",
    "maximum_contribution_surface",
)
FORMAL_GEOMETRY_THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)


class ReceiptContractError(ValueError):
    """Raised when a receipt violates the unified baseline contract."""


def _reject_unknown_keys(
    payload: Mapping[str, Any], allowed: Iterable[str], label: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise ReceiptContractError(f"{label} has unknown fields: {unknown}")


def canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptContractError(f"value is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptContractError(f"{label} must be an object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReceiptContractError(f"{label} must be a nonempty string")
    return value.strip()


def _require_sha(value: Any, label: str) -> str:
    text = _require_nonempty_string(value, label).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise ReceiptContractError(f"{label} must be a lowercase SHA-256")
    return text


def _require_finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ReceiptContractError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReceiptContractError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ReceiptContractError(f"{label} must be {qualifier}")
    return number


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptContractError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ReceiptContractError(f"{label} must be >= {minimum}")
    return value


def _require_string_list(
    value: Any, label: str, *, nonempty: bool = False
) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a nonempty list" if nonempty else "a list"
        raise ReceiptContractError(f"{label} must be {qualifier}")
    result = [_require_nonempty_string(item, f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise ReceiptContractError(f"{label} must not contain duplicates")
    return result


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    material = copy.deepcopy(dict(payload))
    material.pop(field, None)
    return canonical_json_sha256(material)


def _finalize_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(dict(payload))
    result[field] = _self_hash(result, field)
    return result


def _validate_self_hash(payload: Mapping[str, Any], field: str, label: str) -> str:
    declared = _require_sha(payload.get(field), f"{label}.{field}")
    actual = _self_hash(payload, field)
    if declared != actual:
        raise ReceiptContractError(
            f"{label} self-hash mismatch: declared={declared} actual={actual}"
        )
    return declared


def _identity_hash_material(entries: Any, label: str) -> list[dict[str, str]]:
    if not isinstance(entries, list):
        raise ReceiptContractError(f"{label} must be a list")
    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for offset, raw in enumerate(entries):
        entry = _require_mapping(raw, f"{label}[{offset}]")
        _reject_unknown_keys(
            entry,
            {"role", "logical_id", "path", "sha256", "size_bytes"},
            f"{label}[{offset}]",
        )
        role = _require_nonempty_string(entry.get("role"), f"{label}[{offset}].role")
        digest = _require_sha(entry.get("sha256"), f"{label}[{offset}].sha256")
        logical_id = str(entry.get("logical_id", "")).strip()
        if "path" in entry and not isinstance(entry["path"], str):
            raise ReceiptContractError(f"{label}[{offset}].path must be a string")
        if "size_bytes" in entry:
            _require_int(entry["size_bytes"], f"{label}[{offset}].size_bytes", minimum=0)
        key = (role, logical_id)
        if key in seen:
            raise ReceiptContractError(f"{label} has duplicate role/logical_id: {key}")
        seen.add(key)
        item = {"role": role, "sha256": digest}
        if logical_id:
            item["logical_id"] = logical_id
        normalized.append(item)
    normalized.sort(key=lambda item: (item["role"], item.get("logical_id", ""), item["sha256"]))
    return normalized


def scoped_provenance_sha256(provenance: Mapping[str, Any]) -> str:
    """Hash only execution-relevant provenance, excluding paths/repo metadata."""

    payload = _require_mapping(provenance, "provenance")
    _reject_unknown_keys(
        payload,
        {
            "scope",
            "relevant_files",
            "inputs",
            "settings",
            "environment",
            "repository",
            "scope_sha256",
        },
        "provenance",
    )
    scope = _require_nonempty_string(payload.get("scope"), "provenance.scope")
    files = _identity_hash_material(payload.get("relevant_files"), "provenance.relevant_files")
    inputs = _identity_hash_material(payload.get("inputs"), "provenance.inputs")
    settings = _require_mapping(payload.get("settings"), "provenance.settings")
    environment = _require_mapping(payload.get("environment"), "provenance.environment")
    material = {
        "scope": scope,
        "relevant_files": files,
        "inputs": inputs,
        "settings": copy.deepcopy(dict(settings)),
        "environment": copy.deepcopy(dict(environment)),
    }
    return canonical_json_sha256(material)


def make_scoped_provenance(
    *,
    scope: str,
    relevant_files: Sequence[Mapping[str, Any]],
    inputs: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    environment: Mapping[str, Any],
    repository: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create provenance whose digest is stable under unrelated repo changes."""

    result: dict[str, Any] = {
        "scope": str(scope),
        "relevant_files": [copy.deepcopy(dict(item)) for item in relevant_files],
        "inputs": [copy.deepcopy(dict(item)) for item in inputs],
        "settings": copy.deepcopy(dict(settings)),
        "environment": copy.deepcopy(dict(environment)),
        "repository": copy.deepcopy(dict(repository or {})),
    }
    result["scope_sha256"] = scoped_provenance_sha256(result)
    return result


def validate_scoped_provenance(
    provenance: Any, *, expected_scope: str, label: str
) -> str:
    payload = _require_mapping(provenance, label)
    if _require_nonempty_string(payload.get("scope"), f"{label}.scope") != expected_scope:
        raise ReceiptContractError(f"{label}.scope must be {expected_scope!r}")
    _identity_hash_material(payload.get("relevant_files"), f"{label}.relevant_files")
    _identity_hash_material(payload.get("inputs"), f"{label}.inputs")
    _require_mapping(payload.get("settings"), f"{label}.settings")
    _require_mapping(payload.get("environment"), f"{label}.environment")
    if "repository" in payload:
        _require_mapping(payload["repository"], f"{label}.repository")
    declared = _require_sha(payload.get("scope_sha256"), f"{label}.scope_sha256")
    actual = scoped_provenance_sha256(payload)
    if declared != actual:
        raise ReceiptContractError(
            f"{label} scoped provenance mismatch: declared={declared} actual={actual}"
        )
    return declared


def _artifact_material(artifacts: Any, label: str) -> list[dict[str, str]]:
    return _identity_hash_material(artifacts, label)


def _validate_model_profile(profile: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(profile, label)
    _reject_unknown_keys(
        payload,
        {"representation", "model_size_bytes", "parameter_count", "gaussian_count"},
        label,
    )
    representation = _require_nonempty_string(
        payload.get("representation"), f"{label}.representation"
    )
    if representation not in {"gaussian", "non_gaussian"}:
        raise ReceiptContractError(
            f"{label}.representation must be 'gaussian' or 'non_gaussian'"
        )
    model_size = _require_int(
        payload.get("model_size_bytes"), f"{label}.model_size_bytes", minimum=0
    )
    parameter_count = _require_int(
        payload.get("parameter_count"), f"{label}.parameter_count", minimum=0
    )
    result: dict[str, Any] = {
        "representation": representation,
        "model_size_bytes": model_size,
        "parameter_count": parameter_count,
    }
    if representation == "gaussian":
        result["gaussian_count"] = _require_int(
            payload.get("gaussian_count"), f"{label}.gaussian_count", minimum=0
        )
    elif "gaussian_count" in payload:
        raise ReceiptContractError(
            f"{label}.gaussian_count is only valid for gaussian representations"
        )
    return result


def _validate_method_origin(origin: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(origin, label)
    fields = {
        "kind",
        "repository_url",
        "source_commit",
        "license",
        "adapter_commit",
        "compatibility_patch_sha256",
        "official_recipe_id",
        "input_conversion",
        "camera_mapping",
        "split_mapping",
        "output_conversion",
        "semantic_limitations",
        "adapter_changes_method",
    }
    _reject_unknown_keys(payload, fields, label)
    kind = _require_nonempty_string(payload.get("kind"), f"{label}.kind")
    if kind not in {"internal", "external"}:
        raise ReceiptContractError(f"{label}.kind must be internal or external")
    result: dict[str, Any] = {"kind": kind}
    for field in (
        "repository_url",
        "license",
        "official_recipe_id",
        "input_conversion",
        "camera_mapping",
        "split_mapping",
        "output_conversion",
    ):
        result[field] = _require_nonempty_string(payload.get(field), f"{label}.{field}")
    for field in ("source_commit", "adapter_commit"):
        commit = _require_nonempty_string(payload.get(field), f"{label}.{field}").lower()
        if GIT_COMMIT_RE.fullmatch(commit) is None:
            raise ReceiptContractError(f"{label}.{field} must be a 40- or 64-hex commit")
        result[field] = commit
    result["compatibility_patch_sha256"] = _require_sha(
        payload.get("compatibility_patch_sha256"),
        f"{label}.compatibility_patch_sha256",
    )
    result["semantic_limitations"] = _require_string_list(
        payload.get("semantic_limitations"), f"{label}.semantic_limitations"
    )
    if payload.get("adapter_changes_method") is not False:
        raise ReceiptContractError(f"{label}.adapter_changes_method must be false")
    result["adapter_changes_method"] = False
    return result


def endpoint_reuse_key(receipt: Mapping[str, Any]) -> str:
    scene = _require_nonempty_string(receipt.get("scene"), "adapter.scene")
    adapter_id = _require_nonempty_string(receipt.get("adapter_id"), "adapter.adapter_id")
    run_scope = _require_nonempty_string(receipt.get("run_scope"), "adapter.run_scope")
    method_origin = _validate_method_origin(
        receipt.get("method_origin"), "adapter.method_origin"
    )
    training_sha = validate_scoped_provenance(
        receipt.get("training_provenance"),
        expected_scope="training_endpoint",
        label="adapter.training_provenance",
    )
    endpoint = _require_mapping(receipt.get("endpoint"), "adapter.endpoint")
    artifacts = _artifact_material(endpoint.get("artifacts"), "adapter.endpoint.artifacts")
    endpoint_kind = _require_nonempty_string(
        endpoint.get("endpoint_kind"), "adapter.endpoint.endpoint_kind"
    )
    model_profile = _validate_model_profile(
        endpoint.get("model_profile"), "adapter.endpoint.model_profile"
    )
    return canonical_json_sha256(
        {
            "scene": scene,
            "adapter_id": adapter_id,
            "run_scope": run_scope,
            "method_origin": method_origin,
            "training_scope_sha256": training_sha,
            "endpoint_kind": endpoint_kind,
            "artifacts": artifacts,
            "model_profile": model_profile,
        }
    )


def evaluation_reuse_key(receipt: Mapping[str, Any]) -> str:
    endpoint_key = _require_sha(
        receipt.get("endpoint_reuse_key"), "evaluator.endpoint_reuse_key"
    )
    evaluator_id = _require_nonempty_string(
        receipt.get("evaluator_id"), "evaluator.evaluator_id"
    )
    evaluation_sha = validate_scoped_provenance(
        receipt.get("evaluation_provenance"),
        expected_scope="endpoint_evaluation",
        label="evaluator.evaluation_provenance",
    )
    return canonical_json_sha256(
        {
            "endpoint_reuse_key": endpoint_key,
            "evaluator_id": evaluator_id,
            "evaluation_scope_sha256": evaluation_sha,
        }
    )


def _validate_terminal_status(receipt: Mapping[str, Any], label: str) -> str:
    status = _require_nonempty_string(receipt.get("status"), f"{label}.status")
    if status not in TERMINAL_STATUSES:
        raise ReceiptContractError(
            f"{label}.status must be one of {sorted(TERMINAL_STATUSES)}"
        )
    if status == FAILED:
        failure = _require_mapping(receipt.get("failure"), f"{label}.failure")
        _reject_unknown_keys(
            failure,
            {"code", "message", "signature_sha256", "log_sha256"},
            f"{label}.failure",
        )
        _require_nonempty_string(failure.get("code"), f"{label}.failure.code")
        _require_nonempty_string(failure.get("message"), f"{label}.failure.message")
        for field in ("signature_sha256", "log_sha256"):
            if field in failure:
                _require_sha(failure[field], f"{label}.failure.{field}")
    elif status == UNSUPPORTED:
        unsupported = _require_mapping(
            receipt.get("unsupported"), f"{label}.unsupported"
        )
        _reject_unknown_keys(
            unsupported,
            {"reason", "code", "evidence_sha256"},
            f"{label}.unsupported",
        )
        _require_nonempty_string(
            unsupported.get("reason"), f"{label}.unsupported.reason"
        )
        if "code" in unsupported:
            _require_nonempty_string(unsupported["code"], f"{label}.unsupported.code")
        if "evidence_sha256" in unsupported:
            _require_sha(
                unsupported["evidence_sha256"],
                f"{label}.unsupported.evidence_sha256",
            )
    return status


def _validate_render_profile(value: Any, label: str) -> dict[str, Any]:
    render = _require_mapping(value, label)
    _reject_unknown_keys(
        render,
        {
            "view_count",
            "width",
            "height",
            "warmup_views",
            "total_time_s",
            "ms_per_view",
            "fps",
        },
        label,
    )
    view_count = _require_int(render.get("view_count"), f"{label}.view_count", minimum=1)
    width = _require_int(render.get("width"), f"{label}.width", minimum=1)
    height = _require_int(render.get("height"), f"{label}.height", minimum=1)
    warmup = _require_int(render.get("warmup_views"), f"{label}.warmup_views", minimum=0)
    total = _require_finite(
        render.get("total_time_s"), f"{label}.total_time_s", nonnegative=True
    )
    if total <= 0.0:
        raise ReceiptContractError(f"{label}.total_time_s must be > 0")
    ms_per_view = _require_finite(
        render.get("ms_per_view"), f"{label}.ms_per_view", nonnegative=True
    )
    fps = _require_finite(render.get("fps"), f"{label}.fps", nonnegative=True)
    expected_ms = total * 1000.0 / view_count
    expected_fps = view_count / total
    if not math.isclose(ms_per_view, expected_ms, rel_tol=1e-3, abs_tol=1e-6):
        raise ReceiptContractError(
            f"{label}.ms_per_view is inconsistent with total/view_count"
        )
    if not math.isclose(fps, expected_fps, rel_tol=1e-3, abs_tol=1e-6):
        raise ReceiptContractError(f"{label}.fps is inconsistent with total/view_count")
    return {
        "view_count": view_count,
        "width": width,
        "height": height,
        "warmup_views": warmup,
        "total_time_s": total,
        "ms_per_view": ms_per_view,
        "fps": fps,
    }


def _validate_cost_records(
    records: Any, *, label: str, alias: bool = False
) -> list[dict[str, Any]]:
    if records is None:
        return []
    if not isinstance(records, list):
        raise ReceiptContractError(f"{label} must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for offset, raw in enumerate(records):
        item_label = f"{label}[{offset}]"
        record = _require_mapping(raw, item_label)
        _reject_unknown_keys(
            record,
            {
                "cost_id",
                "cost_class",
                "stage",
                "timing_scope",
                "execution_environment",
                "batch_execution",
                "reported_method",
            },
            item_label,
        )
        cost_id = _require_nonempty_string(record.get("cost_id"), f"{item_label}.cost_id")
        if cost_id in seen:
            raise ReceiptContractError(f"{label} has duplicate cost_id {cost_id!r}")
        seen.add(cost_id)
        stage = _require_nonempty_string(record.get("stage"), f"{item_label}.stage")
        cost_class = _require_nonempty_string(
            record.get("cost_class"), f"{item_label}.cost_class"
        )
        if cost_class not in {"method_specific", "shared_excluded"}:
            raise ReceiptContractError(
                f"{item_label}.cost_class must be method_specific or shared_excluded"
            )
        execution_environment = _require_mapping(
            record.get("execution_environment"), f"{item_label}.execution_environment"
        )
        environment_fields = {
            "device_id",
            "gpu_model",
            "gpu_uuid",
            "cuda_version",
            "driver_version",
            "torch_version",
            "python_version",
            "runtime_id",
        }
        _reject_unknown_keys(
            execution_environment, environment_fields, f"{item_label}.execution_environment"
        )
        normalized_environment = {
            field: _require_nonempty_string(
                execution_environment.get(field),
                f"{item_label}.execution_environment.{field}",
            )
            for field in sorted(environment_fields)
        }
        timing_scope = _require_mapping(
            record.get("timing_scope"), f"{item_label}.timing_scope"
        )
        _reject_unknown_keys(
            timing_scope,
            {"scope_id", "included_operations", "excluded_operations"},
            f"{item_label}.timing_scope",
        )
        scope_id = _require_nonempty_string(
            timing_scope.get("scope_id"), f"{item_label}.timing_scope.scope_id"
        )
        included = _require_string_list(
            timing_scope.get("included_operations"),
            f"{item_label}.timing_scope.included_operations",
            nonempty=True,
        )
        excluded = _require_string_list(
            timing_scope.get("excluded_operations", []),
            f"{item_label}.timing_scope.excluded_operations",
        )
        batch = _require_mapping(
            record.get("batch_execution"), f"{item_label}.batch_execution"
        )
        _reject_unknown_keys(
            batch,
            {
                "batch_id",
                "wall_time_s",
                "gpu_seconds",
                "peak_allocated_vram_mb",
                "peak_reserved_vram_mb",
                "disk_bytes",
                "render_profile",
            },
            f"{item_label}.batch_execution",
        )
        batch_id = _require_nonempty_string(
            batch.get("batch_id"), f"{item_label}.batch_execution.batch_id"
        )
        batch_wall = _require_finite(
            batch.get("wall_time_s"),
            f"{item_label}.batch_execution.wall_time_s",
            nonnegative=True,
        )
        batch_gpu = _require_finite(
            batch.get("gpu_seconds"),
            f"{item_label}.batch_execution.gpu_seconds",
            nonnegative=True,
        )
        batch_allocated_vram = _require_finite(
            batch.get("peak_allocated_vram_mb"),
            f"{item_label}.batch_execution.peak_allocated_vram_mb",
            nonnegative=True,
        )
        batch_reserved_vram = _require_finite(
            batch.get("peak_reserved_vram_mb"),
            f"{item_label}.batch_execution.peak_reserved_vram_mb",
            nonnegative=True,
        )
        batch_disk = _require_int(
            batch.get("disk_bytes"),
            f"{item_label}.batch_execution.disk_bytes",
            minimum=0,
        )
        reported = _require_mapping(
            record.get("reported_method"), f"{item_label}.reported_method"
        )
        _reject_unknown_keys(
            reported,
            {
                "accounting",
                "source_cost_id",
                "wall_time_s",
                "gpu_seconds",
                "peak_allocated_vram_mb",
                "peak_reserved_vram_mb",
                "disk_bytes",
                "exclusion_reason",
                "render_profile",
            },
            f"{item_label}.reported_method",
        )
        accounting = _require_nonempty_string(
            reported.get("accounting"),
            f"{item_label}.reported_method.accounting",
        )
        if accounting not in {"measured", "inherited_exact_endpoint", "excluded_shared"}:
            raise ReceiptContractError(
                f"{item_label}.reported_method.accounting must be measured or "
                "inherited_exact_endpoint, or excluded_shared"
            )
        method_wall = _require_finite(
            reported.get("wall_time_s"),
            f"{item_label}.reported_method.wall_time_s",
            nonnegative=True,
        )
        method_gpu = _require_finite(
            reported.get("gpu_seconds"),
            f"{item_label}.reported_method.gpu_seconds",
            nonnegative=True,
        )
        method_allocated_vram = _require_finite(
            reported.get("peak_allocated_vram_mb"),
            f"{item_label}.reported_method.peak_allocated_vram_mb",
            nonnegative=True,
        )
        method_reserved_vram = _require_finite(
            reported.get("peak_reserved_vram_mb"),
            f"{item_label}.reported_method.peak_reserved_vram_mb",
            nonnegative=True,
        )
        method_disk = _require_int(
            reported.get("disk_bytes"),
            f"{item_label}.reported_method.disk_bytes",
            minimum=0,
        )
        source_cost_id = reported.get("source_cost_id")
        if accounting == "inherited_exact_endpoint":
            source_cost_id = _require_nonempty_string(
                source_cost_id, f"{item_label}.reported_method.source_cost_id"
            )
        elif source_cost_id is not None:
            raise ReceiptContractError(
                f"{item_label}.reported_method.source_cost_id is only valid for inherited cost"
            )
        if alias and (cost_class != "method_specific" or accounting != "inherited_exact_endpoint"):
            raise ReceiptContractError(
                "alias reported-method cost must inherit the exact source endpoint"
            )
        if not alias and cost_class == "method_specific" and accounting != "measured":
            raise ReceiptContractError(
                "non-alias reported-method cost must use measured accounting"
            )
        if cost_class == "shared_excluded":
            if alias or accounting != "excluded_shared":
                raise ReceiptContractError(
                    "shared_excluded cost must use excluded_shared accounting on a non-alias receipt"
                )
            _require_nonempty_string(
                reported.get("exclusion_reason"),
                f"{item_label}.reported_method.exclusion_reason",
            )
            if any(
                value != 0
                for value in (
                    method_wall,
                    method_gpu,
                    method_allocated_vram,
                    method_reserved_vram,
                    method_disk,
                )
            ):
                raise ReceiptContractError(
                    "shared_excluded cost must contribute zero to reported method cost"
                )
        elif "exclusion_reason" in reported:
            raise ReceiptContractError(
                "exclusion_reason is only valid for shared_excluded costs"
            )
        if alias and any(
            value != 0
            for value in (
                batch_wall,
                batch_gpu,
                batch_allocated_vram,
                batch_reserved_vram,
                batch_disk,
            )
        ):
            raise ReceiptContractError(
                "alias incremental batch execution must be zero for every cost field"
            )
        batch_render_profile = None
        method_render_profile = None
        if stage == "rendering":
            if alias:
                if "render_profile" in batch:
                    raise ReceiptContractError(
                        f"{item_label}.batch_execution.render_profile must be absent "
                        "for a zero-execution alias"
                    )
            else:
                batch_render_profile = _validate_render_profile(
                    batch.get("render_profile"),
                    f"{item_label}.batch_execution.render_profile",
                )
            method_render_profile = _validate_render_profile(
                reported.get("render_profile"),
                f"{item_label}.reported_method.render_profile",
            )
        elif "render_profile" in batch or "render_profile" in reported:
            raise ReceiptContractError(
                f"{item_label} render_profile is only valid for stage='rendering'"
            )
        normalized_reported = {
            "accounting": accounting,
            "wall_time_s": method_wall,
            "gpu_seconds": method_gpu,
            "peak_allocated_vram_mb": method_allocated_vram,
            "peak_reserved_vram_mb": method_reserved_vram,
            "disk_bytes": method_disk,
        }
        if source_cost_id is not None:
            normalized_reported["source_cost_id"] = source_cost_id
        if "exclusion_reason" in reported:
            normalized_reported["exclusion_reason"] = reported["exclusion_reason"]
        if method_render_profile is not None:
            normalized_reported["render_profile"] = method_render_profile
        normalized_batch = {
            "batch_id": batch_id,
            "wall_time_s": batch_wall,
            "gpu_seconds": batch_gpu,
            "peak_allocated_vram_mb": batch_allocated_vram,
            "peak_reserved_vram_mb": batch_reserved_vram,
            "disk_bytes": batch_disk,
        }
        if batch_render_profile is not None:
            normalized_batch["render_profile"] = batch_render_profile
        normalized_record: dict[str, Any] = {
            "cost_id": cost_id,
            "cost_class": cost_class,
            "stage": stage,
            "execution_environment": normalized_environment,
            "timing_scope": {
                "scope_id": scope_id,
                "included_operations": included,
                "excluded_operations": excluded,
            },
            "batch_execution": normalized_batch,
            "reported_method": normalized_reported,
        }
        normalized.append(
            normalized_record
        )
    return normalized


def validate_adapter_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(_require_mapping(receipt, "adapter receipt")))
    _reject_unknown_keys(
        payload,
        {
            "schema",
            "status",
            "scene",
            "reported_method_id",
            "adapter_id",
            "run_scope",
            "method_origin",
            "training_provenance",
            "endpoint",
            "failure",
            "unsupported",
            "cost_records",
            "receipt_sha256",
        },
        "adapter receipt",
    )
    if payload.get("schema") != ADAPTER_SCHEMA:
        raise ReceiptContractError(f"adapter receipt schema must be {ADAPTER_SCHEMA}")
    _validate_self_hash(payload, "receipt_sha256", "adapter receipt")
    status = _validate_terminal_status(payload, "adapter receipt")
    _require_nonempty_string(payload.get("scene"), "adapter receipt.scene")
    _require_nonempty_string(
        payload.get("reported_method_id"), "adapter receipt.reported_method_id"
    )
    _require_nonempty_string(payload.get("adapter_id"), "adapter receipt.adapter_id")
    run_scope = _require_nonempty_string(
        payload.get("run_scope"), "adapter receipt.run_scope"
    )
    if run_scope not in {"formal", "smoke", "preliminary"}:
        raise ReceiptContractError("adapter receipt.run_scope is invalid")
    _validate_method_origin(payload.get("method_origin"), "adapter receipt.method_origin")
    validate_scoped_provenance(
        payload.get("training_provenance"),
        expected_scope="training_endpoint",
        label="adapter receipt.training_provenance",
    )
    normalized_costs = _validate_cost_records(
        payload.get("cost_records", []), label="adapter receipt.cost_records"
    )
    if status == SUCCEEDED:
        if "failure" in payload or "unsupported" in payload:
            raise ReceiptContractError(
                "successful adapter must not publish failure/unsupported details"
            )
        endpoint = _require_mapping(payload.get("endpoint"), "adapter receipt.endpoint")
        _reject_unknown_keys(
            endpoint,
            {"endpoint_id", "endpoint_kind", "artifacts", "model_profile", "reuse_key"},
            "adapter receipt.endpoint",
        )
        _require_nonempty_string(endpoint.get("endpoint_id"), "adapter receipt.endpoint.endpoint_id")
        _require_nonempty_string(
            endpoint.get("endpoint_kind"), "adapter receipt.endpoint.endpoint_kind"
        )
        _artifact_material(endpoint.get("artifacts"), "adapter receipt.endpoint.artifacts")
        _validate_model_profile(
            endpoint.get("model_profile"), "adapter receipt.endpoint.model_profile"
        )
        declared = _require_sha(
            endpoint.get("reuse_key"), "adapter receipt.endpoint.reuse_key"
        )
        actual = endpoint_reuse_key(payload)
        if declared != actual:
            raise ReceiptContractError(
                f"adapter endpoint reuse key mismatch: declared={declared} actual={actual}"
            )
        if run_scope == "formal":
            training_costs = [
                record
                for record in normalized_costs
                if record["cost_class"] == "method_specific"
                and record["stage"] == "training"
            ]
            if not training_costs:
                raise ReceiptContractError(
                    "successful formal adapter requires method-specific training cost"
                )
            for record in training_costs:
                execution = record["batch_execution"]
                if (
                    execution["wall_time_s"] <= 0
                    or execution["peak_allocated_vram_mb"] <= 0
                    or execution["peak_reserved_vram_mb"] <= 0
                    or execution["peak_reserved_vram_mb"]
                    < execution["peak_allocated_vram_mb"]
                ):
                    raise ReceiptContractError(
                        "formal training cost requires positive wall time and valid "
                        "allocated/reserved peak VRAM"
                    )
    elif payload.get("endpoint") not in (None, {}):
        raise ReceiptContractError("failed/unsupported adapter must not publish an endpoint")
    if status == FAILED and "unsupported" in payload:
        raise ReceiptContractError("failed adapter must not publish unsupported details")
    if status == UNSUPPORTED and "failure" in payload:
        raise ReceiptContractError("unsupported adapter must not publish failure details")
    return payload


def finalize_adapter_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(receipt))
    if payload.get("status") == SUCCEEDED:
        endpoint = copy.deepcopy(dict(_require_mapping(payload.get("endpoint"), "adapter.endpoint")))
        endpoint["reuse_key"] = endpoint_reuse_key({**payload, "endpoint": endpoint})
        payload["endpoint"] = endpoint
    payload = _finalize_self_hash(payload, "receipt_sha256")
    return validate_adapter_receipt(payload)


def _validate_metric_scope(scope: Any, label: str) -> dict[str, Any]:
    payload = _require_mapping(scope, label)
    _reject_unknown_keys(
        payload,
        {
            "metric_definition_id",
            "protocol_sha256",
            "collection_sha256",
            "split_sha256",
            "camera_sha256",
            "split",
            "aggregation",
            "support_sha256",
            "view_count",
            "sample_count",
            "resolution",
            "depth_definition",
            "threshold_m",
        },
        label,
    )
    split = _require_nonempty_string(payload.get("split"), f"{label}.split")
    if split not in {"train", "guard", "test", "train_guard"}:
        raise ReceiptContractError(
            f"{label}.split must be train, guard, test, or train_guard"
        )
    resolution = _require_mapping(payload.get("resolution"), f"{label}.resolution")
    _reject_unknown_keys(resolution, {"width", "height"}, f"{label}.resolution")
    result: dict[str, Any] = {
        "metric_definition_id": _require_nonempty_string(
            payload.get("metric_definition_id"), f"{label}.metric_definition_id"
        ),
        "protocol_sha256": _require_sha(
            payload.get("protocol_sha256"), f"{label}.protocol_sha256"
        ),
        "collection_sha256": _require_sha(
            payload.get("collection_sha256"), f"{label}.collection_sha256"
        ),
        "split_sha256": _require_sha(
            payload.get("split_sha256"), f"{label}.split_sha256"
        ),
        "camera_sha256": _require_sha(
            payload.get("camera_sha256"), f"{label}.camera_sha256"
        ),
        "split": split,
        "aggregation": _require_nonempty_string(
            payload.get("aggregation"), f"{label}.aggregation"
        ),
        "support_sha256": _require_sha(
            payload.get("support_sha256"), f"{label}.support_sha256"
        ),
        "view_count": _require_int(
            payload.get("view_count"), f"{label}.view_count", minimum=1
        ),
        "sample_count": _require_int(
            payload.get("sample_count"), f"{label}.sample_count", minimum=1
        ),
        "resolution": {
            "width": _require_int(
                resolution.get("width"), f"{label}.resolution.width", minimum=1
            ),
            "height": _require_int(
                resolution.get("height"), f"{label}.resolution.height", minimum=1
            ),
        },
    }
    if "depth_definition" in payload:
        depth_definition = _require_nonempty_string(
            payload["depth_definition"], f"{label}.depth_definition"
        )
        if depth_definition not in FORMAL_DEPTH_DEFINITIONS:
            raise ReceiptContractError(
                f"{label}.depth_definition is outside the frozen formal definitions"
            )
        result["depth_definition"] = depth_definition
    if "threshold_m" in payload:
        threshold = _require_finite(
            payload["threshold_m"], f"{label}.threshold_m", nonnegative=True
        )
        if threshold not in FORMAL_GEOMETRY_THRESHOLDS_M:
            raise ReceiptContractError(
                f"{label}.threshold_m is outside the frozen eight thresholds"
            )
        result["threshold_m"] = threshold
    return result


def metric_pairing_key(evaluation_scope: Mapping[str, Any]) -> str:
    """Bind a paired metric to its exact definition, split, support and resolution."""

    return canonical_json_sha256(
        _validate_metric_scope(evaluation_scope, "metric evaluation_scope")
    )


def _validate_metrics(metrics: Any, label: str) -> dict[str, dict[str, Any]]:
    payload = _require_mapping(metrics, label)
    result: dict[str, dict[str, Any]] = {}
    for metric_id, raw in payload.items():
        name = _require_nonempty_string(metric_id, f"{label} metric id")
        if name != metric_id:
            raise ReceiptContractError(f"{label} metric IDs must not contain edge whitespace")
        metric_label = f"{label}.{name}"
        metric = _require_mapping(raw, metric_label)
        _reject_unknown_keys(
            metric,
            {
                "status",
                "value",
                "unit",
                "pairing_key",
                "evaluation_scope",
                "failure",
                "unsupported",
            },
            metric_label,
        )
        status = _validate_terminal_status(metric, metric_label)
        unit = _require_nonempty_string(metric.get("unit"), f"{metric_label}.unit")
        scope = _validate_metric_scope(
            metric.get("evaluation_scope"), f"{metric_label}.evaluation_scope"
        )
        pairing_key = _require_sha(metric.get("pairing_key"), f"{metric_label}.pairing_key")
        expected_pairing = canonical_json_sha256(scope)
        if pairing_key != expected_pairing:
            raise ReceiptContractError(
                f"{metric_label}.pairing_key mismatch: "
                f"declared={pairing_key} actual={expected_pairing}"
            )
        normalized: dict[str, Any] = {
            "status": status,
            "unit": unit,
            "pairing_key": pairing_key,
            "evaluation_scope": scope,
        }
        if status == SUCCEEDED:
            if "failure" in metric or "unsupported" in metric:
                raise ReceiptContractError(
                    f"{metric_label} succeeded but contains failure/unsupported details"
                )
            normalized["value"] = _require_finite(
                metric.get("value"), f"{metric_label}.value"
            )
        else:
            if "value" in metric:
                raise ReceiptContractError(
                    f"{metric_label} failed/unsupported and must not publish a value"
                )
            if status == FAILED:
                if "unsupported" in metric:
                    raise ReceiptContractError(
                        f"{metric_label} failed but contains unsupported details"
                    )
                normalized["failure"] = copy.deepcopy(metric["failure"])
            else:
                if "failure" in metric:
                    raise ReceiptContractError(
                        f"{metric_label} unsupported but contains failure details"
                    )
                normalized["unsupported"] = copy.deepcopy(metric["unsupported"])
        result[name] = normalized
    return result


def validate_evaluator_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(_require_mapping(receipt, "evaluator receipt")))
    _reject_unknown_keys(
        payload,
        {
            "schema",
            "status",
            "scene",
            "reported_method_id",
            "evaluator_id",
            "endpoint_reuse_key",
            "evaluation_provenance",
            "evaluation_reuse_key",
            "metrics",
            "failure",
            "unsupported",
            "cost_records",
            "receipt_sha256",
        },
        "evaluator receipt",
    )
    if payload.get("schema") != EVALUATOR_SCHEMA:
        raise ReceiptContractError(f"evaluator receipt schema must be {EVALUATOR_SCHEMA}")
    _validate_self_hash(payload, "receipt_sha256", "evaluator receipt")
    status = _validate_terminal_status(payload, "evaluator receipt")
    _require_nonempty_string(payload.get("scene"), "evaluator receipt.scene")
    _require_nonempty_string(
        payload.get("reported_method_id"), "evaluator receipt.reported_method_id"
    )
    _require_nonempty_string(payload.get("evaluator_id"), "evaluator receipt.evaluator_id")
    _require_sha(payload.get("endpoint_reuse_key"), "evaluator receipt.endpoint_reuse_key")
    validate_scoped_provenance(
        payload.get("evaluation_provenance"),
        expected_scope="endpoint_evaluation",
        label="evaluator receipt.evaluation_provenance",
    )
    _validate_cost_records(payload.get("cost_records", []), label="evaluator receipt.cost_records")
    declared = _require_sha(
        payload.get("evaluation_reuse_key"), "evaluator receipt.evaluation_reuse_key"
    )
    actual = evaluation_reuse_key(payload)
    if declared != actual:
        raise ReceiptContractError(
            f"evaluation reuse key mismatch: declared={declared} actual={actual}"
        )
    metrics = payload.get("metrics", {})
    if status == SUCCEEDED:
        if "failure" in payload or "unsupported" in payload:
            raise ReceiptContractError(
                "successful evaluator must not publish top-level failure/unsupported details"
            )
        if not _validate_metrics(metrics, "evaluator receipt.metrics"):
            raise ReceiptContractError("successful evaluator receipt must contain metrics")
    elif metrics not in ({}, None):
        raise ReceiptContractError("failed/unsupported evaluator must not publish metrics")
    if status == FAILED and "unsupported" in payload:
        raise ReceiptContractError("failed evaluator must not publish unsupported details")
    if status == UNSUPPORTED and "failure" in payload:
        raise ReceiptContractError("unsupported evaluator must not publish failure details")
    return payload


def finalize_evaluator_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(receipt))
    payload["evaluation_reuse_key"] = evaluation_reuse_key(payload)
    payload = _finalize_self_hash(payload, "receipt_sha256")
    return validate_evaluator_receipt(payload)


def validate_alias_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(_require_mapping(receipt, "alias receipt")))
    _reject_unknown_keys(
        payload,
        {
            "schema",
            "status",
            "scene",
            "source_method_id",
            "reported_method_id",
            "source_endpoint_reuse_key",
            "reuse",
            "zero_modification",
            "cost_records",
            "receipt_sha256",
        },
        "alias receipt",
    )
    if payload.get("schema") != ALIAS_SCHEMA:
        raise ReceiptContractError(f"alias receipt schema must be {ALIAS_SCHEMA}")
    _validate_self_hash(payload, "receipt_sha256", "alias receipt")
    if payload.get("status") != SUCCEEDED:
        raise ReceiptContractError("alias receipt must have status SUCCEEDED")
    _require_nonempty_string(payload.get("scene"), "alias receipt.scene")
    source = _require_nonempty_string(
        payload.get("source_method_id"), "alias receipt.source_method_id"
    )
    alias = _require_nonempty_string(
        payload.get("reported_method_id"), "alias receipt.reported_method_id"
    )
    if source == alias:
        raise ReceiptContractError("alias source and reported method IDs must differ")
    _require_sha(payload.get("source_endpoint_reuse_key"), "alias receipt.source_endpoint_reuse_key")
    reuse = _require_mapping(payload.get("reuse"), "alias receipt.reuse")
    _reject_unknown_keys(reuse, {"exact_endpoint", "roles"}, "alias receipt.reuse")
    if reuse.get("exact_endpoint") is not True:
        raise ReceiptContractError("alias receipt must certify exact_endpoint=true")
    _require_string_list(reuse.get("roles"), "alias receipt.reuse.roles", nonempty=True)
    zero_modification = _require_mapping(
        payload.get("zero_modification"), "alias receipt.zero_modification"
    )
    _reject_unknown_keys(
        zero_modification,
        {"manifest_sha256", "modified_count", "total_gaussian_count", "checked_fields"},
        "alias receipt.zero_modification",
    )
    _require_sha(
        zero_modification.get("manifest_sha256"),
        "alias receipt.zero_modification.manifest_sha256",
    )
    if _require_int(
        zero_modification.get("modified_count"),
        "alias receipt.zero_modification.modified_count",
        minimum=0,
    ) != 0:
        raise ReceiptContractError("exact alias requires modified_count=0")
    _require_int(
        zero_modification.get("total_gaussian_count"),
        "alias receipt.zero_modification.total_gaussian_count",
        minimum=1,
    )
    checked_fields = _require_string_list(
        zero_modification.get("checked_fields"),
        "alias receipt.zero_modification.checked_fields",
        nonempty=True,
    )
    required_alias_fields = {"xyz", "scaling", "rotation", "opacity", "topology"}
    if not required_alias_fields.issubset(checked_fields):
        raise ReceiptContractError(
            "exact alias zero-modification manifest must check geometry/opacity/topology"
        )
    _validate_cost_records(
        payload.get("cost_records", []), label="alias receipt.cost_records", alias=True
    )
    return payload


def finalize_alias_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    payload = _finalize_self_hash(receipt, "receipt_sha256")
    return validate_alias_receipt(payload)


def validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    schema = receipt.get("schema") if isinstance(receipt, Mapping) else None
    if schema == ADAPTER_SCHEMA:
        return validate_adapter_receipt(receipt)
    if schema == EVALUATOR_SCHEMA:
        return validate_evaluator_receipt(receipt)
    if schema == ALIAS_SCHEMA:
        return validate_alias_receipt(receipt)
    raise ReceiptContractError(f"unsupported receipt schema: {schema!r}")


def classify_reuse(
    old_adapter: Mapping[str, Any],
    new_adapter: Mapping[str, Any],
    old_evaluator: Mapping[str, Any] | None = None,
    new_evaluator: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old_a = validate_adapter_receipt(old_adapter)
    new_a = validate_adapter_receipt(new_adapter)
    if old_a["status"] != SUCCEEDED or new_a["status"] != SUCCEEDED:
        return {
            "endpoint_reusable": False,
            "evaluation_reusable": False,
            "action": "run_or_recover_training",
        }
    endpoint_same = old_a["endpoint"]["reuse_key"] == new_a["endpoint"]["reuse_key"]
    if not endpoint_same:
        return {
            "endpoint_reusable": False,
            "evaluation_reusable": False,
            "action": "retrain_and_reevaluate",
        }
    if old_evaluator is None or new_evaluator is None:
        return {
            "endpoint_reusable": True,
            "evaluation_reusable": False,
            "action": "reevaluate_only",
        }
    old_e = validate_evaluator_receipt(old_evaluator)
    new_e = validate_evaluator_receipt(new_evaluator)
    evaluation_same = (
        old_e["endpoint_reuse_key"] == new_e["endpoint_reuse_key"]
        and old_e["evaluation_reuse_key"] == new_e["evaluation_reuse_key"]
        and old_e["status"] == SUCCEEDED
        and new_e["status"] == SUCCEEDED
    )
    return {
        "endpoint_reusable": True,
        "evaluation_reusable": evaluation_same,
        "action": "reuse_endpoint_and_evaluation" if evaluation_same else "reevaluate_only",
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReceiptContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReceiptContractError(f"JSON root must be an object: {path}")
    return payload


def _load_bound_receipts(
    identities: Any, *, base: Path, label: str, expected_schema: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if not isinstance(identities, list):
        raise ReceiptContractError(f"study.{label} must be a list")
    result = []
    for offset, raw in enumerate(identities):
        identity = _require_mapping(raw, f"study.{label}[{offset}]")
        _reject_unknown_keys(
            identity,
            {"path", "sha256", "size_bytes"},
            f"study.{label}[{offset}]",
        )
        path_text = _require_nonempty_string(
            identity.get("path"), f"study.{label}[{offset}].path"
        )
        path = Path(path_text)
        if not path.is_absolute():
            path = (base / path).resolve()
        if not path.is_file():
            raise ReceiptContractError(f"receipt file is missing: {path}")
        observed = sha256_file(path)
        expected = _require_sha(identity.get("sha256"), f"study.{label}[{offset}].sha256")
        if "size_bytes" in identity:
            expected_size = _require_int(
                identity["size_bytes"],
                f"study.{label}[{offset}].size_bytes",
                minimum=0,
            )
            if expected_size != path.stat().st_size:
                raise ReceiptContractError(
                    f"receipt file size mismatch: {path} "
                    f"expected={expected_size} actual={path.stat().st_size}"
                )
        if observed != expected:
            raise ReceiptContractError(
                f"receipt file SHA mismatch: {path} expected={expected} actual={observed}"
            )
        receipt = validate_receipt(_load_json(path))
        if receipt["schema"] != expected_schema:
            raise ReceiptContractError(
                f"study.{label}[{offset}] binds {receipt['schema']!r}; "
                f"expected {expected_schema!r}"
            )
        result.append(
            (
                receipt,
                {"path": str(path), "sha256": observed, "size_bytes": int(path.stat().st_size)},
            )
        )
    return result


def finalize_study_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = _finalize_self_hash(manifest, "manifest_sha256")
    validate_study_manifest(payload)
    return payload


def validate_study_manifest(
    manifest: Mapping[str, Any], *, base: Path | None = None
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(_require_mapping(manifest, "study manifest")))
    _reject_unknown_keys(
        payload,
        {
            "schema",
            "protocol",
            "matrix_contract",
            "geometry_contract",
            "required_metric_catalog",
            "scenes",
            "reported_methods",
            "metrics",
            "comparisons",
            "adapter_receipts",
            "evaluator_receipts",
            "alias_receipts",
            "manifest_sha256",
        },
        "study manifest",
    )
    if payload.get("schema") != STUDY_SCHEMA:
        raise ReceiptContractError(f"study manifest schema must be {STUDY_SCHEMA}")
    _validate_self_hash(payload, "manifest_sha256", "study manifest")
    protocol = _require_mapping(payload.get("protocol"), "study.protocol")
    _reject_unknown_keys(
        protocol,
        {
            "path",
            "sha256",
            "protocol_id",
            "protocol_version",
            "collection_hash",
            "collection_split_hash",
            "formal_rule_hash",
        },
        "study.protocol",
    )
    protocol_path_text = _require_nonempty_string(
        protocol.get("path"), "study.protocol.path"
    )
    protocol_file_sha = _require_sha(protocol.get("sha256"), "study.protocol.sha256")
    _require_nonempty_string(protocol.get("protocol_id"), "study.protocol.protocol_id")
    _require_nonempty_string(
        protocol.get("protocol_version"), "study.protocol.protocol_version"
    )
    _require_sha(protocol.get("collection_hash"), "study.protocol.collection_hash")
    _require_sha(
        protocol.get("collection_split_hash"), "study.protocol.collection_split_hash"
    )
    _require_sha(protocol.get("formal_rule_hash"), "study.protocol.formal_rule_hash")
    formal_protocol = None
    if base is not None:
        protocol_path = Path(protocol_path_text)
        if not protocol_path.is_absolute():
            protocol_path = (base / protocol_path).resolve()
        if not protocol_path.is_file():
            raise ReceiptContractError(f"formal protocol file is missing: {protocol_path}")
        actual_protocol_sha = sha256_file(protocol_path)
        if actual_protocol_sha != protocol_file_sha:
            raise ReceiptContractError(
                f"formal protocol SHA mismatch: expected={protocol_file_sha} "
                f"actual={actual_protocol_sha}"
            )
        formal_protocol = _load_json(protocol_path)
        observed_binding = {
            "protocol_id": formal_protocol.get("protocol_id"),
            "protocol_version": formal_protocol.get("protocol_version"),
            "collection_hash": formal_protocol.get("collection_hash"),
            "collection_split_hash": formal_protocol.get("collection", {}).get(
                "collection_split_hash"
            ),
            "formal_rule_hash": formal_protocol.get("collection", {}).get(
                "formal_rule_hash"
            ),
        }
        declared_binding = {key: protocol[key] for key in observed_binding}
        if observed_binding != declared_binding:
            raise ReceiptContractError(
                f"study/formal protocol binding mismatch: "
                f"declared={declared_binding} observed={observed_binding}"
            )
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ReceiptContractError("study.scenes must be a nonempty list")
    scene_ids = [_require_nonempty_string(scene, "study scene") for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        raise ReceiptContractError("study scenes must be unique")
    matrix = _require_mapping(payload.get("matrix_contract"), "study.matrix_contract")
    _reject_unknown_keys(
        matrix,
        {
            "matrix_id",
            "method_ids",
            "scene_ids",
            "complete_cross_product",
            "expected_cell_count",
        },
        "study.matrix_contract",
    )
    _require_nonempty_string(matrix.get("matrix_id"), "study.matrix_contract.matrix_id")
    matrix_methods = _require_string_list(
        matrix.get("method_ids"), "study.matrix_contract.method_ids", nonempty=True
    )
    matrix_scenes = _require_string_list(
        matrix.get("scene_ids"), "study.matrix_contract.scene_ids", nonempty=True
    )
    if matrix.get("complete_cross_product") is not True:
        raise ReceiptContractError("study matrix must declare complete_cross_product=true")
    expected_cells = _require_int(
        matrix.get("expected_cell_count"),
        "study.matrix_contract.expected_cell_count",
        minimum=1,
    )
    if expected_cells != len(matrix_methods) * len(matrix_scenes):
        raise ReceiptContractError("study matrix expected_cell_count is not the cross product")
    if not set(matrix_scenes).issubset(scene_ids):
        raise ReceiptContractError("study matrix contains scenes outside study.scenes")

    geometry = _require_mapping(
        payload.get("geometry_contract"), "study.geometry_contract"
    )
    _reject_unknown_keys(
        geometry,
        {"depth_definitions", "thresholds_m"},
        "study.geometry_contract",
    )
    depth_definitions = _require_string_list(
        geometry.get("depth_definitions"),
        "study.geometry_contract.depth_definitions",
        nonempty=True,
    )
    thresholds_raw = geometry.get("thresholds_m")
    if not isinstance(thresholds_raw, list):
        raise ReceiptContractError("study.geometry_contract.thresholds_m must be a list")
    thresholds = [
        _require_finite(value, "study.geometry_contract threshold", nonnegative=True)
        for value in thresholds_raw
    ]
    if tuple(depth_definitions) != FORMAL_DEPTH_DEFINITIONS:
        raise ReceiptContractError(
            f"depth definitions must equal the frozen formal order {FORMAL_DEPTH_DEFINITIONS}"
        )
    if tuple(thresholds) != FORMAL_GEOMETRY_THRESHOLDS_M:
        raise ReceiptContractError(
            f"geometry thresholds must equal {FORMAL_GEOMETRY_THRESHOLDS_M}"
        )

    required_metric_catalog = _require_mapping(
        payload.get("required_metric_catalog"), "study.required_metric_catalog"
    )
    for family, values in required_metric_catalog.items():
        _require_nonempty_string(family, "study.required_metric_catalog family")
        _require_string_list(
            values, f"study.required_metric_catalog.{family}", nonempty=True
        )
    if formal_protocol is not None:
        if matrix.get("matrix_id") == "phase1_internal_representative":
            phase1 = next(
                phase
                for phase in formal_protocol.get("execution_phases", [])
                if phase.get("phase_id") == 1
            )
            if matrix_methods != phase1.get("methods") or matrix_scenes != phase1.get("scenes"):
                raise ReceiptContractError(
                    "formal Phase-1 matrix must be the exact three-internal x six-representative cross product"
                )
        if depth_definitions != formal_protocol.get("depth_definitions"):
            raise ReceiptContractError("study depth definitions differ from formal protocol")
        if thresholds != [float(value) for value in formal_protocol.get("geometry_thresholds_m", [])]:
            raise ReceiptContractError("study geometry thresholds differ from formal protocol")
        if dict(required_metric_catalog) != formal_protocol.get("common_metrics"):
            raise ReceiptContractError("study required metric catalog differs from formal protocol")
    methods = payload.get("reported_methods")
    if not isinstance(methods, list) or not methods:
        raise ReceiptContractError("study.reported_methods must be a nonempty list")
    method_ids = []
    for offset, raw in enumerate(methods):
        method = _require_mapping(raw, f"study.reported_methods[{offset}]")
        _reject_unknown_keys(
            method,
            {"method_id", "display_name", "expected_scenes"},
            f"study.reported_methods[{offset}]",
        )
        method_ids.append(
            _require_nonempty_string(
                method.get("method_id"), f"study.reported_methods[{offset}].method_id"
            )
        )
        _require_nonempty_string(
            method.get("display_name"), f"study.reported_methods[{offset}].display_name"
        )
        expected_scenes = _require_string_list(
            method.get("expected_scenes"),
            f"study.reported_methods[{offset}].expected_scenes",
            nonempty=True,
        )
        unknown_scenes = sorted(set(expected_scenes) - set(scene_ids))
        if unknown_scenes:
            raise ReceiptContractError(
                f"study.reported_methods[{offset}] has unknown expected scenes: {unknown_scenes}"
            )
    if len(method_ids) != len(set(method_ids)):
        raise ReceiptContractError("study reported method IDs must be unique")
    if set(matrix_methods) != set(method_ids):
        raise ReceiptContractError(
            "study reported methods must exactly equal matrix_contract.method_ids"
        )
    expected_by_method = {
        str(method["method_id"]): list(method["expected_scenes"]) for method in methods
    }
    for method_id in matrix_methods:
        if not set(matrix_scenes).issubset(expected_by_method[method_id]):
            raise ReceiptContractError(
                f"method {method_id!r} does not cover every required matrix scene"
            )
    metrics = payload.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ReceiptContractError("study.metrics must be a nonempty list")
    metric_ids = []
    for offset, raw in enumerate(metrics):
        metric = _require_mapping(raw, f"study.metrics[{offset}]")
        _reject_unknown_keys(
            metric,
            {
                "metric_id",
                "evaluator_id",
                "direction",
                "metric_definition_id",
                "unit",
                "split",
                "aggregation",
                "macro_aggregation",
                "tie_tolerance",
                "depth_definition",
                "threshold_m",
            },
            f"study.metrics[{offset}]",
        )
        metric_ids.append(
            _require_nonempty_string(metric.get("metric_id"), f"study.metrics[{offset}].metric_id")
        )
        _require_nonempty_string(
            metric.get("evaluator_id"), f"study.metrics[{offset}].evaluator_id"
        )
        _require_nonempty_string(
            metric.get("metric_definition_id"),
            f"study.metrics[{offset}].metric_definition_id",
        )
        _require_nonempty_string(metric.get("unit"), f"study.metrics[{offset}].unit")
        split = _require_nonempty_string(
            metric.get("split"), f"study.metrics[{offset}].split"
        )
        if split not in {"train", "guard", "test", "train_guard"}:
            raise ReceiptContractError(
                f"study.metrics[{offset}].split is invalid"
            )
        _require_nonempty_string(
            metric.get("aggregation"), f"study.metrics[{offset}].aggregation"
        )
        if metric.get("macro_aggregation") not in {"mean", "median"}:
            raise ReceiptContractError(
                f"study.metrics[{offset}].macro_aggregation must be mean or median"
            )
        _require_finite(
            metric.get("tie_tolerance"),
            f"study.metrics[{offset}].tie_tolerance",
            nonnegative=True,
        )
        if "depth_definition" in metric:
            if metric["depth_definition"] not in FORMAL_DEPTH_DEFINITIONS:
                raise ReceiptContractError(
                    f"study.metrics[{offset}].depth_definition is not frozen-formal"
                )
        if "threshold_m" in metric:
            threshold = _require_finite(
                metric["threshold_m"],
                f"study.metrics[{offset}].threshold_m",
                nonnegative=True,
            )
            if threshold not in FORMAL_GEOMETRY_THRESHOLDS_M:
                raise ReceiptContractError(
                    f"study.metrics[{offset}].threshold_m is not frozen-formal"
                )
        if metric.get("direction") not in ("min", "max"):
            raise ReceiptContractError("metric direction must be min or max")
    if len(metric_ids) != len(set(metric_ids)):
        raise ReceiptContractError("study metric IDs must be unique")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, list):
        raise ReceiptContractError("study.comparisons must be a list")
    comparison_ids = set()
    for offset, raw in enumerate(comparisons):
        comparison = _require_mapping(raw, f"study.comparisons[{offset}]")
        _reject_unknown_keys(
            comparison,
            {"comparison_id", "method_ids", "baseline_method_id"},
            f"study.comparisons[{offset}]",
        )
        comparison_id = _require_nonempty_string(
            comparison.get("comparison_id"), f"study.comparisons[{offset}].comparison_id"
        )
        if comparison_id in comparison_ids:
            raise ReceiptContractError(f"duplicate comparison ID {comparison_id!r}")
        comparison_ids.add(comparison_id)
        group_methods = _require_string_list(
            comparison.get("method_ids"),
            f"study.comparisons[{offset}].method_ids",
            nonempty=True,
        )
        if len(group_methods) < 2:
            raise ReceiptContractError("comparison groups require at least two methods")
        if set(group_methods) != set(matrix_methods):
            raise ReceiptContractError(
                "metric comparison groups may not shrink the declared matrix method set"
            )
        for method in group_methods:
            if method not in method_ids:
                raise ReceiptContractError(f"comparison references unknown method {method!r}")
        baseline = _require_nonempty_string(
            comparison.get("baseline_method_id"),
            f"study.comparisons[{offset}].baseline_method_id",
        )
        if baseline not in group_methods:
            raise ReceiptContractError("comparison baseline_method_id must be in method_ids")
    for field in ("adapter_receipts", "evaluator_receipts", "alias_receipts"):
        if not isinstance(payload.get(field), list):
            raise ReceiptContractError(f"study.{field} must be a list")
    return payload


def _index_receipts(
    adapters: Sequence[dict[str, Any]],
    evaluators: Sequence[dict[str, Any]],
    aliases: Sequence[dict[str, Any]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    adapter_index: dict[tuple[str, str], dict[str, Any]] = {}
    evaluator_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    alias_index: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in adapters:
        key = (receipt["scene"], receipt["reported_method_id"])
        if key in adapter_index:
            raise ReceiptContractError(f"duplicate adapter receipt for {key}")
        adapter_index[key] = receipt
    for receipt in evaluators:
        key = (
            receipt["scene"],
            receipt["reported_method_id"],
            receipt["evaluator_id"],
        )
        if key in evaluator_index:
            raise ReceiptContractError(f"duplicate evaluator receipt for {key}")
        evaluator_index[key] = receipt
    for receipt in aliases:
        key = (receipt["scene"], receipt["reported_method_id"])
        if key in alias_index or key in adapter_index:
            raise ReceiptContractError(f"duplicate adapter/alias endpoint for {key}")
        alias_index[key] = receipt

    for key, evaluator in evaluator_index.items():
        adapter_key = key[:2]
        adapter = adapter_index.get(adapter_key)
        if adapter is None or adapter["status"] != SUCCEEDED:
            raise ReceiptContractError(
                f"evaluator {key} has no successful adapter endpoint"
            )
        if evaluator["endpoint_reuse_key"] != adapter["endpoint"]["reuse_key"]:
            raise ReceiptContractError(f"evaluator {key} binds a different endpoint")
    for key, alias in alias_index.items():
        source_key = (key[0], alias["source_method_id"])
        source = adapter_index.get(source_key)
        if source is None or source["status"] != SUCCEEDED:
            raise ReceiptContractError(f"alias {key} has no successful source adapter")
        if alias["source_endpoint_reuse_key"] != source["endpoint"]["reuse_key"]:
            raise ReceiptContractError(f"alias {key} source endpoint binding mismatch")
        if source_key in alias_index:
            raise ReceiptContractError("alias chains are not supported")
    return adapter_index, evaluator_index, alias_index


def _metric_state(
    *,
    scene: str,
    method_id: str,
    metric_spec: Mapping[str, Any],
    protocol: Mapping[str, Any],
    adapters: Mapping[tuple[str, str], dict[str, Any]],
    evaluators: Mapping[tuple[str, str, str], dict[str, Any]],
    aliases: Mapping[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    metric_id = str(metric_spec["metric_id"])
    evaluator_id = str(metric_spec["evaluator_id"])
    key = (scene, method_id)
    alias = aliases.get(key)
    source_method_id = method_id
    alias_sha = None
    if alias is not None:
        source_method_id = alias["source_method_id"]
        alias_sha = alias["receipt_sha256"]
    adapter = adapters.get((scene, source_method_id))
    base = {
        "scene": scene,
        "method_id": method_id,
        "metric_id": metric_id,
        "source_method_id": source_method_id,
        "alias_receipt_sha256": alias_sha,
    }
    if adapter is None:
        return {**base, "status": "MISSING_ADAPTER"}
    if adapter["status"] != SUCCEEDED:
        detail = (
            adapter.get("failure")
            if adapter["status"] == FAILED
            else adapter.get("unsupported")
        )
        return {
            **base,
            "status": adapter["status"],
            "phase": "adapter",
            "detail": copy.deepcopy(detail),
            "receipt_sha256": adapter["receipt_sha256"],
        }
    evaluator = evaluators.get((scene, source_method_id, evaluator_id))
    if evaluator is None:
        return {**base, "status": "MISSING_EVALUATION"}
    if evaluator["status"] != SUCCEEDED:
        detail = (
            evaluator.get("failure")
            if evaluator["status"] == FAILED
            else evaluator.get("unsupported")
        )
        return {
            **base,
            "status": evaluator["status"],
            "phase": "evaluator",
            "detail": copy.deepcopy(detail),
            "receipt_sha256": evaluator["receipt_sha256"],
        }
    metric = evaluator.get("metrics", {}).get(metric_id)
    if metric is None:
        return {
            **base,
            "status": "MISSING_METRIC",
            "receipt_sha256": evaluator["receipt_sha256"],
        }
    validated = _validate_metrics({metric_id: metric}, "metric state")[metric_id]
    scope = validated["evaluation_scope"]
    expected_scope = {
        "metric_definition_id": metric_spec["metric_definition_id"],
        "unit": metric_spec["unit"],
        "split": metric_spec["split"],
        "aggregation": metric_spec["aggregation"],
        "protocol_sha256": protocol["sha256"],
        "collection_sha256": protocol["collection_hash"],
    }
    observed_scope = {
        "metric_definition_id": scope["metric_definition_id"],
        "unit": validated["unit"],
        "split": scope["split"],
        "aggregation": scope["aggregation"],
        "protocol_sha256": scope["protocol_sha256"],
        "collection_sha256": scope["collection_sha256"],
    }
    for optional_field in ("depth_definition", "threshold_m"):
        if optional_field in metric_spec:
            expected_scope[optional_field] = metric_spec[optional_field]
            observed_scope[optional_field] = scope.get(optional_field)
        elif optional_field in scope:
            raise ReceiptContractError(
                f"metric scope for {(scene, method_id, metric_id)} declares unexpected "
                f"{optional_field}"
            )
    if observed_scope != expected_scope:
        raise ReceiptContractError(
            f"metric scope mismatch for {(scene, method_id, metric_id)}: "
            f"expected={expected_scope} observed={observed_scope}"
        )
    if validated["status"] != SUCCEEDED:
        detail = (
            validated.get("failure")
            if validated["status"] == FAILED
            else validated.get("unsupported")
        )
        return {
            **base,
            "status": validated["status"],
            "phase": "metric",
            "detail": copy.deepcopy(detail),
            "receipt_sha256": evaluator["receipt_sha256"],
        }
    return {
        **base,
        "status": SUCCEEDED,
        "value": validated["value"],
        "unit": validated["unit"],
        "pairing_key": validated["pairing_key"],
        "evaluation_scope": scope,
        "endpoint_reuse_key": adapter["endpoint"]["reuse_key"],
        "evaluator_receipt_sha256": evaluator["receipt_sha256"],
    }


def _cost_summary(receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    receipt_list = list(receipts)
    batches: dict[str, dict[str, Any]] = {}
    method_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    normalized_by_receipt: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for receipt in receipt_list:
        alias = receipt.get("schema") == ALIAS_SCHEMA
        normalized_by_receipt.append(
            (
                receipt,
                _validate_cost_records(
                    receipt.get("cost_records", []),
                    label=f"{receipt.get('schema')}.cost_records",
                    alias=alias,
                ),
            )
        )

    def add_batch(record: Mapping[str, Any]) -> None:
        batch = record["batch_execution"]
        batch_key = batch["batch_id"]
        batch_material = {
            "batch_id": batch_key,
            "cost_class": record["cost_class"],
            "stage": record["stage"],
            "timing_scope": copy.deepcopy(record["timing_scope"]),
            "execution_environment": copy.deepcopy(record["execution_environment"]),
            "wall_time_s": batch["wall_time_s"],
            "gpu_seconds": batch["gpu_seconds"],
            "peak_allocated_vram_mb": batch["peak_allocated_vram_mb"],
            "peak_reserved_vram_mb": batch["peak_reserved_vram_mb"],
            "disk_bytes": batch["disk_bytes"],
        }
        if "render_profile" in batch:
            batch_material["render_profile"] = copy.deepcopy(batch["render_profile"])
        if batch_key in batches and batches[batch_key] != batch_material:
            raise ReceiptContractError(
                f"batch execution {batch_key!r} has inconsistent declarations"
            )
        batches[batch_key] = batch_material

    def add_method_record(
        receipt: Mapping[str, Any], record: Mapping[str, Any]
    ) -> None:
        scene = str(receipt.get("scene", ""))
        method_id = str(receipt.get("reported_method_id", ""))
        method_key = (scene, method_id, record["cost_id"])
        method_material = {
            "scene": scene,
            "method_id": method_id,
            "cost_id": record["cost_id"],
            "cost_class": record["cost_class"],
            "stage": record["stage"],
            "timing_scope": copy.deepcopy(record["timing_scope"]),
            "execution_environment": copy.deepcopy(record["execution_environment"]),
            **record["reported_method"],
        }
        if method_key in method_records and method_records[method_key] != method_material:
            raise ReceiptContractError(
                f"reported method cost has conflicting duplicate: {method_key}"
            )
        method_records[method_key] = method_material

    for receipt, records in normalized_by_receipt:
        if receipt.get("schema") == ALIAS_SCHEMA:
            continue
        for record in records:
            add_batch(record)
            if record["cost_class"] == "method_specific":
                add_method_record(receipt, record)

    for receipt, records in normalized_by_receipt:
        if receipt.get("schema") != ALIAS_SCHEMA:
            continue
        source_method_id = str(receipt["source_method_id"])
        scene = str(receipt["scene"])
        inherited_source_ids: set[str] = set()
        for record in records:
            add_batch(record)
            source_cost_id = str(record["reported_method"]["source_cost_id"])
            inherited_source_ids.add(source_cost_id)
            source = method_records.get((scene, source_method_id, source_cost_id))
            if source is None:
                raise ReceiptContractError(
                    f"alias {(scene, receipt['reported_method_id'])} references missing "
                    f"source cost {source_cost_id!r}"
                )
            alias_material = {
                "stage": record["stage"],
                "timing_scope": record["timing_scope"],
                "execution_environment": record["execution_environment"],
                "wall_time_s": record["reported_method"]["wall_time_s"],
                "gpu_seconds": record["reported_method"]["gpu_seconds"],
                "peak_allocated_vram_mb": record["reported_method"]["peak_allocated_vram_mb"],
                "peak_reserved_vram_mb": record["reported_method"]["peak_reserved_vram_mb"],
                "disk_bytes": record["reported_method"]["disk_bytes"],
                "render_profile": record["reported_method"].get("render_profile"),
            }
            source_material = {
                field: source.get(field)
                for field in alias_material
            }
            if alias_material != source_material:
                raise ReceiptContractError(
                    f"alias cost {record['cost_id']!r} does not exactly inherit "
                    f"source cost {source_cost_id!r}"
                )
            add_method_record(receipt, record)
        expected_source_ids = {
            key[2]
            for key in method_records
            if key[0] == scene and key[1] == source_method_id
        }
        if inherited_source_ids != expected_source_ids:
            raise ReceiptContractError(
                f"alias {(scene, receipt['reported_method_id'])} must inherit every source "
                f"cost record: expected={sorted(expected_source_ids)} "
                f"observed={sorted(inherited_source_ids)}"
            )
    batch_rows = sorted(batches.values(), key=lambda row: row["batch_id"])
    method_rows = sorted(
        method_records.values(),
        key=lambda row: (row["scene"], row["method_id"], row["cost_id"]),
    )
    method_totals: dict[tuple[str, str], dict[str, Any]] = {}
    for row in method_rows:
        key = (row["scene"], row["method_id"])
        total = method_totals.setdefault(
            key,
            {
                "scene": key[0],
                "method_id": key[1],
                "wall_time_s": 0.0,
                "gpu_seconds": 0.0,
                "peak_allocated_vram_mb": 0.0,
                "peak_reserved_vram_mb": 0.0,
                "disk_bytes": 0,
                "cost_record_count": 0,
                "stage_totals": {},
            },
        )
        total["wall_time_s"] += row["wall_time_s"]
        total["gpu_seconds"] += row["gpu_seconds"]
        total["peak_allocated_vram_mb"] = max(
            total["peak_allocated_vram_mb"], row["peak_allocated_vram_mb"]
        )
        total["peak_reserved_vram_mb"] = max(
            total["peak_reserved_vram_mb"], row["peak_reserved_vram_mb"]
        )
        total["disk_bytes"] += row["disk_bytes"]
        total["cost_record_count"] += 1
        stage_total = total["stage_totals"].setdefault(
            row["stage"],
            {
                "wall_time_s": 0.0,
                "gpu_seconds": 0.0,
                "peak_allocated_vram_mb": 0.0,
                "peak_reserved_vram_mb": 0.0,
                "disk_bytes": 0,
            },
        )
        stage_total["wall_time_s"] += row["wall_time_s"]
        stage_total["gpu_seconds"] += row["gpu_seconds"]
        stage_total["peak_allocated_vram_mb"] = max(
            stage_total["peak_allocated_vram_mb"], row["peak_allocated_vram_mb"]
        )
        stage_total["peak_reserved_vram_mb"] = max(
            stage_total["peak_reserved_vram_mb"], row["peak_reserved_vram_mb"]
        )
        stage_total["disk_bytes"] += row["disk_bytes"]
    model_profiles: dict[tuple[str, str], dict[str, Any]] = {}
    adapter_profiles: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipt_list:
        if receipt.get("schema") != ADAPTER_SCHEMA or receipt.get("status") != SUCCEEDED:
            continue
        key = (str(receipt["scene"]), str(receipt["reported_method_id"]))
        profile = _validate_model_profile(
            receipt["endpoint"]["model_profile"], "cost model profile"
        )
        adapter_profiles[key] = profile
        model_profiles[key] = profile
    for receipt in receipt_list:
        if receipt.get("schema") != ALIAS_SCHEMA:
            continue
        source_key = (str(receipt["scene"]), str(receipt["source_method_id"]))
        profile = adapter_profiles.get(source_key)
        if profile is None:
            raise ReceiptContractError(f"alias has no source model profile: {source_key}")
        model_profiles[(str(receipt["scene"]), str(receipt["reported_method_id"]))] = profile
    return {
        "batch_accounting": {
            "scope": "all_incremental_execution_including_explicit_shared_excluded_rows",
            "deduplication_key": "batch_id",
            "records": batch_rows,
            "unique_batch_count": len(batch_rows),
            "nonzero_batch_count": sum(
                1
                for row in batch_rows
                if any(
                    row[field] != 0
                    for field in (
                        "wall_time_s",
                        "gpu_seconds",
                        "peak_allocated_vram_mb",
                        "peak_reserved_vram_mb",
                        "disk_bytes",
                    )
                )
            ),
            "wall_time_s_sum": sum(row["wall_time_s"] for row in batch_rows),
            "gpu_seconds_sum": sum(row["gpu_seconds"] for row in batch_rows),
            "peak_allocated_vram_mb_max": max(
                (row["peak_allocated_vram_mb"] for row in batch_rows), default=0.0
            ),
            "peak_reserved_vram_mb_max": max(
                (row["peak_reserved_vram_mb"] for row in batch_rows), default=0.0
            ),
            "disk_bytes_sum": sum(row["disk_bytes"] for row in batch_rows),
        },
        "method_specific_batch_execution": {
            "records": [row for row in batch_rows if row["cost_class"] == "method_specific"],
            "wall_time_s_sum": sum(
                row["wall_time_s"]
                for row in batch_rows
                if row["cost_class"] == "method_specific"
            ),
        },
        "shared_excluded_execution": {
            "records": [row for row in batch_rows if row["cost_class"] == "shared_excluded"],
            "wall_time_s_sum": sum(
                row["wall_time_s"]
                for row in batch_rows
                if row["cost_class"] == "shared_excluded"
            ),
            "enters_reported_method_cost": False,
        },
        "reported_method_accounting": {
            "records": method_rows,
            "totals": sorted(
                method_totals.values(), key=lambda row: (row["scene"], row["method_id"])
            ),
            "model_profiles": [
                {"scene": key[0], "method_id": key[1], **profile}
                for key, profile in sorted(model_profiles.items())
            ],
            "note": "reported-method costs are attribution values; batch totals are deduplicated separately",
        },
    }


def summarize_study(manifest_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = validate_study_manifest(_load_json(manifest_path), base=manifest_path.parent)
    base = manifest_path.parent
    adapter_pairs = _load_bound_receipts(
        manifest["adapter_receipts"],
        base=base,
        label="adapter_receipts",
        expected_schema=ADAPTER_SCHEMA,
    )
    evaluator_pairs = _load_bound_receipts(
        manifest["evaluator_receipts"],
        base=base,
        label="evaluator_receipts",
        expected_schema=EVALUATOR_SCHEMA,
    )
    alias_pairs = _load_bound_receipts(
        manifest["alias_receipts"],
        base=base,
        label="alias_receipts",
        expected_schema=ALIAS_SCHEMA,
    )
    adapters = [receipt for receipt, _ in adapter_pairs]
    evaluators = [receipt for receipt, _ in evaluator_pairs]
    aliases = [receipt for receipt, _ in alias_pairs]
    adapter_index, evaluator_index, alias_index = _index_receipts(
        adapters, evaluators, aliases
    )

    scenes = [str(value) for value in manifest["scenes"]]
    methods = [str(value["method_id"]) for value in manifest["reported_methods"]]
    expected_scenes_by_method = {
        str(value["method_id"]): [str(scene) for scene in value["expected_scenes"]]
        for value in manifest["reported_methods"]
    }
    for receipt in [*adapters, *aliases]:
        method_id = str(receipt["reported_method_id"])
        scene = str(receipt["scene"])
        if method_id not in expected_scenes_by_method:
            raise ReceiptContractError(
                f"receipt references undeclared reported method {method_id!r}"
            )
        if scene not in expected_scenes_by_method[method_id]:
            raise ReceiptContractError(
                f"receipt {(scene, method_id)} is outside the method-specific expected matrix"
            )
    if manifest["matrix_contract"]["matrix_id"] == "phase1_internal_representative":
        nonformal = [
            (receipt["scene"], receipt["reported_method_id"])
            for receipt in adapters
            if receipt["run_scope"] != "formal"
        ]
        if nonformal:
            raise ReceiptContractError(
                f"formal Phase-1 study contains non-formal adapter receipts: {nonformal}"
            )
    for receipt in evaluators:
        method_id = str(receipt["reported_method_id"])
        scene = str(receipt["scene"])
        if method_id not in expected_scenes_by_method or scene not in expected_scenes_by_method[method_id]:
            raise ReceiptContractError(
                f"evaluator receipt {(scene, method_id)} is outside the expected matrix"
            )
    metric_specs = [dict(value) for value in manifest["metrics"]]
    states: dict[tuple[str, str, str], dict[str, Any]] = {}
    coverage_rows = []
    for method_id in methods:
        for metric in metric_specs:
            metric_id = str(metric["metric_id"])
            scene_states = []
            counts: dict[str, int] = {}
            for scene in scenes:
                if scene not in expected_scenes_by_method[method_id]:
                    state = {
                        "scene": scene,
                        "method_id": method_id,
                        "metric_id": metric_id,
                        "status": NOT_REQUIRED,
                        "reason": "outside_method_expected_matrix",
                    }
                else:
                    state = _metric_state(
                        scene=scene,
                        method_id=method_id,
                        metric_spec=metric,
                        protocol=manifest["protocol"],
                        adapters=adapter_index,
                        evaluators=evaluator_index,
                        aliases=alias_index,
                    )
                states[(scene, method_id, metric_id)] = state
                scene_states.append(state)
                if state["status"] != NOT_REQUIRED:
                    counts[state["status"]] = counts.get(state["status"], 0) + 1
            coverage_rows.append(
                {
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "expected_scene_count": len(expected_scenes_by_method[method_id]),
                    "expected_scenes": expected_scenes_by_method[method_id],
                    "not_required_scenes": [
                        scene
                        for scene in scenes
                        if scene not in expected_scenes_by_method[method_id]
                    ],
                    "status_counts": dict(sorted(counts.items())),
                    "alias_scene_count": sum(
                        1
                        for scene in expected_scenes_by_method[method_id]
                        if (scene, method_id) in alias_index
                    ),
                    "scene_status": scene_states,
                }
            )

    method_coverage_rows = []
    for method_id in methods:
        scene_rows = []
        for scene in expected_scenes_by_method[method_id]:
            alias = alias_index.get((scene, method_id))
            source_method_id = alias["source_method_id"] if alias is not None else method_id
            adapter = adapter_index.get((scene, source_method_id))
            row: dict[str, Any] = {
                "scene": scene,
                "method_id": method_id,
                "source_method_id": source_method_id,
                "alias": alias is not None,
            }
            if adapter is None:
                row["status"] = "MISSING_ADAPTER"
            elif adapter["status"] != SUCCEEDED:
                row["status"] = adapter["status"]
                row["phase"] = "adapter"
                row["detail"] = copy.deepcopy(
                    adapter.get("failure")
                    if adapter["status"] == FAILED
                    else adapter.get("unsupported")
                )
            else:
                row["status"] = SUCCEEDED
            if row["status"] in {FAILED, UNSUPPORTED}:
                row["failure_signature"] = canonical_json_sha256(
                    {
                        "phase": row.get("phase"),
                        "status": row["status"],
                        "detail": row.get("detail"),
                    }
                )
            scene_rows.append(row)
        status_counts: dict[str, int] = {}
        for row in scene_rows:
            status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        method_coverage_rows.append(
            {
                "method_id": method_id,
                "collection_scene_count": len(scenes),
                "expected_scene_count": len(expected_scenes_by_method[method_id]),
                "expected_scenes": expected_scenes_by_method[method_id],
                "valid_scene_count": status_counts.get(SUCCEEDED, 0),
                "failed_scene_count": status_counts.get(FAILED, 0),
                "unsupported_scene_count": status_counts.get(UNSUPPORTED, 0),
                "alias_scene_count": sum(1 for row in scene_rows if row["alias"]),
                "missing_scene_count": sum(
                    count
                    for status, count in status_counts.items()
                    if status.startswith("MISSING_")
                ),
                "completion_fraction_of_expected": (
                    status_counts.get(SUCCEEDED, 0) / len(expected_scenes_by_method[method_id])
                ),
                "status_counts": dict(sorted(status_counts.items())),
                "failure_signatures": sorted(
                    {
                        row["failure_signature"]
                        for row in scene_rows
                        if "failure_signature" in row
                    }
                ),
                "scene_status": scene_rows,
            }
        )

    available_scene_macro = []
    for method_id in methods:
        for metric in metric_specs:
            metric_id = str(metric["metric_id"])
            valid = [
                states[(scene, method_id, metric_id)]
                for scene in expected_scenes_by_method[method_id]
                if states[(scene, method_id, metric_id)]["status"] == SUCCEEDED
            ]
            aggregate = (
                statistics.fmean
                if metric["macro_aggregation"] == "mean"
                else statistics.median
            )
            available_scene_macro.append(
                {
                    "method_id": method_id,
                    "metric_id": metric_id,
                    "role": "secondary_available_scene_macro",
                    "aggregation": metric["macro_aggregation"],
                    "n_valid": len(valid),
                    "scenes": [row["scene"] for row in valid],
                    "value": (
                        float(aggregate([row["value"] for row in valid]))
                        if valid
                        else None
                    ),
                }
            )

    primary_macro = []
    primary_group_macro = []
    comparison_coverage = []
    for comparison in manifest["comparisons"]:
        group_methods = [str(method) for method in comparison["method_ids"]]
        baseline_method = str(comparison["baseline_method_id"])
        comparison_scenes = [
            scene
            for scene in scenes
            if all(scene in expected_scenes_by_method[method] for method in group_methods)
        ]
        for metric in metric_specs:
            metric_id = str(metric["metric_id"])
            common_rows = []
            excluded = []
            for scene in comparison_scenes:
                scene_states = {
                    method: states[(scene, method, metric_id)] for method in group_methods
                }
                if any(state["status"] != SUCCEEDED for state in scene_states.values()):
                    excluded.append(
                        {
                            "scene": scene,
                            "reason": "non_success_status",
                            "method_status": {
                                method: state["status"]
                                for method, state in scene_states.items()
                            },
                        }
                    )
                    continue
                pairing_keys = {
                    method: state["pairing_key"]
                    for method, state in scene_states.items()
                }
                if len(set(pairing_keys.values())) != 1:
                    excluded.append(
                        {
                            "scene": scene,
                            "reason": "pairing_key_mismatch",
                            "method_pairing_keys": pairing_keys,
                        }
                    )
                    continue
                common_rows.append(
                    {
                        "scene": scene,
                        "method_values": {
                            method: state["value"] for method, state in scene_states.items()
                        },
                        "pairing_key": next(iter(pairing_keys.values())),
                    }
                )
            comparison_coverage.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "method_ids": group_methods,
                    "baseline_method_id": baseline_method,
                    "metric_id": metric_id,
                    "expected_scene_count": len(comparison_scenes),
                    "expected_scenes": comparison_scenes,
                    "common_scene_count": len(common_rows),
                    "common_scenes": [row["scene"] for row in common_rows],
                    "excluded": excluded,
                }
            )
            if not common_rows:
                continue
            aggregate = (
                statistics.fmean if metric["macro_aggregation"] == "mean" else statistics.median
            )
            method_values = {
                method: float(
                    aggregate([row["method_values"][method] for row in common_rows])
                )
                for method in group_methods
            }
            primary_group_macro.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "method_ids": group_methods,
                    "baseline_method_id": baseline_method,
                    "metric_id": metric_id,
                    "direction": metric["direction"],
                    "aggregation": (
                        f"unweighted_{metric['macro_aggregation']}_over_"
                        "metric_specific_group_common_scenes"
                    ),
                    "common_scene_count": len(common_rows),
                    "common_scenes": [row["scene"] for row in common_rows],
                    "method_values": method_values,
                }
            )
            for compared_method in group_methods:
                if compared_method == baseline_method:
                    continue
                baseline_value = method_values[baseline_method]
                compared_value = method_values[compared_method]
                raw_delta = compared_value - baseline_value
                preferred_delta = raw_delta if metric["direction"] == "max" else -raw_delta
                wins = ties = losses = 0
                tolerance = float(metric["tie_tolerance"])
                for row in common_rows:
                    scene_delta = (
                        row["method_values"][compared_method]
                        - row["method_values"][baseline_method]
                    )
                    preferred_scene_delta = (
                        scene_delta if metric["direction"] == "max" else -scene_delta
                    )
                    if preferred_scene_delta > tolerance:
                        wins += 1
                    elif preferred_scene_delta < -tolerance:
                        losses += 1
                    else:
                        ties += 1
                primary_macro.append(
                    {
                        "comparison_id": comparison["comparison_id"],
                        "comparison_group_method_ids": group_methods,
                        "left_method_id": baseline_method,
                        "right_method_id": compared_method,
                        "metric_id": metric_id,
                        "direction": metric["direction"],
                        "aggregation": (
                            f"unweighted_{metric['macro_aggregation']}_over_"
                            "metric_specific_group_common_scenes"
                        ),
                        "common_scene_count": len(common_rows),
                        "common_scenes": [row["scene"] for row in common_rows],
                        "left_value": baseline_value,
                        "right_value": compared_value,
                        "right_minus_left": raw_delta,
                        "right_preferred_delta": preferred_delta,
                        "tie_tolerance": tolerance,
                        "right_win_tie_loss": {
                            "win": wins,
                            "tie": ties,
                            "loss": losses,
                        },
                    }
                )

    failures = []
    for phase, receipts in (("adapter", adapters), ("evaluator", evaluators)):
        for receipt in receipts:
            if receipt["status"] == SUCCEEDED:
                continue
            detail = receipt.get("failure") if receipt["status"] == FAILED else receipt.get("unsupported")
            failures.append(
                {
                    "phase": phase,
                    "scene": receipt["scene"],
                    "method_id": receipt["reported_method_id"],
                    "status": receipt["status"],
                    "detail": copy.deepcopy(detail),
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )
    for receipt in evaluators:
        if receipt["status"] != SUCCEEDED:
            continue
        for metric_id, metric in _validate_metrics(
            receipt.get("metrics", {}), "failure summary metrics"
        ).items():
            if metric["status"] == SUCCEEDED:
                continue
            detail = (
                metric.get("failure")
                if metric["status"] == FAILED
                else metric.get("unsupported")
            )
            failures.append(
                {
                    "phase": "metric",
                    "scene": receipt["scene"],
                    "method_id": receipt["reported_method_id"],
                    "metric_id": metric_id,
                    "status": metric["status"],
                    "detail": copy.deepcopy(detail),
                    "receipt_sha256": receipt["receipt_sha256"],
                }
            )

    endpoint_groups: dict[str, list[dict[str, str]]] = {}
    for receipt in adapters:
        if receipt["status"] == SUCCEEDED:
            endpoint_groups.setdefault(receipt["endpoint"]["reuse_key"], []).append(
                {"scene": receipt["scene"], "method_id": receipt["reported_method_id"]}
            )
    for receipt in aliases:
        endpoint_groups.setdefault(receipt["source_endpoint_reuse_key"], []).append(
            {"scene": receipt["scene"], "method_id": receipt["reported_method_id"]}
        )

    receipt_identities = [
        {"kind": kind, **identity}
        for kind, pairs in (
            ("adapter", adapter_pairs),
            ("evaluator", evaluator_pairs),
            ("alias", alias_pairs),
        )
        for _, identity in pairs
    ]
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "protocol": copy.deepcopy(manifest["protocol"]),
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "source_receipts": sorted(receipt_identities, key=lambda row: (row["kind"], row["path"])),
        "policies": {
            "endpoint_provenance_scope": "training_endpoint_only",
            "evaluation_provenance_scope": "endpoint_evaluation_only",
            "repository_metadata_in_scope_hash": False,
            "absolute_paths_in_scope_hash": False,
            "primary_macro_status": SUCCEEDED,
            "failed_or_unsupported_in_primary_macro": False,
            "macro_scene_policy": (
                "metric-specific common-scene intersection across every method in each "
                "declared comparison group; every pair reuses that exact set"
            ),
            "cost_policy": (
                "incremental batch execution deduplicated by batch_id; exact aliases execute "
                "zero additional work but inherit the source endpoint's reported method cost; "
                "shared_excluded work is retained for audit and never enters method cost"
            ),
        },
        "endpoint_reuse": {
            "groups": [
                {
                    "endpoint_reuse_key": key,
                    "reported_endpoints": sorted(
                        rows, key=lambda row: (row["scene"], row["method_id"])
                    ),
                }
                for key, rows in sorted(endpoint_groups.items())
            ]
        },
        "primary_common_scene_macro": primary_group_macro,
        "primary_paired_macro": primary_macro,
        "secondary_available_scene_macro": available_scene_macro,
        "coverage": {
            "endpoint_completion": method_coverage_rows,
            "method_metric": coverage_rows,
            "comparison_metric": comparison_coverage,
        },
        "failures_and_unsupported": sorted(
            failures, key=lambda row: (row["phase"], row["scene"], row["method_id"])
        ),
        "cost": _cost_summary([*adapters, *evaluators, *aliases]),
    }
    summary = _finalize_self_hash(summary, "receipt_sha256")
    if output_path is not None:
        output = output_path.resolve()
        if output.exists():
            raise ReceiptContractError(f"refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate", help="Validate one receipt")
    validate_parser.add_argument("--receipt", required=True, type=Path)
    summarize_parser = subparsers.add_parser("summarize", help="Build study summary receipt")
    summarize_parser.add_argument("--manifest", required=True, type=Path)
    summarize_parser.add_argument("--output", required=True, type=Path)
    schema_parser = subparsers.add_parser("schema", help="Copy the bundled JSON Schema")
    schema_parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "validate":
        path = args.receipt.resolve()
        receipt = validate_receipt(_load_json(path))
        result = {
            "schema": receipt["schema"],
            "status": receipt["status"],
            "receipt_sha256": receipt["receipt_sha256"],
            "file_sha256": sha256_file(path),
        }
    elif args.command == "summarize":
        result = summarize_study(args.manifest, args.output)
    else:
        source = Path(__file__).resolve().parent / "schemas" / SCHEMA_BUNDLE
        if not source.is_file():
            raise ReceiptContractError(f"bundled JSON Schema is missing: {source}")
        output = args.output.resolve()
        if output.exists():
            raise ReceiptContractError(f"refusing to overwrite output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes())
        result = {
            "schema_file": str(output),
            "sha256": sha256_file(output),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
