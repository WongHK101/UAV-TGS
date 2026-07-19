#!/usr/bin/env python3
"""Minimal Hold-8 v2 endpoint receipts and paired scene-macro aggregation.

This module is deliberately independent from ``baseline_evaluation_receipts``.
The latter implements the richer guard4/v1 contract and remains available for
the archived exploratory study.  Hold-8 v2 keeps one flat training receipt,
one flat evaluation receipt, and (only when needed) one exact no-op alias
record.  The two signatures are scoped provenance digests, not recursive
receipt self-hashes.

The formal aggregate is scene-macro.  Its scene intersection is computed
separately for every metric, so an unsupported RGB output cannot remove an
otherwise valid thermal or temperature result.
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


PROTOCOL_ID = "uav-tgs-aaai27-hold8-v2"
TRAINING_SCHEMA = "uav-tgs-aaai27-hold8-training-receipt-v2"
EVALUATION_SCHEMA = "uav-tgs-aaai27-hold8-evaluation-receipt-v2"
ALIAS_SCHEMA = "uav-tgs-aaai27-hold8-noop-alias-v2"
SUMMARY_SCHEMA = "uav-tgs-aaai27-hold8-six-scene-summary-v2"

FORMAL_SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)

SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
UNSUPPORTED = "UNSUPPORTED"
TERMINAL_STATUSES = frozenset({SUCCEEDED, FAILED, UNSUPPORTED})

METRIC_VALID = "VALID"
METRIC_NA = "N/A"
METRIC_STATUSES = frozenset({METRIC_VALID, METRIC_NA})

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

TRAINING_SIGNATURE_FIELDS = (
    "protocol_id",
    "method_id",
    "repository",
    # Whole-repository commits are intentionally informational.  The scoped
    # digest below binds only code that can affect this endpoint, so later
    # matrix-runner/report-only commits do not invalidate a reusable result.
    "source_scoped_code_sha256",
    "runtime_patch_sha256",
    "recipe",
    "config_sha256",
    "seed",
    "split_manifest_sha256",
    "data_sha256",
    "camera_sha256",
    "range_sha256",
    "lut_sha256",
    "command",
)

EVALUATION_SIGNATURE_FIELDS = (
    "protocol_id",
    "training_signature",
    "endpoint_sha256",
    "evaluator_code_sha256",
    "evaluator_config_sha256",
    "split_manifest_sha256",
    "reference_sha256",
)

_TRAINING_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "scene",
        "method_id",
        "repository",
        "repository_commit",
        "source_training_commit",
        "current_runner_commit",
        "source_scoped_code_sha256",
        "reused_endpoint",
        "reuse_reason",
        "runtime_patch_sha256",
        "recipe",
        "config_sha256",
        "seed",
        "split_manifest_sha256",
        "data_sha256",
        "camera_sha256",
        "range_sha256",
        "lut_sha256",
        "host",
        "gpu",
        "command",
        "endpoint_sha256",
        "completion_status",
        "failure_reason",
        "batch_execution_wall_time_s",
        "reported_method_wall_time_s",
        "peak_vram_bytes",
        "model_size_bytes",
        "gaussian_count",
        "training_signature",
    }
)

_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "scene",
        "method_id",
        "training_signature",
        "endpoint_sha256",
        "evaluator_code_sha256",
        "evaluator_config_sha256",
        "split_manifest_sha256",
        "reference_sha256",
        "completion_status",
        "failure_reason",
        "metrics",
        "render_fps",
        "evaluation_signature",
    }
)

_ALIAS_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "scene",
        "alias_method_id",
        "source_method_id",
        "source_endpoint_sha256",
        "source_training_signature",
        "source_evaluation_signature",
        "proof_sha256",
        "alias_reason",
        "modified_count",
        "completion_status",
        "independent_endpoint_run",
        "independent_performance_claim",
        "batch_execution_wall_time_s",
        "additional_alias_cost_s",
        "reported_method_wall_time_s",
    }
)


class Hold8ReceiptError(ValueError):
    """Raised when a minimal Hold-8 receipt violates the contract."""


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 of canonical JSON without accepting NaN/Inf."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Hold8ReceiptError(
            f"value is not canonical-JSON serializable: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_unknown_keys(
    payload: Mapping[str, Any], allowed: Iterable[str], label: str
) -> None:
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise Hold8ReceiptError(f"{label} has unknown fields: {unknown}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Hold8ReceiptError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Hold8ReceiptError(f"{label} must be a nonempty string")
    return value.strip()


def _require_sha(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _require_string(value, label).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise Hold8ReceiptError(f"{label} must be a lowercase SHA-256")
    return text


def _require_git_commit(value: Any, label: str) -> str:
    text = _require_string(value, label).lower()
    if GIT_COMMIT_RE.fullmatch(text) is None:
        raise Hold8ReceiptError(f"{label} must be a full lowercase 40-hex Git commit")
    return text


def _require_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Hold8ReceiptError(f"{label} must be a boolean")
    return value


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Hold8ReceiptError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise Hold8ReceiptError(f"{label} must be >= {minimum}")
    return value


def _require_finite(
    value: Any,
    label: str,
    *,
    nonnegative: bool = False,
    optional: bool = False,
) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise Hold8ReceiptError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise Hold8ReceiptError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise Hold8ReceiptError(f"{label} must be {qualifier}")
    return number


def _require_status(value: Any, label: str) -> str:
    status = _require_string(value, label)
    if status not in TERMINAL_STATUSES:
        raise Hold8ReceiptError(
            f"{label} must be one of {sorted(TERMINAL_STATUSES)}"
        )
    return status


def _require_scene(value: Any, label: str = "scene") -> str:
    scene = _require_string(value, label)
    if scene not in FORMAL_SCENES:
        raise Hold8ReceiptError(
            f"{label} must be one of the six formal scenes: {list(FORMAL_SCENES)}"
        )
    return scene


def _signature(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    missing = [field for field in fields if field not in payload]
    if missing:
        raise Hold8ReceiptError(f"signature material is missing fields: {missing}")
    return canonical_json_sha256({field: payload[field] for field in fields})


def training_signature(payload: Mapping[str, Any]) -> str:
    """Compute the training-scoped signature from a flat receipt draft."""

    return _signature(payload, TRAINING_SIGNATURE_FIELDS)


def evaluation_signature(payload: Mapping[str, Any]) -> str:
    """Compute the evaluation-scoped signature from a flat receipt draft."""

    return _signature(payload, EVALUATION_SIGNATURE_FIELDS)


def _validate_completion(
    payload: Mapping[str, Any], *, label: str, endpoint_field: bool
) -> str:
    status = _require_status(payload.get("completion_status"), f"{label}.completion_status")
    reason = payload.get("failure_reason")
    if status == SUCCEEDED:
        if reason is not None:
            raise Hold8ReceiptError(
                f"{label}.failure_reason must be null after success"
            )
        if endpoint_field:
            _require_sha(payload.get("endpoint_sha256"), f"{label}.endpoint_sha256")
    else:
        _require_string(reason, f"{label}.failure_reason")
        if endpoint_field and payload.get("endpoint_sha256") is not None:
            raise Hold8ReceiptError(
                f"{label}.endpoint_sha256 must be null unless training succeeded"
            )
    return status


def validate_training_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one flat training receipt."""

    receipt = copy.deepcopy(dict(_require_mapping(payload, "training receipt")))
    _reject_unknown_keys(receipt, _TRAINING_FIELDS, "training receipt")
    if receipt.get("schema") != TRAINING_SCHEMA:
        raise Hold8ReceiptError(f"training receipt.schema must be {TRAINING_SCHEMA!r}")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise Hold8ReceiptError(f"training receipt.protocol_id must be {PROTOCOL_ID!r}")

    _require_scene(receipt.get("scene"))
    for field in (
        "method_id",
        "repository",
        "recipe",
        "host",
        "gpu",
        "command",
    ):
        _require_string(receipt.get(field), f"training receipt.{field}")
    for field in (
        "repository_commit",
        "source_training_commit",
        "current_runner_commit",
    ):
        _require_git_commit(receipt.get(field), f"training receipt.{field}")
    _require_sha(
        receipt.get("runtime_patch_sha256"),
        "training receipt.runtime_patch_sha256",
        optional=True,
    )
    for field in (
        "source_scoped_code_sha256",
        "config_sha256",
        "split_manifest_sha256",
        "data_sha256",
        "camera_sha256",
        "range_sha256",
        "lut_sha256",
    ):
        _require_sha(receipt.get(field), f"training receipt.{field}")
    _require_int(receipt.get("seed"), "training receipt.seed", minimum=0)
    status = _validate_completion(receipt, label="training receipt", endpoint_field=True)

    batch_time = _require_finite(
        receipt.get("batch_execution_wall_time_s"),
        "training receipt.batch_execution_wall_time_s",
        nonnegative=True,
    )
    reported = _require_finite(
        receipt.get("reported_method_wall_time_s"),
        "training receipt.reported_method_wall_time_s",
        nonnegative=True,
        optional=status != SUCCEEDED,
    )
    if status == SUCCEEDED and (reported is None or reported <= 0.0):
        raise Hold8ReceiptError(
            "training receipt.reported_method_wall_time_s must preserve a positive formal cost after success"
        )

    reused = _require_bool(
        receipt.get("reused_endpoint"), "training receipt.reused_endpoint"
    )
    reuse_reason = receipt.get("reuse_reason")
    if reused:
        if status != SUCCEEDED:
            raise Hold8ReceiptError(
                "training receipt.reused_endpoint requires a successful endpoint"
            )
        _require_string(reuse_reason, "training receipt.reuse_reason")
        if batch_time != 0.0:
            raise Hold8ReceiptError(
                "training receipt.batch_execution_wall_time_s must be 0 for a reused endpoint"
            )
        if reported is None or reported <= 0.0:
            raise Hold8ReceiptError(
                "training receipt.reported_method_wall_time_s must preserve a positive formal cost for a reused endpoint"
            )
    elif reuse_reason is not None:
        raise Hold8ReceiptError(
            "training receipt.reuse_reason must be null for a fresh endpoint"
        )
    peak_vram = _require_finite(
        receipt.get("peak_vram_bytes"),
        "training receipt.peak_vram_bytes",
        nonnegative=True,
        optional=status != SUCCEEDED,
    )
    model_size = _require_finite(
        receipt.get("model_size_bytes"),
        "training receipt.model_size_bytes",
        nonnegative=True,
        optional=status != SUCCEEDED,
    )
    if status == SUCCEEDED and (peak_vram is None or model_size is None):
        raise Hold8ReceiptError(
            "successful training requires peak_vram_bytes and model_size_bytes"
        )
    if receipt.get("gaussian_count") is not None:
        _require_int(
            receipt.get("gaussian_count"),
            "training receipt.gaussian_count",
            minimum=0,
        )

    declared = _require_sha(
        receipt.get("training_signature"), "training receipt.training_signature"
    )
    actual = training_signature(receipt)
    if declared != actual:
        raise Hold8ReceiptError(
            "training receipt signature mismatch: "
            f"declared={declared} actual={actual}"
        )
    return receipt


def finalize_training_receipt(draft: Mapping[str, Any]) -> dict[str, Any]:
    """Add the flat schema/signature fields and validate a training draft."""

    result = copy.deepcopy(dict(_require_mapping(draft, "training draft")))
    result.setdefault("schema", TRAINING_SCHEMA)
    result.setdefault("protocol_id", PROTOCOL_ID)
    computed = training_signature(result)
    declared = result.get("training_signature")
    if declared is not None and declared != computed:
        raise Hold8ReceiptError(
            "training draft contains a stale or incorrect training_signature"
        )
    result["training_signature"] = computed
    return validate_training_receipt(result)


def _validate_metrics(value: Any, *, completion_status: str) -> dict[str, dict[str, Any]]:
    metrics = dict(_require_mapping(value, "evaluation receipt.metrics"))
    normalized: dict[str, dict[str, Any]] = {}
    for metric_name, raw in metrics.items():
        name = _require_string(metric_name, "metric name")
        entry = dict(_require_mapping(raw, f"metric {name!r}"))
        status = _require_string(entry.get("status"), f"metric {name!r}.status")
        if status not in METRIC_STATUSES:
            raise Hold8ReceiptError(
                f"metric {name!r}.status must be one of {sorted(METRIC_STATUSES)}"
            )
        if status == METRIC_VALID:
            _reject_unknown_keys(entry, {"status", "value"}, f"metric {name!r}")
            number = _require_finite(entry.get("value"), f"metric {name!r}.value")
            normalized[name] = {"status": METRIC_VALID, "value": number}
        else:
            _reject_unknown_keys(entry, {"status", "reason"}, f"metric {name!r}")
            normalized[name] = {
                "status": METRIC_NA,
                "reason": _require_string(entry.get("reason"), f"metric {name!r}.reason"),
            }
    if completion_status != SUCCEEDED and normalized:
        raise Hold8ReceiptError(
            "a failed/unsupported evaluation must not publish metric values"
        )
    return normalized


def validate_evaluation_receipt(
    payload: Mapping[str, Any],
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an evaluation receipt and optionally bind it to training."""

    receipt = copy.deepcopy(dict(_require_mapping(payload, "evaluation receipt")))
    _reject_unknown_keys(receipt, _EVALUATION_FIELDS, "evaluation receipt")
    if receipt.get("schema") != EVALUATION_SCHEMA:
        raise Hold8ReceiptError(
            f"evaluation receipt.schema must be {EVALUATION_SCHEMA!r}"
        )
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise Hold8ReceiptError(
            f"evaluation receipt.protocol_id must be {PROTOCOL_ID!r}"
        )
    _require_scene(receipt.get("scene"))
    _require_string(receipt.get("method_id"), "evaluation receipt.method_id")
    for field in (
        "training_signature",
        "endpoint_sha256",
        "evaluator_code_sha256",
        "evaluator_config_sha256",
        "split_manifest_sha256",
    ):
        _require_sha(receipt.get(field), f"evaluation receipt.{field}")
    _require_sha(
        receipt.get("reference_sha256"),
        "evaluation receipt.reference_sha256",
        optional=True,
    )
    status = _validate_completion(receipt, label="evaluation receipt", endpoint_field=False)
    receipt["metrics"] = _validate_metrics(
        receipt.get("metrics"), completion_status=status
    )
    render_fps = _require_finite(
        receipt.get("render_fps"),
        "evaluation receipt.render_fps",
        nonnegative=True,
        optional=status != SUCCEEDED,
    )
    if status == SUCCEEDED and render_fps is None:
        raise Hold8ReceiptError("successful evaluation requires render_fps")

    declared = _require_sha(
        receipt.get("evaluation_signature"),
        "evaluation receipt.evaluation_signature",
    )
    actual = evaluation_signature(receipt)
    if declared != actual:
        raise Hold8ReceiptError(
            "evaluation receipt signature mismatch: "
            f"declared={declared} actual={actual}"
        )

    if training is not None:
        bound = validate_training_receipt(training)
        for field in ("protocol_id", "scene", "method_id", "training_signature"):
            if receipt[field] != bound[field]:
                raise Hold8ReceiptError(
                    f"evaluation/training binding mismatch for {field}"
                )
        if bound["completion_status"] != SUCCEEDED:
            raise Hold8ReceiptError("evaluation cannot bind to unsuccessful training")
        if receipt["endpoint_sha256"] != bound["endpoint_sha256"]:
            raise Hold8ReceiptError("evaluation/training endpoint hash mismatch")
        if receipt["split_manifest_sha256"] != bound["split_manifest_sha256"]:
            raise Hold8ReceiptError("evaluation/training split hash mismatch")
    return receipt


def finalize_evaluation_receipt(
    draft: Mapping[str, Any], training: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind a flat evaluation draft to a successful training receipt."""

    bound = validate_training_receipt(training)
    if bound["completion_status"] != SUCCEEDED:
        raise Hold8ReceiptError("cannot evaluate an unsuccessful training receipt")
    result = copy.deepcopy(dict(_require_mapping(draft, "evaluation draft")))
    result.setdefault("schema", EVALUATION_SCHEMA)
    result.setdefault("protocol_id", PROTOCOL_ID)
    bindings = {
        "scene": bound["scene"],
        "method_id": bound["method_id"],
        "training_signature": bound["training_signature"],
        "endpoint_sha256": bound["endpoint_sha256"],
        "split_manifest_sha256": bound["split_manifest_sha256"],
    }
    for field, expected in bindings.items():
        if field in result and result[field] != expected:
            raise Hold8ReceiptError(
                f"evaluation draft conflicts with training for {field}"
            )
        result[field] = expected
    computed = evaluation_signature(result)
    declared = result.get("evaluation_signature")
    if declared is not None and declared != computed:
        raise Hold8ReceiptError(
            "evaluation draft contains a stale or incorrect evaluation_signature"
        )
    result["evaluation_signature"] = computed
    return validate_evaluation_receipt(result, bound)


def make_noop_alias_receipt(
    source_training: Mapping[str, Any],
    source_evaluation: Mapping[str, Any],
    *,
    alias_method_id: str,
    proof_sha256: str,
    alias_reason: str = "zero_modified_gaussians",
) -> dict[str, Any]:
    """Create an exact no-op alias with zero batch cost and inherited method cost."""

    training = validate_training_receipt(source_training)
    evaluation = validate_evaluation_receipt(source_evaluation, training)
    if training["completion_status"] != SUCCEEDED or evaluation["completion_status"] != SUCCEEDED:
        raise Hold8ReceiptError("an alias source must have successful training and evaluation")
    alias = _require_string(alias_method_id, "alias_method_id")
    if alias == training["method_id"]:
        raise Hold8ReceiptError("alias_method_id must differ from source_method_id")
    proof = _require_sha(proof_sha256, "proof_sha256")
    result = {
        "schema": ALIAS_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "scene": training["scene"],
        "alias_method_id": alias,
        "source_method_id": training["method_id"],
        "source_endpoint_sha256": training["endpoint_sha256"],
        "source_training_signature": training["training_signature"],
        "source_evaluation_signature": evaluation["evaluation_signature"],
        "proof_sha256": proof,
        "alias_reason": _require_string(alias_reason, "alias_reason"),
        "modified_count": 0,
        "completion_status": SUCCEEDED,
        "independent_endpoint_run": False,
        "independent_performance_claim": False,
        "batch_execution_wall_time_s": 0.0,
        "additional_alias_cost_s": 0.0,
        "reported_method_wall_time_s": training["reported_method_wall_time_s"],
    }
    return validate_noop_alias_receipt(result, training, evaluation)


def validate_noop_alias_receipt(
    payload: Mapping[str, Any],
    source_training: Mapping[str, Any] | None = None,
    source_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate exact alias semantics, optionally against its source records."""

    receipt = copy.deepcopy(dict(_require_mapping(payload, "alias receipt")))
    _reject_unknown_keys(receipt, _ALIAS_FIELDS, "alias receipt")
    if receipt.get("schema") != ALIAS_SCHEMA:
        raise Hold8ReceiptError(f"alias receipt.schema must be {ALIAS_SCHEMA!r}")
    if receipt.get("protocol_id") != PROTOCOL_ID:
        raise Hold8ReceiptError(f"alias receipt.protocol_id must be {PROTOCOL_ID!r}")
    _require_scene(receipt.get("scene"))
    alias = _require_string(receipt.get("alias_method_id"), "alias_method_id")
    source = _require_string(receipt.get("source_method_id"), "source_method_id")
    if alias == source:
        raise Hold8ReceiptError("alias and source method IDs must differ")
    for field in (
        "source_endpoint_sha256",
        "source_training_signature",
        "source_evaluation_signature",
        "proof_sha256",
    ):
        _require_sha(receipt.get(field), f"alias receipt.{field}")
    _require_string(receipt.get("alias_reason"), "alias receipt.alias_reason")
    if _require_int(receipt.get("modified_count"), "alias receipt.modified_count", minimum=0) != 0:
        raise Hold8ReceiptError("an exact no-op alias requires modified_count=0")
    if receipt.get("completion_status") != SUCCEEDED:
        raise Hold8ReceiptError("an alias receipt must be SUCCEEDED")
    if receipt.get("independent_endpoint_run") is not False:
        raise Hold8ReceiptError("independent_endpoint_run must be false for an alias")
    if receipt.get("independent_performance_claim") is not False:
        raise Hold8ReceiptError(
            "independent_performance_claim must be false for an alias"
        )
    for field in ("batch_execution_wall_time_s", "additional_alias_cost_s"):
        value = _require_finite(
            receipt.get(field), f"alias receipt.{field}", nonnegative=True
        )
        if value != 0.0:
            raise Hold8ReceiptError(f"alias receipt.{field} must equal zero")
    _require_finite(
        receipt.get("reported_method_wall_time_s"),
        "alias receipt.reported_method_wall_time_s",
        nonnegative=True,
    )

    if (source_training is None) != (source_evaluation is None):
        raise Hold8ReceiptError(
            "source_training and source_evaluation must be supplied together"
        )
    if source_training is not None and source_evaluation is not None:
        training = validate_training_receipt(source_training)
        evaluation = validate_evaluation_receipt(source_evaluation, training)
        expected = {
            "scene": training["scene"],
            "source_method_id": training["method_id"],
            "source_endpoint_sha256": training["endpoint_sha256"],
            "source_training_signature": training["training_signature"],
            "source_evaluation_signature": evaluation["evaluation_signature"],
            "reported_method_wall_time_s": training["reported_method_wall_time_s"],
        }
        for field, value in expected.items():
            if receipt[field] != value:
                raise Hold8ReceiptError(f"alias/source binding mismatch for {field}")
    return receipt


def _index_unique(
    receipts: Sequence[Mapping[str, Any]],
    *,
    kind: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in receipts:
        if kind == "training":
            receipt = validate_training_receipt(raw)
            method_field = "method_id"
        elif kind == "evaluation":
            receipt = validate_evaluation_receipt(raw)
            method_field = "method_id"
        elif kind == "alias":
            receipt = validate_noop_alias_receipt(raw)
            method_field = "alias_method_id"
        else:  # pragma: no cover - internal programming error
            raise AssertionError(kind)
        key = (receipt["scene"], receipt[method_field])
        if key in indexed:
            raise Hold8ReceiptError(f"duplicate {kind} receipt for {key}")
        indexed[key] = receipt
    return indexed


def summarize_six_scene_study(
    *,
    methods: Sequence[str],
    training_receipts: Sequence[Mapping[str, Any]],
    evaluation_receipts: Sequence[Mapping[str, Any]],
    alias_receipts: Sequence[Mapping[str, Any]] = (),
    metrics: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the formal per-metric common-scene macro and coverage summary.

    A scene contributes to one metric only when every requested method has a
    successful (or exact-aliased) endpoint and a ``VALID`` value for that
    metric.  FAILED/UNSUPPORTED/not-run rows and metric-level ``N/A`` entries
    are never converted to zero and never leak into a macro denominator.
    """

    method_list = [_require_string(item, "method") for item in methods]
    if not method_list or len(method_list) != len(set(method_list)):
        raise Hold8ReceiptError("methods must be a nonempty unique list")

    training_by_key = _index_unique(training_receipts, kind="training")
    evaluation_by_key = _index_unique(evaluation_receipts, kind="evaluation")
    alias_by_key = _index_unique(alias_receipts, kind="alias")

    for key, evaluation in evaluation_by_key.items():
        training = training_by_key.get(key)
        if training is None:
            raise Hold8ReceiptError(f"evaluation has no training receipt: {key}")
        validate_evaluation_receipt(evaluation, training)

    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for scene in FORMAL_SCENES:
        for method in method_list:
            key = (scene, method)
            alias = alias_by_key.get(key)
            training = training_by_key.get(key)
            evaluation = evaluation_by_key.get(key)
            if alias is not None and (training is not None or evaluation is not None):
                raise Hold8ReceiptError(
                    f"alias {key} must not also claim an independent endpoint"
                )

            if alias is not None:
                source_key = (scene, alias["source_method_id"])
                source_training = training_by_key.get(source_key)
                source_evaluation = evaluation_by_key.get(source_key)
                if source_training is None or source_evaluation is None:
                    raise Hold8ReceiptError(
                        f"alias {key} is missing source records for {source_key}"
                    )
                validate_noop_alias_receipt(alias, source_training, source_evaluation)
                resolved[key] = {
                    "state": "COMPLETED",
                    "kind": "ALIAS",
                    "metrics": source_evaluation["metrics"],
                    "batch_execution_wall_time_s": 0.0,
                    "reported_method_wall_time_s": alias[
                        "reported_method_wall_time_s"
                    ],
                    "reason": None,
                }
                continue

            if training is None:
                resolved[key] = {
                    "state": "NOT_RUN",
                    "kind": "INDEPENDENT",
                    "metrics": {},
                    "batch_execution_wall_time_s": 0.0,
                    "reported_method_wall_time_s": None,
                    "reason": "missing training receipt",
                }
                continue
            if training["completion_status"] != SUCCEEDED:
                resolved[key] = {
                    "state": training["completion_status"],
                    "kind": "INDEPENDENT",
                    "metrics": {},
                    "batch_execution_wall_time_s": training[
                        "batch_execution_wall_time_s"
                    ],
                    "reported_method_wall_time_s": None,
                    "reason": training["failure_reason"],
                }
                continue
            if evaluation is None:
                resolved[key] = {
                    "state": "NOT_EVALUATED",
                    "kind": "INDEPENDENT",
                    "metrics": {},
                    "batch_execution_wall_time_s": training[
                        "batch_execution_wall_time_s"
                    ],
                    "reported_method_wall_time_s": training[
                        "reported_method_wall_time_s"
                    ],
                    "reason": "missing evaluation receipt",
                }
                continue
            if evaluation["completion_status"] != SUCCEEDED:
                resolved[key] = {
                    "state": evaluation["completion_status"],
                    "kind": "INDEPENDENT",
                    "metrics": {},
                    "batch_execution_wall_time_s": training[
                        "batch_execution_wall_time_s"
                    ],
                    "reported_method_wall_time_s": training[
                        "reported_method_wall_time_s"
                    ],
                    "reason": evaluation["failure_reason"],
                }
                continue
            resolved[key] = {
                "state": "COMPLETED",
                "kind": "INDEPENDENT",
                "metrics": evaluation["metrics"],
                "batch_execution_wall_time_s": training[
                    "batch_execution_wall_time_s"
                ],
                "reported_method_wall_time_s": training[
                    "reported_method_wall_time_s"
                ],
                "reason": None,
            }

    if metrics is None:
        metric_names = sorted(
            {
                metric
                for row in resolved.values()
                for metric in row["metrics"]
            }
        )
    else:
        metric_names = [_require_string(item, "metric") for item in metrics]
        if len(metric_names) != len(set(metric_names)):
            raise Hold8ReceiptError("metrics must not contain duplicates")

    method_coverage: dict[str, Any] = {}
    for method in method_list:
        rows = [resolved[(scene, method)] for scene in FORMAL_SCENES]
        failures = [
            {
                "scene": scene,
                "state": resolved[(scene, method)]["state"],
                "reason": resolved[(scene, method)]["reason"],
            }
            for scene in FORMAL_SCENES
            if resolved[(scene, method)]["state"] != "COMPLETED"
        ]
        method_coverage[method] = {
            "completed_scene_count": sum(row["state"] == "COMPLETED" for row in rows),
            "formal_scene_count": len(FORMAL_SCENES),
            "failed_scene_count": sum(row["state"] == FAILED for row in rows),
            "unsupported_scene_count": sum(
                row["state"] == UNSUPPORTED for row in rows
            ),
            "not_run_scene_count": sum(row["state"] == "NOT_RUN" for row in rows),
            "not_evaluated_scene_count": sum(
                row["state"] == "NOT_EVALUATED" for row in rows
            ),
            "alias_scene_count": sum(row["kind"] == "ALIAS" for row in rows),
            "independent_completed_scene_count": sum(
                row["state"] == "COMPLETED" and row["kind"] == "INDEPENDENT"
                for row in rows
            ),
            "batch_execution_wall_time_s": float(
                sum(row["batch_execution_wall_time_s"] for row in rows)
            ),
            "reported_method_wall_time_s": float(
                sum(
                    row["reported_method_wall_time_s"]
                    for row in rows
                    if row["reported_method_wall_time_s"] is not None
                )
            ),
            "incomplete_scenes": failures,
        }

    metric_summaries: dict[str, Any] = {}
    for metric in metric_names:
        common_scenes: list[str] = []
        for scene in FORMAL_SCENES:
            entries = [resolved[(scene, method)] for method in method_list]
            if all(
                row["state"] == "COMPLETED"
                and metric in row["metrics"]
                and row["metrics"][metric]["status"] == METRIC_VALID
                for row in entries
            ):
                common_scenes.append(scene)

        method_means: dict[str, float | None] = {}
        method_metric_coverage: dict[str, Any] = {}
        for method in method_list:
            values = [
                resolved[(scene, method)]["metrics"][metric]["value"]
                for scene in common_scenes
            ]
            method_means[method] = (
                float(statistics.fmean(values)) if values else None
            )
            valid_scenes: list[str] = []
            na_scenes: list[str] = []
            unavailable_scenes: list[str] = []
            for scene in FORMAL_SCENES:
                row = resolved[(scene, method)]
                entry = row["metrics"].get(metric)
                if row["state"] != "COMPLETED" or entry is None:
                    unavailable_scenes.append(scene)
                elif entry["status"] == METRIC_VALID:
                    valid_scenes.append(scene)
                else:
                    na_scenes.append(scene)
            method_metric_coverage[method] = {
                "valid_scene_count": len(valid_scenes),
                "valid_scenes": valid_scenes,
                "na_scene_count": len(na_scenes),
                "na_scenes": na_scenes,
                "unavailable_scene_count": len(unavailable_scenes),
                "unavailable_scenes": unavailable_scenes,
            }
        metric_summaries[metric] = {
            "common_scene_count": len(common_scenes),
            "common_scenes": common_scenes,
            "scene_macro": method_means,
            "method_coverage": method_metric_coverage,
        }

    return {
        "schema": SUMMARY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "formal_scenes": list(FORMAL_SCENES),
        "methods": method_list,
        "all_methods_complete_all_scenes": all(
            row["state"] == "COMPLETED" for row in resolved.values()
        ),
        "coverage": method_coverage,
        "metrics": metric_summaries,
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Hold8ReceiptError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_paths(base: Path, values: Any, label: str) -> list[Path]:
    if not isinstance(values, list):
        raise Hold8ReceiptError(f"{label} must be a list of JSON paths")
    paths: list[Path] = []
    for item in values:
        path = Path(_require_string(item, f"{label} item"))
        paths.append(path if path.is_absolute() else base / path)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("finalize-training")
    training.add_argument("--input", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)

    evaluation = subparsers.add_parser("finalize-evaluation")
    evaluation.add_argument("--input", type=Path, required=True)
    evaluation.add_argument("--training", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)

    alias = subparsers.add_parser("make-alias")
    alias.add_argument("--source-training", type=Path, required=True)
    alias.add_argument("--source-evaluation", type=Path, required=True)
    alias.add_argument("--alias-method", required=True)
    alias.add_argument("--proof-sha256", required=True)
    alias.add_argument("--alias-reason", default="zero_modified_gaussians")
    alias.add_argument("--output", type=Path, required=True)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "finalize-training":
        payload = finalize_training_receipt(_read_json(args.input))
    elif args.command == "finalize-evaluation":
        payload = finalize_evaluation_receipt(
            _read_json(args.input), _read_json(args.training)
        )
    elif args.command == "make-alias":
        payload = make_noop_alias_receipt(
            _read_json(args.source_training),
            _read_json(args.source_evaluation),
            alias_method_id=args.alias_method,
            proof_sha256=args.proof_sha256,
            alias_reason=args.alias_reason,
        )
    elif args.command == "summarize":
        manifest = dict(_require_mapping(_read_json(args.manifest), "study manifest"))
        base = args.manifest.parent
        payload = summarize_six_scene_study(
            methods=manifest.get("methods", []),
            metrics=manifest.get("metrics"),
            training_receipts=[
                _read_json(path)
                for path in _resolve_paths(
                    base, manifest.get("training_receipts", []), "training_receipts"
                )
            ],
            evaluation_receipts=[
                _read_json(path)
                for path in _resolve_paths(
                    base,
                    manifest.get("evaluation_receipts", []),
                    "evaluation_receipts",
                )
            ],
            alias_receipts=[
                _read_json(path)
                for path in _resolve_paths(
                    base, manifest.get("alias_receipts", []), "alias_receipts"
                )
            ],
        )
    else:  # pragma: no cover - argparse enforces a valid subcommand
        raise AssertionError(args.command)
    _write_json(args.output, payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
