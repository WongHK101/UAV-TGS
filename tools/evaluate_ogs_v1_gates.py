#!/usr/bin/env python3
"""Fail-closed OGS-v1 gradient-smoke and scale-safety evidence gates.

This sidecar never changes a checkpoint.  It consumes the fixed-anchor OGS
cache, trusted RGB checkpoints, and the append-only training smoke log.  The
reported update quantities are LR-scaled gradient proxies, not Adam updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from utils.ogs_v1 import load_ogs_cache, ogs_v1_loss, summarize_scale_safety


SMOKE_SCHEMA = "uav-tgs-ogs-v1-gradient-smoke-v1"
SMOKE_REPORT_SCHEMA = "uav-tgs-ogs-v1-gradient-smoke-gate-v1"
SCALE_REPORT_SCHEMA = "uav-tgs-ogs-v1-scale-safety-v1"
PAIRED_REPORT_SCHEMA = "uav-tgs-ogs-v1-paired-smoke-gate-v1"
RGB_INPUT_SCHEMA = "uav-tgs-ogs-v1-rgb-direction-input-v1"
RGB_REPORT_SCHEMA = "uav-tgs-ogs-v1-rgb-direction-gate-v1"
THERMAL_INPUT_SCHEMA = "uav-tgs-ogs-v1-thermal-direction-input-v1"
THERMAL_REPORT_SCHEMA = "uav-tgs-ogs-v1-thermal-direction-gate-v1"
RGB_GROUPS = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


class OgsGateEvidenceError(RuntimeError):
    """Raised when required OGS gate evidence is absent or inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise OgsGateEvidenceError(f"{label} must be numeric, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OgsGateEvidenceError(f"{label} is missing or non-numeric") from exc
    if not math.isfinite(result):
        raise OgsGateEvidenceError(f"{label} is NaN or Inf")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise OgsGateEvidenceError(f"{label} must be an integer, not bool")
    if isinstance(value, str):
        token = value.strip()
        unsigned = token[1:] if token[:1] in ("+", "-") else token
        if not unsigned.isdigit():
            raise OgsGateEvidenceError(f"{label} is missing or non-integer")
        return int(token)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise OgsGateEvidenceError(f"{label} is missing or non-integer") from exc
    try:
        exact = bool(result == value)
    except Exception:
        exact = False
    if not exact:
        raise OgsGateEvidenceError(f"{label} is not an exact integer")
    return result


def _percentile_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise OgsGateEvidenceError("cannot summarize empty or non-finite values")
    return {
        "count": int(array.size),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OgsGateEvidenceError(f"JSON file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OgsGateEvidenceError(f"cannot read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise OgsGateEvidenceError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> Path:
    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"report exists; pass --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _nested(record: Mapping[str, Any], keys: Sequence[str], label: str) -> Any:
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise OgsGateEvidenceError(f"{label} is missing")
        current = current[key]
    return current


def summarize_gradient_smoke(
    log_path: Path,
    *,
    expected_updates: int = 200,
    gate_start: int = 21,
    gate_end: int = 200,
    minimum_ratio: float = 0.02,
    maximum_ratio: float = 0.20,
) -> dict[str, Any]:
    """Validate and summarize the O35 200-update gradient-smoke JSONL."""

    path = log_path.resolve()
    if expected_updates != 200:
        raise OgsGateEvidenceError("OGS-v1 smoke protocol requires exactly 200 updates")
    if (gate_start, gate_end) != (21, 200):
        raise OgsGateEvidenceError(
            "OGS-v1 resolved gate window is fixed to updates 21..200 inclusive"
        )
    if not (0 <= minimum_ratio <= maximum_ratio):
        raise OgsGateEvidenceError("invalid combined-ratio bounds")
    if not path.is_file():
        raise OgsGateEvidenceError(f"gradient smoke log does not exist: {path}")

    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise OgsGateEvidenceError(f"blank JSONL record at line {line_number}")
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise OgsGateEvidenceError(
                f"invalid JSONL record at line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise OgsGateEvidenceError(f"JSONL line {line_number} is not an object")
        records.append(record)

    if len(records) != expected_updates:
        raise OgsGateEvidenceError(
            f"expected {expected_updates} smoke records, found {len(records)}"
        )
    updates = [_integer(row.get("continuation_update"), "continuation_update") for row in records]
    expected_sequence = list(range(1, expected_updates + 1))
    if updates != expected_sequence:
        duplicates = sorted({value for value in updates if updates.count(value) > 1})
        raise OgsGateEvidenceError(
            "smoke updates must appear exactly once in order 1..200; "
            f"duplicates={duplicates}"
        )
    iterations = [_integer(row.get("iteration"), "iteration") for row in records]
    if iterations != list(range(iterations[0], iterations[0] + expected_updates)):
        raise OgsGateEvidenceError("smoke iterations are not consecutive and ordered")
    offsets = {iteration - update for iteration, update in zip(iterations, updates)}
    if len(offsets) != 1:
        raise OgsGateEvidenceError("iteration/update offsets are inconsistent")

    required_series: dict[str, list[float]] = {
        "l_rgb": [],
        "l_ogs": [],
        "weighted_l_ogs": [],
        "combined_ratio": [],
        "scaling_ratio": [],
        "rotation_ratio": [],
        "scaling_cosine": [],
        "rotation_cosine": [],
        "rgb_scaling_gradient_norm": [],
        "rgb_rotation_gradient_norm": [],
        "ogs_scaling_gradient_norm": [],
        "ogs_rotation_gradient_norm": [],
        "weighted_ogs_scaling_gradient_norm": [],
        "weighted_ogs_rotation_gradient_norm": [],
        "rgb_scaling_update_proxy": [],
        "rgb_rotation_update_proxy": [],
        "weighted_ogs_scaling_update_proxy": [],
        "weighted_ogs_rotation_update_proxy": [],
        "rgb_combined_update_proxy": [],
        "weighted_ogs_combined_update_proxy": [],
    }
    active_counts: list[int] = []
    eligible_counts: list[int] = []
    for index, record in enumerate(records, start=1):
        if record.get("schema") != SMOKE_SCHEMA:
            raise OgsGateEvidenceError(f"unexpected smoke schema at update {index}")
        if record.get("finite") is not True:
            raise OgsGateEvidenceError(f"finite flag is not true at update {index}")
        required_series["l_rgb"].append(_finite_number(record.get("l_rgb"), "l_rgb"))
        required_series["l_ogs"].append(_finite_number(record.get("l_ogs"), "l_ogs"))
        required_series["weighted_l_ogs"].append(
            _finite_number(record.get("weighted_l_ogs"), "weighted_l_ogs")
        )
        for name in ("combined_ratio", "scaling_ratio", "rotation_ratio"):
            required_series[name].append(
                _finite_number(
                    _nested(record, ("gradient_probe", "lr_scaled", name), name),
                    name,
                )
            )
        for name in ("scaling_cosine", "rotation_cosine"):
            cosine = _finite_number(
                _nested(record, ("gradient_probe", "raw", name), name), name
            )
            if cosine < -1.000001 or cosine > 1.000001:
                raise OgsGateEvidenceError(f"{name} is outside [-1, 1]")
            required_series[name].append(cosine)
        raw_field_map = {
            "rgb_scaling_gradient_norm": "rgb_scaling_norm",
            "rgb_rotation_gradient_norm": "rgb_rotation_norm",
            "ogs_scaling_gradient_norm": "ogs_scaling_norm",
            "ogs_rotation_gradient_norm": "ogs_rotation_norm",
            "weighted_ogs_scaling_gradient_norm": "weighted_ogs_scaling_norm",
            "weighted_ogs_rotation_gradient_norm": "weighted_ogs_rotation_norm",
        }
        for output_name, log_name in raw_field_map.items():
            required_series[output_name].append(
                _finite_number(
                    _nested(
                        record,
                        ("gradient_probe", "raw", log_name),
                        output_name,
                    ),
                    output_name,
                )
            )
        lr_field_map = {
            "rgb_scaling_update_proxy": "rgb_scaling_update_proxy",
            "rgb_rotation_update_proxy": "rgb_rotation_update_proxy",
            "weighted_ogs_scaling_update_proxy": "weighted_ogs_scaling_update_proxy",
            "weighted_ogs_rotation_update_proxy": "weighted_ogs_rotation_update_proxy",
            "rgb_combined_update_proxy": "rgb_combined_update_proxy",
            "weighted_ogs_combined_update_proxy": "weighted_ogs_combined_update_proxy",
        }
        for output_name, log_name in lr_field_map.items():
            required_series[output_name].append(
                _finite_number(
                    _nested(
                        record,
                        ("gradient_probe", "lr_scaled", log_name),
                        output_name,
                    ),
                    output_name,
                )
            )
        eligible = _integer(record.get("eligible_count"), "eligible_count")
        active = _integer(record.get("active_count"), "active_count")
        if eligible <= 0 or active < 0 or active > eligible:
            raise OgsGateEvidenceError(f"invalid active/eligible count at update {index}")
        eligible_counts.append(eligible)
        active_counts.append(active)
    if len(set(eligible_counts)) != 1:
        raise OgsGateEvidenceError("fixed eligible count changed during smoke")

    # The source text simultaneously says "20..200 inclusive" and "the first
    # 20 are not gated".  The latter is the clearer causal warm-up rule, so the
    # fail-closed resolution excludes updates 1..20 and gates updates 21..200.
    gate_indices = [
        index for index, update in enumerate(updates) if gate_start <= update <= gate_end
    ]
    if len(gate_indices) != gate_end - gate_start + 1:
        raise OgsGateEvidenceError("gate window is incomplete")
    gate_series = {
        name: [values[index] for index in gate_indices]
        for name, values in required_series.items()
    }
    summaries = {
        name: _percentile_summary(values) for name, values in gate_series.items()
    }
    active_gate = [active_counts[index] for index in gate_indices]
    eligible_gate = [eligible_counts[index] for index in gate_indices]
    summaries["active_count"] = _percentile_summary(active_gate)
    summaries["eligible_count"] = _percentile_summary(eligible_gate)
    median_ratio = summaries["combined_ratio"]["p50"]
    decisions = {
        "exact_updates_1_through_200": True,
        "all_required_values_finite": True,
        "fixed_positive_eligible_population": True,
        "combined_ratio_median_gte_minimum": median_ratio >= minimum_ratio,
        "combined_ratio_median_lte_maximum": median_ratio <= maximum_ratio,
    }
    passed = all(decisions.values())
    return {
        "schema": SMOKE_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "source": {"path": str(path), "sha256": _sha256(path)},
        "protocol": {
            "expected_updates": expected_updates,
            "observed_absolute_iterations": [iterations[0], iterations[-1]],
            "continuation_anchor_iteration": next(iter(offsets)),
            "logged_warmup_updates": [1, 20],
            "gate_window_updates_inclusive": [gate_start, gate_end],
            "gate_window_count": len(gate_indices),
            "source_wording_conflict": {
                "numeric_wording": "iterations 20..200 inclusive",
                "causal_warmup_wording": "first 20 logged but excluded from gate",
                "resolved_policy": (
                    "exclude updates 1..20; gate updates 21..200 inclusive "
                    "(180 records)"
                ),
            },
            "combined_ratio_bounds_inclusive": [minimum_ratio, maximum_ratio],
            "quantity_definition": "LR-scaled gradient proxy; not an Adam update",
        },
        "summaries_gate_window": summaries,
        "eligible_count": eligible_counts[0],
        "active_count_first": active_counts[0],
        "active_count_last": active_counts[-1],
        "finite": True,
        "decisions": decisions,
    }


def _load_checkpoint(path: Path, expected_iteration: int, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise OgsGateEvidenceError(f"{label} checkpoint does not exist: {resolved}")
    payload = torch.load(str(resolved), map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise OgsGateEvidenceError(f"{label} is not a (model_params, iteration) checkpoint")
    model, iteration = payload
    if _integer(iteration, f"{label} iteration") != expected_iteration:
        raise OgsGateEvidenceError(
            f"{label} iteration mismatch: expected={expected_iteration} actual={iteration}"
        )
    if not isinstance(model, (tuple, list)) or len(model) < 12:
        raise OgsGateEvidenceError(f"{label} payload does not match Gaussian capture()")
    required = {
        "xyz": (1, 3),
        "f_dc": (2, None),
        "f_rest": (3, None),
        "raw_scaling": (4, 3),
        "raw_rotation": (5, 4),
        "raw_opacity": (6, 1),
    }
    gaussian_count: int | None = None
    tensors: dict[str, torch.Tensor] = {}
    for name, (position, tail) in required.items():
        tensor = model[position]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise OgsGateEvidenceError(f"{label} {name} is not a per-Gaussian tensor")
        if gaussian_count is None:
            gaussian_count = int(tensor.shape[0])
        if int(tensor.shape[0]) != gaussian_count:
            raise OgsGateEvidenceError(f"{label} per-Gaussian tensor counts differ")
        if tail is not None and (tensor.ndim != 2 or int(tensor.shape[1]) != tail):
            raise OgsGateEvidenceError(f"{label} {name} has unexpected shape {tuple(tensor.shape)}")
        if not bool(torch.isfinite(tensor).all()):
            raise OgsGateEvidenceError(f"{label} {name} contains NaN/Inf")
        tensors[name] = tensor.detach().cpu().contiguous()
    if gaussian_count is None or gaussian_count <= 0:
        raise OgsGateEvidenceError(f"{label} has zero Gaussians")

    # Capture-side buffers must preserve the same row cardinality as the model.
    for position, name in ((7, "max_radii2D"), (8, "xyz_gradient_accum"), (9, "denom")):
        tensor = model[position]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1:
            raise OgsGateEvidenceError(f"{label} {name} buffer is malformed")
        if int(tensor.shape[0]) != gaussian_count:
            raise OgsGateEvidenceError(f"{label} {name} count differs from geometry")

    optimizer = model[10]
    if not isinstance(optimizer, Mapping):
        raise OgsGateEvidenceError(f"{label} checkpoint has no optimizer state")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list):
        raise OgsGateEvidenceError(f"{label} optimizer param_groups are malformed")
    group_names = tuple(group.get("name") for group in groups)
    if group_names != RGB_GROUPS:
        raise OgsGateEvidenceError(
            f"{label} optimizer is not the six-group RGB optimizer: {group_names}"
        )
    return {
        "path": resolved,
        "sha256": _sha256(resolved),
        "iteration": expected_iteration,
        "gaussian_count": gaussian_count,
        "model": model,
        "tensors": tensors,
        "optimizer_group_names": list(group_names),
    }


def _validate_continuation_protocol(
    path: Path,
    *,
    anchor: Mapping[str, Any],
    current_iteration: int,
    expect_ogs: bool | None,
) -> dict[str, Any]:
    payload = _read_json(path.resolve())
    required_exact = {
        "schema": "uav-tgs-rgb-continuation-protocol-v1",
        "recipe": "fixed_topology",
        "anchor_iteration": anchor["iteration"],
        "final_iteration": current_iteration,
        "topology_fixed": True,
        "densification": False,
        "pruning": False,
        "opacity_reset": False,
        "artifact_save_semantics": "aligned",
        "optimizer_step_at_final_iteration": True,
    }
    for key, expected in required_exact.items():
        if payload.get(key) != expected:
            raise OgsGateEvidenceError(
                f"continuation protocol {key} mismatch: expected={expected!r} "
                f"actual={payload.get(key)!r}"
            )
    if int(payload.get("requested_updates", -1)) != current_iteration - anchor["iteration"]:
        raise OgsGateEvidenceError("continuation protocol update count is inconsistent")
    if int(payload.get("scheduler_horizon", -1)) != anchor["iteration"]:
        raise OgsGateEvidenceError("continuation scheduler horizon changed")
    if expect_ogs is not None and payload.get("ogs_v1") is not expect_ogs:
        raise OgsGateEvidenceError(
            f"continuation OGS mode mismatch: expected={expect_ogs}"
        )
    declared_anchor = Path(str(payload.get("start_checkpoint", ""))).resolve()
    try:
        same_anchor = os.path.samefile(declared_anchor, anchor["path"])
    except OSError:
        same_anchor = declared_anchor == anchor["path"]
    if not same_anchor:
        raise OgsGateEvidenceError(
            "continuation protocol does not name the supplied trusted anchor checkpoint"
        )
    return payload


def _rename_max_keys(summary: Any) -> Any:
    if isinstance(summary, dict):
        return {
            ("max" if key == "p100" else key): _rename_max_keys(value)
            for key, value in summary.items()
        }
    if isinstance(summary, list):
        return [_rename_max_keys(value) for value in summary]
    return summary


def _risk_percentiles(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    if reference.size == 0 or not np.all(np.isfinite(reference)):
        raise OgsGateEvidenceError("eligible risk reference is empty or non-finite")
    ordered = np.sort(reference, kind="stable")
    return 100.0 * np.searchsorted(ordered, query, side="right") / ordered.size


def _read_mapping_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise OgsGateEvidenceError(f"audit mapping CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise OgsGateEvidenceError(f"audit mapping CSV is empty: {path}")
    return rows


def _load_audit_mapping(directory: Path, gaussian_count: int) -> dict[str, Any]:
    root = directory.resolve()
    clamp_path = root / "clamp20_diagnostics.csv"
    control_path = root / "scale_matched_controls.csv"
    clamp_rows = _read_mapping_csv(clamp_path)
    control_rows = _read_mapping_csv(control_path)
    clamp_indices = [_integer(row.get("gaussian_index"), "clamp gaussian_index") for row in clamp_rows]
    if len(set(clamp_indices)) != len(clamp_indices):
        raise OgsGateEvidenceError("clamp audit mapping contains duplicate indices")
    if any(index < 0 or index >= gaussian_count for index in clamp_indices):
        raise OgsGateEvidenceError("clamp audit mapping contains out-of-range indices")
    target_set = set(clamp_indices)
    control_indices: list[int] = []
    for row in control_rows:
        target = _integer(row.get("target_index"), "control target_index")
        control = _integer(row.get("control_index"), "control_index")
        if target not in target_set:
            raise OgsGateEvidenceError("control mapping names a non-clamp target")
        if control < 0 or control >= gaussian_count or control in target_set:
            raise OgsGateEvidenceError("control mapping contains an invalid control index")
        control_indices.append(control)
    return {
        "root": str(root),
        "clamp_csv_sha256": _sha256(clamp_path),
        "control_csv_sha256": _sha256(control_path),
        "clamp_indices": clamp_indices,
        "control_indices": control_indices,
        "clamp_rows": clamp_rows,
        "control_rows": control_rows,
    }


def _group_diagnostics(
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cache: Mapping[str, Any],
    indices: Sequence[int],
) -> dict[str, Any]:
    if not indices:
        raise OgsGateEvidenceError("diagnostic group is empty")
    result = ogs_v1_loss(scales, rotations, cache)
    selected = torch.as_tensor(indices, dtype=torch.long, device=scales.device)
    eligible = cache["eligible_mask"].to(device=scales.device, dtype=torch.bool)
    observability = cache["observability"].to(device=scales.device)
    visible = cache["visible_count"].to(device=scales.device)
    risk = result["risk"].detach().cpu().numpy()
    eligible_risk = risk[eligible.detach().cpu().numpy()]
    selected_risk = risk[np.asarray(indices, dtype=np.int64)]
    risk_pct = _risk_percentiles(eligible_risk, selected_risk)

    def values(tensor: torch.Tensor) -> list[float]:
        return tensor.index_select(0, selected).detach().cpu().to(torch.float64).tolist()

    active = result["active_mask"].index_select(0, selected)
    selected_eligible = eligible.index_select(0, selected)
    return {
        "count": len(indices),
        "unique_count": len(set(indices)),
        "eligible_count": int(selected_eligible.sum().item()),
        "active_count": int(active.sum().item()),
        "active_fraction": float(active.to(torch.float32).mean().item()),
        "visible_count": _percentile_summary(values(visible)),
        "observability": _percentile_summary(values(observability)),
        "q": _percentile_summary(values(result["q_current"])),
        "unweighted_penalty": _percentile_summary(values(result["penalty"])),
        "risk": _percentile_summary(selected_risk.tolist()),
        "risk_percentile": _percentile_summary(risk_pct.tolist()),
    }


def _clamp_control_diagnostics(
    mapping_directory: Path,
    *,
    anchor_scales: torch.Tensor,
    anchor_rotations: torch.Tensor,
    current_scales: torch.Tensor,
    current_rotations: torch.Tensor,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    mapping = _load_audit_mapping(mapping_directory, int(anchor_scales.shape[0]))
    clamp = mapping["clamp_indices"]
    controls = mapping["control_indices"]
    static: dict[str, Any] = {}
    for name, rows in (
        ("clamp20", mapping["clamp_rows"]),
        ("scale_matched_controls", mapping["control_rows"]),
    ):
        eigengap = []
        for row in rows:
            value = row.get("weak_eigengap")
            if value not in (None, ""):
                eigengap.append(_finite_number(value, f"{name} weak_eigengap"))
        if eigengap:
            static[name] = {"weak_eigengap": _percentile_summary(eigengap)}
    return {
        "mapping": {
            key: value
            for key, value in mapping.items()
            if key not in ("clamp_rows", "control_rows", "clamp_indices", "control_indices")
        },
        "index_counts": {
            "clamp20": len(clamp),
            "control_rows": len(controls),
            "unique_controls": len(set(controls)),
        },
        "static_anchor_audit_fields": static,
        "before_anchor": {
            "clamp20": _group_diagnostics(anchor_scales, anchor_rotations, cache, clamp),
            "scale_matched_controls": _group_diagnostics(
                anchor_scales, anchor_rotations, cache, controls
            ),
        },
        "after_current": {
            "clamp20": _group_diagnostics(current_scales, current_rotations, cache, clamp),
            "scale_matched_controls": _group_diagnostics(
                current_scales, current_rotations, cache, controls
            ),
        },
        "use_restriction": (
            "clamp20/control are fixed mechanism diagnostics only; not training, "
            "threshold selection, supervision, or parameter tuning"
        ),
    }


DEFAULT_SCALE_LIMITS = {
    "max_fraction_any_axis_gt_2": 0.01,
    "max_fraction_any_axis_lt_0_5": 0.01,
    "minimum_trace_ratio_p1": 0.10,
    "maximum_trace_ratio_p99": 10.0,
    "minimum_volume_ratio_p1": 0.01,
    "maximum_volume_ratio_p99": 100.0,
    "max_new_extreme_fraction": 0.001,
}


def evaluate_scale_safety(
    *,
    anchor_checkpoint: Path,
    current_checkpoint: Path,
    cache_path: Path,
    continuation_protocol: Path,
    current_iteration: int,
    group: str,
    expect_ogs: bool | None,
    audit_mapping_directory: Path | None = None,
    limits: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Compare one fixed-topology checkpoint to its trusted 30k anchor."""

    cache_probe = load_ogs_cache(cache_path, device="cpu")
    metadata = cache_probe.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise OgsGateEvidenceError("OGS cache metadata is malformed")
    anchor_iteration = _integer(
        metadata.get("checkpoint_iteration"), "cache checkpoint_iteration"
    )
    if anchor_iteration != 30000:
        raise OgsGateEvidenceError("scale audit requires the formal RGB 30000 anchor")
    anchor = _load_checkpoint(anchor_checkpoint, anchor_iteration, "anchor")
    current = _load_checkpoint(current_checkpoint, current_iteration, "current")
    if metadata.get("checkpoint_sha256") != anchor["sha256"]:
        raise OgsGateEvidenceError("trusted anchor SHA-256 differs from fixed OGS cache")
    if int(cache_probe.get("gaussian_count", -1)) != anchor["gaussian_count"]:
        raise OgsGateEvidenceError("OGS cache Gaussian count differs from anchor")
    if current["gaussian_count"] != anchor["gaussian_count"]:
        raise OgsGateEvidenceError("fixed topology Gaussian count changed")
    protocol = _validate_continuation_protocol(
        continuation_protocol,
        anchor=anchor,
        current_iteration=current_iteration,
        expect_ogs=expect_ogs,
    )
    if expect_ogs is True and protocol.get("ogs_cache_sha256") != cache_probe.get(
        "cache_sha256"
    ):
        raise OgsGateEvidenceError("O35 protocol names a different OGS anchor cache")

    anchor_shapes = {
        name: list(tensor.shape) for name, tensor in anchor["tensors"].items()
    }
    current_shapes = {
        name: list(tensor.shape) for name, tensor in current["tensors"].items()
    }
    if current_shapes != anchor_shapes:
        raise OgsGateEvidenceError("per-Gaussian row shapes changed during continuation")

    anchor_scales = torch.exp(anchor["tensors"]["raw_scaling"].to(torch.float64))
    current_scales = torch.exp(current["tensors"]["raw_scaling"].to(torch.float64))
    anchor_rotations = anchor["tensors"]["raw_rotation"].to(torch.float64)
    current_rotations = current["tensors"]["raw_rotation"].to(torch.float64)
    if not bool(torch.isfinite(anchor_scales).all()) or not bool(
        torch.isfinite(current_scales).all()
    ):
        raise OgsGateEvidenceError("activated scales contain NaN/Inf")
    cache = load_ogs_cache(
        cache_path,
        device="cpu",
        expected_gaussian_count=anchor["gaussian_count"],
        expected_metadata={"checkpoint_sha256": anchor["sha256"]},
    )
    anchor_summary = _rename_max_keys(
        summarize_scale_safety(anchor_scales, anchor_scales, anchor_rotations, cache)
    )
    current_summary = _rename_max_keys(
        summarize_scale_safety(anchor_scales, current_scales, current_rotations, cache)
    )

    policy = dict(DEFAULT_SCALE_LIMITS)
    if limits is not None:
        unknown = set(limits) - set(policy)
        if unknown:
            raise OgsGateEvidenceError(f"unknown scale-safety limits: {sorted(unknown)}")
        policy.update({key: float(value) for key, value in limits.items()})
    trace = current_summary["covariance_trace_ratio_percentiles"]
    volume = current_summary["covariance_volume_ratio_percentiles"]
    new_extremes = max(
        0,
        int(current_summary["extreme_ellipsoid_count"])
        - int(anchor_summary["extreme_ellipsoid_count"]),
    )
    new_extreme_fraction = new_extremes / float(anchor["gaussian_count"])
    decisions = {
        "all_values_finite": bool(current_summary["finite"]),
        "fraction_any_axis_gt_2_within_limit": (
            current_summary["fraction_any_axis_gt_2"]
            <= policy["max_fraction_any_axis_gt_2"]
        ),
        "fraction_any_axis_lt_0_5_within_limit": (
            current_summary["fraction_any_axis_lt_0_5"]
            <= policy["max_fraction_any_axis_lt_0_5"]
        ),
        "trace_ratio_p1_above_floor": trace["p1"] >= policy["minimum_trace_ratio_p1"],
        "trace_ratio_p99_below_ceiling": trace["p99"]
        <= policy["maximum_trace_ratio_p99"],
        "volume_ratio_p1_above_floor": volume["p1"]
        >= policy["minimum_volume_ratio_p1"],
        "volume_ratio_p99_below_ceiling": volume["p99"]
        <= policy["maximum_volume_ratio_p99"],
        "new_extreme_fraction_within_limit": new_extreme_fraction
        <= policy["max_new_extreme_fraction"],
    }
    passed = all(decisions.values())
    diagnostic = None
    if audit_mapping_directory is not None:
        diagnostic = _clamp_control_diagnostics(
            audit_mapping_directory,
            anchor_scales=anchor_scales,
            anchor_rotations=anchor_rotations,
            current_scales=current_scales,
            current_rotations=current_rotations,
            cache=cache,
        )
    return {
        "schema": SCALE_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "group": group,
        "anchor_iteration": anchor_iteration,
        "current_iteration": current_iteration,
        "continuation_updates": current_iteration - anchor_iteration,
        "inputs": {
            "anchor_checkpoint": {
                "path": str(anchor["path"]),
                "sha256": anchor["sha256"],
            },
            "current_checkpoint": {
                "path": str(current["path"]),
                "sha256": current["sha256"],
            },
            "ogs_cache": {
                "path": str(cache_path.resolve()),
                "file_sha256": _sha256(cache_path.resolve()),
                "semantic_sha256": cache.get("cache_sha256"),
            },
            "continuation_protocol": {
                "path": str(continuation_protocol.resolve()),
                "sha256": _sha256(continuation_protocol.resolve()),
                "ordered_camera_sha256": protocol.get("ordered_camera_sha256"),
                "camera_sequence_sha256": protocol.get("camera_sequence_sha256"),
                "ogs_v1": protocol.get("ogs_v1"),
            },
        },
        "topology_and_index_contract": {
            "passed": True,
            "gaussian_count": anchor["gaussian_count"],
            "per_gaussian_shapes_equal": True,
            "six_group_rgb_optimizer_present_both": True,
            "topology_fixed_manifest": True,
            "densification_disabled": True,
            "pruning_disabled": True,
            "opacity_reset_disabled": True,
            "index_correspondence_basis": (
                "same trusted capture row order plus fixed-topology training path "
                "with no row-creating, deleting, or reordering operation"
            ),
            "limitation": (
                "checkpoints contain no persistent Gaussian IDs; index continuity "
                "is proven by the enforced execution contract, not inferred from "
                "mutable xyz/appearance values"
            ),
        },
        "anchor_baseline": anchor_summary,
        "current_vs_anchor": current_summary,
        "scale_safety_policy": {
            **policy,
            "purpose": (
                "conservative collapse/inflation evidence gate; not a loss-weight "
                "search and not a lambda-selection target"
            ),
        },
        "new_extreme_ellipsoid_count": new_extremes,
        "new_extreme_ellipsoid_fraction": new_extreme_fraction,
        "decisions": decisions,
        "clamp20_control_diagnostic": diagnostic,
    }


def evaluate_paired_smoke_gate(
    *,
    gradient_report_path: Path,
    r_scale_report_path: Path,
    o_scale_report_path: Path,
    expected_iteration: int = 30200,
) -> dict[str, Any]:
    """Combine O35 gradient and paired R35/O35 scale-safety gate evidence."""

    gradient = _read_json(gradient_report_path.resolve())
    r_scale = _read_json(r_scale_report_path.resolve())
    o_scale = _read_json(o_scale_report_path.resolve())
    if gradient.get("schema") != SMOKE_REPORT_SCHEMA:
        raise OgsGateEvidenceError("unexpected gradient report schema")
    for label, report in (("R35", r_scale), ("O35", o_scale)):
        if report.get("schema") != SCALE_REPORT_SCHEMA:
            raise OgsGateEvidenceError(f"unexpected {label} scale report schema")
        if int(report.get("current_iteration", -1)) != expected_iteration:
            raise OgsGateEvidenceError(f"{label} scale report has wrong iteration")
    r_inputs = r_scale.get("inputs", {})
    o_inputs = o_scale.get("inputs", {})
    if _nested(r_inputs, ("anchor_checkpoint", "sha256"), "R anchor SHA") != _nested(
        o_inputs, ("anchor_checkpoint", "sha256"), "O anchor SHA"
    ):
        raise OgsGateEvidenceError("paired smoke reports do not share one RGB anchor")
    if _nested(r_inputs, ("ogs_cache", "semantic_sha256"), "R cache SHA") != _nested(
        o_inputs, ("ogs_cache", "semantic_sha256"), "O cache SHA"
    ):
        raise OgsGateEvidenceError("paired smoke reports do not share one OGS cache")
    if _nested(
        r_inputs, ("continuation_protocol", "camera_sequence_sha256"), "R camera sequence"
    ) != _nested(
        o_inputs, ("continuation_protocol", "camera_sequence_sha256"), "O camera sequence"
    ):
        raise OgsGateEvidenceError("paired smoke reports used different camera sequences")
    decisions = {
        "o35_gradient_ratio_and_finite_gate_passed": gradient.get("status") == "passed",
        "r35_scale_safety_passed": r_scale.get("status") == "passed",
        "o35_scale_safety_passed": o_scale.get("status") == "passed",
        "shared_anchor_cache_camera_sequence": True,
        "paired_iteration_30200": expected_iteration == 30200,
    }
    passed = all(decisions.values())
    return {
        "schema": PAIRED_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "expected_iteration": expected_iteration,
        "inputs": {
            "gradient_report": {
                "path": str(gradient_report_path.resolve()),
                "sha256": _sha256(gradient_report_path.resolve()),
            },
            "r_scale_report": {
                "path": str(r_scale_report_path.resolve()),
                "sha256": _sha256(r_scale_report_path.resolve()),
            },
            "o_scale_report": {
                "path": str(o_scale_report_path.resolve()),
                "sha256": _sha256(o_scale_report_path.resolve()),
            },
        },
        "decisions": decisions,
        "action": (
            "continue_to_paired_5000_update_R35_O35"
            if passed
            else "stop_without_lambda_grid_and_report_gpt"
        ),
    }


def _bounded_metric(
    group: Mapping[str, Any],
    path: Sequence[str],
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _finite_number(_nested(group, path, label), label)
    if minimum is not None and value < minimum:
        raise OgsGateEvidenceError(f"{label} is below {minimum}")
    if maximum is not None and value > maximum:
        raise OgsGateEvidenceError(f"{label} is above {maximum}")
    return value


def _relative_improvement(reference: float, candidate: float, label: str) -> float:
    if reference <= 0:
        raise OgsGateEvidenceError(f"{label} reference must be positive")
    return (reference - candidate) / reference


def evaluate_rgb_direction_gate(input_path: Path) -> dict[str, Any]:
    """Apply the fixed Phase-C R35/O35 RGB and reference-depth gates."""

    path = input_path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != RGB_INPUT_SCHEMA:
        raise OgsGateEvidenceError("unexpected RGB direction input schema")
    r35 = payload.get("r35")
    o35 = payload.get("o35")
    if not isinstance(r35, Mapping) or not isinstance(o35, Mapping):
        raise OgsGateEvidenceError("RGB input requires r35 and o35 objects")

    def metrics(group: Mapping[str, Any], prefix: str) -> dict[str, float]:
        return {
            "psnr": _bounded_metric(
                group, ("appearance", "psnr"), f"{prefix}.appearance.psnr"
            ),
            "ssim": _bounded_metric(
                group,
                ("appearance", "ssim"),
                f"{prefix}.appearance.ssim",
                minimum=0.0,
                maximum=1.0,
            ),
            "lpips": _bounded_metric(
                group,
                ("appearance", "lpips"),
                f"{prefix}.appearance.lpips",
                minimum=0.0,
            ),
            "front_at_1m": _bounded_metric(
                group,
                ("depth", "front_at_1m"),
                f"{prefix}.depth.front_at_1m",
                minimum=0.0,
                maximum=1.0,
            ),
            "mean_error_m": _bounded_metric(
                group,
                ("depth", "mean_error_m"),
                f"{prefix}.depth.mean_error_m",
                minimum=0.0,
            ),
            "missing_rate": _bounded_metric(
                group,
                ("depth", "missing_rate"),
                f"{prefix}.depth.missing_rate",
                minimum=0.0,
                maximum=1.0,
            ),
        }

    reference = metrics(r35, "r35")
    candidate = metrics(o35, "o35")
    qualitative = payload.get("qualitative_thin_structure_pass")
    if not isinstance(qualitative, bool):
        raise OgsGateEvidenceError(
            "qualitative_thin_structure_pass must be an explicit boolean"
        )
    deltas = {
        "psnr_loss_db": reference["psnr"] - candidate["psnr"],
        "ssim_loss": reference["ssim"] - candidate["ssim"],
        "lpips_increase": candidate["lpips"] - reference["lpips"],
        "front_at_1m_improvement": candidate["front_at_1m"]
        - reference["front_at_1m"],
        "mean_error_relative_improvement": _relative_improvement(
            reference["mean_error_m"],
            candidate["mean_error_m"],
            "mean depth error",
        ),
        "missing_rate_increase": candidate["missing_rate"]
        - reference["missing_rate"],
    }
    decisions = {
        "psnr_loss_lte_0_10_db": deltas["psnr_loss_db"] <= 0.10,
        "ssim_loss_lte_0_003": deltas["ssim_loss"] <= 0.003,
        "lpips_increase_lte_0_005": deltas["lpips_increase"] <= 0.005,
        "geometry_direction_gate": (
            deltas["front_at_1m_improvement"] >= 0.05
            or deltas["mean_error_relative_improvement"] >= 0.05
        ),
        "missing_rate_increase_lte_0_002": deltas["missing_rate_increase"]
        <= 0.002,
        "qualitative_thin_structure_pass": qualitative,
    }
    passed = all(decisions.values())
    return {
        "schema": RGB_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "source": {"path": str(path), "sha256": _sha256(path)},
        "normalized_input": {"r35": reference, "o35": candidate},
        "deltas_o35_vs_r35": deltas,
        "thresholds": {
            "maximum_psnr_loss_db": 0.10,
            "maximum_ssim_loss": 0.003,
            "maximum_lpips_increase": 0.005,
            "minimum_front_at_1m_improvement": 0.05,
            "minimum_mean_error_relative_improvement": 0.05,
            "geometry_rule": "front improvement OR mean-error relative improvement",
            "maximum_missing_rate_increase": 0.002,
            "qualitative_thin_structure_must_pass": True,
        },
        "decisions": decisions,
        "action": (
            "continue_to_phase_d_fixed_f3"
            if passed
            else "stop_before_thermal_and_report_gpt"
        ),
    }


def evaluate_thermal_direction_gate(input_path: Path) -> dict[str, Any]:
    """Apply the fixed Phase-D R35+F3/O35+F3 thermal gates."""

    path = input_path.resolve()
    payload = _read_json(path)
    if payload.get("schema") != THERMAL_INPUT_SCHEMA:
        raise OgsGateEvidenceError("unexpected thermal direction input schema")
    r35 = payload.get("r35_f3")
    o35 = payload.get("o35_f3")
    if not isinstance(r35, Mapping) or not isinstance(o35, Mapping):
        raise OgsGateEvidenceError("thermal input requires r35_f3 and o35_f3 objects")

    def metrics(group: Mapping[str, Any], prefix: str) -> dict[str, float]:
        return {
            "t_psnr": _bounded_metric(
                group, ("appearance", "t_psnr"), f"{prefix}.appearance.t_psnr"
            ),
            "ssim": _bounded_metric(
                group,
                ("appearance", "ssim"),
                f"{prefix}.appearance.ssim",
                minimum=0.0,
                maximum=1.0,
            ),
            "lpips": _bounded_metric(
                group,
                ("appearance", "lpips"),
                f"{prefix}.appearance.lpips",
                minimum=0.0,
            ),
            "temperature_mae_c": _bounded_metric(
                group,
                ("temperature", "mae_c"),
                f"{prefix}.temperature.mae_c",
                minimum=0.0,
            ),
            "temperature_rmse_c": _bounded_metric(
                group,
                ("temperature", "rmse_c"),
                f"{prefix}.temperature.rmse_c",
                minimum=0.0,
            ),
            "off_lut": _bounded_metric(
                group,
                ("temperature", "off_lut"),
                f"{prefix}.temperature.off_lut",
                minimum=0.0,
            ),
        }

    reference = metrics(r35, "r35_f3")
    candidate = metrics(o35, "o35_f3")
    off_lut_policy = payload.get("off_lut_policy")
    if not isinstance(off_lut_policy, Mapping):
        raise OgsGateEvidenceError("thermal input requires an off_lut_policy object")
    if off_lut_policy.get("declared_before_evaluation") is not True:
        raise OgsGateEvidenceError(
            "off_lut_policy.declared_before_evaluation must be true"
        )
    metric_name = off_lut_policy.get("metric_name")
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise OgsGateEvidenceError("off_lut_policy.metric_name must be non-empty")
    maximum_off_lut_absolute = _bounded_metric(
        off_lut_policy,
        ("maximum_absolute_increase",),
        "off_lut_policy.maximum_absolute_increase",
        minimum=0.0,
    )
    maximum_off_lut_relative_raw = off_lut_policy.get("maximum_relative_increase")
    maximum_off_lut_relative = None
    if maximum_off_lut_relative_raw is not None:
        maximum_off_lut_relative = _finite_number(
            maximum_off_lut_relative_raw,
            "off_lut_policy.maximum_relative_increase",
        )
        if maximum_off_lut_relative < 0:
            raise OgsGateEvidenceError(
                "off_lut_policy.maximum_relative_increase must be nonnegative"
            )

    def temperature_change(name: str) -> dict[str, float | bool | None]:
        before = reference[name]
        after = candidate[name]
        increase = after - before
        relative_for_gate = increase / before if before > 0 else (
            0.0 if increase <= 0 else float("inf")
        )
        return {
            "absolute_increase_c": increase,
            "relative_increase": (
                relative_for_gate if math.isfinite(relative_for_gate) else None
            ),
            "pass_absolute_or_relative": increase <= 0.1
            or relative_for_gate <= 0.05,
        }

    mae = temperature_change("temperature_mae_c")
    rmse = temperature_change("temperature_rmse_c")
    off_lut_increase = candidate["off_lut"] - reference["off_lut"]
    off_lut_relative = (
        off_lut_increase / reference["off_lut"]
        if reference["off_lut"] > 0
        else (0.0 if off_lut_increase <= 0 else float("inf"))
    )
    off_lut_absolute_pass = off_lut_increase <= maximum_off_lut_absolute
    off_lut_relative_pass = (
        True
        if maximum_off_lut_relative is None
        else off_lut_relative <= maximum_off_lut_relative
    )
    decisions = {
        "t_psnr_loss_lte_0_15_db": reference["t_psnr"] - candidate["t_psnr"]
        <= 0.15,
        "ssim_loss_lte_0_006": reference["ssim"] - candidate["ssim"] <= 0.006,
        "lpips_increase_lte_0_008": candidate["lpips"] - reference["lpips"]
        <= 0.008,
        "temperature_mae_absolute_or_relative_gate": bool(
            mae["pass_absolute_or_relative"]
        ),
        "temperature_rmse_absolute_or_relative_gate": bool(
            rmse["pass_absolute_or_relative"]
        ),
        "off_lut_declared_absolute_limit_pass": off_lut_absolute_pass,
        "off_lut_declared_relative_limit_pass": off_lut_relative_pass,
    }
    passed = all(decisions.values())
    return {
        "schema": THERMAL_REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "source": {"path": str(path), "sha256": _sha256(path)},
        "normalized_input": {"r35_f3": reference, "o35_f3": candidate},
        "deltas_o35_f3_vs_r35_f3": {
            "t_psnr_loss_db": reference["t_psnr"] - candidate["t_psnr"],
            "ssim_loss": reference["ssim"] - candidate["ssim"],
            "lpips_increase": candidate["lpips"] - reference["lpips"],
            "temperature_mae": mae,
            "temperature_rmse": rmse,
            "off_lut_increase": off_lut_increase,
            "off_lut_relative_increase": (
                off_lut_relative if math.isfinite(off_lut_relative) else None
            ),
        },
        "thresholds": {
            "maximum_t_psnr_loss_db": 0.15,
            "maximum_ssim_loss": 0.006,
            "maximum_lpips_increase": 0.008,
            "temperature_error_rule": (
                "each of MAE and RMSE: absolute increase <=0.1 C OR "
                "relative increase <=5%"
            ),
            "off_lut_policy": {
                "metric_name": metric_name,
                "maximum_absolute_increase": maximum_off_lut_absolute,
                "maximum_relative_increase": maximum_off_lut_relative,
                "rule": (
                    "absolute limit AND relative limit when a relative limit is declared"
                ),
                "declared_before_evaluation": True,
            },
        },
        "decisions": decisions,
        "action": (
            "phase_d_thermal_gate_passed"
            if passed
            else "stop_and_report_gpt_without_parameter_search"
        ),
    }


def _add_common_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="summarize the O35 gradient smoke JSONL")
    smoke.add_argument("--log", required=True, type=Path)
    smoke.add_argument("--minimum-ratio", type=float, default=0.02)
    smoke.add_argument("--maximum-ratio", type=float, default=0.20)
    _add_common_output(smoke)

    scale = commands.add_parser("scale", help="audit one fixed-topology checkpoint")
    scale.add_argument("--anchor-checkpoint", required=True, type=Path)
    scale.add_argument("--current-checkpoint", required=True, type=Path)
    scale.add_argument("--ogs-cache", required=True, type=Path)
    scale.add_argument("--continuation-protocol", required=True, type=Path)
    scale.add_argument("--current-iteration", required=True, type=int, choices=(30200, 35000))
    scale.add_argument("--group", required=True)
    mode = scale.add_mutually_exclusive_group(required=True)
    mode.add_argument("--expect-ogs", action="store_true")
    mode.add_argument("--expect-reference", action="store_true")
    scale.add_argument("--audit-mapping-directory", type=Path)
    for key, value in DEFAULT_SCALE_LIMITS.items():
        scale.add_argument(
            "--" + key.replace("_", "-"),
            dest=key,
            type=float,
            default=value,
        )
    _add_common_output(scale)

    paired = commands.add_parser("paired", help="combine the paired 30200 smoke gates")
    paired.add_argument("--gradient-report", required=True, type=Path)
    paired.add_argument("--r-scale-report", required=True, type=Path)
    paired.add_argument("--o-scale-report", required=True, type=Path)
    _add_common_output(paired)

    rgb = commands.add_parser("rgb", help="apply the fixed Phase-C RGB gate")
    rgb.add_argument("--input", required=True, type=Path)
    _add_common_output(rgb)

    thermal = commands.add_parser(
        "thermal", help="apply the fixed Phase-D thermal gate"
    )
    thermal.add_argument("--input", required=True, type=Path)
    _add_common_output(thermal)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        payload = summarize_gradient_smoke(
            args.log,
            minimum_ratio=args.minimum_ratio,
            maximum_ratio=args.maximum_ratio,
        )
    elif args.command == "scale":
        limits = {key: getattr(args, key) for key in DEFAULT_SCALE_LIMITS}
        payload = evaluate_scale_safety(
            anchor_checkpoint=args.anchor_checkpoint,
            current_checkpoint=args.current_checkpoint,
            cache_path=args.ogs_cache,
            continuation_protocol=args.continuation_protocol,
            current_iteration=args.current_iteration,
            group=args.group,
            expect_ogs=args.expect_ogs,
            audit_mapping_directory=args.audit_mapping_directory,
            limits=limits,
        )
    elif args.command == "paired":
        payload = evaluate_paired_smoke_gate(
            gradient_report_path=args.gradient_report,
            r_scale_report_path=args.r_scale_report,
            o_scale_report_path=args.o_scale_report,
        )
    elif args.command == "rgb":
        payload = evaluate_rgb_direction_gate(args.input)
    else:
        payload = evaluate_thermal_direction_gate(args.input)
    report = _write_json(args.report, payload, args.overwrite)
    print(json.dumps({"status": payload["status"], "report": str(report)}, sort_keys=True))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
