#!/usr/bin/env python3
"""Build a fail-closed train/guard-only conditional-resplat decision receipt.

This sidecar deliberately does *not* open formal-test NPZ files.  It derives
front-error concentration and per-Gaussian diagnostics from train/guard views,
binds the decision to the already frozen SCSP support-anomaly set, and writes a
receipt before test-only final reporting is allowed.

The v2 decision is the exact intersection of the locked SCSP anomaly set and
the train/guard top-0.1% front-responsibility sets.  The sidecar never modifies
a model, split, reference, or dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geometric_repeatability.evaluate_depth_definitions import (
    DEPTH_DEFINITIONS,
    DIAGNOSTIC_DEPTH_SEMANTICS,
    FORMAL_SPLIT_LABELS,
    _as_hw,
    _formal_split_records_by_stem,
    _image_abs_residual,
    _is_sha256,
    _load_bound_index_set,
    _load_json,
    _load_temperature_arrays,
    _manifest_scene,
    _normalized_stem,
    _resolve_view_npz,
    _sha256,
    _temperature_contract,
    _validate_joint_diagnostic_arrays,
    _view_map,
    _view_metadata,
)


PROTOCOL = "uav-tgs-conditional-resplat-decision-v2"
SELECTION_SPLITS = ("train", "guard")
FRONT_SIGNAL = "front_max_contribution"
LOCKED_SCSP_MANIFEST_SHA256 = "a6f95331ceb5f9bc36ad0f3b52fd802ff9bce8cc2a4eb73c00680bef330a84a1"

# Frozen v2 candidate contract.  There are intentionally no CLI knobs.
TOP_001_FRACTION = 0.001
TOP_01_FRACTION = 0.01
DEPTH_MIN_M = 1.0e-6
OPACITY_THRESHOLD = 0.5
# Existing smallest formal front-curve threshold.  Selection mass is only the
# residual *beyond* this material-error boundary; it is not a CLI parameter.
MATERIAL_FRONT_THRESHOLD_M = 0.25


@dataclass(frozen=True)
class SelectionView:
    image_name: str
    split: str
    block: str
    reference_path: Path
    model_path: Path
    expected_shape: tuple[int, int]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _top_responsibility_indices(
    mass: np.ndarray,
    fraction: float,
    *,
    gaussian_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the frozen top-fraction set over valid responsibility entries.

    Gaussian indices are implicit array positions, so exact length is the
    index-validity contract.  Negative responsibility is a protocol error.
    Non-finite entries are excluded from ``N_valid`` and reported.
    """

    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    if values.size != int(gaussian_count):
        raise ValueError(
            "responsibility length does not match Gaussian index space: "
            f"expected={gaussian_count} actual={values.size}"
        )
    negative = np.flatnonzero(values < 0.0)
    if negative.size:
        raise ValueError(
            "front responsibility contains negative entries; first indices="
            + ",".join(str(int(index)) for index in negative[:20])
        )
    valid = np.isfinite(values) & (values >= 0.0)
    valid_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
    n_valid = int(valid_indices.size)
    if n_valid == 0:
        raise ValueError("front responsibility has no finite nonnegative entries")
    k = max(1, int(math.ceil(float(fraction) * float(n_valid))))
    ordered = valid_indices[
        np.lexsort((valid_indices, -values[valid_indices]))
    ].astype(np.int64, copy=False)
    selected = ordered[:k]
    return selected, {
        "n_valid": n_valid,
        "nonfinite_count": int(values.size - n_valid),
        "k": k,
        "fraction": float(fraction),
        "ordering": "responsibility_descending_then_gaussian_index_ascending",
    }


def _top_indices(mass: np.ndarray, fraction: float, *, gaussian_count: int) -> np.ndarray:
    selected, _ = _top_responsibility_indices(
        mass, fraction, gaussian_count=gaussian_count
    )
    return selected


def _mass_share(mass: np.ndarray, indices: np.ndarray) -> float:
    values = np.asarray(mass, dtype=np.float64).reshape(-1)
    total = float(np.sum(np.where(np.isfinite(values) & (values > 0.0), values, 0.0), dtype=np.float64))
    if total <= 0.0 or indices.size == 0:
        return 0.0
    return float(np.sum(values[np.asarray(indices, dtype=np.int64)], dtype=np.float64)) / total


def _concentration(entries: Mapping[str, float]) -> dict[str, Any]:
    ordered = sorted(
        ((str(key), max(0.0, float(value))) for key, value in entries.items()),
        key=lambda item: (-item[1], item[0]),
    )
    total = float(sum(value for _, value in ordered))
    return {
        "entry_count": len(ordered),
        "total_mass": total,
        "top1_share": (ordered[0][1] / total) if ordered and total > 0.0 else 0.0,
        "top_entries": [
            {"label": label, "mass": value, "share": value / total if total > 0.0 else 0.0}
            for label, value in ordered[:20]
        ],
    }


def _overlap_evidence(first: np.ndarray, second: np.ndarray, population: int) -> dict[str, Any]:
    first_set = {int(value) for value in np.asarray(first, dtype=np.int64).tolist()}
    second_set = {int(value) for value in np.asarray(second, dtype=np.int64).tolist()}
    intersection = first_set & second_set
    expected = float(len(first_set) * len(second_set)) / float(population) if population > 0 else 0.0
    enrichment = float(len(intersection)) / expected if expected > 0.0 else 0.0
    return {
        "first_count": len(first_set),
        "second_count": len(second_set),
        "intersection_count": len(intersection),
        "expected_random_intersection": expected,
        "enrichment": enrichment,
        "first_recall": float(len(intersection)) / float(len(first_set)) if first_set else 0.0,
        "second_recall": float(len(intersection)) / float(len(second_set)) if second_set else 0.0,
        "intersection_indices": sorted(intersection),
    }


def _support_overlap(candidate: np.ndarray, support_indices: frozenset[int], population: int) -> dict[str, Any]:
    evidence = _overlap_evidence(candidate, np.fromiter(sorted(support_indices), dtype=np.int64), population)
    return {
        **evidence,
        "candidate_count": evidence["first_count"],
        "support_count": evidence["second_count"],
        "support_recall": evidence["second_recall"],
    }


def _scatter_sum(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = (indices >= 0) & (indices < target.size) & np.isfinite(values) & (values > 0.0)
    if np.any(valid):
        target += np.bincount(indices[valid], weights=values[valid], minlength=target.size)


def _scatter_finite(target: np.ndarray, indices: np.ndarray, values: np.ndarray, valid: np.ndarray) -> None:
    """Scatter signed finite values under an explicit validity mask."""

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = (
        np.asarray(valid, dtype=bool).reshape(-1)
        & (indices >= 0)
        & (indices < target.size)
        & np.isfinite(values)
    )
    if np.any(valid):
        target += np.bincount(indices[valid], weights=values[valid], minlength=target.size)


def _scatter_count(target: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1) & (indices >= 0) & (indices < target.size)
    if np.any(valid):
        target += np.bincount(indices[valid], minlength=target.size).astype(target.dtype, copy=False)


def _increment_unique(target: np.ndarray, indices: np.ndarray, valid: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1) & (indices >= 0) & (indices < target.size)
    if np.any(valid):
        target[np.unique(indices[valid])] += 1


def _new_split_state(gaussian_count: int) -> dict[str, Any]:
    zeros = lambda dtype=np.float64: np.zeros((gaussian_count,), dtype=dtype)
    return {
        "front_mass": zeros(),
        "rgb_abs_mass": zeros(),
        "temperature_abs_mass": zeros(),
        "top1_opacity_contribution": zeros(),
        "top1_projected_footprint_pixels": zeros(np.int64),
        "top1_assigned_view_count": zeros(np.int64),
        "front_assigned_view_count": zeros(np.int64),
        "temperature_observation_weight": zeros(),
        "temperature_observation_sum": zeros(),
        "temperature_observation_sum_sq": zeros(),
        "temperature_observation_pixel_count": zeros(np.int64),
        "temperature_view_mean_count": zeros(np.int64),
        "temperature_view_mean_sum": zeros(),
        "temperature_view_mean_sum_sq": zeros(),
        "raw_front_mass": 0.0,
        "assigned_front_mass": 0.0,
        "view_mass": {},
        "block_mass": {},
        "view_count": 0,
    }


def _accumulate_temperature_view(
    state: MutableMapping[str, Any], indices: np.ndarray, weights: np.ndarray, temperatures: np.ndarray
) -> None:
    valid = (
        (indices >= 0)
        & np.isfinite(weights)
        & (weights > 0.0)
        & np.isfinite(temperatures)
    )
    if not np.any(valid):
        return
    raw_indices = np.asarray(indices[valid], dtype=np.int64)
    raw_weights = np.asarray(weights[valid], dtype=np.float64)
    raw_temperatures = np.asarray(temperatures[valid], dtype=np.float64)
    unique, inverse = np.unique(raw_indices, return_inverse=True)
    weight_sum = np.bincount(inverse, weights=raw_weights)
    weighted_temp = np.bincount(inverse, weights=raw_weights * raw_temperatures)
    view_mean = weighted_temp / weight_sum
    state["temperature_view_mean_count"][unique] += 1
    state["temperature_view_mean_sum"][unique] += view_mean
    state["temperature_view_mean_sum_sq"][unique] += view_mean * view_mean


def collect_selection_signals(
    views: Sequence[SelectionView],
    *,
    gaussian_count: int,
    appearance_modality: str,
    temperature_contract: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Open only train/guard view arrays and accumulate selection diagnostics."""

    state = {split: _new_split_state(gaussian_count) for split in SELECTION_SPLITS}
    for view in views:
        if view.split not in SELECTION_SPLITS:
            # This branch is the central no-test-I/O guarantee.
            continue
        current = state[view.split]
        current["view_count"] += 1
        with np.load(view.reference_path, allow_pickle=False) as reference, np.load(
            view.model_path, allow_pickle=False
        ) as model:
            top_index, top_weight = _validate_joint_diagnostic_arrays(
                model,
                expected_shape=view.expected_shape,
                gaussian_count=gaussian_count,
                label=view.image_name,
            )
            reference_depth = _as_hw(np.asarray(reference["depth"], dtype=np.float64), label="reference depth")
            reference_valid = _as_hw(np.asarray(reference["valid_mask"]), label="reference valid").astype(bool)
            if reference_depth.shape != view.expected_shape or reference_valid.shape != view.expected_shape:
                raise ValueError(f"{view.image_name}: reference/camera dimensions mismatch")
            rendered_depth = _as_hw(
                np.asarray(model[DEPTH_DEFINITIONS["max_contribution"]], dtype=np.float64),
                label="max-contribution depth",
            )
            opacity = _as_hw(np.asarray(model["accumulated_opacity"], dtype=np.float64), label="opacity")
            model_valid = (
                np.isfinite(rendered_depth)
                & (rendered_depth > DEPTH_MIN_M)
                & np.isfinite(opacity)
                & (opacity >= OPACITY_THRESHOLD)
            )
            front_mass = np.where(
                reference_valid & model_valid,
                np.maximum(reference_depth - rendered_depth - MATERIAL_FRONT_THRESHOLD_M, 0.0),
                0.0,
            )
            positive_front = np.isfinite(front_mass) & (front_mass > 0.0)
            assigned_front = positive_front & (top_index >= 0) & (top_index < gaussian_count)
            current["raw_front_mass"] += float(np.sum(front_mass[positive_front], dtype=np.float64))
            current["assigned_front_mass"] += float(np.sum(front_mass[assigned_front], dtype=np.float64))
            _scatter_sum(current["front_mass"], top_index, front_mass)
            assigned_view_mass = float(np.sum(front_mass[assigned_front], dtype=np.float64))
            current["view_mass"][view.image_name] = assigned_view_mass
            current["block_mass"][view.block] = current["block_mass"].get(view.block, 0.0) + assigned_view_mass
            _scatter_sum(current["top1_opacity_contribution"], top_index, top_weight)
            positive_top = np.isfinite(top_weight) & (top_weight > 0.0)
            _scatter_count(current["top1_projected_footprint_pixels"], top_index, positive_top)
            _increment_unique(current["top1_assigned_view_count"], top_index, positive_top)
            _increment_unique(current["front_assigned_view_count"], top_index, assigned_front)

            if appearance_modality == "rgb":
                if "render_rgb" not in model or "target_rgb" not in model:
                    raise KeyError(f"{view.image_name}: RGB bundle lacks render_rgb/target_rgb")
                rgb_mass = _image_abs_residual(model["render_rgb"], model["target_rgb"])
                _scatter_sum(current["rgb_abs_mass"], top_index, rgb_mass)

            temperature_keys = {
                "render_temperature_c",
                "target_temperature_c",
                "temperature_valid_mask",
            }
            if temperature_contract is None and bool(temperature_keys & set(model.files)):
                raise ValueError("Temperature arrays require an explicit temperature-responsibility contract")
            if temperature_contract is not None:
                rendered_temperature, target_temperature, temperature_valid = _load_temperature_arrays(
                    model,
                    temperature_contract,
                    expected_shape=view.expected_shape,
                    label=view.image_name,
                )
                observation_valid = temperature_valid & positive_top
                observation_weight = np.where(observation_valid, top_weight, 0.0)
                temperature_abs_mass = np.where(
                    temperature_valid,
                    np.abs(rendered_temperature.astype(np.float64) - target_temperature.astype(np.float64)),
                    0.0,
                )
                _scatter_sum(current["temperature_abs_mass"], top_index, temperature_abs_mass)
                _scatter_sum(current["temperature_observation_weight"], top_index, observation_weight)
                _scatter_finite(
                    current["temperature_observation_sum"],
                    top_index,
                    observation_weight * target_temperature,
                    observation_valid,
                )
                _scatter_finite(
                    current["temperature_observation_sum_sq"],
                    top_index,
                    observation_weight * target_temperature.astype(np.float64) ** 2,
                    observation_valid,
                )
                _scatter_count(
                    current["temperature_observation_pixel_count"],
                    top_index,
                    observation_valid,
                )
                _accumulate_temperature_view(
                    current,
                    top_index,
                    observation_weight,
                    target_temperature,
                )
    return state


def _finalize_temperature(state: MutableMapping[str, Any]) -> None:
    weight = state["temperature_observation_weight"]
    mean = np.full(weight.shape, np.nan, dtype=np.float64)
    variance = np.full(weight.shape, np.nan, dtype=np.float64)
    valid = weight > 0.0
    mean[valid] = state["temperature_observation_sum"][valid] / weight[valid]
    variance[valid] = np.maximum(
        state["temperature_observation_sum_sq"][valid] / weight[valid] - mean[valid] ** 2,
        0.0,
    )
    view_count = state["temperature_view_mean_count"]
    view_mean = np.full(weight.shape, np.nan, dtype=np.float64)
    view_variance = np.full(weight.shape, np.nan, dtype=np.float64)
    view_valid = view_count > 0
    view_mean[view_valid] = state["temperature_view_mean_sum"][view_valid] / view_count[view_valid]
    view_variance[view_valid] = np.maximum(
        state["temperature_view_mean_sum_sq"][view_valid] / view_count[view_valid]
        - view_mean[view_valid] ** 2,
        0.0,
    )
    state["temperature_observation_mean_c"] = mean
    state["temperature_observation_variance_c2"] = variance
    state["temperature_cross_view_mean_c"] = view_mean
    state["temperature_cross_view_variance_c2"] = view_variance


def compute_fixed_decision(
    states: Mapping[str, Mapping[str, Any]],
    *,
    gaussian_count: int,
    scsp_indices: frozenset[int],
    finite_support_indices: frozenset[int],
    candidate_block_counts: Mapping[int, int],
    support_distances: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Apply the frozen v2 train/guard candidate rule without test data."""

    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive")
    invalid_scsp = sorted(
        int(index)
        for index in scsp_indices
        if int(index) < 0 or int(index) >= int(gaussian_count)
    )
    if invalid_scsp:
        raise ValueError(f"SCSP anomaly indices are outside Gaussian index space: {invalid_scsp[:20]}")

    split_evidence: dict[str, Any] = {}
    top_sets: dict[str, dict[str, np.ndarray]] = {}
    for split in SELECTION_SPLITS:
        state = states[split]
        mass = np.asarray(state["front_mass"], dtype=np.float64)
        top001, top001_contract = _top_responsibility_indices(
            mass, TOP_001_FRACTION, gaussian_count=gaussian_count
        )
        top01, top01_contract = _top_responsibility_indices(
            mass, TOP_01_FRACTION, gaussian_count=gaussian_count
        )
        top_sets[split] = {"top_0.1pct": top001, "top_1pct": top01}
        raw_mass = float(state["raw_front_mass"])
        assigned_mass = float(state["assigned_front_mass"])
        assigned_fraction = assigned_mass / raw_mass if raw_mass > 0.0 else 0.0
        view_concentration = _concentration(state["view_mass"])
        block_concentration = _concentration(state["block_mass"])
        support = _support_overlap(top001, scsp_indices, gaussian_count)
        split_evidence[split] = {
            "raw_front_mass": raw_mass,
            "assigned_front_mass": assigned_mass,
            "assigned_mass_fraction": assigned_fraction,
            "top_0.1pct": {
                **top001_contract,
                "count": int(top001.size),
                "mass_share": _mass_share(mass, top001),
                "indices": top001.tolist(),
                "indices_sha256": _canonical_json_sha256(top001.tolist()),
            },
            "top_1pct": {
                **top01_contract,
                "count": int(top01.size),
                "mass_share": _mass_share(mass, top01),
                "indices_sha256": _canonical_json_sha256(top01.tolist()),
            },
            "view_concentration": view_concentration,
            "block_concentration": block_concentration,
            "scsp_overlap_top_0.1pct": support,
        }

    stability = _overlap_evidence(
        top_sets["train"]["top_0.1pct"],
        top_sets["guard"]["top_0.1pct"],
        gaussian_count,
    )

    train_top = {int(index) for index in top_sets["train"]["top_0.1pct"].tolist()}
    guard_top = {int(index) for index in top_sets["guard"]["top_0.1pct"].tolist()}
    candidates = sorted(scsp_indices.intersection(train_top).intersection(guard_top))

    support_distances = support_distances or {}
    train_views = np.asarray(states["train"]["front_assigned_view_count"], dtype=np.int64)
    guard_views = np.asarray(states["guard"]["front_assigned_view_count"], dtype=np.int64)
    for label, values in (("train", train_views), ("guard", guard_views)):
        if values.size != gaussian_count or bool((values < 0).any()):
            raise ValueError(f"{label} front-view counts do not match the Gaussian index space")

    candidate_rows = []
    finite_distances = []
    for index in candidates:
        distance = float(support_distances.get(index, float("nan")))
        finite_positive_support = (
            index in finite_support_indices and math.isfinite(distance) and distance > 0.0
        )
        if finite_positive_support:
            finite_distances.append(distance)
        candidate_rows.append(
            {
                "gaussian_index": int(index),
                "train_front_mass": float(states["train"]["front_mass"][index]),
                "guard_front_mass": float(states["guard"]["front_mass"][index]),
                "train_front_views": int(train_views[index]),
                "guard_front_views": int(guard_views[index]),
                "combined_front_views": int(train_views[index] + guard_views[index]),
                "train_guard_front_blocks": int(candidate_block_counts.get(index, 0)),
                "local_support_distance_m": distance if finite_positive_support else None,
                "finite_positive_support": bool(finite_positive_support),
            }
        )

    candidate_mass: dict[str, Any] = {}
    for split in SELECTION_SPLITS:
        mass = np.asarray(states[split]["front_mass"], dtype=np.float64)
        valid = np.isfinite(mass) & (mass >= 0.0)
        total = float(np.sum(mass[valid], dtype=np.float64))
        selected_mass = float(
            np.sum(mass[np.asarray(candidates, dtype=np.int64)], dtype=np.float64)
        ) if candidates else 0.0
        candidate_mass[split] = {
            "total_valid_responsibility_mass": total,
            "candidate_responsibility_mass": selected_mass,
            "candidate_mass_share": selected_mass / total if total > 0.0 else 0.0,
        }

    combined_views = [row["combined_front_views"] for row in candidate_rows]
    block_counts = [row["train_guard_front_blocks"] for row in candidate_rows]
    support_summary = {
        "finite_positive_count": len(finite_distances),
        "missing_or_nonpositive_count": len(candidates) - len(finite_distances),
        "minimum_distance_m": min(finite_distances) if finite_distances else None,
        "maximum_distance_m": max(finite_distances) if finite_distances else None,
        "mean_distance_m": (
            float(sum(finite_distances)) / float(len(finite_distances))
            if finite_distances
            else None
        ),
        "used_for_candidate_selection": False,
    }
    view_block_summary = {
        "minimum_combined_front_views": min(combined_views) if combined_views else None,
        "maximum_combined_front_views": max(combined_views) if combined_views else None,
        "minimum_train_guard_front_blocks": min(block_counts) if block_counts else None,
        "maximum_train_guard_front_blocks": max(block_counts) if block_counts else None,
        "used_for_candidate_selection": False,
    }

    conditions = {"nonempty_exact_candidate_set": bool(candidates)}
    triggered = bool(candidates)
    failed = sorted(key for key, value in conditions.items() if not value)
    return {
        "decision": "execute_one_deterministic_resplat" if triggered else "skip_resplat",
        "triggered": triggered,
        "failed_conditions": failed,
        "conditions": conditions,
        "split_evidence": split_evidence,
        "train_guard_top_0.1pct_stability": stability,
        "candidate_rule": {
            "formula": "SCSP_support_anomaly AND train_top_0.1pct AND guard_top_0.1pct",
            "scsp_support_anomaly_count": len(scsp_indices),
            "scsp_support_anomaly_indices_sha256": _canonical_json_sha256(sorted(scsp_indices)),
            "train_top_0.1pct_count": int(top_sets["train"]["top_0.1pct"].size),
            "guard_top_0.1pct_count": int(top_sets["guard"]["top_0.1pct"].size),
            "candidate_count": len(candidates),
            "candidate_indices": candidates,
            "candidate_indices_sha256": _canonical_json_sha256(candidates),
            "responsibility_mass": candidate_mass,
            "support_summary": support_summary,
            "view_block_summary": view_block_summary,
            "candidate_diagnostics": candidate_rows,
        },
        # Retain this alias so old consumers can read a v2 receipt while the
        # candidate_rule object remains the authoritative schema.
        "eligible_candidate_indices": candidates,
    }


def _load_model_properties(model_manifest: Mapping[str, Any], gaussian_count: int) -> dict[str, np.ndarray]:
    from plyfile import PlyData

    identity = model_manifest.get("model_point_cloud")
    if not isinstance(identity, Mapping):
        raise ValueError("Model manifest is missing model_point_cloud identity")
    path = Path(str(identity.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if str(identity.get("sha256", "")).lower() != _sha256(path):
        raise ValueError("Rendered model PLY SHA-256 mismatch")
    if int(identity.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ValueError("Rendered model PLY size mismatch")
    vertex = PlyData.read(str(path))["vertex"]
    if len(vertex.data) != gaussian_count:
        raise ValueError("Rendered model PLY Gaussian count mismatch")
    names = set(vertex.data.dtype.names or ())
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2"}
    if not required <= names:
        raise ValueError(f"Rendered model PLY lacks fields: {sorted(required - names)}")
    xyz = np.stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")], axis=1)
    raw_scale = np.stack(
        [np.asarray(vertex[name], dtype=np.float64) for name in ("scale_0", "scale_1", "scale_2")],
        axis=1,
    )
    raw_opacity = np.asarray(vertex["opacity"], dtype=np.float64)
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(raw_scale)) or not np.all(np.isfinite(raw_opacity)):
        raise ValueError("Rendered model PLY properties contain NaN/Inf")
    activated_scale = np.exp(raw_scale)
    activated_opacity = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -80.0, 80.0)))
    return {
        "xyz": xyz,
        "activated_scale": activated_scale,
        "activated_opacity": activated_opacity,
        "ply_path": np.asarray(str(path)),
    }


def _query_support(
    xyz: np.ndarray,
    indices: np.ndarray,
    *,
    sparse_root: Path,
    scsp_manifest: Mapping[str, Any],
) -> tuple[dict[int, float], dict[str, Any]]:
    import torch

    from tools.build_adaptive_scale_anchor import query_second_support_distance

    support = scsp_manifest.get("sparse_support")
    if not isinstance(support, Mapping):
        raise ValueError("SCSP manifest lacks sparse_support binding")
    voxel_size = float(support.get("voxel_size", 0.0))
    radius = int(support.get("max_voxel_radius", -1))
    if voxel_size <= 0.0 or radius < 0:
        raise ValueError("SCSP sparse-support query parameters are invalid")
    selected_xyz = torch.as_tensor(xyz[indices], dtype=torch.float32, device="cpu")
    distances, observed = query_second_support_distance(
        selected_xyz,
        sparse_root,
        voxel_size,
        radius,
        65536,
    )
    expected_points_sha = str(support.get("points3d_sha256", "")).lower()
    if expected_points_sha != str(observed.get("points3d_sha256", "")).lower():
        raise ValueError("Current sparse support does not match the SCSP-bound points3D hash")
    values = distances.detach().cpu().numpy().astype(np.float64, copy=False)
    return {int(index): float(value) for index, value in zip(indices.tolist(), values.tolist())}, observed


def _candidate_block_counts(views: Sequence[SelectionView], candidate_indices: np.ndarray) -> dict[int, int]:
    candidate = {int(index) for index in np.asarray(candidate_indices, dtype=np.int64).tolist()}
    seen: dict[int, set[str]] = {index: set() for index in candidate}
    if not candidate:
        return {}
    for view in views:
        if view.split not in SELECTION_SPLITS:
            continue
        with np.load(view.reference_path, allow_pickle=False) as reference, np.load(
            view.model_path, allow_pickle=False
        ) as model:
            index = _as_hw(np.asarray(model["top_contributor_index"]), label="top contributor").astype(np.int64)
            depth = _as_hw(
                np.asarray(model[DEPTH_DEFINITIONS["max_contribution"]], dtype=np.float64),
                label="max depth",
            )
            opacity = _as_hw(np.asarray(model["accumulated_opacity"], dtype=np.float64), label="opacity")
            reference_depth = _as_hw(np.asarray(reference["depth"], dtype=np.float64), label="reference depth")
            reference_valid = _as_hw(np.asarray(reference["valid_mask"]), label="reference valid").astype(bool)
            front = (
                reference_valid
                & np.isfinite(depth)
                & (depth > DEPTH_MIN_M)
                & (opacity >= OPACITY_THRESHOLD)
                & ((reference_depth - depth) > MATERIAL_FRONT_THRESHOLD_M)
            )
            for value in np.unique(index[front]).tolist():
                if int(value) in candidate:
                    # A physical route/view block can contain both train and
                    # guard frames.  Count it once rather than once per split.
                    seen[int(value)].add(view.block)
    return {index: len(blocks) for index, blocks in seen.items()}


def _selection_views(
    reference_manifest_path: Path,
    model_manifest_path: Path,
    formal_split_manifest_path: Path,
) -> tuple[list[SelectionView], dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference_manifest = _load_json(reference_manifest_path)
    model_manifest = _load_json(model_manifest_path)
    formal_manifest = _load_json(formal_split_manifest_path)
    scenes = {
        _manifest_scene(reference_manifest, label="reference manifest"),
        _manifest_scene(model_manifest, label="model manifest"),
        _manifest_scene(formal_manifest, label="formal split manifest"),
    }
    if len(scenes) != 1:
        raise ValueError(f"Selection input scene mismatch: {sorted(scenes)}")
    split_sha = _sha256(formal_split_manifest_path)
    if str(reference_manifest.get("depth_semantics", "")) != "metric_camera_z_reference_mesh":
        raise ValueError("Selection reference is not a metric OpenMVS reference-depth manifest")
    reference_binding = reference_manifest.get("all_split_reference_binding")
    if (
        not isinstance(reference_binding, Mapping)
        or set(reference_binding.get("bound_split_labels", [])) != FORMAL_SPLIT_LABELS
        or str(reference_binding.get("formal_split_manifest_sha256", "")).lower() != split_sha
    ):
        raise ValueError("Selection reference lacks an exact all-formal-split binding")
    for manifest, label in ((reference_manifest, "reference"), (model_manifest, "model")):
        identity = manifest.get("formal_split_manifest_identity")
        if not isinstance(identity, Mapping) or str(identity.get("sha256", "")).lower() != split_sha:
            raise ValueError(f"{label}/formal split identity mismatch")
    diagnostic = model_manifest.get("depth_diagnostics")
    if not isinstance(diagnostic, Mapping) or diagnostic.get("enabled") is not True:
        raise ValueError("Model manifest lacks enabled depth diagnostics")
    for key, expected in DIAGNOSTIC_DEPTH_SEMANTICS.items():
        if str(diagnostic.get(key, "")) != expected:
            raise ValueError(f"Model diagnostic semantics mismatch for {key}")
    reference_views = _view_map(reference_manifest, label="reference manifest")
    model_views = _view_map(model_manifest, label="model manifest")
    if set(reference_views) != set(model_views):
        raise ValueError("Reference/model view sets differ")
    if str(reference_manifest.get("camera_set_sha256", "")).lower() != str(
        model_manifest.get("camera_set_sha256", "")
    ).lower():
        raise ValueError("Reference/model camera-set identity mismatch")
    formal_records = _formal_split_records_by_stem(formal_manifest)
    result: list[SelectionView] = []
    matched_record_ids: set[int] = set()
    for image_name in sorted(model_views):
        stem = _normalized_stem(image_name)
        record = formal_records.get(stem)
        if record is None:
            raise ValueError(f"{image_name}: no formal split record")
        matched_record_ids.add(id(record))
        split = str(record.get("split", "")).strip().lower()
        if split not in FORMAL_SPLIT_LABELS:
            raise ValueError(f"{image_name}: invalid split {split!r}")
        model_view = model_views[image_name]
        reference_view = reference_views[image_name]
        if str(model_view.get("bound_split", "")).strip().lower() != split or str(
            reference_view.get("bound_split", "")
        ).strip().lower() != split:
            raise ValueError(f"{image_name}: manifest/formal split disagreement")
        height, width = int(model_view["height"]), int(model_view["width"])
        if (height, width) != (int(reference_view["height"]), int(reference_view["width"])):
            raise ValueError(f"{image_name}: reference/model dimensions differ")
        metadata = _view_metadata(reference_view, model_view, model_manifest, supplemental_view=record)
        if split in SELECTION_SPLITS:
            # Resolve and hash-check only selection-eligible arrays.
            reference_path = _resolve_view_npz(reference_manifest_path, reference_view)
            model_path = _resolve_view_npz(model_manifest_path, model_view)
        else:
            # Deliberately retain no test path that could be opened later.
            reference_path = Path("__formal_test_not_opened__")
            model_path = Path("__formal_test_not_opened__")
        result.append(
            SelectionView(
                image_name=image_name,
                split=split,
                block=metadata["block"],
                reference_path=reference_path,
                model_path=model_path,
                expected_shape=(height, width),
            )
        )
    formal_record_count = sum(isinstance(record, Mapping) for record in formal_manifest.get("records", []))
    if len(result) != formal_record_count or len(matched_record_ids) != formal_record_count:
        raise ValueError("Formal split/model view coverage is not exact")
    return result, reference_manifest, model_manifest, formal_manifest


def run(
    *,
    reference_manifest_path: Path,
    model_manifest_path: Path,
    formal_split_manifest_path: Path,
    scsp_manifest_path: Path,
    sparse_root: Path,
    out_dir: Path,
    temperature_responsibility_manifest_path: Path | None = None,
    support_query: Callable[..., tuple[dict[int, float], dict[str, Any]]] = _query_support,
) -> dict[str, Any]:
    inputs = [reference_manifest_path, model_manifest_path, formal_split_manifest_path, scsp_manifest_path]
    if temperature_responsibility_manifest_path is not None:
        inputs.append(temperature_responsibility_manifest_path)
    for path in inputs:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    scsp_manifest_identity = _identity(scsp_manifest_path.resolve())
    if scsp_manifest_identity["sha256"] != LOCKED_SCSP_MANIFEST_SHA256:
        raise ValueError(
            "SCSP manifest is not the locked formal support-anomaly receipt: "
            f"expected={LOCKED_SCSP_MANIFEST_SHA256} "
            f"actual={scsp_manifest_identity['sha256']}"
        )
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.partial-", dir=str(out_dir.parent)))
    try:
        views, reference_manifest, model_manifest, formal_manifest = _selection_views(
            reference_manifest_path.resolve(),
            model_manifest_path.resolve(),
            formal_split_manifest_path.resolve(),
        )
        gaussian_count = int(model_manifest.get("gaussian_count", 0))
        if gaussian_count <= 0:
            raise ValueError("Model manifest gaussian_count must be positive")
        index_anchor = model_manifest.get("gaussian_index_anchor")
        index_binding = model_manifest.get("gaussian_index_binding")
        if not isinstance(index_anchor, Mapping) or not isinstance(index_binding, Mapping):
            raise ValueError("Model manifest lacks Gaussian index binding")
        anchor_sha = str(index_anchor.get("sha256", "")).lower()
        if (
            not _is_sha256(anchor_sha)
            or index_binding.get("status") != "verified"
            or int(index_binding.get("gaussian_count", -1)) != gaussian_count
        ):
            raise ValueError("Model Gaussian index binding is not verified")
        scsp = _load_bound_index_set(
            scsp_manifest_path.resolve(),
            label="SCSP support-anomaly set",
            explicit_anchor_sha256="",
            model_index_anchor_sha256=anchor_sha,
            gaussian_count=gaussian_count,
        )
        scsp_manifest = _load_json(scsp_manifest_path.resolve())
        if (
            scsp_manifest.get("status") != "passed"
            or scsp_manifest.get("method") != "scsp"
            or scsp_manifest.get("invariants", {}).get("no_training") is not True
        ):
            raise ValueError("SCSP manifest is not a passed, no-training support-anomaly receipt")
        model_views = _view_map(model_manifest, label="model manifest")
        temperature_contract = _temperature_contract(
            temperature_responsibility_manifest_path.resolve()
            if temperature_responsibility_manifest_path is not None
            else None,
            scene_name=_manifest_scene(model_manifest, label="model manifest"),
            model_manifest_path=model_manifest_path.resolve(),
            model_manifest=model_manifest,
            model_views=model_views,
            temperature_fields_present=temperature_responsibility_manifest_path is not None,
            formal_split_sha256=_sha256(formal_split_manifest_path.resolve()),
            formal_split_counts={
                split: sum(
                    1
                    for record in formal_manifest.get("records", [])
                    if isinstance(record, Mapping) and str(record.get("split", "")).strip().lower() == split
                )
                for split in FORMAL_SPLIT_LABELS
            },
        )
        appearance_modality = str(model_manifest.get("appearance_modality", "")).strip().lower()
        if appearance_modality not in {"rgb", "thermal_canonical", "none"}:
            raise ValueError("Unsupported appearance modality")
        states = collect_selection_signals(
            views,
            gaussian_count=gaussian_count,
            appearance_modality=appearance_modality,
            temperature_contract=temperature_contract,
        )
        for state in states.values():
            _finalize_temperature(state)

        top_union: set[int] = set()
        for split in SELECTION_SPLITS:
            top_union.update(
                _top_indices(
                    states[split]["front_mass"],
                    TOP_01_FRACTION,
                    gaussian_count=gaussian_count,
                ).tolist()
            )
        diagnostics_indices = np.asarray(
            sorted(top_union | set(scsp.indices)), dtype=np.int64
        )
        properties = _load_model_properties(model_manifest, gaussian_count)
        support_distance, support_receipt = support_query(
            properties["xyz"],
            diagnostics_indices,
            sparse_root=sparse_root.resolve(),
            scsp_manifest=scsp_manifest,
        )
        finite_support = frozenset(
            index for index, value in support_distance.items() if math.isfinite(value) and value > 0.0
        )
        block_counts = _candidate_block_counts(views, diagnostics_indices)
        decision = compute_fixed_decision(
            states,
            gaussian_count=gaussian_count,
            scsp_indices=scsp.indices,
            finite_support_indices=finite_support,
            candidate_block_counts=block_counts,
            support_distances=support_distance,
        )

        train_top_v2 = set(
            int(index)
            for index in decision["split_evidence"]["train"]["top_0.1pct"]["indices"]
        )
        guard_top_v2 = set(
            int(index)
            for index in decision["split_evidence"]["guard"]["top_0.1pct"]["indices"]
        )
        candidate_v2 = set(
            int(index) for index in decision["candidate_rule"]["candidate_indices"]
        )

        arrays: dict[str, np.ndarray] = {}
        for split in SELECTION_SPLITS:
            for key in (
                "front_mass",
                "rgb_abs_mass",
                "temperature_abs_mass",
                "top1_opacity_contribution",
                "top1_projected_footprint_pixels",
                "top1_assigned_view_count",
                "front_assigned_view_count",
                "temperature_observation_pixel_count",
                "temperature_observation_mean_c",
                "temperature_observation_variance_c2",
                "temperature_view_mean_count",
                "temperature_cross_view_mean_c",
                "temperature_cross_view_variance_c2",
            ):
                arrays[f"{split}__{key}"] = np.asarray(states[split][key])
        np.savez_compressed(temporary / "selection_signals_train_guard_only.npz", **arrays)

        candidate_path = temporary / "candidate_diagnostics_train_guard_only.csv"
        with candidate_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "gaussian_index",
                "scsp_support_anomaly",
                "train_top_0.1pct_v2",
                "guard_top_0.1pct_v2",
                "exact_candidate_v2",
                "local_support_distance_m",
                "activated_opacity",
                "scale_0_m",
                "scale_1_m",
                "scale_2_m",
                "max_scale_m",
                "max_scale_over_support",
                "train_front_mass",
                "guard_front_mass",
                "train_front_views",
                "guard_front_views",
                "train_top1_assigned_views",
                "guard_top1_assigned_views",
                "train_guard_front_blocks",
                "train_projected_footprint_pixels_top1",
                "guard_projected_footprint_pixels_top1",
                "train_opacity_contribution_sum_top1",
                "guard_opacity_contribution_sum_top1",
                "train_rgb_abs_mass_top1_approx",
                "guard_rgb_abs_mass_top1_approx",
                "train_temperature_abs_mass_top1_approx_c",
                "guard_temperature_abs_mass_top1_approx_c",
                "train_temperature_mean_c",
                "guard_temperature_mean_c",
                "train_temperature_variance_c2",
                "guard_temperature_variance_c2",
                "train_temperature_cross_view_variance_c2",
                "guard_temperature_cross_view_variance_c2",
                "train_temperature_view_mean_count",
                "guard_temperature_view_mean_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in diagnostics_indices.tolist():
                distance = float(support_distance.get(index, float("inf")))
                scale = properties["activated_scale"][index]
                writer.writerow(
                    {
                        "gaussian_index": index,
                        "scsp_support_anomaly": index in scsp.indices,
                        "train_top_0.1pct_v2": index in train_top_v2,
                        "guard_top_0.1pct_v2": index in guard_top_v2,
                        "exact_candidate_v2": index in candidate_v2,
                        "local_support_distance_m": distance if math.isfinite(distance) else "",
                        "activated_opacity": float(properties["activated_opacity"][index]),
                        "scale_0_m": float(scale[0]),
                        "scale_1_m": float(scale[1]),
                        "scale_2_m": float(scale[2]),
                        "max_scale_m": float(np.max(scale)),
                        "max_scale_over_support": float(np.max(scale) / distance)
                        if math.isfinite(distance) and distance > 0.0
                        else "",
                        "train_front_mass": float(states["train"]["front_mass"][index]),
                        "guard_front_mass": float(states["guard"]["front_mass"][index]),
                        "train_front_views": int(states["train"]["front_assigned_view_count"][index]),
                        "guard_front_views": int(states["guard"]["front_assigned_view_count"][index]),
                        "train_top1_assigned_views": int(states["train"]["top1_assigned_view_count"][index]),
                        "guard_top1_assigned_views": int(states["guard"]["top1_assigned_view_count"][index]),
                        "train_guard_front_blocks": int(block_counts.get(index, 0)),
                        "train_projected_footprint_pixels_top1": int(
                            states["train"]["top1_projected_footprint_pixels"][index]
                        ),
                        "guard_projected_footprint_pixels_top1": int(
                            states["guard"]["top1_projected_footprint_pixels"][index]
                        ),
                        "train_opacity_contribution_sum_top1": float(
                            states["train"]["top1_opacity_contribution"][index]
                        ),
                        "guard_opacity_contribution_sum_top1": float(
                            states["guard"]["top1_opacity_contribution"][index]
                        ),
                        "train_rgb_abs_mass_top1_approx": float(states["train"]["rgb_abs_mass"][index]),
                        "guard_rgb_abs_mass_top1_approx": float(states["guard"]["rgb_abs_mass"][index]),
                        "train_temperature_abs_mass_top1_approx_c": float(
                            states["train"]["temperature_abs_mass"][index]
                        ),
                        "guard_temperature_abs_mass_top1_approx_c": float(
                            states["guard"]["temperature_abs_mass"][index]
                        ),
                        "train_temperature_mean_c": states["train"]["temperature_observation_mean_c"][index],
                        "guard_temperature_mean_c": states["guard"]["temperature_observation_mean_c"][index],
                        "train_temperature_variance_c2": states["train"][
                            "temperature_observation_variance_c2"
                        ][index],
                        "guard_temperature_variance_c2": states["guard"][
                            "temperature_observation_variance_c2"
                        ][index],
                        "train_temperature_cross_view_variance_c2": states["train"][
                            "temperature_cross_view_variance_c2"
                        ][index],
                        "guard_temperature_cross_view_variance_c2": states["guard"][
                            "temperature_cross_view_variance_c2"
                        ][index],
                        "train_temperature_view_mean_count": int(
                            states["train"]["temperature_view_mean_count"][index]
                        ),
                        "guard_temperature_view_mean_count": int(
                            states["guard"]["temperature_view_mean_count"][index]
                        ),
                    }
                )

        selection_npz = temporary / "selection_signals_train_guard_only.npz"
        receipt = {
            "protocol": PROTOCOL,
            "status": "passed",
            "scene_name": _manifest_scene(model_manifest, label="model manifest"),
            "producer": {
                "tool": "tools/geometric_repeatability/build_conditional_resplat_decision.py",
                "tool_identity": _identity(Path(__file__).resolve()),
            },
            "decision_contract": {
                "version": 2,
                "logic": (
                    "exact set intersection: locked SCSP support anomaly AND "
                    "train top-0.1% front responsibility AND guard top-0.1% "
                    "front responsibility"
                ),
                "constants": {
                    "top_0.1_fraction": TOP_001_FRACTION,
                    "n_valid": (
                        "finite nonnegative responsibility entries whose implicit "
                        "array position is a valid Gaussian index"
                    ),
                    "k_rule": "max(1,ceil(0.001*N_valid))",
                    "ordering": "responsibility descending, Gaussian index ascending",
                    "negative_responsibility": "fail_closed",
                    "nonfinite_responsibility": "excluded_from_N_valid",
                    "locked_scsp_manifest_sha256": LOCKED_SCSP_MANIFEST_SHA256,
                    "material_front_threshold_m": MATERIAL_FRONT_THRESHOLD_M,
                },
                "support_view_block_role": "reported_only_not_candidate_filters",
                "balanced_or_log_iqr_eligibility_used": False,
                "clamp20_used": False,
                "rgb_or_temperature_used_to_trigger": False,
                "front_mass_semantics": (
                    "sum(max(D_ref-D_max_contribution-0.25m,0)); 0.25m is the fixed "
                    "smallest formal front-curve threshold"
                ),
            },
            "selection_policy": {
                "eligible_splits": list(SELECTION_SPLITS),
                "formal_test_npz_open_count_before_receipt": 0,
                "formal_test_metrics_used": False,
                "test_is_final_report_only_after_this_receipt": True,
                "skipped_formal_test_view_count": sum(view.split == "test" for view in views),
            },
            "inputs": {
                "reference_manifest": _identity(reference_manifest_path.resolve()),
                "model_manifest": _identity(model_manifest_path.resolve()),
                "formal_split_manifest": _identity(formal_split_manifest_path.resolve()),
                "scsp_manifest": scsp_manifest_identity,
                "temperature_responsibility_manifest": _identity(
                    temperature_responsibility_manifest_path.resolve()
                )
                if temperature_responsibility_manifest_path is not None
                else None,
                "gaussian_index_anchor_sha256": anchor_sha,
                "gaussian_count": gaussian_count,
                "sparse_support": support_receipt,
            },
            "artifacts": {
                "selection_signals": {
                    "file": selection_npz.name,
                    "sha256": _sha256(selection_npz),
                    "contains_test_arrays": False,
                },
                "candidate_diagnostics": {
                    "file": candidate_path.name,
                    "sha256": _sha256(candidate_path),
                    "contains_test_rows": False,
                },
            },
            "decision": decision,
        }
        # The receipt is intentionally the last published file in the partial tree.
        _atomic_json(temporary / "conditional_resplat_decision_receipt.json", receipt)
        if out_dir.exists():
            out_dir.rmdir()
        os.replace(temporary, out_dir)
        return receipt
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--formal-split-manifest", required=True)
    parser.add_argument("--scsp-manifest", required=True)
    parser.add_argument("--sparse-root", required=True)
    parser.add_argument("--temperature-responsibility-manifest", default="")
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = run(
        reference_manifest_path=Path(args.reference_manifest),
        model_manifest_path=Path(args.model_manifest),
        formal_split_manifest_path=Path(args.formal_split_manifest),
        scsp_manifest_path=Path(args.scsp_manifest),
        sparse_root=Path(args.sparse_root),
        out_dir=Path(args.out_dir),
        temperature_responsibility_manifest_path=(
            Path(args.temperature_responsibility_manifest)
            if str(args.temperature_responsibility_manifest).strip()
            else None
        ),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
