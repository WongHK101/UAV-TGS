#!/usr/bin/env python3
"""Formal OCT test evaluator with byte-exact frozen-training compatibility.

The 30k OCT checkpoints were trained by commit ``1336a18``.  This evaluator is
an evaluation-only sidecar: it never weakens or edits the immutable training
runner.  Before creating output it proves that every source file bound by the
training protocol is byte-identical, validates the exact checkpoint/binding,
pins the train-only hotspot definition, and records its own clean evaluator
provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oct_gs.formal import FormalOCTTargetStore, sha256_file, sha256_json
from oct_gs.losses import OCTLossWeights, oct_rendering_loss
from oct_gs.protocol import (
    OCTStageCostTracker,
    load_oct_checkpoint,
    load_oct_protocol_manifest,
    validate_training_source_provenance,
)
from oct_gs.rendering import OCTRendererContext
from tools import oct_gs_formal as frozen_runner


EVALUATION_SCHEMA = "uav-tgs-oct-formal-evaluation-v2"
EVALUATION_SOURCE_SCHEMA = "uav-tgs-oct-evaluation-source-v1"
COMPATIBILITY_SCHEMA = "uav-tgs-oct-training-evaluation-compatibility-v1"
FROZEN_TRAINING_COMMIT = "1336a18abde68b042f8f5be5d14839e63c814cf9"
FORMAL_CHECKPOINT_STEP = 30_000
FORMAL_HOTSPOT_QUANTILE = 0.95
FORMAL_HOTSPOT_BINS = 65_536
HOTSPOT_EVALUATION_BINS = 4_096
ALLOWED_POST_TRAINING_PATHS = {
    "tests/test_conditional_resplat_decision.py",
    "tests/test_depth_definitions.py",
    "tests/test_evaluate_formal_baseline_hotspots.py",
    "tests/test_oct_gs.py",
    "tests/test_temperature_responsibility_bundle.py",
    "tools/evaluate_oct_gs_formal_v2.py",
    "tools/evaluate_formal_baseline_hotspots.py",
    "tools/geometric_repeatability/build_conditional_resplat_decision.py",
    "tools/geometric_repeatability/build_temperature_responsibility_bundle.py",
    "tools/geometric_repeatability/evaluate_depth_definitions.py",
}


def _git_bytes(arguments: list[str]) -> bytes:
    return frozen_runner._git_bytes(arguments)


def _source_records(paths: list[Path]) -> list[dict[str, Any]]:
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git_bytes(["ls-files", "-z"]).split(b"\0")
        if item
    }
    records: list[dict[str, Any]] = []
    for source in sorted(path.resolve() for path in paths):
        if not source.is_file():
            raise FileNotFoundError(f"evaluation source file is missing: {source}")
        relative = source.relative_to(REPO_ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"evaluation source is not Git-tracked: {relative}")
        data = source.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    records.sort(key=lambda record: record["path"])
    return records


def _evaluation_source_paths() -> list[Path]:
    paths = set(frozen_runner._formal_source_paths())
    paths.update(
        {
            Path(__file__).resolve(),
            (REPO_ROOT / "metrics.py").resolve(),
            (REPO_ROOT / "utils" / "loss_utils.py").resolve(),
            (REPO_ROOT / "utils" / "image_utils.py").resolve(),
            (
                REPO_ROOT / "tools" / "thermal_radiometry" / "evaluate_temperature.py"
            ).resolve(),
            (REPO_ROOT / "tools" / "thermal_radiometry" / "palette_lut.py").resolve(),
        }
    )
    paths.update(
        path.resolve()
        for path in (REPO_ROOT / "lpipsPyTorch").rglob("*.py")
        if path.is_file()
    )
    return sorted(paths)


def _evaluation_source_provenance() -> dict[str, Any]:
    commit = _git_bytes(["rev-parse", "HEAD"]).decode("ascii").strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("formal OCT evaluation requires a full lowercase Git commit")
    status = _git_bytes(["status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        preview = status.decode("utf-8", errors="replace").splitlines()[:10]
        raise RuntimeError(
            "formal OCT evaluation refuses a dirty Git worktree: " + "; ".join(preview)
        )
    files = _source_records(_evaluation_source_paths())
    payload = {
        "schema": EVALUATION_SOURCE_SCHEMA,
        "git_commit": commit,
        "git_clean": True,
        "git_status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "files": files,
        "files_sha256": sha256_json(files),
    }
    return payload


def _changed_paths_since_frozen_training() -> list[str]:
    raw = _git_bytes(
        ["diff", "--name-only", f"{FROZEN_TRAINING_COMMIT}..HEAD", "--"]
    ).decode("utf-8", errors="strict")
    changed = sorted(
        line.strip().replace("\\", "/") for line in raw.splitlines() if line.strip()
    )
    unexpected = sorted(set(changed) - ALLOWED_POST_TRAINING_PATHS)
    if unexpected:
        raise RuntimeError(
            "post-training commit contains non-approved changes; no generic commit "
            f"mismatch waiver is allowed: {unexpected}"
        )
    return changed


def _training_source_compatibility(
    training_protocol: Mapping[str, Any],
) -> dict[str, Any]:
    stored = validate_training_source_provenance(
        training_protocol.get("source_provenance", {})
    )
    if stored["git_commit"] != FROZEN_TRAINING_COMMIT:
        raise RuntimeError(
            "formal evaluation-v2 accepts only the frozen 1336a18 OCT training commit"
        )
    current_files = _source_records(frozen_runner._formal_source_paths())
    current_files_sha256 = sha256_json(current_files)
    if current_files != stored["files"]:
        raise RuntimeError(
            "current frozen-training source bytes differ from the stored OCT protocol"
        )
    if current_files_sha256 != stored["files_sha256"]:
        raise RuntimeError("frozen-training source inventory hash differs")
    changed_paths = _changed_paths_since_frozen_training()
    return {
        "schema": COMPATIBILITY_SCHEMA,
        "status": "passed",
        "generic_commit_mismatch_waiver": False,
        "training_git_commit_exact": True,
        "training_git_commit": stored["git_commit"],
        "training_source_files_byte_exact": True,
        "training_source_files_sha256": stored["files_sha256"],
        "post_training_changed_paths": changed_paths,
        "post_training_changed_paths_allowlisted": True,
        "training_runner_modified": False,
        "evaluation_v2_is_sidecar_only": True,
    }


def _validate_formal_hotspot_threshold(
    payload: Mapping[str, Any], binding: Any
) -> None:
    if (
        payload.get("source_split") != "train"
        or payload.get("test_statistics_used") is not False
    ):
        raise ValueError("formal hotspot threshold must be train-only")
    if float(payload.get("quantile", math.nan)) != FORMAL_HOTSPOT_QUANTILE:
        raise ValueError(
            f"formal hotspot quantile must equal {FORMAL_HOTSPOT_QUANTILE}"
        )
    bins = payload.get("histogram_bins")
    if type(bins) is not int or bins != FORMAL_HOTSPOT_BINS:
        raise ValueError(
            f"formal hotspot histogram_bins must equal {FORMAL_HOTSPOT_BINS}"
        )

    threshold_c = float(payload.get("threshold_c", math.nan))
    if not math.isfinite(threshold_c) or not (
        float(binding.tmin_c) <= threshold_c <= float(binding.tmax_c)
    ):
        raise ValueError("formal hotspot threshold_c is non-finite/outside scene range")
    if payload.get("range_c") != [float(binding.tmin_c), float(binding.tmax_c)]:
        raise ValueError("formal hotspot range_c differs from the exact scene range")
    valid_train_pixels = payload.get("valid_train_pixels")
    if type(valid_train_pixels) is not int or valid_train_pixels <= 0:
        raise ValueError("formal hotspot valid_train_pixels must be a positive integer")


def _exact_display_temperature_c(
    temperature_c: torch.Tensor,
    tmin_c: float,
    tmax_c: float,
) -> torch.Tensor:
    """Recover temperature from the same exact 8-bit bin saved to Hot-Iron."""

    if temperature_c.dtype != torch.float32:
        raise TypeError("exact-display temperature input must be float32")
    low, high = float(tmin_c), float(tmax_c)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("invalid exact-display temperature range")
    position = ((temperature_c - low) / (high - low)).clamp(0.0, 1.0) * 255.0
    index = torch.round(position)
    return temperature_c.new_tensor(low) + index * temperature_c.new_tensor(
        (high - low) / 255.0
    )


def _update_visible_temperature_moments(
    mean_c: torch.Tensor,
    m2_c2: torch.Tensor,
    visible_count: torch.Tensor,
    temperature_c: torch.Tensor,
    visible: torch.Tensor,
) -> None:
    # Vectorized per-Gaussian Welford update over projected-visible views.
    shape = tuple(mean_c.shape)
    values = (m2_c2, visible_count, temperature_c, visible)
    if any(tuple(value.shape) != shape for value in values):
        raise ValueError("per-Gaussian moment tensors must have identical shape")
    if mean_c.dtype != torch.float32 or m2_c2.dtype != torch.float32:
        raise TypeError("per-Gaussian moment accumulators must be float32")
    if temperature_c.dtype != torch.float32 or visible.dtype != torch.bool:
        raise TypeError("temperature/visibility must be float32/bool")
    if visible_count.dtype not in (torch.int32, torch.int64):
        raise TypeError("visible count must be an integer tensor")
    if not bool(visible.any().item()):
        return
    previous_count = visible_count[visible].to(torch.float32)
    next_count = previous_count + 1.0
    samples = temperature_c[visible]
    previous_mean = mean_c[visible]
    delta = samples - previous_mean
    next_mean = previous_mean + delta / next_count
    mean_c[visible] = next_mean
    m2_c2[visible] += delta * (samples - next_mean)
    visible_count[visible] += 1


def _population_variance_from_moments(
    m2_c2: torch.Tensor,
    visible_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(m2_c2.shape) != tuple(visible_count.shape):
        raise ValueError("M2/count shapes differ")
    if m2_c2.dtype != torch.float32 or visible_count.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError("M2/count must be float32/integer")
    valid = visible_count >= 2
    variance = torch.zeros_like(m2_c2)
    variance[valid] = (m2_c2[valid] / visible_count[valid].to(torch.float32)).clamp_min(
        0.0
    )
    return variance, valid


def _histogram_auprc(
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> float | None:
    positive = np.asarray(positive_counts, dtype=np.int64)
    negative = np.asarray(negative_counts, dtype=np.int64)
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ValueError("hotspot histograms must be same-shape vectors")
    if bool((positive < 0).any()) or bool((negative < 0).any()):
        raise ValueError("hotspot histogram counts must be nonnegative")
    positives = int(positive.sum())
    if positives == 0:
        return None
    cumulative_tp = np.cumsum(positive[::-1], dtype=np.int64)
    cumulative_fp = np.cumsum(negative[::-1], dtype=np.int64)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def _occupancy_invariant_evidence(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    ordered_fields = expected.get("ordered_fields")
    if (
        not isinstance(ordered_fields, list)
        or observed.get("ordered_fields") != ordered_fields
    ):
        raise RuntimeError("shared occupancy field order differs")
    expected_fields = expected.get("fields")
    observed_fields = observed.get("fields")
    if not isinstance(expected_fields, Mapping) or not isinstance(
        observed_fields, Mapping
    ):
        raise ValueError("shared occupancy snapshots lack field records")
    field_hashes: dict[str, Any] = {}
    for name in ordered_fields:
        left = expected_fields.get(name)
        right = observed_fields.get(name)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise ValueError(f"shared occupancy snapshot lacks field {name}")
        if left != right:
            raise RuntimeError(f"shared occupancy field differs: {name}")
        field_hashes[str(name)] = {
            "expected_sha256": left.get("sha256"),
            "observed_sha256": right.get("sha256"),
            "exact": True,
        }
    expected_count = int(expected.get("topology_count", -1))
    observed_count = int(observed.get("topology_count", -2))
    if expected_count <= 0 or expected_count != observed_count:
        raise RuntimeError("shared occupancy topology count differs")
    topology_basis = {
        "topology_count": expected_count,
        "ordered_fields": ordered_fields,
    }
    expected_overall = expected.get("overall_sha256")
    observed_overall = observed.get("overall_sha256")
    if not isinstance(expected_overall, str) or expected_overall != observed_overall:
        raise RuntimeError("shared occupancy overall digest differs")
    return {
        "status": "passed",
        "exact": True,
        "topology_count": expected_count,
        "topology_sha256": sha256_json(topology_basis),
        "expected_overall_sha256": expected_overall,
        "observed_overall_sha256": observed_overall,
        "ordered_fields": list(ordered_fields),
        "field_hashes": field_hashes,
    }


def _load_and_validate_endpoint_receipt(
    *,
    endpoint_receipt_path: Path,
    checkpoint_path: Path,
    protocol_path: Path,
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    binding: Any,
    runtime: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    receipt_path = endpoint_receipt_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    protocol_path = protocol_path.resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(f"formal endpoint receipt is missing: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("formal endpoint receipt must contain a JSON object")
    if payload.get("schema") != frozen_runner.ENDPOINT_SCHEMA:
        raise ValueError("formal endpoint receipt schema mismatch")
    supplied_endpoint_sha = payload.get("endpoint_sha256")
    basis = dict(payload)
    basis.pop("endpoint_sha256", None)
    if supplied_endpoint_sha != sha256_json(basis):
        raise ValueError("formal endpoint receipt self-hash mismatch")
    if (
        payload.get("step") != FORMAL_CHECKPOINT_STEP
        or payload.get("sequence_offset") != FORMAL_CHECKPOINT_STEP
    ):
        raise ValueError("formal endpoint receipt is not the final 30k endpoint")

    run_root = receipt_path.parent.parent
    if (
        receipt_path.parent.name != "endpoints"
        or receipt_path.name != f"step_{FORMAL_CHECKPOINT_STEP}.json"
        or checkpoint_path.parent.name != "checkpoints"
        or checkpoint_path.name != f"step_{FORMAL_CHECKPOINT_STEP}.pt"
        or checkpoint_path.parent.parent != run_root
        or protocol_path != run_root / "protocol.json"
    ):
        raise ValueError(
            "endpoint/checkpoint/protocol paths do not identify one formal run root"
        )

    actual_checkpoint_sha = sha256_file(checkpoint_path)
    if payload.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise RuntimeError("endpoint receipt/checkpoint file SHA-256 mismatch")
    if checkpoint.get("checkpoint_sha256") != actual_checkpoint_sha:
        raise RuntimeError("loaded checkpoint identity differs from endpoint receipt")
    checkpoint_protocol = checkpoint.get("protocol_receipt")
    if not isinstance(checkpoint_protocol, Mapping):
        raise ValueError("checkpoint lacks immutable protocol receipt")
    if protocol.get("manifest_file_sha256") != sha256_file(protocol_path):
        raise ValueError("current protocol file SHA differs from its validated record")
    if payload.get("protocol_manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("endpoint logical protocol SHA differs")
    if checkpoint_protocol.get("manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("checkpoint logical protocol SHA differs")
    if checkpoint_protocol.get("manifest_file_sha256") != protocol.get(
        "manifest_file_sha256"
    ):
        raise ValueError("checkpoint protocol file SHA differs")
    if payload.get("formal_protocol_sha256") != binding.formal_protocol_sha256:
        raise ValueError("endpoint formal binding SHA differs")
    if (
        checkpoint_protocol.get("formal_protocol_sha256")
        != binding.formal_protocol_sha256
    ):
        raise ValueError("checkpoint formal binding SHA differs")

    anchor_sha = runtime.anchor_snapshot.get("overall_sha256")
    if payload.get("anchor_occupancy_sha256") != anchor_sha:
        raise ValueError("endpoint anchor occupancy SHA differs")
    if checkpoint_protocol.get("anchor_occupancy_sha256") != anchor_sha:
        raise ValueError("checkpoint anchor occupancy SHA differs")
    if protocol.get("anchor_snapshot", {}).get("overall_sha256") != anchor_sha:
        raise ValueError("protocol anchor occupancy SHA differs")
    training_source = validate_training_source_provenance(
        protocol.get("source_provenance", {})
    )
    if payload.get("source_files_sha256") != training_source["files_sha256"]:
        raise ValueError("endpoint frozen-training source inventory SHA differs")

    endpoint_cost = payload.get("cost")
    checkpoint_cost = checkpoint.get("cost_summary")
    if not isinstance(endpoint_cost, Mapping) or endpoint_cost != checkpoint_cost:
        raise ValueError("endpoint/checkpoint training cost receipt mismatch")
    if endpoint_cost.get("schema") != "uav-tgs-oct-cost-v1":
        raise ValueError("endpoint training cost schema mismatch")
    if endpoint_cost.get("status") != f"endpoint_{FORMAL_CHECKPOINT_STEP}":
        raise ValueError("endpoint training cost status is not final")
    if endpoint_cost.get("cumulative_optimizer_steps") != FORMAL_CHECKPOINT_STEP:
        raise ValueError("endpoint cumulative optimizer step cost is not 30k")
    segment_start = endpoint_cost.get("segment_start_step")
    segment_steps = endpoint_cost.get("segment_optimizer_steps")
    if (
        type(segment_start) is not int
        or type(segment_steps) is not int
        or segment_start < 0
        or segment_steps <= 0
        or segment_start + segment_steps != FORMAL_CHECKPOINT_STEP
    ):
        raise ValueError("endpoint training segment cost does not terminate at 30k")
    if (
        endpoint_cost.get("optimizer_steps") != segment_steps
        or endpoint_cost.get("rendered_views") != segment_steps
        or endpoint_cost.get("raster_passes") != segment_steps
        or endpoint_cost.get("raster_passes_per_view") != 1.0
    ):
        raise ValueError("endpoint training cost step/view/raster accounting differs")
    for key in (
        "wall_time_s",
        "end_to_end_wall_time_s",
        "pre_training_setup_wall_time_s",
        "ms_per_step",
    ):
        value = endpoint_cost.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"endpoint training cost {key} is not finite/nonnegative")
    if float(endpoint_cost["end_to_end_wall_time_s"]) < float(
        endpoint_cost["wall_time_s"]
    ):
        raise ValueError("endpoint end-to-end cost is shorter than training wall time")
    if endpoint_cost.get("peak_memory_reset_succeeded") is not True:
        raise ValueError("endpoint CUDA peak-memory reset was not confirmed")
    device = endpoint_cost.get("device")
    if not isinstance(device, Mapping) or device.get("cuda_available") is not True:
        raise ValueError("endpoint training cost lacks a CUDA device receipt")
    if not isinstance(device.get("device_name"), str) or not device["device_name"]:
        raise ValueError("endpoint training cost lacks a CUDA device name")
    allocated = device.get("peak_torch_allocated_bytes")
    reserved = device.get("peak_torch_reserved_bytes")
    if (
        type(allocated) is not int
        or type(reserved) is not int
        or allocated < 0
        or reserved < allocated
    ):
        raise ValueError("endpoint CUDA peak-memory accounting is invalid")
    cost_metadata = endpoint_cost.get("metadata")
    if not isinstance(cost_metadata, Mapping):
        raise ValueError("endpoint training cost metadata is missing")
    if (
        cost_metadata.get("formal_protocol_sha256")
        != binding.formal_protocol_sha256
        or cost_metadata.get("scene") != protocol.get("scene_name")
        or cost_metadata.get("variant") != protocol.get("variant")
        or cost_metadata.get("segment_start_step") != segment_start
    ):
        raise ValueError("endpoint training cost metadata differs from formal protocol")
    recent_loss = float(payload.get("recent_loss_mean", math.nan))
    if not math.isfinite(recent_loss):
        raise ValueError("endpoint recent_loss_mean is non-finite")

    identity = {
        "path": str(receipt_path),
        "file_sha256": sha256_file(receipt_path),
        "endpoint_sha256": supplied_endpoint_sha,
        "run_root": str(run_root),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha,
        "protocol_path": str(protocol_path),
        "protocol_manifest_file_sha256": protocol["manifest_file_sha256"],
        "protocol_manifest_sha256": protocol["manifest_sha256"],
    }
    flags = {
        "endpoint_receipt_schema_exact": True,
        "endpoint_receipt_self_hash_exact": True,
        "endpoint_receipt_step_sequence_exact_30000": True,
        "endpoint_checkpoint_path_identity_exact": True,
        "endpoint_checkpoint_sha256_exact": True,
        "endpoint_protocol_file_logical_sha256_exact": True,
        "endpoint_formal_binding_sha256_exact": True,
        "endpoint_anchor_occupancy_sha256_exact": True,
        "endpoint_source_files_sha256_exact": True,
        "endpoint_checkpoint_training_cost_exact": True,
    }
    return payload, identity, flags


def _checkpoint_compatibility(
    *,
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    binding: Any,
    runtime: Any,
    variant: str,
) -> dict[str, Any]:
    if int(checkpoint.get("step", -1)) != FORMAL_CHECKPOINT_STEP:
        raise ValueError("formal evaluation-v2 requires the final 30k checkpoint")
    if checkpoint.get("sequence_offset") != FORMAL_CHECKPOINT_STEP:
        raise ValueError("checkpoint sequence offset is not the final 30k step")
    if protocol.get("scene_name") != binding.scene_name:
        raise ValueError("checkpoint protocol scene differs from formal binding")
    if protocol.get("variant") != variant:
        raise ValueError("checkpoint protocol variant differs from request")
    if checkpoint.get("field_config") != protocol.get("field", {}).get("config"):
        raise ValueError("checkpoint field config differs from protocol")
    if checkpoint.get("anchor_snapshot") != runtime.anchor_snapshot:
        raise ValueError("checkpoint anchor snapshot differs from current exact anchor")
    receipt = checkpoint.get("protocol_receipt")
    if not isinstance(receipt, Mapping):
        raise ValueError("checkpoint lacks immutable protocol receipt")
    if receipt.get("manifest_file_sha256") != protocol.get("manifest_file_sha256"):
        raise ValueError("checkpoint protocol file SHA differs")
    if receipt.get("manifest_sha256") != protocol.get("manifest_sha256"):
        raise ValueError("checkpoint logical protocol SHA differs")
    if receipt.get("formal_protocol_sha256") != binding.formal_protocol_sha256:
        raise ValueError("checkpoint formal binding SHA differs")
    return {
        "checkpoint_step_exact_30000": True,
        "checkpoint_sequence_offset_exact": True,
        "protocol_manifest_file_sha256_exact": True,
        "protocol_manifest_logical_sha256_exact": True,
        "formal_binding_sha256_exact": True,
        "anchor_snapshot_exact": True,
        "field_config_exact": True,
    }


def command_eval_v2(args: argparse.Namespace) -> int:
    process_start_perf = time.perf_counter()
    if int(args.resolution) != -1:
        raise ValueError("formal OCT test evaluation-v2 is fixed to r1/full-resolution")

    # All gates below complete before output_root is created or any metric is
    # written.  Training/evaluation commit divergence is accepted only through
    # explicit byte equality plus the narrow changed-path allowlist above.
    evaluation_source = _evaluation_source_provenance()
    protocol_path = Path(args.protocol_manifest).resolve()
    protocol = load_oct_protocol_manifest(protocol_path)
    compatibility = _training_source_compatibility(protocol)
    runtime = frozen_runner.Runtime(args)
    binding = runtime.binding()
    variant = str(args.variant)
    checkpoint_path = Path(args.checkpoint).resolve()
    field, checkpoint = load_oct_checkpoint(
        checkpoint_path,
        anchor=runtime.anchor,
        protocol_manifest=protocol_path,
        formal_binding=binding,
        map_location="cpu",
    )
    if field.config.variant != variant:
        raise ValueError("checkpoint variant differs from requested evaluation variant")
    checkpoint_flags = _checkpoint_compatibility(
        checkpoint=checkpoint,
        protocol=protocol,
        binding=binding,
        runtime=runtime,
        variant=variant,
    )
    threshold = frozen_runner._load_hotspot_threshold(
        args.hotspot_threshold_manifest, binding
    )
    _validate_formal_hotspot_threshold(threshold, binding)
    endpoint_receipt, endpoint_identity, endpoint_flags = (
        _load_and_validate_endpoint_receipt(
            endpoint_receipt_path=Path(args.endpoint_receipt),
            checkpoint_path=checkpoint_path,
            protocol_path=protocol_path,
            checkpoint=checkpoint,
            protocol=protocol,
            binding=binding,
            runtime=runtime,
        )
    )
    compatibility.update(checkpoint_flags)
    compatibility.update(endpoint_flags)
    compatibility.update(
        {
            "hotspot_threshold_train_only": True,
            "hotspot_quantile_exact_0_95": True,
            "hotspot_histogram_bins_exact_65536": True,
            "evaluation_git_commit": evaluation_source["git_commit"],
            "evaluation_source_files_sha256": evaluation_source["files_sha256"],
        }
    )
    # Evaluation never consumes the serialized Adam moments.  Drop them before
    # moving the compact thermometric field to CUDA.
    checkpoint.pop("optimizer_state_dict", None)

    frozen_runner._require_isolated_output(
        args.eval_output, args, label="formal OCT evaluation-v2 output"
    )
    output_root = frozen_runner._require_fresh_directory(
        args.eval_output, "OCT evaluation-v2 output"
    )

    field = field.cuda().eval()
    hotspot_threshold = float(threshold["threshold_c"])
    targets = FormalOCTTargetStore(binding)
    targets.preload(binding.names("test"), evaluation_support=True)
    context = OCTRendererContext(runtime.anchor, runtime.proxy)
    tracker = OCTStageCostTracker(
        {"scene": binding.scene_name, "variant": variant, "mode": "test-r1-v2"}
    )
    tracker.start()
    rows: list[dict[str, Any]] = []
    absolute_errors: list[np.ndarray] = []
    signed_sum = squared_sum = 0.0
    valid_count = 0
    hotspot_intersection = hotspot_union = 0
    direct_hotspot_intersection = direct_hotspot_union = 0
    hotspot_pos = np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    hotspot_neg = np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    direct_hotspot_pos = np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    direct_hotspot_neg = np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    gaussian_mean = torch.zeros(field.num_gaussians, device="cuda", dtype=torch.float32)
    gaussian_m2 = torch.zeros_like(gaussian_mean)
    gaussian_visible_count = torch.zeros(
        field.num_gaussians, device="cuda", dtype=torch.int32
    )
    render_time_ms: list[float] = []
    appearance_method = f"{variant}_step_{int(checkpoint['step'])}_eval_v2"
    appearance_root = output_root / "formal_appearance"
    appearance_renders = appearance_root / "test" / appearance_method / "renders"
    appearance_gt = appearance_root / "test" / appearance_method / "gt"
    expected_appearance_names: list[str] = []

    with torch.no_grad():
        for camera in runtime.test_cameras:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()
            output = context.render(camera, field, runtime.pipe, exact_display=True)
            render_end.record()
            render_end.synchronize()
            render_time_ms.append(float(render_start.elapsed_time(render_end)))
            height, width = output["temperature_c"].shape[-2:]
            record = binding.by_camera[str(camera.image_name)]
            if (height, width) != record.shape_hw:
                raise ValueError("formal test render is not native target resolution")
            target_t, target_rgb, support = targets.get(
                camera.image_name,
                height,
                width,
                "cuda",
                evaluation_support=True,
            )
            valid = support[0]
            direct_temperature = output["temperature_c"][valid]
            signed = direct_temperature - target_t[0][valid]
            if not bool(torch.isfinite(signed).all().item()):
                raise FloatingPointError("non-finite formal test temperature")
            abs_cpu = signed.abs().float().cpu().numpy()
            absolute_errors.append(abs_cpu)
            signed_sum += float(signed.double().sum().item())
            squared_sum += float(signed.double().square().sum().item())
            valid_count += int(signed.numel())
            color_error = (output["hot_iron"] - target_rgb)[:, valid]
            mse = float(color_error.double().square().mean().item())
            metric_loss = oct_rendering_loss(
                output["temperature_c"],
                target_t,
                output["hot_iron"],
                radiance_proxy=runtime.proxy,
                target_hot_iron=target_rgb,
                mask=support,
                weights=OCTLossWeights(1.0, 0.0, 1.0, 0.0),
                thermometric_domain="celsius",
            )

            comparable_t = _exact_display_temperature_c(
                output["temperature_c"], binding.tmin_c, binding.tmax_c
            )[valid]
            target_hot = target_t[0][valid] >= hotspot_threshold
            comparable_hot = comparable_t >= hotspot_threshold
            hotspot_intersection += int((comparable_hot & target_hot).sum().item())
            hotspot_union += int((comparable_hot | target_hot).sum().item())
            comparable_score = (
                (comparable_t - binding.tmin_c) / (binding.tmax_c - binding.tmin_c)
            ).clamp(0.0, 1.0)
            comparable_index = torch.clamp(
                (comparable_score * (HOTSPOT_EVALUATION_BINS - 1)).long(),
                0,
                HOTSPOT_EVALUATION_BINS - 1,
            )
            hotspot_pos += (
                torch.bincount(
                    comparable_index[target_hot], minlength=HOTSPOT_EVALUATION_BINS
                )
                .cpu()
                .numpy()
            )
            hotspot_neg += (
                torch.bincount(
                    comparable_index[~target_hot], minlength=HOTSPOT_EVALUATION_BINS
                )
                .cpu()
                .numpy()
            )

            direct_hot = direct_temperature >= hotspot_threshold
            direct_hotspot_intersection += int((direct_hot & target_hot).sum().item())
            direct_hotspot_union += int((direct_hot | target_hot).sum().item())
            direct_score = (
                (direct_temperature - binding.tmin_c)
                / (binding.tmax_c - binding.tmin_c)
            ).clamp(0.0, 1.0)
            direct_index = torch.clamp(
                (direct_score * (HOTSPOT_EVALUATION_BINS - 1)).long(),
                0,
                HOTSPOT_EVALUATION_BINS - 1,
            )
            direct_hotspot_pos += (
                torch.bincount(
                    direct_index[target_hot], minlength=HOTSPOT_EVALUATION_BINS
                )
                .cpu()
                .numpy()
            )
            direct_hotspot_neg += (
                torch.bincount(
                    direct_index[~target_hot], minlength=HOTSPOT_EVALUATION_BINS
                )
                .cpu()
                .numpy()
            )

            gaussian_t = output["gaussian_temperature_c"][:, 0]
            radii = output.get("radii")
            if not isinstance(radii, torch.Tensor) or radii.shape != gaussian_t.shape:
                raise RuntimeError("formal OCT evaluation requires per-Gaussian radii")
            _update_visible_temperature_moments(
                gaussian_mean,
                gaussian_m2,
                gaussian_visible_count,
                gaussian_t,
                radii > 0,
            )
            tracker.record_step(raster_passes=int(output["raster_passes"]))
            stem = Path(camera.image_name).stem
            render_t = (
                output["temperature_c"]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            (output_root / "temperature_c").mkdir(parents=True, exist_ok=True)
            np.save(
                output_root / "temperature_c" / f"{stem}.npy",
                render_t,
                allow_pickle=False,
            )
            appearance_name = f"{stem}.png"
            frozen_runner._save_exact_png(
                appearance_renders / appearance_name, output["hot_iron"]
            )
            frozen_runner._copy_immutable_reference(
                record.canonical_path, appearance_gt / appearance_name
            )
            expected_appearance_names.append(appearance_name)
            rows.append(
                {
                    "camera_name": camera.image_name,
                    "mae_c": float(np.mean(abs_cpu)),
                    "rmse_c": float(
                        np.sqrt(np.mean(np.square(abs_cpu, dtype=np.float64)))
                    ),
                    "bias_c": float(signed.double().mean().item()),
                    "p95_abs_c": float(np.quantile(abs_cpu, 0.95)),
                    "support_masked_psnr_db": (
                        None if mse == 0.0 else float(-10.0 * math.log10(mse))
                    ),
                    "support_masked_ssim": float(
                        1.0 - metric_loss["color_dssim"].item()
                    ),
                    "valid_pixels": int(signed.numel()),
                }
            )

    if valid_count <= 0:
        raise ValueError("formal test support contains no valid pixels")
    all_abs = np.concatenate(absolute_errors)
    auprc = _histogram_auprc(hotspot_pos, hotspot_neg)
    direct_auprc = _histogram_auprc(direct_hotspot_pos, direct_hotspot_neg)
    variance, variance_valid = _population_variance_from_moments(
        gaussian_m2, gaussian_visible_count
    )
    measured_variance = variance[variance_valid]
    if measured_variance.numel() == 0:
        raise ValueError("no Gaussian is visible in at least two formal test views")
    if variant == "oct_scalar" and not bool(
        torch.equal(measured_variance, torch.zeros_like(measured_variance))
    ):
        raise RuntimeError("OCT-Scalar Welford variance is not bit-exact zero")

    observed_occupancy = context.verify_anchor_unchanged()
    occupancy_evidence = _occupancy_invariant_evidence(
        runtime.anchor_snapshot, observed_occupancy
    )
    appearance = frozen_runner._run_formal_appearance_evaluator(
        appearance_root,
        method_name=appearance_method,
        expected_names=expected_appearance_names,
    )
    palette_temperature = frozen_runner._run_formal_temperature_evaluator(
        output_root,
        render_root=appearance_renders,
        ground_truth_root=Path(args.temperature_root).resolve(),
        evaluation_support_root=(Path(args.evaluation_support_root).resolve() / "bool"),
        range_manifest=Path(args.range_manifest).resolve(),
        split_manifest=Path(args.bound_split).resolve(),
        expected_count=len(runtime.test_cameras),
    )
    post_preload_tracker = tracker.finish()

    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "scene_name": binding.scene_name,
        "variant": variant,
        "split": "test",
        "resolution": "r1/full-resolution",
        "formal_protocol_sha256": binding.formal_protocol_sha256,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_step": checkpoint["step"],
        "endpoint_receipt_identity": endpoint_identity,
        "verified_training_endpoint_cost": dict(endpoint_receipt["cost"]),
        "training_source_provenance": protocol["source_provenance"],
        "evaluation_source_provenance": evaluation_source,
        "training_evaluation_compatibility": compatibility,
        "metrics": {
            "temperature_mae_c": float(all_abs.mean()),
            "temperature_rmse_c": float(math.sqrt(squared_sum / valid_count)),
            "temperature_bias_c": float(signed_sum / valid_count),
            "temperature_p95_abs_c": float(np.quantile(all_abs, 0.95)),
            "temperature_semantics": (
                "direct OCT rendered apparent-temperature vs float32 TSDK target; "
                "not palette-inverted"
            ),
            "palette_inverted_comparable": palette_temperature[
                "primary_supported_pixels"
            ],
            "support_masked_psnr_db_mean_per_view": (
                None
                if any(row["support_masked_psnr_db"] is None for row in rows)
                else float(np.mean([row["support_masked_psnr_db"] for row in rows]))
            ),
            "support_masked_ssim_mean_per_view": float(
                np.mean([row["support_masked_ssim"] for row in rows])
            ),
            "formal_full_frame_psnr_db": appearance["metrics"]["PSNR"],
            "formal_full_frame_ssim": appearance["metrics"]["SSIM"],
            "formal_full_frame_lpips": appearance["metrics"]["LPIPS"],
            "hotspot_iou": float(hotspot_intersection / max(hotspot_union, 1)),
            "hotspot_auprc_histogram_4096": auprc,
            "hotspot_primary_semantics": (
                "exact-display 8-bit Hot-Iron LUT-equivalent apparent temperature; "
                "comparable to palette-inverted F3/legacy"
            ),
            "hotspot_positive_pixels": int(hotspot_pos.sum()),
            "hotspot_negative_pixels": int(hotspot_neg.sum()),
            "hotspot_direct_temperature_noncomparable_diagnostic": {
                "iou": float(
                    direct_hotspot_intersection / max(direct_hotspot_union, 1)
                ),
                "auprc_histogram_4096": direct_auprc,
                "positive_pixels": int(direct_hotspot_pos.sum()),
                "negative_pixels": int(direct_hotspot_neg.sum()),
                "reason_noncomparable": (
                    "OCT direct float temperature retains sub-bin information that "
                    "palette-only F3/legacy outputs do not expose"
                ),
            },
            "off_lut_distance": palette_temperature["off_lut_distance"][
                "supported_pixels"
            ],
            "off_lut_note": (
                "verified exact fixed-LUT display property; not treated as an accuracy win"
            ),
        },
        "hotspot_threshold": threshold,
        "cross_view_gaussian_temperature_variance_c2": {
            "accumulator": "vectorized Welford population variance",
            "status": (
                "zero_by_construction_and_bit_exact"
                if variant == "oct_scalar"
                else "measured_projected_visible_views"
            ),
            "population": "Gaussians with radii>0 in at least two formal test views",
            "population_count": int(measured_variance.numel()),
            "visible_view_count_p50": float(
                torch.quantile(
                    gaussian_visible_count[variance_valid].float(), 0.50
                ).item()
            ),
            "visible_view_count_p95": float(
                torch.quantile(
                    gaussian_visible_count[variance_valid].float(), 0.95
                ).item()
            ),
            "mean": float(measured_variance.mean().item()),
            "p95": float(torch.quantile(measured_variance, 0.95).item()),
            "max": float(measured_variance.max().item()),
        },
        "uncertainty_calibration": "N/A (OCT-GS v1 disables sigma/NLL)",
        "shared_occupancy_invariant": occupancy_evidence,
        "alpha_backend": context.alpha_backend,
        "cost": {
            "post_preload_evaluation_tracker": post_preload_tracker,
            "verified_training_endpoint": dict(endpoint_receipt["cost"]),
            "pure_render": {
                "synchronized_cuda_event": True,
                "views": len(render_time_ms),
                "total_ms": float(np.sum(render_time_ms)),
                "mean_ms_per_view": float(np.mean(render_time_ms)),
                "fps": float(1000.0 / np.mean(render_time_ms)),
            },
            "formal_appearance_evaluator_wall_time_s": appearance["wall_time_s"],
            "formal_temperature_evaluator_wall_time_s": palette_temperature[
                "wall_time_s"
            ],
        },
        "appearance_evaluator": appearance,
        "palette_inverted_temperature_evaluator": palette_temperature,
        "per_view": rows,
    }
    report["cost"]["process_wall_time_setup_to_report_s"] = max(
        0.0, time.perf_counter() - process_start_perf
    )
    frozen_runner._atomic_json(output_root / "evaluation_v2.json", report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    frozen_runner._add_common(parser)
    frozen_runner._add_training(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--endpoint-receipt", required=True)
    parser.add_argument("--protocol-manifest", required=True)
    parser.add_argument("--hotspot-threshold-manifest", required=True)
    parser.add_argument("--eval-output", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frozen_runner._validate_recipe_args(args)
    return command_eval_v2(args)


if __name__ == "__main__":
    raise SystemExit(main())
