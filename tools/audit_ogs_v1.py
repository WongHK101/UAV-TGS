#!/usr/bin/env python3
"""Build the fixed OGS-v1 observability cache and audit an RGB anchor.

This is intentionally a separate, no-gradient process.  It loads the formal
RGB checkpoint/PLY, sweeps the explicit training cameras once, and writes a
cache that the continuation trainer can consume without repeating the sweep.
Visibility is the renderer's ``radii > 0`` predicate; no rasterizer changes or
training RNG are involved.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.camera_sequence import camera_parameters_hash, camera_parameters_payload


AUDIT_SCHEMA = "uav-tgs-ogs-v1-anchor-audit-v1"
CACHE_SCHEMA = "ogs_v1_anchor_cache_v1"
RHO = 3.0
EPS = 1e-8
VARIANCE_EPS = 1e-16
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUDIT_OUTPUT_FILENAMES = frozenset(
    {
        "clamp20_control_summary.json",
        "clamp20_diagnostics.csv",
        "manifest.json",
        "observability_audit.json",
        "ogs_v1_anchor_cache.pt",
        "ordered_camera_names.json",
        "scale_matched_controls.csv",
        "top_risk_gaussians.csv",
    }
)


class OgsAuditError(RuntimeError):
    """Raised when an OGS audit input or invariant fails closed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def verify_optional_sha256(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if expected:
        normalized = expected.strip().lower()
        if not SHA256_RE.fullmatch(normalized):
            raise OgsAuditError(f"{label} expected SHA-256 is invalid: {expected!r}")
        if actual != normalized:
            raise OgsAuditError(
                f"{label} SHA-256 mismatch: expected={normalized} actual={actual} path={path}"
            )
    return actual


def tensor_sha256(tensor: Any) -> str:
    """Hash a tensor's dtype, shape, and raw contiguous CPU bytes."""
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def composite_tensor_sha256(tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(tensors[name]).encode("ascii"))
    return digest.hexdigest()


def read_name_list(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise OgsAuditError(f"training camera list is empty: {path}")
    if len(names) != len(set(names)):
        raise OgsAuditError(f"training camera list contains duplicate names: {path}")
    return names


def load_binding_manifest(
    path: Path,
    *,
    scene_name: str,
    train_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OgsAuditError(f"cannot read binding manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise OgsAuditError("binding manifest root must be an object")
    if payload.get("status") not in (None, "passed"):
        raise OgsAuditError(
            f"binding manifest is not passed: status={payload.get('status')!r}"
        )
    declared_scene = payload.get("scene")
    if declared_scene is not None and str(declared_scene) != str(scene_name):
        raise OgsAuditError(
            f"binding scene mismatch: expected={scene_name!r} actual={declared_scene!r}"
        )

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise OgsAuditError("binding manifest must contain a non-empty files list")
    bound_train = []
    for row in files:
        if not isinstance(row, dict):
            raise OgsAuditError("binding files entries must be objects")
        if row.get("split") == "train":
            name = row.get("camera_name")
            if not isinstance(name, str) or not name:
                raise OgsAuditError("binding train entry has no camera_name")
            bound_train.append(name)
    if len(bound_train) != len(set(bound_train)):
        raise OgsAuditError("binding manifest contains duplicate train camera names")
    if set(bound_train) != set(train_names):
        missing = sorted(set(train_names) - set(bound_train))
        extra = sorted(set(bound_train) - set(train_names))
        raise OgsAuditError(
            "binding/train-list mismatch: "
            f"missing_from_binding={missing[:8]} extra_in_binding={extra[:8]}"
        )
    counts = payload.get("counts")
    if isinstance(counts, dict) and counts.get("train") is not None:
        if int(counts["train"]) != len(bound_train):
            raise OgsAuditError(
                "binding train count mismatch: "
                f"declared={counts['train']} actual={len(bound_train)}"
            )

    hash_keys = (
        "binding_hash",
        "collection_hash",
        "collection_split_hash",
        "formal_rule_hash",
        "scene_split_hash",
        "scene_rule_hash",
        "decode_protocol_hash",
    )
    split_hashes = {
        key: str(payload[key])
        for key in hash_keys
        if isinstance(payload.get(key), str) and payload[key]
    }
    return payload, split_hashes


def parse_clamp_indices(
    path: Path,
    *,
    gaussian_count: int,
    expected_count: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the established clamp sidecar without recomputing its selection."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OgsAuditError(f"cannot read clamp index sidecar: {path}") from exc
    if not isinstance(payload, dict):
        raise OgsAuditError("clamp index sidecar root must be an object")
    declared_count = payload.get("gaussian_count")
    if declared_count is not None and int(declared_count) != int(gaussian_count):
        raise OgsAuditError(
            "clamp sidecar Gaussian count mismatch: "
            f"declared={declared_count} anchor={gaussian_count}"
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise OgsAuditError("clamp sidecar must contain a records list")
    indices = []
    for row in records:
        if not isinstance(row, dict) or "gaussian_index" not in row:
            raise OgsAuditError("clamp sidecar record is missing gaussian_index")
        value = row["gaussian_index"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise OgsAuditError(f"invalid clamp Gaussian index: {value!r}")
        indices.append(value)
    if len(indices) != len(set(indices)):
        raise OgsAuditError("clamp index sidecar contains duplicate indices")
    if any(index < 0 or index >= gaussian_count for index in indices):
        raise OgsAuditError("clamp index sidecar contains an out-of-range index")
    declared_selected = payload.get("selected_count")
    if declared_selected is not None and int(declared_selected) != len(indices):
        raise OgsAuditError(
            "clamp selected_count mismatch: "
            f"declared={declared_selected} actual={len(indices)}"
        )
    if expected_count is not None and len(indices) != int(expected_count):
        raise OgsAuditError(
            f"clamp index count mismatch: expected={expected_count} actual={len(indices)}"
        )
    return np.asarray(sorted(indices), dtype=np.int64), payload


def distribution(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {"count": 0, "percentiles": {}}
    if not np.all(np.isfinite(array)):
        raise OgsAuditError("cannot summarize non-finite values")
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "percentiles": {},
    }
    for percentile in PERCENTILES:
        result["percentiles"][f"p{percentile}"] = float(
            np.percentile(array, percentile)
        )
    return result


def risk_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    query = np.asarray(values, dtype=np.float64).reshape(-1)
    if ref.size == 0:
        raise OgsAuditError("risk percentile reference population is empty")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(query)):
        raise OgsAuditError("risk percentile inputs contain non-finite values")
    ordered = np.sort(ref, kind="stable")
    ranks = np.searchsorted(ordered, query, side="right")
    return ranks.astype(np.float64) * (100.0 / float(ordered.size))


def match_scale_controls(
    raw_log_scales: np.ndarray,
    eligible_mask: np.ndarray,
    clamp_indices: np.ndarray,
    *,
    controls_per_target: int = 5,
) -> dict[str, Any]:
    """Wrap the shared matcher while exposing flat audit CSV rows.

    Matching never receives observability, q, risk, depth, final metrics, or
    test views.  The shared implementation excludes clamp indices from the
    fixed-eligible candidate pool, globally prefers no replacement, and uses
    index order to break distance ties.
    """
    from utils.ogs_v1 import match_scale_controls as core_match_scale_controls

    scales = np.asarray(raw_log_scales, dtype=np.float64)
    eligible = np.asarray(eligible_mask, dtype=bool).reshape(-1)
    clamps = np.asarray(clamp_indices, dtype=np.int64).reshape(-1)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise OgsAuditError(f"raw_log_scales must have shape [N,3], got {scales.shape}")
    if eligible.shape != (scales.shape[0],):
        raise OgsAuditError("eligible mask shape does not match scale array")
    if controls_per_target <= 0:
        raise OgsAuditError("controls_per_target must be positive")
    if clamps.size == 0:
        raise OgsAuditError("cannot match controls for an empty clamp set")
    if len(set(clamps.tolist())) != clamps.size:
        raise OgsAuditError("clamp indices for matching are not unique")
    if np.any(clamps < 0) or np.any(clamps >= scales.shape[0]):
        raise OgsAuditError("clamp indices for matching are out of range")
    if not np.all(np.isfinite(scales)):
        raise OgsAuditError("raw log scales contain non-finite values")

    try:
        activated_scales = np.exp(scales)
        matched = core_match_scale_controls(
            activated_scales,
            eligible,
            clamps.tolist(),
            controls_per_target=controls_per_target,
            standardization_mask=eligible,
        )
    except (ValueError, IndexError, FloatingPointError) as exc:
        raise OgsAuditError(f"scale control matching failed: {exc}") from exc
    rows = []
    for record in matched["records"]:
        for rank, (control, distance, reused) in enumerate(
            zip(
                record["control_indices"],
                record["squared_distances"],
                record["reused"],
            ),
            start=1,
        ):
            rows.append(
                {
                    "clamp_index": int(record["target_index"]),
                    "control_rank": rank,
                    "control_index": int(control),
                    "squared_standardized_scale_distance": float(distance),
                    "reused": bool(reused),
                }
            )
    return {
        "rows": rows,
        "controls_per_target": int(matched["controls_per_target"]),
        "candidate_count": int(matched["candidate_count"]),
        "unique_control_count": int(matched["unique_control_count"]),
        "reuse_count": int(matched["reuse_count"]),
        "reuse_required": bool(matched["reuse_required"]),
        "standardization": matched["standardization"],
    }


def evaluate_complete_non_support(
    *,
    eligible_active_clamp_count: int,
    clamp_risk_percentiles: np.ndarray,
    clamp_risk: np.ndarray,
    control_risk: np.ndarray,
) -> dict[str, Any]:
    """Apply the three-way conservative stop rule exactly as specified."""
    clamp_percentiles = np.asarray(clamp_risk_percentiles, dtype=np.float64)
    clamp_values = np.asarray(clamp_risk, dtype=np.float64)
    control_values = np.asarray(control_risk, dtype=np.float64)
    if clamp_percentiles.size == 0 or clamp_values.size == 0 or control_values.size == 0:
        raise OgsAuditError("complete-non-support gate received an empty population")
    conditions = {
        "eligible_and_active_clamp_count_lt_5": int(eligible_active_clamp_count) < 5,
        "median_clamp_risk_percentile_lte_50": float(
            np.median(clamp_percentiles)
        ) <= 50.0,
        "median_clamp_risk_lte_control": float(np.median(clamp_values))
        <= float(np.median(control_values)),
    }
    return {
        "conditions": conditions,
        "complete_non_support": bool(all(conditions.values())),
        "eligible_and_active_clamp_count": int(eligible_active_clamp_count),
        "median_clamp_risk_percentile": float(np.median(clamp_percentiles)),
        "median_clamp_risk": float(np.median(clamp_values)),
        "median_control_risk": float(np.median(control_values)),
        "policy": (
            "stop only when all three conditions are true; weak enrichment does "
            "not change thresholds or lambda"
        ),
    }


def ordered_names_sha256(names: Sequence[str]) -> str:
    # Same canonical JSON definition used by utils.camera_sequence.
    return canonical_json_sha256(list(names))


def build_camera_audit_payload(
    cameras: Sequence[object],
) -> tuple[list[dict[str, Any]], str]:
    """Serialize camera rows and bind the digest written by the audit."""

    payload = camera_parameters_payload(cameras)
    digest = camera_parameters_hash(cameras)
    if canonical_json_sha256(payload) != digest:
        raise OgsAuditError("camera payload and camera-parameter hash disagree")
    return payload, digest


def build_ordered_camera_manifest(
    scene_name: str,
    ordered_camera_names: Sequence[str],
    cameras: Sequence[object],
) -> dict[str, Any]:
    """Build the complete ordered-camera sidecar without GPU dependencies."""

    names = list(ordered_camera_names)
    if len(names) != len(cameras):
        raise OgsAuditError("ordered camera names and camera payload lengths differ")
    camera_payload, camera_parameters_sha = build_camera_audit_payload(cameras)
    return {
        "schema": "uav-tgs-ogs-v1-ordered-train-cameras-v1",
        "scene_name": scene_name,
        "ordered_camera_names": names,
        "ordered_camera_sha256": ordered_names_sha256(names),
        "camera_parameters_sha256": camera_parameters_sha,
        "cameras": camera_payload,
    }


def _read_ply_raw_geometry(path: Path) -> dict[str, np.ndarray]:
    from plyfile import PlyData

    try:
        vertex = PlyData.read(str(path))["vertex"].data
    except (KeyError, ValueError) as exc:
        raise OgsAuditError(f"cannot read anchor PLY vertex data: {path}") from exc
    names = set(vertex.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "opacity",
    }
    missing = sorted(required - names)
    if missing:
        raise OgsAuditError(f"anchor PLY is missing geometry fields: {missing}")
    return {
        "xyz": np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1),
        "scaling": np.stack(
            [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1
        ),
        "rotation": np.stack(
            [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]],
            axis=1,
        ),
        "opacity": np.asarray(vertex["opacity"])[:, None],
    }


def _validate_checkpoint_ply_exact(
    checkpoint_raw: Mapping[str, Any],
    ply_raw: Mapping[str, np.ndarray],
) -> None:
    for name in ("xyz", "scaling", "rotation", "opacity"):
        checkpoint_array = checkpoint_raw[name].detach().cpu().contiguous().numpy()
        ply_array = np.asarray(ply_raw[name])
        if checkpoint_array.dtype != ply_array.dtype:
            raise OgsAuditError(
                f"checkpoint/PLY dtype mismatch for {name}: "
                f"{checkpoint_array.dtype} vs {ply_array.dtype}"
            )
        if checkpoint_array.shape != ply_array.shape:
            raise OgsAuditError(
                f"checkpoint/PLY shape mismatch for {name}: "
                f"{checkpoint_array.shape} vs {ply_array.shape}"
            )
        if not np.array_equal(checkpoint_array, ply_array):
            max_abs = float(
                np.max(
                    np.abs(
                        checkpoint_array.astype(np.float64)
                        - ply_array.astype(np.float64)
                    )
                )
            )
            raise OgsAuditError(
                f"checkpoint/PLY raw field is not exact for {name}; max_abs={max_abs}"
            )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise OgsAuditError(f"audit output exists but is not a directory: {path}")
    entries = list(path.iterdir())
    if entries and not overwrite:
        raise OgsAuditError(
            f"refusing to overwrite non-empty audit output directory: {path}"
        )
    if overwrite:
        unknown = [
            entry
            for entry in entries
            if entry.name not in AUDIT_OUTPUT_FILENAMES
            or not (entry.is_file() or entry.is_symlink())
        ]
        if unknown:
            raise OgsAuditError(
                "refusing --overwrite because the output contains unknown entries: "
                f"{[entry.name for entry in unknown[:8]]}"
            )
        for entry in entries:
            entry.unlink()


def _run_gpu_audit(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import render
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.ogs_v1 import (
        OGS_V1_FORMULA_SHA256,
        OGS_V1_LOSS_EPS,
        OGS_V1_MIN_ACTIVATED_OPACITY,
        OGS_V1_MIN_VISIBLE_VIEWS,
        OGS_V1_RHO,
        OGS_V1_VARIANCE_EPS,
        accumulate_camera_bearings,
        build_ogs_eligibility_mask,
        build_ogs_cache,
        compute_observability_from_moments,
        covariance_thickness,
        initialize_bearing_moment_accumulators,
        save_ogs_cache,
        verify_ogs_v1_formula_hash,
    )

    if not torch.cuda.is_available():
        raise OgsAuditError("OGS observability audit requires the CUDA renderer")

    checkpoint_path = args.checkpoint.resolve()
    anchor_ply_path = args.anchor_ply.resolve()
    train_list_path = Path(args.train_list).resolve()
    binding_path = args.binding_manifest.resolve()
    for path, label in (
        (checkpoint_path, "checkpoint"),
        (anchor_ply_path, "anchor PLY"),
        (train_list_path, "train list"),
        (binding_path, "binding manifest"),
    ):
        if not path.is_file():
            raise OgsAuditError(f"{label} does not exist: {path}")

    checkpoint_sha = verify_optional_sha256(
        checkpoint_path, args.expected_checkpoint_sha256, "checkpoint"
    )
    ply_sha = verify_optional_sha256(
        anchor_ply_path, args.expected_ply_sha256, "anchor PLY"
    )
    train_list_sha = verify_optional_sha256(
        train_list_path, args.expected_train_list_sha256, "train list"
    )
    binding_sha = verify_optional_sha256(
        binding_path, args.expected_binding_sha256, "binding manifest"
    )
    train_names = read_name_list(train_list_path)
    declared_scene_train_sha = str(getattr(args, "train_list_sha256", "") or "")
    if declared_scene_train_sha and declared_scene_train_sha.lower() != train_list_sha:
        raise OgsAuditError(
            "ModelParams train_list_sha256 disagrees with the audited train list: "
            f"declared={declared_scene_train_sha} actual={train_list_sha}"
        )
    # Make the Scene reader independently verify the same explicit list bytes.
    args.train_list_sha256 = train_list_sha
    _, split_hashes = load_binding_manifest(
        binding_path, scene_name=args.scene_name, train_names=train_names
    )

    expected_ply = (
        Path(args.model_path)
        / "point_cloud"
        / f"iteration_{args.anchor_iteration}"
        / "point_cloud.ply"
    ).resolve()
    try:
        same_anchor = os.path.samefile(anchor_ply_path, expected_ply)
    except OSError:
        same_anchor = anchor_ply_path == expected_ply
    if not same_anchor:
        raise OgsAuditError(
            "--anchor-ply must be the exact Scene anchor loaded at the requested "
            f"iteration: explicit={anchor_ply_path} scene={expected_ply}"
        )

    # This is the explicitly named, locally produced formal anchor checkpoint.
    # PyTorch 2.6 defaults to weights_only=True, but Gaussian capture() includes
    # trusted optimizer/state objects that require the scoped opt-out here.
    checkpoint_payload = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False
    )
    if (
        not isinstance(checkpoint_payload, (tuple, list))
        or len(checkpoint_payload) != 2
    ):
        raise OgsAuditError("checkpoint must be a (model_params, iteration) pair")
    model_params, checkpoint_iteration = checkpoint_payload
    if int(checkpoint_iteration) != int(args.anchor_iteration):
        raise OgsAuditError(
            "checkpoint iteration mismatch: "
            f"expected={args.anchor_iteration} actual={checkpoint_iteration}"
        )
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 12:
        raise OgsAuditError("checkpoint model payload does not match Gaussian capture()")
    active_sh_degree = int(model_params[0])
    checkpoint_raw = {
        "xyz": model_params[1].detach().cpu().contiguous().clone(),
        "scaling": model_params[4].detach().cpu().contiguous().clone(),
        "rotation": model_params[5].detach().cpu().contiguous().clone(),
        "opacity": model_params[6].detach().cpu().contiguous().clone(),
    }
    # The audit does not use Adam moments.  Release the large trusted checkpoint
    # payload after copying the four anchor tensors needed by the protocol.
    del model_params, checkpoint_payload
    gc.collect()
    raw_tensor_hashes = {
        name: tensor_sha256(tensor) for name, tensor in checkpoint_raw.items()
    }
    geometry_raw_sha = composite_tensor_sha256(checkpoint_raw)
    gaussian_count = int(checkpoint_raw["xyz"].shape[0])
    expected_shapes = {
        "xyz": (gaussian_count, 3),
        "scaling": (gaussian_count, 3),
        "rotation": (gaussian_count, 4),
        "opacity": (gaussian_count, 1),
    }
    for name, expected_shape in expected_shapes.items():
        tensor = checkpoint_raw[name]
        if tuple(tensor.shape) != expected_shape:
            raise OgsAuditError(
                f"checkpoint {name} shape mismatch: {tuple(tensor.shape)} vs {expected_shape}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise OgsAuditError(f"checkpoint {name} contains NaN/Inf")

    ply_raw = _read_ply_raw_geometry(anchor_ply_path)
    _validate_checkpoint_ply_exact(checkpoint_raw, ply_raw)

    # Scene(shuffle=False) preserves the reader's deterministic sorted order and
    # does not call random.shuffle.  This executable is a separate process from
    # training, so the sweep cannot consume continuation RNG state.
    dataset = ModelParams(argparse.ArgumentParser(add_help=False)).extract(args)
    pipeline = PipelineParams(argparse.ArgumentParser(add_help=False)).extract(args)
    gaussians = GaussianModel(dataset.sh_degree, args.optimizer_type)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.anchor_iteration,
        shuffle=False,
    )
    cameras = list(scene.getTrainCameras())
    ordered_camera_names = [str(camera.image_name) for camera in cameras]
    if len(ordered_camera_names) != len(set(ordered_camera_names)):
        raise OgsAuditError("loaded training cameras contain duplicate image names")
    if set(ordered_camera_names) != set(train_names):
        raise OgsAuditError(
            "loaded train cameras do not exactly match the explicit train list"
        )

    # Replace Scene's PLY geometry with the byte-exact checkpoint tensors.
    with torch.no_grad():
        gaussians._xyz = nn.Parameter(
            checkpoint_raw["xyz"].cuda().requires_grad_(False)
        )
        gaussians._scaling = nn.Parameter(
            checkpoint_raw["scaling"].cuda().requires_grad_(False)
        )
        gaussians._rotation = nn.Parameter(
            checkpoint_raw["rotation"].cuda().requires_grad_(False)
        )
        gaussians._opacity = nn.Parameter(
            checkpoint_raw["opacity"].cuda().requires_grad_(False)
        )
        gaussians.active_sh_degree = active_sh_degree

    ordered_camera_manifest = build_ordered_camera_manifest(
        args.scene_name, ordered_camera_names, cameras
    )
    camera_parameters_sha = ordered_camera_manifest["camera_parameters_sha256"]
    ordered_camera_names_sha = ordered_camera_manifest[
        "ordered_camera_sha256"
    ]

    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    moment_sum, visible_count = initialize_bearing_moment_accumulators(
        gaussian_count, torch.device("cuda")
    )
    with torch.no_grad():
        for camera_index, camera in enumerate(cameras):
            render_package = render(
                camera,
                gaussians,
                pipeline,
                background,
                use_trained_exp=False,
                separate_sh=False,
            )
            radii = render_package["radii"]
            if tuple(radii.shape) != (gaussian_count,):
                raise OgsAuditError(
                    f"renderer radii shape mismatch at camera {camera_index}: {radii.shape}"
                )
            visibility = radii > 0
            accumulate_camera_bearings(
                moment_sum,
                visible_count,
                gaussians.get_xyz,
                camera.camera_center,
                visibility,
            )

    core_result = compute_observability_from_moments(
        moment_sum, visible_count, chunk_size=args.eig_chunk_size
    )
    observability = core_result["observability"]
    weakest_direction = core_result["weakest_direction"]
    eigenvalues = core_result["eigenvalues"]
    weak_eigengap = core_result["weak_eigengap"]
    min_eigengap = core_result["min_eigengap"]
    trace_m_residual = core_result["trace_m_residual"]
    trace_h_residual = core_result["trace_h_residual"]
    finite_mask = (
        core_result["eigen_finite"].to(dtype=torch.bool)
        & core_result["eigen_in_range"].to(dtype=torch.bool)
    )
    seen = visible_count > 0
    if not bool(finite_mask[seen].all()):
        raise OgsAuditError("observability eigendecomposition has invalid seen Gaussians")
    if not bool(torch.isfinite(eigenvalues[seen]).all()):
        raise OgsAuditError("H eigenvalues contain NaN/Inf")
    eigen_tolerance = float(args.eigen_tolerance)
    if bool((eigenvalues[seen] < -eigen_tolerance).any()) or bool(
        (eigenvalues[seen] > 1.0 + eigen_tolerance).any()
    ):
        raise OgsAuditError(
            "H eigenvalues exceed the theoretical [0,1] range beyond tolerance"
        )
    max_trace_m_residual = torch.max(torch.abs(trace_m_residual[seen]))
    max_trace_h_residual = torch.max(torch.abs(trace_h_residual[seen]))
    if bool(max_trace_m_residual > args.trace_tolerance) or bool(
        max_trace_h_residual > args.trace_tolerance
    ):
        raise OgsAuditError(
            "M/H trace residual exceeds tolerance: "
            f"max_M={float(max_trace_m_residual.item())} "
            f"max_H={float(max_trace_h_residual.item())}"
        )

    formula_sha256 = verify_ogs_v1_formula_hash(OGS_V1_FORMULA_SHA256)
    activated_opacity = torch.sigmoid(checkpoint_raw["opacity"].cuda()).squeeze(1)
    eligible = build_ogs_eligibility_mask(visible_count, activated_opacity)
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        raise OgsAuditError("OGS eligible set is empty; refusing to write a cache")

    anchor_scales = torch.exp(
        checkpoint_raw["scaling"].cuda().to(dtype=torch.float64)
    )
    anchor_rotations = checkpoint_raw["rotation"].cuda().to(dtype=torch.float64)
    thickness = covariance_thickness(
        anchor_scales,
        anchor_rotations,
        weakest_direction.to(dtype=torch.float64),
        variance_eps=OGS_V1_VARIANCE_EPS,
    )
    parallel = thickness["parallel"]
    perpendicular = thickness["perpendicular"]
    q_anchor = parallel / (perpendicular + EPS)
    covariance_trace = thickness["trace"]
    log_penalty = torch.relu(
        torch.log(parallel + OGS_V1_LOSS_EPS)
        - torch.log(OGS_V1_RHO * perpendicular + OGS_V1_LOSS_EPS)
    ).square()
    risk = (1.0 - observability).square() * log_penalty
    active = eligible & (parallel > OGS_V1_RHO * perpendicular)
    for label, values in (
        ("observability", observability),
        ("anchor penalty", log_penalty),
        ("anchor risk", risk),
    ):
        if not bool(torch.isfinite(values).all()):
            raise OgsAuditError(f"{label} contains NaN/Inf")

    metadata = {
        "schema_version": CACHE_SCHEMA,
        "scene_name": args.scene_name,
        "gaussian_count": gaussian_count,
        "checkpoint_iteration": int(checkpoint_iteration),
        "checkpoint_sha256": checkpoint_sha,
        "ply_sha256": ply_sha,
        "raw_tensor_sha256": raw_tensor_hashes,
        "geometry_raw_tensor_sha256": geometry_raw_sha,
        "train_list_sha256": train_list_sha,
        "ordered_train_camera_names_sha256": ordered_camera_names_sha,
        "camera_parameters_sha256": camera_parameters_sha,
        "binding_sha256": binding_sha,
        "split_hashes": split_hashes,
        "rho": OGS_V1_RHO,
        "eps": OGS_V1_LOSS_EPS,
        "variance_eps": OGS_V1_VARIANCE_EPS,
        "ogs_v1_formula_sha256": formula_sha256,
        "renderer_visibility": "radii>0",
        "eligibility": {
            "minimum_visible_views": OGS_V1_MIN_VISIBLE_VIEWS,
            "minimum_activated_opacity_strict_gt": (
                OGS_V1_MIN_ACTIVATED_OPACITY
            ),
        },
        "camera_count": len(cameras),
        "separate_process_no_training_rng": True,
        "moment_accumulation_dtype": "float64",
        "moment_storage": "full_symmetric_3x3_float64",
        "moment_storage_bytes": int(moment_sum.numel() * moment_sum.element_size()),
    }
    cache = build_ogs_cache(
        observability,
        weakest_direction,
        visible_count,
        activated_opacity,
        anchor_scales,
        anchor_rotations,
        metadata=metadata,
        min_visible_views=OGS_V1_MIN_VISIBLE_VIEWS,
        min_activated_opacity=OGS_V1_MIN_ACTIVATED_OPACITY,
        eps=OGS_V1_VARIANCE_EPS,
    )
    if not torch.equal(cache["eligible_mask"], eligible.detach().cpu()):
        raise OgsAuditError("core cache eligibility differs from audit eligibility")
    cached_perpendicular = cache["perpendicular_thickness"].to(
        device=perpendicular.device, dtype=perpendicular.dtype
    )
    if not bool(
        torch.allclose(cached_perpendicular, perpendicular, rtol=2e-6, atol=1e-8)
    ):
        raise OgsAuditError("core cache perpendicular thickness differs from audit")

    output = args.output.resolve()
    _prepare_output_directory(output, overwrite=args.overwrite)
    _write_json(
        output / "ordered_camera_names.json",
        ordered_camera_manifest,
    )
    cache_path = output / "ogs_v1_anchor_cache.pt"
    semantic_cache_sha = save_ogs_cache(cache_path, cache)

    to_numpy = lambda tensor: tensor.detach().cpu().numpy()
    arrays = {
        "visible_count": to_numpy(visible_count),
        "opacity": to_numpy(activated_opacity),
        "observability": to_numpy(observability),
        "weakest_direction": to_numpy(weakest_direction),
        "eigenvalues": to_numpy(eigenvalues),
        "weak_eigengap": to_numpy(weak_eigengap),
        "min_eigengap": to_numpy(min_eigengap),
        "trace_m_residual": to_numpy(trace_m_residual),
        "trace_h_residual": to_numpy(trace_h_residual),
        "parallel": to_numpy(parallel),
        "perpendicular": to_numpy(perpendicular),
        "q": to_numpy(q_anchor),
        "penalty": to_numpy(log_penalty),
        "risk": to_numpy(risk),
        "covariance_trace": to_numpy(covariance_trace),
        "eligible": to_numpy(eligible).astype(bool),
        "active": to_numpy(active).astype(bool),
        "xyz": checkpoint_raw["xyz"].numpy(),
        "raw_scaling": checkpoint_raw["scaling"].numpy(),
        "raw_rotation": checkpoint_raw["rotation"].numpy(),
    }
    seen_np = arrays["visible_count"] > 0
    eligible_np = arrays["eligible"]
    active_np = arrays["active"]
    global_summary = {
        "gaussian_count": gaussian_count,
        "camera_count": len(cameras),
        "seen_count": int(np.sum(seen_np)),
        "eligible_count": eligible_count,
        "active_count": int(np.sum(active_np)),
        "visible_count": distribution(arrays["visible_count"]),
        "activated_opacity": distribution(arrays["opacity"]),
        "observability_seen": distribution(arrays["observability"][seen_np]),
        "observability_eligible": distribution(arrays["observability"][eligible_np]),
        "h_eigenvalue_0_seen": distribution(arrays["eigenvalues"][seen_np, 0]),
        "h_eigenvalue_1_seen": distribution(arrays["eigenvalues"][seen_np, 1]),
        "h_eigenvalue_2_seen": distribution(arrays["eigenvalues"][seen_np, 2]),
        "weak_eigengap_seen": distribution(arrays["weak_eigengap"][seen_np]),
        "minimum_adjacent_eigengap_seen": distribution(
            arrays["min_eigengap"][seen_np]
        ),
        "moment_trace_m_residual_seen": distribution(
            arrays["trace_m_residual"][seen_np]
        ),
        "hessian_trace_h_residual_seen": distribution(
            arrays["trace_h_residual"][seen_np]
        ),
        "parallel_thickness_eligible": distribution(
            arrays["parallel"][eligible_np]
        ),
        "perpendicular_thickness_eligible": distribution(
            arrays["perpendicular"][eligible_np]
        ),
        "q_anchor_eligible": distribution(arrays["q"][eligible_np]),
        "unweighted_penalty_eligible": distribution(
            arrays["penalty"][eligible_np]
        ),
        "weighted_risk_eligible": distribution(arrays["risk"][eligible_np]),
        "covariance_trace_eligible": distribution(
            arrays["covariance_trace"][eligible_np]
        ),
        "finite_flags": {
            "all_seen_eigendecompositions_finite": True,
            "all_observability_finite": True,
            "all_anchor_thickness_finite": True,
            "all_anchor_risk_finite": True,
        },
    }

    eligible_indices = np.flatnonzero(eligible_np)
    risk_order = eligible_indices[
        np.lexsort((eligible_indices, -arrays["risk"][eligible_indices]))
    ]
    top_indices = risk_order[: min(args.top_risk_count, risk_order.size)]
    top_rows = []
    for rank, index in enumerate(top_indices.tolist(), start=1):
        row = {
            "risk_rank": rank,
            "gaussian_index": index,
            "x": float(arrays["xyz"][index, 0]),
            "y": float(arrays["xyz"][index, 1]),
            "z": float(arrays["xyz"][index, 2]),
            "visible_count": int(arrays["visible_count"][index]),
            "activated_opacity": float(arrays["opacity"][index]),
            "observability": float(arrays["observability"][index]),
            "weak_n_x": float(arrays["weakest_direction"][index, 0]),
            "weak_n_y": float(arrays["weakest_direction"][index, 1]),
            "weak_n_z": float(arrays["weakest_direction"][index, 2]),
            "raw_scale_0": float(arrays["raw_scaling"][index, 0]),
            "raw_scale_1": float(arrays["raw_scaling"][index, 1]),
            "raw_scale_2": float(arrays["raw_scaling"][index, 2]),
            "parallel_thickness": float(arrays["parallel"][index]),
            "perpendicular_thickness": float(arrays["perpendicular"][index]),
            "q_anchor": float(arrays["q"][index]),
            "unweighted_penalty": float(arrays["penalty"][index]),
            "risk": float(arrays["risk"][index]),
            "active": bool(arrays["active"][index]),
            "h_eigenvalue_0": float(arrays["eigenvalues"][index, 0]),
            "h_eigenvalue_1": float(arrays["eigenvalues"][index, 1]),
            "h_eigenvalue_2": float(arrays["eigenvalues"][index, 2]),
            "weak_eigengap": float(arrays["weak_eigengap"][index]),
            "minimum_adjacent_eigengap": float(arrays["min_eigengap"][index]),
        }
        top_rows.append(row)
    top_fields = list(top_rows[0]) if top_rows else []
    _write_csv(output / "top_risk_gaussians.csv", top_rows, top_fields)

    clamp_summary = None
    phase_b_allowed = True
    if args.clamp_indices is not None:
        clamp_path = args.clamp_indices.resolve()
        if not clamp_path.is_file():
            raise OgsAuditError(f"clamp index sidecar does not exist: {clamp_path}")
        clamp_sha = verify_optional_sha256(
            clamp_path, args.expected_clamp_sha256, "clamp index sidecar"
        )
        clamp_indices, _ = parse_clamp_indices(
            clamp_path,
            gaussian_count=gaussian_count,
            expected_count=args.expected_clamp_count,
        )
        matching = match_scale_controls(
            arrays["raw_scaling"],
            eligible_np,
            clamp_indices,
            controls_per_target=args.controls_per_clamp,
        )
        control_indices = np.asarray(
            [row["control_index"] for row in matching["rows"]], dtype=np.int64
        )
        reference_risk = arrays["risk"][eligible_np]
        clamp_percentile = risk_percentiles(
            reference_risk, arrays["risk"][clamp_indices]
        )
        control_percentile = risk_percentiles(
            reference_risk, arrays["risk"][control_indices]
        )
        gate = evaluate_complete_non_support(
            eligible_active_clamp_count=int(
                np.sum(eligible_np[clamp_indices] & active_np[clamp_indices])
            ),
            clamp_risk_percentiles=clamp_percentile,
            clamp_risk=arrays["risk"][clamp_indices],
            control_risk=arrays["risk"][control_indices],
        )
        phase_b_allowed = not gate["complete_non_support"]

        def group_summary(indices: np.ndarray, percentiles: np.ndarray) -> dict[str, Any]:
            return {
                "count": int(indices.size),
                "eligible_count": int(np.sum(eligible_np[indices])),
                "active_count": int(np.sum(active_np[indices])),
                "eligible_and_active_count": int(
                    np.sum(eligible_np[indices] & active_np[indices])
                ),
                "visible_count": distribution(arrays["visible_count"][indices]),
                "observability": distribution(arrays["observability"][indices]),
                "q_anchor": distribution(arrays["q"][indices]),
                "unweighted_penalty": distribution(arrays["penalty"][indices]),
                "risk": distribution(arrays["risk"][indices]),
                "risk_percentile": distribution(percentiles),
                "weak_eigengap": distribution(arrays["weak_eigengap"][indices]),
                "minimum_adjacent_eigengap": distribution(
                    arrays["min_eigengap"][indices]
                ),
            }

        clamp_summary = {
            "input": {
                "path": str(clamp_path),
                "sha256": clamp_sha,
                "count": int(clamp_indices.size),
            },
            "matching": {
                key: value for key, value in matching.items() if key != "rows"
            },
            "clamp20": group_summary(clamp_indices, clamp_percentile),
            "scale_matched_controls": group_summary(
                control_indices, control_percentile
            ),
            "complete_non_support_gate": gate,
            "phase_b_allowed": phase_b_allowed,
        }
        control_rows = []
        for row in matching["rows"]:
            index = int(row["control_index"])
            control_rows.append(
                {
                    **row,
                    "visible_count": int(arrays["visible_count"][index]),
                    "observability": float(arrays["observability"][index]),
                    "q_anchor": float(arrays["q"][index]),
                    "unweighted_penalty": float(arrays["penalty"][index]),
                    "risk": float(arrays["risk"][index]),
                    "risk_percentile": float(
                        risk_percentiles(reference_risk, arrays["risk"][[index]])[0]
                    ),
                    "active": bool(active_np[index]),
                    "weak_eigengap": float(arrays["weak_eigengap"][index]),
                    "minimum_adjacent_eigengap": float(
                        arrays["min_eigengap"][index]
                    ),
                }
            )
        _write_csv(
            output / "scale_matched_controls.csv",
            control_rows,
            list(control_rows[0]) if control_rows else [],
        )
        clamp_rows = []
        for index, percentile in zip(clamp_indices.tolist(), clamp_percentile.tolist()):
            clamp_rows.append(
                {
                    "gaussian_index": index,
                    "visible_count": int(arrays["visible_count"][index]),
                    "eligible": bool(eligible_np[index]),
                    "active": bool(active_np[index]),
                    "observability": float(arrays["observability"][index]),
                    "q_anchor": float(arrays["q"][index]),
                    "unweighted_penalty": float(arrays["penalty"][index]),
                    "risk": float(arrays["risk"][index]),
                    "risk_percentile": float(percentile),
                    "weak_eigengap": float(arrays["weak_eigengap"][index]),
                    "minimum_adjacent_eigengap": float(
                        arrays["min_eigengap"][index]
                    ),
                }
            )
        _write_csv(
            output / "clamp20_diagnostics.csv",
            clamp_rows,
            list(clamp_rows[0]) if clamp_rows else [],
        )
        _write_json(output / "clamp20_control_summary.json", clamp_summary)

    audit_payload = {
        "schema": AUDIT_SCHEMA,
        "status": "passed",
        "created_utc": utc_now(),
        "metadata": metadata,
        "global": global_summary,
        "clamp_diagnostic": clamp_summary,
        "phase_b_allowed": phase_b_allowed,
        "phase_b_decision": (
            "continue"
            if phase_b_allowed
            else "stop_complete_non_support_and_report_gpt"
        ),
    }
    audit_path = output / "observability_audit.json"
    _write_json(audit_path, audit_payload)

    output_hashes = {}
    for path in sorted(output.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json":
            output_hashes[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "schema": AUDIT_SCHEMA,
        "status": "passed",
        "created_utc": utc_now(),
        "scene_name": args.scene_name,
        "input": {
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
            "anchor_ply": {"path": str(anchor_ply_path), "sha256": ply_sha},
            "train_list": {
                "path": str(train_list_path),
                "sha256": train_list_sha,
            },
            "binding_manifest": {
                "path": str(binding_path),
                "sha256": binding_sha,
            },
        },
        "output": output_hashes,
        "cache_sha256": output_hashes[cache_path.name]["sha256"],
        "cache_semantic_sha256": semantic_cache_sha,
        "phase_b_allowed": phase_b_allowed,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        },
        "audit_storage": {
            "moment_storage": metadata["moment_storage"],
            "moment_storage_bytes": metadata["moment_storage_bytes"],
            "training_cache_is_compact": True,
        },
    }
    _write_json(output / "manifest.json", manifest)
    return {
        "status": "passed",
        "output": str(output),
        "cache": str(cache_path),
        "cache_sha256": manifest["cache_sha256"],
        "cache_semantic_sha256": semantic_cache_sha,
        "eligible_count": eligible_count,
        "active_count": int(np.sum(active_np)),
        "phase_b_allowed": phase_b_allowed,
    }


def build_parser() -> argparse.ArgumentParser:
    from arguments import ModelParams, PipelineParams

    parser = argparse.ArgumentParser(description=__doc__)
    ModelParams(parser)
    PipelineParams(parser)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--anchor-ply", required=True, type=Path)
    parser.add_argument("--anchor-iteration", required=True, type=int)
    parser.add_argument("--binding-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--clamp-indices", type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--expected-ply-sha256", default="")
    parser.add_argument("--expected-train-list-sha256", default="")
    parser.add_argument("--expected-binding-sha256", default="")
    parser.add_argument("--expected-clamp-sha256", default="")
    parser.add_argument("--expected-clamp-count", type=int)
    parser.add_argument("--controls-per-clamp", type=int, default=5)
    parser.add_argument("--eig-chunk-size", type=int, default=100_000)
    parser.add_argument("--top-risk-count", type=int, default=200)
    parser.add_argument("--trace-tolerance", type=float, default=1e-6)
    parser.add_argument("--eigen-tolerance", type=float, default=1e-7)
    parser.add_argument("--optimizer_type", default="default")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise OgsAuditError("--model_path is required")
    if not args.source_path:
        raise OgsAuditError("--source_path is required")
    if not args.train_list:
        raise OgsAuditError("--train_list is required")
    if args.anchor_iteration <= 0:
        raise OgsAuditError("--anchor-iteration must be positive")
    if args.eig_chunk_size <= 0:
        raise OgsAuditError("--eig-chunk-size must be positive")
    if args.top_risk_count <= 0:
        raise OgsAuditError("--top-risk-count must be positive")
    if args.controls_per_clamp <= 0:
        raise OgsAuditError("--controls-per-clamp must be positive")
    if args.expected_clamp_count is not None and args.expected_clamp_count <= 0:
        raise OgsAuditError("--expected-clamp-count must be positive")
    if args.clamp_indices is None and (
        args.expected_clamp_sha256 or args.expected_clamp_count is not None
    ):
        raise OgsAuditError(
            "clamp expected hash/count requires --clamp-indices"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_cli_args(args)
        result = _run_gpu_audit(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
