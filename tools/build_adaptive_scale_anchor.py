#!/usr/bin/env python3
"""Build scene-tail or sparse-context scale-projected RGB anchors.

This is a zero-update sidecar.  It changes only the largest activated scale
axis of rows that violate the selected robust fence, preserves Gaussian order
and topology, and writes a new model directory with a fail-closed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import numpy as np
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.build_shared_clamped_anchor import (
    COPIED_MODEL_FILES,
    MODEL_FIELDS,
    UNCHANGED_MODEL_FIELDS,
    SharedAnchorError,
    _array_sha256,
    _canonical_json_sha256,
    _checkpoint_paths,
    _infer_sh_degree,
    _load_checkpoint,
    _load_gaussian_model_class,
    _model_tensor_hashes,
    _percentile_summary,
    _ply_fields,
    _require_sha,
    _scale_distribution,
    _sha256,
    _validate_checkpoint_ply_exact,
)
from utils.sparse_support import (
    VoxelHashNN,
    load_colmap_points3D,
    resolve_colmap_model_dir,
)


class AdaptiveScaleError(SharedAnchorError):
    """Fail-closed adaptive scale projection error."""


EPS = 1e-12
ROBUST_IQR_MULTIPLIER = 3.0


def robust_upper_fence(values: torch.Tensor) -> dict[str, float]:
    """Return Q3 + 3*IQR for a finite non-empty one-dimensional tensor."""

    flat = values.detach().to(dtype=torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise AdaptiveScaleError("robust fence values must be finite and non-empty")
    q1 = torch.quantile(flat, 0.25)
    q3 = torch.quantile(flat, 0.75)
    iqr = q3 - q1
    cutoff = q3 + ROBUST_IQR_MULTIPLIER * iqr
    return {
        "q1": float(q1.item()),
        "q3": float(q3.item()),
        "iqr": float(iqr.item()),
        "multiplier": ROBUST_IQR_MULTIPLIER,
        "cutoff": float(cutoff.item()),
    }


def scene_tail_thresholds(activated_scaling: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one scene-level activated-scale threshold for every Gaussian."""

    if activated_scaling.ndim != 2 or activated_scaling.shape[1] != 3:
        raise AdaptiveScaleError("activated scaling must have shape (N, 3)")
    smax = activated_scaling.max(dim=1).values
    if not bool(torch.isfinite(smax).all()) or bool((smax <= 0).any()):
        raise AdaptiveScaleError("activated scale must be finite and positive")
    fence = robust_upper_fence(torch.log(smax + EPS))
    tau_scene = math.exp(fence["cutoff"])
    if not math.isfinite(tau_scene) or tau_scene <= 0:
        raise AdaptiveScaleError("scene-tail threshold is not finite and positive")
    return torch.full_like(smax, tau_scene), {**fence, "tau_scene": tau_scene}


def scsp_thresholds(
    activated_scaling: torch.Tensor,
    local_support: torch.Tensor,
    scene_thresholds: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute local sparse-context thresholds with scene-tail fallback."""

    smax = activated_scaling.max(dim=1).values
    local = local_support.to(device=smax.device, dtype=smax.dtype).reshape(-1)
    if local.shape != smax.shape or scene_thresholds.shape != smax.shape:
        raise AdaptiveScaleError("SCSP tensor lengths do not match Gaussian count")
    valid = torch.isfinite(local) & (local > 0)
    if not bool(valid.any()):
        return scene_thresholds.clone(), {
            "valid_support_count": 0,
            "fallback_count": int(smax.numel()),
            "local_fence": None,
        }
    ratio_log = torch.log((smax[valid] + EPS) / (local[valid] + EPS))
    fence = robust_upper_fence(ratio_log)
    local_multiplier = math.exp(fence["cutoff"])
    local_tau = torch.full_like(smax, float("inf"))
    local_tau[valid] = local[valid] * local_multiplier
    threshold = torch.minimum(local_tau, scene_thresholds)
    threshold[~valid] = scene_thresholds[~valid]
    if not bool(torch.isfinite(threshold).all()) or bool((threshold <= 0).any()):
        raise AdaptiveScaleError("SCSP thresholds must be finite and positive")
    return threshold, {
        "valid_support_count": int(valid.sum().item()),
        "fallback_count": int((~valid).sum().item()),
        "local_fence": {**fence, "exp_cutoff": local_multiplier},
    }


def project_largest_scale_axis(
    raw_log_scaling: torch.Tensor, thresholds: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project only the largest principal scale axis to its row threshold."""

    activated = torch.exp(raw_log_scaling.detach())
    smax, max_axis = activated.max(dim=1)
    thresholds = thresholds.to(device=activated.device, dtype=activated.dtype).reshape(-1)
    if thresholds.shape != smax.shape:
        raise AdaptiveScaleError("threshold count does not match Gaussian count")
    selected = smax > thresholds
    output = raw_log_scaling.detach().clone()
    rows = torch.nonzero(selected, as_tuple=False).flatten()
    if rows.numel() > 0:
        output[rows, max_axis[rows]] = torch.log(thresholds[rows])
    changed_per_row = (output != raw_log_scaling.detach()).sum(dim=1)
    if bool((changed_per_row > 1).any()):
        raise AdaptiveScaleError("more than one scale axis changed in a Gaussian")
    if int((changed_per_row > 0).sum().item()) != int(rows.numel()):
        raise AdaptiveScaleError("selected and changed scaling rows disagree")
    return output, selected, max_axis


def query_second_support_distance(
    xyz: torch.Tensor,
    sparse_root: Path,
    voxel_size: float,
    max_voxel_radius: int,
    chunk_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build the established voxel-centroid support and query second distance."""

    model_dir = resolve_colmap_model_dir(sparse_root)
    point_cloud = load_colmap_points3D(model_dir)
    if point_cloud.xyz.shape[0] == 0:
        return torch.full((xyz.shape[0],), float("inf"), device=xyz.device), {
            "model_dir": str(model_dir.resolve()),
            "raw_point_count": 0,
            "voxel_centroid_count": 0,
            "points3d_sha256": None,
        }
    support = torch.as_tensor(point_cloud.xyz, dtype=torch.float32, device=xyz.device)
    index = VoxelHashNN(support, voxel_size=voxel_size).to(xyz.device)
    pieces = []
    for start in range(0, int(xyz.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(xyz.shape[0]))
        distances = index.query_topk_torch(
            xyz[start:stop], k=2, max_voxel_radius=max_voxel_radius
        )
        pieces.append(distances[:, 1])
    second = torch.cat(pieces, dim=0) if pieces else torch.empty(0, device=xyz.device)
    points_path = model_dir / "points3D.bin"
    if not points_path.is_file():
        points_path = model_dir / "points3D.txt"
    return second, {
        "model_dir": str(model_dir.resolve()),
        "raw_point_count": int(point_cloud.xyz.shape[0]),
        "voxel_centroid_count": int(index._centroids_sorted.shape[0]),
        "voxel_size": float(voxel_size),
        "max_voxel_radius": int(max_voxel_radius),
        "local_support_statistic": "second_nearest_voxel_centroid_distance",
        "points3d_path": str(points_path.resolve()),
        "points3d_sha256": _sha256(points_path),
    }


def _assign_model(gaussians: Any, params: list[Any], new_scaling: torch.Tensor) -> None:
    gaussians.active_sh_degree = int(params[0])
    for name in ("xyz", "features_dc", "features_rest", "rotation", "opacity"):
        tensor = params[MODEL_FIELDS[name]]
        setattr(gaussians, f"_{name}", nn.Parameter(tensor.detach().clone().requires_grad_(True)))
    gaussians._scaling = nn.Parameter(new_scaling.detach().clone().requires_grad_(True))
    gaussians.max_radii2D = params[MODEL_FIELDS["max_radii2D"]].detach().clone()
    gaussians.xyz_gradient_accum = params[MODEL_FIELDS["xyz_gradient_accum"]].detach().clone()
    gaussians.denom = params[MODEL_FIELDS["denom"]].detach().clone()
    gaussians.spatial_lr_scale = params[11]


def _load_absolute_indices(path: str | None, gaussian_count: int) -> tuple[set[int], dict[str, Any] | None]:
    if not path:
        return set(), None
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    indices = payload.get("clamped_indices")
    if not isinstance(indices, list) or any(not isinstance(value, int) for value in indices):
        raise AdaptiveScaleError("absolute clamp manifest has invalid clamped_indices")
    unique = set(indices)
    if len(unique) != len(indices) or any(value < 0 or value >= gaussian_count for value in unique):
        raise AdaptiveScaleError("absolute clamp indices are duplicate or out of range")
    return unique, {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "count": len(unique),
    }


def build_adaptive_anchor(args: argparse.Namespace) -> dict[str, Any]:
    input_model = Path(args.input_model_dir).resolve()
    output_model = Path(args.output_model_dir).resolve()
    if input_model == output_model:
        raise AdaptiveScaleError("input and output model directories must differ")
    if output_model.exists():
        raise AdaptiveScaleError(f"refusing to overwrite output: {output_model}")
    if not input_model.is_dir():
        raise AdaptiveScaleError(f"input model directory is missing: {input_model}")

    iteration = int(args.anchor_iteration)
    input_checkpoint, input_ply = _checkpoint_paths(input_model, iteration)
    if not input_checkpoint.is_file() or not input_ply.is_file():
        raise AdaptiveScaleError("input checkpoint or PLY is missing")
    checkpoint_sha = _require_sha(input_checkpoint, args.expected_checkpoint_sha256, "input checkpoint")
    ply_sha = _require_sha(input_ply, args.expected_ply_sha256, "input PLY")
    params, gaussian_count = _load_checkpoint(input_checkpoint, iteration)
    input_hashes = _model_tensor_hashes(params)
    input_ply_fields = _ply_fields(input_ply)
    input_ply_hashes = _validate_checkpoint_ply_exact(params, input_ply_fields, "input")

    raw_scaling = params[MODEL_FIELDS["scaling"]].detach()
    activated_before = torch.exp(raw_scaling)
    scene_threshold, scene_stats = scene_tail_thresholds(activated_before)
    support_distance = torch.full_like(scene_threshold, float("inf"))
    support_manifest = None
    scsp_stats = None
    if args.method == "scene_tail":
        row_thresholds = scene_threshold
    elif args.method == "scsp":
        if not args.sparse_root:
            raise AdaptiveScaleError("--sparse-root is required for SCSP")
        support_distance, support_manifest = query_second_support_distance(
            params[MODEL_FIELDS["xyz"]].detach(),
            Path(args.sparse_root),
            float(args.support_voxel_size),
            int(args.support_max_voxel_radius),
            int(args.query_chunk_size),
        )
        row_thresholds, scsp_stats = scsp_thresholds(
            activated_before, support_distance, scene_threshold
        )
    else:
        raise AdaptiveScaleError(f"unsupported method: {args.method}")

    new_scaling, selected_mask, max_axis = project_largest_scale_axis(raw_scaling, row_thresholds)
    indices = torch.nonzero(selected_mask, as_tuple=False).flatten().cpu().tolist()
    activated_after = torch.exp(new_scaling)
    abs_indices, abs_manifest = _load_absolute_indices(args.absolute_clamp_manifest, gaussian_count)
    overlap = sorted(abs_indices.intersection(indices))

    GaussianModel = _load_gaussian_model_class()
    sh_degree = _infer_sh_degree(params[MODEL_FIELDS["features_rest"]])
    gaussians = GaussianModel(sh_degree)
    _assign_model(gaussians, params, new_scaling)
    output_params = list(params)
    output_params[MODEL_FIELDS["scaling"]] = gaussians._scaling
    prewrite_hashes = _model_tensor_hashes(output_params)
    changed_fields = sorted(name for name in MODEL_FIELDS if input_hashes[name] != prewrite_hashes[name])
    expected_changed_fields = ["scaling"] if indices else []
    if changed_fields != expected_changed_fields:
        raise AdaptiveScaleError(f"unexpected changed model fields: {changed_fields}")

    row_audit = []
    for index in indices:
        local_value = float(support_distance[index].item())
        fallback = not math.isfinite(local_value) or local_value <= 0
        row_audit.append(
            {
                "gaussian_index": int(index),
                "maximum_axis": int(max_axis[index].item()),
                "support_source": (
                    "scene_tail"
                    if args.method == "scene_tail"
                    else (
                        "scene_tail_fallback"
                        if fallback
                        else "second_nearest_voxel_centroid"
                    )
                ),
                "local_support_distance": None if fallback else local_value,
                "row_threshold": float(row_thresholds[index].item()),
                "activated_scale_before": [float(v) for v in activated_before[index].cpu().tolist()],
                "activated_scale_after": [float(v) for v in activated_after[index].cpu().tolist()],
            }
        )

    partial = output_model.with_name(f".{output_model.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        output_checkpoint, output_ply = _checkpoint_paths(partial, iteration)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        torch.save((tuple(output_params), iteration), str(output_checkpoint))
        gaussians.save_ply(str(output_ply))
        copied_files = {}
        for name in COPIED_MODEL_FILES:
            source = input_model / name
            if not source.is_file():
                if name == "cfg_args":
                    raise AdaptiveScaleError("input model is missing required cfg_args")
                continue
            destination = partial / name
            shutil.copy2(source, destination)
            if _sha256(source) != _sha256(destination):
                raise AdaptiveScaleError(f"copied model file changed: {name}")
            copied_files[name] = {"sha256": _sha256(destination), "byte_exact": True}

        reloaded, reloaded_count = _load_checkpoint(output_checkpoint, iteration)
        output_hashes = _model_tensor_hashes(reloaded)
        output_ply_hashes = _validate_checkpoint_ply_exact(reloaded, _ply_fields(output_ply), "output")
        changed_after = sorted(name for name in MODEL_FIELDS if input_hashes[name] != output_hashes[name])
        if changed_after != expected_changed_fields:
            raise AdaptiveScaleError(f"unexpected changed fields after reload: {changed_after}")
        for name in UNCHANGED_MODEL_FIELDS:
            if input_hashes[name] != output_hashes[name]:
                raise AdaptiveScaleError(f"model field changed unexpectedly: {name}")
        for name, before_hash in input_ply_hashes.items():
            if name != "scaling" and before_hash != output_ply_hashes[name]:
                raise AdaptiveScaleError(f"PLY group changed unexpectedly: {name}")
        if reloaded_count != gaussian_count:
            raise AdaptiveScaleError("Gaussian count changed")

        output_scaling = reloaded[MODEL_FIELDS["scaling"]].detach()
        changed_elements_per_row = (output_scaling != raw_scaling).sum(dim=1)
        if bool((changed_elements_per_row > 1).any()):
            raise AdaptiveScaleError("output changed more than one scale axis per row")
        changed_rows = int((changed_elements_per_row > 0).sum().item())
        if changed_rows != len(indices):
            raise AdaptiveScaleError("output changed row count disagrees with selection")
        if _sha256(input_checkpoint) != checkpoint_sha or _sha256(input_ply) != ply_sha:
            raise AdaptiveScaleError("input anchor changed during construction")

        shrink = (
            activated_before[selected_mask].max(dim=1).values
            / activated_after[selected_mask].max(dim=1).values
        ).detach().cpu().numpy() if indices else np.empty(0, dtype=np.float32)
        manifest = {
            "schema": "uav-tgs-adaptive-scale-anchor-v1",
            "status": "passed",
            "scene": str(args.scene_name),
            "method": str(args.method),
            "anchor_iteration": iteration,
            "source_code": {"commit": str(args.code_commit), "tool": "tools/build_adaptive_scale_anchor.py"},
            "input": {
                "model_dir": str(input_model),
                "checkpoint": str(input_checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "ply": str(input_ply),
                "ply_sha256": ply_sha,
                "read_only_unchanged": True,
            },
            "output": {
                "model_dir": str(output_model),
                "checkpoint_sha256": _sha256(output_checkpoint),
                "ply_sha256": _sha256(output_ply),
                "copied_model_files": copied_files,
            },
            "formula": {
                "scene_tail": "exp(Q3(log(smax+eps)) + 3*IQR(log(smax+eps)))",
                "scsp_local": "d_i * exp(Q3(log((smax+eps)/(d_i+eps))) + 3*IQR)",
                "row_threshold": "scene_tail" if args.method == "scene_tail" else "min(scsp_local, scene_tail); scene_tail fallback",
                "projection": "replace only largest principal activated scale axis",
                "eps": EPS,
            },
            "thresholds": {"scene_tail": scene_stats, "scsp": scsp_stats},
            "sparse_support": support_manifest,
            "absolute_clamp_diagnostic": {
                "manifest": abs_manifest,
                "overlap_count": len(overlap),
                "overlap_fraction_of_adaptive": len(overlap) / len(indices) if indices else 0.0,
                "overlap_fraction_of_absolute": len(overlap) / len(abs_indices) if abs_indices else None,
                "overlap_indices": overlap,
            },
            "counts": {
                "gaussian_count_before": gaussian_count,
                "gaussian_count_after": reloaded_count,
                "modified_gaussians": len(indices),
                "fallback_gaussians_all": int(scsp_stats["fallback_count"]) if scsp_stats is not None else 0,
                "fallback_gaussians_modified": sum(row["support_source"] == "scene_tail_fallback" for row in row_audit),
                "changed_scaling_elements": int(changed_elements_per_row.sum().item()),
            },
            "scale_summary": {
                "all_before": _scale_distribution(activated_before),
                "all_after": _scale_distribution(activated_after),
                "modified_before": _scale_distribution(activated_before, indices),
                "modified_after": _scale_distribution(activated_after, indices),
                "shrink_ratio": _percentile_summary(shrink),
            },
            "modified_indices": indices,
            "modified_indices_sha256": _canonical_json_sha256(indices),
            "modified_rows": row_audit,
            "tensor_hashes": {"input": input_hashes, "output": output_hashes},
            "invariants": {
                "only_scaling_changed": changed_after == expected_changed_fields,
                "at_most_one_scale_axis_changed_per_row": True,
                "xyz_exact": input_hashes["xyz"] == output_hashes["xyz"],
                "rotation_exact": input_hashes["rotation"] == output_hashes["rotation"],
                "opacity_exact": input_hashes["opacity"] == output_hashes["opacity"],
                "sh_exact": input_hashes["features_dc"] == output_hashes["features_dc"] and input_hashes["features_rest"] == output_hashes["features_rest"],
                "schema_index_count_exact": True,
                "topology_exact": gaussian_count == reloaded_count,
                "input_anchor_unchanged": True,
                "no_training": True,
            },
        }
        manifest_path = partial / "adaptive_scale_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (partial / "ADAPTIVE_SCALE_STATUS").write_text("passed\n", encoding="ascii")
        os.replace(partial, output_model)
        return manifest
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--method", choices=["scene_tail", "scsp"], required=True)
    parser.add_argument("--input-model-dir", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--anchor-iteration", type=int, default=30000)
    parser.add_argument("--sparse-root")
    parser.add_argument("--support-voxel-size", type=float, default=1.5)
    parser.add_argument("--support-max-voxel-radius", type=int, default=2)
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument("--absolute-clamp-manifest")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-ply-sha256")
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.anchor_iteration < 0:
        raise SystemExit("--anchor-iteration must be nonnegative")
    if not math.isfinite(args.support_voxel_size) or args.support_voxel_size <= 0:
        raise SystemExit("--support-voxel-size must be finite and positive")
    if args.support_max_voxel_radius < 0 or args.query_chunk_size <= 0:
        raise SystemExit("support radius must be nonnegative and chunk size positive")
    try:
        manifest = build_adaptive_anchor(args)
    except (AdaptiveScaleError, FileNotFoundError, ValueError) as error:
        print(f"adaptive scale anchor failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
