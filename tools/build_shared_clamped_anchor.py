#!/usr/bin/env python3
"""Build a shared RGB anchor with the legacy Stage-2 scale clamp.

This tool performs no optimization.  It loads one formal RGB checkpoint,
invokes ``GaussianModel.clamp_scaling_max_`` exactly once, and writes an
independent model directory.  The original model is treated as read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import numpy as np
from plyfile import PlyData
import torch
from torch import nn


class SharedAnchorError(RuntimeError):
    """Fail-closed shared-anchor construction error."""


MODEL_FIELDS = {
    "xyz": 1,
    "features_dc": 2,
    "features_rest": 3,
    "scaling": 4,
    "rotation": 5,
    "opacity": 6,
    "max_radii2D": 7,
    "xyz_gradient_accum": 8,
    "denom": 9,
}
UNCHANGED_MODEL_FIELDS = tuple(name for name in MODEL_FIELDS if name != "scaling")
COPIED_MODEL_FILES = ("cfg_args", "cameras.json", "exposure.json")
SCALE_PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _percentile_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {}
    if not np.all(np.isfinite(array)):
        raise SharedAnchorError("scale distribution contains NaN/Inf")
    return {
        f"p{percentile}": float(np.percentile(array, percentile))
        for percentile in SCALE_PERCENTILES
    }


def _scale_distribution(
    activated_scaling: torch.Tensor,
    selected_indices: list[int] | None = None,
) -> dict[str, Any]:
    scales = activated_scaling.detach().cpu().numpy().astype(np.float64, copy=False)
    if selected_indices is not None:
        scales = scales[np.asarray(selected_indices, dtype=np.int64)]
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise SharedAnchorError(
            f"activated scaling must have shape (N, 3), got {scales.shape}"
        )
    if scales.shape[0] == 0:
        return {"count": 0, "percentiles": {}}
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise SharedAnchorError("activated scaling distribution must be finite and positive")

    maximum = np.max(scales, axis=1)
    minimum = np.min(scales, axis=1)
    geometric_mean = np.exp(np.mean(np.log(scales), axis=1))
    return {
        "count": int(scales.shape[0]),
        "percentiles": {
            "axis_0": _percentile_summary(scales[:, 0]),
            "axis_1": _percentile_summary(scales[:, 1]),
            "axis_2": _percentile_summary(scales[:, 2]),
            "per_gaussian_max": _percentile_summary(maximum),
            "per_gaussian_min": _percentile_summary(minimum),
            "anisotropy_max_over_min": _percentile_summary(maximum / minimum),
            "geometric_mean": _percentile_summary(geometric_mean),
            "volume_proxy": _percentile_summary(np.prod(scales, axis=1)),
        },
    }


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha(path: Path, expected: str | None, label: str) -> str:
    actual = _sha256(path)
    if expected is not None and actual != expected.lower():
        raise SharedAnchorError(
            f"{label} SHA-256 mismatch: expected={expected.lower()} actual={actual}"
        )
    return actual


def _checkpoint_paths(model_dir: Path, iteration: int) -> tuple[Path, Path]:
    checkpoint = model_dir / f"chkpnt{iteration}.pth"
    ply = model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    return checkpoint, ply


def _load_checkpoint(path: Path, iteration: int) -> tuple[list[Any], int]:
    # Formal checkpoints are trusted project artifacts and include optimizer
    # objects that are intentionally incompatible with weights_only=True.
    payload = torch.load(str(path), weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise SharedAnchorError("checkpoint must be a (model_params, iteration) pair")
    model_params, actual_iteration = payload
    if int(actual_iteration) != int(iteration):
        raise SharedAnchorError(
            f"checkpoint iteration mismatch: expected={iteration} actual={actual_iteration}"
        )
    if not isinstance(model_params, (tuple, list)) or len(model_params) != 12:
        raise SharedAnchorError(
            "checkpoint model payload does not match GaussianModel.capture()"
        )
    params = list(model_params)
    gaussian_count = int(params[MODEL_FIELDS["xyz"]].shape[0])
    shapes = {
        "xyz": (gaussian_count, 3),
        "features_dc": (gaussian_count, 1, 3),
        "scaling": (gaussian_count, 3),
        "rotation": (gaussian_count, 4),
        "opacity": (gaussian_count, 1),
    }
    for name, expected_shape in shapes.items():
        tensor = params[MODEL_FIELDS[name]]
        if not isinstance(tensor, torch.Tensor):
            raise SharedAnchorError(f"checkpoint field {name} is not a tensor")
        if tuple(tensor.shape) != expected_shape:
            raise SharedAnchorError(
                f"checkpoint {name} shape mismatch: "
                f"expected={expected_shape} actual={tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise SharedAnchorError(f"checkpoint field {name} contains NaN/Inf")
    features_rest = params[MODEL_FIELDS["features_rest"]]
    if (
        not isinstance(features_rest, torch.Tensor)
        or features_rest.ndim != 3
        or features_rest.shape[0] != gaussian_count
        or features_rest.shape[2] != 3
        or not bool(torch.isfinite(features_rest).all())
    ):
        raise SharedAnchorError("checkpoint features_rest has invalid shape or values")
    return params, gaussian_count


def _infer_sh_degree(features_rest: torch.Tensor) -> int:
    coefficient_count = int(features_rest.shape[1]) + 1
    root = int(round(math.sqrt(coefficient_count)))
    if root * root != coefficient_count or root < 1:
        raise SharedAnchorError(
            f"cannot infer SH degree from {coefficient_count} coefficients"
        )
    return root - 1


def _ply_fields(path: Path) -> dict[str, np.ndarray]:
    ply = PlyData.read(str(path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise SharedAnchorError("PLY must contain exactly one vertex element")
    vertex = ply.elements[0]
    result: dict[str, np.ndarray] = {}
    for prop in vertex.properties:
        result[prop.name] = np.asarray(vertex[prop.name]).copy()
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required - set(result))
    if missing:
        raise SharedAnchorError(f"PLY is missing required fields: {missing}")
    return result


def _stack(fields: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    names = sorted(
        (name for name in fields if name.startswith(prefix)),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    return np.stack([fields[name] for name in names], axis=1)


def _checkpoint_ply_arrays(params: list[Any]) -> dict[str, np.ndarray]:
    xyz = params[MODEL_FIELDS["xyz"]].detach().cpu().numpy()
    features_dc = (
        params[MODEL_FIELDS["features_dc"]]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    features_rest = (
        params[MODEL_FIELDS["features_rest"]]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    return {
        "xyz": xyz,
        "features_dc": features_dc,
        "features_rest": features_rest,
        "opacity": params[MODEL_FIELDS["opacity"]].detach().cpu().numpy(),
        "scaling": params[MODEL_FIELDS["scaling"]].detach().cpu().numpy(),
        "rotation": params[MODEL_FIELDS["rotation"]].detach().cpu().numpy(),
    }


def _ply_group_arrays(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "xyz": np.stack([fields["x"], fields["y"], fields["z"]], axis=1),
        "features_dc": _stack(fields, "f_dc_"),
        "features_rest": _stack(fields, "f_rest_"),
        "opacity": fields["opacity"][:, None],
        "scaling": _stack(fields, "scale_"),
        "rotation": _stack(fields, "rot_"),
    }


def _validate_checkpoint_ply_exact(
    params: list[Any], fields: dict[str, np.ndarray], label: str
) -> dict[str, str]:
    checkpoint_arrays = _checkpoint_ply_arrays(params)
    ply_arrays = _ply_group_arrays(fields)
    hashes: dict[str, str] = {}
    for name, checkpoint_array in checkpoint_arrays.items():
        ply_array = ply_arrays[name]
        if checkpoint_array.shape != ply_array.shape:
            raise SharedAnchorError(
                f"{label} checkpoint/PLY {name} shape mismatch: "
                f"{checkpoint_array.shape} vs {ply_array.shape}"
            )
        if not np.array_equal(checkpoint_array, ply_array, equal_nan=False):
            raise SharedAnchorError(
                f"{label} checkpoint/PLY {name} values are not byte-exact"
            )
        hashes[name] = _array_sha256(checkpoint_array)
    return hashes


def _model_tensor_hashes(params: list[Any]) -> dict[str, str]:
    return {
        name: _tensor_sha256(params[index])
        for name, index in MODEL_FIELDS.items()
    }


def _assign_gaussian_fields(gaussians: Any, params: list[Any]) -> None:
    gaussians.active_sh_degree = int(params[0])
    for name in (
        "xyz",
        "features_dc",
        "features_rest",
        "scaling",
        "rotation",
        "opacity",
    ):
        tensor = params[MODEL_FIELDS[name]]
        setattr(
            gaussians,
            f"_{name}",
            nn.Parameter(tensor.detach().clone().requires_grad_(True)),
        )
    gaussians.max_radii2D = params[MODEL_FIELDS["max_radii2D"]].detach().clone()
    gaussians.xyz_gradient_accum = (
        params[MODEL_FIELDS["xyz_gradient_accum"]].detach().clone()
    )
    gaussians.denom = params[MODEL_FIELDS["denom"]].detach().clone()
    gaussians.spatial_lr_scale = params[11]


def _load_gaussian_model_class():
    """Load the exact GaussianModel file without importing scene.__init__.

    ``scene.__init__`` imports the full camera/render stack, which is unrelated
    to this checkpoint-only operation.  Loading the module file directly keeps
    the sidecar usable in a minimal CPU test environment while invoking the
    repository's canonical ``clamp_scaling_max_`` implementation.
    """

    module_path = Path(__file__).resolve().parents[1] / "scene" / "gaussian_model.py"
    repository_root = str(module_path.parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)
    spec = importlib.util.spec_from_file_location(
        "_uav_tgs_shared_anchor_gaussian_model", module_path
    )
    if spec is None or spec.loader is None:
        raise SharedAnchorError(f"cannot load GaussianModel module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GaussianModel


def build_shared_anchor(args: argparse.Namespace) -> dict[str, Any]:
    input_model = Path(args.input_model_dir).resolve()
    output_model = Path(args.output_model_dir).resolve()
    if input_model == output_model:
        raise SharedAnchorError("input and output model directories must differ")
    if output_model.exists():
        raise SharedAnchorError(f"refusing to overwrite output: {output_model}")
    if not input_model.is_dir():
        raise SharedAnchorError(f"input model directory is missing: {input_model}")

    iteration = int(args.anchor_iteration)
    input_checkpoint, input_ply = _checkpoint_paths(input_model, iteration)
    if not input_checkpoint.is_file() or not input_ply.is_file():
        raise SharedAnchorError("input checkpoint or PLY is missing")
    input_checkpoint_sha = _require_sha(
        input_checkpoint, args.expected_checkpoint_sha256, "input checkpoint"
    )
    input_ply_sha = _require_sha(input_ply, args.expected_ply_sha256, "input PLY")

    params, gaussian_count = _load_checkpoint(input_checkpoint, iteration)
    input_tensor_hashes = _model_tensor_hashes(params)
    input_ply_fields = _ply_fields(input_ply)
    input_ply_group_hashes = _validate_checkpoint_ply_exact(
        params, input_ply_fields, "input"
    )
    sh_degree = _infer_sh_degree(params[MODEL_FIELDS["features_rest"]])

    raw_scaling = params[MODEL_FIELDS["scaling"]]
    before_activated = torch.exp(raw_scaling.detach())
    threshold_log = math.log(float(args.max_scale))
    selected_mask = raw_scaling.detach().max(dim=1).values > threshold_log
    selected_indices = (
        torch.nonzero(selected_mask, as_tuple=False).flatten().cpu().tolist()
    )
    if len(selected_indices) != int(args.expected_clamped_count):
        raise SharedAnchorError(
            "legacy clamp row count mismatch: "
            f"expected={args.expected_clamped_count} actual={len(selected_indices)}"
        )

    # Import lazily so metadata-only CLI parsing and unit-test discovery do not
    # require the compiled CUDA extensions.
    GaussianModel = _load_gaussian_model_class()

    gaussians = GaussianModel(sh_degree)
    _assign_gaussian_fields(gaussians, params)
    stats = gaussians.clamp_scaling_max_(float(args.max_scale))
    clamped_count, total, before_smax, after_smax = stats
    if clamped_count != len(selected_indices) or total != gaussian_count:
        raise SharedAnchorError(
            "GaussianModel.clamp_scaling_max_ disagrees with the precomputed "
            f"selection: method={clamped_count}/{total} "
            f"selection={len(selected_indices)}/{gaussian_count}"
        )
    after_activated = gaussians.get_scaling.detach()
    if not bool(torch.isfinite(after_activated).all()):
        raise SharedAnchorError("clamped scaling contains NaN/Inf")
    if float(after_activated.max().item()) > float(args.max_scale) + 1e-6:
        raise SharedAnchorError("clamped activated scaling exceeds the threshold")

    output_params = list(params)
    output_params[MODEL_FIELDS["scaling"]] = gaussians._scaling
    output_tensor_hashes_prewrite = _model_tensor_hashes(output_params)
    changed_fields = sorted(
        name
        for name in MODEL_FIELDS
        if input_tensor_hashes[name] != output_tensor_hashes_prewrite[name]
    )
    if changed_fields != ["scaling"]:
        raise SharedAnchorError(
            f"unexpected changed model fields before write: {changed_fields}"
        )

    row_audit = []
    for index in selected_indices:
        row_audit.append(
            {
                "gaussian_index": int(index),
                "activated_scale_before": [
                    float(value) for value in before_activated[index].cpu().tolist()
                ],
                "activated_scale_after": [
                    float(value) for value in after_activated[index].cpu().tolist()
                ],
                "raw_log_scale_before": [
                    float(value) for value in raw_scaling[index].detach().cpu().tolist()
                ],
                "raw_log_scale_after": [
                    float(value)
                    for value in gaussians._scaling[index].detach().cpu().tolist()
                ],
            }
        )

    partial = output_model.with_name(
        f".{output_model.name}.partial-{uuid.uuid4().hex}"
    )
    partial.mkdir(parents=True, exist_ok=False)
    try:
        output_checkpoint, output_ply = _checkpoint_paths(partial, iteration)
        output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        torch.save((tuple(output_params), iteration), str(output_checkpoint))
        gaussians.save_ply(str(output_ply))

        copied_files: dict[str, dict[str, Any]] = {}
        for name in COPIED_MODEL_FILES:
            source = input_model / name
            if not source.is_file():
                if name == "cfg_args":
                    raise SharedAnchorError("input model is missing required cfg_args")
                continue
            destination = partial / name
            shutil.copy2(source, destination)
            copied_files[name] = {
                "input_sha256": _sha256(source),
                "output_sha256": _sha256(destination),
                "byte_exact": _sha256(source) == _sha256(destination),
            }
            if not copied_files[name]["byte_exact"]:
                raise SharedAnchorError(f"copied model file changed: {name}")

        output_checkpoint_sha = _sha256(output_checkpoint)
        output_ply_sha = _sha256(output_ply)
        reloaded_params, reloaded_count = _load_checkpoint(
            output_checkpoint, iteration
        )
        if reloaded_count != gaussian_count:
            raise SharedAnchorError("output checkpoint Gaussian count changed")
        output_tensor_hashes = _model_tensor_hashes(reloaded_params)
        output_ply_fields = _ply_fields(output_ply)
        output_ply_group_hashes = _validate_checkpoint_ply_exact(
            reloaded_params, output_ply_fields, "output"
        )

        changed_after_reload = sorted(
            name
            for name in MODEL_FIELDS
            if input_tensor_hashes[name] != output_tensor_hashes[name]
        )
        if changed_after_reload != ["scaling"]:
            raise SharedAnchorError(
                f"unexpected changed model fields after reload: {changed_after_reload}"
            )
        for name in UNCHANGED_MODEL_FIELDS:
            if input_tensor_hashes[name] != output_tensor_hashes[name]:
                raise SharedAnchorError(f"model field changed unexpectedly: {name}")
        for name in input_ply_group_hashes:
            if name == "scaling":
                continue
            if input_ply_group_hashes[name] != output_ply_group_hashes[name]:
                raise SharedAnchorError(f"PLY field group changed unexpectedly: {name}")

        output_scaling = reloaded_params[MODEL_FIELDS["scaling"]].detach()
        changed_elements = int((output_scaling != raw_scaling.detach()).sum().item())
        changed_rows = int(
            torch.any(output_scaling != raw_scaling.detach(), dim=1).sum().item()
        )
        if changed_rows != len(selected_indices):
            raise SharedAnchorError(
                f"changed scaling row count mismatch: {changed_rows}"
            )

        # Prove that the read-only inputs were not modified.
        input_checkpoint_sha_after = _sha256(input_checkpoint)
        input_ply_sha_after = _sha256(input_ply)
        if (
            input_checkpoint_sha_after != input_checkpoint_sha
            or input_ply_sha_after != input_ply_sha
        ):
            raise SharedAnchorError("input anchor changed during construction")

        final_checkpoint = output_model / f"chkpnt{iteration}.pth"
        final_ply = (
            output_model
            / "point_cloud"
            / f"iteration_{iteration}"
            / "point_cloud.ply"
        )
        manifest = {
            "schema": "uav-tgs-shared-clamped-rgb-anchor-v1",
            "status": "passed",
            "scene": str(args.scene_name),
            "anchor_iteration": iteration,
            "operation": {
                "training_updates": 0,
                "legacy_method": "GaussianModel.clamp_scaling_max_",
                "invocation_count": 1,
                "max_activated_scale": float(args.max_scale),
                "threshold_raw_log": threshold_log,
                "selection_rule": "max(raw_log_scaling, axis=1) > log(max_scale)",
                "changed_model_fields": ["scaling"],
                "optimizer_state_policy": (
                    "preserved from the RGB anchor for provenance; the optional "
                    "downstream F3 uses a fresh appearance-only optimizer"
                ),
            },
            "source_code": {
                "commit": str(args.code_commit),
                "tool": "tools/build_shared_clamped_anchor.py",
            },
            "input": {
                "model_dir": str(input_model),
                "checkpoint": str(input_checkpoint),
                "checkpoint_sha256_before": input_checkpoint_sha,
                "checkpoint_sha256_after": input_checkpoint_sha_after,
                "ply": str(input_ply),
                "ply_sha256_before": input_ply_sha,
                "ply_sha256_after": input_ply_sha_after,
                "read_only_unchanged": True,
            },
            "output": {
                "model_dir": str(output_model),
                "checkpoint": str(final_checkpoint),
                "checkpoint_sha256": output_checkpoint_sha,
                "ply": str(final_ply),
                "ply_sha256": output_ply_sha,
                "copied_model_files": copied_files,
            },
            "counts": {
                "gaussian_count_before": gaussian_count,
                "gaussian_count_after": reloaded_count,
                "expected_clamped_gaussians": int(args.expected_clamped_count),
                "actual_clamped_gaussians": clamped_count,
                "changed_scaling_rows": changed_rows,
                "changed_scaling_elements": changed_elements,
            },
            "scale_summary": {
                "before_global_max": before_smax,
                "after_global_max": after_smax,
                "all_gaussians_before": _scale_distribution(before_activated),
                "all_gaussians_after": _scale_distribution(after_activated),
                "clamped_rows_before": _scale_distribution(
                    before_activated, selected_indices
                ),
                "clamped_rows_after": _scale_distribution(
                    after_activated, selected_indices
                ),
            },
            "clamped_indices": selected_indices,
            "clamped_indices_sha256": _canonical_json_sha256(selected_indices),
            "clamped_rows": row_audit,
            "tensor_hashes": {
                "input_checkpoint": input_tensor_hashes,
                "output_checkpoint": output_tensor_hashes,
                "input_ply_groups": input_ply_group_hashes,
                "output_ply_groups": output_ply_group_hashes,
            },
            "invariants": {
                "only_scaling_changed": changed_after_reload == ["scaling"],
                "xyz_exact": input_tensor_hashes["xyz"]
                == output_tensor_hashes["xyz"],
                "rotation_exact": input_tensor_hashes["rotation"]
                == output_tensor_hashes["rotation"],
                "opacity_exact": input_tensor_hashes["opacity"]
                == output_tensor_hashes["opacity"],
                "sh_exact": (
                    input_tensor_hashes["features_dc"]
                    == output_tensor_hashes["features_dc"]
                    and input_tensor_hashes["features_rest"]
                    == output_tensor_hashes["features_rest"]
                ),
                "topology_exact": gaussian_count == reloaded_count,
                "gaussian_count_exact": gaussian_count == reloaded_count,
                "checkpoint_ply_exact_after": True,
                "input_anchor_unchanged": True,
                "no_training": True,
            },
        }
        manifest_path = partial / "shared_clamp_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (partial / "SHARED_ANCHOR_STATUS").write_text("passed\n", encoding="ascii")
        os.replace(partial, output_model)
        return manifest
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--input-model-dir", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--anchor-iteration", type=int, default=30000)
    parser.add_argument("--max-scale", type=float, required=True)
    parser.add_argument("--expected-clamped-count", type=int, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-ply-sha256")
    parser.add_argument("--code-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.anchor_iteration < 0:
        raise SystemExit("--anchor-iteration must be nonnegative")
    if not math.isfinite(args.max_scale) or args.max_scale <= 0:
        raise SystemExit("--max-scale must be finite and positive")
    if args.expected_clamped_count < 0:
        raise SystemExit("--expected-clamped-count must be nonnegative")
    try:
        manifest = build_shared_anchor(args)
    except SharedAnchorError as error:
        print(f"shared-anchor clamp failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
