#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import hashlib
import json
import os
import torch
from efficiency_probe import TorchStageProbe
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.temperature_loss import (
    FORMULA_VERSION as TEMPERATURE_LOSS_FORMULA_VERSION,
    SCALAR_GRADIENT_TARGET,
    SPATIAL_GRADIENT_TARGET,
    TemperatureTargetStore,
    adjacent_lut_tau,
    canonical_lut_tensor,
    load_calibration_manifest,
    temperature_consistency_losses,
)

# Optional: pseudo-color thermal structure loss (added for improved thermal texture preservation)
try:
    from utils.loss_utils import structure_grad_loss
except Exception:
    structure_grad_loss = None

from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
from utils.camera_sequence import (
    camera_parameters_hash,
    camera_lookup,
    load_sequence_manifest,
    ordered_camera_names,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


_STRICT_FREEZE_FIELDS = ("_xyz", "_scaling", "_rotation", "_opacity")
_OPACITY_ADAPTIVE_FREEZE_FIELDS = ("_xyz", "_scaling", "_rotation")
_RGB_CONTINUATION_GROUPS = (
    "xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"
)
_RGB_APPEARANCE_GROUPS = ("f_dc", "f_rest")


def _optimizer_state_step(value):
    if torch.is_tensor(value):
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise RuntimeError("Optimizer step must be one finite scalar")
        return int(value.detach().cpu().item())
    return int(value)


def _validate_rgb_continuation_checkpoint(
    model_params, checkpoint_iteration, expected_anchor_iteration
):
    """Fail closed unless the anchor contains a restorable six-group RGB Adam."""

    if int(checkpoint_iteration) != int(expected_anchor_iteration):
        raise RuntimeError(
            "RGB continuation anchor iteration mismatch: "
            f"expected={expected_anchor_iteration} actual={checkpoint_iteration}"
        )
    if not isinstance(model_params, (tuple, list)) or len(model_params) != 12:
        raise RuntimeError("RGB continuation requires the standard 12-field checkpoint")
    optimizer_state = model_params[10]
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("RGB continuation checkpoint has no optimizer state_dict")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, dict):
        raise RuntimeError("RGB continuation checkpoint optimizer state is malformed")
    names = tuple(group.get("name") for group in groups)
    if names != _RGB_CONTINUATION_GROUPS:
        raise RuntimeError(
            "RGB continuation requires the standard six-group RGB Adam: "
            f"expected={_RGB_CONTINUATION_GROUPS} actual={names}"
        )

    steps = {}
    for group in groups:
        parameters = group.get("params")
        name = group.get("name")
        if not isinstance(parameters, list) or len(parameters) != 1:
            raise RuntimeError(f"RGB optimizer group {name!r} must contain one tensor")
        state = states.get(parameters[0])
        if not isinstance(state, dict):
            raise RuntimeError(f"RGB optimizer group {name!r} has no Adam state")
        if "step" not in state or "exp_avg" not in state or "exp_avg_sq" not in state:
            raise RuntimeError(f"RGB optimizer group {name!r} has incomplete Adam state")
        if not torch.is_tensor(state["exp_avg"]) or not torch.is_tensor(state["exp_avg_sq"]):
            raise RuntimeError(f"RGB optimizer group {name!r} has invalid moments")
        if state["exp_avg"].shape != state["exp_avg_sq"].shape:
            raise RuntimeError(f"RGB optimizer group {name!r} moment shapes differ")
        if not torch.isfinite(state["exp_avg"]).all() or not torch.isfinite(state["exp_avg_sq"]).all():
            raise RuntimeError(f"RGB optimizer group {name!r} has non-finite moments")
        steps[name] = _optimizer_state_step(state["step"])

    unique_steps = sorted(set(steps.values()))
    if (
        len(unique_steps) != 1
        or unique_steps[0] <= 0
        or unique_steps[0] > int(expected_anchor_iteration)
    ):
        raise RuntimeError(
            "RGB continuation Adam steps are not a coherent anchor state: "
            f"steps={steps} required=uniform_positive_at_most_"
            f"{int(expected_anchor_iteration)}"
        )
    return {
        "groups": list(names),
        "adam_steps": steps,
        "uniform_adam_step": unique_steps[0],
    }


def _validate_restored_rgb_optimizer(gaussians, expected_summary):
    names = tuple(group.get("name") for group in gaussians.optimizer.param_groups)
    if names != _RGB_CONTINUATION_GROUPS:
        raise RuntimeError(
            "Restored RGB optimizer group mismatch: "
            f"expected={_RGB_CONTINUATION_GROUPS} actual={names}"
        )
    restored_steps = {}
    for group in gaussians.optimizer.param_groups:
        name = group.get("name")
        if len(group.get("params", [])) != 1:
            raise RuntimeError(f"Restored RGB optimizer group {name!r} is malformed")
        state = gaussians.optimizer.state.get(group["params"][0])
        if not isinstance(state, dict) or "step" not in state:
            raise RuntimeError(f"Restored RGB optimizer group {name!r} lost Adam state")
        restored_steps[name] = _optimizer_state_step(state["step"])
    if restored_steps != expected_summary["adam_steps"]:
        raise RuntimeError(
            "Restored RGB Adam steps differ from checkpoint: "
            f"expected={expected_summary['adam_steps']} actual={restored_steps}"
        )
    return restored_steps


def _should_optimizer_step(iteration, final_iteration, step_at_final_iteration):
    return int(iteration) < int(final_iteration) or bool(step_at_final_iteration)


def _validate_rgb_continuation_schedule(
    anchor_iteration,
    scheduler_horizon,
    updates,
    final_iteration,
    step_at_final_iteration,
):
    if int(anchor_iteration) != 30000:
        raise ValueError("RGB continuation requires the formal RGB 30000 anchor")
    if int(scheduler_horizon) != int(anchor_iteration):
        raise ValueError("Scheduler horizon must remain at the anchor iteration")
    if int(updates) not in (200, 5000):
        raise ValueError("RGB continuation updates must be 200 or 5000")
    expected_final = int(anchor_iteration) + int(updates)
    if int(final_iteration) != expected_final:
        raise ValueError(f"RGB continuation final iteration must be {expected_final}")
    if not bool(step_at_final_iteration):
        raise ValueError("RGB continuation requires the final optimizer step")
    return expected_final


def _vector_norm(gradient):
    if gradient is None:
        return 0.0
    return float(torch.linalg.vector_norm(gradient.detach()).item())


def _gradient_cosine(left, right, eps=1e-30):
    if left is None or right is None:
        return None
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    denominator = torch.linalg.vector_norm(left_flat) * torch.linalg.vector_norm(right_flat)
    if float(denominator.item()) <= eps:
        return None
    return float(torch.dot(left_flat, right_flat).div(denominator).item())


def _lr_scaled_gradient_probe(rgb_grads, ogs_grads, scaling_lr, rotation_lr, lambda_ogs):
    rgb_scaling = _vector_norm(rgb_grads[0])
    rgb_rotation = _vector_norm(rgb_grads[1])
    ogs_scaling = _vector_norm(ogs_grads[0])
    ogs_rotation = _vector_norm(ogs_grads[1])
    weighted_ogs_scaling = float(lambda_ogs) * ogs_scaling
    weighted_ogs_rotation = float(lambda_ogs) * ogs_rotation
    rgb_proxy = (
        (float(scaling_lr) * rgb_scaling) ** 2
        + (float(rotation_lr) * rgb_rotation) ** 2
    ) ** 0.5
    ogs_proxy = (
        (float(scaling_lr) * weighted_ogs_scaling) ** 2
        + (float(rotation_lr) * weighted_ogs_rotation) ** 2
    ) ** 0.5
    return {
        "raw": {
            "rgb_scaling_norm": rgb_scaling,
            "rgb_rotation_norm": rgb_rotation,
            "ogs_scaling_norm": ogs_scaling,
            "ogs_rotation_norm": ogs_rotation,
            "weighted_ogs_scaling_norm": weighted_ogs_scaling,
            "weighted_ogs_rotation_norm": weighted_ogs_rotation,
            "scaling_cosine": _gradient_cosine(rgb_grads[0], ogs_grads[0]),
            "rotation_cosine": _gradient_cosine(rgb_grads[1], ogs_grads[1]),
        },
        "lr_scaled": {
            "scaling_lr": float(scaling_lr),
            "rotation_lr": float(rotation_lr),
            "rgb_scaling_update_proxy": float(scaling_lr) * rgb_scaling,
            "rgb_rotation_update_proxy": float(rotation_lr) * rgb_rotation,
            "weighted_ogs_scaling_update_proxy": float(scaling_lr) * weighted_ogs_scaling,
            "weighted_ogs_rotation_update_proxy": float(rotation_lr) * weighted_ogs_rotation,
            "rgb_combined_update_proxy": rgb_proxy,
            "weighted_ogs_combined_update_proxy": ogs_proxy,
            "combined_ratio": (ogs_proxy / rgb_proxy) if rgb_proxy > 0.0 else None,
            "scaling_ratio": (
                weighted_ogs_scaling / rgb_scaling if rgb_scaling > 0.0 else None
            ),
            "rotation_ratio": (
                weighted_ogs_rotation / rgb_rotation if rgb_rotation > 0.0 else None
            ),
        },
    }


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def _tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _ogs_anchor_tensor_sha256(tensor):
    """Match tools/audit_ogs_v1.py's dtype+shape+bytes tensor hash."""

    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _combined_gradient_norm(gradients):
    finite = [gradient.detach() for gradient in gradients if gradient is not None]
    if not finite:
        return 0.0
    squared = sum(float(gradient.pow(2).sum().item()) for gradient in finite)
    return squared ** 0.5


def _calibrate_temperature_loss(
    scene,
    gaussians,
    pipe,
    background,
    dataset,
    opt,
    target_store,
    tau,
    calibration_views,
):
    """One train-only batch calibration of auxiliary SH gradient magnitudes."""

    train_cameras = sorted(scene.getTrainCameras(), key=lambda camera: str(camera.image_name))
    count = min(int(calibration_views), len(train_cameras))
    if count <= 0:
        raise RuntimeError("temperature loss calibration has no train cameras")
    indices = sorted({min((index * len(train_cameras)) // count, len(train_cameras) - 1) for index in range(count)})
    cameras = [train_cameras[index] for index in indices]
    appearance_parameters = (gaussians._features_dc, gaussians._features_rest)
    image_norms, scalar_norms, gradient_norms = [], [], []
    loss_records = []
    for camera in cameras:
        rendered = render(
            camera,
            gaussians,
            pipe,
            background,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )["render"]
        alpha = None
        if camera.alpha_mask is not None:
            alpha = camera.alpha_mask.cuda()
            rendered = rendered * alpha
        ground_truth = camera.original_image.cuda()
        image_l1 = l1_loss(rendered, ground_truth)
        if FUSED_SSIM_AVAILABLE:
            image_ssim = fused_ssim(rendered.unsqueeze(0), ground_truth.unsqueeze(0))
        else:
            image_ssim = ssim(rendered, ground_truth)
        image_loss = (1.0 - opt.lambda_dssim) * image_l1 + opt.lambda_dssim * (1.0 - image_ssim)
        if float(getattr(args, "t_struct_grad_w", 0.0)) > 0.0:
            if structure_grad_loss is None:
                raise RuntimeError("temperature calibration requires structure_grad_loss")
            image_loss = image_loss + float(args.t_struct_grad_w) * structure_grad_loss(
                rendered,
                ground_truth,
                mask=alpha,
                normalize=bool(getattr(args, "t_struct_grad_norm", True)),
            )
        target_u, support = target_store.get(
            camera.image_name,
            int(rendered.shape[-2]),
            int(rendered.shape[-1]),
            rendered.device,
        )
        if alpha is not None:
            support = support * alpha.to(device=support.device, dtype=support.dtype)
        auxiliary = temperature_consistency_losses(
            rendered,
            target_u,
            mask=support,
            tau=tau,
            chunk_pixels=int(getattr(args, "temperature_lut_chunk_pixels", 16384)),
        )
        image_gradients = torch.autograd.grad(
            image_loss, appearance_parameters, retain_graph=True, allow_unused=True
        )
        image_norm = _combined_gradient_norm(image_gradients)
        del image_gradients
        scalar_gradients = torch.autograd.grad(
            auxiliary["scalar"], appearance_parameters, retain_graph=True, allow_unused=True
        )
        scalar_norm = _combined_gradient_norm(scalar_gradients)
        del scalar_gradients
        spatial_gradients = torch.autograd.grad(
            auxiliary["gradient"], appearance_parameters, retain_graph=False, allow_unused=True
        )
        gradient_norm = _combined_gradient_norm(spatial_gradients)
        del spatial_gradients
        values = (image_norm, scalar_norm, gradient_norm)
        if any(not torch.isfinite(torch.tensor(value)).item() for value in values):
            raise RuntimeError("temperature calibration produced non-finite gradient norms")
        image_norms.append(image_norm)
        scalar_norms.append(scalar_norm)
        gradient_norms.append(gradient_norm)
        loss_records.append(
            {
                "image_name": str(camera.image_name),
                "image_loss": float(image_loss.detach().item()),
                "scalar_loss": float(auxiliary["scalar"].detach().item()),
                "gradient_loss": float(auxiliary["gradient"].detach().item()),
                "image_sh_grad_norm": image_norm,
                "scalar_sh_grad_norm": scalar_norm,
                "gradient_sh_grad_norm": gradient_norm,
                "valid_fraction": float(auxiliary["valid_fraction"]),
            }
        )
        del rendered, ground_truth, image_loss, auxiliary, target_u, support
    mean_image = sum(image_norms) / len(image_norms)
    mean_scalar = sum(scalar_norms) / len(scalar_norms)
    mean_gradient = sum(gradient_norms) / len(gradient_norms)
    if min(mean_image, mean_scalar, mean_gradient) <= 0.0:
        raise RuntimeError(
            "temperature calibration requires positive mean SH gradient norms: "
            f"image={mean_image} scalar={mean_scalar} gradient={mean_gradient}"
        )
    return {
        "schema": "uav-tgs-temperature-loss-calibration-v1",
        "status": "passed",
        "formula_version": TEMPERATURE_LOSS_FORMULA_VERSION,
        "lut_sha256": target_store.metadata()["lut_sha256"],
        "tau": float(tau),
        "tau_rule": "median positive adjacent squared distance of normalized fixed LUT",
        "camera_selection": "evenly_spaced_over_sorted_train_image_names",
        "camera_names": [str(camera.image_name) for camera in cameras],
        "train_only": True,
        "targets": {
            "scalar_aux_over_image_sh_gradient": SCALAR_GRADIENT_TARGET,
            "gradient_aux_over_image_sh_gradient": SPATIAL_GRADIENT_TARGET,
        },
        "mean_gradient_norms": {
            "image": mean_image,
            "scalar_unweighted": mean_scalar,
            "gradient_unweighted": mean_gradient,
        },
        "lambda_temp": SCALAR_GRADIENT_TARGET * mean_image / mean_scalar,
        "lambda_grad": SPATIAL_GRADIENT_TARGET * mean_image / mean_gradient,
        "records": loss_records,
        "temperature_target": target_store.metadata(),
    }


def _normalized_sha256(value, label):
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise RuntimeError(f"{label} must be a lowercase 64-character SHA-256")
    return normalized


def _verified_sha256_binding(label, expected, actual):
    expected_sha256 = _normalized_sha256(expected, f"expected {label} SHA-256")
    actual_sha256 = _normalized_sha256(actual, f"actual {label} SHA-256")
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return {
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "verified": True,
    }


def _snapshot_strict_freeze_fields(gaussians):
    return {
        name: getattr(gaussians, name).detach().cpu().contiguous().clone()
        for name in _STRICT_FREEZE_FIELDS
    }


def _write_strict_freeze_audit(
    model_path,
    gaussians,
    before,
    start_iteration,
    final_iteration,
    *,
    schema="uav-tgs-strict-freeze-audit-v1",
    filename="strict_freeze_audit.json",
    label="StrictFreezeAudit",
):
    fields = {}
    all_unchanged = True
    for name, expected in before.items():
        actual = getattr(gaussians, name).detach().cpu().contiguous()
        same_shape = tuple(actual.shape) == tuple(expected.shape)
        same_dtype = actual.dtype == expected.dtype
        unchanged = bool(same_shape and same_dtype and torch.equal(actual, expected))
        all_unchanged = all_unchanged and unchanged
        max_abs_diff = None
        if same_shape and actual.numel() > 0:
            max_abs_diff = float(torch.max(torch.abs(actual - expected)).item())
        elif same_shape:
            max_abs_diff = 0.0
        fields[name] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "before_sha256": _tensor_sha256(expected),
            "after_sha256": _tensor_sha256(actual),
            "max_abs_diff": max_abs_diff,
            "unchanged": unchanged,
        }

    payload = {
        "schema": schema,
        "status": "passed" if all_unchanged else "failed",
        "start_iteration": int(start_iteration),
        "final_iteration": int(final_iteration),
        "fields": fields,
    }
    output_path = os.path.join(model_path, filename)
    temporary_path = output_path + f".tmp-{os.getpid()}"
    os.makedirs(model_path, exist_ok=True)
    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    if not all_unchanged:
        changed = [name for name, item in fields.items() if not item["unchanged"]]
        raise RuntimeError(
            "Strict thermal freeze invariant failed for: " + ", ".join(changed)
        )
    print(f"[INFO] {label}: passed path={output_path}")
    return payload


def _snapshot_opacity_adaptive_freeze_fields(gaussians):
    return {
        name: getattr(gaussians, name).detach().cpu().contiguous().clone()
        for name in _OPACITY_ADAPTIVE_FREEZE_FIELDS
    }


def _write_opacity_adaptive_freeze_audit(
    model_path, gaussians, before, start_iteration, final_iteration
):
    fields = {}
    all_unchanged = True
    for name, expected in before.items():
        actual = getattr(gaussians, name).detach().cpu().contiguous()
        same_shape = tuple(actual.shape) == tuple(expected.shape)
        same_dtype = actual.dtype == expected.dtype
        unchanged = bool(same_shape and same_dtype and torch.equal(actual, expected))
        all_unchanged = all_unchanged and unchanged
        max_abs_diff = None
        if same_shape and actual.numel() > 0:
            max_abs_diff = float(torch.max(torch.abs(actual - expected)).item())
        elif same_shape:
            max_abs_diff = 0.0
        fields[name] = {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "before_sha256": _tensor_sha256(expected),
            "after_sha256": _tensor_sha256(actual),
            "max_abs_diff": max_abs_diff,
            "unchanged": unchanged,
        }

    opacity = gaussians.get_opacity.detach().cpu().contiguous().reshape(-1)
    opacity_finite = bool(torch.isfinite(opacity).all().item())
    opacity_summary = {
        "count": int(opacity.numel()),
        "finite": opacity_finite,
        "min": float(torch.min(opacity).item()) if opacity.numel() and opacity_finite else None,
        "max": float(torch.max(opacity).item()) if opacity.numel() and opacity_finite else None,
    }
    passed = all_unchanged and opacity_finite
    payload = {
        "schema": "uav-tgs-opacity-adaptive-freeze-audit-v1",
        "status": "passed" if passed else "failed",
        "start_iteration": int(start_iteration),
        "final_iteration": int(final_iteration),
        "frozen_fields": fields,
        "activated_opacity": opacity_summary,
    }
    output_path = os.path.join(model_path, "opacity_adaptive_freeze_audit.json")
    temporary_path = output_path + f".tmp-{os.getpid()}"
    os.makedirs(model_path, exist_ok=True)
    with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    if not passed:
        changed = [name for name, item in fields.items() if not item["unchanged"]]
        if not opacity_finite:
            changed.append("activated_opacity_nonfinite")
        raise RuntimeError(
            "Opacity-adaptive thermal invariant failed for: " + ", ".join(changed)
        )
    print(f"[INFO] OpacityAdaptiveFreezeAudit: passed path={output_path}")
    return payload


def _save_iteration_artifacts(
    scene,
    gaussians,
    iteration,
    *,
    save_gaussians,
    save_checkpoint,
    thermal_max_sh_degree=None,
):
    """Save requested artifacts from one unchanged in-memory model state."""
    if not (save_gaussians or save_checkpoint):
        return
    if thermal_max_sh_degree is not None:
        gaussians.zero_sh_above_degree_(thermal_max_sh_degree)
    if save_gaussians:
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        scene.save(iteration)
    if save_checkpoint:
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save(
            (gaussians.capture(), iteration),
            os.path.join(scene.model_path, "chkpnt" + str(iteration) + ".pth"),
        )


def _resolve_artifact_save_semantics(thermal_recipe, requested_semantics):
    """Resolve the endpoint-save protocol without changing legacy defaults."""
    recipe = str(thermal_recipe)
    requested = None if requested_semantics is None else str(requested_semantics)
    if requested not in (None, "legacy", "aligned"):
        raise ValueError(f"Unsupported artifact save semantics: {requested!r}")
    if recipe in ("aaai_strict", "geometry_frozen_opacity_adaptive"):
        if requested == "legacy":
            raise ValueError(
                f"--thermal_recipe {recipe} requires "
                "--artifact_save_semantics aligned"
            )
        return "aligned"
    return "legacy" if requested is None else requested


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, ss_args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    thermal_freeze_mode = str(getattr(args, "thermal_freeze_mode", "legacy"))
    rgb_continuation_recipe = str(
        getattr(args, "rgb_continuation_recipe", "legacy")
    )
    rgb_continuation = rgb_continuation_recipe in (
        "fixed_topology", "appearance_only"
    )
    rgb_appearance_refit = rgb_continuation_recipe == "appearance_only"
    ogs_enabled = bool(getattr(args, "ogs_v1", False))
    strict_freeze = thermal_freeze_mode == "strict"
    opacity_adaptive = thermal_freeze_mode == "geometry_frozen_opacity_adaptive"
    topology_frozen = rgb_continuation or thermal_freeze_mode in (
        "strict", "continuous_unfrozen", "geometry_frozen_opacity_adaptive"
    )
    artifact_save_semantics = _resolve_artifact_save_semantics(
        getattr(args, "thermal_recipe", "legacy"),
        getattr(args, "artifact_save_semantics", None),
    )
    aligned_artifact_saves = (
        artifact_save_semantics == "aligned" or rgb_continuation
    )
    if aligned_artifact_saves:
        print(
            "[INFO] AAAIArtifactSaveSemantics: "
            "semantics=aligned aligned_post_optimizer_step=1"
        )
    if topology_frozen and not checkpoint:
        raise RuntimeError(
            "Fixed-topology training requires a start checkpoint: "
            f"rgb_continuation={rgb_continuation} "
            f"thermal_freeze_mode={thermal_freeze_mode}"
        )

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # The fixed sequence manifest is keyed to Scene(shuffle=False) camera
    # ordering emitted by the audit sidecar.  Legacy training retains its
    # historical shuffled Scene construction.
    scene = Scene(dataset, gaussians, shuffle=(not rgb_continuation))


    # Optional: sparse support gating for densification (disabled by default).
    # NOTE: When enabled, the index is built and pinned to the model device once
    # before training starts. There must be no CPU/Numpy fallback in the densify hot-path.
    if ss_args is not None and getattr(ss_args, "ss_enable", False) and not topology_frozen:
        try:
            from utils import sparse_support as _ss
            from scene.colmap_loader import load_colmap_sparse_xyz as _load_colmap_sparse_xyz

            dev = gaussians.get_xyz.device

            support_t = None
            src_used = None

            # 1) Prefer COLMAP sparse points (if requested).
            if getattr(ss_args, "ss_source", "colmap_sparse") == "colmap_sparse":
                model_dir = _ss.resolve_colmap_model_dir(os.path.join(dataset.source_path, "sparse"))
                xyz_np = _load_colmap_sparse_xyz(model_dir)
                if xyz_np is not None and getattr(xyz_np, "shape", None) is not None and xyz_np.shape[0] > 0:
                    # Build support tensor directly on the same device as gaussians.
                    support_t = torch.tensor(xyz_np, dtype=torch.float32, device=dev)
                    src_used = "colmap_sparse"

            # 2) Fallback: initial gaussian point cloud (already on dev).
            if support_t is None or support_t.numel() == 0:
                support_t = gaussians.get_xyz.detach()
                src_used = "init_pcd"

            # 3) Guard: empty/invalid support => disable.
            if support_t is None or support_t.ndim != 2 or support_t.shape[-1] != 3 or support_t.shape[0] == 0:
                print("[WARN] SparseSupport enabled but support points are empty/invalid; disabling sparse support.")
            else:
                use_aabb = bool(getattr(ss_args, "ss_use_aabb", True))
                aabb_arg = None
                margin = float(getattr(ss_args, "ss_aabb_margin", 0.0) or 0.0)
                if use_aabb:
                    # AABB (+ optional margin)
                    lo = support_t.min(dim=0).values
                    hi = support_t.max(dim=0).values
                    if margin != 0.0:
                        lo = lo - margin
                        hi = hi + margin
                    aabb_arg = (lo, hi)

                voxel = getattr(ss_args, "ss_voxel_size", None)
                nn_thr = getattr(ss_args, "ss_nn_dist_thr", None)
                vhnn = None

                # If user requests NN gating but no voxel index can be built, gracefully ignore NN gating.
                if nn_thr is not None and voxel is None:
                    print("[WARN] SparseSupport: ss_nn_dist_thr set but ss_voxel_size is None; ignoring nn_dist_thr (AABB-only).")
                    nn_thr = None

                if voxel is not None:
                    # Build index on the model device; GaussianModel will cache/pin it once.
                    vhnn = _ss.VoxelHashNN(support_t, voxel_size=float(voxel))

                # NN-only mode: when AABB is disabled, NN params must be valid.
                if (not use_aabb) and (vhnn is None or nn_thr is None):
                    print("[WARN] SparseSupport: ss_use_aabb=False requires both ss_voxel_size and ss_nn_dist_thr; disabling sparse support.")
                else:
                    gaussians.set_sparse_support(
                        aabb=aabb_arg,
                        index=vhnn,
                        nn_dist_thr=nn_thr,
                        adaptive_nn=bool(getattr(ss_args, "ss_adaptive_nn", False)),
                        adaptive_alpha=float(getattr(ss_args, "ss_adaptive_alpha", 1.0)),
                        adaptive_beta=float(getattr(ss_args, "ss_adaptive_beta", 0.0)),
                        adaptive_max_scale=float(getattr(ss_args, "ss_adaptive_max_scale", 1.5)),
                        trim_tail_pct=float(getattr(ss_args, "ss_trim_tail_pct", 0.0)),
                        drop_small_islands=int(getattr(ss_args, "ss_drop_small_islands", 0)),
                        island_radius=getattr(ss_args, "ss_island_radius", None),
                    )
                    print(f"[INFO] SparseSupport enabled: source={src_used}, use_aabb={use_aabb}, margin={margin}, voxel={voxel}, nn_thr={nn_thr}")
                    if bool(getattr(ss_args, "ss_adaptive_nn", False)) or float(getattr(ss_args, "ss_trim_tail_pct", 0.0)) > 0.0 or int(getattr(ss_args, "ss_drop_small_islands", 0)) > 0:
                        print(
                            f"[INFO] SparseSupport refine: adaptive_nn={bool(getattr(ss_args, 'ss_adaptive_nn', False))} "
                            f"alpha={float(getattr(ss_args, 'ss_adaptive_alpha', 1.0))} "
                            f"beta={float(getattr(ss_args, 'ss_adaptive_beta', 0.0))} "
                            f"max_scale={float(getattr(ss_args, 'ss_adaptive_max_scale', 1.5))} "
                            f"trim_tail_pct={float(getattr(ss_args, 'ss_trim_tail_pct', 0.0))} "
                            f"drop_small_islands={int(getattr(ss_args, 'ss_drop_small_islands', 0))} "
                            f"island_radius={getattr(ss_args, 'ss_island_radius', None)}"
                        )

        except Exception:
            print("[WARN] SparseSupport enabled but initialization failed; disabling sparse support.")
    gaussians.training_setup(opt)
    rgb_optimizer_summary = None
    if checkpoint:
        # Checkpoints are generated locally by this training pipeline and
        # contain NumPy/Python objects in addition to tensor weights.  PyTorch
        # 2.6+ defaults to weights_only=True, which rejects that established
        # checkpoint format before thermal fine-tuning can start.
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        optimizer_restore_mode = (
            str(getattr(args, "rgb_optimizer_state", "restore"))
            if rgb_continuation
            else str(getattr(args, "thermal_optimizer_state", "restore"))
        )
        if rgb_continuation:
            if rgb_appearance_refit:
                if optimizer_restore_mode != "fresh":
                    raise RuntimeError("RGB appearance refit requires a fresh optimizer")
                if int(first_iter) != int(
                    getattr(args, "rgb_continuation_anchor_iteration", 30000)
                ):
                    raise RuntimeError(
                        "RGB appearance-refit anchor iteration mismatch: "
                        f"expected={getattr(args, 'rgb_continuation_anchor_iteration', 30000)} "
                        f"actual={first_iter}"
                    )
                if not isinstance(model_params, (tuple, list)) or len(model_params) != 12:
                    raise RuntimeError(
                        "RGB appearance refit requires the standard 12-field checkpoint"
                    )
                rgb_optimizer_summary = {
                    "groups": list(_RGB_APPEARANCE_GROUPS),
                    "adam_steps": {name: 0 for name in _RGB_APPEARANCE_GROUPS},
                    "uniform_adam_step": 0,
                    "initialization": "fresh",
                }
            else:
                if optimizer_restore_mode != "restore":
                    raise RuntimeError(
                        "Fixed-topology RGB continuation requires restored optimizer state"
                    )
                rgb_optimizer_summary = _validate_rgb_continuation_checkpoint(
                    model_params,
                    first_iter,
                    getattr(args, "rgb_continuation_anchor_iteration", 30000),
                )
        if opacity_adaptive and optimizer_restore_mode == "restore":
            checkpoint_groups = tuple(
                group.get("name") for group in model_params[10].get("param_groups", [])
            )
            expected_checkpoint_groups = ("f_dc", "f_rest", "opacity")
            if checkpoint_groups != expected_checkpoint_groups:
                raise RuntimeError(
                    "Opacity-adaptive resume requires an A3 checkpoint optimizer: "
                    f"expected={expected_checkpoint_groups} actual={checkpoint_groups}"
                )
        gaussians.restore(model_params, opt, optimizer_restore_mode=optimizer_restore_mode)
        if rgb_continuation and not rgb_appearance_refit:
            restored_steps = _validate_restored_rgb_optimizer(
                gaussians, rgb_optimizer_summary
            )
            if gaussians.active_sh_degree != gaussians.max_sh_degree:
                raise RuntimeError(
                    "RGB continuation requires the formal SH3 anchor: "
                    f"active={gaussians.active_sh_degree} max={gaussians.max_sh_degree}"
                )
            if dataset.train_test_exp:
                raise RuntimeError(
                    "This RGB anchor checkpoint does not serialize exposure Adam; "
                    "a train_test_exp continuation cannot be restored exactly"
                )
            # Recreate both scheduler functions with the original Stage-1
            # horizon, not the 35k continuation endpoint.
            scheduler_horizon = int(
                getattr(args, "rgb_continuation_scheduler_horizon", 30000)
            )
            gaussians.xyz_scheduler_args = get_expon_lr_func(
                lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
                lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
                lr_delay_mult=opt.position_lr_delay_mult,
                max_steps=scheduler_horizon,
            )
            gaussians.exposure_scheduler_args = get_expon_lr_func(
                opt.exposure_lr_init,
                opt.exposure_lr_final,
                lr_delay_steps=opt.exposure_lr_delay_steps,
                lr_delay_mult=opt.exposure_lr_delay_mult,
                max_steps=scheduler_horizon,
            )
            print(
                "[INFO] RGBContinuationRestore: "
                f"groups={rgb_optimizer_summary['groups']} "
                f"adam_steps={restored_steps} scheduler_horizon={scheduler_horizon}"
            )
        elif rgb_appearance_refit:
            if gaussians.active_sh_degree != gaussians.max_sh_degree:
                raise RuntimeError(
                    "RGB appearance refit requires the formal SH3 anchor: "
                    f"active={gaussians.active_sh_degree} max={gaussians.max_sh_degree}"
                )
            gaussians.training_setup_appearance_only(
                opt,
                sh_degree_cap=gaussians.max_sh_degree,
                preserve_feature_state=False,
            )
            optimizer_groups = tuple(
                group.get("name") for group in gaussians.optimizer.param_groups
            )
            if optimizer_groups != _RGB_APPEARANCE_GROUPS:
                raise RuntimeError(
                    "RGB appearance-refit optimizer groups mismatch: "
                    f"expected={_RGB_APPEARANCE_GROUPS} actual={optimizer_groups}"
                )
            if gaussians.optimizer.state:
                raise RuntimeError("RGB appearance refit inherited Adam state")
            print(
                "[INFO] RGBAppearanceRefit: appearance_only=1 optimizer=fresh "
                f"groups={list(optimizer_groups)}"
            )
        if optimizer_restore_mode == "fresh":
            state_entries = len(gaussians.optimizer.state)
            step_entries = sum(
                1
                for state in gaussians.optimizer.state.values()
                if isinstance(state, dict) and "step" in state
            )
            if state_entries != 0 or step_entries != 0:
                raise RuntimeError(
                    "Fresh optimizer restore inherited RGB optimizer state: "
                    f"state_entries={state_entries} step_entries={step_entries}"
                )
            print("[INFO] FreshOptimizerRestore: state_entries=0 step_entries=0")
    start_iteration = int(first_iter)
    if getattr(args, "ss_prune_before_thermal", False) and checkpoint and not topology_frozen:
        if not getattr(args, "ss_enable", False):
            print("[WARN] ss_prune_before_thermal set but ss_enable=False; skipping prune.")
        elif not (getattr(gaussians, "_ss_enabled", False) and gaussians._ss_is_enabled()):
            print("[WARN] ss_prune_before_thermal set but SparseSupport not configured; skipping prune.")
        else:
            prune_stats = gaussians.prune_outside_sparse_support()
            if prune_stats is not None:
                before, after = prune_stats
                removed = before - after
                keep_ratio = (after / float(before)) if before > 0 else 1.0
                print(f"[INFO] SparseSupport prune_before_thermal: before={before} after={after} removed={removed} keep_ratio={keep_ratio:.6f}")

    thermal_scale_clamp = str(getattr(args, "thermal_scale_clamp", "legacy"))
    if (
        thermal_scale_clamp != "off"
        and not strict_freeze
        and args.clamp_scale_max is not None
        and checkpoint
        and getattr(args, "start_checkpoint", None)
    ):
        clamped_gauss, total, before_smax, after_smax = gaussians.clamp_scaling_max_(args.clamp_scale_max)
        if clamped_gauss > 0:
            print(
                f"[INFO] ClampScaling thermal_after_restore: max_scale={args.clamp_scale_max} "
                f"clamped_gauss={clamped_gauss}/{total} before_smax={before_smax:.6f} after_smax={after_smax:.6f}"
            )

    if getattr(args, "thermal_reset_features", False) and checkpoint and getattr(args, "start_checkpoint", None):
        with torch.no_grad():
            if gaussians._features_dc is not None:
                gaussians._features_dc.zero_()
            if gaussians._features_rest is not None:
                gaussians._features_rest.zero_()
        adam_cleared = False
        try:
            for group in gaussians.optimizer.param_groups:
                name = group.get("name", None)
                if name not in ("f_dc", "f_rest"):
                    continue
                if not group.get("params"):
                    continue
                p = group["params"][0]
                st = gaussians.optimizer.state.get(p, None)
                if st is None:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in st and torch.is_tensor(st[key]):
                        st[key].zero_()
                        adam_cleared = True
        except Exception:
            adam_cleared = False
        print(f"[INFO] ThermalResetFeatures: sh_zeroed=1 adam_state_cleared={1 if adam_cleared else 0}")

    thermal_max_sh_degree = getattr(args, "thermal_max_sh_degree", None)
    if thermal_max_sh_degree is not None:
        if not (checkpoint and getattr(args, "start_checkpoint", None)):
            raise RuntimeError("thermal_max_sh_degree requires a start checkpoint")
        thermal_max_sh_degree = int(thermal_max_sh_degree)
        cold_restart = bool(getattr(args, "thermal_reset_features", False))
        zeroed_bands = gaussians.configure_sh_degree_cap_(
            thermal_max_sh_degree, cold_restart=cold_restart
        )

        # A cap without a full SH reset must also remove inherited Adam
        # momentum for inactive coefficients, otherwise zero-valued higher
        # bands can move even when the renderer supplies zero gradients.
        keep_rest = (thermal_max_sh_degree + 1) ** 2 - 1
        for group in gaussians.optimizer.param_groups:
            if group.get("name", None) != "f_rest" or not group.get("params"):
                continue
            param = group["params"][0]
            state = gaussians.optimizer.state.get(param, None)
            if not isinstance(state, dict):
                continue
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key, None)
                if torch.is_tensor(value) and value.shape == param.shape:
                    with torch.no_grad():
                        value[:, keep_rest:, :].zero_()
        print(
            "[INFO] ThermalSHColdRestart: "
            f"enabled={1 if cold_restart else 0} active_degree={gaussians.active_sh_degree} "
            f"max_degree={thermal_max_sh_degree} zeroed_rest_bands={zeroed_bands}"
        )

    strict_freeze_snapshot = None
    rgb_appearance_freeze_snapshot = None
    opacity_adaptive_freeze_snapshot = None
    if strict_freeze:
        strict_freeze_snapshot = _snapshot_strict_freeze_fields(gaussians)
        preserve_feature_state = str(
            getattr(args, "thermal_optimizer_state", "restore")
        ) == "restore"
        gaussians.training_setup_appearance_only(
            opt,
            sh_degree_cap=thermal_max_sh_degree,
            preserve_feature_state=preserve_feature_state,
        )
        optimizer_groups = [group.get("name") for group in gaussians.optimizer.param_groups]
        expected_groups = ["f_dc"] if thermal_max_sh_degree == 0 else ["f_dc", "f_rest"]
        if optimizer_groups != expected_groups:
            raise RuntimeError(
                "Strict thermal optimizer contains unexpected groups: "
                f"expected={expected_groups} actual={optimizer_groups}"
            )
        if gaussians.exposure_optimizer is not None:
            raise RuntimeError("Strict thermal mode must disable the exposure optimizer")
        print(
            "[INFO] StrictThermalFreeze: appearance_only=1 "
            f"optimizer_groups={optimizer_groups} preserve_feature_state={int(preserve_feature_state)}"
        )
    elif rgb_appearance_refit:
        rgb_appearance_freeze_snapshot = _snapshot_strict_freeze_fields(gaussians)
    elif opacity_adaptive:
        opacity_adaptive_freeze_snapshot = _snapshot_opacity_adaptive_freeze_fields(gaussians)
        preserve_optimizer_state = str(
            getattr(args, "thermal_optimizer_state", "restore")
        ) == "restore"
        gaussians.training_setup_geometry_frozen_opacity_adaptive(
            opt,
            sh_degree_cap=thermal_max_sh_degree,
            preserve_optimizer_state=preserve_optimizer_state,
        )
        optimizer_groups = [group.get("name") for group in gaussians.optimizer.param_groups]
        expected_groups = ["f_dc", "f_rest", "opacity"]
        if optimizer_groups != expected_groups:
            raise RuntimeError(
                "Opacity-adaptive thermal optimizer contains unexpected groups: "
                f"expected={expected_groups} actual={optimizer_groups}"
            )
        trainability = {
            "xyz": bool(gaussians._xyz.requires_grad),
            "f_dc": bool(gaussians._features_dc.requires_grad),
            "f_rest": bool(gaussians._features_rest.requires_grad),
            "opacity": bool(gaussians._opacity.requires_grad),
            "scaling": bool(gaussians._scaling.requires_grad),
            "rotation": bool(gaussians._rotation.requires_grad),
            "exposure": bool(gaussians._exposure.requires_grad),
        }
        expected_trainability = {
            "xyz": False,
            "f_dc": True,
            "f_rest": True,
            "opacity": True,
            "scaling": False,
            "rotation": False,
            "exposure": False,
        }
        if trainability != expected_trainability:
            raise RuntimeError(
                "Opacity-adaptive parameter trainability mismatch: "
                f"expected={expected_trainability} actual={trainability}"
            )
        if gaussians.exposure_optimizer is not None:
            raise RuntimeError("Opacity-adaptive thermal mode must disable the exposure optimizer")
        optimizer_lrs = {
            group.get("name"): float(group.get("lr", 0.0))
            for group in gaussians.optimizer.param_groups
        }
        if not torch.isfinite(torch.tensor(optimizer_lrs["opacity"])) or optimizer_lrs["opacity"] <= 0.0:
            raise RuntimeError(f"Invalid opacity-adaptive opacity LR: {optimizer_lrs['opacity']}")
        runtime_protocol = {
            "schema": "uav-tgs-opacity-adaptive-runtime-protocol-v1",
            "status": "passed",
            "thermal_recipe": str(getattr(args, "thermal_recipe", "legacy")),
            "thermal_freeze_mode": thermal_freeze_mode,
            "thermal_optimizer_state": str(getattr(args, "thermal_optimizer_state", "restore")),
            "thermal_max_sh_degree": thermal_max_sh_degree,
            "thermal_scale_clamp": str(getattr(args, "thermal_scale_clamp", "legacy")),
            "artifact_save_semantics": artifact_save_semantics,
            "optimizer_groups": optimizer_groups,
            "optimizer_lrs": optimizer_lrs,
            "trainability": trainability,
            "topology_frozen": True,
            "densification": False,
            "pruning": False,
            "opacity_reset": False,
        }
        runtime_path = os.path.join(scene.model_path, "opacity_adaptive_protocol.json")
        runtime_temporary = runtime_path + f".tmp-{os.getpid()}"
        with open(runtime_temporary, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(runtime_protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(runtime_temporary, runtime_path)
        print(
            "[INFO] GeometryFrozenOpacityAdaptive: "
            f"optimizer_groups={optimizer_groups} preserve_optimizer_state={int(preserve_optimizer_state)} "
            f"opacity_lr={optimizer_lrs['opacity']:.8f}"
        )

    def _reapply_lrs_after_restore() -> None:
        for group in gaussians.optimizer.param_groups:
            name = group.get("name", None)
            if name == "xyz":
                group["lr"] = opt.position_lr_init * gaussians.spatial_lr_scale
            elif name == "f_dc":
                group["lr"] = opt.feature_lr
            elif name == "f_rest":
                group["lr"] = opt.feature_lr / 20.0
            elif name == "opacity":
                group["lr"] = opt.opacity_lr
            elif name == "scaling":
                group["lr"] = opt.scaling_lr
            elif name == "rotation":
                group["lr"] = opt.rotation_lr

    if (
        checkpoint
        and getattr(args, "start_checkpoint", None)
        and not getattr(args, "sgf_disable", False)
        and not rgb_continuation
    ):
        _reapply_lrs_after_restore()

    debug_stats = bool(getattr(args, "debug_gaussian_stats", False)) and bool(checkpoint)
    def _log_gaussian_stats(tag: str) -> None:
        if not debug_stats:
            return
        with torch.no_grad():
            scales = gaussians.get_scaling
            if scales is None or scales.numel() == 0:
                return
            smax = torch.max(scales, dim=1).values if scales.dim() >= 2 else scales.reshape(-1)
            smax = smax.reshape(-1)
            scount = int(smax.numel())
            if scount > 0:
                sq = torch.quantile(smax, torch.tensor([0.5, 0.9, 0.95, 0.99], device=smax.device))
                smax_max = float(torch.max(smax).item())
            else:
                sq = torch.tensor([0.0, 0.0, 0.0, 0.0], device=smax.device)
                smax_max = 0.0

            op = gaussians.get_opacity
            op = op.reshape(-1)
            ocount = int(op.numel())
            if ocount > 0:
                oq = torch.quantile(op, torch.tensor([0.5, 0.9, 0.95, 0.99], device=op.device))
                op_max = float(torch.max(op).item())
            else:
                oq = torch.tensor([0.0, 0.0, 0.0, 0.0], device=op.device)
                op_max = 0.0

            msg = (
                f"[INFO] GaussianStats {tag}: "
                f"smax_count={scount} smax_p50={float(sq[0]):.6f} smax_p90={float(sq[1]):.6f} "
                f"smax_p95={float(sq[2]):.6f} smax_p99={float(sq[3]):.6f} smax_max={smax_max:.6f} "
                f"op_count={ocount} op_p50={float(oq[0]):.6f} op_p90={float(oq[1]):.6f} "
                f"op_p95={float(oq[2]):.6f} op_p99={float(oq[3]):.6f} op_max={op_max:.6f}"
            )
            if tag == "after_restore":
                lr_map = {}
                for g in gaussians.optimizer.param_groups:
                    n = g.get("name", None)
                    if n is not None:
                        lr_map[n] = g.get("lr", None)
                if "xyz" in lr_map:
                    msg += f" lr_xyz={lr_map['xyz']:.6f}"
                if "f_dc" in lr_map:
                    msg += f" lr_f_dc={lr_map['f_dc']:.6f}"
                if "f_rest" in lr_map:
                    msg += f" lr_f_rest={lr_map['f_rest']:.6f}"
                if "opacity" in lr_map:
                    msg += f" lr_opacity={lr_map['opacity']:.6f}"
                if "scaling" in lr_map:
                    msg += f" lr_scaling={lr_map['scaling']:.6f}"
                if "rotation" in lr_map:
                    msg += f" lr_rotation={lr_map['rotation']:.6f}"
            print(msg)

    _log_gaussian_stats("after_restore")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    temperature_loss_mode = str(getattr(args, "temperature_loss_mode", "none"))
    temperature_target_store = None
    temperature_loss_weights = {"lambda_temp": 0.0, "lambda_grad": 0.0}
    temperature_tau = None
    temperature_protocol = None
    if temperature_loss_mode != "none":
        if not strict_freeze:
            raise RuntimeError("temperature loss requires strict appearance-only Stage 2")
        temperature_target_store = TemperatureTargetStore(
            getattr(args, "temperature_gt_root"),
            getattr(args, "temperature_range_manifest"),
            getattr(args, "temperature_support_root", None) or None,
            max_cache_items=int(getattr(args, "temperature_target_cache_items", 1024)),
        )
        temperature_tau = adjacent_lut_tau(
            canonical_lut_tensor(torch.device("cuda"), torch.float32)
        )
        calibration_source = str(
            getattr(args, "temperature_loss_calibration", "manifest")
        )
        if calibration_source == "calibrate":
            calibration = _calibrate_temperature_loss(
                scene,
                gaussians,
                pipe,
                background,
                dataset,
                opt,
                temperature_target_store,
                temperature_tau,
                int(getattr(args, "temperature_calibration_views", 8)),
            )
            calibration_path = (
                getattr(args, "temperature_loss_calibration_manifest", "")
                or os.path.join(scene.model_path, "temperature_loss_calibration.json")
            )
            os.makedirs(os.path.dirname(os.path.abspath(calibration_path)), exist_ok=True)
            temporary_path = calibration_path + f".tmp-{os.getpid()}"
            with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(calibration, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, calibration_path)
            calibration = {
                **calibration,
                "manifest_path": os.path.abspath(calibration_path),
                "manifest_sha256": _file_sha256(calibration_path),
            }
        elif calibration_source == "manifest":
            calibration_path = getattr(args, "temperature_loss_calibration_manifest", "")
            if not calibration_path:
                raise RuntimeError("temperature loss manifest mode requires a calibration manifest")
            calibration = load_calibration_manifest(calibration_path, temperature_tau)
        else:
            raise RuntimeError(f"unsupported temperature calibration source: {calibration_source}")
        temperature_loss_weights = {
            "lambda_temp": float(calibration["lambda_temp"]),
            "lambda_grad": (
                float(calibration["lambda_grad"])
                if temperature_loss_mode == "scalar_grad"
                else 0.0
            ),
        }
        temperature_protocol = {
            "schema": "uav-tgs-temperature-loss-runtime-v1",
            "status": "passed",
            "mode": temperature_loss_mode,
            "formula_version": TEMPERATURE_LOSS_FORMULA_VERSION,
            "tau": temperature_tau,
            "tau_rule": "median positive adjacent squared distance of normalized fixed LUT",
            "lambda_temp": temperature_loss_weights["lambda_temp"],
            "lambda_grad": temperature_loss_weights["lambda_grad"],
            "calibration_source": calibration_source,
            "calibration_manifest_path": calibration["manifest_path"],
            "calibration_manifest_sha256": calibration["manifest_sha256"],
            "calibration_scene_target": calibration.get("temperature_target"),
            "runtime_scene_target": temperature_target_store.metadata(),
            "image_loss_preserved": True,
            "manifold_loss": False,
            "geometry_opacity_frozen": True,
        }
        protocol_path = os.path.join(scene.model_path, "temperature_loss_protocol.json")
        os.makedirs(scene.model_path, exist_ok=True)
        temporary_path = protocol_path + f".tmp-{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(temperature_protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, protocol_path)
        print(
            "[INFO] TemperatureConsistencyLoss: "
            f"mode={temperature_loss_mode} tau={temperature_tau:.9g} "
            f"lambda_temp={temperature_loss_weights['lambda_temp']:.9g} "
            f"lambda_grad={temperature_loss_weights['lambda_grad']:.9g}"
        )

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = (
        opt.optimizer_type == "sparse_adam"
        and SPARSE_ADAM_AVAILABLE
        and not topology_frozen
    )
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_cameras = scene.getTrainCameras().copy()
    viewpoint_stack = train_cameras.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    fixed_camera_sequence = None
    fixed_camera_manifest = None
    fixed_camera_map = None
    continuation_checkpoint_sha256 = None
    continuation_train_list_sha256 = None
    ogs_cache_file_sha256 = None
    ogs_formula_sha256 = None
    ogs_binding_verification = None
    ogs_lambda_calibration = None
    ogs_camera_parameters_sha256 = None
    if rgb_continuation:
        camera_names = ordered_camera_names(train_cameras)
        fixed_camera_manifest = load_sequence_manifest(
            getattr(args, "fixed_camera_sequence"),
            camera_names=camera_names,
            expected_steps=5000,
        )
        fixed_camera_sequence = fixed_camera_manifest["sequence"]
        fixed_camera_map = camera_lookup(train_cameras)
        sequence_metadata = fixed_camera_manifest.get("metadata", {})
        if not isinstance(sequence_metadata, dict):
            raise RuntimeError("Fixed camera sequence metadata is malformed")
        expected_checkpoint_sha = str(
            sequence_metadata.get("anchor_sha256", "")
        ).strip().lower()
        expected_train_list_sha = str(
            sequence_metadata.get("split_sha256", "")
        ).strip().lower()
        if not expected_checkpoint_sha or not expected_train_list_sha:
            raise RuntimeError(
                "Fixed camera sequence must pin anchor_sha256 and split_sha256"
            )
        continuation_checkpoint_sha256 = _file_sha256(checkpoint)
        if continuation_checkpoint_sha256 != expected_checkpoint_sha:
            raise RuntimeError(
                "Fixed camera sequence checkpoint SHA-256 mismatch: "
                f"expected={expected_checkpoint_sha} "
                f"actual={continuation_checkpoint_sha256}"
            )
        train_list_path = str(getattr(dataset, "train_list", "") or "")
        if not train_list_path or not os.path.isfile(train_list_path):
            raise RuntimeError(
                "Fixed-topology RGB continuation requires the formal train list file"
            )
        continuation_train_list_sha256 = _file_sha256(train_list_path)
        if continuation_train_list_sha256 != expected_train_list_sha:
            raise RuntimeError(
                "Fixed camera sequence train-list SHA-256 mismatch: "
                f"expected={expected_train_list_sha} "
                f"actual={continuation_train_list_sha256}"
            )
        declared_train_list_sha = str(
            getattr(dataset, "train_list_sha256", "") or ""
        ).strip().lower()
        if (
            declared_train_list_sha
            and declared_train_list_sha != continuation_train_list_sha256
        ):
            raise RuntimeError(
                "ModelParams train_list_sha256 does not match the actual train list"
            )
        print(
            "[INFO] FixedCameraSequence: "
            f"steps={len(fixed_camera_sequence)} "
            f"camera_hash={fixed_camera_manifest['ordered_camera_sha256']} "
            f"sequence_sha256={fixed_camera_manifest['sequence_sha256']}"
        )

    ogs_cache = None
    if ogs_enabled:
        from utils.ogs_v1 import (
            OGS_V1_FORMULA_SHA256,
            OGS_V1_INITIAL_LAMBDA_MODE,
            OGS_V1_LOSS_EPS,
            OGS_V1_MIN_ACTIVATED_OPACITY,
            OGS_V1_MIN_VISIBLE_VIEWS,
            OGS_V1_RECALIBRATED_LAMBDA_MODE,
            OGS_V1_RHO,
            OGS_V1_VARIANCE_EPS,
            compute_ogs_cache_hash,
            load_ogs_cache,
            ogs_v1_recalibration_manifest,
            verify_ogs_v1_formula_hash,
        )

        lambda_protocol = str(
            getattr(args, "ogs_v1_lambda_protocol", OGS_V1_INITIAL_LAMBDA_MODE)
        )
        if lambda_protocol not in (
            OGS_V1_INITIAL_LAMBDA_MODE,
            OGS_V1_RECALIBRATED_LAMBDA_MODE,
        ):
            raise RuntimeError(f"Unsupported OGS-v1 lambda protocol: {lambda_protocol}")

        expected_formula_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_formula_sha256", ""),
                "expected OGS formula SHA-256",
            )
            if getattr(args, "expected_ogs_formula_sha256", "")
            else OGS_V1_FORMULA_SHA256
        )
        ogs_formula_sha256 = verify_ogs_v1_formula_hash(expected_formula_sha)

        expected_anchor_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_anchor_sha256", ""),
                "expected OGS anchor SHA-256",
            )
            if getattr(args, "expected_ogs_anchor_sha256", "")
            else continuation_checkpoint_sha256
        )
        expected_camera_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_camera_sha256", ""),
                "expected OGS ordered-camera SHA-256",
            )
            if getattr(args, "expected_ogs_camera_sha256", "")
            else fixed_camera_manifest["ordered_camera_sha256"]
        )
        ogs_camera_parameters_sha256 = camera_parameters_hash(train_cameras)
        expected_camera_parameters_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_camera_parameters_sha256", ""),
                "expected OGS camera-parameters SHA-256",
            )
            if getattr(args, "expected_ogs_camera_parameters_sha256", "")
            else ogs_camera_parameters_sha256
        )
        expected_sequence_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_sequence_sha256", ""),
                "expected OGS fixed-sequence SHA-256",
            )
            if getattr(args, "expected_ogs_sequence_sha256", "")
            else fixed_camera_manifest["sequence_sha256"]
        )
        expected_sequence_manifest_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_sequence_manifest_sha256", ""),
                "expected OGS fixed-sequence manifest SHA-256",
            )
            if getattr(args, "expected_ogs_sequence_manifest_sha256", "")
            else fixed_camera_manifest["manifest_sha256"]
        )
        anchor_binding = _verified_sha256_binding(
            "OGS-v1 anchor",
            expected_anchor_sha,
            continuation_checkpoint_sha256,
        )
        ordered_camera_binding = _verified_sha256_binding(
            "OGS-v1 ordered-camera",
            expected_camera_sha,
            fixed_camera_manifest["ordered_camera_sha256"],
        )
        camera_parameters_binding = _verified_sha256_binding(
            "OGS-v1 camera-parameters",
            expected_camera_parameters_sha,
            ogs_camera_parameters_sha256,
        )
        sequence_binding = _verified_sha256_binding(
            "OGS-v1 fixed-sequence",
            expected_sequence_sha,
            fixed_camera_manifest["sequence_sha256"],
        )
        sequence_manifest_binding = _verified_sha256_binding(
            "OGS-v1 fixed-sequence manifest",
            expected_sequence_manifest_sha,
            fixed_camera_manifest["manifest_sha256"],
        )

        cache_path = str(getattr(args, "ogs_cache"))
        ogs_cache_file_sha256 = _file_sha256(cache_path)
        expected_cache_file_sha = (
            _normalized_sha256(
                getattr(args, "expected_ogs_cache_sha256", ""),
                "expected OGS cache file SHA-256",
            )
            if getattr(args, "expected_ogs_cache_sha256", "")
            else ogs_cache_file_sha256
        )
        if ogs_cache_file_sha256 != expected_cache_file_sha:
            raise RuntimeError(
                "OGS-v1 cache file SHA-256 mismatch: "
                f"expected={expected_cache_file_sha} "
                f"actual={ogs_cache_file_sha256}"
            )

        expected_cache_metadata = {
            "scene_name": sequence_metadata.get("scene"),
            "checkpoint_sha256": sequence_metadata.get("anchor_sha256"),
            "train_list_sha256": sequence_metadata.get("split_sha256"),
            "ordered_train_camera_names_sha256": fixed_camera_manifest[
                "ordered_camera_sha256"
            ],
            "camera_parameters_sha256": expected_camera_parameters_sha,
        }
        missing_expected = [
            key for key, value in expected_cache_metadata.items() if not value
        ]
        if missing_expected:
            raise RuntimeError(
                "OGS fixed sequence lacks anchor identity metadata: "
                + ", ".join(missing_expected)
            )
        ogs_cache = load_ogs_cache(
            cache_path,
            device=gaussians.get_xyz.device,
            expected_gaussian_count=int(gaussians.get_xyz.shape[0]),
            expected_metadata=expected_cache_metadata,
        )
        cache_semantic_declared_sha = ogs_cache["cache_sha256"]
        cache_semantic_actual_sha = compute_ogs_cache_hash(ogs_cache)
        if cache_semantic_actual_sha != cache_semantic_declared_sha:
            raise RuntimeError(
                "OGS-v1 cache semantic SHA-256 mismatch after device load: "
                f"declared={cache_semantic_declared_sha} "
                f"actual={cache_semantic_actual_sha}"
            )
        current_raw_hashes = {
            "xyz": _ogs_anchor_tensor_sha256(gaussians._xyz),
            "scaling": _ogs_anchor_tensor_sha256(gaussians._scaling),
            "rotation": _ogs_anchor_tensor_sha256(gaussians._rotation),
            "opacity": _ogs_anchor_tensor_sha256(gaussians._opacity),
        }
        cached_raw_hashes = ogs_cache.get("metadata", {}).get(
            "raw_tensor_sha256"
        )
        if cached_raw_hashes != current_raw_hashes:
            raise RuntimeError(
                "OGS cache Gaussian index/tensor identity does not match the "
                f"restored anchor: expected={cached_raw_hashes} "
                f"actual={current_raw_hashes}"
            )
        if int(ogs_cache["eligible_mask"].sum().item()) <= 0:
            raise RuntimeError("OGS-v1 cache has zero eligible Gaussians")
        cache_metadata = ogs_cache.get("metadata", {})
        formula_metadata_checks = {
            "rho": cache_metadata.get("rho") == OGS_V1_RHO,
            "eps": cache_metadata.get("eps") == OGS_V1_LOSS_EPS,
            "variance_eps": (
                cache_metadata.get("variance_eps") == OGS_V1_VARIANCE_EPS
            ),
            "renderer_visibility": (
                cache_metadata.get("renderer_visibility") == "radii>0"
            ),
            "minimum_visible_views": (
                cache_metadata.get("eligibility", {}).get(
                    "minimum_visible_views"
                )
                == OGS_V1_MIN_VISIBLE_VIEWS
            ),
            "minimum_activated_opacity_strict_gt": (
                cache_metadata.get("eligibility", {}).get(
                    "minimum_activated_opacity_strict_gt"
                )
                == OGS_V1_MIN_ACTIVATED_OPACITY
            ),
        }
        if not all(formula_metadata_checks.values()):
            failed = [
                key for key, passed in formula_metadata_checks.items() if not passed
            ]
            raise RuntimeError(
                "OGS-v1 cache metadata is incompatible with the pinned formula: "
                + ", ".join(failed)
            )
        cache_formula_sha = cache_metadata.get("ogs_v1_formula_sha256")
        if cache_formula_sha is not None and cache_formula_sha != ogs_formula_sha256:
            raise RuntimeError(
                "OGS-v1 cache formula SHA-256 mismatch: "
                f"cache={cache_formula_sha} current={ogs_formula_sha256}"
            )
        if lambda_protocol == OGS_V1_RECALIBRATED_LAMBDA_MODE:
            ogs_lambda_calibration = ogs_v1_recalibration_manifest()

        ogs_binding_verification = {
            "status": "passed",
            "anchor_checkpoint": anchor_binding,
            "ordered_camera_set": ordered_camera_binding,
            "camera_parameters": {
                **camera_parameters_binding,
                "cache_sha256": cache_metadata[
                    "camera_parameters_sha256"
                ],
                "cache_verified": (
                    cache_metadata["camera_parameters_sha256"]
                    == camera_parameters_binding["actual_sha256"]
                ),
            },
            "fixed_5000_camera_sequence": {
                "sequence": sequence_binding,
                "manifest": sequence_manifest_binding,
                "internal_manifest_integrity_verified": True,
            },
            "cache_file": {
                "expected_sha256": expected_cache_file_sha,
                "actual_sha256": ogs_cache_file_sha256,
                "verified": True,
            },
            "cache_semantic": {
                "expected_sha256": cache_semantic_declared_sha,
                "actual_sha256": cache_semantic_actual_sha,
                "verified": True,
            },
            "formula": {
                "expected_sha256": expected_formula_sha,
                "actual_sha256": ogs_formula_sha256,
                "repository_pin_sha256": OGS_V1_FORMULA_SHA256,
                "cache_declared_sha256": cache_formula_sha,
                "cache_metadata_contract_verified": formula_metadata_checks,
                "verified": True,
            },
        }
        print(
            "[INFO] OGSV1Cache: "
            f"eligible={int(ogs_cache['eligible_mask'].sum().item())} "
            f"cache_sha256={ogs_cache.get('cache_sha256')}"
        )

    if rgb_continuation:
        protocol = {
            "schema": "uav-tgs-rgb-continuation-protocol-v1",
            "recipe": rgb_continuation_recipe,
            "start_checkpoint": str(checkpoint),
            "start_checkpoint_sha256": continuation_checkpoint_sha256,
            "anchor_iteration": int(start_iteration),
            "scheduler_horizon": int(
                getattr(args, "rgb_continuation_scheduler_horizon", 30000)
            ),
            "scheduler_parameters": {
                "position_lr_init": float(opt.position_lr_init),
                "position_lr_final": float(opt.position_lr_final),
                "position_lr_delay_mult": float(opt.position_lr_delay_mult),
                "position_lr_max_steps": int(opt.position_lr_max_steps),
                "exposure_lr_init": float(opt.exposure_lr_init),
                "exposure_lr_final": float(opt.exposure_lr_final),
                "exposure_lr_delay_steps": int(opt.exposure_lr_delay_steps),
                "exposure_lr_delay_mult": float(opt.exposure_lr_delay_mult),
            },
            "requested_updates": int(
                getattr(args, "rgb_continuation_updates", 5000)
            ),
            "final_iteration": int(opt.iterations),
            "optimizer_step_at_final_iteration": bool(
                getattr(args, "optimizer_step_at_final_iteration", False)
            ),
            "optimizer_restore": rgb_optimizer_summary,
            "optimizer_initialization": (
                "fresh" if rgb_appearance_refit else "restored_anchor_state"
            ),
            "trainable_fields": (
                ["f_dc", "f_rest"]
                if rgb_appearance_refit
                else list(_RGB_CONTINUATION_GROUPS)
            ),
            "topology_fixed": True,
            "densification": False,
            "pruning": False,
            "opacity_reset": False,
            "artifact_save_semantics": "aligned",
            "camera_sequence_path": str(getattr(args, "fixed_camera_sequence")),
            "ordered_camera_sha256": fixed_camera_manifest["ordered_camera_sha256"],
            "camera_sequence_sha256": fixed_camera_manifest["sequence_sha256"],
            "train_list_sha256": continuation_train_list_sha256,
            "rng_seeds": {
                "python": 0,
                "numpy": 0,
                "torch": 0,
                "cuda": 0,
                "camera_sequence": int(fixed_camera_manifest["seed"]),
            },
            "random_background": bool(opt.random_background),
            "rgb_objective": {
                "lambda_dssim": float(opt.lambda_dssim),
                "depth_l1_weight_init": float(opt.depth_l1_weight_init),
                "depth_l1_weight_final": float(opt.depth_l1_weight_final),
                "thermal_structure_gradient_weight": float(
                    getattr(args, "t_struct_grad_w", 0.0)
                ),
            },
            "cuda_determinism_note": (
                "Fixed seeds and camera inputs do not guarantee bitwise "
                "determinism for all CUDA rasterizer kernels."
            ),
            "ogs_v1": ogs_enabled,
            "lambda_ogs": float(getattr(args, "lambda_ogs", 1e-3)),
            "ogs_rho": float(getattr(args, "ogs_rho", 3.0)),
            "ogs_cache_path": str(getattr(args, "ogs_cache", "")) if ogs_enabled else None,
            "ogs_cache_sha256": (
                ogs_cache.get("cache_sha256") if ogs_enabled else None
            ),
            "ogs_cache_file_sha256": ogs_cache_file_sha256,
            "ogs_v1_lambda_protocol": str(
                getattr(args, "ogs_v1_lambda_protocol", "initial_1e-3")
            ),
            "ogs_lambda_calibration": ogs_lambda_calibration,
            "ogs_v1_formula_sha256": ogs_formula_sha256,
            "ogs_v1_verified_bindings": ogs_binding_verification,
        }
        protocol_path = os.path.join(scene.model_path, "rgb_continuation_protocol.json")
        temporary_path = protocol_path + f".tmp-{os.getpid()}"
        with open(temporary_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(protocol, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, protocol_path)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    ss_logged_densify_trigger = False
    clamp_logged_rgb = False
    ss_gate_densify = bool(getattr(args, "ss_gate_densify", False))
    if getattr(args, "ss_enable", False) and (not ss_gate_densify):
        print("[INFO] SparseSupport densify gating disabled (prune-only mode).")
    if getattr(args, "ss_enable", False):
        densify_possible = False
        try:
            interval = int(opt.densification_interval)
        except Exception:
            interval = 0
        if interval > 0:
            start_iter = first_iter
            end_iter = min(int(opt.iterations), int(opt.densify_until_iter) - 1)
            start_iter = max(start_iter, int(opt.densify_from_iter) + 1)
            if start_iter <= end_iter:
                k = ((start_iter + interval - 1) // interval) * interval
                if k <= end_iter:
                    densify_possible = True
        if not densify_possible:
            print("[WARN] ss_enable=True but densify is disabled in this stage; SS gating logs won't appear.")
    struct_warned = False
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree(max_degree=thermal_max_sh_degree)

        # Paired RGB continuations consume a pre-generated sequence and never
        # call the process-global Python RNG during camera selection.
        if fixed_camera_sequence is not None:
            sequence_index = int(iteration) - int(start_iteration) - 1
            if sequence_index < 0 or sequence_index >= len(fixed_camera_sequence):
                raise RuntimeError(
                    "Fixed camera sequence exhausted at continuation update "
                    f"{sequence_index + 1}"
                )
            viewpoint_cam = fixed_camera_map[fixed_camera_sequence[sequence_index]]
        else:
            # Legacy random-pop-without-replacement path remains byte-for-byte
            # equivalent in behavior when RGB continuation is not requested.
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        current_alpha_mask = None
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            current_alpha_mask = alpha_mask
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        # Optional: pseudo-color thermal structure gradient loss (disabled by default)
        if getattr(args, "t_struct_grad_w", 0.0) > 0.0:
            if structure_grad_loss is None:
                if not struct_warned:
                    print("[WARN] t_struct_grad_w>0 but structure_grad_loss is unavailable. Skipping structure loss.")
                    struct_warned = True
            else:
                # Guard: require 3-channel inputs (pseudo-color thermal or RGB)
                ch_img = int(image.shape[0]) if image.dim() == 3 else (int(image.shape[1]) if image.dim() == 4 else -1)
                ch_gt = int(gt_image.shape[0]) if gt_image.dim() == 3 else (int(gt_image.shape[1]) if gt_image.dim() == 4 else -1)
                if ch_img != 3 or ch_gt != 3:
                    if not struct_warned:
                        print("[WARN] t_struct_grad_w>0 but image/gt_image is not 3-channel. Skipping structure loss.")
                        struct_warned = True
                else:
                    try:
                        _mask = alpha_mask if "alpha_mask" in locals() else None
                        loss_struct = structure_grad_loss(
                            image, gt_image, mask=_mask,
                            normalize=getattr(args, "t_struct_grad_norm", True),
                        )
                        loss = loss + float(getattr(args, "t_struct_grad_w", 0.0)) * loss_struct
                    except Exception as _e:
                        if not struct_warned:
                            print(f"[WARN] structure_grad_loss failed once; skipping. err={_e}")
                            struct_warned = True

        temperature_result = None
        if temperature_target_store is not None:
            target_u, temperature_support = temperature_target_store.get(
                viewpoint_cam.image_name,
                int(image.shape[-2]),
                int(image.shape[-1]),
                image.device,
            )
            if current_alpha_mask is not None:
                temperature_support = temperature_support * current_alpha_mask.to(
                    device=temperature_support.device,
                    dtype=temperature_support.dtype,
                )
            temperature_result = temperature_consistency_losses(
                image,
                target_u,
                mask=temperature_support,
                tau=temperature_tau,
                chunk_pixels=int(getattr(args, "temperature_lut_chunk_pixels", 16384)),
            )
            if not (
                torch.isfinite(temperature_result["scalar"]).item()
                and torch.isfinite(temperature_result["gradient"]).item()
            ):
                raise RuntimeError(
                    f"non-finite temperature loss at iteration {iteration}"
                )
            loss = (
                loss
                + temperature_loss_weights["lambda_temp"]
                * temperature_result["scalar"]
                + temperature_loss_weights["lambda_grad"]
                * temperature_result["gradient"]
            )
            if iteration == first_iter or iteration % 1000 == 0:
                temperature_log_path = os.path.join(
                    scene.model_path, "temperature_loss_train.jsonl"
                )
                _append_jsonl(
                    temperature_log_path,
                    {
                        "schema": "uav-tgs-temperature-loss-training-v1",
                        "iteration": int(iteration),
                        "image_name": str(viewpoint_cam.image_name),
                        "mode": temperature_loss_mode,
                        "scalar_unweighted": float(
                            temperature_result["scalar"].detach().item()
                        ),
                        "gradient_unweighted": float(
                            temperature_result["gradient"].detach().item()
                        ),
                        "scalar_weighted": float(
                            (
                                temperature_loss_weights["lambda_temp"]
                                * temperature_result["scalar"]
                            ).detach().item()
                        ),
                        "gradient_weighted": float(
                            (
                                temperature_loss_weights["lambda_grad"]
                                * temperature_result["gradient"]
                            ).detach().item()
                        ),
                        "valid_fraction": float(
                            temperature_result["valid_fraction"]
                        ),
                        "finite": bool(
                            torch.isfinite(temperature_result["scalar"]).item()
                            and torch.isfinite(temperature_result["gradient"]).item()
                        ),
                    },
                )

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        rgb_objective = loss
        ogs_result = None
        if ogs_enabled:
            from utils.ogs_v1 import ogs_v1_loss

            ogs_result = ogs_v1_loss(
                scales=gaussians.get_scaling,
                rotations=gaussians._rotation,
                cache=ogs_cache,
                rho=float(getattr(args, "ogs_rho", 3.0)),
                eps=1e-8,
            )
            ogs_loss = ogs_result["loss"]
            lambda_ogs = float(getattr(args, "lambda_ogs", 1e-3))

            if iteration == first_iter:
                forbidden_parameters = (
                    gaussians._xyz,
                    gaussians._features_dc,
                    gaussians._features_rest,
                    gaussians._opacity,
                    gaussians._exposure,
                )
                forbidden_gradients = torch.autograd.grad(
                    ogs_loss,
                    forbidden_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                leaked = []
                for name, gradient in zip(
                    ("xyz", "f_dc", "f_rest", "opacity", "exposure"),
                    forbidden_gradients,
                ):
                    if gradient is not None and torch.count_nonzero(gradient).item() != 0:
                        leaked.append(name)
                if leaked:
                    raise RuntimeError(
                        "OGS-v1 gradient leaked outside scaling/rotation: "
                        + ", ".join(leaked)
                    )

            smoke_step = int(iteration) - int(start_iteration)
            smoke_limit = int(getattr(args, "ogs_gradient_smoke_steps", 200))
            if smoke_step <= smoke_limit:
                rgb_grads = torch.autograd.grad(
                    rgb_objective,
                    (gaussians._scaling, gaussians._rotation),
                    retain_graph=True,
                    allow_unused=True,
                )
                ogs_grads = torch.autograd.grad(
                    ogs_loss,
                    (gaussians._scaling, gaussians._rotation),
                    retain_graph=True,
                    allow_unused=True,
                )
                lr_map = {
                    group.get("name"): float(group.get("lr", 0.0))
                    for group in gaussians.optimizer.param_groups
                }
                probe = _lr_scaled_gradient_probe(
                    rgb_grads,
                    ogs_grads,
                    lr_map["scaling"],
                    lr_map["rotation"],
                    lambda_ogs,
                )
                eligible_q = ogs_result["q_current"][
                    ogs_cache["eligible_mask"]
                ].detach()
                finite_values = [
                    rgb_objective.detach(),
                    ogs_loss.detach(),
                    eligible_q,
                    *(gradient.detach() for gradient in rgb_grads if gradient is not None),
                    *(gradient.detach() for gradient in ogs_grads if gradient is not None),
                ]
                payload = {
                    "schema": "uav-tgs-ogs-v1-gradient-smoke-v1",
                    "iteration": int(iteration),
                    "continuation_update": smoke_step,
                    "camera_image_name": str(viewpoint_cam.image_name),
                    "l_rgb": float(rgb_objective.detach().item()),
                    "l_ogs": float(ogs_loss.detach().item()),
                    "weighted_l_ogs": float((lambda_ogs * ogs_loss).detach().item()),
                    "penalty_mean": float(
                        torch.as_tensor(ogs_result["penalty_mean"]).detach().item()
                    ),
                    "eligible_count": int(ogs_result["eligible_count"]),
                    "active_count": int(ogs_result["active_count"]),
                    "q_current_eligible_mean": float(eligible_q.mean().item()),
                    "q_current_eligible_max": float(eligible_q.max().item()),
                    "gradient_probe": probe,
                    "finite": bool(
                        all(torch.isfinite(value).all().item() for value in finite_values)
                    ),
                }
                smoke_path = (
                    getattr(args, "ogs_gradient_log", "")
                    or os.path.join(scene.model_path, "ogs_gradient_smoke.jsonl")
                )
                _append_jsonl(smoke_path, payload)
                if not payload["finite"]:
                    raise RuntimeError(
                        f"Non-finite OGS-v1 smoke value at iteration {iteration}"
                    )

            loss = rgb_objective + lambda_ogs * ogs_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # Legacy compatibility: PLY is written before the optimizer step.
            if (not aligned_artifact_saves) and (iteration in saving_iterations):
                if thermal_max_sh_degree is not None:
                    gaussians.zero_sh_above_degree_(thermal_max_sh_degree)
                if debug_stats and iteration == opt.iterations:
                    _log_gaussian_stats("before_save")
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (not topology_frozen) and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if getattr(args, "ss_enable", False) and not ss_logged_densify_trigger:
                        print(f"[INFO] densify triggered at iter={iteration} (ss_enable=True)")
                        ss_logged_densify_trigger = True
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    ss_enabled_backup = None
                    if getattr(args, "ss_enable", False) and (not ss_gate_densify) and getattr(gaussians, "_ss_enabled", False):
                        ss_enabled_backup = gaussians._ss_enabled
                        gaussians._ss_enabled = False
                    try:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    finally:
                        if ss_enabled_backup is not None:
                            gaussians._ss_enabled = ss_enabled_backup
                    if args.clamp_scale_after_densify and args.clamp_scale_max is not None:
                        clamped_gauss, total, before_smax, after_smax = gaussians.clamp_scaling_max_(args.clamp_scale_max)
                        if clamped_gauss > 0 and not clamp_logged_rgb:
                            print(
                                f"[INFO] ClampScaling rgb_after_densify: max_scale={args.clamp_scale_max} "
                                f"clamped_gauss={clamped_gauss}/{total} before_smax={before_smax:.6f} after_smax={after_smax:.6f}"
                            )
                            clamp_logged_rgb = True
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if _should_optimizer_step(
                iteration,
                opt.iterations,
                getattr(args, "optimizer_step_at_final_iteration", False),
            ):
                if gaussians.exposure_optimizer is not None:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if aligned_artifact_saves:
                # AAAI recipe: both artifacts are written after the same optimizer
                # step, so equal iteration labels identify one model/Adam state.
                save_gaussians = iteration in saving_iterations
                save_checkpoint = iteration in checkpoint_iterations
                if debug_stats and save_gaussians and iteration == opt.iterations:
                    _log_gaussian_stats("before_save")
                _save_iteration_artifacts(
                    scene,
                    gaussians,
                    iteration,
                    save_gaussians=save_gaussians,
                    save_checkpoint=save_checkpoint,
                    thermal_max_sh_degree=thermal_max_sh_degree,
                )
            elif iteration in checkpoint_iterations:
                # Preserve the historical post-step checkpoint behavior.
                if thermal_max_sh_degree is not None:
                    gaussians.zero_sh_above_degree_(thermal_max_sh_degree)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    # Optional: one-shot final clamp at the end of RGB stage.
    # This runs after normal training saves and overwrites final artifacts.
    if getattr(args, "clamp_scale_after_rgb_final", False) and not checkpoint:
        if args.clamp_scale_max is None:
            print("[WARN] clamp_scale_after_rgb_final set but clamp_scale_max is None; skipping final clamp.")
        else:
            clamped_gauss, total, before_smax, after_smax = gaussians.clamp_scaling_max_(args.clamp_scale_max)
            if clamped_gauss > 0:
                print(
                    f"[INFO] ClampScaling rgb_after_train: max_scale={args.clamp_scale_max} "
                    f"clamped_gauss={clamped_gauss}/{total} before_smax={before_smax:.6f} after_smax={after_smax:.6f}"
                )
            scene.save(opt.iterations)
            torch.save((gaussians.capture(), opt.iterations), scene.model_path + "/chkpnt" + str(opt.iterations) + ".pth")

    # Optional: one-shot sparse-support prune at the end of RGB stage.
    # This runs after normal training saves and overwrites final artifacts
    # with the pruned model so downstream stage-2 can reuse the cleaned checkpoint.
    if getattr(args, "ss_prune_after_rgb", False) and not checkpoint:
        if not getattr(args, "ss_enable", False):
            print("[WARN] ss_prune_after_rgb set but ss_enable=False; skipping prune.")
        elif not (getattr(gaussians, "_ss_enabled", False) and gaussians._ss_is_enabled()):
            print("[WARN] ss_prune_after_rgb set but SparseSupport not configured; skipping prune.")
        else:
            prune_stats = gaussians.prune_outside_sparse_support()
            if prune_stats is not None:
                before, after = prune_stats
                removed = before - after
                keep_ratio = (after / float(before)) if before > 0 else 1.0
                print(
                    f"[INFO] SparseSupport prune_after_rgb: before={before} after={after} "
                    f"removed={removed} keep_ratio={keep_ratio:.6f}"
                )
                scene.save(opt.iterations)
                torch.save((gaussians.capture(), opt.iterations), scene.model_path + "/chkpnt" + str(opt.iterations) + ".pth")

    if strict_freeze:
        _write_strict_freeze_audit(
            scene.model_path,
            gaussians,
            strict_freeze_snapshot,
            start_iteration=start_iteration,
            final_iteration=int(opt.iterations),
        )
    elif rgb_appearance_refit:
        _write_strict_freeze_audit(
            scene.model_path,
            gaussians,
            rgb_appearance_freeze_snapshot,
            start_iteration=start_iteration,
            final_iteration=int(opt.iterations),
            schema="uav-tgs-rgb-appearance-refit-freeze-audit-v1",
            filename="rgb_appearance_refit_freeze_audit.json",
            label="RGBAppearanceRefitFreezeAudit",
        )
    elif opacity_adaptive:
        _write_opacity_adaptive_freeze_audit(
            scene.model_path,
            gaussians,
            opacity_adaptive_freeze_snapshot,
            start_iteration=start_iteration,
            final_iteration=int(opt.iterations),
        )

    optimizer_updates_executed = max(
        0,
        int(opt.iterations)
        - int(start_iteration)
        - (0 if getattr(args, "optimizer_step_at_final_iteration", False) else 1),
    )
    if rgb_continuation:
        final_steps = {}
        for group in gaussians.optimizer.param_groups:
            state = gaussians.optimizer.state.get(group["params"][0])
            if not isinstance(state, dict) or "step" not in state:
                raise RuntimeError(
                    f"RGB continuation final Adam state missing for {group.get('name')}"
                )
            final_steps[group.get("name")] = _optimizer_state_step(state["step"])
        expected_groups = (
            _RGB_APPEARANCE_GROUPS
            if rgb_appearance_refit
            else _RGB_CONTINUATION_GROUPS
        )
        if tuple(final_steps) != expected_groups:
            raise RuntimeError(
                "RGB continuation final optimizer groups mismatch: "
                f"expected={expected_groups} actual={tuple(final_steps)}"
            )
        step_deltas = {
            name: final_steps[name] - rgb_optimizer_summary["adam_steps"][name]
            for name in expected_groups
        }
        expected_updates = int(getattr(args, "rgb_continuation_updates", 5000))
        if set(step_deltas.values()) != {expected_updates}:
            raise RuntimeError(
                "RGB continuation did not execute the exact requested optimizer "
                f"updates: expected={expected_updates} actual={step_deltas}"
            )
        if optimizer_updates_executed != expected_updates:
            raise RuntimeError(
                "RGB continuation loop/update count mismatch: "
                f"loop={optimizer_updates_executed} expected={expected_updates}"
            )
        print(
            "[INFO] RGBContinuationUpdates: "
            f"exact_updates={expected_updates} final_adam_steps={final_steps}"
        )

    return {
        "start_iteration": start_iteration,
        "final_iteration": int(opt.iterations),
        "iterations_executed": max(0, int(opt.iterations) - start_iteration),
        "optimizer_updates_executed": optimizer_updates_executed,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "model_path": str(scene.model_path),
    }

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    def _str2bool(v):
        """Parse a boolean value from CLI/config strings (robust across Python versions)."""
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        raise ValueError(f"invalid bool: {v}")

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--benchmark_efficiency", action="store_true", default=False,
                        help="Write boundary-only training time and peak PyTorch CUDA memory (default: off).")
    parser.add_argument("--efficiency_output", type=str, default="",
                        help="Training efficiency JSON path (default: <model_path>/train_efficiency.json).")
    parser.add_argument("--efficiency_stage", type=str, default="auto",
                        help="Stage label stored in efficiency JSON (default: auto -> rgb/thermal).")

    # Sparse Support (disabled by default): gates densification using sparse COLMAP support / init point cloud.
    parser.add_argument("--ss_enable", action="store_true", default=False)
    parser.add_argument("--ss_source", type=str, choices=["colmap_sparse", "init_pcd"], default="colmap_sparse")
    parser.add_argument("--ss_use_aabb", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ss_aabb_margin", type=float, default=0.0)
    parser.add_argument("--ss_voxel_size", type=float, default=None)
    parser.add_argument("--ss_nn_dist_thr", type=float, default=None)
    parser.add_argument("--ss_adaptive_nn", action="store_true", default=False)
    parser.add_argument("--ss_adaptive_alpha", type=float, default=1.0)
    parser.add_argument("--ss_adaptive_beta", type=float, default=0.0)
    parser.add_argument("--ss_adaptive_max_scale", type=float, default=1.5)
    parser.add_argument("--ss_trim_tail_pct", type=float, default=0.0)
    parser.add_argument("--ss_drop_small_islands", type=int, default=0)
    parser.add_argument("--ss_island_radius", type=float, default=None)
    parser.add_argument("--ss_gate_densify", action="store_true", default=False)
    parser.add_argument("--ss_prune_before_thermal", action="store_true", default=False)
    parser.add_argument("--ss_prune_after_rgb", action="store_true", default=True)
    parser.add_argument("--debug_gaussian_stats", action="store_true", default=False)
    parser.add_argument("--clamp_scale_max", type=float, default=None)
    parser.add_argument("--clamp_scale_after_densify", action="store_true", default=False)
    parser.add_argument("--clamp_scale_after_rgb_final", action="store_true", default=False)
    parser.add_argument("--thermal_reset_features", action="store_true", default=False)
    parser.add_argument(
        "--thermal_max_sh_degree", type=int, choices=[0, 1, 3], default=None,
        help="Thermal-only SH cap. With --thermal_reset_features, restart active SH at degree 0."
    )
    parser.add_argument(
        "--thermal_optimizer_state", choices=["restore", "fresh"], default="restore",
        help="Thermal checkpoint optimizer state (default: restore). fresh starts a new Adam state."
    )
    parser.add_argument(
        "--thermal_recipe",
        choices=["legacy", "aaai_strict", "geometry_frozen_opacity_adaptive"],
        default="legacy",
        help="Stage-2 protocol preset. legacy preserves the ACM behavior by default."
    )
    parser.add_argument(
        "--artifact_save_semantics",
        choices=["legacy", "aligned"],
        default=None,
        help=(
            "PLY/checkpoint save ordering. Unset resolves to legacy, while "
            "formal Stage-2 recipes resolve to aligned post-step endpoints."
        ),
    )
    parser.add_argument(
        "--optimizer_step_at_final_iteration",
        action="store_true",
        default=False,
        help=(
            "Execute the optimizer update at the labeled final iteration. "
            "Only valid with aligned artifact saves or the fixed-topology RGB recipe."
        ),
    )
    parser.add_argument(
        "--rgb_continuation_recipe",
        choices=["legacy", "fixed_topology", "appearance_only"],
        default="legacy",
        help=(
            "RGB 30k-anchor continuation: fixed_topology restores the full "
            "RGB Adam; appearance_only creates a fresh SH-only optimizer."
        ),
    )
    parser.add_argument(
        "--rgb_optimizer_state",
        choices=["restore", "fresh"],
        default="restore",
        help="Optimizer initialization for an explicit RGB continuation recipe.",
    )
    parser.add_argument(
        "--rgb_continuation_anchor_iteration", type=int, default=30000
    )
    parser.add_argument(
        "--rgb_continuation_scheduler_horizon", type=int, default=30000
    )
    parser.add_argument(
        "--rgb_continuation_updates",
        type=int,
        choices=[200, 5000],
        default=5000,
        help="Use 200 for smoke or 5000 for the formal continuation.",
    )
    parser.add_argument(
        "--fixed_camera_sequence",
        type=str,
        default="",
        help="Pre-generated 5000-entry image_name sequence manifest.",
    )
    parser.add_argument("--ogs_v1", action="store_true", default=False)
    parser.add_argument("--ogs_cache", type=str, default="")
    parser.add_argument("--lambda_ogs", type=float, default=1e-3)
    parser.add_argument(
        "--ogs_v1_lambda_protocol",
        choices=[
            "initial_1e-3",
            "train_only_gradient_probe_recalibrated_1_1",
        ],
        default="initial_1e-3",
        help=(
            "Explicit OGS-v1 lambda provenance. The recalibrated mode pins "
            "lambda=1.1 and requires all expected hash bindings."
        ),
    )
    parser.add_argument("--expected_ogs_cache_sha256", type=str, default="")
    parser.add_argument("--expected_ogs_anchor_sha256", type=str, default="")
    parser.add_argument("--expected_ogs_camera_sha256", type=str, default="")
    parser.add_argument(
        "--expected_ogs_camera_parameters_sha256", type=str, default=""
    )
    parser.add_argument("--expected_ogs_sequence_sha256", type=str, default="")
    parser.add_argument(
        "--expected_ogs_sequence_manifest_sha256", type=str, default=""
    )
    parser.add_argument("--expected_ogs_formula_sha256", type=str, default="")
    parser.add_argument("--ogs_rho", type=float, default=3.0)
    parser.add_argument("--ogs_gradient_smoke_steps", type=int, default=200)
    parser.add_argument("--ogs_gradient_log", type=str, default="")
    parser.add_argument(
        "--thermal_freeze_mode",
        choices=[
            "legacy", "strict", "continuous_unfrozen",
            "geometry_frozen_opacity_adaptive",
        ],
        default="legacy",
        help=(
            "Stage-2 parameter update mode. strict trains SH appearance only; "
            "geometry_frozen_opacity_adaptive trains SH plus opacity."
        ),
    )
    parser.add_argument(
        "--thermal_scale_clamp", choices=["legacy", "off"], default="legacy",
        help="Control the one-shot thermal restore scale clamp.",
    )
    parser.add_argument("--sgf_disable", action="store_true", default=False)
    parser.add_argument("--baseline_modules_off", action="store_true", default=False)
    parser.add_argument("--baseline_restore_ssp", action="store_true", default=False)
    parser.add_argument("--baseline_restore_stt", action="store_true", default=False)

    # Improved-4 (optional): pseudo-color thermal structure-gradient loss
    parser.add_argument("--t_struct_grad_w", type=float, default=0.0)
    parser.add_argument("--t_struct_grad_norm", type=_str2bool, default=True)
    parser.add_argument(
        "--temperature_loss_mode",
        choices=["none", "scalar", "scalar_grad"],
        default="none",
        help="Optional normalized apparent-temperature auxiliary objective.",
    )
    parser.add_argument("--temperature_gt_root", type=str, default="")
    parser.add_argument("--temperature_support_root", type=str, default="")
    parser.add_argument("--temperature_range_manifest", type=str, default="")
    parser.add_argument(
        "--temperature_loss_calibration",
        choices=["calibrate", "manifest"],
        default="manifest",
    )
    parser.add_argument("--temperature_loss_calibration_manifest", type=str, default="")
    parser.add_argument("--temperature_calibration_views", type=int, default=8)
    parser.add_argument("--temperature_lut_chunk_pixels", type=int, default=16384)
    parser.add_argument("--temperature_target_cache_items", type=int, default=1024)

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    def _option_was_set(name):
        return any(token == name or token.startswith(name + "=") for token in cli_args)

    if args.thermal_max_sh_degree is not None and not args.start_checkpoint:
        parser.error("--thermal_max_sh_degree requires --start_checkpoint")
    if args.thermal_optimizer_state == "fresh" and not args.start_checkpoint:
        parser.error("--thermal_optimizer_state fresh requires --start_checkpoint")
    if args.temperature_loss_mode == "none":
        extra_temperature_options = [
            token.split("=", 1)[0]
            for token in cli_args
            if token.startswith("--temperature_")
            and not token.startswith("--temperature_loss_mode")
        ]
        if extra_temperature_options:
            parser.error("temperature options require --temperature_loss_mode scalar or scalar_grad")
    else:
        if args.thermal_recipe != "aaai_strict":
            parser.error("temperature loss requires --thermal_recipe aaai_strict")
        if not args.temperature_gt_root or not args.temperature_range_manifest:
            parser.error("temperature loss requires --temperature_gt_root and --temperature_range_manifest")
        if args.temperature_loss_calibration == "manifest" and not args.temperature_loss_calibration_manifest:
            parser.error("temperature manifest calibration requires --temperature_loss_calibration_manifest")
        if args.temperature_calibration_views <= 0:
            parser.error("--temperature_calibration_views must be positive")
        if args.temperature_lut_chunk_pixels <= 0 or args.temperature_target_cache_items <= 0:
            parser.error("temperature LUT chunk and target cache sizes must be positive")

    if (args.baseline_restore_ssp or args.baseline_restore_stt) and (not args.baseline_modules_off):
        parser.error("--baseline_restore_ssp/--baseline_restore_stt require --baseline_modules_off")

    if args.baseline_modules_off:
        args.ss_enable = False
        args.ss_prune_before_thermal = False
        args.ss_prune_after_rgb = False
        args.clamp_scale_max = None
        args.clamp_scale_after_densify = False
        args.clamp_scale_after_rgb_final = False
        args.thermal_reset_features = False
        args.t_struct_grad_w = 0.0
        args.t_struct_grad_norm = True
        args.sgf_disable = True

    if args.baseline_restore_ssp:
        # Selectively restore the stage-1 SSP package on top of the baseline transfer recipe.
        args.ss_enable = True
        args.ss_source = "colmap_sparse"
        args.ss_use_aabb = False
        args.ss_aabb_margin = 0.0
        args.ss_voxel_size = 1.5
        args.ss_nn_dist_thr = 3.5
        args.ss_adaptive_nn = True
        args.ss_adaptive_alpha = 1.2
        args.ss_adaptive_beta = 0.2
        args.ss_adaptive_max_scale = 2.0
        args.ss_trim_tail_pct = 0.0
        args.ss_drop_small_islands = 10
        args.ss_island_radius = 10.0
        args.ss_prune_after_rgb = True
        args.ss_prune_before_thermal = False

    if args.baseline_restore_stt:
        # Selectively restore the stage-2 STT package on top of the baseline transfer recipe.
        args.clamp_scale_max = 10.0
        args.thermal_reset_features = True
        args.t_struct_grad_w = 0.006
        args.t_struct_grad_norm = True
        args.sgf_disable = False

    try:
        args.artifact_save_semantics = _resolve_artifact_save_semantics(
            args.thermal_recipe, args.artifact_save_semantics
        )
    except ValueError as error:
        parser.error(str(error))

    rgb_continuation_requested = args.rgb_continuation_recipe in (
        "fixed_topology", "appearance_only"
    )
    if rgb_continuation_requested:
        if not args.start_checkpoint:
            parser.error(
                "--rgb_continuation_recipe requires --start_checkpoint"
            )
        forbidden_prefixes = (
            "--thermal_",
            "--temperature_",
            "--baseline_",
            "--sgf_disable",
            "--t_struct_grad_",
            "--clamp_scale_",
            "--ss_",
        )
        forbidden_options = sorted(
            token.split("=", 1)[0]
            for token in cli_args
            if token.startswith(forbidden_prefixes)
        )
        if forbidden_options:
            parser.error(
                "Fixed-topology RGB continuation rejects thermal/baseline/"
                "SSP/clamp flags: " + ", ".join(forbidden_options)
            )
        if args.thermal_recipe != "legacy":
            parser.error("RGB continuation cannot use a thermal recipe")
        expected_rgb_optimizer_state = (
            "fresh"
            if args.rgb_continuation_recipe == "appearance_only"
            else "restore"
        )
        if args.rgb_optimizer_state != expected_rgb_optimizer_state:
            parser.error(
                f"--rgb_continuation_recipe {args.rgb_continuation_recipe} "
                f"requires --rgb_optimizer_state {expected_rgb_optimizer_state}"
            )
        if args.thermal_freeze_mode != "legacy":
            parser.error("RGB continuation cannot use a thermal freeze mode")
        if args.thermal_reset_features or args.thermal_max_sh_degree is not None:
            parser.error("RGB continuation cannot reset or cap thermal SH")
        if args.baseline_modules_off or args.baseline_restore_ssp or args.baseline_restore_stt:
            parser.error("RGB continuation cannot use baseline recipe flags")
        if args.sgf_disable or args.ss_enable or args.ss_prune_before_thermal:
            parser.error("RGB continuation cannot use SGF/SSP stage flags")
        if (
            args.clamp_scale_max is not None
            or args.clamp_scale_after_densify
            or args.clamp_scale_after_rgb_final
        ):
            parser.error("RGB continuation cannot use scale clamps")
        if args.t_struct_grad_w != 0.0:
            parser.error("RGB continuation uses the ordinary RGB objective only")
        if args.optimizer_type != "default":
            parser.error("RGB continuation requires the standard dense Adam")
        if not args.fixed_camera_sequence:
            parser.error("RGB continuation requires --fixed_camera_sequence")
        try:
            expected_endpoint = _validate_rgb_continuation_schedule(
                args.rgb_continuation_anchor_iteration,
                args.rgb_continuation_scheduler_horizon,
                args.rgb_continuation_updates,
                args.iterations,
                args.optimizer_step_at_final_iteration,
            )
        except ValueError as error:
            parser.error(str(error))
        if args.position_lr_max_steps != args.rgb_continuation_scheduler_horizon:
            parser.error(
                "--position_lr_max_steps must equal the original 30000 scheduler horizon"
            )
        if expected_endpoint not in args.checkpoint_iterations:
            parser.error(
                "RGB continuation requires a checkpoint at the aligned final endpoint"
            )
        if (
            _option_was_set("--artifact_save_semantics")
            and args.artifact_save_semantics != "aligned"
        ):
            parser.error("RGB continuation requires aligned artifact saves")
        args.artifact_save_semantics = "aligned"
        # Fixed topology makes these no-ops, but setting them explicitly keeps
        # the runtime manifest unambiguous.
        args.ss_enable = False
        args.ss_prune_before_thermal = False
        args.ss_prune_after_rgb = False

        if args.ogs_v1 and args.rgb_continuation_recipe != "fixed_topology":
            parser.error("OGS-v1 requires --rgb_continuation_recipe fixed_topology")
        if args.ogs_v1:
            if not args.ogs_cache:
                parser.error("--ogs_v1 requires --ogs_cache")
            if args.ogs_v1_lambda_protocol == "initial_1e-3":
                if args.lambda_ogs != 1e-3:
                    parser.error(
                        "Initial OGS-v1 pilot pins --lambda_ogs to 1e-3"
                    )
            elif (
                args.ogs_v1_lambda_protocol
                == "train_only_gradient_probe_recalibrated_1_1"
            ):
                if args.lambda_ogs != 1.1:
                    parser.error(
                        "Recalibrated OGS-v1 pilot pins --lambda_ogs to 1.1"
                    )
                required_hashes = {
                    "--expected_ogs_cache_sha256": args.expected_ogs_cache_sha256,
                    "--expected_ogs_anchor_sha256": args.expected_ogs_anchor_sha256,
                    "--expected_ogs_camera_sha256": args.expected_ogs_camera_sha256,
                    "--expected_ogs_camera_parameters_sha256": (
                        args.expected_ogs_camera_parameters_sha256
                    ),
                    "--expected_ogs_sequence_sha256": (
                        args.expected_ogs_sequence_sha256
                    ),
                    "--expected_ogs_sequence_manifest_sha256": (
                        args.expected_ogs_sequence_manifest_sha256
                    ),
                    "--expected_ogs_formula_sha256": args.expected_ogs_formula_sha256,
                }
                missing_hashes = [
                    option
                    for option, value in required_hashes.items()
                    if not str(value).strip()
                ]
                if missing_hashes:
                    parser.error(
                        "Recalibrated OGS-v1 requires explicit hash bindings: "
                        + ", ".join(missing_hashes)
                    )
            if args.ogs_rho != 3.0:
                parser.error("OGS-v1 pilot pins --ogs_rho to 3")
            if args.ogs_gradient_smoke_steps != 200:
                parser.error("OGS-v1 pilot pins the gradient smoke to 200 updates")
        elif (
            _option_was_set("--ogs_cache")
            or _option_was_set("--lambda_ogs")
            or _option_was_set("--ogs_v1_lambda_protocol")
            or _option_was_set("--expected_ogs_cache_sha256")
            or _option_was_set("--expected_ogs_anchor_sha256")
            or _option_was_set("--expected_ogs_camera_sha256")
            or _option_was_set("--expected_ogs_camera_parameters_sha256")
            or _option_was_set("--expected_ogs_sequence_sha256")
            or _option_was_set("--expected_ogs_sequence_manifest_sha256")
            or _option_was_set("--expected_ogs_formula_sha256")
            or _option_was_set("--ogs_rho")
            or _option_was_set("--ogs_gradient_smoke_steps")
            or _option_was_set("--ogs_gradient_log")
        ):
            parser.error("OGS options require --ogs_v1")
    else:
        continuation_only = (
            args.ogs_v1
            or bool(args.fixed_camera_sequence)
            or _option_was_set("--rgb_optimizer_state")
            or _option_was_set("--rgb_continuation_anchor_iteration")
            or _option_was_set("--rgb_continuation_scheduler_horizon")
            or _option_was_set("--rgb_continuation_updates")
            or _option_was_set("--expected_ogs_cache_sha256")
            or _option_was_set("--expected_ogs_anchor_sha256")
            or _option_was_set("--expected_ogs_camera_sha256")
            or _option_was_set("--expected_ogs_camera_parameters_sha256")
            or _option_was_set("--expected_ogs_sequence_sha256")
            or _option_was_set("--expected_ogs_sequence_manifest_sha256")
            or _option_was_set("--expected_ogs_formula_sha256")
        )
        if continuation_only:
            parser.error(
                "Fixed camera/OGS continuation options require "
                "an explicit --rgb_continuation_recipe"
            )

    if (
        args.optimizer_step_at_final_iteration
        and args.artifact_save_semantics != "aligned"
    ):
        parser.error(
            "--optimizer_step_at_final_iteration requires aligned artifact saves"
        )

    if args.thermal_recipe == "aaai_strict":
        if not args.start_checkpoint:
            parser.error("--thermal_recipe aaai_strict requires --start_checkpoint")
        if args.baseline_modules_off or args.baseline_restore_ssp or args.baseline_restore_stt:
            parser.error("--thermal_recipe aaai_strict cannot be combined with baseline recipe flags")
        if _option_was_set("--sgf_disable"):
            parser.error("--thermal_recipe aaai_strict does not use --sgf_disable")
        if _option_was_set("--thermal_optimizer_state") and args.thermal_optimizer_state != "fresh":
            parser.error("--thermal_recipe aaai_strict requires --thermal_optimizer_state fresh")
        if _option_was_set("--thermal_freeze_mode") and args.thermal_freeze_mode != "strict":
            parser.error("--thermal_recipe aaai_strict requires --thermal_freeze_mode strict")
        if _option_was_set("--thermal_scale_clamp") and args.thermal_scale_clamp != "off":
            parser.error("--thermal_recipe aaai_strict requires --thermal_scale_clamp off")
        if _option_was_set("--ss_enable") or _option_was_set("--ss_prune_before_thermal"):
            parser.error("--thermal_recipe aaai_strict does not permit Stage-2 sparse-support pruning")

        args.thermal_reset_features = True
        if args.thermal_max_sh_degree is None:
            args.thermal_max_sh_degree = 1
        args.thermal_optimizer_state = "fresh"
        args.thermal_freeze_mode = "strict"
        args.thermal_scale_clamp = "off"
        args.ss_enable = False
        args.ss_prune_before_thermal = False
        args.sgf_disable = False

    if args.thermal_recipe == "geometry_frozen_opacity_adaptive":
        if not args.start_checkpoint:
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive requires --start_checkpoint"
            )
        if args.baseline_modules_off or args.baseline_restore_ssp or args.baseline_restore_stt:
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive cannot be combined "
                "with baseline recipe flags"
            )
        if _option_was_set("--sgf_disable"):
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive does not use --sgf_disable"
            )
        if _option_was_set("--thermal_max_sh_degree") and args.thermal_max_sh_degree != 3:
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive requires "
                "--thermal_max_sh_degree 3"
            )
        if _option_was_set("--thermal_freeze_mode") and (
            args.thermal_freeze_mode != "geometry_frozen_opacity_adaptive"
        ):
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive requires "
                "--thermal_freeze_mode geometry_frozen_opacity_adaptive"
            )
        if _option_was_set("--thermal_scale_clamp") and args.thermal_scale_clamp != "off":
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive requires "
                "--thermal_scale_clamp off"
            )
        if _option_was_set("--opacity_lr") and args.opacity_lr != 2e-4:
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive requires --opacity_lr 0.0002"
            )
        if _option_was_set("--ss_enable") or _option_was_set("--ss_prune_before_thermal"):
            parser.error(
                "--thermal_recipe geometry_frozen_opacity_adaptive does not permit "
                "Stage-2 sparse-support pruning"
            )
        resume_mode = (
            _option_was_set("--thermal_optimizer_state")
            and args.thermal_optimizer_state == "restore"
        )
        if resume_mode and _option_was_set("--thermal_reset_features"):
            parser.error(
                "Opacity-adaptive resume must not repeat --thermal_reset_features"
            )
        args.thermal_max_sh_degree = 3
        args.thermal_optimizer_state = "restore" if resume_mode else "fresh"
        args.thermal_freeze_mode = "geometry_frozen_opacity_adaptive"
        args.thermal_scale_clamp = "off"
        args.opacity_lr = 2e-4
        args.thermal_reset_features = not resume_mode
        args.ss_enable = False
        args.ss_prune_before_thermal = False
        args.sgf_disable = False

    if args.thermal_freeze_mode in (
        "strict", "continuous_unfrozen", "geometry_frozen_opacity_adaptive"
    ):
        if not args.start_checkpoint:
            parser.error(f"--thermal_freeze_mode {args.thermal_freeze_mode} requires --start_checkpoint")
        args.ss_prune_before_thermal = False
        args.ss_enable = False
    if args.thermal_freeze_mode == "strict" and args.thermal_scale_clamp != "off":
        parser.error("--thermal_freeze_mode strict requires --thermal_scale_clamp off")
    if (
        args.thermal_freeze_mode == "geometry_frozen_opacity_adaptive"
        and args.thermal_scale_clamp != "off"
    ):
        parser.error(
            "--thermal_freeze_mode geometry_frozen_opacity_adaptive requires "
            "--thermal_scale_clamp off"
        )

    # Thermal-stage default: if user did not explicitly pass --opacity_lr and a
    # checkpoint is provided, use a conservative opacity lr to avoid geometry drift.
    opacity_flag_set = any((a == "--opacity_lr") or a.startswith("--opacity_lr=") for a in cli_args)
    if (
        args.start_checkpoint
        and (not opacity_flag_set)
        and args.rgb_continuation_recipe == "legacy"
    ):
        if args.baseline_restore_stt:
            args.opacity_lr = 2e-4
        else:
            args.opacity_lr = 0.025 if args.baseline_modules_off else 2e-4

    args.save_iterations.append(args.iterations)
    if (not args.start_checkpoint) and args.ss_prune_after_rgb and (not args.ss_enable):
        args.ss_enable = True
        print("[INFO] ss_prune_after_rgb enabled by default; auto-enabling ss_enable for RGB stage.")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if args.benchmark_efficiency and not (args.efficiency_output or args.model_path):
        parser.error("--benchmark_efficiency requires -m/--model_path or --efficiency_output")
    efficiency_stage = args.efficiency_stage
    if efficiency_stage == "auto":
        efficiency_stage = "thermal" if args.start_checkpoint else "rgb"
    efficiency_output = args.efficiency_output or os.path.join(args.model_path, "train_efficiency.json")
    efficiency_probe = TorchStageProbe(
        args.benchmark_efficiency,
        efficiency_output,
        efficiency_stage,
        torch,
        metadata={
            "source_path": str(args.source_path),
            "model_path": str(args.model_path),
            "requested_final_iteration": int(args.iterations),
            "start_checkpoint": str(args.start_checkpoint) if args.start_checkpoint else None,
            "resolution": int(args.resolution),
            "thermal_recipe": str(args.thermal_recipe),
            "artifact_save_semantics": str(args.artifact_save_semantics),
            "thermal_max_sh_degree": args.thermal_max_sh_degree,
            "thermal_optimizer_state": str(args.thermal_optimizer_state),
            "thermal_freeze_mode": str(args.thermal_freeze_mode),
            "thermal_scale_clamp": str(args.thermal_scale_clamp),
            "temperature_loss_mode": str(args.temperature_loss_mode),
            "temperature_loss_calibration": str(
                args.temperature_loss_calibration
            ),
            "temperature_loss_calibration_manifest": str(
                args.temperature_loss_calibration_manifest
            ),
            "optimizer_step_at_final_iteration": bool(
                args.optimizer_step_at_final_iteration
            ),
            "rgb_continuation_recipe": str(args.rgb_continuation_recipe),
            "rgb_continuation_updates": int(args.rgb_continuation_updates),
            "fixed_camera_sequence": str(args.fixed_camera_sequence),
            "ogs_v1": bool(args.ogs_v1),
            "lambda_ogs": float(args.lambda_ogs),
            "ogs_v1_lambda_protocol": str(args.ogs_v1_lambda_protocol),
        },
    )
    efficiency_probe.start()
    try:
        training_result = training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
            ss_args=(args if args.ss_enable else None),
        )
    except BaseException as error:
        efficiency_probe.finish("failed", error=error)
        raise
    efficiency_probe.finish("completed", result=training_result)

    # All done
    print("\nTraining complete.")
