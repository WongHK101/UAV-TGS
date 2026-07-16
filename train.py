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

# Optional: pseudo-color thermal structure loss (added for improved thermal texture preservation)
try:
    from utils.loss_utils import structure_grad_loss
except Exception:
    structure_grad_loss = None

from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
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


def _tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _snapshot_strict_freeze_fields(gaussians):
    return {
        name: getattr(gaussians, name).detach().cpu().contiguous().clone()
        for name in _STRICT_FREEZE_FIELDS
    }


def _write_strict_freeze_audit(model_path, gaussians, before, start_iteration, final_iteration):
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
        "schema": "uav-tgs-strict-freeze-audit-v1",
        "status": "passed" if all_unchanged else "failed",
        "start_iteration": int(start_iteration),
        "final_iteration": int(final_iteration),
        "fields": fields,
    }
    output_path = os.path.join(model_path, "strict_freeze_audit.json")
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
    print(f"[INFO] StrictFreezeAudit: passed path={output_path}")
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
    strict_freeze = thermal_freeze_mode == "strict"
    opacity_adaptive = thermal_freeze_mode == "geometry_frozen_opacity_adaptive"
    topology_frozen = thermal_freeze_mode in (
        "strict", "continuous_unfrozen", "geometry_frozen_opacity_adaptive"
    )
    artifact_save_semantics = _resolve_artifact_save_semantics(
        getattr(args, "thermal_recipe", "legacy"),
        getattr(args, "artifact_save_semantics", None),
    )
    aligned_artifact_saves = artifact_save_semantics == "aligned"
    if aligned_artifact_saves:
        print(
            "[INFO] AAAIArtifactSaveSemantics: "
            "semantics=aligned aligned_post_optimizer_step=1"
        )
    if topology_frozen and not checkpoint:
        raise RuntimeError(f"thermal_freeze_mode={thermal_freeze_mode} requires a start checkpoint")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)


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
    if checkpoint:
        # Checkpoints are generated locally by this training pipeline and
        # contain NumPy/Python objects in addition to tensor weights.  PyTorch
        # 2.6+ defaults to weights_only=True, which rejects that established
        # checkpoint format before thermal fine-tuning can start.
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        optimizer_restore_mode = str(getattr(args, "thermal_optimizer_state", "restore"))
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

    if checkpoint and getattr(args, "start_checkpoint", None) and not getattr(args, "sgf_disable", False):
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

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = (
        opt.optimizer_type == "sparse_adam"
        and SPARSE_ADAM_AVAILABLE
        and not topology_frozen
    )
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
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

        # Pick a random Camera
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

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
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
            if iteration < opt.iterations:
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
    elif opacity_adaptive:
        _write_opacity_adaptive_freeze_audit(
            scene.model_path,
            gaussians,
            opacity_adaptive_freeze_snapshot,
            start_iteration=start_iteration,
            final_iteration=int(opt.iterations),
        )

    return {
        "start_iteration": start_iteration,
        "final_iteration": int(opt.iterations),
        "iterations_executed": max(0, int(opt.iterations) - start_iteration),
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

    cli_args = sys.argv[1:]
    args = parser.parse_args(cli_args)

    def _option_was_set(name):
        return any(token == name or token.startswith(name + "=") for token in cli_args)

    if args.thermal_max_sh_degree is not None and not args.start_checkpoint:
        parser.error("--thermal_max_sh_degree requires --start_checkpoint")
    if args.thermal_optimizer_state == "fresh" and not args.start_checkpoint:
        parser.error("--thermal_optimizer_state fresh requires --start_checkpoint")

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
    if args.start_checkpoint and (not opacity_flag_set):
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
