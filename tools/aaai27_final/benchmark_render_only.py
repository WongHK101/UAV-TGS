#!/usr/bin/env python3
"""Uniform thermal render-only benchmark for frozen UAV-TGS endpoints.

Each invocation loads exactly one method/scene endpoint in a clean process.
Model loading, dataset scanning and correctness I/O occur outside the timed
region.  The timed region covers one full formal test-camera pass from the
method's per-view inference entry point to the final in-memory GPU thermal
tensor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


SCHEMA = "uav-tgs-aaai27-unified-render-only-v1"
METHODS = (
    "raw_f3",
    "scsp_refit_f3",
    "adaptive_opacity_scale_clamp",
    "thermalgaussian_ommg",
    "mmone",
    "thermal3dgs",
    "thermonerf",
    "physir_splat_sh",
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _ensure_loader_file_limit(minimum: int = 65536) -> dict[str, int] | None:
    """Raise the Linux soft descriptor limit for upstream PIL camera loaders."""
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, minimum), hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    current, current_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    return {"soft": int(current), "hard": int(current_hard)}


def _trusted_cfg(path: Path) -> argparse.Namespace:
    """Read a frozen 3DGS Namespace cfg_args file.

    cfg_args is executable Python produced by the upstream repositories.  It
    is trusted endpoint metadata, not user input.  Only Namespace and
    torch.device are exposed to evaluation.
    """
    text = path.read_text(encoding="utf-8").strip()
    value = eval(  # noqa: S307 - trusted frozen endpoint artifact
        text,
        {"__builtins__": {}, "Namespace": argparse.Namespace, "device": torch.device},
    )
    if not isinstance(value, argparse.Namespace):
        raise ValueError(f"cfg_args is not an argparse Namespace: {path}")
    return value


def _merge_cfg(parser: argparse.ArgumentParser, cfg_path: Path) -> argparse.Namespace:
    args = parser.parse_args([])
    frozen = _trusted_cfg(cfg_path)
    for key, value in vars(frozen).items():
        setattr(args, key, value)
    return args


def _read_names(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid or duplicate camera list: {path}")
    return values


def _stem(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem.casefold()


def _order_views(views: Sequence[Any], formal_names: Sequence[str]) -> list[Any]:
    mapping: dict[str, Any] = {}
    for view in views:
        name = getattr(view, "image_name", None)
        if not isinstance(name, str):
            raise ValueError("camera object lacks image_name")
        key = _stem(name)
        if key in mapping:
            raise ValueError(f"duplicate loaded camera stem: {key}")
        mapping[key] = view
    requested = [_stem(value) for value in formal_names]
    missing = [key for key in requested if key not in mapping]
    extra = sorted(set(mapping) - set(requested))
    if missing or extra:
        raise ValueError(f"formal/loaded test cameras differ: missing={missing[:8]} extra={extra[:8]}")
    return [mapping[key] for key in requested]


def _formal_names(args: argparse.Namespace) -> list[str]:
    if args.test_list is not None:
        return _read_names(args.test_list)
    if args.adapter_manifest is not None:
        payload = _load_json(args.adapter_manifest)
        # Phase-A manifests used test_names; the later adapter revision records
        # the same exact COLMAP order explicitly as test_camera_names.
        names = payload.get("test_camera_names", payload.get("test_names"))
        if not isinstance(names, list) or not names or not all(isinstance(item, str) for item in names):
            raise ValueError("adapter manifest lacks test_camera_names")
        if len(names) != len(set(names)):
            raise ValueError("adapter test_camera_names contains duplicates")
        return names
    raise ValueError("either --test-list or --adapter-manifest is required")


def _set_common_endpoint_args(
    value: argparse.Namespace, args: argparse.Namespace
) -> argparse.Namespace:
    value.model_path = str(args.model_root.resolve())
    value.source_path = str(args.dataset_path.resolve())
    value.eval = True
    return value


def _resize_rgb_to_formal(value: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """GPU equivalent of the frozen RGB PIL-default resize adapter."""
    target = (int(args.formal_height), int(args.formal_width))
    if tuple(value.shape[-2:]) == target:
        return value.clamp(0.0, 1.0)
    # The formal adapter serializes the native render before PIL resize.  Keep
    # the same 8-bit boundary while excluding image encoding and CPU I/O.
    quantized = torch.round(value.clamp(0.0, 1.0) * 255.0) / 255.0
    return F.interpolate(
        quantized.unsqueeze(0),
        size=target,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0].clamp(0.0, 1.0)


def _import_repo(repo: Path) -> None:
    resolved = str(repo.resolve())
    if sys.path[0] != resolved:
        sys.path.insert(0, resolved)


def _endpoint_files(method: str, root: Path, iteration: int) -> list[Path]:
    patterns: dict[str, tuple[str, ...]] = {
        "raw_f3": (f"point_cloud/iteration_{iteration}/point_cloud.ply",),
        "scsp_refit_f3": (f"point_cloud/iteration_{iteration}/point_cloud.ply",),
        "adaptive_opacity_scale_clamp": (f"point_cloud/iteration_{iteration}/point_cloud.ply",),
        "thermalgaussian_ommg": (f"point_cloud/iteration_{iteration}/point_cloud.ply",),
        "mmone": (f"point_cloud/iteration_{iteration}/point_cloud.ply", "thermal_chkpnt30000.pth"),
        "thermal3dgs": (
            f"point_cloud/iteration_{iteration}/point_cloud.ply",
            f"ATF/iteration_{iteration}/ATF.pth",
            f"TCM/iteration_{iteration}/TCM.pth",
        ),
    }
    if method == "thermonerf":
        matches = sorted(root.rglob("*.ckpt")) + sorted(root.rglob("config.yml"))
    elif method == "physir_splat_sh":
        matches = [root / f"point_cloud/iteration_{iteration}/point_cloud.ply"]
        matches += [root / f"Geometry/iteration_{iteration}/geometry.pth"]
    else:
        matches = [root / value for value in patterns[method]]
    if method != "thermonerf":
        matches.append(root / "cfg_args")
    missing = [path for path in matches if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing endpoint assets: {missing}")
    if not matches:
        raise ValueError(f"no endpoint assets found under {root}")
    return matches


@dataclass
class LoadedRenderer:
    views: list[Any]
    render_view: Callable[[Any], torch.Tensor]
    loaded_iteration: int
    inference_dtype: str
    metadata: dict[str, Any]


def _load_standard_3dgs(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cfg = _set_common_endpoint_args(_merge_cfg(parser, args.model_root / "cfg_args"), args)
    dataset = model.extract(cfg)
    pipe = pipeline.extract(cfg)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = _order_views(scene.getTestCameras(), _formal_names(args))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    def render_view(view: Any) -> torch.Tensor:
        return render(view, gaussians, pipe, background)["render"].clamp(0.0, 1.0)

    return LoadedRenderer(
        views,
        render_view,
        int(scene.loaded_iter),
        str(gaussians.get_xyz.dtype),
        {"renderer_path": "UAV-TGS gaussian_renderer.render[render]"},
    )


def _load_ommg(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cfg = _set_common_endpoint_args(_merge_cfg(parser, args.model_root / "cfg_args"), args)
    dataset = model.extract(cfg)
    pipe = pipeline.extract(cfg)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    views = _order_views(scene.getTestCameras(), _formal_names(args))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    def render_view(view: Any) -> torch.Tensor:
        # OMMG computes RGB and thermal jointly in one unavoidable call.
        value = render(view, gaussians, pipe, background)["render_thermal"]
        return _resize_rgb_to_formal(value, args)

    return LoadedRenderer(
        views,
        render_view,
        int(scene.loaded_iter),
        str(gaussians.get_xyz.dtype),
        {"renderer_path": "OMMG gaussian_renderer.render[render_thermal]", "joint_rgb_thermal": True},
    )


def _load_mmone(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    from arguments import ModelParams, PipelineParams
    from gaussian_renderer import GaussianModel, render
    from scene import Scene

    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cfg = _set_common_endpoint_args(_merge_cfg(parser, args.model_root / "cfg_args"), args)
    cfg.joint = True
    cfg.rgb_thermal = True
    cfg.thermal_density = True
    cfg.include_language = False
    cfg.language_density = False
    dataset = model.extract(cfg)
    pipe = pipeline.extract(cfg)
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    checkpoint = args.model_root / "thermal_chkpnt30000.pth"
    model_params, restored_iteration = torch.load(checkpoint, map_location="cuda", weights_only=False)
    gaussians.restore(model_params, cfg, mode="test")
    views = _order_views(scene.getTestCameras(), _formal_names(args))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    def render_view(view: Any) -> torch.Tensor:
        # The official MMOne path rasterizes RGB and thermal in the same call.
        value = render(view, gaussians, pipe, background, cfg)["thermals"]
        return _resize_rgb_to_formal(value, args)

    return LoadedRenderer(
        views,
        render_view,
        int(restored_iteration),
        str(gaussians.get_xyz.dtype),
        {"renderer_path": "MMOne gaussian_renderer.render[thermals]", "joint_rgb_thermal": True},
    )


def _load_thermal3dgs(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    from arguments import ModelParams, PipelineParams
    import gaussian_renderer as renderer_module
    from scene import ATFModel, Scene, TCMModel

    # The frozen formal runtime uses the repository's depth rasterizer (color,
    # radii, depth), while this pinned renderer revision consumes only color
    # and radii.  Adapt only that return arity in the clean benchmark process;
    # the color tensor and all CUDA work remain the official path.
    rasterizer_type = renderer_module.GaussianRasterizer

    class _ColorRadiiRasterizer:
        def __init__(self, *values: Any, **named: Any) -> None:
            self._delegate = rasterizer_type(*values, **named)

        def __call__(self, *values: Any, **named: Any) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self._delegate(*values, **named)
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                raise ValueError("unexpected Thermal3D-GS rasterizer output")
            return outputs[0], outputs[1]

    renderer_module.GaussianRasterizer = _ColorRadiiRasterizer
    GaussianModel = renderer_module.GaussianModel
    render = renderer_module.render

    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    cfg = _set_common_endpoint_args(_merge_cfg(parser, args.model_root / "cfg_args"), args)
    dataset = model.extract(cfg)
    pipe = pipeline.extract(cfg)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    atf = ATFModel(dataset.is_blender)
    atf.load_weights(dataset.model_path)
    tcm = TCMModel()
    tcm.load_weights(dataset.model_path)
    if hasattr(atf, "eval"):
        atf.eval()
    if hasattr(tcm, "eval"):
        tcm.eval()
    views = _order_views(scene.getTestCameras(), _formal_names(args))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    def render_view(view: Any) -> torch.Tensor:
        if bool(getattr(dataset, "load2gpu_on_the_fly", False)):
            view.load2gpu()
        fid = view.fid
        xyz = gaussians.get_xyz
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        absorption, scattering, distance = atf.step(xyz, time_input)
        d_rgb = torch.exp((absorption + scattering) * distance)
        value = render(view, gaussians, pipe, background, d_rgb)["render"]
        return (value + tcm.step(value)).clamp(0.0, 1.0)

    return LoadedRenderer(
        views,
        render_view,
        int(scene.loaded_iter),
        str(gaussians.get_xyz.dtype),
        {
            "renderer_path": "Thermal3D-GS ATF + splat + TCM",
            "runtime_compatibility": "depth-rasterizer color/radii return-arity adapter",
        },
    )


def _hotiron_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Repository-owned 256-entry LUT, reproduced with integer-identical rules.
    index = np.arange(256, dtype=np.int32)
    red = np.clip(3 * index, 0, 255)
    green = np.clip(3 * index - 255, 0, 255)
    blue = np.clip(3 * index - 510, 0, 255)
    lut = np.stack((red, green, blue), axis=1).astype(np.uint8)
    if np.unique(lut, axis=0).shape[0] != 256:
        raise RuntimeError("Hot-Iron LUT is not unique")
    gray = ((lut[:, 0].astype(np.int32) * 4899 + lut[:, 1].astype(np.int32) * 9617 +
             lut[:, 2].astype(np.int32) * 1868 + 8192) >> 14).astype(np.int16)
    inverse = np.empty(256, dtype=np.int64)
    for value in range(256):
        inverse[value] = int(np.argmin(np.abs(gray - value)))
    return (
        torch.as_tensor(inverse, dtype=torch.long, device=device),
        torch.as_tensor(lut, dtype=torch.float32, device=device) / 255.0,
    )


def _load_thermonerf(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    from thermo_nerf.render.renderer import Renderer

    pipeline, _config, loaded_step = Renderer.extract_pipeline(
        args.model_root.resolve(), args.dataset_path.resolve()
    )
    pipeline.eval()
    datamanager = pipeline.datamanager
    datamanager.setup_eval()
    loader = datamanager.fixed_indices_eval_dataloader
    if loader is None:
        raise RuntimeError("ThermoNeRF fixed eval dataloader is unavailable")
    views = [camera for camera, _batch in loader]
    names = _formal_names(args)
    if len(views) != len(names):
        raise ValueError(f"ThermoNeRF/formal test count differs: {len(views)} != {len(names)}")
    inverse_lut, rgb_lut = _hotiron_tables(torch.device("cuda"))
    formal_size = (int(args.formal_height), int(args.formal_width))

    def render_view(cameras: Any) -> torch.Tensor:
        camera_indices = torch.arange(cameras.camera_to_worlds.shape[0])
        ray_bundle = cameras.generate_rays(camera_indices)
        flat = ray_bundle.flatten()
        pipeline.model.camera_optimizer.apply_to_raybundle(flat)
        ray_bundle = flat.reshape(ray_bundle.shape)
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        value = outputs["thermal"]
        while value.ndim > 2 and value.shape[-1] == 1:
            value = value[..., 0]
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"ThermoNeRF thermal tensor is not HxW: {tuple(value.shape)}")
        # The frozen formal adapter converts official grayscale output to 8-bit,
        # resizes with nearest-neighbour, inverts Hot-Iron luma and emits RGB.
        value = torch.round(value.clamp(0.0, 1.0) * 255.0).to(torch.long)
        value = torch.round(
            F.interpolate(
                value[None, None].float(),
                size=formal_size,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )[0, 0].clamp(0.0, 255.0)
        ).to(torch.long)
        indices = inverse_lut[value]
        return rgb_lut[indices].permute(2, 0, 1).contiguous()

    parameter = next(pipeline.model.parameters())
    return LoadedRenderer(
        views,
        render_view,
        int(loaded_step),
        str(parameter.dtype),
        {
            "renderer_path": "ThermoNeRF official rays/model + frozen Hot-Iron grayscale adapter",
            "native_to_formal_in_timed_path": True,
        },
    )


def _load_physir(args: argparse.Namespace) -> LoadedRenderer:
    _import_repo(args.source_repo)
    entry_path = args.source_repo / "render.py"
    spec = importlib.util.spec_from_file_location("_uav_tgs_physir_render_entry", entry_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load PhysIR render entry: {entry_path}")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from gaussian_renderer import GaussianModel
    from scene import GeoRefineModel, Scene

    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser(add_help=False)
    model = ModelParams(parser, device, sentinel=True)
    pipeline = PipelineParams(parser)
    optimization = OptimizationParams(parser, sentinel=True)
    cfg = _set_common_endpoint_args(_merge_cfg(parser, args.model_root / "cfg_args"), args)
    dataset = model.extract(cfg)
    pipe = pipeline.extract(cfg)
    opt = optimization.extract(cfg)
    gaussians = GaussianModel(dataset.sh_degree, device)
    if getattr(dataset, "load_model_path", ""):
        dataset.load_model_path = ""
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    geo_model = GeoRefineModel(
        device,
        dataset.is_blender,
        dataset.is_6dof,
        time_multires=(opt.geometry_time_multires if opt.geometry_time_multires >= 0 else None),
    )
    geo_model.load_weights(dataset.model_path)
    if hasattr(geo_model, "eval"):
        geo_model.eval()
    views = _order_views(scene.getTestCameras(), _formal_names(args))
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    time_interval = 1.0 / max(len(views), 1)
    no_eval_noise = lambda _: 0.0

    def render_view(view: Any) -> torch.Tensor:
        if bool(getattr(dataset, "load2gpu_on_the_fly", False)):
            view.load2device()
        fid = view.fid
        if entry._use_static_temporal_geometry(gaussians, view, False):
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            d_xyz, d_rotation, d_scaling = entry.compute_geo_fields(
                view,
                gaussians,
                geo_model,
                pipe,
                background,
                opt,
                int(scene.loaded_iter),
                no_eval_noise,
                time_interval,
                device,
                is_blender=True,
                feature_set="thermal",
                eval_mode=True,
            )
            d_xyz, _, _, _ = entry.apply_temporal_pose_delta(
                d_xyz, gaussians, fid, None, opt, int(scene.loaded_iter), scene.cameras_extent
            )
        value = entry.render(
            view,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            device,
            dataset.is_6dof,
            override_color=entry._temporal_override_color(gaussians, view),
            feature_set="thermal",
            opacity_threshold=float(getattr(opt, "ir_eval_opacity_threshold", 0.0)),
        )["render"]
        return value.clamp(0.0, 1.0)

    return LoadedRenderer(
        views,
        render_view,
        int(scene.loaded_iter),
        str(gaussians.get_xyz.dtype),
        {
            "renderer_path": "PhysIR frozen geometry field + thermal SH splat",
            "physical_renderer": False,
            "vggt_ir": False,
        },
    )


def _load_renderer(args: argparse.Namespace) -> LoadedRenderer:
    if args.method in {"raw_f3", "scsp_refit_f3", "adaptive_opacity_scale_clamp"}:
        return _load_standard_3dgs(args)
    if args.method == "thermalgaussian_ommg":
        return _load_ommg(args)
    if args.method == "mmone":
        return _load_mmone(args)
    if args.method == "thermal3dgs":
        return _load_thermal3dgs(args)
    if args.method == "thermonerf":
        return _load_thermonerf(args)
    if args.method == "physir_splat_sh":
        return _load_physir(args)
    raise ValueError(args.method)


def _tensor_u8(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().clamp(0.0, 1.0)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3:
        raise ValueError(f"render tensor must be CHW/HW, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    if tensor.shape[0] != 3:
        raise ValueError(f"render tensor must have 1 or 3 channels, got {tuple(tensor.shape)}")
    return torch.round(tensor * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def _correctness(value: torch.Tensor, reference: Path | None) -> dict[str, Any]:
    finite = bool(torch.isfinite(value).all().item())
    payload: dict[str, Any] = {
        "shape_chw": list(value.shape),
        "finite": finite,
        "reference_compared": reference is not None,
    }
    if not finite:
        raise ValueError("non-finite Building correctness tensor")
    if reference is None:
        return payload
    expected = np.asarray(Image.open(reference).convert("RGB"), dtype=np.uint8)
    actual = _tensor_u8(value)
    if actual.shape != expected.shape:
        raise ValueError(f"correctness shape mismatch: {actual.shape} != {expected.shape}")
    difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    p99_abs = float(np.quantile(difference, 0.99))
    tail_within_tolerance = (
        difference.max() <= args_global.correctness_max_abs_u8
        if args_global.correctness_p99_abs_u8 is None
        else p99_abs <= args_global.correctness_p99_abs_u8
    )
    payload.update(
        {
            "reference_path": str(reference.resolve()),
            "reference_sha256": _sha256(reference),
            "max_abs_u8": int(difference.max()),
            "mean_abs_u8": float(difference.mean()),
            "p99_abs_u8": p99_abs,
            "within_tolerance": bool(
                tail_within_tolerance
                and difference.mean() <= args_global.correctness_mean_abs_u8
            ),
            "tolerance": {
                "max_abs_u8": int(args_global.correctness_max_abs_u8),
                "p99_abs_u8": (
                    float(args_global.correctness_p99_abs_u8)
                    if args_global.correctness_p99_abs_u8 is not None
                    else None
                ),
                "mean_abs_u8": float(args_global.correctness_mean_abs_u8),
            },
        }
    )
    if not payload["within_tolerance"]:
        raise ValueError(f"Building correctness drift exceeds tolerance: {payload}")
    return payload


def benchmark(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal render-only benchmark")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark receipt: {args.output}")
    nofile_limit = _ensure_loader_file_limit()
    loaded = _load_renderer(args)
    if len(loaded.views) != args.expected_view_count:
        raise ValueError(f"test view count mismatch: {len(loaded.views)} != {args.expected_view_count}")

    endpoint_files = _endpoint_files(args.method, args.model_root, loaded.loaded_iteration)
    provenance = [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in endpoint_files
    ]

    with torch.inference_mode():
        # One fixed Building view correctness check, outside the timed passes.
        first = loaded.render_view(loaded.views[0])
        torch.cuda.synchronize()
        correctness = _correctness(first, args.correctness_reference)
        del first

        # Required complete-list warm-up.
        for view in loaded.views:
            value = loaded.render_view(view)
        torch.cuda.synchronize()
        if not torch.isfinite(value).all().item():
            raise ValueError("non-finite warm-up render")
        output_shape = list(value.shape)
        output_dtype = str(value.dtype)
        del value

        torch.cuda.reset_peak_memory_stats()
        passes: list[dict[str, Any]] = []
        for index in range(args.passes):
            torch.cuda.synchronize()
            started_ns = time.perf_counter_ns()
            for view in loaded.views:
                value = loaded.render_view(view)
            torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - started_ns
            if not torch.isfinite(value).all().item():
                raise ValueError(f"non-finite timed render in pass {index + 1}")
            elapsed_s = elapsed_ns / 1_000_000_000.0
            ms_per_view = elapsed_ns / 1_000_000.0 / len(loaded.views)
            passes.append(
                {
                    "pass": index + 1,
                    "elapsed_ns": elapsed_ns,
                    "elapsed_s": elapsed_s,
                    "view_count": len(loaded.views),
                    "ms_per_view": ms_per_view,
                    "views_per_s": 1000.0 / ms_per_view,
                }
            )

    med_ms = float(statistics.median(item["ms_per_view"] for item in passes))
    allocated = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    source_commit = args.source_commit
    if source_commit is None:
        source_commit = subprocess.run(
            ["git", "-C", str(args.source_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if len(source_commit) != 40 or any(value not in "0123456789abcdef" for value in source_commit):
        raise ValueError(f"invalid source commit: {source_commit}")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "scene": args.scene,
        "method": args.method,
        "display_name": args.display_name,
        "benchmark_scope": "thermal render-only end-to-end in-memory latency",
        "timing_includes": [
            "per-view formal inference call",
            "runtime-required camera/ray generation",
            "runtime-required modality fusion or joint RGB/T compute",
            "all CUDA kernels",
            "necessary model processing before final GPU thermal tensor",
        ],
        "timing_excludes": [
            "model loading",
            "dataset scan and GT reads",
            "image encoding or saving",
            "GPU-to-CPU copy",
            "metric and depth evaluation",
            "TSDK decode and common postprocessing",
        ],
        "warmup": {"complete_test_passes": 1, "view_count": len(loaded.views)},
        "passes": passes,
        "scene_result": {
            "median_ms_per_view": med_ms,
            "views_per_s": 1000.0 / med_ms,
        },
        "view_count": len(loaded.views),
        "output_shape_chw": output_shape,
        "output_resolution_wh": [int(output_shape[-1]), int(output_shape[-2])],
        "batch_size": 1,
        "inference_dtype": loaded.inference_dtype,
        "output_dtype": output_dtype,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "gpu": {
            "name": gpu.name,
            "total_memory_bytes": int(gpu.total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "loader_nofile_limit": nofile_limit,
        "loaded_iteration": loaded.loaded_iteration,
        "benchmark_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "source_repository": {
            "path": str(args.source_repo.resolve()),
            "commit": source_commit,
        },
        "correctness": correctness,
        "renderer": loaded.metadata,
        "endpoint_assets": provenance,
        "formal_camera_source": {
            "test_list": str(args.test_list.resolve()) if args.test_list else None,
            "test_list_sha256": _sha256(args.test_list) if args.test_list else None,
            "adapter_manifest": str(args.adapter_manifest.resolve()) if args.adapter_manifest else None,
            "adapter_manifest_sha256": _sha256(args.adapter_manifest) if args.adapter_manifest else None,
        },
    }
    _atomic_json(args.output, payload)
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--adapter-manifest", type=Path)
    parser.add_argument("--expected-view-count", type=int, required=True)
    parser.add_argument("--formal-width", type=int, default=320)
    parser.add_argument("--formal-height", type=int, default=256)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--correctness-reference", type=Path)
    parser.add_argument("--correctness-max-abs-u8", type=int, default=1)
    parser.add_argument("--correctness-p99-abs-u8", type=float)
    parser.add_argument("--correctness-mean-abs-u8", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser


# Correctness thresholds are part of the receipt; this module-global avoids
# threading those immutable CLI values through a method-specific closure.
args_global: argparse.Namespace


def main() -> int:
    global args_global
    args_global = build_parser().parse_args()
    if args_global.passes != 3:
        raise ValueError("formal protocol requires exactly three timed passes")
    target = benchmark(args_global)
    print(json.dumps({"status": "completed", "receipt": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
