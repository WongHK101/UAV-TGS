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

import numpy as np
import hashlib
import os
import torch
from tqdm import tqdm
from os import makedirs
from pathlib import Path
from scene import Scene
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from efficiency_probe import atomic_write_json, benchmark_render_calls, now_iso
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


_RENDERER_SUPPORT_KEYS = (
    "alpha",
    "accumulation",
    "accum_alpha",
    "opacity",
    "support",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _view_source_name(view):
    source_name = str(getattr(view, "image_name", ""))
    if not source_name:
        raise ValueError("Every rendered view must have a non-empty image_name")
    return source_name


def _image_name_output_stems(views):
    """Return safe output stems and reject cross-platform collisions up front."""
    stems = []
    seen = {}
    for view in views:
        source_name = _view_source_name(view)
        stem = Path(source_name.replace("\\", "/")).stem
        if not stem or stem in {".", ".."}:
            raise ValueError(f"Invalid output stem derived from image_name: {source_name!r}")
        collision_key = stem.casefold()
        if collision_key in seen:
            raise ValueError(
                "Duplicate image_name stem is not allowed for named render output: "
                f"{seen[collision_key]!r} and {source_name!r} both map to {stem!r}"
            )
        seen[collision_key] = source_name
        stems.append(stem)
    return stems


def _renderer_support_array(render_package, expected_height, expected_width):
    """Extract a native per-pixel renderer support output without synthesising one."""
    source_key = next(
        (key for key in _RENDERER_SUPPORT_KEYS if key in render_package),
        None,
    )
    if source_key is None:
        available = ", ".join(sorted(str(key) for key in render_package.keys()))
        raise RuntimeError(
            "--save_renderer_support requested, but the renderer returned no native "
            "alpha/accumulation/opacity/support map. Refusing to synthesize a map. "
            f"Available render-package keys: {available or '<none>'}"
        )

    values = render_package[source_key]
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "float"):
        values = values.float()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    values = np.asarray(values)

    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    elif values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise RuntimeError(
            f"Renderer support key {source_key!r} must be a single-channel map, "
            f"got shape {values.shape}"
        )
    if values.shape != (int(expected_height), int(expected_width)):
        raise RuntimeError(
            f"Renderer support key {source_key!r} has shape {values.shape}, expected "
            f"{(int(expected_height), int(expected_width))}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Renderer support key {source_key!r} contains non-finite values")
    return np.asarray(values, dtype=np.float32), source_key


def _opacity_proxy_array(proxy_render, expected_height, expected_width):
    """Match the established depth evaluator's white-override opacity proxy."""
    values = proxy_render
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "float"):
        values = values.float()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[0] < 1:
        raise RuntimeError(
            "White-override opacity proxy must be a channel-first color render, "
            f"got shape {values.shape}"
        )
    values = values[0]
    if values.shape != (int(expected_height), int(expected_width)):
        raise RuntimeError(
            f"White-override opacity proxy has shape {values.shape}, expected "
            f"{(int(expected_height), int(expected_width))}"
        )
    if not np.all(np.isfinite(values)):
        raise RuntimeError("White-override opacity proxy contains non-finite values")
    return np.asarray(values, dtype=np.float32)


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    save_by_image_name=False,
    save_renderer_support=False,
    save_opacity_proxy=False,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    if save_by_image_name:
        views = list(views)
        output_stems = _image_name_output_stems(views)
    else:
        output_stems = None

    support_path = os.path.join(
        model_path,
        name,
        "ours_{}".format(iteration),
        "renderer_support",
    )
    opacity_proxy_path = os.path.join(
        model_path,
        name,
        "ours_{}".format(iteration),
        "opacity_proxy",
    )

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if save_renderer_support:
        makedirs(support_path, exist_ok=True)
    if save_opacity_proxy:
        makedirs(opacity_proxy_path, exist_ok=True)

    if save_opacity_proxy:
        black_background = torch.zeros_like(background)
        white_override = torch.ones(
            (gaussians.get_xyz.shape[0], 3),
            dtype=torch.float32,
            device=background.device,
        )
    else:
        black_background = None
        white_override = None

    mapping_entries = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_package = render(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
        )
        rendering = render_package["render"]
        gt = view.original_image[0:3, :, :]

        support_values = None
        support_source_key = None
        if save_renderer_support:
            support_values, support_source_key = _renderer_support_array(
                render_package,
                rendering.shape[-2],
                rendering.shape[-1],
            )

        opacity_proxy_values = None
        if save_opacity_proxy:
            opacity_package = render(
                view,
                gaussians,
                pipeline,
                black_background,
                scaling_modifier=1.0,
                separate_sh=False,
                override_color=white_override,
                use_trained_exp=False,
            )
            opacity_proxy_values = _opacity_proxy_array(
                opacity_package["render"],
                rendering.shape[-2],
                rendering.shape[-1],
            )

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
            if support_values is not None:
                support_values = support_values[..., support_values.shape[-1] // 2:]
            if opacity_proxy_values is not None:
                opacity_proxy_values = opacity_proxy_values[
                    ..., opacity_proxy_values.shape[-1] // 2:
                ]

        output_stem = output_stems[idx] if output_stems is not None else '{0:05d}'.format(idx)
        render_relative = os.path.join("renders", output_stem + ".png").replace("\\", "/")
        gt_relative = os.path.join("gt", output_stem + ".png").replace("\\", "/")
        render_output_path = os.path.join(render_path, output_stem + ".png")
        gt_output_path = os.path.join(gts_path, output_stem + ".png")
        torchvision.utils.save_image(rendering, render_output_path)
        torchvision.utils.save_image(gt, gt_output_path)

        support_relative = None
        support_output_path = None
        if save_renderer_support:
            support_relative = os.path.join(
                "renderer_support",
                output_stem + ".npy",
            ).replace("\\", "/")
            support_output_path = os.path.join(support_path, output_stem + ".npy")
            np.save(
                support_output_path,
                support_values,
                allow_pickle=False,
            )

        opacity_proxy_relative = None
        opacity_proxy_output_path = None
        if save_opacity_proxy:
            opacity_proxy_relative = os.path.join(
                "opacity_proxy",
                output_stem + ".npy",
            ).replace("\\", "/")
            opacity_proxy_output_path = os.path.join(
                opacity_proxy_path,
                output_stem + ".npy",
            )
            np.save(
                opacity_proxy_output_path,
                opacity_proxy_values,
                allow_pickle=False,
            )

        if save_by_image_name or save_renderer_support or save_opacity_proxy:
            mapping_entries.append(
                {
                    "split": str(name),
                    "iteration": int(iteration),
                    "source_image_name": _view_source_name(view),
                    "output": {
                        "render": render_relative,
                        "ground_truth": gt_relative,
                        "renderer_support": support_relative,
                        "opacity_proxy": opacity_proxy_relative,
                    },
                    "output_sha256": {
                        "render": _sha256(render_output_path),
                        "ground_truth": _sha256(gt_output_path),
                        "renderer_support": (
                            _sha256(support_output_path)
                            if support_output_path is not None
                            else None
                        ),
                        "opacity_proxy": (
                            _sha256(opacity_proxy_output_path)
                            if opacity_proxy_output_path is not None
                            else None
                        ),
                    },
                    "renderer_support_source_key": support_source_key,
                    "opacity_proxy_semantics": (
                        "black_bg_plus_white_override_color_render"
                        if save_opacity_proxy
                        else None
                    ),
                    "support_threshold_applied": False,
                    "support_threshold": None,
                }
            )

    if save_by_image_name or save_renderer_support or save_opacity_proxy:
        manifest_path = os.path.join(
            model_path,
            name,
            "ours_{}".format(iteration),
            "render_mapping_manifest.json",
        )
        atomic_write_json(
            manifest_path,
            {
                "schema_name": "uav-tgs-render-output-mapping",
                "schema_version": 1,
                "split": str(name),
                "iteration": int(iteration),
                "name_mode": "image_name_stem" if save_by_image_name else "sequential_index",
                "renderer_support_saved": bool(save_renderer_support),
                "opacity_proxy_saved": bool(save_opacity_proxy),
                "opacity_proxy_semantics": (
                    "black_bg_plus_white_override_color_render"
                    if save_opacity_proxy
                    else None
                ),
                "support_threshold_applied": False,
                "support_threshold": None,
                "entries": mapping_entries,
            },
        )

def _view_resolution(view):
    width = getattr(view, "image_width", None)
    height = getattr(view, "image_height", None)
    if width is None or height is None:
        image = getattr(view, "original_image", None)
        if image is not None and len(image.shape) >= 2:
            height, width = image.shape[-2:]
    try:
        return {"width": int(width), "height": int(height)}
    except (TypeError, ValueError):
        return {"width": None, "height": None}


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    benchmark_efficiency: bool = False,
    benchmark_warmup_views: int = 10,
    benchmark_repeats: int = 3,
    benchmark_output: str = "",
    benchmark_only: bool = False,
    save_by_image_name: bool = False,
    save_renderer_support: bool = False,
    save_opacity_proxy: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if benchmark_efficiency:
            test_views = list(scene.getTestCameras())
            if not test_views:
                raise RuntimeError("Efficiency benchmark requires at least one test camera")

            def _render_once(view):
                return render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    separate_sh=separate_sh,
                )["render"]

            benchmark = benchmark_render_calls(
                _render_once,
                test_views,
                repeats=benchmark_repeats,
                warmup_views=benchmark_warmup_views,
                torch_module=torch,
            )
            resolutions = []
            for view in test_views:
                resolution = _view_resolution(view)
                if resolution not in resolutions:
                    resolutions.append(resolution)
            output_path = benchmark_output or os.path.join(
                dataset.model_path,
                "test",
                "ours_{}".format(scene.loaded_iter),
                "render_efficiency.json",
            )
            atomic_write_json(
                output_path,
                {
                    "schema_name": "uav-tgs-efficiency",
                    "schema_version": 1,
                    "kind": "render",
                    "status": "completed",
                    "created_at": now_iso(),
                    "model_path": str(dataset.model_path),
                    "iteration": int(scene.loaded_iter),
                    "split": "test",
                    "gaussian_count": int(gaussians.get_xyz.shape[0]),
                    "resolutions": resolutions,
                    "benchmark": benchmark,
                },
            )
            print("Efficiency benchmark written to " + str(output_path))

        # Keep benchmark warm-up controlled and comparable: when both actions
        # are requested, measure before the full PNG render pass.
        if not benchmark_only and not skip_train:
             render_set(
                 dataset.model_path,
                 "train",
                 scene.loaded_iter,
                 scene.getTrainCameras(),
                 gaussians,
                 pipeline,
                 background,
                 dataset.train_test_exp,
                 separate_sh,
                 save_by_image_name,
                 save_renderer_support,
                 save_opacity_proxy,
             )

        if not benchmark_only and not skip_test:
             render_set(
                 dataset.model_path,
                 "test",
                 scene.loaded_iter,
                 scene.getTestCameras(),
                 gaussians,
                 pipeline,
                 background,
                 dataset.train_test_exp,
                 separate_sh,
                 save_by_image_name,
                 save_renderer_support,
                 save_opacity_proxy,
             )

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--benchmark_efficiency", action="store_true", default=False,
                        help="Benchmark render-only test-view throughput and peak PyTorch CUDA memory.")
    parser.add_argument("--benchmark_warmup_views", type=int, default=10)
    parser.add_argument("--benchmark_repeats", type=int, default=3)
    parser.add_argument("--benchmark_output", type=str, default="")
    parser.add_argument("--benchmark_only", action="store_true", default=False,
                        help="Skip PNG rendering and run only the opt-in efficiency benchmark.")
    parser.add_argument("--save_by_image_name", action="store_true", default=False,
                        help="Opt in to image_name-stem render/GT filenames and a deterministic mapping manifest.")
    parser.add_argument("--save_renderer_support", action="store_true", default=False,
                        help="Save a native renderer alpha/accumulation/opacity/support map as float32 NPY; fail if unavailable.")
    parser.add_argument("--save_opacity_proxy", action="store_true", default=False,
                        help="Save the established black-background/white-override opacity proxy as float32 NPY.")
    args = get_combined_args(parser)
    if args.benchmark_only and not args.benchmark_efficiency:
        parser.error("--benchmark_only requires --benchmark_efficiency")
    if args.benchmark_efficiency and args.skip_test:
        parser.error("--benchmark_efficiency requires the test split (remove --skip_test)")
    if args.benchmark_warmup_views < 0:
        parser.error("--benchmark_warmup_views must be non-negative")
    if args.benchmark_repeats <= 0:
        parser.error("--benchmark_repeats must be positive")
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        benchmark_efficiency=args.benchmark_efficiency,
        benchmark_warmup_views=args.benchmark_warmup_views,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_output=args.benchmark_output,
        benchmark_only=args.benchmark_only,
        save_by_image_name=args.save_by_image_name,
        save_renderer_support=args.save_renderer_support,
        save_opacity_proxy=args.save_opacity_proxy,
    )
