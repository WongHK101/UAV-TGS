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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
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


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

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
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not benchmark_only and not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

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
    )
