from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import Camera
from tools.geometric_repeatability.depth_reference_common import (
    apply_world_transform_to_camera_to_world,
    estimate_strict_to_native_world_transform,
    load_json,
    load_native_camera_entries,
)
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _camera_to_world_from_view(view) -> List[List[float]]:
    w2c = getWorld2View2(view.R, view.T, view.trans, view.scale).astype(np.float64)
    c2w = np.linalg.inv(w2c)
    return c2w.tolist()


def _tensor_hwc_to_numpy_hw(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[0] in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f"Unsupported 3D tensor shape for image conversion: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D tensor after conversion, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _render_depth_and_opacity(view, gaussians, pipeline, black_bg: torch.Tensor, white_override: torch.Tensor) -> Dict[str, np.ndarray]:
    with torch.no_grad():
        depth_out = render(
            view,
            gaussians,
            pipeline,
            black_bg,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=None,
            use_trained_exp=False,
        )
        alpha_out = render(
            view,
            gaussians,
            pipeline,
            black_bg,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=white_override,
            use_trained_exp=False,
        )
    depth = _tensor_hwc_to_numpy_hw(depth_out["depth"])
    opacity_render = alpha_out["render"].detach().float().cpu().numpy()
    if opacity_render.ndim != 3 or opacity_render.shape[0] < 1:
        raise ValueError(f"Unexpected opacity render shape: {opacity_render.shape}")
    opacity = np.asarray(opacity_render[0], dtype=np.float64)
    return {"depth": depth, "opacity": opacity}


def _camera_from_manifest_view(
    *,
    manifest_view: Dict[str, Any],
    camera_to_world: np.ndarray,
    image_root: Path,
    uid: int,
    data_device: str,
) -> Camera:
    image_name = str(manifest_view["image_name"])
    image_path = image_root / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image for manifest view not found: {image_path}")
    width = int(manifest_view["width"])
    height = int(manifest_view["height"])
    fx = float(manifest_view["fx"])
    fy = float(manifest_view["fy"])
    w2c = np.linalg.inv(np.asarray(camera_to_world, dtype=np.float64))
    rot = w2c[:3, :3].T
    trans = w2c[:3, 3]
    return Camera(
        resolution=(width, height),
        colmap_id=int(uid),
        R=rot,
        T=trans,
        FoVx=float(focal2fov(fx, width)),
        FoVy=float(focal2fov(fy, height)),
        depth_params=None,
        image=Image.open(image_path),
        invdepthmap=None,
        image_name=image_name,
        uid=int(uid),
        data_device=data_device,
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=True,
    )


def _infer_scene_name(dataset) -> str:
    src = Path(dataset.source_path)
    if src.name.lower() in {"thermal_ud", "rgb_ud", "thermal", "rgb", "images"} and src.parent.name:
        return src.parent.name
    if src.name:
        return src.name
    return Path(dataset.model_path).parent.name


def export_probe_bundle(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    out_dir: Path,
    split_label: str,
    max_views: int | None,
    scene_name_override: str,
    camera_frame_mode: str,
    probe_camera_manifest_path: Path | None,
    native_cameras_json_path: Path | None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        black_bg = torch.zeros(3, dtype=torch.float32, device="cuda")
        white_override = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")

        manifest_views: List[Dict[str, Any]] = []
        split_dir = out_dir / "views"
        split_dir.mkdir(parents=True, exist_ok=True)

        strict_to_native_alignment: Dict[str, Any] | None = None
        if camera_frame_mode == "scene_test":
            views = scene.getTestCameras()
            if max_views is not None:
                views = views[: int(max_views)]
            for idx, view in enumerate(views):
                arrays = _render_depth_and_opacity(
                    view=view,
                    gaussians=gaussians,
                    pipeline=pipeline,
                    black_bg=black_bg,
                    white_override=white_override,
                )
                view_rel = Path("views") / f"{idx:05d}.npz"
                view_path = out_dir / view_rel
                np.savez_compressed(
                    view_path,
                    depth=np.asarray(arrays["depth"], dtype=np.float64),
                    opacity=np.asarray(arrays["opacity"], dtype=np.float64),
                )
                manifest_views.append(
                    {
                        "view_id": f"{idx:05d}",
                        "image_name": str(view.image_name),
                        "width": int(view.image_width),
                        "height": int(view.image_height),
                        "fx": float(fov2focal(view.FoVx, view.image_width)),
                        "fy": float(fov2focal(view.FoVy, view.image_height)),
                        "cx": float(view.image_width / 2.0),
                        "cy": float(view.image_height / 2.0),
                        "camera_to_world": _camera_to_world_from_view(view),
                        "npz_file": str(view_rel).replace("\\", "/"),
                    }
                )
        elif camera_frame_mode == "probe_manifest_native_align":
            if probe_camera_manifest_path is None:
                raise ValueError("probe_camera_manifest_path is required when camera_frame_mode=probe_manifest_native_align")
            if native_cameras_json_path is None:
                raise ValueError("native_cameras_json_path is required when camera_frame_mode=probe_manifest_native_align")
            probe_manifest = load_json(probe_camera_manifest_path)
            probe_views = list(probe_manifest["views"])
            if max_views is not None:
                probe_views = probe_views[: int(max_views)]
            native_cameras_by_stem = load_native_camera_entries(native_cameras_json_path)
            strict_to_native_alignment = estimate_strict_to_native_world_transform(
                strict_views=probe_manifest["views"],
                native_cameras_by_stem=native_cameras_by_stem,
            )
            strict_to_native = np.asarray(strict_to_native_alignment["strict_to_native_transform"], dtype=np.float64)
            image_root = Path(dataset.source_path) / str(dataset.images)
            for idx, strict_view in enumerate(probe_views):
                native_c2w = apply_world_transform_to_camera_to_world(
                    np.asarray(strict_view["camera_to_world"], dtype=np.float64),
                    strict_to_native,
                )
                render_view = _camera_from_manifest_view(
                    manifest_view=strict_view,
                    camera_to_world=native_c2w,
                    image_root=image_root,
                    uid=idx,
                    data_device=dataset.data_device,
                )
                arrays = _render_depth_and_opacity(
                    view=render_view,
                    gaussians=gaussians,
                    pipeline=pipeline,
                    black_bg=black_bg,
                    white_override=white_override,
                )
                view_rel = Path("views") / f"{idx:05d}.npz"
                view_path = out_dir / view_rel
                np.savez_compressed(
                    view_path,
                    depth=np.asarray(arrays["depth"], dtype=np.float64),
                    opacity=np.asarray(arrays["opacity"], dtype=np.float64),
                )
                manifest_views.append(
                    {
                        "view_id": str(strict_view["view_id"]),
                        "image_name": str(strict_view["image_name"]),
                        "width": int(strict_view["width"]),
                        "height": int(strict_view["height"]),
                        "fx": float(strict_view["fx"]),
                        "fy": float(strict_view["fy"]),
                        "cx": float(strict_view["cx"]),
                        "cy": float(strict_view["cy"]),
                        "camera_to_world": strict_view["camera_to_world"],
                        "native_camera_to_world": native_c2w.tolist(),
                        "npz_file": str(view_rel).replace("\\", "/"),
                    }
                )
        else:
            raise ValueError(f"Unsupported camera_frame_mode: {camera_frame_mode!r}")

    split_manifest = {
        "bundle_type": "gaussian_probe_split_bundle_v1",
        "scene_name": scene_name_override if scene_name_override else _infer_scene_name(dataset),
        "split_label": split_label,
        "model_path": str(Path(dataset.model_path).resolve()),
        "source_path": str(Path(dataset.source_path).resolve()),
        "iteration": int(scene.loaded_iter),
        # The rasterizer returns inverse depth rather than metric camera-z depth.
        "depth_semantics": "inverse_camera_z_from_renderer",
        "opacity_semantics": "black_bg_plus_white_override_color_render",
        "camera_frame_mode": str(camera_frame_mode),
        "render_resolution": {
            "resolution_arg": int(dataset.resolution),
        },
        "probe_camera_manifest": str(probe_camera_manifest_path.resolve()) if probe_camera_manifest_path is not None else "",
        "native_cameras_json": str(native_cameras_json_path.resolve()) if native_cameras_json_path is not None else "",
        "strict_to_native_alignment": strict_to_native_alignment,
        "views": manifest_views,
    }
    split_manifest_path = out_dir / "split_manifest.json"
    _save_json(split_manifest_path, split_manifest)
    return split_manifest_path


def build_argparser() -> tuple[argparse.ArgumentParser, ModelParams, PipelineParams]:
    parser = argparse.ArgumentParser(description="Export probe-view depth/opacity bundle from a Gaussian model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split_label", required=True)
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--scene_name_override", default="")
    parser.add_argument(
        "--camera_frame_mode",
        default="scene_test",
        choices=["scene_test", "probe_manifest_native_align"],
        help="scene_test preserves baseline behavior; probe_manifest_native_align uses a strict probe manifest and aligns it into the model's native frame before rendering.",
    )
    parser.add_argument("--probe_camera_manifest", default="")
    parser.add_argument("--native_cameras_json", default="")
    parser.add_argument("--quiet", action="store_true")
    return parser, model, pipeline


def main() -> None:
    parser, model_params, pipeline_params = build_argparser()
    args = get_combined_args(parser)
    if not hasattr(args, "max_views"):
        args.max_views = None
    safe_state(args.quiet)
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    manifest_path = export_probe_bundle(
        dataset=dataset,
        iteration=int(args.iteration),
        pipeline=pipeline,
        out_dir=Path(args.out_dir).resolve(),
        split_label=str(args.split_label),
        max_views=args.max_views,
        scene_name_override=str(args.scene_name_override),
        camera_frame_mode=str(args.camera_frame_mode),
        probe_camera_manifest_path=Path(args.probe_camera_manifest).resolve() if str(args.probe_camera_manifest).strip() else None,
        native_cameras_json_path=Path(args.native_cameras_json).resolve() if str(args.native_cameras_json).strip() else None,
    )
    print(f"PROBE_BUNDLE_SAVED {manifest_path}")


if __name__ == "__main__":
    main()
