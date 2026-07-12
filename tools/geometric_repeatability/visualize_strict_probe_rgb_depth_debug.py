from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from visualize_depth_reference_views import (
    _add_depth_colorbar,
    _depth_to_rgb,
    _load_gt_image,
    _load_json,
    _load_npz,
    _make_model_valid_mask,
    _raw_depth_to_metric_camera_z,
    _resolve_gt_root,
    _resize_rgb_image,
    _save_json,
    _select_views,
)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Debug strict-probe RGB/depth alignment for selected held-out views.")
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--bundle_manifest", required=True)
    ap.add_argument("--adapter_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--image_names", nargs="*", default=[])
    ap.add_argument("--view_ids", nargs="*", default=[])
    ap.add_argument("--first_n", type=int, default=0)
    ap.add_argument("--gt_images_root", default="")
    ap.add_argument("--gt_images_dir_name", default="")
    ap.add_argument("--depth_percentile_low", type=float, default=2.0)
    ap.add_argument("--depth_percentile_high", type=float, default=98.0)
    ap.add_argument("--depth_cmap", default="turbo_r")
    ap.add_argument("--invalid_depth_rgb", default="0.94,0.94,0.94")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--quiet", action="store_true")
    return ap


def _parse_rgb_triplet(text: str) -> np.ndarray:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B triplet, got: {text!r}")
    rgb = np.asarray([float(part) for part in parts], dtype=np.float64)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError(f"RGB triplet values must lie in [0, 1], got: {text!r}")
    return rgb


def _camera_to_world_from_view(view) -> np.ndarray:
    w2c = getWorld2View2(view.R, view.T, view.trans, view.scale).astype(np.float64)
    return np.linalg.inv(w2c)


def _rotation_delta_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = rot_a @ rot_b.T
    cos_angle = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) / 2.0))
    return float(math.degrees(math.acos(cos_angle)))


def _build_scene_from_bundle(bundle_manifest: Dict[str, Any], quiet: bool):
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "strict_probe_debug",
            "--model_path",
            str(bundle_manifest["model_path"]),
            "--source_path",
            str(bundle_manifest["source_path"]),
            "--images",
            "images",
            "--resolution",
            str(bundle_manifest["render_resolution"]["resolution_arg"]),
            "--eval",
        ]
        args = get_combined_args(parser)
    finally:
        sys.argv = argv_backup

    safe_state(bool(quiet))
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=int(bundle_manifest["iteration"]), shuffle=False)
    return dataset, pipeline, gaussians, scene


def _render_rgb_and_depth(view, gaussians, pipeline) -> tuple[np.ndarray, np.ndarray]:
    black_bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        out = render(
            view,
            gaussians,
            pipeline,
            black_bg,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=None,
            use_trained_exp=False,
        )
    rgb = out["render"].detach().float().cpu().numpy()
    depth = out["depth"].detach().float().cpu().numpy()
    if rgb.ndim != 3 or rgb.shape[0] < 3:
        raise ValueError(f"Unexpected RGB render shape: {rgb.shape}")
    rgb = np.moveaxis(rgb[:3], 0, -1)
    rgb = np.clip(rgb, 0.0, 1.0)
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    depth = np.asarray(depth, dtype=np.float64)
    return rgb, depth


def _native_camera_lookup(model_path: Path, image_name: str) -> Dict[str, Any] | None:
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        return None
    cameras = _load_json(cameras_json)
    for item in cameras:
        if str(item.get("img_name", "")) == str(image_name):
            return item
    return None


def _native_render_path(model_path: Path, iteration: int, camera_entry: Dict[str, Any] | None) -> Path | None:
    if camera_entry is None:
        return None
    render_path = model_path / "test" / f"ours_{int(iteration)}" / "renders" / f"{int(camera_entry['id']):05d}.png"
    return render_path if render_path.exists() else None


def _compute_depth_display_range(reference_manifest_path: Path, selected_views: List[Dict[str, Any]]) -> tuple[float, float]:
    ref_root = reference_manifest_path.parent
    values: List[np.ndarray] = []
    for view in selected_views:
        ref_npz = _load_npz(ref_root, str(view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        valid_values = ref_depth[ref_valid]
        if valid_values.size:
            values.append(valid_values)
    if not values:
        return 0.0, 1.0
    concat = np.concatenate(values, axis=0)
    depth_min = float(np.percentile(concat, 2.0))
    depth_max = float(np.percentile(concat, 98.0))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        depth_min = float(np.min(concat))
        depth_max = float(np.max(concat))
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    return depth_min, depth_max


def _render_panel(
    out_path: Path,
    image_name: str,
    gt_rgb: np.ndarray,
    ref_depth_rgb: np.ndarray,
    strict_rgb: np.ndarray,
    strict_depth_rgb: np.ndarray,
    bundle_depth_rgb: np.ndarray,
    native_rgb: np.ndarray | None,
    dpi: int,
    depth_min: float,
    depth_max: float,
    cmap_name: str,
) -> None:
    titles = ["Strict Probe GT", "Strict Probe\nReference Depth", "Strict Probe\nRGB Render", "Strict Probe\nDepth", "Bundle\nDepth"]
    images = [gt_rgb, ref_depth_rgb, strict_rgb, strict_depth_rgb, bundle_depth_rgb]
    if native_rgb is not None:
        titles.append("Native Test\nRGB Render")
        images.append(native_rgb)
    ncols = len(images)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.8), squeeze=False)
    for idx, (title, img) in enumerate(zip(titles, images)):
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(f"{title}\n{image_name}")
        axes[0, idx].axis("off")
    fig.tight_layout(rect=(0.0, 0.0, 0.92, 0.94))
    _add_depth_colorbar(
        fig=fig,
        depth_min=depth_min,
        depth_max=depth_max,
        cmap_name=cmap_name,
        cbar_rect=(0.93, 0.18, 0.015, 0.64),
        label="Depth (m)",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_argparser().parse_args()
    reference_manifest_path = Path(args.reference_manifest).resolve()
    bundle_manifest_path = Path(args.bundle_manifest).resolve()
    adapter_manifest_path = Path(args.adapter_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = _load_json(reference_manifest_path)
    bundle_manifest = _load_json(bundle_manifest_path)
    adapter_manifest = _load_json(adapter_manifest_path)

    selected_views = _select_views(
        reference_manifest=reference_manifest,
        image_names=args.image_names,
        view_ids=args.view_ids,
        first_n=int(args.first_n),
    )
    gt_root = _resolve_gt_root(reference_manifest, override_root=str(args.gt_images_root), override_images_dir=str(args.gt_images_dir_name))
    invalid_rgb = _parse_rgb_triplet(str(args.invalid_depth_rgb))
    depth_min_vis, depth_max_vis = _compute_depth_display_range(reference_manifest_path, selected_views)

    _, pipeline, gaussians, scene = _build_scene_from_bundle(bundle_manifest, quiet=bool(args.quiet))
    scene_views = {str(v.image_name): v for v in scene.getTestCameras()}
    bundle_views_by_name = {str(v["image_name"]): v for v in bundle_manifest["views"]}
    ref_views_by_name = {str(v["image_name"]): v for v in reference_manifest["views"]}

    summary_views: List[Dict[str, Any]] = []
    model_path = Path(bundle_manifest["model_path"]).resolve()
    iteration = int(bundle_manifest["iteration"])
    depth_semantics = str(adapter_manifest["depth_semantics"])
    validity_rule = adapter_manifest["validity_rule"]

    for ref_view in selected_views:
        image_name = str(ref_view["image_name"])
        strict_view = scene_views[image_name]
        bundle_view = bundle_views_by_name[image_name]
        ref_bundle_view = ref_views_by_name[image_name]

        strict_rgb, strict_depth_raw = _render_rgb_and_depth(strict_view, gaussians, pipeline)
        strict_c2w = _camera_to_world_from_view(strict_view)
        bundle_c2w = np.asarray(bundle_view["camera_to_world"], dtype=np.float64)
        c2w_delta = float(np.max(np.abs(strict_c2w - bundle_c2w)))

        ref_npz = _load_npz(reference_manifest_path.parent, str(ref_bundle_view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        ref_depth_rgb = _depth_to_rgb(
            ref_depth,
            ref_valid,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_rgb,
        )

        bundle_npz = _load_npz(bundle_manifest_path.parent, str(bundle_view["npz_file"]))
        bundle_depth = _raw_depth_to_metric_camera_z(np.asarray(bundle_npz["depth"], dtype=np.float64), depth_semantics=depth_semantics)
        bundle_opacity = np.asarray(bundle_npz["opacity"], dtype=np.float64)
        bundle_valid = _make_model_valid_mask(
            metric_depth=bundle_depth,
            opacity=bundle_opacity,
            depth_min=float(validity_rule.get("depth_min", 1e-6)),
            opacity_threshold=float(validity_rule.get("opacity_threshold", 0.5)),
        )
        bundle_depth_rgb = _depth_to_rgb(
            bundle_depth,
            bundle_valid,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_rgb,
        )

        strict_depth_metric = _raw_depth_to_metric_camera_z(strict_depth_raw, depth_semantics=depth_semantics)
        strict_depth_valid = np.isfinite(strict_depth_metric) & (strict_depth_metric > float(validity_rule.get("depth_min", 1e-6)))
        strict_depth_rgb = _depth_to_rgb(
            strict_depth_metric,
            strict_depth_valid,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_rgb,
        )

        gt_rgb = _resize_rgb_image(
            _load_gt_image(gt_root / image_name),
            width=int(ref_view["width"]),
            height=int(ref_view["height"]),
        )

        native_camera = _native_camera_lookup(model_path, image_name)
        native_rgb = None
        native_render = _native_render_path(model_path, iteration, native_camera)
        native_translation_delta = None
        native_rotation_delta = None
        if native_render is not None and native_camera is not None:
            native_rgb = _load_gt_image(native_render)
            native_rgb = _resize_rgb_image(native_rgb, width=int(ref_view["width"]), height=int(ref_view["height"]))
            native_rot = np.asarray(native_camera["rotation"], dtype=np.float64)
            native_pos = np.asarray(native_camera["position"], dtype=np.float64)
            native_translation_delta = float(np.linalg.norm(native_pos - bundle_c2w[:3, 3]))
            native_rotation_delta = _rotation_delta_deg(native_rot, bundle_c2w[:3, :3])

        panel_path = out_dir / f"{str(ref_view['view_id'])}_{image_name}__strict_probe_debug.png"
        _render_panel(
            out_path=panel_path,
            image_name=image_name,
            gt_rgb=gt_rgb,
            ref_depth_rgb=ref_depth_rgb,
            strict_rgb=strict_rgb,
            strict_depth_rgb=strict_depth_rgb,
            bundle_depth_rgb=bundle_depth_rgb,
            native_rgb=native_rgb,
            dpi=int(args.dpi),
            depth_min=depth_min_vis,
            depth_max=depth_max_vis,
            cmap_name=str(args.depth_cmap),
        )

        summary_views.append(
            {
                "view_id": str(ref_view["view_id"]),
                "image_name": image_name,
                "panel_png": str(panel_path),
                "strict_probe_camera_to_world_max_abs_delta": c2w_delta,
                "native_render_png": str(native_render) if native_render is not None else "",
                "native_camera_translation_delta_m": native_translation_delta,
                "native_camera_rotation_delta_deg": native_rotation_delta,
            }
        )

    _save_json(
        out_dir / "manifest.json",
        {
            "reference_manifest": str(reference_manifest_path),
            "bundle_manifest": str(bundle_manifest_path),
            "adapter_manifest": str(adapter_manifest_path),
            "gt_root": str(gt_root),
            "depth_cmap": str(args.depth_cmap),
            "depth_display_range_m": {"min": float(depth_min_vis), "max": float(depth_max_vis)},
            "views": summary_views,
        },
    )
    print(f"STRICT_PROBE_DEBUG_READY {out_dir}")


if __name__ == "__main__":
    main()
