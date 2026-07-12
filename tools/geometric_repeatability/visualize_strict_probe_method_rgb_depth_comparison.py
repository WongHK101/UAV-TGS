from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
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
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
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
    ap = argparse.ArgumentParser(description="Visualize strict-probe RGB renders and evaluated depth maps across multiple methods.")
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene_name", default="")
    ap.add_argument("--gt_images_root", default="")
    ap.add_argument("--gt_images_dir_name", default="")
    ap.add_argument("--image_names", nargs="*", default=[])
    ap.add_argument("--view_ids", nargs="*", default=[])
    ap.add_argument("--first_n", type=int, default=0)
    ap.add_argument("--random_n", type=int, default=0)
    ap.add_argument("--random_seed", type=int, default=20260501)
    ap.add_argument("--depth_percentile_low", type=float, default=2.0)
    ap.add_argument("--depth_percentile_high", type=float, default=98.0)
    ap.add_argument("--depth_cmap", default="turbo_r")
    ap.add_argument("--invalid_depth_rgb", default="0.94,0.94,0.94")
    ap.add_argument("--show_depth_colorbar", action="store_true")
    ap.add_argument(
        "--include_unmasked_reference_depth",
        action="store_true",
        help="Also show the raw finite-only reference depth without applying the evaluation valid mask; default OFF for backward compatibility.",
    )
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec in label=method_dir or label|bundle_manifest|adapter_manifest format.",
    )
    return ap


def _parse_rgb_triplet(text: str) -> np.ndarray:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B triplet, got: {text!r}")
    rgb = np.asarray([float(part) for part in parts], dtype=np.float64)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError(f"RGB triplet values must lie in [0, 1], got: {text!r}")
    return rgb


def _parse_method_specs(method_specs: Sequence[str]) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for spec in method_specs:
        if "|" in spec:
            parts = [part.strip() for part in spec.split("|")]
            if len(parts) != 3:
                raise ValueError(f"Expected label|bundle_manifest|adapter_manifest format, got: {spec}")
            label, bundle_text, adapter_text = parts
            bundle_manifest = Path(bundle_text).resolve()
            adapter_manifest = Path(adapter_text).resolve()
            method_dir = bundle_manifest.parent.parent
        else:
            if "=" not in spec:
                raise ValueError(f"Expected label=method_dir or label|bundle_manifest|adapter_manifest format, got: {spec}")
            label, dir_text = spec.split("=", 1)
            method_dir = Path(dir_text.strip()).resolve()
            bundle_manifest = method_dir / "bundle" / "split_manifest.json"
            adapter_manifest = method_dir / "depth_adapter_manifest.json"
        if not bundle_manifest.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest}")
        if not adapter_manifest.exists():
            raise FileNotFoundError(f"Adapter manifest not found: {adapter_manifest}")
        parsed.append(
            {
                "label": label.strip(),
                "method_dir": method_dir,
                "bundle_manifest_path": bundle_manifest,
                "adapter_manifest_path": adapter_manifest,
            }
        )
    return parsed


def _select_random_views(reference_manifest: Dict[str, Any], random_n: int, random_seed: int) -> List[Dict[str, Any]]:
    views = list(reference_manifest["views"])
    if random_n <= 0 or random_n >= len(views):
        return views
    rng = random.Random(int(random_seed))
    selected = rng.sample(views, int(random_n))
    return sorted(selected, key=lambda item: str(item["view_id"]))


def _choose_views(reference_manifest: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.image_names or args.view_ids or int(args.first_n) > 0:
        return _select_views(
            reference_manifest=reference_manifest,
            image_names=args.image_names,
            view_ids=args.view_ids,
            first_n=int(args.first_n),
        )
    if int(args.random_n) > 0:
        return _select_random_views(reference_manifest=reference_manifest, random_n=int(args.random_n), random_seed=int(args.random_seed))
    return _select_views(reference_manifest=reference_manifest, image_names=[], view_ids=[], first_n=10)


def _compute_depth_display_range(reference_manifest_path: Path, selected_views: Sequence[Dict[str, Any]], low_pct: float, high_pct: float) -> tuple[float, float]:
    valid_depths: List[np.ndarray] = []
    ref_root = reference_manifest_path.parent
    for view in selected_views:
        ref_npz = _load_npz(ref_root, str(view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        values = ref_depth[ref_valid]
        if values.size > 0:
            valid_depths.append(values)
    if not valid_depths:
        return 0.0, 1.0
    concat = np.concatenate(valid_depths, axis=0)
    depth_min = float(np.percentile(concat, low_pct))
    depth_max = float(np.percentile(concat, high_pct))
    if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_max <= depth_min:
        depth_min = float(np.min(concat))
        depth_max = float(np.max(concat))
    if depth_max <= depth_min:
        depth_max = depth_min + 1.0
    return depth_min, depth_max


def _build_scene_from_bundle(bundle_manifest: Dict[str, Any], quiet: bool):
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "strict_probe_method_rgb_depth",
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


def _render_rgb(view, gaussians, pipeline) -> np.ndarray:
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
    if rgb.ndim != 3 or rgb.shape[0] < 3:
        raise ValueError(f"Unexpected RGB render shape: {rgb.shape}")
    rgb = np.moveaxis(rgb[:3], 0, -1)
    return np.clip(rgb, 0.0, 1.0)


def _camera_from_bundle_view(
    *,
    bundle_view: Dict[str, Any],
    image_root: Path,
    uid: int,
    data_device: str,
) -> Camera:
    camera_to_world = np.asarray(bundle_view.get("native_camera_to_world", bundle_view["camera_to_world"]), dtype=np.float64)
    image_name = str(bundle_view["image_name"])
    image_path = image_root / image_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image for bundle view not found: {image_path}")
    width = int(bundle_view["width"])
    height = int(bundle_view["height"])
    fx = float(bundle_view["fx"])
    fy = float(bundle_view["fy"])
    w2c = np.linalg.inv(camera_to_world)
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


def _render_per_view_panel(
    out_path: Path,
    gt_rgb: np.ndarray,
    ref_depth_rgb: np.ndarray,
    ref_depth_unmasked_rgb: np.ndarray | None,
    method_panels: Sequence[Dict[str, Any]],
    image_name: str,
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
    show_depth_colorbar: bool,
    dpi: int,
) -> None:
    base_cols = 3 if ref_depth_unmasked_rgb is not None else 2
    ncols = base_cols + 2 * len(method_panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 3.8), squeeze=False)
    axes_row = axes[0]
    axes_row[0].imshow(gt_rgb)
    axes_row[0].set_title(f"Strict Probe GT\n{image_name}")
    axes_row[0].axis("off")
    axes_row[1].imshow(ref_depth_rgb)
    axes_row[1].set_title("Strict Probe\nReference Depth")
    axes_row[1].axis("off")
    col_idx = 2
    if ref_depth_unmasked_rgb is not None:
        axes_row[2].imshow(ref_depth_unmasked_rgb)
        axes_row[2].set_title("Reference Depth\n(No Valid Mask)")
        axes_row[2].axis("off")
        col_idx = 3
    for panel in method_panels:
        axes_row[col_idx].imshow(panel["rgb_render"])
        axes_row[col_idx].set_title(f"{panel['label']}\nStrict Probe RGB")
        axes_row[col_idx].axis("off")
        axes_row[col_idx + 1].imshow(panel["depth_rgb"])
        axes_row[col_idx + 1].set_title(f"{panel['label']}\nEvaluated Depth")
        axes_row[col_idx + 1].axis("off")
        col_idx += 2
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.0, right_margin, 0.95))
    if show_depth_colorbar:
        _add_depth_colorbar(
            fig=fig,
            depth_min=depth_min_vis,
            depth_max=depth_max_vis,
            cmap_name=cmap_name,
            cbar_rect=(0.93, 0.18, 0.015, 0.64),
            label="Depth (m)",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(
    out_path: Path,
    scene_name: str,
    rows_payload: Sequence[Dict[str, Any]],
    method_labels: Sequence[str],
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
    show_depth_colorbar: bool,
    dpi: int,
) -> None:
    nrows = len(rows_payload)
    include_unmasked_reference = rows_payload and rows_payload[0].get("ref_depth_unmasked_rgb") is not None
    base_cols = 3 if include_unmasked_reference else 2
    ncols = base_cols + 2 * len(method_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)
    for row_idx, payload in enumerate(rows_payload):
        axes[row_idx, 0].imshow(payload["gt_rgb"])
        axes[row_idx, 0].set_title(f"Strict Probe GT\n{payload['image_name']}")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(payload["ref_depth_rgb"])
        axes[row_idx, 1].set_title("Strict Probe\nReference Depth")
        axes[row_idx, 1].axis("off")
        col_idx = 2
        if include_unmasked_reference:
            axes[row_idx, 2].imshow(payload["ref_depth_unmasked_rgb"])
            axes[row_idx, 2].set_title("Reference Depth\n(No Valid Mask)")
            axes[row_idx, 2].axis("off")
            col_idx = 3
        for method_label in method_labels:
            panel = payload["method_panels"][method_label]
            axes[row_idx, col_idx].imshow(panel["rgb_render"])
            axes[row_idx, col_idx].set_title(f"{method_label}\nStrict Probe RGB")
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx + 1].imshow(panel["depth_rgb"])
            axes[row_idx, col_idx + 1].set_title(f"{method_label}\nEvaluated Depth")
            axes[row_idx, col_idx + 1].axis("off")
            col_idx += 2
    fig.suptitle(f"{scene_name}: Strict-Probe RGB + Evaluated Depth")
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.0, right_margin, 0.97))
    if show_depth_colorbar:
        _add_depth_colorbar(
            fig=fig,
            depth_min=depth_min_vis,
            depth_max=depth_max_vis,
            cmap_name=cmap_name,
            cbar_rect=(0.93, 0.12, 0.015, 0.76),
            label="Depth (m)",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_argparser().parse_args()
    reference_manifest_path = Path(args.reference_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = _load_json(reference_manifest_path)
    method_specs = _parse_method_specs(args.method)
    scene_name = str(args.scene_name) if str(args.scene_name) else str(reference_manifest.get("scene_name", "Scene"))
    gt_root = _resolve_gt_root(reference_manifest, override_root=str(args.gt_images_root), override_images_dir=str(args.gt_images_dir_name))
    selected_views = _choose_views(reference_manifest=reference_manifest, args=args)
    invalid_depth_rgb = _parse_rgb_triplet(str(args.invalid_depth_rgb))
    depth_min_vis, depth_max_vis = _compute_depth_display_range(
        reference_manifest_path=reference_manifest_path,
        selected_views=selected_views,
        low_pct=float(args.depth_percentile_low),
        high_pct=float(args.depth_percentile_high),
    )

    ref_by_name = {str(v["image_name"]): v for v in reference_manifest["views"]}
    methods_payload: List[Dict[str, Any]] = []
    for spec in method_specs:
        bundle_manifest = _load_json(spec["bundle_manifest_path"])
        adapter_manifest = _load_json(spec["adapter_manifest_path"])
        dataset, pipeline, gaussians, scene = _build_scene_from_bundle(bundle_manifest, quiet=bool(args.quiet))
        methods_payload.append(
            {
                **spec,
                "bundle_manifest": bundle_manifest,
                "adapter_manifest": adapter_manifest,
                "views_by_name": {str(v["image_name"]): v for v in bundle_manifest["views"]},
                "scene_views_by_name": {str(v.image_name): v for v in scene.getTestCameras()},
                "scene_image_root": Path(bundle_manifest["source_path"]) / "images",
                "data_device": str(dataset.data_device),
                "pipeline": pipeline,
                "gaussians": gaussians,
            }
        )

    rows_payload: List[Dict[str, Any]] = []
    manifest_payload = {
        "scene_name": scene_name,
        "reference_manifest": str(reference_manifest_path),
        "gt_root": str(gt_root),
        "selected_views": [],
        "methods": [],
    }

    for view in selected_views:
        image_name = str(view["image_name"])
        ref_view = ref_by_name[image_name]
        ref_npz = _load_npz(reference_manifest_path.parent, str(ref_view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        gt_rgb = _resize_rgb_image(
            _load_gt_image(gt_root / image_name),
            width=int(ref_view["width"]),
            height=int(ref_view["height"]),
        )
        ref_depth_rgb = _depth_to_rgb(
            ref_depth,
            ref_valid,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_depth_rgb,
        )
        ref_depth_unmasked_rgb = None
        if bool(args.include_unmasked_reference_depth):
            ref_depth_unmasked_rgb = _depth_to_rgb(
                ref_depth,
                np.isfinite(ref_depth) & (ref_depth > 0.0),
                depth_min_vis,
                depth_max_vis,
                cmap_name=str(args.depth_cmap),
                invalid_rgb=invalid_depth_rgb,
            )

        method_panels: Dict[str, Dict[str, Any]] = {}
        per_method_manifest: List[Dict[str, Any]] = []
        ordered_panels: List[Dict[str, Any]] = []
        for payload in methods_payload:
            bundle_view = payload["views_by_name"][image_name]
            bundle_npz = _load_npz(payload["bundle_manifest_path"].parent, str(bundle_view["npz_file"]))
            model_depth = _raw_depth_to_metric_camera_z(
                np.asarray(bundle_npz["depth"], dtype=np.float64),
                depth_semantics=str(payload["adapter_manifest"]["depth_semantics"]),
            )
            model_opacity = np.asarray(bundle_npz["opacity"], dtype=np.float64)
            validity_rule = payload["adapter_manifest"]["validity_rule"]
            model_valid = _make_model_valid_mask(
                metric_depth=model_depth,
                opacity=model_opacity,
                depth_min=float(validity_rule.get("depth_min", 1e-6)),
                opacity_threshold=float(validity_rule.get("opacity_threshold", 0.5)),
            )
            model_depth_rgb = _depth_to_rgb(
                model_depth,
                model_valid,
                depth_min_vis,
                depth_max_vis,
                cmap_name=str(args.depth_cmap),
                invalid_rgb=invalid_depth_rgb,
            )
            if payload["bundle_manifest"].get("camera_frame_mode") == "probe_manifest_native_align" and "native_camera_to_world" in bundle_view:
                render_view = _camera_from_bundle_view(
                    bundle_view=bundle_view,
                    image_root=payload["scene_image_root"],
                    uid=int(bundle_view["view_id"]),
                    data_device=str(payload["data_device"]),
                )
            else:
                render_view = payload["scene_views_by_name"][image_name]
            strict_rgb = _render_rgb(render_view, payload["gaussians"], payload["pipeline"])
            method_panels[payload["label"]] = {
                "rgb_render": strict_rgb,
                "depth_rgb": model_depth_rgb,
            }
            ordered_panels.append(
                {
                    "label": payload["label"],
                    "rgb_render": strict_rgb,
                    "depth_rgb": model_depth_rgb,
                }
            )
            per_method_manifest.append(
                {
                    "label": payload["label"],
                    "bundle_manifest": str(payload["bundle_manifest_path"]),
                    "adapter_manifest": str(payload["adapter_manifest_path"]),
                }
            )

        panel_path = out_dir / "per_view_panels" / f"{str(view['view_id'])}_{image_name}__strict_probe_rgb_depth.png"
        _render_per_view_panel(
            out_path=panel_path,
            gt_rgb=gt_rgb,
            ref_depth_rgb=ref_depth_rgb,
            ref_depth_unmasked_rgb=ref_depth_unmasked_rgb,
            method_panels=ordered_panels,
            image_name=image_name,
            depth_min_vis=depth_min_vis,
            depth_max_vis=depth_max_vis,
            cmap_name=str(args.depth_cmap),
            show_depth_colorbar=bool(args.show_depth_colorbar),
            dpi=int(args.dpi),
        )

        rows_payload.append(
            {
                "image_name": image_name,
                "gt_rgb": gt_rgb,
                "ref_depth_rgb": ref_depth_rgb,
                "ref_depth_unmasked_rgb": ref_depth_unmasked_rgb,
                "method_panels": method_panels,
            }
        )
        manifest_payload["selected_views"].append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "per_view_panel_png": str(panel_path),
                "methods": per_method_manifest,
            }
        )

    contact_sheet_path = out_dir / f"{scene_name}_strict_probe_rgb_depth_contact_sheet.png"
    _render_contact_sheet(
        out_path=contact_sheet_path,
        scene_name=scene_name,
        rows_payload=rows_payload,
        method_labels=[payload["label"] for payload in methods_payload],
        depth_min_vis=depth_min_vis,
        depth_max_vis=depth_max_vis,
        cmap_name=str(args.depth_cmap),
        show_depth_colorbar=bool(args.show_depth_colorbar),
        dpi=int(args.dpi),
    )

    manifest_payload["contact_sheet_png"] = str(contact_sheet_path)
    manifest_payload["depth_display_range_m"] = {
        "min": float(depth_min_vis),
        "max": float(depth_max_vis),
    }
    manifest_payload["include_unmasked_reference_depth"] = bool(args.include_unmasked_reference_depth)
    _save_json(out_dir / "manifest.json", manifest_payload)
    print(f"STRICT_PROBE_METHOD_RGB_DEPTH_READY {out_dir}")


if __name__ == "__main__":
    main()
