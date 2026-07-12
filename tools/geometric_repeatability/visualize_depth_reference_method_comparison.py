from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from visualize_depth_reference_views import (
    _add_depth_colorbar,
    _classification_legend_handles,
    _classification_rgb,
    _compute_depth_display_range,
    _depth_to_rgb,
    _load_gt_image,
    _load_json,
    _load_npz,
    _make_model_valid_mask,
    _parse_rgb_triplet,
    _raw_depth_to_metric_camera_z,
    _resolve_gt_root,
    _resize_rgb_image,
    _sanitize_stem,
    _save_json,
    _select_views,
    _write_csv,
)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Create held-out view comparison sheets across multiple methods.")
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold_m", type=float, required=True)
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
    ap.add_argument("--depth_cmap", default="viridis")
    ap.add_argument("--invalid_depth_rgb", default="0.0,0.0,0.0")
    ap.add_argument("--show_depth_colorbar", action="store_true")
    ap.add_argument("--resize_gt_to_view", action="store_true")
    ap.add_argument("--gt_title", default="GT")
    ap.add_argument("--reference_title", default="Reference\nMesh Depth")
    ap.add_argument("--method_panel_layout", choices=["error_only", "depth_and_error"], default="error_only")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method spec in label=method_dir format, where method_dir contains bundle/split_manifest.json and depth_adapter_manifest.json.",
    )
    return ap


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
            if not bundle_manifest.exists():
                raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest}")
            if not adapter_manifest.exists():
                raise FileNotFoundError(f"Adapter manifest not found: {adapter_manifest}")
            parsed.append(
                {
                    "label": label,
                    "method_dir": bundle_manifest.parent.parent,
                    "bundle_manifest_path": bundle_manifest,
                    "adapter_manifest_path": adapter_manifest,
                }
            )
            continue
        if "=" not in spec:
            raise ValueError(f"Expected label=method_dir or label|bundle_manifest|adapter_manifest format, got: {spec}")
        label, dir_text = spec.split("=", 1)
        label = label.strip()
        method_dir = Path(dir_text.strip()).resolve()
        bundle_manifest = method_dir / "bundle" / "split_manifest.json"
        adapter_manifest = method_dir / "depth_adapter_manifest.json"
        if not bundle_manifest.exists():
            raise FileNotFoundError(f"Bundle manifest not found: {bundle_manifest}")
        if not adapter_manifest.exists():
            raise FileNotFoundError(f"Adapter manifest not found: {adapter_manifest}")
        parsed.append(
            {
                "label": label,
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


def _render_per_view_comparison(
    out_path: Path,
    gt_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    method_panels: Sequence[Dict[str, Any]],
    image_name: str,
    threshold_m: float,
    dpi: int,
    method_panel_layout: str,
    show_depth_colorbar: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
    gt_title: str,
    reference_title: str,
) -> None:
    ncols = 2 + len(method_panels) if method_panel_layout == "error_only" else 2 + 2 * len(method_panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.15 * ncols, 3.6), squeeze=False)
    axes_row = axes[0]
    axes_row[0].imshow(gt_rgb)
    axes_row[0].set_title(f"{gt_title}\n{image_name}")
    axes_row[0].axis("off")

    axes_row[1].imshow(depth_rgb)
    axes_row[1].set_title(reference_title)
    axes_row[1].axis("off")

    if method_panel_layout == "error_only":
        for idx, panel in enumerate(method_panels, start=2):
            axes_row[idx].imshow(panel["class_rgb"])
            axes_row[idx].set_title(panel["label"])
            axes_row[idx].axis("off")
    else:
        col_idx = 2
        for panel in method_panels:
            axes_row[col_idx].imshow(panel["depth_rgb"])
            axes_row[col_idx].set_title(f"{panel['label']}\nDepth")
            axes_row[col_idx].axis("off")
            axes_row[col_idx + 1].imshow(panel["class_rgb"])
            axes_row[col_idx + 1].set_title(f"{panel['label']}\nError @ {threshold_m:.2f} m")
            axes_row[col_idx + 1].axis("off")
            col_idx += 2

    fig.legend(handles=_classification_legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{image_name} | Threshold = {threshold_m:.2f} m")
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.08, right_margin, 0.92))
    if show_depth_colorbar:
        _add_depth_colorbar(
            fig=fig,
            depth_min=depth_min_vis,
            depth_max=depth_max_vis,
            cmap_name=cmap_name,
            cbar_rect=(0.93, 0.16, 0.015, 0.66),
            label="Depth (m)",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_scene_contact_sheet(
    out_path: Path,
    scene_name: str,
    threshold_m: float,
    rows_payload: Sequence[Dict[str, Any]],
    method_labels: Sequence[str],
    dpi: int,
    method_panel_layout: str,
    show_depth_colorbar: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
    gt_title: str,
    reference_title: str,
) -> None:
    nrows = len(rows_payload)
    ncols = 2 + len(method_labels) if method_panel_layout == "error_only" else 2 + 2 * len(method_labels)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows), squeeze=False)

    for row_idx, payload in enumerate(rows_payload):
        axes[row_idx, 0].imshow(payload["gt_rgb"])
        axes[row_idx, 0].set_title(f"{gt_title}\n{payload['image_name']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(payload["depth_rgb"])
        axes[row_idx, 1].set_title(reference_title)
        axes[row_idx, 1].axis("off")

        if method_panel_layout == "error_only":
            for col_idx, method_label in enumerate(method_labels, start=2):
                axes[row_idx, col_idx].imshow(payload["method_panels"][method_label]["class_rgb"])
                axes[row_idx, col_idx].set_title(method_label)
                axes[row_idx, col_idx].axis("off")
        else:
            col_idx = 2
            for method_label in method_labels:
                axes[row_idx, col_idx].imshow(payload["method_panels"][method_label]["depth_rgb"])
                axes[row_idx, col_idx].set_title(f"{method_label}\nDepth")
                axes[row_idx, col_idx].axis("off")
                axes[row_idx, col_idx + 1].imshow(payload["method_panels"][method_label]["class_rgb"])
                axes[row_idx, col_idx + 1].set_title(f"{method_label}\nError")
                axes[row_idx, col_idx + 1].axis("off")
                col_idx += 2

    fig.legend(handles=_classification_legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{scene_name}: Held-out View Comparison @ {threshold_m:.2f} m")
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.05, right_margin, 0.96))
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
    invalid_depth_rgb = _parse_rgb_triplet(str(args.invalid_depth_rgb), default=(0.0, 0.0, 0.0))
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
        methods_payload.append(
            {
                **spec,
                "bundle_manifest": bundle_manifest,
                "adapter_manifest": adapter_manifest,
                "views_by_name": {str(v["image_name"]): v for v in bundle_manifest["views"]},
            }
        )

    rows_payload: List[Dict[str, Any]] = []
    stats_rows: List[List[Any]] = []
    manifest_payload = {
        "scene_name": scene_name,
        "reference_manifest": str(reference_manifest_path),
        "gt_root": str(gt_root),
        "threshold_m": float(args.threshold_m),
        "random_seed": int(args.random_seed),
        "method_panel_layout": str(args.method_panel_layout),
        "depth_display_settings": {
            "depth_cmap": str(args.depth_cmap),
            "invalid_depth_rgb": [float(x) for x in invalid_depth_rgb],
            "show_depth_colorbar": bool(args.show_depth_colorbar),
            "resize_gt_to_view": bool(args.resize_gt_to_view),
        },
        "column_titles": {
            "gt_title": str(args.gt_title),
            "reference_title": str(args.reference_title),
        },
        "selected_views": [],
        "methods": [
            {
                "label": payload["label"],
                "method_dir": str(payload["method_dir"]),
                "bundle_manifest": str(payload["bundle_manifest_path"]),
                "adapter_manifest": str(payload["adapter_manifest_path"]),
            }
            for payload in methods_payload
        ],
    }

    for view in selected_views:
        image_name = str(view["image_name"])
        ref_view = ref_by_name[image_name]
        ref_npz = _load_npz(reference_manifest_path.parent, str(ref_view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        gt_rgb = _load_gt_image(gt_root / image_name)
        if bool(args.resize_gt_to_view):
            gt_rgb = _resize_rgb_image(gt_rgb, width=int(ref_view["width"]), height=int(ref_view["height"]))
        depth_rgb = _depth_to_rgb(
            ref_depth,
            ref_valid,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_depth_rgb,
        )

        method_panels: Dict[str, Dict[str, np.ndarray]] = {}
        method_counts: Dict[str, Dict[str, int]] = {}
        method_panels_ordered: List[Dict[str, Any]] = []
        for payload in methods_payload:
            if image_name not in payload["views_by_name"]:
                raise KeyError(f"{payload['label']} is missing selected image {image_name}")
            model_view = payload["views_by_name"][image_name]
            model_npz = _load_npz(payload["bundle_manifest_path"].parent, str(model_view["npz_file"]))
            raw_model_depth = np.asarray(model_npz["depth"], dtype=np.float64)
            model_opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
            adapter_semantics = str(payload["adapter_manifest"]["depth_semantics"])
            validity_rule = payload["adapter_manifest"]["validity_rule"]
            model_depth = _raw_depth_to_metric_camera_z(raw_model_depth, depth_semantics=adapter_semantics)
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
            class_rgb, counts = _classification_rgb(
                ref_valid=ref_valid,
                model_valid=model_valid,
                ref_depth=ref_depth,
                model_depth=model_depth,
                threshold_m=float(args.threshold_m),
            )
            method_panels[payload["label"]] = {
                "depth_rgb": model_depth_rgb,
                "class_rgb": class_rgb,
            }
            method_panels_ordered.append(
                {
                    "label": payload["label"],
                    "depth_rgb": model_depth_rgb,
                    "class_rgb": class_rgb,
                }
            )
            method_counts[payload["label"]] = counts
            stats_rows.append(
                [
                    scene_name,
                    str(view["view_id"]),
                    image_name,
                    payload["label"],
                    counts["reference_valid_pixels"],
                    counts["model_valid_pixels"],
                    counts["correct_pixels"],
                    counts["too_shallow_pixels"],
                    counts["too_deep_pixels"],
                    counts["missing_model_pixels"],
                    counts["correct_pixels"] / max(1, counts["reference_valid_pixels"]),
                    counts["too_shallow_pixels"] / max(1, counts["reference_valid_pixels"]),
                    counts["too_deep_pixels"] / max(1, counts["reference_valid_pixels"]),
                    counts["missing_model_pixels"] / max(1, counts["reference_valid_pixels"]),
                ]
            )

        stem = f"{str(view['view_id'])}_{_sanitize_stem(image_name)}"
        strip_path = out_dir / "per_view_comparisons" / f"{stem}__comparison.png"
        _render_per_view_comparison(
            out_path=strip_path,
            gt_rgb=gt_rgb,
            depth_rgb=depth_rgb,
            method_panels=method_panels_ordered,
            image_name=image_name,
            threshold_m=float(args.threshold_m),
            dpi=int(args.dpi),
            method_panel_layout=str(args.method_panel_layout),
            show_depth_colorbar=bool(args.show_depth_colorbar),
            depth_min_vis=depth_min_vis,
            depth_max_vis=depth_max_vis,
            cmap_name=str(args.depth_cmap),
            gt_title=str(args.gt_title),
            reference_title=str(args.reference_title),
        )

        rows_payload.append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "gt_rgb": gt_rgb,
                "depth_rgb": depth_rgb,
                "method_panels": method_panels,
            }
        )
        manifest_payload["selected_views"].append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "per_view_comparison_png": str(strip_path),
                "method_counts": method_counts,
            }
        )

    contact_sheet_path = out_dir / f"{scene_name}_comparison_contact_sheet.png"
    _render_scene_contact_sheet(
        out_path=contact_sheet_path,
        scene_name=scene_name,
        threshold_m=float(args.threshold_m),
        rows_payload=rows_payload,
        method_labels=[payload["label"] for payload in methods_payload],
        dpi=int(args.dpi),
        method_panel_layout=str(args.method_panel_layout),
        show_depth_colorbar=bool(args.show_depth_colorbar),
        depth_min_vis=depth_min_vis,
        depth_max_vis=depth_max_vis,
        cmap_name=str(args.depth_cmap),
        gt_title=str(args.gt_title),
        reference_title=str(args.reference_title),
    )

    _write_csv(
        out_dir / "method_comparison_stats.csv",
        [
            "scene_name",
            "view_id",
            "image_name",
            "method_label",
            "reference_valid_pixels",
            "model_valid_pixels",
            "correct_pixels",
            "too_shallow_pixels",
            "too_deep_pixels",
            "missing_model_pixels",
            "correct_rate",
            "too_shallow_rate",
            "too_deep_rate",
            "missing_model_rate",
        ],
        stats_rows,
    )
    manifest_payload["contact_sheet_png"] = str(contact_sheet_path)
    manifest_payload["depth_display_range_m"] = {
        "min": float(depth_min_vis),
        "max": float(depth_max_vis),
    }
    _save_json(out_dir / "method_comparison_manifest.json", manifest_payload)
    print(f"DEPTH_METHOD_COMPARISON_READY {out_dir}")


if __name__ == "__main__":
    main()
