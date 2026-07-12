from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


def _parse_rgb_triplet(text: str, default: Sequence[float] | None = None) -> np.ndarray:
    if not text:
        if default is None:
            raise ValueError("RGB triplet text is empty and no default was provided.")
        return np.asarray(default, dtype=np.float64)
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B triplet, got: {text!r}")
    rgb = np.asarray([float(part) for part in parts], dtype=np.float64)
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        raise ValueError(f"RGB triplet values must lie in [0, 1], got: {text!r}")
    return rgb


def _raw_depth_to_metric_camera_z(raw_depth: np.ndarray, depth_semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if depth_semantics == "metric_camera_z_from_renderer":
        return raw_depth
    if depth_semantics == "inverse_camera_z_from_renderer":
        metric = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        metric[positive] = 1.0 / raw_depth[positive]
        return metric
    raise ValueError(f"Unsupported depth semantics: {depth_semantics!r}")


def _make_model_valid_mask(metric_depth: np.ndarray, opacity: np.ndarray, depth_min: float, opacity_threshold: float) -> np.ndarray:
    return (
        np.isfinite(metric_depth)
        & np.isfinite(opacity)
        & (metric_depth > float(depth_min))
        & (opacity >= float(opacity_threshold))
    )


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Visualize GT / reference mesh depth / categorical depth agreement for selected held-out views.")
    ap.add_argument("--reference_manifest", required=True)
    ap.add_argument("--model_manifest", required=True)
    ap.add_argument("--adapter_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold_m", type=float, required=True)
    ap.add_argument("--image_names", nargs="*", default=[])
    ap.add_argument("--view_ids", nargs="*", default=[])
    ap.add_argument("--first_n", type=int, default=0)
    ap.add_argument("--gt_images_root", default="")
    ap.add_argument("--gt_images_dir_name", default="")
    ap.add_argument("--depth_percentile_low", type=float, default=2.0)
    ap.add_argument("--depth_percentile_high", type=float, default=98.0)
    ap.add_argument("--depth_cmap", default="viridis")
    ap.add_argument("--invalid_depth_rgb", default="0.0,0.0,0.0")
    ap.add_argument("--show_depth_colorbar", action="store_true")
    ap.add_argument("--resize_gt_to_view", action="store_true")
    ap.add_argument("--dpi", type=int, default=180)
    return ap


def _resolve_gt_root(reference_manifest: Dict[str, Any], override_root: str, override_images_dir: str) -> Path:
    if override_root:
        gt_root = Path(override_root)
        if override_images_dir:
            gt_root = gt_root / override_images_dir
        return gt_root.resolve()

    camera_manifest_path = reference_manifest.get("camera_manifest_path", "")
    if camera_manifest_path:
        camera_manifest = _load_json(Path(camera_manifest_path))
        source_path = Path(camera_manifest["source_path"])
        images_dir_name = str(camera_manifest["images_dir_name"])
        return (source_path / images_dir_name).resolve()

    raise ValueError("Could not infer GT image root from reference manifest; pass --gt_images_root explicitly.")


def _select_views(reference_manifest: Dict[str, Any], image_names: Sequence[str], view_ids: Sequence[str], first_n: int) -> List[Dict[str, Any]]:
    views = reference_manifest["views"]
    views_by_name = {str(v["image_name"]): v for v in views}
    views_by_id = {str(v["view_id"]): v for v in views}

    selected: List[Dict[str, Any]] = []
    if image_names:
        for name in image_names:
            if name not in views_by_name:
                raise KeyError(f"Image name not found in reference manifest: {name}")
            selected.append(views_by_name[name])
    elif view_ids:
        for view_id in view_ids:
            if view_id not in views_by_id:
                raise KeyError(f"View id not found in reference manifest: {view_id}")
            selected.append(views_by_id[view_id])
    else:
        count = int(first_n) if int(first_n) > 0 else 3
        selected = list(views[:count])

    seen = set()
    deduped: List[Dict[str, Any]] = []
    for view in selected:
        key = str(view["view_id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(view)
    return deduped


def _load_npz(base_dir: Path, rel_path: str) -> Dict[str, np.ndarray]:
    path = base_dir / rel_path
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


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


def _depth_to_rgb(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    depth_min: float,
    depth_max: float,
    cmap_name: str = "viridis",
    invalid_rgb: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    normalized = np.zeros(depth.shape, dtype=np.float64)
    if depth_max > depth_min:
        normalized = (depth - depth_min) / (depth_max - depth_min)
    normalized = np.clip(normalized, 0.0, 1.0)
    colored = plt.get_cmap(cmap_name)(normalized)[..., :3]
    colored = np.asarray(colored, dtype=np.float64)
    invalid_rgb_arr = np.asarray(invalid_rgb if invalid_rgb is not None else (0.0, 0.0, 0.0), dtype=np.float64)
    colored[~valid_mask] = invalid_rgb_arr
    return colored


def _classification_rgb(
    ref_valid: np.ndarray,
    model_valid: np.ndarray,
    ref_depth: np.ndarray,
    model_depth: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, Dict[str, int]]:
    colors_map = {
        "invalid_reference": np.array([0.10, 0.10, 0.10], dtype=np.float64),
        "missing_model": np.array([1.00, 0.68, 0.12], dtype=np.float64),
        "correct": np.array([0.18, 0.72, 0.24], dtype=np.float64),
        "too_shallow": np.array([0.92, 0.22, 0.18], dtype=np.float64),
        "too_deep": np.array([0.18, 0.40, 0.95], dtype=np.float64),
    }
    out = np.zeros(ref_depth.shape + (3,), dtype=np.float64)
    out[...] = colors_map["invalid_reference"]

    ref_valid = np.asarray(ref_valid, dtype=bool)
    model_valid = np.asarray(model_valid, dtype=bool)
    valid_joint = ref_valid & model_valid
    missing = ref_valid & (~model_valid)
    too_shallow = valid_joint & (model_depth < (ref_depth - threshold_m))
    too_deep = valid_joint & (model_depth > (ref_depth + threshold_m))
    correct = valid_joint & (~too_shallow) & (~too_deep)

    out[missing] = colors_map["missing_model"]
    out[correct] = colors_map["correct"]
    out[too_shallow] = colors_map["too_shallow"]
    out[too_deep] = colors_map["too_deep"]

    counts = {
        "reference_valid_pixels": int(np.count_nonzero(ref_valid)),
        "model_valid_pixels": int(np.count_nonzero(valid_joint)),
        "missing_model_pixels": int(np.count_nonzero(missing)),
        "correct_pixels": int(np.count_nonzero(correct)),
        "too_shallow_pixels": int(np.count_nonzero(too_shallow)),
        "too_deep_pixels": int(np.count_nonzero(too_deep)),
    }
    return out, counts


def _load_gt_image(gt_path: Path) -> np.ndarray:
    image = mpimg.imread(str(gt_path))
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    image = image.astype(np.float64)
    if image.max() > 1.0:
        image = image / 255.0
    return image



def _resize_rgb_image(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    arr = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
    image = Image.fromarray(arr)
    resized = image.resize((int(width), int(height)), resample=Image.BILINEAR)
    out = np.asarray(resized, dtype=np.float64) / 255.0
    return out


def _sanitize_stem(text: str) -> str:
    return str(text).replace("\\", "_").replace("/", "_").replace(":", "_").replace(" ", "_")


def _classification_legend_handles() -> List[Patch]:
    return [
        Patch(color=[0.18, 0.72, 0.24], label="Correct"),
        Patch(color=[0.18, 0.40, 0.95], label="Too Deep"),
        Patch(color=[0.92, 0.22, 0.18], label="Too Shallow"),
        Patch(color=[1.00, 0.68, 0.12], label="Missing Model"),
        Patch(color=[0.10, 0.10, 0.10], label="Invalid Reference"),
    ]


def _add_depth_colorbar(
    fig: plt.Figure,
    depth_min: float,
    depth_max: float,
    cmap_name: str,
    cbar_rect: Sequence[float],
    label: str = "Depth (m)",
) -> None:
    cax = fig.add_axes(list(cbar_rect))
    norm = colors.Normalize(vmin=float(depth_min), vmax=float(depth_max))
    sm = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    cbar.set_label(label)


def _render_single_strip(
    out_path: Path,
    gt_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    class_rgb: np.ndarray,
    image_name: str,
    threshold_m: float,
    counts: Dict[str, int],
    dpi: int,
    show_depth_colorbar: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
    ax_gt, ax_depth, ax_class = axes[0]
    ax_gt.imshow(gt_rgb)
    ax_gt.set_title(f"GT: {image_name}")
    ax_gt.axis("off")

    ax_depth.imshow(depth_rgb)
    ax_depth.set_title("Reference Mesh Depth")
    ax_depth.axis("off")

    ax_class.imshow(class_rgb)
    ax_class.set_title(f"Threshold Classification @ {threshold_m:.2f} m")
    ax_class.axis("off")

    fig.legend(handles=_classification_legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"{image_name} | valid={counts['reference_valid_pixels']} | "
        f"correct={counts['correct_pixels']} | shallow={counts['too_shallow_pixels']} | "
        f"deep={counts['too_deep_pixels']} | missing={counts['missing_model_pixels']}"
    )
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.08, right_margin, 0.90))
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


def _render_contact_sheet(
    out_path: Path,
    rows_payload: Sequence[Dict[str, Any]],
    threshold_m: float,
    dpi: int,
    show_depth_colorbar: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    cmap_name: str,
) -> None:
    row_count = len(rows_payload)
    fig, axes = plt.subplots(row_count, 3, figsize=(12, 4 * row_count), squeeze=False)
    for row_idx, payload in enumerate(rows_payload):
        axes[row_idx, 0].imshow(payload["gt_rgb"])
        axes[row_idx, 0].set_title(f"GT: {payload['image_name']}")
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(payload["depth_rgb"])
        axes[row_idx, 1].set_title("Reference Mesh Depth")
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(payload["class_rgb"])
        axes[row_idx, 2].set_title(f"Classification @ {threshold_m:.2f} m")
        axes[row_idx, 2].axis("off")

    fig.legend(handles=_classification_legend_handles(), loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"Held-out View Visualization @ {threshold_m:.2f} m")
    right_margin = 0.92 if show_depth_colorbar else 1.0
    fig.tight_layout(rect=(0.0, 0.06, right_margin, 0.96))
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
    model_manifest_path = Path(args.model_manifest).resolve()
    adapter_manifest_path = Path(args.adapter_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = _load_json(reference_manifest_path)
    model_manifest = _load_json(model_manifest_path)
    adapter_manifest = _load_json(adapter_manifest_path)

    gt_root = _resolve_gt_root(reference_manifest, override_root=str(args.gt_images_root), override_images_dir=str(args.gt_images_dir_name))
    selected_views = _select_views(
        reference_manifest=reference_manifest,
        image_names=args.image_names,
        view_ids=args.view_ids,
        first_n=int(args.first_n),
    )

    ref_by_name = {str(v["image_name"]): v for v in reference_manifest["views"]}
    model_by_name = {str(v["image_name"]): v for v in model_manifest["views"]}
    adapter_semantics = str(adapter_manifest["depth_semantics"])
    validity_rule = adapter_manifest["validity_rule"]
    depth_min = float(validity_rule.get("depth_min", 1e-6))
    opacity_threshold = float(validity_rule.get("opacity_threshold", 0.5))
    invalid_depth_rgb = _parse_rgb_triplet(str(args.invalid_depth_rgb), default=(0.0, 0.0, 0.0))

    depth_min_vis, depth_max_vis = _compute_depth_display_range(
        reference_manifest_path=reference_manifest_path,
        selected_views=selected_views,
        low_pct=float(args.depth_percentile_low),
        high_pct=float(args.depth_percentile_high),
    )

    rows_payload: List[Dict[str, Any]] = []
    stats_rows: List[List[Any]] = []
    selection_payload = {
        "reference_manifest": str(reference_manifest_path),
        "model_manifest": str(model_manifest_path),
        "adapter_manifest": str(adapter_manifest_path),
        "gt_root": str(gt_root),
        "threshold_m": float(args.threshold_m),
        "depth_display_settings": {
            "depth_cmap": str(args.depth_cmap),
            "invalid_depth_rgb": [float(x) for x in invalid_depth_rgb],
            "show_depth_colorbar": bool(args.show_depth_colorbar),
            "resize_gt_to_view": bool(args.resize_gt_to_view),
        },
        "selected_views": [],
    }

    for view in selected_views:
        image_name = str(view["image_name"])
        if image_name not in ref_by_name or image_name not in model_by_name:
            raise KeyError(f"Selected view is missing from reference/model manifests: {image_name}")
        ref_view = ref_by_name[image_name]
        model_view = model_by_name[image_name]
        ref_npz = _load_npz(reference_manifest_path.parent, str(ref_view["npz_file"]))
        model_npz = _load_npz(model_manifest_path.parent, str(model_view["npz_file"]))

        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        raw_model_depth = np.asarray(model_npz["depth"], dtype=np.float64)
        model_opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
        model_depth = _raw_depth_to_metric_camera_z(raw_model_depth, depth_semantics=adapter_semantics)
        model_valid = _make_model_valid_mask(model_depth, model_opacity, depth_min=depth_min, opacity_threshold=opacity_threshold)

        gt_path = gt_root / image_name
        if not gt_path.exists():
            raise FileNotFoundError(f"GT image not found: {gt_path}")
        gt_rgb = _load_gt_image(gt_path)
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
        class_rgb, counts = _classification_rgb(
            ref_valid=ref_valid,
            model_valid=model_valid,
            ref_depth=ref_depth,
            model_depth=model_depth,
            threshold_m=float(args.threshold_m),
        )

        stem = f"{str(view['view_id'])}_{_sanitize_stem(image_name)}"
        strip_path = out_dir / "strips" / f"{stem}__triptych.png"
        _render_single_strip(
            out_path=strip_path,
            gt_rgb=gt_rgb,
            depth_rgb=depth_rgb,
            class_rgb=class_rgb,
            image_name=image_name,
            threshold_m=float(args.threshold_m),
            counts=counts,
            dpi=int(args.dpi),
            show_depth_colorbar=bool(args.show_depth_colorbar),
            depth_min_vis=depth_min_vis,
            depth_max_vis=depth_max_vis,
            cmap_name=str(args.depth_cmap),
        )

        rows_payload.append(
            {
                "image_name": image_name,
                "view_id": str(view["view_id"]),
                "gt_rgb": gt_rgb,
                "depth_rgb": depth_rgb,
                "class_rgb": class_rgb,
            }
        )
        stats_rows.append(
            [
                str(view["view_id"]),
                image_name,
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
        selection_payload["selected_views"].append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "gt_image": str(gt_path),
                "strip_png": str(strip_path),
                **counts,
            }
        )

    contact_sheet_path = out_dir / "selected_views_contact_sheet.png"
    _render_contact_sheet(
        out_path=contact_sheet_path,
        rows_payload=rows_payload,
        threshold_m=float(args.threshold_m),
        dpi=int(args.dpi),
        show_depth_colorbar=bool(args.show_depth_colorbar),
        depth_min_vis=depth_min_vis,
        depth_max_vis=depth_max_vis,
        cmap_name=str(args.depth_cmap),
    )

    _write_csv(
        out_dir / "selected_views_stats.csv",
        [
            "view_id",
            "image_name",
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
    selection_payload["contact_sheet_png"] = str(contact_sheet_path)
    selection_payload["depth_display_range_m"] = {
        "min": float(depth_min_vis),
        "max": float(depth_max_vis),
    }
    _save_json(out_dir / "selected_views_manifest.json", selection_payload)
    print(f"DEPTH_VIEW_VIS_READY {out_dir}")


if __name__ == "__main__":
    main()
