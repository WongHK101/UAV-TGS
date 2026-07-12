from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Patch
import numpy as np

from visualize_depth_reference_views import (
    _add_depth_colorbar,
    _compute_depth_display_range,
    _depth_to_rgb,
    _load_gt_image,
    _load_json,
    _load_npz,
    _parse_rgb_triplet,
    _resolve_gt_root,
    _sanitize_stem,
    _save_json,
    _select_views,
    _write_csv,
)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Visualize reference validity diagnostics for selected held-out views.")
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
    ap.add_argument("--support_cmap", default="magma")
    ap.add_argument("--invalid_depth_rgb", default="0.94,0.94,0.94")
    ap.add_argument("--show_colorbars", action="store_true")
    ap.add_argument("--dpi", type=int, default=180)
    return ap


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


def _support_to_rgb(support_count: np.ndarray, finite_depth: np.ndarray, vmax: float, cmap_name: str, invalid_rgb: np.ndarray) -> np.ndarray:
    vmax = max(float(vmax), 1.0)
    normalized = np.clip(np.asarray(support_count, dtype=np.float64) / vmax, 0.0, 1.0)
    colored = plt.get_cmap(cmap_name)(normalized)[..., :3]
    colored = np.asarray(colored, dtype=np.float64)
    colored[~finite_depth] = invalid_rgb
    return colored


def _valid_mask_to_rgb(valid_mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros(valid_mask.shape + (3,), dtype=np.float64)
    rgb[~valid_mask] = np.array([0.15, 0.15, 0.15], dtype=np.float64)
    rgb[valid_mask] = np.array([0.20, 0.78, 0.28], dtype=np.float64)
    return rgb


def _inside_roi_to_rgb(inside_roi: np.ndarray) -> np.ndarray:
    rgb = np.zeros(inside_roi.shape + (3,), dtype=np.float64)
    rgb[~inside_roi] = np.array([0.16, 0.16, 0.16], dtype=np.float64)
    rgb[inside_roi] = np.array([0.98, 0.80, 0.20], dtype=np.float64)
    return rgb


def _invalid_reason_rgb(
    finite_depth: np.ndarray,
    inside_roi: np.ndarray,
    support_count: np.ndarray,
    valid_mask: np.ndarray,
    support_min_count: int,
) -> tuple[np.ndarray, Dict[str, int]]:
    finite_depth = np.asarray(finite_depth, dtype=bool)
    inside_roi = np.asarray(inside_roi, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    support_count = np.asarray(support_count, dtype=np.int32)

    no_mesh_hit = ~finite_depth
    outside_roi = finite_depth & (~inside_roi)
    low_support = finite_depth & inside_roi & (~valid_mask) & (support_count < int(support_min_count))
    other_invalid = (~valid_mask) & (~no_mesh_hit) & (~outside_roi) & (~low_support)
    valid_reference = valid_mask

    rgb = np.zeros(valid_mask.shape + (3,), dtype=np.float64)
    rgb[valid_reference] = np.array([0.20, 0.78, 0.28], dtype=np.float64)
    rgb[no_mesh_hit] = np.array([0.00, 0.00, 0.00], dtype=np.float64)
    rgb[outside_roi] = np.array([0.96, 0.68, 0.18], dtype=np.float64)
    rgb[low_support] = np.array([0.86, 0.22, 0.82], dtype=np.float64)
    rgb[other_invalid] = np.array([0.40, 0.40, 0.40], dtype=np.float64)

    counts = {
        "finite_depth_pixels": int(np.count_nonzero(finite_depth)),
        "valid_reference_pixels": int(np.count_nonzero(valid_reference)),
        "outside_roi_pixels": int(np.count_nonzero(outside_roi)),
        "low_support_pixels": int(np.count_nonzero(low_support)),
        "no_mesh_hit_pixels": int(np.count_nonzero(no_mesh_hit)),
        "other_invalid_pixels": int(np.count_nonzero(other_invalid)),
    }
    return rgb, counts


def _support_vmax(reference_manifest_path: Path, selected_views: Sequence[Dict[str, Any]], high_pct: float) -> float:
    values: List[np.ndarray] = []
    ref_root = reference_manifest_path.parent
    for view in selected_views:
        ref_npz = _load_npz(ref_root, str(view["npz_file"]))
        support = np.asarray(ref_npz["support_count"], dtype=np.float64)
        finite_depth = np.isfinite(np.asarray(ref_npz["depth"], dtype=np.float64))
        cur = support[finite_depth]
        if cur.size > 0:
            values.append(cur)
    if not values:
        return 1.0
    concat = np.concatenate(values, axis=0)
    vmax = float(np.percentile(concat, high_pct))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(concat)) if concat.size else 1.0
    return max(vmax, 1.0)


def _render_per_view_diagnostic(
    out_path: Path,
    gt_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    valid_rgb: np.ndarray,
    inside_roi_rgb: np.ndarray,
    support_rgb: np.ndarray,
    invalid_reason_rgb: np.ndarray,
    image_name: str,
    summary_text: str,
    dpi: int,
    show_colorbars: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    depth_cmap: str,
    support_vmax: float,
    support_cmap: str,
) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(20, 4), squeeze=False)
    row = axes[0]
    panels = [
        (gt_rgb, f"GT\n{image_name}"),
        (depth_rgb, "Reference\nMesh Depth"),
        (valid_rgb, "Reference\nValid Mask"),
        (inside_roi_rgb, "Inside ROI"),
        (support_rgb, "Support Count"),
        (invalid_reason_rgb, "Invalid Reason"),
    ]
    for ax, (img, title) in zip(row, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    legend_handles = [
        Patch(color=[0.20, 0.78, 0.28], label="Valid Reference"),
        Patch(color=[0.96, 0.68, 0.18], label="Outside ROI"),
        Patch(color=[0.86, 0.22, 0.82], label="Low Support"),
        Patch(color=[0.00, 0.00, 0.00], label="No Mesh Hit"),
        Patch(color=[0.40, 0.40, 0.40], label="Other Invalid"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(summary_text)
    right_margin = 0.92 if show_colorbars else 1.0
    fig.tight_layout(rect=(0.0, 0.10, right_margin, 0.90))
    if show_colorbars:
        _add_depth_colorbar(fig, depth_min_vis, depth_max_vis, depth_cmap, (0.93, 0.55, 0.015, 0.25), label="Depth (m)")
        _add_depth_colorbar(fig, 0.0, support_vmax, support_cmap, (0.93, 0.17, 0.015, 0.25), label="Support Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _render_contact_sheet(
    out_path: Path,
    scene_name: str,
    rows_payload: Sequence[Dict[str, Any]],
    dpi: int,
    show_colorbars: bool,
    depth_min_vis: float,
    depth_max_vis: float,
    depth_cmap: str,
    support_vmax: float,
    support_cmap: str,
) -> None:
    nrows = len(rows_payload)
    fig, axes = plt.subplots(nrows, 6, figsize=(20, 3.2 * nrows), squeeze=False)
    for row_idx, payload in enumerate(rows_payload):
        panels = [
            (payload["gt_rgb"], f"GT\n{payload['image_name']}"),
            (payload["depth_rgb"], "Reference\nMesh Depth"),
            (payload["valid_rgb"], "Reference\nValid Mask"),
            (payload["inside_roi_rgb"], "Inside ROI"),
            (payload["support_rgb"], "Support Count"),
            (payload["invalid_reason_rgb"], "Invalid Reason"),
        ]
        for col_idx, (img, title) in enumerate(panels):
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].set_title(title)
            axes[row_idx, col_idx].axis("off")

    legend_handles = [
        Patch(color=[0.20, 0.78, 0.28], label="Valid Reference"),
        Patch(color=[0.96, 0.68, 0.18], label="Outside ROI"),
        Patch(color=[0.86, 0.22, 0.82], label="Low Support"),
        Patch(color=[0.00, 0.00, 0.00], label="No Mesh Hit"),
        Patch(color=[0.40, 0.40, 0.40], label="Other Invalid"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(f"{scene_name}: Reference Validity Diagnostics")
    right_margin = 0.92 if show_colorbars else 1.0
    fig.tight_layout(rect=(0.0, 0.06, right_margin, 0.96))
    if show_colorbars:
        _add_depth_colorbar(fig, depth_min_vis, depth_max_vis, depth_cmap, (0.93, 0.55, 0.015, 0.28), label="Depth (m)")
        _add_depth_colorbar(fig, 0.0, support_vmax, support_cmap, (0.93, 0.17, 0.015, 0.28), label="Support Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _build_argparser().parse_args()
    reference_manifest_path = Path(args.reference_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_manifest = _load_json(reference_manifest_path)
    scene_name = str(args.scene_name) if str(args.scene_name) else str(reference_manifest.get("scene_name", "Scene"))
    gt_root = _resolve_gt_root(reference_manifest, override_root=str(args.gt_images_root), override_images_dir=str(args.gt_images_dir_name))
    selected_views = _choose_views(reference_manifest=reference_manifest, args=args)
    invalid_depth_rgb = _parse_rgb_triplet(str(args.invalid_depth_rgb), default=(0.94, 0.94, 0.94))
    depth_min_vis, depth_max_vis = _compute_depth_display_range(
        reference_manifest_path=reference_manifest_path,
        selected_views=selected_views,
        low_pct=float(args.depth_percentile_low),
        high_pct=float(args.depth_percentile_high),
    )
    support_vmax = _support_vmax(reference_manifest_path, selected_views, high_pct=98.0)
    support_rule = reference_manifest.get("support_rule", {})
    support_min_count = int(support_rule.get("min_support_count", 1))

    rows_payload: List[Dict[str, Any]] = []
    stats_rows: List[List[Any]] = []
    manifest_payload = {
        "scene_name": scene_name,
        "reference_manifest": str(reference_manifest_path),
        "gt_root": str(gt_root),
        "random_seed": int(args.random_seed),
        "support_rule": support_rule,
        "depth_display_settings": {
            "depth_cmap": str(args.depth_cmap),
            "support_cmap": str(args.support_cmap),
            "invalid_depth_rgb": [float(x) for x in invalid_depth_rgb],
            "show_colorbars": bool(args.show_colorbars),
        },
        "selected_views": [],
    }

    for view in selected_views:
        image_name = str(view["image_name"])
        ref_npz = _load_npz(reference_manifest_path.parent, str(view["npz_file"]))
        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        support_count = np.asarray(ref_npz["support_count"], dtype=np.int32)
        valid_mask = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        inside_roi = np.asarray(ref_npz["inside_roi"], dtype=np.uint8).astype(bool)
        finite_depth = np.isfinite(ref_depth)

        gt_rgb = _load_gt_image(gt_root / image_name)
        depth_rgb = _depth_to_rgb(
            ref_depth,
            valid_mask,
            depth_min_vis,
            depth_max_vis,
            cmap_name=str(args.depth_cmap),
            invalid_rgb=invalid_depth_rgb,
        )
        valid_rgb = _valid_mask_to_rgb(valid_mask)
        inside_roi_rgb = _inside_roi_to_rgb(inside_roi)
        support_rgb = _support_to_rgb(support_count, finite_depth, support_vmax, str(args.support_cmap), invalid_depth_rgb)
        invalid_reason_rgb, counts = _invalid_reason_rgb(
            finite_depth=finite_depth,
            inside_roi=inside_roi,
            support_count=support_count,
            valid_mask=valid_mask,
            support_min_count=support_min_count,
        )

        total_pixels = int(ref_depth.size)
        valid_ratio = counts["valid_reference_pixels"] / max(1, total_pixels)
        summary_text = (
            f"{scene_name} | {image_name} | valid={counts['valid_reference_pixels']} ({valid_ratio:.3f}) | "
            f"outside_roi={counts['outside_roi_pixels']} | low_support={counts['low_support_pixels']} | no_hit={counts['no_mesh_hit_pixels']}"
        )

        stem = f"{str(view['view_id'])}_{_sanitize_stem(image_name)}"
        strip_path = out_dir / "per_view_reference_diagnostics" / f"{stem}__reference_diagnostic.png"
        _render_per_view_diagnostic(
            out_path=strip_path,
            gt_rgb=gt_rgb,
            depth_rgb=depth_rgb,
            valid_rgb=valid_rgb,
            inside_roi_rgb=inside_roi_rgb,
            support_rgb=support_rgb,
            invalid_reason_rgb=invalid_reason_rgb,
            image_name=image_name,
            summary_text=summary_text,
            dpi=int(args.dpi),
            show_colorbars=bool(args.show_colorbars),
            depth_min_vis=depth_min_vis,
            depth_max_vis=depth_max_vis,
            depth_cmap=str(args.depth_cmap),
            support_vmax=support_vmax,
            support_cmap=str(args.support_cmap),
        )

        rows_payload.append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "gt_rgb": gt_rgb,
                "depth_rgb": depth_rgb,
                "valid_rgb": valid_rgb,
                "inside_roi_rgb": inside_roi_rgb,
                "support_rgb": support_rgb,
                "invalid_reason_rgb": invalid_reason_rgb,
            }
        )
        stats_rows.append(
            [
                scene_name,
                str(view["view_id"]),
                image_name,
                total_pixels,
                counts["finite_depth_pixels"],
                counts["valid_reference_pixels"],
                counts["outside_roi_pixels"],
                counts["low_support_pixels"],
                counts["no_mesh_hit_pixels"],
                counts["other_invalid_pixels"],
                counts["finite_depth_pixels"] / max(1, total_pixels),
                counts["valid_reference_pixels"] / max(1, total_pixels),
                counts["outside_roi_pixels"] / max(1, total_pixels),
                counts["low_support_pixels"] / max(1, total_pixels),
                counts["no_mesh_hit_pixels"] / max(1, total_pixels),
            ]
        )
        manifest_payload["selected_views"].append(
            {
                "view_id": str(view["view_id"]),
                "image_name": image_name,
                "diagnostic_png": str(strip_path),
                **counts,
            }
        )

    contact_sheet_path = out_dir / f"{scene_name}_reference_validity_contact_sheet.png"
    _render_contact_sheet(
        out_path=contact_sheet_path,
        scene_name=scene_name,
        rows_payload=rows_payload,
        dpi=int(args.dpi),
        show_colorbars=bool(args.show_colorbars),
        depth_min_vis=depth_min_vis,
        depth_max_vis=depth_max_vis,
        depth_cmap=str(args.depth_cmap),
        support_vmax=support_vmax,
        support_cmap=str(args.support_cmap),
    )

    _write_csv(
        out_dir / "reference_validity_stats.csv",
        [
            "scene_name",
            "view_id",
            "image_name",
            "total_pixels",
            "finite_depth_pixels",
            "valid_reference_pixels",
            "outside_roi_pixels",
            "low_support_pixels",
            "no_mesh_hit_pixels",
            "other_invalid_pixels",
            "finite_depth_ratio",
            "valid_reference_ratio",
            "outside_roi_ratio",
            "low_support_ratio",
            "no_mesh_hit_ratio",
        ],
        stats_rows,
    )
    manifest_payload["contact_sheet_png"] = str(contact_sheet_path)
    manifest_payload["depth_display_range_m"] = {
        "min": float(depth_min_vis),
        "max": float(depth_max_vis),
    }
    manifest_payload["support_display_max"] = float(support_vmax)
    _save_json(out_dir / "reference_validity_manifest.json", manifest_payload)
    print(f"REFERENCE_VALIDITY_VIS_READY {out_dir}")


if __name__ == "__main__":
    main()
