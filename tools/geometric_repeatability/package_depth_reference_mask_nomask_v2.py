from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

TOOL_DIR = Path(__file__).resolve().parent
REPO = TOOL_DIR.parent.parent
sys.path.insert(0, str(TOOL_DIR))

from visualize_depth_reference_views import (  # noqa: E402
    _compute_depth_display_range,
    _depth_to_rgb,
    _load_gt_image,
    _load_json,
    _load_npz,
    _make_model_valid_mask,
    _raw_depth_to_metric_camera_z,
    _resolve_gt_root,
    _resize_rgb_image,
)


FORMAL = Path(".")
MESHFIX = Path(".")
V1 = Path(".")
OUT = Path("depth_reference_package")
ZIP = OUT.with_suffix(".zip")
RECOMPUTE_METRICS = False
REQUIRE_NATIVE_ALIGN = False
DEPRECATED_TARGETS: List[Path] = []

SCENES = ["Building", "PVpanel", "Road", "TransmissionTower", "Orchard"]
METHODS = [
    "Ours_M00_full",
    "Ours_G01_full",
    "Ours_G02_full",
    "Ours_M01_full",
    "Thermal3D_GS_full",
    "ThermalGaussian_OMMG_full",
    "ThermalGaussian_MSMG_full",
    "ThermalGaussian_MFTG_full",
]
SOTA5 = [
    "Ours_M01_full",
    "Thermal3D_GS_full",
    "ThermalGaussian_MFTG_full",
    "ThermalGaussian_MSMG_full",
    "ThermalGaussian_OMMG_full",
]
ABLATION4 = ["Ours_M00_full", "Ours_G01_full", "Ours_G02_full", "Ours_M01_full"]
GROUPS = {"SOTA5": SOTA5, "Ablation4": ABLATION4}
DISPLAY = {
    "SOTA5": {
        "Ours_M01_full": "Ours",
        "Thermal3D_GS_full": "Thermal3D-GS",
        "ThermalGaussian_MFTG_full": "ThermalGaussian-MFTG",
        "ThermalGaussian_MSMG_full": "ThermalGaussian-MSMG",
        "ThermalGaussian_OMMG_full": "ThermalGaussian-OMMG",
    },
    "Ablation4": {
        "Ours_M00_full": "Baseline",
        "Ours_G01_full": "+SSP",
        "Ours_G02_full": "+STT",
        "Ours_M01_full": "Full model",
    },
    "global": {
        "Ours_M00_full": "Baseline",
        "Ours_G01_full": "+SSP",
        "Ours_G02_full": "+STT",
        "Ours_M01_full": "Ours",
        "Thermal3D_GS_full": "Thermal3D-GS",
        "ThermalGaussian_MFTG_full": "ThermalGaussian-MFTG",
        "ThermalGaussian_MSMG_full": "ThermalGaussian-MSMG",
        "ThermalGaussian_OMMG_full": "ThermalGaussian-OMMG",
    },
}
THRESHOLDS = [0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 30.00]
MASKED_REFS = {
    "Building": FORMAL / "Building" / "reference" / "reference_depth_manifest.json",
    "Orchard": FORMAL / "Orchard" / "reference" / "reference_depth_manifest.json",
    "PVpanel": MESHFIX / "PVpanel" / "reference" / "reference_depth_manifest.json",
    "Road": MESHFIX / "Road" / "reference" / "reference_depth_manifest.json",
    "TransmissionTower": MESHFIX / "TransmissionTower" / "reference" / "reference_depth_manifest.json",
}

INVALID_RGB = (1.0, 1.0, 1.0)
DEPTH_CMAP = "turbo"
DPI_CONTACT = 135
DPI_STRIP = 130

CLASS_COLORS = {
    "correct": (0.15, 0.70, 0.25),
    "front_intrusion": (0.90, 0.10, 0.10),
    "too_deep": (0.15, 0.30, 0.95),
    "missing": (0.05, 0.05, 0.05),
    "ignored": (1.0, 1.0, 1.0),
}
LEGEND = [
    Patch(facecolor=CLASS_COLORS["correct"], label="Within threshold"),
    Patch(facecolor=CLASS_COLORS["front_intrusion"], label="Front intrusion"),
    Patch(facecolor=CLASS_COLORS["too_deep"], label="Too deep"),
    Patch(facecolor=CLASS_COLORS["missing"], label="Missing model"),
    Patch(facecolor=CLASS_COLORS["ignored"], edgecolor="0.6", label="Not evaluated"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a mask/no-mask depth-reference result package")
    parser.add_argument("--formal_root", required=True, help="Root containing method model bundles and Building/Orchard references")
    parser.add_argument("--meshfix_root", required=True, help="Root containing repaired PVpanel/Road/TransmissionTower references")
    parser.add_argument("--previous_package_root", default="", help="Previous package root used when copying metrics")
    parser.add_argument("--out", required=True, help="Output package directory")
    parser.add_argument("--zip", default="", help="Output zip path; defaults to <out>.zip")
    parser.add_argument(
        "--recompute_metrics",
        action="store_true",
        help="Re-evaluate masked/no-mask metrics from the selected references and model bundles instead of copying previous metrics.",
    )
    parser.add_argument(
        "--require_native_align",
        action="store_true",
        help="Require every model bundle to use camera_frame_mode=probe_manifest_native_align.",
    )
    parser.add_argument(
        "--deprecated_target",
        action="append",
        default=[],
        help="Old package folder or zip path to mark as deprecated via sidecar marker files.",
    )
    args = parser.parse_args()
    if not args.recompute_metrics and not str(args.previous_package_root).strip():
        parser.error("--previous_package_root is required unless --recompute_metrics is set")
    return args


def configure_from_args(args: argparse.Namespace) -> None:
    global FORMAL, MESHFIX, V1, OUT, ZIP, MASKED_REFS, RECOMPUTE_METRICS, REQUIRE_NATIVE_ALIGN, DEPRECATED_TARGETS
    FORMAL = Path(args.formal_root).resolve()
    MESHFIX = Path(args.meshfix_root).resolve()
    V1 = Path(args.previous_package_root).resolve() if str(args.previous_package_root).strip() else Path()
    OUT = Path(args.out).resolve()
    ZIP = Path(args.zip).resolve() if str(args.zip).strip() else OUT.with_suffix(".zip")
    RECOMPUTE_METRICS = bool(args.recompute_metrics)
    REQUIRE_NATIVE_ALIGN = bool(args.require_native_align)
    DEPRECATED_TARGETS = [Path(p).resolve() for p in args.deprecated_target]
    MASKED_REFS = {
        "Building": FORMAL / "Building" / "reference" / "reference_depth_manifest.json",
        "Orchard": FORMAL / "Orchard" / "reference" / "reference_depth_manifest.json",
        "PVpanel": MESHFIX / "PVpanel" / "reference" / "reference_depth_manifest.json",
        "Road": MESHFIX / "Road" / "reference" / "reference_depth_manifest.json",
        "TransmissionTower": MESHFIX / "TransmissionTower" / "reference" / "reference_depth_manifest.json",
    }


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_name(text: str) -> str:
    text = text.replace("+", "plus_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def thr_dir(delta: float) -> str:
    return f"thr_{delta:.2f}m".replace(".", "p")


def refs_to_str(refs: Dict[str, Dict[str, Path]]) -> Dict[str, Dict[str, str]]:
    return {mask: {scene: str(path) for scene, path in scenes.items()} for mask, scenes in refs.items()}


def prepare_output() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    if ZIP.exists():
        ZIP.unlink()
    for subdir in [
        "metrics",
        "tables",
        "plots",
        "visualizations",
        "reference_manifests",
        "protocol_notes",
        "code_snapshot/tools/geometric_repeatability",
    ]:
        (OUT / subdir).mkdir(parents=True, exist_ok=True)


def create_nomask_reference(scene: str, masked_manifest: Path) -> tuple[Path, int, int]:
    dst_root = OUT / "reference_manifests" / "nomask" / scene
    dst_root.mkdir(parents=True, exist_ok=True)
    src_root = masked_manifest.parent
    manifest = load_json(masked_manifest)
    manifest["reference_mask_variant"] = "no_mask_finite_depth_only"
    manifest["reference_mask_note"] = (
        "valid_mask is finite(reference depth) & reference depth > 0; support_count and inside_roi are preserved for audit but not used as validity."
    )
    manifest["masked_source_reference_manifest"] = str(masked_manifest)
    masked_valid_total = 0
    nomask_valid_total = 0
    for view in manifest["views"]:
        arr = np.load(src_root / view["npz_file"])
        depth = np.asarray(arr["depth"])
        support_count = np.asarray(arr["support_count"]) if "support_count" in arr.files else np.zeros(depth.shape, dtype=np.int32)
        inside_roi = np.asarray(arr["inside_roi"]) if "inside_roi" in arr.files else np.ones(depth.shape, dtype=np.uint8)
        src_valid = np.asarray(arr["valid_mask"]).astype(bool) if "valid_mask" in arr.files else (np.isfinite(depth) & (depth > 0))
        nomask_valid = np.isfinite(depth) & (depth > 0)
        masked_valid_total += int(np.count_nonzero(src_valid))
        nomask_valid_total += int(np.count_nonzero(nomask_valid))
        dst_npz = dst_root / view["npz_file"]
        dst_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dst_npz,
            depth=depth,
            support_count=support_count,
            valid_mask=nomask_valid.astype(np.uint8),
            inside_roi=inside_roi,
        )
    manifest["nomask_total_valid_pixels"] = nomask_valid_total
    manifest["masked_source_total_valid_pixels"] = masked_valid_total
    out_manifest = dst_root / "reference_depth_manifest.json"
    save_json(out_manifest, manifest)
    return out_manifest, masked_valid_total, nomask_valid_total


def prepare_references() -> tuple[Dict[str, Dict[str, Path]], List[Dict[str, Any]]]:
    refs: Dict[str, Dict[str, Path]] = {"masked": {}, "nomask": {}}
    rows: List[Dict[str, Any]] = []
    for scene in SCENES:
        masked = MASKED_REFS[scene]
        if not masked.exists():
            raise FileNotFoundError(masked)
        refs["masked"][scene] = masked
        dst = OUT / "reference_manifests" / "masked" / scene
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["reference_depth_manifest.json", "reference_build_manifest.json", "reference_roi.json", "probe_camera_manifest.json"]:
            src = masked.parent / name
            if src.exists():
                shutil.copy2(src, dst / name)
        nomask, masked_valid, nomask_valid = create_nomask_reference(scene, masked)
        refs["nomask"][scene] = nomask
        rows.append(
            {
                "scene_name": scene,
                "masked_reference_manifest": str(masked),
                "nomask_reference_manifest": str(nomask),
                "masked_valid_pixels": masked_valid,
                "nomask_finite_depth_pixels": nomask_valid,
                "nomask_over_masked_ratio": nomask_valid / masked_valid if masked_valid else math.nan,
            }
        )
        log(f"Reference {scene}: masked={masked_valid:,}, nomask={nomask_valid:,}")
    save_json(OUT / "reference_manifests" / "reference_manifest_index.json", refs_to_str(refs))
    write_csv(OUT / "tables" / "reference_mask_valid_area_summary.csv", rows)
    return refs, rows


def copy_final_reference_meshes() -> List[Dict[str, Any]]:
    """Copy the exact reference meshes recorded by the reference build manifests.

    These meshes are the geometry sources used to render the packaged
    reference-depth manifests. Dense fused clouds are intentionally not copied
    because they are large intermediate artifacts; their source paths and all
    construction parameters remain recorded for reproducibility.
    """
    rows: List[Dict[str, Any]] = []
    mesh_root = OUT / "reference_meshes"
    for scene in SCENES:
        build_manifest_path = MASKED_REFS[scene].parent / "reference_build_manifest.json"
        build_manifest = load_json(build_manifest_path)
        source_mesh = Path(str(build_manifest.get("reference_mesh_path", "")))
        if not source_mesh.exists() or source_mesh.stat().st_size <= 0:
            raise FileNotFoundError(f"Final reference mesh is missing or empty for {scene}: {source_mesh}")
        scene_dir = mesh_root / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        copied_mesh = scene_dir / source_mesh.name
        shutil.copy2(source_mesh, copied_mesh)
        for name in ["reference_build_manifest.json", "reference_roi.json", "probe_camera_manifest.json", "reference_depth_manifest.json"]:
            src = MASKED_REFS[scene].parent / name
            if src.exists():
                shutil.copy2(src, scene_dir / name)

        overrides = build_manifest.get("reference_construction_overrides", {})
        support_rule = build_manifest.get("support_rule", {})
        row = {
            "scene_name": scene,
            "mesh_backend": build_manifest.get("reference_mesh_backend", ""),
            "source_reference_mesh_path": str(source_mesh),
            "packaged_reference_mesh_path": str(copied_mesh),
            "source_reference_mesh_size_bytes": source_mesh.stat().st_size,
            "reference_mesher_input_ply": build_manifest.get("reference_mesher_input_ply", ""),
            "reference_fused_ply": build_manifest.get("reference_fused_ply", ""),
            "roi_path": build_manifest.get("roi_path", ""),
            "patch_match_max_image_size": overrides.get("patch_match_max_image_size", ""),
            "patch_match_auto_source_count": overrides.get("patch_match_auto_source_count", ""),
            "patch_match_window_radius": overrides.get("patch_match_window_radius", ""),
            "patch_match_num_iterations": overrides.get("patch_match_num_iterations", ""),
            "patch_match_geom_consistency": overrides.get("patch_match_geom_consistency", ""),
            "patch_match_filter": overrides.get("patch_match_filter", ""),
            "stereo_fusion_max_image_size": overrides.get("stereo_fusion_max_image_size", ""),
            "stereo_fusion_min_num_pixels": overrides.get("stereo_fusion_min_num_pixels", ""),
            "stereo_fusion_max_reproj_error": overrides.get("stereo_fusion_max_reproj_error", ""),
            "stereo_fusion_max_depth_error": overrides.get("stereo_fusion_max_depth_error", ""),
            "stereo_fusion_max_normal_error": overrides.get("stereo_fusion_max_normal_error", ""),
            "mesh_backend_preference": overrides.get("mesh_backend_preference", ""),
            "poisson_depth": overrides.get("poisson_depth", ""),
            "poisson_trim": overrides.get("poisson_trim", ""),
            "poisson_point_weight": overrides.get("poisson_point_weight", ""),
            "mesh_crop_fused_to_roi": overrides.get("mesh_crop_fused_to_roi", ""),
            "support_min_count": support_rule.get("min_support_count", ""),
            "support_radius_px": support_rule.get("support_radius_px", ""),
            "support_depth_tolerance_m": support_rule.get("support_depth_tolerance_m", ""),
        }
        rows.append(row)
        save_json(
            scene_dir / "mesh_reproduction_parameters.json",
            {
                "scene_name": scene,
                "mesh_backend": row["mesh_backend"],
                "source_reference_mesh_path": row["source_reference_mesh_path"],
                "packaged_reference_mesh_path": row["packaged_reference_mesh_path"],
                "reference_build_manifest": build_manifest,
                "note": (
                    "This is the exact mesh recorded by reference_build_manifest.json and used to render the packaged "
                    "reference-depth manifest for this scene."
                ),
            },
        )
        log(f"Copied final mesh {scene}: {copied_mesh} ({source_mesh.stat().st_size / (1024**2):.1f} MiB)")
    write_csv(OUT / "tables" / "reference_mesh_reproduction_parameters.csv", rows)
    save_json(OUT / "reference_meshes" / "reference_mesh_index.json", rows)
    return rows


def copy_metrics_from_v1() -> None:
    if not V1.exists():
        raise FileNotFoundError(f"Previous package not found: {V1}")
    count = 0
    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            for method in METHODS:
                src_dir = V1 / "metrics" / mask / scene / method
                dst_dir = OUT / "metrics" / mask / scene / method
                if not (src_dir / "metrics_summary.json").exists():
                    raise FileNotFoundError(src_dir / "metrics_summary.json")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                count += 1
    log(f"Copied metrics JSON sets: {count}")


def assert_model_bundle_ready(scene: str, method: str, manifest_path: Path) -> None:
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = load_json(manifest_path)
    if REQUIRE_NATIVE_ALIGN and manifest.get("camera_frame_mode") != "probe_manifest_native_align":
        raise ValueError(f"Expected native-aligned model bundle for {scene}/{method}: {manifest_path}")
    if REQUIRE_NATIVE_ALIGN:
        if not manifest.get("strict_to_native_alignment"):
            raise ValueError(f"Missing strict_to_native_alignment for {scene}/{method}: {manifest_path}")
        views = manifest.get("views") or []
        if not views or "native_camera_to_world" not in views[0]:
            raise ValueError(f"Missing native_camera_to_world for {scene}/{method}: {manifest_path}")


def assert_metrics_reference_native_align_bundles() -> None:
    if not REQUIRE_NATIVE_ALIGN:
        return
    checked_metrics = 0
    checked_models = set()
    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            for method in METHODS:
                metrics_path = OUT / "metrics" / mask / scene / method / "metrics_summary.json"
                if not metrics_path.exists():
                    raise FileNotFoundError(metrics_path)
                metrics = load_json(metrics_path)
                if str(metrics.get("scene_name", "")) != scene:
                    raise ValueError(f"Metric scene mismatch in {metrics_path}")
                if str(metrics.get("method_name", "")) != method:
                    raise ValueError(f"Metric method mismatch in {metrics_path}")
                model_manifest_value = metrics.get("model_manifest")
                if not model_manifest_value:
                    raise ValueError(f"Missing model_manifest in {metrics_path}")
                model_manifest = Path(str(model_manifest_value))
                assert_model_bundle_ready(scene, method, model_manifest)
                checked_metrics += 1
                checked_models.add(str(model_manifest.resolve()))
    log(f"Validated native-aligned metric model manifests: {checked_metrics} metrics, {len(checked_models)} unique bundles")


def recompute_metrics_from_bundles(refs: Dict[str, Dict[str, Path]]) -> None:
    count = 0
    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            reference_manifest = Path(refs[mask][scene])
            for method in METHODS:
                model_manifest = FORMAL / scene / method / "bundle" / "split_manifest.json"
                adapter_manifest = FORMAL / scene / method / "depth_adapter_manifest.json"
                out_dir = OUT / "metrics" / mask / scene / method
                assert_model_bundle_ready(scene, method, model_manifest)
                if not adapter_manifest.exists():
                    raise FileNotFoundError(adapter_manifest)
                if out_dir.exists():
                    shutil.rmtree(out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    sys.executable,
                    str(TOOL_DIR / "evaluate_depth_reference.py"),
                    "--reference_manifest",
                    str(reference_manifest),
                    "--model_manifest",
                    str(model_manifest),
                    "--adapter_manifest",
                    str(adapter_manifest),
                    "--out_dir",
                    str(out_dir),
                    "--enable_agreement_metrics",
                ]
                subprocess.run(cmd, cwd=str(TOOL_DIR), check=True)
                count += 1
                if count % 10 == 0:
                    log(f"Recomputed metrics {count}/80")
    log(f"Recomputed metrics JSON sets: {count}")


def metric_mean_row(
    mask: str,
    method: str,
    display: str,
    delta: float,
    rows: Sequence[Dict[str, Any]],
    group: str = "",
) -> Dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
        return sum(values) / len(values) if values else math.nan

    return {
        "group_name": group,
        "mask_variant": mask,
        "method_id": method,
        "display_method_name": display,
        "threshold_m": delta,
        "num_scenes": len(rows),
        "FrontIntrusionRate_mean": mean("FrontIntrusionRate"),
        "FrontIntrusionMagnitude_mean": mean("FrontIntrusionMagnitude"),
        "TooDeepRate_mean": mean("TooDeepRate"),
        "DepthAgreementRate_mean": mean("DepthAgreementRate"),
        "MissingRate_mean": mean("MissingRate"),
        "AbsDepthError_Median_mean": mean("AbsDepthError_Median"),
    }


def collect_metrics_tables() -> None:
    all_rows: List[Dict[str, Any]] = []
    secondary_rows: List[Dict[str, Any]] = []
    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            for method in METHODS:
                path = OUT / "metrics" / mask / scene / method / "metrics_summary.json"
                data = load_json(path)
                sec = data.get("secondary_metrics", {})
                counts = data.get("counts", {})
                global_name = DISPLAY["global"][method]
                secondary_rows.append(
                    {
                        "mask_variant": mask,
                        "scene_name": scene,
                        "method_id": method,
                        "display_method_name": global_name,
                        "reference_valid_pixels": counts.get("reference_valid_pixels", ""),
                        "model_valid_on_reference_pixels": counts.get("model_valid_on_reference_pixels", ""),
                        "missing_pixels": counts.get("missing_pixels", ""),
                        "ModelValidOnReferenceRate": sec.get("ModelValidOnReferenceRate", ""),
                        "MissingRate": sec.get("MissingRate", ""),
                        "AbsDepthError_Mean": sec.get("AbsDepthError_Mean", ""),
                        "AbsDepthError_Median": sec.get("AbsDepthError_Median", ""),
                        "SignedDepthBias_Mean": sec.get("SignedDepthBias_Mean", ""),
                        "metrics_json": str(path),
                    }
                )
                for tm in data.get("threshold_metrics", []):
                    all_rows.append(
                        {
                            "mask_variant": mask,
                            "scene_name": scene,
                            "method_id": method,
                            "display_method_name": global_name,
                            "threshold_m": float(tm["threshold_m"]),
                            "FrontIntrusionRate": float(tm.get("FrontIntrusionRate", math.nan)),
                            "FrontIntrusionMagnitude": float(tm.get("FrontIntrusionMagnitude", math.nan)),
                            "TooDeepRate": float(tm.get("TooDeepRate", math.nan)),
                            "DepthAgreementRate": float(tm.get("DepthAgreementRate", math.nan)),
                            "MissingRate": float(sec.get("MissingRate", math.nan)),
                            "AbsDepthError_Mean": float(sec.get("AbsDepthError_Mean", math.nan)),
                            "AbsDepthError_Median": float(sec.get("AbsDepthError_Median", math.nan)),
                            "SignedDepthBias_Mean": float(sec.get("SignedDepthBias_Mean", math.nan)),
                            "metrics_json": str(path),
                        }
                    )
    write_csv(OUT / "tables" / "all_threshold_metrics_9thr_mask_and_nomask.csv", all_rows)
    write_csv(OUT / "tables" / "secondary_metrics_by_scene_method_mask.csv", secondary_rows)

    macro: List[Dict[str, Any]] = []
    for mask in ["masked", "nomask"]:
        for method in METHODS:
            for delta in THRESHOLDS:
                values = [row for row in all_rows if row["mask_variant"] == mask and row["method_id"] == method and abs(float(row["threshold_m"]) - delta) < 1e-9]
                macro.append(metric_mean_row(mask, method, DISPLAY["global"][method], delta, values))
    write_csv(OUT / "tables" / "macro_average_by_method_threshold_mask.csv", macro)

    for group, methods in GROUPS.items():
        group_rows: List[Dict[str, Any]] = []
        group_macro: List[Dict[str, Any]] = []
        for row in all_rows:
            if row["method_id"] not in methods:
                continue
            group_row = dict(row)
            group_row["group_name"] = group
            group_row["display_method_name"] = DISPLAY[group][row["method_id"]]
            group_rows.append(group_row)
        for mask in ["masked", "nomask"]:
            for method in methods:
                for delta in THRESHOLDS:
                    values = [row for row in group_rows if row["mask_variant"] == mask and row["method_id"] == method and abs(float(row["threshold_m"]) - delta) < 1e-9]
                    group_macro.append(metric_mean_row(mask, method, DISPLAY[group][method], delta, values, group))
        write_csv(OUT / "tables" / f"{group}_scene_level_9thr_mask_and_nomask.csv", group_rows)
        write_csv(OUT / "tables" / f"{group}_macro_average_9thr_mask_and_nomask.csv", group_macro)


def selected_views_for_scene(reference_manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    views = list(reference_manifest["views"])
    count = len(views)
    if count <= 10:
        return views
    selected_indices = []
    seen = set()
    for idx in (int(math.floor(k * count / 10.0)) for k in range(10)):
        idx = min(idx, count - 1)
        if idx not in seen:
            selected_indices.append(idx)
            seen.add(idx)
    while len(selected_indices) < min(10, count):
        for idx in range(count):
            if idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
                break
    return [views[idx] for idx in selected_indices[:10]]


def model_payload(scene: str, method: str) -> Dict[str, Any]:
    method_dir = FORMAL / scene / method
    return {
        "method_id": method,
        "method_dir": method_dir,
        "bundle_manifest_path": method_dir / "bundle" / "split_manifest.json",
        "adapter_manifest_path": method_dir / "depth_adapter_manifest.json",
        "bundle_manifest": load_json(method_dir / "bundle" / "split_manifest.json"),
        "adapter_manifest": load_json(method_dir / "depth_adapter_manifest.json"),
    }


def classify_custom(
    ref_valid: np.ndarray,
    model_valid: np.ndarray,
    ref_depth: np.ndarray,
    model_depth: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, Dict[str, int]]:
    rgb = np.ones((*ref_valid.shape, 3), dtype=np.float32)
    valid_joint = ref_valid & model_valid
    missing = ref_valid & ~model_valid
    front = valid_joint & (model_depth < ref_depth - delta)
    deep = valid_joint & (model_depth > ref_depth + delta)
    correct = valid_joint & ~(front | deep)
    rgb[correct] = CLASS_COLORS["correct"]
    rgb[front] = CLASS_COLORS["front_intrusion"]
    rgb[deep] = CLASS_COLORS["too_deep"]
    rgb[missing] = CLASS_COLORS["missing"]
    return rgb, {
        "reference_valid_pixels": int(np.count_nonzero(ref_valid)),
        "model_valid_pixels": int(np.count_nonzero(valid_joint)),
        "correct_pixels": int(np.count_nonzero(correct)),
        "too_shallow_pixels": int(np.count_nonzero(front)),
        "too_deep_pixels": int(np.count_nonzero(deep)),
        "missing_model_pixels": int(np.count_nonzero(missing)),
    }


def load_view_assets(
    ref_root: Path,
    ref_view: Dict[str, Any],
    gt_root: Path,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_npz = _load_npz(ref_root, str(ref_view["npz_file"]))
    ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
    ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
    gt = _load_gt_image(gt_root / str(ref_view["image_name"]))
    gt = _resize_rgb_image(gt, width=int(ref_view["width"]), height=int(ref_view["height"]))
    ref_rgb = _depth_to_rgb(ref_depth, ref_valid, depth_min, depth_max, cmap_name=DEPTH_CMAP, invalid_rgb=INVALID_RGB)
    return ref_depth, ref_valid, gt, ref_rgb


def load_model_depth(payload: Dict[str, Any], image_name: str) -> tuple[np.ndarray, np.ndarray]:
    views_by_name = {str(view["image_name"]): view for view in payload["bundle_manifest"]["views"]}
    view = views_by_name[image_name]
    model_npz = _load_npz(payload["bundle_manifest_path"].parent, str(view["npz_file"]))
    raw = np.asarray(model_npz["depth"], dtype=np.float64)
    opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
    semantics = str(payload["adapter_manifest"]["depth_semantics"])
    validity = payload["adapter_manifest"]["validity_rule"]
    depth = _raw_depth_to_metric_camera_z(raw, depth_semantics=semantics)
    valid = _make_model_valid_mask(
        depth,
        opacity,
        depth_min=float(validity.get("depth_min", 1e-6)),
        opacity_threshold=float(validity.get("opacity_threshold", 0.5)),
    )
    return depth, valid


def render_single_method(
    path: Path,
    gt: np.ndarray,
    ref_rgb: np.ndarray,
    model_rgb: np.ndarray,
    class_rgb: np.ndarray,
    display_name: str,
    scene: str,
    mask: str,
    delta: float,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 2.2), dpi=DPI_STRIP)
    for ax, image, title in zip(axes, [gt, ref_rgb, model_rgb, class_rgb], ["GT", "Reference", "Model depth", "Error"]):
        ax.imshow(image)
        ax.set_title(title, fontsize=8, pad=2)
        ax.axis("off")
    fig.suptitle(f"{display_name} | {scene} | {mask} | {delta:g}m", fontsize=9, y=0.98)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.02, wspace=0.015)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI_STRIP)
    plt.close(fig)


def render_per_view(
    path: Path,
    gt: np.ndarray,
    ref_rgb: np.ndarray,
    method_images: Sequence[Dict[str, Any]],
    group: str,
    scene: str,
    mask: str,
    delta: float,
) -> None:
    ncols = len(method_images) + 2
    fig, axes = plt.subplots(2, ncols, figsize=(2.05 * ncols, 3.65), dpi=DPI_CONTACT)
    axes[0, 0].imshow(gt)
    axes[0, 0].set_title("GT", fontsize=9, pad=2)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")
    axes[0, 1].imshow(ref_rgb)
    axes[0, 1].set_title("Reference", fontsize=9, pad=2)
    axes[0, 1].axis("off")
    axes[1, 1].axis("off")
    for idx, item in enumerate(method_images):
        col = idx + 2
        axes[0, col].imshow(item["model_rgb"])
        axes[0, col].set_title(item["display"], fontsize=9, pad=2)
        axes[0, col].axis("off")
        axes[1, col].imshow(item["class_rgb"])
        axes[1, col].axis("off")
    fig.suptitle(f"{group} | {scene} | {mask} | {delta:g}m", fontsize=10, y=0.995)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.003, right=0.997, top=0.90, bottom=0.09, wspace=0.01, hspace=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI_CONTACT)
    plt.close(fig)


def render_contact(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    methods: Sequence[Dict[str, Any]],
    group: str,
    scene: str,
    mask: str,
    delta: float,
) -> None:
    ncols = 2 + 2 * len(methods)
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.62 * ncols, 1.08 * nrows), dpi=DPI_CONTACT)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    headers = ["GT", "Reference"]
    for method in methods:
        headers.extend([method["display"], "Error"])
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=8, pad=2)
    for row_idx, row in enumerate(rows):
        axes[row_idx, 0].imshow(row["gt"])
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].imshow(row["ref_rgb"])
        axes[row_idx, 1].axis("off")
        col = 2
        for method in methods:
            payload = row["methods"][method["method_id"]]
            axes[row_idx, col].imshow(payload["model_rgb"])
            axes[row_idx, col].axis("off")
            axes[row_idx, col + 1].imshow(payload["class_rgb"])
            axes[row_idx, col + 1].axis("off")
            col += 2
    fig.suptitle(f"{group} | {scene} | {mask} | {delta:g}m", fontsize=10, y=0.998)
    fig.legend(handles=LEGEND, loc="lower center", ncol=5, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.002, right=0.998, top=0.965, bottom=0.035, wspace=0.003, hspace=0.003)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI_CONTACT)
    plt.close(fig)


def render_visualizations(refs: Dict[str, Dict[str, Path]]) -> None:
    records: List[Dict[str, Any]] = []
    all_stats: List[Dict[str, Any]] = []
    selected_index: Dict[str, List[Dict[str, str]]] = {}
    total_jobs = 2 * len(SCENES) * len(THRESHOLDS) * len(GROUPS)
    job = 0
    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            ref_path = Path(refs[mask][scene])
            ref_manifest = load_json(ref_path)
            ref_root = ref_path.parent
            gt_root = _resolve_gt_root(ref_manifest, override_root="", override_images_dir="")
            selected = selected_views_for_scene(ref_manifest)
            selected_index[f"{mask}::{scene}"] = [{"view_id": str(view["view_id"]), "image_name": str(view["image_name"])} for view in selected]
            depth_min, depth_max = _compute_depth_display_range(ref_path, selected, low_pct=2.0, high_pct=98.0)
            payloads = {method: model_payload(scene, method) for method in METHODS}
            cache: Dict[str, Any] = {}
            for ref_view in selected:
                image_name = str(ref_view["image_name"])
                ref_depth, ref_valid, gt, ref_rgb = load_view_assets(ref_root, ref_view, gt_root, depth_min, depth_max)
                cache[image_name] = {"view": ref_view, "ref_depth": ref_depth, "ref_valid": ref_valid, "gt": gt, "ref_rgb": ref_rgb, "models": {}}
                for method in METHODS:
                    depth, valid = load_model_depth(payloads[method], image_name)
                    model_rgb = _depth_to_rgb(depth, valid, depth_min, depth_max, cmap_name=DEPTH_CMAP, invalid_rgb=INVALID_RGB)
                    cache[image_name]["models"][method] = {"depth": depth, "valid": valid, "model_rgb": model_rgb}
            for delta in THRESHOLDS:
                unique_single_paths: Dict[tuple[str, str], str] = {}
                for group, method_ids in GROUPS.items():
                    job += 1
                    group_dir = OUT / "visualizations" / mask / scene / thr_dir(delta) / group
                    per_view_dir = group_dir / "per_view"
                    methods_meta = [{"method_id": method, "display": DISPLAY[group][method], "method_dir": str(FORMAL / scene / method)} for method in method_ids]
                    contact_rows = []
                    stats_rows = []
                    manifest_views = []
                    for ref_view in selected:
                        image_name = str(ref_view["image_name"])
                        view_id = str(ref_view["view_id"])
                        stem = f"{view_id}_{Path(image_name).stem}"
                        item = cache[image_name]
                        row_methods: Dict[str, Dict[str, np.ndarray]] = {}
                        per_view_methods = []
                        for method in method_ids:
                            model = item["models"][method]
                            class_rgb, counts = classify_custom(item["ref_valid"], model["valid"], item["ref_depth"], model["depth"], delta)
                            display = DISPLAY[group][method]
                            unique_key = (method, image_name)
                            if unique_key not in unique_single_paths:
                                method_slug = safe_name(DISPLAY["global"][method])
                                single_path = OUT / "visualizations" / mask / scene / thr_dir(delta) / "per_method_unique" / method_slug / f"{stem}.png"
                                render_single_method(
                                    single_path,
                                    item["gt"],
                                    item["ref_rgb"],
                                    model["model_rgb"],
                                    class_rgb,
                                    DISPLAY["global"][method],
                                    scene,
                                    mask,
                                    delta,
                                )
                                unique_single_paths[unique_key] = str(single_path)
                            single_path = Path(unique_single_paths[unique_key])
                            row_methods[method] = {"model_rgb": model["model_rgb"], "class_rgb": class_rgb}
                            per_view_methods.append({"method_id": method, "display": display, "model_rgb": model["model_rgb"], "class_rgb": class_rgb})
                            reference_count = max(1, counts["reference_valid_pixels"])
                            stats = {
                                "mask_variant": mask,
                                "scene_name": scene,
                                "threshold_m": delta,
                                "group_name": group,
                                "view_id": view_id,
                                "image_name": image_name,
                                "method_id": method,
                                "display_method_name": display,
                                **counts,
                                "correct_rate": counts["correct_pixels"] / reference_count,
                                "FrontIntrusionRate": counts["too_shallow_pixels"] / reference_count,
                                "TooDeepRate": counts["too_deep_pixels"] / reference_count,
                                "MissingRate": counts["missing_model_pixels"] / reference_count,
                                "per_method_png": str(single_path),
                            }
                            stats_rows.append(stats)
                            all_stats.append(stats)
                        per_view_path = per_view_dir / f"{stem}.png"
                        render_per_view(per_view_path, item["gt"], item["ref_rgb"], per_view_methods, group, scene, mask, delta)
                        contact_rows.append({"gt": item["gt"], "ref_rgb": item["ref_rgb"], "methods": row_methods})
                        manifest_views.append({"view_id": view_id, "image_name": image_name, "per_view_png": str(per_view_path)})
                    contact_path = group_dir / "contact_sheet.png"
                    render_contact(contact_path, contact_rows, methods_meta, group, scene, mask, delta)
                    write_csv(group_dir / "stats.csv", stats_rows)
                    save_json(
                        group_dir / "manifest.json",
                        {
                            "mask_variant": mask,
                            "scene_name": scene,
                            "threshold_m": delta,
                            "group_name": group,
                            "reference_manifest": str(ref_path),
                            "gt_root": str(gt_root),
                            "selected_view_rule": "scene-local equal-spaced indices: floor(k*N/10), k=0..9, over that scene held-out manifest order",
                            "selected_views": manifest_views,
                            "methods": methods_meta,
                            "contact_sheet_png": str(contact_path),
                        },
                    )
                    records.append(
                        {
                            "mask_variant": mask,
                            "scene_name": scene,
                            "threshold_m": delta,
                            "group_name": group,
                            "contact_sheet_png": str(contact_path),
                            "out_dir": str(group_dir),
                        }
                    )
                    if job % 10 == 0 or job == total_jobs:
                        log(f"Visualizations {job}/{total_jobs}: {mask} {scene} {delta:g} {group}")
    write_csv(OUT / "visualizations" / "visualization_manifest.csv", records)
    write_csv(OUT / "visualizations" / "all_per_view_method_stats.csv", all_stats)
    save_json(OUT / "visualizations" / "selected_views_by_mask_scene.json", selected_index)


def plot_curves() -> List[Path]:
    rows: List[Dict[str, str]] = []
    for group in GROUPS:
        with (OUT / "tables" / f"{group}_macro_average_9thr_mask_and_nomask.csv").open("r", encoding="utf-8-sig", newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    metrics = [
        ("FrontIntrusionRate_mean", "Front intrusion rate (lower is better)"),
        ("TooDeepRate_mean", "Too-deep rate (lower is better)"),
        ("DepthAgreementRate_mean", "Depth agreement rate (higher is better)"),
    ]
    xlabels = [f"{threshold:g}m" for threshold in THRESHOLDS]
    plot_paths: List[Path] = []
    for group, method_ids in GROUPS.items():
        for mask in ["masked", "nomask"]:
            subset = [row for row in rows if row["group_name"] == group and row["mask_variant"] == mask]
            for metric_key, ylabel in metrics:
                fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
                for method in method_ids:
                    display = DISPLAY[group][method]
                    values = []
                    for threshold in THRESHOLDS:
                        matches = [row for row in subset if row["method_id"] == method and abs(float(row["threshold_m"]) - threshold) < 1e-9]
                        values.append(float(matches[0][metric_key]) if matches else math.nan)
                    ax.plot(range(len(THRESHOLDS)), values, marker="o", linewidth=1.8, label=display)
                ax.set_xticks(range(len(THRESHOLDS)), xlabels)
                ax.set_xlabel("Threshold (equal-spaced labels)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"{group} macro-average | {mask}")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=8, ncol=2)
                fig.tight_layout()
                path = OUT / "plots" / f"{group}_{mask}_{metric_key}_curve_equal_spaced.png"
                fig.savefig(path)
                plt.close(fig)
                plot_paths.append(path)
    return plot_paths


def copy_code_snapshot() -> None:
    dst = OUT / "code_snapshot" / "tools" / "geometric_repeatability"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(TOOL_DIR, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "artifacts"))
    if (REPO / "AGENTS.md").exists():
        shutil.copy2(REPO / "AGENTS.md", OUT / "code_snapshot" / "AGENTS.md")
    notes = OUT / "protocol_notes"
    for src in [
        FORMAL / "MESH_FAILURE_DIAGNOSIS.md",
        FORMAL / "mesh_failure_diagnosis_summary.csv",
        MESHFIX / "MESH_FIX_ROICROP_STATUS.md",
    ]:
        if src.exists():
            shutil.copy2(src, notes / src.name)


def completeness_check() -> Dict[str, Any]:
    metrics_json_count = len(list((OUT / "metrics").rglob("metrics_summary.json")))
    per_method_png_count = len(list((OUT / "visualizations").glob("*/*/*/per_method_unique/*/*.png")))
    contact_sheet_count = len(list((OUT / "visualizations").glob("*/*/*/*/contact_sheet.png")))
    per_view_group_png_count = len(list((OUT / "visualizations").glob("*/*/*/*/per_view/*.png")))
    plot_count = len(list((OUT / "plots").glob("*.png")))
    final_mesh_count = len(list((OUT / "reference_meshes").glob("*/*.ply")))
    rows: List[Dict[str, Any]] = [
        {"check_name": "metrics_json_count", "expected": 80, "actual": metrics_json_count, "pass": metrics_json_count == 80},
        {"check_name": "per_method_png_count", "expected": 7200, "actual": per_method_png_count, "pass": per_method_png_count == 7200},
        {"check_name": "contact_sheet_count", "expected": 180, "actual": contact_sheet_count, "pass": contact_sheet_count == 180},
        {"check_name": "per_view_group_png_count", "expected": 1800, "actual": per_view_group_png_count, "pass": per_view_group_png_count == 1800},
        {"check_name": "plot_count", "expected": 12, "actual": plot_count, "pass": plot_count == 12},
        {"check_name": "final_reference_mesh_count", "expected": 5, "actual": final_mesh_count, "pass": final_mesh_count == 5},
    ]
    selected = load_json(OUT / "visualizations" / "selected_views_by_mask_scene.json")
    for key, views in selected.items():
        scene = key.split("::", 1)[1]
        reference = load_json(MASKED_REFS[scene])
        expected = min(10, len(reference["views"]))
        rows.append({"check_name": f"selected_views_{key}", "expected": expected, "actual": len(views), "pass": len(views) == expected})
    write_csv(OUT / "tables" / "completeness_check.csv", rows)
    return {
        "metrics_json_count": metrics_json_count,
        "per_method_png_count": per_method_png_count,
        "contact_sheet_count": contact_sheet_count,
        "per_view_group_png_count": per_view_group_png_count,
        "plot_count": plot_count,
        "final_reference_mesh_count": final_mesh_count,
        "all_checks_pass": all(bool(row["pass"]) for row in rows),
    }


def write_readme(completeness: Dict[str, Any]) -> None:
    readme = f"""# Depth-Reference Geometry Evaluation Package v2

Package root: `{OUT}`

This package is prepared for research/writing assistance. It contains mask/no-mask depth-reference metrics and visualizations for 5 scenes, 8 methods, and 9 threshold levels.

## Evaluation Goal

This is a **reference-depth-based geometry consistency** and **front-intrusion analysis** package. It compares each method's rendered thermal-model depth against a training-side MVS/mesh reference on held-out views.

Do **not** describe this as absolute 3D ground-truth accuracy or ground-truth depth accuracy. The safer wording is: held-out geometry consistency against a training-only reference surface.

## Method Display Names

SOTA5 group:

- `Ours`
- `Thermal3D-GS`
- `ThermalGaussian-MFTG`
- `ThermalGaussian-MSMG`
- `ThermalGaussian-OMMG`

Ablation4 group:

- `Baseline`
- `+SSP`
- `+STT`
- `Full model`

Note: `Ours` in SOTA5 and `Full model` in Ablation4 correspond to the same internal method id, `Ours_M01_full`. The package uses group-specific display names to match the paper tables.

## Metrics

Let `D_ref` be the rendered reference mesh depth, `D_model` be the model-rendered depth, `delta` be the threshold, and `M_ref` be the reference-valid pixel set.

- `FrontIntrusionRate@delta = count(D_model < D_ref - delta) / count(M_ref)`. Lower is better.
- `TooDeepRate@delta = count(D_model > D_ref + delta) / count(M_ref)`. Lower is better.
- `DepthAgreementRate@delta = count(abs(D_model - D_ref) <= delta) / count(M_ref)`. Higher is better.
- `MissingRate = count(model invalid on M_ref) / count(M_ref)`. Lower is better.
- `FrontIntrusionMagnitude@delta` is the mean positive amount `D_ref - D_model` over front-intrusion pixels.

Depths are metric camera-z values in meters. Model validity uses each method's frozen depth adapter and opacity threshold.

## Thresholds

All metrics and visualizations are provided for 9 thresholds:

`0.10 / 0.25 / 0.50 / 1 / 2 / 5 / 10 / 20 / 30 m`

Curve plots use equal-spaced threshold labels so the small thresholds remain visible.

## Mask Variants

- `masked`: uses finite reference depth plus the reference support/validity mask.
- `nomask`: uses finite positive mesh-hit reference depth only.

Recommended use: use `masked` for conservative numeric reporting, and use `nomask` as qualitative/sensitivity support when masks make visual panels look unnecessarily incomplete.

## Mesh Reference Note

The five scenes do not need to be reconstructed with identical MVS/mesh parameters. Scene-specific stable reference construction is used to obtain a reliable training-side reference surface for each scene. Since the final reference for a scene is shared by all evaluated methods, parameter differences across scenes do not affect fairness among methods within that scene. In writing, mention this briefly as robust scene-wise reference construction rather than emphasizing it as a caveat.

The exact final meshes used by the packaged reference-depth manifests are included in:

`{OUT / 'reference_meshes'}`

For reproducibility, each scene folder contains the final `.ply` mesh, `reference_build_manifest.json`, `reference_roi.json`, and `mesh_reproduction_parameters.json`. A compact cross-scene parameter table is available at:

`{OUT / 'tables' / 'reference_mesh_reproduction_parameters.csv'}`

## Directory Guide

- Metrics JSON: `{OUT / 'metrics'}`
- Tables: `{OUT / 'tables'}`
- Curves: `{OUT / 'plots'}`
- Visualizations: `{OUT / 'visualizations'}`
- Reference manifests: `{OUT / 'reference_manifests'}`
- Protocol notes: `{OUT / 'protocol_notes'}`
- Final reference meshes and mesh parameters: `{OUT / 'reference_meshes'}`
- Full depth-evaluation code snapshot: `{OUT / 'code_snapshot' / 'tools' / 'geometric_repeatability'}`

## Camera-Frame Alignment

Model depth bundles are read from:

`{FORMAL}`

For this package, every model bundle is expected to be rendered with `camera_frame_mode=probe_manifest_native_align`. The strict held-out probe cameras are aligned into each model's native COLMAP frame before rendering. This avoids the previously observed GT/SOTA view mismatch.

PVpanel, Road, and TransmissionTower references are read from the repaired mesh reference root:

`{MESHFIX}`

## Main Tables

- Full 9-threshold table: `{OUT / 'tables' / 'all_threshold_metrics_9thr_mask_and_nomask.csv'}`
- Macro table: `{OUT / 'tables' / 'macro_average_by_method_threshold_mask.csv'}`
- SOTA5 macro table: `{OUT / 'tables' / 'SOTA5_macro_average_9thr_mask_and_nomask.csv'}`
- Ablation4 macro table: `{OUT / 'tables' / 'Ablation4_macro_average_9thr_mask_and_nomask.csv'}`
- Completeness check: `{OUT / 'tables' / 'completeness_check.csv'}`

## Visualization Layout

Each visualization folder is organized as:

`visualizations/<mask>/<scene>/<threshold>/<group>/`

where `group` is either `SOTA5` or `Ablation4`.

Each group folder contains:

- `contact_sheet.png`: 10 held-out views, method names only at the top, compact spacing.
- `per_view/*.png`: one image per selected held-out view, comparing all methods in that group.
- `manifest.json`: exact scene/mask/threshold/method/view mapping.
- `stats.csv`: pixel counts and rates for the displayed views.

Single-method panels are stored once per unique method at:

`visualizations/<mask>/<scene>/<threshold>/per_method_unique/<method>/*.png`

This avoids duplicating the same internal model twice when `Ours_M01_full` is displayed as `Ours` in SOTA5 and `Full model` in Ablation4.

## Held-Out View Sampling

For each scene independently, 10 held-out views are selected by scene-local equal spacing over the reference/probe manifest order: `floor(k * N / 10)`, `k=0..9`, where `N` is the number of held-out views in that scene. The same selected views are reused for all masks, thresholds, and methods in that scene.

## Completeness Summary

- Metrics JSON: `{completeness['metrics_json_count']}` / 80
- Single-method per-view PNGs: `{completeness['per_method_png_count']}` / 7200
- Contact sheets: `{completeness['contact_sheet_count']}` / 180
- Per-view group PNGs: `{completeness['per_view_group_png_count']}` / 1800
- Curve plots: `{completeness['plot_count']}` / 12
- Final reference meshes: `{completeness['final_reference_mesh_count']}` / 5

## Suggested Writing Use

- For a main-text method comparison: start from `SOTA5_macro_average_9thr_mask_and_nomask.csv` and the SOTA5 curve plots.
- For ablation discussion: use `Ablation4_macro_average_9thr_mask_and_nomask.csv` and Ablation4 curve plots.
- For qualitative figures: inspect `nomask` first for visual clarity, then verify the corresponding `masked` panel for conservative support.
- Avoid overclaiming; describe the result as reference-based geometry consistency and front-intrusion suppression.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def package_summary(
    completeness: Dict[str, Any],
    reference_rows: Sequence[Dict[str, Any]],
    mesh_rows: Sequence[Dict[str, Any]],
    plot_paths: Sequence[Path],
) -> None:
    save_json(
        OUT / "package_summary.json",
        {
            "package_root": str(OUT),
            "zip_path": str(ZIP),
            "model_bundle_root": str(FORMAL),
            "meshfix_reference_root": str(MESHFIX),
            "metrics_source": "recomputed_from_selected_model_bundles_and_references" if RECOMPUTE_METRICS else str(V1),
            "require_native_align": REQUIRE_NATIVE_ALIGN,
            "scenes": SCENES,
            "thresholds_m": THRESHOLDS,
            "mask_variants": ["masked", "nomask"],
            "groups": {
                "SOTA5": [{"method_id": method, "display_method_name": DISPLAY["SOTA5"][method]} for method in SOTA5],
                "Ablation4": [{"method_id": method, "display_method_name": DISPLAY["Ablation4"][method]} for method in ABLATION4],
            },
            "reference_valid_area_summary": list(reference_rows),
            "reference_mesh_reproduction_parameters": list(mesh_rows),
            "completeness": completeness,
            "plot_paths": [str(path) for path in plot_paths],
        },
    )


def mark_deprecated_targets() -> None:
    if not DEPRECATED_TARGETS:
        return
    marker_text = f"""# Deprecated Depth-Reference Result Package

This package is deprecated and should not be used for paper writing or figures.

Reason: it was generated before the native camera-frame alignment rerun, so GT/reference and SOTA model rendered views may be inconsistent.

Replacement package:

`{OUT}`

Replacement zip:

`{ZIP}`
"""
    for target in DEPRECATED_TARGETS:
        if target.suffix.lower() == ".zip":
            sidecar = target.with_name(target.name + ".DEPRECATED.txt")
            sidecar.write_text(marker_text, encoding="utf-8")
            log(f"Wrote deprecated sidecar: {sidecar}")
        elif target.exists() and target.is_dir():
            marker = target / "DEPRECATED_DO_NOT_USE.md"
            marker.write_text(marker_text, encoding="utf-8")
            log(f"Wrote deprecated marker: {marker}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            sidecar = target.with_name(target.name + ".DEPRECATED.txt")
            sidecar.write_text(marker_text, encoding="utf-8")
            log(f"Wrote deprecated sidecar for missing target: {sidecar}")


def make_zip() -> None:
    if ZIP.exists():
        ZIP.unlink()
    shutil.make_archive(str(OUT), "zip", root_dir=str(OUT))


def main() -> None:
    configure_from_args(parse_args())
    started = time.time()
    log("Preparing output")
    prepare_output()
    log("Copying full depth-evaluation code snapshot")
    copy_code_snapshot()
    log("Preparing masked/no-mask references")
    refs, reference_rows = prepare_references()
    log("Copying final reference meshes and mesh parameters")
    mesh_rows = copy_final_reference_meshes()
    if RECOMPUTE_METRICS:
        log("Recomputing metrics from selected references and model bundles")
        recompute_metrics_from_bundles(refs)
    else:
        log("Copying metrics from previous package")
        copy_metrics_from_v1()
    if REQUIRE_NATIVE_ALIGN:
        log("Validating native-aligned metric model manifests")
        assert_metrics_reference_native_align_bundles()
    log("Collecting metric tables")
    collect_metrics_tables()
    log("Plotting curves")
    plot_paths = plot_curves()
    log("Rendering full visualization set")
    render_visualizations(refs)
    log("Running completeness check")
    completeness = completeness_check()
    log(f"Completeness: {completeness}")
    log("Writing README and package summary")
    write_readme(completeness)
    package_summary(completeness, reference_rows, mesh_rows, plot_paths)
    log("Creating zip")
    make_zip()
    log("Marking deprecated package targets")
    mark_deprecated_targets()
    log(f"DONE in {(time.time() - started) / 60.0:.1f} min")
    log(f"OUT={OUT}")
    log(f"ZIP={ZIP}")


if __name__ == "__main__":
    main()
