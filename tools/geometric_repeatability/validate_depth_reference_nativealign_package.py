from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


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
MESHFIX_SCENES = {"PVpanel", "Road", "TransmissionTower"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a native-align depth-reference result package.")
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--formal_root", required=True)
    parser.add_argument("--meshfix_root", required=True)
    parser.add_argument("--zip", default="", help="Zip path to test; defaults to <package_root>.zip")
    parser.add_argument("--check_zip", action="store_true", help="Run zipfile.testzip on the package zip.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def norm(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def under(path: str | Path, root: str | Path) -> bool:
    path_norm = norm(path)
    root_norm = norm(root)
    return path_norm == root_norm or path_norm.startswith(root_norm + os.sep)


def add_error(errors: List[str], message: str) -> None:
    errors.append(message)


def validate_bundle(model_manifest: Path, errors: List[str]) -> None:
    if not model_manifest.exists():
        add_error(errors, f"model manifest missing: {model_manifest}")
        return
    manifest = load_json(model_manifest)
    if manifest.get("camera_frame_mode") != "probe_manifest_native_align":
        add_error(errors, f"bad camera_frame_mode in {model_manifest}: {manifest.get('camera_frame_mode')!r}")
    alignment = manifest.get("strict_to_native_alignment")
    if not isinstance(alignment, dict) or alignment.get("common_count", 0) <= 0:
        add_error(errors, f"missing/empty strict_to_native_alignment: {model_manifest}")
    transform = alignment.get("strict_to_native_transform") if isinstance(alignment, dict) else None
    if not isinstance(transform, list) or len(transform) != 4:
        add_error(errors, f"invalid strict_to_native_transform: {model_manifest}")
    views = manifest.get("views") or []
    if not views:
        add_error(errors, f"bundle has no views: {model_manifest}")
        return
    missing_native = sum(1 for view in views if "native_camera_to_world" not in view)
    if missing_native:
        add_error(errors, f"{missing_native} views missing native_camera_to_world: {model_manifest}")


def validate_package(package_root: Path, formal_root: Path, meshfix_root: Path, zip_path: Path, check_zip: bool) -> Dict[str, Any]:
    errors: List[str] = []
    summary_path = package_root / "package_summary.json"
    if not summary_path.exists():
        add_error(errors, f"package_summary.json missing: {summary_path}")
        return {"pass": False, "errors": errors}

    summary = load_json(summary_path)
    completeness = summary.get("completeness") or {}
    if not summary.get("require_native_align"):
        add_error(errors, "package_summary require_native_align is not true")
    if summary.get("metrics_source") != "recomputed_from_selected_model_bundles_and_references":
        add_error(errors, "metrics_source is not recomputed_from_selected_model_bundles_and_references")
    if not completeness.get("all_checks_pass"):
        add_error(errors, "package_summary completeness all_checks_pass is not true")
    if not under(summary.get("model_bundle_root", ""), formal_root):
        add_error(errors, "package_summary model_bundle_root does not match formal_root")
    if not under(summary.get("meshfix_reference_root", ""), meshfix_root):
        add_error(errors, "package_summary meshfix_reference_root does not match meshfix_root")

    counts = {
        "metrics_summary": len(list((package_root / "metrics").rglob("metrics_summary.json"))),
        "contact_sheet": len(list((package_root / "visualizations").rglob("contact_sheet.png"))),
        "per_view": sum(1 for path in (package_root / "visualizations").rglob("*.png") if f"{os.sep}per_view{os.sep}" in str(path)),
        "per_method": sum(
            1 for path in (package_root / "visualizations").rglob("*.png") if f"{os.sep}per_method_unique{os.sep}" in str(path)
        ),
        "plots": len(list((package_root / "plots").glob("*.png"))),
        "final_reference_meshes": len(list((package_root / "reference_meshes").glob("*/*.ply"))),
        "zip_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
    }
    expected_counts = {
        "metrics_summary": 80,
        "contact_sheet": 180,
        "per_view": 1800,
        "per_method": 7200,
        "plots": 12,
        "final_reference_meshes": 5,
    }
    for key, expected in expected_counts.items():
        if counts[key] != expected:
            add_error(errors, f"{key} count {counts[key]} != {expected}")
    if check_zip and (not zip_path.exists() or counts["zip_bytes"] <= 0):
        add_error(errors, f"zip missing or empty: {zip_path}")

    reference_rows = summary.get("reference_valid_area_summary") or []
    masked_refs = {str(row.get("scene_name")): str(row.get("masked_reference_manifest")) for row in reference_rows}
    nomask_refs = {str(row.get("scene_name")): str(row.get("nomask_reference_manifest")) for row in reference_rows}
    for scene in SCENES:
        masked_ref_value = masked_refs.get(scene, "")
        nomask_ref_value = nomask_refs.get(scene, "")
        if not masked_ref_value:
            add_error(errors, f"masked reference path missing for {scene}")
            continue
        if not nomask_ref_value:
            add_error(errors, f"nomask reference path missing for {scene}")
            continue
        masked_ref = Path(masked_ref_value)
        nomask_ref = Path(nomask_ref_value)
        if not masked_ref.exists():
            add_error(errors, f"masked reference missing for {scene}: {masked_ref}")
        if not nomask_ref.exists():
            add_error(errors, f"nomask reference missing for {scene}: {nomask_ref}")
        if scene in MESHFIX_SCENES and not under(masked_ref, meshfix_root):
            add_error(errors, f"{scene} masked reference is not under meshfix_root: {masked_ref}")

    metric_counter: Counter[tuple[str, str]] = Counter()
    method_counter: Counter[str] = Counter()
    model_manifests = set()
    for metrics_path in (package_root / "metrics").rglob("metrics_summary.json"):
        rel = metrics_path.relative_to(package_root / "metrics").parts
        if len(rel) < 4:
            add_error(errors, f"unexpected metrics path: {metrics_path}")
            continue
        mask, scene, method = rel[0], rel[1], rel[2]
        metrics = load_json(metrics_path)
        metric_counter[(mask, scene)] += 1
        method_counter[method] += 1
        if scene not in SCENES or method not in METHODS or mask not in {"masked", "nomask"}:
            add_error(errors, f"unexpected metrics identity: {metrics_path}")
        if metrics.get("scene_name") != scene or metrics.get("method_name") != method:
            add_error(errors, f"metrics payload identity mismatch: {metrics_path}")
        model_manifest_value = metrics.get("model_manifest")
        if not model_manifest_value:
            add_error(errors, f"model_manifest missing in {metrics_path}")
            continue
        model_manifest = Path(str(model_manifest_value))
        model_manifests.add(norm(model_manifest))
        if not under(model_manifest, formal_root):
            add_error(errors, f"model_manifest not under formal_root: {model_manifest}")
        validate_bundle(model_manifest, errors)

    for mask in ["masked", "nomask"]:
        for scene in SCENES:
            if metric_counter[(mask, scene)] != len(METHODS):
                add_error(errors, f"{mask}/{scene} metric count {metric_counter[(mask, scene)]} != {len(METHODS)}")
    for method in METHODS:
        if method_counter[method] != 2 * len(SCENES):
            add_error(errors, f"{method} metric count {method_counter[method]} != {2 * len(SCENES)}")
    if len(model_manifests) != len(SCENES) * len(METHODS):
        add_error(errors, f"unique model manifests {len(model_manifests)} != {len(SCENES) * len(METHODS)}")

    zip_test = "skipped"
    if check_zip and zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad_member = archive.testzip()
        zip_test = "ok" if bad_member is None else f"bad member: {bad_member}"
        if bad_member is not None:
            add_error(errors, f"zip test failed: {bad_member}")

    return {
        "pass": not errors,
        "errors": errors,
        "package_root": str(package_root),
        "zip_path": str(zip_path),
        "counts": counts,
        "summary_completeness": completeness,
        "unique_model_manifests": len(model_manifests),
        "zip_test": zip_test,
    }


def main() -> None:
    args = parse_args()
    package_root = Path(args.package_root).resolve()
    zip_path = Path(args.zip).resolve() if str(args.zip).strip() else package_root.with_suffix(".zip")
    result = validate_package(
        package_root=package_root,
        formal_root=Path(args.formal_root).resolve(),
        meshfix_root=Path(args.meshfix_root).resolve(),
        zip_path=zip_path,
        check_zip=bool(args.check_zip),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
