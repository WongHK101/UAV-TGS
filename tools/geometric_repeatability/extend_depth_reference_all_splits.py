"""Raycast a fixed, hashed OpenMVS reference mesh for all formal splits.

This tool does not rebuild SfM, dense geometry, or the mesh.  It extends a
test-view reference manifest by raycasting the exact same mesh and applying
the exact same ROI/support rule to a fail-closed all-split camera manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geometric_repeatability.build_all_split_probe_camera_manifest import (
    camera_set_sha256,
    camera_sha256,
)
from tools.geometric_repeatability.depth_reference_common import (
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    load_ply_mesh,
    load_ply_points_xyz,
    render_mesh_depth_for_view,
    render_support_count_for_view,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _stem(value: Any) -> str:
    return Path(str(value).strip().replace("\\", "/")).stem.lower()


def _view_map(payload: Mapping[str, Any], label: str) -> Dict[str, Dict[str, Any]]:
    views = payload.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{label}.views must be non-empty")
    mapped: Dict[str, Dict[str, Any]] = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"{label}.views entries must be objects")
        name = str(view.get("image_name", ""))
        if not name or name in mapped:
            raise ValueError(f"{label} has missing/duplicate image_name {name!r}")
        mapped[name] = view
    return mapped


def _formal_records(payload: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Formal split records must be a list")
    labels = {str(record.get("split", "")).lower() for record in records if isinstance(record, dict)}
    if labels != {"train", "guard", "test"}:
        raise ValueError(f"Formal split labels must be exactly train/guard/test, got {sorted(labels)}")
    mapped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Formal split entries must be objects")
        aliases = [record.get("pair_id", ""), record.get("filename", ""), record.get("camera_name", ""), record.get("thermal_camera_name", "")]
        original = record.get("original_files")
        if isinstance(original, dict):
            aliases.extend(original.values())
        for alias in aliases:
            key = _stem(alias)
            if not key:
                continue
            if key in mapped and mapped[key] is not record:
                raise ValueError(f"Duplicate formal camera alias {key!r}")
            mapped[key] = record
    return mapped


def _npz_path(manifest_path: Path, view: Mapping[str, Any]) -> Path:
    root = manifest_path.parent.resolve()
    path = (root / str(view.get("npz_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Reference view escapes manifest root: {path}") from exc
    if not path.is_file() or _sha256(path) != str(view.get("npz_sha256", "")).lower():
        raise RuntimeError(f"Base reference NPZ identity mismatch: {path}")
    return path


def _save_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _camera_equal(first: Mapping[str, Any], second: Mapping[str, Any], label: str) -> None:
    if camera_sha256(first) != camera_sha256(second):
        raise ValueError(f"Camera binding mismatch for {label}")


def extend_reference(
    *,
    base_reference_manifest_path: Path,
    all_split_probe_manifest_path: Path,
    formal_split_manifest_path: Path,
    expected_base_reference_sha256: str,
    expected_reference_mesh_sha256: str,
    formal_reference_lock_path: Path,
    out_dir: Path,
) -> Path:
    base_reference_manifest_path = base_reference_manifest_path.resolve()
    all_split_probe_manifest_path = all_split_probe_manifest_path.resolve()
    formal_split_manifest_path = formal_split_manifest_path.resolve()
    formal_reference_lock_path = formal_reference_lock_path.resolve()
    out_dir = out_dir.resolve()
    expected_base_reference_sha256 = str(expected_base_reference_sha256).strip().lower()
    expected_reference_mesh_sha256 = str(expected_reference_mesh_sha256).strip().lower()
    if len(expected_base_reference_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_base_reference_sha256
    ):
        raise ValueError("expected_base_reference_sha256 must be a lowercase SHA-256")
    if len(expected_reference_mesh_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_reference_mesh_sha256
    ):
        raise ValueError("expected_reference_mesh_sha256 must be a lowercase SHA-256")
    actual_base_sha = _sha256(base_reference_manifest_path)
    if actual_base_sha != expected_base_reference_sha256:
        raise ValueError(
            "Operator-pinned base reference manifest SHA-256 mismatch: "
            f"{actual_base_sha} != {expected_base_reference_sha256}"
        )
    lock = _load_json(formal_reference_lock_path)
    if str(lock.get("protocol", "")) != "uav-tgs-formal-reference-lock-v1" or str(
        lock.get("status", "")
    ).lower() != "approved":
        raise ValueError("Formal reference lock must be an approved uav-tgs-formal-reference-lock-v1 receipt")
    lock_bindings = {
        "base_reference_manifest_sha256": expected_base_reference_sha256,
        "reference_mesh_sha256": expected_reference_mesh_sha256,
        "formal_split_manifest_sha256": _sha256(formal_split_manifest_path),
    }
    for key, expected in lock_bindings.items():
        if str(lock.get(key, "")).strip().lower() != expected:
            raise ValueError(f"Formal reference lock mismatch for {key}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = _load_json(base_reference_manifest_path)
    probe = _load_json(all_split_probe_manifest_path)
    formal = _load_json(formal_split_manifest_path)
    scenes = {
        str(base.get("scene_name", "")),
        str(probe.get("scene_name", "")),
        str(formal.get("scene_name", formal.get("scene", ""))),
    }
    if "" in scenes:
        formal_scenes = {
            str(record.get("scene_name", record.get("scene", "")))
            for record in formal.get("records", [])
            if isinstance(record, dict)
        }
        formal_scenes.discard("")
        scenes.discard("")
        scenes.update(formal_scenes)
    if len(scenes) != 1:
        raise ValueError(f"Scene mismatch: {sorted(scenes)}")
    scene_name = next(iter(scenes))
    if str(lock.get("scene_name", lock.get("scene", ""))).strip() != scene_name:
        raise ValueError("Formal reference lock scene mismatch")
    if str(base.get("depth_semantics", "")) != "metric_camera_z_reference_mesh":
        raise ValueError("Base reference depth semantics mismatch")
    if str(base.get("reference_construction_protocol", "")) != "openmvs-reference-mesh-v1":
        raise ValueError("Base reference is not the frozen OpenMVS protocol")
    mesh_backend = str(base.get("reference_mesh_backend", ""))
    if not mesh_backend.startswith("openmvs_"):
        raise ValueError("Base reference mesh backend is not OpenMVS")

    mesh_path = Path(str(base.get("reference_mesh_path", ""))).resolve()
    dense_path = Path(str(base.get("reference_dense_ply", ""))).resolve()
    roi_path = Path(str(base.get("roi_path", ""))).resolve()
    for path in (mesh_path, dense_path, roi_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    mesh_sha = _sha256(mesh_path)
    dense_sha = _sha256(dense_path)
    if mesh_sha != str(base.get("reference_mesh_sha256", "")).lower():
        raise RuntimeError("Frozen reference mesh SHA-256 mismatch")
    if mesh_sha != expected_reference_mesh_sha256:
        raise ValueError(
            "Operator-pinned reference mesh SHA-256 mismatch: "
            f"{mesh_sha} != {expected_reference_mesh_sha256}"
        )
    if dense_sha != str(base.get("reference_dense_ply_sha256", "")).lower():
        raise RuntimeError("Frozen reference dense PLY SHA-256 mismatch")
    roi = _load_json(roi_path)
    if str(roi.get("scene_name", "")) != scene_name:
        raise ValueError("Reference ROI scene mismatch")
    if str(roi.get("source_points_sha256", "")).lower() != dense_sha:
        raise ValueError("Reference ROI is not bound to the frozen dense PLY")
    bbox_min = np.asarray(roi.get("bbox_min"), dtype=np.float64)
    bbox_max = np.asarray(roi.get("bbox_max"), dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,) or not np.all(bbox_min < bbox_max):
        raise ValueError("Reference ROI bounds are invalid")
    support_rule = base.get("support_rule")
    if (
        not isinstance(support_rule, dict)
        or support_rule.get("type") != "training_dense_projected_support_count"
        or support_rule.get("source_backend") != "openmvs_densify_point_cloud"
    ):
        raise ValueError("Base support rule is missing or changed")
    support_min_count = int(support_rule["min_support_count"])
    support_radius_px = int(support_rule["support_radius_px"])
    support_tolerance_m = float(support_rule["support_depth_tolerance_m"])
    if support_min_count < 1 or support_radius_px < 0 or not np.isfinite(support_tolerance_m) or support_tolerance_m <= 0.0:
        raise ValueError("Base support rule parameters are invalid")

    base_views = _view_map(base, "base reference")
    probe_views = _view_map(probe, "all-split probe")
    records = _formal_records(formal)
    if str(probe.get("camera_manifest_type", "")) != "formal_all_split_probe_camera_manifest_v1":
        raise ValueError("Probe manifest is not a formal all-split camera binding")
    if str(probe.get("bound_split_manifest_identity", {}).get("sha256", "")).lower() != _sha256(formal_split_manifest_path):
        raise ValueError("Probe/formal split hash mismatch")
    if str(probe.get("camera_set_sha256", "")).lower() != camera_set_sha256(list(probe_views.values())):
        raise ValueError("Probe camera-set hash mismatch")
    matched_records: set[int] = set()
    for name, view in probe_views.items():
        record = records.get(_stem(view.get("pair_id", name))) or records.get(_stem(name))
        if record is None:
            raise KeyError(f"Probe camera {name!r} is absent from the formal split")
        matched_records.add(id(record))
        split = str(record.get("split", "")).lower()
        if str(view.get("bound_split", "")).lower() != split:
            raise ValueError(f"Probe bound_split mismatch for {name}")
        if str(view.get("camera_sha256", "")).lower() != camera_sha256(view):
            raise ValueError(f"Probe camera hash mismatch for {name}")
    if len(matched_records) != len(formal.get("records", [])) or len(probe_views) != len(formal.get("records", [])):
        raise ValueError("All-split probe must bind every formal record exactly once")
    base_by_stem = {_stem(name): (name, view) for name, view in base_views.items()}
    probe_by_stem = {_stem(name): (name, view) for name, view in probe_views.items()}
    if len(base_by_stem) != len(base_views) or len(probe_by_stem) != len(probe_views):
        raise ValueError("Base/probe manifests contain duplicate frame stems")
    if not set(base_by_stem).issubset(probe_by_stem):
        raise ValueError("Base reference cameras are not a subset of the all-split probe")
    expected_test_stems = {
        _stem(name)
        for name, view in probe_views.items()
        if str(view.get("bound_split", "")).lower() == "test"
    }
    if set(base_by_stem) != expected_test_stems:
        raise ValueError(
            "Base held-out reference must cover every formal test camera exactly; "
            f"missing={sorted(expected_test_stems - set(base_by_stem))[:8]} "
            f"extra={sorted(set(base_by_stem) - expected_test_stems)[:8]}"
        )
    for stem, (base_name, base_view) in base_by_stem.items():
        probe_name, probe_view = probe_by_stem[stem]
        _camera_equal(base_view, probe_view, f"{base_name}/{probe_name}")
        record = records.get(stem)
        if record is None or str(record.get("split", "")).lower() != "test":
            raise ValueError(f"Base held-out reference view {base_name!r} is not formal test")

    vertices, faces = load_ply_mesh(mesh_path)
    dense_points = load_ply_points_xyz(dense_path)
    roi_rule = roi.get("roi_rule")
    if (
        str(roi.get("protocol_name", "")) != "reference-depth-based-geometric-evaluation-v1"
        or str(roi.get("reference_construction_protocol", "")) != "openmvs-reference-mesh-v1"
        or not isinstance(roi_rule, dict)
        or roi_rule.get("type") != "training_reference_dense_quantile_aabb"
    ):
        raise ValueError("Reference ROI protocol/rule mismatch")
    recomputed_roi = compute_quantile_bbox(
        dense_points,
        lower_quantile=float(roi_rule["lower_quantile"]),
        upper_quantile=float(roi_rule["upper_quantile"]),
        padding_ratio_of_robust_diagonal=float(roi_rule["padding_ratio_of_robust_diagonal"]),
    )
    if not np.allclose(np.asarray(recomputed_roi["bbox_min"]), bbox_min, rtol=0.0, atol=1e-12):
        raise ValueError("Reference ROI bbox_min is not reproducible from the frozen dense PLY/rule")
    if not np.allclose(np.asarray(recomputed_roi["bbox_max"]), bbox_max, rtol=0.0, atol=1e-12):
        raise ValueError("Reference ROI bbox_max is not reproducible from the frozen dense PLY/rule")
    if not np.isclose(float(recomputed_roi["scene_diagonal"]), float(roi.get("scene_diagonal")), rtol=0.0, atol=1e-12):
        raise ValueError("Reference ROI scene diagonal is not reproducible from the frozen dense PLY/rule")
    manifest_views: list[Dict[str, Any]] = []
    for index, name in enumerate(sorted(probe_views)):
        view = probe_views[name]
        depth = render_mesh_depth_for_view(vertices, faces, view)
        support_count = render_support_count_for_view(
            dense_points,
            view,
            depth_tolerance_m=support_tolerance_m,
            support_radius_px=support_radius_px,
        )
        finite = np.isfinite(depth) & (depth > 0.0)
        inside_roi = (
            compute_inside_bbox_mask(depth, view, bbox_min=bbox_min, bbox_max=bbox_max)
            if np.any(finite)
            else np.zeros_like(finite, dtype=bool)
        )
        valid_mask = finite & inside_roi & (support_count >= support_min_count)
        arrays = {
            "depth": np.asarray(depth, dtype=np.float64),
            "inside_roi": np.asarray(inside_roi, dtype=np.uint8),
            "support_count": np.asarray(support_count, dtype=np.int32),
            "valid_mask": np.asarray(valid_mask, dtype=np.uint8),
        }
        base_pair = base_by_stem.get(_stem(name))
        if base_pair is not None:
            base_name, base_view = base_pair
            base_npz_path = _npz_path(base_reference_manifest_path, base_view)
            with np.load(base_npz_path, allow_pickle=False) as previous:
                for key, value in arrays.items():
                    if key not in previous or not np.array_equal(np.asarray(previous[key]), value, equal_nan=True):
                        raise RuntimeError(f"Fixed-mesh extension changed base test array {base_name}/{key}")
        relative = Path("views") / f"{index:05d}.npz"
        path = out_dir / relative
        _save_npz_deterministic(path, arrays)
        record = records.get(_stem(view.get("pair_id", name))) or records[_stem(name)]
        entry: Dict[str, Any] = {
            key: view[key]
            for key in ("view_id", "image_name", "width", "height", "fx", "fy", "cx", "cy", "camera_to_world")
        }
        entry["view_id"] = f"{index:05d}"
        entry["bound_split"] = str(record["split"]).lower()
        entry["split"] = entry["bound_split"]
        entry["camera_sha256"] = camera_sha256(entry)
        for key in ("block_id", "block", "block_index", "strip_id", "stratum", "gimbal_pitch_deg"):
            if key in record:
                entry[key] = record[key]
        entry.update(
            {
                "npz_file": str(relative).replace("\\", "/"),
                "npz_size_bytes": int(path.stat().st_size),
                "npz_sha256": _sha256(path),
            }
        )
        manifest_views.append(entry)

    base_identity = _identity(base_reference_manifest_path)
    probe_identity = _identity(all_split_probe_manifest_path)
    split_identity = _identity(formal_split_manifest_path)
    output: Dict[str, Any] = {
        **{key: value for key, value in base.items() if key != "views"},
        "reference_view_scope": "all_formal_splits",
        "base_probe_camera_manifest_path": str(base.get("camera_manifest_path", "")),
        "camera_manifest_path": str(all_split_probe_manifest_path),
        "base_reference_manifest_identity": base_identity,
        "probe_camera_manifest_identity": probe_identity,
        "formal_split_manifest_identity": split_identity,
        "formal_reference_lock_identity": _identity(formal_reference_lock_path),
        "camera_set_sha256": camera_set_sha256(manifest_views),
        "all_split_reference_binding": {
            "extension_protocol": "fixed-openmvs-mesh-all-formal-splits-v1",
            "base_reference_manifest_sha256": base_identity["sha256"],
            "reference_mesh_sha256": mesh_sha,
            "reference_mesh_backend": mesh_backend,
            "reference_dense_ply_sha256": dense_sha,
            "roi_sha256": _sha256(roi_path),
            "probe_camera_manifest_sha256": probe_identity["sha256"],
            "formal_split_manifest_sha256": split_identity["sha256"],
            "bound_split_labels": ["train", "guard", "test"],
            "mesh_or_backend_rebuilt": False,
            "operator_pinned_base_reference_sha256": expected_base_reference_sha256,
            "operator_pinned_reference_mesh_sha256": expected_reference_mesh_sha256,
            "formal_reference_lock_sha256": _sha256(formal_reference_lock_path),
        },
        "views": manifest_views,
    }
    manifest_path = out_dir / "reference_depth_manifest.json"
    manifest_path.write_text(json.dumps(output, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_reference_manifest", required=True)
    parser.add_argument("--all_split_probe_manifest", required=True)
    parser.add_argument("--formal_split_manifest", required=True)
    parser.add_argument("--expected_base_reference_sha256", required=True)
    parser.add_argument("--expected_reference_mesh_sha256", required=True)
    parser.add_argument("--formal_reference_lock", required=True)
    parser.add_argument("--out_dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = extend_reference(
        base_reference_manifest_path=Path(args.base_reference_manifest),
        all_split_probe_manifest_path=Path(args.all_split_probe_manifest),
        formal_split_manifest_path=Path(args.formal_split_manifest),
        expected_base_reference_sha256=str(args.expected_base_reference_sha256),
        expected_reference_mesh_sha256=str(args.expected_reference_mesh_sha256),
        formal_reference_lock_path=Path(args.formal_reference_lock),
        out_dir=Path(args.out_dir),
    )
    print(f"ALL_SPLIT_REFERENCE_MANIFEST {manifest}")


if __name__ == "__main__":
    main()
