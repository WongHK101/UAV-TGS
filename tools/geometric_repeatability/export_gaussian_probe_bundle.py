from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args
from tools.geometric_repeatability.depth_reference_common import (
    apply_world_transform_to_camera_to_world,
    estimate_strict_to_native_world_transform,
    load_json,
    load_native_camera_entries,
)
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2


ALIGNMENT_MAX_TRANSLATION_ERROR_M = 1.0e-4
ALIGNMENT_MAX_ROTATION_ERROR_DEG = 5.0e-2
PRINCIPAL_POINT_CENTER_TOLERANCE_PX = 1.0e-6


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_file_identity(value: str | Path | None) -> Dict[str, Any]:
    if value is None or not str(value).strip():
        return {"path": "", "size_bytes": 0, "sha256": ""}
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _require_file_identity(path: Path, expected: Any, *, label: str) -> Dict[str, Any]:
    actual = _optional_file_identity(path)
    if not isinstance(expected, dict):
        raise ValueError(f"{label} is missing a file identity")
    expected_sha = str(expected.get("sha256", "")).strip().lower()
    expected_size = int(expected.get("size_bytes", -1))
    if expected_sha != actual["sha256"] or expected_size != actual["size_bytes"]:
        raise ValueError(
            f"{label} identity mismatch: expected sha/size "
            f"{expected_sha}/{expected_size}, got {actual['sha256']}/{actual['size_bytes']}"
        )
    return actual


def _ply_xyz_sequence_identity(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")
    vertex = ply["vertex"]
    for field in ("x", "y", "z"):
        if field not in vertex.data.dtype.names:
            raise ValueError(f"PLY vertex element is missing {field!r}: {path}")
    xyz = np.column_stack([vertex[field] for field in ("x", "y", "z")]).astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(xyz.shape, dtype="<i8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(xyz).tobytes(order="C"))
    return {
        "vertex_count": int(xyz.shape[0]),
        "dtype": "float32_le",
        "sequence_sha256": digest.hexdigest(),
    }


def _validate_gaussian_index_binding(
    *,
    model_point_cloud_path: Path,
    index_anchor_path: Path,
    gaussian_count: int,
    binding_manifest_path: Path | None,
) -> Dict[str, Any]:
    model_identity = _optional_file_identity(model_point_cloud_path)
    anchor_identity = _optional_file_identity(index_anchor_path)
    model_xyz = _ply_xyz_sequence_identity(model_point_cloud_path)
    anchor_xyz = _ply_xyz_sequence_identity(index_anchor_path)
    if model_xyz["vertex_count"] != int(gaussian_count) or anchor_xyz["vertex_count"] != int(gaussian_count):
        raise ValueError("Rendered PLY/index-anchor PLY Gaussian counts do not match the loaded model")

    if model_identity["sha256"] == anchor_identity["sha256"]:
        proof = "identical_ply_sha256"
        receipt_identity = None
    elif model_xyz["sequence_sha256"] == anchor_xyz["sequence_sha256"]:
        proof = "exact_ordered_xyz_sequence"
        receipt_identity = None
    else:
        if binding_manifest_path is None:
            raise ValueError(
                "Rendered PLY differs in ordered XYZ from the Gaussian index anchor; "
                "an explicit fixed-topology index binding receipt is required"
            )
        binding_manifest_path = binding_manifest_path.resolve()
        receipt_identity = _optional_file_identity(binding_manifest_path)
        payload = load_json(binding_manifest_path)
        if str(payload.get("protocol", "")) != "uav-tgs-gaussian-index-binding-v1":
            raise ValueError("Unsupported Gaussian index binding receipt protocol")
        if str(payload.get("status", "")) != "verified":
            raise ValueError("Gaussian index binding receipt is not verified")
        expected = {
            "rendered_model_point_cloud_sha256": model_identity["sha256"],
            "gaussian_index_anchor_sha256": anchor_identity["sha256"],
        }
        for key, value in expected.items():
            if str(payload.get(key, "")).strip().lower() != value:
                raise ValueError(f"Gaussian index binding receipt mismatch for {key}")
        if int(payload.get("gaussian_count", -1)) != int(gaussian_count):
            raise ValueError("Gaussian index binding receipt count mismatch")
        if payload.get("topology_fixed") is not True or payload.get("index_order_preserved") is not True:
            raise ValueError("Gaussian index binding receipt must prove fixed topology and preserved index order")
        evidence = payload.get("evidence_manifest_identity")
        if not isinstance(evidence, dict):
            raise ValueError("Gaussian index binding receipt must identify its invariant-audit evidence manifest")
        evidence_path = Path(str(evidence.get("path", ""))).resolve()
        _require_file_identity(evidence_path, evidence, label="Gaussian index invariant-audit evidence")
        evidence_payload = load_json(evidence_path)
        if str(evidence_payload.get("schema", "")) != "uav-tgs-rgb-continuation-protocol-v1":
            raise ValueError("Gaussian index evidence is not a supported RGB continuation protocol")
        if evidence_payload.get("topology_fixed") is not True:
            raise ValueError("Gaussian index evidence does not declare fixed topology")
        for key in ("densification", "pruning", "opacity_reset"):
            if evidence_payload.get(key) is not False:
                raise ValueError(f"Gaussian index evidence must declare {key}=false")
        if str(evidence_payload.get("artifact_save_semantics", "")) != "aligned":
            raise ValueError("Gaussian index evidence does not use aligned endpoint saves")
        for receipt_key, evidence_key in (
            ("anchor_iteration", "anchor_iteration"),
            ("final_iteration", "final_iteration"),
        ):
            if int(payload.get(receipt_key, -1)) != int(evidence_payload.get(evidence_key, -2)):
                raise ValueError(f"Gaussian index binding/evidence mismatch for {receipt_key}")
        for key in ("anchor_checkpoint_identity", "final_checkpoint_identity"):
            identity = payload.get(key)
            if not isinstance(identity, dict):
                raise ValueError(f"Gaussian index binding receipt is missing {key}")
            _require_file_identity(Path(str(identity.get("path", ""))), identity, label=key)
        if str(payload["anchor_checkpoint_identity"]["sha256"]).lower() != str(
            evidence_payload.get("start_checkpoint_sha256", "")
        ).lower():
            raise ValueError("Gaussian index binding start-checkpoint/evidence mismatch")
        ordered_proof = payload.get("ordered_xyz_proof")
        if not isinstance(ordered_proof, dict) or any(
            ordered_proof.get(key) is not True
            for key in (
                "anchor_ply_equals_start_checkpoint",
                "rendered_ply_equals_final_checkpoint",
                "continuation_has_no_topology_mutation",
            )
        ):
            raise ValueError("Gaussian index binding receipt lacks endpoint ordered-XYZ proof")
        if str(ordered_proof.get("anchor_ply", {}).get("sequence_sha256", "")).lower() != anchor_xyz[
            "sequence_sha256"
        ] or str(ordered_proof.get("rendered_ply", {}).get("sequence_sha256", "")).lower() != model_xyz[
            "sequence_sha256"
        ]:
            raise ValueError("Gaussian index binding receipt ordered-XYZ hashes do not match endpoint PLYs")
        producer = payload.get("producer_identity")
        expected_generator = REPO_ROOT / "tools" / "geometric_repeatability" / "build_gaussian_index_binding_receipt.py"
        if not isinstance(producer, dict) or not isinstance(producer.get("script"), dict):
            raise ValueError("Gaussian index binding receipt is missing producer identity")
        _require_file_identity(expected_generator, producer["script"], label="Gaussian index receipt generator")
        proof = "fixed_topology_invariant_audit_receipt"

    return {
        "status": "verified",
        "proof": proof,
        "gaussian_count": int(gaussian_count),
        "rendered_model_point_cloud": model_identity,
        "gaussian_index_anchor": anchor_identity,
        "rendered_ordered_xyz": model_xyz,
        "anchor_ordered_xyz": anchor_xyz,
        "binding_receipt_identity": receipt_identity,
    }


def _normalized_stem(value: Any) -> str:
    return Path(str(value).strip().replace("\\", "/")).stem.lower()


def _formal_records_by_stem(path: Path) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Formal split manifest records must be a list")
    labels = {str(record.get("split", "")).strip().lower() for record in records if isinstance(record, dict)}
    if labels != {"train", "guard", "test"}:
        raise ValueError(f"Formal split manifest must contain train/guard/test, observed={sorted(labels)}")
    mapped: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Formal split records must be objects")
        values: List[Any] = [
            record.get("pair_id", ""),
            record.get("filename", ""),
            record.get("camera_name", ""),
            record.get("thermal_camera_name", ""),
        ]
        original = record.get("original_files")
        if isinstance(original, dict):
            values.extend(original.values())
        for value in values:
            stem = _normalized_stem(value)
            if not stem:
                continue
            if stem in mapped and mapped[stem] is not record:
                raise ValueError(f"Duplicate formal split stem {stem!r}")
            mapped[stem] = record
    return mapped, payload


def _camera_sha256_for_matrix(view: Dict[str, Any], camera_to_world: Any) -> str:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise ValueError("camera_to_world must be finite 4x4")
    payload = {
        "width": int(view["width"]),
        "height": int(view["height"]),
        "fx_fy_cx_cy": [float(view[key]) for key in ("fx", "fy", "cx", "cy")],
        "camera_to_world": c2w.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _camera_sha256(view: Dict[str, Any]) -> str:
    return _camera_sha256_for_matrix(view, view["camera_to_world"])


def _render_camera_sha256(view: Dict[str, Any]) -> str:
    return _camera_sha256_for_matrix(view, view["native_camera_to_world"])


def _camera_set_sha256(views: List[Dict[str, Any]]) -> str:
    payload = [(str(view["image_name"]), _camera_sha256(view)) for view in sorted(views, key=lambda row: str(row["image_name"]))]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _render_camera_set_sha256(views: List[Dict[str, Any]]) -> str:
    payload = [
        (str(view["image_name"]), _render_camera_sha256(view))
        for view in sorted(views, key=lambda row: str(row["image_name"]))
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _producer_identity() -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_error = ""
    except Exception as exc:
        commit = ""
        status = "git-identity-unavailable"
        git_error = f"{type(exc).__name__}: {exc}"
    return {
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "repo_root": str(REPO_ROOT.resolve()),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_error": git_error,
    }


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


def _tensor_chw_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    if array.ndim != 3:
        raise ValueError(f"Expected a CxHxW tensor, got {array.shape}")
    return np.asarray(array, dtype=np.float32)


def _tensor_index_hw_numpy(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1xHxW integer tensor, got {array.shape}")
    return np.asarray(array, dtype=np.int32)


def _index_probe_images(image_root: Path) -> Dict[str, List[Path]]:
    image_root = image_root.resolve()
    if not image_root.is_dir():
        raise FileNotFoundError(f"Probe image root is missing: {image_root}")
    indexed: Dict[str, List[Path]] = {}
    for path in sorted(image_root.iterdir(), key=lambda item: item.name.casefold()):
        if path.is_file():
            indexed.setdefault(path.stem.casefold(), []).append(path.resolve())
    return indexed


def _resolve_manifest_image_path(
    *, image_root: Path, image_name: str, indexed_images: Dict[str, List[Path]]
) -> Path:
    image_root = image_root.resolve()
    relative = Path(str(image_name).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe probe manifest image_name: {image_name!r}")
    exact = (image_root / relative).resolve()
    try:
        exact.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(f"Probe image escapes image root: {image_name!r}") from exc
    if exact.is_file():
        return exact
    candidates = indexed_images.get(relative.stem.casefold(), [])
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Probe image has no exact match and its stem is not unique: "
            f"image_name={image_name!r}, candidates={[path.name for path in candidates]}"
        )
    return candidates[0]


def _render_depth_and_opacity(
    view,
    gaussians,
    pipeline,
    black_bg: torch.Tensor,
    white_override: torch.Tensor,
    *,
    depth_diagnostics: bool,
    appearance_modality: str,
) -> Dict[str, np.ndarray]:
    from gaussian_renderer import render

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
            return_diagnostics=depth_diagnostics,
        )
    depth = _tensor_hwc_to_numpy_hw(depth_out["depth"])
    if depth_diagnostics:
        arrays: Dict[str, np.ndarray] = {
            "depth": depth,
            "opacity": _tensor_hwc_to_numpy_hw(depth_out["accumulated_opacity"]),
            "depth_expected_alpha_normalized": _tensor_hwc_to_numpy_hw(depth_out["depth_expected_alpha_normalized"]),
            "depth_transmittance_median": _tensor_hwc_to_numpy_hw(depth_out["depth_transmittance_median"]),
            "depth_max_contribution": _tensor_hwc_to_numpy_hw(depth_out["depth_max_contribution"]),
            "top_contributor_index": _tensor_index_hw_numpy(depth_out["top_contributor_index"]),
            "top_contributor_weight": _tensor_hwc_to_numpy_hw(depth_out["top_contributor_weight"]),
            "accumulated_opacity": _tensor_hwc_to_numpy_hw(depth_out["accumulated_opacity"]),
        }
        if appearance_modality == "rgb":
            arrays["render_rgb"] = _tensor_chw_numpy(depth_out["render"])
            arrays["target_rgb"] = _tensor_chw_numpy(view.original_image)
        elif appearance_modality == "thermal_canonical":
            arrays["render_thermal_canonical"] = _tensor_chw_numpy(depth_out["render"])
            arrays["target_thermal_canonical"] = _tensor_chw_numpy(view.original_image)
        elif appearance_modality != "none":
            raise ValueError(f"Unsupported appearance_modality: {appearance_modality!r}")
        return arrays

    with torch.no_grad():
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
    opacity_render = alpha_out["render"].detach().float().cpu().numpy()
    if opacity_render.ndim != 3 or opacity_render.shape[0] < 1:
        raise ValueError(f"Unexpected opacity render shape: {opacity_render.shape}")
    opacity = np.asarray(opacity_render[0], dtype=np.float64)
    return {"depth": depth, "opacity": opacity}


def _camera_from_manifest_view(
    *,
    manifest_view: Dict[str, Any],
    camera_to_world: np.ndarray,
    image_path: Path,
    uid: int,
    data_device: str,
) -> Camera:
    from scene.cameras import Camera

    image_name = str(manifest_view["image_name"])
    image_path = image_path.resolve()
    if not image_path.is_file():
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
    formal_split_manifest_path: Path | None,
    gaussian_index_anchor_path: Path | None,
    gaussian_index_binding_manifest_path: Path | None,
    depth_diagnostics: bool,
    appearance_modality: str,
) -> Path:
    from gaussian_renderer import GaussianModel
    from scene import Scene

    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to write diagnostic bundle into non-empty directory: {out_dir}")
    if appearance_modality not in {"rgb", "thermal_canonical", "none"}:
        raise ValueError(f"Unsupported appearance_modality: {appearance_modality!r}")
    formal_records: Dict[str, Dict[str, Any]] = {}
    formal_split_payload: Dict[str, Any] | None = None
    if depth_diagnostics:
        if camera_frame_mode != "probe_manifest_native_align":
            raise ValueError("Formal depth diagnostics require camera_frame_mode=probe_manifest_native_align")
        if probe_camera_manifest_path is None or formal_split_manifest_path is None:
            raise ValueError("Formal depth diagnostics require probe_camera_manifest and formal_split_manifest")
        if max_views is not None:
            raise ValueError("Formal depth diagnostics forbid max_views; all bound split views are required")
        formal_records, formal_split_payload = _formal_records_by_stem(formal_split_manifest_path)
        probe_binding = load_json(probe_camera_manifest_path)
        if str(probe_binding.get("camera_manifest_type", "")) != "formal_all_split_probe_camera_manifest_v1":
            raise ValueError("Depth diagnostics require a formal all-split probe-camera manifest")
        bound_identity = probe_binding.get("bound_split_manifest_identity")
        if not isinstance(bound_identity, dict) or str(bound_identity.get("sha256", "")).lower() != _sha256_file(formal_split_manifest_path):
            raise ValueError("Probe-camera/formal-split hash binding mismatch")
        if native_cameras_json_path is None:
            raise ValueError("Formal depth diagnostics require native_cameras_json")
        _require_file_identity(
            native_cameras_json_path,
            probe_binding.get("model_cameras_json_identity"),
            label="Probe-bound model cameras.json",
        )
        for view in probe_binding.get("views", []):
            if str(view.get("camera_sha256", "")).strip().lower() != _camera_sha256(view):
                raise ValueError(f"Probe camera hash mismatch for {view.get('image_name', '')!r}")
            if not np.isclose(float(view["cx"]), float(view["width"]) / 2.0, rtol=0.0, atol=PRINCIPAL_POINT_CENTER_TOLERANCE_PX) or not np.isclose(
                float(view["cy"]),
                float(view["height"]) / 2.0,
                rtol=0.0,
                atol=PRINCIPAL_POINT_CENTER_TOLERANCE_PX,
            ):
                raise ValueError(
                    "The repository Camera/rasterizer path assumes a centered principal point; "
                    f"probe camera {view.get('image_name', '')!r} is not centered"
                )
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        model_point_cloud = (
            Path(dataset.model_path).resolve()
            / "point_cloud"
            / f"iteration_{int(scene.loaded_iter)}"
            / "point_cloud.ply"
        )
        if not model_point_cloud.is_file():
            raise FileNotFoundError(f"Loaded Gaussian point cloud is missing: {model_point_cloud}")
        index_anchor_path = gaussian_index_anchor_path.resolve() if gaussian_index_anchor_path is not None else model_point_cloud
        index_anchor_identity = _optional_file_identity(index_anchor_path)
        gaussian_index_binding = _validate_gaussian_index_binding(
            model_point_cloud_path=model_point_cloud,
            index_anchor_path=index_anchor_path,
            gaussian_count=int(gaussians.get_xyz.shape[0]),
            binding_manifest_path=gaussian_index_binding_manifest_path,
        )
        black_bg = torch.zeros(3, dtype=torch.float32, device="cuda")
        white_override = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")

        manifest_views: List[Dict[str, Any]] = []
        split_dir = out_dir / "views"
        split_dir.mkdir(parents=True, exist_ok=True)

        strict_to_native_alignment: Dict[str, Any] | None = None
        native_camera_coverage: Dict[str, Any] | None = None
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
                    depth_diagnostics=depth_diagnostics,
                    appearance_modality=appearance_modality,
                )
                view_rel = Path("views") / f"{idx:05d}.npz"
                view_path = out_dir / view_rel
                np.savez_compressed(view_path, **arrays)
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
                        "split": str(split_label),
                        "npz_file": str(view_rel).replace("\\", "/"),
                        "npz_size_bytes": int(view_path.stat().st_size),
                        "npz_sha256": _sha256_file(view_path),
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
            common_stems = {_normalized_stem(name) for name in strict_to_native_alignment["common_image_names"]}
            if common_stems != set(native_cameras_by_stem):
                raise ValueError("Strict/native alignment did not use every camera bound by model cameras.json")
            direct_center_errors: List[float] = []
            direct_rotation_errors: List[float] = []
            image_root = Path(dataset.source_path) / str(dataset.images)
            indexed_images = _index_probe_images(image_root)
            for idx, strict_view in enumerate(probe_views):
                native_c2w = apply_world_transform_to_camera_to_world(
                    np.asarray(strict_view["camera_to_world"], dtype=np.float64),
                    strict_to_native,
                )
                resolved_image_path = _resolve_manifest_image_path(
                    image_root=image_root,
                    image_name=str(strict_view["image_name"]),
                    indexed_images=indexed_images,
                )
                render_view = _camera_from_manifest_view(
                    manifest_view=strict_view,
                    camera_to_world=native_c2w,
                    image_path=resolved_image_path,
                    uid=idx,
                    data_device=dataset.data_device,
                )
                arrays = _render_depth_and_opacity(
                    view=render_view,
                    gaussians=gaussians,
                    pipeline=pipeline,
                    black_bg=black_bg,
                    white_override=white_override,
                    depth_diagnostics=depth_diagnostics,
                    appearance_modality=appearance_modality,
                )
                view_rel = Path("views") / f"{idx:05d}.npz"
                view_path = out_dir / view_rel
                np.savez_compressed(view_path, **arrays)
                diagnostic_metadata = {
                    key: strict_view[key]
                    for key in (
                        "split",
                        "partition",
                        "block_id",
                        "block",
                        "block_index",
                        "strip_id",
                        "orientation",
                        "view_class",
                        "view_type",
                        "stratum",
                        "gimbal_pitch_deg",
                        "pitch_deg",
                    )
                    if key in strict_view
                }
                formal_record = None
                if depth_diagnostics:
                    for candidate in (
                        strict_view.get("pair_id", ""),
                        strict_view.get("image_name", ""),
                        strict_view.get("source_image", ""),
                    ):
                        formal_record = formal_records.get(_normalized_stem(candidate))
                        if formal_record is not None:
                            break
                if depth_diagnostics and formal_record is None:
                    raise KeyError(f"Probe view {strict_view['image_name']!r} is absent from the formal split manifest")
                if formal_record is not None:
                    diagnostic_metadata = {
                        **diagnostic_metadata,
                        **{
                            key: formal_record[key]
                            for key in (
                                "split",
                                "block_id",
                                "block",
                                "block_index",
                                "strip_id",
                                "stratum",
                                "gimbal_pitch_deg",
                            )
                            if key in formal_record
                        },
                    }
                bound_split = str(
                    diagnostic_metadata.get("split", diagnostic_metadata.get("partition", split_label))
                ).strip().lower()
                view_entry = {
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
                    "split": bound_split,
                    "bound_split": bound_split,
                    "input_image_identity": {
                        **_optional_file_identity(resolved_image_path),
                        "relative_path": resolved_image_path.relative_to(image_root.resolve()).as_posix(),
                        "resolved_by": (
                            "exact_name"
                            if resolved_image_path == (image_root / str(strict_view["image_name"])).resolve()
                            else "unique_stem"
                        ),
                    },
                    **{key: value for key, value in diagnostic_metadata.items() if key not in {"split", "partition"}},
                    "npz_file": str(view_rel).replace("\\", "/"),
                    "npz_size_bytes": int(view_path.stat().st_size),
                    "npz_sha256": _sha256_file(view_path),
                }
                view_entry["camera_sha256"] = _camera_sha256(view_entry)
                view_entry["render_camera_sha256"] = _render_camera_sha256(view_entry)
                native_entry = native_cameras_by_stem.get(_normalized_stem(strict_view["image_name"]))
                if native_entry is not None:
                    direct_native = np.eye(4, dtype=np.float64)
                    direct_native[:3, :3] = np.asarray(native_entry["rotation"], dtype=np.float64)
                    direct_native[:3, 3] = np.asarray(native_entry["position"], dtype=np.float64)
                    center_error = float(np.linalg.norm(native_c2w[:3, 3] - direct_native[:3, 3]))
                    relative_rotation = native_c2w[:3, :3] @ direct_native[:3, :3].T
                    cosine = np.clip((float(np.trace(relative_rotation)) - 1.0) / 2.0, -1.0, 1.0)
                    rotation_error = float(np.degrees(np.arccos(cosine)))
                    direct_center_errors.append(center_error)
                    direct_rotation_errors.append(rotation_error)
                    view_entry["bound_native_camera_sha256"] = _camera_sha256_for_matrix(view_entry, direct_native)
                    view_entry["alignment_center_error_m"] = center_error
                    view_entry["alignment_rotation_error_deg"] = rotation_error
                    view_entry["render_camera_is_alignment_extrapolation"] = False
                else:
                    view_entry["bound_native_camera_sha256"] = ""
                    view_entry["alignment_center_error_m"] = None
                    view_entry["alignment_rotation_error_deg"] = None
                    view_entry["render_camera_is_alignment_extrapolation"] = True
                manifest_views.append(view_entry)
            strict_to_native_alignment["revalidated_against_bound_native_cameras"] = {
                "count": len(direct_center_errors),
                "translation_error_mean_m": float(np.mean(direct_center_errors)),
                "translation_error_max_m": float(np.max(direct_center_errors)),
                "rotation_error_mean_deg": float(np.mean(direct_rotation_errors)),
                "rotation_error_max_deg": float(np.max(direct_rotation_errors)),
                "maximum_allowed_translation_error_m": ALIGNMENT_MAX_TRANSLATION_ERROR_M,
                "maximum_allowed_rotation_error_deg": ALIGNMENT_MAX_ROTATION_ERROR_DEG,
                "status": "passed",
            }
            if max(direct_center_errors) > ALIGNMENT_MAX_TRANSLATION_ERROR_M:
                raise ValueError(
                    "Strict/native camera alignment translation residual exceeds the formal integrity limit: "
                    f"{max(direct_center_errors)} > {ALIGNMENT_MAX_TRANSLATION_ERROR_M} m"
                )
            if max(direct_rotation_errors) > ALIGNMENT_MAX_ROTATION_ERROR_DEG:
                raise ValueError(
                    "Strict/native camera alignment rotation residual exceeds the formal integrity limit: "
                    f"{max(direct_rotation_errors)} > {ALIGNMENT_MAX_ROTATION_ERROR_DEG} deg"
                )
        else:
            raise ValueError(f"Unsupported camera_frame_mode: {camera_frame_mode!r}")

    if depth_diagnostics:
        bound_labels = {str(view.get("bound_split", "")).lower() for view in manifest_views}
        if bound_labels != {"train", "guard", "test"}:
            raise ValueError(
                "Formal diagnostic probe manifest must provide actual train/guard/test views; "
                f"observed={sorted(bound_labels)}"
            )
        if native_cameras_json_path is None:
            raise ValueError("Formal diagnostics require an explicitly validated model cameras.json")
        native_names = set(load_native_camera_entries(native_cameras_json_path))
        by_split = {
            split: {_normalized_stem(view["image_name"]).lower() for view in manifest_views if view["bound_split"] == split}
            for split in ("train", "guard", "test")
        }
        expected_native = by_split["train"] | by_split["test"]
        if native_names != expected_native:
            raise ValueError(
                "Model cameras.json must be exactly the formal train+test camera set; "
                f"missing={sorted(expected_native - native_names)[:8]} extra={sorted(native_names - expected_native)[:8]}"
            )
        if native_names & by_split["guard"]:
            raise ValueError("Guard cameras unexpectedly appear in the existing model cameras.json")
        native_camera_coverage = {
            "policy": "existing model cameras.json is train+test only; guard cameras come from the all-split formal probe",
            "counts": {split: len(by_split[split]) for split in ("train", "guard", "test")},
            "model_cameras_json_count": len(native_names),
            "guard_missing_count": len(by_split["guard"]),
            "guard_silently_reused": False,
            "probe_bound_model_cameras_json_identity_verified": True,
            "render_camera_set_sha256": _render_camera_set_sha256(manifest_views),
        }

    scene_name = scene_name_override if scene_name_override else _infer_scene_name(dataset)
    if formal_split_payload is not None:
        formal_scenes = {
            str(record.get("scene_name", record.get("scene", ""))).strip()
            for record in formal_split_payload.get("records", [])
            if isinstance(record, dict)
        }
        formal_scenes.discard("")
        if formal_scenes and formal_scenes != {scene_name}:
            raise ValueError(f"Formal split scene mismatch: model={scene_name!r}, formal={sorted(formal_scenes)}")
    split_manifest = {
        "bundle_type": "gaussian_probe_split_bundle_v1",
        "producer_identity": _producer_identity(),
        "scene_name": scene_name,
        "split_label": split_label,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "appearance_modality": str(appearance_modality),
        "model_path": str(Path(dataset.model_path).resolve()),
        "source_path": str(Path(dataset.source_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "model_point_cloud": {
            "path": str(model_point_cloud),
            "size_bytes": int(model_point_cloud.stat().st_size),
            "sha256": _sha256_file(model_point_cloud),
        },
        "gaussian_index_anchor": index_anchor_identity,
        "gaussian_index_binding": gaussian_index_binding,
        "train_list": _optional_file_identity(getattr(dataset, "train_list", "")),
        "test_list": _optional_file_identity(getattr(dataset, "test_list", "")),
        # The rasterizer returns inverse depth rather than metric camera-z depth.
        "depth_semantics": "inverse_camera_z_from_renderer",
        "opacity_semantics": (
            "exact_sum_of_accepted_alpha_times_transmittance"
            if depth_diagnostics
            else "black_bg_plus_white_override_color_render"
        ),
        "depth_diagnostics": {
            "enabled": bool(depth_diagnostics),
            "depth_expected_alpha_normalized": "metric camera-z; sum(alpha*T*z)/sum(alpha*T)",
            "depth_transmittance_median": "metric camera-z at first accepted contributor where transmittance <= 0.5; zero if absent",
            "depth_max_contribution": "metric camera-z of Gaussian maximizing alpha*T",
            "top_contributor_index": "zero-based Gaussian index maximizing alpha*T; -1 if absent",
            "top_contributor_weight": "unnormalized compositing weight alpha*T",
            "accumulated_opacity": "sum of accepted alpha*T weights",
        },
        "camera_frame_mode": str(camera_frame_mode),
        "render_resolution": {
            "resolution_arg": int(dataset.resolution),
        },
        "probe_camera_manifest": str(probe_camera_manifest_path.resolve()) if probe_camera_manifest_path is not None else "",
        "probe_camera_manifest_identity": _optional_file_identity(probe_camera_manifest_path),
        "formal_split_manifest": str(formal_split_manifest_path.resolve()) if formal_split_manifest_path is not None else "",
        "formal_split_manifest_identity": _optional_file_identity(formal_split_manifest_path),
        "camera_set_sha256": _camera_set_sha256(manifest_views) if depth_diagnostics else "",
        "render_camera_set_sha256": _render_camera_set_sha256(manifest_views) if depth_diagnostics else "",
        "native_cameras_json": str(native_cameras_json_path.resolve()) if native_cameras_json_path is not None else "",
        "native_cameras_json_identity": _optional_file_identity(native_cameras_json_path),
        "native_camera_coverage": native_camera_coverage,
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
    parser.add_argument("--formal_split_manifest", default="")
    parser.add_argument(
        "--gaussian_index_anchor_ply",
        default="",
        help="Optional topology/index-space anchor PLY (for example the pre-SCSP raw anchor); defaults to the rendered PLY.",
    )
    parser.add_argument(
        "--gaussian_index_binding_manifest",
        default="",
        help="Required fixed-topology/index-order audit receipt when the rendered PLY and anchor have different ordered XYZ.",
    )
    parser.add_argument(
        "--depth_diagnostics",
        action="store_true",
        help="Opt in to metric expected/median/max-contribution depth and top-contributor outputs; legacy output is unchanged by default.",
    )
    parser.add_argument(
        "--appearance_modality",
        choices=["rgb", "thermal_canonical", "none"],
        default="none",
        help="Explicitly declares the bundle appearance source; only rgb enables RGB residual responsibility.",
    )
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
        formal_split_manifest_path=(
            Path(args.formal_split_manifest).resolve() if str(args.formal_split_manifest).strip() else None
        ),
        gaussian_index_anchor_path=(
            Path(args.gaussian_index_anchor_ply).resolve() if str(args.gaussian_index_anchor_ply).strip() else None
        ),
        gaussian_index_binding_manifest_path=(
            Path(args.gaussian_index_binding_manifest).resolve()
            if str(args.gaussian_index_binding_manifest).strip()
            else None
        ),
        depth_diagnostics=bool(args.depth_diagnostics),
        appearance_modality=str(args.appearance_modality),
    )
    print(f"PROBE_BUNDLE_SAVED {manifest_path}")


if __name__ == "__main__":
    main()
