"""Build a fail-closed train/guard/test probe-camera manifest.

The formal training-camera set remains the only input to the repository's
NerfPP normalization calculation.  Guard and test cameras are loaded together
as held-out views from the same all-view COLMAP workspace.  An existing model
``cameras.json`` is evidence only: for the current protocol it must contain
exactly train+test and must not be silently treated as if it contained guard.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.geometric_repeatability.depth_reference_common import compute_scaled_resolution
from utils.graphics_utils import fov2focal, getWorld2View2
from utils.read_write_model import read_images_binary, read_images_text


SPLITS = ("train", "guard", "test")
MODEL_CAMERA_MAX_POSITION_ERROR_M = 1.0e-4
MODEL_CAMERA_MAX_ROTATION_ERROR_DEG = 5.0e-2


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


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stem(value: Any) -> str:
    return Path(str(value).strip().replace("\\", "/")).stem.lower()


def _camera_to_world(camera: Any) -> np.ndarray:
    w2c = getWorld2View2(
        camera.R,
        camera.T,
        np.zeros((3,), dtype=np.float64),
        1.0,
    ).astype(np.float64)
    return np.linalg.inv(w2c)


def camera_sha256(view: Mapping[str, Any]) -> str:
    payload = {
        "width": int(view["width"]),
        "height": int(view["height"]),
        "fx_fy_cx_cy": [float(view[key]) for key in ("fx", "fy", "cx", "cy")],
        "camera_to_world": np.asarray(view["camera_to_world"], dtype=np.float64).tolist(),
    }
    return _canonical_sha(payload)


def camera_set_sha256(views: list[Mapping[str, Any]]) -> str:
    return _canonical_sha(
        [(str(view["image_name"]), camera_sha256(view)) for view in sorted(views, key=lambda row: str(row["image_name"]))]
    )


def _write_name_list(path: Path, names: list[str]) -> Dict[str, Any]:
    data = "".join(f"{name}\n" for name in names).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError(f"Existing formal camera list differs: {path}")
    path.write_bytes(data)
    return _identity(path)


def _registered_names(source_path: Path) -> tuple[set[str], Dict[str, Any]]:
    sparse = source_path / "sparse" / "0"
    image_bin = sparse / "images.bin"
    image_txt = sparse / "images.txt"
    if image_bin.is_file():
        images = read_images_binary(str(image_bin))
        image_identity = _identity(image_bin)
    elif image_txt.is_file():
        images = read_images_text(str(image_txt))
        image_identity = _identity(image_txt)
    else:
        raise FileNotFoundError(f"Missing COLMAP images.bin/images.txt under {sparse}")
    names = {str(item.name) for item in images.values()}
    if len(names) != len(images):
        raise ValueError("COLMAP workspace contains duplicate registered image names")
    artifacts: Dict[str, Any] = {"registered_images": image_identity}
    for stem in ("cameras", "points3D"):
        candidates = [sparse / f"{stem}.bin", sparse / f"{stem}.txt"]
        selected = next((candidate for candidate in candidates if candidate.is_file()), None)
        if selected is None:
            raise FileNotFoundError(f"Missing COLMAP {stem}.bin/.txt under {sparse}")
        artifacts[stem] = _identity(selected)
    return names, artifacts


def _model_camera_map(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Model cameras.json must contain a list")
    mapped: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Model cameras.json entries must be objects")
        key = _stem(item.get("img_name", ""))
        if not key or key in mapped:
            raise ValueError(f"Model cameras.json has missing/duplicate camera {key!r}")
        for field in ("width", "height", "fx", "fy", "position", "rotation"):
            if field not in item:
                raise ValueError(f"Model camera {key!r} is missing {field}")
        mapped[key] = item
    return mapped


def _validate_model_camera(entry: Mapping[str, Any], camera: Any, name: str) -> None:
    c2w = _camera_to_world(camera)
    if (int(entry["width"]), int(entry["height"])) != (int(camera.width), int(camera.height)):
        raise ValueError(f"{name}: model/source camera dimensions mismatch")
    expected_focal = np.asarray(
        [fov2focal(camera.FovX, camera.width), fov2focal(camera.FovY, camera.height)],
        dtype=np.float64,
    )
    observed_focal = np.asarray([entry["fx"], entry["fy"]], dtype=np.float64)
    if not np.allclose(observed_focal, expected_focal, rtol=1e-7, atol=1e-5):
        raise ValueError(f"{name}: model/source focal mismatch")
    observed_position = np.asarray(entry["position"], dtype=np.float64)
    observed_rotation = np.asarray(entry["rotation"], dtype=np.float64)
    if observed_position.shape != (3,) or observed_rotation.shape != (3, 3):
        raise ValueError(f"{name}: model camera pose has invalid shape")
    position_error = float(np.linalg.norm(observed_position - c2w[:3, 3]))
    relative_rotation = observed_rotation @ c2w[:3, :3].T
    cosine = np.clip((float(np.trace(relative_rotation)) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = float(np.degrees(np.arccos(cosine)))
    if not np.isfinite(position_error) or position_error > MODEL_CAMERA_MAX_POSITION_ERROR_M:
        raise ValueError(
            f"{name}: model/source camera position mismatch "
            f"({position_error} m > {MODEL_CAMERA_MAX_POSITION_ERROR_M} m)"
        )
    if not np.isfinite(rotation_error) or rotation_error > MODEL_CAMERA_MAX_ROTATION_ERROR_DEG:
        raise ValueError(
            f"{name}: model/source camera rotation mismatch "
            f"({rotation_error} deg > {MODEL_CAMERA_MAX_ROTATION_ERROR_DEG} deg)"
        )


def build_manifest(
    *,
    source_path: Path,
    images_dir_name: str,
    resolution_arg: int,
    bound_split_path: Path,
    model_cameras_json_path: Path,
    camera_name_field: str,
    scene_name: str,
    out_path: Path,
) -> Dict[str, Any]:
    source_path = source_path.resolve()
    bound_split_path = bound_split_path.resolve()
    model_cameras_json_path = model_cameras_json_path.resolve()
    out_path = out_path.resolve()
    if out_path.exists():
        raise FileExistsError(f"Refusing to replace an existing camera manifest: {out_path}")
    if not (source_path / images_dir_name).is_dir():
        raise FileNotFoundError(source_path / images_dir_name)
    points3d_ply = source_path / "sparse" / "0" / "points3D.ply"
    if not points3d_ply.is_file():
        raise FileNotFoundError(
            "Camera-manifest construction is read-only and refuses to let readColmapSceneInfo materialize "
            f"a missing sparse points3D.ply; prepare it in the derived workspace first: {points3d_ply}"
        )

    bound = _load_json(bound_split_path)
    bound_scene = str(bound.get("scene_name", bound.get("scene", ""))).strip()
    if bound_scene != scene_name:
        raise ValueError(f"Bound split scene mismatch: {bound_scene!r} != {scene_name!r}")
    records = bound.get("records")
    if not isinstance(records, list):
        raise ValueError("Bound split records must be a list")
    record_by_stem: Dict[str, Dict[str, Any]] = {}
    split_names: Dict[str, list[str]] = {split: [] for split in SPLITS}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Bound split entries must be objects")
        split = str(record.get("split", "")).strip().lower()
        if split not in split_names:
            raise ValueError(f"Invalid formal split label: {split!r}")
        camera_name = str(record.get(camera_name_field, "")).strip()
        key = _stem(camera_name)
        if not camera_name or not key or key in record_by_stem:
            raise ValueError(f"Missing/duplicate formal camera {camera_name!r}")
        record_by_stem[key] = record
        split_names[split].append(camera_name)
    for split in SPLITS:
        split_names[split] = sorted(split_names[split])
        if not split_names[split] or len(split_names[split]) != len(set(split_names[split])):
            raise ValueError(f"Invalid formal {split} membership")
    expected_names = {name for names in split_names.values() for name in names}
    registered_names, sparse_artifacts = _registered_names(source_path)
    if registered_names != expected_names:
        raise ValueError(
            "All-view COLMAP workspace registration differs from the formal 3-way partition; "
            f"missing={sorted(expected_names - registered_names)[:8]} extra={sorted(registered_names - expected_names)[:8]}"
        )

    list_root = out_path.parent / "camera_lists"
    train_list = list_root / "train.txt"
    heldout_list = list_root / "guard_plus_test.txt"
    train_identity = _write_name_list(train_list, split_names["train"])
    heldout_identity = _write_name_list(heldout_list, sorted(split_names["guard"] + split_names["test"]))
    # The scene reader imports CUDA-only Gaussian dependencies.  Delay that
    # import until actual manifest construction so the pure hashing helpers
    # remain usable in CPU-only validation and unit tests.
    from scene.dataset_readers import readColmapSceneInfo

    with contextlib.redirect_stdout(io.StringIO()):
        scene_info = readColmapSceneInfo(
            path=str(source_path),
            images=images_dir_name,
            depths="",
            eval=True,
            train_test_exp=False,
            train_list=str(train_list),
            test_list=str(heldout_list),
        )
    loaded_train = {camera.image_name for camera in scene_info.train_cameras}
    loaded_heldout = {camera.image_name for camera in scene_info.test_cameras}
    if loaded_train != set(split_names["train"]):
        raise ValueError("Loaded train cameras differ from the formal 470-train membership")
    if loaded_heldout != set(split_names["guard"] + split_names["test"]):
        raise ValueError("Loaded held-out cameras differ from formal guard+test membership")

    cameras = {camera.image_name: camera for camera in (*scene_info.train_cameras, *scene_info.test_cameras)}
    if set(cameras) != expected_names:
        raise ValueError("Loaded camera union is not the complete formal partition")
    model_cameras = _model_camera_map(model_cameras_json_path)
    expected_model_stems = {_stem(name) for name in split_names["train"] + split_names["test"]}
    guard_stems = {_stem(name) for name in split_names["guard"]}
    if set(model_cameras) != expected_model_stems:
        raise ValueError(
            "Existing model cameras.json must be exactly train+test (guard is intentionally absent); "
            f"missing={sorted(expected_model_stems - set(model_cameras))[:8]} "
            f"extra={sorted(set(model_cameras) - expected_model_stems)[:8]}"
        )
    if set(model_cameras) & guard_stems:
        raise ValueError("Existing model cameras.json unexpectedly contains guard cameras")
    for name, camera in cameras.items():
        key = _stem(name)
        if key in model_cameras:
            _validate_model_camera(model_cameras[key], camera, name)

    views: list[Dict[str, Any]] = []
    for index, name in enumerate(sorted(cameras)):
        camera = cameras[name]
        record = record_by_stem[_stem(name)]
        width, height = compute_scaled_resolution(int(camera.width), int(camera.height), int(resolution_arg))
        view: Dict[str, Any] = {
            "view_id": f"{index:05d}",
            "image_name": name,
            "pair_id": str(record.get("pair_id", _stem(name))),
            "width": int(width),
            "height": int(height),
            "fx": float(fov2focal(camera.FovX, width)),
            "fy": float(fov2focal(camera.FovY, height)),
            "cx": float(width / 2.0),
            "cy": float(height / 2.0),
            "camera_to_world": _camera_to_world(camera).tolist(),
            "bound_split": str(record["split"]).lower(),
            "split": str(record["split"]).lower(),
        }
        for key in ("block_id", "block", "block_index", "strip_id", "stratum", "gimbal_pitch_deg"):
            if key in record:
                view[key] = record[key]
        view["camera_sha256"] = camera_sha256(view)
        views.append(view)

    normalization = {
        "translate": np.asarray(scene_info.nerf_normalization["translate"], dtype=np.float64).tolist(),
        "radius": float(scene_info.nerf_normalization["radius"]),
    }
    normalization["sha256"] = _canonical_sha(normalization)
    payload: Dict[str, Any] = {
        "camera_manifest_type": "formal_all_split_probe_camera_manifest_v1",
        "scene_name": scene_name,
        "source_path": str(source_path),
        "images_dir_name": images_dir_name,
        "resolution_arg": int(resolution_arg),
        "bound_split_manifest_identity": _identity(bound_split_path),
        "source_workspace_identity": {
            "path": str(source_path),
            "sparse_artifacts": sparse_artifacts,
            "registered_camera_count": len(registered_names),
            "registered_names_sha256": _canonical_sha(sorted(registered_names)),
        },
        "model_cameras_json_identity": _identity(model_cameras_json_path),
        "model_camera_coverage": {
            "policy": "existing model cameras.json is train+test only; guard is loaded from the all-view workspace",
            "train_count": len(split_names["train"]),
            "guard_count": len(split_names["guard"]),
            "test_count": len(split_names["test"]),
            "model_camera_count": len(model_cameras),
            "guard_silently_reused": False,
        },
        "formal_train_normalization": {
            "computed_from": "formal train cameras only",
            "train_camera_count": len(split_names["train"]),
            **normalization,
        },
        "generated_camera_lists": {"train": train_identity, "guard_plus_test": heldout_identity},
        "camera_set_sha256": camera_set_sha256(views),
        "views": views,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_path", required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--bound_split", required=True)
    parser.add_argument("--model_cameras_json", required=True)
    parser.add_argument("--camera_name_field", choices=["camera_name", "thermal_camera_name"], required=True)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--out_manifest", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = build_manifest(
        source_path=Path(args.source_path),
        images_dir_name=str(args.images),
        resolution_arg=int(args.resolution),
        bound_split_path=Path(args.bound_split),
        model_cameras_json_path=Path(args.model_cameras_json),
        camera_name_field=str(args.camera_name_field),
        scene_name=str(args.scene_name),
        out_path=Path(args.out_manifest),
    )
    print(f"FORMAL_ALL_SPLIT_CAMERA_MANIFEST {Path(args.out_manifest).resolve()}")
    print(f"FORMAL_ALL_SPLIT_CAMERA_COUNT {len(payload['views'])}")


if __name__ == "__main__":
    main()
