from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import Camera, Image, Point3D, read_model, write_model


PROTOCOL_NAME = "formal-reference-depth-filtered-train-workspace-v2"
SCHEMA_NAME = "uav_tgs_train_only_colmap_workspace"
SCHEMA_VERSION = 2
FORMAL_BINDING_SCHEMA = "uav_tgs_formal_scene_decode_binding"
FORMAL_SFM_SCOPE = "shared_sfm_all_images"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object: {path}")
    return payload


def _array_semantic_record(value: Any) -> Dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": [int(item) for item in array.shape],
        "sha256": digest.hexdigest(),
    }


def _model_semantic_sha256(
    cameras: Mapping[int, Camera],
    images: Mapping[int, Image],
    points3d: Mapping[int, Point3D],
) -> str:
    """Hash decoded COLMAP semantics independently of BIN/TXT serialization."""
    payload = {
        "cameras": [
            {
                "id": int(camera_id),
                "model": str(camera.model),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": _array_semantic_record(camera.params),
            }
            for camera_id, camera in sorted(cameras.items())
        ],
        "images": [
            {
                "id": int(image_id),
                "camera_id": int(image.camera_id),
                "name": _portable_name(image.name),
                "qvec": _array_semantic_record(image.qvec),
                "tvec": _array_semantic_record(image.tvec),
                "xys": _array_semantic_record(image.xys),
                "point3D_ids": _array_semantic_record(image.point3D_ids),
            }
            for image_id, image in sorted(images.items())
        ],
        "points3D": [
            {
                "id": int(point_id),
                "xyz": _array_semantic_record(point.xyz),
                "rgb": _array_semantic_record(point.rgb),
                "error": float(point.error),
                "image_ids": _array_semantic_record(point.image_ids),
                "point2D_idxs": _array_semantic_record(point.point2D_idxs),
            }
            for point_id, point in sorted(points3d.items())
        ],
    }
    return _canonical_sha256(payload)


def _portable_name(raw_name: str) -> str:
    text = str(raw_name).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or text.startswith("/"):
        raise ValueError(f"Image name must be a non-empty relative path: {raw_name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe image name: {raw_name!r}")
    return path.as_posix()


def _name_key(raw_name: str) -> str:
    return _portable_name(raw_name).casefold()


def _load_name_list(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} list not found: {path}")
    names = [
        _portable_name(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"{label} list is empty: {path}")
    keys = [_name_key(name) for name in names]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} list contains duplicate or case-colliding image names: {path}")
    return names


def _validate_binding_provenance(
    *,
    binding_manifest_path: Path,
    sfm_image_scope: str,
    train_list_path: Path,
    test_list_path: Path,
    guard_list_path: Path,
    train_names: Sequence[str],
    test_names: Sequence[str],
    guard_names: Sequence[str],
) -> Dict[str, Any]:
    if sfm_image_scope != FORMAL_SFM_SCOPE:
        raise ValueError(
            "Formal reference construction requires "
            f"--sfm-image-scope {FORMAL_SFM_SCOPE}; got {sfm_image_scope!r}"
        )
    payload = _load_json_object(binding_manifest_path, "formal binding manifest")
    if payload.get("schema_name") != FORMAL_BINDING_SCHEMA:
        raise ValueError(
            "Unexpected formal binding schema: "
            f"{payload.get('schema_name')!r} != {FORMAL_BINDING_SCHEMA!r}"
        )
    if payload.get("status") != "passed":
        raise ValueError("Formal binding manifest status is not passed")
    if payload.get("sfm_image_scope") != sfm_image_scope:
        raise ValueError(
            "Formal binding SfM scope mismatch: "
            f"{payload.get('sfm_image_scope')!r} != {sfm_image_scope!r}"
        )
    if not str(payload.get("binding_hash", "")).strip():
        raise ValueError("Formal binding manifest has no binding_hash")

    expected_counts = {
        "total": len(train_names) + len(test_names) + len(guard_names),
        "train": len(train_names),
        "test": len(test_names),
        "guard": len(guard_names),
    }
    declared_counts = payload.get("counts")
    if declared_counts != expected_counts:
        raise ValueError(
            f"Formal binding counts mismatch: {declared_counts!r} != {expected_counts!r}"
        )

    declared_outputs = payload.get("outputs")
    if not isinstance(declared_outputs, Mapping):
        raise ValueError("Formal binding manifest has no outputs object")
    list_rows: Dict[str, Any] = {}
    for label, path, names in (
        ("train", train_list_path, train_names),
        ("test", test_list_path, test_names),
        ("guard", guard_list_path, guard_names),
    ):
        output_name = f"{label}_list.txt"
        declared = declared_outputs.get(output_name)
        if not isinstance(declared, Mapping):
            raise ValueError(f"Formal binding manifest is missing {output_name}")
        actual_sha = _sha256_file(path)
        if declared.get("sha256") != actual_sha:
            raise ValueError(
                f"Formal binding {label} list SHA mismatch: "
                f"{declared.get('sha256')!r} != {actual_sha!r}"
            )
        list_rows[label] = {
            "path": str(path),
            "file_sha256": actual_sha,
            "ordered_names_semantic_sha256": _canonical_sha256(
                [_portable_name(name) for name in names]
            ),
            "count": len(names),
            "binding_declared_path": str(declared.get("path", "")),
        }

    partition_semantics = {
        "sfm_image_scope": sfm_image_scope,
        "counts": expected_counts,
        "train": [_portable_name(name) for name in train_names],
        "test": [_portable_name(name) for name in test_names],
        "guard": [_portable_name(name) for name in guard_names],
    }
    return {
        "path": str(binding_manifest_path),
        "file_sha256": _sha256_file(binding_manifest_path),
        "semantic_sha256": _canonical_sha256(payload),
        "schema_name": payload.get("schema_name"),
        "schema_version": payload.get("schema_version"),
        "binding_hash": payload.get("binding_hash"),
        "scene": payload.get("scene"),
        "sfm_image_scope": sfm_image_scope,
        "counts": expected_counts,
        "partition_semantic_sha256": _canonical_sha256(partition_semantics),
        "lists": list_rows,
        "upstream_hashes": {
            key: payload.get(key)
            for key in (
                "collection_hash",
                "collection_split_hash",
                "formal_rule_hash",
                "scene_split_hash",
                "scene_manifest_sha256",
                "decode_manifest_sha256",
                "decode_protocol_sha256",
                "decode_protocol_hash",
            )
        },
    }


def _detect_model_ext(model_root: Path) -> str:
    complete: list[str] = []
    for ext in (".bin", ".txt"):
        if all((model_root / f"{stem}{ext}").is_file() for stem in ("cameras", "images", "points3D")):
            complete.append(ext)
    if not complete:
        raise FileNotFoundError(
            f"No complete COLMAP BIN or TXT model found under {model_root}"
        )
    # Match utils.read_write_model.read_model's deterministic preference.
    return ".bin" if ".bin" in complete else ".txt"


def _indexed_names(names: Iterable[str], label: str) -> Dict[str, str]:
    indexed: Dict[str, str] = {}
    for raw_name in names:
        name = _portable_name(raw_name)
        key = _name_key(name)
        if key in indexed:
            raise ValueError(
                f"{label} contains case-colliding names: {indexed[key]!r} and {name!r}"
            )
        indexed[key] = name
    return indexed


def _validate_partition(
    images: Mapping[int, Image],
    train_names: Sequence[str],
    test_names: Sequence[str],
    guard_names: Sequence[str],
) -> tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, Image]]:
    train = _indexed_names(train_names, "train list")
    test = _indexed_names(test_names, "test list")
    guard = _indexed_names(guard_names, "guard list")
    overlaps = {
        "train_test": sorted(set(train) & set(test)),
        "train_guard": sorted(set(train) & set(guard)),
        "test_guard": sorted(set(test) & set(guard)),
    }
    nonempty = {label: values for label, values in overlaps.items() if values}
    if nonempty:
        raise ValueError(f"Train/test/guard lists overlap; refusing materialization: {nonempty}")

    model_by_key: Dict[str, Image] = {}
    for image in images.values():
        key = _name_key(image.name)
        if key in model_by_key:
            raise ValueError(
                "COLMAP model contains case-colliding image names: "
                f"{model_by_key[key].name!r} and {image.name!r}"
            )
        model_by_key[key] = image

    partition_keys = set(train) | set(test) | set(guard)
    unclassified_model = sorted(set(model_by_key) - partition_keys)
    if unclassified_model:
        samples = [model_by_key[key].name for key in unclassified_model[:10]]
        raise ValueError(
            "All-view COLMAP model contains images outside the supplied train/test/guard "
            f"partition: {samples}"
        )
    return train, test, guard, model_by_key


def _validate_bidirectional_model(
    images: Mapping[int, Image],
    points3d: Mapping[int, Point3D],
    *,
    label: str,
) -> None:
    track_entry_count = 0
    for point_id, point in points3d.items():
        if len(point.image_ids) != len(point.point2D_idxs):
            raise ValueError(f"{label}: point {point_id} has mismatched track arrays")
        pairs: set[tuple[int, int]] = set()
        for image_id_raw, point2d_idx_raw in zip(point.image_ids, point.point2D_idxs):
            image_id = int(image_id_raw)
            point2d_idx = int(point2d_idx_raw)
            pair = (image_id, point2d_idx)
            if pair in pairs:
                raise ValueError(f"{label}: point {point_id} contains duplicate track entry {pair}")
            pairs.add(pair)
            image = images.get(image_id)
            if image is None:
                raise ValueError(f"{label}: point {point_id} tracks missing image {image_id}")
            if point2d_idx < 0 or point2d_idx >= len(image.point3D_ids):
                raise ValueError(
                    f"{label}: point {point_id} track index {point2d_idx} is invalid for image {image_id}"
                )
            if int(image.point3D_ids[point2d_idx]) != int(point_id):
                raise ValueError(
                    f"{label}: point {point_id} track ({image_id}, {point2d_idx}) has no matching reverse reference"
                )
            track_entry_count += 1

    positive_image_reference_count = 0
    for image_id, image in images.items():
        if len(image.xys) != len(image.point3D_ids):
            raise ValueError(f"{label}: image {image_id} has mismatched xys/point3D_ids arrays")
        for point2d_idx, point_id_raw in enumerate(image.point3D_ids):
            point_id = int(point_id_raw)
            if point_id < 0:
                continue
            positive_image_reference_count += 1
            if point_id not in points3d:
                raise ValueError(f"{label}: image {image_id} references missing point {point_id}")
    # Every unique track entry has already been proven to point back to the
    # corresponding image observation. Equal cardinality therefore proves the
    # converse without retaining a model-sized set of track tuples in memory.
    if positive_image_reference_count != track_entry_count:
        raise ValueError(
            f"{label}: reverse-reference count differs from track-entry count "
            f"({positive_image_reference_count} != {track_entry_count})"
        )


def _filter_model(
    cameras: Mapping[int, Camera],
    images: Mapping[int, Image],
    points3d: Mapping[int, Point3D],
    selected_image_ids: set[int],
) -> tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    _validate_bidirectional_model(images, points3d, label="source model")

    filtered_points: Dict[int, Point3D] = {}
    for point_id in sorted(points3d):
        point = points3d[point_id]
        keep = np.fromiter(
            (int(image_id) in selected_image_ids for image_id in point.image_ids),
            dtype=bool,
            count=len(point.image_ids),
        )
        if not bool(np.any(keep)):
            continue
        filtered_points[point_id] = Point3D(
            id=point.id,
            xyz=np.array(point.xyz, copy=True),
            rgb=np.array(point.rgb, copy=True),
            error=point.error,
            image_ids=np.array(point.image_ids[keep], copy=True),
            point2D_idxs=np.array(point.point2D_idxs[keep], copy=True),
        )

    retained_point_ids = set(filtered_points)
    filtered_images: Dict[int, Image] = {}
    for image_id in sorted(selected_image_ids):
        image = images[image_id]
        point_ids = np.array(image.point3D_ids, copy=True)
        point_ids[
            np.fromiter(
                (int(point_id) >= 0 and int(point_id) not in retained_point_ids for point_id in point_ids),
                dtype=bool,
                count=len(point_ids),
            )
        ] = -1
        filtered_images[image_id] = Image(
            id=image.id,
            qvec=np.array(image.qvec, copy=True),
            tvec=np.array(image.tvec, copy=True),
            camera_id=image.camera_id,
            name=image.name,
            xys=np.array(image.xys, copy=True),
            point3D_ids=point_ids,
        )

    used_camera_ids = {int(image.camera_id) for image in filtered_images.values()}
    missing_cameras = sorted(used_camera_ids - set(cameras))
    if missing_cameras:
        raise ValueError(f"Selected images reference missing cameras: {missing_cameras}")
    filtered_cameras = {camera_id: cameras[camera_id] for camera_id in sorted(used_camera_ids)}
    _validate_bidirectional_model(filtered_images, filtered_points, label="filtered model")
    return filtered_cameras, filtered_images, filtered_points


def _camera_pose_semantic_sha256(
    cameras: Mapping[int, Camera], images: Mapping[int, Image]
) -> str:
    payload = {
        "cameras": [
            {
                "id": int(camera_id),
                "model": str(camera.model),
                "width": int(camera.width),
                "height": int(camera.height),
                "params": _array_semantic_record(camera.params),
            }
            for camera_id, camera in sorted(cameras.items())
        ],
        "images": [
            {
                "id": int(image_id),
                "camera_id": int(image.camera_id),
                "name": _portable_name(image.name),
                "qvec": _array_semantic_record(image.qvec),
                "tvec": _array_semantic_record(image.tvec),
            }
            for image_id, image in sorted(images.items())
        ],
    }
    return _canonical_sha256(payload)


def _validate_reloaded_camera_pose_preservation(
    source_cameras: Mapping[int, Camera],
    source_images: Mapping[int, Image],
    reloaded_cameras: Mapping[int, Camera],
    reloaded_images: Mapping[int, Image],
) -> Dict[str, Any]:
    """Compare every retained camera and image pose after serialization/reload."""
    if set(reloaded_cameras) != set(source_cameras):
        raise RuntimeError(
            "Reloaded camera IDs differ from the retained source camera IDs: "
            f"source={sorted(source_cameras)} output={sorted(reloaded_cameras)}"
        )
    if set(reloaded_images) != set(source_images):
        raise RuntimeError(
            "Reloaded image IDs differ from the retained source image IDs: "
            f"source={sorted(source_images)} output={sorted(reloaded_images)}"
        )

    for camera_id in sorted(source_cameras):
        before = source_cameras[camera_id]
        after = reloaded_cameras[camera_id]
        scalar_equal = (
            int(after.id) == int(before.id)
            and str(after.model) == str(before.model)
            and int(after.width) == int(before.width)
            and int(after.height) == int(before.height)
        )
        params_equal = np.array_equal(
            np.asarray(after.params), np.asarray(before.params)
        )
        if not scalar_equal or not params_equal:
            raise RuntimeError(
                f"Reloaded camera {camera_id} differs from its retained source camera"
            )

    for image_id in sorted(source_images):
        before = source_images[image_id]
        after = reloaded_images[image_id]
        scalar_equal = (
            int(after.id) == int(before.id)
            and int(after.camera_id) == int(before.camera_id)
            and str(after.name) == str(before.name)
        )
        qvec_equal = np.array_equal(np.asarray(after.qvec), np.asarray(before.qvec))
        tvec_equal = np.array_equal(np.asarray(after.tvec), np.asarray(before.tvec))
        if not scalar_equal or not qvec_equal or not tvec_equal:
            raise RuntimeError(
                f"Reloaded image {image_id} camera/name/qvec/tvec differs from source"
            )

    source_semantic_sha = _camera_pose_semantic_sha256(
        source_cameras, source_images
    )
    output_semantic_sha = _camera_pose_semantic_sha256(
        reloaded_cameras, reloaded_images
    )
    if source_semantic_sha != output_semantic_sha:
        raise RuntimeError("Reloaded camera/pose semantic hash differs from source")
    return {
        "status": "passed",
        "comparison": "np.array_equal plus exact scalar/string equality",
        "camera_count": len(source_cameras),
        "image_count": len(source_images),
        "camera_fields_checked": ["id", "model", "width", "height", "params"],
        "image_fields_checked": ["id", "camera_id", "name", "qvec", "tvec"],
        "source_semantic_sha256": source_semantic_sha,
        "output_semantic_sha256": output_semantic_sha,
        "all_retained_camera_parameters_exact": True,
        "all_retained_image_poses_exact": True,
    }


def _resolve_image_file(image_root: Path, actual_name: str) -> Path:
    relative = Path(*PurePosixPath(_portable_name(actual_name)).parts)
    direct = image_root / relative
    if direct.is_file():
        return direct

    # Windows paths are case-insensitive, whereas formal workspaces are also
    # materialized on Linux. Resolve each component case-insensitively but fail
    # on ambiguity so a case mismatch cannot silently select the wrong frame.
    current = image_root
    for part in relative.parts:
        if not current.is_dir():
            raise FileNotFoundError(f"RGB image not found: {direct}")
        matches = [candidate for candidate in current.iterdir() if candidate.name.casefold() == part.casefold()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"RGB image path component {part!r} has {len(matches)} case-insensitive matches under {current}"
            )
        current = matches[0]
    if not current.is_file():
        raise FileNotFoundError(f"RGB image not found: {direct}")
    return current


def _copy_or_link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:  # Defensive guard for direct Python callers.
        raise ValueError(f"Unsupported image mode: {mode}")


def _file_record(path: Path) -> Dict[str, Any]:
    return {"sha256": _sha256_file(path), "size_bytes": int(path.stat().st_size)}


def _bundle_hash(records: Mapping[str, Mapping[str, Any]]) -> str:
    return _canonical_sha256({name: dict(record) for name, record in sorted(records.items())})


def materialize_train_only_workspace(
    *,
    source_model_root: Path,
    binding_manifest_path: Path,
    sfm_image_scope: str,
    train_list_path: Path,
    test_list_path: Path,
    guard_list_path: Path,
    image_root: Path,
    output_workspace: Path,
    image_mode: str = "copy",
) -> Dict[str, Any]:
    raw_output_workspace = output_workspace.expanduser()
    if raw_output_workspace.exists() or raw_output_workspace.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output workspace: {raw_output_workspace}")
    source_model_root = source_model_root.resolve()
    binding_manifest_path = binding_manifest_path.resolve()
    train_list_path = train_list_path.resolve()
    test_list_path = test_list_path.resolve()
    guard_list_path = guard_list_path.resolve()
    image_root = image_root.resolve()
    output_workspace = raw_output_workspace.resolve()

    if image_mode not in {"copy", "hardlink"}:
        raise ValueError(f"Unsupported image mode: {image_mode}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"RGB image root not found: {image_root}")
    if output_workspace.exists() or output_workspace.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output workspace: {output_workspace}")
    if image_root == output_workspace or image_root in output_workspace.parents:
        raise ValueError(f"Output workspace must not be created inside the RGB image root: {image_root}")
    if source_model_root == output_workspace or source_model_root in output_workspace.parents:
        raise ValueError(f"Output workspace must not be created inside the source sparse model: {source_model_root}")

    ext = _detect_model_ext(source_model_root)
    cameras, images, points3d = read_model(str(source_model_root), ext=ext)
    if not images:
        raise ValueError(f"Source COLMAP model has no registered images: {source_model_root}")

    train_names = _load_name_list(train_list_path, "train")
    test_names = _load_name_list(test_list_path, "test")
    guard_names = _load_name_list(guard_list_path, "guard")
    train, test, guard, model_by_key = _validate_partition(
        images, train_names, test_names, guard_names
    )
    binding_provenance = _validate_binding_provenance(
        binding_manifest_path=binding_manifest_path,
        sfm_image_scope=sfm_image_scope,
        train_list_path=train_list_path,
        test_list_path=test_list_path,
        guard_list_path=guard_list_path,
        train_names=train_names,
        test_names=test_names,
        guard_names=guard_names,
    )

    selected_images = {
        int(model_by_key[key].id): model_by_key[key]
        for key in train
        if key in model_by_key
    }
    if not selected_images:
        raise ValueError("No train-list images are registered in the source COLMAP model")
    filtered_cameras, filtered_images, filtered_points = _filter_model(
        cameras,
        images,
        points3d,
        set(selected_images),
    )

    # Every partition member is resolved before output creation. This prevents a
    # partially materialized workspace and proves that excluded views exist in
    # the same RGB source collection even if they were not registered by SfM.
    actual_name_by_partition_key = {
        key: model_by_key[key].name if key in model_by_key else declared
        for partition in (train, test, guard)
        for key, declared in partition.items()
    }
    source_file_by_key = {
        key: _resolve_image_file(image_root, actual_name)
        for key, actual_name in actual_name_by_partition_key.items()
    }

    output_workspace.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_workspace.name}.tmp-",
            dir=str(output_workspace.parent),
        )
    )
    try:
        output_images = staging / "images"
        output_model = staging / "sparse" / "0"
        output_images.mkdir(parents=True)
        output_model.mkdir(parents=True)

        source_image_records: Dict[str, Dict[str, Any]] = {}
        output_image_records: Dict[str, Dict[str, Any]] = {}
        actual_train_names: list[str] = []
        for key in train:
            actual_name = actual_name_by_partition_key[key]
            source = source_file_by_key[key]
            destination = output_images / Path(*PurePosixPath(_portable_name(actual_name)).parts)
            _copy_or_link(source, destination, image_mode)
            source_record = _file_record(source)
            output_record = _file_record(destination)
            if source_record != output_record:
                raise RuntimeError(f"Materialized RGB image differs from source: {actual_name}")
            source_image_records[actual_name] = source_record
            output_image_records[actual_name] = output_record
            actual_train_names.append(actual_name)

        write_model(
            filtered_cameras,
            filtered_images,
            filtered_points,
            str(output_model),
            ext=ext,
        )
        reloaded_cameras, reloaded_images, reloaded_points = read_model(
            str(output_model), ext=ext
        )
        _validate_bidirectional_model(
            reloaded_images, reloaded_points, label="reloaded output model"
        )
        reload_preservation = _validate_reloaded_camera_pose_preservation(
            filtered_cameras,
            filtered_images,
            reloaded_cameras,
            reloaded_images,
        )

        model_filenames = [f"{stem}{ext}" for stem in ("cameras", "images", "points3D")]
        source_model_records = {
            name: _file_record(source_model_root / name) for name in model_filenames
        }
        output_model_records = {name: _file_record(output_model / name) for name in model_filenames}
        output_model_names = {_name_key(image.name) for image in reloaded_images.values()}
        if not output_model_names <= set(train):
            raise RuntimeError("Reloaded output sparse model contains a test, guard, or unclassified image")
        if output_model_names & (set(test) | set(guard)):
            raise RuntimeError("Reloaded output sparse model contains an excluded test/guard image")

        dropped_empty_points = len(points3d) - len(filtered_points)
        source_track_entries = sum(len(point.image_ids) for point in points3d.values())
        output_track_entries = sum(len(point.image_ids) for point in filtered_points.values())
        manifest: Dict[str, Any] = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "protocol_name": PROTOCOL_NAME,
            "status": "passed",
            "formal_provenance": binding_provenance,
            "source": {
                "model_root": str(source_model_root),
                "model_format": ext.lstrip("."),
                "model_files": source_model_records,
                "model_bundle_sha256": _bundle_hash(source_model_records),
                "model_semantic_sha256": _model_semantic_sha256(
                    cameras, images, points3d
                ),
                "sfm_image_scope": sfm_image_scope,
                "sfm_semantics": (
                    "shared all-image SfM supplies cameras, poses, observations, "
                    "and tracks; this tool only filters that model for reference construction"
                ),
                "image_root": str(image_root),
                "selected_train_images": source_image_records,
                "selected_train_images_bundle_sha256": _bundle_hash(source_image_records),
                "counts": {
                    "cameras": len(cameras),
                    "registered_images": len(images),
                    "points3D": len(points3d),
                    "track_entries": source_track_entries,
                },
            },
            "partition": {
                "train_list": {
                    "path": str(train_list_path),
                    "sha256": _sha256_file(train_list_path),
                    "count": len(train),
                    "ordered_names_sha256": _canonical_sha256(train_names),
                },
                "test_list": {
                    "path": str(test_list_path),
                    "sha256": _sha256_file(test_list_path),
                    "count": len(test),
                    "ordered_names_sha256": _canonical_sha256(test_names),
                },
                "guard_list": {
                    "path": str(guard_list_path),
                    "sha256": _sha256_file(guard_list_path),
                    "count": len(guard),
                    "ordered_names_sha256": _canonical_sha256(guard_names),
                },
                "registered": {
                    "train": len(set(model_by_key) & set(train)),
                    "test": len(set(model_by_key) & set(test)),
                    "guard": len(set(model_by_key) & set(guard)),
                },
                "unregistered_train_count": len(set(train) - set(model_by_key)),
                "validation": {
                    "pairwise_disjoint": True,
                    "covers_all_registered_source_images": True,
                    "test_images_rejected_from_output": True,
                    "guard_images_rejected_from_output": True,
                },
            },
            "filtering": {
                "removed_unreferenced_cameras": len(cameras) - len(filtered_cameras),
                "removed_nontrain_registered_images": len(images) - len(filtered_images),
                "removed_points_with_empty_train_track": dropped_empty_points,
                "removed_nontrain_track_entries": source_track_entries - output_track_entries,
                "point2D_arrays_preserved_with_removed_references_set_to_minus_one": True,
                "reload_exact_camera_pose_preservation": reload_preservation,
            },
            "output": {
                "workspace": str(output_workspace),
                "semantics": (
                    "training-view filtered workspace derived from shared all-image SfM; "
                    "not an independently reconstructed train-only SfM model"
                ),
                "image_mode": image_mode,
                "images": output_image_records,
                "images_bundle_sha256": _bundle_hash(output_image_records),
                "sparse_model_files": output_model_records,
                "sparse_model_bundle_sha256": _bundle_hash(output_model_records),
                "sparse_model_semantic_sha256": _model_semantic_sha256(
                    reloaded_cameras, reloaded_images, reloaded_points
                ),
                "counts": {
                    "image_files": len(output_image_records),
                    "cameras": len(filtered_cameras),
                    "registered_images": len(filtered_images),
                    "points3D": len(filtered_points),
                    "track_entries": output_track_entries,
                },
                "actual_train_image_names_sha256": _canonical_sha256(actual_train_names),
            },
            "invariants": {
                "output_image_files_equal_train_list": len(output_image_records) == len(train),
                "output_registered_images_subset_of_train": True,
                "no_test_or_guard_image_files": True,
                "no_test_or_guard_registered_images": True,
                "bidirectional_point_track_references_valid": True,
                "retained_camera_parameters_exact_after_reload": bool(
                    reload_preservation["all_retained_camera_parameters_exact"]
                ),
                "retained_image_poses_exact_after_reload": bool(
                    reload_preservation["all_retained_image_poses_exact"]
                ),
                "source_and_output_train_image_hashes_equal": (
                    _bundle_hash(source_image_records) == _bundle_hash(output_image_records)
                ),
            },
        }
        if not all(bool(value) for value in manifest["invariants"].values()):
            raise RuntimeError(f"Train-only materialization invariants failed: {manifest['invariants']}")
        _write_json(staging / "train_only_colmap_manifest.json", manifest)
        os.replace(staging, output_workspace)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a fail-closed train-only RGB/COLMAP workspace from an all-view "
            "sparse model for formal OpenMVS reference-depth construction."
        )
    )
    parser.add_argument("--source-model", required=True, help="All-view COLMAP model directory (cameras/images/points3D)")
    parser.add_argument(
        "--binding-manifest",
        required=True,
        help="Passed formal scene decode/split binding manifest",
    )
    parser.add_argument(
        "--sfm-image-scope",
        required=True,
        choices=(FORMAL_SFM_SCOPE,),
        help=(
            "Required provenance label for the source reconstruction. The source "
            "SfM used all scene images; only this derived workspace is filtered."
        ),
    )
    parser.add_argument("--train-list", required=True, help="Newline-delimited train camera/image names")
    parser.add_argument("--test-list", required=True, help="Newline-delimited held-out test camera/image names")
    parser.add_argument("--guard-list", required=True, help="Newline-delimited guard camera/image names")
    parser.add_argument("--image-root", required=True, help="RGB image root containing the full partition")
    parser.add_argument("--output-workspace", required=True, help="New workspace to create; existing paths are refused")
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="How to materialize train RGB files (default: copy; hardlink fails if unsupported)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = materialize_train_only_workspace(
        source_model_root=Path(args.source_model),
        binding_manifest_path=Path(args.binding_manifest),
        sfm_image_scope=str(args.sfm_image_scope),
        train_list_path=Path(args.train_list),
        test_list_path=Path(args.test_list),
        guard_list_path=Path(args.guard_list),
        image_root=Path(args.image_root),
        output_workspace=Path(args.output_workspace),
        image_mode=args.image_mode,
    )
    print(f"TRAIN_ONLY_WORKSPACE {manifest['output']['workspace']}")
    print(f"TRAIN_IMAGES {manifest['partition']['train_list']['count']}")
    print(f"REGISTERED_TRAIN_IMAGES {manifest['output']['counts']['registered_images']}")
    print(f"OUTPUT_SPARSE_SHA256 {manifest['output']['sparse_model_bundle_sha256']}")


if __name__ == "__main__":
    main()
