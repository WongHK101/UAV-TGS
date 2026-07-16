#!/usr/bin/env python3
"""Remap distorted float32 Celsius maps with a COLMAP undistorter model.

The input sparse model describes the distorted images used by COLMAP.  The
output sparse model is the ``sparse/0`` model written by COLMAP's
``image_undistorter``.  Every output image is paired with an input image and a
temperature NPY by exact name, relative stem, or (only when unique) basename
stem.  Ambiguous or incomplete associations fail before any output is
published.

Invalid border samples are filled by border replication so that the resulting
temperature maps remain finite.  They MUST be excluded with the accompanying
``valid_support`` boolean masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as error:  # pragma: no cover - depends on runtime environment
    raise ImportError(
        "undistort_temperature requires OpenCV; install opencv-python-headless"
    ) from error


SCHEMA = "uav-tgs-undistorted-temperature-v1"
SUPPORTED_INPUT_MODELS = frozenset(
    {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL"}
)
SUPPORTED_OUTPUT_MODELS = frozenset({"SIMPLE_PINHOLE", "PINHOLE"})

_CAMERA_MODEL_BY_ID = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
_CAMERA_MODEL_BY_NAME = {name: (model_id, count) for model_id, (name, count) in _CAMERA_MODEL_BY_ID.items()}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: Tuple[float, ...]


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    qvec: Tuple[float, float, float, float]
    tvec: Tuple[float, float, float]
    camera_id: int
    name: str


@dataclass(frozen=True)
class SparseModel:
    root: Path
    model_format: str
    cameras: Mapping[int, Camera]
    images: Mapping[int, ImageRecord]
    files: Tuple[Path, ...]


@dataclass(frozen=True)
class RemapJob:
    output_image: ImageRecord
    input_image: ImageRecord
    input_camera: Camera
    output_camera: Camera
    temperature_path: Path
    output_relative: Path
    image_match_mode: str
    temperature_match_mode: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(stream: Any, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"Unexpected end of COLMAP binary file (wanted {size} bytes, got {len(value)})")
    return value


def _unpack(stream: Any, fmt: str) -> Tuple[Any, ...]:
    size = struct.calcsize("<" + fmt)
    return struct.unpack("<" + fmt, _read_exact(stream, size))


def _read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(stream, "iiQQ")
            if model_id not in _CAMERA_MODEL_BY_ID:
                raise ValueError(f"Unknown COLMAP camera model id {model_id} in {path}")
            model, param_count = _CAMERA_MODEL_BY_ID[model_id]
            params = tuple(float(value) for value in _unpack(stream, "d" * param_count))
            if camera_id in cameras:
                raise ValueError(f"Duplicate camera id {camera_id} in {path}")
            cameras[camera_id] = Camera(camera_id, model, int(width), int(height), params)
        if stream.read(1):
            raise ValueError(f"Trailing bytes in COLMAP camera file: {path}")
    return cameras


def _read_images_binary(path: Path) -> Dict[int, ImageRecord]:
    images: Dict[int, ImageRecord] = {}
    with path.open("rb") as stream:
        (count,) = _unpack(stream, "Q")
        for _ in range(count):
            values = _unpack(stream, "i" + "d" * 7 + "i")
            image_id = int(values[0])
            qvec = tuple(float(value) for value in values[1:5])
            tvec = tuple(float(value) for value in values[5:8])
            camera_id = int(values[8])
            name_bytes = bytearray()
            while True:
                value = _read_exact(stream, 1)
                if value == b"\x00":
                    break
                name_bytes.extend(value)
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"Non-UTF-8 image name in {path}") from error
            (point_count,) = _unpack(stream, "Q")
            _read_exact(stream, int(point_count) * 24)
            if image_id in images:
                raise ValueError(f"Duplicate image id {image_id} in {path}")
            images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name)
        if stream.read(1):
            raise ValueError(f"Trailing bytes in COLMAP image file: {path}")
    return images


def _read_cameras_text(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed camera line {line_number} in {path}")
        camera_id = int(fields[0])
        model = fields[1]
        if model not in _CAMERA_MODEL_BY_NAME:
            raise ValueError(f"Unknown COLMAP camera model {model!r} in {path}")
        expected = _CAMERA_MODEL_BY_NAME[model][1]
        params = tuple(float(value) for value in fields[4:])
        if len(params) != expected:
            raise ValueError(
                f"Camera {camera_id} model {model} expects {expected} parameters, got {len(params)}"
            )
        if camera_id in cameras:
            raise ValueError(f"Duplicate camera id {camera_id} in {path}")
        cameras[camera_id] = Camera(camera_id, model, int(fields[2]), int(fields[3]), params)
    return cameras


def _read_images_text(path: Path) -> Dict[int, ImageRecord]:
    images: Dict[int, ImageRecord] = {}
    expect_image = True
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if not expect_image:
            # A COLMAP image record always has a following (possibly empty)
            # POINTS2D line.  Its contents are irrelevant for this operation.
            expect_image = True
            continue
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 10:
            raise ValueError(f"Malformed image line {line_number} in {path}")
        image_id = int(fields[0])
        qvec = tuple(float(value) for value in fields[1:5])
        tvec = tuple(float(value) for value in fields[5:8])
        camera_id = int(fields[8])
        name = " ".join(fields[9:])
        if image_id in images:
            raise ValueError(f"Duplicate image id {image_id} in {path}")
        images[image_id] = ImageRecord(image_id, qvec, tvec, camera_id, name)
        expect_image = False
    return images


def read_sparse_model(root: Path, *, model_format: str = "auto") -> SparseModel:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"COLMAP model directory does not exist: {resolved}")
    complete = {
        "bin": (resolved / "cameras.bin", resolved / "images.bin"),
        "txt": (resolved / "cameras.txt", resolved / "images.txt"),
    }
    available = [name for name, pair in complete.items() if all(path.is_file() for path in pair)]
    if model_format == "auto":
        if len(available) != 1:
            raise ValueError(
                f"Expected exactly one complete COLMAP model format in {resolved}; found {available}. "
                "Select --input-model-format/--output-model-format explicitly if both are intentional."
            )
        selected = available[0]
    else:
        selected = model_format
        if selected not in complete:
            raise ValueError(f"Unsupported model format: {selected}")
        if not all(path.is_file() for path in complete[selected]):
            raise FileNotFoundError(f"Incomplete {selected} COLMAP model in {resolved}")
    camera_path, image_path = complete[selected]
    if selected == "bin":
        cameras = _read_cameras_binary(camera_path)
        images = _read_images_binary(image_path)
    else:
        cameras = _read_cameras_text(camera_path)
        images = _read_images_text(image_path)
    if not cameras or not images:
        raise ValueError(f"COLMAP model contains no cameras or registered images: {resolved}")
    missing_cameras = sorted({image.camera_id for image in images.values()} - set(cameras))
    if missing_cameras:
        raise ValueError(f"Images reference missing cameras in {resolved}: {missing_cameras}")
    model_files = tuple(
        path
        for path in sorted(resolved.iterdir(), key=lambda value: value.name)
        if path.is_file() and path.suffix.lower() == f".{selected}"
    )
    return SparseModel(resolved, selected, cameras, images, model_files)


def _normalise_name(name: str) -> str:
    value = name.replace("\\", "/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not value
        or any(part in ("", ".", "..") or ":" in part for part in path.parts)
    ):
        raise ValueError(f"Unsafe or empty image name in COLMAP model: {name!r}")
    return path.as_posix()


def _relative_stem(name: str) -> str:
    return str(PurePosixPath(_normalise_name(name)).with_suffix(""))


def _resolve_unique(
    target_name: str,
    candidates: Sequence[Any],
    *,
    name_function: Any,
    permit_exact_name: bool,
) -> Tuple[Any, str]:
    target = _normalise_name(target_name)
    strategies = []
    if permit_exact_name:
        strategies.append(("exact_name", lambda value: _normalise_name(value), target))
    target_relative_stem = _relative_stem(target)
    strategies.extend(
        [
            ("relative_stem", lambda value: _relative_stem(value), target_relative_stem),
            ("basename_stem", lambda value: PurePosixPath(_relative_stem(value)).name, PurePosixPath(target_relative_stem).name),
        ]
    )
    for mode, key_function, key in strategies:
        matches = [item for item in candidates if key_function(name_function(item)) == key]
        if len(matches) == 1:
            return matches[0], mode
        if len(matches) > 1:
            names = sorted(name_function(item) for item in matches)
            raise ValueError(f"Ambiguous {mode} match for {target!r}: {names}")
    raise KeyError(f"No same-name/same-stem match for {target!r}")


def _qvec_to_rotmat(qvec: Sequence[float]) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"Invalid COLMAP quaternion: {tuple(qvec)}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def _validate_camera(camera: Camera, *, output: bool) -> None:
    supported = SUPPORTED_OUTPUT_MODELS if output else SUPPORTED_INPUT_MODELS
    role = "output" if output else "input"
    if camera.model not in supported:
        raise ValueError(f"Unsupported {role} camera model {camera.model} for camera {camera.camera_id}")
    if camera.width <= 0 or camera.height <= 0:
        raise ValueError(f"Invalid dimensions for {role} camera {camera.camera_id}")
    if not np.all(np.isfinite(camera.params)):
        raise ValueError(f"Non-finite parameters for {role} camera {camera.camera_id}")
    if camera.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        focal_values = camera.params[:1]
    else:
        focal_values = camera.params[:2]
    if any(value <= 0.0 for value in focal_values):
        raise ValueError(f"Non-positive focal length for {role} camera {camera.camera_id}")


def build_remap(input_camera: Camera, output_camera: Camera) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build output-pixel to distorted-input-pixel maps."""
    _validate_camera(input_camera, output=False)
    _validate_camera(output_camera, output=True)
    columns, rows = np.meshgrid(
        np.arange(output_camera.width, dtype=np.float64),
        np.arange(output_camera.height, dtype=np.float64),
    )
    if output_camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = output_camera.params
        x = (columns - cx) / focal
        y = (rows - cy) / focal
    else:
        fx, fy, cx, cy = output_camera.params
        x = (columns - cx) / fx
        y = (rows - cy) / fy

    if input_camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = input_camera.params
        map_x = focal * x + cx
        map_y = focal * y + cy
    elif input_camera.model == "PINHOLE":
        fx, fy, cx, cy = input_camera.params
        map_x = fx * x + cx
        map_y = fy * y + cy
    else:
        if input_camera.model == "SIMPLE_RADIAL":
            focal, cx, cy, k1 = input_camera.params
            k2 = 0.0
        else:
            focal, cx, cy, k1, k2 = input_camera.params
        radius2 = x * x + y * y
        distortion = 1.0 + k1 * radius2 + k2 * radius2 * radius2
        map_x = focal * distortion * x + cx
        map_y = focal * distortion * y + cy
    if not np.all(np.isfinite(map_x)) or not np.all(np.isfinite(map_y)):
        raise ValueError("Remap contains NaN or infinite coordinates")
    valid = (
        (map_x >= 0.0)
        & (map_y >= 0.0)
        & (map_x <= float(input_camera.width - 1))
        & (map_y <= float(input_camera.height - 1))
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid


def _camera_payload(camera: Camera) -> Dict[str, Any]:
    return {
        "camera_id": camera.camera_id,
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": list(camera.params),
    }


def _model_payload(model: SparseModel) -> Dict[str, Any]:
    return {
        "root": str(model.root),
        "format": model.model_format,
        "camera_count": len(model.cameras),
        "registered_image_count": len(model.images),
        "files": [
            {
                "relative_path": path.relative_to(model.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in model.files
        ],
        "cameras": [_camera_payload(model.cameras[key]) for key in sorted(model.cameras)],
    }


def _build_jobs(
    temperature_root: Path,
    input_model: SparseModel,
    output_model: SparseModel,
    *,
    pose_tolerance: float,
) -> Tuple[Sequence[RemapJob], Sequence[Path]]:
    if len(input_model.images) != len(output_model.images):
        raise ValueError(
            "Input/output registered image counts differ: "
            f"{len(input_model.images)} != {len(output_model.images)}"
        )
    input_images = list(input_model.images.values())
    npy_files = sorted(
        (path for path in temperature_root.rglob("*.npy") if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    if not npy_files:
        raise FileNotFoundError(f"No temperature NPY files found under {temperature_root}")
    jobs = []
    used_input_ids = set()
    used_temperature_paths = set()
    used_outputs = set()
    for output_image in sorted(output_model.images.values(), key=lambda image: _normalise_name(image.name)):
        input_image, image_match_mode = _resolve_unique(
            output_image.name,
            input_images,
            name_function=lambda image: image.name,
            permit_exact_name=True,
        )
        if input_image.image_id in used_input_ids:
            raise ValueError(f"Input image was matched more than once: {input_image.name}")
        used_input_ids.add(input_image.image_id)
        rotation_error = float(
            np.max(np.abs(_qvec_to_rotmat(input_image.qvec) - _qvec_to_rotmat(output_image.qvec)))
        )
        translation_error = float(
            np.max(np.abs(np.asarray(input_image.tvec) - np.asarray(output_image.tvec)))
        )
        if rotation_error > pose_tolerance or translation_error > pose_tolerance:
            raise ValueError(
                f"Input/output poses differ for {output_image.name!r}: "
                f"rotation max-abs={rotation_error}, translation max-abs={translation_error}"
            )
        try:
            temperature_path, temperature_match_mode = _resolve_unique(
                output_image.name,
                npy_files,
                name_function=lambda path: path.relative_to(temperature_root).as_posix(),
                permit_exact_name=False,
            )
        except KeyError as error:
            raise FileNotFoundError(str(error)) from error
        if temperature_path in used_temperature_paths:
            raise ValueError(f"Temperature map was matched more than once: {temperature_path}")
        used_temperature_paths.add(temperature_path)
        input_camera = input_model.cameras[input_image.camera_id]
        output_camera = output_model.cameras[output_image.camera_id]
        _validate_camera(input_camera, output=False)
        _validate_camera(output_camera, output=True)
        values = np.load(temperature_path, allow_pickle=False, mmap_mode="r")
        if values.dtype != np.float32:
            raise ValueError(f"Temperature map must be float32 Celsius: {temperature_path} has {values.dtype}")
        expected_shape = (input_camera.height, input_camera.width)
        if values.ndim != 2 or values.shape != expected_shape:
            raise ValueError(
                f"Temperature shape must match input camera for {temperature_path}: "
                f"expected {expected_shape}, got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Temperature map contains NaN or infinite values: {temperature_path}")
        output_relative = Path(_normalise_name(output_image.name)).with_suffix(".npy")
        output_key = os.path.normcase(output_relative.as_posix())
        if output_key in used_outputs:
            raise ValueError(f"Output stem collision: {output_relative}")
        used_outputs.add(output_key)
        jobs.append(
            RemapJob(
                output_image,
                input_image,
                input_camera,
                output_camera,
                temperature_path,
                output_relative,
                image_match_mode,
                temperature_match_mode,
            )
        )
    unused_input = sorted(set(input_model.images) - used_input_ids)
    if unused_input:
        raise ValueError(f"Input model images were not matched: {unused_input}")
    extra_temperature = sorted(set(npy_files) - used_temperature_paths, key=lambda path: path.as_posix())
    if extra_temperature:
        raise ValueError(
            "Temperature root contains files not represented by the COLMAP output model: "
            + ", ".join(str(path) for path in extra_temperature[:10])
            + (" ..." if len(extra_temperature) > 10 else "")
        )
    return jobs, npy_files


def remap_temperature_tree(
    temperature_root: Path,
    input_model_root: Path,
    output_model_root: Path,
    output_root: Path,
    *,
    input_model_format: str = "auto",
    output_model_format: str = "auto",
    pose_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    source_root = Path(temperature_root).resolve()
    destination_root = Path(output_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Temperature root does not exist: {source_root}")
    if pose_tolerance < 0.0 or not np.isfinite(pose_tolerance):
        raise ValueError("pose_tolerance must be finite and non-negative")
    if destination_root.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {destination_root}")
    try:
        destination_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output root must be outside the temperature source tree")

    input_model = read_sparse_model(input_model_root, model_format=input_model_format)
    output_model = read_sparse_model(output_model_root, model_format=output_model_format)
    jobs, _ = _build_jobs(
        source_root,
        input_model,
        output_model,
        pose_tolerance=pose_tolerance,
    )
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{destination_root.name}.tmp-", dir=destination_root.parent))
    records = []
    try:
        for job in jobs:
            source_values = np.load(job.temperature_path, allow_pickle=False)
            map_x, map_y, valid = build_remap(job.input_camera, job.output_camera)
            remapped = cv2.remap(
                source_values,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            remapped = np.asarray(remapped, dtype=np.float32)
            if remapped.shape != (job.output_camera.height, job.output_camera.width):
                raise RuntimeError(f"Unexpected remap shape for {job.output_image.name}")
            if not np.all(np.isfinite(remapped)):
                raise RuntimeError(f"Remap produced non-finite values for {job.output_image.name}")
            temperature_target = temporary_root / "temperature_c" / job.output_relative
            support_target = temporary_root / "valid_support" / job.output_relative
            temperature_target.parent.mkdir(parents=True, exist_ok=True)
            support_target.parent.mkdir(parents=True, exist_ok=True)
            np.save(temperature_target, remapped, allow_pickle=False)
            np.save(support_target, valid.astype(np.bool_, copy=False), allow_pickle=False)
            final_temperature = destination_root / "temperature_c" / job.output_relative
            final_support = destination_root / "valid_support" / job.output_relative
            record = {
                "image_name": _normalise_name(job.output_image.name),
                "input_image_id": job.input_image.image_id,
                "output_image_id": job.output_image.image_id,
                "image_match_mode": job.image_match_mode,
                "temperature_match_mode": job.temperature_match_mode,
                "input_temperature": {
                    "path": str(job.temperature_path),
                    "relative_path": job.temperature_path.relative_to(source_root).as_posix(),
                    "sha256": _sha256(job.temperature_path),
                    "dtype": str(source_values.dtype),
                    "shape": list(source_values.shape),
                },
                "input_camera": _camera_payload(job.input_camera),
                "output_camera": _camera_payload(job.output_camera),
                "output_temperature": {
                    "path": str(final_temperature),
                    "relative_path": (Path("temperature_c") / job.output_relative).as_posix(),
                    "sha256": _sha256(temperature_target),
                    "dtype": str(remapped.dtype),
                    "shape": list(remapped.shape),
                    "observed_min_c": float(np.min(remapped)),
                    "observed_max_c": float(np.max(remapped)),
                },
                "valid_support": {
                    "path": str(final_support),
                    "relative_path": (Path("valid_support") / job.output_relative).as_posix(),
                    "sha256": _sha256(support_target),
                    "dtype": str(valid.dtype),
                    "shape": list(valid.shape),
                    "valid_pixels": int(np.count_nonzero(valid)),
                    "total_pixels": int(valid.size),
                    "valid_ratio": float(np.count_nonzero(valid) / valid.size),
                },
            }
            records.append(record)
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "temperature_source_root": str(source_root),
            "output_root": str(destination_root),
            "input_model": _model_payload(input_model),
            "undistorted_output_model": _model_payload(output_model),
            "remap": {
                "direction": "undistorted output pixel to distorted input pixel",
                "interpolation": "OpenCV INTER_LINEAR",
                "border_fill": "OpenCV BORDER_REPLICATE; exclude with valid_support mask",
                "opencv_version": cv2.__version__,
                "supported_input_models": sorted(SUPPORTED_INPUT_MODELS),
                "supported_output_models": sorted(SUPPORTED_OUTPUT_MODELS),
                "pose_tolerance_max_abs": pose_tolerance,
            },
            "summary": {
                "file_count": len(records),
                "valid_pixels": sum(record["valid_support"]["valid_pixels"] for record in records),
                "total_pixels": sum(record["valid_support"]["total_pixels"] for record in records),
            },
            "files": records,
        }
        payload["summary"]["valid_ratio"] = float(
            payload["summary"]["valid_pixels"] / payload["summary"]["total_pixels"]
        )
        manifest = temporary_root / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_root, destination_root)
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    payload["manifest_path"] = str(destination_root / "manifest.json")
    payload["manifest_sha256"] = _sha256(destination_root / "manifest.json")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--input-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--input-model-format", choices=("auto", "bin", "txt"), default="auto")
    parser.add_argument("--output-model-format", choices=("auto", "bin", "txt"), default="auto")
    parser.add_argument("--pose-tolerance", type=float, default=1e-6)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = remap_temperature_tree(
        args.temperature_root,
        args.input_model,
        args.output_model,
        args.output_root,
        input_model_format=args.input_model_format,
        output_model_format=args.output_model_format,
        pose_tolerance=args.pose_tolerance,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest": result["manifest_path"],
                "files": result["summary"]["file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
