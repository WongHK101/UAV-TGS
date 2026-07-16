#!/usr/bin/env python3
"""Clone a COLMAP sparse model while changing only image-name extensions.

This is used to keep canonical thermal observations as real PNG files through
``colmap image_undistorter``.  Camera parameters, poses, observations and point
tracks are copied unchanged; any other mutation fails the post-write audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from utils.read_write_model import read_model, write_model


SCHEMA_NAME = "uav_tgs_colmap_image_extension_clone"
SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(root: Path, extension: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stem in ("cameras", "images", "points3D"):
        path = root / f"{stem}{extension}"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[path.name] = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    return result


def _normalised_extension(value: str) -> str:
    extension = str(value).strip()
    if not extension.startswith(".") or len(extension) < 2 or any(char in extension for char in "/\\"):
        raise ValueError("target extension must be a simple suffix such as .png")
    return extension


def _detect_extension(root: Path) -> str:
    available = [
        extension
        for extension in (".bin", ".txt")
        if all((root / f"{stem}{extension}").is_file() for stem in ("cameras", "images", "points3D"))
    ]
    if len(available) != 1:
        raise ValueError(f"expected exactly one complete COLMAP model format in {root}, found {available}")
    return available[0]


def _arrays_equal(first: Any, second: Any) -> bool:
    return np.array_equal(np.asarray(first), np.asarray(second), equal_nan=True)


def clone_with_extension(
    source_model: Path,
    output_model: Path,
    *,
    target_extension: str = ".png",
    output_format: str = ".bin",
) -> dict[str, Any]:
    source_model = source_model.resolve()
    output_model = output_model.resolve()
    target_extension = _normalised_extension(target_extension)
    if output_format not in (".bin", ".txt"):
        raise ValueError("output_format must be .bin or .txt")
    if output_model.exists():
        raise FileExistsError(f"refusing to replace existing output model: {output_model}")
    source_format = _detect_extension(source_model)
    cameras, images, points = read_model(str(source_model), ext=source_format)
    if not images:
        raise ValueError("source COLMAP model contains no images")

    renamed = {}
    seen_names: set[str] = set()
    rows = []
    for image_id, image in sorted(images.items()):
        path = PurePosixPath(str(image.name).replace("\\", "/"))
        if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
            raise ValueError(f"unsafe COLMAP image name: {image.name!r}")
        new_name = path.with_suffix(target_extension).as_posix()
        if new_name in seen_names:
            raise ValueError(f"renaming would create a duplicate image name: {new_name}")
        seen_names.add(new_name)
        renamed[image_id] = image._replace(name=new_name)
        rows.append({"image_id": int(image_id), "source_name": image.name, "output_name": new_name})

    output_model.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_model.name + ".tmp-", dir=output_model.parent))
    try:
        write_model(cameras, renamed, points, str(temporary), ext=output_format)
        check_cameras, check_images, check_points = read_model(str(temporary), ext=output_format)
        if set(check_cameras) != set(cameras) or set(check_images) != set(images) or set(check_points) != set(points):
            raise RuntimeError("post-write COLMAP entity IDs changed")
        for key in cameras:
            if check_cameras[key].model != cameras[key].model or check_cameras[key].width != cameras[key].width \
                    or check_cameras[key].height != cameras[key].height or not _arrays_equal(check_cameras[key].params, cameras[key].params):
                raise RuntimeError(f"camera changed while renaming images: {key}")
        for key in images:
            before, after = images[key], check_images[key]
            if after.name != renamed[key].name or after.camera_id != before.camera_id \
                    or not _arrays_equal(after.qvec, before.qvec) or not _arrays_equal(after.tvec, before.tvec) \
                    or not _arrays_equal(after.xys, before.xys) or not _arrays_equal(after.point3D_ids, before.point3D_ids):
                raise RuntimeError(f"image geometry/observations changed while renaming: {key}")
        for key in points:
            before, after = points[key], check_points[key]
            if not _arrays_equal(after.xyz, before.xyz) or not _arrays_equal(after.rgb, before.rgb) \
                    or float(after.error) != float(before.error) or not _arrays_equal(after.image_ids, before.image_ids) \
                    or not _arrays_equal(after.point2D_idxs, before.point2D_idxs):
                raise RuntimeError(f"point track changed while renaming images: {key}")
        os.replace(temporary, output_model)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source_model": str(source_model),
        "source_format": source_format,
        "source_files": _model_files(source_model, source_format),
        "output_model": str(output_model),
        "output_format": output_format,
        "output_files": _model_files(output_model, output_format),
        "target_extension": target_extension,
        "counts": {"cameras": len(cameras), "images": len(images), "points3D": len(points)},
        "invariants": {
            "camera_parameters_exact": True,
            "poses_exact": True,
            "observations_exact": True,
            "point_tracks_exact": True,
            "image_names_only_mutation": True,
        },
        "renames": rows,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output-model", required=True, type=Path)
    parser.add_argument("--target-extension", default=".png")
    parser.add_argument("--output-format", choices=(".bin", ".txt"), default=".bin")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = clone_with_extension(
        args.source_model,
        args.output_model,
        target_extension=args.target_extension,
        output_format=args.output_format,
    )
    _atomic_json(args.manifest.resolve(), result)
    print(json.dumps({"status": "passed", "manifest": str(args.manifest.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
