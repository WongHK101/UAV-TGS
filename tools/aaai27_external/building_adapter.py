#!/usr/bin/env python3
"""Materialize read-only Building views for the five external baselines.

The adapter never rewrites the frozen CFR RGB images, canonical thermal PNGs,
COLMAP model, or split lists.  It creates only relative symlinks and a compact
receipt under an experiment-owned directory.  Filename aliases ending in JPG
may point at PNG payloads; Pillow detects the payload format from its header,
so this preserves the canonical lossless bytes while satisfying legacy
loaders that hard-code a JPG suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Iterable, Sequence


SCHEMA = "uav-tgs-aaai27-external-building-adapter-v1"
METHODS = (
    "thermal3dgs",
    "thermalgaussian_ommg",
    "mmone",
    "thermonerf",
    "physir_splat",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_split(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"split must be nonempty and duplicate-free: {path}")
    return names


def pair_id(name: str) -> int:
    stem = Path(name).stem
    if not stem.isdigit():
        raise ValueError(f"non-numeric pair ID: {name}")
    return int(stem)


def validate_hold8(train: Sequence[str], test: Sequence[str]) -> list[str]:
    all_names = sorted([*train, *test], key=lambda item: (pair_id(item), item))
    if set(train) & set(test):
        raise ValueError("train/test overlap")
    expected_test = [name for index, name in enumerate(all_names) if index % 8 == 0]
    expected_train = [name for index, name in enumerate(all_names) if index % 8 != 0]
    if list(test) != expected_test or list(train) != expected_train:
        raise ValueError("split is not the frozen natural-order hold-8 manifest")
    return all_names


def _relative_symlink(target: Path, link: Path) -> None:
    if not target.is_file():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(os.path.relpath(target, start=link.parent))


def _link_tree(target: Path, link: Path) -> None:
    if not target.is_dir():
        raise FileNotFoundError(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(os.path.relpath(target, start=link.parent), target_is_directory=True)


def _write_compact_colmap_images(source: Path, output: Path) -> int:
    """Drop unused 2D observations while preserving every formal pose record."""

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    expect_image = True
    with source.open("r", encoding="utf-8") as reader, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for raw in reader:
            stripped = raw.strip()
            if stripped.startswith("#"):
                writer.write(stripped + "\n")
                continue
            if not stripped:
                if not expect_image:
                    writer.write("\n")
                    expect_image = True
                continue
            if expect_image:
                fields = stripped.split()
                if len(fields) < 10:
                    raise ValueError(f"malformed COLMAP image record: {stripped[:120]}")
                writer.write(stripped + "\n")
                count += 1
                expect_image = False
            else:
                # The official GS text loader accepts an empty POINTS2D line.
                writer.write("\n")
                expect_image = True
    if not expect_image:
        raise ValueError("COLMAP images.txt ended before its POINTS2D record")
    if count == 0:
        raise ValueError(f"no COLMAP image records in {source}")
    return count


def _materialize_compact_sparse(source: Path, output: Path) -> dict[str, object]:
    target = output / "sparse" / "0"
    target.mkdir(parents=True, exist_ok=True)
    _relative_symlink(source / "cameras.txt", target / "cameras.txt")
    _relative_symlink(source / "points3D.ply", target / "points3D.ply")
    count = _write_compact_colmap_images(source / "images.txt", target / "images.txt")
    return {
        "policy": "formal_pose_records_with_points2D_observations_omitted",
        "image_record_count": count,
        "compact_images_txt_sha256": sha256_file(target / "images.txt"),
        "compact_images_txt_size_bytes": (target / "images.txt").stat().st_size,
    }


def _thermal_source(thermal_dir: Path, rgb_name: str) -> Path:
    candidates = (
        thermal_dir / f"{Path(rgb_name).stem}.png",
        thermal_dir / rgb_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"canonical thermal image missing for {rgb_name}")


def _parse_colmap_cameras_text(path: Path) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id = int(fields[0])
        result[camera_id] = {
            "model": fields[1],
            "w": int(fields[2]),
            "h": int(fields[3]),
            "params": [float(value) for value in fields[4:]],
        }
    if not result:
        raise ValueError(f"no COLMAP cameras in {path}")
    return result


def _parse_colmap_images_text(path: Path) -> dict[str, dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, dict[str, object]] = {}
    expect_image = True
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not expect_image:
            expect_image = True
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"malformed COLMAP image record: {line[:120]}")
        name = fields[9]
        result[name] = {
            "qvec": [float(value) for value in fields[1:5]],
            "tvec": [float(value) for value in fields[5:8]],
            "camera_id": int(fields[8]),
        }
        expect_image = False
    if not result:
        raise ValueError(f"no COLMAP images in {path}")
    return result


def _qvec_to_rotmat(qvec: Sequence[float]) -> list[list[float]]:
    w, x, y, z = qvec
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[matrix[row][col] for row in range(3)] for col in range(3)]


def _matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def _colmap_to_nerfstudio_c2w(qvec: Sequence[float], tvec: Sequence[float]) -> list[list[float]]:
    # COLMAP stores world-to-camera.  Nerfstudio consumes OpenGL-style c2w.
    rotation_wc = _qvec_to_rotmat(qvec)
    rotation_cw = _transpose(rotation_wc)
    center = [-value for value in _matvec(rotation_cw, tvec)]
    matrix = [[0.0] * 4 for _ in range(4)]
    for row in range(3):
        for col in range(3):
            matrix[row][col] = rotation_cw[row][col]
        matrix[row][3] = center[row]
    matrix[3][3] = 1.0
    # COLMAP camera axes: +x right, +y down, +z forward. OpenGL flips y/z.
    for row in range(3):
        matrix[row][1] *= -1.0
        matrix[row][2] *= -1.0
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError("non-finite camera transform")
    return matrix


def _camera_intrinsics(camera: dict[str, object]) -> dict[str, object]:
    model = str(camera["model"])
    params = list(camera["params"])
    if model == "PINHOLE" and len(params) >= 4:
        fl_x, fl_y, cx, cy = params[:4]
    elif model == "SIMPLE_PINHOLE" and len(params) >= 3:
        fl_x, cx, cy = params[:3]
        fl_y = fl_x
    else:
        raise ValueError(f"unsupported formal undistorted camera model: {model}")
    return {
        "camera_model": "OPENCV",
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": camera["w"],
        "h": camera["h"],
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def _materialize_split_dirs(
    output: Path,
    rgb_dir: Path,
    thermal_dir: Path,
    train: Sequence[str],
    test: Sequence[str],
) -> None:
    for split, names in (("train", train), ("test", test)):
        for name in names:
            _relative_symlink(rgb_dir / name, output / "rgb" / split / name)
            _relative_symlink(
                _thermal_source(thermal_dir, name), output / "thermal" / split / name
            )


def materialize(
    *,
    project_root: Path,
    method: str,
    output: Path,
    replace: bool = False,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    formal = project_root / "derived" / "aaai27_hold8_v2" / "Building"
    binding = (
        project_root
        / "derived"
        / "thermal_radiometry"
        / "aaai27_hold8_v2"
        / "Building"
        / "binding"
    )
    rgb_dir = formal / "workspace" / "images"
    thermal_dir = formal / "thermal_benchmark" / "images"
    sparse_dir = formal / "workspace" / "sparse" / "0"
    train_file = binding / "train_list.txt"
    test_file = binding / "test_list.txt"
    train = read_split(train_file)
    test = read_split(test_file)
    all_names = validate_hold8(train, test)
    if (len(all_names), len(train), len(test)) != (614, 537, 77):
        raise ValueError("Building collection cardinality mismatch")

    for name in all_names:
        if method != "thermal3dgs" and not (rgb_dir / name).is_file():
            raise FileNotFoundError(rgb_dir / name)
        _thermal_source(thermal_dir, name)

    if output.exists() or output.is_symlink():
        if not replace:
            raise FileExistsError(output)
        resolved = output.resolve()
        parent = output.parent.resolve()
        if resolved == Path("/") or parent not in resolved.parents:
            raise ValueError(f"unsafe output replacement: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    compact_sparse = None
    if method in {"thermalgaussian_ommg", "physir_splat"}:
        compact_sparse = _materialize_compact_sparse(sparse_dir, output)
        _materialize_split_dirs(output, rgb_dir, thermal_dir, train, test)
    elif method == "thermal3dgs":
        compact_sparse = _materialize_compact_sparse(sparse_dir, output)
        for name in all_names:
            _relative_symlink(_thermal_source(thermal_dir, name), output / "images" / name)
    elif method == "mmone":
        compact_sparse = _materialize_compact_sparse(sparse_dir, output)
        for name in all_names:
            _relative_symlink(rgb_dir / name, output / "images" / name)
            thermal_alias = f"{Path(name).stem}.jpg"
            _relative_symlink(
                _thermal_source(thermal_dir, name), output / "thermal" / thermal_alias
            )
    elif method == "thermonerf":
        cameras = _parse_colmap_cameras_text(sparse_dir / "cameras.txt")
        images = _parse_colmap_images_text(sparse_dir / "images.txt")
        frames: list[dict[str, object]] = []
        for name in all_names:
            if name not in images:
                raise KeyError(f"formal COLMAP image missing: {name}")
            record = images[name]
            camera = cameras[int(record["camera_id"])]
            frame = {
                "file_path": f"images/{name}",
                "thermal_file_path": f"thermal/{Path(name).stem}.png",
                "transform_matrix": _colmap_to_nerfstudio_c2w(
                    record["qvec"], record["tvec"]
                ),
            }
            frame.update(_camera_intrinsics(camera))
            frames.append(frame)
            _relative_symlink(rgb_dir / name, output / "images" / name)
            _relative_symlink(
                _thermal_source(thermal_dir, name),
                output / "thermal" / f"{Path(name).stem}.png",
            )
        transforms = {
            "frames": frames,
            "train_filenames": [f"images/{name}" for name in train],
            "val_filenames": [f"images/{name}" for name in test],
            "test_filenames": [f"images/{name}" for name in test],
        }
        (output / "transforms.json").write_text(
            json.dumps(transforms, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if compact_sparse is not None and compact_sparse["image_record_count"] != len(all_names):
        raise ValueError(
            "compact sparse pose count mismatch: "
            f"{compact_sparse['image_record_count']} != {len(all_names)}"
        )

    source_hashes = {
        "train_list_sha256": sha256_file(train_file),
        "test_list_sha256": sha256_file(test_file),
        "cameras_txt_sha256": sha256_file(sparse_dir / "cameras.txt"),
        "images_txt_sha256": sha256_file(sparse_dir / "images.txt"),
    }
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "scene": "Building",
        "protocol": "aaai27_hold8_v2",
        "method": method,
        "payload_policy": "relative_symlink_only",
        "thermal_payload": "canonical_hotiron_lossless_png",
        "jpg_alias_payload_note": "legacy JPG suffix may point to unchanged PNG bytes",
        "total_count": len(all_names),
        "train_count": len(train),
        "test_count": len(test),
        "train_names": list(train),
        "test_names": list(test),
        "source_hashes": source_hashes,
        "sparse_adapter": compact_sparse,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    (output / "adapter_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = materialize(
        project_root=args.project_root.resolve(),
        method=args.method,
        output=args.output,
        replace=args.replace,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
