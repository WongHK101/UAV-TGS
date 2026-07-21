"""Shared, fail-closed helpers for external expected-depth exports.

The final Hold-8 protocol evaluates one geometry definition: normalized
alpha/volume-weighted *camera-z* depth.  Method-specific exporters use this
module to bind their arrays to the frozen split without duplicating receipt
and hashing code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.hold8_expected_depth_evaluator import (
    EXPECTED_DEPTH_EPSILON,
    _MODEL_DEPTH_CONTRACT,
    _validate_authoritative_split,
)


SCHEMA = "uav-tgs-aaai27-external-expected-depth-export-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalized_stem(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).stem.casefold()


def formal_test_records(scene_split_manifest: Path) -> list[dict[str, Any]]:
    split = load_json(scene_split_manifest)
    rows = split.get("records")
    if not isinstance(rows, list):
        raise ValueError("scene split manifest has no records")
    result = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("split") == "test"]
    if not result:
        raise ValueError("scene split manifest has no test records")
    return result


def validate_render_binding(
    render_binding_manifest: Path, test_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    payload = load_json(render_binding_manifest)
    if payload.get("protocol") != "aaai27_hold8_v2":
        raise ValueError("render binding is not Hold-8 v2")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(test_records):
        raise ValueError("render binding/test record cardinality mismatch")
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["raw_index"]))
    if [int(row["raw_index"]) for row in ordered] != list(range(len(ordered))):
        raise ValueError("render binding raw_index is not contiguous")
    expected = [str(row["pair_id"]) for row in test_records]
    actual = [str(row.get("pair_id", "")) for row in ordered]
    if actual != expected:
        raise ValueError("render binding order differs from the authoritative test order")
    return ordered


def validate_adapter_sources(
    *,
    adapter_manifest: Path,
    train_list: Path,
    test_list: Path,
    cameras_txt: Path,
    images_txt: Path,
) -> dict[str, Any]:
    payload = load_json(adapter_manifest)
    if payload.get("protocol") != "aaai27_hold8_v2":
        raise ValueError("external adapter is not Hold-8 v2")
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise ValueError("external adapter lacks source hashes")
    expected = {
        "train_list_sha256": sha256_file(train_list),
        "test_list_sha256": sha256_file(test_list),
        "cameras_txt_sha256": sha256_file(cameras_txt),
        "images_txt_sha256": sha256_file(images_txt),
    }
    for key, value in expected.items():
        if str(source_hashes.get(key, "")).lower() != value:
            raise ValueError(f"external adapter/formal source mismatch for {key}")
    return {
        "path": str(Path(adapter_manifest).resolve()),
        "size_bytes": Path(adapter_manifest).stat().st_size,
        "sha256": sha256_file(adapter_manifest),
        "formal_source_hashes": expected,
    }


def camera_z_from_ray_distance(
    ray_distance: np.ndarray, directions_norm: np.ndarray, dataparser_scale: float
) -> np.ndarray:
    """Convert normalized Nerfstudio ray distance to original metric camera-z."""

    distance = np.asarray(ray_distance, dtype=np.float64)
    norm = np.asarray(directions_norm, dtype=np.float64)
    scale = float(dataparser_scale)
    if distance.shape != norm.shape:
        raise ValueError(f"distance/directions_norm shape mismatch: {distance.shape} vs {norm.shape}")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("dataparser_scale must be finite and positive")
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise ValueError("directions_norm must be finite and positive")
    return (distance / norm / scale).astype(np.float32)


def write_view_npz(
    *, output_root: Path, index: int, expected_depth: np.ndarray, weight_sum: np.ndarray
) -> tuple[Path, dict[str, Any]]:
    output_root = Path(output_root)
    views = output_root / "views"
    views.mkdir(parents=True, exist_ok=True)
    expected = np.asarray(expected_depth, dtype=np.float32)
    weight = np.asarray(weight_sum, dtype=np.float32)
    if expected.ndim != 2 or expected.shape != weight.shape:
        raise ValueError("expected depth and weight sum must be matching HxW arrays")
    if np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weight sum must be finite and non-negative")
    positive = (
        np.isfinite(expected)
        & (expected > 0.0)
        & (weight > EXPECTED_DEPTH_EPSILON)
    )
    target = views / f"{index:05d}.npz"
    np.savez_compressed(
        target,
        expected_depth_camera_z=expected,
        weight_sum=weight,
        has_finite_positive_depth_sample=positive,
    )
    return target, {
        "npz_file": target.relative_to(output_root).as_posix(),
        "npz_size_bytes": target.stat().st_size,
        "npz_sha256": sha256_file(target),
    }


def write_model_manifest(
    *,
    output_root: Path,
    method_name: str,
    scene_name: str,
    collection_manifest: Path,
    scene_split_manifest: Path,
    views: Sequence[Mapping[str, Any]],
    exporter_metadata: Mapping[str, Any],
) -> Path:
    output_root = Path(output_root).resolve()
    binding, test_pair_ids = _validate_authoritative_split(
        collection_manifest_path=Path(collection_manifest).resolve(),
        scene_split_manifest_path=Path(scene_split_manifest).resolve(),
        scene_name=scene_name,
    )
    actual = [str(view.get("pair_id", "")) for view in views]
    if actual != test_pair_ids:
        raise ValueError("exported view order differs from authoritative Hold-8 test order")
    payload = {
        "schema_version": SCHEMA,
        "status": "completed",
        "kind": "model",
        "scene_name": scene_name,
        "split": "test",
        **binding,
        "method_name": method_name,
        "depth_contract": _MODEL_DEPTH_CONTRACT,
        "exporter_metadata": dict(exporter_metadata),
        "views": [dict(view) for view in views],
    }
    target = output_root / "manifest.json"
    atomic_json(target, payload)
    return target
