#!/usr/bin/env python3
"""Bind renderer/reference NPZ bundles to the frozen Hold-8 test contract.

The legacy geometry tools emit renderer-native manifests.  This adapter keeps
their arrays unchanged for OpenMVS references and writes a compact, expected-
depth-only model bundle for the final Hold-8 evaluator.  It never reads train
pixels or any metric value when establishing test membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from tools.hold8_expected_depth_evaluator import (
    EXPECTED_DEPTH_EPSILON,
    _MODEL_DEPTH_CONTRACT,
    _validate_authoritative_split,
)


SCHEMA = "uav-tgs-aaai27-hold8-expected-depth-binding-v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stem(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).stem.casefold()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_views(source: Mapping[str, Any], source_manifest: Path) -> dict[str, tuple[Mapping[str, Any], Path]]:
    rows = source.get("views")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no views")
    indexed: dict[str, tuple[Mapping[str, Any], Path]] = {}
    root = source_manifest.parent.resolve()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("source view must be an object")
        key = _stem(row.get("image_name", ""))
        if not key or key in indexed:
            raise ValueError(f"empty/duplicate source image stem: {key!r}")
        path = (root / str(row.get("npz_file", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source NPZ escapes bundle root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_size = int(row.get("npz_size_bytes", -1))
        expected_sha = str(row.get("npz_sha256", "")).lower()
        if expected_size != path.stat().st_size or expected_sha != _sha(path):
            raise RuntimeError(f"source NPZ identity mismatch: {path}")
        indexed[key] = (row, path)
    return indexed


def _test_records(scene_split: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = scene_split.get("records")
    if not isinstance(records, list):
        raise ValueError("scene split has no records")
    result = [row for row in records if isinstance(row, Mapping) and row.get("split") == "test"]
    if not result:
        raise ValueError("scene split has no test records")
    return result


def _record_source_key(row: Mapping[str, Any], *, kind: str) -> str:
    fields: Sequence[str] = (
        ("camera_name", "pair_id") if kind == "reference" else ("thermal_camera_name", "pair_id")
    )
    values = [_stem(row.get(field, "")) for field in fields]
    values = [value for value in values if value]
    if not values:
        raise ValueError(f"test record lacks source identity: {row}")
    return values[0]


def _hardlink_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _write_reference(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as arrays:
        if "depth" not in arrays or "valid_mask" not in arrays:
            raise ValueError(f"reference NPZ lacks depth/valid_mask: {source}")
        depth = np.asarray(arrays["depth"])
        valid = np.asarray(arrays["valid_mask"])
        if depth.shape != valid.shape or depth.ndim != 2:
            raise ValueError(f"reference arrays have invalid shapes: {source}")
    _hardlink_or_copy(source, target)


def _write_model(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=False) as arrays:
        if "depth_expected_alpha_normalized" not in arrays or "accumulated_opacity" not in arrays:
            raise ValueError(f"model NPZ lacks expected depth/opacity: {source}")
        expected = np.asarray(arrays["depth_expected_alpha_normalized"], dtype=np.float32)
        weight = np.asarray(arrays["accumulated_opacity"], dtype=np.float32)
    if expected.shape != weight.shape or expected.ndim != 2:
        raise ValueError(f"model arrays have invalid shapes: {source}")
    positive = (
        np.isfinite(expected)
        & (expected > 0.0)
        & np.isfinite(weight)
        & (weight > EXPECTED_DEPTH_EPSILON)
    )
    np.savez_compressed(
        target,
        expected_depth_camera_z=expected,
        weight_sum=weight,
        has_finite_positive_depth_sample=positive,
    )


def bind_bundle(
    *,
    kind: str,
    source_manifest: Path,
    collection_manifest: Path,
    scene_split_manifest: Path,
    output_root: Path,
    method_name: str = "",
) -> Path:
    if kind not in {"reference", "model"}:
        raise ValueError(f"unsupported bundle kind: {kind}")
    if output_root.exists():
        raise FileExistsError(output_root)
    collection_manifest = collection_manifest.resolve()
    scene_split_manifest = scene_split_manifest.resolve()
    source_manifest = source_manifest.resolve()
    source = _load(source_manifest)
    split = _load(scene_split_manifest)
    scene = str(split.get("scene", ""))
    binding, test_pair_ids = _validate_authoritative_split(
        collection_manifest_path=collection_manifest,
        scene_split_manifest_path=scene_split_manifest,
        scene_name=scene,
    )
    records = _test_records(split)
    if [str(row.get("pair_id", "")) for row in records] != test_pair_ids:
        raise ValueError("scene test order differs from authoritative modulo-8 order")
    sources = _source_views(source, source_manifest)
    expected_keys = [_record_source_key(row, kind=kind) for row in records]
    if set(sources) != set(expected_keys) or len(sources) != len(expected_keys):
        raise ValueError(
            "source bundle does not exactly cover Hold-8 test views: "
            f"missing={sorted(set(expected_keys)-set(sources))[:8]} "
            f"extra={sorted(set(sources)-set(expected_keys))[:8]}"
        )
    output_root.mkdir(parents=True)
    views_root = output_root / "views"
    views_root.mkdir()
    output_views: list[dict[str, Any]] = []
    for index, (row, pair_id, key) in enumerate(zip(records, test_pair_ids, expected_keys)):
        _source_row, source_path = sources[key]
        target = views_root / f"{index:05d}.npz"
        if kind == "reference":
            _write_reference(source_path, target)
            image_name = str(row.get("camera_name", ""))
        else:
            _write_model(source_path, target)
            image_name = str(row.get("thermal_camera_name", ""))
        output_views.append(
            {
                "pair_id": pair_id,
                "image_name": image_name,
                "npz_file": target.relative_to(output_root).as_posix(),
                "npz_size_bytes": target.stat().st_size,
                "npz_sha256": _sha(target),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "completed",
        "kind": kind,
        "scene_name": scene,
        "split": "test",
        **binding,
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": _sha(source_manifest),
        },
        "views": output_views,
    }
    if kind == "reference":
        payload["depth_semantics"] = "metric_camera_z"
    else:
        if not method_name.strip():
            raise ValueError("model binding requires method_name")
        payload["method_name"] = method_name.strip()
        payload["depth_contract"] = _MODEL_DEPTH_CONTRACT
    manifest = output_root / "manifest.json"
    _atomic_json(manifest, payload)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("reference", "model"), required=True)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--scene-split-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--method-name", default="")
    return parser


def main() -> None:
    args = _parser().parse_args()
    manifest = bind_bundle(
        kind=args.kind,
        source_manifest=args.source_manifest,
        collection_manifest=args.collection_manifest,
        scene_split_manifest=args.scene_split_manifest,
        output_root=args.output_root,
        method_name=args.method_name,
    )
    print(f"HOLD8_EXPECTED_DEPTH_BOUND {manifest}")


if __name__ == "__main__":
    main()
