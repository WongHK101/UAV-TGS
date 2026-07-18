#!/usr/bin/env python3
"""Build a fail-closed receipt for a formal thermal-canonical probe bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.thermal_radiometry.palette_lut import (
    PALETTE_NAME,
    hot_iron_lut,
    lut_sha256,
)


PROTOCOL = "uav-tgs-formal-thermal-render-binding-v1"
CANONICAL_SCHEMA = "uav-tgs-canonical-hot-iron-v1"
FORMAL_SPLITS = frozenset({"train", "guard", "test"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _load(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _scene(payload: Mapping[str, Any], label: str) -> str:
    value = str(payload.get("scene_name", payload.get("scene", ""))).strip()
    if not value:
        raise ValueError(f"{label} is missing scene/scene_name")
    return value


def _stem(value: Any) -> str:
    return Path(str(value or "").strip().replace("\\", "/")).stem.casefold()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_output_isolated(output: Path, protected: Mapping[str, Path]) -> None:
    parent = output.resolve().parent
    for label, root in protected.items():
        root = root.resolve()
        if parent == root or _is_within(parent, root) or _is_within(root, parent):
            raise ValueError(
                "Thermal render binding output must be tree-isolated from "
                f"{label}: output_parent={parent}, protected={root}"
            )


def _formal_index(payload: Mapping[str, Any]) -> dict[str, str]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("formal split records must be a non-empty list")
    result: dict[str, str] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError("formal split records must be objects")
        stem = _stem(row.get("pair_id"))
        split = str(row.get("split", "")).strip().lower()
        if not stem or stem in result or split not in FORMAL_SPLITS:
            raise ValueError("formal split has missing/duplicate pair_id or invalid split")
        result[stem] = split
    return result


def _source_index(payload: Mapping[str, Any]) -> dict[str, str]:
    views = payload.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("source model views must be a non-empty list")
    result: dict[str, str] = {}
    for row in views:
        if not isinstance(row, Mapping):
            raise ValueError("source model views must be objects")
        stem = _stem(row.get("image_name"))
        split = str(row.get("bound_split", row.get("split", ""))).strip().lower()
        if not stem or stem in result or split not in FORMAL_SPLITS:
            raise ValueError("source model has missing/duplicate image_name or invalid split")
        result[stem] = split
    return result


def build_thermal_render_binding(
    *,
    source_model_manifest_path: Path,
    formal_split_manifest_path: Path,
    range_manifest_path: Path,
    canonical_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    source_path = Path(source_model_manifest_path).resolve()
    split_path = Path(formal_split_manifest_path).resolve()
    range_path = Path(range_manifest_path).resolve()
    canonical_path = Path(canonical_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace thermal render binding: {output_path}")
    _require_output_isolated(
        output_path,
        {
            "source probe bundle": source_path.parent,
            "formal split tree": split_path.parent,
            "range tree": range_path.parent,
            "canonical tree": canonical_path.parent,
        },
    )

    source = _load(source_path, "source model manifest")
    formal = _load(split_path, "formal split manifest")
    range_manifest = _load(range_path, "range manifest")
    canonical = _load(canonical_path, "canonical manifest")
    scene = _scene(source, "source model manifest")
    for payload, label in ((formal, "formal split"), (range_manifest, "range manifest")):
        if _scene(payload, label) != scene:
            raise ValueError(f"{label} scene mismatch")
    canonical_scene = str(canonical.get("scene_name", canonical.get("scene", ""))).strip()
    if canonical_scene and canonical_scene != scene:
        raise ValueError("canonical manifest scene mismatch")
    if str(source.get("appearance_modality", "")).strip().lower() != "thermal_canonical":
        raise ValueError("source model must use appearance_modality=thermal_canonical")

    split_identity = _identity(split_path)
    bound_split = source.get("formal_split_manifest_identity")
    if not isinstance(bound_split, Mapping) or str(bound_split.get("sha256", "")).lower() != split_identity["sha256"]:
        raise ValueError("source model/formal split identity mismatch")
    formal_index = _formal_index(formal)
    source_index = _source_index(source)
    if source_index != formal_index:
        raise ValueError("source model view coverage/splits differ from formal split")

    model_artifact = source.get("model_point_cloud")
    if not isinstance(model_artifact, Mapping):
        raise ValueError("source model is missing model_point_cloud identity")
    model_path = Path(str(model_artifact.get("path", ""))).resolve()
    model_identity = _identity(model_path)
    if (
        str(model_artifact.get("sha256", "")).strip().lower() != model_identity["sha256"]
        or int(model_artifact.get("size_bytes", -1)) != model_identity["size_bytes"]
    ):
        raise ValueError("source model point-cloud identity does not match the artifact")

    range_identity = _identity(range_path)
    if str(range_manifest.get("source_split_manifest_sha256", "")).lower() != split_identity["sha256"]:
        raise ValueError("range/formal split SHA-256 binding mismatch")
    if str(canonical.get("schema", "")) != CANONICAL_SCHEMA:
        raise ValueError(f"canonical manifest must use {CANONICAL_SCHEMA}")
    palette = canonical.get("palette")
    if not isinstance(palette, Mapping):
        raise ValueError("canonical manifest is missing palette metadata")
    fixed_lut_sha = lut_sha256(hot_iron_lut())
    if (
        str(palette.get("name", "")) != PALETTE_NAME
        or str(palette.get("sha256_uint8_rgb", "")).lower() != fixed_lut_sha
    ):
        raise ValueError("canonical manifest does not bind the fixed Hot-Iron LUT")
    temperature_range = canonical.get("temperature_range")
    range_source = temperature_range.get("source") if isinstance(temperature_range, Mapping) else None
    if not isinstance(range_source, Mapping) or str(range_source.get("sha256", "")).lower() != range_identity["sha256"]:
        raise ValueError("canonical/range manifest SHA-256 binding mismatch")
    canonical_files = canonical.get("files")
    if not isinstance(canonical_files, list):
        raise ValueError("canonical manifest files must be a list")
    canonical_stems = [_stem(row.get("relative_id")) for row in canonical_files if isinstance(row, Mapping)]
    if len(canonical_stems) != len(formal_index) or set(canonical_stems) != set(formal_index):
        raise ValueError("canonical manifest does not cover every formal view exactly")

    payload: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "complete",
        "scene_name": scene,
        "appearance_modality": "thermal_canonical",
        "source_model_manifest_sha256": _sha256(source_path),
        "formal_split_manifest_sha256": split_identity["sha256"],
        "range_manifest_sha256": range_identity["sha256"],
        "canonical_manifest_sha256": _sha256(canonical_path),
        "lut_sha256_uint8_rgb": fixed_lut_sha,
        "model_point_cloud_sha256": model_identity["sha256"],
        "model_point_cloud_size_bytes": model_identity["size_bytes"],
        "coverage": {
            "view_count": len(formal_index),
            "split_counts": {
                split: sum(value == split for value in formal_index.values())
                for split in sorted(FORMAL_SPLITS)
            },
        },
        "inputs": {
            "source_model_manifest": _identity(source_path),
            "formal_split_manifest": split_identity,
            "range_manifest": range_identity,
            "canonical_manifest": _identity(canonical_path),
            "model_point_cloud": model_identity,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-manifest", required=True, type=Path)
    parser.add_argument("--formal-split-manifest", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_thermal_render_binding(
        source_model_manifest_path=args.source_model_manifest,
        formal_split_manifest_path=args.formal_split_manifest,
        range_manifest_path=args.range_manifest,
        canonical_manifest_path=args.canonical_manifest,
        output_path=args.output,
    )
    print(json.dumps({"status": "complete", "receipt_sha256": payload["receipt_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
