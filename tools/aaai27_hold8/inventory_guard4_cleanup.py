#!/usr/bin/env python3
"""Inventory, but never delete, guard4 v1 cleanup candidates."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA = "uav-tgs-guard4-cleanup-inventory-v1"

CATEGORY_PATTERNS = {
    "openmvs_intermediate": (
        "*/reference_openmvs_v1/**/depth*.dmap",
        "*/reference_openmvs_v1/**/reference_openmvs_dense.ply",
        "*/reference_openmvs_v1/**/reference_openmvs_mesh.ply",
        "*/reference_openmvs_v1/**/_openmvs_input/**",
    ),
    "optimizer_checkpoint": ("*/phase1_internal_v1/**/chkpnt*.pth",),
    "raw_evaluation_array": (
        "*/phase1_internal_v1/evaluation/*/metrics/geometry/contribution_bundle/views/*.npz",
        "*/phase1_internal_v1/guard_evidence/*/geometry/guard_partition/views/*.npz",
        "*/phase1_internal_v1/evaluation/*/models/**/opacity_proxy/*.npy",
    ),
}

PRESERVE_PATTERNS = {
    "model_recovery": (
        "**/point_cloud.ply",
        "**/cfg_args",
        "**/cameras.json",
        "**/exposure.json",
    ),
    "protocol_and_scalar_evidence": (
        "**/*receipt*.json",
        "**/*metrics*.json",
        "**/*summary*.json",
        "**/*manifest*.json",
        "**/*.yaml",
        "**/*.yml",
    ),
    "reference_recovery": (
        "**/*mesh_refined*.ply",
        "**/compact_reference_bundle/**",
    ),
    "qualitative_and_logs": (
        "**/qualitative/**/*.png",
        "**/fixed_views/**/*.png",
        "**/*.log",
    ),
}

PROTECTED_BASENAMES = {
    "reference_openmvs_mesh_refined.ply",
    "point_cloud.ply",
    "cameras.json",
    "exposure.json",
    "cfg_args",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(relative: str) -> str | None:
    normalized = relative.replace("\\", "/")
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns):
            return category
    return None


def classify_preserved(relative: str) -> str | None:
    normalized = relative.replace("\\", "/")
    for category, patterns in PRESERVE_PATTERNS.items():
        if any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns):
            return category
    return None


def iter_files(root: Path) -> Iterable[Path]:
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = [name for name in names if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                yield path


def build_inventory(root: Path, *, hash_files: bool) -> dict[str, Any]:
    root = root.resolve()
    if root != Path("/root/autodl-tmp/UAV-TGS/experiments"):
        raise ValueError("cleanup inventory is confined to the authoritative 900 experiments root")
    if not root.is_dir():
        raise FileNotFoundError(root)
    candidates: list[dict[str, Any]] = []
    preserved_assets: list[dict[str, Any]] = []
    category_bytes: dict[str, int] = {key: 0 for key in CATEGORY_PATTERNS}
    preserved_category_bytes: dict[str, int] = {key: 0 for key in PRESERVE_PATTERNS}
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        category = classify(relative)
        stat = path.stat()
        preserve_category = classify_preserved(relative)
        if category is not None:
            if (
                preserve_category is not None
                or path.name in PROTECTED_BASENAMES
                or "mesh_refined" in path.name
            ):
                raise RuntimeError(f"protected file matched cleanup patterns: {relative}")
            record = {
                "relative_path": relative,
                "category": category,
                "size_bytes": stat.st_size,
                "inode": stat.st_ino,
                "hardlink_count": stat.st_nlink,
                "sha256": sha256(path) if hash_files else None,
                "requires_ply_render_equivalence": category == "optimizer_checkpoint",
            }
            candidates.append(record)
            category_bytes[category] += stat.st_size
        elif preserve_category is not None:
            preserved_assets.append(
                {
                    "relative_path": relative,
                    "category": preserve_category,
                    "size_bytes": stat.st_size,
                    "sha256": sha256(path) if hash_files else None,
                }
            )
            preserved_category_bytes[preserve_category] += stat.st_size
    candidates.sort(key=lambda item: item["relative_path"])
    preserved_assets.sort(key=lambda item: item["relative_path"])
    preserve_rules = [
        "final point_cloud.ply plus cfg_args/cameras.json/exposure.json",
        "all scalar metrics, receipts, logs, fixed qualitative images",
        "reference_openmvs_mesh_refined.ply and compact reference bundles/manifests",
        "CFR, raw data, full-frame Celsius NPY, shared SfM/cameras, LUT and support assets",
    ]
    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "inventory_mode": "sha256" if hash_files else "size_only_dry_run",
        "candidate_count": len(candidates),
        "category_bytes": category_bytes,
        "candidate_bytes_gross": sum(category_bytes.values()),
        "candidates": candidates,
        "preserved_asset_count": len(preserved_assets),
        "preserved_category_bytes": preserved_category_bytes,
        "preserved_bytes": sum(preserved_category_bytes.values()),
        "preserved_assets": preserved_assets,
        "preserve_rules": preserve_rules,
        "candidate_preserve_disjoint": not (
            {item["relative_path"] for item in candidates}
            & {item["relative_path"] for item in preserved_assets}
        ),
        "deletion_authorized": False,
        "cleanup_ready": False,
        "required_before_cleanup": [
            "checkpoint and final-PLY fixed-view render equivalence receipt for every checkpoint",
            "compact reference bundle and refined-mesh hash validation",
            "authoritative 900 archive verification",
            "separate explicit cleanup command after GPT review",
        ],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hash-files", action="store_true")
    args = parser.parse_args(argv)
    payload = build_inventory(args.root, hash_files=args.hash_files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": payload["candidate_count"], "candidate_bytes_gross": payload["candidate_bytes_gross"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
