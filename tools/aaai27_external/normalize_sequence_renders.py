#!/usr/bin/env python3
"""Bind sequential official baseline renders to the frozen Building test IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from tools.aaai27_external.building_adapter import read_split, sha256_file


SCHEMA = "uav-tgs-aaai27-external-render-binding-v1"


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        raise FileExistsError(link)
    link.symlink_to(os.path.relpath(target.resolve(), start=link.parent))


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def bind(
    *,
    raw_render_root: Path,
    formal_gt_root: Path,
    test_list: Path,
    output_model_root: Path,
    method: str,
    modality: str,
    raw_gt_root: Path | None = None,
    replace: bool = False,
) -> dict[str, object]:
    names = read_split(test_list.resolve())
    raw_renders = sorted(raw_render_root.resolve().glob("*.png"))
    if len(raw_renders) != len(names):
        raise ValueError(
            f"render/test cardinality mismatch: {len(raw_renders)} != {len(names)}"
        )
    raw_gts = None
    if raw_gt_root is not None:
        raw_gts = sorted(raw_gt_root.resolve().glob("*.png"))
        if len(raw_gts) != len(names):
            raise ValueError("raw GT/test cardinality mismatch")
    if output_model_root.exists():
        if not replace:
            raise FileExistsError(output_model_root)
        shutil.rmtree(output_model_root)
    output = output_model_root / "test" / "ours_formal"
    render_output = output / "renders"
    gt_output = output / "gt"
    rows = []
    max_abs_gt_drift = 0
    for index, (name, render_path) in enumerate(zip(names, raw_renders)):
        stem = Path(name).stem
        target_name = f"{stem}.png"
        formal_candidates = [
            formal_gt_root / name,
            formal_gt_root / f"{stem}.png",
            formal_gt_root / f"{stem}.jpg",
            formal_gt_root / f"{stem}.JPG",
        ]
        formal_gt = next((path.resolve() for path in formal_candidates if path.is_file()), None)
        if formal_gt is None:
            raise FileNotFoundError(f"formal GT missing for {name}")
        if raw_gts is not None:
            raw_array = _rgb(raw_gts[index])
            formal_array = _rgb(formal_gt)
            if raw_array.shape != formal_array.shape:
                raise ValueError(f"raw/formal GT shape mismatch for {name}")
            drift = int(np.max(np.abs(raw_array.astype(np.int16) - formal_array.astype(np.int16))))
            max_abs_gt_drift = max(max_abs_gt_drift, drift)
            if drift != 0:
                raise ValueError(f"raw/formal GT pixel mismatch for {name}: max_abs={drift}")
        _link(render_path, render_output / target_name)
        _link(formal_gt, gt_output / target_name)
        rows.append(
            {
                "pair_id": stem,
                "raw_index": index,
                "raw_render_name": render_path.name,
                "raw_render_sha256": sha256_file(render_path),
                "formal_gt_name": formal_gt.name,
                "formal_gt_sha256": sha256_file(formal_gt),
            }
        )
    manifest: dict[str, object] = {
        "schema": SCHEMA,
        "scene": "Building",
        "protocol": "aaai27_hold8_v2",
        "method": method,
        "modality": modality,
        "mapping_rule": "official_test_iteration_order_to_natural_hold8_test_order",
        "test_list_sha256": sha256_file(test_list),
        "test_count": len(names),
        "raw_gt_verified_pixel_exact": raw_gts is not None,
        "raw_gt_max_abs_drift_u8": max_abs_gt_drift if raw_gts is not None else None,
        "rows": rows,
    }
    material = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["payload_sha256"] = hashlib.sha256(material).hexdigest()
    (output_model_root / "render_binding_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-render-root", type=Path, required=True)
    parser.add_argument("--raw-gt-root", type=Path)
    parser.add_argument("--formal-gt-root", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output-model-root", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--modality", choices=("rgb", "thermal"), required=True)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    value = bind(**vars(args))
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

