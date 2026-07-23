#!/usr/bin/env python3
"""Render float32 Celsius maps as fixed-range lossless canonical Hot-Iron PNGs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from palette_lut import palette_metadata, resolve_temperature_range, temperature_to_rgb
except ImportError:  # pragma: no cover - module execution path
    from .palette_lut import palette_metadata, resolve_temperature_range, temperature_to_rgb


SCHEMA = "uav-tgs-canonical-hot-iron-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_temperature_maps(root: Path) -> Sequence[Path]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Temperature root is not a directory: {resolved}")
    maps = sorted((path for path in resolved.rglob("*.npy") if path.is_file()), key=lambda p: p.as_posix())
    if not maps:
        raise FileNotFoundError(f"No .npy temperature maps found under: {resolved}")
    return maps


def render_temperature_file(
    input_path: Path,
    output_path: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite canonical PNG: {destination}")
    values = np.load(source, allow_pickle=False)
    if values.dtype != np.float32:
        raise ValueError(f"Temperature map must be float32 Celsius: {source} has {values.dtype}")
    rgb, clipping_mask = temperature_to_rgb(values, tmin_c, tmax_c)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.png")
    try:
        Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG", compress_level=6, optimize=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    clipped_low = int(np.count_nonzero(values < tmin_c))
    clipped_high = int(np.count_nonzero(values > tmax_c))
    return {
        "input_path": str(source),
        "input_sha256": _sha256(source),
        "output_path": str(destination),
        "output_sha256": _sha256(destination),
        "canonical_rgb_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
        "height": int(values.shape[0]),
        "width": int(values.shape[1]),
        "temperature_dtype": str(values.dtype),
        "observed_min_c": float(np.min(values)),
        "observed_max_c": float(np.max(values)),
        "pixels": int(values.size),
        "clipped_low_pixels": clipped_low,
        "clipped_high_pixels": clipped_high,
        "clipping_ratio": float(np.count_nonzero(clipping_mask) / values.size),
    }


def _render_job(
    job: Tuple[str, str, float, float, bool, str, str, str]
) -> Dict[str, Any]:
    (
        source,
        destination,
        tmin_c,
        tmax_c,
        overwrite,
        relative_id,
        relative_input,
        relative_output,
    ) = job
    record = render_temperature_file(
        Path(source),
        Path(destination),
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        overwrite=overwrite,
    )
    record["relative_id"] = relative_id
    record["relative_input"] = relative_input
    record["relative_output"] = relative_output
    return record


def render_tree(
    temperature_root: Path,
    output_root: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    range_provenance: Optional[Dict[str, Any]] = None,
    manifest_path: Optional[Path] = None,
    overwrite: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    source_root = Path(temperature_root).resolve()
    destination_root = Path(output_root).resolve()
    if source_root == destination_root or _is_within(destination_root, source_root):
        raise ValueError("Canonical output must be outside the temperature source tree")
    maps = discover_temperature_maps(source_root)
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    jobs = []
    seen_outputs = set()
    for source in maps:
        relative = source.relative_to(source_root).with_suffix(".png")
        destination = destination_root / relative
        canonical_key = os.path.normcase(str(destination.resolve()))
        if canonical_key in seen_outputs:
            raise RuntimeError(f"Output collision for {relative}")
        seen_outputs.add(canonical_key)
        jobs.append(
            (
                str(source),
                str(destination),
                float(tmin_c),
                float(tmax_c),
                bool(overwrite),
                relative.with_suffix("").as_posix(),
                source.relative_to(source_root).as_posix(),
                relative.as_posix(),
            )
        )
    if workers == 1:
        records = [_render_job(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_render_job, jobs))
    total_pixels = sum(item["pixels"] for item in records)
    clipped = sum(item["clipped_low_pixels"] + item["clipped_high_pixels"] for item in records)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_root": str(destination_root),
        "temperature_range": {
            "tmin_c": float(tmin_c),
            "tmax_c": float(tmax_c),
            "source": range_provenance or {"source": "function-argument"},
        },
        "palette": palette_metadata(),
        "image_encoding": {"format": "PNG", "mode": "RGB", "lossless": True, "gamma": 1.0},
        "files": records,
        "summary": {
            "file_count": len(records),
            "pixels": total_pixels,
            "clipped_pixels": clipped,
            "clipping_ratio": float(clipped / total_pixels),
        },
    }
    target = Path(manifest_path).resolve() if manifest_path else destination_root / "canonical_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    payload["manifest_path"] = str(target)
    payload["manifest_sha256"] = _sha256(target)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tmin-c", type=float)
    parser.add_argument("--tmax-c", type=float)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tmin_c, tmax_c, provenance = resolve_temperature_range(
        tmin_c=args.tmin_c,
        tmax_c=args.tmax_c,
        range_manifest=args.range_manifest,
    )
    result = render_tree(
        args.temperature_root,
        args.output_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        range_provenance=provenance,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    print(json.dumps({"status": result["status"], "manifest": result["manifest_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
