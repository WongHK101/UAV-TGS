#!/usr/bin/env python3
"""Evaluate rendered thermal PNGs against float32 TSDK-referenced Celsius maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from palette_lut import palette_metadata, resolve_temperature_range, rgb_to_temperature
except ImportError:  # pragma: no cover - module execution path
    from .palette_lut import palette_metadata, resolve_temperature_range, rgb_to_temperature


SCHEMA = "uav-tgs-apparent-temperature-evaluation-v1"
METRIC_NAME = "TSDK-referenced apparent-temperature consistency"
DEFAULT_HISTOGRAM_BINS = 65536


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_temperature(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32:
        raise TypeError(f"Ground-truth temperature must be float32 Celsius: {path} ({values.dtype})")
    if values.ndim != 2:
        raise ValueError(f"Ground-truth temperature must be two-dimensional: {path} ({values.shape})")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Ground-truth temperature is empty or nonfinite: {path}")
    return values


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Thermal render must be an RGB image: {path} ({image.mode})")
        return np.asarray(image, dtype=np.uint8)


def discover_pairs(ground_truth_root: Path, render_root: Path) -> Sequence[Tuple[Path, Path, str]]:
    gt_root = Path(ground_truth_root).resolve()
    image_root = Path(render_root).resolve()
    if not gt_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError("ground-truth and render roots must both be directories")
    ground_truth = sorted(gt_root.rglob("*.npy"), key=lambda p: p.as_posix())
    if not ground_truth:
        raise FileNotFoundError(f"No float32 NPY maps found under: {gt_root}")
    pairs = []
    missing = []
    for gt_path in ground_truth:
        relative = gt_path.relative_to(gt_root)
        render_path = image_root / relative.with_suffix(".png")
        if not render_path.is_file():
            missing.append(relative.with_suffix(".png").as_posix())
            continue
        pairs.append((gt_path, render_path, relative.with_suffix("").as_posix()))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} rendered PNG(s), first: {preview}")
    return pairs


class _ErrorAccumulator:
    def __init__(self, histogram_max: float, bins: int):
        if bins < 2:
            raise ValueError("histogram bins must be at least 2")
        self.histogram_max = max(float(histogram_max), np.finfo(np.float64).eps)
        self.bins = int(bins)
        self.histogram = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.sum_abs = 0.0
        self.sum_squared = 0.0
        self.sum_signed = 0.0
        self.max_abs = 0.0

    def update(self, signed_error: np.ndarray) -> None:
        signed = np.asarray(signed_error, dtype=np.float64).reshape(-1)
        if signed.size == 0:
            return
        absolute = np.abs(signed)
        self.count += int(signed.size)
        self.sum_abs += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared += float(np.sum(signed * signed, dtype=np.float64))
        self.sum_signed += float(np.sum(signed, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(absolute)))
        counts, _ = np.histogram(absolute, bins=self.bins, range=(0.0, self.histogram_max))
        self.histogram += counts

    def _percentile_upper_edge(self, percentile: float) -> float:
        if self.count == 0:
            return float("nan")
        if self.max_abs == 0.0:
            return 0.0
        rank = int(math.ceil((percentile / 100.0) * self.count))
        index = int(np.searchsorted(np.cumsum(self.histogram), rank, side="left"))
        index = min(max(index, 0), self.bins - 1)
        return (index + 1) * self.histogram_max / self.bins

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "pixels": 0,
                "mae_c": None,
                "rmse_c": None,
                "signed_bias_c": None,
                "p95_abs_error_c": None,
                "max_abs_error_c": None,
            }
        return {
            "pixels": self.count,
            "mae_c": self.sum_abs / self.count,
            "rmse_c": math.sqrt(self.sum_squared / self.count),
            "signed_bias_c": self.sum_signed / self.count,
            "p95_abs_error_c": self._percentile_upper_edge(95.0),
            "max_abs_error_c": self.max_abs,
            "p95_estimator": "fixed histogram upper edge",
            "p95_histogram_bin_width_c": self.histogram_max / self.bins,
        }


def _off_lut_summary(accumulator: _ErrorAccumulator) -> Dict[str, Any]:
    summary = accumulator.summary()
    return {
        "pixels": summary["pixels"],
        "mean_rgb_distance": summary["mae_c"],
        "rms_rgb_distance": summary["rmse_c"],
        "p95_rgb_distance": summary["p95_abs_error_c"],
        "max_rgb_distance": summary["max_abs_error_c"],
        "p95_estimator": summary.get("p95_estimator"),
        "p95_histogram_bin_width": summary.get("p95_histogram_bin_width_c"),
    }


def evaluate_temperature_tree(
    ground_truth_root: Path,
    render_root: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    range_provenance: Optional[Dict[str, Any]] = None,
    chunk_pixels: int = 32768,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
) -> Dict[str, Any]:
    pairs = discover_pairs(ground_truth_root, render_root)
    global_min = math.inf
    global_max = -math.inf
    total_pixels = 0
    for gt_path, _, _ in pairs:
        values = _load_temperature(gt_path)
        global_min = min(global_min, float(np.min(values)))
        global_max = max(global_max, float(np.max(values)))
        total_pixels += int(values.size)
    maximum_possible_error = max(
        abs(global_min - tmin_c),
        abs(global_min - tmax_c),
        abs(global_max - tmin_c),
        abs(global_max - tmax_c),
    )
    all_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    in_range_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    off_lut = _ErrorAccumulator(math.sqrt(3.0 * 255.0 * 255.0), histogram_bins)
    clipped_low = 0
    clipped_high = 0
    files = []
    for gt_path, render_path, relative_id in pairs:
        ground_truth = _load_temperature(gt_path)
        render = _load_rgb(render_path)
        if render.shape[:2] != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch for {relative_id}: GT={ground_truth.shape}, render={render.shape[:2]}"
            )
        recovered, off_distance, _ = rgb_to_temperature(
            render,
            tmin_c,
            tmax_c,
            chunk_pixels=chunk_pixels,
        )
        signed = recovered.astype(np.float64) - ground_truth.astype(np.float64)
        below = ground_truth < tmin_c
        above = ground_truth > tmax_c
        in_range = ~(below | above)
        all_errors.update(signed)
        in_range_errors.update(signed[in_range])
        off_lut.update(off_distance)
        clipped_low += int(np.count_nonzero(below))
        clipped_high += int(np.count_nonzero(above))
        absolute = np.abs(signed)
        files.append(
            {
                "relative_id": relative_id,
                "ground_truth_path": str(gt_path.resolve()),
                "ground_truth_sha256": _sha256(gt_path),
                "render_path": str(render_path.resolve()),
                "render_sha256": _sha256(render_path),
                "pixels": int(ground_truth.size),
                "mae_c": float(np.mean(absolute)),
                "rmse_c": float(np.sqrt(np.mean(signed * signed))),
                "signed_bias_c": float(np.mean(signed)),
                "p95_abs_error_c": float(np.percentile(absolute, 95.0)),
                "max_abs_error_c": float(np.max(absolute)),
                "clipped_low_pixels": int(np.count_nonzero(below)),
                "clipped_high_pixels": int(np.count_nonzero(above)),
                "off_lut_mean_rgb_distance": float(np.mean(off_distance)),
                "off_lut_p95_rgb_distance": float(np.percentile(off_distance, 95.0)),
                "off_lut_max_rgb_distance": float(np.max(off_distance)),
            }
        )
    clipped = clipped_low + clipped_high
    return {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": METRIC_NAME,
        "claim_boundary": "apparent-temperature consistency, not absolute thermometry or true surface temperature",
        "ground_truth": "float32 Celsius maps decoded by the configured TSDK protocol",
        "ground_truth_root": str(Path(ground_truth_root).resolve()),
        "render_root": str(Path(render_root).resolve()),
        "temperature_range": {
            "tmin_c": float(tmin_c),
            "tmax_c": float(tmax_c),
            "source": range_provenance or {"source": "function-argument"},
        },
        "palette": palette_metadata(),
        "summary": {
            "file_count": len(files),
            "pixels": total_pixels,
            "temperature_error_all_pixels": all_errors.summary(),
            "temperature_error_in_range_pixels": in_range_errors.summary(),
            "clipping": {
                "low_pixels": clipped_low,
                "high_pixels": clipped_high,
                "clipped_pixels": clipped,
                "clipping_ratio": clipped / total_pixels,
            },
            "off_lut_distance": _off_lut_summary(off_lut),
        },
        "files": files,
    }


def write_report(path: Path, payload: Dict[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tmin-c", type=float)
    parser.add_argument("--tmax-c", type=float)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument("--chunk-pixels", type=int, default=32768)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tmin_c, tmax_c, provenance = resolve_temperature_range(
        tmin_c=args.tmin_c,
        tmax_c=args.tmax_c,
        range_manifest=args.range_manifest,
    )
    payload = evaluate_temperature_tree(
        args.ground_truth_root,
        args.render_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        range_provenance=provenance,
        chunk_pixels=args.chunk_pixels,
        histogram_bins=args.histogram_bins,
    )
    report = write_report(args.report, payload, overwrite=args.overwrite)
    print(json.dumps({"status": "complete", "report": str(report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
