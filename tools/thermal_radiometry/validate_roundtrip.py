#!/usr/bin/env python3
"""Fail-closed temperature -> canonical PNG -> LUT inverse round-trip QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

try:
    from evaluate_temperature import evaluate_temperature_tree, write_report
    from palette_lut import PALETTE_SIZE, resolve_temperature_range
except ImportError:  # pragma: no cover - module execution path
    from .evaluate_temperature import evaluate_temperature_tree, write_report
    from .palette_lut import PALETTE_SIZE, resolve_temperature_range


SCHEMA = "uav-tgs-canonical-roundtrip-validation-v1"


class RoundTripValidationError(RuntimeError):
    pass


def validate_roundtrip(
    temperature_root: Path,
    canonical_root: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    range_provenance=None,
    chunk_pixels: int = 32768,
    histogram_bins: int = 65536,
    tolerance_c: float = 1e-5,
    max_clipping_ratio: float = 0.01,
):
    if tolerance_c < 0.0:
        raise ValueError("tolerance_c must be non-negative")
    if not 0.0 <= max_clipping_ratio <= 1.0:
        raise ValueError("max_clipping_ratio must be in [0, 1]")
    evaluation = evaluate_temperature_tree(
        temperature_root,
        canonical_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        range_provenance=range_provenance,
        chunk_pixels=chunk_pixels,
        histogram_bins=histogram_bins,
    )
    summary = evaluation["summary"]
    bin_width_c = (tmax_c - tmin_c) / float(PALETTE_SIZE - 1)
    half_bin_limit_c = 0.5 * bin_width_c + tolerance_c
    in_range_max = summary["temperature_error_in_range_pixels"]["max_abs_error_c"]
    off_lut_max = summary["off_lut_distance"]["max_rgb_distance"]
    clipping_ratio = summary["clipping"]["clipping_ratio"]
    checks = {
        "in_range_max_error_within_half_bin": bool(
            in_range_max is None or in_range_max <= half_bin_limit_c
        ),
        "canonical_pixels_are_exact_lut_entries": bool(off_lut_max == 0.0),
        "clipping_ratio_within_limit": bool(clipping_ratio <= max_clipping_ratio),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "temperature_range": evaluation["temperature_range"],
        "palette": evaluation["palette"],
        "limits": {
            "temperature_bin_width_c": bin_width_c,
            "half_bin_plus_tolerance_c": half_bin_limit_c,
            "tolerance_c": tolerance_c,
            "max_clipping_ratio": max_clipping_ratio,
        },
        "checks": checks,
        "evaluation": evaluation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tmin-c", type=float)
    parser.add_argument("--tmax-c", type=float)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument("--chunk-pixels", type=int, default=32768)
    parser.add_argument("--histogram-bins", type=int, default=65536)
    parser.add_argument("--tolerance-c", type=float, default=1e-5)
    parser.add_argument("--max-clipping-ratio", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tmin_c, tmax_c, provenance = resolve_temperature_range(
        tmin_c=args.tmin_c,
        tmax_c=args.tmax_c,
        range_manifest=args.range_manifest,
    )
    payload = validate_roundtrip(
        args.temperature_root,
        args.canonical_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        range_provenance=provenance,
        chunk_pixels=args.chunk_pixels,
        histogram_bins=args.histogram_bins,
        tolerance_c=args.tolerance_c,
        max_clipping_ratio=args.max_clipping_ratio,
    )
    report = write_report(args.report, payload, overwrite=args.overwrite)
    print(json.dumps({"status": payload["status"], "report": str(report)}, sort_keys=True))
    if payload["status"] != "passed":
        raise RoundTripValidationError(f"Canonical round-trip validation failed: {payload['checks']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RoundTripValidationError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
