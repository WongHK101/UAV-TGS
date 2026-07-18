#!/usr/bin/env python3
"""Small non-training utilities for the OCT-GS sidecar protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oct_gs.field import OCTConfig, OCTGaussianField
from oct_gs.formal import FORMAL_EXPERIMENT_RECIPE, sha256_json
from oct_gs.protocol import BuildingGradientCalibrator, inspect_oct_checkpoint
from oct_gs.radiance import BandRadianceProxy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibrate(args: argparse.Namespace) -> int:
    source = Path(args.gradient_records).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("scene_name") != "Building" or payload.get("split") != "train":
        raise ValueError("gradient records must explicitly identify Building/train")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("gradient records must contain a non-empty records list")
    thermometric_domain = str(payload.get("thermometric_domain", "celsius"))
    receipt_path = Path(args.source_receipt).resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    calibrator = BuildingGradientCalibrator(
        receipt, thermometric_domain=thermometric_domain
    )
    for record in records:
        calibrator.add_gradient_norms(
            str(record["view_id"]),
            str(record["variant"]),
            {
                "thermometric": record["thermometric"],
                "color_l1": record["color_l1"],
                "color_dssim": record["color_dssim"],
            },
        )
    result = calibrator.freeze(
        args.output,
        metadata={
            "gradient_records_path": str(source),
            "gradient_records_sha256": _sha256(source),
            "source_receipt_path": str(receipt_path),
            "source_receipt_sha256": _sha256(receipt_path),
            "experiment_recipe": dict(FORMAL_EXPERIMENT_RECIPE),
            "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
        },
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "weights": result["weights"]}, indent=2))
    return 0


def _inspect(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = inspect_oct_checkpoint(args.checkpoint)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _self_check(_: argparse.Namespace) -> int:
    proxy = BandRadianceProxy(5.0, 45.0)
    temperatures = torch.linspace(5.0, 45.0, 257, requires_grad=True)
    recovered = proxy.inverse(proxy(temperatures))
    max_error = float(torch.max(torch.abs(recovered - temperatures)).item())
    recovered.mean().backward()
    field = OCTGaussianField(
        OCTConfig(
            num_gaussians=4,
            tmin_c=5.0,
            tmax_c=45.0,
            variant="oct_residual",
        )
    )
    result = {
        "radiance_roundtrip_max_abs_c": max_error,
        "radiance_gradient_finite": bool(torch.isfinite(temperatures.grad).all()),
        "field": field.protocol_metadata(),
    }
    print(json.dumps(result, indent=2))
    return 0 if max_error < 0.01 and result["radiance_gradient_finite"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate = subparsers.add_parser(
        "calibrate", help="Freeze Building/train-only gradient magnitude weights"
    )
    calibrate.add_argument("--gradient-records", required=True)
    calibrate.add_argument("--source-receipt", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(function=_calibrate)
    inspect = subparsers.add_parser("inspect-checkpoint")
    inspect.add_argument("--checkpoint", required=True)
    inspect.set_defaults(function=_inspect)
    self_check = subparsers.add_parser("self-check")
    self_check.set_defaults(function=_self_check)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
