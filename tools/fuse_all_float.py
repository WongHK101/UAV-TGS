#!/usr/bin/env python3
"""Canonical all-float and hotspot-gated model-space interpolation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.thermal_radiometry.palette_lut import (
    hot_iron_lut,
    lut_sha256,
    resolve_temperature_range,
)


SH_C0 = 0.28209479177387814


class FusionError(RuntimeError):
    """Fail-closed model fusion error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_coefficients(value: str) -> list[float]:
    result = [float(item) for item in value.replace(",", " ").split()]
    if not result or any(not math.isfinite(item) or item < 0 or item > 1 for item in result):
        raise FusionError("coefficients must be finite values in [0,1]")
    if len(set(result)) != len(result):
        raise FusionError("coefficients must be unique")
    return result


def coefficient_directory(value: float) -> str:
    if value == 0.0:
        return "0"
    if value == 1.0:
        return "1"
    return format(value, ".10g")


def load_vertex(path: Path) -> tuple[PlyData, np.ndarray]:
    ply = PlyData.read(str(path))
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise FusionError("PLY must contain exactly one vertex element")
    return ply, ply["vertex"].data


def rotation_fields(names: tuple[str, ...]) -> list[str]:
    fields = sorted(
        (name for name in names if name.startswith("rot_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(fields) != 4:
        raise FusionError(f"expected four quaternion fields, found {fields}")
    return fields


def dc_fields(names: tuple[str, ...]) -> list[str]:
    fields = sorted(
        (name for name in names if name.startswith("f_dc_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if len(fields) != 3:
        raise FusionError(f"expected three DC fields, found {fields}")
    return fields


def nlerp_quaternion(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Shortest-path normalized lerp with exact callers handling endpoints."""

    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64).copy()
    if q0.shape != q1.shape or q0.ndim != 2 or q0.shape[1] != 4:
        raise FusionError("quaternions must have shape (N,4)")
    dot = np.sum(q0 * q1, axis=1)
    q1[dot < 0] *= -1.0
    w = np.asarray(weight, dtype=np.float64).reshape(-1, 1)
    if w.shape[0] not in (1, q0.shape[0]):
        raise FusionError("quaternion weight count mismatch")
    blended = (1.0 - w) * q0 + w * q1
    norm = np.linalg.norm(blended, axis=1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0):
        raise FusionError("quaternion interpolation produced invalid norm")
    return blended / norm


def _field_group(vertex: np.ndarray, fields: list[str]) -> np.ndarray:
    return np.stack([np.asarray(vertex[name]) for name in fields], axis=1)


def validate_pair(rgb: np.ndarray, thermal: np.ndarray, require_shared_geometry: bool) -> dict[str, Any]:
    if len(rgb) != len(thermal):
        raise FusionError(f"Gaussian count mismatch: {len(rgb)} vs {len(thermal)}")
    if rgb.dtype.descr != thermal.dtype.descr:
        raise FusionError("PLY schema/name/order/dtype mismatch")
    names = tuple(rgb.dtype.names or ())
    rotation_fields(names)
    dc_fields(names)
    geometry = ["x", "y", "z", "opacity"]
    geometry += sorted(name for name in names if name.startswith("scale_"))
    geometry += rotation_fields(names)
    missing = [name for name in geometry if name not in names]
    if missing:
        raise FusionError(f"missing geometry fields: {missing}")
    shared = {name: bool(np.array_equal(rgb[name], thermal[name])) for name in geometry}
    if require_shared_geometry and not all(shared.values()):
        changed = [name for name, exact in shared.items() if not exact]
        raise FusionError("strict shared-geometry invariant failed: " + ", ".join(changed))
    return {
        "gaussian_count": len(rgb),
        "schema": list(names),
        "schema_sha256": hashlib.sha256(json.dumps(rgb.dtype.descr).encode("utf-8")).hexdigest(),
        "shared_geometry_fields": shared,
        "row_alignment_basis": "strict Stage-2 unchanged geometry and identical Gaussian count/schema/order",
    }


def all_float_vertex(rgb: np.ndarray, thermal: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return rgb.copy()
    if alpha == 1.0:
        return thermal.copy()
    output = rgb.copy()
    names = tuple(rgb.dtype.names or ())
    rotations = rotation_fields(names)
    for name in names:
        if name in rotations:
            continue
        if np.issubdtype(rgb[name].dtype, np.floating):
            output[name] = (
                (1.0 - alpha) * np.asarray(rgb[name], dtype=np.float64)
                + alpha * np.asarray(thermal[name], dtype=np.float64)
            ).astype(rgb[name].dtype)
    q = nlerp_quaternion(
        _field_group(rgb, rotations),
        _field_group(thermal, rotations),
        np.asarray([alpha]),
    )
    for index, name in enumerate(rotations):
        output[name] = q[:, index].astype(rgb[name].dtype)
    return output


def dc_apparent_temperature(
    thermal: np.ndarray, tmin_c: float, tmax_c: float
) -> tuple[np.ndarray, dict[str, Any]]:
    fields = dc_fields(tuple(thermal.dtype.names or ()))
    dc = _field_group(thermal, fields).astype(np.float64)
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    lut = hot_iron_lut().astype(np.float64) / 255.0
    rgb_norm = np.sum(rgb * rgb, axis=1, keepdims=True)
    lut_norm = np.sum(lut * lut, axis=1)[None, :]
    distance2 = np.maximum(rgb_norm + lut_norm - 2.0 * rgb @ lut.T, 0.0)
    index = np.argmin(distance2, axis=1)
    temperature = tmin_c + (index.astype(np.float64) / 255.0) * (tmax_c - tmin_c)
    off_lut = np.sqrt(distance2[np.arange(len(index)), index])
    return temperature, {
        "dc_rgb_clipped_fraction": float(np.mean((0.5 + SH_C0 * dc < 0) | (0.5 + SH_C0 * dc > 1))),
        "off_lut_mean_normalized_rgb": float(np.mean(off_lut)),
        "off_lut_p95_normalized_rgb": float(np.percentile(off_lut, 95)),
    }


def train_temperature_quantile(
    temperature_root: Path, train_list: Path, quantile: float
) -> tuple[float, dict[str, Any]]:
    if not 0.0 < quantile < 1.0:
        raise FusionError("train quantile must be in (0,1)")
    names = [Path(line.strip()).stem for line in train_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise FusionError("train list is empty")
    # Per-frame quantiles keep memory bounded and prevent long frames from
    # silently receiving different weights if dimensions ever differ.
    frame_quantiles = []
    paths = []
    for name in names:
        direct = temperature_root / f"{name}.npy"
        matches = [direct] if direct.is_file() else list(temperature_root.glob(f"*/{name}.npy"))
        if len(matches) != 1:
            raise FusionError(f"expected one train temperature NPY for {name}, found {len(matches)}")
        values = np.load(matches[0], mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise FusionError(f"invalid train temperature map: {matches[0]}")
        frame_quantiles.append(float(np.quantile(values, quantile)))
        paths.append(str(matches[0].resolve()))
    threshold = float(np.median(frame_quantiles))
    return threshold, {
        "source": "median_of_per_train_frame_quantiles",
        "quantile": float(quantile),
        "train_frame_count": len(names),
        "train_list": str(train_list.resolve()),
        "train_list_sha256": sha256(train_list),
        "temperature_root": str(temperature_root.resolve()),
        "per_frame_quantile_percentiles": {
            "p0": float(np.min(frame_quantiles)),
            "p50": float(np.median(frame_quantiles)),
            "p100": float(np.max(frame_quantiles)),
        },
    }


def hotspot_weights(
    apparent_temperature: np.ndarray,
    threshold_c: float,
    softness_c: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(softness_c) or softness_c <= 0:
        raise FusionError("softness_c must be finite and positive")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise FusionError("hotspot strength must be in [0,1]")
    logits = np.clip((apparent_temperature - threshold_c) / softness_c, -60.0, 60.0)
    gate = 1.0 / (1.0 + np.exp(-logits))
    return strength * gate, gate


def weighted_all_float_vertex(
    rgb: np.ndarray, thermal: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    output = rgb.copy()
    names = tuple(rgb.dtype.names or ())
    rotations = rotation_fields(names)
    weight = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weight.shape[0] != len(rgb):
        raise FusionError("hotspot weight count mismatch")
    for name in names:
        if name in rotations:
            continue
        if np.issubdtype(rgb[name].dtype, np.floating):
            output[name] = (
                (1.0 - weight) * np.asarray(rgb[name], dtype=np.float64)
                + weight * np.asarray(thermal[name], dtype=np.float64)
            ).astype(rgb[name].dtype)
    q = nlerp_quaternion(
        _field_group(rgb, rotations), _field_group(thermal, rotations), weight
    )
    for index, name in enumerate(rotations):
        output[name] = q[:, index].astype(rgb[name].dtype)
    return output


def gate_vertex(rgb: np.ndarray, gate: np.ndarray) -> np.ndarray:
    output = rgb.copy()
    dc = dc_fields(tuple(rgb.dtype.names or ()))
    rest = [name for name in (rgb.dtype.names or ()) if name.startswith("f_rest_")]
    coefficient = (np.asarray(gate, dtype=np.float64) - 0.5) / SH_C0
    for name in dc:
        output[name] = coefficient.astype(output[name].dtype)
    for name in rest:
        output[name] = np.zeros_like(output[name])
    return output


def rewrite_cfg(cfg: str, model_path: Path) -> str:
    normalized = str(model_path.resolve()).replace("\\", "/")
    import re
    output, count = re.subn(
        r"(model_path\s*=\s*)(['\"])[^'\"]+(['\"])",
        lambda match: f"{match.group(1)}{match.group(2)}{normalized}{match.group(2)}",
        cfg,
        count=1,
    )
    if count != 1:
        raise FusionError("cfg_args does not contain exactly one model_path")
    return output


def write_model(
    output_dir: Path,
    vertex: np.ndarray,
    template_ply: PlyData,
    cfg: str,
    iteration: int,
) -> Path:
    point_path = output_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    point_path.parent.mkdir(parents=True, exist_ok=False)
    PlyData(
        [PlyElement.describe(vertex, "vertex")], text=getattr(template_ply, "text", False)
    ).write(str(point_path))
    (output_dir / "cfg_args").write_text(rewrite_cfg(cfg, output_dir), encoding="utf-8")
    return point_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    rgb_model = Path(args.rgb_model_dir).resolve()
    thermal_model = Path(args.thermal_model_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FusionError(f"refusing to overwrite output root: {output_root}")
    rgb_ply_path = rgb_model / "point_cloud" / f"iteration_{args.rgb_iteration}" / "point_cloud.ply"
    thermal_ply_path = thermal_model / "point_cloud" / f"iteration_{args.thermal_iteration}" / "point_cloud.ply"
    rgb_ply, rgb = load_vertex(rgb_ply_path)
    _, thermal = load_vertex(thermal_ply_path)
    pair = validate_pair(rgb, thermal, bool(args.require_shared_geometry))
    cfg = (rgb_model / "cfg_args").read_text(encoding="utf-8")
    partial = output_root.with_name(f".{output_root.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True, exist_ok=False)
    outputs = []
    hot_manifest = None
    try:
        if args.mode == "all_float":
            coefficients = parse_coefficients(args.coefficients)
            for alpha in coefficients:
                model_dir = partial / coefficient_directory(alpha)
                output_vertex = all_float_vertex(rgb, thermal, alpha)
                point_path = write_model(model_dir, output_vertex, rgb_ply, cfg, args.output_iteration)
                endpoint_exact = (
                    bool(np.array_equal(output_vertex, rgb)) if alpha == 0 else
                    bool(np.array_equal(output_vertex, thermal)) if alpha == 1 else None
                )
                if endpoint_exact is False:
                    raise FusionError(f"strict endpoint failed at alpha={alpha}")
                outputs.append(
                    {
                        "coefficient": alpha,
                        "semantic_name": "model_space_interpolation_coefficient",
                        "relative_model_dir": str(model_dir.relative_to(partial).as_posix()),
                        "ply_sha256": sha256(point_path),
                        "strict_endpoint_exact": endpoint_exact,
                    }
                )
        elif args.mode == "hot_all_float":
            strengths = parse_coefficients(args.hotspot_strengths)
            tmin, tmax, range_metadata = resolve_temperature_range(
                range_manifest=Path(args.range_manifest)
            )
            apparent, dc_stats = dc_apparent_temperature(thermal, tmin, tmax)
            if args.threshold_c is not None:
                threshold = float(args.threshold_c)
                threshold_source = {"source": "explicit_celsius", "threshold_c": threshold}
            else:
                if args.train_quantile is None or not args.train_temperature_root or not args.train_list:
                    raise FusionError("hot mode requires --threshold-c or the complete train-quantile inputs")
                threshold, threshold_source = train_temperature_quantile(
                    Path(args.train_temperature_root), Path(args.train_list), float(args.train_quantile)
                )
                threshold_source["threshold_c"] = threshold
            _, gate = hotspot_weights(apparent, threshold, args.softness_c, 1.0)
            gate_dir = partial / "gate_model"
            gate_path = write_model(
                gate_dir, gate_vertex(rgb, gate), rgb_ply, cfg, args.output_iteration
            )
            for strength in strengths:
                weights, _ = hotspot_weights(apparent, threshold, args.softness_c, strength)
                model_dir = partial / coefficient_directory(strength)
                point_path = write_model(
                    model_dir,
                    weighted_all_float_vertex(rgb, thermal, weights),
                    rgb_ply,
                    cfg,
                    args.output_iteration,
                )
                outputs.append(
                    {
                        "hotspot_strength": strength,
                        "semantic_name": "hotspot_gate_strength_not_endpoint",
                        "relative_model_dir": str(model_dir.relative_to(partial).as_posix()),
                        "ply_sha256": sha256(point_path),
                        "weight_percentiles": {
                            f"p{p}": float(np.percentile(weights, p))
                            for p in (0, 25, 50, 75, 95, 99, 100)
                        },
                    }
                )
            hot_manifest = {
                "range": range_metadata,
                "threshold": threshold_source,
                "softness_c": float(args.softness_c),
                "apparent_temperature_percentiles_c": {
                    f"p{p}": float(np.percentile(apparent, p))
                    for p in (0, 25, 50, 75, 95, 99, 100)
                },
                "gate_percentiles": {
                    f"p{p}": float(np.percentile(gate, p))
                    for p in (0, 25, 50, 75, 95, 99, 100)
                },
                "dc_approximation": "RGB=clip(0.5+C0*f_dc,0,1), nearest fixed Hot-Iron LUT",
                "dc_diagnostics": dc_stats,
                "gate_model": {
                    "relative_model_dir": str(gate_dir.relative_to(partial).as_posix()),
                    "ply_sha256": sha256(gate_path),
                    "purpose": "render held-out gate maps with the standard renderer",
                },
                "semantic_claim_pending": "temperature-gated only if held-out gate validation passes; otherwise high-thermal-response visualization",
            }
        else:
            raise FusionError(f"unsupported mode: {args.mode}")

        manifest = {
            "schema": "uav-tgs-canonical-model-fusion-v1",
            "status": "passed",
            "mode": args.mode,
            "source_code_commit": str(args.code_commit),
            "operation": {
                "all_float_fields": [
                    name for name in (rgb.dtype.names or ()) if np.issubdtype(rgb[name].dtype, np.floating)
                ],
                "quaternion": "q/-q shortest-path alignment then normalized lerp; all_float endpoints direct copy",
                "non_float_fields": "RGB copy for intermediate outputs",
            },
            "inputs": {
                "rgb_ply": str(rgb_ply_path),
                "rgb_ply_sha256": sha256(rgb_ply_path),
                "thermal_ply": str(thermal_ply_path),
                "thermal_ply_sha256": sha256(thermal_ply_path),
            },
            "pair_invariants": pair,
            "outputs": outputs,
            "hotspot": hot_manifest,
            "lut_sha256": lut_sha256() if args.mode == "hot_all_float" else None,
        }
        (partial / "fusion_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (partial / "FUSION_STATUS").write_text("passed\n", encoding="ascii")
        os.replace(partial, output_root)
        return manifest
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--mode", choices=["all_float", "hot_all_float"], required=True)
    result.add_argument("--rgb-model-dir", required=True)
    result.add_argument("--rgb-iteration", type=int, required=True)
    result.add_argument("--thermal-model-dir", required=True)
    result.add_argument("--thermal-iteration", type=int, required=True)
    result.add_argument("--output-root", required=True)
    result.add_argument("--output-iteration", type=int, required=True)
    result.add_argument("--coefficients", default="0,0.25,0.5,0.75,1")
    result.add_argument("--hotspot-strengths", default="0.25,0.5,0.75,1")
    result.add_argument("--range-manifest")
    result.add_argument("--threshold-c", type=float)
    result.add_argument("--train-quantile", type=float)
    result.add_argument("--train-temperature-root")
    result.add_argument("--train-list")
    result.add_argument("--softness-c", type=float, default=1.0)
    result.add_argument("--require-shared-geometry", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--code-commit", required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.mode == "hot_all_float" and not args.range_manifest:
        raise SystemExit("hot_all_float requires --range-manifest")
    try:
        manifest = run(args)
    except (FusionError, FileNotFoundError, ValueError) as error:
        print(f"fusion failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
