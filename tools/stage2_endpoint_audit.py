#!/usr/bin/env python3
"""Fail-closed PLY audit for Stage-2 thermal endpoint experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from plyfile import PlyData


SCHEMA_VERSION = "uav-tgs.stage2-endpoint-audit.v1"
GEOMETRY_FIELDS = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
    "opacity",
)


class EndpointAuditError(RuntimeError):
    """Raised when an endpoint violates a requested invariant."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _vertex_data(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    ply = PlyData.read(str(path))
    try:
        data = ply["vertex"].data
    except KeyError as exc:
        raise EndpointAuditError(f"PLY has no vertex element: {path}") from exc
    if data.dtype.names is None:
        raise EndpointAuditError(f"PLY vertex element has no named fields: {path}")
    return data


def _numbered_fields(names: Sequence[str], prefix: str) -> list[str]:
    selected: list[tuple[int, str]] = []
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if not suffix.isdigit():
            raise EndpointAuditError(f"Non-numeric {prefix} field: {name}")
        selected.append((int(suffix), name))
    selected.sort()
    indices = [index for index, _ in selected]
    if indices != list(range(len(indices))):
        raise EndpointAuditError(f"Non-contiguous {prefix} fields: {indices}")
    return [name for _, name in selected]


def _inactive_rest_fields(names: Sequence[str], max_sh_degree: int) -> list[str]:
    rest = _numbered_fields(names, "f_rest_")
    if not rest or len(rest) % 3:
        raise EndpointAuditError(
            f"Expected three equal f_rest channel blocks, found {len(rest)} fields"
        )
    coefficients_per_channel = len(rest) // 3
    expected_coefficients = (3 + 1) ** 2 - 1
    if coefficients_per_channel != expected_coefficients:
        raise EndpointAuditError(
            "Expected SH3 PLY schema with 15 rest coefficients per channel, "
            f"found {coefficients_per_channel}"
        )
    active_rest = (max_sh_degree + 1) ** 2 - 1
    if not 0 <= active_rest <= coefficients_per_channel:
        raise EndpointAuditError(f"Unsupported thermal max SH degree: {max_sh_degree}")
    inactive: list[str] = []
    for channel in range(3):
        offset = channel * coefficients_per_channel
        inactive.extend(rest[offset + active_rest : offset + coefficients_per_channel])
    return inactive


def _geometry_comparison(reference: np.ndarray, endpoint: np.ndarray) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in GEOMETRY_FIELDS:
        before = np.asarray(reference[name])
        after = np.asarray(endpoint[name])
        finite = bool(np.all(np.isfinite(before)) and np.all(np.isfinite(after)))
        unchanged = bool(finite and np.array_equal(before, after))
        max_abs_diff = (
            float(np.max(np.abs(after.astype(np.float64) - before.astype(np.float64))))
            if finite and before.size
            else (0.0 if finite else None)
        )
        report[name] = {
            "finite": finite,
            "unchanged": unchanged,
            "max_abs_diff": max_abs_diff,
            "reference_sha256": _array_sha256(before),
            "endpoint_sha256": _array_sha256(after),
        }
    return report


def _inactive_sh_audit(endpoint: np.ndarray, max_sh_degree: int) -> dict[str, Any]:
    names = list(endpoint.dtype.names or ())
    inactive = _inactive_rest_fields(names, max_sh_degree)
    nonzero_fields: list[str] = []
    nonfinite_fields: list[str] = []
    maximum = 0.0
    for name in inactive:
        values = np.asarray(endpoint[name])
        if not np.all(np.isfinite(values)):
            nonfinite_fields.append(name)
            continue
        field_max = float(np.max(np.abs(values))) if values.size else 0.0
        maximum = max(maximum, field_max)
        if field_max != 0.0:
            nonzero_fields.append(name)
    return {
        "thermal_max_sh_degree": max_sh_degree,
        "inactive_field_count": len(inactive),
        "inactive_fields": inactive,
        "nonfinite_fields": nonfinite_fields,
        "nonzero_fields": nonzero_fields,
        "max_abs_value": maximum,
        "all_inactive_fields_exact_zero": not nonzero_fields and not nonfinite_fields,
    }


def audit_endpoint_group(
    *,
    rgb_ply: Path,
    model_root: Path,
    group: str,
    iterations: Sequence[int],
    thermal_max_sh_degree: int,
    strict_geometry: bool,
) -> dict[str, Any]:
    rgb_path = rgb_ply.resolve()
    reference = _vertex_data(rgb_path)
    reference_names = list(reference.dtype.names or ())
    missing_geometry = [name for name in GEOMETRY_FIELDS if name not in reference_names]
    if missing_geometry:
        raise EndpointAuditError(f"RGB PLY missing geometry fields: {missing_geometry}")
    _inactive_rest_fields(reference_names, thermal_max_sh_degree)

    endpoints: list[dict[str, Any]] = []
    overall_passed = True
    for iteration in iterations:
        endpoint_path = (
            model_root.resolve()
            / "point_cloud"
            / f"iteration_{int(iteration)}"
            / "point_cloud.ply"
        )
        endpoint = _vertex_data(endpoint_path)
        endpoint_names = list(endpoint.dtype.names or ())
        schema_equal = endpoint_names == reference_names
        count_equal = len(endpoint) == len(reference)
        endpoint_report: dict[str, Any] = {
            "iteration": int(iteration),
            "path": str(endpoint_path),
            "ply_sha256": _sha256(endpoint_path),
            "gaussian_count": len(endpoint),
            "gaussian_count_equal_rgb": count_equal,
            "schema_equal_rgb": schema_equal,
        }
        if schema_equal:
            sh_report = _inactive_sh_audit(endpoint, thermal_max_sh_degree)
        else:
            sh_report = {
                "thermal_max_sh_degree": thermal_max_sh_degree,
                "all_inactive_fields_exact_zero": False,
                "error": "schema differs from RGB PLY",
            }
        endpoint_report["inactive_sh"] = sh_report

        if strict_geometry and schema_equal and count_equal:
            geometry = _geometry_comparison(reference, endpoint)
            geometry_passed = all(item["unchanged"] for item in geometry.values())
            endpoint_report["geometry"] = {
                "requested": True,
                "passed": geometry_passed,
                "fields": geometry,
            }
        elif strict_geometry:
            geometry_passed = False
            endpoint_report["geometry"] = {
                "requested": True,
                "passed": False,
                "error": "schema or Gaussian count differs from RGB PLY",
            }
        else:
            geometry_passed = True
            endpoint_report["geometry"] = {"requested": False, "passed": None}

        passed = bool(
            schema_equal
            and count_equal
            and sh_report["all_inactive_fields_exact_zero"]
            and geometry_passed
        )
        endpoint_report["status"] = "passed" if passed else "failed"
        overall_passed = overall_passed and passed
        endpoints.append(endpoint_report)

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if overall_passed else "failed",
        "group": group,
        "configuration": {
            "iterations": [int(value) for value in iterations],
            "thermal_max_sh_degree": thermal_max_sh_degree,
            "strict_geometry": strict_geometry,
            "geometry_fields": list(GEOMETRY_FIELDS),
            "inactive_sh_rule": "SH3 PLY schema retained; coefficients above thermal max must be finite exact zero",
        },
        "rgb_reference": {
            "path": str(rgb_path),
            "ply_sha256": _sha256(rgb_path),
            "gaussian_count": len(reference),
            "fields": reference_names,
        },
        "endpoints": endpoints,
    }


def _write_report(path: Path, payload: Mapping[str, Any], overwrite: bool) -> Path:
    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-ply", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--iterations", nargs="+", type=int, required=True)
    parser.add_argument("--thermal-max-sh-degree", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--strict-geometry", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = audit_endpoint_group(
        rgb_ply=args.rgb_ply,
        model_root=args.model_root,
        group=args.group,
        iterations=args.iterations,
        thermal_max_sh_degree=args.thermal_max_sh_degree,
        strict_geometry=args.strict_geometry,
    )
    report = _write_report(args.report, payload, args.overwrite)
    print(
        json.dumps(
            {"status": payload["status"], "group": args.group, "report": str(report)},
            sort_keys=True,
        )
    )
    if payload["status"] != "passed":
        raise EndpointAuditError(f"Stage-2 endpoint audit failed for group {args.group}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EndpointAuditError, FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
