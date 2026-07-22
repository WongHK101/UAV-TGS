#!/usr/bin/env python3
"""Stat only the frozen assets needed to deploy each formal table row.

The collector deliberately excludes optimizer state, intermediate checkpoints,
render outputs, logs, caches, and small configuration files.  It reads file
metadata only; model payloads are never loaded or rewritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "uav-tgs-aaai27-deployable-model-assets-v1"
INTERNAL_METHODS = (
    "raw_f3",
    "scsp_refit_f3",
    "adaptive_opacity_scale_clamp",
)
EXTERNAL_METHODS = (
    "ThermalGaussian-OMMG",
    "MMOne",
    "Thermal3D-GS",
    "ThermoNeRF",
    "PhysIR-Splat-SH",
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _asset(role: str, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"role": role, "path": str(resolved), "size_bytes": resolved.stat().st_size}


def _row(scene: str, method: str, host: str, assets: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    # Count the same physical path only once even if a caller assigns it more
    # than one semantic role.
    by_path: dict[Path, list[str]] = {}
    for role, path in assets:
        resolved = path.resolve()
        by_path.setdefault(resolved, []).append(role)
    rows = []
    for path, roles in sorted(by_path.items(), key=lambda item: str(item[0])):
        value = _asset("+".join(sorted(roles)), path)
        rows.append(value)
    return {
        "scene": scene,
        "method": method,
        "host": host,
        "assets": rows,
        "asset_count": len(rows),
        "model_size_bytes": sum(item["size_bytes"] for item in rows),
    }


def _internal_rows(root: Path, host: str, scenes: Iterable[str]) -> list[dict[str, Any]]:
    experiment = root / "experiments/aaai27_hold8_v2_native"
    rows = []
    for scene in scenes:
        scene_root = experiment / scene
        rgb = scene_root / "rgb_anchor/Model_RGB/point_cloud/iteration_30000/point_cloud.ply"
        for method in INTERNAL_METHODS:
            iteration = 65000 if method == "scsp_refit_f3" else 60000
            thermal = scene_root / f"methods/{method}/model/point_cloud/iteration_{iteration}/point_cloud.ply"
            rgb_endpoint = (
                scene_root
                / "methods/scsp_refit_f3/rgb_refit/point_cloud/iteration_35000/point_cloud.ply"
                if method == "scsp_refit_f3"
                else rgb
            )
            rows.append(
                _row(
                    scene,
                    method,
                    host,
                    (("formal_rgb_endpoint", rgb_endpoint), ("formal_thermal_endpoint", thermal)),
                )
            )
    return rows


def _external_model_root(root: Path, scene: str, method: str) -> Path:
    phase = "external_phase_a" if scene == "Building" else "external_phase_b"
    base = root / phase / "experiments" / scene
    suffix = {
        "ThermalGaussian-OMMG": "ThermalGaussian-OMMG/formal/model",
        "MMOne": "MMOne/formal/model",
        "Thermal3D-GS": "Thermal3D-GS/formal_stream_sync/model",
        "ThermoNeRF": "ThermoNeRF/formal/model",
        "PhysIR-Splat-SH": "PhysIR-Splat/formal_default_sh_no_opacity_reset/model",
    }[method]
    return base / suffix


def _one_checkpoint(model: Path) -> Path:
    matches = sorted(model.rglob("step-000029999.ckpt"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one final ThermoNeRF checkpoint under {model}, found {len(matches)}")
    return matches[0]


def _external_assets(model: Path, method: str) -> tuple[tuple[str, Path], ...]:
    ply = model / "point_cloud/iteration_30000/point_cloud.ply"
    if method == "ThermalGaussian-OMMG":
        return (("joint_rgb_thermal_ply", ply),)
    if method == "MMOne":
        return (("rgb_geometry_ply", ply), ("thermal_checkpoint", model / "thermal_chkpnt30000.pth"))
    if method == "Thermal3D-GS":
        return (
            ("gaussian_ply", ply),
            ("atf_checkpoint", model / "ATF/iteration_30000/ATF.pth"),
            ("tcm_checkpoint", model / "TCM/iteration_30000/TCM.pth"),
        )
    if method == "ThermoNeRF":
        return (("joint_rgb_thermal_checkpoint", _one_checkpoint(model)),)
    if method == "PhysIR-Splat-SH":
        return (
            ("thermal_sh_gaussian_ply", ply),
            ("frozen_geometry_checkpoint", model / "Geometry/iteration_30000/geometry.pth"),
        )
    raise ValueError(method)


def _parse_external(value: str) -> tuple[str, str]:
    try:
        scene, method = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("external selection must be SCENE=METHOD") from error
    if method not in EXTERNAL_METHODS:
        raise argparse.ArgumentTypeError(f"unknown external method: {method}")
    return scene, method


def _external_rows(
    root: Path, host: str, selections: Iterable[tuple[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    for scene, method in selections:
        model = _external_model_root(root, scene, method)
        rows.append(_row(scene, method, host, _external_assets(model, method)))
    return rows


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    rows = _internal_rows(root, args.host_label, args.internal_scene)
    rows += _external_rows(root, args.host_label, args.external)
    keys = [(row["scene"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/method in host inventory")
    _write(
        args.output,
        {
            "schema": SCHEMA,
            "created_at": _now(),
            "host": args.host_label,
            "project_root": str(root),
            "rows": rows,
        },
    )
    return args.output


def merge(args: argparse.Namespace) -> Path:
    rows: list[dict[str, Any]] = []
    inputs = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"unexpected inventory schema: {path}")
        rows.extend(payload["rows"])
        inputs.append(str(path.resolve()))
    keys = [(row["scene"], row["method"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/method across host inventories")
    _write(
        args.output,
        {"schema": SCHEMA, "created_at": _now(), "inputs": inputs, "rows": rows},
    )
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--project-root", type=Path, required=True)
    collect_parser.add_argument("--host-label", required=True)
    collect_parser.add_argument("--internal-scene", action="append", default=[])
    collect_parser.add_argument("--external", action="append", type=_parse_external, default=[])
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.set_defaults(handler=collect)
    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--input", action="append", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.set_defaults(handler=merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.handler(args)
    print(json.dumps({"status": "completed", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
