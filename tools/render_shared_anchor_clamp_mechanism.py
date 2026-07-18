#!/usr/bin/env python3
"""Render fixed free-view ellipsoid diagnostics for shared-anchor clamp rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


class MechanismRenderError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MechanismRenderError("manifest root must be an object")
    return value


def _load_params(path: Path):
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise MechanismRenderError(f"invalid checkpoint: {path}")
    params, iteration = payload
    if not isinstance(params, (tuple, list)) or len(params) != 12:
        raise MechanismRenderError(f"invalid Gaussian payload: {path}")
    return params, int(iteration)


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise MechanismRenderError("invalid quaternion")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ellipsoid(
    center: np.ndarray,
    scaling: np.ndarray,
    rotation: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sphere = np.stack(
        [
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
        ],
        axis=0,
    ).reshape(3, -1)
    transformed = center[:, None] + rotation @ (scaling[:, None] * sphere)
    shape = (len(u), len(v))
    return tuple(transformed[index].reshape(shape) for index in range(3))


def render(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise MechanismRenderError(f"refusing to overwrite: {output_dir}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "passed":
        raise MechanismRenderError("shared-anchor manifest is not passed")
    adaptive = manifest.get("schema") == "uav-tgs-adaptive-scale-anchor-v1"
    input_checkpoint = Path(manifest["input"]["checkpoint"]).resolve()
    if adaptive:
        output_checkpoint = Path(manifest["output"]["checkpoint"]).resolve()
        input_checkpoint_sha = manifest["input"]["checkpoint_sha256"]
        indices = np.asarray(manifest["modified_indices"], dtype=np.int64)
        expected_count = manifest["counts"]["modified_gaussians"]
        indices_sha = manifest["modified_indices_sha256"]
        method_label = str(manifest["method"])
    else:
        output_checkpoint = Path(manifest["output"]["checkpoint"]).resolve()
        input_checkpoint_sha = manifest["input"]["checkpoint_sha256_before"]
        indices = np.asarray(manifest["clamped_indices"], dtype=np.int64)
        expected_count = manifest["counts"]["actual_clamped_gaussians"]
        indices_sha = manifest["clamped_indices_sha256"]
        method_label = "absolute_clamp"
    if (
        _sha256(input_checkpoint) != input_checkpoint_sha
        or _sha256(output_checkpoint) != manifest["output"]["checkpoint_sha256"]
    ):
        raise MechanismRenderError("checkpoint identity differs from the manifest")
    raw_params, raw_iteration = _load_params(input_checkpoint)
    shared_params, shared_iteration = _load_params(output_checkpoint)
    if raw_iteration != shared_iteration or raw_iteration != manifest["anchor_iteration"]:
        raise MechanismRenderError("checkpoint iteration mismatch")
    xyz = raw_params[1].detach().numpy()
    raw_scale = np.exp(raw_params[4].detach().numpy())
    shared_scale = np.exp(shared_params[4].detach().numpy())
    rotation = raw_params[5].detach().numpy()
    if len(indices) != expected_count:
        raise MechanismRenderError("manifest clamp count/index list mismatch")
    if (
        not np.array_equal(raw_params[1].detach().numpy(), shared_params[1].detach().numpy())
        or not np.array_equal(rotation, shared_params[5].detach().numpy())
    ):
        raise MechanismRenderError("xyz or rotation changed between anchors")

    if len(indices) == 0:
        output_dir.mkdir(parents=True)
        figure = plt.figure(figsize=(8, 3), dpi=160)
        axis = figure.add_subplot(1, 1, 1)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            f"{manifest['scene']} {method_label}\nNo Gaussian exceeds the robust fence",
            ha="center",
            va="center",
            fontsize=15,
        )
        image_path = output_dir / "ellipsoid_free_views.png"
        figure.savefig(image_path, bbox_inches="tight")
        plt.close(figure)
        report = {
            "schema": "uav-tgs-scale-projection-mechanism-render-v1",
            "status": "complete",
            "scene": manifest["scene"],
            "anchor_iteration": raw_iteration,
            "method": method_label,
            "clamped_gaussian_count": 0,
            "modified_indices_sha256": indices_sha,
            "input_manifest": str(manifest_path),
            "input_manifest_sha256": _sha256(manifest_path),
            "raw_checkpoint_sha256": _sha256(input_checkpoint),
            "shared_checkpoint_sha256": _sha256(output_checkpoint),
            "views": [],
            "render": str(image_path),
            "render_sha256": _sha256(image_path),
            "interpretation": "the robust fence selected no Gaussian; output is exactly the raw anchor",
        }
        (output_dir / "mechanism_render_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report

    centers = xyz[indices]
    context_count = min(int(args.context_points), xyz.shape[0])
    stride = max(xyz.shape[0] // max(context_count, 1), 1)
    context = xyz[::stride][:context_count]
    u = np.linspace(0.0, 2.0 * np.pi, 13)
    v = np.linspace(0.0, np.pi, 9)
    ellipsoids: list[tuple[Any, Any]] = []
    bounds = [centers]
    for position, index in enumerate(indices):
        matrix = _rotation_matrix(rotation[index])
        raw_surface = _ellipsoid(
            centers[position], raw_scale[index], matrix, u, v
        )
        shared_surface = _ellipsoid(
            centers[position], shared_scale[index], matrix, u, v
        )
        ellipsoids.append((raw_surface, shared_surface))
        bounds.extend(
            [
                np.stack(raw_surface, axis=-1).reshape(-1, 3),
                np.stack(shared_surface, axis=-1).reshape(-1, 3),
            ]
        )
    bound_points = np.concatenate(bounds, axis=0)
    lower = np.min(bound_points, axis=0)
    upper = np.max(bound_points, axis=0)
    center = (lower + upper) / 2.0
    half = max(float(np.max(upper - lower)) / 2.0, 1e-3) * 1.08

    views = [
        {"elevation_deg": 25.0, "azimuth_deg": -60.0},
        {"elevation_deg": 25.0, "azimuth_deg": 0.0},
        {"elevation_deg": 25.0, "azimuth_deg": 60.0},
    ]
    figure = plt.figure(figsize=(18, 6), dpi=160)
    for panel, view in enumerate(views, start=1):
        axis = figure.add_subplot(1, 3, panel, projection="3d")
        axis.scatter(
            context[:, 0],
            context[:, 1],
            context[:, 2],
            s=0.25,
            c="#777777",
            alpha=0.12,
            rasterized=True,
        )
        for raw_surface, shared_surface in ellipsoids:
            axis.plot_wireframe(
                *raw_surface,
                rstride=2,
                cstride=2,
                color="#d95f02",
                linewidth=0.45,
                alpha=0.55,
            )
            axis.plot_wireframe(
                *shared_surface,
                rstride=2,
                cstride=2,
                color="#1b9e77",
                linewidth=0.55,
                alpha=0.75,
            )
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=5,
            c="black",
            alpha=0.8,
        )
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(
            elev=view["elevation_deg"], azim=view["azimuth_deg"]
        )
        axis.set_title(
            f"elev={view['elevation_deg']:.0f}°, "
            f"azim={view['azimuth_deg']:.0f}°"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    figure.suptitle(
        f"{manifest['scene']} {method_label}: "
        "raw (orange) vs projected anchor (green)"
    )
    figure.tight_layout()
    output_dir.mkdir(parents=True)
    image_path = output_dir / "ellipsoid_free_views.png"
    figure.savefig(image_path, bbox_inches="tight")
    plt.close(figure)

    report = {
        "schema": "uav-tgs-scale-projection-mechanism-render-v1",
        "status": "complete",
        "scene": manifest["scene"],
        "anchor_iteration": raw_iteration,
        "clamped_gaussian_count": len(indices),
        "method": method_label,
        "modified_indices_sha256": indices_sha,
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": _sha256(manifest_path),
        "raw_checkpoint_sha256": _sha256(input_checkpoint),
        "shared_checkpoint_sha256": _sha256(output_checkpoint),
        "views": views,
        "context_point_sampling": {
            "rule": "deterministic global stride",
            "requested": int(args.context_points),
            "actual": int(context.shape[0]),
            "stride": int(stride),
        },
        "render": str(image_path),
        "render_sha256": _sha256(image_path),
        "interpretation": (
            "mechanism-only free-view visualization of the fixed clamp rows; "
            "not a geometry metric or method contribution"
        ),
    }
    report_path = output_dir / "mechanism_render_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-points", type=int, default=5000)
    args = parser.parse_args()
    try:
        payload = render(args)
    except (MechanismRenderError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
