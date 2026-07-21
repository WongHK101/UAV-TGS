#!/usr/bin/env python3
"""Benchmark a frozen ThermoNeRF endpoint on the formal test render sequence.

The timed loop includes camera-ray generation, model forward, GPU
synchronization, thermal image transfer, and JPEG writing.  Model loading and
dataset setup are intentionally outside the loop and are still captured by the
outer process receipt.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aaai27_external.depth_export_common import (
    atomic_json,
    formal_test_records,
    load_json,
    sha256_file,
    validate_adapter_sources,
    validate_render_binding,
)


SCHEMA = "uav-tgs-aaai27-thermonerf-render-benchmark-v1"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _single_camera_size(cameras: Any) -> tuple[int, int]:
    widths = cameras.width.detach().cpu().reshape(-1).tolist()
    heights = cameras.height.detach().cpu().reshape(-1).tolist()
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("render benchmark requires one camera per formal view")
    return int(widths[0]), int(heights[0])


def _thermal_u8(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or np.any(~np.isfinite(array)):
        raise ValueError(f"thermal render is not a finite HxW array: {array.shape}")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _formal_target_resolution(binding_payload: dict[str, Any]) -> tuple[int, int]:
    """Return the single frozen evaluation-target resolution.

    ThermoNeRF can have a native training/render raster that differs slightly
    from the canonical thermal target (for example, RGB-sized paired inputs
    versus the valid thermal crop).  That difference is handled by the frozen
    normalization adapter and must not invalidate a native-render timing run.
    """
    resolutions = binding_payload.get("formal_gt_resolutions_wh")
    if (
        not isinstance(resolutions, list)
        or len(resolutions) != 1
        or not isinstance(resolutions[0], list)
        or len(resolutions[0]) != 2
    ):
        raise ValueError(
            f"render binding requires one formal target resolution: {resolutions}"
        )
    width, height = resolutions[0]
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError(f"invalid formal target resolution: {resolutions}")
    return width, height


def benchmark(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    images_root = output_root / "thermal"
    images_root.mkdir(parents=True, exist_ok=True)

    source_repo = args.source_repo.resolve()
    if str(source_repo) not in sys.path:
        sys.path.insert(0, str(source_repo))
    from thermo_nerf.render.renderer import Renderer

    adapter_identity = validate_adapter_sources(
        adapter_manifest=args.adapter_manifest.resolve(),
        train_list=args.train_list.resolve(),
        test_list=args.test_list.resolve(),
        cameras_txt=(args.camera_source_path / "sparse" / "0" / "cameras.txt").resolve(),
        images_txt=(args.camera_source_path / "sparse" / "0" / "images.txt").resolve(),
    )
    test_records = formal_test_records(args.scene_split_manifest.resolve())
    binding_rows = validate_render_binding(args.render_binding_manifest.resolve(), test_records)
    binding_payload = load_json(args.render_binding_manifest.resolve())
    formal_width, formal_height = _formal_target_resolution(binding_payload)

    model_root = args.model_root.resolve()
    configs = list(model_root.rglob("config.yml"))
    checkpoints = list(model_root.rglob("*.ckpt"))
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError("endpoint must contain exactly one config.yml and one checkpoint")
    pipeline, _config, loaded_step = Renderer.extract_pipeline(
        model_root, args.dataset_path.resolve()
    )
    datamanager = pipeline.datamanager
    datamanager.setup_eval()
    loader = datamanager.fixed_indices_eval_dataloader
    if loader is None:
        raise RuntimeError("ThermoNeRF fixed eval dataloader is unavailable")

    image_entries: list[dict[str, Any]] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index, (cameras, _batch) in enumerate(loader):
        if index >= len(binding_rows):
            raise ValueError("eval loader has more views than the formal binding")
        width, height = _single_camera_size(cameras)
        if (width, height) != (int(args.width), int(args.height)):
            raise ValueError(
                f"native ThermoNeRF raster mismatch at {index}: "
                f"{width}x{height} vs {args.width}x{args.height}"
            )
        camera_indices = torch.arange(cameras.camera_to_worlds.shape[0])
        ray_bundle = cameras.generate_rays(camera_indices)
        flat = ray_bundle.flatten()
        pipeline.model.camera_optimizer.apply_to_raybundle(flat)
        ray_bundle = flat.reshape(ray_bundle.shape)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        if "thermal" not in outputs:
            raise ValueError("ThermoNeRF output lacks the official thermal channel")
        thermal = _thermal_u8(outputs["thermal"])
        target = images_root / f"thermal_{index:05d}.jpg"
        Image.fromarray(thermal, mode="L").save(target)
        image_entries.append(
            {
                "pair_id": str(binding_rows[index]["pair_id"]),
                "file": target.relative_to(output_root).as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
        del outputs, ray_bundle, flat, thermal
    torch.cuda.synchronize()
    render_wall_time_s = time.perf_counter() - started
    if len(image_entries) != len(binding_rows):
        raise ValueError(
            f"eval loader/formal view mismatch: {len(image_entries)} vs {len(binding_rows)}"
        )

    payload = {
        "schema": SCHEMA,
        "status": "completed",
        "created_at": _now(),
        "method": "ThermoNeRF",
        "scene": args.scene_name,
        "split": "test",
        "timing_scope": (
            "camera ray generation + frozen model forward + CUDA synchronization + "
            "thermal CPU transfer + JPEG write; excludes model load and dataset setup"
        ),
        "view_count": len(image_entries),
        "width": int(args.width),
        "height": int(args.height),
        "native_render_resolution_wh": [int(args.width), int(args.height)],
        "formal_evaluation_target_resolution_wh": [formal_width, formal_height],
        "native_raster_matches_formal_target": (
            (int(args.width), int(args.height)) == (formal_width, formal_height)
        ),
        "render_wall_time_s": render_wall_time_s,
        "render_test_views_per_s": len(image_entries) / render_wall_time_s,
        "loaded_step": int(loaded_step),
        "model_config": {
            "path": str(configs[0]),
            "size_bytes": configs[0].stat().st_size,
            "sha256": sha256_file(configs[0]),
        },
        "checkpoint": {
            "path": str(checkpoints[0]),
            "size_bytes": checkpoints[0].stat().st_size,
            "sha256": sha256_file(checkpoints[0]),
        },
        "adapter": adapter_identity,
        "render_binding": {
            "path": str(args.render_binding_manifest.resolve()),
            "size_bytes": args.render_binding_manifest.stat().st_size,
            "sha256": sha256_file(args.render_binding_manifest),
        },
        "collection_manifest_sha256": sha256_file(args.collection_manifest.resolve()),
        "scene_split_manifest_sha256": sha256_file(args.scene_split_manifest.resolve()),
        "images": image_entries,
    }
    target = output_root / "benchmark.json"
    atomic_json(target, payload)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", default="Building")
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--camera-source-path", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--render-binding-manifest", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--scene-split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1257)
    parser.add_argument("--height", type=int, default=1006)
    return parser


def main() -> int:
    target = benchmark(build_parser().parse_args())
    print(json.dumps({"status": "completed", "benchmark": str(target)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
