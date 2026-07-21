#!/usr/bin/env python3
"""Export ThermoNeRF's native expected depth in metric camera-z units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aaai27_external.depth_export_common import (
    camera_z_from_ray_distance,
    formal_test_records,
    load_json,
    sha256_file,
    validate_adapter_sources,
    validate_render_binding,
    write_model_manifest,
    write_view_npz,
)


def _as_hw(tensor: torch.Tensor, label: str) -> np.ndarray:
    array = tensor.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{label} did not produce an HxW array: {array.shape}")
    return array.astype(np.float32, copy=False)


def _dataparser_scale(datamanager: Any) -> float:
    dataset = datamanager.eval_dataset
    outputs = getattr(dataset, "_dataparser_outputs", None)
    if outputs is None:
        raise ValueError("ThermoNeRF eval dataset does not expose dataparser outputs")
    scale = float(getattr(outputs, "dataparser_scale", float("nan")))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid ThermoNeRF dataparser scale: {scale}")
    return scale


def _reference_shapes(reference_manifest: Path) -> dict[str, tuple[int, int]]:
    manifest = load_json(reference_manifest)
    root = reference_manifest.resolve().parent
    result: dict[str, tuple[int, int]] = {}
    for row in manifest.get("views", []):
        pair_id = str(row.get("pair_id", ""))
        path = (root / str(row.get("npz_file", ""))).resolve()
        if not pair_id or pair_id in result or not path.is_file():
            raise ValueError("reference manifest has invalid view identities")
        with np.load(path, allow_pickle=False) as arrays:
            depth = np.asarray(arrays["depth"])
        if depth.ndim != 2:
            raise ValueError("reference depth is not HxW")
        result[pair_id] = (int(depth.shape[1]), int(depth.shape[0]))
    if not result:
        raise ValueError("reference manifest has no views")
    return result


def _rescale_cameras_exact(cameras: Any, target_width: int, target_height: int) -> None:
    """Match the frozen reference raster without interpolating rendered depth."""

    current_width = float(cameras.width.reshape(-1)[0].item())
    current_height = float(cameras.height.reshape(-1)[0].item())
    if current_width <= 0.0 or current_height <= 0.0:
        raise ValueError("invalid native ThermoNeRF camera size")
    scale_x = float(target_width) / current_width
    scale_y = float(target_height) / current_height
    cameras.fx *= scale_x
    cameras.cx *= scale_x
    cameras.fy *= scale_y
    cameras.cy *= scale_y
    cameras.width[...] = int(target_width)
    cameras.height[...] = int(target_height)


def export(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

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
    reference_shapes = _reference_shapes(args.reference_manifest.resolve())
    expected_pairs = {str(row["pair_id"]) for row in binding_rows}
    if set(reference_shapes) != expected_pairs:
        raise ValueError("reference/render binding view sets differ")

    model_root = args.model_root.resolve()
    config_paths = list(model_root.rglob("config.yml"))
    checkpoint_paths = list(model_root.rglob("*.ckpt"))
    if len(config_paths) != 1 or len(checkpoint_paths) != 1:
        raise ValueError(
            "ThermoNeRF endpoint must contain exactly one config.yml and one checkpoint"
        )
    model_config = config_paths[0]
    checkpoint = checkpoint_paths[0]
    pipeline, config, loaded_step = Renderer.extract_pipeline(
        model_root, args.dataset_path.resolve()
    )
    datamanager = pipeline.datamanager
    datamanager.setup_eval()
    loader = datamanager.fixed_indices_eval_dataloader
    if loader is None:
        raise RuntimeError("ThermoNeRF fixed eval dataloader is unavailable")
    scale = _dataparser_scale(datamanager)

    view_entries: list[dict[str, Any]] = []
    for index, (cameras, _batch) in enumerate(loader):
        if index >= len(binding_rows):
            raise ValueError("ThermoNeRF eval loader has more views than the formal binding")
        row = binding_rows[index]
        target_width, target_height = reference_shapes[str(row["pair_id"])]
        _rescale_cameras_exact(cameras, target_width, target_height)
        camera_indices = torch.arange(cameras.camera_to_worlds.shape[0])
        ray_bundle = cameras.generate_rays(camera_indices)
        flat = ray_bundle.flatten()
        pipeline.model.camera_optimizer.apply_to_raybundle(flat)
        ray_bundle = flat.reshape(ray_bundle.shape)
        directions_norm_tensor = ray_bundle.metadata.get("directions_norm")
        if directions_norm_tensor is None:
            raise ValueError("ThermoNeRF ray bundle lacks directions_norm")
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        expected_distance = _as_hw(outputs["expected_depth"], "expected ray depth")
        accumulation = _as_hw(outputs["accumulation"], "accumulation")
        directions_norm = _as_hw(directions_norm_tensor, "directions_norm")
        expected_camera_z = camera_z_from_ray_distance(
            expected_distance, directions_norm, scale
        )
        target, identity = write_view_npz(
            output_root=output_root,
            index=index,
            expected_depth=expected_camera_z,
            weight_sum=accumulation,
        )
        view_entries.append(
            {
                "pair_id": str(row["pair_id"]),
                "image_name": str(row["formal_gt_name"]),
                "width": int(expected_camera_z.shape[1]),
                "height": int(expected_camera_z.shape[0]),
                **identity,
            }
        )
        del outputs, ray_bundle, flat
        torch.cuda.empty_cache()
    if len(view_entries) != len(binding_rows):
        raise ValueError(
            f"ThermoNeRF eval loader/formal view mismatch: {len(view_entries)} vs {len(binding_rows)}"
        )

    metadata = {
        "adapter_type": "native ThermoNeRF expected-depth export",
        "training_or_endpoint_mutated": False,
        "native_depth_semantics": "normalized volume-weighted distance along unit ray",
        "camera_z_conversion": "ray_distance / directions_norm / dataparser_scale",
        "dataparser_scale": scale,
        "camera_output_policy": "per-axis intrinsic rescale to frozen reference raster",
        "reference_manifest": {
            "path": str(args.reference_manifest.resolve()),
            "size_bytes": args.reference_manifest.stat().st_size,
            "sha256": sha256_file(args.reference_manifest),
        },
        "model_config": {
            "path": str(model_config),
            "size_bytes": model_config.stat().st_size,
            "sha256": sha256_file(model_config),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
            "loaded_step": int(loaded_step),
        },
        "external_adapter_manifest": adapter_identity,
        "appearance_render_binding": {
            "path": str(args.render_binding_manifest.resolve()),
            "size_bytes": args.render_binding_manifest.stat().st_size,
            "sha256": sha256_file(args.render_binding_manifest),
        },
    }
    return write_model_manifest(
        output_root=output_root,
        method_name=args.method_name,
        scene_name=args.scene_name,
        collection_manifest=args.collection_manifest,
        scene_split_manifest=args.scene_split_manifest,
        views=view_entries,
        exporter_metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-name", default="ThermoNeRF")
    parser.add_argument("--scene-name", default="Building")
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--camera-source-path", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--render-binding-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--scene-split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    manifest = export(build_parser().parse_args())
    print(json.dumps({"status": "completed", "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
