#!/usr/bin/env python3
"""Export common camera-z expected depth from a frozen external GS endpoint.

This is a depth-only evaluation adapter.  It does not retrain or mutate the
external method.  Geometry is rendered at the frozen shared Hold-8 cameras
with the repository's diagnostic extension of the standard 3DGS compositor.
MMOne may explicitly replace the RGB opacity logits stored in its PLY with the
thermal-opacity logits from its official thermal checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaussian_renderer import render
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from tools.aaai27_external.depth_export_common import (
    formal_test_records,
    sha256_file,
    validate_adapter_sources,
    validate_render_binding,
    write_model_manifest,
    write_view_npz,
)
from utils.camera_utils import cameraList_from_camInfos


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_mmone_thermal_opacity(
    *, checkpoint: Path, gaussians: GaussianModel
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("MMOne checkpoint must contain (model_params, iteration)")
    model_params, iteration = payload
    if not isinstance(model_params, (tuple, list)) or len(model_params) != 15:
        raise ValueError(
            "MMOne formal rgb_thermal+thermal_density checkpoint must have 15 model fields"
        )
    checkpoint_xyz = model_params[1].detach().cpu()
    checkpoint_scaling = model_params[6].detach().cpu()
    checkpoint_rotation = model_params[7].detach().cpu()
    thermal_opacity = model_params[9].detach().cpu()
    comparisons = (
        ("xyz", checkpoint_xyz, gaussians._xyz.detach().cpu()),
        ("scaling", checkpoint_scaling, gaussians._scaling.detach().cpu()),
        ("rotation", checkpoint_rotation, gaussians._rotation.detach().cpu()),
    )
    for name, checkpoint_tensor, ply_tensor in comparisons:
        if checkpoint_tensor.shape != ply_tensor.shape or not torch.equal(
            checkpoint_tensor, ply_tensor
        ):
            raise ValueError(f"MMOne checkpoint/PLY {name} is not byte-exact")
    if thermal_opacity.shape != gaussians._opacity.shape:
        raise ValueError("MMOne thermal opacity/PLY Gaussian count mismatch")
    gaussians._opacity = nn.Parameter(
        thermal_opacity.to(device="cuda", dtype=torch.float32), requires_grad=False
    )
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_iteration": int(iteration),
        "opacity_semantics": "MMOne official thermal-density opacity logits",
        "thermal_opacity_tensor_sha256": _tensor_sha256(thermal_opacity),
        "geometry_checkpoint_ply_exact": True,
    }


def _camera_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        resolution=int(args.resolution),
        data_device="cuda",
        train_test_exp=False,
    )


def _squeeze_hw(value: torch.Tensor, label: str) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{label} did not produce an HxW array: {array.shape}")
    return array.astype(np.float32, copy=False)


def export(args: argparse.Namespace) -> Path:
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    adapter_identity = validate_adapter_sources(
        adapter_manifest=args.adapter_manifest.resolve(),
        train_list=args.train_list.resolve(),
        test_list=args.test_list.resolve(),
        cameras_txt=(args.formal_sparse_root / "cameras.txt").resolve(),
        images_txt=(args.formal_sparse_root / "images.txt").resolve(),
    )
    test_records = formal_test_records(args.scene_split_manifest.resolve())
    binding_rows = validate_render_binding(args.render_binding_manifest.resolve(), test_records)

    camera_train_list = (args.camera_train_list or args.train_list).resolve()
    camera_test_list = (args.camera_test_list or args.test_list).resolve()
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        str(args.camera_source_path.resolve()),
        args.images,
        "",
        True,
        False,
        train_list=str(camera_train_list),
        test_list=str(camera_test_list),
        train_list_sha256=sha256_file(camera_train_list),
        test_list_sha256=sha256_file(camera_test_list),
    )
    cameras = cameraList_from_camInfos(
        scene_info.test_cameras,
        1.0,
        _camera_args(args),
        scene_info.is_nerf_synthetic,
        True,
    )
    if len(cameras) != len(binding_rows):
        raise ValueError("formal camera/render-binding cardinality mismatch")
    for camera, row in zip(cameras, binding_rows):
        if Path(str(camera.image_name)).stem.casefold() != str(row["pair_id"]).casefold():
            raise ValueError("formal camera order differs from render binding")

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(args.model_ply.resolve()))
    opacity_metadata: dict[str, Any] = {
        "opacity_semantics": "standard activated opacity from external final PLY",
    }
    if args.mmone_thermal_checkpoint is not None:
        opacity_metadata = _load_mmone_thermal_opacity(
            checkpoint=args.mmone_thermal_checkpoint.resolve(), gaussians=gaussians
        )

    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    override = torch.ones(
        (gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda"
    )
    view_entries: list[dict[str, Any]] = []
    max_alpha_identity_error = 0.0
    with torch.no_grad():
        for index, (camera, row) in enumerate(zip(cameras, binding_rows)):
            result = render(
                camera,
                gaussians,
                pipeline,
                background,
                override_color=override,
                return_diagnostics=True,
            )
            expected = _squeeze_hw(result["depth_expected_alpha_normalized"], "expected depth")
            weight = _squeeze_hw(result["accumulated_opacity"], "accumulated opacity")
            white = _squeeze_hw(result["render"][0:1], "white override render")
            identity_error = float(np.max(np.abs(white.astype(np.float64) - weight.astype(np.float64))))
            max_alpha_identity_error = max(max_alpha_identity_error, identity_error)
            if identity_error > 2.0e-5:
                raise RuntimeError(
                    f"white-render/accumulated-opacity identity failed at view {index}: {identity_error}"
                )
            target, identity = write_view_npz(
                output_root=output_root,
                index=index,
                expected_depth=expected,
                weight_sum=weight,
            )
            view_entries.append(
                {
                    "pair_id": str(row["pair_id"]),
                    "image_name": str(row["formal_gt_name"]),
                    "camera_name": str(camera.image_name),
                    "width": int(camera.image_width),
                    "height": int(camera.image_height),
                    **identity,
                }
            )
            del result, expected, weight, white
            torch.cuda.empty_cache()

    model_ply = args.model_ply.resolve()
    metadata = {
        "adapter_type": "depth-only standard-3dgs diagnostic sidecar",
        "training_or_endpoint_mutated": False,
        "camera_policy": "shared formal Hold-8 RGB camera-z frame",
        "camera_resolution_argument": int(args.resolution),
        "camera_train_list_sha256": sha256_file(camera_train_list),
        "camera_test_list_sha256": sha256_file(camera_test_list),
        "raster_weight_policy": "standard accepted alpha-times-transmittance weights",
        "model_ply": {
            "path": str(model_ply),
            "size_bytes": model_ply.stat().st_size,
            "sha256": sha256_file(model_ply),
        },
        "external_adapter_manifest": adapter_identity,
        "appearance_render_binding": {
            "path": str(args.render_binding_manifest.resolve()),
            "size_bytes": args.render_binding_manifest.stat().st_size,
            "sha256": sha256_file(args.render_binding_manifest),
        },
        "opacity": opacity_metadata,
        "gaussian_count": int(gaussians.get_xyz.shape[0]),
        "white_render_accumulated_opacity_max_abs_error": max_alpha_identity_error,
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
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--scene-name", default="Building")
    parser.add_argument("--model-ply", type=Path, required=True)
    parser.add_argument("--mmone-thermal-checkpoint", type=Path)
    parser.add_argument("--camera-source-path", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--camera-train-list", type=Path)
    parser.add_argument("--camera-test-list", type=Path)
    parser.add_argument("--formal-sparse-root", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--render-binding-manifest", type=Path, required=True)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--scene-split-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=4)
    return parser


def main() -> int:
    manifest = export(build_parser().parse_args())
    print(json.dumps({"status": "completed", "manifest": str(manifest)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
