from __future__ import annotations

"""Zero-update raw/resplat checkpoint diagnostic on four fixed formal views.

This module deliberately contains no candidate-selection, resplatting, training,
or parameter-tuning code.  It restores two checkpoints without constructing an
optimizer, renders the same four train/guard cameras, and measures only output
drift.  Heavy Gaussian-splatting imports stay lazy so the deterministic
selection, metric, and panel helpers can be unit-tested on CPU.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROTOCOL = "uav-tgs-zero-update-resplat-diagnostic-v1"
SELECTION_RECEIPT_PROTOCOL = "uav-tgs-zero-update-resplat-selection-receipt-v1"
ALPHA_THRESHOLD = 0.01
GROUP_BLOCK_OFFSET = 7
SELECTION_GROUPS = (
    ("train", "nadir"),
    ("train", "oblique"),
    ("guard", "nadir"),
    ("guard", "oblique"),
)
PSNR_MSE_FLOOR = 1.0e-12


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _normalized_stem(value: Any) -> str:
    return Path(str(value).strip().replace("\\", "/")).stem.casefold()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_orientation(record: Mapping[str, Any]) -> str:
    for key in ("orientation", "view_class", "view_type", "stratum"):
        value = str(record.get(key, "")).strip().casefold()
        if "nadir" in value:
            return "nadir"
        if "oblique" in value:
            return "oblique"
    for key in ("gimbal_pitch_deg", "pitch_deg"):
        value = record.get(key)
        if value is None or str(value).strip() == "":
            continue
        pitch = float(value)
        if not math.isfinite(pitch):
            break
        return "nadir" if pitch <= -75.0 else "oblique"
    raise ValueError(f"Cannot determine nadir/oblique orientation for record {record.get('pair_id', '')!r}")


def _finite_int(value: Any, *, default: int = 2**31 - 1) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return numeric


def _formal_order_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("strip_id", "")).casefold(),
        _finite_int(record.get("block_index")),
        _finite_int(record.get("block_offset")),
        _finite_int(record.get("position_in_strip")),
        str(record.get("timestamp_utc", "")),
        str(record.get("pair_id", record.get("camera_name", record.get("filename", "")))).casefold(),
    )


def _record_public_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "scene",
        "scene_name",
        "pair_id",
        "camera_name",
        "filename",
        "split",
        "stratum",
        "gimbal_pitch_deg",
        "strip_id",
        "block_index",
        "block_offset",
        "position_in_strip",
        "source_record_hash",
        "hash",
    )
    return {key: record[key] for key in keys if key in record}


def select_fixed_formal_records(split_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Select exactly train/guard x nadir/oblique at immutable group offset 7.

    Guard views are adjacent frames and the formal manifests contain no guard
    record whose *formal* ``block_offset`` equals seven.  Therefore the fixed
    protocol orders each eligible group by immutable formal fields and selects
    its zero-based item 7.  The actual formal block fields are retained in the
    result so this distinction cannot be hidden.
    """

    records = split_payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Formal split manifest records must be a list")
    eligible: dict[tuple[str, str], list[Mapping[str, Any]]] = {group: [] for group in SELECTION_GROUPS}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Formal split records must be objects")
        split = str(record.get("split", "")).strip().casefold()
        if split not in {"train", "guard"}:
            # Test records are deliberately not classified, ordered, or selected.
            continue
        orientation = _record_orientation(record)
        group = (split, orientation)
        if group in eligible:
            eligible[group].append(record)

    selected: list[dict[str, Any]] = []
    observed_stems: set[str] = set()
    for split, orientation in SELECTION_GROUPS:
        ordered = sorted(eligible[(split, orientation)], key=_formal_order_key)
        if len(ordered) <= GROUP_BLOCK_OFFSET:
            raise ValueError(
                f"Need at least {GROUP_BLOCK_OFFSET + 1} records for {split}/{orientation}, got {len(ordered)}"
            )
        record = ordered[GROUP_BLOCK_OFFSET]
        stem = _record_stem(record)
        if not stem or stem in observed_stems:
            raise ValueError(f"Fixed selection produced an empty or duplicate image stem: {stem!r}")
        observed_stems.add(stem)
        selected.append(
            {
                "selection_id": f"{split}_{orientation}",
                "split": split,
                "orientation": orientation,
                "ordered_group_count": len(ordered),
                "ordered_group_offset_zero_based": GROUP_BLOCK_OFFSET,
                "formal_record": _record_public_fields(record),
                "image_stem": stem,
            }
        )
    return selected


def _record_stem(record: Mapping[str, Any]) -> str:
    for key in ("pair_id", "camera_name", "filename", "thermal_camera_name"):
        stem = _normalized_stem(record.get(key, ""))
        if stem:
            return stem
    original = record.get("original_files")
    if isinstance(original, dict):
        for value in original.values():
            stem = _normalized_stem(value)
            if stem:
                return stem
    return ""


def _views_by_stem(probe_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    views = probe_payload.get("views")
    if not isinstance(views, list):
        raise ValueError("Probe camera manifest views must be a list")
    mapped: dict[str, dict[str, Any]] = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError("Probe camera views must be objects")
        stem = _normalized_stem(view.get("image_name", view.get("pair_id", "")))
        if not stem:
            raise ValueError("Probe camera view has no image name")
        if stem in mapped:
            raise ValueError(f"Duplicate probe camera stem {stem!r}")
        mapped[stem] = view
    return mapped


def bind_selected_views(
    selected: Sequence[Mapping[str, Any]], probe_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    mapped = _views_by_stem(probe_payload)
    bound: list[dict[str, Any]] = []
    for entry in selected:
        stem = str(entry["image_stem"])
        view = mapped.get(stem)
        if view is None:
            raise KeyError(f"Fixed formal record {stem!r} is absent from the probe camera manifest")
        split = str(view.get("bound_split", view.get("split", ""))).strip().casefold()
        if split != str(entry["split"]):
            raise ValueError(f"Probe/formal split mismatch for {stem}: {split!r} != {entry['split']!r}")
        if split == "test":
            raise ValueError("Test cameras are forbidden by the zero-update diagnostic")
        bound.append({**dict(entry), "camera_view": dict(view)})
    return bound


def _validate_rgb_hwc(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{label} must be HxWx3 or 3xHxW, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN/Inf")
    return np.clip(array, 0.0, 1.0)


def _validate_alpha_hw(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{label} must be HxW, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN/Inf")
    return np.clip(array, 0.0, 1.0)


def compute_pair_metrics(
    raw_rgb: Any,
    resplat_rgb: Any,
    raw_alpha: Any,
    resplat_alpha: Any,
    *,
    ssim_value: float,
    lpips_value: float,
    alpha_threshold: float = ALPHA_THRESHOLD,
) -> dict[str, float]:
    raw_rgb_array = _validate_rgb_hwc(raw_rgb, label="raw_rgb")
    resplat_rgb_array = _validate_rgb_hwc(resplat_rgb, label="resplat_rgb")
    raw_alpha_array = _validate_alpha_hw(raw_alpha, label="raw_alpha")
    resplat_alpha_array = _validate_alpha_hw(resplat_alpha, label="resplat_alpha")
    if raw_rgb_array.shape != resplat_rgb_array.shape:
        raise ValueError(f"RGB render shape mismatch: {raw_rgb_array.shape} != {resplat_rgb_array.shape}")
    if raw_alpha_array.shape != resplat_alpha_array.shape or raw_alpha_array.shape != raw_rgb_array.shape[:2]:
        raise ValueError("RGB/alpha render dimensions do not match")
    if not 0.0 <= float(alpha_threshold) <= 1.0:
        raise ValueError("alpha_threshold must be in [0,1]")

    raw_mask = raw_alpha_array > float(alpha_threshold)
    resplat_mask = resplat_alpha_array > float(alpha_threshold)
    union = int(np.count_nonzero(raw_mask | resplat_mask))
    intersection = int(np.count_nonzero(raw_mask & resplat_mask))
    alpha_abs = np.abs(raw_alpha_array - resplat_alpha_array)
    rgb_delta = raw_rgb_array - resplat_rgb_array
    mse = float(np.mean(np.square(rgb_delta)))
    psnr_db = float(-10.0 * math.log10(max(mse, PSNR_MSE_FLOOR)))
    metrics = {
        "alpha_coverage_iou": float(intersection / union) if union else 1.0,
        "alpha_mae": float(np.mean(alpha_abs)),
        "alpha_p95_abs": float(np.percentile(alpha_abs, 95.0)),
        "rgb_psnr_db": psnr_db,
        "rgb_ssim": float(ssim_value),
        "rgb_lpips": float(lpips_value),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite diagnostic metric: {metrics}")
    return metrics


def _to_uint8_rgb(value: Any) -> np.ndarray:
    return np.rint(_validate_rgb_hwc(value, label="panel_rgb") * 255.0).astype(np.uint8)


def _heatmap_gray(value: np.ndarray) -> np.ndarray:
    gray = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.stack((gray, np.sqrt(gray), 1.0 - gray), axis=-1)


def save_comparison_panel(
    path: Path,
    *,
    raw_rgb: Any,
    resplat_rgb: Any,
    raw_alpha: Any,
    resplat_alpha: Any,
    title: str,
) -> None:
    raw = _validate_rgb_hwc(raw_rgb, label="raw_rgb")
    resplat = _validate_rgb_hwc(resplat_rgb, label="resplat_rgb")
    alpha_delta = np.abs(
        _validate_alpha_hw(raw_alpha, label="raw_alpha")
        - _validate_alpha_hw(resplat_alpha, label="resplat_alpha")
    )
    rgb_delta = np.mean(np.abs(raw - resplat), axis=-1)
    arrays = (
        _to_uint8_rgb(raw),
        _to_uint8_rgb(resplat),
        np.rint(_heatmap_gray(np.minimum(1.0, alpha_delta * 10.0)) * 255.0).astype(np.uint8),
        np.rint(_heatmap_gray(np.minimum(1.0, rgb_delta * 10.0)) * 255.0).astype(np.uint8),
    )
    labels = ("Raw RGB", "Resplat RGB", "|alpha delta| x10", "mean |RGB delta| x10")
    height, width = arrays[0].shape[:2]
    if any(array.shape[:2] != (height, width) for array in arrays):
        raise ValueError("Panel inputs have different dimensions")
    header = 42
    canvas = Image.new("RGB", (width * len(arrays), height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill="black")
    for index, (array, label) in enumerate(zip(arrays, labels)):
        x = index * width
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, header))
        draw.text((x + 6, 22), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=False)


def render_model_view(
    camera: Any,
    gaussians: Any,
    pipeline: Any,
    *,
    render_fn: Callable[..., Mapping[str, Any]] | None = None,
    white_background: bool = False,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Render RGB and the established white-override/black-bg alpha proxy."""

    import torch

    if render_fn is None:
        from gaussian_renderer import render as render_fn

    xyz = gaussians.get_xyz
    if int(xyz.shape[0]) <= 0:
        raise ValueError("Cannot render an empty Gaussian model")
    device = xyz.device
    rgb_background = torch.ones(3, dtype=torch.float32, device=device) if white_background else torch.zeros(
        3, dtype=torch.float32, device=device
    )
    black_background = torch.zeros(3, dtype=torch.float32, device=device)
    white_override = torch.ones((int(xyz.shape[0]), 3), dtype=torch.float32, device=device)
    with torch.no_grad():
        rgb_package = render_fn(
            camera,
            gaussians,
            pipeline,
            rgb_background,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=None,
            use_trained_exp=False,
        )
        alpha_package = render_fn(
            camera,
            gaussians,
            pipeline,
            black_background,
            scaling_modifier=1.0,
            separate_sh=False,
            override_color=white_override,
            use_trained_exp=False,
        )
    rgb_tensor = rgb_package["render"].detach().float().clamp(0.0, 1.0)
    alpha_tensor = alpha_package["render"].detach().float()[0].clamp(0.0, 1.0)
    rgb = np.moveaxis(rgb_tensor.cpu().numpy(), 0, -1)
    alpha = alpha_tensor.cpu().numpy()
    _validate_rgb_hwc(rgb, label="render_rgb")
    _validate_alpha_hw(alpha, label="render_alpha")
    return rgb, alpha, rgb_tensor


def _appearance_metrics(raw_tensor: Any, resplat_tensor: Any) -> tuple[float, float]:
    import torch
    from lpipsPyTorch import lpips
    from utils.loss_utils import ssim

    with torch.no_grad():
        raw_batch = raw_tensor.unsqueeze(0)
        resplat_batch = resplat_tensor.unsqueeze(0)
        ssim_value = float(ssim(raw_batch, resplat_batch).detach().cpu().item())
        lpips_value = float(lpips(raw_batch, resplat_batch, net_type="vgg").detach().cpu().item())
    return ssim_value, lpips_value


def restore_checkpoint_for_render(
    checkpoint_path: Path,
    *,
    sh_degree: int,
    device: str,
    model_factory: Callable[[int], Any] | None = None,
) -> tuple[Any, int]:
    """Restore render tensors only; never instantiate an optimizer or training state."""

    import torch
    if model_factory is None:
        from scene.gaussian_model import GaussianModel

        model_factory = GaussianModel

    checkpoint_path = checkpoint_path.resolve()
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise ValueError("Expected repository checkpoint payload (model_args, iteration)")
    model_args, iteration = payload
    if not isinstance(model_args, (tuple, list)) or len(model_args) != 12:
        raise ValueError("Expected GaussianModel.capture() 12-field payload")
    active_degree = int(model_args[0])
    if active_degree < 0 or active_degree > int(sh_degree):
        raise ValueError(f"Checkpoint active SH degree {active_degree} is incompatible with --sh_degree={sh_degree}")
    model = model_factory(int(sh_degree))
    fields = (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_scaling",
        "_rotation",
        "_opacity",
        "max_radii2D",
        "xyz_gradient_accum",
        "denom",
    )
    for name, value in zip(fields, model_args[1:10]):
        if not torch.is_tensor(value):
            raise ValueError(f"Checkpoint field {name} is not a tensor")
        setattr(model, name, value.detach().to(device=device).requires_grad_(False))
    count = int(model._xyz.shape[0])
    if count <= 0 or any(int(getattr(model, name).shape[0]) != count for name in fields[1:6]):
        raise ValueError("Checkpoint Gaussian tensor counts are inconsistent")
    model.active_sh_degree = active_degree
    model.spatial_lr_scale = float(model_args[11])
    model.optimizer = None
    return model, int(iteration)


def _camera_from_view(view: Mapping[str, Any], *, uid: int, data_device: str) -> Any:
    from scene.cameras import Camera
    from utils.graphics_utils import focal2fov

    width = int(view["width"])
    height = int(view["height"])
    if width <= 0 or height <= 0:
        raise ValueError("Camera dimensions must be positive")
    cx = float(view["cx"])
    cy = float(view["cy"])
    if not np.isclose(cx, width / 2.0, atol=1.0e-6, rtol=0.0) or not np.isclose(
        cy, height / 2.0, atol=1.0e-6, rtol=0.0
    ):
        raise ValueError("Repository rasterizer path requires centered principal points")
    c2w_value = view.get("native_camera_to_world", view.get("camera_to_world"))
    c2w = np.asarray(c2w_value, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise ValueError("Render camera_to_world must be finite 4x4")
    w2c = np.linalg.inv(c2w)
    blank = Image.new("RGB", (width, height), "black")
    return Camera(
        resolution=(width, height),
        colmap_id=uid,
        R=w2c[:3, :3].T,
        T=w2c[:3, 3],
        FoVx=float(focal2fov(float(view["fx"]), width)),
        FoVy=float(focal2fov(float(view["fy"]), height)),
        depth_params=None,
        image=blank,
        invdepthmap=None,
        image_name=str(view["image_name"]),
        uid=uid,
        data_device=data_device,
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    )


def _native_render_views(
    bound: Sequence[Mapping[str, Any]],
    probe_payload: Mapping[str, Any],
    native_cameras_json: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if all("native_camera_to_world" in entry["camera_view"] for entry in bound):
        return [dict(entry["camera_view"]) for entry in bound], {
            "mode": "prebound_native_camera_to_world",
            "source_splits": ["train", "guard"],
        }
    if native_cameras_json is None:
        raise ValueError(
            "Probe views lack native_camera_to_world; provide --native_cameras_json for train-only alignment"
        )
    from tools.geometric_repeatability.depth_reference_common import (
        apply_world_transform_to_camera_to_world,
        estimate_strict_to_native_world_transform,
        load_native_camera_entries,
    )

    train_views = [
        view
        for view in probe_payload.get("views", [])
        if str(view.get("bound_split", view.get("split", ""))).strip().casefold() == "train"
    ]
    if len(train_views) < 3:
        raise ValueError("Need at least three formal train cameras for strict/native alignment")
    native = load_native_camera_entries(native_cameras_json.resolve())
    alignment = estimate_strict_to_native_world_transform(train_views, native)
    if float(alignment["translation_error_max_m"]) > 1.0e-4 or float(
        alignment["rotation_error_max_deg"]
    ) > 5.0e-2:
        raise ValueError("Train-only strict/native camera alignment exceeds the formal integrity limits")
    transform = np.asarray(alignment["strict_to_native_transform"], dtype=np.float64)
    rendered: list[dict[str, Any]] = []
    for entry in bound:
        view = dict(entry["camera_view"])
        view["native_camera_to_world"] = apply_world_transform_to_camera_to_world(
            np.asarray(view["camera_to_world"], dtype=np.float64), transform
        ).tolist()
        rendered.append(view)
    return rendered, {
        "mode": "strict_to_native_rigid_alignment",
        "alignment_input_split": "train_only",
        "test_views_used": False,
        "alignment": alignment,
    }


def _camera_sha(view: Mapping[str, Any]) -> str:
    payload = {
        "image_name": str(view["image_name"]),
        "width": int(view["width"]),
        "height": int(view["height"]),
        "fx_fy_cx_cy": [float(view[key]) for key in ("fx", "fy", "cx", "cy")],
        "native_camera_to_world": np.asarray(view["native_camera_to_world"], dtype=np.float64).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_selection_receipt(
    *,
    scene_name: str,
    formal_split: Path,
    probe_camera_manifest: Path,
    native_cameras_json: Path | None,
    receipt_path: Path,
) -> Path:
    """Freeze four cameras before a resplat checkpoint is generated."""

    receipt_path = receipt_path.resolve()
    if receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite selection receipt: {receipt_path}")
    split_payload = _load_json(formal_split.resolve())
    probe_payload = _load_json(probe_camera_manifest.resolve())
    bound_identity = probe_payload.get("bound_split_manifest_identity")
    if isinstance(bound_identity, dict):
        expected_split_sha = _sha256_file(formal_split.resolve())
        if str(bound_identity.get("sha256", "")).lower() != expected_split_sha:
            raise ValueError("Probe-camera/formal-split hash binding mismatch")
    selected = select_fixed_formal_records(split_payload)
    selection_sha256 = _canonical_sha256(selected)
    bound = bind_selected_views(selected, probe_payload)
    render_views, camera_binding = _native_render_views(bound, probe_payload, native_cameras_json)
    frozen_views: list[dict[str, Any]] = []
    for entry, view in zip(bound, render_views):
        if str(entry["split"]) == "test":
            raise RuntimeError("Selection receipt cannot contain test views")
        frozen_views.append(
            {
                **{key: value for key, value in entry.items() if key != "camera_view"},
                "camera_view": view,
                "render_camera_sha256": _camera_sha(view),
            }
        )
    core: dict[str, Any] = {
        "protocol": SELECTION_RECEIPT_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_name": str(scene_name),
        "selection_protocol": {
            "groups": [f"{split}/{orientation}" for split, orientation in SELECTION_GROUPS],
            "order_key": [
                "strip_id",
                "block_index",
                "block_offset",
                "position_in_strip",
                "timestamp_utc",
                "pair_id",
            ],
            "ordered_group_offset_zero_based": GROUP_BLOCK_OFFSET,
            "selected_records_sha256": selection_sha256,
            "test_records_eligible": False,
            "test_views_selected": False,
            "note": (
                "Formal guard records have no formal block_offset=7; each eligible group is ordered by "
                "immutable formal fields and its zero-based item 7 is frozen. Actual block fields remain in receipt."
            ),
        },
        "inputs": {
            "formal_split": _file_identity(formal_split),
            "probe_camera_manifest": _file_identity(probe_camera_manifest),
            "native_cameras_json": _file_identity(native_cameras_json) if native_cameras_json else None,
        },
        "camera_binding": camera_binding,
        "frozen_views": frozen_views,
    }
    core["receipt_content_sha256"] = _canonical_sha256(core)
    _save_json(receipt_path, core)
    return receipt_path


def load_and_verify_selection_receipt(
    *,
    receipt_path: Path,
    scene_name: str,
    formal_split: Path,
    probe_camera_manifest: Path,
    native_cameras_json: Path | None,
) -> dict[str, Any]:
    """Verify a pre-resplat receipt without re-running view selection."""

    receipt_path = receipt_path.resolve()
    receipt = _load_json(receipt_path)
    if str(receipt.get("protocol", "")) != SELECTION_RECEIPT_PROTOCOL:
        raise ValueError("Unsupported zero-update selection receipt protocol")
    if str(receipt.get("scene_name", "")) != str(scene_name):
        raise ValueError("Selection receipt scene mismatch")
    expected_inputs = {
        "formal_split": _file_identity(formal_split),
        "probe_camera_manifest": _file_identity(probe_camera_manifest),
        "native_cameras_json": _file_identity(native_cameras_json) if native_cameras_json else None,
    }
    if receipt.get("inputs") != expected_inputs:
        raise ValueError("Selection receipt input identities no longer match")
    claimed_content_sha = str(receipt.get("receipt_content_sha256", "")).lower()
    content = dict(receipt)
    content.pop("receipt_content_sha256", None)
    if claimed_content_sha != _canonical_sha256(content):
        raise ValueError("Selection receipt content hash mismatch")
    frozen = receipt.get("frozen_views")
    if not isinstance(frozen, list) or len(frozen) != 4:
        raise ValueError("Selection receipt must contain exactly four frozen views")
    groups = [(str(row.get("split", "")), str(row.get("orientation", ""))) for row in frozen]
    if groups != list(SELECTION_GROUPS):
        raise ValueError(f"Selection receipt groups mismatch: {groups}")
    selected_for_hash: list[dict[str, Any]] = []
    for row in frozen:
        if row["split"] not in {"train", "guard"}:
            raise ValueError("Selection receipt contains a forbidden split")
        view = row.get("camera_view")
        if not isinstance(view, dict) or str(view.get("bound_split", view.get("split", ""))).lower() != row["split"]:
            raise ValueError("Selection receipt camera/split binding mismatch")
        if str(row.get("render_camera_sha256", "")).lower() != _camera_sha(view):
            raise ValueError("Selection receipt render camera hash mismatch")
        selected_for_hash.append(
            {
                key: row[key]
                for key in (
                    "selection_id",
                    "split",
                    "orientation",
                    "ordered_group_count",
                    "ordered_group_offset_zero_based",
                    "formal_record",
                    "image_stem",
                )
            }
        )
    claimed_selection_sha = str(receipt.get("selection_protocol", {}).get("selected_records_sha256", ""))
    if claimed_selection_sha != _canonical_sha256(selected_for_hash):
        raise ValueError("Selection receipt selected-record hash mismatch")
    return receipt


def _producer_identity() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = ""
    return {
        "script": _file_identity(Path(__file__)),
        "git_commit": commit,
    }


def evaluate(
    *,
    scene_name: str,
    raw_checkpoint: Path,
    resplat_checkpoint: Path,
    formal_split: Path,
    probe_camera_manifest: Path,
    native_cameras_json: Path | None,
    selection_receipt: Path,
    output_dir: Path,
    pipeline: Any,
    sh_degree: int,
    device: str,
    data_device: str,
    white_background: bool,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    # The compare phase may not re-run selection.  It consumes the immutable
    # receipt that must have been generated before the resplat checkpoint.
    receipt = load_and_verify_selection_receipt(
        receipt_path=selection_receipt,
        scene_name=scene_name,
        formal_split=formal_split,
        probe_camera_manifest=probe_camera_manifest,
        native_cameras_json=native_cameras_json,
    )
    bound = list(receipt["frozen_views"])
    render_views = [dict(entry["camera_view"]) for entry in bound]
    camera_binding = receipt["camera_binding"]
    selection_sha256 = str(receipt["selection_protocol"]["selected_records_sha256"])
    raw_identity_before = _file_identity(raw_checkpoint)
    resplat_identity_before = _file_identity(resplat_checkpoint)
    raw_model, raw_iteration = restore_checkpoint_for_render(raw_checkpoint, sh_degree=sh_degree, device=device)
    resplat_model, resplat_iteration = restore_checkpoint_for_render(
        resplat_checkpoint, sh_degree=sh_degree, device=device
    )
    output_dir.mkdir(parents=True)
    panel_dir = output_dir / "panels"
    per_view: list[dict[str, Any]] = []
    for uid, (entry, view) in enumerate(zip(bound, render_views)):
        camera = _camera_from_view(view, uid=uid, data_device=data_device)
        raw_rgb, raw_alpha, raw_tensor = render_model_view(
            camera, raw_model, pipeline, white_background=white_background
        )
        resplat_rgb, resplat_alpha, resplat_tensor = render_model_view(
            camera, resplat_model, pipeline, white_background=white_background
        )
        ssim_value, lpips_value = _appearance_metrics(raw_tensor, resplat_tensor)
        metrics = compute_pair_metrics(
            raw_rgb,
            resplat_rgb,
            raw_alpha,
            resplat_alpha,
            ssim_value=ssim_value,
            lpips_value=lpips_value,
        )
        panel_path = panel_dir / f"{uid:02d}_{entry['selection_id']}_{entry['image_stem']}.png"
        save_comparison_panel(
            panel_path,
            raw_rgb=raw_rgb,
            resplat_rgb=resplat_rgb,
            raw_alpha=raw_alpha,
            resplat_alpha=resplat_alpha,
            title=f"{entry['selection_id']} | {view['image_name']}",
        )
        per_view.append(
            {
                **{key: value for key, value in entry.items() if key != "camera_view"},
                "image_name": str(view["image_name"]),
                "render_camera_sha256": _camera_sha(view),
                "metrics": metrics,
                "panel": _file_identity(panel_path),
            }
        )

    metric_names = tuple(per_view[0]["metrics"])
    aggregate = {
        name: {
            "mean": float(np.mean([row["metrics"][name] for row in per_view])),
            "min": float(np.min([row["metrics"][name] for row in per_view])),
            "max": float(np.max([row["metrics"][name] for row in per_view])),
        }
        for name in metric_names
    }
    raw_identity_after = _file_identity(raw_checkpoint)
    resplat_identity_after = _file_identity(resplat_checkpoint)
    if raw_identity_after != raw_identity_before or resplat_identity_after != resplat_identity_before:
        raise RuntimeError("Input checkpoint identity changed during zero-update diagnostic")
    manifest = {
        "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_name": str(scene_name),
        "producer": _producer_identity(),
        "guardrails": {
            "zero_update": True,
            "optimizer_constructed": False,
            "grad_enabled_for_render": False,
            "candidate_selection": False,
            "resplatting_performed": False,
            "training_performed": False,
            "tuning_performed": False,
            "eligible_splits": ["train", "guard"],
            "test_images_loaded": False,
            "test_views_rendered": False,
        },
        "selection_protocol": {
            "groups": [f"{split}/{orientation}" for split, orientation in SELECTION_GROUPS],
            "order_key": [
                "strip_id",
                "block_index",
                "block_offset",
                "position_in_strip",
                "timestamp_utc",
                "pair_id",
            ],
            "ordered_group_offset_zero_based": GROUP_BLOCK_OFFSET,
            "selected_records_sha256_before_checkpoint_load": selection_sha256,
            "note": (
                "Formal guard records are adjacent frames and have no formal block_offset=7; "
                "the immutable protocol therefore selects item 7 after group ordering and reports "
                "the selected record's actual formal block fields."
            ),
        },
        "inputs": {
            "raw_checkpoint": raw_identity_before,
            "resplat_checkpoint": resplat_identity_before,
            "formal_split": _file_identity(formal_split),
            "probe_camera_manifest": _file_identity(probe_camera_manifest),
            "native_cameras_json": _file_identity(native_cameras_json) if native_cameras_json else None,
            "selection_receipt": _file_identity(selection_receipt),
        },
        "checkpoint_iterations": {"raw": raw_iteration, "resplat": resplat_iteration},
        "gaussian_counts": {
            "raw": int(raw_model.get_xyz.shape[0]),
            "resplat": int(resplat_model.get_xyz.shape[0]),
        },
        "alpha_proxy": {
            "semantics": "black_bg_plus_white_override_color_render_first_channel",
            "coverage_threshold_strict_greater_than": ALPHA_THRESHOLD,
        },
        "rgb_drift": {
            "comparison": "raw_checkpoint_render_vs_resplat_checkpoint_render",
            "psnr_mse_floor": PSNR_MSE_FLOOR,
            "lpips_backbone": "vgg",
        },
        "render_configuration": {
            "sh_degree": int(sh_degree),
            "device": str(device),
            "data_device": str(data_device),
            "white_background_for_rgb": bool(white_background),
            "pipeline": {
                key: bool(getattr(pipeline, key, False))
                for key in ("convert_SHs_python", "compute_cov3D_python", "debug", "antialiasing")
            },
        },
        "camera_binding": camera_binding,
        "views": per_view,
        "aggregate": aggregate,
        "all_metrics_finite": True,
        "panel_count": len(per_view),
    }
    if len(per_view) != 4:
        raise RuntimeError(f"Expected exactly four diagnostic views, got {len(per_view)}")
    manifest_path = output_dir / "manifest.json"
    _save_json(manifest_path, manifest)
    return manifest_path


def build_parser() -> tuple[argparse.ArgumentParser, Any]:
    from arguments import PipelineParams

    parser = argparse.ArgumentParser(description=__doc__)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--scene_name", required=True)
    parser.add_argument("--raw_checkpoint", default="")
    parser.add_argument("--resplat_checkpoint", default="")
    parser.add_argument("--formal_split", required=True)
    parser.add_argument("--probe_camera_manifest", required=True)
    parser.add_argument("--native_cameras_json", default="")
    parser.add_argument("--selection_receipt", required=True)
    parser.add_argument("--create_selection_receipt_only", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data_device", default="cuda")
    parser.add_argument("--white_background", action="store_true")
    return parser, pipeline_params


def main() -> None:
    parser, pipeline_params = build_parser()
    args = parser.parse_args()
    native_cameras_json = Path(args.native_cameras_json) if str(args.native_cameras_json).strip() else None
    if args.create_selection_receipt_only:
        create_selection_receipt(
            scene_name=args.scene_name,
            formal_split=Path(args.formal_split),
            probe_camera_manifest=Path(args.probe_camera_manifest),
            native_cameras_json=native_cameras_json,
            receipt_path=Path(args.selection_receipt),
        )
        return
    for flag, value in (
        ("--raw_checkpoint", args.raw_checkpoint),
        ("--resplat_checkpoint", args.resplat_checkpoint),
        ("--output_dir", args.output_dir),
    ):
        if not str(value).strip():
            parser.error(f"{flag} is required unless --create_selection_receipt_only is used")
    evaluate(
        scene_name=args.scene_name,
        raw_checkpoint=Path(args.raw_checkpoint),
        resplat_checkpoint=Path(args.resplat_checkpoint),
        formal_split=Path(args.formal_split),
        probe_camera_manifest=Path(args.probe_camera_manifest),
        native_cameras_json=native_cameras_json,
        selection_receipt=Path(args.selection_receipt),
        output_dir=Path(args.output_dir),
        pipeline=pipeline_params.extract(args),
        sh_degree=int(args.sh_degree),
        device=str(args.device),
        data_device=str(args.data_device),
        white_background=bool(args.white_background),
    )


if __name__ == "__main__":
    main()
