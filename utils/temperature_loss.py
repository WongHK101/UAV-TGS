"""Differentiable canonical Hot-Iron inverse and temperature losses."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tools.thermal_radiometry.palette_lut import (
    PALETTE_SIZE,
    hot_iron_lut,
    lut_sha256,
    resolve_temperature_range,
)


FORMULA_VERSION = "uav-tgs-soft-hotiron-inverse-v1"
SCALAR_GRADIENT_TARGET = 0.10
SPATIAL_GRADIENT_TARGET = 0.05


def canonical_lut_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(
        hot_iron_lut().astype(np.float32) / 255.0,
        device=device,
        dtype=dtype,
    )


def adjacent_lut_tau(lut: torch.Tensor) -> float:
    """Fixed temperature rule: median positive adjacent squared LUT distance."""

    if lut.shape != (PALETTE_SIZE, 3):
        raise ValueError(f"LUT must have shape ({PALETTE_SIZE}, 3)")
    adjacent = (lut[1:] - lut[:-1]).pow(2).sum(dim=1)
    positive = adjacent[adjacent > 0]
    if positive.numel() == 0 or not bool(torch.isfinite(positive).all()):
        raise ValueError("LUT adjacent distances are invalid")
    tau = float(torch.median(positive.detach().to(torch.float64)).item())
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("soft inverse temperature must be finite and positive")
    return tau


def soft_lut_inverse(
    image: torch.Tensor,
    *,
    tau: float | None = None,
    chunk_pixels: int = 16384,
) -> torch.Tensor:
    """Map RGB to normalized apparent temperature without HxWx256x3 storage."""

    if image.ndim not in (3, 4):
        raise ValueError(f"image must be CHW or BCHW, got {tuple(image.shape)}")
    unbatched = image.ndim == 3
    value = image.unsqueeze(0) if unbatched else image
    if value.shape[1] != 3:
        raise ValueError("soft LUT inverse requires three-channel RGB")
    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")
    lut = canonical_lut_tensor(value.device, value.dtype)
    resolved_tau = adjacent_lut_tau(lut) if tau is None else float(tau)
    if not np.isfinite(resolved_tau) or resolved_tau <= 0:
        raise ValueError("tau must be finite and positive")
    lut_norm = lut.pow(2).sum(dim=1).view(1, -1)
    indices = torch.linspace(
        0.0, 1.0, PALETTE_SIZE, device=value.device, dtype=value.dtype
    )
    output_batches = []
    for batch in value:
        pixels = batch.permute(1, 2, 0).reshape(-1, 3)
        parts = []
        for start in range(0, int(pixels.shape[0]), int(chunk_pixels)):
            part = pixels[start : start + int(chunk_pixels)]
            distance2 = (
                part.pow(2).sum(dim=1, keepdim=True)
                + lut_norm
                - 2.0 * (part @ lut.transpose(0, 1))
            ).clamp_min(0.0)
            probability = torch.softmax(-distance2 / resolved_tau, dim=1)
            parts.append(probability @ indices)
        output_batches.append(torch.cat(parts).view(1, batch.shape[1], batch.shape[2]))
    result = torch.stack(output_batches, dim=0)
    return result[0] if unbatched else result


def _mask4(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference)
    value = mask.to(device=reference.device, dtype=reference.dtype)
    if value.ndim == 2:
        value = value[None, None]
    elif value.ndim == 3:
        value = value.unsqueeze(0) if value.shape[0] == 1 else value.unsqueeze(1)
    elif value.ndim != 4:
        raise ValueError(f"mask has unsupported shape: {tuple(value.shape)}")
    if value.shape[1] != 1:
        value = value[:, :1]
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value, size=reference.shape[-2:], mode="nearest")
    if value.shape[0] == 1 and reference.shape[0] > 1:
        value = value.expand(reference.shape[0], -1, -1, -1)
    if value.shape != reference.shape:
        raise ValueError(f"mask/reference shapes cannot broadcast: {value.shape} vs {reference.shape}")
    return value.clamp(0.0, 1.0)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    denominator = mask.sum().clamp_min(eps)
    return (value * mask).sum() / denominator


def _sobel(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = value.dtype, value.device
    kx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3) / 8.0
    return F.conv2d(value, kx, padding=1), F.conv2d(value, ky, padding=1)


def temperature_consistency_losses(
    image: torch.Tensor,
    target_u: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    tau: float | None = None,
    chunk_pixels: int = 16384,
    charbonnier_eps: float = 1e-3,
) -> dict[str, torch.Tensor | float]:
    """Return scalar and Sobel-gradient Charbonnier losses in normalized units."""

    predicted = soft_lut_inverse(image, tau=tau, chunk_pixels=chunk_pixels)
    pred4 = predicted.unsqueeze(0) if predicted.ndim == 3 else predicted
    target = target_u.to(device=pred4.device, dtype=pred4.dtype)
    if target.ndim == 2:
        target = target[None, None]
    elif target.ndim == 3:
        target = target.unsqueeze(0) if target.shape[0] == 1 else target.unsqueeze(1)
    if target.shape[-2:] != pred4.shape[-2:]:
        target = F.interpolate(target, size=pred4.shape[-2:], mode="bilinear", align_corners=False)
    if target.shape != pred4.shape:
        raise ValueError(f"target/prediction shapes differ: {target.shape} vs {pred4.shape}")
    valid = _mask4(mask, pred4)
    eps2 = float(charbonnier_eps) ** 2
    scalar = _masked_mean(torch.sqrt((pred4 - target).pow(2) + eps2), valid, 1e-12)
    pred_x, pred_y = _sobel(pred4)
    target_x, target_y = _sobel(target)
    gradient_error = 0.5 * (
        torch.sqrt((pred_x - target_x).pow(2) + eps2)
        + torch.sqrt((pred_y - target_y).pow(2) + eps2)
    )
    gradient = _masked_mean(gradient_error, valid, 1e-12)
    resolved_tau = adjacent_lut_tau(canonical_lut_tensor(image.device, image.dtype)) if tau is None else float(tau)
    return {
        "scalar": scalar,
        "gradient": gradient,
        "u_pred": predicted,
        "tau": resolved_tau,
        "valid_fraction": float(valid.detach().mean().item()),
    }


class TemperatureTargetStore:
    """Lazily cache per-camera normalized temperature and support on CPU."""

    def __init__(
        self,
        temperature_root: str | Path,
        range_manifest: str | Path,
        support_root: str | Path | None = None,
        max_cache_items: int = 1024,
    ) -> None:
        self.temperature_root = Path(temperature_root).resolve()
        self.support_root = None if support_root is None else Path(support_root).resolve()
        self.range_manifest = Path(range_manifest).resolve()
        self.tmin_c, self.tmax_c, self.range_metadata = resolve_temperature_range(
            range_manifest=self.range_manifest
        )
        self.max_cache_items = int(max_cache_items)
        if self.max_cache_items <= 0:
            raise ValueError("max_cache_items must be positive")
        self._cache: OrderedDict[tuple[str, int, int], tuple[torch.Tensor, torch.Tensor]] = OrderedDict()

    @staticmethod
    def _stem(image_name: str) -> str:
        return Path(str(image_name)).stem

    def _path(self, root: Path, image_name: str) -> Path:
        direct = root / f"{self._stem(image_name)}.npy"
        if direct.is_file():
            return direct
        matches = list(root.glob(f"*/{self._stem(image_name)}.npy"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one NPY for {image_name!r} under {root}, found {len(matches)}")
        return matches[0]

    def get(
        self,
        image_name: str,
        height: int,
        width: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (self._stem(image_name), int(height), int(width))
        cached = self._cache.get(key)
        if cached is None:
            temperature = np.load(self._path(self.temperature_root, image_name), allow_pickle=False)
            if temperature.ndim != 2 or not np.all(np.isfinite(temperature)):
                raise ValueError(f"invalid temperature map for {image_name!r}")
            target = np.clip(
                (temperature.astype(np.float32) - self.tmin_c) / (self.tmax_c - self.tmin_c),
                0.0,
                1.0,
            )
            target_tensor = torch.from_numpy(target)[None, None]
            target_tensor = F.interpolate(
                target_tensor, size=(int(height), int(width)), mode="bilinear", align_corners=False
            )[0].contiguous()
            if self.support_root is None:
                support_tensor = torch.ones((1, int(height), int(width)), dtype=torch.float32)
            else:
                support = np.load(self._path(self.support_root, image_name), allow_pickle=False)
                if support.dtype != np.bool_ or support.ndim != 2:
                    raise ValueError(f"support mask must be boolean 2D for {image_name!r}")
                support_tensor = F.interpolate(
                    torch.from_numpy(support.astype(np.float32))[None, None],
                    size=(int(height), int(width)),
                    mode="nearest",
                )[0].contiguous()
            cached = (target_tensor, support_tensor)
            self._cache[key] = cached
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return cached[0].to(device=device, non_blocking=True), cached[1].to(device=device, non_blocking=True)

    def metadata(self) -> dict[str, Any]:
        return {
            "temperature_root": str(self.temperature_root),
            "support_root": None if self.support_root is None else str(self.support_root),
            "range_manifest": str(self.range_manifest),
            "range_manifest_sha256": hashlib.sha256(self.range_manifest.read_bytes()).hexdigest(),
            "tmin_c": self.tmin_c,
            "tmax_c": self.tmax_c,
            "lut_sha256": lut_sha256(),
            "formula_version": FORMULA_VERSION,
        }


def load_calibration_manifest(path: str | Path, tau: float) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "uav-tgs-temperature-loss-calibration-v1":
        raise ValueError("temperature calibration manifest schema mismatch")
    if payload.get("formula_version") != FORMULA_VERSION:
        raise ValueError("temperature calibration formula version mismatch")
    if payload.get("lut_sha256") != lut_sha256():
        raise ValueError("temperature calibration LUT hash mismatch")
    if not math_isclose(float(payload.get("tau")), float(tau)):
        raise ValueError("temperature calibration tau mismatch")
    for name in ("lambda_temp", "lambda_grad"):
        value = float(payload.get(name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"invalid calibrated {name}")
    return {**payload, "manifest_path": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-12, 1e-7 * max(abs(left), abs(right)))
