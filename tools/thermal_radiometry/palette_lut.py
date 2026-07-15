#!/usr/bin/env python3
"""Fixed repository-owned Hot-Iron LUT and temperature conversion helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


PALETTE_NAME = "uav-tgs-hot-iron-v1"
PALETTE_SIZE = 256


def hot_iron_lut() -> np.ndarray:
    """Return the fixed 256x3 uint8 black-red-yellow-white LUT.

    This is a repository-owned canonical palette.  It is deliberately not
    described as, or assumed to be, a proprietary DJI display palette.
    """
    index = np.arange(PALETTE_SIZE, dtype=np.int16)
    red = np.minimum(255, index * 3)
    green = np.clip((index - 85) * 3, 0, 255)
    blue = np.clip((index - 170) * 3, 0, 255)
    lut = np.stack((red, green, blue), axis=1).astype(np.uint8)
    if np.unique(lut, axis=0).shape[0] != PALETTE_SIZE:
        raise RuntimeError("Canonical Hot-Iron LUT must contain 256 unique colors")
    return lut


def lut_sha256(lut: Optional[np.ndarray] = None) -> str:
    array = hot_iron_lut() if lut is None else np.asarray(lut, dtype=np.uint8)
    if array.shape != (PALETTE_SIZE, 3):
        raise ValueError(f"Expected a {PALETTE_SIZE}x3 LUT, got {array.shape}")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def palette_metadata() -> Dict[str, Any]:
    lut = hot_iron_lut()
    return {
        "name": PALETTE_NAME,
        "entries": int(lut.shape[0]),
        "unique_rgb_entries": int(np.unique(lut, axis=0).shape[0]),
        "sha256_uint8_rgb": lut_sha256(lut),
        "mapping": "linear-nearest-bin",
        "gamma": 1.0,
        "provenance": "repository-owned canonical palette; not a DJI native palette",
    }


def validate_temperature_range(tmin_c: float, tmax_c: float) -> Tuple[float, float]:
    low = float(tmin_c)
    high = float(tmax_c)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError(f"Invalid temperature range: Tmin={low!r}, Tmax={high!r}")
    return low, high


def _range_from_mapping(payload: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    mappings = [payload]
    for key in ("range", "temperature_range", "scene_range"):
        value = payload.get(key)
        if isinstance(value, dict):
            mappings.append(value)
    key_pairs = (
        ("tmin_c", "tmax_c"),
        ("Tmin", "Tmax"),
        ("tmin", "tmax"),
        ("min_c", "max_c"),
    )
    for mapping in mappings:
        for low_key, high_key in key_pairs:
            if low_key in mapping and high_key in mapping:
                return validate_temperature_range(mapping[low_key], mapping[high_key])
    return None


def resolve_temperature_range(
    *,
    tmin_c: Optional[float] = None,
    tmax_c: Optional[float] = None,
    range_manifest: Optional[Path] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Resolve a range from an explicit pair or a JSON manifest."""
    explicit = tmin_c is not None or tmax_c is not None
    if explicit and range_manifest is not None:
        raise ValueError("Use either --tmin-c/--tmax-c or --range-manifest, not both")
    if explicit:
        if tmin_c is None or tmax_c is None:
            raise ValueError("Both Tmin and Tmax are required")
        low, high = validate_temperature_range(tmin_c, tmax_c)
        return low, high, {"source": "explicit", "tmin_c": low, "tmax_c": high}
    if range_manifest is None:
        raise ValueError("A temperature range is required")
    path = Path(range_manifest).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Range manifest must contain a JSON object: {path}")
    resolved = _range_from_mapping(payload)
    if resolved is None:
        raise ValueError(f"Could not find Tmin/Tmax in range manifest: {path}")
    low, high = resolved
    return low, high, {
        "source": "manifest",
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tmin_c": low,
        "tmax_c": high,
    }


def temperature_to_indices(
    temperature_c: np.ndarray,
    tmin_c: float,
    tmax_c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    low, high = validate_temperature_range(tmin_c, tmax_c)
    values = np.asarray(temperature_c)
    if values.ndim != 2:
        raise ValueError(f"Temperature map must be two-dimensional, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Temperature map contains NaN or infinite values")
    clipped = np.clip(values.astype(np.float64, copy=False), low, high)
    normalized = (clipped - low) / (high - low)
    indices = np.rint(normalized * (PALETTE_SIZE - 1)).astype(np.uint8)
    clipping_mask = (values < low) | (values > high)
    return indices, clipping_mask


def temperature_to_rgb(
    temperature_c: np.ndarray,
    tmin_c: float,
    tmax_c: float,
    lut: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    table = hot_iron_lut() if lut is None else np.asarray(lut, dtype=np.uint8)
    if table.shape != (PALETTE_SIZE, 3):
        raise ValueError(f"Expected a {PALETTE_SIZE}x3 LUT, got {table.shape}")
    indices, clipping_mask = temperature_to_indices(temperature_c, tmin_c, tmax_c)
    return table[indices], clipping_mask


def nearest_lut_indices(
    rgb: np.ndarray,
    *,
    lut: Optional[np.ndarray] = None,
    chunk_pixels: int = 32768,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project RGB pixels to the LUT without building an HxWx256x3 tensor."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {image.shape}")
    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")
    table_u8 = hot_iron_lut() if lut is None else np.asarray(lut, dtype=np.uint8)
    if table_u8.shape != (PALETTE_SIZE, 3):
        raise ValueError(f"Expected a {PALETTE_SIZE}x3 LUT, got {table_u8.shape}")
    flat_image = image.reshape(-1, 3)
    pixels = flat_image.astype(np.float32, copy=False)
    table = table_u8.astype(np.float32)
    table_norm = np.sum(table * table, axis=1)[None, :]
    indices = np.empty(pixels.shape[0], dtype=np.uint8)
    distances = np.empty(pixels.shape[0], dtype=np.float32)
    # PNG canonical observations and many rendered pixels are already exact LUT
    # entries.  Resolve those by a compact 24-bit RGB key and reserve the
    # 256-way distance projection for genuinely off-LUT pixels.  Stable sorting
    # preserves np.argmin's first-entry tie behaviour if a custom LUT contains
    # duplicate colours.
    exact_u8_input = image.dtype == np.uint8
    table_keys = (
        (table_u8[:, 0].astype(np.uint32) << 16)
        | (table_u8[:, 1].astype(np.uint32) << 8)
        | table_u8[:, 2].astype(np.uint32)
    )
    table_key_order = np.argsort(table_keys, kind="stable")
    sorted_table_keys = table_keys[table_key_order]
    for start in range(0, pixels.shape[0], chunk_pixels):
        stop = min(start + chunk_pixels, pixels.shape[0])
        part = pixels[start:stop]
        part_indices = indices[start:stop]
        part_distances = distances[start:stop]
        exact = np.zeros(stop - start, dtype=bool)
        if exact_u8_input:
            part_u8 = flat_image[start:stop]
            part_keys = (
                (part_u8[:, 0].astype(np.uint32) << 16)
                | (part_u8[:, 1].astype(np.uint32) << 8)
                | part_u8[:, 2].astype(np.uint32)
            )
            positions = np.searchsorted(sorted_table_keys, part_keys)
            candidates = np.minimum(positions, PALETTE_SIZE - 1)
            exact = (positions < PALETTE_SIZE) & (
                sorted_table_keys[candidates] == part_keys
            )
            if np.any(exact):
                part_indices[exact] = table_key_order[candidates[exact]].astype(np.uint8)
                part_distances[exact] = 0.0
        unmatched = ~exact
        if np.any(unmatched):
            projected = part[unmatched]
            d2 = (
                np.sum(projected * projected, axis=1, keepdims=True)
                + table_norm
                - 2.0 * (projected @ table.T)
            )
            np.maximum(d2, 0.0, out=d2)
            nearest = np.argmin(d2, axis=1)
            part_indices[unmatched] = nearest.astype(np.uint8)
            part_distances[unmatched] = np.sqrt(
                d2[np.arange(projected.shape[0]), nearest]
            )
    return indices.reshape(image.shape[:2]), distances.reshape(image.shape[:2])


def indices_to_temperature(indices: np.ndarray, tmin_c: float, tmax_c: float) -> np.ndarray:
    low, high = validate_temperature_range(tmin_c, tmax_c)
    values = np.asarray(indices, dtype=np.float32)
    return (low + values * ((high - low) / float(PALETTE_SIZE - 1))).astype(np.float32)


def rgb_to_temperature(
    rgb: np.ndarray,
    tmin_c: float,
    tmax_c: float,
    *,
    lut: Optional[np.ndarray] = None,
    chunk_pixels: int = 32768,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, distances = nearest_lut_indices(rgb, lut=lut, chunk_pixels=chunk_pixels)
    return indices_to_temperature(indices, tmin_c, tmax_c), distances, indices
