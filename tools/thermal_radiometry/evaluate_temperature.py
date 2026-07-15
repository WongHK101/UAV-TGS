#!/usr/bin/env python3
"""Evaluate palette-inverted thermal renders against float32 Celsius maps.

The primary model-render result is evaluated only on supported pixels.  A
separate all-pixel result is retained as a diagnostic so that invalid or empty
render regions cannot silently improve the reported score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from palette_lut import palette_metadata, resolve_temperature_range, rgb_to_temperature
except ImportError:  # pragma: no cover - module execution path
    from .palette_lut import palette_metadata, resolve_temperature_range, rgb_to_temperature


SCHEMA = "uav-tgs-apparent-temperature-evaluation-v2"
METRIC_NAME = "TSDK-referenced apparent-temperature consistency"
MODEL_RENDER_METRIC_NAME = "palette-inverted TSDK-referenced apparent-temperature error"
DEFAULT_HISTOGRAM_BINS = 65536
VALID_SUBSETS = ("train", "test", "guard")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_temperature(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32:
        raise TypeError(f"Ground-truth temperature must be float32 Celsius: {path} ({values.dtype})")
    if values.ndim != 2:
        raise ValueError(f"Ground-truth temperature must be two-dimensional: {path} ({values.shape})")
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"Ground-truth temperature is empty or nonfinite: {path}")
    return values


def _load_render(path: Path) -> tuple[np.ndarray, Optional[np.ndarray]]:
    with Image.open(path) as image:
        if image.mode == "RGB":
            return np.asarray(image, dtype=np.uint8), None
        if image.mode == "RGBA":
            values = np.asarray(image, dtype=np.uint8)
            return values[..., :3], values[..., 3].astype(np.float32) / 255.0
        raise ValueError(f"Thermal render must be an RGB or RGBA image: {path} ({image.mode})")


def _load_support_values(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            if image.mode == "RGBA":
                values = np.asarray(image, dtype=np.uint8)[..., 3]
            elif image.mode in {"1", "L", "I", "F"}:
                values = np.asarray(image)
            else:
                values = np.asarray(image.convert("L"), dtype=np.uint8)
    values = np.asarray(values)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Support mask must be two-dimensional: {path} ({values.shape})")
    if values.dtype == np.bool_:
        return values.astype(np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Support mask contains nonfinite values: {path}")
    result = values.astype(np.float32)
    minimum = float(np.min(result)) if result.size else 0.0
    maximum = float(np.max(result)) if result.size else 0.0
    if minimum < 0.0:
        raise ValueError(f"Support mask contains negative values: {path}")
    if maximum > 1.0:
        if maximum <= 255.0:
            result /= 255.0
        else:
            raise ValueError(f"Support mask values must lie in [0,1] or [0,255]: {path}")
    return result


def _relative_stem(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix()


def _index_files(root: Path, suffixes: set[str]) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {resolved}")
    relative: dict[str, Path] = {}
    basenames: dict[str, list[Path]] = {}
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = _relative_stem(path, resolved)
        if key in relative:
            raise ValueError(
                f"Multiple files share the same relative stem under {resolved}: "
                f"{relative[key]} and {path}"
            )
        relative[key] = path
        basenames.setdefault(path.stem, []).append(path)
    return relative, basenames


def _resolve_indexed_file(
    relative_id: str,
    relative_index: Mapping[str, Path],
    basename_index: Mapping[str, Sequence[Path]],
) -> Optional[Path]:
    normalised = Path(str(relative_id).replace("\\", "/")).with_suffix("").as_posix()
    if normalised in relative_index:
        return relative_index[normalised]
    matches = basename_index.get(Path(normalised).name, ())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous basename in evaluation tree: {relative_id}")
    return None


def _split_records(path: Path, subset: Optional[str]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise ValueError("split manifest must be an object containing a records list")
    records = payload["records"]
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("every split manifest record must be an object")
    if subset is not None:
        records = [record for record in records if record.get("split") == subset]
    if not records:
        suffix = "" if subset is None else f" for subset {subset!r}"
        raise ValueError(f"split manifest selects no records{suffix}")
    return list(records), {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(Path(path)),
        "schema_name": payload.get("schema_name"),
        "schema_version": payload.get("schema_version"),
        "scene": payload.get("scene"),
        "split_hash": payload.get("split_hash"),
        "subset": subset,
    }


def _select_ground_truth(
    ground_truth_root: Path,
    split_manifest: Optional[Path],
    subset: Optional[str],
) -> tuple[list[tuple[Path, str, Optional[Mapping[str, Any]]]], Optional[dict[str, Any]]]:
    gt_root = Path(ground_truth_root).resolve()
    gt_relative, gt_basenames = _index_files(gt_root, {".npy"})
    if not gt_relative:
        raise FileNotFoundError(f"No float32 NPY maps found under: {gt_root}")
    if split_manifest is None:
        if subset is not None:
            raise ValueError("subset requires split_manifest")
        return [
            (path, relative_id, None)
            for relative_id, path in sorted(gt_relative.items())
        ], None

    records, provenance = _split_records(Path(split_manifest).resolve(), subset)
    selected: list[tuple[Path, str, Optional[Mapping[str, Any]]]] = []
    used_paths: set[Path] = set()
    for record in records:
        candidates: list[str] = []
        temperature_npy = record.get("temperature_npy")
        if temperature_npy not in (None, ""):
            candidate = Path(str(temperature_npy).replace("\\", "/"))
            if candidate.is_absolute():
                try:
                    candidates.append(candidate.resolve().relative_to(gt_root).with_suffix("").as_posix())
                except ValueError:
                    candidates.append(candidate.stem)
            else:
                candidates.extend((candidate.with_suffix("").as_posix(), candidate.stem))
        for key in ("pair_id", "frame_id", "id", "stem"):
            if record.get(key) not in (None, ""):
                candidates.append(str(record[key]))
        resolved_path: Optional[Path] = None
        for candidate in candidates:
            resolved_path = _resolve_indexed_file(candidate, gt_relative, gt_basenames)
            if resolved_path is not None:
                break
        if resolved_path is None:
            raise FileNotFoundError(
                f"Selected split record has no ground-truth NPY: {dict(record)}"
            )
        if resolved_path in used_paths:
            raise ValueError(f"Split manifest selects a ground-truth frame more than once: {resolved_path}")
        used_paths.add(resolved_path)
        selected.append((resolved_path, _relative_stem(resolved_path, gt_root), record))
    return selected, provenance


def discover_pairs(ground_truth_root: Path, render_root: Path) -> Sequence[Tuple[Path, Path, str]]:
    gt_root = Path(ground_truth_root).resolve()
    image_root = Path(render_root).resolve()
    if not gt_root.is_dir() or not image_root.is_dir():
        raise FileNotFoundError("ground-truth and render roots must both be directories")
    ground_truth = sorted(gt_root.rglob("*.npy"), key=lambda p: p.as_posix())
    if not ground_truth:
        raise FileNotFoundError(f"No float32 NPY maps found under: {gt_root}")
    pairs = []
    missing = []
    for gt_path in ground_truth:
        relative = gt_path.relative_to(gt_root)
        render_path = image_root / relative.with_suffix(".png")
        if not render_path.is_file():
            missing.append(relative.with_suffix(".png").as_posix())
            continue
        pairs.append((gt_path, render_path, relative.with_suffix("").as_posix()))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} rendered PNG(s), first: {preview}")
    return pairs


class _ErrorAccumulator:
    def __init__(self, histogram_max: float, bins: int):
        if bins < 2:
            raise ValueError("histogram bins must be at least 2")
        self.histogram_max = max(float(histogram_max), np.finfo(np.float64).eps)
        self.bins = int(bins)
        self.histogram = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.sum_abs = 0.0
        self.sum_squared = 0.0
        self.sum_signed = 0.0
        self.max_abs = 0.0

    def update(self, signed_error: np.ndarray) -> None:
        signed = np.asarray(signed_error, dtype=np.float64).reshape(-1)
        if signed.size == 0:
            return
        absolute = np.abs(signed)
        self.count += int(signed.size)
        self.sum_abs += float(np.sum(absolute, dtype=np.float64))
        self.sum_squared += float(np.sum(signed * signed, dtype=np.float64))
        self.sum_signed += float(np.sum(signed, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(absolute)))
        counts, _ = np.histogram(absolute, bins=self.bins, range=(0.0, self.histogram_max))
        self.histogram += counts

    def _percentile_upper_edge(self, percentile: float) -> float:
        if self.count == 0:
            return float("nan")
        if self.max_abs == 0.0:
            return 0.0
        rank = int(math.ceil((percentile / 100.0) * self.count))
        index = int(np.searchsorted(np.cumsum(self.histogram), rank, side="left"))
        index = min(max(index, 0), self.bins - 1)
        return (index + 1) * self.histogram_max / self.bins

    def summary(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "pixels": 0,
                "mae_c": None,
                "rmse_c": None,
                "signed_bias_c": None,
                "p95_abs_error_c": None,
                "max_abs_error_c": None,
            }
        return {
            "pixels": self.count,
            "mae_c": self.sum_abs / self.count,
            "rmse_c": math.sqrt(self.sum_squared / self.count),
            "signed_bias_c": self.sum_signed / self.count,
            "p95_abs_error_c": self._percentile_upper_edge(95.0),
            "max_abs_error_c": self.max_abs,
            "p95_estimator": "fixed histogram upper edge",
            "p95_histogram_bin_width_c": self.histogram_max / self.bins,
        }


def _off_lut_summary(accumulator: _ErrorAccumulator) -> Dict[str, Any]:
    summary = accumulator.summary()
    return {
        "pixels": summary["pixels"],
        "mean_rgb_distance": summary["mae_c"],
        "rms_rgb_distance": summary["rmse_c"],
        "p95_rgb_distance": summary["p95_abs_error_c"],
        "max_rgb_distance": summary["max_abs_error_c"],
        "p95_estimator": summary.get("p95_estimator"),
        "p95_histogram_bin_width": summary.get("p95_histogram_bin_width_c"),
    }


def _exact_error_summary(signed_error: np.ndarray) -> Dict[str, Any]:
    signed = np.asarray(signed_error, dtype=np.float64).reshape(-1)
    if signed.size == 0:
        return {
            "pixels": 0,
            "mae_c": None,
            "rmse_c": None,
            "signed_bias_c": None,
            "p95_abs_error_c": None,
            "max_abs_error_c": None,
        }
    absolute = np.abs(signed)
    return {
        "pixels": int(signed.size),
        "mae_c": float(np.mean(absolute)),
        "rmse_c": float(np.sqrt(np.mean(signed * signed))),
        "signed_bias_c": float(np.mean(signed)),
        "p95_abs_error_c": float(np.percentile(absolute, 95.0)),
        "max_abs_error_c": float(np.max(absolute)),
    }


def _exact_off_lut_summary(distance: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(distance, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            "pixels": 0,
            "mean_rgb_distance": None,
            "rms_rgb_distance": None,
            "p95_rgb_distance": None,
            "max_rgb_distance": None,
        }
    return {
        "pixels": int(values.size),
        "mean_rgb_distance": float(np.mean(values)),
        "rms_rgb_distance": float(np.sqrt(np.mean(values * values))),
        "p95_rgb_distance": float(np.percentile(values, 95.0)),
        "max_rgb_distance": float(np.max(values)),
    }


def _frame_macro(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = (
        "mae_c",
        "rmse_c",
        "signed_bias_c",
        "p95_abs_error_c",
        "max_abs_error_c",
    )
    output: Dict[str, Any] = {"frame_count": len(summaries)}
    for metric in metrics:
        values = np.asarray(
            [summary[metric] for summary in summaries if summary.get(metric) is not None],
            dtype=np.float64,
        )
        output[metric] = {
            "mean": None if values.size == 0 else float(np.mean(values)),
            "std": None if values.size == 0 else float(np.std(values, ddof=0)),
            "frame_count": int(values.size),
        }
    return output


def _clipping_summary(low: int, high: int, pixels: int) -> Dict[str, Any]:
    clipped = int(low) + int(high)
    return {
        "pixels": int(pixels),
        "low_pixels": int(low),
        "high_pixels": int(high),
        "clipped_pixels": clipped,
        "clipping_ratio": None if pixels == 0 else clipped / int(pixels),
    }


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else int(numerator) / int(denominator)


def _split_fields(record: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if record is None:
        return None
    keys = (
        "pair_id",
        "split",
        "stratum",
        "strip_id",
        "position_in_strip",
        "block_index",
        "hash",
    )
    return {key: record.get(key) for key in keys if key in record}


def evaluate_temperature_tree(
    ground_truth_root: Path,
    render_root: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    range_provenance: Optional[Dict[str, Any]] = None,
    chunk_pixels: int = 32768,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    split_manifest: Optional[Path] = None,
    subset: Optional[str] = None,
    mask_root: Optional[Path] = None,
    alpha_threshold: float = 0.0,
    require_support: bool = False,
) -> Dict[str, Any]:
    if not math.isfinite(tmin_c) or not math.isfinite(tmax_c) or tmax_c <= tmin_c:
        raise ValueError("temperature range must be finite and tmax_c > tmin_c")
    if subset is not None and subset not in VALID_SUBSETS:
        raise ValueError(f"subset must be one of {VALID_SUBSETS}")
    if not math.isfinite(alpha_threshold) or not 0.0 <= alpha_threshold <= 1.0:
        raise ValueError("alpha_threshold must lie in [0,1]")

    selected, split_provenance = _select_ground_truth(
        ground_truth_root,
        split_manifest,
        subset,
    )
    render_root_resolved = Path(render_root).resolve()
    render_relative, render_basenames = _index_files(render_root_resolved, {".png"})
    mask_root_resolved = None if mask_root is None else Path(mask_root).resolve()
    if mask_root_resolved is None:
        mask_relative: Mapping[str, Path] = {}
        mask_basenames: Mapping[str, Sequence[Path]] = {}
    else:
        mask_relative, mask_basenames = _index_files(
            mask_root_resolved,
            {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"},
        )

    global_min = math.inf
    global_max = -math.inf
    total_pixels = 0
    expected_clipped_low = 0
    expected_clipped_high = 0
    selected_metadata: list[dict[str, Any]] = []
    for gt_path, relative_id, split_record in selected:
        values = _load_temperature(gt_path)
        global_min = min(global_min, float(np.min(values)))
        global_max = max(global_max, float(np.max(values)))
        total_pixels += int(values.size)
        expected_clipped_low += int(np.count_nonzero(values < tmin_c))
        expected_clipped_high += int(np.count_nonzero(values > tmax_c))
        selected_metadata.append(
            {
                "ground_truth_path": gt_path,
                "relative_id": relative_id,
                "split_record": split_record,
                "shape": values.shape,
                "pixels": int(values.size),
            }
        )
    maximum_possible_error = max(
        abs(global_min - tmin_c),
        abs(global_min - tmax_c),
        abs(global_max - tmin_c),
        abs(global_max - tmax_c),
    )
    all_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    all_in_range_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    supported_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    supported_in_range_errors = _ErrorAccumulator(maximum_possible_error, histogram_bins)
    all_off_lut = _ErrorAccumulator(math.sqrt(3.0 * 255.0 * 255.0), histogram_bins)
    supported_off_lut = _ErrorAccumulator(math.sqrt(3.0 * 255.0 * 255.0), histogram_bins)
    all_clipped_low = 0
    all_clipped_high = 0
    supported_clipped_low = 0
    supported_clipped_high = 0
    render_available_pixels = 0
    supported_pixels = 0
    unsupported_pixels = 0
    missing_render_pixels = 0
    missing_mask_pixels = 0
    render_available_frames = 0
    support_available_frames = 0
    missing_render_frames = 0
    missing_mask_frames = 0
    frames_without_supported_pixels = 0
    explicit_support_frames = 0
    implicit_support_frames = 0
    explicit_support_domain_pixels = 0
    implicit_support_domain_pixels = 0
    all_frame_summaries: list[Dict[str, Any]] = []
    supported_frame_summaries: list[Dict[str, Any]] = []
    files = []
    for metadata in selected_metadata:
        gt_path = metadata["ground_truth_path"]
        relative_id = metadata["relative_id"]
        split_record = metadata["split_record"]
        ground_truth = _load_temperature(gt_path)
        render_path = _resolve_indexed_file(relative_id, render_relative, render_basenames)
        base_file = {
            "relative_id": relative_id,
            "ground_truth_path": str(gt_path.resolve()),
            "ground_truth_sha256": _sha256(gt_path),
            "pixels": int(ground_truth.size),
            "split_assignment": _split_fields(split_record),
        }
        if render_path is None:
            missing_render_frames += 1
            missing_render_pixels += int(ground_truth.size)
            files.append(
                {
                    **base_file,
                    "status": "missing_render",
                    "expected_render_path": str(
                        render_root_resolved / Path(relative_id).with_suffix(".png")
                    ),
                    "supported_pixels": 0,
                    "unsupported_pixels": 0,
                    "missing_pixels": int(ground_truth.size),
                    "support_is_explicit": False,
                }
            )
            continue

        render, render_alpha = _load_render(render_path)
        if render.shape[:2] != ground_truth.shape:
            raise ValueError(
                f"Shape mismatch for {relative_id}: GT={ground_truth.shape}, render={render.shape[:2]}"
            )
        recovered, off_distance, _ = rgb_to_temperature(
            render,
            tmin_c,
            tmax_c,
            chunk_pixels=chunk_pixels,
        )
        signed = recovered.astype(np.float64) - ground_truth.astype(np.float64)
        below = ground_truth < tmin_c
        above = ground_truth > tmax_c
        in_range = ~(below | above)
        all_errors.update(signed)
        all_in_range_errors.update(signed[in_range])
        all_off_lut.update(off_distance)
        all_clipped_low += int(np.count_nonzero(below))
        all_clipped_high += int(np.count_nonzero(above))
        render_available_frames += 1
        render_available_pixels += int(ground_truth.size)
        all_frame = _exact_error_summary(signed)
        all_frame_summaries.append(all_frame)

        mask_path: Optional[Path] = None
        if mask_root_resolved is not None:
            mask_path = _resolve_indexed_file(relative_id, mask_relative, mask_basenames)
            if mask_path is None:
                support = None
                support_source = "missing_external_mask"
                missing_mask_frames += 1
                missing_mask_pixels += int(ground_truth.size)
            else:
                support_values = _load_support_values(mask_path)
                if support_values.shape != ground_truth.shape:
                    raise ValueError(
                        f"Mask shape mismatch for {relative_id}: "
                        f"GT={ground_truth.shape}, mask={support_values.shape}"
                    )
                support = support_values > alpha_threshold
                support_source = "external_mask"
                frame_support_is_explicit = True
        elif render_alpha is not None:
            support = render_alpha > alpha_threshold
            support_source = "render_alpha"
            frame_support_is_explicit = True
        else:
            support = np.ones(ground_truth.shape, dtype=bool)
            support_source = "implicit_full_frame"
            frame_support_is_explicit = False

        if support is None:
            frame_support_is_explicit = False

        supported_frame: Optional[Dict[str, Any]] = None
        supported_off_frame: Optional[Dict[str, Any]] = None
        if support is not None:
            support_available_frames += 1
            if frame_support_is_explicit:
                explicit_support_frames += 1
                explicit_support_domain_pixels += int(ground_truth.size)
            else:
                implicit_support_frames += 1
                implicit_support_domain_pixels += int(ground_truth.size)
            frame_supported = int(np.count_nonzero(support))
            frame_unsupported = int(ground_truth.size) - frame_supported
            supported_pixels += frame_supported
            unsupported_pixels += frame_unsupported
            supported_errors.update(signed[support])
            supported_in_range_errors.update(signed[support & in_range])
            supported_off_lut.update(off_distance[support])
            supported_clipped_low += int(np.count_nonzero(below & support))
            supported_clipped_high += int(np.count_nonzero(above & support))
            supported_frame = _exact_error_summary(signed[support])
            supported_off_frame = _exact_off_lut_summary(off_distance[support])
            if frame_supported:
                supported_frame_summaries.append(supported_frame)
            else:
                frames_without_supported_pixels += 1
        else:
            frame_supported = 0
            frame_unsupported = 0

        all_off_frame = _exact_off_lut_summary(off_distance)
        files.append(
            {
                **base_file,
                "status": "complete" if support is not None else "missing_mask",
                "render_path": str(render_path.resolve()),
                "render_sha256": _sha256(render_path),
                "mask_path": None if mask_path is None else str(mask_path.resolve()),
                "mask_sha256": None if mask_path is None else _sha256(mask_path),
                "support_source": support_source,
                "support_is_explicit": frame_support_is_explicit,
                "alpha_threshold": float(alpha_threshold),
                "supported_pixels": frame_supported,
                "unsupported_pixels": frame_unsupported,
                "missing_pixels": int(ground_truth.size) if support is None else 0,
                "supported_pixel_temperature_error": supported_frame,
                "all_pixel_temperature_error_diagnostic": all_frame,
                "supported_pixel_off_lut_distance": supported_off_frame,
                "all_pixel_off_lut_distance_diagnostic": all_off_frame,
                # Legacy per-frame keys retain their all-pixel diagnostic meaning.
                "mae_c": all_frame["mae_c"],
                "rmse_c": all_frame["rmse_c"],
                "signed_bias_c": all_frame["signed_bias_c"],
                "p95_abs_error_c": all_frame["p95_abs_error_c"],
                "max_abs_error_c": all_frame["max_abs_error_c"],
                "clipped_low_pixels": int(np.count_nonzero(below)),
                "clipped_high_pixels": int(np.count_nonzero(above)),
                "off_lut_mean_rgb_distance": all_off_frame["mean_rgb_distance"],
                "off_lut_p95_rgb_distance": all_off_frame["p95_rgb_distance"],
                "off_lut_max_rgb_distance": all_off_frame["max_rgb_distance"],
            }
        )

    missing_pixels = missing_render_pixels + missing_mask_pixels
    unsupported_or_missing = unsupported_pixels + missing_pixels
    primary_metric_valid = supported_pixels > 0
    completed_with_missing = missing_pixels > 0
    support_is_explicit = (
        explicit_support_frames == len(selected_metadata)
        and len(selected_metadata) > 0
    )
    support_warnings: list[str] = []
    if implicit_support_frames:
        support_warnings.append(
            "implicit_full_frame support is a legacy compatibility fallback, "
            "not an explicit model support mask"
        )
    if missing_render_frames:
        support_warnings.append("selected frames are missing thermal renders")
    if missing_mask_frames:
        support_warnings.append("selected rendered frames are missing requested external masks")
    if not primary_metric_valid:
        support_warnings.append("primary supported-pixel metric has no evaluable pixels")

    if require_support:
        failures: list[str] = []
        if missing_render_frames:
            failures.append(f"missing renders={missing_render_frames}")
        if missing_mask_frames:
            failures.append(f"missing masks={missing_mask_frames}")
        if implicit_support_frames:
            failures.append(f"implicit full-frame support={implicit_support_frames}")
        if frames_without_supported_pixels:
            failures.append(f"frames without supported pixels={frames_without_supported_pixels}")
        if not primary_metric_valid:
            failures.append("no supported pixels")
        if failures:
            raise ValueError("Explicit support is required; " + ", ".join(failures))

    if not primary_metric_valid:
        evaluation_status = "invalid_no_supported_pixels"
    elif completed_with_missing:
        evaluation_status = "completed_with_missing"
    else:
        evaluation_status = "complete"
    all_error_summary = all_errors.summary()
    all_in_range_summary = all_in_range_errors.summary()
    supported_error_summary = supported_errors.summary()
    supported_in_range_summary = supported_in_range_errors.summary()
    all_off_lut_summary = _off_lut_summary(all_off_lut)
    supported_off_lut_summary = _off_lut_summary(supported_off_lut)
    expected_clipping = _clipping_summary(
        expected_clipped_low, expected_clipped_high, total_pixels
    )
    all_clipping = _clipping_summary(
        all_clipped_low, all_clipped_high, render_available_pixels
    )
    supported_clipping = _clipping_summary(
        supported_clipped_low, supported_clipped_high, supported_pixels
    )
    support_coverage = {
        "expected_frames": len(selected_metadata),
        "render_available_frames": render_available_frames,
        "support_available_frames": support_available_frames,
        "frames_without_supported_pixels": frames_without_supported_pixels,
        "explicit_support_frames": explicit_support_frames,
        "implicit_support_frames": implicit_support_frames,
        "missing_render_frames": missing_render_frames,
        "missing_mask_frames": missing_mask_frames,
        "expected_pixels": total_pixels,
        "render_available_pixels": render_available_pixels,
        "supported_pixels": supported_pixels,
        "unsupported_pixels": unsupported_pixels,
        "missing_render_pixels": missing_render_pixels,
        "missing_mask_pixels": missing_mask_pixels,
        "missing_pixels": missing_pixels,
        "explicit_support_domain_pixels": explicit_support_domain_pixels,
        "implicit_support_domain_pixels": implicit_support_domain_pixels,
        "unsupported_or_missing_pixels": unsupported_or_missing,
        "supported_ratio": _ratio(supported_pixels, total_pixels),
        "unsupported_ratio": _ratio(unsupported_pixels, total_pixels),
        "missing_ratio": _ratio(missing_pixels, total_pixels),
        "unsupported_or_missing_ratio": _ratio(unsupported_or_missing, total_pixels),
    }
    return {
        "schema": SCHEMA,
        "status": evaluation_status,
        "primary_metric_valid": primary_metric_valid,
        "completed_with_missing": completed_with_missing,
        "support_is_explicit": support_is_explicit,
        "warnings": support_warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": METRIC_NAME,
        "model_render_metric_name": MODEL_RENDER_METRIC_NAME,
        "claim_boundary": (
            "palette-inverted TSDK-referenced apparent-temperature error; "
            "not absolute thermometry or true surface temperature"
        ),
        "ground_truth": "float32 Celsius maps decoded by the configured TSDK protocol",
        "ground_truth_root": str(Path(ground_truth_root).resolve()),
        "render_root": str(render_root_resolved),
        "split": split_provenance,
        "support_policy": {
            "mask_root": None if mask_root_resolved is None else str(mask_root_resolved),
            "alpha_threshold": float(alpha_threshold),
            "require_support": bool(require_support),
            "support_is_explicit": support_is_explicit,
            "warnings": support_warnings,
            "precedence": "external mask, then render alpha, then implicit full frame",
            "comparison": "support_value > alpha_threshold",
            "primary_metric_domain": "supported pixels",
            "all_pixel_domain": "diagnostic only",
        },
        "temperature_range": {
            "tmin_c": float(tmin_c),
            "tmax_c": float(tmax_c),
            "source": range_provenance or {"source": "function-argument"},
        },
        "palette": palette_metadata(),
        "summary": {
            "file_count": len(files),
            "evaluated_file_count": render_available_frames,
            "pixels": total_pixels,
            "support_coverage": support_coverage,
            "pixel_micro_aggregate": {
                "supported_pixels": supported_error_summary,
                "supported_in_range_pixels": supported_in_range_summary,
                "all_pixels_diagnostic": all_error_summary,
                "all_in_range_pixels_diagnostic": all_in_range_summary,
            },
            "frame_macro_mean_std": {
                "supported_pixels": _frame_macro(supported_frame_summaries),
                "all_pixels_diagnostic": _frame_macro(all_frame_summaries),
            },
            "clipping_aggregates": {
                "reference_all_expected_pixels": expected_clipping,
                "supported_pixels": supported_clipping,
                "all_pixels_diagnostic": all_clipping,
            },
            "off_lut_distance_aggregates": {
                "supported_pixels": supported_off_lut_summary,
                "all_pixels_diagnostic": all_off_lut_summary,
            },
            "temperature_error_supported_pixels": supported_error_summary,
            "temperature_error_supported_in_range_pixels": supported_in_range_summary,
            "temperature_error_all_pixels_diagnostic": all_error_summary,
            # Legacy aggregate keys retain their all-pixel diagnostic meaning.
            "temperature_error_all_pixels": all_error_summary,
            "temperature_error_in_range_pixels": all_in_range_summary,
            "clipping": expected_clipping,
            "off_lut_distance": all_off_lut_summary,
        },
        "files": files,
    }


def write_report(path: Path, payload: Dict[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Report exists; pass --overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--tmin-c", type=float)
    parser.add_argument("--tmax-c", type=float)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="optional deterministic split manifest used to select frames",
    )
    parser.add_argument(
        "--subset",
        choices=VALID_SUBSETS,
        help="evaluate only train, test, or guard records from --split-manifest",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        help="optional mask/alpha tree mirroring render relative paths",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.0,
        help="normalised support threshold in [0,1]; supported iff value is greater",
    )
    parser.add_argument(
        "--require-support",
        action="store_true",
        help=(
            "fail unless every selected frame has a render, an explicit external-mask "
            "or RGBA-alpha support domain, and at least one supported pixel"
        ),
    )
    parser.add_argument("--chunk-pixels", type=int, default=32768)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.subset is not None and args.split_manifest is None:
        raise ValueError("--subset requires --split-manifest")
    tmin_c, tmax_c, provenance = resolve_temperature_range(
        tmin_c=args.tmin_c,
        tmax_c=args.tmax_c,
        range_manifest=args.range_manifest,
    )
    payload = evaluate_temperature_tree(
        args.ground_truth_root,
        args.render_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        range_provenance=provenance,
        chunk_pixels=args.chunk_pixels,
        histogram_bins=args.histogram_bins,
        split_manifest=args.split_manifest,
        subset=args.subset,
        mask_root=args.mask_root,
        alpha_threshold=args.alpha_threshold,
        require_support=args.require_support,
    )
    report = write_report(args.report, payload, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "primary_metric_valid": payload["primary_metric_valid"],
                "report": str(report),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
