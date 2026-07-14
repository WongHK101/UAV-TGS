#!/usr/bin/env python3
"""Estimate a fixed scene temperature range from train frames only.

Every finite pixel from every training ``.npy`` file contributes to a
two-pass, fixed-memory histogram.  The default range uses global p0.1/p99.9
plus a 2% span margin.  Test frames are read only after the range is fixed and
are used exclusively for clipping QA; guard frames are not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_NAME = "uav_tgs_train_only_scene_temperature_range"
SCHEMA_VERSION = 1
DEFAULT_LOW_PERCENTILE = 0.1
DEFAULT_HIGH_PERCENTILE = 99.9
DEFAULT_MARGIN_FRACTION = 0.02
DEFAULT_HISTOGRAM_BINS = 65536
DEFAULT_CHUNK_PIXELS = 1_048_576


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(record: Mapping[str, Any], dotted_names: Iterable[str]) -> Any:
    for dotted_name in dotted_names:
        value: Any = record
        found = True
        for part in dotted_name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value not in (None, ""):
            return value
    return None


def _load_split_manifest(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise ValueError("split manifest must be an object containing a records list")
    return payload


def _temperature_path(
    record: Mapping[str, Any],
    *,
    manifest_dir: Path,
    npy_root: Path | None,
    npy_field: str | None,
) -> Path:
    names = (npy_field,) if npy_field else (
        "temperature_npy",
        "temperature_path",
        "npy_path",
        "output_path",
        "outputs.temperature_npy",
        "files.temperature_npy",
        "derived.temperature_npy",
    )
    value = _lookup(record, (name for name in names if name))
    if value is None:
        raise ValueError(f"record {record.get('pair_id')} has no temperature .npy path")
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (npy_root if npy_root is not None else manifest_dir) / path


def _open_array(path: Path) -> np.ndarray:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("float32"):
        raise TypeError(f"temperature array must be float32 Celsius: {path} ({array.dtype})")
    if array.ndim != 2:
        raise ValueError(f"temperature array must be two-dimensional: {path} ({array.shape})")
    if array.size == 0:
        raise ValueError(f"temperature array is empty: {path}")
    return array


def _finite_chunks(array: np.ndarray, chunk_pixels: int) -> Iterable[np.ndarray]:
    iterator = np.nditer(
        array,
        flags=("external_loop", "buffered"),
        op_flags=("readonly",),
        order="K",
        buffersize=chunk_pixels,
    )
    for values in iterator:
        chunk = np.asarray(values)
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            yield finite


def _scan_bounds(path: Path, chunk_pixels: int) -> dict[str, Any]:
    array = _open_array(path)
    valid_count = 0
    minimum = math.inf
    maximum = -math.inf
    for finite in _finite_chunks(array, chunk_pixels):
        valid_count += int(finite.size)
        minimum = min(minimum, float(np.min(finite)))
        maximum = max(maximum, float(np.max(finite)))
    if valid_count == 0:
        raise ValueError(f"temperature array contains no finite values: {path}")
    if valid_count != array.size:
        raise ValueError(f"temperature array contains NaN or infinity: {path}")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "pixel_count": int(array.size),
        "valid_count": valid_count,
        "nonfinite_count": int(array.size - valid_count),
        "minimum": minimum,
        "maximum": maximum,
    }


def _histogram_for(
    path: Path,
    *,
    minimum: float,
    maximum: float,
    bins: int,
    chunk_pixels: int,
) -> np.ndarray:
    histogram = np.zeros(bins, dtype=np.int64)
    if minimum == maximum:
        histogram[0] = _scan_bounds(path, chunk_pixels)["valid_count"]
        return histogram
    array = _open_array(path)
    for finite in _finite_chunks(array, chunk_pixels):
        counts, _ = np.histogram(finite, bins=bins, range=(minimum, maximum))
        histogram += counts
    return histogram


def _histogram_quantile(
    histogram: np.ndarray,
    *,
    minimum: float,
    maximum: float,
    percentile: float,
) -> float:
    total = int(histogram.sum())
    if total <= 0:
        raise ValueError("cannot compute a percentile from an empty histogram")
    if minimum == maximum:
        return minimum
    rank = (percentile / 100.0) * (total - 1)
    cumulative = np.cumsum(histogram, dtype=np.int64)
    index = int(np.searchsorted(cumulative, rank + 1, side="left"))
    index = min(max(index, 0), len(histogram) - 1)
    previous = 0 if index == 0 else int(cumulative[index - 1])
    count = int(histogram[index])
    width = (maximum - minimum) / len(histogram)
    if count <= 0:
        fraction = 0.5
    else:
        fraction = min(max((rank - previous + 0.5) / count, 0.0), 1.0)
    return minimum + (index + fraction) * width


def _clip_counts(
    path: Path,
    *,
    tmin: float,
    tmax: float,
    chunk_pixels: int,
) -> dict[str, int | float]:
    array = _open_array(path)
    valid_count = 0
    low_count = 0
    high_count = 0
    for finite in _finite_chunks(array, chunk_pixels):
        valid_count += int(finite.size)
        low_count += int(np.count_nonzero(finite < tmin))
        high_count += int(np.count_nonzero(finite > tmax))
    clipped_count = low_count + high_count
    return {
        "valid_count": valid_count,
        "low_count": low_count,
        "high_count": high_count,
        "clipped_count": clipped_count,
        "low_fraction": low_count / valid_count if valid_count else 0.0,
        "high_fraction": high_count / valid_count if valid_count else 0.0,
        "clipped_fraction": clipped_count / valid_count if valid_count else 0.0,
    }


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": record.get("pair_id"),
        "split": record.get("split"),
        "stratum": record.get("stratum"),
        "record_hash": record.get("hash"),
        "source_record_hash": record.get("source_record_hash"),
    }


def _aggregate_clipping(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = sum(int(row["clipping"]["valid_count"]) for row in rows)
    low = sum(int(row["clipping"]["low_count"]) for row in rows)
    high = sum(int(row["clipping"]["high_count"]) for row in rows)
    clipped = low + high
    return {
        "frame_count": len(rows),
        "valid_count": valid,
        "low_count": low,
        "high_count": high,
        "clipped_count": clipped,
        "low_fraction": low / valid if valid else 0.0,
        "high_fraction": high / valid if valid else 0.0,
        "clipped_fraction": clipped / valid if valid else 0.0,
    }


def estimate_scene_range(
    split_manifest: Mapping[str, Any],
    *,
    manifest_dir: Path,
    npy_root: Path | None = None,
    npy_field: str | None = None,
    low_percentile: float = DEFAULT_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIGH_PERCENTILE,
    margin_fraction: float = DEFAULT_MARGIN_FRACTION,
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS,
    chunk_pixels: int = DEFAULT_CHUNK_PIXELS,
) -> dict[str, Any]:
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if margin_fraction < 0.0 or histogram_bins < 2 or chunk_pixels <= 0:
        raise ValueError("margin must be non-negative; bins >= 2; chunk_pixels > 0")

    records = split_manifest["records"]
    train_records = [record for record in records if record.get("split") == "train"]
    test_records = [record for record in records if record.get("split") == "test"]
    if not train_records:
        raise ValueError("split manifest contains no training records")

    resolved: dict[str, list[tuple[Mapping[str, Any], Path]]] = {"train": [], "test": []}
    for split, subset in (("train", train_records), ("test", test_records)):
        for record in subset:
            path = _temperature_path(
                record,
                manifest_dir=manifest_dir,
                npy_root=npy_root,
                npy_field=npy_field,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            resolved[split].append((record, path))

    train_bounds: list[dict[str, Any]] = []
    global_minimum = math.inf
    global_maximum = -math.inf
    global_valid_count = 0
    for record, path in resolved["train"]:
        bounds = _scan_bounds(path, chunk_pixels)
        global_minimum = min(global_minimum, bounds["minimum"])
        global_maximum = max(global_maximum, bounds["maximum"])
        global_valid_count += bounds["valid_count"]
        train_bounds.append({"record": record, "path": path, "bounds": bounds})

    global_histogram = np.zeros(histogram_bins, dtype=np.int64)
    train_quantiles: dict[str, tuple[float, float]] = {}
    for item in train_bounds:
        histogram = _histogram_for(
            item["path"],
            minimum=global_minimum,
            maximum=global_maximum,
            bins=histogram_bins,
            chunk_pixels=chunk_pixels,
        )
        global_histogram += histogram
        train_quantiles[str(item["path"])] = (
            _histogram_quantile(
                histogram,
                minimum=global_minimum,
                maximum=global_maximum,
                percentile=low_percentile,
            ),
            _histogram_quantile(
                histogram,
                minimum=global_minimum,
                maximum=global_maximum,
                percentile=high_percentile,
            ),
        )

    global_q_low = _histogram_quantile(
        global_histogram,
        minimum=global_minimum,
        maximum=global_maximum,
        percentile=low_percentile,
    )
    global_q_high = _histogram_quantile(
        global_histogram,
        minimum=global_minimum,
        maximum=global_maximum,
        percentile=high_percentile,
    )
    q_low = min(item[0] for item in train_quantiles.values())
    q_high = max(item[1] for item in train_quantiles.values())
    if q_high <= q_low:
        raise ValueError("train temperature quantiles do not define a positive scene range")
    margin = (q_high - q_low) * margin_fraction
    tmin = q_low - margin
    tmax = q_high + margin

    per_frame: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for item in train_bounds:
        record = item["record"]
        path = item["path"]
        frame_q_low, frame_q_high = train_quantiles[str(path)]
        row = {
            **_record_identity(record),
            "temperature_npy": str(path),
            **item["bounds"],
            f"p{low_percentile:g}": frame_q_low,
            f"p{high_percentile:g}": frame_q_high,
            "clipping": _clip_counts(path, tmin=tmin, tmax=tmax, chunk_pixels=chunk_pixels),
        }
        per_frame.append(row)
        by_split["train"].append(row)

    for record, path in resolved["test"]:
        bounds = _scan_bounds(path, chunk_pixels)
        histogram = _histogram_for(
            path,
            minimum=bounds["minimum"],
            maximum=bounds["maximum"],
            bins=histogram_bins,
            chunk_pixels=chunk_pixels,
        )
        row = {
            **_record_identity(record),
            "temperature_npy": str(path),
            **bounds,
            f"p{low_percentile:g}": _histogram_quantile(
                histogram,
                minimum=bounds["minimum"],
                maximum=bounds["maximum"],
                percentile=low_percentile,
            ),
            f"p{high_percentile:g}": _histogram_quantile(
                histogram,
                minimum=bounds["minimum"],
                maximum=bounds["maximum"],
                percentile=high_percentile,
            ),
            "clipping": _clip_counts(path, tmin=tmin, tmax=tmax, chunk_pixels=chunk_pixels),
        }
        per_frame.append(row)
        by_split["test"].append(row)

    configuration = {
        "low_percentile": low_percentile,
        "high_percentile": high_percentile,
        "margin_fraction": margin_fraction,
        "histogram_bins": histogram_bins,
        "chunk_pixels": chunk_pixels,
        "estimator": "two_pass_uniform_histogram_per_frame_quantile_envelope",
        "range_rule": "min(train_frame_low_quantile), max(train_frame_high_quantile), plus span margin",
        "test_role": "qa_only_not_used_for_estimation",
        "guard_role": "not_read",
    }
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": split_manifest.get("scene"),
        "split_hash": split_manifest.get("split_hash"),
        "configuration": configuration,
        "train_estimation": {
            "frame_count": len(train_records),
            "valid_count": global_valid_count,
            "global_minimum": global_minimum,
            "global_maximum": global_maximum,
            "global_pixel_low_quantile": global_q_low,
            "global_pixel_high_quantile": global_q_high,
            f"p{low_percentile:g}": q_low,
            f"p{high_percentile:g}": q_high,
            "margin": margin,
        },
        "Tmin": tmin,
        "Tmax": tmax,
        "per_frame_quantiles": per_frame,
        "clipping_stats": {
            "train": _aggregate_clipping(by_split["train"]),
            "test": _aggregate_clipping(by_split["test"]),
        },
    }
    result["range_hash"] = _hash_json(
        {
            "scene": result["scene"],
            "split_hash": result["split_hash"],
            "configuration": configuration,
            "Tmin": tmin,
            "Tmax": tmax,
        }
    )
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--npy-root", type=Path)
    parser.add_argument("--npy-field", help="optional dotted record field containing the .npy path")
    parser.add_argument("--low-percentile", type=float, default=DEFAULT_LOW_PERCENTILE)
    parser.add_argument("--high-percentile", type=float, default=DEFAULT_HIGH_PERCENTILE)
    parser.add_argument("--margin-fraction", type=float, default=DEFAULT_MARGIN_FRACTION)
    parser.add_argument("--histogram-bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--chunk-pixels", type=int, default=DEFAULT_CHUNK_PIXELS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.resolve() == args.split_manifest.resolve():
        raise ValueError("range output must not replace the split manifest")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"range output exists; pass --overwrite: {args.output}")
    split_manifest = _load_split_manifest(args.split_manifest)
    result = estimate_scene_range(
        split_manifest,
        manifest_dir=args.split_manifest.resolve().parent,
        npy_root=None if args.npy_root is None else args.npy_root.resolve(),
        npy_field=args.npy_field,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        margin_fraction=args.margin_fraction,
        histogram_bins=args.histogram_bins,
        chunk_pixels=args.chunk_pixels,
    )
    result["source_split_manifest"] = str(args.split_manifest.resolve())
    result["source_split_manifest_sha256"] = _file_sha256(args.split_manifest)
    _write_json(args.output, result)
    print(
        f"scene={result['scene']} Tmin={result['Tmin']:.6f} "
        f"Tmax={result['Tmax']:.6f} range_hash={result['range_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
