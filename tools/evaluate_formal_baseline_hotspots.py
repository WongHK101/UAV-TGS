#!/usr/bin/env python3
"""Evaluate formal palette-rendered baselines with one frozen hotspot threshold.

This is an evaluation-only sidecar for Legacy-L/F3/SCSP-style RGB thermal
renders.  It validates the complete radiometry chain before opening any test
artifact, then compares nearest-fixed-LUT (8-bit display-equivalent)
temperatures with direct float32 TSDK targets on the formal evaluation support.
The hotspot threshold is accepted only from the immutable train-only q95/65536
receipt produced by ``tools/oct_gs_formal.py freeze-hotspot-threshold``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oct_gs.formal import sha256_file, sha256_json
from oct_gs.protocol import load_oct_protocol_manifest
from oct_gs.radiance import METHOD_SEMANTICS, TARGET_SEMANTICS
from tools.thermal_radiometry.palette_lut import (
    PALETTE_NAME,
    lut_sha256,
    rgb_to_temperature,
)


REPORT_SCHEMA = "uav-tgs-formal-baseline-hotspot-evaluation-v1"
THRESHOLD_SCHEMA = "uav-tgs-oct-train-only-hotspot-threshold-v1"
CANONICAL_SCHEMA = "uav-tgs-canonical-hot-iron-v1"
OPTIMIZATION_SUPPORT_SCHEMA = "uav-tgs-undistorted-temperature-v1"
EVALUATION_SUPPORT_SCHEMA = "uav-tgs-formal-temperature-support"
FORMAL_QUANTILE = 0.95
FORMAL_THRESHOLD_BINS = 65_536
HOTSPOT_EVALUATION_BINS = 4_096
FORMAL_SPLITS = frozenset({"train", "guard", "test"})
EVALUATION_SUPPORT_POLICY = {
    "expression": "valid_support AND (opacity_proxy > opacity_threshold)",
    "opacity_threshold": 0.01,
    "comparison": "strict_greater_than",
    "opacity_proxy_semantics": "black_bg_plus_white_override_color_render",
    "threshold_applied_only_by_this_combiner": True,
}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _identity(record: Mapping[str, Any]) -> str:
    for key in ("pair_id", "relative_id", "image_name", "thermal_camera_name", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.replace("\\", "/")).stem
    raise ValueError("formal record has no usable pair identity")


def _indexed(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} records must be objects")
        key = _identity(row)
        if key in result:
            raise ValueError(f"{label} has duplicate identity {key!r}")
        result[key] = row
    if not result:
        raise ValueError(f"{label} is empty")
    return result


def _require_sha(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return text


def _close(left: Any, right: Any) -> bool:
    try:
        left_f, right_f = float(left), float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(left_f) and math.isfinite(right_f) and abs(left_f - right_f) <= max(
        1e-7, 1e-7 * max(abs(left_f), abs(right_f))
    )


def _support_relative(record: Mapping[str, Any]) -> str:
    nested = record.get("valid_support")
    if isinstance(nested, Mapping):
        value = nested.get("relative_path", nested.get("path"))
        if isinstance(value, str) and value.strip():
            return value
    for key in ("support_npy", "relative_path", "mask_path", "path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"support record {_identity(record)!r} has no path")


def _support_sha(record: Mapping[str, Any]) -> str:
    nested = record.get("valid_support")
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return _require_sha(nested["sha256"], f"support {_identity(record)}")
    for key in ("support_sha256", "mask_sha256", "sha256"):
        if record.get(key):
            return _require_sha(record[key], f"support {_identity(record)}")
    raise ValueError(f"support record {_identity(record)!r} has no SHA-256")


def _evaluation_support_record(record: Mapping[str, Any]) -> tuple[str, str]:
    outputs = record.get("outputs")
    binary = outputs.get("bool") if isinstance(outputs, Mapping) else None
    if not isinstance(binary, Mapping) or binary.get("dtype") != "bool":
        raise ValueError(f"evaluation support {_identity(record)!r} lacks bool output")
    relative = binary.get("relative_path")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"evaluation support {_identity(record)!r} lacks bool path")
    if Path(relative.replace("\\", "/")).parts[:1] != ("bool",):
        raise ValueError("evaluation support bool output must live under bool/")
    return relative, _require_sha(binary.get("sha256"), f"evaluation support {_identity(record)}")


def _resolve_under(root: Path, relative: str, pair_id: str) -> Path:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    portable = str(relative).replace("\\", "/")
    pure = PurePosixPath(portable)
    if (
        not portable
        or pure.is_absolute()
        or ".." in pure.parts
        or re.match(r"^[A-Za-z]:", portable) is not None
    ):
        raise ValueError(f"unsafe relative artifact path for {pair_id!r}: {relative!r}")
    candidate = Path(*pure.parts)
    direct = (root / candidate).resolve()
    try:
        direct.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes its declared root: {relative!r}") from error
    if direct.is_file():
        return direct
    matches = []
    for path in root.rglob(candidate.name):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            matches.append(resolved)
    matches.sort()
    if len(matches) == 1:
        return matches[0]
    stem_matches = []
    for path in root.rglob(f"{pair_id}.*"):
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            stem_matches.append(resolved)
    stem_matches.sort()
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError(
        f"cannot uniquely resolve {relative!r} for {pair_id!r} under {root}"
    )


def _index_pngs(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    relative: dict[str, Path] = {}
    basenames: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.suffix.casefold() != ".png":
            continue
        key = path.relative_to(root).with_suffix("").as_posix()
        if key in relative:
            raise ValueError(f"duplicate relative render identity under {root}: {key}")
        relative[key] = path.resolve()
        basenames.setdefault(path.stem, []).append(path.resolve())
    if not relative:
        raise FileNotFoundError(f"no PNG renders under {root}")
    return relative, basenames


def _resolve_render(
    root: Path,
    pair_id: str,
    camera_name: str,
    relative: Mapping[str, Path],
    basenames: Mapping[str, Sequence[Path]],
) -> Path:
    candidates = [
        Path(camera_name.replace("\\", "/")).with_suffix("").as_posix(),
        pair_id,
    ]
    for candidate in candidates:
        if candidate in relative:
            return relative[candidate]
        matches = basenames.get(Path(candidate).name, ())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"ambiguous render basename {candidate!r} under {root}")
    raise FileNotFoundError(f"missing formal test render for {pair_id!r} under {root}")


def _load_float_temperature(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype("float32") or value.ndim != 2 or value.size == 0:
        raise TypeError(f"direct TSDK target must be non-empty float32 HW: {path}")
    if not bool(np.isfinite(value).all()):
        raise ValueError(f"direct TSDK target contains NaN/Inf: {path}")
    return value


def _load_bool_support(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.dtype("bool") or value.ndim != 2 or value.size == 0:
        raise TypeError(f"formal evaluation support must be non-empty bool HW: {path}")
    return value


def _load_rgb_render(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError(f"thermal render must be RGB/RGBA PNG: {path} ({image.mode})")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _support_index(
    pair_ids: Sequence[str], support_by_pair: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        row = support_by_pair[pair_id]
        relative = _support_relative(row)
        encoding = (
            "bool-npy"
            if Path(relative.replace("\\", "/")).suffix.casefold() == ".npy"
            else "binary-image-0-255"
        )
        records.append(
            {"pair_id": pair_id, "sha256": _support_sha(row), "encoding": encoding}
        )
    return records


def _hotspot_source_receipt(
    *,
    scene_name: str,
    split_sha256: str,
    split_hash: str,
    decode_manifest_sha256: str,
    decode_protocol_sha256: str,
    range_sha256: str,
    range_hash: str,
    support_sha256: str,
    support_index_sha256: str,
    train_camera_names: Sequence[str],
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema": "uav-tgs-oct-hotspot-source-receipt-v1",
        "scene_name": scene_name,
        "bound_split_sha256": split_sha256,
        "split_hash": split_hash,
        "decode_manifest_sha256": decode_manifest_sha256,
        "decode_protocol_sha256": decode_protocol_sha256,
        "range_manifest_sha256": range_sha256,
        "range_hash": range_hash,
        "optimization_support_manifest_sha256": support_sha256,
        "optimization_support_index_sha256": support_index_sha256,
        "train_view_ids_sha256": sha256_json(sorted(train_camera_names)),
    }
    receipt["receipt_sha256"] = sha256_json(receipt)
    return receipt


def _validate_threshold_manifest(
    path: Path,
    *,
    scene_name: str,
    source_receipt: Mapping[str, Any],
    train_camera_names: Sequence[str],
    tmin_c: float,
    tmax_c: float,
) -> tuple[dict[str, Any], str, float]:
    threshold = _load_json(path, "hotspot threshold")
    supplied_hash = threshold.get("threshold_sha256")
    basis = dict(threshold)
    basis.pop("threshold_sha256", None)
    if threshold.get("schema") != THRESHOLD_SCHEMA or supplied_hash != sha256_json(basis):
        raise ValueError("hotspot threshold manifest/hash mismatch")
    if threshold.get("scene_name") != scene_name or threshold.get("source_receipt") != source_receipt:
        raise ValueError("hotspot threshold radiometry/split receipt mismatch")
    if threshold.get("source_split") != "train" or threshold.get("test_statistics_used") is not False:
        raise ValueError("hotspot threshold is not train-only")
    if float(threshold.get("quantile", math.nan)) != FORMAL_QUANTILE:
        raise ValueError(f"formal hotspot quantile must equal {FORMAL_QUANTILE}")
    if type(threshold.get("histogram_bins")) is not int or threshold["histogram_bins"] != FORMAL_THRESHOLD_BINS:
        raise ValueError(f"formal hotspot bins must equal {FORMAL_THRESHOLD_BINS}")
    if threshold.get("train_view_ids_sha256") != sha256_json(sorted(train_camera_names)):
        raise ValueError("hotspot threshold train membership mismatch")
    valid_train_pixels = threshold.get("valid_train_pixels")
    if type(valid_train_pixels) is not int or valid_train_pixels <= 0:
        raise ValueError("hotspot threshold valid_train_pixels must be positive")
    threshold_c = float(threshold.get("threshold_c", math.nan))
    if not math.isfinite(threshold_c) or not tmin_c <= threshold_c <= tmax_c:
        raise ValueError("hotspot threshold is non-finite/outside scene range")
    if threshold.get("range_c") != [tmin_c, tmax_c]:
        raise ValueError("hotspot threshold scene range mismatch")
    return threshold, str(supplied_hash), threshold_c


@dataclass(frozen=True)
class _FormalInputs:
    scene_name: str
    tmin_c: float
    tmax_c: float
    threshold_c: float
    split_records: tuple[Mapping[str, Any], ...]
    canonical_by_pair: Mapping[str, Mapping[str, Any]]
    evaluation_support_by_pair: Mapping[str, Mapping[str, Any]]
    threshold: Mapping[str, Any]
    provenance: Mapping[str, Any]
    formal_binding: Mapping[str, Any]
    pair_parameter_index_sha256: str
    temperature_error_histogram_max_c: float


def _validate_formal_inputs(args: argparse.Namespace) -> _FormalInputs:
    """Validate every train-only/radiometry gate before test data is opened."""

    paths = {
        "formal_protocol_manifest": Path(args.formal_protocol_manifest).resolve(),
        "bound_split": Path(args.bound_split).resolve(),
        "decode_manifest": Path(args.decode_manifest).resolve(),
        "decode_protocol": Path(args.decode_protocol).resolve(),
        "range_manifest": Path(args.range_manifest).resolve(),
        "canonical_manifest": Path(args.canonical_manifest).resolve(),
        "optimization_support_manifest": Path(args.optimization_support_manifest).resolve(),
        "hotspot_threshold_manifest": Path(args.hotspot_threshold_manifest).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    oct_protocol = load_oct_protocol_manifest(paths["formal_protocol_manifest"])
    formal_binding = oct_protocol.get("formal_binding")
    if not isinstance(formal_binding, Mapping):
        raise ValueError("formal OCT protocol lacks its immutable binding")

    split = _load_json(paths["bound_split"], "bound split")
    split_records_raw = split.get("records")
    if not isinstance(split_records_raw, list) or not split_records_raw:
        raise ValueError("bound split has no records")
    split_records = tuple(split_records_raw)
    split_by_pair = _indexed(split_records, "bound split")
    pair_ids = list(split_by_pair)
    split_values = [str(row.get("split", "")) for row in split_records]
    if set(split_values) != FORMAL_SPLITS:
        raise ValueError("bound split must contain train/guard/test and no other labels")
    counts = {
        "total": len(split_records),
        **{name: split_values.count(name) for name in sorted(FORMAL_SPLITS)},
    }
    if split.get("counts") != counts:
        raise ValueError("bound split counts mismatch")
    scene_name = str(split.get("scene", ""))
    if not scene_name or scene_name != str(args.scene_name):
        raise ValueError("bound split/CLI scene mismatch")
    if formal_binding.get("scene_name") != scene_name:
        raise ValueError("formal protocol/split scene mismatch")
    split_hash = _require_sha(split.get("split_hash"), "bound split hash")
    split_sha = sha256_file(paths["bound_split"])
    formal_split = formal_binding.get("bound_split")
    if (
        not isinstance(formal_split, Mapping)
        or formal_split.get("sha256") != split_sha
        or formal_split.get("split_hash") != split_hash
        or formal_split.get("counts") != counts
    ):
        raise ValueError("formal protocol does not bind this exact split")
    camera_by_pair: dict[str, str] = {}
    for pair_id, row in split_by_pair.items():
        camera_name = str(row.get("thermal_camera_name", ""))
        if not camera_name:
            raise ValueError(f"bound split lacks thermal_camera_name for {pair_id}")
        camera_by_pair[pair_id] = camera_name
    if len(set(camera_by_pair.values())) != len(camera_by_pair):
        raise ValueError("bound split thermal camera names are not unique")

    decode_binding = split.get("decode_binding")
    if not isinstance(decode_binding, Mapping):
        raise ValueError("bound split lacks formal decode binding")
    if decode_binding.get("adapter_backend") != "official-dji-irp":
        raise ValueError("formal target must use official-dji-irp")
    for key in ("verified_decode_requests", "verified_raw_rjpeg_hashes", "verified_temperature_file_hashes"):
        if decode_binding.get(key) is not True:
            raise ValueError(f"bound split decode binding does not prove {key}")
    decode_manifest_sha = sha256_file(paths["decode_manifest"])
    decode_protocol_sha = sha256_file(paths["decode_protocol"])
    if decode_binding.get("decode_manifest_sha256") != decode_manifest_sha:
        raise ValueError("decode manifest differs from bound split")
    if decode_binding.get("decode_protocol_sha256") != decode_protocol_sha:
        raise ValueError("decode protocol differs from bound split")
    formal_tsdk = formal_binding.get("tsdk_target")
    if (
        not isinstance(formal_tsdk, Mapping)
        or formal_tsdk.get("decode_manifest_sha256") != decode_manifest_sha
        or formal_tsdk.get("decode_protocol_sha256") != decode_protocol_sha
    ):
        raise ValueError("formal protocol does not bind this exact TSDK decode")
    protocol_hash = _require_sha(decode_binding.get("protocol_hash"), "decode protocol hash")
    decode_by_pair = _indexed(_load_jsonl(paths["decode_manifest"], "decode manifest"), "decode manifest")
    protocol_by_pair = _indexed(_load_jsonl(paths["decode_protocol"], "decode protocol"), "decode protocol")
    if set(decode_by_pair) != set(pair_ids) or set(protocol_by_pair) != set(pair_ids):
        raise ValueError("decode/protocol identities differ from bound split")
    parameter_keys = {
        "distance_m",
        "humidity_percent",
        "emissivity",
        "ambient_c",
        "reflected_c",
    }
    tsdk_index: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        decode_row = decode_by_pair[pair_id]
        protocol_row = protocol_by_pair[pair_id]
        if (
            decode_row.get("scene") != scene_name
            or decode_row.get("success") is not True
            or decode_row.get("dtype") != "float32"
        ):
            raise ValueError(f"decode record is not successful float32 for {pair_id}")
        if (
            protocol_row.get("scene") != scene_name
            or protocol_row.get("protocol_hash") != protocol_hash
            or protocol_row.get("schema_version") != "uav-tgs.radiometry-protocol.v1"
        ):
            raise ValueError(f"decode protocol hash mismatch for {pair_id}")
        parameters = protocol_row.get("decode_parameters")
        if not isinstance(parameters, Mapping) or set(parameters) != parameter_keys:
            raise ValueError(f"decode parameters are incomplete for {pair_id}")
        normalized_parameters: dict[str, dict[str, Any]] = {}
        for name in sorted(parameter_keys):
            entry = parameters[name]
            if not isinstance(entry, Mapping) or not isinstance(entry.get("source"), str):
                raise ValueError(f"decode parameter {name} lacks value/source for {pair_id}")
            value = float(entry.get("value"))
            if not math.isfinite(value):
                raise ValueError(f"decode parameter {name} is non-finite for {pair_id}")
            normalized_parameters[name] = {"value": value, "source": entry["source"]}
        tsdk_index.append(
            {
                "pair_id": pair_id,
                "output_sha256": _require_sha(
                    decode_row.get("output_sha256"), f"decode target {pair_id}"
                ),
                "parameters_sha256": sha256_json(normalized_parameters),
            }
        )
    pair_parameter_index_sha = sha256_json(tsdk_index)
    if (
        formal_tsdk.get("pair_parameter_index_sha256") != pair_parameter_index_sha
        or formal_tsdk.get("protocol_hash") != protocol_hash
        or formal_tsdk.get("target_semantics") != TARGET_SEMANTICS
        or formal_tsdk.get("method_semantics") != METHOD_SEMANTICS
        or formal_tsdk.get("adapter_backend") != "official-dji-irp"
        or formal_tsdk.get("absolute_thermometry_claimed") is not False
    ):
        raise ValueError("formal protocol TSDK semantics/parameter index mismatch")

    range_payload = _load_json(paths["range_manifest"], "range manifest")
    range_sha = sha256_file(paths["range_manifest"])
    if str(range_payload.get("scene", "")) != scene_name:
        raise ValueError("range manifest scene mismatch")
    if range_payload.get("source_split_manifest_sha256") != split_sha:
        raise ValueError("range manifest is not bound to the exact formal split")
    if range_payload.get("split_hash") != split_hash:
        raise ValueError("range/formal split hash mismatch")
    range_hash = _require_sha(range_payload.get("range_hash"), "range hash")
    configuration = range_payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("range manifest lacks configuration")
    if configuration.get("guard_role") != "not_read":
        raise ValueError("formal range must not read guard")
    if configuration.get("test_role") != "qa_only_not_used_for_estimation":
        raise ValueError("formal range may use test only for post-estimation QA")
    tmin_c, tmax_c = float(range_payload.get("Tmin")), float(range_payload.get("Tmax"))
    if not math.isfinite(tmin_c) or not math.isfinite(tmax_c) or tmax_c <= tmin_c:
        raise ValueError("formal temperature range is invalid")
    range_basis = {
        "scene": range_payload.get("scene"),
        "split_hash": range_payload.get("split_hash"),
        "configuration": range_payload.get("configuration"),
        "Tmin": range_payload.get("Tmin"),
        "Tmax": range_payload.get("Tmax"),
    }
    if range_hash != sha256_json(range_basis):
        raise ValueError("formal range logical hash mismatch")
    formal_range = formal_binding.get("temperature_range")
    if (
        not isinstance(formal_range, Mapping)
        or formal_range.get("sha256") != range_sha
        or formal_range.get("range_hash") != range_hash
        or not _close(formal_range.get("tmin_c"), tmin_c)
        or not _close(formal_range.get("tmax_c"), tmax_c)
    ):
        raise ValueError("formal protocol does not bind this exact temperature range")
    quantiles = range_payload.get("per_frame_quantiles")
    if not isinstance(quantiles, list):
        raise ValueError("formal range lacks per-frame QA records")
    test_extrema = [
        (float(row.get("minimum")), float(row.get("maximum")))
        for row in quantiles
        if isinstance(row, Mapping) and row.get("split") == "test"
    ]
    if len(test_extrema) != counts["test"] or not all(
        math.isfinite(low) and math.isfinite(high) and high >= low
        for low, high in test_extrema
    ):
        raise ValueError("formal range test QA extrema coverage is incomplete")
    observed_low = min(low for low, _ in test_extrema)
    observed_high = max(high for _, high in test_extrema)
    temperature_error_histogram_max_c = max(
        abs(observed_low - tmin_c),
        abs(observed_low - tmax_c),
        abs(observed_high - tmin_c),
        abs(observed_high - tmax_c),
        np.finfo(np.float64).eps,
    )

    canonical = _load_json(paths["canonical_manifest"], "canonical manifest")
    canonical_sha = sha256_file(paths["canonical_manifest"])
    if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("status") != "complete":
        raise ValueError("canonical Hot-Iron manifest is incomplete/unsupported")
    palette = canonical.get("palette")
    if not isinstance(palette, Mapping):
        raise ValueError("canonical manifest lacks palette metadata")
    if palette.get("name") != PALETTE_NAME or palette.get("sha256_uint8_rgb") != lut_sha256():
        raise ValueError("canonical manifest does not bind the fixed Hot-Iron LUT")
    encoding = canonical.get("image_encoding")
    if encoding != {"format": "PNG", "mode": "RGB", "lossless": True, "gamma": 1.0}:
        raise ValueError("canonical Hot-Iron encoding mismatch")
    canonical_range = canonical.get("temperature_range")
    if not isinstance(canonical_range, Mapping):
        raise ValueError("canonical manifest lacks temperature range")
    if not _close(canonical_range.get("tmin_c"), tmin_c) or not _close(
        canonical_range.get("tmax_c"), tmax_c
    ):
        raise ValueError("canonical/range temperatures differ")
    range_source = canonical_range.get("source")
    if not isinstance(range_source, Mapping) or range_source.get("sha256") != range_sha:
        raise ValueError("canonical manifest is not bound to the exact range manifest")
    canonical_rows = canonical.get("files")
    if not isinstance(canonical_rows, list):
        raise ValueError("canonical manifest lacks files")
    canonical_by_pair = _indexed(canonical_rows, "canonical manifest")
    if set(canonical_by_pair) != set(pair_ids):
        raise ValueError("canonical identities differ from bound split")
    formal_canonical = formal_binding.get("canonical_target")
    if (
        not isinstance(formal_canonical, Mapping)
        or formal_canonical.get("manifest_sha256") != canonical_sha
        or formal_canonical.get("lut_sha256") != lut_sha256()
    ):
        raise ValueError("formal protocol does not bind this canonical LUT/manifest")

    support = _load_json(paths["optimization_support_manifest"], "optimization support")
    support_sha = sha256_file(paths["optimization_support_manifest"])
    if support.get("schema") != OPTIMIZATION_SUPPORT_SCHEMA or support.get("status") != "complete":
        raise ValueError("optimization support manifest is incomplete/unsupported")
    support_rows = support.get("files")
    if not isinstance(support_rows, list):
        raise ValueError("optimization support manifest lacks files")
    support_by_pair = _indexed(support_rows, "optimization support")
    if set(support_by_pair) != set(pair_ids):
        raise ValueError("optimization support identities differ from bound split")
    support_index = _support_index(pair_ids, support_by_pair)
    support_index_sha = sha256_json(support_index)
    formal_support = formal_binding.get("support")
    formal_optimization = (
        formal_support.get("optimization") if isinstance(formal_support, Mapping) else None
    )
    if (
        not isinstance(formal_optimization, Mapping)
        or formal_optimization.get("manifest_sha256") != support_sha
        or formal_optimization.get("support_index_sha256") != support_index_sha
    ):
        raise ValueError("formal protocol does not bind this optimization support")
    for pair_id in pair_ids:
        canonical_row = canonical_by_pair[pair_id]
        support_row = support_by_pair[pair_id]
        raw_target = support_row.get("input_temperature")
        undistorted_target = support_row.get("output_temperature")
        if not isinstance(raw_target, Mapping) or not isinstance(undistorted_target, Mapping):
            raise ValueError(f"support target provenance is incomplete for {pair_id}")
        if raw_target.get("dtype") != "float32" or undistorted_target.get("dtype") != "float32":
            raise TypeError(f"support target provenance is not float32 for {pair_id}")
        if _require_sha(raw_target.get("sha256"), f"raw target {pair_id}") != _require_sha(
            decode_by_pair[pair_id].get("output_sha256"), f"decode target {pair_id}"
        ):
            raise ValueError(f"support/decode target SHA mismatch for {pair_id}")
        if canonical_row.get("temperature_dtype") != "float32":
            raise TypeError(f"canonical target is not float32 for {pair_id}")
        if _require_sha(canonical_row.get("input_sha256"), f"canonical target {pair_id}") != _require_sha(
            undistorted_target.get("sha256"), f"undistorted target {pair_id}"
        ):
            raise ValueError(f"canonical/undistorted target SHA mismatch for {pair_id}")

    # The threshold receipt is fully validated before even stat/read access to
    # the test-only evaluation-support manifest or any test artifact root.
    train_camera_names = sorted(
        camera_by_pair[pair_id]
        for pair_id, row in split_by_pair.items()
        if row.get("split") == "train"
    )
    source_receipt = _hotspot_source_receipt(
        scene_name=scene_name,
        split_sha256=split_sha,
        split_hash=split_hash,
        decode_manifest_sha256=decode_manifest_sha,
        decode_protocol_sha256=decode_protocol_sha,
        range_sha256=range_sha,
        range_hash=range_hash,
        support_sha256=support_sha,
        support_index_sha256=support_index_sha,
        train_camera_names=train_camera_names,
    )
    threshold, supplied_threshold_hash, threshold_c = _validate_threshold_manifest(
        paths["hotspot_threshold_manifest"],
        scene_name=scene_name,
        source_receipt=source_receipt,
        train_camera_names=train_camera_names,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
    )

    evaluation_support_path = Path(args.evaluation_support_manifest).resolve()
    if not evaluation_support_path.is_file():
        raise FileNotFoundError(
            f"missing evaluation_support_manifest: {evaluation_support_path}"
        )
    paths["evaluation_support_manifest"] = evaluation_support_path

    evaluation_support = _load_json(paths["evaluation_support_manifest"], "evaluation support")
    evaluation_support_sha = sha256_file(paths["evaluation_support_manifest"])
    if (
        evaluation_support.get("schema_name") != EVALUATION_SUPPORT_SCHEMA
        or evaluation_support.get("schema_version") != 1
        or evaluation_support.get("split") != "test"
    ):
        raise ValueError("evaluation support must be formal schema v1/test")
    if evaluation_support.get("expected_test_count") != counts["test"]:
        raise ValueError("evaluation support test count mismatch")
    if evaluation_support.get("policy") != EVALUATION_SUPPORT_POLICY:
        raise ValueError("evaluation support policy differs from formal F3/legacy policy")
    sources = evaluation_support.get("source_manifests")
    if not isinstance(sources, Mapping):
        raise ValueError("evaluation support lacks source manifests")
    if not isinstance(sources.get("split"), Mapping) or sources["split"].get("sha256") != split_sha:
        raise ValueError("evaluation support is bound to another split")
    if not isinstance(sources.get("valid_support"), Mapping) or sources["valid_support"].get("sha256") != support_sha:
        raise ValueError("evaluation support is bound to another optimization support")
    evaluation_rows = evaluation_support.get("records")
    if not isinstance(evaluation_rows, list):
        raise ValueError("evaluation support lacks records")
    evaluation_support_by_pair = _indexed(evaluation_rows, "evaluation support")
    test_pairs = {
        pair_id for pair_id, row in split_by_pair.items() if row.get("split") == "test"
    }
    if set(evaluation_support_by_pair) != test_pairs:
        raise ValueError("evaluation support identities differ from formal test split")
    evaluation_index: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        if pair_id not in test_pairs:
            continue
        relative, digest = _evaluation_support_record(evaluation_support_by_pair[pair_id])
        encoding = (
            "bool-npy"
            if Path(relative.replace("\\", "/")).suffix.casefold() == ".npy"
            else "binary-image-0-255"
        )
        evaluation_index.append(
            {"pair_id": pair_id, "sha256": digest, "encoding": encoding}
        )
    formal_evaluation = (
        formal_support.get("evaluation") if isinstance(formal_support, Mapping) else None
    )
    if (
        not isinstance(formal_evaluation, Mapping)
        or formal_evaluation.get("manifest_sha256") != evaluation_support_sha
        or formal_evaluation.get("support_index_sha256") != sha256_json(evaluation_index)
        or formal_evaluation.get("policy") != EVALUATION_SUPPORT_POLICY
        or formal_evaluation.get("split") != "test"
    ):
        raise ValueError("formal protocol does not bind this test evaluation support")

    provenance: dict[str, Any] = {
        label: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for label, path in paths.items()
    }
    provenance.update(
        {
            "fixed_lut_sha256_uint8_rgb": lut_sha256(),
            "split_hash": split_hash,
            "range_hash": range_hash,
            "hotspot_source_receipt": source_receipt,
            "threshold_sha256": supplied_threshold_hash,
            "formal_protocol_manifest_sha256": oct_protocol["manifest_sha256"],
            "formal_protocol_sha256": formal_binding["formal_protocol_sha256"],
            "pair_parameter_index_sha256": pair_parameter_index_sha,
        }
    )
    return _FormalInputs(
        scene_name=scene_name,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        threshold_c=threshold_c,
        split_records=split_records,
        canonical_by_pair=canonical_by_pair,
        evaluation_support_by_pair=evaluation_support_by_pair,
        threshold=threshold,
        provenance=provenance,
        formal_binding=formal_binding,
        pair_parameter_index_sha256=pair_parameter_index_sha,
        temperature_error_histogram_max_c=temperature_error_histogram_max_c,
    )


def _histogram_auprc(positive_counts: np.ndarray, negative_counts: np.ndarray) -> float | None:
    positive = np.asarray(positive_counts, dtype=np.int64)
    negative = np.asarray(negative_counts, dtype=np.int64)
    if positive.ndim != 1 or positive.shape != negative.shape:
        raise ValueError("hotspot histograms must be same-shape vectors")
    if bool((positive < 0).any()) or bool((negative < 0).any()):
        raise ValueError("hotspot histogram counts must be nonnegative")
    positives = int(positive.sum())
    if positives == 0:
        return None
    cumulative_tp = np.cumsum(positive[::-1], dtype=np.int64)
    cumulative_fp = np.cumsum(negative[::-1], dtype=np.int64)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def _validate_canonical_target_index(
    formal: _FormalInputs, temperature_root: Path
) -> tuple[dict[str, Path], str]:
    resolved: dict[str, Path] = {}
    index: list[dict[str, Any]] = []
    for split_record in formal.split_records:
        pair_id = _identity(split_record)
        canonical = formal.canonical_by_pair[pair_id]
        path = _resolve_under(
            temperature_root, str(canonical.get("relative_input", "")), pair_id
        )
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype("float32") or value.ndim != 2 or value.size == 0:
            raise TypeError(f"formal target must be non-empty float32 HW: {path}")
        resolved[pair_id] = path
        index.append(
            {
                "pair_id": pair_id,
                "split": str(split_record.get("split")),
                "camera_name": str(split_record.get("thermal_camera_name")),
                "temperature_sha256": _require_sha(
                    canonical.get("input_sha256"), f"target {pair_id}"
                ),
                "canonical_sha256": _require_sha(
                    canonical.get("output_sha256"), f"canonical {pair_id}"
                ),
                "shape_hw": [int(value.shape[0]), int(value.shape[1])],
            }
        )
    digest = sha256_json(index)
    canonical_binding = formal.formal_binding.get("canonical_target")
    if (
        not isinstance(canonical_binding, Mapping)
        or canonical_binding.get("target_index_sha256") != digest
    ):
        raise ValueError("formal protocol canonical target index mismatch")
    return resolved, digest


def _histogram_percentile_upper_edge(
    histogram: np.ndarray, *, maximum: float, count: int, percentile: float
) -> float:
    values = np.asarray(histogram, dtype=np.int64)
    if values.ndim != 1 or values.size < 2 or int(values.sum()) != int(count):
        raise ValueError("percentile histogram/count mismatch")
    rank = int(math.ceil((float(percentile) / 100.0) * int(count)))
    index = int(np.searchsorted(np.cumsum(values), rank, side="left"))
    index = min(max(index, 0), values.size - 1)
    return float((index + 1) * float(maximum) / values.size)


@dataclass
class _Accumulator:
    error_histogram_max_c: float
    valid_pixels: int = 0
    target_hot_pixels: int = 0
    prediction_hot_pixels: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_signed_error: float = 0.0
    max_abs_error: float = 0.0
    hot_sum_abs_error: float = 0.0
    hot_pixel_count: int = 0
    nonhot_sum_abs_error: float = 0.0
    nonhot_pixel_count: int = 0
    off_lut_sum: float = 0.0
    off_lut_squared_sum: float = 0.0
    off_lut_max: float = 0.0
    positive_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    )
    negative_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(HOTSPOT_EVALUATION_BINS, dtype=np.int64)
    )
    absolute_error_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(65_536, dtype=np.int64)
    )
    off_lut_histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(65_536, dtype=np.int64)
    )

    def update(
        self,
        prediction_c: np.ndarray,
        target_c: np.ndarray,
        off_lut_distance: np.ndarray,
        *,
        threshold_c: float,
        tmin_c: float,
        tmax_c: float,
    ) -> None:
        prediction = np.asarray(prediction_c, dtype=np.float32).reshape(-1)
        target = np.asarray(target_c, dtype=np.float32).reshape(-1)
        distance = np.asarray(off_lut_distance, dtype=np.float32).reshape(-1)
        if prediction.size == 0 or prediction.shape != target.shape or prediction.shape != distance.shape:
            raise ValueError("hotspot accumulator inputs must be equal-size and non-empty")
        if not bool(np.isfinite(prediction).all() and np.isfinite(target).all() and np.isfinite(distance).all()):
            raise ValueError("hotspot accumulator inputs contain NaN/Inf")
        target_hot = target >= np.float32(threshold_c)
        prediction_hot = prediction >= np.float32(threshold_c)
        signed = prediction.astype(np.float64) - target.astype(np.float64)
        absolute = np.abs(signed)
        if float(absolute.max()) > self.error_histogram_max_c + max(
            1e-6, 1e-7 * self.error_histogram_max_c
        ):
            raise ValueError("temperature error exceeds the formal range-QA histogram bound")
        self.valid_pixels += int(prediction.size)
        self.target_hot_pixels += int(target_hot.sum())
        self.prediction_hot_pixels += int(prediction_hot.sum())
        self.true_positive += int(np.count_nonzero(target_hot & prediction_hot))
        self.false_positive += int(np.count_nonzero(~target_hot & prediction_hot))
        self.false_negative += int(np.count_nonzero(target_hot & ~prediction_hot))
        self.sum_abs_error += float(absolute.sum(dtype=np.float64))
        self.sum_squared_error += float(np.square(signed).sum(dtype=np.float64))
        self.sum_signed_error += float(signed.sum(dtype=np.float64))
        self.max_abs_error = max(self.max_abs_error, float(absolute.max()))
        self.hot_sum_abs_error += float(absolute[target_hot].sum(dtype=np.float64))
        self.hot_pixel_count += int(target_hot.sum())
        self.nonhot_sum_abs_error += float(absolute[~target_hot].sum(dtype=np.float64))
        self.nonhot_pixel_count += int((~target_hot).sum())
        distance64 = distance.astype(np.float64)
        self.off_lut_sum += float(distance64.sum(dtype=np.float64))
        self.off_lut_squared_sum += float(np.square(distance64).sum(dtype=np.float64))
        self.off_lut_max = max(self.off_lut_max, float(distance64.max()))
        self.absolute_error_histogram += np.histogram(
            np.minimum(absolute, self.error_histogram_max_c),
            bins=self.absolute_error_histogram.size,
            range=(0.0, self.error_histogram_max_c),
        )[0]
        off_lut_maximum = math.sqrt(3.0 * 255.0 * 255.0)
        self.off_lut_histogram += np.histogram(
            np.minimum(distance64, off_lut_maximum),
            bins=self.off_lut_histogram.size,
            range=(0.0, off_lut_maximum),
        )[0]
        score = np.clip(
            (prediction - np.float32(tmin_c)) / np.float32(tmax_c - tmin_c),
            np.float32(0.0),
            np.float32(1.0),
        )
        index = np.clip(
            (score * np.float32(HOTSPOT_EVALUATION_BINS - 1)).astype(np.int64),
            0,
            HOTSPOT_EVALUATION_BINS - 1,
        )
        self.positive_histogram += np.bincount(
            index[target_hot], minlength=HOTSPOT_EVALUATION_BINS
        )
        self.negative_histogram += np.bincount(
            index[~target_hot], minlength=HOTSPOT_EVALUATION_BINS
        )

    def summary(self) -> dict[str, Any]:
        count = self.valid_pixels
        if count <= 0:
            raise ValueError("hotspot metric accumulator is empty")
        union = self.true_positive + self.false_positive + self.false_negative
        precision_denominator = self.true_positive + self.false_positive
        recall_denominator = self.true_positive + self.false_negative
        hot_mae = (
            None if self.hot_pixel_count == 0 else self.hot_sum_abs_error / self.hot_pixel_count
        )
        nonhot_mae = (
            None
            if self.nonhot_pixel_count == 0
            else self.nonhot_sum_abs_error / self.nonhot_pixel_count
        )
        return {
            "valid_pixels": count,
            "target_hot_pixels": self.target_hot_pixels,
            "target_hot_prevalence": self.target_hot_pixels / count,
            "prediction_hot_pixels": self.prediction_hot_pixels,
            "confusion": {
                "true_positive": self.true_positive,
                "false_positive": self.false_positive,
                "false_negative": self.false_negative,
            },
            "hotspot_iou": 0.0 if union == 0 else self.true_positive / union,
            "hotspot_precision": (
                None if precision_denominator == 0 else self.true_positive / precision_denominator
            ),
            "hotspot_recall": (
                None if recall_denominator == 0 else self.true_positive / recall_denominator
            ),
            "hotspot_auprc_histogram_4096": _histogram_auprc(
                self.positive_histogram, self.negative_histogram
            ),
            "temperature_error": {
                "mae_c": self.sum_abs_error / count,
                "rmse_c": math.sqrt(self.sum_squared_error / count),
                "signed_bias_c": self.sum_signed_error / count,
                "p95_abs_error_c": (
                    0.0
                    if self.max_abs_error == 0.0
                    else _histogram_percentile_upper_edge(
                        self.absolute_error_histogram,
                        maximum=self.error_histogram_max_c,
                        count=count,
                        percentile=95.0,
                    )
                ),
                "p95_estimator": "fixed histogram upper edge",
                "p95_histogram_bin_width_c": (
                    self.error_histogram_max_c / self.absolute_error_histogram.size
                ),
            },
            "temperature_mae_by_target_hotspot": {
                "hot_pixels": self.hot_pixel_count,
                "hot_mae_c": hot_mae,
                "nonhot_pixels": self.nonhot_pixel_count,
                "nonhot_mae_c": nonhot_mae,
                "hot_minus_nonhot_mae_c": (
                    None if hot_mae is None or nonhot_mae is None else hot_mae - nonhot_mae
                ),
            },
            "off_lut_distance_rgb": {
                "mean": self.off_lut_sum / count,
                "rms": math.sqrt(self.off_lut_squared_sum / count),
                "max": self.off_lut_max,
                "p95": (
                    0.0
                    if self.off_lut_max == 0.0
                    else _histogram_percentile_upper_edge(
                        self.off_lut_histogram,
                        maximum=math.sqrt(3.0 * 255.0 * 255.0),
                        count=count,
                        percentile=95.0,
                    )
                ),
                "p95_estimator": "fixed histogram upper edge",
                "p95_histogram_bin_width": (
                    math.sqrt(3.0 * 255.0 * 255.0)
                    / self.off_lut_histogram.size
                ),
            },
        }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float], *, ranked: bool) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or x.shape != y.shape or not bool(np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    if ranked:
        x, y = _average_ranks(x), _average_ranks(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator == 0.0 else float(np.dot(x, y) / denominator)


def _association(per_view: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    pairs: list[tuple[float, float]] = []
    for row in per_view:
        value = row["metrics"].get(metric)
        mae = row["metrics"]["temperature_error"]["mae_c"]
        if value is not None and math.isfinite(float(value)) and math.isfinite(float(mae)):
            pairs.append((float(mae), float(value)))
    return {
        "views": len(pairs),
        "temperature_mae_vs_metric_pearson": _correlation(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs], ranked=False
        ),
        "temperature_mae_vs_metric_spearman": _correlation(
            [pair[0] for pair in pairs], [pair[1] for pair in pairs], ranked=True
        ),
    }


def _block_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "stratum": record.get("stratum"),
        "strip_id": record.get("strip_id"),
        "block_index": record.get("block_index"),
    }


def evaluate_formal_baseline_hotspots(args: argparse.Namespace) -> dict[str, Any]:
    formal = _validate_formal_inputs(args)
    # Test artifacts are deliberately touched only after the full train-only
    # threshold/radiometry contract above has passed.
    temperature_root = Path(args.temperature_root).resolve()
    support_root = Path(args.evaluation_support_root).resolve()
    render_root = Path(args.render_root).resolve()
    resolved_targets, target_index_sha = _validate_canonical_target_index(
        formal, temperature_root
    )
    render_relative, render_basenames = _index_pngs(render_root)
    test_records = [record for record in formal.split_records if record.get("split") == "test"]
    if not test_records:
        raise ValueError("formal split has no test records")

    aggregate = _Accumulator(formal.temperature_error_histogram_max_c)
    block_accumulators: dict[str, tuple[dict[str, Any], _Accumulator, list[str]]] = {}
    per_view: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for split_record in test_records:
        pair_id = _identity(split_record)
        camera_name = str(split_record.get("thermal_camera_name", ""))
        canonical = formal.canonical_by_pair[pair_id]
        target_path = resolved_targets[pair_id]
        expected_target_sha = _require_sha(canonical.get("input_sha256"), f"target {pair_id}")
        actual_target_sha = sha256_file(target_path)
        if actual_target_sha != expected_target_sha:
            raise ValueError(f"direct TSDK target SHA mismatch for {pair_id}")
        support_row = formal.evaluation_support_by_pair[pair_id]
        support_relative, expected_support_sha = _evaluation_support_record(support_row)
        support_path = _resolve_under(support_root, support_relative, pair_id)
        actual_support_sha = sha256_file(support_path)
        if actual_support_sha != expected_support_sha:
            raise ValueError(f"formal evaluation support SHA mismatch for {pair_id}")
        render_path = _resolve_render(
            render_root, pair_id, camera_name, render_relative, render_basenames
        )
        render_sha = sha256_file(render_path)

        target = _load_float_temperature(target_path)
        support = _load_bool_support(support_path)
        render = _load_rgb_render(render_path)
        if render.shape[:2] != target.shape or support.shape != target.shape:
            raise ValueError(
                f"formal render/target/support shape mismatch for {pair_id}: "
                f"render={render.shape[:2]}, target={target.shape}, support={support.shape}"
            )
        if not bool(support.any()):
            raise ValueError(f"formal evaluation support is empty for {pair_id}")
        prediction, off_lut, _ = rgb_to_temperature(
            render, formal.tmin_c, formal.tmax_c, chunk_pixels=int(args.chunk_pixels)
        )
        selected_prediction = prediction[support]
        selected_target = target[support]
        selected_off_lut = off_lut[support]
        view_accumulator = _Accumulator(formal.temperature_error_histogram_max_c)
        for accumulator in (view_accumulator, aggregate):
            accumulator.update(
                selected_prediction,
                selected_target,
                selected_off_lut,
                threshold_c=formal.threshold_c,
                tmin_c=formal.tmin_c,
                tmax_c=formal.tmax_c,
            )
        block = _block_identity(split_record)
        block_key = sha256_json(block)
        if block_key not in block_accumulators:
            block_accumulators[block_key] = (
                block,
                _Accumulator(formal.temperature_error_histogram_max_c),
                [],
            )
        _, block_accumulator, block_pairs = block_accumulators[block_key]
        block_accumulator.update(
            selected_prediction,
            selected_target,
            selected_off_lut,
            threshold_c=formal.threshold_c,
            tmin_c=formal.tmin_c,
            tmax_c=formal.tmax_c,
        )
        block_pairs.append(pair_id)
        view_metrics = view_accumulator.summary()
        per_view.append(
            {
                "pair_id": pair_id,
                "camera_name": camera_name,
                "split": "test",
                "stratum": split_record.get("stratum"),
                "strip_id": split_record.get("strip_id"),
                "block_index": split_record.get("block_index"),
                "metrics": view_metrics,
            }
        )
        input_records.append(
            {
                "pair_id": pair_id,
                "camera_name": camera_name,
                "shape_hw": list(target.shape),
                "target": {
                    "path": str(target_path),
                    "sha256": actual_target_sha,
                    "dtype": "float32",
                },
                "evaluation_support": {
                    "path": str(support_path),
                    "sha256": actual_support_sha,
                    "dtype": "bool",
                },
                "render": {
                    "path": str(render_path),
                    "sha256": render_sha,
                    "mode": "RGB-nearest-fixed-LUT-evaluated",
                },
            }
        )

    if len({row["pair_id"] for row in input_records}) != len(test_records):
        raise RuntimeError("formal test evaluation did not cover each pair exactly once")
    per_view.sort(key=lambda row: row["pair_id"])
    input_records.sort(key=lambda row: row["pair_id"])
    per_block: list[dict[str, Any]] = []
    for block_key in sorted(block_accumulators):
        block, accumulator, pairs = block_accumulators[block_key]
        per_block.append(
            {
                **block,
                "block_identity_sha256": block_key,
                "pair_ids": sorted(pairs),
                "pair_ids_sha256": sha256_json(sorted(pairs)),
                "metrics": accumulator.summary(),
            }
        )

    evaluator_path = Path(__file__).resolve()
    provenance = dict(formal.provenance)
    provenance.update(
        {
            "render_root": str(render_root),
            "temperature_root": str(temperature_root),
            "evaluation_support_root": str(support_root),
            "selected_test_inputs": input_records,
            "selected_test_inputs_sha256": sha256_json(input_records),
            "canonical_target_index_sha256": target_index_sha,
            "evaluator_source": {
                "path": str(evaluator_path),
                "sha256": sha256_file(evaluator_path),
                "bytes": evaluator_path.stat().st_size,
            },
        }
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "method_name": str(args.method_name),
        "scene_name": formal.scene_name,
        "split": "test",
        "selection_boundary": {
            "threshold_source_split": "train",
            "test_statistics_used_for_threshold": False,
            "quantile": FORMAL_QUANTILE,
            "threshold_histogram_bins": FORMAL_THRESHOLD_BINS,
            "threshold_c": formal.threshold_c,
            "test_role": "final_report_only",
        },
        "display_semantics": {
            "primary": (
                "nearest fixed 256-entry Hot-Iron LUT projection followed by exact "
                "8-bit display-bin apparent-temperature recovery"
            ),
            "comparable_to_oct_evaluator_v2": True,
            "hotspot_auprc_score": "display-equivalent temperature normalized to scene range",
            "hotspot_auprc_estimator": "4096-bin descending precision-recall integral",
            "target": "direct float32 TSDK-referenced apparent-temperature; never palette-inverted",
            "support": "formal test evaluation support; no resize",
            "off_lut_distance_role": "diagnostic only",
        },
        "temperature_range_c": [formal.tmin_c, formal.tmax_c],
        "formal_binding_compatibility": {
            "formal_protocol_manifest_sha256": formal.provenance[
                "formal_protocol_manifest_sha256"
            ],
            "formal_protocol_sha256": formal.provenance["formal_protocol_sha256"],
            "pair_parameter_index_sha256": formal.pair_parameter_index_sha256,
            "canonical_target_index_sha256": target_index_sha,
            "target_semantics": TARGET_SEMANTICS,
            "method_semantics": METHOD_SEMANTICS,
            "status": "passed",
        },
        "hotspot_threshold": formal.threshold,
        "metrics": aggregate.summary(),
        "temperature_mae_association_across_views": {
            metric: _association(per_view, metric)
            for metric in (
                "hotspot_iou",
                "hotspot_precision",
                "hotspot_recall",
                "hotspot_auprc_histogram_4096",
            )
        },
        "per_view": per_view,
        "per_block": per_block,
        "inputs": provenance,
        "inputs_sha256": sha256_json(provenance),
    }
    report["report_payload_sha256"] = sha256_json(report)
    return report


def write_atomic_report(path: Path, report: Mapping[str, Any]) -> Path:
    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        raise ValueError("formal hotspot report schema mismatch")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping) or report.get("inputs_sha256") != sha256_json(inputs):
        raise ValueError("formal hotspot report input inventory hash mismatch")
    basis = dict(report)
    supplied_report_hash = basis.pop("report_payload_sha256", None)
    if supplied_report_hash != sha256_json(basis):
        raise ValueError("formal hotspot report self-hash mismatch")

    def verify_file_identities(value: Any) -> None:
        if isinstance(value, Mapping):
            path_value, digest_value = value.get("path"), value.get("sha256")
            if isinstance(path_value, str) and isinstance(digest_value, str):
                source = Path(path_value)
                if not source.is_file() or sha256_file(source) != digest_value:
                    raise ValueError(f"formal hotspot input changed before write: {source}")
                if "bytes" in value and int(value["bytes"]) != source.stat().st_size:
                    raise ValueError(f"formal hotspot input size changed before write: {source}")
            for nested in value.values():
                verify_file_identities(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                verify_file_identities(nested)

    verify_file_identities(inputs)
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace formal hotspot report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale formal hotspot temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--formal-protocol-manifest", required=True, type=Path)
    parser.add_argument("--bound-split", required=True, type=Path)
    parser.add_argument("--decode-manifest", required=True, type=Path)
    parser.add_argument("--decode-protocol", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--optimization-support-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-support-manifest", required=True, type=Path)
    parser.add_argument("--hotspot-threshold-manifest", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--evaluation-support-root", required=True, type=Path)
    parser.add_argument("--render-root", required=True, type=Path)
    parser.add_argument("--chunk-pixels", type=int, default=32768)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not str(args.method_name).strip():
        raise ValueError("--method-name must be non-empty")
    if int(args.chunk_pixels) <= 0:
        raise ValueError("--chunk-pixels must be positive")
    report = evaluate_formal_baseline_hotspots(args)
    output = write_atomic_report(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "method_name": report["method_name"],
                "report": str(output),
                "report_payload_sha256": report["report_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
