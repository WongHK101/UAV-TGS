#!/usr/bin/env python3
"""Freeze the formal train-only hotspot threshold for any scene.

This sidecar deliberately does not depend on the OCT training binding.  It
authenticates the formal radiometry inputs, reads only train temperature and
valid-support arrays, and emits the threshold receipt already consumed by the
formal hotspot evaluators.  Non-train records are checked as manifest
identities only; their payload paths are never resolved or opened.  Both the
legacy train/guard/test protocol and Hold-8 train/test protocol are accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


THRESHOLD_SCHEMA = "uav-tgs-oct-train-only-hotspot-threshold-v1"
RANGE_SCHEMA = "uav_tgs_train_only_scene_temperature_range"
SUPPORT_SCHEMA = "uav-tgs-undistorted-temperature-v1"
PROTOCOL_SCHEMA = "uav-tgs.radiometry-protocol.v1"
FORMAL_SPLITS = frozenset({"train", "guard", "test"})
HOLD8_SPLITS = frozenset({"train", "test"})
SUPPORTED_SPLIT_SETS = frozenset({FORMAL_SPLITS, HOLD8_SPLITS})
FORMAL_QUANTILE = 0.95
FORMAL_HISTOGRAM_BINS = 65_536
FORMAL_PARAMETER_KEYS = frozenset(
    {"distance_m", "humidity_percent", "emissivity", "ambient_c", "reflected_c"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
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


def _require_sha(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return token


def _identity(record: Mapping[str, Any]) -> str:
    for key in ("pair_id", "relative_id", "image_name", "thermal_camera_name", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.replace("\\", "/")).stem
    raise ValueError("formal record has no usable pair identity")


def _indexed(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} records must be objects")
        pair_id = _identity(row)
        if pair_id in result:
            raise ValueError(f"{label} has duplicate identity {pair_id!r}")
        result[pair_id] = row
    if not result:
        raise ValueError(f"{label} is empty")
    return result


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


def _support_index(
    pair_ids: Sequence[str], support_by_pair: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        row = support_by_pair[pair_id]
        relative = _support_relative(row)
        result.append(
            {
                "pair_id": pair_id,
                "sha256": _support_sha(row),
                "encoding": (
                    "bool-npy"
                    if Path(relative.replace("\\", "/")).suffix.casefold() == ".npy"
                    else "binary-image-0-255"
                ),
            }
        )
    return result


def hotspot_source_receipt(
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
    """Return the byte-compatible receipt used by existing OCT evaluators."""

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


def _portable_relative(value: Any, label: str) -> PurePosixPath:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        raise ValueError(f"{label} is missing")
    if re.match(r"^[A-Za-z]:", text) or text.startswith("/"):
        # Runtime roots are authoritative after cross-platform transfer.  For
        # an absolute provenance path only its basename is portable.
        text = PurePosixPath(text).name
    path = PurePosixPath(text)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} contains an unsafe path: {value!r}")
    return path


def _resolve_train_payload(
    root: Path, relative: Any, *, pair_id: str, label: str, expected_suffix: str
) -> Path:
    """Resolve one train artifact without enumerating any guard/test path."""

    root = root.resolve()
    portable = _portable_relative(relative, label)
    candidates: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} escapes its declared root: {relative!r}") from error
        if resolved not in candidates:
            candidates.append(resolved)

    add(root.joinpath(*portable.parts))
    if portable.parts and portable.parts[0].casefold() == root.name.casefold():
        add(root.joinpath(*portable.parts[1:]))
    add(root / portable.name)
    if len(portable.parts) == 1:
        add(root / f"{pair_id}{expected_suffix}")

    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"cannot uniquely resolve train {label} for {pair_id!r} under {root}: {existing}"
        )
    if existing[0].suffix.casefold() != expected_suffix.casefold() and label == "temperature":
        raise ValueError(f"train temperature must be {expected_suffix}: {existing[0]}")
    return existing[0]


def _output_temperature_relative(record: Mapping[str, Any]) -> str:
    value = record.get("output_temperature")
    if not isinstance(value, Mapping):
        raise ValueError(f"support record {_identity(record)!r} lacks output_temperature")
    relative = value.get("relative_path", value.get("path"))
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"support record {_identity(record)!r} lacks output temperature path")
    return relative


def _load_train_temperature(path: Path, expected_sha: str) -> np.ndarray:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"train temperature SHA mismatch: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.dtype != np.dtype("float32") or value.ndim != 2 or value.size == 0:
        raise TypeError(f"train temperature must be non-empty float32 HW: {path}")
    return value


def _load_train_support(path: Path, expected_sha: str) -> np.ndarray:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"train support SHA mismatch: {path}")
    if path.suffix.casefold() == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype("bool") or value.ndim != 2 or value.size == 0:
            raise TypeError(f"train NPY support must be non-empty bool HW: {path}")
        return value
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    if gray.ndim != 2 or gray.size == 0:
        raise TypeError(f"train image support must be non-empty HW: {path}")
    unique = set(int(item) for item in np.unique(gray))
    if not unique <= {0, 255}:
        raise ValueError(f"train image support must contain only 0/255: {path}")
    return gray == 255


def _validate_parameter_record(row: Mapping[str, Any], pair_id: str) -> None:
    parameters = row.get("decode_parameters")
    if not isinstance(parameters, Mapping) or set(parameters) != FORMAL_PARAMETER_KEYS:
        raise ValueError(f"decode parameters are incomplete for {pair_id}")
    for name in sorted(FORMAL_PARAMETER_KEYS):
        entry = parameters[name]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("source"), str):
            raise ValueError(f"decode parameter {name} lacks value/source for {pair_id}")
        try:
            number = float(entry.get("value"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"decode parameter {name} is not numeric for {pair_id}") from error
        if not math.isfinite(number) or not entry["source"].strip():
            raise ValueError(f"decode parameter {name} is invalid for {pair_id}")


def _validate_inputs(
    *,
    scene_name: str,
    bound_split_path: Path,
    decode_manifest_path: Path,
    decode_protocol_path: Path,
    range_manifest_path: Path,
    support_manifest_path: Path,
) -> dict[str, Any]:
    paths = (
        bound_split_path,
        decode_manifest_path,
        decode_protocol_path,
        range_manifest_path,
        support_manifest_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    bound = _load_json(bound_split_path, "bound split")
    if str(bound.get("scene", "")) != scene_name:
        raise ValueError("bound split/CLI scene mismatch")
    records_raw = bound.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("bound split has no records")
    records = tuple(records_raw)
    split_by_pair = _indexed(records, "bound split")
    pair_ids = list(split_by_pair)
    split_values = [str(row.get("split", "")) for row in records]
    split_labels = frozenset(split_values)
    if split_labels not in SUPPORTED_SPLIT_SETS:
        raise ValueError(
            "bound split must contain exactly train/test or train/guard/test labels"
        )
    counts = {
        "total": len(records),
        **{name: split_values.count(name) for name in sorted(split_labels)},
    }
    if bound.get("counts") != counts:
        raise ValueError("bound split counts mismatch")
    split_hash = _require_sha(bound.get("split_hash"), "bound split hash")
    split_sha = sha256_file(bound_split_path)

    camera_by_pair: dict[str, str] = {}
    for pair_id, row in split_by_pair.items():
        camera = str(row.get("thermal_camera_name", "")).strip()
        if not camera:
            raise ValueError(f"bound split lacks thermal_camera_name for {pair_id}")
        camera_by_pair[pair_id] = camera
    if len(set(camera_by_pair.values())) != len(camera_by_pair):
        raise ValueError("bound split thermal camera names are not unique")

    decode_binding = bound.get("decode_binding")
    if not isinstance(decode_binding, Mapping):
        raise ValueError("bound split lacks formal decode binding")
    if decode_binding.get("adapter_backend") != "official-dji-irp":
        raise ValueError("formal target must use official-dji-irp")
    for key in (
        "verified_decode_requests",
        "verified_raw_rjpeg_hashes",
        "verified_temperature_file_hashes",
    ):
        if decode_binding.get(key) is not True:
            raise ValueError(f"bound split decode binding does not prove {key}")
    decode_manifest_sha = sha256_file(decode_manifest_path)
    decode_protocol_sha = sha256_file(decode_protocol_path)
    if decode_binding.get("decode_manifest_sha256") != decode_manifest_sha:
        raise ValueError("decode manifest differs from bound split")
    if decode_binding.get("decode_protocol_sha256") != decode_protocol_sha:
        raise ValueError("decode protocol differs from bound split")
    protocol_hash = _require_sha(decode_binding.get("protocol_hash"), "decode protocol hash")

    decode_rows = _indexed(_load_jsonl(decode_manifest_path, "decode manifest"), "decode manifest")
    protocol_rows = _indexed(_load_jsonl(decode_protocol_path, "decode protocol"), "decode protocol")
    if set(decode_rows) != set(pair_ids) or set(protocol_rows) != set(pair_ids):
        raise ValueError("decode/protocol identities differ from bound split")
    for pair_id in pair_ids:
        decoded = decode_rows[pair_id]
        protocol = protocol_rows[pair_id]
        if (
            decoded.get("scene") != scene_name
            or decoded.get("success") is not True
            or decoded.get("dtype") != "float32"
        ):
            raise ValueError(f"decode record is not successful float32 for {pair_id}")
        _require_sha(decoded.get("output_sha256"), f"decode target {pair_id}")
        if (
            protocol.get("scene") != scene_name
            or protocol.get("protocol_hash") != protocol_hash
            or protocol.get("schema_version") != PROTOCOL_SCHEMA
        ):
            raise ValueError(f"decode protocol hash/schema mismatch for {pair_id}")
        _validate_parameter_record(protocol, pair_id)

    range_payload = _load_json(range_manifest_path, "range manifest")
    if (
        range_payload.get("schema_name") != RANGE_SCHEMA
        or range_payload.get("schema_version") != 1
        or range_payload.get("scene") != scene_name
    ):
        raise ValueError("formal range schema/scene mismatch")
    if range_payload.get("source_split_manifest_sha256") != split_sha:
        raise ValueError("range manifest is not bound to the exact formal split")
    if range_payload.get("split_hash") != split_hash:
        raise ValueError("range/formal split hash mismatch")
    configuration = range_payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("range manifest lacks configuration")
    guard_role = configuration.get("guard_role")
    if split_labels == FORMAL_SPLITS and guard_role != "not_read":
        raise ValueError("legacy formal range must not read guard")
    if split_labels == HOLD8_SPLITS and guard_role not in (None, "not_read"):
        raise ValueError("Hold-8 range declares an invalid guard role")
    if configuration.get("test_role") != "qa_only_not_used_for_estimation":
        raise ValueError("formal range may use test only for post-estimation QA")
    try:
        tmin_c = float(range_payload.get("Tmin"))
        tmax_c = float(range_payload.get("Tmax"))
    except (TypeError, ValueError) as error:
        raise ValueError("formal temperature range is not numeric") from error
    if not math.isfinite(tmin_c) or not math.isfinite(tmax_c) or tmax_c <= tmin_c:
        raise ValueError("formal temperature range is invalid")
    range_hash = _require_sha(range_payload.get("range_hash"), "range hash")
    range_basis = {
        "scene": range_payload.get("scene"),
        "split_hash": range_payload.get("split_hash"),
        "configuration": configuration,
        "Tmin": range_payload.get("Tmin"),
        "Tmax": range_payload.get("Tmax"),
    }
    if range_hash != sha256_json(range_basis):
        raise ValueError("formal range logical hash mismatch")
    train_estimation = range_payload.get("train_estimation")
    clipping_stats = range_payload.get("clipping_stats")
    if not isinstance(train_estimation, Mapping) or not isinstance(clipping_stats, Mapping):
        raise ValueError("formal range train/clipping provenance is missing")
    if int(train_estimation.get("frame_count", -1)) != counts["train"]:
        raise ValueError("formal range train frame count mismatch")
    for split in ("train", "test"):
        item = clipping_stats.get(split)
        if not isinstance(item, Mapping) or int(item.get("frame_count", -1)) != counts[split]:
            raise ValueError(f"formal range {split} QA frame count mismatch")
    quantiles = range_payload.get("per_frame_quantiles")
    if not isinstance(quantiles, list):
        raise ValueError("formal range lacks per-frame quantiles")
    quantile_counts = Counter(
        str(row.get("split")) for row in quantiles if isinstance(row, Mapping)
    )
    if quantile_counts != Counter({"train": counts["train"], "test": counts["test"]}):
        raise ValueError("formal range per-frame split coverage mismatch")

    support_payload = _load_json(support_manifest_path, "optimization support")
    if support_payload.get("schema") != SUPPORT_SCHEMA or support_payload.get("status") != "complete":
        raise ValueError("optimization support manifest is incomplete/unsupported")
    support_rows_raw = support_payload.get("files")
    if not isinstance(support_rows_raw, list):
        raise ValueError("optimization support manifest lacks files")
    support_by_pair = _indexed(support_rows_raw, "optimization support")
    if set(support_by_pair) != set(pair_ids):
        raise ValueError("optimization support identities differ from bound split")
    for pair_id in pair_ids:
        row = support_by_pair[pair_id]
        raw = row.get("input_temperature")
        output = row.get("output_temperature")
        valid = row.get("valid_support")
        if not isinstance(raw, Mapping) or not isinstance(output, Mapping) or not isinstance(valid, Mapping):
            raise ValueError(f"optimization support provenance is incomplete for {pair_id}")
        if raw.get("dtype") != "float32" or output.get("dtype") != "float32":
            raise TypeError(f"optimization support temperature dtype is not float32 for {pair_id}")
        if valid.get("dtype") != "bool":
            raise TypeError(f"optimization support mask dtype is not bool for {pair_id}")
        if _require_sha(raw.get("sha256"), f"raw target {pair_id}") != _require_sha(
            decode_rows[pair_id].get("output_sha256"), f"decode target {pair_id}"
        ):
            raise ValueError(f"optimization support/decode target SHA mismatch for {pair_id}")
        _require_sha(output.get("sha256"), f"undistorted target {pair_id}")
        _support_sha(row)
        _output_temperature_relative(row)
        _support_relative(row)

    support_sha = sha256_file(support_manifest_path)
    support_index_sha = sha256_json(_support_index(pair_ids, support_by_pair))
    train_pairs = [
        pair_id for pair_id in pair_ids if split_by_pair[pair_id].get("split") == "train"
    ]
    train_camera_names = sorted(camera_by_pair[pair_id] for pair_id in train_pairs)
    if not train_pairs or len(train_camera_names) != counts["train"]:
        raise ValueError("formal train membership is empty/inconsistent")

    return {
        "pair_ids": pair_ids,
        "train_pairs": train_pairs,
        "train_camera_names": train_camera_names,
        "decode_by_pair": decode_rows,
        "support_by_pair": support_by_pair,
        "tmin_c": tmin_c,
        "tmax_c": tmax_c,
        "source_receipt": hotspot_source_receipt(
            scene_name=scene_name,
            split_sha256=split_sha,
            split_hash=split_hash,
            decode_manifest_sha256=decode_manifest_sha,
            decode_protocol_sha256=decode_protocol_sha,
            range_sha256=sha256_file(range_manifest_path),
            range_hash=range_hash,
            support_sha256=support_sha,
            support_index_sha256=support_index_sha,
            train_camera_names=train_camera_names,
        ),
    }


def freeze_train_hotspot_threshold(
    *,
    scene_name: str,
    bound_split_path: str | Path,
    decode_manifest_path: str | Path,
    decode_protocol_path: str | Path,
    range_manifest_path: str | Path,
    temperature_root: str | Path,
    support_manifest_path: str | Path,
    support_root: str | Path,
    chunk_pixels: int = 1_048_576,
) -> dict[str, Any]:
    """Validate formal inputs and compute the fixed train-only threshold."""

    if chunk_pixels <= 0:
        raise ValueError("chunk_pixels must be positive")
    bound_split_path = Path(bound_split_path).resolve()
    decode_manifest_path = Path(decode_manifest_path).resolve()
    decode_protocol_path = Path(decode_protocol_path).resolve()
    range_manifest_path = Path(range_manifest_path).resolve()
    support_manifest_path = Path(support_manifest_path).resolve()
    temperature_root = Path(temperature_root).resolve()
    support_root = Path(support_root).resolve()
    if not temperature_root.is_dir():
        raise FileNotFoundError(f"temperature root: {temperature_root}")
    if not support_root.is_dir():
        raise FileNotFoundError(f"optimization support root: {support_root}")

    validated = _validate_inputs(
        scene_name=str(scene_name),
        bound_split_path=bound_split_path,
        decode_manifest_path=decode_manifest_path,
        decode_protocol_path=decode_protocol_path,
        range_manifest_path=range_manifest_path,
        support_manifest_path=support_manifest_path,
    )
    tmin_c = float(validated["tmin_c"])
    tmax_c = float(validated["tmax_c"])
    edges = np.linspace(tmin_c, tmax_c, FORMAL_HISTOGRAM_BINS + 1, dtype=np.float64)
    counts = np.zeros(FORMAL_HISTOGRAM_BINS, dtype=np.int64)

    support_by_pair = validated["support_by_pair"]
    for pair_id in validated["train_pairs"]:
        row = support_by_pair[pair_id]
        output = row["output_temperature"]
        target = _resolve_train_payload(
            temperature_root,
            _output_temperature_relative(row),
            pair_id=pair_id,
            label="temperature",
            expected_suffix=".npy",
        )
        support = _resolve_train_payload(
            support_root,
            _support_relative(row),
            pair_id=pair_id,
            label="support",
            expected_suffix=Path(_support_relative(row)).suffix,
        )
        temperature = _load_train_temperature(
            target, _require_sha(output.get("sha256"), f"undistorted target {pair_id}")
        )
        mask = _load_train_support(support, _support_sha(row))
        if tuple(temperature.shape) != tuple(mask.shape):
            raise ValueError(f"train temperature/support shape mismatch for {pair_id}")
        flat_temperature = temperature.reshape(-1)
        flat_mask = mask.reshape(-1)
        for start in range(0, int(flat_temperature.size), chunk_pixels):
            stop = min(start + chunk_pixels, int(flat_temperature.size))
            values = np.asarray(flat_temperature[start:stop], dtype=np.float32)
            if not bool(np.isfinite(values).all()):
                raise ValueError(f"train temperature contains NaN/Inf for {pair_id}")
            selected = values[np.asarray(flat_mask[start:stop], dtype=np.bool_)]
            if selected.size:
                selected = np.clip(selected, tmin_c, tmax_c)
                counts += np.histogram(selected, bins=edges)[0]

    total = int(counts.sum())
    if total <= 0:
        raise ValueError("formal train support contains no valid pixels")
    target_rank = int(math.ceil(FORMAL_QUANTILE * total))
    index = int(np.searchsorted(np.cumsum(counts), target_rank, side="left"))
    threshold = float(edges[min(index + 1, FORMAL_HISTOGRAM_BINS)])
    payload: dict[str, Any] = {
        "schema": THRESHOLD_SCHEMA,
        "scene_name": str(scene_name),
        "source_receipt": validated["source_receipt"],
        "source_split": "train",
        "test_statistics_used": False,
        "quantile": FORMAL_QUANTILE,
        "histogram_bins": FORMAL_HISTOGRAM_BINS,
        "valid_train_pixels": total,
        "threshold_c": threshold,
        "range_c": [tmin_c, tmax_c],
        "train_view_ids_sha256": sha256_json(validated["train_camera_names"]),
    }
    payload["threshold_sha256"] = sha256_json(payload)
    return payload


def write_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"hotspot threshold output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--bound-split", required=True, type=Path)
    parser.add_argument("--decode-manifest", required=True, type=Path)
    parser.add_argument("--decode-protocol", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--optimization-support-manifest", required=True, type=Path)
    parser.add_argument("--optimization-support-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-pixels", type=int, default=1_048_576)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = freeze_train_hotspot_threshold(
        scene_name=args.scene,
        bound_split_path=args.bound_split,
        decode_manifest_path=args.decode_manifest,
        decode_protocol_path=args.decode_protocol,
        range_manifest_path=args.range_manifest,
        temperature_root=args.temperature_root,
        support_manifest_path=args.optimization_support_manifest,
        support_root=args.optimization_support_root,
        chunk_pixels=args.chunk_pixels,
    )
    output = write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "scene": args.scene,
                "output": str(output),
                "threshold_c": payload["threshold_c"],
                "threshold_sha256": payload["threshold_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
