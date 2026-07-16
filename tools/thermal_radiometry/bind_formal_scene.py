#!/usr/bin/env python3
"""Bind a frozen formal scene split to verified decoded Celsius maps.

The frozen split is never edited.  This tool joins it to an existing successful
TSDK decode manifest by ``(scene, pair_id)``, verifies the decoded file hashes,
and writes a derived split plus explicit train/test/guard camera lists.  The
result is the narrow bridge used by formal training and train-only range
estimation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_NAME = "uav_tgs_formal_scene_decode_binding"
SCHEMA_VERSION = 2
ALLOWED_SFM_SCOPES = ("shared_sfm_all_images", "train_only")
FORMAL_PROTOCOL_SCHEMA = "uav-tgs.radiometry-protocol.v1"
FORMAL_DECODE_SCHEMA = "uav-tgs.temperature-decode.v1"
FORMAL_ADAPTER = "builtin:dji_irp"
FORMAL_ADAPTER_BACKEND = "official-dji-irp"

# This is deliberately the protocol basis used by the manifest that actually
# produced the frozen Building Celsius maps (decode_protocol_used_v1.jsonl).
# A later regenerated protocol added robust-inlier fields and therefore has a
# different collection hash even when many resolved parameter values happen to
# be equal.  Formal binding must not silently relabel old arrays with that newer
# protocol.
FORMAL_PROTOCOL_BASIS_FIELDS = (
    "scene",
    "frame_id",
    "pair_id",
    "source_path",
    "strip_id",
    "decode_parameters",
    "raw_lrf_distance_m",
    "raw_lrf_status",
    "raw_lrf_valid",
    "used_distance_m",
    "used_distance_source",
    "distance_fallback_reason",
    "source_audit_record_hash",
    "protocol_record_hash",
)
LATER_PROTOCOL_ONLY_FIELDS = (
    "raw_lrf_robust_inlier",
    "raw_lrf_robust_outlier",
)
PARAMETER_NAMES = (
    "distance_m",
    "humidity_percent",
    "emissivity",
    "ambient_c",
    "reflected_c",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(label: str, value: Any) -> str:
    token = str(value or "").strip().lower()
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{label} must be a 64-character lowercase/uppercase SHA-256")
    return token


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _portable_path(value: Any, *, label: str) -> PurePosixPath:
    text = str(value or "").strip().replace("\\", "/")
    if not text:
        raise ValueError(f"{label} is missing")
    path = PurePosixPath(text)
    if any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} contains an unsafe path: {value!r}")
    return path


def _require_path_suffix(value: Any, suffix: Sequence[str], *, label: str) -> None:
    path = _portable_path(value, label=label)
    actual = tuple(part.casefold() for part in path.parts[-len(suffix):])
    expected = tuple(str(part).casefold() for part in suffix)
    if actual != expected:
        raise ValueError(f"{label} has suffix {actual!r}, expected {expected!r}")


def _require_mapping(label: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _parameter_values(parameters: Any, *, label: str) -> dict[str, float]:
    mapping = _require_mapping(label, parameters)
    if set(mapping) != set(PARAMETER_NAMES):
        raise ValueError(
            f"{label} must contain exactly {list(PARAMETER_NAMES)}, got {sorted(mapping)}"
        )
    result: dict[str, float] = {}
    for name in PARAMETER_NAMES:
        item = _require_mapping(f"{label}.{name}", mapping[name])
        if set(item) != {"value", "source"}:
            raise ValueError(f"{label}.{name} must contain exactly value/source")
        try:
            number = float(item["value"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}.{name}.value must be numeric") from error
        if not math.isfinite(number) or not str(item["source"]).strip():
            raise ValueError(f"{label}.{name} must have a finite value and non-empty source")
        result[name] = number
    return result


def _validate_protocol_collection(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_protocol_hash: str,
) -> str:
    basis: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if row.get("schema_version") != FORMAL_PROTOCOL_SCHEMA:
            raise ValueError(
                f"protocol row {index} schema mismatch: {row.get('schema_version')!r}"
            )
        later_fields = [field for field in LATER_PROTOCOL_ONLY_FIELDS if field in row]
        if later_fields:
            raise ValueError(
                "decode protocol is not the frozen decode_protocol_used_v1 schema; "
                f"later-only fields present in row {index}: {later_fields}"
            )
        missing = [field for field in FORMAL_PROTOCOL_BASIS_FIELDS if field not in row]
        if missing:
            raise ValueError(f"protocol row {index} is missing frozen-basis fields: {missing}")
        record_hash = _require_sha256(
            f"protocol row {index} protocol_record_hash", row.get("protocol_record_hash")
        )
        record_payload = {
            key: value
            for key, value in row.items()
            if key not in {"protocol_record_hash", "protocol_hash"}
        }
        _require_equal(
            f"protocol row {index} protocol_record_hash",
            _json_hash(record_payload),
            record_hash,
        )
        _parameter_values(row.get("decode_parameters"), label=f"protocol row {index} parameters")
        basis.append({field: row[field] for field in FORMAL_PROTOCOL_BASIS_FIELDS})

    computed = _json_hash(basis)
    expected = _require_sha256("expected decode protocol hash", expected_protocol_hash)
    _require_equal("computed decode protocol hash", computed, expected)
    for index, row in enumerate(rows, 1):
        row_hash = _require_sha256(f"protocol row {index} protocol_hash", row.get("protocol_hash"))
        _require_equal(f"protocol row {index} collection protocol hash", row_hash, computed)
    return computed


def _validate_decode_lineage(
    *,
    scene: str,
    pair_id: str,
    decode: Mapping[str, Any],
    protocol: Mapping[str, Any],
    decode_manifest_path: Path,
    raw_thermal_root: Path,
    expected_adapter: str,
    expected_adapter_executable_sha256: str,
) -> dict[str, Any]:
    label = f"{scene}/{pair_id}"
    _require_equal(f"decode schema for {label}", decode.get("schema_version"), FORMAL_DECODE_SCHEMA)
    for field, expected in (
        ("scene", scene),
        ("pair_id", pair_id),
        ("frame_id", str(protocol.get("frame_id", ""))),
        ("strip_id", str(protocol.get("strip_id", ""))),
    ):
        _require_equal(f"decode/protocol {field} for {label}", str(decode.get(field, "")), str(expected))

    protocol_parameters = protocol.get("decode_parameters")
    protocol_parameter_values = _parameter_values(
        protocol_parameters, label=f"protocol parameters for {label}"
    )
    _require_equal(
        f"decode parameters for {label}",
        decode.get("parameters"),
        protocol_parameters,
    )
    _require_equal(
        f"decode source path for {label}",
        decode.get("source_path"),
        protocol.get("source_path"),
    )
    _require_equal(
        f"decode metadata for {label}",
        decode.get("metadata"),
        protocol.get("metadata"),
    )
    _require_equal(f"decode adapter for {label}", decode.get("adapter"), expected_adapter)

    request_filename = f"{scene}--{pair_id}.json"
    request_path = decode_manifest_path.parent / "decode_requests" / request_filename
    if not request_path.is_file():
        raise FileNotFoundError(f"decode request is missing for {label}: {request_path}")
    _require_path_suffix(
        decode.get("request_path"),
        ("decode_requests", request_filename),
        label=f"decode request_path for {label}",
    )
    request = _load_json(request_path)
    for key, request_value in request.items():
        _require_equal(f"decode/request {key} for {label}", decode.get(key), request_value)
    _require_equal(f"request schema for {label}", request.get("schema_version"), FORMAL_DECODE_SCHEMA)
    _require_equal(f"request parameters for {label}", request.get("parameters"), protocol_parameters)
    _require_equal(f"request source path for {label}", request.get("source_path"), protocol.get("source_path"))
    _require_equal(f"request metadata for {label}", request.get("metadata"), protocol.get("metadata"))
    _require_equal(f"request adapter for {label}", request.get("adapter"), expected_adapter)
    _require_equal(
        f"decode output/temperature path for {label}",
        decode.get("output_path"),
        decode.get("temperature_npy"),
    )
    _require_path_suffix(
        decode.get("temperature_npy"),
        ("temperature_c", scene, f"{pair_id}.npy"),
        label=f"decode temperature_npy for {label}",
    )

    protocol_source = _portable_path(protocol.get("source_path"), label=f"protocol source_path for {label}")
    if protocol_source.stem != pair_id:
        raise ValueError(
            f"protocol source stem for {label} is {protocol_source.stem!r}, expected {pair_id!r}"
        )
    raw_path = raw_thermal_root / protocol_source.name
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw R-JPEG is missing for {label}: {raw_path}")
    raw_sha = _sha256(raw_path)
    expected_raw_sha = _require_sha256(f"decode source SHA-256 for {label}", decode.get("source_sha256"))
    _require_equal(f"raw R-JPEG SHA-256 for {label}", raw_sha, expected_raw_sha)
    _require_equal(
        f"raw R-JPEG size for {label}",
        int(raw_path.stat().st_size),
        int(decode.get("source_size_bytes", -1)),
    )

    diagnostics = _require_mapping(
        f"adapter diagnostics for {label}", decode.get("adapter_diagnostics")
    )
    _require_equal(
        f"adapter backend for {label}", diagnostics.get("backend"), FORMAL_ADAPTER_BACKEND
    )
    executable_sha = _require_sha256(
        f"adapter executable SHA-256 for {label}", diagnostics.get("executable_sha256")
    )
    _require_equal(
        f"adapter executable SHA-256 for {label}",
        executable_sha,
        expected_adapter_executable_sha256,
    )
    applied = _require_mapping(
        f"adapter parameters_applied for {label}", diagnostics.get("parameters_applied")
    )
    if set(applied) != set(PARAMETER_NAMES):
        raise ValueError(
            f"adapter parameters_applied for {label} must contain exactly {list(PARAMETER_NAMES)}"
        )
    for name, expected_value in protocol_parameter_values.items():
        try:
            actual_value = float(applied[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"adapter applied parameter {name} for {label} is not numeric") from error
        _require_equal(f"adapter applied parameter {name} for {label}", actual_value, expected_value)
    resolution = _require_mapping(f"adapter resolution for {label}", diagnostics.get("resolution"))
    _require_equal(f"adapter width for {label}", int(resolution.get("width", -1)), 1280)
    _require_equal(f"adapter height for {label}", int(resolution.get("height", -1)), 1024)
    for field in ("dirp_api_version", "rjpeg_version"):
        if not str(diagnostics.get(field, "")).strip():
            raise ValueError(f"adapter diagnostics {field} is missing for {label}")

    return {
        "raw_relative_path": raw_path.relative_to(raw_thermal_root).as_posix(),
        "raw_sha256": raw_sha,
        "raw_size_bytes": int(raw_path.stat().st_size),
        "request_relative_path": request_path.relative_to(decode_manifest_path.parent).as_posix(),
        "request_sha256": _sha256(request_path),
        "protocol_record_hash": str(protocol["protocol_record_hash"]),
        "adapter": expected_adapter,
        "adapter_backend": FORMAL_ADAPTER_BACKEND,
        "adapter_executable_sha256": executable_sha,
        "dirp_api_version": str(diagnostics["dirp_api_version"]),
        "rjpeg_version": str(diagnostics["rjpeg_version"]),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL manifest is empty: {path}")
    return rows


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _unique_index(
    rows: Iterable[Mapping[str, Any]], *, scene: str, label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("scene", "")) != scene:
            raise ValueError(
                f"{label} manifest contains a row outside scene {scene!r}: "
                f"{row.get('scene')!r}"
            )
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"{label} record is missing pair_id")
        if pair_id in index:
            raise ValueError(f"duplicate {label} pair_id for {scene}: {pair_id}")
        index[pair_id] = row
    return index


def bind_formal_scene(
    *,
    scene_manifest_path: Path,
    collection_manifest_path: Path,
    decode_manifest_path: Path,
    decode_protocol_path: Path,
    temperature_root: Path,
    raw_thermal_root: Path,
    scene: str,
    sfm_image_scope: str,
    expected_collection_manifest_sha256: str,
    expected_collection_hash: str,
    expected_collection_split_hash: str,
    expected_formal_rule_hash: str,
    expected_scene_manifest_sha256: str,
    expected_scene_split_hash: str,
    expected_scene_rule_hash: str,
    expected_decode_protocol_hash: str,
    expected_adapter_executable_sha256: str,
    expected_adapter: str = FORMAL_ADAPTER,
    camera_extension: str = ".JPG",
    thermal_camera_extension: str = ".png",
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    scene_manifest_path = scene_manifest_path.resolve()
    collection_manifest_path = collection_manifest_path.resolve()
    decode_manifest_path = decode_manifest_path.resolve()
    decode_protocol_path = decode_protocol_path.resolve()
    temperature_root = temperature_root.resolve()
    raw_thermal_root = raw_thermal_root.resolve()
    if sfm_image_scope not in ALLOWED_SFM_SCOPES:
        raise ValueError(f"unsupported sfm_image_scope: {sfm_image_scope}")
    if not raw_thermal_root.is_dir():
        raise FileNotFoundError(f"raw thermal root does not exist: {raw_thermal_root}")
    if expected_adapter != FORMAL_ADAPTER:
        raise ValueError(
            f"formal decode binding only accepts adapter {FORMAL_ADAPTER!r}, got {expected_adapter!r}"
        )
    expected_adapter_executable_sha256 = _require_sha256(
        "expected adapter executable SHA-256", expected_adapter_executable_sha256
    )
    expected_collection_manifest_sha256 = _require_sha256(
        "expected collection manifest SHA-256", expected_collection_manifest_sha256
    )
    expected_scene_manifest_sha256 = _require_sha256(
        "expected scene manifest SHA-256", expected_scene_manifest_sha256
    )
    expected_collection_hash = _require_sha256("expected collection hash", expected_collection_hash)
    expected_collection_split_hash = _require_sha256(
        "expected collection split hash", expected_collection_split_hash
    )
    expected_formal_rule_hash = _require_sha256(
        "expected formal rule hash", expected_formal_rule_hash
    )
    expected_scene_split_hash = _require_sha256(
        "expected scene split hash", expected_scene_split_hash
    )
    expected_scene_rule_hash = _require_sha256(
        "expected scene rule hash", expected_scene_rule_hash
    )
    expected_decode_protocol_hash = _require_sha256(
        "expected decode protocol hash", expected_decode_protocol_hash
    )
    for label, extension in (
        ("camera_extension", camera_extension),
        ("thermal_camera_extension", thermal_camera_extension),
    ):
        if not extension.startswith(".") or "/" in extension or "\\" in extension:
            raise ValueError(f"{label} must be a simple extension such as .JPG or .png")

    scene_manifest_sha = _sha256(scene_manifest_path)
    collection_manifest_sha = _sha256(collection_manifest_path)
    _require_equal(
        "collection manifest SHA-256",
        collection_manifest_sha,
        expected_collection_manifest_sha256,
    )
    _require_equal(
        "scene manifest SHA-256",
        scene_manifest_sha,
        expected_scene_manifest_sha256,
    )
    scene_manifest = _load_json(scene_manifest_path)
    collection = _load_json(collection_manifest_path)
    _require_equal("collection hash", collection.get("collection_hash"), expected_collection_hash)
    _require_equal(
        "collection split hash",
        collection.get("collection_split_hash"),
        expected_collection_split_hash,
    )
    _require_equal(
        "formal rule hash", collection.get("formal_rule_hash"), expected_formal_rule_hash
    )
    _require_equal(
        "scene split hash", scene_manifest.get("split_hash"), expected_scene_split_hash
    )
    _require_equal(
        "scene rule hash", scene_manifest.get("rule_hash"), expected_scene_rule_hash
    )
    if scene_manifest.get("scene") != scene:
        raise ValueError("scene manifest scene does not match --scene")
    records = scene_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("scene manifest must contain non-empty records")
    if scene_manifest.get("validation", {}).get("status") != "passed":
        raise ValueError("frozen scene split validation is not passed")
    if collection.get("validation", {}).get("status") != "passed":
        raise ValueError("formal collection validation is not passed")

    collection_scene_rows = [row for row in collection.get("scenes", []) if row.get("scene") == scene]
    if len(collection_scene_rows) != 1:
        raise ValueError(f"formal collection must contain exactly one {scene} row")
    collection_scene = collection_scene_rows[0]
    scene_file_sha = scene_manifest_sha
    if scene_file_sha != collection_scene.get("manifest_sha256"):
        raise ValueError("scene manifest SHA does not match frozen collection")
    if scene_manifest.get("split_hash") != collection_scene.get("split_hash"):
        raise ValueError("scene split hash does not match frozen collection")
    if scene_manifest.get("rule_hash") != collection_scene.get("rule_hash"):
        raise ValueError("scene rule hash does not match frozen collection")

    split_counts = Counter(str(record.get("split", "")) for record in records)
    expected_counts = dict(collection_scene.get("counts", {}))
    actual_counts = {
        "total": len(records),
        "train": split_counts["train"],
        "test": split_counts["test"],
        "guard": split_counts["guard"],
    }
    if split_counts.keys() - {"train", "test", "guard"}:
        raise ValueError(f"unknown split labels: {sorted(split_counts.keys())}")
    if actual_counts != expected_counts:
        raise ValueError(f"formal split counts mismatch: {actual_counts} != {expected_counts}")

    pair_ids = [str(record.get("pair_id", "")).strip() for record in records]
    if any(not value for value in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("formal scene pair IDs must be non-empty and unique")

    decode_rows = _load_jsonl(decode_manifest_path)
    protocol_rows = _load_jsonl(decode_protocol_path)
    protocol_collection_hash = _validate_protocol_collection(
        protocol_rows,
        expected_protocol_hash=expected_decode_protocol_hash,
    )
    decode_index = _unique_index(decode_rows, scene=scene, label="decode")
    protocol_index = _unique_index(protocol_rows, scene=scene, label="protocol")
    expected_pairs = set(pair_ids)
    if set(decode_index) != expected_pairs:
        raise ValueError(
            f"decode/formal pair mismatch: missing={sorted(expected_pairs-set(decode_index))[:8]} "
            f"extra={sorted(set(decode_index)-expected_pairs)[:8]}"
        )
    if set(protocol_index) != expected_pairs:
        raise ValueError(
            f"protocol/formal pair mismatch: missing={sorted(expected_pairs-set(protocol_index))[:8]} "
            f"extra={sorted(set(protocol_index)-expected_pairs)[:8]}"
        )

    resolved_records: list[dict[str, Any]] = []
    lists = {
        "train": [], "test": [], "guard": [],
        "thermal_train": [], "thermal_test": [], "thermal_guard": [],
    }
    file_rows: list[dict[str, Any]] = []
    protocol_hashes: set[str] = set()
    dirp_api_versions: set[str] = set()
    rjpeg_versions: set[str] = set()
    for record in records:
        pair_id = str(record["pair_id"])
        split = str(record["split"])
        decode = decode_index[pair_id]
        protocol = protocol_index[pair_id]
        if decode.get("success") is not True:
            raise ValueError(f"decode was not successful for {scene}/{pair_id}")
        if record.get("source_record_hash") != protocol.get("source_audit_record_hash"):
            raise ValueError(f"formal/protocol source record hash mismatch for {pair_id}")
        if record.get("strip_id") != protocol.get("strip_id"):
            raise ValueError(f"formal/protocol strip mismatch for {pair_id}")
        protocol_hash = str(protocol.get("protocol_hash", ""))
        if not protocol_hash:
            raise ValueError(f"protocol hash is missing for {pair_id}")
        protocol_hashes.add(protocol_hash)

        lineage = _validate_decode_lineage(
            scene=scene,
            pair_id=pair_id,
            decode=decode,
            protocol=protocol,
            decode_manifest_path=decode_manifest_path,
            raw_thermal_root=raw_thermal_root,
            expected_adapter=expected_adapter,
            expected_adapter_executable_sha256=expected_adapter_executable_sha256,
        )
        dirp_api_versions.add(str(lineage["dirp_api_version"]))
        rjpeg_versions.add(str(lineage["rjpeg_version"]))

        npy_path = temperature_root / scene / f"{pair_id}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(npy_path)
        if str(decode.get("dtype")) != "float32" or list(decode.get("shape_hw", [])) != [1024, 1280]:
            raise ValueError(f"unexpected decode dtype/shape for {pair_id}")
        array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype("float32") or array.shape != (1024, 1280):
            raise ValueError(f"temperature file dtype/shape mismatch for {pair_id}: {array.dtype} {array.shape}")
        actual_sha = _sha256(npy_path)
        expected_sha = _require_sha256(
            f"decode output SHA-256 for {scene}/{pair_id}", decode.get("output_sha256")
        )
        if actual_sha != expected_sha:
            raise ValueError(f"temperature file SHA mismatch for {pair_id}")

        camera_name = f"{pair_id}{camera_extension}"
        thermal_camera_name = f"{pair_id}{thermal_camera_extension}"
        lists[split].append(camera_name)
        lists[f"thermal_{split}"].append(thermal_camera_name)
        resolved = dict(record)
        resolved["temperature_npy"] = f"{scene}/{pair_id}.npy"
        resolved["camera_name"] = camera_name
        resolved["thermal_camera_name"] = thermal_camera_name
        resolved_records.append(resolved)
        file_rows.append(
            {
                "pair_id": pair_id,
                "split": split,
                "camera_name": camera_name,
                "thermal_camera_name": thermal_camera_name,
                "temperature_npy": f"{scene}/{pair_id}.npy",
                "temperature_sha256": expected_sha,
                "verified_sha256": actual_sha,
                "raw_thermal": lineage,
            }
        )

    if len(protocol_hashes) != 1:
        raise ValueError(f"expected exactly one decode protocol hash, got {sorted(protocol_hashes)}")
    _require_equal(
        "validated collection protocol hash",
        next(iter(protocol_hashes)),
        protocol_collection_hash,
    )
    if len(dirp_api_versions) != 1 or len(rjpeg_versions) != 1:
        raise ValueError(
            "formal decode collection must use one DIRP API and one R-JPEG version: "
            f"DIRP={sorted(dirp_api_versions)}, R-JPEG={sorted(rjpeg_versions)}"
        )
    for names in lists.values():
        names.sort()
    if set(lists["train"]) & set(lists["test"]):
        raise ValueError("derived train/test lists overlap")
    if (set(lists["train"]) | set(lists["test"])) & set(lists["guard"]):
        raise ValueError("derived guard list overlaps train/test")
    if set(lists["thermal_train"]) & set(lists["thermal_test"]):
        raise ValueError("derived thermal train/test lists overlap")
    if (set(lists["thermal_train"]) | set(lists["thermal_test"])) & set(lists["thermal_guard"]):
        raise ValueError("derived thermal guard list overlaps train/test")

    bound_split = dict(scene_manifest)
    bound_split["records"] = resolved_records
    bound_split["formal_source_manifest"] = str(scene_manifest_path)
    bound_split["formal_source_manifest_sha256"] = scene_file_sha
    bound_split["decode_binding"] = {
        "decode_manifest_sha256": _sha256(decode_manifest_path),
        "decode_protocol_sha256": _sha256(decode_protocol_path),
        "protocol_hash": protocol_collection_hash,
        "temperature_root": str(temperature_root),
        "raw_thermal_root": str(raw_thermal_root),
        "verified_temperature_file_hashes": True,
        "verified_raw_rjpeg_hashes": True,
        "verified_decode_requests": True,
        "adapter": expected_adapter,
        "adapter_backend": FORMAL_ADAPTER_BACKEND,
        "adapter_executable_sha256": expected_adapter_executable_sha256,
        "dirp_api_version": next(iter(dirp_api_versions)),
        "rjpeg_version": next(iter(rjpeg_versions)),
    }
    binding_basis = {
        "collection_hash": collection.get("collection_hash"),
        "collection_split_hash": collection.get("collection_split_hash"),
        "formal_rule_hash": collection.get("formal_rule_hash"),
        "scene_split_hash": scene_manifest.get("split_hash"),
        "scene_rule_hash": scene_manifest.get("rule_hash"),
        "scene_manifest_sha256": scene_file_sha,
        "collection_manifest_sha256": collection_manifest_sha,
        "decode_manifest_sha256": _sha256(decode_manifest_path),
        "decode_protocol_sha256": _sha256(decode_protocol_path),
        "decode_protocol_hash": protocol_collection_hash,
        "formal_protocol_schema": FORMAL_PROTOCOL_SCHEMA,
        "formal_decode_schema": FORMAL_DECODE_SCHEMA,
        "adapter": expected_adapter,
        "adapter_backend": FORMAL_ADAPTER_BACKEND,
        "adapter_executable_sha256": expected_adapter_executable_sha256,
        "dirp_api_version": next(iter(dirp_api_versions)),
        "rjpeg_version": next(iter(rjpeg_versions)),
        "sfm_image_scope": sfm_image_scope,
        "counts": actual_counts,
        "files": file_rows,
    }
    binding_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        **binding_basis,
        "collection_manifest": str(collection_manifest_path),
        "collection_manifest_sha256": collection_manifest_sha,
        "scene": scene,
        "camera_extension": camera_extension,
        "thermal_camera_extension": thermal_camera_extension,
        "binding_hash": _json_hash(binding_basis),
    }
    return bound_split, lists, binding_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", required=True, type=Path)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--decode-manifest", required=True, type=Path)
    parser.add_argument("--decode-protocol", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument(
        "--raw-thermal-root",
        required=True,
        type=Path,
        help="Directory containing this scene's original R-JPEG files directly",
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sfm-image-scope", required=True, choices=ALLOWED_SFM_SCOPES)
    parser.add_argument("--expected-collection-manifest-sha256", required=True)
    parser.add_argument("--expected-collection-hash", required=True)
    parser.add_argument("--expected-collection-split-hash", required=True)
    parser.add_argument("--expected-formal-rule-hash", required=True)
    parser.add_argument("--expected-scene-manifest-sha256", required=True)
    parser.add_argument("--expected-scene-split-hash", required=True)
    parser.add_argument("--expected-scene-rule-hash", required=True)
    parser.add_argument("--expected-decode-protocol-hash", required=True)
    parser.add_argument("--expected-adapter", default=FORMAL_ADAPTER, choices=(FORMAL_ADAPTER,))
    parser.add_argument("--expected-adapter-executable-sha256", required=True)
    parser.add_argument("--camera-extension", default=".JPG")
    parser.add_argument("--thermal-camera-extension", default=".png")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.resolve()
    outputs = [
        output_root / "bound_split.json",
        output_root / "train_list.txt",
        output_root / "test_list.txt",
        output_root / "guard_list.txt",
        output_root / "thermal_train_list.txt",
        output_root / "thermal_test_list.txt",
        output_root / "thermal_guard_list.txt",
        output_root / "binding_manifest.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing binding outputs: {existing[:3]}")
    bound, lists, manifest = bind_formal_scene(
        scene_manifest_path=args.scene_manifest,
        collection_manifest_path=args.collection_manifest,
        decode_manifest_path=args.decode_manifest,
        decode_protocol_path=args.decode_protocol,
        temperature_root=args.temperature_root,
        raw_thermal_root=args.raw_thermal_root,
        scene=args.scene,
        sfm_image_scope=args.sfm_image_scope,
        expected_collection_manifest_sha256=args.expected_collection_manifest_sha256,
        expected_collection_hash=args.expected_collection_hash,
        expected_collection_split_hash=args.expected_collection_split_hash,
        expected_formal_rule_hash=args.expected_formal_rule_hash,
        expected_scene_manifest_sha256=args.expected_scene_manifest_sha256,
        expected_scene_split_hash=args.expected_scene_split_hash,
        expected_scene_rule_hash=args.expected_scene_rule_hash,
        expected_decode_protocol_hash=args.expected_decode_protocol_hash,
        expected_adapter=args.expected_adapter,
        expected_adapter_executable_sha256=args.expected_adapter_executable_sha256,
        camera_extension=args.camera_extension,
        thermal_camera_extension=args.thermal_camera_extension,
    )
    _atomic_json(outputs[0], bound)
    for path, label in zip(
        outputs[1:7],
        ("train", "test", "guard", "thermal_train", "thermal_test", "thermal_guard"),
    ):
        _atomic_text(path, "".join(f"{name}\n" for name in lists[label]))
    manifest = dict(manifest)
    manifest["outputs"] = {
        path.name: {"path": str(path), "sha256": _sha256(path)} for path in outputs[:7]
    }
    _atomic_json(outputs[7], manifest)
    print(json.dumps({"status": "passed", "binding_manifest": str(outputs[7])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
