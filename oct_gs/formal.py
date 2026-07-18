"""Fail-closed formal data binding and target access for OCT-GS."""

from __future__ import annotations

from collections import OrderedDict, Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from tools.thermal_radiometry.palette_lut import lut_sha256
from utils.camera_sequence import validate_sequence_manifest

from .field import OCTConfig, OCT_VARIANTS
from .radiance import BandRadianceProxy, METHOD_SEMANTICS, TARGET_SEMANTICS


FORMAL_BINDING_SCHEMA = "uav-tgs-oct-formal-binding-v1"
FORMAL_EXPERIMENT_RECIPE = {
    "schema": "uav-tgs-oct-paired-recipe-v1",
    "variants": ["oct_scalar", "oct_residual"],
    "thermometric_domain": "celsius",
    "residual_bound_fraction": 0.05,
    "temperature_lr": 1.0e-2,
    "residual_lr": 2.5e-3,
    "adam_eps": 1.0e-15,
    "schedule": "constant",
    "steps": 30_000,
    "endpoints": [10_000, 20_000, 30_000],
    "resolution": -1,
    "sequence_seed": 0,
    "background_temperature_policy": "formal_scene_Tmin",
}
_RANGE_SCHEMA = "uav_tgs_train_only_scene_temperature_range"
_CANONICAL_SCHEMA = "uav-tgs-canonical-hot-iron-v1"
_OPTIMIZATION_SUPPORT_SCHEMA = "uav-tgs-undistorted-temperature-v1"
_EVALUATION_SUPPORT_SCHEMA = "uav-tgs-formal-temperature-support"
_EVALUATION_SUPPORT_POLICY = {
    "expression": "valid_support AND (opacity_proxy > opacity_threshold)",
    "opacity_threshold": 0.01,
    "comparison": "strict_greater_than",
    "opacity_proxy_semantics": "black_bg_plus_white_override_color_render",
    "threshold_applied_only_by_this_combiner": True,
}
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
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {source}")
    return value


def load_jsonl_objects(path: str | Path, label: str) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty: {source}")
    return rows


def _require_sha(value: Any, label: str) -> str:
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return text


def _file_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _close(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= max(
        1e-7, 1e-7 * max(abs(float(left)), abs(float(right)))
    )


def _optimization_support_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if payload.get("schema") != _OPTIMIZATION_SUPPORT_SCHEMA:
        raise ValueError("optimization support must be the bound undistorted-temperature manifest")
    if payload.get("status") != "complete":
        raise ValueError("optimization support manifest is not complete")
    value = payload.get("files")
    if not isinstance(value, list) or not value:
        raise ValueError("optimization support manifest must contain files")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError("optimization support records must be objects")
    return list(value)


def _evaluation_support_rows(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
    bound_split_sha256: str,
    optimization_support_sha256: str,
) -> list[Mapping[str, Any]]:
    if payload.get("schema_name") != _EVALUATION_SUPPORT_SCHEMA:
        raise ValueError("evaluation support has an unsupported schema")
    if payload.get("schema_version") != 1 or payload.get("split") != "test":
        raise ValueError("evaluation support must be schema v1 for the test split")
    if int(payload.get("expected_test_count", -1)) != int(expected_count):
        raise ValueError("evaluation support test count mismatch")
    if payload.get("policy") != _EVALUATION_SUPPORT_POLICY:
        raise ValueError("evaluation support policy differs from the formal F3/legacy policy")
    sources = payload.get("source_manifests")
    if not isinstance(sources, Mapping):
        raise ValueError("evaluation support lacks source provenance")
    split_source = sources.get("split")
    valid_source = sources.get("valid_support")
    if not isinstance(split_source, Mapping) or not isinstance(valid_source, Mapping):
        raise ValueError("evaluation support source provenance is incomplete")
    if split_source.get("sha256") != bound_split_sha256:
        raise ValueError("evaluation support is bound to another formal split")
    if valid_source.get("sha256") != optimization_support_sha256:
        raise ValueError("evaluation support is bound to another valid-support manifest")
    value = payload.get("records")
    if not isinstance(value, list) or len(value) != int(expected_count):
        raise ValueError("evaluation support records/count mismatch")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError("evaluation support records must be objects")
    return list(value)


def _identity(record: Mapping[str, Any]) -> str:
    for key in ("pair_id", "relative_id", "image_name", "thermal_camera_name", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).stem
    raise ValueError("formal record has no usable pair identity")


def _support_relative(record: Mapping[str, Any]) -> str:
    nested = record.get("valid_support")
    if isinstance(nested, Mapping):
        value = nested.get("relative_path", nested.get("path"))
        if isinstance(value, str) and value.strip():
            return value
    for key in ("support_npy", "relative_path", "mask_path", "path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if key in ("mask_path", "path") and path.is_absolute():
                return path.name
            return value
    raise ValueError(f"support record {_identity(record)!r} has no path")


def _support_sha(record: Mapping[str, Any]) -> str:
    nested = record.get("valid_support")
    if isinstance(nested, Mapping) and nested.get("sha256"):
        return _require_sha(
            nested["sha256"], f"support {_identity(record)} valid_support.sha256"
        )
    for key in ("support_sha256", "mask_sha256", "sha256"):
        if record.get(key):
            return _require_sha(record[key], f"support {_identity(record)} {key}")
    raise ValueError(f"support record {_identity(record)!r} has no SHA-256")


def _evaluation_support_relative(record: Mapping[str, Any]) -> str:
    outputs = record.get("outputs")
    binary = outputs.get("bool") if isinstance(outputs, Mapping) else None
    if not isinstance(binary, Mapping) or binary.get("dtype") != "bool":
        raise ValueError(f"evaluation support {_identity(record)!r} lacks a bool output")
    relative = binary.get("relative_path")
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"evaluation support {_identity(record)!r} lacks a bool path")
    if Path(relative).parts[:1] != ("bool",):
        raise ValueError("evaluation support bool output must live under bool/")
    return relative


def _evaluation_support_sha(record: Mapping[str, Any]) -> str:
    outputs = record.get("outputs")
    binary = outputs.get("bool") if isinstance(outputs, Mapping) else None
    if not isinstance(binary, Mapping):
        raise ValueError(f"evaluation support {_identity(record)!r} lacks a bool output")
    return _require_sha(binary.get("sha256"), f"evaluation support {_identity(record)}")


def _resolve_under(root: Path, relative: str, *, fallback_id: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        # Formal manifests are frequently transferred across operating systems;
        # only their relative identity is portable.  The explicit root remains
        # authoritative at runtime.
        candidate = Path(candidate.name)
    direct = (root / candidate).resolve()
    if direct.is_file():
        return direct
    matches = [path.resolve() for path in root.rglob(candidate.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    stem_matches = [path.resolve() for path in root.rglob(f"{fallback_id}.*") if path.is_file()]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError(f"cannot uniquely resolve {relative!r} under {root}")


def _validate_target_array(path: Path, expected_sha: str, verify_hash: bool) -> tuple[int, int]:
    if verify_hash and sha256_file(path) != expected_sha:
        raise ValueError(f"temperature target SHA mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("float32") or array.ndim != 2 or array.size == 0:
        raise TypeError(f"temperature target must be non-empty float32 HW: {path}")
    if verify_hash and not bool(np.isfinite(array).all()):
        raise ValueError(f"temperature target contains NaN/Inf: {path}")
    return int(array.shape[0]), int(array.shape[1])


def _validate_canonical_image(
    path: Path, expected_sha: str, shape_hw: tuple[int, int], verify_hash: bool
) -> None:
    if verify_hash and sha256_file(path) != expected_sha:
        raise ValueError(f"canonical target SHA mismatch: {path}")
    with Image.open(path) as image:
        if image.mode != "RGB" or (image.height, image.width) != shape_hw:
            raise ValueError(f"canonical target must be RGB and match temperature shape: {path}")


def _validate_support(path: Path, expected_sha: str, shape_hw: tuple[int, int], verify_hash: bool) -> str:
    if verify_hash and sha256_file(path) != expected_sha:
        raise ValueError(f"support SHA mismatch: {path}")
    if path.suffix.casefold() == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype("bool") or value.ndim != 2:
            raise TypeError(f"NPY support must be boolean HW: {path}")
        if tuple(value.shape) != shape_hw:
            raise ValueError(f"support/target shape mismatch: {path}")
        return "bool-npy"
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8)
    if tuple(value.shape) != shape_hw:
        raise ValueError(f"support/target shape mismatch: {path}")
    unique = set(int(item) for item in np.unique(value))
    if not unique <= {0, 255}:
        raise ValueError(f"image support must contain only 0/255: {path}")
    return "binary-image-0-255"


@dataclass(frozen=True)
class FormalOCTRecord:
    pair_id: str
    split: str
    camera_name: str
    temperature_path: Path
    temperature_sha256: str
    canonical_path: Path
    canonical_sha256: str
    support_path: Path
    support_sha256: str
    support_encoding: str
    evaluation_support_path: Path | None
    evaluation_support_sha256: str | None
    evaluation_support_encoding: str | None
    shape_hw: tuple[int, int]


class FormalOCTBinding:
    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        records: Sequence[FormalOCTRecord],
    ) -> None:
        self.payload = dict(payload)
        self.records = tuple(records)
        self.by_camera = {record.camera_name: record for record in records}
        if len(self.by_camera) != len(self.records):
            raise ValueError("formal OCT camera names must be unique")

    @property
    def formal_protocol_sha256(self) -> str:
        return str(self.payload["formal_protocol_sha256"])

    @property
    def scene_name(self) -> str:
        return str(self.payload["scene_name"])

    @property
    def tmin_c(self) -> float:
        return float(self.payload["temperature_range"]["tmin_c"])

    @property
    def tmax_c(self) -> float:
        return float(self.payload["temperature_range"]["tmax_c"])

    def names(self, split: str) -> list[str]:
        return sorted(
            record.camera_name for record in self.records if record.split == split
        )

    def immutable_summary(self) -> dict[str, Any]:
        keys = (
            "schema",
            "formal_protocol_sha256",
            "scene_name",
            "bound_split",
            "temperature_range",
            "tsdk_target",
            "canonical_target",
            "support",
            "camera_sequence",
            "anchor",
            "field_configs",
            "radiance_proxy",
            "experiment_recipe",
        )
        return {key: self.payload[key] for key in keys}

    def calibration_receipt(self) -> dict[str, Any]:
        """Portable Building-train receipt pinned into loss calibration."""

        train_names = self.names("train")
        receipt = {
            "schema": "uav-tgs-oct-calibration-source-receipt-v1",
            "scene_name": self.scene_name,
            "formal_protocol_sha256": self.formal_protocol_sha256,
            "bound_split_sha256": self.payload["bound_split"]["sha256"],
            "decode_manifest_sha256": self.payload["tsdk_target"]["decode_manifest_sha256"],
            "decode_protocol_sha256": self.payload["tsdk_target"]["decode_protocol_sha256"],
            "tsdk_protocol_hash": self.payload["tsdk_target"]["protocol_hash"],
            "range_manifest_sha256": self.payload["temperature_range"]["sha256"],
            "canonical_manifest_sha256": self.payload["canonical_target"]["manifest_sha256"],
            "target_index_sha256": self.payload["canonical_target"]["target_index_sha256"],
            "optimization_support_manifest_sha256": self.payload["support"]["optimization"]["manifest_sha256"],
            "evaluation_support_manifest_sha256": self.payload["support"]["evaluation"]["manifest_sha256"],
            "anchor_artifact_sha256": self.payload["anchor"]["artifact_sha256"],
            "anchor_occupancy_sha256": self.payload["anchor"]["occupancy_sha256"],
            "camera_sequence_manifest_sha256": self.payload["camera_sequence"]["manifest_sha256"],
            "camera_sequence_sha256": self.payload["camera_sequence"]["sequence_sha256"],
            "camera_parameters_sha256": self.payload["camera_sequence"]["camera_parameters_sha256"],
            "field_configs_sha256": self.payload["field_configs"]["sha256"],
            "field_configs": self.payload["field_configs"]["payload"],
            "experiment_recipe_sha256": self.payload["experiment_recipe"]["sha256"],
            "train_view_ids": train_names,
            "train_view_ids_sha256": sha256_json(train_names),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt

    def hotspot_receipt(self) -> dict[str, Any]:
        """Train-only radiometry receipt independent of model anchor/sequence."""

        train_names = self.names("train")
        receipt = {
            "schema": "uav-tgs-oct-hotspot-source-receipt-v1",
            "scene_name": self.scene_name,
            "bound_split_sha256": self.payload["bound_split"]["sha256"],
            "split_hash": self.payload["bound_split"]["split_hash"],
            "decode_manifest_sha256": self.payload["tsdk_target"]["decode_manifest_sha256"],
            "decode_protocol_sha256": self.payload["tsdk_target"]["decode_protocol_sha256"],
            "range_manifest_sha256": self.payload["temperature_range"]["sha256"],
            "range_hash": self.payload["temperature_range"]["range_hash"],
            "optimization_support_manifest_sha256": self.payload["support"]["optimization"]["manifest_sha256"],
            "optimization_support_index_sha256": self.payload["support"]["optimization"]["support_index_sha256"],
            "train_view_ids_sha256": sha256_json(train_names),
        }
        receipt["receipt_sha256"] = sha256_json(receipt)
        return receipt

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"formal binding already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return destination


def build_formal_binding(
    *,
    scene_name: str,
    bound_split_path: str | Path,
    decode_manifest_path: str | Path,
    decode_protocol_path: str | Path,
    range_manifest_path: str | Path,
    canonical_manifest_path: str | Path,
    temperature_root: str | Path,
    canonical_root: str | Path,
    support_manifest_path: str | Path,
    support_root: str | Path,
    evaluation_support_manifest_path: str | Path,
    evaluation_support_root: str | Path,
    camera_sequence_path: str | Path,
    camera_parameters_sha256: str,
    anchor_artifact_path: str | Path,
    anchor_snapshot: Mapping[str, Any],
    field_configs: Mapping[str, OCTConfig],
    radiance_proxy: BandRadianceProxy,
    verify_payload_files: bool = True,
) -> FormalOCTBinding:
    scene = str(scene_name)
    if scene not in ("Building", "InternalRoad"):
        raise ValueError("formal OCT v1 is restricted to Building and InternalRoad")
    if set(field_configs) != set(OCT_VARIANTS):
        raise ValueError("formal OCT binding must pin both v1 variants")
    for name, config in field_configs.items():
        config.validate()
        if config.variant != name:
            raise ValueError("field config key/variant mismatch")
        if not _close(
            config.residual_bound_fraction,
            FORMAL_EXPERIMENT_RECIPE["residual_bound_fraction"],
        ):
            raise ValueError("field residual bound differs from the frozen paired recipe")
    bound_path = Path(bound_split_path).resolve()
    decode_manifest_path = Path(decode_manifest_path).resolve()
    decode_protocol_path = Path(decode_protocol_path).resolve()
    range_path = Path(range_manifest_path).resolve()
    canonical_manifest_path = Path(canonical_manifest_path).resolve()
    support_manifest_path = Path(support_manifest_path).resolve()
    evaluation_support_manifest_path = Path(evaluation_support_manifest_path).resolve()
    sequence_path = Path(camera_sequence_path).resolve()
    anchor_artifact_path = Path(anchor_artifact_path).resolve()
    input_records = {
        "bound_split": _file_record(bound_path),
        "decode_manifest": _file_record(decode_manifest_path),
        "decode_protocol": _file_record(decode_protocol_path),
        "range_manifest": _file_record(range_path),
        "canonical_manifest": _file_record(canonical_manifest_path),
        "support_manifest": _file_record(support_manifest_path),
        "evaluation_support_manifest": _file_record(evaluation_support_manifest_path),
        "camera_sequence": _file_record(sequence_path),
        "anchor_artifact": _file_record(anchor_artifact_path),
    }
    bound = load_json_object(bound_path, "bound split")
    if bound.get("scene") != scene or not isinstance(bound.get("records"), list):
        raise ValueError("bound split scene/records mismatch")
    split_hash = _require_sha(bound.get("split_hash"), "bound split_hash")
    split_records = bound["records"]
    pair_ids = [str(record.get("pair_id", "")) for record in split_records]
    if any(not item for item in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("bound split pair IDs must be non-empty and unique")
    splits = [str(record.get("split", "")) for record in split_records]
    if set(splits) != {"train", "test", "guard"}:
        raise ValueError("bound split must contain train/test/guard and no other labels")
    actual_counts = Counter(splits)
    declared_counts = bound.get("counts")
    expected_counts = {
        "total": len(split_records),
        "train": actual_counts["train"],
        "test": actual_counts["test"],
        "guard": actual_counts["guard"],
    }
    if declared_counts != expected_counts:
        raise ValueError("bound split counts mismatch")
    camera_names = []
    split_by_pair: dict[str, str] = {}
    camera_by_pair: dict[str, str] = {}
    for record in split_records:
        pair = str(record["pair_id"])
        camera = str(record.get("thermal_camera_name", ""))
        if not camera:
            raise ValueError(f"bound split is missing thermal_camera_name for {pair}")
        camera_names.append(camera)
        split_by_pair[pair] = str(record["split"])
        camera_by_pair[pair] = camera
    if len(camera_names) != len(set(camera_names)):
        raise ValueError("bound split thermal camera names are not unique")

    decode_binding = bound.get("decode_binding")
    if not isinstance(decode_binding, Mapping):
        raise ValueError("bound split has no formal TSDK decode binding")
    if decode_binding.get("adapter_backend") != "official-dji-irp":
        raise ValueError("formal OCT target must use the official DJI IRP backend")
    if decode_binding.get("verified_temperature_file_hashes") is not True:
        raise ValueError("bound split does not prove temperature target hashes")
    if decode_binding.get("decode_manifest_sha256") != input_records["decode_manifest"]["sha256"]:
        raise ValueError("supplied TSDK decode manifest differs from bound split")
    if decode_binding.get("decode_protocol_sha256") != input_records["decode_protocol"]["sha256"]:
        raise ValueError("supplied TSDK protocol differs from bound split")
    protocol_hash = _require_sha(decode_binding.get("protocol_hash"), "TSDK protocol_hash")

    decode_rows = load_jsonl_objects(decode_manifest_path, "TSDK decode manifest")
    protocol_rows = load_jsonl_objects(decode_protocol_path, "TSDK protocol")
    decode_by_pair = {str(row.get("pair_id", "")): row for row in decode_rows}
    protocol_by_pair = {str(row.get("pair_id", "")): row for row in protocol_rows}
    if len(decode_by_pair) != len(decode_rows) or set(decode_by_pair) != set(pair_ids):
        raise ValueError("TSDK decode identities differ from formal split")
    if len(protocol_by_pair) != len(protocol_rows) or set(protocol_by_pair) != set(pair_ids):
        raise ValueError("TSDK protocol identities differ from formal split")
    parameter_keys = {
        "distance_m",
        "humidity_percent",
        "emissivity",
        "ambient_c",
        "reflected_c",
    }
    tsdk_index: list[dict[str, Any]] = []
    for pair in pair_ids:
        decoded = decode_by_pair[pair]
        protocol_row = protocol_by_pair[pair]
        if decoded.get("scene") != scene or decoded.get("success") is not True:
            raise ValueError(f"invalid TSDK decode status/scene for {pair}")
        if decoded.get("dtype") != "float32":
            raise TypeError(f"TSDK target is not float32 for {pair}")
        output_sha = _require_sha(decoded.get("output_sha256"), f"TSDK output {pair}")
        if protocol_row.get("scene") != scene or protocol_row.get("protocol_hash") != protocol_hash:
            raise ValueError(f"TSDK protocol scene/hash mismatch for {pair}")
        if protocol_row.get("schema_version") != "uav-tgs.radiometry-protocol.v1":
            raise ValueError(f"unsupported TSDK protocol schema for {pair}")
        parameters = protocol_row.get("decode_parameters")
        if not isinstance(parameters, Mapping) or set(parameters) != parameter_keys:
            raise ValueError(f"incomplete TSDK decode parameters for {pair}")
        normalized_parameters: dict[str, dict[str, Any]] = {}
        for name in sorted(parameter_keys):
            entry = parameters[name]
            if not isinstance(entry, Mapping) or not isinstance(entry.get("source"), str):
                raise ValueError(f"TSDK parameter {name} lacks value/source for {pair}")
            value = float(entry.get("value"))
            if not np.isfinite(value):
                raise ValueError(f"TSDK parameter {name} is non-finite for {pair}")
            normalized_parameters[name] = {"value": value, "source": entry["source"]}
        tsdk_index.append(
            {
                "pair_id": pair,
                "output_sha256": output_sha,
                "parameters_sha256": sha256_json(normalized_parameters),
            }
        )

    range_payload = load_json_object(range_path, "range manifest")
    if (
        range_payload.get("schema_name") != _RANGE_SCHEMA
        or range_payload.get("schema_version") != 1
        or range_payload.get("scene") != scene
    ):
        raise ValueError("formal range schema/scene mismatch")
    if range_payload.get("split_hash") != split_hash:
        raise ValueError("formal range split_hash mismatch")
    if range_payload.get("source_split_manifest_sha256") != input_records["bound_split"]["sha256"]:
        raise ValueError("formal range is not bound to the supplied bound_split")
    range_basis = {
        "scene": range_payload["scene"],
        "split_hash": range_payload["split_hash"],
        "configuration": range_payload["configuration"],
        "Tmin": range_payload["Tmin"],
        "Tmax": range_payload["Tmax"],
    }
    if range_payload.get("range_hash") != sha256_json(range_basis):
        raise ValueError("formal range_hash mismatch")
    range_configuration = range_payload.get("configuration")
    if not isinstance(range_configuration, Mapping):
        raise ValueError("formal range configuration is missing")
    if range_configuration.get("guard_role") != "not_read":
        raise ValueError("formal range must not read guard frames")
    if range_configuration.get("test_role") != "qa_only_not_used_for_estimation":
        raise ValueError("formal range must use test frames for QA only")
    train_estimation = range_payload.get("train_estimation")
    clipping_stats = range_payload.get("clipping_stats")
    if not isinstance(train_estimation, Mapping) or not isinstance(clipping_stats, Mapping):
        raise ValueError("formal range train/clipping provenance is missing")
    if int(train_estimation.get("frame_count", -1)) != expected_counts["train"]:
        raise ValueError("formal range train frame count mismatch")
    for split in ("train", "test"):
        stats = clipping_stats.get(split)
        if not isinstance(stats, Mapping) or int(stats.get("frame_count", -1)) != expected_counts[split]:
            raise ValueError(f"formal range {split} QA frame count mismatch")
    quantiles = range_payload.get("per_frame_quantiles")
    if not isinstance(quantiles, list):
        raise ValueError("formal range lacks per-frame quantiles")
    quantile_counts = Counter(str(row.get("split")) for row in quantiles if isinstance(row, Mapping))
    if quantile_counts != Counter({"train": expected_counts["train"], "test": expected_counts["test"]}):
        raise ValueError("formal range per-frame split coverage mismatch")
    tmin_c, tmax_c = float(range_payload["Tmin"]), float(range_payload["Tmax"])
    if not _close(tmin_c, radiance_proxy.tmin_c) or not _close(tmax_c, radiance_proxy.tmax_c):
        raise ValueError("radiance proxy range differs from formal range")
    for config in field_configs.values():
        if not _close(config.tmin_c, tmin_c) or not _close(config.tmax_c, tmax_c):
            raise ValueError("field config range differs from formal range")

    canonical = load_json_object(canonical_manifest_path, "canonical manifest")
    if canonical.get("schema") != _CANONICAL_SCHEMA or canonical.get("status") != "complete":
        raise ValueError("canonical Hot-Iron manifest is incomplete/unsupported")
    palette = canonical.get("palette", {})
    if palette.get("sha256_uint8_rgb") != lut_sha256():
        raise ValueError("canonical Hot-Iron LUT SHA mismatch")
    encoding = canonical.get("image_encoding", {})
    if encoding != {"format": "PNG", "mode": "RGB", "lossless": True, "gamma": 1.0}:
        raise ValueError("canonical Hot-Iron encoding mismatch")
    canonical_range = canonical.get("temperature_range", {})
    if not _close(canonical_range.get("tmin_c"), tmin_c) or not _close(
        canonical_range.get("tmax_c"), tmax_c
    ):
        raise ValueError("canonical/formal temperature range mismatch")
    if canonical_range.get("source", {}).get("sha256") != input_records["range_manifest"]["sha256"]:
        raise ValueError("canonical target is not bound to the supplied range manifest")
    canonical_rows = canonical.get("files")
    if not isinstance(canonical_rows, list) or len(canonical_rows) != len(pair_ids):
        raise ValueError("canonical target file count mismatch")
    canonical_by_pair = {_identity(record): record for record in canonical_rows}
    if set(canonical_by_pair) != set(pair_ids):
        raise ValueError("canonical target identities differ from bound split")

    support_payload = load_json_object(support_manifest_path, "optimization support manifest")
    support_rows = _optimization_support_rows(support_payload)
    support_by_pair = {_identity(record): record for record in support_rows}
    if set(support_by_pair) != set(pair_ids):
        raise ValueError("optimization support identities differ from bound split")
    evaluation_support_payload = load_json_object(
        evaluation_support_manifest_path, "evaluation support manifest"
    )
    evaluation_support_rows = _evaluation_support_rows(
        evaluation_support_payload,
        expected_count=expected_counts["test"],
        bound_split_sha256=input_records["bound_split"]["sha256"],
        optimization_support_sha256=input_records["support_manifest"]["sha256"],
    )
    evaluation_support_by_pair = {
        _identity(record): record for record in evaluation_support_rows
    }
    expected_test_pairs = {
        pair for pair in pair_ids if split_by_pair[pair] == "test"
    }
    if set(evaluation_support_by_pair) != expected_test_pairs:
        raise ValueError("evaluation support identities differ from the formal test split")

    target_root = Path(temperature_root).resolve()
    color_root = Path(canonical_root).resolve()
    support_root_path = Path(support_root).resolve()
    evaluation_support_root_path = Path(evaluation_support_root).resolve()
    for root, label in (
        (target_root, "temperature root"),
        (color_root, "canonical root"),
        (support_root_path, "support root"),
        (evaluation_support_root_path, "evaluation support root"),
    ):
        if not root.is_dir():
            raise FileNotFoundError(f"{label}: {root}")
    records: list[FormalOCTRecord] = []
    target_index: list[dict[str, Any]] = []
    support_index: list[dict[str, Any]] = []
    evaluation_support_index: list[dict[str, Any]] = []
    for pair in pair_ids:
        canonical_row = canonical_by_pair[pair]
        if canonical_row.get("temperature_dtype") != "float32":
            raise TypeError(f"canonical manifest target dtype is not float32: {pair}")
        temperature_path = _resolve_under(
            target_root, str(canonical_row["relative_input"]), fallback_id=pair
        )
        temperature_sha = _require_sha(canonical_row.get("input_sha256"), f"target {pair}")
        support_row = support_by_pair[pair]
        undistorted_target = support_row.get("output_temperature")
        raw_target = support_row.get("input_temperature")
        if not isinstance(undistorted_target, Mapping) or not isinstance(raw_target, Mapping):
            raise ValueError(f"incomplete undistorted temperature provenance for {pair}")
        if undistorted_target.get("dtype") != "float32" or raw_target.get("dtype") != "float32":
            raise TypeError(f"undistorted/raw TSDK target is not float32 for {pair}")
        if _require_sha(
            undistorted_target.get("sha256"), f"undistorted target {pair}"
        ) != temperature_sha:
            raise ValueError(f"canonical/undistorted target SHA mismatch for {pair}")
        if _require_sha(
            raw_target.get("sha256"), f"raw TSDK target {pair}"
        ) != decode_by_pair[pair].get("output_sha256"):
            raise ValueError(f"undistortion/TSDK source SHA mismatch for {pair}")
        shape_hw = _validate_target_array(
            temperature_path, temperature_sha, verify_payload_files
        )
        canonical_path = _resolve_under(
            color_root, str(canonical_row["relative_output"]), fallback_id=pair
        )
        canonical_sha = _require_sha(canonical_row.get("output_sha256"), f"canonical {pair}")
        _validate_canonical_image(
            canonical_path, canonical_sha, shape_hw, verify_payload_files
        )
        support_path = _resolve_under(
            support_root_path, _support_relative(support_row), fallback_id=pair
        )
        support_sha = _support_sha(support_row)
        support_encoding = _validate_support(
            support_path, support_sha, shape_hw, verify_payload_files
        )
        evaluation_support_path: Path | None = None
        evaluation_support_sha: str | None = None
        evaluation_support_encoding: str | None = None
        if split_by_pair[pair] == "test":
            evaluation_support_row = evaluation_support_by_pair[pair]
            evaluation_support_path = _resolve_under(
                evaluation_support_root_path,
                _evaluation_support_relative(evaluation_support_row),
                fallback_id=pair,
            )
            evaluation_support_sha = _evaluation_support_sha(evaluation_support_row)
            evaluation_support_encoding = _validate_support(
                evaluation_support_path,
                evaluation_support_sha,
                shape_hw,
                verify_payload_files,
            )
            declared_shape = evaluation_support_row.get("shape")
            if declared_shape != list(shape_hw):
                raise ValueError(f"evaluation support declared shape mismatch for {pair}")
            evaluation_support_index.append(
                {
                    "pair_id": pair,
                    "sha256": evaluation_support_sha,
                    "encoding": evaluation_support_encoding,
                }
            )
        records.append(
            FormalOCTRecord(
                pair_id=pair,
                split=split_by_pair[pair],
                camera_name=camera_by_pair[pair],
                temperature_path=temperature_path,
                temperature_sha256=temperature_sha,
                canonical_path=canonical_path,
                canonical_sha256=canonical_sha,
                support_path=support_path,
                support_sha256=support_sha,
                support_encoding=support_encoding,
                evaluation_support_path=evaluation_support_path,
                evaluation_support_sha256=evaluation_support_sha,
                evaluation_support_encoding=evaluation_support_encoding,
                shape_hw=shape_hw,
            )
        )
        target_index.append(
            {
                "pair_id": pair,
                "split": split_by_pair[pair],
                "camera_name": camera_by_pair[pair],
                "temperature_sha256": temperature_sha,
                "canonical_sha256": canonical_sha,
                "shape_hw": list(shape_hw),
            }
        )
        support_index.append(
            {"pair_id": pair, "sha256": support_sha, "encoding": support_encoding}
        )

    sequence_payload = load_json_object(sequence_path, "camera sequence")
    train_names = sorted(
        camera_by_pair[pair] for pair in pair_ids if split_by_pair[pair] == "train"
    )
    sequence_payload = validate_sequence_manifest(
        sequence_payload,
        camera_names=train_names,
        expected_steps=int(sequence_payload.get("steps", -1)),
    )
    if int(sequence_payload["steps"]) != 30_000:
        raise ValueError("formal OCT camera sequence must contain exactly 30000 steps")
    expected_seed = int(FORMAL_EXPERIMENT_RECIPE["sequence_seed"])
    if (
        type(sequence_payload.get("seed")) is not int
        or int(sequence_payload["seed"]) != expected_seed
    ):
        raise ValueError(
            f"formal OCT camera sequence must use the frozen seed={expected_seed}"
        )
    metadata = sequence_payload.get("metadata", {})
    expected_metadata = {
        "scene": scene,
        "bound_split_sha256": input_records["bound_split"]["sha256"],
        "decode_manifest_sha256": input_records["decode_manifest"]["sha256"],
        "decode_protocol_sha256": input_records["decode_protocol"]["sha256"],
        "range_manifest_sha256": input_records["range_manifest"]["sha256"],
        "canonical_manifest_sha256": input_records["canonical_manifest"]["sha256"],
        "support_manifest_sha256": input_records["support_manifest"]["sha256"],
        "evaluation_support_manifest_sha256": input_records["evaluation_support_manifest"]["sha256"],
        "anchor_artifact_sha256": input_records["anchor_artifact"]["sha256"],
        "anchor_occupancy_sha256": anchor_snapshot.get("overall_sha256"),
        "camera_parameters_sha256": _require_sha(
            camera_parameters_sha256, "camera_parameters_sha256"
        ),
        "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"camera sequence metadata mismatch for {key}")

    config_payload = {name: field_configs[name].to_dict() for name in sorted(field_configs)}
    core = {
        "schema": FORMAL_BINDING_SCHEMA,
        "scene_name": scene,
        "bound_split": {
            "sha256": input_records["bound_split"]["sha256"],
            "split_hash": split_hash,
            "counts": expected_counts,
        },
        "temperature_range": {
            "sha256": input_records["range_manifest"]["sha256"],
            "range_hash": range_payload["range_hash"],
            "tmin_c": tmin_c,
            "tmax_c": tmax_c,
        },
        "tsdk_target": {
            "target_semantics": TARGET_SEMANTICS,
            "method_semantics": METHOD_SEMANTICS,
            "decode_manifest_sha256": input_records["decode_manifest"]["sha256"],
            "decode_protocol_sha256": input_records["decode_protocol"]["sha256"],
            "protocol_hash": protocol_hash,
            "adapter_backend": decode_binding["adapter_backend"],
            "adapter_executable_sha256": _require_sha(
                decode_binding.get("adapter_executable_sha256"),
                "TSDK adapter executable SHA",
            ),
            "pair_parameter_index_sha256": sha256_json(tsdk_index),
            "target_geometric_transform": "float32 temperature-domain undistortion with bound valid_support",
            "target_transform_manifest_sha256": input_records["support_manifest"]["sha256"],
            "radiometric_correction_after_tsdk_decode": False,
            "environmental_correction_reapplied_by_oct": False,
            "fixed_decode_parameters_are_metadata_only": True,
            "absolute_thermometry_claimed": False,
        },
        "canonical_target": {
            "manifest_sha256": input_records["canonical_manifest"]["sha256"],
            "lut_sha256": lut_sha256(),
            "target_index_sha256": sha256_json(target_index),
        },
        "support": {
            "optimization": {
                "schema": _OPTIMIZATION_SUPPORT_SCHEMA,
                "policy": "undistortion valid_support; train/guard/test target validity",
                "manifest_sha256": input_records["support_manifest"]["sha256"],
                "support_index_sha256": sha256_json(support_index),
            },
            "evaluation": {
                "schema": _EVALUATION_SUPPORT_SCHEMA,
                "split": "test",
                "policy": dict(_EVALUATION_SUPPORT_POLICY),
                "manifest_sha256": input_records["evaluation_support_manifest"]["sha256"],
                "support_index_sha256": sha256_json(evaluation_support_index),
            },
        },
        "camera_sequence": {
            "manifest_sha256": input_records["camera_sequence"]["sha256"],
            "content_sha256": sequence_payload["manifest_sha256"],
            "sequence_sha256": sequence_payload["sequence_sha256"],
            "ordered_camera_sha256": sequence_payload["ordered_camera_sha256"],
            "camera_parameters_sha256": expected_metadata["camera_parameters_sha256"],
            "steps": 30_000,
            "seed": expected_seed,
        },
        "anchor": {
            "artifact_sha256": input_records["anchor_artifact"]["sha256"],
            "occupancy_sha256": anchor_snapshot.get("overall_sha256"),
            "topology_count": anchor_snapshot.get("topology_count"),
        },
        "field_configs": {
            "payload": config_payload,
            "sha256": sha256_json(config_payload),
        },
        "radiance_proxy": radiance_proxy.metadata(),
        "experiment_recipe": {
            "payload": dict(FORMAL_EXPERIMENT_RECIPE),
            "sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
        },
    }
    core["formal_protocol_sha256"] = sha256_json(core)
    payload = {
        **core,
        "status": "validated",
        "payload_files_verified": bool(verify_payload_files),
        "inputs": input_records,
        "runtime_roots": {
            "temperature_root": str(target_root),
            "canonical_root": str(color_root),
            "support_root": str(support_root_path),
            "evaluation_support_root": str(evaluation_support_root_path),
        },
        "records": target_index,
    }
    return FormalOCTBinding(payload=payload, records=records)


class FormalOCTTargetStore:
    """Native-resolution formal targets with separate optimization/evaluation support."""

    def __init__(
        self, binding: FormalOCTBinding, max_cache_items: int | None = None
    ) -> None:
        self.binding = binding
        self.max_cache_items = (
            len(binding.records) if max_cache_items is None else int(max_cache_items)
        )
        if self.max_cache_items <= 0:
            raise ValueError("max_cache_items must be positive")
        self._cache: OrderedDict[
            tuple[str, bool], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = OrderedDict()

    def _load(
        self, camera_name: str, evaluation_support: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.binding.by_camera.get(camera_name)
        if record is None:
            raise KeyError(f"camera is not in formal binding: {camera_name}")
        temperature = np.load(record.temperature_path, allow_pickle=False)
        if temperature.dtype != np.dtype("float32") or not bool(np.isfinite(temperature).all()):
            raise ValueError(f"invalid formal temperature target: {record.temperature_path}")
        with Image.open(record.canonical_path) as image:
            color_u8 = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        support_path = record.support_path
        if evaluation_support:
            if record.split != "test" or record.evaluation_support_path is None:
                raise ValueError("evaluation support is defined only for formal test cameras")
            support_path = record.evaluation_support_path
        if support_path.suffix.casefold() == ".npy":
            support = np.load(support_path, allow_pickle=False)
        else:
            with Image.open(support_path) as image:
                support = np.asarray(image.convert("L"), dtype=np.uint8) == 255
        if support.dtype != np.dtype("bool"):
            raise TypeError("formal support must resolve to boolean")
        return (
            torch.from_numpy(np.asarray(temperature, dtype=np.float32))[None],
            torch.from_numpy(color_u8).permute(2, 0, 1).contiguous(),
            torch.from_numpy(np.asarray(support, dtype=np.bool_))[None],
        )

    def get(
        self,
        camera_name: str,
        height: int,
        width: int,
        device: str | torch.device,
        *,
        evaluation_support: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.binding.by_camera.get(camera_name)
        if record is None:
            raise KeyError(f"camera is not in formal binding: {camera_name}")
        if (int(height), int(width)) != record.shape_hw:
            raise ValueError(
                "formal OCT targets cannot be resized; camera/target native dimensions differ"
            )
        key = (camera_name, bool(evaluation_support))
        if key not in self._cache:
            self._cache[key] = self._load(camera_name, bool(evaluation_support))
            while len(self._cache) > self.max_cache_items:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        temperature, color_u8, support = self._cache[key]
        return (
            temperature.to(device=device, non_blocking=True),
            (color_u8.to(torch.float32) / 255.0).to(device=device, non_blocking=True),
            support.to(device=device, non_blocking=True),
        )

    def preload(self, camera_names: Sequence[str], *, evaluation_support: bool = False) -> None:
        """Load each native target once; later shuffled epochs perform no disk I/O."""

        for camera_name in camera_names:
            key = (str(camera_name), bool(evaluation_support))
            if key not in self._cache:
                self._cache[key] = self._load(*key)
        if len(self._cache) > self.max_cache_items:
            raise RuntimeError("preload exceeds the configured target cache capacity")
