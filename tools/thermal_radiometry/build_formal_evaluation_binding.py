#!/usr/bin/env python3
"""Build a method-independent formal radiometry evaluation binding.

This receipt binds only the artifacts needed by the palette-rendered baseline
temperature/hotspot evaluator.  It intentionally carries no OCT field,
optimizer, anchor, camera sequence, or checkpoint contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oct_gs.formal import sha256_file, sha256_json
from oct_gs.radiance import METHOD_SEMANTICS, TARGET_SEMANTICS
from tools.evaluate_formal_baseline_hotspots import (
    CANONICAL_SCHEMA,
    EVALUATION_SUPPORT_POLICY,
    EVALUATION_SUPPORT_SCHEMA,
    GENERIC_BINDING_CORE_SCHEMA,
    GENERIC_BINDING_SCHEMA,
    OPTIMIZATION_SUPPORT_SCHEMA,
    _evaluation_support_record,
    _indexed,
    _load_json,
    _load_jsonl,
    _require_sha,
    _resolve_under,
    _support_index,
)
from tools.thermal_radiometry.palette_lut import PALETTE_NAME, lut_sha256


def expected_split_labels(split: Mapping[str, Any]) -> set[str]:
    """Return the exact partition labels allowed by the frozen split protocol."""

    if split.get("protocol_id") == "uav-tgs-aaai27-hold8-v2":
        return {"train", "test"}
    return {"train", "guard", "test"}


def _input(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def build_formal_evaluation_binding(
    *,
    scene_name: str,
    bound_split_path: str | Path,
    decode_manifest_path: str | Path,
    decode_protocol_path: str | Path,
    range_manifest_path: str | Path,
    canonical_manifest_path: str | Path,
    optimization_support_manifest_path: str | Path,
    evaluation_support_manifest_path: str | Path,
    temperature_root: str | Path,
) -> dict[str, Any]:
    """Validate and bind the formal radiometry/evaluation artifact graph."""

    scene = str(scene_name).strip()
    if not scene:
        raise ValueError("scene_name must be non-empty")
    paths = {
        "bound_split": Path(bound_split_path).resolve(),
        "decode_manifest": Path(decode_manifest_path).resolve(),
        "decode_protocol": Path(decode_protocol_path).resolve(),
        "range_manifest": Path(range_manifest_path).resolve(),
        "canonical_manifest": Path(canonical_manifest_path).resolve(),
        "optimization_support_manifest": Path(optimization_support_manifest_path).resolve(),
        "evaluation_support_manifest": Path(evaluation_support_manifest_path).resolve(),
    }
    inputs = {name: _input(path) for name, path in paths.items()}

    split = _load_json(paths["bound_split"], "bound split")
    records = split.get("records")
    if split.get("scene") != scene or not isinstance(records, list) or not records:
        raise ValueError("bound split scene/records mismatch")
    split_by_pair = _indexed(records, "bound split")
    pair_ids = list(split_by_pair)
    split_values = [str(row.get("split", "")) for row in records]
    expected_splits = expected_split_labels(split)
    if set(split_values) != expected_splits:
        raise ValueError(
            "bound split labels do not match its protocol: "
            f"expected={sorted(expected_splits)} actual={sorted(set(split_values))}"
        )
    counts = {
        "total": len(records),
        **{name: split_values.count(name) for name in sorted(expected_splits)},
    }
    if split.get("counts") != counts:
        raise ValueError("bound split counts mismatch")
    split_hash = _require_sha(split.get("split_hash"), "bound split hash")
    camera_by_pair = {pair: str(row.get("thermal_camera_name", "")) for pair, row in split_by_pair.items()}
    if any(not name for name in camera_by_pair.values()) or len(set(camera_by_pair.values())) != len(pair_ids):
        raise ValueError("bound split thermal camera names must be non-empty and unique")

    decode_binding = split.get("decode_binding")
    if not isinstance(decode_binding, Mapping) or decode_binding.get("adapter_backend") != "official-dji-irp":
        raise ValueError("bound split lacks official-dji-irp formal decode binding")
    for key in (
        "verified_decode_requests",
        "verified_raw_rjpeg_hashes",
        "verified_temperature_file_hashes",
    ):
        if decode_binding.get(key) is not True:
            raise ValueError(f"bound split decode binding does not prove {key}")
    if decode_binding.get("decode_manifest_sha256") != inputs["decode_manifest"]["sha256"]:
        raise ValueError("decode manifest differs from bound split")
    if decode_binding.get("decode_protocol_sha256") != inputs["decode_protocol"]["sha256"]:
        raise ValueError("decode protocol differs from bound split")
    protocol_hash = _require_sha(decode_binding.get("protocol_hash"), "decode protocol hash")
    decode_by_pair = _indexed(_load_jsonl(paths["decode_manifest"], "decode manifest"), "decode manifest")
    protocol_by_pair = _indexed(_load_jsonl(paths["decode_protocol"], "decode protocol"), "decode protocol")
    if set(decode_by_pair) != set(pair_ids) or set(protocol_by_pair) != set(pair_ids):
        raise ValueError("decode/protocol identities differ from bound split")
    parameter_keys = {"distance_m", "humidity_percent", "emissivity", "ambient_c", "reflected_c"}
    parameter_index: list[dict[str, Any]] = []
    for pair in pair_ids:
        decoded, protocol = decode_by_pair[pair], protocol_by_pair[pair]
        if decoded.get("scene") != scene or decoded.get("success") is not True or decoded.get("dtype") != "float32":
            raise ValueError(f"decode record is not successful float32 for {pair}")
        if (
            protocol.get("scene") != scene
            or protocol.get("protocol_hash") != protocol_hash
            or protocol.get("schema_version") != "uav-tgs.radiometry-protocol.v1"
        ):
            raise ValueError(f"decode protocol mismatch for {pair}")
        parameters = protocol.get("decode_parameters")
        if not isinstance(parameters, Mapping) or set(parameters) != parameter_keys:
            raise ValueError(f"decode parameters are incomplete for {pair}")
        normalized: dict[str, Any] = {}
        for name in sorted(parameter_keys):
            entry = parameters[name]
            if not isinstance(entry, Mapping) or not isinstance(entry.get("source"), str):
                raise ValueError(f"decode parameter {name} lacks value/source for {pair}")
            value = float(entry.get("value"))
            if not math.isfinite(value):
                raise ValueError(f"decode parameter {name} is non-finite for {pair}")
            normalized[name] = {"value": value, "source": entry["source"]}
        parameter_index.append(
            {
                "pair_id": pair,
                "output_sha256": _require_sha(decoded.get("output_sha256"), f"decode target {pair}"),
                "parameters_sha256": sha256_json(normalized),
            }
        )

    range_payload = _load_json(paths["range_manifest"], "range manifest")
    if range_payload.get("scene") != scene:
        raise ValueError("range manifest scene mismatch")
    if (
        range_payload.get("source_split_manifest_sha256") != inputs["bound_split"]["sha256"]
        or range_payload.get("split_hash") != split_hash
    ):
        raise ValueError("range manifest is not bound to the exact split")
    configuration = range_payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("range manifest lacks configuration")
    if configuration.get("guard_role") != "not_read" or configuration.get("test_role") != "qa_only_not_used_for_estimation":
        raise ValueError("formal range train/guard/test roles mismatch")
    tmin_c, tmax_c = float(range_payload.get("Tmin")), float(range_payload.get("Tmax"))
    if not math.isfinite(tmin_c) or not math.isfinite(tmax_c) or tmax_c <= tmin_c:
        raise ValueError("formal temperature range is invalid")
    range_basis = {
        "scene": range_payload.get("scene"),
        "split_hash": range_payload.get("split_hash"),
        "configuration": configuration,
        "Tmin": range_payload.get("Tmin"),
        "Tmax": range_payload.get("Tmax"),
    }
    range_hash = _require_sha(range_payload.get("range_hash"), "range hash")
    if range_hash != sha256_json(range_basis):
        raise ValueError("range logical hash mismatch")

    canonical = _load_json(paths["canonical_manifest"], "canonical manifest")
    if canonical.get("schema") != CANONICAL_SCHEMA or canonical.get("status") != "complete":
        raise ValueError("canonical Hot-Iron manifest is incomplete/unsupported")
    palette = canonical.get("palette")
    if not isinstance(palette, Mapping) or palette.get("name") != PALETTE_NAME or palette.get("sha256_uint8_rgb") != lut_sha256():
        raise ValueError("canonical manifest does not bind the fixed Hot-Iron LUT")
    if canonical.get("image_encoding") != {"format": "PNG", "mode": "RGB", "lossless": True, "gamma": 1.0}:
        raise ValueError("canonical Hot-Iron encoding mismatch")
    canonical_range = canonical.get("temperature_range")
    if not isinstance(canonical_range, Mapping):
        raise ValueError("canonical manifest lacks temperature range")
    if float(canonical_range.get("tmin_c")) != tmin_c or float(canonical_range.get("tmax_c")) != tmax_c:
        raise ValueError("canonical/range temperatures differ")
    if not isinstance(canonical_range.get("source"), Mapping) or canonical_range["source"].get("sha256") != inputs["range_manifest"]["sha256"]:
        raise ValueError("canonical manifest is not bound to the range manifest")
    canonical_rows = canonical.get("files")
    if not isinstance(canonical_rows, list):
        raise ValueError("canonical manifest lacks files")
    canonical_by_pair = _indexed(canonical_rows, "canonical manifest")
    if set(canonical_by_pair) != set(pair_ids):
        raise ValueError("canonical identities differ from bound split")

    support = _load_json(paths["optimization_support_manifest"], "optimization support")
    if support.get("schema") != OPTIMIZATION_SUPPORT_SCHEMA or support.get("status") != "complete":
        raise ValueError("optimization support manifest is incomplete/unsupported")
    support_rows = support.get("files")
    if not isinstance(support_rows, list):
        raise ValueError("optimization support manifest lacks files")
    support_by_pair = _indexed(support_rows, "optimization support")
    if set(support_by_pair) != set(pair_ids):
        raise ValueError("optimization support identities differ from bound split")
    support_index = _support_index(pair_ids, support_by_pair)

    evaluation = _load_json(paths["evaluation_support_manifest"], "evaluation support")
    if (
        evaluation.get("schema_name") != EVALUATION_SUPPORT_SCHEMA
        or evaluation.get("schema_version") != 1
        or evaluation.get("split") != "test"
        or evaluation.get("policy") != EVALUATION_SUPPORT_POLICY
        or evaluation.get("expected_test_count") != counts["test"]
    ):
        raise ValueError("evaluation support contract mismatch")
    sources = evaluation.get("source_manifests")
    if not isinstance(sources, Mapping):
        raise ValueError("evaluation support lacks source manifests")
    if not isinstance(sources.get("split"), Mapping) or sources["split"].get("sha256") != inputs["bound_split"]["sha256"]:
        raise ValueError("evaluation support is bound to another split")
    if not isinstance(sources.get("valid_support"), Mapping) or sources["valid_support"].get("sha256") != inputs["optimization_support_manifest"]["sha256"]:
        raise ValueError("evaluation support is bound to another optimization support")
    evaluation_rows = evaluation.get("records")
    if not isinstance(evaluation_rows, list):
        raise ValueError("evaluation support lacks records")
    evaluation_by_pair = _indexed(evaluation_rows, "evaluation support")
    test_pairs = {pair for pair, row in split_by_pair.items() if row.get("split") == "test"}
    if set(evaluation_by_pair) != test_pairs:
        raise ValueError("evaluation support identities differ from test split")
    evaluation_index = []
    for pair in pair_ids:
        if pair in test_pairs:
            relative, digest = _evaluation_support_record(evaluation_by_pair[pair])
            encoding = "bool-npy" if Path(relative.replace("\\", "/")).suffix.casefold() == ".npy" else "binary-image-0-255"
            evaluation_index.append({"pair_id": pair, "sha256": digest, "encoding": encoding})

    root = Path(temperature_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    target_index = []
    for pair in pair_ids:
        row = canonical_by_pair[pair]
        if row.get("temperature_dtype") != "float32":
            raise TypeError(f"canonical target is not float32 for {pair}")
        target = _resolve_under(root, str(row.get("relative_input", "")), pair)
        target_sha = _require_sha(row.get("input_sha256"), f"target {pair}")
        if sha256_file(target) != target_sha:
            raise ValueError(f"formal target SHA mismatch for {pair}")
        value = np.load(target, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.dtype("float32") or value.ndim != 2 or value.size == 0:
            raise TypeError(f"formal target must be non-empty float32 HW for {pair}")
        target_index.append(
            {
                "pair_id": pair,
                "split": str(split_by_pair[pair].get("split")),
                "camera_name": camera_by_pair[pair],
                "temperature_sha256": target_sha,
                "canonical_sha256": _require_sha(row.get("output_sha256"), f"canonical {pair}"),
                "shape_hw": [int(value.shape[0]), int(value.shape[1])],
            }
        )

    core: dict[str, Any] = {
        "schema": GENERIC_BINDING_CORE_SCHEMA,
        "scene_name": scene,
        "bound_split": {"sha256": inputs["bound_split"]["sha256"], "split_hash": split_hash, "counts": counts},
        "temperature_range": {"sha256": inputs["range_manifest"]["sha256"], "range_hash": range_hash, "tmin_c": tmin_c, "tmax_c": tmax_c},
        "tsdk_target": {
            "decode_manifest_sha256": inputs["decode_manifest"]["sha256"],
            "decode_protocol_sha256": inputs["decode_protocol"]["sha256"],
            "protocol_hash": protocol_hash,
            "pair_parameter_index_sha256": sha256_json(parameter_index),
            "target_semantics": TARGET_SEMANTICS,
            "method_semantics": METHOD_SEMANTICS,
            "adapter_backend": "official-dji-irp",
            "absolute_thermometry_claimed": False,
        },
        "canonical_target": {
            "manifest_sha256": inputs["canonical_manifest"]["sha256"],
            "lut_sha256": lut_sha256(),
            "target_index_sha256": sha256_json(target_index),
        },
        "support": {
            "optimization": {
                "manifest_sha256": inputs["optimization_support_manifest"]["sha256"],
                "support_index_sha256": sha256_json(support_index),
            },
            "evaluation": {
                "manifest_sha256": inputs["evaluation_support_manifest"]["sha256"],
                "support_index_sha256": sha256_json(evaluation_index),
                "policy": EVALUATION_SUPPORT_POLICY,
                "split": "test",
            },
        },
    }
    core["formal_protocol_sha256"] = sha256_json(core)
    payload: dict[str, Any] = {
        "schema": GENERIC_BINDING_SCHEMA,
        "schema_version": 1,
        "status": "complete",
        "purpose": "method-independent formal radiometry evaluation only",
        "formal_binding": core,
    }
    payload["binding_manifest_sha256"] = sha256_json(payload)
    return payload


def write_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"formal evaluation binding already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--bound-split", required=True, type=Path)
    parser.add_argument("--decode-manifest", required=True, type=Path)
    parser.add_argument("--decode-protocol", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--optimization-support-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-support-manifest", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_formal_evaluation_binding(
        scene_name=args.scene_name,
        bound_split_path=args.bound_split,
        decode_manifest_path=args.decode_manifest,
        decode_protocol_path=args.decode_protocol,
        range_manifest_path=args.range_manifest,
        canonical_manifest_path=args.canonical_manifest,
        optimization_support_manifest_path=args.optimization_support_manifest,
        evaluation_support_manifest_path=args.evaluation_support_manifest,
        temperature_root=args.temperature_root,
    )
    output = write_atomic(args.output, payload)
    print(json.dumps({"status": "complete", "scene_name": args.scene_name, "output": str(output), "binding_manifest_sha256": payload["binding_manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
