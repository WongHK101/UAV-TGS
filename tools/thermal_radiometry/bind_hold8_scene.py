#!/usr/bin/env python3
"""Bind one frozen Hold-8 scene split to verified radiometry assets.

The Hold-8 source manifests are immutable.  This sidecar authenticates the
collection and scene manifests, joins every pair to the frozen DJI decode
manifest/protocol, verifies the raw/decode/float32 lineage, and materializes a
derived split plus explicit RGB and thermal train/test lists.  It deliberately
has no guard or validation output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from tools.thermal_radiometry import bind_formal_scene as lineage
from tools.thermal_radiometry.build_hold8_split import (
    HOLDOUT_PERIOD,
    PROTOCOL_ID,
    SCHEMA_NAME as SPLIT_SCHEMA_NAME,
)


SCHEMA_NAME = "uav_tgs_hold8_scene_decode_binding"
SCHEMA_VERSION = 1
FORMAL_SFM_IMAGE_SCOPE = "shared_sfm_all_images"
ALLOWED_SFM_SCOPES = (FORMAL_SFM_IMAGE_SCOPE,)
PROTOCOL_BASIS_CHOICES = lineage.PROTOCOL_BASIS_CHOICES
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _list_bytes(values: Sequence[str]) -> bytes:
    return ("\n".join(values) + ("\n" if values else "")).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return token


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} line {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {path}")
    return rows


def _unique_scene_index(
    rows: Sequence[Mapping[str, Any]], *, scene: str, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("scene", "")) != scene:
            raise ValueError(
                f"{label} contains a row outside scene {scene!r}: {row.get('scene')!r}"
            )
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or pair_id in result:
            raise ValueError(f"{label} contains a missing/duplicate pair_id: {pair_id!r}")
        result[pair_id] = row
    return result


def _validate_extension(value: str, label: str) -> str:
    if not value.startswith(".") or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a simple extension such as .JPG or .png")
    return value


def _validate_hold8_manifests(
    *,
    scene_manifest_path: Path,
    collection_manifest_path: Path,
    scene: str,
    expected_collection_manifest_sha256: str,
    expected_collection_hash: str,
    expected_collection_split_hash: str,
    expected_scene_manifest_sha256: str,
    expected_scene_split_hash: str,
) -> tuple[dict[str, Any], dict[str, Any], list[Mapping[str, Any]]]:
    scene_sha = _file_sha256(scene_manifest_path)
    collection_sha = _file_sha256(collection_manifest_path)
    _require_equal(
        "collection manifest SHA-256",
        collection_sha,
        _require_sha(expected_collection_manifest_sha256, "expected collection manifest SHA-256"),
    )
    _require_equal(
        "scene manifest SHA-256",
        scene_sha,
        _require_sha(expected_scene_manifest_sha256, "expected scene manifest SHA-256"),
    )
    scene_manifest = _load_json(scene_manifest_path, "Hold-8 scene manifest")
    collection = _load_json(collection_manifest_path, "Hold-8 collection manifest")
    if (
        scene_manifest.get("schema_name") != SPLIT_SCHEMA_NAME
        or int(scene_manifest.get("schema_version", -1)) != 1
        or scene_manifest.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("scene manifest is not an AAAI27 Hold-8 v2 scene split")
    if (
        collection.get("schema_name") != f"{SPLIT_SCHEMA_NAME}_collection"
        or int(collection.get("schema_version", -1)) != 1
        or collection.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("collection manifest is not an AAAI27 Hold-8 v2 collection")
    if scene_manifest.get("scene") != scene:
        raise ValueError("scene manifest/CLI scene mismatch")
    for payload, label in ((scene_manifest, "scene"), (collection, "collection")):
        validation = payload.get("validation")
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            raise ValueError(f"Hold-8 {label} validation is not passed")
    _require_equal(
        "collection hash",
        collection.get("collection_hash"),
        _require_sha(expected_collection_hash, "expected collection hash"),
    )
    _require_equal(
        "collection split hash",
        collection.get("collection_split_hash"),
        _require_sha(expected_collection_split_hash, "expected collection split hash"),
    )
    _require_equal(
        "scene split hash",
        scene_manifest.get("split_hash"),
        _require_sha(expected_scene_split_hash, "expected scene split hash"),
    )
    _require_equal(
        "scene/collection collection hash",
        scene_manifest.get("collection_hash"),
        collection.get("collection_hash"),
    )

    scene_rows = [row for row in collection.get("scenes", []) if row.get("scene") == scene]
    if len(scene_rows) != 1:
        raise ValueError(f"collection must contain exactly one scene row for {scene}")
    scene_row = scene_rows[0]
    _require_equal("collection scene manifest SHA", scene_row.get("manifest_sha256"), scene_sha)
    _require_equal(
        "collection scene split hash", scene_row.get("split_hash"), scene_manifest.get("split_hash")
    )

    records = scene_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Hold-8 scene manifest contains no records")
    if not all(isinstance(row, Mapping) for row in records):
        raise ValueError("Hold-8 scene records must be JSON objects")
    pair_ids = [str(row.get("pair_id", "")).strip() for row in records]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Hold-8 pair IDs must be non-empty and unique")
    labels = [str(row.get("split", "")) for row in records]
    if set(labels) != {"train", "test"}:
        raise ValueError("Hold-8 scene split must contain exactly train/test labels")
    for index, row in enumerate(records):
        if int(row.get("zero_based_sorted_index", -1)) != index:
            raise ValueError("Hold-8 sorted indices are not contiguous/in-order")
        expected_label = "test" if index % HOLDOUT_PERIOD == 0 else "train"
        if row.get("split") != expected_label:
            raise ValueError(f"Hold-8 modulo assignment mismatch at index {index}")

    observed_counts = {
        "total": len(records),
        "train": labels.count("train"),
        "test": labels.count("test"),
    }
    if scene_manifest.get("counts") != observed_counts or scene_row.get("counts") != observed_counts:
        raise ValueError("Hold-8 scene counts disagree with records/collection")
    expected_train_ids = [pair_id for pair_id, label in zip(pair_ids, labels) if label == "train"]
    expected_test_ids = [pair_id for pair_id, label in zip(pair_ids, labels) if label == "test"]
    hash_payload = scene_manifest.get("hashes")
    if not isinstance(hash_payload, Mapping):
        raise ValueError("Hold-8 scene manifest lacks list hashes")
    for label, values in (
        ("pair_ordering", pair_ids),
        ("train_list", expected_train_ids),
        ("test_list", expected_test_ids),
    ):
        _require_equal(
            f"Hold-8 {label} hash",
            _bytes_sha256(_list_bytes(values)),
            _require_sha(hash_payload.get(f"{label}_sha256"), f"Hold-8 {label} hash"),
        )
    return scene_manifest, collection, records


def bind_hold8_scene(
    *,
    scene_manifest_path: str | Path,
    collection_manifest_path: str | Path,
    decode_manifest_path: str | Path,
    decode_protocol_path: str | Path,
    temperature_root: str | Path,
    raw_thermal_root: str | Path,
    scene: str,
    sfm_image_scope: str,
    expected_collection_manifest_sha256: str,
    expected_collection_hash: str,
    expected_collection_split_hash: str,
    expected_scene_manifest_sha256: str,
    expected_scene_split_hash: str,
    expected_decode_manifest_sha256: str,
    expected_decode_protocol_sha256: str,
    expected_decode_protocol_hash: str,
    expected_adapter_executable_sha256: str,
    expected_adapter: str = lineage.FORMAL_ADAPTER,
    decode_protocol_basis: str = lineage.LEGACY_PROTOCOL_BASIS,
    camera_extension: str = ".JPG",
    thermal_camera_extension: str = ".png",
) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, Any]]:
    """Authenticate and bind one Hold-8 scene without creating guard outputs."""

    paths = {
        "scene": Path(scene_manifest_path).resolve(),
        "collection": Path(collection_manifest_path).resolve(),
        "decode": Path(decode_manifest_path).resolve(),
        "protocol": Path(decode_protocol_path).resolve(),
        "temperature_root": Path(temperature_root).resolve(),
        "raw_root": Path(raw_thermal_root).resolve(),
    }
    for label in ("scene", "collection", "decode", "protocol"):
        if not paths[label].is_file():
            raise FileNotFoundError(paths[label])
    for label in ("temperature_root", "raw_root"):
        if not paths[label].is_dir():
            raise FileNotFoundError(paths[label])
    if sfm_image_scope != FORMAL_SFM_IMAGE_SCOPE:
        raise ValueError(
            "Hold-8 formal binding requires sfm_image_scope="
            f"{FORMAL_SFM_IMAGE_SCOPE!r}; observed {sfm_image_scope!r}"
        )
    if decode_protocol_basis not in PROTOCOL_BASIS_CHOICES:
        raise ValueError(f"unsupported decode protocol basis: {decode_protocol_basis!r}")
    if expected_adapter != lineage.FORMAL_ADAPTER:
        raise ValueError(f"unsupported formal decode adapter: {expected_adapter!r}")
    camera_extension = _validate_extension(camera_extension, "camera_extension")
    thermal_camera_extension = _validate_extension(
        thermal_camera_extension, "thermal_camera_extension"
    )

    expected_decode_sha = _require_sha(
        expected_decode_manifest_sha256, "expected decode manifest SHA-256"
    )
    expected_protocol_sha = _require_sha(
        expected_decode_protocol_sha256, "expected decode protocol SHA-256"
    )
    expected_adapter_sha = _require_sha(
        expected_adapter_executable_sha256, "expected adapter executable SHA-256"
    )
    _require_equal("decode manifest SHA-256", _file_sha256(paths["decode"]), expected_decode_sha)
    _require_equal("decode protocol SHA-256", _file_sha256(paths["protocol"]), expected_protocol_sha)

    scene_manifest, collection, records = _validate_hold8_manifests(
        scene_manifest_path=paths["scene"],
        collection_manifest_path=paths["collection"],
        scene=scene,
        expected_collection_manifest_sha256=expected_collection_manifest_sha256,
        expected_collection_hash=expected_collection_hash,
        expected_collection_split_hash=expected_collection_split_hash,
        expected_scene_manifest_sha256=expected_scene_manifest_sha256,
        expected_scene_split_hash=expected_scene_split_hash,
    )
    decode_rows = _load_jsonl(paths["decode"], "decode manifest")
    protocol_rows = _load_jsonl(paths["protocol"], "decode protocol")
    protocol_hash = lineage._validate_protocol_collection(
        protocol_rows,
        expected_protocol_hash=expected_decode_protocol_hash,
        protocol_basis_variant=decode_protocol_basis,
    )
    decode_by_pair = _unique_scene_index(decode_rows, scene=scene, label="decode manifest")
    protocol_by_pair = _unique_scene_index(protocol_rows, scene=scene, label="decode protocol")
    pair_ids = [str(row["pair_id"]) for row in records]
    expected_pairs = set(pair_ids)
    for label, indexed in (("decode", decode_by_pair), ("protocol", protocol_by_pair)):
        if set(indexed) != expected_pairs:
            raise ValueError(
                f"{label}/Hold-8 pair mismatch: "
                f"missing={sorted(expected_pairs - set(indexed))[:8]} "
                f"extra={sorted(set(indexed) - expected_pairs)[:8]}"
            )

    lists: dict[str, list[str]] = {
        "train": [],
        "test": [],
        "thermal_train": [],
        "thermal_test": [],
    }
    resolved_records: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    dirp_versions: set[str] = set()
    rjpeg_versions: set[str] = set()
    for record in records:
        pair_id = str(record["pair_id"])
        split = str(record["split"])
        decode = decode_by_pair[pair_id]
        protocol = protocol_by_pair[pair_id]
        if decode.get("success") is not True:
            raise ValueError(f"decode was not successful for {scene}/{pair_id}")
        source_hash = record.get("source_record_hash")
        if source_hash is not None and source_hash != protocol.get("source_audit_record_hash"):
            raise ValueError(f"split/protocol source record hash mismatch for {pair_id}")
        if record.get("strip_id") is not None and record.get("strip_id") != protocol.get("strip_id"):
            raise ValueError(f"split/protocol strip mismatch for {pair_id}")
        row_protocol_hash = _require_sha(protocol.get("protocol_hash"), f"protocol hash {pair_id}")
        _require_equal(f"protocol collection hash {pair_id}", row_protocol_hash, protocol_hash)
        raw_lineage = lineage._validate_decode_lineage(
            scene=scene,
            pair_id=pair_id,
            decode=decode,
            protocol=protocol,
            decode_manifest_path=paths["decode"],
            raw_thermal_root=paths["raw_root"],
            expected_adapter=expected_adapter,
            expected_adapter_executable_sha256=expected_adapter_sha,
        )
        dirp_versions.add(str(raw_lineage["dirp_api_version"]))
        rjpeg_versions.add(str(raw_lineage["rjpeg_version"]))

        npy_path = paths["temperature_root"] / scene / f"{pair_id}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(npy_path)
        if decode.get("dtype") != "float32" or list(decode.get("shape_hw", [])) != [1024, 1280]:
            raise ValueError(f"decode dtype/shape is not float32 1024x1280 for {pair_id}")
        array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.dtype("float32") or array.shape != (1024, 1280):
            raise ValueError(f"temperature dtype/shape mismatch for {pair_id}")
        actual_temperature_sha = _file_sha256(npy_path)
        expected_temperature_sha = _require_sha(
            decode.get("output_sha256"), f"decode output SHA-256 {pair_id}"
        )
        _require_equal(
            f"temperature file SHA-256 {pair_id}",
            actual_temperature_sha,
            expected_temperature_sha,
        )

        camera_name = f"{pair_id}{camera_extension}"
        thermal_camera_name = f"{pair_id}{thermal_camera_extension}"
        lists[split].append(camera_name)
        lists[f"thermal_{split}"].append(thermal_camera_name)
        resolved = dict(record)
        resolved.update(
            {
                "camera_name": camera_name,
                "thermal_camera_name": thermal_camera_name,
                "temperature_npy": f"{scene}/{pair_id}.npy",
            }
        )
        resolved_records.append(resolved)
        files.append(
            {
                "pair_id": pair_id,
                "split": split,
                "camera_name": camera_name,
                "thermal_camera_name": thermal_camera_name,
                "temperature_npy": f"{scene}/{pair_id}.npy",
                "temperature_sha256": actual_temperature_sha,
                "raw_thermal": raw_lineage,
            }
        )

    if len(dirp_versions) != 1 or len(rjpeg_versions) != 1:
        raise ValueError("decode collection must use one DIRP API and one R-JPEG version")
    rgb_membership = set(lists["train"]) | set(lists["test"])
    thermal_membership = set(lists["thermal_train"]) | set(lists["thermal_test"])
    if set(lists["train"]) & set(lists["test"]) or len(rgb_membership) != len(records):
        raise ValueError("derived RGB train/test membership is not disjoint and complete")
    if set(lists["thermal_train"]) & set(lists["thermal_test"]) or len(thermal_membership) != len(records):
        raise ValueError("derived thermal train/test membership is not disjoint and complete")

    counts = Counter(str(row["split"]) for row in records)
    count_payload = {"total": len(records), "train": counts["train"], "test": counts["test"]}
    list_hashes = {label: _bytes_sha256(_list_bytes(values)) for label, values in lists.items()}
    bound_split = dict(scene_manifest)
    bound_split["records"] = resolved_records
    # Keep the derived split byte-stable after the frozen input tree is copied
    # between 900 and 901; the SHA is authoritative, not a host-local path.
    bound_split["hold8_source_manifest"] = paths["scene"].name
    bound_split["hold8_source_manifest_sha256"] = _file_sha256(paths["scene"])
    bound_split["decode_binding"] = {
        "schema_name": SCHEMA_NAME,
        "decode_manifest_sha256": expected_decode_sha,
        "decode_protocol_sha256": expected_protocol_sha,
        "protocol_hash": protocol_hash,
        "protocol_basis_variant": decode_protocol_basis,
        "verified_temperature_file_hashes": True,
        "verified_raw_rjpeg_hashes": True,
        "verified_decode_requests": True,
        "adapter": expected_adapter,
        "adapter_backend": lineage.FORMAL_ADAPTER_BACKEND,
        "adapter_executable_sha256": expected_adapter_sha,
        "dirp_api_version": next(iter(dirp_versions)),
        "rjpeg_version": next(iter(rjpeg_versions)),
    }
    basis = {
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "collection_hash": collection["collection_hash"],
        "collection_split_hash": collection["collection_split_hash"],
        "scene_split_hash": scene_manifest["split_hash"],
        "collection_manifest_sha256": _file_sha256(paths["collection"]),
        "scene_manifest_sha256": _file_sha256(paths["scene"]),
        "decode_manifest_sha256": expected_decode_sha,
        "decode_protocol_sha256": expected_protocol_sha,
        "decode_protocol_hash": protocol_hash,
        "decode_protocol_basis": decode_protocol_basis,
        "adapter": expected_adapter,
        "adapter_backend": lineage.FORMAL_ADAPTER_BACKEND,
        "adapter_executable_sha256": expected_adapter_sha,
        "dirp_api_version": next(iter(dirp_versions)),
        "rjpeg_version": next(iter(rjpeg_versions)),
        "sfm_image_scope": sfm_image_scope,
        "counts": count_payload,
        "list_sha256": list_hashes,
        "files": files,
    }
    binding_manifest = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        **basis,
        "binding_hash": _json_hash(basis),
    }
    return bound_split, lists, binding_manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def materialize_hold8_binding(
    output_root: str | Path,
    *,
    overwrite: bool = False,
    **kwargs: Any,
) -> dict[str, Path]:
    """Write the six Hold-8 binding artifacts atomically."""

    output_root = Path(output_root).resolve()
    outputs = {
        "bound_split": output_root / "bound_split.json",
        "train": output_root / "train_list.txt",
        "test": output_root / "test_list.txt",
        "thermal_train": output_root / "thermal_train_list.txt",
        "thermal_test": output_root / "thermal_test_list.txt",
        "binding_manifest": output_root / "binding_manifest.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite Hold-8 binding outputs: {existing[:3]}")
    bound, lists, manifest = bind_hold8_scene(**kwargs)
    _atomic_json(outputs["bound_split"], bound)
    for label in ("train", "test", "thermal_train", "thermal_test"):
        _atomic_text(outputs[label], _list_bytes(lists[label]).decode("utf-8"))
    _atomic_json(outputs["binding_manifest"], manifest)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", required=True, type=Path)
    parser.add_argument("--collection-manifest", required=True, type=Path)
    parser.add_argument("--decode-manifest", required=True, type=Path)
    parser.add_argument("--decode-protocol", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--raw-thermal-root", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sfm-image-scope", required=True, choices=ALLOWED_SFM_SCOPES)
    parser.add_argument("--expected-collection-manifest-sha256", required=True)
    parser.add_argument("--expected-collection-hash", required=True)
    parser.add_argument("--expected-collection-split-hash", required=True)
    parser.add_argument("--expected-scene-manifest-sha256", required=True)
    parser.add_argument("--expected-scene-split-hash", required=True)
    parser.add_argument("--expected-decode-manifest-sha256", required=True)
    parser.add_argument("--expected-decode-protocol-sha256", required=True)
    parser.add_argument("--expected-decode-protocol-hash", required=True)
    parser.add_argument("--expected-adapter-executable-sha256", required=True)
    parser.add_argument("--expected-adapter", default=lineage.FORMAL_ADAPTER)
    parser.add_argument(
        "--decode-protocol-basis",
        choices=PROTOCOL_BASIS_CHOICES,
        default=lineage.LEGACY_PROTOCOL_BASIS,
    )
    parser.add_argument("--camera-extension", default=".JPG")
    parser.add_argument("--thermal-camera-extension", default=".png")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = materialize_hold8_binding(
        args.output_root,
        overwrite=args.overwrite,
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
        expected_scene_manifest_sha256=args.expected_scene_manifest_sha256,
        expected_scene_split_hash=args.expected_scene_split_hash,
        expected_decode_manifest_sha256=args.expected_decode_manifest_sha256,
        expected_decode_protocol_sha256=args.expected_decode_protocol_sha256,
        expected_decode_protocol_hash=args.expected_decode_protocol_hash,
        expected_adapter_executable_sha256=args.expected_adapter_executable_sha256,
        expected_adapter=args.expected_adapter,
        decode_protocol_basis=args.decode_protocol_basis,
        camera_extension=args.camera_extension,
        thermal_camera_extension=args.thermal_camera_extension,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "scene": args.scene,
                "bound_split": str(outputs["bound_split"]),
                "binding_manifest": str(outputs["binding_manifest"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
