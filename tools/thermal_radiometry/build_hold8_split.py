#!/usr/bin/env python3
"""Build the deterministic AAAI27 Hold-8 v2 train/test manifests.

This module is intentionally independent from :mod:`build_split`, which
implements the retired block/guard protocol.  Hold-8 orders each scene by its
canonical RGB--T ``pair_id`` using a numeric-aware natural ordering, assigns
zero-based positions divisible by eight to test, and assigns every other pair
to train.  There is no guard or validation partition.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "uav_tgs_aaai27_hold8_split"
SCHEMA_VERSION = 1
PROTOCOL_ID = "uav-tgs-aaai27-hold8-v2"
HOLDOUT_PERIOD = 8

EXPECTED_SCENE_COUNTS: dict[str, int] = {
    "Building": 614,
    "Garden": 656,
    "InternalRoad": 559,
    "Orchard": 588,
    "PVpanel": 289,
    "Plaza": 668,
    "Road": 467,
    "TransmissionTower": 673,
    "Urban100K": 1671,
    "Urban20K": 748,
    "Urban50K": 1299,
}

_RESERVED_RECORD_KEYS = frozenset(
    {"zero_based_sorted_index", "split", "split_rule", "source_record_sha256"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Hold8ValidationError(ValueError):
    """Raised when a formal Hold-8 input or invariant fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pair_list_bytes(pair_ids: Sequence[str]) -> bytes:
    # The trailing LF is part of the frozen list-file contract, including for
    # an empty list (which is represented by an empty file).
    if not pair_ids:
        return b""
    return ("\n".join(pair_ids) + "\n").encode("utf-8")


def _input_records_hash(records: Sequence[Mapping[str, Any]]) -> str:
    """Preserve the collection-identity hash used by the locked audit set."""

    return _hash_json(
        [
            {
                "source_index": source_index,
                "pair_id": record["pair_id"],
                "source_record_hash": _hash_json(record),
            }
            for source_index, record in enumerate(records)
        ]
    )


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Return a cross-input-order deterministic numeric-aware sort key."""

    tokens: list[tuple[int, Any]] = []
    for part in re.split(r"(\d+)", value):
        if not part:
            continue
        if part.isdigit():
            # Length and original digits give deterministic ordering for
            # numerically equal spellings such as ``2`` and ``02``.
            tokens.append((1, (int(part), len(part), part)))
        else:
            tokens.append((0, (part.casefold(), part)))
    return tuple(tokens)


def _validate_sha256(value: str, field: str) -> str:
    normalized = str(value).lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise Hold8ValidationError(f"{field} must be a 64-character SHA-256")
    return normalized


def _validate_code_commit(value: str) -> str:
    normalized = str(value).lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise Hold8ValidationError("code_commit must be a full 40-character Git commit")
    return normalized


def _extract_records(payload: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
        metadata: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        records = None
        for key in ("records", "frames", "pairs", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise Hold8ValidationError(
                "manifest must contain a records, frames, pairs, or items array"
            )
        metadata = payload
    else:
        raise Hold8ValidationError("manifest must be a JSON object or array")
    if not records:
        raise Hold8ValidationError("manifest contains no records")
    if not all(isinstance(record, Mapping) for record in records):
        raise Hold8ValidationError("every record must be a JSON object")
    return list(records), metadata


def load_manifest(path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _extract_records(records)
    return _extract_records(json.loads(path.read_text(encoding="utf-8")))


def resolve_generator_identity(repo_root: Path | None = None) -> dict[str, str]:
    source_path = Path(__file__).resolve()
    if repo_root is None:
        repo_root = source_path.parents[2]
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Hold8ValidationError("unable to resolve split generator Git commit") from exc
    return {
        "code_commit": _validate_code_commit(commit),
        "generator_source_sha256": _file_sha256(source_path),
    }


def _validate_input_record(record: Mapping[str, Any], scene: str) -> str:
    collisions = sorted(_RESERVED_RECORD_KEYS & set(record))
    if collisions:
        raise Hold8ValidationError(
            f"source record contains reserved Hold-8 keys: {collisions}"
        )
    record_scene = record.get("scene")
    if not isinstance(record_scene, str) or not record_scene.strip():
        raise Hold8ValidationError("every record must contain a non-empty string scene")
    if record_scene != scene:
        raise Hold8ValidationError(
            f"scene mismatch: expected {scene!r}, observed {record_scene!r}"
        )
    pair_id = record.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id.strip():
        raise Hold8ValidationError(
            f"every {scene} record must contain a non-empty string pair_id"
        )
    if pair_id != pair_id.strip():
        raise Hold8ValidationError(f"pair_id has leading/trailing whitespace: {pair_id!r}")
    if "\n" in pair_id or "\r" in pair_id:
        raise Hold8ValidationError(f"pair_id cannot contain a newline: {pair_id!r}")
    return pair_id


def build_scene_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    source_manifest: str,
    source_manifest_sha256: str,
    collection_hash: str,
    code_commit: str,
    generator_source_sha256: str,
    expected_total: int | None = None,
) -> dict[str, Any]:
    """Build one deterministic scene manifest while preserving audit fields."""

    if not records:
        raise Hold8ValidationError(f"scene {scene!r} contains no records")
    if not isinstance(scene, str) or not scene:
        raise Hold8ValidationError("scene must be a non-empty string")
    source_manifest_sha256 = _validate_sha256(
        source_manifest_sha256, "source_manifest_sha256"
    )
    collection_hash = _validate_sha256(collection_hash, "collection_hash")
    code_commit = _validate_code_commit(code_commit)
    generator_source_sha256 = _validate_sha256(
        generator_source_sha256, "generator_source_sha256"
    )
    if expected_total is not None and len(records) != int(expected_total):
        raise Hold8ValidationError(
            f"scene count mismatch for {scene}: expected {expected_total}, observed {len(records)}"
        )

    by_pair: dict[str, Mapping[str, Any]] = {}
    for record in records:
        pair_id = _validate_input_record(record, scene)
        if pair_id in by_pair:
            raise Hold8ValidationError(f"duplicate pair_id in {scene}: {pair_id!r}")
        by_pair[pair_id] = record

    ordered_pair_ids = sorted(by_pair, key=lambda value: (natural_sort_key(value), value))
    train_pair_ids: list[str] = []
    test_pair_ids: list[str] = []
    output_records: list[dict[str, Any]] = []
    for index, pair_id in enumerate(ordered_pair_ids):
        split = "test" if index % HOLDOUT_PERIOD == 0 else "train"
        (test_pair_ids if split == "test" else train_pair_ids).append(pair_id)
        source_record = by_pair[pair_id]
        output_record = copy.deepcopy(dict(source_record))
        output_record.update(
            {
                "zero_based_sorted_index": index,
                "split": split,
                "split_rule": "zero_based_sorted_index_mod_8",
                "source_record_sha256": _hash_json(source_record),
            }
        )
        output_records.append(output_record)

    expected_test = (len(records) + HOLDOUT_PERIOD - 1) // HOLDOUT_PERIOD
    expected_train = len(records) - expected_test
    counts = {
        "total": len(output_records),
        "train": len(train_pair_ids),
        "test": len(test_pair_ids),
    }
    if counts != {"total": len(records), "train": expected_train, "test": expected_test}:
        raise Hold8ValidationError(f"Hold-8 count invariant failed for {scene}: {counts}")
    if set(train_pair_ids) & set(test_pair_ids):
        raise Hold8ValidationError(f"train/test overlap detected for {scene}")
    if set(train_pair_ids) | set(test_pair_ids) != set(ordered_pair_ids):
        raise Hold8ValidationError(f"train/test coverage mismatch for {scene}")

    hashes = {
        "input_records_sha256": _input_records_hash(records),
        "source_records_content_sha256": _hash_json(list(records)),
        "pair_ordering_sha256": _sha256_bytes(_pair_list_bytes(ordered_pair_ids)),
        "train_list_sha256": _sha256_bytes(_pair_list_bytes(train_pair_ids)),
        "test_list_sha256": _sha256_bytes(_pair_list_bytes(test_pair_ids)),
        "split_records_sha256": _hash_json(output_records),
    }
    rule = {
        "ordering": "numeric_aware_natural_sort_of_canonical_pair_id",
        "index_base": 0,
        "test_condition": "zero_based_sorted_index % 8 == 0",
        "train_condition": "zero_based_sorted_index % 8 != 0",
        "holdout_period": HOLDOUT_PERIOD,
        "guard_partition": False,
        "validation_partition": False,
        "missing_pair_ids_change_position_rule": False,
    }
    split_basis = {
        "protocol_id": PROTOCOL_ID,
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "collection_hash": collection_hash,
        "source_manifest_sha256": source_manifest_sha256,
        "generator": {
            "code_commit": code_commit,
            "source_sha256": generator_source_sha256,
        },
        "rule": rule,
        "counts": counts,
        "hashes": hashes,
    }
    result: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "scene": scene,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "collection_hash": collection_hash,
        "generator": {
            "code_commit": code_commit,
            "source_sha256": generator_source_sha256,
        },
        "rule": rule,
        "counts": counts,
        "hashes": hashes,
        "lists": {
            "ordering": {"count": len(ordered_pair_ids), "sha256": hashes["pair_ordering_sha256"]},
            "train": {"count": len(train_pair_ids), "sha256": hashes["train_list_sha256"]},
            "test": {"count": len(test_pair_ids), "sha256": hashes["test_list_sha256"]},
        },
        "records": output_records,
        "validation": {
            "status": "passed",
            "expected_total": expected_total,
            "count_formula_passed": True,
            "pair_ids_unique": True,
            "train_test_disjoint": True,
            "train_test_complete": True,
            "labels_exactly_train_test": True,
            "zero_based_indices_contiguous": True,
            "source_audit_fields_preserved": True,
        },
    }
    result["split_hash"] = _hash_json(split_basis)
    return result


def build_collection_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    source_manifest: str,
    source_manifest_sha256: str,
    code_commit: str,
    generator_source_sha256: str,
    expected_scene_counts: Mapping[str, int] = EXPECTED_SCENE_COUNTS,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the complete fail-closed scene collection and its scene manifests."""

    if not records:
        raise Hold8ValidationError("collection contains no records")
    source_manifest_sha256 = _validate_sha256(
        source_manifest_sha256, "source_manifest_sha256"
    )
    code_commit = _validate_code_commit(code_commit)
    generator_source_sha256 = _validate_sha256(
        generator_source_sha256, "generator_source_sha256"
    )
    expected = {str(scene): int(count) for scene, count in expected_scene_counts.items()}
    if not expected or any(not scene or count <= 0 for scene, count in expected.items()):
        raise Hold8ValidationError("expected_scene_counts must contain positive scene counts")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        scene = record.get("scene")
        if not isinstance(scene, str) or not scene.strip():
            raise Hold8ValidationError("every collection record must contain a scene")
        grouped.setdefault(scene, []).append(record)
    if set(grouped) != set(expected):
        raise Hold8ValidationError(
            "collection scene set mismatch: "
            f"missing={sorted(set(expected) - set(grouped))} "
            f"extra={sorted(set(grouped) - set(expected))}"
        )
    observed_counts = {scene: len(scene_records) for scene, scene_records in grouped.items()}
    if observed_counts != expected:
        mismatches = {
            scene: {"expected": expected[scene], "observed": observed_counts[scene]}
            for scene in sorted(expected)
            if observed_counts[scene] != expected[scene]
        }
        raise Hold8ValidationError(f"collection scene count mismatch: {mismatches}")

    collection_hash_basis = {
        "source_manifest_sha256": source_manifest_sha256,
        "record_count": len(records),
        "scenes": [
            {
                "scene": scene,
                "record_count": len(grouped[scene]),
                "input_records_hash": _input_records_hash(grouped[scene]),
            }
            for scene in sorted(grouped)
        ],
    }
    collection_hash = _hash_json(collection_hash_basis)
    scene_manifests = {
        scene: build_scene_manifest(
            grouped[scene],
            scene=scene,
            source_manifest=source_manifest,
            source_manifest_sha256=source_manifest_sha256,
            collection_hash=collection_hash,
            code_commit=code_commit,
            generator_source_sha256=generator_source_sha256,
            expected_total=expected[scene],
        )
        for scene in sorted(grouped)
    }
    counts = {
        key: sum(manifest["counts"][key] for manifest in scene_manifests.values())
        for key in ("total", "train", "test")
    }
    expected_test = sum((count + HOLDOUT_PERIOD - 1) // HOLDOUT_PERIOD for count in expected.values())
    expected_counts = {
        "total": sum(expected.values()),
        "train": sum(expected.values()) - expected_test,
        "test": expected_test,
    }
    if counts != expected_counts:
        raise Hold8ValidationError(
            f"aggregate Hold-8 count mismatch: expected {expected_counts}, observed {counts}"
        )

    collection_split_basis = {
        "protocol_id": PROTOCOL_ID,
        "collection_hash": collection_hash,
        "generator": {
            "code_commit": code_commit,
            "source_sha256": generator_source_sha256,
        },
        "scenes": [
            {"scene": scene, "split_hash": manifest["split_hash"]}
            for scene, manifest in sorted(scene_manifests.items())
        ],
    }
    collection = {
        "schema_name": f"{SCHEMA_NAME}_collection",
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "collection_hash_basis": collection_hash_basis,
        "collection_hash": collection_hash,
        "generator": {
            "code_commit": code_commit,
            "source_sha256": generator_source_sha256,
        },
        "rule": next(iter(scene_manifests.values()))["rule"],
        "counts": counts,
        "expected_scene_counts": dict(sorted(expected.items())),
        "scene_count": len(scene_manifests),
        "scenes": [
            {
                "scene": scene,
                "counts": manifest["counts"],
                "input_records_sha256": manifest["hashes"]["input_records_sha256"],
                "pair_ordering_sha256": manifest["hashes"]["pair_ordering_sha256"],
                "train_list_sha256": manifest["hashes"]["train_list_sha256"],
                "test_list_sha256": manifest["hashes"]["test_list_sha256"],
                "split_hash": manifest["split_hash"],
            }
            for scene, manifest in sorted(scene_manifests.items())
        ],
        "collection_split_hash": _hash_json(collection_split_basis),
        "validation": {
            "status": "passed",
            "scene_set_exact": True,
            "scene_counts_exact": True,
            "aggregate_counts_exact": True,
            "all_scene_validations_passed": True,
            "labels_exactly_train_test": True,
        },
    }
    return collection, scene_manifests


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def materialize_collection(
    manifest_path: Path,
    output_root: Path,
    *,
    expected_scene_counts: Mapping[str, int] = EXPECTED_SCENE_COUNTS,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    records, _ = load_manifest(manifest_path)
    source_manifest_sha256 = _file_sha256(manifest_path)
    identity = resolve_generator_identity()
    collection, scene_manifests = build_collection_manifest(
        records,
        # The source SHA is authoritative.  Keeping only the basename makes
        # copied formal manifests byte-identical on Windows, 900, and 901.
        source_manifest=manifest_path.name,
        source_manifest_sha256=source_manifest_sha256,
        expected_scene_counts=expected_scene_counts,
        **identity,
    )

    intended = [output_root / "collection_manifest.json"]
    for scene in scene_manifests:
        intended.extend(
            [
                output_root / "scenes" / f"{scene}.split.json",
                output_root / "lists" / f"{scene}.ordering.txt",
                output_root / "lists" / f"{scene}.train.txt",
                output_root / "lists" / f"{scene}.test.txt",
            ]
        )
    existing = [path for path in intended if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Hold-8 output exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    scene_entries: list[dict[str, Any]] = []
    for scene, scene_manifest in sorted(scene_manifests.items()):
        pair_ids = [record["pair_id"] for record in scene_manifest["records"]]
        train_ids = [record["pair_id"] for record in scene_manifest["records"] if record["split"] == "train"]
        test_ids = [record["pair_id"] for record in scene_manifest["records"] if record["split"] == "test"]
        list_paths = {
            "ordering": output_root / "lists" / f"{scene}.ordering.txt",
            "train": output_root / "lists" / f"{scene}.train.txt",
            "test": output_root / "lists" / f"{scene}.test.txt",
        }
        list_values = {"ordering": pair_ids, "train": train_ids, "test": test_ids}
        for label, path in list_paths.items():
            content = _pair_list_bytes(list_values[label])
            expected_sha = scene_manifest["lists"][label]["sha256"]
            if _sha256_bytes(content) != expected_sha:
                raise Hold8ValidationError(f"{scene} {label} list hash changed before write")
            _write_bytes(path, content)

        scene_path = output_root / "scenes" / f"{scene}.split.json"
        _write_json(scene_path, scene_manifest)
        scene_entries.append(
            {
                **next(item for item in collection["scenes"] if item["scene"] == scene),
                "manifest": scene_path.relative_to(output_root).as_posix(),
                "manifest_sha256": _file_sha256(scene_path),
                "list_files": {
                    label: path.relative_to(output_root).as_posix()
                    for label, path in list_paths.items()
                },
            }
        )

    materialized = copy.deepcopy(collection)
    materialized["scenes"] = scene_entries
    materialized["materialization"] = {
        "output_root": ".",
        "path_semantics": "relative_to_manifest_directory",
        "scene_manifest_count": len(scene_entries),
        "list_file_count": 3 * len(scene_entries),
        "all_written_hashes_verified": True,
    }
    _write_json(output_root / "collection_manifest.json", materialized)
    return materialized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_collection(
        args.manifest,
        args.output_root,
        # The formal CLI is deliberately pinned to the registered eleven-scene
        # collection.  Unit tests and synthetic fixtures may still call the
        # Python API with an explicit count map, but a formal invocation cannot
        # override the collection contract from the command line.
        expected_scene_counts=EXPECTED_SCENE_COUNTS,
        overwrite=args.overwrite,
    )
    print(
        f"protocol={result['protocol_id']} scenes={result['scene_count']} "
        f"train={result['counts']['train']} test={result['counts']['test']} "
        f"collection_hash={result['collection_hash']} "
        f"split_hash={result['collection_split_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
