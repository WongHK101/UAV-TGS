#!/usr/bin/env python3
"""Freeze the deterministic 11-scene AAAI split protocol.

The formal recipe is deliberately not configurable: scene-level block budget,
stratum-aware allocation, 16-frame blocks, period 8, four guard frames on each
side, and at least 16 remaining train frames in every related strip/stratum.
Guard=2 remains available only through ``split_qa.py`` as development evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # pragma: no cover - direct-script import path
    from . import build_split
except ImportError:  # pragma: no cover
    import build_split  # type: ignore


SCHEMA_NAME = "uav_tgs_aaai27_formal_split_collection"
SCHEMA_VERSION = 1
FORMAL_BLOCK_SIZE = 16
FORMAL_TEST_PERIOD_BLOCKS = 8
FORMAL_GUARD_FRAMES = 4
FORMAL_MINIMUM_TRAIN_FRAMES = 16
DEFAULT_SEED = "uav-tgs-aaai27-v1"
EXPECTED_SCENES = (
    "Building",
    "Garden",
    "InternalRoad",
    "Orchard",
    "PVpanel",
    "Plaza",
    "Road",
    "TransmissionTower",
    "Urban100K",
    "Urban20K",
    "Urban50K",
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_summary_csv(path: Path, scenes: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "scene",
                "total",
                "train",
                "test",
                "guard",
                "test_block_target",
                "test_block_selected",
                "strata_without_test_count",
                "ordering_mode",
                "input_records_hash",
                "selected_test_blocks_hash",
                "split_hash",
                "manifest_sha256",
            ),
        )
        writer.writeheader()
        for scene in scenes:
            writer.writerow(
                {
                    "scene": scene["scene"],
                    **scene["counts"],
                    "test_block_target": scene["test_block_budget"]["target"],
                    "test_block_selected": scene["test_block_budget"]["selected"],
                    "strata_without_test_count": len(scene["strata_without_test"]),
                    "ordering_mode": scene["ordering_mode"],
                    "input_records_hash": scene["input_records_hash"],
                    "selected_test_blocks_hash": scene["selected_test_blocks_hash"],
                    "split_hash": scene["split_hash"],
                    "manifest_sha256": scene["manifest_sha256"],
                }
            )
    os.replace(temporary, path)


def _formal_rule(seed: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "scene_test_block_budget": (
            "round_half_up(scene_frame_count / "
            "(block_size_frames * test_period_blocks))"
        ),
        "stratum_allocation": "largest_remainder_over_feasible_capacity",
        "block_size_frames": FORMAL_BLOCK_SIZE,
        "test_period_blocks": FORMAL_TEST_PERIOD_BLOCKS,
        "guard_frames_each_side": FORMAL_GUARD_FRAMES,
        "minimum_train_frames_per_related_strip_or_stratum": (
            FORMAL_MINIMUM_TRAIN_FRAMES
        ),
        "strata_without_test_allowed": True,
        "partial_tail_block_can_be_test": False,
        "fail_closed_on": [
            "test_block_budget_shortfall",
            "strip_without_train",
            "stratum_without_train",
            "selected_strip_below_16_train",
            "selected_stratum_below_16_train",
            "test_fraction_out_of_range",
        ],
        "guard_2_status": "development_qa_only",
    }


def build_formal_collection(
    records: Sequence[Mapping[str, Any]],
    *,
    source_manifest: Path,
    source_manifest_sha256: str,
    seed: str = DEFAULT_SEED,
    expected_scenes: Sequence[str] = EXPECTED_SCENES,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if seed != DEFAULT_SEED:
        raise ValueError(
            f"formal seed is frozen to {DEFAULT_SEED!r}; received {seed!r}"
        )
    if build_split.MIN_TRAIN_FRAMES_AFTER_TEST != FORMAL_MINIMUM_TRAIN_FRAMES:
        raise RuntimeError("build_split minimum-train constant diverges from formal protocol")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        scene_value = record.get("scene")
        if scene_value in (None, ""):
            raise ValueError("every formal collection record must contain scene")
        grouped.setdefault(str(scene_value), []).append(record)

    expected = set(str(scene) for scene in expected_scenes)
    actual = set(grouped)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"formal scene set mismatch: missing={missing} extra={extra}")

    source_path = str(source_manifest.resolve())
    scene_manifests: dict[str, dict[str, Any]] = {}
    for scene in sorted(grouped):
        scene_manifests[scene] = build_split.build_split_manifest(
            grouped[scene],
            scene=scene,
            seed=seed,
            source_manifest=source_path,
            source_manifest_sha256=source_manifest_sha256,
            block_size=FORMAL_BLOCK_SIZE,
            test_period_blocks=FORMAL_TEST_PERIOD_BLOCKS,
            guard_frames=FORMAL_GUARD_FRAMES,
            fail_on_invalid=True,
        )

    collection_basis = {
        "source_manifest_sha256": source_manifest_sha256,
        "record_count": len(records),
        "scenes": [
            {
                "scene": scene,
                "record_count": manifest["counts"]["total"],
                "input_records_hash": manifest["input_records_hash"],
            }
            for scene, manifest in sorted(scene_manifests.items())
        ],
    }
    collection_hash = _hash_json(collection_basis)
    formal_rule = _formal_rule(seed)
    formal_rule_hash = _hash_json(formal_rule)
    split_basis = {
        "collection_hash": collection_hash,
        "formal_rule_hash": formal_rule_hash,
        "scenes": [
            {
                "scene": scene,
                "split_hash": manifest["split_hash"],
                "selected_test_blocks_hash": manifest["selected_test_blocks_hash"],
            }
            for scene, manifest in sorted(scene_manifests.items())
        ],
    }

    validation = {
        "status": "passed",
        "scene_set_exact": True,
        "all_scene_validations_passed": all(
            manifest["validation"]["status"] == "passed"
            for manifest in scene_manifests.values()
        ),
        "no_strip_without_train": all(
            manifest["validation"]["no_strip_without_train"]
            for manifest in scene_manifests.values()
        ),
        "no_stratum_without_train": all(
            manifest["validation"]["no_stratum_without_train"]
            for manifest in scene_manifests.values()
        ),
        "strata_without_test_allowed": True,
    }
    if not all(value is True for key, value in validation.items() if key != "status"):
        raise build_split.SplitAllocationError(
            "formal collection failed an aggregate split invariant"
        )

    collection = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "source_manifest": source_path,
        "source_manifest_sha256": source_manifest_sha256,
        "collection_hash_basis": collection_basis,
        "collection_hash": collection_hash,
        "formal_rule": formal_rule,
        "formal_rule_hash": formal_rule_hash,
        "collection_split_hash": _hash_json(split_basis),
        "counts": {
            key: sum(manifest["counts"][key] for manifest in scene_manifests.values())
            for key in ("total", "train", "test", "guard")
        },
        "scene_count": len(scene_manifests),
        "expected_scenes": sorted(expected),
        "validation": validation,
    }
    return collection, scene_manifests


def materialise_formal_collection(
    manifest: Path,
    output_root: Path,
    *,
    seed: str = DEFAULT_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest = manifest.resolve()
    output_root = output_root.resolve()
    records, _ = build_split.load_manifest(manifest)
    source_sha256 = _file_sha256(manifest)
    collection, scene_manifests = build_formal_collection(
        records,
        source_manifest=manifest,
        source_manifest_sha256=source_sha256,
        seed=seed,
    )

    scene_dir = output_root / "scenes"
    collection_path = output_root / "collection_manifest.json"
    summary_path = output_root / "scene_summary.csv"
    intended_paths = [
        collection_path,
        summary_path,
        *(scene_dir / f"{scene}.split.json" for scene in sorted(scene_manifests)),
    ]
    existing = [path for path in intended_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "formal split output exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    scene_entries: list[dict[str, Any]] = []
    for scene, scene_manifest in sorted(scene_manifests.items()):
        scene_path = scene_dir / f"{scene}.split.json"
        _write_json(scene_path, scene_manifest)
        scene_entries.append(
            {
                "scene": scene,
                "manifest": scene_path.relative_to(output_root).as_posix(),
                "manifest_sha256": _file_sha256(scene_path),
                "counts": scene_manifest["counts"],
                "test_block_budget": scene_manifest["test_block_budget"],
                "ordering_mode": scene_manifest["rule"]["ordering_mode"],
                "input_records_hash": scene_manifest["input_records_hash"],
                "rule_hash": scene_manifest["rule_hash"],
                "allocation_hash": scene_manifest["allocation_hash"],
                "selected_test_blocks_hash": scene_manifest[
                    "selected_test_blocks_hash"
                ],
                "selected_candidate_hashes": scene_manifest[
                    "selected_candidate_hashes"
                ],
                "split_hash": scene_manifest["split_hash"],
                "strata_without_test": [
                    allocation["stratum"]
                    for allocation in scene_manifest["stratum_allocations"]
                    if allocation["without_test"]
                ],
                "validation": scene_manifest["validation"],
            }
        )

    collection["scenes"] = scene_entries
    collection["scene_manifests_hash"] = _hash_json(
        [
            {
                "scene": entry["scene"],
                "manifest_sha256": entry["manifest_sha256"],
            }
            for entry in scene_entries
        ]
    )
    collection["collection_manifest_content_hash"] = _hash_json(collection)
    _write_json(collection_path, collection)
    _write_summary_csv(summary_path, scene_entries)
    return collection


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialise_formal_collection(
        args.manifest,
        args.output_root,
        overwrite=bool(args.overwrite),
    )
    print(
        f"scenes={result['scene_count']} train={result['counts']['train']} "
        f"test={result['counts']['test']} guard={result['counts']['guard']} "
        f"collection_hash={result['collection_hash']} "
        f"collection_split_hash={result['collection_split_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
