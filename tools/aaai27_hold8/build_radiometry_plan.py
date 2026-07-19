#!/usr/bin/env python3
"""Build a deterministic train-only radiometry plan for Hold-8 v2.

The plan binds immutable Hold-8 split manifests to split-independent decoded
temperature and camera assets.  It does not estimate a range or render images;
those CPU jobs consume the plan during Phase 1 preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.thermal_radiometry.build_hold8_split import (
    EXPECTED_SCENE_COUNTS,
    HOLDOUT_PERIOD,
    PROTOCOL_ID,
    SCHEMA_NAME as HOLD8_SPLIT_SCHEMA,
    natural_sort_key,
)
from tools.thermal_radiometry.palette_lut import hot_iron_lut, lut_sha256


SCHEMA = "uav-tgs-aaai27-hold8-radiometry-plan-v2"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def json_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def pair_list_hash(values: Sequence[str]) -> str:
    payload = ("\n".join(values) + ("\n" if values else "")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_tree_hash(path: Path) -> str:
    """Hash regular files by relative path, byte count, and content digest."""

    rows: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"directory asset contains a symlink: {child}")
        if child.is_file():
            rows.append(
                {
                    "relative_path": child.relative_to(path).as_posix(),
                    "size_bytes": child.stat().st_size,
                    "sha256": file_hash(child),
                }
            )
    return json_hash(rows)


def load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def require_sha(value: Any, label: str) -> str:
    token = str(value or "").lower()
    if not SHA_RE.fullmatch(token):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return token


def require_commit(value: Any, label: str) -> str:
    token = str(value or "").lower()
    if not COMMIT_RE.fullmatch(token):
        raise ValueError(f"{label} must be a full lowercase 40-hex Git commit")
    return token


def resolve_asset(path_text: Any, *, inventory_dir: Path, label: str) -> Path:
    path = Path(str(path_text or ""))
    if not str(path):
        raise ValueError(f"{label} path is missing")
    if not path.is_absolute():
        path = inventory_dir / path
    path = path.resolve()
    if not path.is_file() and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def verify_asset(record: Mapping[str, Any], *, inventory_dir: Path, label: str) -> dict[str, Any]:
    path = resolve_asset(record.get("path"), inventory_dir=inventory_dir, label=label)
    kind = "directory" if path.is_dir() else "file"
    result = {"path": str(path), "kind": kind}
    if kind == "file":
        actual = file_hash(path)
        expected = record.get("sha256")
        if expected is not None and actual != require_sha(expected, f"{label}.sha256"):
            raise ValueError(f"{label} SHA mismatch")
        result["sha256"] = actual
    else:
        file_count = sum(1 for child in path.rglob("*") if child.is_file())
        expected_count = record.get("file_count")
        if expected_count is not None and file_count != int(expected_count):
            raise ValueError(
                f"{label} file-count mismatch: expected {expected_count}, observed {file_count}"
            )
        result["file_count"] = file_count
        tree_hash = record.get("tree_sha256")
        if tree_hash is not None:
            expected_tree_hash = require_sha(tree_hash, f"{label}.tree_sha256")
            actual_tree_hash = directory_tree_hash(path)
            if actual_tree_hash != expected_tree_hash:
                raise ValueError(f"{label} directory-tree SHA mismatch")
            result["tree_sha256"] = actual_tree_hash
    return result


def require_content_hash(asset: Mapping[str, Any], label: str) -> None:
    if asset.get("kind") == "directory" and asset.get("tree_sha256") is None:
        raise ValueError(f"{label} directory requires tree_sha256")


def verify_jsonl_scene_count(path: Path, *, scene: str, expected_count: int, label: str) -> None:
    pair_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} line {line_number} is not an object")
            if row.get("scene") != scene:
                raise ValueError(f"{label} contains another scene")
            pair_id = str(row.get("pair_id", "")).strip()
            if not pair_id or pair_id in pair_ids:
                raise ValueError(f"{label} contains a missing/duplicate pair ID")
            pair_ids.add(pair_id)
    if len(pair_ids) != expected_count:
        raise ValueError(
            f"{label} record-count mismatch: expected {expected_count}, observed {len(pair_ids)}"
        )


def _scene_entry_index(collection: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if (
        collection.get("schema_name") != f"{HOLD8_SPLIT_SCHEMA}_collection"
        or int(collection.get("schema_version", -1)) != 1
        or collection.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError("collection is not the formal AAAI27 Hold-8 v2 schema")
    validation = collection.get("validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "passed":
        raise ValueError("Hold-8 collection validation is not passed")
    entries = collection.get("scenes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Hold-8 collection must contain scene entries")
    result: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Hold-8 collection scene entries must be objects")
        scene = str(entry.get("scene", ""))
        if not scene or scene in result:
            raise ValueError("Hold-8 collection has missing/duplicate scene")
        result[scene] = entry
    return result


def build_plan(
    *,
    collection_path: Path,
    split_root: Path,
    asset_inventory_path: Path,
    code_commit: str,
    enforce_formal_collection: bool = True,
) -> dict[str, Any]:
    collection_path = collection_path.resolve()
    split_root = split_root.resolve()
    asset_inventory_path = asset_inventory_path.resolve()
    collection = load_object(collection_path, "Hold-8 collection")
    assets = load_object(asset_inventory_path, "asset inventory")
    code_commit = require_commit(code_commit, "code_commit")
    collection_hash = require_sha(collection.get("collection_hash"), "collection_hash")
    collection_split_hash = require_sha(
        collection.get("collection_split_hash"), "collection_split_hash"
    )
    if assets.get("collection_hash") != collection_hash:
        raise ValueError("asset inventory collection hash differs from Hold-8 collection")
    asset_scenes = assets.get("scenes")
    if not isinstance(asset_scenes, Mapping):
        raise ValueError("asset inventory must contain a scenes object")
    scene_entries = _scene_entry_index(collection)
    if enforce_formal_collection:
        expected_counts = {
            scene: {
                "total": total,
                "train": total - ((total + HOLDOUT_PERIOD - 1) // HOLDOUT_PERIOD),
                "test": (total + HOLDOUT_PERIOD - 1) // HOLDOUT_PERIOD,
            }
            for scene, total in EXPECTED_SCENE_COUNTS.items()
        }
        observed_counts = {
            scene: dict(entry.get("counts", {}))
            for scene, entry in scene_entries.items()
        }
        if set(scene_entries) != set(EXPECTED_SCENE_COUNTS):
            raise ValueError("Hold-8 plan requires the exact registered eleven-scene set")
        if observed_counts != expected_counts:
            raise ValueError("Hold-8 plan scene counts differ from the registered collection")
        aggregate = {
            key: sum(row[key] for row in expected_counts.values())
            for key in ("total", "train", "test")
        }
        if collection.get("counts") != aggregate:
            raise ValueError("Hold-8 plan aggregate counts differ from 8232/7198/1034")
    if set(asset_scenes) != set(scene_entries):
        raise ValueError("asset inventory scene set differs from Hold-8 collection")

    lut_record = assets.get("fixed_hotiron_lut")
    if not isinstance(lut_record, Mapping):
        raise ValueError("asset inventory lacks fixed_hotiron_lut")
    lut = verify_asset(lut_record, inventory_dir=asset_inventory_path.parent, label="LUT")
    lut["palette_name"] = str(lut_record.get("palette_name", "uav-tgs-hot-iron-v1"))
    generated_lut = hot_iron_lut()
    generated_lut_sha = lut_sha256(generated_lut)
    generated_unique_count = int(len({tuple(row) for row in generated_lut.tolist()}))
    declared_lut_sha = require_sha(lut_record.get("lut_rgb_sha256"), "LUT.lut_rgb_sha256")
    if declared_lut_sha != generated_lut_sha:
        raise ValueError("fixed Hot-Iron LUT RGB SHA differs from the repository implementation")
    unique_color_count = int(lut_record.get("unique_color_count", -1))
    if unique_color_count != 256 or unique_color_count != generated_unique_count:
        raise ValueError("fixed Hot-Iron LUT must contain 256 verified unique colors")
    lut["lut_rgb_sha256"] = generated_lut_sha
    lut["unique_color_count"] = generated_unique_count
    lut["verification"] = "generated_from_repository_palette_lut_and_compared_to_inventory"

    planned_scenes: list[dict[str, Any]] = []
    availability = {
        "decoded_temperature_available": [],
        "decoded_temperature_requires_decode": [],
        "shared_sfm_available": [],
        "shared_sfm_requires_build": [],
        "valid_mask_available": [],
        "valid_mask_requires_build": [],
    }
    for scene in sorted(scene_entries):
        entry = scene_entries[scene]
        relative_manifest = str(entry.get("manifest", f"scenes/{scene}.split.json"))
        split_path = (split_root / relative_manifest).resolve()
        if not split_path.is_file():
            raise FileNotFoundError(split_path)
        split_sha = file_hash(split_path)
        declared_sha = entry.get("manifest_sha256")
        if declared_sha is not None and split_sha != require_sha(
            declared_sha, f"{scene}.manifest_sha256"
        ):
            raise ValueError(f"{scene} split manifest SHA mismatch")
        split = load_object(split_path, f"{scene} split")
        if (
            split.get("schema_name") != HOLD8_SPLIT_SCHEMA
            or int(split.get("schema_version", -1)) != 1
            or split.get("protocol_id") != PROTOCOL_ID
            or split.get("scene") != scene
        ):
            raise ValueError(f"{scene} is not a formal Hold-8 v2 scene manifest")
        validation = split.get("validation")
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            raise ValueError(f"{scene} Hold-8 validation is not passed")
        if split.get("collection_hash") != collection_hash:
            raise ValueError(f"{scene} collection hash mismatch")
        if entry.get("split_hash") != split.get("split_hash"):
            raise ValueError(f"{scene} collection entry split hash mismatch")
        records = split.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"{scene} split has no records")
        labels = [str(record.get("split", "")) for record in records if isinstance(record, Mapping)]
        if len(labels) != len(records) or set(labels) != {"train", "test"}:
            raise ValueError(f"{scene} split must contain only train/test records")
        pair_ids: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"{scene} split record is not an object")
            pair_id = str(record.get("pair_id", "")).strip()
            if not pair_id or int(record.get("zero_based_sorted_index", -1)) != index:
                raise ValueError(f"{scene} pair/index contract failed")
            expected_label = "test" if index % HOLDOUT_PERIOD == 0 else "train"
            if record.get("split") != expected_label:
                raise ValueError(f"{scene} modulo-8 membership failed")
            pair_ids.append(pair_id)
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(f"{scene} has duplicate pair IDs")
        if pair_ids != sorted(pair_ids, key=lambda value: (natural_sort_key(value), value)):
            raise ValueError(f"{scene} pair IDs are not in canonical natural order")
        counts = {
            "total": len(labels),
            "train": labels.count("train"),
            "test": labels.count("test"),
        }
        if split.get("counts") != counts:
            raise ValueError(f"{scene} split counts mismatch")
        hashes = split.get("hashes")
        if not isinstance(hashes, Mapping):
            raise ValueError(f"{scene} split lacks hashes")
        train_ids = [
            pair_id
            for pair_id, label in zip(pair_ids, labels)
            if label == "train"
        ]
        test_ids = [
            pair_id
            for pair_id, label in zip(pair_ids, labels)
            if label == "test"
        ]
        for label, values in (
            ("pair_ordering", pair_ids),
            ("train_list", train_ids),
            ("test_list", test_ids),
        ):
            if pair_list_hash(values) != require_sha(
                hashes.get(f"{label}_sha256"), f"{scene}.{label}_sha256"
            ):
                raise ValueError(f"{scene} {label} hash mismatch")
        split_hash = require_sha(split.get("split_hash"), f"{scene}.split_hash")
        if entry.get("split_hash") != split_hash:
            raise ValueError(f"{scene} split hash is not bound by collection")

        scene_assets = asset_scenes[scene]
        if not isinstance(scene_assets, Mapping):
            raise ValueError(f"asset inventory scene {scene} must be an object")
        raw_thermal_root = None
        if scene_assets.get("raw_thermal_root") is not None:
            raw_thermal_root = verify_asset(
                scene_assets["raw_thermal_root"],
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.raw_thermal_root",
            )
            if raw_thermal_root.get("file_count") != counts["total"]:
                raise ValueError(f"{scene} raw thermal count differs from Hold-8 collection")
            require_content_hash(raw_thermal_root, f"{scene}.raw_thermal_root")

        if scene_assets.get("cfr_manifest") is None:
            raise ValueError(f"{scene} requires a frozen CFR manifest")
        cfr_manifest = verify_asset(
            scene_assets["cfr_manifest"],
            inventory_dir=asset_inventory_path.parent,
            label=f"{scene}.cfr_manifest",
        )

        radiometry_status = str(
            scene_assets.get(
                "radiometry_status",
                "available"
                if scene_assets.get("decode_manifest") is not None
                and scene_assets.get("temperature_root") is not None
                else "decode_required",
            )
        )
        if radiometry_status not in {"available", "decode_required"}:
            raise ValueError(f"{scene} has unsupported radiometry_status")
        decoded = None
        decode_protocol = None
        temperature_root = None
        if radiometry_status == "available":
            decoded = verify_asset(
                scene_assets.get("decode_manifest", {}),
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.decode_manifest",
            )
            verify_jsonl_scene_count(
                Path(decoded["path"]),
                scene=scene,
                expected_count=counts["total"],
                label=f"{scene}.decode_manifest",
            )
            if scene_assets.get("decode_protocol") is None:
                raise ValueError(f"{scene} available radiometry needs decode_protocol")
            decode_protocol = verify_asset(
                scene_assets["decode_protocol"],
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.decode_protocol",
            )
            verify_jsonl_scene_count(
                Path(decode_protocol["path"]),
                scene=scene,
                expected_count=counts["total"],
                label=f"{scene}.decode_protocol",
            )
            temperature_root = verify_asset(
                scene_assets.get("temperature_root", {}),
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.temperature_root",
            )
            require_content_hash(temperature_root, f"{scene}.temperature_root")
            if temperature_root.get("file_count") != counts["total"]:
                raise ValueError(f"{scene} temperature count differs from Hold-8 collection")
            availability["decoded_temperature_available"].append(scene)
        else:
            if raw_thermal_root is None:
                raise ValueError(f"{scene} decode_required needs raw_thermal_root")
            require_content_hash(raw_thermal_root, f"{scene}.raw_thermal_root")
            availability["decoded_temperature_requires_decode"].append(scene)

        camera_status = str(
            scene_assets.get(
                "camera_status",
                "available"
                if scene_assets.get("camera_manifest") is not None
                or scene_assets.get("sparse_model") is not None
                else "build_required",
            )
        )
        if camera_status not in {"available", "build_required"}:
            raise ValueError(f"{scene} has unsupported camera_status")
        camera = None
        if camera_status == "available":
            camera_record = scene_assets.get("sparse_model", scene_assets.get("camera_manifest"))
            if not isinstance(camera_record, Mapping):
                raise ValueError(f"{scene} available camera asset is missing")
            camera = verify_asset(
                camera_record,
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.shared_sfm",
            )
            require_content_hash(camera, f"{scene}.shared_sfm")
            if str(scene_assets.get("sfm_scope")) != "shared_sfm_all_images":
                raise ValueError(f"{scene} must declare sfm_scope=shared_sfm_all_images")
            availability["shared_sfm_available"].append(scene)
        else:
            if cfr_manifest is None:
                raise ValueError(f"{scene} build_required needs a CFR manifest")
            availability["shared_sfm_requires_build"].append(scene)
        valid_mask_status = str(
            scene_assets.get(
                "valid_mask_status",
                "available"
                if scene_assets.get("valid_mask_manifest") is not None
                else "build_required",
            )
        )
        if valid_mask_status not in {"available", "build_required"}:
            raise ValueError(f"{scene} has unsupported valid_mask_status")
        valid_mask = None
        if valid_mask_status == "available":
            if not isinstance(scene_assets["valid_mask_manifest"], Mapping):
                raise ValueError(f"{scene}.valid_mask_manifest must be an object")
            valid_mask = verify_asset(
                scene_assets["valid_mask_manifest"],
                inventory_dir=asset_inventory_path.parent,
                label=f"{scene}.valid_mask_manifest",
            )
            require_content_hash(valid_mask, f"{scene}.valid_mask_manifest")
            availability["valid_mask_available"].append(scene)
        else:
            availability["valid_mask_requires_build"].append(scene)
        output_base = f"derived/thermal_radiometry/aaai27_hold8_v2/{scene}"
        stages: list[dict[str, Any]] = []
        if radiometry_status == "decode_required":
            stages.append(
                {
                    "id": "decode_split_independent_full_frame_temperature",
                    "reads": ["all raw R-JPEGs", "frozen radiometry protocol"],
                    "writes": f"{output_base}/temperature_c",
                    "test_membership_used_for_fitting": False,
                }
            )
        stages.extend(
            [
                {
                    "id": "bind_decoded_temperature",
                    "reads": ["all pair identities", "full-frame float32 Celsius"],
                    "writes": f"{output_base}/binding/bound_split.json",
                },
                {
                    "id": "estimate_train_only_range",
                    "reads": ["train temperature only"],
                    "test_role": "post-estimation clipping QA only",
                    "writes": f"{output_base}/radiometry/range_manifest.json",
                },
                {
                    "id": "render_canonical_hotiron",
                    "reads": ["all temperature maps", "frozen train-only range", "fixed LUT"],
                    "writes": f"{output_base}/canonical_hotiron",
                },
            ]
        )
        if valid_mask_status == "build_required":
            stages.append(
                {
                    "id": "materialize_split_independent_valid_masks",
                    "reads": ["decoded/undistorted support for all frames"],
                    "writes": f"{output_base}/qa/valid_support_manifest.json",
                    "test_membership_used_for_fitting": False,
                }
            )
        stages.append(
            {
                "id": "freeze_train_only_hotspot",
                "reads": ["train temperature and split-independent valid masks only"],
                "writes": f"{output_base}/radiometry/hotspot_threshold_train_q95.json",
            }
        )
        if camera_status == "build_required":
            stages.append(
                {
                    "id": "build_shared_full_collection_sfm",
                    "reads": ["all CFR RGB observations"],
                    "writes": f"derived/aaai27_hold8_v2/{scene}/shared_sfm",
                    "cost_scope": "common preprocessing",
                }
            )
        stages.append(
            {
                "id": "build_train_only_reference",
                "reads": ["train RGB images", "shared full-collection poses"],
                "test_role": "render reference depth only",
                "writes": f"experiments/aaai27_hold8_v2/{scene}/reference_train_only",
            }
        )
        pending_assets = [
            label
            for label, status in (
                ("full_frame_temperature_decode", radiometry_status),
                ("shared_full_collection_sfm", camera_status),
                ("split_independent_valid_mask_manifest", valid_mask_status),
            )
            if status != "available"
        ]
        planned_scenes.append(
            {
                "scene": scene,
                "counts": counts,
                "preparation_status": (
                    "inputs_verified" if not pending_assets else "asset_build_required"
                ),
                "pending_assets": pending_assets,
                "split": {
                    "path": str(split_path),
                    "sha256": split_sha,
                    "split_hash": require_sha(split.get("split_hash"), f"{scene}.split_hash"),
                },
                "split_independent_inputs": {
                    "raw_thermal_root": raw_thermal_root,
                    "cfr_manifest": cfr_manifest,
                    "radiometry_status": radiometry_status,
                    "decode_manifest": decoded,
                    "decode_protocol": decode_protocol,
                    "temperature_root": temperature_root,
                    "valid_mask_status": valid_mask_status,
                    "valid_mask_manifest": valid_mask,
                    "camera_status": camera_status,
                    "shared_sfm": camera,
                    "sfm_scope": (
                        "shared_sfm_all_images" if camera_status == "available" else "to_build"
                    ),
                },
                "stages": stages,
            }
        )

    basis = {
        "schema": SCHEMA,
        "collection_hash": collection_hash,
        "collection_split_hash": collection_split_hash,
        "collection_manifest_sha256": file_hash(collection_path),
        "fixed_hotiron_lut": lut,
        "code_commit": code_commit,
        "scenes": planned_scenes,
        "availability": availability,
        "preparation_summary": {
            "inputs_verified_scenes": [
                item["scene"]
                for item in planned_scenes
                if item["preparation_status"] == "inputs_verified"
            ],
            "asset_build_required_scenes": [
                item["scene"]
                for item in planned_scenes
                if item["preparation_status"] == "asset_build_required"
            ],
            "formal_training_authorized": False,
        },
        "data_roles": {
            "train": ["range estimation", "hotspot threshold", "photometric fitting", "OpenMVS reconstruction"],
            "test": ["range clipping QA", "final evaluation only"],
            "guard": "absent",
            "validation": "absent",
        },
        "pose_policy": (
            "pose reconstruction is common preprocessing; photometric model training reads "
            "only Hold-8 train images"
        ),
    }
    basis["plan_sha256"] = json_hash(basis)
    return basis


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--asset-inventory", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_plan(
        collection_path=args.collection,
        split_root=args.split_root,
        asset_inventory_path=args.asset_inventory,
        code_commit=args.code_commit,
    )
    write_json(args.output.resolve(), payload)
    print(json.dumps({"status": "planned", "plan_sha256": payload["plan_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
