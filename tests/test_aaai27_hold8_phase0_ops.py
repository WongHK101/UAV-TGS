from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.aaai27_hold8.build_phase1_queue import build_queue
from tools.aaai27_hold8.build_radiometry_plan import (
    build_plan,
    directory_tree_hash,
    file_hash,
)
from tools.aaai27_hold8.inventory_guard4_cleanup import classify, classify_preserved
from tools.thermal_radiometry.build_hold8_split import build_collection_manifest
from tools.thermal_radiometry.build_hold8_split import (
    EXPECTED_SCENE_COUNTS,
    PROTOCOL_ID,
    SCHEMA_NAME as SPLIT_SCHEMA_NAME,
)
from tools.thermal_radiometry.palette_lut import hot_iron_lut, lut_sha256


SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path):
    split_root = tmp_path / "splits"
    source_records = [
        {"scene": scene, "pair_id": f"{index:04d}"}
        for scene in SCENES
        for index in range(9)
    ]
    collection, generated_scenes = build_collection_manifest(
        source_records,
        source_manifest="fixture.jsonl",
        source_manifest_sha256=_sha(b"fixture-source"),
        code_commit="a" * 40,
        generator_source_sha256=_sha(b"fixture-generator"),
        expected_scene_counts={scene: 9 for scene in SCENES},
    )
    scene_entries = []
    inventory_scenes = {}
    for scene in SCENES:
        split = generated_scenes[scene]
        split_path = split_root / "scenes" / f"{scene}.split.json"
        _write_json(split_path, split)
        generated_entry = next(item for item in collection["scenes"] if item["scene"] == scene)
        scene_entries.append(
            {
                **generated_entry,
                "manifest": f"scenes/{scene}.split.json",
                "manifest_sha256": file_hash(split_path),
            }
        )
        decode = tmp_path / "assets" / scene / "decode.jsonl"
        protocol = tmp_path / "assets" / scene / "decode_protocol.jsonl"
        camera = tmp_path / "assets" / scene / "cameras.json"
        cfr = tmp_path / "assets" / scene / "cfr.json"
        raw = tmp_path / "assets" / scene / "raw"
        temperature = tmp_path / "assets" / scene / "temperature"
        decode.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"scene": scene, "pair_id": f"{index:04d}", "success": True}
            for index in range(9)
        ]
        jsonl = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        decode.write_text(jsonl, encoding="utf-8")
        protocol.write_text(jsonl, encoding="utf-8")
        camera.write_text("{}\n", encoding="utf-8")
        cfr.write_text(json.dumps({"scene": scene}) + "\n", encoding="utf-8")
        raw.mkdir()
        temperature.mkdir()
        for index in range(9):
            (raw / f"{index:04d}.JPG").write_bytes(f"raw-{index}".encode())
            (temperature / f"{index:04d}.npy").write_bytes(f"temperature-{index}".encode())
        inventory_scenes[scene] = {
            "decode_manifest": {"path": str(decode), "sha256": file_hash(decode)},
            "decode_protocol": {"path": str(protocol), "sha256": file_hash(protocol)},
            "temperature_root": {
                "path": str(temperature),
                "file_count": 9,
                "tree_sha256": directory_tree_hash(temperature),
            },
            "raw_thermal_root": {
                "path": str(raw),
                "file_count": 9,
                "tree_sha256": directory_tree_hash(raw),
            },
            "cfr_manifest": {"path": str(cfr), "sha256": file_hash(cfr)},
            "camera_manifest": {"path": str(camera), "sha256": file_hash(camera)},
            "sfm_scope": "shared_sfm_all_images",
        }
    collection["scenes"] = scene_entries
    collection_path = split_root / "collection_manifest.json"
    _write_json(collection_path, collection)
    lut = tmp_path / "assets" / "hotiron.npy"
    lut.parent.mkdir(parents=True, exist_ok=True)
    lut.write_bytes(b"fixed-lut")
    inventory = {
        "collection_hash": collection["collection_hash"],
        "fixed_hotiron_lut": {
            "path": str(lut),
            "sha256": file_hash(lut),
            "lut_rgb_sha256": lut_sha256(hot_iron_lut()),
            "unique_color_count": 256,
        },
        "scenes": inventory_scenes,
    }
    inventory_path = tmp_path / "assets" / "inventory.json"
    _write_json(inventory_path, inventory)
    return collection, collection_path, split_root, inventory_path


def _formal_queue_collection() -> dict:
    scene_rows = []
    aggregate = {"total": 0, "train": 0, "test": 0}
    for scene, total in sorted(EXPECTED_SCENE_COUNTS.items()):
        test = (total + 7) // 8
        counts = {"total": total, "train": total - test, "test": test}
        for key in aggregate:
            aggregate[key] += counts[key]
        scene_rows.append(
            {
                "scene": scene,
                "counts": counts,
                "split_hash": _sha(f"split:{scene}".encode()),
            }
        )
    collection_hash_basis = {
        "source_manifest_sha256": _sha(b"formal-source"),
        "record_count": aggregate["total"],
        "scenes": [
            {
                "scene": row["scene"],
                "record_count": row["counts"]["total"],
                "input_records_hash": _sha(f"records:{row['scene']}".encode()),
            }
            for row in scene_rows
        ],
    }
    collection_hash = hashlib.sha256(
        json.dumps(
            collection_hash_basis,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    generator = {"code_commit": "a" * 40, "source_sha256": "b" * 64}
    collection_split_basis = {
        "protocol_id": PROTOCOL_ID,
        "collection_hash": collection_hash,
        "generator": generator,
        "scenes": [
            {"scene": row["scene"], "split_hash": row["split_hash"]}
            for row in scene_rows
        ],
    }
    collection_split_hash = hashlib.sha256(
        json.dumps(
            collection_split_basis,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return {
        "schema_name": f"{SPLIT_SCHEMA_NAME}_collection",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "collection_hash_basis": collection_hash_basis,
        "collection_hash": collection_hash,
        "generator": generator,
        "collection_split_hash": collection_split_hash,
        "expected_scene_counts": dict(sorted(EXPECTED_SCENE_COUNTS.items())),
        "scene_count": len(EXPECTED_SCENE_COUNTS),
        "counts": aggregate,
        "scenes": scene_rows,
        "validation": {
            "status": "passed",
            "scene_set_exact": True,
            "scene_counts_exact": True,
            "aggregate_counts_exact": True,
            "all_scene_validations_passed": True,
            "labels_exactly_train_test": True,
        },
    }


def test_radiometry_plan_is_train_only_and_deterministic(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    first = build_plan(
        collection_path=collection_path,
        split_root=split_root,
        asset_inventory_path=inventory_path,
        code_commit="a" * 40,
        enforce_formal_collection=False,
    )
    second = build_plan(
        collection_path=collection_path,
        split_root=split_root,
        asset_inventory_path=inventory_path,
        code_commit="a" * 40,
        enforce_formal_collection=False,
    )
    assert first == second
    assert first["data_roles"]["guard"] == "absent"
    assert first["data_roles"]["validation"] == "absent"
    assert all(scene["counts"] == {"total": 9, "train": 7, "test": 2} for scene in first["scenes"])
    assert all(
        scene["stages"][1]["reads"] == ["train temperature only"] for scene in first["scenes"]
    )


def test_radiometry_plan_rejects_guard(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    path = split_root / "scenes" / "Building.split.json"
    split = json.loads(path.read_text(encoding="utf-8"))
    split["records"][1]["split"] = "guard"
    _write_json(path, split)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    for entry in collection["scenes"]:
        if entry["scene"] == "Building":
            entry["manifest_sha256"] = file_hash(path)
    _write_json(collection_path, collection)
    with pytest.raises(ValueError, match="only train/test"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
            enforce_formal_collection=False,
        )


def test_radiometry_plan_rejects_nonformal_or_unbound_split(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    collection["protocol_id"] = "wrong"
    _write_json(collection_path, collection)
    with pytest.raises(ValueError, match="not the formal"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
            enforce_formal_collection=False,
        )

    _, collection_path, split_root, inventory_path = _fixture(tmp_path / "second")
    split_path = split_root / "scenes" / "Building.split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["collection_hash"] = "f" * 64
    _write_json(split_path, split)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    entry = next(item for item in collection["scenes"] if item["scene"] == "Building")
    entry["manifest_sha256"] = file_hash(split_path)
    _write_json(collection_path, collection)
    with pytest.raises(ValueError, match="collection hash mismatch"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
            enforce_formal_collection=False,
        )


def test_radiometry_plan_records_decode_and_sfm_build_gaps(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    scene = "Building"
    raw_root = tmp_path / "assets" / scene / "raw"
    cfr_manifest = tmp_path / "assets" / scene / "cfr_manifest.json"
    _write_json(cfr_manifest, {"scene": scene, "count": 9})
    inventory["scenes"][scene] = {
        "radiometry_status": "decode_required",
        "camera_status": "build_required",
        "raw_thermal_root": {
            "path": str(raw_root),
            "file_count": 9,
            "tree_sha256": directory_tree_hash(raw_root),
        },
        "cfr_manifest": {
            "path": str(cfr_manifest),
            "sha256": file_hash(cfr_manifest),
        },
    }
    _write_json(inventory_path, inventory)

    plan = build_plan(
        collection_path=collection_path,
        split_root=split_root,
        asset_inventory_path=inventory_path,
        code_commit="a" * 40,
        enforce_formal_collection=False,
    )
    building = next(item for item in plan["scenes"] if item["scene"] == scene)
    stage_ids = [item["id"] for item in building["stages"]]
    assert stage_ids[0] == "decode_split_independent_full_frame_temperature"
    assert "build_shared_full_collection_sfm" in stage_ids
    assert "materialize_split_independent_valid_masks" in stage_ids
    assert scene in plan["availability"]["decoded_temperature_requires_decode"]
    assert scene in plan["availability"]["shared_sfm_requires_build"]
    assert scene in plan["availability"]["valid_mask_requires_build"]


def test_radiometry_plan_rejects_raw_count_mismatch(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    raw_root = tmp_path / "assets" / "Building" / "raw_mismatch"
    raw_root.mkdir()
    (raw_root / "only-one.JPG").write_bytes(b"raw")
    inventory["scenes"]["Building"]["raw_thermal_root"] = {
        "path": str(raw_root),
        "file_count": 1,
        "tree_sha256": directory_tree_hash(raw_root),
    }
    _write_json(inventory_path, inventory)
    with pytest.raises(ValueError, match="raw thermal count"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
            enforce_formal_collection=False,
        )


def test_radiometry_plan_formal_mode_locks_collection_commit_and_lut(tmp_path: Path) -> None:
    _, collection_path, split_root, inventory_path = _fixture(tmp_path)
    with pytest.raises(ValueError, match="eleven-scene"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
        )
    with pytest.raises(ValueError, match="40-hex"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="short",
            enforce_formal_collection=False,
        )

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory["fixed_hotiron_lut"]["lut_rgb_sha256"] = "f" * 64
    _write_json(inventory_path, inventory)
    with pytest.raises(ValueError, match="LUT RGB SHA"):
        build_plan(
            collection_path=collection_path,
            split_root=split_root,
            asset_inventory_path=inventory_path,
            code_commit="a" * 40,
            enforce_formal_collection=False,
        )


def test_phase1_queue_has_exact_six_scene_eighteen_rows(tmp_path: Path) -> None:
    collection = _formal_queue_collection()
    queue = build_queue(
        collection,
        code_commit="b" * 40,
        radiometry_plan_sha256="c" * 64,
        host_preflight_sha256={"900": "d" * 64, "901": "e" * 64},
    )
    assert len(queue["jobs"]) == 6
    assert len(queue["endpoint_jobs"]) == 18
    assert len({row["job_id"] for row in queue["endpoint_jobs"]}) == 18
    assert queue["formal_internal_endpoint_rows"] == 18
    assert queue["rgb_anchor_training_count"] == 6
    assert {job["scene"] for job in queue["jobs"]} == set(SCENES)
    assert sum(job["host"] == "900" for job in queue["jobs"]) == 3
    assert sum(job["host"] == "901" for job in queue["jobs"]) == 3
    assert queue["scsp_alias_policy"].startswith("recompute")
    assert sum(row["alias_allowed"] for row in queue["endpoint_jobs"]) == 6
    assert queue["protocol_id"] == PROTOCOL_ID
    assert queue["collection_hash"] == collection["collection_hash"]
    assert queue["collection_split_hash"] == collection["collection_split_hash"]
    assert queue["prerequisite_hashes"] == {
        "radiometry_plan_sha256": "c" * 64,
        "host_preflight_sha256": {"900": "d" * 64, "901": "e" * 64},
    }


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("protocol", "protocol_id"),
        ("validation", "validation flags"),
        ("collection_hash", "collection_hash"),
        ("collection_split_hash", "collection_split_hash"),
        ("representative_count", "counts mismatch for Building"),
    ],
)
def test_phase1_queue_rejects_nonformal_collection(tamper: str, match: str) -> None:
    collection = copy.deepcopy(_formal_queue_collection())
    if tamper == "protocol":
        collection["protocol_id"] = "wrong"
    elif tamper == "validation":
        collection["validation"]["scene_counts_exact"] = False
    elif tamper == "collection_hash":
        collection["collection_hash"] = "not-a-sha"
    elif tamper == "collection_split_hash":
        collection["collection_split_hash"] = "f" * 64
    else:
        next(row for row in collection["scenes"] if row["scene"] == "Building")[
            "counts"
        ]["train"] -= 1
    with pytest.raises(ValueError, match=match):
        build_queue(collection, code_commit="b" * 40)


def test_phase1_queue_rejects_partial_or_invalid_preflight_hashes() -> None:
    collection = _formal_queue_collection()
    with pytest.raises(ValueError, match="exactly hosts 900 and 901"):
        build_queue(
            collection,
            code_commit="b" * 40,
            host_preflight_sha256={"900": "d" * 64},
        )
    with pytest.raises(ValueError, match="host 901 preflight"):
        build_queue(
            collection,
            code_commit="b" * 40,
            host_preflight_sha256={"900": "d" * 64, "901": "bad"},
        )


def test_cleanup_classifier_is_narrow_and_protects_refined_mesh() -> None:
    assert classify("scene/reference_openmvs_v1/x/depth0001.dmap") == "openmvs_intermediate"
    assert classify("scene/reference_openmvs_v1/x/reference_openmvs_dense.ply") == "openmvs_intermediate"
    assert classify("scene/phase1_internal_v1/raw_f3/model/chkpnt60000.pth") == "optimizer_checkpoint"
    assert (
        classify("scene/phase1_internal_v1/guard_evidence/raw/geometry/guard_partition/views/1.npz")
        == "raw_evaluation_array"
    )
    assert classify("scene/reference_openmvs_v1/x/reference_openmvs_mesh_refined.ply") is None
    assert classify("scene/phase1_internal_v1/raw_f3/model/point_cloud.ply") is None
    assert (
        classify_preserved("scene/phase1_internal_v1/raw_f3/model/point_cloud.ply")
        == "model_recovery"
    )
    assert (
        classify_preserved("scene/reference_openmvs_v1/x/reference_openmvs_mesh_refined.ply")
        == "reference_recovery"
    )
    assert (
        classify_preserved("scene/phase1_internal_v1/raw_f3/metrics/metrics_summary.json")
        == "protocol_and_scalar_evidence"
    )
