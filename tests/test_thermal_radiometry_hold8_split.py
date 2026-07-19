from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.thermal_radiometry import build_hold8_split as hold8


CODE_COMMIT = "a" * 40
GENERATOR_SHA = "b" * 64
SOURCE_SHA = "c" * 64
COLLECTION_SHA = "d" * 64


def _record(scene: str, pair_id: str, marker: int = 0) -> dict[str, object]:
    return {
        "scene": scene,
        "pair_id": pair_id,
        "frame_id": pair_id,
        "rgb_path": f"rgb/{pair_id}.JPG",
        "source_path": f"thermal/{pair_id}.JPG",
        "audit_marker": marker,
        "metadata_sources": {"capture_time": "fixture"},
    }


def _scene_manifest(records: list[dict[str, object]], scene: str = "Fixture") -> dict:
    return hold8.build_scene_manifest(
        records,
        scene=scene,
        source_manifest="fixture.jsonl",
        source_manifest_sha256=SOURCE_SHA,
        collection_hash=COLLECTION_SHA,
        code_commit=CODE_COMMIT,
        generator_source_sha256=GENERATOR_SHA,
        expected_total=len(records),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_numeric_aware_pair_order_and_zero_based_hold8_assignment() -> None:
    pair_ids = ["frame10", "frame2", "frame1", "frame02", "frame9", "frame3", "frame8", "frame7", "frame6"]
    manifest = _scene_manifest([_record("Fixture", pair_id, i) for i, pair_id in enumerate(pair_ids)])

    ordered = [record["pair_id"] for record in manifest["records"]]
    assert ordered == [
        "frame1",
        "frame2",
        "frame02",
        "frame3",
        "frame6",
        "frame7",
        "frame8",
        "frame9",
        "frame10",
    ]
    assert [record["zero_based_sorted_index"] for record in manifest["records"]] == list(range(9))
    assert [record["pair_id"] for record in manifest["records"] if record["split"] == "test"] == ["frame1", "frame10"]
    assert manifest["counts"] == {"total": 9, "train": 7, "test": 2}
    assert set(manifest["counts"]) == {"total", "train", "test"}
    assert {record["split"] for record in manifest["records"]} == {"train", "test"}
    assert all(record["split"] != "guard" for record in manifest["records"])


def test_missing_numeric_ids_do_not_change_position_rule() -> None:
    records = [_record("Fixture", pair_id) for pair_id in ["17", "1", "100", "9", "2"]]
    manifest = _scene_manifest(records)
    assert [record["pair_id"] for record in manifest["records"]] == ["1", "2", "9", "17", "100"]
    assert manifest["records"][0]["split"] == "test"
    assert all(record["split"] == "train" for record in manifest["records"][1:])


def test_source_audit_fields_are_preserved_and_only_reserved_fields_added() -> None:
    source = _record("Fixture", "0001", marker=37)
    manifest = _scene_manifest([source])
    output = manifest["records"][0]
    for key, value in source.items():
        assert output[key] == value
    assert output["zero_based_sorted_index"] == 0
    assert output["split"] == "test"
    assert len(output["source_record_sha256"]) == 64
    assert manifest["validation"]["source_audit_fields_preserved"] is True


def test_order_and_list_hashes_bind_exact_lf_files_and_are_input_order_independent() -> None:
    records = [_record("Fixture", f"{index}") for index in range(1, 18)]
    first = _scene_manifest(records)
    second = _scene_manifest(list(reversed(records)))

    # Split/list identity is based on canonical pair ordering.  The source
    # collection-content hash still notices that its source file changed.
    for key in ("pair_ordering_sha256", "train_list_sha256", "test_list_sha256"):
        assert first["hashes"][key] == second["hashes"][key]
    assert first["hashes"]["input_records_sha256"] != second["hashes"]["input_records_sha256"]
    assert first["split_hash"] != second["split_hash"]

    ordered = [record["pair_id"] for record in first["records"]]
    train = [record["pair_id"] for record in first["records"] if record["split"] == "train"]
    test = [record["pair_id"] for record in first["records"] if record["split"] == "test"]
    assert first["hashes"]["pair_ordering_sha256"] == hashlib.sha256(("\n".join(ordered) + "\n").encode()).hexdigest()
    assert first["hashes"]["train_list_sha256"] == hashlib.sha256(("\n".join(train) + "\n").encode()).hexdigest()
    assert first["hashes"]["test_list_sha256"] == hashlib.sha256(("\n".join(test) + "\n").encode()).hexdigest()


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record("Fixture", "1"), _record("Fixture", "1")], "duplicate pair_id"),
        ([_record("Wrong", "1")], "scene mismatch"),
        ([{"scene": "Fixture", "pair_id": " "}], "non-empty string pair_id"),
        ([{"scene": "Fixture", "pair_id": "1", "split": "train"}], "reserved Hold-8 keys"),
    ],
)
def test_scene_validation_fails_closed(records: list[dict], message: str) -> None:
    with pytest.raises(hold8.Hold8ValidationError, match=message):
        _scene_manifest(records)


def test_expected_scene_count_fails_closed() -> None:
    with pytest.raises(hold8.Hold8ValidationError, match="scene count mismatch"):
        hold8.build_scene_manifest(
            [_record("Fixture", "1")],
            scene="Fixture",
            source_manifest="fixture.jsonl",
            source_manifest_sha256=SOURCE_SHA,
            collection_hash=COLLECTION_SHA,
            code_commit=CODE_COMMIT,
            generator_source_sha256=GENERATOR_SHA,
            expected_total=2,
        )


def test_collection_counts_and_scene_set_fail_closed() -> None:
    expected = {"A": 9, "B": 8}
    records = [*[_record("A", str(i)) for i in range(9)], *[_record("B", str(i)) for i in range(8)]]
    collection, scenes = hold8.build_collection_manifest(
        records,
        source_manifest="fixture.jsonl",
        source_manifest_sha256=SOURCE_SHA,
        code_commit=CODE_COMMIT,
        generator_source_sha256=GENERATOR_SHA,
        expected_scene_counts=expected,
    )
    assert collection["counts"] == {"total": 17, "train": 14, "test": 3}
    assert scenes["A"]["counts"] == {"total": 9, "train": 7, "test": 2}
    assert scenes["B"]["counts"] == {"total": 8, "train": 7, "test": 1}
    assert collection["validation"]["labels_exactly_train_test"] is True

    with pytest.raises(hold8.Hold8ValidationError, match="scene set mismatch"):
        hold8.build_collection_manifest(
            records,
            source_manifest="fixture.jsonl",
            source_manifest_sha256=SOURCE_SHA,
            code_commit=CODE_COMMIT,
            generator_source_sha256=GENERATOR_SHA,
            expected_scene_counts={"A": 9},
        )
    with pytest.raises(hold8.Hold8ValidationError, match="scene count mismatch"):
        hold8.build_collection_manifest(
            records,
            source_manifest="fixture.jsonl",
            source_manifest_sha256=SOURCE_SHA,
            code_commit=CODE_COMMIT,
            generator_source_sha256=GENERATOR_SHA,
            expected_scene_counts={"A": 8, "B": 8},
        )


def test_locked_all11_count_formula() -> None:
    expected_tests = {
        scene: (count + hold8.HOLDOUT_PERIOD - 1) // hold8.HOLDOUT_PERIOD
        for scene, count in hold8.EXPECTED_SCENE_COUNTS.items()
    }
    assert sum(hold8.EXPECTED_SCENE_COUNTS.values()) == 8232
    assert sum(expected_tests.values()) == 1034
    assert sum(hold8.EXPECTED_SCENE_COUNTS.values()) - sum(expected_tests.values()) == 7198
    assert expected_tests["Building"] == 77
    assert expected_tests["InternalRoad"] == 70
    assert expected_tests["Urban20K"] == 94


def test_materialization_writes_verified_manifests_and_explicit_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    records = [_record("A", str(i)) for i in range(1, 10)]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8", newline="\n")
    monkeypatch.setattr(
        hold8,
        "resolve_generator_identity",
        lambda: {"code_commit": CODE_COMMIT, "generator_source_sha256": GENERATOR_SHA},
    )
    output = tmp_path / "out"
    result = hold8.materialize_collection(
        source, output, expected_scene_counts={"A": 9}
    )

    scene_entry = result["scenes"][0]
    scene_manifest = output / scene_entry["manifest"]
    assert _sha256(scene_manifest) == scene_entry["manifest_sha256"]
    for label, relative_path in scene_entry["list_files"].items():
        list_path = output / relative_path
        expected_hash = next(
            item for item in result["scenes"] if item["scene"] == "A"
        )[f"{'pair_ordering' if label == 'ordering' else label + '_list'}_sha256"]
        assert _sha256(list_path) == expected_hash
    assert (output / "lists" / "A.test.txt").read_text().splitlines() == ["1", "9"]
    assert set(result["counts"]) == {"total", "train", "test"}

    with pytest.raises(FileExistsError, match="pass --overwrite"):
        hold8.materialize_collection(source, output, expected_scene_counts={"A": 9})


def test_materialization_is_byte_identical_across_output_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    records = [_record("A", str(i)) for i in range(1, 10)]
    source.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setattr(
        hold8,
        "resolve_generator_identity",
        lambda: {"code_commit": CODE_COMMIT, "generator_source_sha256": GENERATOR_SHA},
    )
    first = tmp_path / "host-a" / "split"
    second = tmp_path / "host-b" / "split"
    hold8.materialize_collection(source, first, expected_scene_counts={"A": 9})
    hold8.materialize_collection(source, second, expected_scene_counts={"A": 9})
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_formal_cli_does_not_allow_expected_count_override() -> None:
    parser = hold8._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--manifest",
                "audit.jsonl",
                "--output-root",
                "hold8",
                "--expected-counts-json",
                "synthetic.json",
            ]
        )


def test_generator_and_collection_hashes_are_bound_into_split_hash() -> None:
    records = [_record("Fixture", str(i)) for i in range(9)]
    base = _scene_manifest(records)
    changed_code = hold8.build_scene_manifest(
        records,
        scene="Fixture",
        source_manifest="fixture.jsonl",
        source_manifest_sha256=SOURCE_SHA,
        collection_hash=COLLECTION_SHA,
        code_commit="e" * 40,
        generator_source_sha256=GENERATOR_SHA,
        expected_total=9,
    )
    changed_collection = hold8.build_scene_manifest(
        records,
        scene="Fixture",
        source_manifest="fixture.jsonl",
        source_manifest_sha256=SOURCE_SHA,
        collection_hash="f" * 64,
        code_commit=CODE_COMMIT,
        generator_source_sha256=GENERATOR_SHA,
        expected_total=9,
    )
    assert base["split_hash"] != changed_code["split_hash"]
    assert base["split_hash"] != changed_collection["split_hash"]
