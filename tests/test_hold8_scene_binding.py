from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools.thermal_radiometry import bind_formal_scene as legacy_binding
from tools.thermal_radiometry.bind_hold8_scene import (
    bind_hold8_scene,
    materialize_hold8_binding,
)
from tools.thermal_radiometry.build_hold8_split import (
    PROTOCOL_ID,
    SCHEMA_NAME as SPLIT_SCHEMA,
    build_scene_manifest,
)
from tools.thermal_radiometry.estimate_scene_range import estimate_scene_range


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path) -> dict[str, Any]:
    scene = "Fixture"
    collection_hash = _token("hold8-collection")
    collection_split_hash = _token("hold8-collection-split")
    adapter_sha = _token("dji-irp-executable")
    source_manifest_sha = _token("audit-source")
    temperature_root = root / "temperature_c"
    raw_root = root / "raw_thermal"
    request_root = root / "manifests" / "decode_requests"
    (temperature_root / scene).mkdir(parents=True)
    raw_root.mkdir(parents=True)

    source_records: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    decode_rows: list[dict[str, Any]] = []
    for index, pair_id in enumerate(("0001", "0002"), 1):
        source_record_hash = _token(f"source-{pair_id}")
        raw_path = raw_root / f"{pair_id}.JPG"
        raw_path.write_bytes((f"rjpeg-{pair_id}" * 19).encode("ascii"))
        npy_path = temperature_root / scene / f"{pair_id}.npy"
        np.save(
            npy_path,
            np.linspace(
                float(index * 10),
                float(index * 10 + 1),
                num=1024 * 1280,
                dtype=np.float32,
            ).reshape(1024, 1280),
        )
        source_records.append(
            {
                "scene": scene,
                "pair_id": pair_id,
                "strip_id": "strip-0",
                "source_record_hash": source_record_hash,
            }
        )
        parameters = {
            "distance_m": {"value": 20.0, "source": "strip_valid_lrf_robust_median"},
            "humidity_percent": {"value": 70.0, "source": "benchmark_assumption"},
            "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
            "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
            "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
        }
        metadata = {
            "radiometry_protocol": {
                "strip_id": "strip-0",
                "used_distance_m": 20.0,
                "used_distance_source": "strip_valid_lrf_robust_median",
            }
        }
        protocol = {
            "schema_version": legacy_binding.FORMAL_PROTOCOL_SCHEMA,
            "scene": scene,
            "frame_id": pair_id,
            "pair_id": pair_id,
            "source_path": str(raw_path.resolve()),
            "strip_id": "strip-0",
            "decode_parameters": parameters,
            "raw_lrf_distance_m": 20.0,
            "raw_lrf_status": "Normal",
            "raw_lrf_valid": True,
            "used_distance_m": 20.0,
            "used_distance_source": "strip_valid_lrf_robust_median",
            "distance_fallback_reason": "none",
            "source_audit_record_hash": source_record_hash,
            "metadata": metadata,
        }
        protocol["protocol_record_hash"] = _json_hash(protocol)
        protocol_rows.append(protocol)

        request = {
            "schema_version": legacy_binding.FORMAL_DECODE_SCHEMA,
            "scene": scene,
            "frame_id": pair_id,
            "pair_id": pair_id,
            "source_path": str(raw_path.resolve()),
            "output_path": str(npy_path.resolve()),
            "temperature_npy": str(npy_path.resolve()),
            "tsdk_root": str(root / "tsdk"),
            "adapter": legacy_binding.FORMAL_ADAPTER,
            "parameters": parameters,
            "strip_id": "strip-0",
            "metadata": metadata,
        }
        request_path = request_root / f"{scene}--{pair_id}.json"
        _write_json(request_path, request)
        decode = dict(request)
        decode.update(
            {
                "request_path": str(request_path.resolve()),
                "success": True,
                "dtype": "float32",
                "shape_hw": [1024, 1280],
                "source_size_bytes": raw_path.stat().st_size,
                "source_sha256": _sha(raw_path),
                "output_sha256": _sha(npy_path),
                "adapter_diagnostics": {
                    "backend": legacy_binding.FORMAL_ADAPTER_BACKEND,
                    "executable_sha256": adapter_sha,
                    "parameters_applied": {
                        name: float(entry["value"]) for name, entry in parameters.items()
                    },
                    "resolution": {"width": 1280, "height": 1024},
                    "dirp_api_version": "0x14",
                    "rjpeg_version": "0x300",
                },
            }
        )
        decode_rows.append(decode)

    basis = [
        {field: row[field] for field in legacy_binding.FORMAL_PROTOCOL_BASIS_FIELDS}
        for row in protocol_rows
    ]
    protocol_hash = _json_hash(basis)
    for row in protocol_rows:
        row["protocol_hash"] = protocol_hash

    scene_manifest = build_scene_manifest(
        source_records,
        scene=scene,
        source_manifest=str(root / "audit.jsonl"),
        source_manifest_sha256=source_manifest_sha,
        collection_hash=collection_hash,
        code_commit="1" * 40,
        generator_source_sha256=_token("generator"),
        expected_total=2,
    )
    scene_path = root / "hold8" / "scenes" / f"{scene}.split.json"
    _write_json(scene_path, scene_manifest)
    collection = {
        "schema_name": f"{SPLIT_SCHEMA}_collection",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "collection_hash": collection_hash,
        "collection_split_hash": collection_split_hash,
        "counts": {"total": 2, "train": 1, "test": 1},
        "validation": {"status": "passed"},
        "scenes": [
            {
                "scene": scene,
                "counts": {"total": 2, "train": 1, "test": 1},
                "split_hash": scene_manifest["split_hash"],
                "manifest_sha256": _sha(scene_path),
            }
        ],
    }
    collection_path = root / "hold8" / "collection_manifest.json"
    _write_json(collection_path, collection)
    decode_path = root / "manifests" / "decode_manifest.jsonl"
    protocol_path = root / "manifests" / "decode_protocol.jsonl"
    _write_jsonl(decode_path, decode_rows)
    _write_jsonl(protocol_path, protocol_rows)
    kwargs = {
        "scene_manifest_path": scene_path,
        "collection_manifest_path": collection_path,
        "decode_manifest_path": decode_path,
        "decode_protocol_path": protocol_path,
        "temperature_root": temperature_root,
        "raw_thermal_root": raw_root,
        "scene": scene,
        "sfm_image_scope": "shared_sfm_all_images",
        "expected_collection_manifest_sha256": _sha(collection_path),
        "expected_collection_hash": collection_hash,
        "expected_collection_split_hash": collection_split_hash,
        "expected_scene_manifest_sha256": _sha(scene_path),
        "expected_scene_split_hash": scene_manifest["split_hash"],
        "expected_decode_manifest_sha256": _sha(decode_path),
        "expected_decode_protocol_sha256": _sha(protocol_path),
        "expected_decode_protocol_hash": protocol_hash,
        "expected_adapter_executable_sha256": adapter_sha,
    }
    return {"kwargs": kwargs, "temperature_root": temperature_root}


def test_binding_has_only_train_test_outputs_and_feeds_range(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    bound, lists, manifest = bind_hold8_scene(**fixture["kwargs"])

    assert lists == {
        "train": ["0002.JPG"],
        "test": ["0001.JPG"],
        "thermal_train": ["0002.png"],
        "thermal_test": ["0001.png"],
    }
    assert bound["counts"] == {"total": 2, "train": 1, "test": 1}
    assert bound["records"][0]["temperature_npy"] == "Fixture/0001.npy"
    assert manifest["counts"] == {"total": 2, "train": 1, "test": 1}
    assert manifest["status"] == "passed"

    range_payload = estimate_scene_range(
        bound,
        manifest_dir=tmp_path,
        npy_root=fixture["temperature_root"],
        histogram_bins=64,
        chunk_pixels=64 * 1024,
    )
    assert range_payload["train_estimation"]["frame_count"] == 1
    assert range_payload["clipping_stats"]["test"]["frame_count"] == 1

    outputs = materialize_hold8_binding(tmp_path / "binding", **fixture["kwargs"])
    assert set(outputs) == {
        "bound_split",
        "train",
        "test",
        "thermal_train",
        "thermal_test",
        "binding_manifest",
    }
    assert not list((tmp_path / "binding").glob("*guard*"))


def test_binding_rejects_train_only_sfm_scope(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["kwargs"]["sfm_image_scope"] = "train_only"
    with pytest.raises(ValueError, match="requires sfm_image_scope"):
        bind_hold8_scene(**fixture["kwargs"])


@pytest.mark.parametrize("tamper", ["count", "pair", "temperature_sha", "decode_manifest_pin"])
def test_binding_tampering_fails_closed(tmp_path: Path, tamper: str) -> None:
    fixture = _fixture(tmp_path)
    kwargs = fixture["kwargs"]
    if tamper == "count":
        payload = json.loads(Path(kwargs["scene_manifest_path"]).read_text())
        payload["counts"]["train"] = 2
        _write_json(Path(kwargs["scene_manifest_path"]), payload)
        kwargs["expected_scene_manifest_sha256"] = _sha(Path(kwargs["scene_manifest_path"]))
        collection = json.loads(Path(kwargs["collection_manifest_path"]).read_text())
        collection["scenes"][0]["manifest_sha256"] = kwargs["expected_scene_manifest_sha256"]
        _write_json(Path(kwargs["collection_manifest_path"]), collection)
        kwargs["expected_collection_manifest_sha256"] = _sha(Path(kwargs["collection_manifest_path"]))
        match = "counts disagree"
    elif tamper == "pair":
        rows = Path(kwargs["decode_manifest_path"]).read_text().splitlines()
        payload = json.loads(rows[0])
        payload["pair_id"] = "9999"
        rows[0] = json.dumps(payload, sort_keys=True)
        Path(kwargs["decode_manifest_path"]).write_text("\n".join(rows) + "\n")
        kwargs["expected_decode_manifest_sha256"] = _sha(Path(kwargs["decode_manifest_path"]))
        match = "pair mismatch"
    elif tamper == "temperature_sha":
        npy_path = fixture["temperature_root"] / "Fixture" / "0002.npy"
        with npy_path.open("ab") as stream:
            stream.write(b"tamper")
        match = "temperature file SHA-256"
    else:
        kwargs["expected_decode_manifest_sha256"] = "f" * 64
        match = "decode manifest SHA-256 mismatch"

    with pytest.raises(ValueError, match=match):
        bind_hold8_scene(**kwargs)
