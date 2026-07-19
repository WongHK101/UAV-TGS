import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from tools.evaluate_formal_baseline_hotspots import (
    _hotspot_source_receipt as existing_hotspot_source_receipt,
    _support_index as existing_support_index,
    _validate_threshold_manifest as existing_validate_threshold_manifest,
)
from tools.thermal_radiometry.freeze_train_hotspot_threshold import (
    FORMAL_HISTOGRAM_BINS,
    FORMAL_QUANTILE,
    freeze_train_hotspot_threshold,
    sha256_file,
    sha256_json,
    write_atomic,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.scene = "PVpanel"  # Deliberately outside the OCT two-scene scope.
        self.temperature_root = root / "undistorted" / "temperature_c"
        self.support_root = root / "undistorted"
        (self.support_root / "valid_support").mkdir(parents=True)
        self.temperature_root.mkdir(parents=True)

        self.records = [
            ("guard0", "guard"),
            ("train0", "train"),
            ("test0", "test"),
            ("train1", "train"),
        ]
        train_values = {
            "train0": np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32),
            "train1": np.asarray([[40.0, 50.0], [60.0, 70.0]], dtype=np.float32),
        }
        train_support = {
            "train0": np.ones((2, 2), dtype=np.bool_),
            "train1": np.asarray([[True, False], [True, True]], dtype=np.bool_),
        }
        for pair_id in train_values:
            np.save(self.temperature_root / f"{pair_id}.npy", train_values[pair_id])
            np.save(
                self.support_root / "valid_support" / f"{pair_id}.npy",
                train_support[pair_id],
            )

        protocol_hash = "a" * 64
        parameters = {
            "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
            "distance_m": {"value": 50.0, "source": "strip_lrf"},
            "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
            "humidity_percent": {"value": 70.0, "source": "benchmark_assumption"},
            "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
        }
        decode_rows: list[dict] = []
        protocol_rows: list[dict] = []
        self.raw_sha: dict[str, str] = {}
        for index, (pair_id, _) in enumerate(self.records):
            target = self.temperature_root / f"{pair_id}.npy"
            digest = sha256_file(target) if target.is_file() else f"{index + 1:x}" * 64
            self.raw_sha[pair_id] = digest
            decode_rows.append(
                {
                    "pair_id": pair_id,
                    "scene": self.scene,
                    "success": True,
                    "dtype": "float32",
                    "output_sha256": digest,
                }
            )
            protocol_rows.append(
                {
                    "pair_id": pair_id,
                    "scene": self.scene,
                    "schema_version": "uav-tgs.radiometry-protocol.v1",
                    "protocol_hash": protocol_hash,
                    "decode_parameters": parameters,
                }
            )
        self.decode_manifest = root / "decode.jsonl"
        self.decode_protocol = root / "protocol.jsonl"
        _write_jsonl(self.decode_manifest, decode_rows)
        _write_jsonl(self.decode_protocol, protocol_rows)

        self.bound_split = root / "bound_split.json"
        split_hash = "b" * 64
        _write_json(
            self.bound_split,
            {
                "scene": self.scene,
                "split_hash": split_hash,
                "counts": {"total": 4, "train": 2, "guard": 1, "test": 1},
                "decode_binding": {
                    "adapter_backend": "official-dji-irp",
                    "decode_manifest_sha256": sha256_file(self.decode_manifest),
                    "decode_protocol_sha256": sha256_file(self.decode_protocol),
                    "protocol_hash": protocol_hash,
                    "verified_decode_requests": True,
                    "verified_raw_rjpeg_hashes": True,
                    "verified_temperature_file_hashes": True,
                },
                "records": [
                    {
                        "pair_id": pair_id,
                        "split": split,
                        "thermal_camera_name": f"{pair_id}.png",
                    }
                    for pair_id, split in self.records
                ],
            },
        )

        self.range_manifest = root / "range.json"
        configuration = {
            "guard_role": "not_read",
            "test_role": "qa_only_not_used_for_estimation",
        }
        range_basis = {
            "scene": self.scene,
            "split_hash": split_hash,
            "configuration": configuration,
            "Tmin": 0.0,
            "Tmax": 100.0,
        }
        _write_json(
            self.range_manifest,
            {
                "schema_name": "uav_tgs_train_only_scene_temperature_range",
                "schema_version": 1,
                **range_basis,
                "range_hash": sha256_json(range_basis),
                "source_split_manifest_sha256": sha256_file(self.bound_split),
                "train_estimation": {"frame_count": 2},
                "clipping_stats": {
                    "train": {"frame_count": 2},
                    "test": {"frame_count": 1},
                },
                "per_frame_quantiles": [
                    {"pair_id": "train0", "split": "train"},
                    {"pair_id": "train1", "split": "train"},
                    {"pair_id": "test0", "split": "test"},
                ],
            },
        )

        support_rows: list[dict] = []
        for index, (pair_id, _) in enumerate(self.records):
            target = self.temperature_root / f"{pair_id}.npy"
            support = self.support_root / "valid_support" / f"{pair_id}.npy"
            output_sha = sha256_file(target) if target.is_file() else f"{index + 5:x}" * 64
            support_sha = sha256_file(support) if support.is_file() else f"{index + 9:x}" * 64
            support_rows.append(
                {
                    "pair_id": pair_id,
                    "input_temperature": {
                        "dtype": "float32",
                        "sha256": self.raw_sha[pair_id],
                    },
                    "output_temperature": {
                        "dtype": "float32",
                        "relative_path": f"temperature_c/{pair_id}.npy",
                        "sha256": output_sha,
                    },
                    "valid_support": {
                        "dtype": "bool",
                        "relative_path": f"valid_support/{pair_id}.npy",
                        "sha256": support_sha,
                    },
                }
            )
        self.support_rows = support_rows
        self.support_manifest = root / "optimization_support.json"
        _write_json(
            self.support_manifest,
            {
                "schema": "uav-tgs-undistorted-temperature-v1",
                "status": "complete",
                "files": support_rows,
            },
        )

        # These paths intentionally do not exist.  A successful freeze proves
        # that guard/test payloads were neither resolved nor opened.
        assert not (self.temperature_root / "guard0.npy").exists()
        assert not (self.temperature_root / "test0.npy").exists()
        assert not (self.support_root / "valid_support" / "guard0.npy").exists()
        assert not (self.support_root / "valid_support" / "test0.npy").exists()

    def kwargs(self) -> dict:
        return {
            "scene_name": self.scene,
            "bound_split_path": self.bound_split,
            "decode_manifest_path": self.decode_manifest,
            "decode_protocol_path": self.decode_protocol,
            "range_manifest_path": self.range_manifest,
            "temperature_root": self.temperature_root,
            "support_manifest_path": self.support_manifest,
            "support_root": self.support_root,
            "chunk_pixels": 3,
        }


def test_third_scene_is_train_only_deterministic_and_existing_receipt_compatible() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _Fixture(Path(temporary))
        first = freeze_train_hotspot_threshold(**fixture.kwargs())
        second = freeze_train_hotspot_threshold(**fixture.kwargs())

        assert first == second
        assert first["scene_name"] == "PVpanel"
        assert first["quantile"] == FORMAL_QUANTILE
        assert first["histogram_bins"] == FORMAL_HISTOGRAM_BINS
        assert first["valid_train_pixels"] == 7
        width = 100.0 / FORMAL_HISTOGRAM_BINS
        expected_threshold = (math.floor(70.0 / width) + 1) * width
        assert first["threshold_c"] == pytest.approx(expected_threshold, abs=1e-12)

        pair_ids = [pair_id for pair_id, _ in fixture.records]
        support_by_pair = {row["pair_id"]: row for row in fixture.support_rows}
        expected_receipt = existing_hotspot_source_receipt(
            scene_name=fixture.scene,
            split_sha256=sha256_file(fixture.bound_split),
            split_hash="b" * 64,
            decode_manifest_sha256=sha256_file(fixture.decode_manifest),
            decode_protocol_sha256=sha256_file(fixture.decode_protocol),
            range_sha256=sha256_file(fixture.range_manifest),
            range_hash=json.loads(fixture.range_manifest.read_text())["range_hash"],
            support_sha256=sha256_file(fixture.support_manifest),
            support_index_sha256=sha256_json(
                existing_support_index(pair_ids, support_by_pair)
            ),
            train_camera_names=["train0.png", "train1.png"],
        )
        assert first["source_receipt"] == expected_receipt

        output_a = write_atomic(fixture.root / "threshold_a.json", first)
        output_b = write_atomic(fixture.root / "threshold_b.json", second)
        assert output_a.read_bytes() == output_b.read_bytes()
        loaded, supplied_hash, threshold = existing_validate_threshold_manifest(
            output_a,
            scene_name=fixture.scene,
            source_receipt=expected_receipt,
            train_camera_names=["train0.png", "train1.png"],
            tmin_c=0.0,
            tmax_c=100.0,
        )
        assert loaded == first
        assert supplied_hash == first["threshold_sha256"]
        assert threshold == first["threshold_c"]


def test_hold8_train_test_only_binding_is_accepted_without_nontrain_payload_reads() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _Fixture(Path(temporary))

        # Convert the legacy fixture to the Hold-8 two-partition contract.  The
        # test payload still does not exist, proving that only train arrays are
        # resolved/opened.
        decode_rows = [
            row
            for row in (
                json.loads(line) for line in fixture.decode_manifest.read_text().splitlines()
            )
            if row["pair_id"] != "guard0"
        ]
        protocol_rows = [
            row
            for row in (
                json.loads(line) for line in fixture.decode_protocol.read_text().splitlines()
            )
            if row["pair_id"] != "guard0"
        ]
        _write_jsonl(fixture.decode_manifest, decode_rows)
        _write_jsonl(fixture.decode_protocol, protocol_rows)

        bound = json.loads(fixture.bound_split.read_text())
        bound["records"] = [row for row in bound["records"] if row["split"] != "guard"]
        bound["counts"] = {"total": 3, "train": 2, "test": 1}
        bound["decode_binding"]["decode_manifest_sha256"] = sha256_file(
            fixture.decode_manifest
        )
        bound["decode_binding"]["decode_protocol_sha256"] = sha256_file(
            fixture.decode_protocol
        )
        _write_json(fixture.bound_split, bound)

        support = json.loads(fixture.support_manifest.read_text())
        support["files"] = [
            row for row in support["files"] if row["pair_id"] != "guard0"
        ]
        _write_json(fixture.support_manifest, support)

        range_payload = json.loads(fixture.range_manifest.read_text())
        range_payload["source_split_manifest_sha256"] = sha256_file(fixture.bound_split)
        # A Hold-8 range may omit the now-inapplicable legacy guard-role field.
        configuration = dict(range_payload["configuration"])
        configuration.pop("guard_role")
        range_payload["configuration"] = configuration
        range_basis = {
            "scene": range_payload["scene"],
            "split_hash": range_payload["split_hash"],
            "configuration": configuration,
            "Tmin": range_payload["Tmin"],
            "Tmax": range_payload["Tmax"],
        }
        range_payload["range_hash"] = sha256_json(range_basis)
        _write_json(fixture.range_manifest, range_payload)

        result = freeze_train_hotspot_threshold(**fixture.kwargs())
        assert result["source_split"] == "train"
        assert result["test_statistics_used"] is False
        assert result["valid_train_pixels"] == 7
        assert not (fixture.temperature_root / "test0.npy").exists()
        assert not (fixture.support_root / "valid_support" / "test0.npy").exists()


@pytest.mark.parametrize("tamper", ["decode_hash", "support_membership", "range_hash", "train_membership"])
def test_hash_and_membership_tampering_fails_closed(tamper: str) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        fixture = _Fixture(Path(temporary))
        if tamper == "decode_hash":
            rows = [json.loads(line) for line in fixture.decode_manifest.read_text().splitlines()]
            rows[0]["output_sha256"] = "f" * 64
            _write_jsonl(fixture.decode_manifest, rows)
            match = "decode manifest differs from bound split"
        elif tamper == "support_membership":
            payload = json.loads(fixture.support_manifest.read_text())
            payload["files"].pop()
            _write_json(fixture.support_manifest, payload)
            match = "optimization support identities differ"
        elif tamper == "range_hash":
            payload = json.loads(fixture.range_manifest.read_text())
            payload["Tmax"] = 101.0
            _write_json(fixture.range_manifest, payload)
            match = "range logical hash mismatch"
        else:
            payload = json.loads(fixture.bound_split.read_text())
            payload["records"][1]["thermal_camera_name"] = "train1.png"
            _write_json(fixture.bound_split, payload)
            match = "thermal camera names are not unique"

        with pytest.raises(ValueError, match=match):
            freeze_train_hotspot_threshold(**fixture.kwargs())
