from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.thermal_radiometry import split_qa


class SplitQaTests(unittest.TestCase):
    @staticmethod
    def _records(scene: str, count: int, *, missing_metadata: bool = False) -> list[dict[str, object]]:
        start = datetime(2026, 5, 25, tzinfo=timezone.utc)
        records: list[dict[str, object]] = []
        for index in range(count):
            record: dict[str, object] = {
                "scene": scene,
                "pair_id": f"{index + 1:04d}",
                "filename": f"frame_{index + 1:04d}_T.JPG",
                "capture_time": (start + timedelta(seconds=index)).isoformat(),
                "gps_latitude": 30.0 + index * 1e-5,
                "gps_longitude": 120.0,
                "gimbal_pitch_deg": -45.0 + index * 0.01,
                "gimbal_yaw_deg": 90.0 + index * 0.02,
                "source_path": f"thermal/{index + 1:04d}.JPG",
            }
            records.append(record)
        if missing_metadata:
            records[3].pop("capture_time")
        return records

    @staticmethod
    def _write_manifest(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_all_guard_candidates_report_required_counts_and_nearest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reliable = root / "reliable.jsonl"
            fallback = root / "fallback.jsonl"
            self._write_manifest(reliable, self._records("Reliable", 160))
            self._write_manifest(fallback, self._records("Fallback", 40, missing_metadata=True))

            report = split_qa.build_split_qa_report(
                [fallback, reliable], seed="fixed-seed"
            )

            self.assertEqual(report["decision_status"], "comparison_only_no_guard_selected")
            self.assertIn("independent minima", report["nearest_neighbor_semantics"]["per_metric_minima"])
            self.assertEqual(report["fixed_rule"]["block_size_frames"], 16)
            self.assertEqual(
                [candidate["guard_frames_each_side"] for candidate in report["candidates"]],
                [2, 4],
            )
            for candidate in report["candidates"]:
                self.assertEqual(candidate["counts"]["total"], 200)
                self.assertEqual(
                    candidate["counts"]["total"],
                    candidate["counts"]["train"]
                    + candidate["counts"]["test"]
                    + candidate["counts"]["guard"],
                )
                self.assertEqual(candidate["metadata_fallback"]["scene_count"], 1)
                self.assertEqual(candidate["metadata_fallback"]["frame_count"], 40)
                self.assertAlmostEqual(candidate["metadata_fallback"]["frame_rate"], 0.2)
                self.assertIn("retained_ratio", candidate["retained_train_test"])
                self.assertIn("train_to_test_ratio", candidate["retained_train_test"])
                self.assertEqual(candidate["invalid_scene_count"], 0)

                scenes = {scene["scene"]: scene for scene in candidate["scenes"]}
                reliable_scene = scenes["Reliable"]
                self.assertEqual(reliable_scene["ordering_mode"], "timestamp_gimbal")
                self.assertTrue(reliable_scene["stratum_counts"])
                self.assertTrue(reliable_scene["strip_counts"])
                self.assertTrue(reliable_scene["test_frame_nearest_train"])
                self.assertEqual(reliable_scene["validation"]["status"], "passed")
                self.assertGreaterEqual(reliable_scene["test_fraction_of_scene"], 0.08)
                self.assertLessEqual(reliable_scene["test_fraction_of_scene"], 0.16)
                self.assertIn("strata_without_test", reliable_scene)
                nearest = reliable_scene["test_frame_nearest_train"][0]
                self.assertIsNotNone(nearest["nearest_train_temporal_gap_s"])
                self.assertIsNotNone(nearest["nearest_train_gps_distance_m"])
                self.assertIsNotNone(nearest["nearest_train_gimbal_pitch_difference_deg"])
                self.assertIsNotNone(nearest["nearest_train_gimbal_yaw_difference_deg"])
                self.assertIn("pair_id", nearest["nearest_train_by_time"])
                self.assertIn("pair_id", nearest["nearest_train_by_gps"])
                self.assertIn("pair_id", nearest["nearest_train_by_gimbal"])
                summary = reliable_scene["nearest_train_summary"]
                self.assertIn("per_metric_independent_minima", summary)
                self.assertIn("nearest_by_time_observation", summary)
                self.assertGreater(
                    summary["nearest_by_time_observation"]["gps_distance_m"]["supported_count"],
                    0,
                )

                fallback_scene = scenes["Fallback"]
                self.assertEqual(fallback_scene["ordering_mode"], "filename_order_fallback")
                self.assertEqual(fallback_scene["metadata_fallback_rate"], 1.0)

    def test_each_guard_candidate_preserves_budget_and_count_conservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "scene.jsonl"
            self._write_manifest(manifest, self._records("Scene", 160))
            report = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            for candidate in report["candidates"]:
                counts = candidate["counts"]
                self.assertEqual(counts["total"], counts["train"] + counts["test"] + counts["guard"])
                scene = candidate["scenes"][0]
                self.assertEqual(scene["test_block_budget"]["target"], 1)
                self.assertEqual(scene["test_block_budget"]["selected"], 1)
                self.assertEqual(counts["test"], 16)
                self.assertEqual(scene["validation"]["status"], "passed")

    def test_short_strip_is_explicitly_reported_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "short.jsonl"
            self._write_manifest(manifest, self._records("Short", 15))
            report = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            for candidate in report["candidates"]:
                scene = candidate["scenes"][0]
                self.assertEqual(scene["counts"]["test"], 0)
                self.assertEqual(len(scene["strips_without_test"]), 1)
                self.assertEqual(candidate["strips_without_test_count"], 1)
                self.assertIsNone(candidate["retained_train_test"]["train_to_test_ratio"])

    def test_single_complete_block_is_kept_entirely_for_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "no_train.jsonl"
            self._write_manifest(manifest, self._records("NoTrain", 16))
            report = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            for candidate in report["candidates"]:
                scene = candidate["scenes"][0]
                self.assertEqual(scene["counts"], {"total": 16, "train": 16, "test": 0, "guard": 0})
                self.assertEqual(scene["strips_without_train"], [])
                self.assertEqual(scene["strata_without_train"], [])
                self.assertEqual(candidate["strips_without_train_count"], 0)
                self.assertEqual(candidate["strata_without_train_count"], 0)
                self.assertEqual(candidate["warning_count"], 0)
                self.assertEqual(scene["validation"]["status"], "passed")
                nearest_summary = candidate["nearest_train_summary"]
                self.assertEqual(
                    nearest_summary["per_metric_independent_minima"]["temporal_gap_s"]["supported_count"],
                    0,
                )
                self.assertEqual(
                    nearest_summary["nearest_by_time_observation"]["gps_distance_m"]["supported_count"],
                    0,
                )

    def test_formal_split_failure_is_reported_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "invalid.jsonl"
            records = self._records("Invalid", 160)
            for index, record in enumerate(records):
                group = index // 16
                record["stratum"] = f"short-{group:02d}"
                record["strip_id"] = f"short-{group:02d}"
            self._write_manifest(manifest, records)

            report = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            for candidate in report["candidates"]:
                scene = candidate["scenes"][0]
                self.assertEqual(scene["validation"]["status"], "failed")
                self.assertEqual(candidate["invalid_scene_count"], 1)
                budget_error = next(
                    warning
                    for warning in scene["warnings"]
                    if warning["code"] == "test_block_budget_shortfall"
                )
                self.assertEqual(budget_error["severity"], "error")

    def test_report_is_deterministic_and_cli_writes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "scene.jsonl"
            output = root / "qa.json"
            self._write_manifest(manifest, self._records("Scene", 128))
            first = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            second = split_qa.build_split_qa_report([manifest], seed="fixed-seed")
            self.assertEqual(first, second)
            self.assertEqual(len(first["qa_hash"]), 64)
            self.assertTrue(first["cross_guard_test_set_comparison"])

            exit_code = split_qa.main(
                [
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--seed",
                    "fixed-seed",
                ]
            )
            self.assertEqual(exit_code, 0)
            persisted = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(persisted["qa_hash"], first["qa_hash"])


if __name__ == "__main__":
    unittest.main()
