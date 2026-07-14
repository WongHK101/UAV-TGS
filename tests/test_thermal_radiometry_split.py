from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from tools.thermal_radiometry import build_split


class BuildSplitTests(unittest.TestCase):
    @staticmethod
    def _records(count: int) -> list[dict[str, object]]:
        start = datetime(2026, 5, 25, tzinfo=timezone.utc)
        return [
            {
                "pair_id": f"{index + 1:04d}",
                "filename": f"frame_{index + 1:04d}_T.JPG",
                "timestamp": (start + timedelta(seconds=index)).isoformat(),
                "gimbal_pitch_deg": -45.0,
                "gimbal_yaw_deg": 90.0,
                "original_files": {
                    "rgb": f"rgb/{index + 1:04d}.jpg",
                    "thermal": f"thermal/{index + 1:04d}.jpg",
                },
                "temperature_npy": f"temperature/{index + 1:04d}.npy",
            }
            for index in range(count)
        ]

    def test_complete_blocks_and_guards_follow_hashed_phase(self) -> None:
        result = build_split.build_split_manifest(
            self._records(160), scene="Fixture", seed="fixed-seed"
        )
        self.assertEqual(result["rule"]["ordering_mode"], "timestamp_gimbal")
        strip = result["strips"][0]
        phase = strip["phase"]
        expected_test_blocks = [index for index in range(10) if index % 8 == phase]
        self.assertEqual(strip["test_block_indices"], expected_test_blocks)

        test_records = [r for r in result["records"] if r["split"] == "test"]
        self.assertEqual(len(test_records), 16 * len(expected_test_blocks))
        self.assertTrue(all(r["block_index"] in expected_test_blocks for r in test_records))
        self.assertTrue(all(r["rule"] == "periodic_complete_test_block" for r in test_records))

        test_positions = {r["position_in_strip"] for r in test_records}
        expected_guards: set[int] = set()
        for block in expected_test_blocks:
            start = block * 16
            end = start + 16
            expected_guards.update(range(max(0, start - 2), start))
            expected_guards.update(range(end, min(160, end + 2)))
        expected_guards.difference_update(test_positions)
        actual_guards = {
            r["position_in_strip"] for r in result["records"] if r["split"] == "guard"
        }
        self.assertEqual(actual_guards, expected_guards)
        self.assertTrue(test_positions.isdisjoint(actual_guards))

        sample = result["records"][0]
        self.assertIn(sample["split"], {"train", "test", "guard"})
        self.assertEqual(sample["pair_id"], "0001")
        self.assertIn("thermal", sample["original_files"])
        self.assertIn("stratum", sample)
        self.assertIn("rule", sample)
        self.assertEqual(len(sample["hash"]), 64)

    def test_same_inputs_produce_same_split_hash(self) -> None:
        first = build_split.build_split_manifest(
            self._records(144), scene="Fixture", seed="fixed-seed"
        )
        second = build_split.build_split_manifest(
            self._records(144), scene="Fixture", seed="fixed-seed"
        )
        self.assertEqual(first["split_hash"], second["split_hash"])
        self.assertEqual(first["records"], second["records"])

    def test_missing_metadata_falls_back_to_natural_filename_order(self) -> None:
        records = self._records(20)
        records[5].pop("timestamp")
        records = list(reversed(records))
        result = build_split.build_split_manifest(
            records, scene="Fallback", seed="fixed-seed"
        )
        self.assertEqual(result["rule"]["ordering_mode"], "filename_order_fallback")
        self.assertFalse(result["metadata_reliability"]["reliable"])
        self.assertEqual(
            [record["pair_id"] for record in result["records"]],
            [f"{index:04d}" for index in range(1, 21)],
        )
        self.assertTrue(all(record["stratum"] == "filename_order" for record in result["records"]))

    def test_audit_capture_time_is_accepted_and_short_strip_gets_a_test_block(self) -> None:
        records = self._records(32)
        for record in records:
            record["capture_time"] = record.pop("timestamp")
        result = build_split.build_split_manifest(
            records, scene="Short", seed="fixed-seed"
        )
        self.assertEqual(result["rule"]["ordering_mode"], "timestamp_gimbal")
        self.assertEqual(result["counts"]["test"], 16)
        self.assertEqual(result["strips"][0]["full_block_count"], 2)


if __name__ == "__main__":
    unittest.main()
