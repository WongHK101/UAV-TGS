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

    def test_scene_budget_selects_complete_blocks_with_guards(self) -> None:
        result = build_split.build_split_manifest(
            self._records(160), scene="Fixture", seed="fixed-seed"
        )
        self.assertEqual(result["rule"]["ordering_mode"], "timestamp_gimbal")
        strip = result["strips"][0]
        self.assertEqual(result["test_block_budget"], {"target": 1, "selected": 1, "shortfall": 0})
        self.assertEqual(len(strip["test_block_indices"]), 1)
        expected_test_blocks = strip["test_block_indices"]

        test_records = [r for r in result["records"] if r["split"] == "test"]
        self.assertEqual(len(test_records), 16 * len(expected_test_blocks))
        self.assertTrue(all(r["block_index"] in expected_test_blocks for r in test_records))
        self.assertTrue(
            all(r["rule"] == "scene_budget_stratum_allocated_test_block" for r in test_records)
        )

        test_positions = {r["position_in_strip"] for r in test_records}
        expected_guards: set[int] = set()
        for block in expected_test_blocks:
            start = block * 16
            end = start + 16
            expected_guards.update(
                range(max(0, start - build_split.DEFAULT_GUARD_FRAMES), start)
            )
            expected_guards.update(
                range(end, min(160, end + build_split.DEFAULT_GUARD_FRAMES))
            )
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
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertAlmostEqual(result["validation"]["test_fraction_of_scene"], 0.1)

    def test_empty_records_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "records must not be empty"):
            build_split.build_split_manifest(
                [], scene="Empty", seed="fixed-seed"
            )

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

    def test_audit_capture_time_is_accepted_and_short_scene_stays_train(self) -> None:
        records = self._records(32)
        for record in records:
            record["capture_time"] = record.pop("timestamp")
        result = build_split.build_split_manifest(
            records, scene="Short", seed="fixed-seed"
        )
        self.assertEqual(result["rule"]["ordering_mode"], "timestamp_gimbal")
        self.assertEqual(result["counts"]["test"], 0)
        self.assertEqual(result["counts"]["train"], 32)
        self.assertEqual(result["strips"][0]["full_block_count"], 2)
        self.assertEqual(result["test_block_budget"]["target"], 0)
        self.assertEqual(result["validation"]["status"], "passed")

    def test_short_stratum_is_train_and_budget_moves_to_supported_stratum(self) -> None:
        records = self._records(256)
        for index, record in enumerate(records):
            if index < 16:
                record["stratum"] = "short"
                record["strip_id"] = "short-strip"
            else:
                record["stratum"] = "supported"
                record["strip_id"] = "supported-strip"
        result = build_split.build_split_manifest(
            records, scene="Strata", seed="fixed-seed"
        )
        self.assertEqual(result["test_block_budget"]["target"], 2)
        short = next(item for item in result["stratum_allocations"] if item["stratum"] == "short")
        supported = next(
            item for item in result["stratum_allocations"] if item["stratum"] == "supported"
        )
        self.assertEqual(short["feasible_capacity"], 0)
        self.assertEqual(short["counts"], {"train": 16, "test": 0, "guard": 0})
        self.assertEqual(supported["selected_block_count"], 2)
        self.assertEqual(result["counts"]["test"], 32)
        self.assertEqual(result["validation"]["status"], "passed")

    def test_pvpanel_shape_uses_scene_budget_instead_of_one_test_per_short_strip(self) -> None:
        records = self._records(289)
        offset = 0
        for strip_index, strip_size in enumerate([65, *([56] * 4)]):
            for record in records[offset : offset + strip_size]:
                record["stratum"] = f"pv-stratum-{strip_index:02d}"
                record["strip_id"] = f"pv-strip-{strip_index:02d}"
            offset += strip_size

        result = build_split.build_split_manifest(
            records, scene="PVpanelLike", seed="fixed-seed"
        )

        self.assertEqual(result["test_block_budget"]["target"], 2)
        self.assertEqual(result["test_block_budget"]["selected"], 2)
        self.assertEqual(result["counts"]["test"], 32)
        self.assertEqual(
            sum(bool(strip["test_block_indices"]) for strip in result["strips"]),
            2,
        )
        self.assertTrue(
            all(strip["train_frame_count"] >= 16 for strip in result["strips"])
        )
        self.assertEqual(result["validation"]["status"], "passed")

    def test_budget_shortfall_fails_closed(self) -> None:
        records = self._records(160)
        for index, record in enumerate(records):
            strip_index = index // 16
            record["stratum"] = f"short-{strip_index:02d}"
            record["strip_id"] = f"short-{strip_index:02d}"

        with self.assertRaisesRegex(
            build_split.SplitAllocationError, "test_block_budget_shortfall"
        ):
            build_split.build_split_manifest(
                records, scene="NoEligibleBlocks", seed="fixed-seed"
            )

        diagnostic = build_split.build_split_manifest(
            records,
            scene="NoEligibleBlocks",
            seed="fixed-seed",
            fail_on_invalid=False,
        )
        self.assertEqual(diagnostic["test_block_budget"]["shortfall"], 1)
        self.assertEqual(diagnostic["validation"]["status"], "failed")

    def test_cumulative_eligibility_preserves_sixteen_training_frames(self) -> None:
        result = build_split.build_split_manifest(
            self._records(47),
            scene="Capacity",
            seed="fixed-seed",
            fail_on_invalid=False,
        )
        allocation = result["stratum_allocations"][0]
        self.assertEqual(result["strips"][0]["full_block_count"], 2)
        self.assertEqual(allocation["feasible_capacity"], 1)

    def test_input_content_and_manifest_hash_are_bound_into_split_hash(self) -> None:
        records = self._records(144)
        first = build_split.build_split_manifest(
            records,
            scene="Fixture",
            seed="fixed-seed",
            source_manifest_sha256="a" * 64,
        )
        changed_records = self._records(144)
        changed_records[0]["unused_metadata"] = "changed"
        changed = build_split.build_split_manifest(
            changed_records,
            scene="Fixture",
            seed="fixed-seed",
            source_manifest_sha256="a" * 64,
        )
        changed_manifest = build_split.build_split_manifest(
            records,
            scene="Fixture",
            seed="fixed-seed",
            source_manifest_sha256="b" * 64,
        )
        self.assertNotEqual(first["input_records_hash"], changed["input_records_hash"])
        self.assertNotEqual(first["split_hash"], changed["split_hash"])
        self.assertNotEqual(first["split_hash"], changed_manifest["split_hash"])

    def test_invalid_test_fraction_fails_closed(self) -> None:
        with self.assertRaisesRegex(build_split.SplitAllocationError, "test_fraction_out_of_range"):
            build_split.build_split_manifest(
                self._records(64), scene="Invalid", seed="fixed-seed"
            )
        diagnostic = build_split.build_split_manifest(
            self._records(64),
            scene="Invalid",
            seed="fixed-seed",
            fail_on_invalid=False,
        )
        self.assertEqual(diagnostic["validation"]["status"], "failed")
        self.assertIn("test_fraction_out_of_range", diagnostic["validation"]["errors"])


if __name__ == "__main__":
    unittest.main()
