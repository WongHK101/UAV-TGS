from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.thermal_radiometry import build_formal_split, build_split


class FormalSplitTests(unittest.TestCase):
    @staticmethod
    def _records(scene: str, count: int = 160) -> list[dict[str, object]]:
        start = datetime(2026, 5, 25, tzinfo=timezone.utc)
        return [
            {
                "scene": scene,
                "pair_id": f"{index + 1:04d}",
                "filename": f"{scene}_{index + 1:04d}_T.JPG",
                "capture_time": (start + timedelta(seconds=index)).isoformat(),
                "gimbal_pitch_deg": -45.0,
                "gimbal_yaw_deg": 90.0,
                "source_path": f"{scene}/thermal/{index + 1:04d}.JPG",
            }
            for index in range(count)
        ]

    def test_formal_constants_match_single_scene_default(self) -> None:
        self.assertEqual(build_split.DEFAULT_GUARD_FRAMES, 4)
        self.assertEqual(build_formal_split.FORMAL_BLOCK_SIZE, 16)
        self.assertEqual(build_formal_split.FORMAL_TEST_PERIOD_BLOCKS, 8)
        self.assertEqual(build_formal_split.FORMAL_GUARD_FRAMES, 4)
        self.assertEqual(build_formal_split.FORMAL_MINIMUM_TRAIN_FRAMES, 16)

    def test_formal_collection_binds_collection_and_selected_block_hashes(self) -> None:
        records = [
            record
            for scene in build_formal_split.EXPECTED_SCENES
            for record in self._records(scene)
        ]
        building = [record for record in records if record["scene"] == "Building"]
        for record in building[:16]:
            record["stratum"] = "short"
            record["strip_id"] = "short-strip"
        for record in building[16:]:
            record["stratum"] = "supported"
            record["strip_id"] = "supported-strip"
        collection, scenes = build_formal_split.build_formal_collection(
            records,
            source_manifest=Path("collection.jsonl"),
            source_manifest_sha256="a" * 64,
        )
        self.assertEqual(collection["scene_count"], 11)
        self.assertEqual(collection["formal_rule"]["guard_frames_each_side"], 4)
        self.assertTrue(collection["formal_rule"]["strata_without_test_allowed"])
        self.assertEqual(collection["validation"]["status"], "passed")
        self.assertEqual(len(collection["collection_hash"]), 64)
        self.assertEqual(len(collection["collection_split_hash"]), 64)
        self.assertEqual(collection["counts"]["total"], 1760)
        short = next(
            item
            for item in scenes["Building"]["stratum_allocations"]
            if item["stratum"] == "short"
        )
        self.assertTrue(short["without_test"])
        self.assertEqual(short["counts"], {"train": 16, "test": 0, "guard": 0})
        for scene in scenes.values():
            self.assertEqual(scene["rule"]["guard_frames_each_side"], 4)
            self.assertEqual(scene["validation"]["status"], "passed")
            self.assertTrue(scene["validation"]["no_strip_without_train"])
            self.assertTrue(scene["validation"]["no_stratum_without_train"])
            self.assertEqual(len(scene["selected_test_blocks_hash"]), 64)
            self.assertEqual(len(scene["selected_candidate_hashes"]), 1)

    def test_missing_formal_scene_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "formal scene set mismatch"):
            build_formal_split.build_formal_collection(
                self._records("Building"),
                source_manifest=Path("collection.jsonl"),
                source_manifest_sha256="a" * 64,
            )

    def test_formal_seed_is_frozen_in_api_and_absent_from_cli(self) -> None:
        records = [
            record
            for scene in build_formal_split.EXPECTED_SCENES
            for record in self._records(scene)
        ]
        with self.assertRaisesRegex(ValueError, "formal seed is frozen"):
            build_formal_split.build_formal_collection(
                records,
                source_manifest=Path("collection.jsonl"),
                source_manifest_sha256="a" * 64,
                seed="different-seed",
            )
        cli_options = {
            option
            for action in build_formal_split._parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--seed", cli_options)

    def test_materialised_collection_is_deterministic_and_overwrite_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "collection.jsonl"
            output = root / "formal"
            records = [
                record
                for scene in build_formal_split.EXPECTED_SCENES
                for record in self._records(scene)
            ]
            source.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            first = build_formal_split.materialise_formal_collection(source, output)
            persisted = json.loads(
                (output / "collection_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first, persisted)
            self.assertEqual(len(list((output / "scenes").glob("*.split.json"))), 11)
            self.assertTrue((output / "scene_summary.csv").is_file())
            with self.assertRaises(FileExistsError):
                build_formal_split.materialise_formal_collection(source, output)
            second = build_formal_split.materialise_formal_collection(
                source, output, overwrite=True
            )
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
