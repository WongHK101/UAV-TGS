from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "geometric_repeatability"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))

import build_depth_reference as backend


class ProbeCameraInterfaceTests(unittest.TestCase):
    def test_legacy_manifest_paths_remain_the_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "reference"
            thermal = root / "thermal"
            train = root / "train.txt"
            probe = root / "probe.txt"
            strict = {
                "artifacts": {
                    "train_union_source_root": str(workspace),
                    "strict_thermal_root": str(thermal),
                },
                "lists": {
                    "train_union": str(train),
                    "probe_test": str(probe),
                },
            }

            resolved = backend._resolve_reference_protocol_paths(strict)

            self.assertEqual(resolved["probe_camera_root"], thermal.resolve())
            self.assertEqual(resolved["probe_camera_train_list"], train.resolve())
            self.assertEqual(resolved["reference_probe_exclusion_list"], probe.resolve())
            self.assertFalse(resolved["extended_probe_camera_interface"])

    def test_formal_manifest_separates_rgb_reference_and_png_probe_camera_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            strict = {
                "artifacts": {
                    "train_union_source_root": str(root / "reference_rgb"),
                    "strict_thermal_root": str(root / "legacy_thermal"),
                    "probe_camera_root": str(root / "probe_camera_png"),
                },
                "lists": {
                    "train_union": str(root / "rgb_train.txt"),
                    "probe_camera_train": str(root / "thermal_train.txt"),
                    "probe_test": str(root / "thermal_test.txt"),
                    "reference_probe_exclusion": str(root / "rgb_test.txt"),
                },
            }

            resolved = backend._resolve_reference_protocol_paths(strict)

            self.assertEqual(resolved["workspace_root"], (root / "reference_rgb").resolve())
            self.assertEqual(
                resolved["probe_camera_root"], (root / "probe_camera_png").resolve()
            )
            self.assertEqual(
                resolved["probe_camera_train_list"], (root / "thermal_train.txt").resolve()
            )
            self.assertEqual(
                resolved["reference_probe_exclusion_list"], (root / "rgb_test.txt").resolve()
            )
            self.assertTrue(resolved["extended_probe_camera_interface"])

    def test_jpg_png_partitions_are_accepted_when_only_extensions_differ(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_train = root / "rgb_train.txt"
            rgb_test = root / "rgb_test.txt"
            thermal_train = root / "thermal_train.txt"
            thermal_test = root / "thermal_test.txt"
            rgb_train.write_text("nested/0001.JPG\n0002.JPG\n", encoding="utf-8")
            rgb_test.write_text("0003.JPG\n", encoding="utf-8")
            thermal_train.write_text("nested/0001.png\n0002.png\n", encoding="utf-8")
            thermal_test.write_text("0003.png\n", encoding="utf-8")

            audit = backend._validate_probe_camera_partition_stems(
                rgb_train,
                rgb_test,
                thermal_train,
                thermal_test,
            )

            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["train_stem_count"], 2)
            self.assertEqual(audit["probe_stem_count"], 1)
            self.assertEqual(audit["reference_train_extensions"], [".jpg"])
            self.assertEqual(audit["probe_camera_train_extensions"], [".png"])

    def test_partition_stem_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_train = root / "rgb_train.txt"
            rgb_test = root / "rgb_test.txt"
            thermal_train = root / "thermal_train.txt"
            thermal_test = root / "thermal_test.txt"
            rgb_train.write_text("0001.JPG\n", encoding="utf-8")
            rgb_test.write_text("0003.JPG\n", encoding="utf-8")
            thermal_train.write_text("0002.png\n", encoding="utf-8")
            thermal_test.write_text("0003.png\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "differ beyond image extensions"):
                backend._validate_probe_camera_partition_stems(
                    rgb_train,
                    rgb_test,
                    thermal_train,
                    thermal_test,
                )

    def test_duplicate_or_overlapping_stems_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb_train = root / "rgb_train.txt"
            rgb_test = root / "rgb_test.txt"
            thermal_train = root / "thermal_train.txt"
            thermal_test = root / "thermal_test.txt"
            rgb_train.write_text("0001.JPG\n0001.png\n", encoding="utf-8")
            rgb_test.write_text("0002.JPG\n", encoding="utf-8")
            thermal_train.write_text("0001.png\n", encoding="utf-8")
            thermal_test.write_text("0002.png\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate image stems"):
                backend._validate_probe_camera_partition_stems(
                    rgb_train,
                    rgb_test,
                    thermal_train,
                    thermal_test,
                )

            rgb_train.write_text("0001.JPG\n", encoding="utf-8")
            rgb_test.write_text("0001.JPG\n", encoding="utf-8")
            thermal_test.write_text("0001.png\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "partitions overlap"):
                backend._validate_probe_camera_partition_stems(
                    rgb_train,
                    rgb_test,
                    thermal_train,
                    thermal_test,
                )


if __name__ == "__main__":
    unittest.main()
