from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.thermal_radiometry import estimate_scene_range as estimate_range


class EstimateSceneRangeTests(unittest.TestCase):
    def _fixture(self, root: Path, test_values: np.ndarray) -> dict[str, object]:
        np.save(root / "train_a.npy", np.arange(0, 100, dtype=np.float32).reshape(10, 10))
        np.save(root / "train_b.npy", np.arange(100, 200, dtype=np.float32).reshape(10, 10))
        np.save(root / "test.npy", test_values.astype(np.float32))
        return {
            "scene": "Fixture",
            "split_hash": "fixture-split",
            "records": [
                {"pair_id": "0001", "split": "train", "temperature_npy": "train_a.npy"},
                {"pair_id": "0002", "split": "train", "temperature_npy": "train_b.npy"},
                {"pair_id": "0003", "split": "test", "temperature_npy": "test.npy"},
                {
                    "pair_id": "0004",
                    "split": "guard",
                    "temperature_npy": "guard_file_must_not_be_read.npy",
                },
            ],
        }

    def test_train_only_range_and_test_clipping_qa(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = self._fixture(root, np.array([[-1000.0, 50.0], [150.0, 1000.0]]))
            result = estimate_range.estimate_scene_range(
                split,
                manifest_dir=root,
                histogram_bins=4096,
                chunk_pixels=17,
            )

            self.assertEqual(result["train_estimation"]["frame_count"], 2)
            self.assertEqual(result["train_estimation"]["valid_count"], 200)
            self.assertGreater(result["Tmin"], -10.0)
            self.assertLess(result["Tmax"], 210.0)
            self.assertEqual(result["clipping_stats"]["test"]["frame_count"], 1)
            self.assertEqual(result["clipping_stats"]["test"]["low_count"], 1)
            self.assertEqual(result["clipping_stats"]["test"]["high_count"], 1)
            self.assertEqual(result["clipping_stats"]["test"]["clipped_fraction"], 0.5)
            self.assertEqual(
                {row["split"] for row in result["per_frame_quantiles"]}, {"train", "test"}
            )

    def test_test_values_do_not_change_estimated_range(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as second_tmp:
            first_root = Path(first_tmp)
            second_root = Path(second_tmp)
            first = estimate_range.estimate_scene_range(
                self._fixture(first_root, np.array([[0.0, 1.0]])),
                manifest_dir=first_root,
                histogram_bins=2048,
                chunk_pixels=13,
            )
            second = estimate_range.estimate_scene_range(
                self._fixture(second_root, np.array([[-1e6, 1e6]])),
                manifest_dir=second_root,
                histogram_bins=2048,
                chunk_pixels=13,
            )
            self.assertEqual(first["Tmin"], second["Tmin"])
            self.assertEqual(first["Tmax"], second["Tmax"])
            self.assertEqual(first["range_hash"], second["range_hash"])

    def test_non_float32_or_nonfinite_maps_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "bad.npy", np.array([[1.0, np.nan]], dtype=np.float32))
            split = {
                "scene": "Fixture",
                "split_hash": "fixture-split",
                "records": [
                    {"pair_id": "bad", "split": "train", "temperature_npy": "bad.npy"}
                ],
            }
            with self.assertRaisesRegex(ValueError, "NaN or infinity"):
                estimate_range.estimate_scene_range(split, manifest_dir=root)


if __name__ == "__main__":
    unittest.main()
