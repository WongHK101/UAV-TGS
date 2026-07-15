from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.thermal_radiometry import evaluate_temperature
from tools.thermal_radiometry import render_canonical_palette


class TemperatureEvaluatorTests(unittest.TestCase):
    @staticmethod
    def _write_split(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema_name": "uav_tgs_deterministic_block_split",
                    "schema_version": 1,
                    "scene": "Fixture",
                    "split_hash": "a" * 64,
                    "records": [
                        {"pair_id": "0001", "split": "train", "strip_id": "tg-0000"},
                        {"pair_id": "0002", "split": "test", "strip_id": "tg-0000"},
                        {"pair_id": "0003", "split": "test", "strip_id": "tg-0000"},
                    ],
                }
            ),
            encoding="utf-8",
        )

    def test_split_mask_support_missing_and_macro_micro_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "renders"
            mask_root = root / "masks"
            gt_root.mkdir()
            mask_root.mkdir()
            for index in range(1, 4):
                values = np.array(
                    [[10.0, 20.0], [30.0, 40.0]], dtype=np.float32
                ) + np.float32(index - 1)
                np.save(gt_root / f"{index:04d}.npy", values, allow_pickle=False)

            # Only one of the two selected test frames is rendered.  Its two
            # unsupported pixels are deliberately changed, which must affect
            # the all-pixel diagnostic but not the primary supported score.
            single_gt = root / "single_gt"
            single_gt.mkdir()
            np.save(
                single_gt / "0002.npy",
                np.load(gt_root / "0002.npy", allow_pickle=False),
                allow_pickle=False,
            )
            render_canonical_palette.render_tree(
                single_gt, render_root, tmin_c=10.0, tmax_c=42.0
            )
            render_path = render_root / "0002.png"
            with Image.open(render_path) as image:
                pixels = np.asarray(image).copy()
            pixels[:, 1] = [0, 0, 0]
            Image.fromarray(pixels, mode="RGB").save(render_path)
            Image.fromarray(
                np.array([[255, 0], [255, 0]], dtype=np.uint8), mode="L"
            ).save(mask_root / "0002.png")

            split_path = root / "split.json"
            self._write_split(split_path)
            result = evaluate_temperature.evaluate_temperature_tree(
                gt_root,
                render_root,
                tmin_c=10.0,
                tmax_c=42.0,
                split_manifest=split_path,
                subset="test",
                mask_root=mask_root,
                alpha_threshold=0.5,
            )

            self.assertEqual(result["status"], "completed_with_missing")
            self.assertTrue(result["primary_metric_valid"])
            self.assertTrue(result["completed_with_missing"])
            self.assertFalse(result["support_is_explicit"])
            self.assertEqual(result["split"]["subset"], "test")
            self.assertEqual(
                result["model_render_metric_name"],
                "palette-inverted TSDK-referenced apparent-temperature error",
            )
            coverage = result["summary"]["support_coverage"]
            self.assertEqual(coverage["expected_frames"], 2)
            self.assertEqual(coverage["missing_render_frames"], 1)
            self.assertEqual(coverage["supported_pixels"], 2)
            self.assertEqual(coverage["unsupported_pixels"], 2)
            self.assertEqual(coverage["missing_render_pixels"], 4)
            self.assertEqual(coverage["missing_mask_pixels"], 0)
            self.assertEqual(coverage["unsupported_ratio"], 0.25)
            self.assertEqual(coverage["missing_ratio"], 0.5)
            self.assertEqual(coverage["unsupported_or_missing_ratio"], 0.75)

            micro = result["summary"]["pixel_micro_aggregate"]
            self.assertEqual(micro["supported_pixels"]["pixels"], 2)
            self.assertEqual(micro["all_pixels_diagnostic"]["pixels"], 4)
            self.assertLess(
                micro["supported_pixels"]["mae_c"],
                micro["all_pixels_diagnostic"]["mae_c"],
            )
            macro = result["summary"]["frame_macro_mean_std"]
            self.assertEqual(macro["supported_pixels"]["frame_count"], 1)
            self.assertEqual(macro["all_pixels_diagnostic"]["frame_count"], 1)
            self.assertIn("std", macro["supported_pixels"]["mae_c"])
            self.assertIn("supported_pixels", result["summary"]["clipping_aggregates"])
            self.assertIn("supported_pixels", result["summary"]["off_lut_distance_aggregates"])
            self.assertEqual(
                {entry["status"] for entry in result["files"]},
                {"complete", "missing_render"},
            )

    def test_rgba_alpha_is_used_when_external_mask_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            rgb_root = root / "rgb"
            render_root = root / "render"
            gt_root.mkdir()
            values = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            np.save(gt_root / "frame.npy", values, allow_pickle=False)
            render_canonical_palette.render_tree(
                gt_root, rgb_root, tmin_c=10.0, tmax_c=40.0
            )
            render_root.mkdir()
            with Image.open(rgb_root / "frame.png") as image:
                rgb = np.asarray(image).copy()
            alpha = np.array([[255, 0], [128, 127]], dtype=np.uint8)
            rgba = np.dstack((rgb, alpha))
            Image.fromarray(rgba, mode="RGBA").save(render_root / "frame.png")

            result = evaluate_temperature.evaluate_temperature_tree(
                gt_root,
                render_root,
                tmin_c=10.0,
                tmax_c=40.0,
                alpha_threshold=0.5,
            )
            coverage = result["summary"]["support_coverage"]
            self.assertEqual(coverage["supported_pixels"], 2)
            self.assertEqual(coverage["unsupported_pixels"], 2)
            self.assertEqual(result["files"][0]["support_source"], "render_alpha")
            self.assertTrue(result["files"][0]["support_is_explicit"])
            self.assertTrue(result["support_is_explicit"])
            self.assertTrue(result["primary_metric_valid"])

    def test_missing_external_mask_keeps_all_pixel_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            mask_root = root / "empty_masks"
            gt_root.mkdir()
            mask_root.mkdir()
            np.save(
                gt_root / "frame.npy",
                np.full((2, 2), 20.0, dtype=np.float32),
                allow_pickle=False,
            )
            render_canonical_palette.render_tree(
                gt_root, render_root, tmin_c=10.0, tmax_c=30.0
            )
            result = evaluate_temperature.evaluate_temperature_tree(
                gt_root,
                render_root,
                tmin_c=10.0,
                tmax_c=30.0,
                mask_root=mask_root,
            )
            coverage = result["summary"]["support_coverage"]
            self.assertEqual(coverage["missing_mask_pixels"], 4)
            self.assertEqual(coverage["missing_ratio"], 1.0)
            self.assertEqual(
                result["summary"]["temperature_error_supported_pixels"]["pixels"], 0
            )
            self.assertEqual(
                result["summary"]["temperature_error_all_pixels_diagnostic"]["pixels"], 4
            )
            self.assertEqual(result["files"][0]["status"], "missing_mask")
            self.assertEqual(result["status"], "invalid_no_supported_pixels")
            self.assertFalse(result["primary_metric_valid"])
            self.assertTrue(result["completed_with_missing"])

    def test_implicit_support_is_marked_and_require_support_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            gt_root.mkdir()
            np.save(
                gt_root / "frame.npy",
                np.full((2, 2), 20.0, dtype=np.float32),
                allow_pickle=False,
            )
            render_canonical_palette.render_tree(
                gt_root, render_root, tmin_c=10.0, tmax_c=30.0
            )
            result = evaluate_temperature.evaluate_temperature_tree(
                gt_root, render_root, tmin_c=10.0, tmax_c=30.0
            )
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["primary_metric_valid"])
            self.assertFalse(result["support_is_explicit"])
            self.assertEqual(
                result["summary"]["support_coverage"]["implicit_support_frames"], 1
            )
            self.assertTrue(any("legacy compatibility" in item for item in result["warnings"]))

            with self.assertRaisesRegex(ValueError, "Explicit support is required"):
                evaluate_temperature.evaluate_temperature_tree(
                    gt_root,
                    render_root,
                    tmin_c=10.0,
                    tmax_c=30.0,
                    require_support=True,
                )

    def test_zero_support_is_not_reported_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            mask_root = root / "mask"
            gt_root.mkdir()
            mask_root.mkdir()
            np.save(
                gt_root / "frame.npy",
                np.full((2, 2), 20.0, dtype=np.float32),
                allow_pickle=False,
            )
            render_canonical_palette.render_tree(
                gt_root, render_root, tmin_c=10.0, tmax_c=30.0
            )
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8), mode="L").save(
                mask_root / "frame.png"
            )
            result = evaluate_temperature.evaluate_temperature_tree(
                gt_root,
                render_root,
                tmin_c=10.0,
                tmax_c=30.0,
                mask_root=mask_root,
            )
            self.assertEqual(result["status"], "invalid_no_supported_pixels")
            self.assertFalse(result["primary_metric_valid"])
            self.assertFalse(result["completed_with_missing"])
            self.assertTrue(result["support_is_explicit"])

    def test_duplicate_mask_stem_and_ambiguous_render_basename_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            mask_root = root / "mask"
            gt_root.mkdir()
            render_root.mkdir()
            mask_root.mkdir()
            np.save(gt_root / "frame.npy", np.ones((1, 1), dtype=np.float32))
            Image.new("RGB", (1, 1)).save(render_root / "frame.png")
            Image.new("L", (1, 1), 255).save(mask_root / "frame.png")
            np.save(mask_root / "frame.npy", np.ones((1, 1), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "same relative stem"):
                evaluate_temperature.evaluate_temperature_tree(
                    gt_root,
                    render_root,
                    tmin_c=0.0,
                    tmax_c=2.0,
                    mask_root=mask_root,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            gt_root.mkdir()
            (render_root / "a").mkdir(parents=True)
            (render_root / "b").mkdir(parents=True)
            np.save(gt_root / "frame.npy", np.ones((1, 1), dtype=np.float32))
            Image.new("RGB", (1, 1)).save(render_root / "a" / "frame.png")
            Image.new("RGB", (1, 1)).save(render_root / "b" / "frame.png")
            with self.assertRaisesRegex(ValueError, "Ambiguous basename"):
                evaluate_temperature.evaluate_temperature_tree(
                    gt_root, render_root, tmin_c=0.0, tmax_c=2.0
                )

    def test_subset_requires_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_root = root / "temperature"
            render_root = root / "render"
            gt_root.mkdir()
            render_root.mkdir()
            np.save(gt_root / "frame.npy", np.ones((1, 1), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "subset requires split_manifest"):
                evaluate_temperature.evaluate_temperature_tree(
                    gt_root,
                    render_root,
                    tmin_c=0.0,
                    tmax_c=2.0,
                    subset="test",
                )


if __name__ == "__main__":
    unittest.main()
