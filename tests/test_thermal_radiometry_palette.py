import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.thermal_radiometry import evaluate_temperature
from tools.thermal_radiometry import palette_lut
from tools.thermal_radiometry import render_canonical_palette
from tools.thermal_radiometry import validate_roundtrip


class CanonicalPaletteTests(unittest.TestCase):
    def test_lut_is_fixed_unique_and_versioned(self):
        lut = palette_lut.hot_iron_lut()
        self.assertEqual(lut.shape, (256, 3))
        self.assertEqual(np.unique(lut, axis=0).shape[0], 256)
        self.assertEqual(
            palette_lut.lut_sha256(lut),
            "c50f9920dc226cbf2db7c62a94a0dd9d9a5a81e0973e73ca67c36cc66a2583d2",
        )

    def test_render_is_deterministic_and_roundtrip_is_within_half_bin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "temperature"
            first = root / "canonical_a"
            second = root / "canonical_b"
            source.mkdir()
            values = np.linspace(10.0, 40.0, 1024, dtype=np.float32).reshape(32, 32)
            np.save(source / "frame.npy", values, allow_pickle=False)
            render_canonical_palette.render_tree(
                source, first, tmin_c=10.0, tmax_c=40.0
            )
            render_canonical_palette.render_tree(
                source, second, tmin_c=10.0, tmax_c=40.0
            )
            png_a = first / "frame.png"
            png_b = second / "frame.png"
            self.assertEqual(hashlib.sha256(png_a.read_bytes()).digest(), hashlib.sha256(png_b.read_bytes()).digest())

            report = validate_roundtrip.validate_roundtrip(
                source,
                first,
                tmin_c=10.0,
                tmax_c=40.0,
                max_clipping_ratio=0.0,
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(
                report["evaluation"]["summary"]["off_lut_distance"]["p95_rgb_distance"],
                0.0,
            )
            half_bin = 0.5 * (40.0 - 10.0) / 255.0
            observed = report["evaluation"]["summary"]["temperature_error_in_range_pixels"][
                "max_abs_error_c"
            ]
            self.assertLessEqual(observed, half_bin + 1e-5)

    def test_evaluator_reports_off_lut_distance_and_uses_float_gt(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "temperature"
            canonical = root / "canonical"
            source.mkdir()
            values = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            np.save(source / "frame.npy", values, allow_pickle=False)
            render_canonical_palette.render_tree(
                source, canonical, tmin_c=10.0, tmax_c=40.0
            )
            png = canonical / "frame.png"
            with Image.open(png) as image:
                pixels = np.array(image)
            pixels[0, 0] = [0, 1, 0]
            Image.fromarray(pixels, mode="RGB").save(png, format="PNG")
            result = evaluate_temperature.evaluate_temperature_tree(
                source, canonical, tmin_c=10.0, tmax_c=40.0
            )
            self.assertEqual(
                result["metric_name"],
                "TSDK-referenced apparent-temperature consistency",
            )
            self.assertGreater(result["summary"]["off_lut_distance"]["max_rgb_distance"], 0.0)
            self.assertIn("float32 Celsius", result["ground_truth"])

    def test_range_manifest_and_source_tree_guard(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "temperature"
            source.mkdir()
            np.save(source / "frame.npy", np.ones((2, 2), dtype=np.float32), allow_pickle=False)
            manifest = root / "range.json"
            manifest.write_text(json.dumps({"Tmin": 0.0, "Tmax": 2.0}), encoding="utf-8")
            low, high, provenance = palette_lut.resolve_temperature_range(range_manifest=manifest)
            self.assertEqual((low, high), (0.0, 2.0))
            self.assertEqual(provenance["source"], "manifest")
            with self.assertRaisesRegex(ValueError, "outside the temperature source tree"):
                render_canonical_palette.render_tree(
                    source,
                    source / "canonical",
                    tmin_c=0.0,
                    tmax_c=2.0,
                )


if __name__ == "__main__":
    unittest.main()
