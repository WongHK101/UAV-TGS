import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.thermal_radiometry import palette_lut
from tools.thermal_radiometry import tsdk_palette_display


def _synthetic_tsdk_bundle(root: Path) -> tsdk_palette_display.TSDKPaletteBundle:
    values = np.arange(256, dtype=np.uint8)
    luts = np.empty((10, 256, 3), dtype=np.uint8)
    for index in range(10):
        luts[index, :, 0] = values
        luts[index, :, 1] = np.bitwise_xor(values, np.uint8(17 + index))
        luts[index, :, 2] = np.bitwise_xor(values, np.uint8(101 + index))
    reference = root / "reference.JPG"
    library = root / "libdirp.mock"
    reference.write_bytes(b"synthetic-rjpeg")
    library.write_bytes(b"synthetic-library")
    return tsdk_palette_display.TSDKPaletteBundle(
        luts_rgb=luts,
        original_palette_index=3,
        original_color_bar_manual=False,
        original_color_bar_low_c=25.0,
        original_color_bar_high_c=30.0,
        original_brightness=50,
        reference_rjpeg=reference,
        reference_rjpeg_sha256=tsdk_palette_display._sha256(reference),
        library_path=library,
        library_sha256=tsdk_palette_display._sha256(library),
    )


class TSDKPaletteDisplayTests(unittest.TestCase):
    def test_exact_canonical_input_exports_reversible_scalar_and_palette(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "renders"
            output = root / "output"
            source.mkdir()
            expected_indices = np.array(
                [[0, 1, 127], [128, 254, 255]], dtype=np.uint8
            )
            Image.fromarray(
                palette_lut.hot_iron_lut()[expected_indices], mode="RGB"
            ).save(source / "frame.png")
            bundle = _synthetic_tsdk_bundle(root)

            result = tsdk_palette_display.export_render_tree(
                source,
                output,
                tmin_c=10.0,
                tmax_c=40.0,
                bundle=bundle,
                palette_names=("hot_iron", "white_hot"),
            )

            recovered_indices = np.load(
                output / "temperature_index" / "frame.npy",
                allow_pickle=False,
            )
            recovered_temperature = np.load(
                output / "apparent_temperature_c" / "frame.npy",
                allow_pickle=False,
            )
            np.testing.assert_array_equal(recovered_indices, expected_indices)
            np.testing.assert_allclose(
                recovered_temperature,
                palette_lut.indices_to_temperature(
                    expected_indices, 10.0, 40.0
                ),
                rtol=0.0,
                atol=0.0,
            )
            hot_iron = np.asarray(
                Image.open(
                    output / "palettes" / "hot_iron" / "frame.png"
                ).convert("RGB")
            )
            np.testing.assert_array_equal(
                hot_iron, bundle.palette("hot_iron")[expected_indices]
            )
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["summary"]["file_count"], 1)
            self.assertEqual(
                result["files"][0]["off_lut_distance_rgb"]["maximum"], 0.0
            )
            self.assertTrue(
                result["files"][0]["palettes"]["hot_iron"][
                    "index_roundtrip_exact"
                ]
            )

    def test_off_lut_input_is_reported_and_optional_map_is_saved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "renders"
            output = root / "output"
            source.mkdir()
            pixels = palette_lut.hot_iron_lut()[
                np.array([[10, 20], [30, 40]], dtype=np.uint8)
            ].copy()
            pixels[0, 0] = [13, 91, 207]
            Image.fromarray(pixels, mode="RGB").save(source / "frame.png")

            result = tsdk_palette_display.export_render_tree(
                source,
                output,
                tmin_c=0.0,
                tmax_c=100.0,
                bundle=_synthetic_tsdk_bundle(root),
                palette_names=("rainbow2",),
                save_off_lut_map=True,
            )

            record = result["files"][0]
            self.assertGreater(
                record["off_lut_distance_rgb"]["maximum"], 0.0
            )
            off_lut = np.load(
                output / "off_lut_distance_rgb" / "frame.npy",
                allow_pickle=False,
            )
            self.assertGreater(float(off_lut[0, 0]), 0.0)
            self.assertEqual(off_lut.dtype, np.float32)
            self.assertFalse(result["conversion"]["model_modified"])
            self.assertFalse(result["conversion"]["formal_metrics_modified"])

    def test_output_inside_source_and_non_unique_palette_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "renders"
            source.mkdir()
            Image.fromarray(
                palette_lut.hot_iron_lut()[
                    np.array([[0, 255]], dtype=np.uint8)
                ],
                mode="RGB",
            ).save(source / "frame.png")
            bundle = _synthetic_tsdk_bundle(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                tsdk_palette_display.export_render_tree(
                    source,
                    source / "output",
                    tmin_c=0.0,
                    tmax_c=1.0,
                    bundle=bundle,
                )
            bad_luts = bundle.luts_rgb.copy()
            bad_luts[3, 1] = bad_luts[3, 0]
            with self.assertRaisesRegex(ValueError, "not one-to-one"):
                tsdk_palette_display.TSDKPaletteBundle(
                    luts_rgb=bad_luts,
                    original_palette_index=3,
                    original_color_bar_manual=False,
                    original_color_bar_low_c=0.0,
                    original_color_bar_high_c=1.0,
                    original_brightness=50,
                    reference_rjpeg=bundle.reference_rjpeg,
                    reference_rjpeg_sha256=bundle.reference_rjpeg_sha256,
                    library_path=bundle.library_path,
                    library_sha256=bundle.library_sha256,
                )


if __name__ == "__main__":
    unittest.main()
