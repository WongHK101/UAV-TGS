from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.thermal_radiometry import rename_colmap_image_extension as rename_model
from utils.read_write_model import Camera, Image, Point3D, read_model, write_model


class RenameColmapImageExtensionTests(unittest.TestCase):
    def test_only_names_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            cameras = {1: Camera(1, "SIMPLE_RADIAL", 8, 6, np.array([4.0, 3.0, 2.0, 0.01]))}
            images = {
                1: Image(1, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "a/0001.JPG",
                         np.array([[1.0, 2.0]]), np.array([7], dtype=np.int64))
            }
            points = {
                7: Point3D(7, np.array([1.0, 2.0, 3.0]), np.array([1, 2, 3], dtype=np.uint8), 0.1,
                           np.array([1], dtype=np.int32), np.array([0], dtype=np.int32))
            }
            write_model(cameras, images, points, str(source), ext=".bin")
            result = rename_model.clone_with_extension(source, output, target_extension=".png")
            out_cameras, out_images, out_points = read_model(str(output), ext=".bin")
            self.assertEqual(out_images[1].name, "a/0001.png")
            np.testing.assert_array_equal(out_images[1].qvec, images[1].qvec)
            np.testing.assert_array_equal(out_points[7].image_ids, points[7].image_ids)
            self.assertTrue(result["invariants"]["image_names_only_mutation"])

    def test_duplicate_target_name_fails_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            cameras = {1: Camera(1, "PINHOLE", 8, 6, np.array([4.0, 4.0, 3.0, 2.0]))}
            common = (np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), 1)
            images = {
                1: Image(1, *common, "0001.JPG", np.empty((0, 2)), np.empty((0,), dtype=np.int64)),
                2: Image(2, *common, "0001.jpeg", np.empty((0, 2)), np.empty((0,), dtype=np.int64)),
            }
            write_model(cameras, images, {}, str(source), ext=".bin")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                rename_model.clone_with_extension(source, output, target_extension=".png")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
