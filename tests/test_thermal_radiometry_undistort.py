import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import cv2
    from tools.thermal_radiometry import undistort_temperature
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    cv2 = None
    undistort_temperature = None


def _write_model(
    root: Path,
    camera_line: str,
    *,
    image_name: str = "frame.jpg",
    qvec: str = "1 0 0 0",
) -> None:
    root.mkdir(parents=True)
    (root / "cameras.txt").write_text(
        "# Camera list\n" + camera_line + "\n", encoding="utf-8"
    )
    (root / "images.txt").write_text(
        f"# Image list\n1 {qvec} 0 0 0 1 {image_name}\n\n", encoding="utf-8"
    )


def _write_binary_model(root: Path) -> None:
    root.mkdir(parents=True)
    with (root / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(struct.pack("<iiQQ", 1, 2, 7, 5))
        stream.write(struct.pack("<dddd", 4.0, 3.0, 2.0, 0.08))
    with (root / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1))
        stream.write(
            struct.pack(
                "<idddddddi",
                1,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1,
            )
        )
        stream.write(b"frame.jpg\x00")
        stream.write(struct.pack("<Q", 0))


@unittest.skipIf(cv2 is None, "opencv-python-headless is not installed")
class UndistortTemperatureTests(unittest.TestCase):
    def test_binary_model_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            _write_binary_model(root)
            model = undistort_temperature.read_sparse_model(root)
            self.assertEqual(model.model_format, "bin")
            self.assertEqual(model.cameras[1].model, "SIMPLE_RADIAL")
            self.assertEqual(model.cameras[1].params, (4.0, 3.0, 2.0, 0.08))
            self.assertEqual(model.images[1].name, "frame.jpg")

    def test_simple_radial_to_pinhole_matches_opencv_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            temperature_root = root / "temperature"
            input_model = root / "input_model"
            output_model = root / "output_model"
            output_root = root / "output"
            temperature_root.mkdir()
            values = (np.arange(35, dtype=np.float32).reshape(5, 7) + 10.0).astype(np.float32)
            np.save(temperature_root / "frame.npy", values, allow_pickle=False)
            _write_model(input_model, "1 SIMPLE_RADIAL 7 5 4 3 2 0.08")
            # q and -q describe the same rotation and must both be accepted.
            _write_model(
                output_model,
                "1 PINHOLE 7 5 4 4 3 2",
                image_name="frame.png",
                qvec="-1 0 0 0",
            )

            result = undistort_temperature.remap_temperature_tree(
                temperature_root,
                input_model,
                output_model,
                output_root,
            )

            camera_matrix = np.array([[4.0, 0.0, 3.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]])
            reference_x, reference_y = cv2.initUndistortRectifyMap(
                camera_matrix,
                np.array([0.08, 0.0, 0.0, 0.0]),
                None,
                camera_matrix,
                (7, 5),
                cv2.CV_32FC1,
            )
            expected = cv2.remap(
                values,
                reference_x,
                reference_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            observed = np.load(output_root / "temperature_c" / "frame.npy", allow_pickle=False)
            support = np.load(output_root / "valid_support" / "frame.npy", allow_pickle=False)
            np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
            self.assertEqual(observed.dtype, np.float32)
            self.assertEqual(support.dtype, np.bool_)
            self.assertLess(np.count_nonzero(support), support.size)
            self.assertEqual(result["summary"]["file_count"], 1)

            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            record = manifest["files"][0]
            self.assertEqual(record["input_camera"]["model"], "SIMPLE_RADIAL")
            self.assertEqual(record["output_camera"]["model"], "PINHOLE")
            self.assertEqual(record["image_match_mode"], "relative_stem")
            self.assertEqual(record["temperature_match_mode"], "relative_stem")
            self.assertEqual(record["input_temperature"]["shape"], [5, 7])
            self.assertEqual(len(record["input_temperature"]["sha256"]), 64)
            self.assertEqual(len(record["output_temperature"]["sha256"]), 64)
            self.assertEqual(len(record["valid_support"]["sha256"]), 64)

    def test_identity_remap_is_exact_and_support_is_full(self) -> None:
        input_camera = undistort_temperature.Camera(1, "SIMPLE_PINHOLE", 4, 3, (2.0, 1.5, 1.0))
        output_camera = undistort_temperature.Camera(1, "PINHOLE", 4, 3, (2.0, 2.0, 1.5, 1.0))
        map_x, map_y, support = undistort_temperature.build_remap(input_camera, output_camera)
        expected_x, expected_y = np.meshgrid(
            np.arange(4, dtype=np.float32), np.arange(3, dtype=np.float32)
        )
        np.testing.assert_array_equal(map_x, expected_x)
        np.testing.assert_array_equal(map_y, expected_y)
        self.assertTrue(np.all(support))

    def test_shape_mismatch_fails_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            temperature_root = root / "temperature"
            input_model = root / "input_model"
            output_model = root / "output_model"
            output_root = root / "output"
            temperature_root.mkdir()
            np.save(temperature_root / "frame.npy", np.zeros((4, 7), dtype=np.float32), allow_pickle=False)
            _write_model(input_model, "1 SIMPLE_RADIAL 7 5 4 3 2 0.08")
            _write_model(output_model, "1 PINHOLE 7 5 4 4 3 2")
            with self.assertRaisesRegex(ValueError, "shape must match input camera"):
                undistort_temperature.remap_temperature_tree(
                    temperature_root,
                    input_model,
                    output_model,
                    output_root,
                )
            self.assertFalse(output_root.exists())

    def test_unsupported_output_distortion_fails_closed(self) -> None:
        input_camera = undistort_temperature.Camera(1, "SIMPLE_RADIAL", 4, 3, (2.0, 1.5, 1.0, 0.1))
        output_camera = undistort_temperature.Camera(1, "SIMPLE_RADIAL", 4, 3, (2.0, 1.5, 1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "Unsupported output camera model"):
            undistort_temperature.build_remap(input_camera, output_camera)


if __name__ == "__main__":
    unittest.main()
