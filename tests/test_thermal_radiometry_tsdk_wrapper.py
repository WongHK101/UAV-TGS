import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.thermal_radiometry import audit_rjpeg
from tools.thermal_radiometry import decode_temperature
from tools.thermal_radiometry import dji_irp_adapter


class RjpegAuditRecordTests(unittest.TestCase):
    def test_selected_dji_metadata_and_pair_delta_are_traceable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "thermal.jpg"
            rgb = Path(temp_dir) / "rgb.jpg"
            source.write_bytes(b"read-only-rjpeg-fixture")
            rgb.write_bytes(b"rgb-fixture")
            thermal_metadata = {
                "EXIF:Model": "ZH20T/H30T fixture",
                "EXIF:ImageWidth": 1280,
                "EXIF:ImageHeight": 1024,
                "EXIF:DateTimeOriginal": "2026:07:15 12:00:00.125+08:00",
                "XMP-drone-dji:RelativeAltitude": 42.5,
                "XMP-drone-dji:GimbalPitchDegree": -90.0,
                "MakerNotes:LRFStatus": "Normal",
                "MakerNotes:LRFTargetDistance": 38.25,
                "MakerNotes:ThermalPalette": "WhiteHot",
                "APP4:ThermalData": "Binary data 2621440 bytes",
                "APP4:ThermalCalibration": "Binary data 512 bytes",
            }
            rgb_metadata = {"EXIF:DateTimeOriginal": "2026:07:15 12:00:00.100+08:00"}
            record = audit_rjpeg.build_audit_record(
                {
                    "thermal_path": str(source),
                    "rgb_path": str(rgb),
                    "scene": "InternalRoad",
                    "frame_id": "000001",
                    "pair_id": "InternalRoad-000001",
                },
                thermal_metadata,
                rgb_metadata,
            )
            self.assertEqual(record["schema_version"], "uav-tgs.rjpeg-audit.v1")
            self.assertEqual(record["native_palette"], "WhiteHot")
            self.assertEqual(record["lrf_distance_m"], 38.25)
            self.assertTrue(record["lrf_distance_valid"])
            self.assertEqual(record["image_width"], 1280)
            self.assertTrue(record["rjpeg_detected"])
            self.assertAlmostEqual(record["rgb_thermal_timestamp_delta_ms"], 25.0)
            self.assertEqual(record["metadata_sources"]["lrf_distance_m"], "MakerNotes:LRFTargetDistance")


class TsdkWrapperTests(unittest.TestCase):
    @staticmethod
    def _write_fake_adapter(path: Path) -> None:
        path.write_text(
            "import numpy as np\n"
            "def decode_rjpeg(*, input_path, output_path, tsdk_root, parameters):\n"
            "    assert input_path.is_file()\n"
            "    assert tsdk_root.is_dir()\n"
            "    assert parameters['emissivity']['value'] == 0.95\n"
            "    np.save(output_path, np.full((3, 4), 21.5, dtype=np.float32))\n"
            "    return {'decoder': 'fixture', 'parameters_used': parameters}\n",
            encoding="utf-8",
        )

    def test_missing_tsdk_root_fails_clearly(self):
        with self.assertRaisesRegex(
            decode_temperature.DecodeConfigurationError, "--tsdk-root PATH or set DJI_TSDK_ROOT"
        ):
            decode_temperature.resolve_tsdk_root(None, {})

    def test_official_utility_is_the_default_adapter_and_layout_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            if __import__("os").name == "nt":
                executable = root / "utility" / "bin" / "windows" / "release_x64" / "dji_irp.exe"
            else:
                executable = root / "utility" / "bin" / "linux" / "release_x64" / "dji_irp"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"fixture")
            self.assertEqual(dji_irp_adapter.resolve_dji_irp(root), executable.resolve())
            args = decode_temperature._build_parser().parse_args(
                ["frame.jpg", "--output-dir", str(root / "out")]
            )
            self.assertEqual(args.adapter, "builtin:dji_irp")

    def test_official_output_parser_records_dynamic_camera_ranges(self):
        text = """
        image  width : 1280
        image height : 1024
        distance: [1,300]
        humidity: [1,100]
        emissivity: [0.1,1]
        ambientTemp: [-40,80]
        reflection: [-40,100]
        """
        self.assertEqual(dji_irp_adapter._parse_resolution(text), (1280, 1024))
        ranges = dji_irp_adapter._parse_parameter_ranges(text)
        self.assertEqual(ranges["distance_m"], {"min": 1.0, "max": 300.0})
        self.assertEqual(ranges["ambient_c"], {"min": -40.0, "max": 80.0})

    def test_probe_writes_validated_float32_and_manifest_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            sdk_dir = root / "sdk"
            output_dir = root / "derived"
            raw_dir.mkdir()
            sdk_dir.mkdir()
            source = raw_dir / "frame_0001.jpg"
            source.write_bytes(b"fake-rjpeg-kept-byte-identical")
            source_before = source.read_bytes()
            adapter = root / "fake_adapter.py"
            self._write_fake_adapter(adapter)

            args = decode_temperature._build_parser().parse_args(
                [
                    str(source),
                    "--output-dir",
                    str(output_dir),
                    "--tsdk-root",
                    str(sdk_dir),
                    "--adapter",
                    f"{adapter}:decode_rjpeg",
                    "--scene",
                    "InternalRoad",
                    "--probe",
                    "--distance-m",
                    "38.25",
                    "--distance-m-source",
                    "per_frame_lrf",
                    "--humidity-percent",
                    "70",
                    "--humidity-percent-source",
                    "scene_embedded_median",
                    "--ambient-c",
                    "25",
                    "--ambient-c-source",
                    "scene_embedded_median",
                    "--reflected-c",
                    "23",
                    "--reflected-c-source",
                    "scene_embedded_median",
                    "--expected-width",
                    "4",
                    "--expected-height",
                    "3",
                ]
            )
            summary = decode_temperature.run(args)
            self.assertEqual(summary["success_count"], 1)
            self.assertEqual(source.read_bytes(), source_before)
            temperature_path = output_dir / "temperature_c" / "InternalRoad" / "frame_0001.npy"
            array = np.load(temperature_path, allow_pickle=False)
            self.assertEqual(array.dtype, np.dtype("float32"))
            self.assertEqual(array.shape, (3, 4))
            manifest_path = output_dir / "manifests" / "decode_manifest.jsonl"
            records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["success"])
            self.assertEqual(records[0]["temperature_npy"], str(temperature_path.resolve()))
            self.assertEqual(records[0]["parameters"]["emissivity"]["source"], "benchmark_assumption")
            self.assertEqual(len(records[0]["output_sha256"]), 64)

    def test_incomplete_parameters_are_rejected_before_adapter(self):
        record = {"source_path": "frame.jpg", "decode_parameters": {"distance_m": 5}}
        globals_ = {name: None for name in decode_temperature.PARAMETER_NAMES}
        globals_["emissivity"] = {"value": 0.95, "source": "benchmark_assumption"}
        with self.assertRaisesRegex(decode_temperature.DecodeConfigurationError, "humidity_percent"):
            decode_temperature.resolve_parameters(record, globals_)


if __name__ == "__main__":
    unittest.main()
