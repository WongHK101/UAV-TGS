from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.thermal_radiometry import build_radiometry_protocol
from tools.thermal_radiometry import decode_temperature


class BuildRadiometryProtocolTests(unittest.TestCase):
    @staticmethod
    def _record(
        root: Path,
        index: int,
        *,
        lrf: float | None,
        lrf_valid: bool,
        altitude: float | None = 30.0,
        pitch: float | None = -60.0,
        timestamp_offset_s: int | None = None,
    ) -> dict[str, object]:
        source = root / "raw" / f"{index:04d}.JPG"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(f"fixture-{index}".encode("ascii"))
        stat = source.stat()
        start = datetime(2026, 7, 15, tzinfo=timezone.utc)
        timestamp = (
            None
            if timestamp_offset_s is None
            else (start + timedelta(seconds=timestamp_offset_s)).isoformat()
        )
        return {
            "schema_version": "uav-tgs.rjpeg-audit.v1",
            "scene": "Fixture",
            "frame_id": f"{index:04d}",
            "pair_id": f"{index:04d}",
            "source_path": str(source),
            "source_size_bytes": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "rjpeg_detected": True,
            "capture_time": timestamp,
            "gimbal_pitch_deg": pitch,
            "gimbal_yaw_deg": 0.0 if pitch is not None else None,
            "relative_altitude_m": altitude,
            "lrf_distance_m": lrf,
            "lrf_status": "Normal" if lrf_valid else "TooClose",
            "lrf_distance_valid": lrf_valid,
            "metadata_sources": {
                "lrf_distance_m": "XMP-drone-dji:LRFTargetDistance",
                "relative_altitude_m": "XMP-drone-dji:RelativeAltitude",
                "gimbal_pitch_deg": "XMP-drone-dji:GimbalPitchDegree",
            },
        }

    @staticmethod
    def _constants() -> dict[str, dict[str, object]]:
        return {
            "humidity_percent": {"value": 70.0, "source": "benchmark_assumption"},
            "ambient_c": {"value": 25.0, "source": "benchmark_assumption"},
            "reflected_c": {"value": 23.0, "source": "benchmark_assumption"},
            "emissivity": {"value": 0.95, "source": "benchmark_assumption"},
        }

    def test_strip_lrf_robust_median_has_priority_and_is_decode_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records = [
                self._record(root, 1, lrf=10.0, lrf_valid=True, timestamp_offset_s=0),
                self._record(root, 2, lrf=11.0, lrf_valid=True, timestamp_offset_s=1),
                self._record(root, 3, lrf=12.0, lrf_valid=True, timestamp_offset_s=2),
                self._record(root, 4, lrf=100.0, lrf_valid=True, timestamp_offset_s=3),
                self._record(root, 5, lrf=0.0, lrf_valid=False, timestamp_offset_s=4),
            ]
            outputs, summary = build_radiometry_protocol.build_protocol(
                records,
                scene="Fixture",
                source_manifest="audit.jsonl",
                source_manifest_sha256="a" * 64,
                scene_constants=self._constants(),
                scene_distance_m=5.0,
                scene_distance_provenance="benchmark_assumption",
            )
            self.assertEqual(len(outputs), 5)
            self.assertTrue(all(record["used_distance_m"] == 11.0 for record in outputs))
            self.assertTrue(
                all(
                    record["used_distance_source"] == "strip_valid_lrf_robust_median"
                    for record in outputs
                )
            )
            self.assertEqual(summary["strips"][0]["valid_lrf_count"], 4)
            self.assertEqual(summary["strips"][0]["robust_lrf_inlier_count"], 3)
            self.assertEqual(summary["strips"][0]["robust_lrf_outlier_count"], 1)
            self.assertTrue(outputs[0]["raw_lrf_robust_inlier"])
            self.assertFalse(outputs[0]["raw_lrf_robust_outlier"])
            self.assertFalse(outputs[3]["raw_lrf_robust_inlier"])
            self.assertTrue(outputs[3]["raw_lrf_robust_outlier"])
            self.assertIn("frame_lrf_robust_outlier", outputs[3]["distance_fallback_reason"])
            self.assertEqual(outputs[-1]["raw_lrf_distance_m"], 0.0)
            self.assertIsNone(outputs[-1]["raw_lrf_robust_inlier"])
            self.assertIn("frame_lrf_invalid", outputs[-1]["distance_fallback_reason"])
            parameters = decode_temperature.resolve_parameters(outputs[0], {})
            self.assertEqual(parameters["distance_m"]["value"], 11.0)
            self.assertEqual(parameters["emissivity"]["source"], "benchmark_assumption")
            self.assertEqual(
                outputs[0]["metadata"]["radiometry_protocol"]["used_distance_m"], 11.0
            )

    def test_geometry_then_scene_assumption_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            geometry_records = [
                self._record(root, 1, lrf=0.0, lrf_valid=False, altitude=30.0, timestamp_offset_s=0),
                self._record(root, 2, lrf=None, lrf_valid=False, altitude=36.0, timestamp_offset_s=1),
            ]
            outputs, _ = build_radiometry_protocol.build_protocol(
                geometry_records,
                scene="Fixture",
                source_manifest="audit.jsonl",
                source_manifest_sha256="b" * 64,
                scene_constants=self._constants(),
                scene_distance_m=5.0,
                scene_distance_provenance="benchmark_assumption",
            )
            expected = ((30.0 / (3.0 ** 0.5 / 2.0)) + (36.0 / (3.0 ** 0.5 / 2.0))) / 2.0
            self.assertAlmostEqual(outputs[0]["used_distance_m"], expected)
            self.assertEqual(
                outputs[0]["used_distance_source"],
                "strip_relative_altitude_gimbal_geometry_estimate",
            )
            self.assertEqual(outputs[0]["distance_fallback_reason"], "strip_has_no_valid_lrf")

            fallback_record = self._record(
                root, 3, lrf=9.0, lrf_valid=True, altitude=None, pitch=None, timestamp_offset_s=None
            )
            fallback, summary = build_radiometry_protocol.build_protocol(
                [fallback_record],
                scene="Fixture",
                source_manifest="audit.jsonl",
                source_manifest_sha256="c" * 64,
                scene_constants=self._constants(),
                scene_distance_m=7.5,
                scene_distance_provenance="benchmark_assumption:legacy_scene_protocol",
            )
            self.assertEqual(fallback[0]["used_distance_m"], 7.5)
            self.assertEqual(fallback[0]["used_distance_source"], "scene_benchmark_assumption")
            self.assertIn("view_flight_strip_unavailable", fallback[0]["distance_fallback_reason"])
            self.assertIn("lrf_pool_rejected", fallback[0]["distance_fallback_reason"])
            self.assertEqual(
                fallback[0]["metadata"]["radiometry_protocol"]["scene_assumption_provenance"],
                "benchmark_assumption:legacy_scene_protocol",
            )
            self.assertEqual(summary["strip_assignment_mode"], "filename_order_fallback")

    def test_embedded_label_without_audited_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            record = self._record(
                Path(temp_dir), 1, lrf=5.0, lrf_valid=True, timestamp_offset_s=0
            )
            constants = self._constants()
            constants["humidity_percent"] = {
                "value": 70.0,
                "source": "scene_embedded_median",
            }
            with self.assertRaisesRegex(
                build_radiometry_protocol.ProtocolConfigurationError,
                "no traceable embedded_humidity_percent",
            ):
                build_radiometry_protocol.build_protocol(
                    [record],
                    scene="Fixture",
                    source_manifest="audit.jsonl",
                    source_manifest_sha256="d" * 64,
                    scene_constants=constants,
                    scene_distance_m=5.0,
                    scene_distance_provenance="benchmark_assumption",
                )

    def test_cli_is_deterministic_preserves_raw_source_and_blocks_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = self._record(root, 1, lrf=8.0, lrf_valid=True, timestamp_offset_s=0)
            source = Path(str(record["source_path"]))
            source_before = source.read_bytes()
            audit = root / "audit.jsonl"
            audit.write_text(json.dumps(record) + "\n", encoding="utf-8")
            common = [
                "--audit-manifest",
                str(audit),
                "--scene-distance-m",
                "5",
                "--humidity-percent",
                "70",
                "--humidity-percent-source",
                "benchmark_assumption",
                "--ambient-c",
                "25",
                "--ambient-c-source",
                "benchmark_assumption",
                "--reflected-c",
                "23",
                "--reflected-c-source",
                "benchmark_assumption",
            ]
            output1 = root / "derived1" / "protocol.jsonl"
            output2 = root / "derived2" / "protocol.jsonl"
            args1 = build_radiometry_protocol._parser().parse_args(
                [*common, "--output", str(output1)]
            )
            args2 = build_radiometry_protocol._parser().parse_args(
                [*common, "--output", str(output2)]
            )
            first = build_radiometry_protocol.run(args1)
            second = build_radiometry_protocol.run(args2)
            self.assertEqual(first["protocol_hash"], second["protocol_hash"])
            self.assertEqual(output1.read_bytes(), output2.read_bytes())
            self.assertEqual(source.read_bytes(), source_before)

            raw_output_args = build_radiometry_protocol._parser().parse_args(
                [*common, "--output", str(source.parent / "protocol.jsonl")]
            )
            with self.assertRaisesRegex(
                build_radiometry_protocol.ProtocolConfigurationError,
                "inside raw source directory",
            ):
                build_radiometry_protocol.run(raw_output_args)


if __name__ == "__main__":
    unittest.main()
