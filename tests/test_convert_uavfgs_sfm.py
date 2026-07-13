from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sqlite3
import struct
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERT_PATH = REPO_ROOT / "convert_uavfgs.py"
PIPELINE_PATH = REPO_ROOT / "run_uavfgs_pipeline.py"
LEGACY_STRICT_MATERIALIZER_PATH = (
    REPO_ROOT / "tools" / "geometric_repeatability" / "materialize_strict_pose_controlled_dataset.py"
)


def load_convert_module():
    spec = importlib.util.spec_from_file_location("convert_uavfgs_under_test", CONVERT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CONVERT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("run_uavfgs_pipeline_under_test", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_text_model(model_dir: Path, names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 16 16 10 10 8 8\n",
        encoding="utf-8",
    )
    image_lines = ["# Image list"]
    for image_id, name in enumerate(names, 1):
        image_lines.extend([f"{image_id} 1 0 0 0 0 0 0 1 {name}", ""])
    (model_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (model_dir / "points3D.txt").write_text(
        "# Point list\n1 0 0 1 255 255 255 0.1 1 0\n",
        encoding="utf-8",
    )


class ConvertUavFgsSfmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_convert_module()
        cls.pipeline = load_pipeline_module()

    def test_legacy_strict_materializer_is_fail_closed_before_colmap(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "legacy_strict_materializer_under_test",
            LEGACY_STRICT_MATERIALIZER_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import {LEGACY_STRICT_MATERIALIZER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(module.subprocess, "run") as subprocess_run:
            with self.assertRaisesRegex(
                RuntimeError,
                "legacy strict-pose materializer is disabled",
            ):
                module.main()
        subprocess_run.assert_not_called()

    def test_colmap4_pose_prior_schema(self) -> None:
        raw_building = self.module._parse_exiftool_gps_records(
            [
                {
                    "SourceFile": "/readonly/Building/rgb/0001.JPG",
                    "EXIF:GPSLatitude": 10.8005365833333,
                    "EXIF:GPSLongitude": 154.200318638889,
                    "EXIF:GPSAltitude": 7424.9375,
                    # Explicit EXIF tags take priority over grouped composite
                    # aliases, and values are not silently corrected.
                    "Composite:GPSLatitude": 1.0,
                    "Composite:GPSLongitude": 2.0,
                    "Composite:GPSAltitude": 3.0,
                }
            ]
        )
        self.assertEqual(
            raw_building["0001.JPG"],
            (10.8005365833333, 154.200318638889, 7424.9375),
        )
        gps = {
            "0001.JPG": raw_building["0001.JPG"],
            "0002.JPG": (30.0, 120.00001, 10.5),
            "0003.JPG": (30.00001, 120.0, 11.0),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)")
            con.executemany(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                [(1, "0001.JPG", 7), (2, "0002.JPG", 7), (3, "0003.JPG", 7)],
            )
            con.execute(
                """
                CREATE TABLE pose_priors(
                    pose_prior_id INTEGER PRIMARY KEY NOT NULL,
                    corr_data_id INTEGER NOT NULL,
                    corr_sensor_id INTEGER NOT NULL,
                    corr_sensor_type INTEGER NOT NULL,
                    position BLOB,
                    position_covariance BLOB,
                    gravity BLOB,
                    coordinate_system INTEGER NOT NULL)
                """
            )
            nan3 = np.full(3, np.nan, dtype=np.float64).tobytes()
            nan9 = np.full((3, 3), np.nan, dtype=np.float64).tobytes()
            con.execute(
                "INSERT INTO pose_priors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (99, 99, 7, 0, np.asarray([0.0, 0.0, 0.0], dtype=np.float64).tobytes(), nan9, nan3, 0),
            )
            con.commit()
            con.close()

            original = self.module.exiftool_extract_gps
            self.module.exiftool_extract_gps = lambda *_args, **_kwargs: gps
            try:
                self.module.populate_pose_priors_from_exif(
                    database,
                    root,
                    exiftool_exe="unused",
                    wgs84_code=0,
                    prior_position_std_m=1.0,
                )
            finally:
                self.module.exiftool_extract_gps = original

            con = sqlite3.connect(database)
            rows = con.execute("SELECT * FROM pose_priors ORDER BY pose_prior_id").fetchall()
            con.close()
            self.assertEqual(len(rows), 3)
            for image_id, row in enumerate(rows, 1):
                self.assertEqual(row[:4], (image_id, image_id, 7, 0))
                self.assertEqual(struct.unpack("3d", row[4]), gps[f"{image_id:04d}.JPG"])
                self.assertTrue(all(math.isfinite(value) for value in struct.unpack("9d", row[5])))
                self.assertTrue(all(math.isnan(value) for value in struct.unpack("3d", row[6])))
                self.assertEqual(row[7], 0)

    def test_swap_latlon_returns_the_exact_values_written_to_database(self) -> None:
        gps = {"0001.JPG": (30.0, 120.0, 10.0)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.execute("INSERT INTO images(image_id, name) VALUES (1, '0001.JPG')")
            con.commit()
            con.close()
            with mock.patch.object(self.module, "exiftool_extract_gps", return_value=gps):
                returned = self.module.populate_pose_priors_from_exif(
                    database,
                    root,
                    exiftool_exe="unused",
                    wgs84_code=0,
                    prior_position_std_m=1.0,
                    swap_latlon=True,
                )
            con = sqlite3.connect(database)
            blob = con.execute("SELECT position FROM pose_priors WHERE image_id=1").fetchone()[0]
            con.close()
            expected = (120.0, 30.0, 10.0)
            self.assertEqual(struct.unpack("3d", blob), expected)
            self.assertEqual(returned["0001.JPG"], expected)

    def test_global_mapper_command_is_gpu_locked(self) -> None:
        cmd = self.module.build_global_mapper_command(
            colmap_exe="/opt/colmap-4.1.0-cuda/bin/colmap",
            database_path=Path("/work/database.db"),
            image_path=Path("/work/input"),
            output_path=Path("/work/distorted/sparse"),
            gpu_index=0,
            random_seed=0,
            extra_args="--GlobalMapper.min_num_matches=20",
        )
        self.assertEqual(cmd[1], "global_mapper")
        self.assertNotIn("mapper", cmd[1:])
        for option, value in (
            ("--GlobalMapper.gp_use_gpu", "1"),
            ("--GlobalMapper.gp_gpu_index", "0"),
            ("--GlobalMapper.ba_ceres_use_gpu", "1"),
            ("--GlobalMapper.ba_ceres_gpu_index", "0"),
            ("--GlobalMapper.random_seed", "0"),
        ):
            index = cmd.index(option)
            self.assertEqual(cmd[index + 1], value)

        with self.assertRaises(ValueError):
            self.module.build_global_mapper_command(
                "colmap",
                Path("db"),
                Path("images"),
                Path("sparse"),
                0,
                0,
                "--GlobalMapper.gp_use_gpu=0",
            )
        with self.assertRaisesRegex(ValueError, "protocol-locked option --output_path"):
            self.module.build_global_mapper_command(
                "colmap",
                Path("db"),
                Path("images"),
                Path("sparse"),
                0,
                0,
                "--output_path=elsewhere",
            )
        with self.assertRaisesRegex(ValueError, "protocol-locked option --project_path"):
            self.module.build_global_mapper_command(
                "colmap",
                Path("db"),
                Path("images"),
                Path("sparse"),
                0,
                0,
                "--project_path=override.ini",
            )
        for abbreviated_override in ("--project_pat=override.ini", "--GlobalMapper.gp_use_g=0"):
            with self.subTest(abbreviated_override=abbreviated_override):
                with self.assertRaisesRegex(ValueError, "protocol-locked option"):
                    self.module.build_global_mapper_command(
                        "colmap",
                        Path("db"),
                        Path("images"),
                        Path("sparse"),
                        0,
                        0,
                        abbreviated_override,
                    )

        matcher_cmd = self.module.build_matcher_command(
            "colmap",
            "exhaustive",
            Path("db"),
            0,
            "--FeatureMatching.guided_matching=1",
        )
        self.assertEqual(matcher_cmd[1], "exhaustive_matcher")
        self.assertEqual(matcher_cmd[matcher_cmd.index("--FeatureMatching.use_gpu") + 1], "1")
        with self.assertRaisesRegex(ValueError, "protocol-locked option --project_path"):
            self.module.build_matcher_command(
                "colmap",
                "exhaustive",
                Path("db"),
                0,
                "--project_path override.ini",
            )
        with self.assertRaisesRegex(ValueError, "protocol-locked option --project_path"):
            self.module.build_matcher_command(
                "colmap",
                "exhaustive",
                Path("db"),
                0,
                "--project_pat override.ini",
            )

        align_cmd, max_error, min_common = self.module.build_model_aligner_command(
            "colmap",
            Path("source"),
            Path("aligned"),
            Path("isolated.db"),
            Path("transform.txt"),
            "--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30",
        )
        seed_index = align_cmd.index("--default_random_seed")
        self.assertEqual(align_cmd[seed_index + 1], "0")
        self.assertEqual(max_error, 30.0)
        self.assertEqual(min_common, 3)
        with self.assertRaisesRegex(ValueError, "protocol-locked option --default_random_seed"):
            self.module.build_model_aligner_command(
                "colmap",
                Path("source"),
                Path("aligned"),
                Path("isolated.db"),
                Path("transform.txt"),
                "--ref_is_gps=1 --alignment_type=enu --default_random_seed=9",
            )
        with self.assertRaisesRegex(ValueError, "protocol-locked option --project_path"):
            self.module.build_model_aligner_command(
                "colmap",
                Path("source"),
                Path("aligned"),
                Path("isolated.db"),
                Path("transform.txt"),
                "--project_path=override.ini",
            )
        for abbreviated_override in ("--project_pat=override.ini", "--default_random_s=9"):
            with self.subTest(abbreviated_override=abbreviated_override):
                with self.assertRaisesRegex(ValueError, "protocol-locked option"):
                    self.module.build_model_aligner_command(
                        "colmap",
                        Path("source"),
                        Path("aligned"),
                        Path("isolated.db"),
                        Path("transform.txt"),
                        abbreviated_override,
                    )
        with self.assertRaisesRegex(ValueError, "spell controlled option --alignment_max_error in full"):
            self.module.build_model_aligner_command(
                "colmap",
                Path("source"),
                Path("aligned"),
                Path("isolated.db"),
                Path("transform.txt"),
                "--alignment_max_e=30",
            )
        for invalid_max_error in ("nan", "inf", "-inf"):
            with self.subTest(invalid_max_error=invalid_max_error):
                with self.assertRaisesRegex(ValueError, "finite alignment_max_error"):
                    self.module.build_model_aligner_command(
                        "colmap",
                        Path("source"),
                        Path("aligned"),
                        Path("isolated.db"),
                        Path("transform.txt"),
                        f"--ref_is_gps=1 --alignment_type=enu --alignment_max_error={invalid_max_error}",
                    )

    def test_colmap_runtime_check_rejects_cpu_or_wrong_version(self) -> None:
        cuda_result = SimpleNamespace(
            returncode=0,
            stdout="COLMAP 4.1.0 (Commit abc with CUDA)\nUsage:\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "colmap"
            executable.write_bytes(b"fake-colmap")
            expected_sha = self.module._sha256_file(executable)
            with mock.patch.object(self.module.subprocess, "run", return_value=cuda_result):
                runtime = self.module.verify_colmap_runtime(
                    str(executable),
                    "4.1.0",
                    True,
                    required_sha256=expected_sha,
                )
            self.assertTrue(runtime["reported_with_cuda"])
            self.assertEqual(runtime["executable_sha256"], expected_sha)

            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                self.module.verify_colmap_runtime(
                    str(executable),
                    "4.1.0",
                    True,
                    required_sha256="0" * 64,
                )

        cpu_result = SimpleNamespace(
            returncode=0,
            stdout="COLMAP 4.1.0 (Commit abc without CUDA)\nUsage:\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "colmap"
            executable.write_bytes(b"fake-colmap")
            with mock.patch.object(self.module.subprocess, "run", return_value=cpu_result):
                with self.assertRaisesRegex(RuntimeError, "CUDA-enabled COLMAP is required"):
                    self.module.verify_colmap_runtime(str(executable), "4.1.0", True)

        old_result = SimpleNamespace(
            returncode=0,
            stdout="COLMAP 3.7\nUsage:\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "colmap"
            executable.write_bytes(b"fake-colmap")
            with mock.patch.object(self.module.subprocess, "run", return_value=old_result):
                with self.assertRaisesRegex(RuntimeError, "Expected COLMAP 4.1.0"):
                    self.module.verify_colmap_runtime(str(executable), "4.1.0", False)

    def test_linux_global_mapper_runtime_requires_resolved_cudss(self) -> None:
        help_result = SimpleNamespace(
            returncode=0,
            stdout="COLMAP 4.1.0 (Commit abc with CUDA)\n",
            stderr="",
        )
        good_ldd = SimpleNamespace(
            returncode=0,
            stdout="libcudss.so.0 => /opt/cudss/lib/libcudss.so.0\nlibcudart.so.12 => /opt/cuda/libcudart.so.12\n",
            stderr="",
        )
        missing_ldd = SimpleNamespace(
            returncode=0,
            stdout="libcudss.so.0 => not found\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "colmap"
            executable.write_bytes(b"fake-colmap")
            with mock.patch.object(self.module.sys, "platform", "linux"), mock.patch.object(
                self.module.subprocess, "run", side_effect=[help_result, good_ldd]
            ):
                runtime = self.module.verify_colmap_runtime(
                    str(executable), "4.1.0", True, require_cudss=True
                )
            self.assertTrue(runtime["ldd_checked"])
            self.assertTrue(runtime["ldd_has_cudss"])

            with mock.patch.object(self.module.sys, "platform", "linux"), mock.patch.object(
                self.module.subprocess, "run", side_effect=[help_result, missing_ldd]
            ):
                with self.assertRaisesRegex(RuntimeError, "unresolved shared libraries"):
                    self.module.verify_colmap_runtime(
                        str(executable), "4.1.0", True, require_cudss=True
                    )

    def test_zero_exit_internal_solver_failure_is_rejected(self) -> None:
        representative_failures = [
            "Ceres Solver Report: Iterations: 5, Termination: FAILURE",
            "Ceres was compiled without CUDA support",
            "Falling back to CPU-based sparse solvers",
            "cuDSS initialization failed; falling back to Eigen CPU",
            "Using CPU-based linear solver",
        ]
        for diagnostic in representative_failures:
            self.assertTrue(
                any(
                    re.search(pattern, diagnostic, re.IGNORECASE)
                    for pattern in self.module.GLOBAL_MAPPER_FORBIDDEN_OUTPUT_PATTERNS
                ),
                diagnostic,
            )
        with self.assertRaisesRegex(RuntimeError, "forbidden solver/GPU fallback diagnostic"):
            self.module.run_cmd(
                [
                    sys.executable,
                    "-c",
                    "print('Ceres Solver Report: Iterations: 5, Termination: FAILURE')",
                ],
                forbidden_output_patterns=[r"Termination:\s*FAILURE"],
            )

    def test_forbidden_gpu_fallback_terminates_process_immediately(self) -> None:
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "was terminated"):
            self.module.run_cmd(
                [
                    sys.executable,
                    "-c",
                    "import time; print('Falling back to CPU-based sparse solvers', flush=True); time.sleep(10)",
                ],
                forbidden_output_patterns=[r"Falling back to CPU-based"],
            )
        self.assertLess(time.monotonic() - started, 5.0)

    def test_empty_global_model_is_rejected_even_if_colmap_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "images.bin").write_bytes(struct.pack("<Q", 0))
            (model / "points3D.bin").write_bytes(struct.pack("<Q", 0))
            with self.assertRaisesRegex(RuntimeError, "No incremental/fallback mapper"):
                self.module.validate_sparse_model(model, min_registered_images=5, mapper_mode="global")

    def test_enu_alignment_audit_matches_colmap_coordinate_convention(self) -> None:
        gps = {
            "0001.JPG": (30.0, 120.0, 10.0),
            "0002.JPG": (30.0, 120.0001, 10.5),
            "0003.JPG": (30.0001, 120.0, 11.0),
            "0004.JPG": (30.0001, 120.0001, 11.5),
        }
        names = list(gps)
        enu = self.module.wgs84_to_enu(
            np.asarray([gps[name] for name in names], dtype=np.float64),
            np.asarray(gps[names[0]], dtype=np.float64),
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.execute(
                "CREATE TABLE pose_priors("
                "image_id INTEGER PRIMARY KEY, position BLOB, "
                "coordinate_system INTEGER NOT NULL, position_covariance BLOB)"
            )
            con.executemany(
                "INSERT INTO images(image_id, name) VALUES (?, ?)",
                [(idx, name) for idx, name in enumerate(names, 1)],
            )
            con.executemany(
                "INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                "VALUES (?, ?, 0, NULL)",
                [
                    (idx, np.asarray(gps[name], dtype=np.float64).tobytes())
                    for idx, name in enumerate(names, 1)
                ],
            )
            con.commit()
            con.close()

            model = root / "sparse_aligned"
            model.mkdir()
            lines = ["# synthetic COLMAP text model"]
            for idx, (name, center) in enumerate(zip(names, enu), 1):
                tvec = -center
                lines.append(
                    f"{idx} 1 0 0 0 {tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} 1 {name}"
                )
                lines.append("")
            (model / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            transform = root / "model_alignment_transform.txt"
            transform.write_text("1 1 0 0 0 0 0 0\n", encoding="utf-8")
            report = root / "model_alignment_audit.json"

            payload = self.module.write_enu_alignment_audit(
                database_path=database,
                aligned_model=model,
                gps_by_basename=gps,
                transform_path=transform,
                report_path=report,
                alignment_max_error=30.0,
                min_common_images=3,
                expected_image_names=names,
            )
            self.assertEqual(payload["status"], "verified")
            self.assertEqual(payload["enu_origin_image"], names[0])
            self.assertEqual(payload["registered_common_images"], 4)
            self.assertLess(payload["error_m"]["max"], 1e-9)
            self.assertTrue(all(value > 0 for value in payload["reference_enu_extent_xyz_m"]))
            self.assertTrue(np.allclose(payload["aligned_camera_extent_xyz_m"], payload["reference_enu_extent_xyz_m"]))
            self.assertTrue(payload["geometry_scale_check"]["verified"])
            self.assertAlmostEqual(
                payload["geometry_scale_check"]["aligned_to_reference_ratio"],
                1.0,
                places=12,
            )
            self.assertTrue(
                self.module._geometry_scale_check_payload_verified(
                    payload["geometry_scale_check"]
                )
            )
            self.assertEqual(payload["formal_expected_images"], 4)
            self.assertEqual(payload["within_threshold_images"], 4)
            self.assertEqual(payload["coordinate_scope"], "relative_local_enu_only")
            self.assertFalse(payload["absolute_geolocation_validated"])
            self.assertEqual(payload["model_aligner_default_random_seed"], 0)
            self.assertTrue(report.is_file())

    def test_enu_alignment_audit_writes_failed_report_for_collapsed_model(self) -> None:
        gps = {
            "0001.JPG": (30.0, 120.0, 10.0),
            "0002.JPG": (30.0, 120.0001, 10.0),
            "0003.JPG": (30.0001, 120.0, 10.0),
            "0004.JPG": (30.0001, 120.0001, 10.0),
        }
        names = list(gps)
        refs = self.module.wgs84_to_enu(
            np.asarray([gps[name] for name in names], dtype=np.float64),
            np.asarray(gps[names[0]], dtype=np.float64),
        )
        collapsed = refs * 0.05

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.execute(
                "CREATE TABLE pose_priors("
                "image_id INTEGER PRIMARY KEY, position BLOB, "
                "coordinate_system INTEGER NOT NULL, position_covariance BLOB)"
            )
            con.executemany(
                "INSERT INTO images(image_id, name) VALUES (?, ?)",
                [(idx, name) for idx, name in enumerate(names, 1)],
            )
            con.executemany(
                "INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                "VALUES (?, ?, 0, NULL)",
                [
                    (idx, np.asarray(gps[name], dtype=np.float64).tobytes())
                    for idx, name in enumerate(names, 1)
                ],
            )
            con.commit()
            con.close()

            model = root / "sparse_aligned"
            model.mkdir()
            lines = ["# collapsed synthetic model"]
            for idx, (name, center) in enumerate(zip(names, collapsed), 1):
                tvec = -center
                lines.extend(
                    [f"{idx} 1 0 0 0 {tvec[0]:.17g} {tvec[1]:.17g} {tvec[2]:.17g} 1 {name}", ""]
                )
            (model / "images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            transform = root / "transform.txt"
            transform.write_text("1 1 0 0 0 0 0 0\n", encoding="utf-8")
            report = root / "audit.json"

            with self.assertRaisesRegex(RuntimeError, "failed the ENU audit"):
                self.module.write_enu_alignment_audit(
                    database,
                    model,
                    gps,
                    transform,
                    report,
                    alignment_max_error=30.0,
                    min_common_images=3,
                )
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["within_threshold_images"], 4)
            self.assertLess(payload["error_m"]["max"], 30.0)
            self.assertFalse(payload["geometry_scale_check"]["verified"])
            self.assertAlmostEqual(
                payload["geometry_scale_check"]["aligned_to_reference_ratio"],
                0.05,
                places=12,
            )
            self.assertLess(
                payload["aligned_camera_extent_xyz_m"][0],
                payload["reference_enu_extent_xyz_m"][0],
            )

    def test_enu_alignment_audit_rejects_unobservable_reference_scale(self) -> None:
        names = ["0001.JPG", "0002.JPG", "0003.JPG"]
        gps = {name: (30.0, 120.0, 10.0) for name in names}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.execute(
                "CREATE TABLE pose_priors("
                "image_id INTEGER PRIMARY KEY, position BLOB, "
                "coordinate_system INTEGER NOT NULL, position_covariance BLOB)"
            )
            con.executemany(
                "INSERT INTO images(image_id, name) VALUES (?, ?)",
                [(idx, name) for idx, name in enumerate(names, 1)],
            )
            con.executemany(
                "INSERT INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                "VALUES (?, ?, 0, NULL)",
                [
                    (idx, np.asarray(gps[name], dtype=np.float64).tobytes())
                    for idx, name in enumerate(names, 1)
                ],
            )
            con.commit()
            con.close()

            model = root / "sparse_aligned"
            write_text_model(model, names)
            transform = root / "transform.txt"
            transform.write_text("1 1 0 0 0 0 0 0\n", encoding="utf-8")
            report = root / "audit.json"
            with self.assertRaisesRegex(RuntimeError, "centered_rms_scale_ratio=unobservable"):
                self.module.write_enu_alignment_audit(
                    database,
                    model,
                    gps,
                    transform,
                    report,
                    alignment_max_error=30.0,
                    min_common_images=3,
                )
            payload = json.loads(report.read_text(encoding="utf-8"))
            check = payload["geometry_scale_check"]
            self.assertFalse(check["reference_scale_observable"])
            self.assertIsNone(check["aligned_to_reference_ratio"])
            self.assertFalse(check["verified"])

    def test_prepare_input_is_an_exact_content_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "a.jpg").write_bytes(b"new-a")
            (source / "b.jpg").write_bytes(b"new-b")
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "a.jpg").write_bytes(b"old-a")
            (input_dir / "stale.jpg").write_bytes(b"stale")

            self.pipeline.prepare_input_dir(source, root, clean=False, link_mode="copy")
            self.assertEqual(sorted(path.name for path in input_dir.iterdir()), ["a.jpg", "b.jpg"])
            self.assertEqual((input_dir / "a.jpg").read_bytes(), b"new-a")

    def test_prepare_input_copy_does_not_modify_a_previous_hardlink_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_source = root / "previous_source"
            current_source = root / "current_source"
            input_dir = root / "input"
            previous_source.mkdir()
            current_source.mkdir()
            input_dir.mkdir()
            previous_image = previous_source / "a.jpg"
            current_image = current_source / "a.jpg"
            previous_image.write_bytes(b"previous-source-must-stay-unchanged")
            current_image.write_bytes(b"current-source")
            try:
                os.link(previous_image, input_dir / "a.jpg")
            except OSError as exc:
                self.skipTest(f"hardlinks are unavailable on this filesystem: {exc}")

            self.pipeline.prepare_input_dir(current_source, root, clean=False, link_mode="copy")
            self.assertEqual(previous_image.read_bytes(), b"previous-source-must-stay-unchanged")
            self.assertEqual((input_dir / "a.jpg").read_bytes(), b"current-source")

    def test_unverified_existing_outputs_require_explicit_replacement_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Refusing to replace existing"):
                self.module.verify_existing_output_ownership(
                    root,
                    allow_unverified=False,
                    include_resized_outputs=False,
                )
            self.module.verify_existing_output_ownership(
                root,
                allow_unverified=True,
                include_resized_outputs=False,
            )

    def test_verified_database_reuse_rejects_stale_inventory_or_sha(self) -> None:
        self.assertTrue(self.module.database_policy_contract("reset")["run_feature_extractor"])
        self.assertFalse(self.module.database_policy_contract("reuse_verified")["run_matcher"])
        adopted_contract = self.module.database_policy_contract("adopt_legacy")
        self.assertFalse(adopted_contract["run_feature_extractor"])
        self.assertFalse(adopted_contract["run_matcher"])
        self.assertFalse(adopted_contract["reuse_verified_eligible"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / "database.db"
            provenance_path = root / "database_provenance.json"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.executemany(
                "INSERT INTO images(image_id, name) VALUES (?, ?)",
                [(1, "a.jpg"), (2, "b.jpg")],
            )
            con.commit()
            con.close()
            core = {
                "input_inventory": {
                    "count": 2,
                    "names": ["a.jpg", "b.jpg"],
                    "entries_sha256": "inventory",
                    "entries": [],
                }
            }
            provenance = {
                "status": "complete",
                "core": core,
                "database_sha256": self.module._sha256_file(database),
                "reuse_verified_eligible": True,
            }
            self.module._atomic_write_json(provenance_path, provenance)
            loaded = self.module._validate_verified_database_reuse(
                database_path=database,
                provenance_path=provenance_path,
                expected_core=core,
            )
            self.assertEqual(loaded["database_sha256"], provenance["database_sha256"])
            self.assertFalse(Path(str(database) + "-wal").exists())
            self.assertFalse(Path(str(database) + "-shm").exists())

            isolated = root / "model_aligner_database.db"
            self.module._copy_closed_sqlite_database(database, isolated)
            pre_sha = self.module._finalize_private_sqlite_database(isolated)
            pre_semantics = self.module._alignment_database_semantic_summary(isolated)
            con = sqlite3.connect(isolated)
            con.execute("PRAGMA user_version=1")
            con.commit()
            con.close()
            post_sha = self.module._finalize_private_sqlite_database(isolated)
            post_semantics = self.module._alignment_database_semantic_summary(isolated)
            self.assertNotEqual(pre_sha, post_sha)
            self.assertEqual(pre_semantics, post_semantics)

            fake_wal = Path(str(database) + "-wal")
            fake_wal.write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "WAL/journal sidecars"):
                self.module._database_image_names(database, immutable=True)
            fake_wal.unlink()

            con = sqlite3.connect(database)
            con.execute("INSERT INTO images(image_id, name) VALUES (3, 'stale.jpg')")
            con.commit()
            con.close()
            provenance["database_sha256"] = self.module._sha256_file(database)
            self.module._atomic_write_json(provenance_path, provenance)
            with self.assertRaisesRegex(RuntimeError, "image inventory differs"):
                self.module._validate_verified_database_reuse(
                    database_path=database,
                    provenance_path=provenance_path,
                    expected_core=core,
                )

            provenance["reuse_verified_eligible"] = False
            self.module._atomic_write_json(provenance_path, provenance)
            with self.assertRaisesRegex(RuntimeError, "ineligible for reuse_verified"):
                self.module._validate_verified_database_reuse(
                    database_path=database,
                    provenance_path=provenance_path,
                    expected_core=core,
                )

    def test_completion_manifest_validates_final_images_model_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["a.jpg", "b.jpg", "c.jpg"]
            for directory in (root / "input", root / "images"):
                directory.mkdir(parents=True)
                for name in names:
                    (directory / name).write_bytes((directory.name + name).encode("ascii"))
            write_text_model(root / "sparse" / "0", names)
            (root / "stereo").mkdir()
            (root / "distorted").mkdir()
            database = root / "distorted" / "database.db"
            con = sqlite3.connect(database)
            con.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
            con.executemany(
                "INSERT INTO images(image_id, name) VALUES (?, ?)",
                [(index, name) for index, name in enumerate(names, 1)],
            )
            con.commit()
            con.close()
            database_sha = self.module._sha256_file(database)
            provenance_path = root / "distorted" / self.module.DATABASE_PROVENANCE_NAME
            lifecycle = {
                "published_database_sha256": database_sha,
                "isolated_model_aligner_database": {"enabled": False, "published": False},
            }
            feature_match_origin = {
                "kind": "fresh_current_run",
                "current_feature_command_executed": True,
                "current_matcher_command_executed": True,
            }
            self.module._atomic_write_json(
                provenance_path,
                {
                    "status": "complete",
                    "database_sha256": database_sha,
                    "hash_lifecycle": lifecycle,
                    "feature_match_origin": feature_match_origin,
                    "reuse_verified_eligible": True,
                },
            )
            arguments = ["-s", str(root), "--min_model_size", "3"]
            payload = {
                "schema_version": self.module.COMPLETION_MANIFEST_SCHEMA_VERSION,
                "status": "complete",
                "arguments": arguments,
                "arguments_sha256": self.module._canonical_sha256(arguments),
                "colmap_runtime": {
                    "executable_sha256": "a" * 64,
                    "reported_with_cuda": True,
                    "ldd_checked": False,
                    "ldd_has_cudss": False,
                },
                "runtime_requirements": {
                    "require_cuda_colmap": True,
                    "require_global_mapper_cudss": True,
                    "cudss_runtime_check_required": False,
                },
                "input_inventory": self.module.build_file_inventory(root / "input", image_files_only=True),
                "database": {
                    "database_sha256": database_sha,
                    "provenance_sha256": self.module._sha256_file(provenance_path),
                    "hash_lifecycle": lifecycle,
                    "feature_match_origin": feature_match_origin,
                    "reuse_verified_eligible": True,
                },
                "mapper_mode": "global",
                "min_registered_images": 3,
                "alignment": {"enabled": False},
                "outputs": self.module.validate_undistorted_layout(root, min_registered_images=3),
            }
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )
            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=3,
            )
            self.assertTrue(ok, reason)

            aligned_model = root / "distorted" / "sparse_aligned"
            write_text_model(aligned_model, names)
            transform_path = root / "distorted" / "model_alignment_transform.txt"
            transform_path.write_text("1 1 0 0 0 0 0 0\n", encoding="utf-8")
            audit_path = root / "distorted" / "model_alignment_audit.json"
            gps_digest = "c" * 64
            valid_geometry_check = {
                "basis": self.module.ENU_GEOMETRY_SCALE_BASIS,
                "inlier_images": 3,
                "aligned_spread_m": 1.0,
                "reference_spread_m": 1.0,
                "reference_observability_tolerance_m": 1e-12,
                "reference_scale_observable": True,
                "aligned_to_reference_ratio": 1.0,
                "accepted_ratio": [
                    self.module.ENU_GEOMETRY_SCALE_RATIO_MIN,
                    self.module.ENU_GEOMETRY_SCALE_RATIO_MAX,
                ],
                "verified": True,
            }
            audit = {
                "status": "verified",
                "coordinate_scope": "relative_local_enu_only",
                "absolute_geolocation_validated": False,
                "model_aligner_default_random_seed": 0,
                "image_embedded_gps_values_sha256": gps_digest,
                "database_image_gps_references": 3,
                "database_priors_matching_current_exif": 3,
                "registered_common_images": 3,
                "within_threshold_images": 3,
                "formal_expected_images": 3,
                "alignment_max_error_m": 30.0,
                "error_m": {"max": 1.0},
                "geometry_scale_check": valid_geometry_check,
            }
            self.module._atomic_write_json(audit_path, audit)
            lifecycle["isolated_model_aligner_database"] = {
                "enabled": True,
                "published": False,
                "semantics_unchanged": True,
                "pre_semantic_sha256": "d" * 64,
                "post_semantic_sha256": "d" * 64,
            }
            self.module._atomic_write_json(
                provenance_path,
                {
                    "status": "complete",
                    "database_sha256": database_sha,
                    "hash_lifecycle": lifecycle,
                    "feature_match_origin": feature_match_origin,
                    "reuse_verified_eligible": True,
                },
            )
            payload["database"]["provenance_sha256"] = self.module._sha256_file(provenance_path)
            payload["alignment"] = {
                "enabled": True,
                "audit_sha256": self.module._sha256_file(audit_path),
                "transform_sha256": self.module._sha256_file(transform_path),
                "aligned_model_inventory": self.module.build_file_inventory(aligned_model),
                "coordinate_scope": "relative_local_enu_only",
                "absolute_geolocation_validated": False,
                "image_embedded_gps_values_sha256": gps_digest,
            }
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )
            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=3,
            )
            self.assertTrue(ok, reason)

            del audit["geometry_scale_check"]
            self.module._atomic_write_json(audit_path, audit)
            payload["alignment"]["audit_sha256"] = self.module._sha256_file(audit_path)
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )
            ok, reason, _ = self.module.validate_conversion_completion_manifest(root)
            self.assertFalse(ok)
            self.assertIn("geometry-scale evidence", reason)

            audit["geometry_scale_check"] = {
                **valid_geometry_check,
                "aligned_spread_m": 0.1,
                "aligned_to_reference_ratio": 0.1,
            }
            self.module._atomic_write_json(audit_path, audit)
            payload["alignment"]["audit_sha256"] = self.module._sha256_file(audit_path)
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )
            ok, reason, _ = self.module.validate_conversion_completion_manifest(root)
            self.assertFalse(ok)
            self.assertIn("geometry-scale evidence", reason)

            audit["geometry_scale_check"] = valid_geometry_check
            self.module._atomic_write_json(audit_path, audit)
            payload["alignment"]["audit_sha256"] = self.module._sha256_file(audit_path)
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )

            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments + ["--different"],
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=3,
            )
            self.assertFalse(ok)
            self.assertIn("argv differs", reason)

            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="b" * 64,
                expected_min_registered_images=3,
            )
            self.assertFalse(ok)
            self.assertIn("COLMAP executable SHA differs", reason)

            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=4,
            )
            self.assertFalse(ok)
            self.assertIn("minimum registered-image count differs", reason)

            payload["colmap_runtime"]["reported_with_cuda"] = False
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )
            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=3,
            )
            self.assertFalse(ok)
            self.assertIn("required CUDA COLMAP runtime", reason)
            payload["colmap_runtime"]["reported_with_cuda"] = True
            self.module._atomic_write_json(
                root / "distorted" / self.module.COMPLETION_MANIFEST_NAME,
                payload,
            )

            (root / "images" / "a.jpg").write_bytes(b"tampered")
            ok, reason, _ = self.module.validate_conversion_completion_manifest(
                root,
                expected_arguments=arguments,
                expected_colmap_sha256="a" * 64,
                expected_min_registered_images=3,
            )
            self.assertFalse(ok)
            self.assertIn("output inventory differs", reason)

    def test_staged_output_transaction_rolls_back_old_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            staging_root = root / ".convert_uavfgs_layout_test"
            old_paths = [root / "images", root / "sparse", root / "stereo"]
            staged_paths = [staging_root / name for name in ("images", "sparse", "stereo")]
            for old, staged in zip(old_paths, staged_paths):
                old.mkdir(parents=True)
                staged.mkdir(parents=True)
                (old / "value.txt").write_text(f"old-{old.name}", encoding="utf-8")
                (staged / "value.txt").write_text(f"new-{staged.name}", encoding="utf-8")
            old_manifest = root / "distorted" / self.module.COMPLETION_MANIFEST_NAME
            staged_manifest = staging_root / self.module.COMPLETION_MANIFEST_NAME
            old_manifest.parent.mkdir()
            old_manifest.write_text("old-manifest", encoding="utf-8")
            staged_manifest.write_text("new-manifest", encoding="utf-8")
            transaction = self.module._swap_staged_items(
                list(zip(staged_paths, old_paths)) + [(staged_manifest, old_manifest)],
                backup_root=root / ".convert_uavfgs_backup_test",
            )
            for old in old_paths:
                self.assertEqual(
                    (old / "value.txt").read_text(encoding="utf-8"),
                    f"new-{old.name}",
                )
            self.assertEqual(old_manifest.read_text(encoding="utf-8"), "new-manifest")
            self.module._rollback_staged_items(transaction)
            for old, staged in zip(old_paths, staged_paths):
                self.assertEqual(
                    (old / "value.txt").read_text(encoding="utf-8"),
                    f"old-{old.name}",
                )
                self.assertEqual(
                    (staged / "value.txt").read_text(encoding="utf-8"),
                    f"new-{staged.name}",
                )
            self.assertEqual(old_manifest.read_text(encoding="utf-8"), "old-manifest")
            self.assertEqual(staged_manifest.read_text(encoding="utf-8"), "new-manifest")

    def test_interrupted_publish_journal_is_rolled_back_on_next_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "images"
            staged_root = root / ".convert_uavfgs_layout_interrupted"
            staged = staged_root / "images"
            old.mkdir()
            staged.mkdir(parents=True)
            (old / "value.txt").write_text("old", encoding="utf-8")
            (staged / "value.txt").write_text("new", encoding="utf-8")
            backup = root / ".convert_uavfgs_backup_interrupted"
            self.module._swap_staged_items(
                [(staged, old)],
                backup_root=backup,
                cleanup_roots=[staged_root],
            )
            self.assertEqual((old / "value.txt").read_text(encoding="utf-8"), "new")

            self.module.recover_incomplete_output_transactions(root)
            self.assertEqual((old / "value.txt").read_text(encoding="utf-8"), "old")
            self.assertFalse(backup.exists())
            self.assertFalse(staged_root.exists())


if __name__ == "__main__":
    unittest.main()
