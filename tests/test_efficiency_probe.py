from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import efficiency_probe


class _CpuOnlyCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _CpuOnlyTorch:
    cuda = _CpuOnlyCuda()


class EfficiencyProbeTests(unittest.TestCase):
    def test_timing_summary_reports_reciprocal_rates(self) -> None:
        summary = efficiency_probe.timing_summary(2.0, 4)
        self.assertEqual(summary["total_s"], 2.0)
        self.assertEqual(summary["ms_per_item"], 500.0)
        self.assertEqual(summary["items_per_s"], 2.0)

        self.assertIsNone(efficiency_probe.timing_summary(None, 4)["items_per_s"])
        self.assertIsNone(efficiency_probe.timing_summary(2.0, 0)["ms_per_item"])
        self.assertIsNone(efficiency_probe.timing_summary(0.0, 4)["items_per_s"])

    def test_ply_vertex_count_and_artifact_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "final model.ply"
            ply_path.write_bytes(
                b"ply\n"
                b"format binary_little_endian 1.0\n"
                b"comment test fixture\n"
                b"element vertex 7\n"
                b"property float x\n"
                b"end_header\n"
                b"fixture-body"
            )

            self.assertEqual(efficiency_probe.ply_vertex_count(ply_path), 7)
            record = efficiency_probe.artifact_record(ply_path, "final_model")
            self.assertEqual(record["label"], "final_model")
            self.assertEqual(record["path"], str(ply_path))
            self.assertTrue(record["exists"])
            self.assertEqual(record["serialized_bytes"], ply_path.stat().st_size)
            self.assertEqual(record["gaussian_count"], 7)

            missing = efficiency_probe.artifact_record(Path(tmp) / "missing.ply", "missing")
            self.assertFalse(missing["exists"])
            self.assertIsNone(missing["serialized_bytes"])
            self.assertIsNone(missing["gaussian_count"])

    def test_cpu_render_helper_excludes_warmup_from_timed_count(self) -> None:
        calls: list[str] = []

        def render_once(view: str) -> str:
            calls.append(view)
            return view

        summary = efficiency_probe.benchmark_render_calls(
            render_once,
            views=["view-a", "view-b"],
            repeats=4,
            warmup_views=3,
            torch_module=_CpuOnlyTorch(),
        )

        self.assertEqual(len(calls), 3 + 2 * 4)
        self.assertEqual(summary["warmup_views"], 3)
        self.assertEqual(summary["num_unique_views"], 2)
        self.assertEqual(summary["repeats"], 4)
        self.assertEqual(summary["timed_views"], 8)
        self.assertTrue(summary["warmup_excluded"])
        self.assertTrue(summary["io_excluded"])
        self.assertIsNone(summary["cuda_event_total_ms"])
        self.assertIsNone(summary["cuda_event_ms_per_view"])
        self.assertFalse(summary["device"]["cuda_available"])

    def test_torch_stage_probe_writes_cpu_only_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "probe" / "train_efficiency.json"
            probe = efficiency_probe.TorchStageProbe(
                enabled=True,
                output_path=output_path,
                stage="rgb",
                torch_module=_CpuOnlyTorch(),
                metadata={"scene": "fixture"},
            )
            probe.start()
            returned = probe.finish(
                "completed",
                result={"iterations_executed": 3, "gaussian_count": 7},
            )

            self.assertIsNotNone(returned)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_name"], efficiency_probe.SCHEMA_NAME)
            self.assertEqual(payload["kind"], "training_stage")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["stage"], "rgb")
            self.assertEqual(payload["metadata"], {"scene": "fixture"})
            self.assertEqual(payload["result"]["iterations_executed"], 3)
            self.assertFalse(payload["device"]["cuda_available"])
            self.assertIsNone(payload["device"]["peak_torch_allocated_bytes"])
            self.assertFalse(payload["peak_memory_reset_succeeded"])
            self.assertIsNone(payload["error"])

    def test_run_command_probe_records_command_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"12345")
            output_path = root / "command_efficiency.json"

            return_code = efficiency_probe.run_command_probe(
                [sys.executable, "-c", "pass"],
                output_path,
                artifacts={"model": artifact},
                poll_gpu=False,
                poll_interval_s=0.1,
            )

            self.assertEqual(return_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "command")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["return_code"], 0)
            self.assertEqual(payload["gpu_memory_sampling"]["backend"], "disabled")
            self.assertEqual(payload["artifacts"][0]["label"], "model")
            self.assertEqual(payload["artifacts"][0]["serialized_bytes"], 5)

    def test_cli_run_accepts_separator_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "model.bin"
            artifact.write_bytes(b"model")
            output_path = root / "cli_efficiency.json"

            return_code = efficiency_probe.main(
                [
                    "run",
                    "--output",
                    str(output_path),
                    "--artifact",
                    f"final_model={artifact}",
                    "--no_gpu_poll",
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ]
            )

            self.assertEqual(return_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["command"], [sys.executable, "-c", "pass"])
            self.assertEqual(payload["artifacts"][0]["label"], "final_model")
            self.assertEqual(payload["artifacts"][0]["serialized_bytes"], 5)

    def test_command_launch_failure_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "launch_failed.json"
            return_code = efficiency_probe.run_command_probe(
                [str(Path(tmp) / "definitely-missing-command")],
                output_path,
                poll_gpu=False,
            )

            self.assertEqual(return_code, 127)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "launch_failed")
            self.assertEqual(payload["return_code"], 127)
            self.assertIsNone(payload["pid"])
            self.assertEqual(payload["error"]["type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
