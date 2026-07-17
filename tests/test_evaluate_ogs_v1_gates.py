from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tools.evaluate_ogs_v1_gates import (
    OgsGateEvidenceError,
    PAIRED_REPORT_SCHEMA,
    evaluate_paired_smoke_gate,
    evaluate_rgb_direction_gate,
    evaluate_scale_safety,
    evaluate_thermal_direction_gate,
    summarize_gradient_smoke,
)
from utils.ogs_v1 import build_ogs_cache, save_ogs_cache


class TestGateCliEntrypoint(unittest.TestCase):
    def test_direct_script_help_resolves_repository_imports(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(repository / "tools" / "evaluate_ogs_v1_gates.py"), "--help"],
            cwd=Path(tempfile.gettempdir()),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)


def _smoke_record(update: int, ratio: float = 0.1) -> dict:
    return {
        "schema": "uav-tgs-ogs-v1-gradient-smoke-v1",
        "iteration": 30000 + update,
        "continuation_update": update,
        "camera_image_name": f"{update:04d}.jpg",
        "l_rgb": 0.1 + update * 1e-6,
        "l_ogs": 0.2,
        "weighted_l_ogs": 0.0002,
        "penalty_mean": 0.3,
        "eligible_count": 3,
        "active_count": 2,
        "gradient_probe": {
            "raw": {
                "rgb_scaling_norm": 2.0,
                "rgb_rotation_norm": 4.0,
                "ogs_scaling_norm": 200.0,
                "ogs_rotation_norm": 400.0,
                "weighted_ogs_scaling_norm": 0.2,
                "weighted_ogs_rotation_norm": 0.4,
                "scaling_cosine": -0.25,
                "rotation_cosine": 0.5,
            },
            "lr_scaled": {
                "scaling_lr": 0.01,
                "rotation_lr": 0.005,
                "rgb_scaling_update_proxy": 0.02,
                "rgb_rotation_update_proxy": 0.02,
                "weighted_ogs_scaling_update_proxy": 0.002,
                "weighted_ogs_rotation_update_proxy": 0.002,
                "rgb_combined_update_proxy": 0.028,
                "weighted_ogs_combined_update_proxy": 0.0028,
                "combined_ratio": ratio,
                "scaling_ratio": 0.1,
                "rotation_ratio": 0.1,
            },
        },
        "finite": True,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in records),
        encoding="utf-8",
    )


def _checkpoint(path: Path, iteration: int, scales: torch.Tensor) -> None:
    count = scales.shape[0]
    names = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")
    groups = [{"name": name, "params": [index], "lr": 1e-3} for index, name in enumerate(names)]
    states = {
        index: {
            "step": torch.tensor(float(iteration)),
            "exp_avg": torch.zeros((count, 1)),
            "exp_avg_sq": torch.zeros((count, 1)),
        }
        for index in range(len(names))
    }
    model = (
        3,
        torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) * 0.01,
        torch.zeros((count, 1, 3)),
        torch.zeros((count, 15, 3)),
        torch.log(scales.to(torch.float32)),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        torch.zeros((count, 1)),
        torch.zeros(count),
        torch.zeros((count, 1)),
        torch.zeros((count, 1)),
        {"param_groups": groups, "state": states},
        1.0,
    )
    torch.save((model, iteration), path)


class OgsGradientSmokeGateTests(unittest.TestCase):
    def test_exact_200_updates_and_first_20_excluded_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "smoke.jsonl"
            _write_jsonl(path, [_smoke_record(index) for index in range(1, 201)])
            report = summarize_gradient_smoke(path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                report["protocol"]["gate_window_updates_inclusive"], [21, 200]
            )
            self.assertEqual(report["summaries_gate_window"]["combined_ratio"]["count"], 180)
            self.assertAlmostEqual(
                report["summaries_gate_window"]["combined_ratio"]["p50"], 0.1
            )
            self.assertEqual(report["eligible_count"], 3)
            self.assertIn(
                "20..200",
                report["protocol"]["source_wording_conflict"]["numeric_wording"],
            )

    def test_ratio_outside_gate_stops_without_structural_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "smoke.jsonl"
            _write_jsonl(
                path, [_smoke_record(index, ratio=0.25) for index in range(1, 201)]
            )
            report = summarize_gradient_smoke(path)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(
                report["decisions"]["combined_ratio_median_lte_maximum"]
            )

    def test_missing_duplicate_or_nonfinite_evidence_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "smoke.jsonl"
            records = [_smoke_record(index) for index in range(1, 201)]
            records[-1]["continuation_update"] = 199
            _write_jsonl(path, records)
            with self.assertRaisesRegex(OgsGateEvidenceError, "exactly once"):
                summarize_gradient_smoke(path)

            records = [_smoke_record(index) for index in range(1, 201)]
            records[40]["gradient_probe"]["lr_scaled"]["combined_ratio"] = float("nan")
            _write_jsonl(path, records)
            with self.assertRaisesRegex(OgsGateEvidenceError, "NaN or Inf"):
                summarize_gradient_smoke(path)


class OgsScaleAndPairedGateTests(unittest.TestCase):
    def _fixture(self, root: Path, *, current_scales: torch.Tensor | None = None):
        anchor_scales = torch.tensor(
            [[1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [1.5, 0.8, 0.6]],
            dtype=torch.float64,
        )
        if current_scales is None:
            current_scales = anchor_scales * torch.tensor(
                [[1.01, 0.99, 1.0], [1.0, 0.98, 1.02], [0.99, 1.01, 1.0]]
            )
        anchor = root / "anchor.pth"
        current = root / "current.pth"
        _checkpoint(anchor, 30000, anchor_scales)
        _checkpoint(current, 30200, current_scales)

        # The cache binds to the exact trusted checkpoint bytes.
        import hashlib

        digest = hashlib.sha256(anchor.read_bytes()).hexdigest()
        rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3, dtype=torch.float64)
        cache = build_ogs_cache(
            observability=torch.tensor([0.0, 0.5, 0.8], dtype=torch.float64),
            weakest_direction=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float64,
            ),
            visible_count=torch.tensor([8, 9, 10]),
            activated_opacity=torch.tensor([0.5, 0.5, 0.5]),
            anchor_scales=anchor_scales,
            anchor_rotations=rotations,
            metadata={
                "checkpoint_iteration": 30000,
                "checkpoint_sha256": digest,
            },
        )
        cache_path = root / "cache.pt"
        save_ogs_cache(cache_path, cache)
        protocol = root / "protocol.json"
        protocol.write_text(
            json.dumps(
                {
                    "schema": "uav-tgs-rgb-continuation-protocol-v1",
                    "recipe": "fixed_topology",
                    "start_checkpoint": str(anchor.resolve()),
                    "anchor_iteration": 30000,
                    "scheduler_horizon": 30000,
                    "requested_updates": 200,
                    "final_iteration": 30200,
                    "optimizer_step_at_final_iteration": True,
                    "topology_fixed": True,
                    "densification": False,
                    "pruning": False,
                    "opacity_reset": False,
                    "artifact_save_semantics": "aligned",
                    "ordered_camera_sha256": "camera-order",
                    "camera_sequence_sha256": "camera-sequence",
                    "ogs_v1": True,
                    "ogs_cache_sha256": cache["cache_sha256"],
                }
            ),
            encoding="utf-8",
        )
        return anchor, current, cache_path, protocol

    @staticmethod
    def _write_mapping(root: Path) -> None:
        with (root / "clamp20_diagnostics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("gaussian_index", "weak_eigengap")
            )
            writer.writeheader()
            writer.writerow({"gaussian_index": 0, "weak_eigengap": 0.25})
        with (root / "scale_matched_controls.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("clamp_index", "control_index", "weak_eigengap"),
            )
            writer.writeheader()
            writer.writerow(
                {"clamp_index": 0, "control_index": 1, "weak_eigengap": 0.5}
            )

    def test_checkpoint_scale_safety_and_fixed_mapping_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor, current, cache, protocol = self._fixture(root)
            self._write_mapping(root)
            report = evaluate_scale_safety(
                anchor_checkpoint=anchor,
                current_checkpoint=current,
                cache_path=cache,
                continuation_protocol=protocol,
                current_iteration=30200,
                group="O35-smoke",
                expect_ogs=True,
                audit_mapping_directory=root,
            )
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["topology_and_index_contract"]["passed"])
            self.assertIn(
                "max",
                report["current_vs_anchor"]["axis_ratio_percentiles"]["axis_0"],
            )
            self.assertEqual(
                report["clamp20_control_diagnostic"]["index_counts"]["clamp20"], 1
            )
            self.assertIn(
                "before_anchor", report["clamp20_control_diagnostic"]
            )
            self.assertIn(
                "after_current", report["clamp20_control_diagnostic"]
            )

    def test_large_scale_collapse_is_an_explicit_failed_evidence_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor_scales = torch.tensor(
                [[1.0, 1.0, 1.0], [2.0, 1.0, 0.5], [1.5, 0.8, 0.6]]
            )
            anchor, current, cache, protocol = self._fixture(
                root, current_scales=anchor_scales * 0.1
            )
            report = evaluate_scale_safety(
                anchor_checkpoint=anchor,
                current_checkpoint=current,
                cache_path=cache,
                continuation_protocol=protocol,
                current_iteration=30200,
                group="O35-smoke",
                expect_ogs=True,
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(
                report["decisions"]["fraction_any_axis_lt_0_5_within_limit"]
            )
            self.assertIn("not a loss-weight search", report["scale_safety_policy"]["purpose"])

    def test_protocol_must_prove_fixed_topology(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anchor, current, cache, protocol = self._fixture(root)
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["pruning"] = True
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(OgsGateEvidenceError, "pruning mismatch"):
                evaluate_scale_safety(
                    anchor_checkpoint=anchor,
                    current_checkpoint=current,
                    cache_path=cache,
                    continuation_protocol=protocol,
                    current_iteration=30200,
                    group="O35-smoke",
                    expect_ogs=True,
                )

    def test_paired_gate_requires_shared_anchor_cache_camera_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gradient = root / "gradient.json"
            r_scale = root / "r.json"
            o_scale = root / "o.json"
            gradient.write_text(
                json.dumps(
                    {
                        "schema": "uav-tgs-ogs-v1-gradient-smoke-gate-v1",
                        "status": "passed",
                    }
                ),
                encoding="utf-8",
            )
            scale_payload = {
                "schema": "uav-tgs-ogs-v1-scale-safety-v1",
                "status": "passed",
                "current_iteration": 30200,
                "inputs": {
                    "anchor_checkpoint": {"sha256": "anchor"},
                    "ogs_cache": {"semantic_sha256": "cache"},
                    "continuation_protocol": {
                        "camera_sequence_sha256": "sequence"
                    },
                },
            }
            r_scale.write_text(json.dumps(scale_payload), encoding="utf-8")
            o_scale.write_text(json.dumps(scale_payload), encoding="utf-8")
            report = evaluate_paired_smoke_gate(
                gradient_report_path=gradient,
                r_scale_report_path=r_scale,
                o_scale_report_path=o_scale,
            )
            self.assertEqual(report["schema"], PAIRED_REPORT_SCHEMA)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                report["action"], "continue_to_paired_5000_update_R35_O35"
            )


class OgsEndpointDirectionGateTests(unittest.TestCase):
    def test_rgb_gate_applies_exact_appearance_depth_and_qualitative_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "uav-tgs-ogs-v1-rgb-direction-input-v1",
                        "r35": {
                            "appearance": {
                                "psnr": 25.0,
                                "ssim": 0.900,
                                "lpips": 0.100,
                            },
                            "depth": {
                                "front_at_1m": 0.20,
                                "mean_error_m": 1.0,
                                "missing_rate": 0.010,
                            },
                        },
                        "o35": {
                            "appearance": {
                                "psnr": 24.91,
                                "ssim": 0.898,
                                "lpips": 0.104,
                            },
                            "depth": {
                                "front_at_1m": 0.145,
                                "mean_error_m": 0.97,
                                "missing_rate": 0.011,
                            },
                        },
                        "qualitative_thin_structure_pass": True,
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_rgb_direction_gate(path)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(report["decisions"]["geometry_direction_gate"])
            self.assertEqual(report["action"], "continue_to_phase_d_fixed_f3")

    def test_rgb_gate_fails_when_geometry_and_qualitative_evidence_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rgb.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "uav-tgs-ogs-v1-rgb-direction-input-v1",
                        "r35": {
                            "appearance": {
                                "psnr": 25.0,
                                "ssim": 0.9,
                                "lpips": 0.1,
                            },
                            "depth": {
                                "front_at_1m": 0.2,
                                "mean_error_m": 1.0,
                                "missing_rate": 0.01,
                            },
                        },
                        "o35": {
                            "appearance": {
                                "psnr": 25.0,
                                "ssim": 0.9,
                                "lpips": 0.1,
                            },
                            "depth": {
                                "front_at_1m": 0.21,
                                "mean_error_m": 0.98,
                                "missing_rate": 0.01,
                            },
                        },
                        "qualitative_thin_structure_pass": False,
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_rgb_direction_gate(path)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["decisions"]["geometry_direction_gate"])
            self.assertFalse(report["decisions"]["qualitative_thin_structure_pass"])

    def test_thermal_gate_uses_or_for_temperature_and_declared_off_lut_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "thermal.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": "uav-tgs-ogs-v1-thermal-direction-input-v1",
                        "r35_f3": {
                            "appearance": {
                                "t_psnr": 22.0,
                                "ssim": 0.80,
                                "lpips": 0.20,
                            },
                            "temperature": {
                                "mae_c": 1.0,
                                "rmse_c": 2.0,
                                "off_lut": 0.010,
                            },
                        },
                        "o35_f3": {
                            "appearance": {
                                "t_psnr": 21.90,
                                "ssim": 0.795,
                                "lpips": 0.207,
                            },
                            "temperature": {
                                # MAE passes absolute (0.08 C); RMSE passes
                                # relative (0.08 / 2 = 4%).
                                "mae_c": 1.08,
                                "rmse_c": 2.08,
                                "off_lut": 0.0105,
                            },
                        },
                        "off_lut_policy": {
                            "metric_name": "mean_off_lut_distance",
                            "maximum_absolute_increase": 0.001,
                            "maximum_relative_increase": 0.10,
                            "declared_before_evaluation": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = evaluate_thermal_direction_gate(path)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(
                report["decisions"]["temperature_mae_absolute_or_relative_gate"]
            )
            self.assertTrue(
                report["decisions"]["temperature_rmse_absolute_or_relative_gate"]
            )

    def test_thermal_off_lut_policy_is_required_and_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "thermal.json"
            payload = {
                "schema": "uav-tgs-ogs-v1-thermal-direction-input-v1",
                "r35_f3": {
                    "appearance": {"t_psnr": 22.0, "ssim": 0.8, "lpips": 0.2},
                    "temperature": {"mae_c": 1.0, "rmse_c": 2.0, "off_lut": 0.01},
                },
                "o35_f3": {
                    "appearance": {"t_psnr": 22.0, "ssim": 0.8, "lpips": 0.2},
                    "temperature": {"mae_c": 1.0, "rmse_c": 2.0, "off_lut": 0.02},
                },
                "off_lut_policy": {
                    "metric_name": "mean_off_lut_distance",
                    "maximum_absolute_increase": 0.001,
                    "maximum_relative_increase": 0.10,
                    "declared_before_evaluation": True,
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            report = evaluate_thermal_direction_gate(path)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(
                report["decisions"]["off_lut_declared_absolute_limit_pass"]
            )
            self.assertFalse(
                report["decisions"]["off_lut_declared_relative_limit_pass"]
            )

            del payload["off_lut_policy"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(OgsGateEvidenceError, "off_lut_policy"):
                evaluate_thermal_direction_gate(path)


if __name__ == "__main__":
    unittest.main()
