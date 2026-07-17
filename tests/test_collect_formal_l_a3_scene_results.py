from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools import collect_formal_l_a3_scene_results
from tools import summarize_a3_three_scene_confirmation


class FormalLA3SceneCollectorTests(unittest.TestCase):
    ITERATIONS = (40000, 50000, 60000)

    @staticmethod
    def _write_json(path: Path, payload: dict, *, allow_nan: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _depth(
        *,
        front_1m: float = 0.10,
        missing: float = 0.01,
    ) -> dict:
        return {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": "InternalRoad",
            "counts": {
                "reference_valid_pixels": 1000,
                "model_valid_on_reference_pixels": int(1000 * (1.0 - missing)),
                "missing_pixels": int(1000 * missing),
            },
            "secondary_metrics": {
                "MissingRate": missing,
                "AbsDepthError_Mean": 0.8,
                "AbsDepthError_Median": 0.5,
                "SignedDepthBias_Mean": -0.1,
            },
            "threshold_metrics": [
                {
                    "threshold_m": threshold,
                    "FrontIntrusionRate": front_1m if threshold == 1.0 else 0.05,
                    "TooDeepRate": 0.08,
                    "DepthAgreementRate": 0.87,
                }
                for threshold in (1.0, 2.0, 5.0)
            ],
        }

    @staticmethod
    def _temperature(
        *,
        mae: float = 1.0,
        rmse: float = 1.4,
        off_lut_mean: float = 3.0,
        off_lut_p95: float = 8.0,
    ) -> dict:
        supported = {
            "mae_c": mae,
            "rmse_c": rmse,
            "signed_bias_c": -0.2,
            "p95_abs_error_c": 2.5,
            "max_abs_error_c": 5.0,
            "pixels": 960,
        }
        return {
            "schema": "uav-tgs-apparent-temperature-evaluation-v2",
            "status": "complete",
            "completed_with_missing": False,
            "primary_metric_valid": True,
            "support_is_explicit": True,
            "summary": {
                "evaluated_file_count": 64,
                "temperature_error_supported_pixels": supported,
                "pixel_micro_aggregate": {"supported_pixels": supported},
                "frame_macro_mean_std": {
                    "supported_pixels": {
                        "frame_count": 64,
                        "mae_c": {"mean": mae + 0.01, "std": 0.1, "frame_count": 64},
                        "rmse_c": {"mean": rmse + 0.01, "std": 0.1, "frame_count": 64},
                        "signed_bias_c": {"mean": -0.2, "std": 0.05, "frame_count": 64},
                        "p95_abs_error_c": {"mean": 2.5, "std": 0.2, "frame_count": 64},
                    }
                },
                "off_lut_distance_aggregates": {
                    "supported_pixels": {
                        "mean_rgb_distance": off_lut_mean,
                        "p95_rgb_distance": off_lut_p95,
                        "max_rgb_distance": 20.0,
                        "rms_rgb_distance": 5.0,
                        "pixels": 960,
                    }
                },
                "clipping_aggregates": {
                    "supported_pixels": {
                        "clipped_pixels": 2,
                        "clipping_ratio": 2.0 / 960.0,
                        "high_pixels": 1,
                        "low_pixels": 1,
                        "pixels": 960,
                    }
                },
                "support_coverage": {
                    "expected_frames": 64,
                    "render_available_frames": 64,
                    "support_available_frames": 64,
                    "frames_without_supported_pixels": 0,
                    "missing_render_frames": 0,
                    "missing_mask_frames": 0,
                    "expected_pixels": 1000,
                    "supported_pixels": 960,
                    "supported_ratio": 0.96,
                    "unsupported_pixels": 30,
                    "unsupported_ratio": 0.03,
                    "missing_pixels": 10,
                    "missing_ratio": 0.01,
                    "unsupported_or_missing_pixels": 40,
                    "unsupported_or_missing_ratio": 0.04,
                },
            },
        }

    @staticmethod
    def _opacity(
        iteration: int,
        *,
        structural: bool = True,
        saturation_detected: bool = False,
    ) -> dict:
        return {
            "schema": "uav-tgs-opacity-adaptation-audit-v1",
            "status": "passed",
            "ply_audit": {
                "gaussian_count": 123,
                "all_structural_fields_exact": structural,
                "activated_opacity": {
                    "semantics": "sigmoid(raw PLY opacity logit)",
                    "anchor": {
                        "count": 123,
                        "mean": 0.30,
                        "median": 0.20,
                        "p95": 0.90,
                        "p99": 0.99,
                        "max": 0.999999,
                    },
                    "a3": {
                        "count": 123,
                        "mean": 0.31,
                        "median": 0.21,
                        "p95": 0.91,
                        "p99": 0.991,
                        "max": 0.9999999,
                    },
                    "a3_minus_anchor": {
                        "absolute": {
                            "count": 123,
                            "mean": 0.02,
                            "median": 0.01,
                            "p95": 0.06,
                            "p99": 0.09,
                            "max": 0.15,
                        },
                        "signed": {
                            "count": 123,
                            "mean": 0.01,
                            "median": 0.0,
                            "p95": 0.05,
                            "p99": 0.08,
                            "max": 0.15,
                        },
                        "rmse": 0.03,
                        "fractions_abs_delta_gt": {
                            "0.01": 0.4,
                            "0.05": 0.1,
                            "0.10": 0.01,
                        },
                    },
                    "catastrophic_saturation": {
                        "definition": (
                            "candidate A3 catastrophic only when at least 0.99 of "
                            "activated opacities collapse to the low or high endpoint"
                        ),
                        "thresholds": {
                            "low_activated_opacity_lte": 1e-4,
                            "high_activated_opacity_gte": 1.0 - 1e-4,
                            "catastrophic_fraction_gte": 0.99,
                        },
                        "anchor": {
                            "low_fraction": 0.0,
                            "high_fraction": 0.0,
                            "detected": False,
                        },
                        "a3": {
                            "low_fraction": 0.99 if saturation_detected else 0.0,
                            "high_fraction": 0.0,
                            "detected": saturation_detected,
                        },
                        "detected": saturation_detected,
                    },
                },
            },
            "opacity_proxy_audit": {
                "frame_macro": {
                    "view_count": 64,
                    "mean_abs_delta_mean": 0.01,
                    "mean_rmse": 0.02,
                    "max_abs_delta": 0.2,
                },
                "pixel_micro": {
                    "count": 1000,
                    "anchor_mean": 0.90,
                    "a3_mean": 0.91,
                    "signed_delta_mean": 0.01,
                    "abs_delta_mean": 0.01,
                    "abs_delta_median": 0.002,
                    "abs_delta_p95": 0.04,
                    "abs_delta_p99": 0.08,
                    "abs_delta_max": 0.2,
                    "rmse": 0.02,
                    "fraction_abs_delta_gt_0.01": 0.2,
                    "fraction_abs_delta_gt_0.05": 0.03,
                    "fraction_abs_delta_gt_0.10": 0.005,
                },
            },
        }

    def _build_fixture(
        self,
        root: Path,
        *,
        a3_psnr: float = 20.0,
        a3_ssim: float = 0.80,
        a3_lpips: float = 0.20,
        l_mae: float = 1.0,
        l_rmse: float = 1.4,
        a3_mae: float = 1.0,
        a3_rmse: float = 1.4,
        a3_front_1m: float = 0.10,
        a3_missing: float = 0.01,
        a3_off_lut_mean: float = 3.0,
        a3_off_lut_p95: float = 8.0,
        saturation_detected: bool = False,
        structural: bool = True,
    ) -> dict[str, Path | str]:
        l_model = root / "models" / "L"
        a3_model = root / "models" / "A3"
        evaluation = root / "evaluation"
        audit_root = root / "audits"
        l_efficiency = root / "efficiency" / "L"
        a3_efficiency = root / "efficiency" / "A3"
        for path in (
            l_model,
            a3_model,
            evaluation,
            audit_root,
            l_efficiency,
            a3_efficiency,
        ):
            path.mkdir(parents=True, exist_ok=True)

        for group, model_root, metrics in (
            ("L", l_model, (20.0, 0.80, 0.20)),
            ("A3", a3_model, (a3_psnr, a3_ssim, a3_lpips)),
        ):
            results = {}
            results_plus = {}
            for iteration in self.ITERATIONS:
                results[f"ours_{iteration}"] = {
                    "PSNR": metrics[0],
                    "SSIM": metrics[1],
                    "LPIPS": metrics[2],
                }
                results_plus[f"ours_{iteration}"] = {"EdgeHaloScore": 0.004}
            self._write_json(model_root / "results.json", results, allow_nan=True)
            self._write_json(model_root / "results_plus.json", results_plus)
            self._write_json(
                model_root / "per_view.json",
                {
                    "ours_60000": {
                        "PSNR": {"0000.png": metrics[0]},
                        "SSIM": {"0000.png": metrics[1]},
                        "LPIPS": {"0000.png": metrics[2]},
                    }
                },
            )

        for group, efficiency_root in (("L", l_efficiency), ("A3", a3_efficiency)):
            self._write_json(
                efficiency_root / f"{group}_train.json",
                {
                    "schema_name": "uav-tgs-efficiency",
                    "schema_version": 1,
                    "kind": "training_stage",
                    "status": "completed",
                    "stage": "thermal",
                    "wall_time_s": 12.0,
                },
            )
            for iteration in self.ITERATIONS:
                self._write_json(
                    efficiency_root / f"{group}_render_{iteration}.json",
                    {
                        "schema_name": "uav-tgs-efficiency",
                        "schema_version": 1,
                        "kind": "render",
                        "status": "completed",
                        "iteration": iteration,
                        "gaussian_count": 123,
                        "benchmark": {"mean_ms_per_view": 2.5},
                    },
                )

        for group in ("L", "A3"):
            for iteration in self.ITERATIONS:
                if group == "L":
                    temp = self._temperature(mae=l_mae, rmse=l_rmse)
                    depth = self._depth()
                else:
                    temp = self._temperature(
                        mae=a3_mae,
                        rmse=a3_rmse,
                        off_lut_mean=a3_off_lut_mean,
                        off_lut_p95=a3_off_lut_p95,
                    )
                    depth = self._depth(front_1m=a3_front_1m, missing=a3_missing)
                self._write_json(
                    evaluation / "temperature" / f"{group}_{iteration}.json", temp
                )
                self._write_json(
                    evaluation
                    / "depth"
                    / group
                    / str(iteration)
                    / "metrics"
                    / "metrics_summary.json",
                    depth,
                )

        runtime = {
            "schema": "uav-tgs-opacity-adaptive-runtime-protocol-v1",
            "status": "passed",
            "thermal_recipe": "geometry_frozen_opacity_adaptive",
            "thermal_freeze_mode": "geometry_frozen_opacity_adaptive",
            "thermal_max_sh_degree": 3,
            "thermal_optimizer_state": "fresh",
            "optimizer_groups": ["f_dc", "f_rest", "opacity"],
            "optimizer_lrs": {"f_dc": 0.001, "f_rest": 0.00005, "opacity": 0.0002},
            "trainability": {
                "xyz": False,
                "f_dc": True,
                "f_rest": True,
                "opacity": True,
                "scaling": False,
                "rotation": False,
                "exposure": False,
            },
            "thermal_scale_clamp": "off",
            "topology_frozen": True,
            "densification": False,
            "pruning": False,
            "opacity_reset": False,
            "artifact_save_semantics": "aligned",
        }
        self._write_json(a3_model / "opacity_adaptive_protocol.json", runtime)
        self._write_json(
            a3_model / "opacity_adaptive_freeze_audit.json",
            {
                "status": "passed",
                "start_iteration": 30000,
                "final_iteration": 60000,
                "activated_opacity": {
                    "count": 123,
                    "finite": True,
                    "min": 0.000001,
                    "max": 0.9999999,
                },
                "frozen_fields": {
                    name: {
                        "unchanged": structural,
                        "max_abs_diff": 0.0 if structural else 0.1,
                        "before_sha256": "a" * 64,
                        "after_sha256": "a" * 64 if structural else "b" * 64,
                    }
                    for name in ("_xyz", "_scaling", "_rotation")
                },
            },
        )
        self._write_json(
            audit_root / "A3_endpoints.json",
            {
                "status": "passed",
                "configuration": {
                    "iterations": list(self.ITERATIONS),
                    "thermal_max_sh_degree": 3,
                    "strict_geometry": False,
                },
                "rgb_reference": {"gaussian_count": 123},
                "endpoints": [
                    {
                        "iteration": iteration,
                        "status": "passed",
                        "gaussian_count": 123,
                        "gaussian_count_equal_rgb": True,
                        "schema_equal_rgb": True,
                    }
                    for iteration in self.ITERATIONS
                ],
            },
        )
        for iteration in self.ITERATIONS:
            self._write_json(
                audit_root / f"A3_{iteration}_opacity.json",
                self._opacity(
                    iteration,
                    structural=structural,
                    saturation_detected=saturation_detected and iteration == 60000,
                ),
            )
            self._write_json(
                audit_root / f"A3_{iteration}_ply_checkpoint_exact.json",
                {
                    "status": "passed",
                    "iteration": iteration,
                    "expected_iteration": iteration,
                    "active_sh_degree": 3,
                    "expected_active_sh_degree": 3,
                    "gaussian_count": 123,
                    "optimizer_group_names": ["f_dc", "f_rest", "opacity"],
                    "expected_optimizer_group_names": ["f_dc", "f_rest", "opacity"],
                    "all_parameter_groups_exact": True,
                    "groups": {
                        name: {"finite": True, "exact": True, "max_abs_diff": 0.0}
                        for name in (
                            "xyz",
                            "f_dc",
                            "f_rest",
                            "opacity",
                            "scaling",
                            "rotation",
                        )
                    },
                },
            )

        anchor_depth = root / "rgb_anchor_depth.json"
        self._write_json(anchor_depth, self._depth(front_1m=0.09, missing=0.005))
        bound_split = root / "bound_split.json"
        split_assignments = ["train"] * 2 + ["test"] * 64 + ["guard"]
        self._write_json(
            bound_split,
            {
                "scene": "InternalRoad",
                "counts": {"train": 2, "test": 64, "guard": 1, "total": 67},
                "records": [
                    {"pair_id": f"{index:04d}", "split": split}
                    for index, split in enumerate(split_assignments)
                ],
            },
        )
        block_analysis = root / "block_analysis.json"
        self._write_json(
            block_analysis,
            {
                "schema": "uav-tgs-formal-test-block-analysis-v1",
                "status": "complete",
                "iteration": 60000,
                "protocol": {
                    "groups": ["L", "A3"],
                    "test_views": 64,
                    "block_count": 4,
                    "views_per_block": 16,
                },
                "classification_counts": {
                    "A3_minus_L": {
                        "PSNR": {"improved": 0, "tied": 4, "declined": 0}
                    }
                },
                "inputs": {
                    "bound_split": {
                        "path": str(bound_split),
                        "sha256": self._sha256(bound_split),
                    },
                    "per_view:L": {
                        "path": str(l_model / "per_view.json"),
                        "sha256": self._sha256(l_model / "per_view.json"),
                    },
                    "per_view:A3": {
                        "path": str(a3_model / "per_view.json"),
                        "sha256": self._sha256(a3_model / "per_view.json"),
                    },
                    "temperature:L": {
                        "path": str(evaluation / "temperature" / "L_60000.json"),
                        "sha256": self._sha256(
                            evaluation / "temperature" / "L_60000.json"
                        ),
                    },
                    "temperature:A3": {
                        "path": str(evaluation / "temperature" / "A3_60000.json"),
                        "sha256": self._sha256(
                            evaluation / "temperature" / "A3_60000.json"
                        ),
                    },
                },
                "blocks": [
                    {
                        "block": {
                            "strip_id": "tg-0000",
                            "block_index": index,
                            "stratum": "nadir:p-090",
                            "size": 16,
                            "views": [f"{index * 16 + i:04d}.png" for i in range(16)],
                        },
                        "group_means": {
                            "L": {"PSNR": 20.0},
                            "A3": {"PSNR": a3_psnr},
                        },
                        "paired_comparisons": {
                            "A3_minus_L": {
                                "combined_non_degraded": True,
                                "metrics": {
                                    "PSNR": {
                                        "raw_a3_minus_baseline": a3_psnr - 20.0,
                                        "classification": "tied",
                                    }
                                },
                                "diagnostic_deltas": {},
                            }
                        },
                    }
                    for index in range(4)
                ],
            },
        )

        assessment = root / "manual_assessment.json"
        self._write_json(
            assessment,
            {
                "schema": "uav-tgs-stage2-scene-manual-assessment-v1",
                "scene": "InternalRoad",
                "pipeline_reference_irreparable_failure": {
                    "assessed": True,
                    "detected": False,
                    "basis": "all formal artifacts completed",
                },
            },
        )
        return {
            "scene": "InternalRoad",
            "l_model_root": l_model,
            "a3_model_root": a3_model,
            "evaluation_root": evaluation,
            "a3_audit_root": audit_root,
            "l_efficiency_root": l_efficiency,
            "a3_efficiency_root": a3_efficiency,
            "bound_split": bound_split,
            "rgb_anchor_depth": anchor_depth,
            "block_analysis": block_analysis,
            "manual_assessment": assessment,
            "next_scene": "Urban20K",
        }

    @staticmethod
    def _collect(fixture: dict[str, Path | str]) -> dict:
        return collect_formal_l_a3_scene_results.collect_scene(**fixture)

    def test_passing_scene_reports_all_metrics_and_continue_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            payload = self._collect(self._build_fixture(Path(temp)))

        self.assertEqual(payload["scene"], "InternalRoad")
        self.assertEqual(
            payload["schema"], "uav-tgs-a3-scene-l-vs-a3-summary-v1"
        )
        self.assertEqual(payload["counts"]["test_views"], 64)
        self.assertEqual(payload["counts"]["test_blocks"], 4)
        self.assertEqual(
            payload["methods"]["A3"]["metrics"]["temperature_rmse_c"], 1.4
        )
        self.assertTrue(payload["invariants"]["all_passed"])
        self.assertFalse(payload["reviews"]["material_off_lut_degradation"])
        self.assertEqual(
            payload["efficiency"]["A3"]["training"]["wall_time_s"], 12.0
        )
        self.assertEqual(payload["decision_endpoint"], 60000)
        self.assertEqual(set(payload["appearance"]), {"L", "A3"})
        self.assertEqual(
            payload["appearance"]["A3"]["60000"]["EdgeHaloScore"], 0.004
        )
        self.assertEqual(
            payload["temperature"]["A3"]["60000"]["pixel_micro"]["mae_c"],
            1.0,
        )
        self.assertEqual(
            payload["depth"]["A3"]["60000"]["thresholds_m"]["1"]
            ["front_intrusion_rate"],
            0.10,
        )
        self.assertTrue(payload["invariants"]["passed"])
        self.assertEqual(payload["normal_gates"]["status"], "passed")
        self.assertEqual(payload["catastrophic_gates"]["status"], "not_detected")
        self.assertEqual(
            payload["continuation_decision"]["action"], "continue_to_next_scene"
        )
        self.assertEqual(payload["continuation_decision"]["target"], "Urban20K")
        macro_input = summarize_a3_three_scene_confirmation._validate_scene_payload(
            payload, "InternalRoad"
        )
        self.assertEqual(macro_input["counts"]["test_views"], 64)
        self.assertIn("temperature_rmse_c", macro_input["methods"]["A3"])

    def test_ordinary_normal_gate_failure_still_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            # 0.20 dB fails the 0.15 normal gate but is below the >0.30
            # catastrophic threshold.
            payload = self._collect(
                self._build_fixture(Path(temp), a3_psnr=19.80)
            )

        self.assertEqual(payload["normal_gates"]["status"], "failed")
        self.assertFalse(
            payload["normal_gates"]["checks"]["psnr_loss_le_0_15_db"]["passed"]
        )
        self.assertEqual(payload["catastrophic_gates"]["status"], "not_detected")
        decision = payload["continuation_decision"]
        self.assertEqual(decision["action"], "continue_to_next_scene")
        self.assertTrue(decision["ordinary_failure_recorded"])
        self.assertIn("non-catastrophic", decision["reason"])

    def test_catastrophic_thresholds_are_strict_and_temperature_uses_and(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            boundary = self._collect(
                self._build_fixture(
                    Path(temp),
                    a3_psnr=19.70,
                    a3_mae=1.20,
                    a3_front_1m=0.13,
                )
            )
        self.assertEqual(boundary["catastrophic_gates"]["status"], "not_detected")

        with tempfile.TemporaryDirectory() as temp:
            psnr = self._collect(
                self._build_fixture(Path(temp), a3_psnr=19.699999)
            )
        self.assertTrue(
            psnr["catastrophic_gates"]["checks"]["psnr_loss_gt_0_30_db"]
            ["detected"]
        )
        self.assertEqual(psnr["continuation_decision"]["action"], "stop")

        with tempfile.TemporaryDirectory() as temp:
            temp_abs_only = self._collect(
                self._build_fixture(Path(temp), a3_mae=1.201)
            )
        self.assertTrue(
            temp_abs_only["catastrophic_gates"]["checks"]
            ["temperature_mae_increase_gt_0_20_c_and_gt_10_percent"]["detected"]
        )

        with tempfile.TemporaryDirectory() as temp:
            # Absolute worsening alone is insufficient: 0.201 C over a 3 C
            # baseline is below the strict 10% arm of the catastrophic AND.
            temp_only_one_arm = self._collect(
                self._build_fixture(Path(temp), l_mae=3.0, a3_mae=3.201)
            )
        self.assertFalse(
            temp_only_one_arm["catastrophic_gates"]["checks"]
            ["temperature_mae_increase_gt_0_20_c_and_gt_10_percent"]["detected"]
        )

        with tempfile.TemporaryDirectory() as temp:
            front = self._collect(
                self._build_fixture(Path(temp), a3_front_1m=0.130001)
            )
        self.assertTrue(
            front["catastrophic_gates"]["checks"]["front_1m_increase_gt_3pp"]
            ["detected"]
        )

    def test_temperature_normal_gate_uses_relative_or_absolute_for_mae_and_rmse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            # Absolute increase is within 0.1 C even though relative MAE is >5%.
            absolute_pass = self._collect(
                self._build_fixture(Path(temp), a3_mae=1.08, a3_rmse=1.49)
            )
        checks = absolute_pass["normal_gates"]["checks"]
        self.assertTrue(checks["temperature_mae_relative_le_5pct_or_absolute_le_0_1c"]["passed"])
        self.assertTrue(checks["temperature_rmse_relative_le_5pct_or_absolute_le_0_1c"]["passed"])

        with tempfile.TemporaryDirectory() as temp:
            failed = self._collect(
                self._build_fixture(Path(temp), a3_mae=1.11, a3_rmse=1.51)
            )
        checks = failed["normal_gates"]["checks"]
        self.assertFalse(checks["temperature_mae_relative_le_5pct_or_absolute_le_0_1c"]["passed"])
        self.assertFalse(checks["temperature_rmse_relative_le_5pct_or_absolute_le_0_1c"]["passed"])

    def test_off_lut_joint_direction_rule_and_saturation_has_no_hidden_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            joint_worse = self._collect(
                self._build_fixture(
                    Path(temp),
                    a3_off_lut_mean=3.1,
                    a3_off_lut_p95=8.1,
                )
            )
        self.assertEqual(joint_worse["normal_gates"]["status"], "failed")
        off_lut_gate = joint_worse["normal_gates"]["checks"][
            "no_simultaneous_material_off_lut_mean_and_p95_degradation"
        ]
        self.assertFalse(off_lut_gate["passed"])
        self.assertTrue(off_lut_gate["simultaneous_directional_worsening"])
        self.assertIn("zero magnitude tolerance", off_lut_gate["material_degradation_rule"])
        self.assertEqual(
            joint_worse["catastrophic_gates"]["status"], "not_detected"
        )
        self.assertEqual(
            joint_worse["continuation_decision"]["action"], "continue_to_next_scene"
        )

        with tempfile.TemporaryDirectory() as temp:
            # A max value near one is evidence to report, but the fixed audit
            # correctly uses the fraction collapsed at the endpoint.
            resolved = self._collect(
                self._build_fixture(
                    Path(temp),
                    a3_off_lut_mean=3.1,
                    a3_off_lut_p95=8.1,
                )
            )
        self.assertEqual(resolved["normal_gates"]["status"], "failed")
        self.assertEqual(resolved["catastrophic_gates"]["status"], "not_detected")
        self.assertEqual(
            resolved["continuation_decision"]["action"], "continue_to_next_scene"
        )
        self.assertTrue(resolved["continuation_decision"]["ordinary_failure_recorded"])
        self.assertGreater(
            resolved["opacity"]["60000"]["activated_opacity_a3"]["max"],
            0.999999,
        )

        with tempfile.TemporaryDirectory() as temp:
            # No magnitude epsilon is hidden in the rule, but both aggregates
            # must worsen.  An exactly unchanged P95 keeps the joint gate open.
            one_sided = self._collect(
                self._build_fixture(
                    Path(temp),
                    a3_off_lut_mean=3.000000001,
                    a3_off_lut_p95=8.0,
                )
            )
        self.assertTrue(
            one_sided["normal_gates"]["checks"]
            ["no_simultaneous_material_off_lut_mean_and_p95_degradation"]["passed"]
        )

        with tempfile.TemporaryDirectory() as temp:
            saturated = self._collect(
                self._build_fixture(Path(temp), saturation_detected=True)
            )
        self.assertTrue(
            saturated["catastrophic_gates"]["checks"]["opacity_saturation"]
            ["detected"]
        )
        self.assertEqual(saturated["continuation_decision"]["action"], "stop")

    def test_invariant_failure_and_nonfinite_are_catastrophic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            invariant = self._collect(
                self._build_fixture(Path(temp), structural=False)
            )
        self.assertFalse(invariant["invariants"]["passed"])
        self.assertTrue(
            invariant["catastrophic_gates"]["checks"]
            ["geometry_or_protocol_invariant_failure"]["detected"]
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            results_path = Path(fixture["a3_model_root"]) / "results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))
            results["ours_60000"]["PSNR"] = math.nan
            self._write_json(results_path, results, allow_nan=True)
            nonfinite = self._collect(fixture)
        self.assertIsNone(nonfinite["appearance"]["A3"]["60000"]["PSNR"])
        self.assertTrue(nonfinite["catastrophic_gates"]["checks"]["nan_or_inf"]["detected"])
        self.assertEqual(nonfinite["continuation_decision"]["action"], "stop")

    def test_cli_writes_json_csv_without_mutating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            input_paths = sorted(path for path in root.rglob("*") if path.is_file())
            before = {path: self._sha256(path) for path in input_paths}
            output_json = root / "outputs" / "scene_summary.json"
            output_csv = root / "outputs" / "scene_summary.csv"
            argv = [
                "--scene",
                str(fixture["scene"]),
                "--l-model-root",
                str(fixture["l_model_root"]),
                "--a3-model-root",
                str(fixture["a3_model_root"]),
                "--evaluation-root",
                str(fixture["evaluation_root"]),
                "--a3-audit-root",
                str(fixture["a3_audit_root"]),
                "--l-efficiency-root",
                str(fixture["l_efficiency_root"]),
                "--a3-efficiency-root",
                str(fixture["a3_efficiency_root"]),
                "--bound-split",
                str(fixture["bound_split"]),
                "--rgb-anchor-depth",
                str(fixture["rgb_anchor_depth"]),
                "--block-analysis",
                str(fixture["block_analysis"]),
                "--manual-assessment",
                str(fixture["manual_assessment"]),
                "--next-scene",
                str(fixture["next_scene"]),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
            self.assertEqual(collect_formal_l_a3_scene_results.main(argv), 0)

            after = {path: self._sha256(path) for path in input_paths}
            self.assertEqual(before, after)
            written = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], collect_formal_l_a3_scene_results.SCHEMA)
            with output_csv.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["category"] == "normal_gate" for row in rows))
            self.assertTrue(any(row["category"] == "catastrophic_gate" for row in rows))
            self.assertTrue(any(row["category"] == "block_delta" for row in rows))

    def test_block_analysis_must_be_complete_l_a3_60k(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            path = Path(fixture["block_analysis"])
            block = json.loads(path.read_text(encoding="utf-8"))
            block["protocol"]["groups"] = ["F3", "A3"]
            self._write_json(path, block)
            with self.assertRaisesRegex(
                collect_formal_l_a3_scene_results.SceneCollectionError,
                "block-analysis groups must contain unique L and A3",
            ):
                self._collect(fixture)

    def test_building_legacy_superset_is_normalized_to_l_a3_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "building_five_blocks_60k.json"
            expected_inputs: dict[str, Path] = {}
            declared_inputs: dict[str, dict[str, str]] = {}
            for label in (
                "bound_split",
                "per_view:L",
                "per_view:A3",
                "temperature:L",
                "temperature:A3",
            ):
                source = root / (label.replace(":", "_") + ".json")
                self._write_json(source, {"source": label})
                expected_inputs[label] = source
                declared_inputs[label] = {
                    "path": f"/legacy/building/{source.name}",
                    "sha256": self._sha256(source),
                }
            blocks = []
            for block_index in range(5):
                comparison = {
                    "combined_non_degraded": True,
                    "metrics": {
                        "PSNR": {
                            "raw_a3_minus_baseline": 0.01,
                            "classification": "tied",
                        }
                    },
                    "diagnostic_deltas": {},
                }
                blocks.append(
                    {
                        "block": {
                            "strip_id": f"tg-{block_index:04d}",
                            "block_index": block_index,
                            "stratum": "nadir:p-090",
                            "size": 16,
                            "views": [
                                f"{block_index * 16 + offset:04d}.png"
                                for offset in range(16)
                            ],
                        },
                        "group_means": {
                            group: {"PSNR": 20.0}
                            for group in ("L", "C3", "F3", "A3")
                        },
                        "paired_comparisons": {
                            "A3_minus_C3": comparison,
                            "A3_minus_F3": comparison,
                            "A3_minus_L": comparison,
                        },
                    }
                )
            self._write_json(
                path,
                {
                    "schema": "uav-tgs-formal-test-block-analysis-v1",
                    "status": "complete",
                    "iteration": 60000,
                    "protocol": {
                        "groups": ["L", "C3", "F3", "A3"],
                        "test_views": 80,
                        "block_count": 5,
                        "views_per_block": 16,
                    },
                    "classification_counts": {
                        "A3_minus_C3": {"PSNR": {"tied": 5}},
                        "A3_minus_F3": {"PSNR": {"tied": 5}},
                        "A3_minus_L": {"PSNR": {"tied": 5}},
                    },
                    "inputs": declared_inputs,
                    "blocks": blocks,
                },
            )

            normalized = (
                collect_formal_l_a3_scene_results._validate_block_analysis(
                    path, expected_inputs=expected_inputs
                )
            )

        self.assertEqual(normalized["protocol"]["groups"], ["L", "A3"])
        self.assertEqual(
            normalized["protocol"]["source_groups"], ["L", "C3", "F3", "A3"]
        )
        self.assertEqual(set(normalized["classification_counts"]), {"A3_minus_L"})
        self.assertEqual(
            set(normalized["collector_input_binding"]), set(expected_inputs)
        )
        self.assertEqual(len(normalized["blocks"]), 5)
        for item in normalized["blocks"]:
            self.assertEqual(set(item["group_means"]), {"L", "A3"})
            self.assertEqual(set(item["paired_comparisons"]), {"A3_minus_L"})

    def test_temperature_count_must_match_formal_block_test_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            path = (
                Path(fixture["evaluation_root"])
                / "temperature"
                / "A3_50000.json"
            )
            report = json.loads(path.read_text(encoding="utf-8"))
            report["summary"]["evaluated_file_count"] = 63
            self._write_json(path, report)
            with self.assertRaisesRegex(
                collect_formal_l_a3_scene_results.SceneCollectionError,
                "evaluated-file count does not match",
            ):
                self._collect(fixture)

    def test_block_analysis_sha_binding_rejects_same_count_wrong_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            path = Path(fixture["block_analysis"])
            block = json.loads(path.read_text(encoding="utf-8"))
            # The report keeps the same complete 64-view shape; only its
            # verified A3 temperature identity is wrong.
            block["inputs"]["temperature:A3"]["sha256"] = "0" * 64
            self._write_json(path, block)
            with self.assertRaisesRegex(
                collect_formal_l_a3_scene_results.SceneCollectionError,
                "block input SHA-256 mismatch for temperature:A3",
            ):
                self._collect(fixture)

    def test_block_analysis_cannot_redeclare_non_16_view_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = self._build_fixture(root)
            path = Path(fixture["block_analysis"])
            block = json.loads(path.read_text(encoding="utf-8"))
            block["protocol"]["views_per_block"] = 15
            block["protocol"]["test_views"] = 60
            for item in block["blocks"]:
                item["block"]["size"] = 15
                item["block"]["views"] = item["block"]["views"][:15]
            self._write_json(path, block)
            with self.assertRaisesRegex(
                collect_formal_l_a3_scene_results.SceneCollectionError,
                "exactly 16 views per block",
            ):
                self._collect(fixture)


if __name__ == "__main__":
    unittest.main()
