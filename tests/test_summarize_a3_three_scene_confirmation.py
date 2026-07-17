from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools import summarize_a3_three_scene_confirmation as summary


class ThreeSceneConfirmationTests(unittest.TestCase):
    BASE_L = {
        "PSNR": 20.0,
        "SSIM": 0.700,
        "LPIPS": 0.300,
        "EdgeHaloScore": 0.004,
        "temperature_mae_c": 1.0,
        "temperature_rmse_c": 1.4,
        "temperature_bias_c": -0.1,
        "temperature_abs_bias_c": 0.1,
        "temperature_p95_c": 2.8,
        "temperature_clipping_ratio": 0.001,
        "off_lut_mean_rgb_distance": 5.0,
        "off_lut_p95_rgb_distance": 20.0,
        "front_intrusion_1m": 0.08,
        "front_intrusion_2m": 0.04,
        "front_intrusion_5m": 0.02,
        "depth_agreement_1m": 0.65,
        "depth_agreement_2m": 0.80,
        "depth_agreement_5m": 0.93,
        "behind_1m": 0.25,
        "behind_2m": 0.14,
        "behind_5m": 0.05,
        "depth_mean_m": 1.7,
        "depth_median_m": 0.55,
        "missing_rate": 0.003,
    }

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _payload(
        self,
        scene: str,
        deltas: dict[str, float] | None = None,
        *,
        counts: dict[str, int] | None = None,
        invariant_passed: bool = True,
        material_off_lut_degradation: bool = False,
        opacity_saturation: bool = False,
        pipeline_reference_irreparable: bool = False,
    ) -> dict[str, object]:
        deltas = deltas or {}
        a3 = {
            metric: value + deltas.get(metric, 0.0)
            for metric, value in self.BASE_L.items()
        }
        return {
            "schema": summary.INPUT_SCHEMA,
            "status": "complete",
            "scene": scene,
            "counts": counts
            or {"train_views": 100, "test_views": 16, "guard_views": 4, "test_blocks": 1},
            "methods": {
                "L": {"metrics": dict(self.BASE_L)},
                "A3": {"metrics": a3},
            },
            "invariants": {
                "all_passed": invariant_passed,
                "checks": {
                    "spatial_fields_exact": invariant_passed,
                    "topology_exact": invariant_passed,
                    "optimizer_groups_exact": invariant_passed,
                },
            },
            "reviews": {
                "material_off_lut_degradation": material_off_lut_degradation,
                "opacity_saturation": opacity_saturation,
                "pipeline_reference_irreparable": pipeline_reference_irreparable,
            },
        }

    def _three_json_inputs(
        self,
        root: Path,
        overrides: dict[str, dict[str, object]] | None = None,
    ) -> dict[str, Path]:
        overrides = overrides or {}
        result: dict[str, Path] = {}
        for index, scene in enumerate(summary.EXPECTED_SCENES, start=1):
            kwargs = dict(overrides.get(scene, {}))
            kwargs.setdefault(
                "counts",
                {
                    "train_views": index * 100,
                    "test_views": index * 16,
                    "guard_views": index * 4,
                    "test_blocks": index,
                },
            )
            payload = self._payload(scene, **kwargs)
            path = root / f"{scene}.json"
            self._write_json(path, payload)
            result[scene] = path
        return result

    def test_freezes_with_two_joint_scene_passes_and_unweighted_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(
                root,
                {
                    "Building": {
                        "deltas": {
                            "PSNR": 0.2,
                            "SSIM": 0.003,
                            "LPIPS": -0.004,
                            "temperature_mae_c": -0.10,
                            "front_intrusion_1m": -0.01,
                        }
                    },
                    "InternalRoad": {"deltas": {}},
                    "Urban20K": {
                        "deltas": {
                            "PSNR": -0.20,
                            "SSIM": -0.007,
                            "LPIPS": 0.009,
                            "temperature_mae_c": 0.15,
                            "front_intrusion_1m": 0.025,
                            "missing_rate": 0.001,
                        }
                    },
                },
            )
            payload = summary.summarize(inputs)

            self.assertTrue(payload["freeze_decision"]["freeze_a3_recipe"])
            self.assertEqual(
                payload["confirmation_counts"][
                    "appearance_and_temperature_improved_or_tied_scenes"
                ],
                2,
            )
            self.assertEqual(payload["confirmation_counts"]["invariant_passed_scenes"], 3)
            self.assertTrue(payload["scene_macro"]["non_inferior"])
            self.assertAlmostEqual(payload["scene_macro"]["A3_minus_L"]["PSNR"], 0.0)
            self.assertAlmostEqual(
                payload["scene_macro"]["A3_minus_L"]["temperature_mae_c"],
                ( -0.10 + 0.0 + 0.15) / 3.0,
            )
            self.assertEqual(
                payload["scene_counts"]["descriptive_sum_for_common_count_fields"][
                    "test_views"
                ],
                96,
            )
            self.assertIn("do not establish Gaussian-center", payload["claim_boundaries"]["geometry"])
            self.assertIn(
                "not absolute thermometry", payload["claim_boundaries"]["temperature"]
            )

    def test_view_counts_do_not_weight_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(
                root,
                {
                    "Building": {
                        "deltas": {"PSNR": 3.0},
                        "counts": {"test_views": 1, "test_blocks": 1},
                    },
                    "InternalRoad": {
                        "deltas": {"PSNR": 0.0},
                        "counts": {"test_views": 100, "test_blocks": 1},
                    },
                    "Urban20K": {
                        "deltas": {"PSNR": 0.0},
                        "counts": {"test_views": 1000, "test_blocks": 1},
                    },
                },
            )
            payload = summary.summarize(inputs)
            self.assertAlmostEqual(payload["scene_macro"]["A3_minus_L"]["PSNR"], 1.0)

    def test_only_one_joint_scene_pass_blocks_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bad = {
                "deltas": {
                    "PSNR": -0.16,
                    "temperature_mae_c": 0.11,
                }
            }
            inputs = self._three_json_inputs(
                Path(temp), {"InternalRoad": bad, "Urban20K": bad}
            )
            payload = summary.summarize(inputs)
            self.assertEqual(
                payload["confirmation_counts"][
                    "appearance_and_temperature_improved_or_tied_scenes"
                ],
                1,
            )
            self.assertFalse(payload["freeze_decision"]["freeze_a3_recipe"])
            self.assertFalse(
                payload["freeze_decision"]["required_gates"][
                    "at_least_2_of_3_appearance_and_temperature_improved_or_tied"
                ]
            )

    def test_macro_non_inferiority_is_independent_required_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(
                Path(temp),
                {
                    "Urban20K": {
                        "deltas": {
                            "PSNR": -0.50,
                            "front_intrusion_1m": 0.025,
                        }
                    }
                },
            )
            payload = summary.summarize(inputs)
            self.assertEqual(
                payload["confirmation_counts"][
                    "appearance_and_temperature_improved_or_tied_scenes"
                ],
                2,
            )
            self.assertFalse(payload["scene_macro"]["non_inferior"])
            self.assertFalse(payload["freeze_decision"]["freeze_a3_recipe"])

    def test_any_invariant_failure_blocks_three_of_three_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(
                Path(temp), {"InternalRoad": {"invariant_passed": False}}
            )
            payload = summary.summarize(inputs)
            self.assertEqual(payload["confirmation_counts"]["invariant_passed_scenes"], 2)
            self.assertFalse(
                payload["freeze_decision"]["required_gates"][
                    "all_3_scene_invariants_passed"
                ]
            )
            self.assertFalse(payload["freeze_decision"]["freeze_a3_recipe"])

    def test_catastrophic_front_intrusion_blocks_freeze_but_boundary_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            boundary_inputs = self._three_json_inputs(
                Path(temp), {"Urban20K": {"deltas": {"front_intrusion_1m": 0.03}}}
            )
            boundary = summary.summarize(boundary_inputs)
            self.assertFalse(
                boundary["scenes"]["Urban20K"]["catastrophic_checks"][
                    "front_intrusion_1m_increase_gt_0_03"
                ]
            )

        with tempfile.TemporaryDirectory() as temp:
            catastrophic_inputs = self._three_json_inputs(
                Path(temp), {"Urban20K": {"deltas": {"front_intrusion_1m": 0.03001}}}
            )
            catastrophic = summary.summarize(catastrophic_inputs)
            self.assertEqual(
                catastrophic["freeze_decision"]["catastrophic_visibility_scenes"],
                ["Urban20K"],
            )
            self.assertFalse(catastrophic["freeze_decision"]["freeze_a3_recipe"])

    def test_temperature_or_rule_and_material_off_lut_review(self) -> None:
        check = summary._temperature_mae_check(10.0, 10.4)
        self.assertFalse(check["absolute_rule_passed"])
        self.assertTrue(check["relative_rule_passed"])
        self.assertTrue(check["passed"])

        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(
                Path(temp),
                {
                    "InternalRoad": {
                        "deltas": {
                            "off_lut_mean_rgb_distance": 1.0,
                            "off_lut_p95_rgb_distance": 1.0,
                        },
                        "material_off_lut_degradation": True,
                    }
                },
            )
            payload = summary.summarize(inputs)
            result = payload["scenes"]["InternalRoad"]
            self.assertTrue(result["reviews"]["both_off_lut_metrics_worse_directionally"])
            self.assertFalse(
                result["scene_classification"][
                    "temperature_improved_or_tied_under_frozen_tolerance"
                ]
            )

    def test_temperature_scene_and_macro_require_both_mae_and_rmse_or_rules(self) -> None:
        relative_rmse = summary._temperature_error_check(
            10.0, 10.4, metric="rmse"
        )
        self.assertFalse(relative_rmse["absolute_rule_passed"])
        self.assertTrue(relative_rmse["relative_rule_passed"])
        self.assertTrue(relative_rmse["passed"])

        with tempfile.TemporaryDirectory() as temp:
            # MAE is unchanged, but RMSE exceeds both its absolute and relative
            # bands in one scene.  That scene cannot cast a temperature pass.
            inputs = self._three_json_inputs(
                Path(temp),
                {"InternalRoad": {"deltas": {"temperature_rmse_c": 0.11}}},
            )
            payload = summary.summarize(inputs)
            internal = payload["scenes"]["InternalRoad"]
            self.assertTrue(
                internal["normal_gate_checks"]["temperature_mae_c"]["passed"]
            )
            self.assertFalse(
                internal["normal_gate_checks"]["temperature_rmse_c"]["passed"]
            )
            self.assertFalse(
                internal["scene_classification"][
                    "temperature_improved_or_tied_under_frozen_tolerance"
                ]
            )
            self.assertIn(
                "temperature_rmse_c", payload["scene_macro"]["normal_gate_checks"]
            )

        absolute_boundary = summary._temperature_error_check(
            1.0, 1.1, metric="rmse"
        )
        self.assertTrue(absolute_boundary["absolute_rule_passed"])
        self.assertTrue(absolute_boundary["passed"])

    def test_nonvisibility_catastrophic_conditions_are_reported_not_fifth_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(
                Path(temp), {"Urban20K": {"opacity_saturation": True}}
            )
            payload = summary.summarize(inputs)
            diagnostic = payload["freeze_decision"]["non_gate_catastrophic_diagnostics"]
            self.assertEqual(
                diagnostic["scenes_by_condition"]["opacity_saturation"], ["Urban20K"]
            )
            self.assertIn("not an additional final freeze gate", diagnostic["role"])
            self.assertTrue(payload["freeze_decision"]["freeze_a3_recipe"])
            self.assertEqual(len(payload["freeze_decision"]["required_gates"]), 4)

    def test_json_contract_fails_closed_on_nonfinite_and_inconsistent_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["Building"].read_text(encoding="utf-8"))
            payload["methods"]["A3"]["metrics"]["PSNR"] = math.nan
            self._write_json(inputs["Building"], payload)
            with self.assertRaisesRegex(summary.ThreeSceneSummaryError, "non-finite"):
                summary.summarize(inputs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["Building"].read_text(encoding="utf-8"))
            payload["invariants"]["all_passed"] = False
            self._write_json(inputs["Building"], payload)
            with self.assertRaisesRegex(summary.ThreeSceneSummaryError, "disagrees"):
                summary.summarize(inputs)

    def test_metric_contract_is_complete_and_macro_uses_the_fixed_full_set(self) -> None:
        self.assertEqual(tuple(self.BASE_L), summary.METHOD_METRICS)
        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(Path(temp))
            payload = summary.summarize(inputs)

        expected = set(summary.METHOD_METRICS)
        self.assertEqual(set(payload["scene_macro"]["metrics_included"]), expected)
        self.assertEqual(set(payload["scene_macro"]["A3_minus_L"]), expected)
        for method in summary.METHODS:
            self.assertEqual(set(payload["scene_macro"]["methods"][method]), expected)
        for scene in summary.EXPECTED_SCENES:
            for method in summary.METHODS:
                self.assertEqual(
                    set(payload["scenes"][scene]["metrics"][method]), expected
                )

    def test_missing_collector_metric_in_one_scene_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["InternalRoad"].read_text(encoding="utf-8"))
            for method in summary.METHODS:
                del payload["methods"][method]["metrics"]["EdgeHaloScore"]
            self._write_json(inputs["InternalRoad"], payload)

            with self.assertRaisesRegex(
                summary.ThreeSceneSummaryError,
                r"InternalRoad/L metric set.*missing=\['EdgeHaloScore'\]",
            ):
                summary.summarize(inputs)

    def test_method_metric_set_mismatch_and_unexpected_metric_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["Building"].read_text(encoding="utf-8"))
            del payload["methods"]["A3"]["metrics"]["temperature_p95_c"]
            self._write_json(inputs["Building"], payload)

            with self.assertRaisesRegex(
                summary.ThreeSceneSummaryError,
                r"Building L and A3 metric sets must match exactly.*temperature_p95_c",
            ):
                summary.summarize(inputs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["Urban20K"].read_text(encoding="utf-8"))
            for method in summary.METHODS:
                payload["methods"][method]["metrics"]["unexpected_metric"] = 0.0
            self._write_json(inputs["Urban20K"], payload)

            with self.assertRaisesRegex(
                summary.ThreeSceneSummaryError,
                r"Urban20K/L metric set.*extra=\['unexpected_metric'\]",
            ):
                summary.summarize(inputs)

    def _write_csv_summary(self, path: Path, payload: dict[str, object]) -> None:
        metrics = payload["methods"]
        counts = payload["counts"]
        checks = payload["invariants"]["checks"]
        fields = [
            "schema",
            "status",
            "scene",
            "method",
            "invariants_passed",
            "material_off_lut_degradation",
            "opacity_saturation",
            "pipeline_reference_irreparable",
            *[f"count__{key}" for key in sorted(counts)],
            *[f"invariant__{key}" for key in sorted(checks)],
            *[f"metric__{key}" for key in sorted(self.BASE_L)],
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for method in summary.METHODS:
                row = {
                    "schema": payload["schema"],
                    "status": payload["status"],
                    "scene": payload["scene"],
                    "method": method,
                    "invariants_passed": payload["invariants"]["all_passed"],
                    **payload["reviews"],
                    **{f"count__{key}": value for key, value in counts.items()},
                    **{f"invariant__{key}": value for key, value in checks.items()},
                    **{
                        f"metric__{key}": value
                        for key, value in metrics[method]["metrics"].items()
                    },
                }
                writer.writerow(row)

    def test_csv_inputs_cli_write_json_and_long_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs: dict[str, Path] = {}
            for scene in summary.EXPECTED_SCENES:
                payload = self._payload(scene, {"PSNR": 0.1})
                path = root / f"{scene}.csv"
                self._write_csv_summary(path, payload)
                inputs[scene] = path
            output = root / "three_scene.json"
            csv_output = root / "three_scene.csv"
            argv: list[str] = []
            for scene in summary.EXPECTED_SCENES:
                argv.extend(["--scene-summary", f"{scene}={inputs[scene]}"])
            argv.extend(["--output", str(output), "--csv", str(csv_output)])

            self.assertEqual(summary.main(argv), 0)
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["schema"], summary.OUTPUT_SCHEMA)
            self.assertTrue(written["freeze_decision"]["freeze_a3_recipe"])
            with csv_output.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            metric_count = len(self.BASE_L)
            self.assertEqual(len(rows), metric_count * 4)
            self.assertEqual(sum(row["row_type"] == "scene_macro" for row in rows), metric_count)

    def test_scene_assignment_set_and_embedded_scene_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            inputs = self._three_json_inputs(Path(temp))
            del inputs["Urban20K"]
            with self.assertRaisesRegex(summary.ThreeSceneSummaryError, "scene inputs must be exactly"):
                summary.summarize(inputs)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = self._three_json_inputs(root)
            payload = json.loads(inputs["Urban20K"].read_text(encoding="utf-8"))
            payload["scene"] = "InternalRoad"
            self._write_json(inputs["Urban20K"], payload)
            with self.assertRaisesRegex(summary.ThreeSceneSummaryError, "scene assignment mismatch"):
                summary.summarize(inputs)


if __name__ == "__main__":
    unittest.main()
