import csv
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from PIL import Image

from tools.export_reassessment_pareto import (
    HOTSPOT_DISPLAY_DOMAIN,
    HOTSPOT_THRESHOLD_BINS,
    HOTSPOT_THRESHOLD_RULE,
    METRIC_DIRECTIONS,
    NOOP_ALIAS_FIGURE_NOTE,
    PARETO_SPECS,
    RENDER_TIMING_COLUMN,
    TRAIN_TIMING_COLUMN,
    build_macro_rows,
    build_worst_rows,
    compute_memberships,
    export_reassessment,
    load_normalized_csvs,
    resolve_macro_scenes,
    scene_visual_encodings,
    scene_rows_for_figure,
    sha256_file,
    verify_manifest_self_hash,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def endpoint_row(scene: str, method_id: str, **overrides):
    row = {
        "schema_version": "endpoint-summary-v1",
        "scene": scene,
        "method_id": method_id,
        "display_name": method_id.upper(),
        "geometry_endpoint_id": f"{method_id}-geometry",
        "rgb_endpoint_id": f"{method_id}-rgb",
        "thermal_endpoint_id": f"{method_id}-thermal",
        "depth_source_id": f"{method_id}-depth",
        "formal_split_sha256": SHA_A,
        "reference_manifest_sha256": SHA_A,
        "evidence_status": "formal",
        "rgb_psnr_db": 30.0,
        "rgb_ssim": 0.91,
        "rgb_lpips": 0.12,
        "thermal_psnr_db": 29.0,
        "thermal_ssim": 0.89,
        "thermal_lpips": 0.20,
        "temp_mae_direct_c": 0.8,
        "temp_rmse_direct_c": 1.1,
        "temp_bias_direct_c": -0.1,
        "temp_p95_direct_c": 2.0,
        "temp_mae_display_c": 1.0,
        "temp_rmse_display_c": 1.3,
        "temp_bias_display_c": 0.1,
        "temp_p95_display_c": 2.4,
        "hotspot_auprc_display": 0.80,
        "hotspot_iou_display": 0.65,
        "hotspot_metric_domain": HOTSPOT_DISPLAY_DOMAIN,
        "hotspot_threshold_rule": HOTSPOT_THRESHOLD_RULE,
        "hotspot_threshold_histogram_bins": HOTSPOT_THRESHOLD_BINS,
        "hotspot_threshold_sha256": SHA_A,
        "front_auc_expected": 0.10,
        "front_auc_median": 0.08,
        "front_auc_max_contribution": 0.06,
        "agreement_auc_expected": 0.82,
        "agreement_auc_median": 0.84,
        "agreement_auc_max_contribution": 0.86,
        "gaussian_count": 1_000_000,
        "train_wall_time_s": 3600.0,
        "render_ms_per_view": 10.0,
        "render_width": 640,
        "render_height": 480,
        "train_timing_scope": "full_training_wall_clock",
        "render_timing_scope": "pure_render_cuda_event",
        "structural_metadata_json": json.dumps({"family": method_id}),
        "appearance_source_sha256": SHA_A,
        "temperature_source_sha256": SHA_A,
        "hotspot_source_sha256": SHA_A,
        "depth_source_sha256": SHA_A,
        "efficiency_source_sha256": SHA_A,
        "alias_receipt_sha256": SHA_A,
    }
    row.update(overrides)
    return row


def write_rows(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path, rows):
    write_rows(path, rows)
    return load_normalized_csvs([path]).rows


def assert_no_design_markers(testcase: unittest.TestCase, value):
    markers = ("feasible", "infeasible", "feasibility")
    if isinstance(value, dict):
        for key, nested in value.items():
            testcase.assertFalse(any(marker in str(key).lower() for marker in markers))
            assert_no_design_markers(testcase, nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            assert_no_design_markers(testcase, nested)
    elif isinstance(value, str):
        testcase.assertFalse(any(marker in value.lower() for marker in markers))


class ReassessmentParetoTests(unittest.TestCase):
    def test_scene_figure_consolidates_explicit_noop_alias_with_present_source(self):
        source = endpoint_row("Building", "raw_f3")
        alias = endpoint_row(
            "Building",
            "scsp_f3",
            structural_metadata_json=json.dumps(
                {
                    "alias_kind": "explicit_noop",
                    "alias_source_method_id": "raw_f3",
                }
            ),
        )
        orphan = endpoint_row(
            "Building",
            "orphan_alias",
            structural_metadata_json=json.dumps(
                {
                    "alias_kind": "explicit_noop",
                    "alias_source_method_id": "missing_source",
                }
            ),
        )
        plotted, consolidated = scene_rows_for_figure([source, alias, orphan])
        self.assertEqual([row["method_id"] for row in plotted], ["raw_f3", "orphan_alias"])
        self.assertEqual(
            consolidated,
            [
                {
                    "scene": "Building",
                    "alias_method_id": "scsp_f3",
                    "source_method_id": "raw_f3",
                }
            ],
        )
        inconsistent = dict(alias)
        inconsistent["thermal_lpips"] = 0.99
        with self.assertRaisesRegex(ValueError, "explicit no-op figure alias differs"):
            scene_rows_for_figure([source, inconsistent])

    def test_input_rejects_feasibility_nonfinite_and_invalid_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            feasible = endpoint_row("A", "m1", isFeasible="true")
            feasible_path = root / "feasible.csv"
            write_rows(feasible_path, [feasible])
            with self.assertRaisesRegex(ValueError, "forbidden feasibility column"):
                load_normalized_csvs([feasible_path])

            nonfinite_path = root / "nonfinite.csv"
            write_rows(nonfinite_path, [endpoint_row("A", "m1", thermal_lpips="NaN")])
            with self.assertRaisesRegex(ValueError, "non-finite thermal_lpips"):
                load_normalized_csvs([nonfinite_path])

            bad_sha_path = root / "bad_sha.csv"
            write_rows(bad_sha_path, [endpoint_row("A", "m1", depth_source_sha256="not-a-hash")])
            with self.assertRaisesRegex(ValueError, "invalid SHA-256 in depth_source_sha256"):
                load_normalized_csvs([bad_sha_path])

    def test_display_hotspot_is_preserved_and_generic_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "alias.csv"
            source = endpoint_row(
                "A",
                "m1",
                hotspot_auprc_display=0.73,
                depth_source_sha256=SHA_A.upper(),
            )
            source.pop("evidence_status")
            loaded = load_rows(
                input_path,
                [source],
            )
            self.assertEqual(loaded[0]["hotspot_auprc_display"], 0.73)
            self.assertEqual(loaded[0]["depth_source_sha256"], SHA_A)
            self.assertEqual(loaded[0]["evidence_status"], "")

            generic_path = root / "generic_hotspot.csv"
            generic = endpoint_row("A", "m1")
            generic["HOTSPOT_AUPRC"] = generic.pop("hotspot_auprc_display")
            write_rows(
                generic_path,
                [generic],
            )
            with self.assertRaisesRegex(ValueError, "generic hotspot column"):
                load_normalized_csvs([generic_path])

    def test_missing_axis_is_excluded_without_imputation_or_metadata_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "missing.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row(
                        "A",
                        "m1",
                        evidence_status="legacy_rejected_label_is_metadata",
                        structural_metadata_json='{"family":"legacy"}',
                        front_auc_expected=0.05,
                        thermal_lpips=0.10,
                    ),
                    endpoint_row(
                        "A",
                        "m2",
                        temp_mae_display_c="null",
                        front_auc_expected=0.20,
                        thermal_lpips=0.30,
                    ),
                ],
            )

            self.assertIsNone(rows[1]["temp_mae_display_c"])
            memberships, exclusions = compute_memberships(rows, level="scene")

            appearance = [
                row
                for row in memberships
                if row["pareto_id"] == "appearance_vs_front_expected"
            ]
            self.assertEqual({row["method_id"] for row in appearance}, {"m1", "m2"})
            self.assertTrue(next(row for row in appearance if row["method_id"] == "m1")["is_member"])

            temperature_ids = {
                row["pareto_id"]
                for row in memberships
                if row["method_id"] == "m2" and row["pareto_id"].startswith("temperature_vs_front_")
            }
            self.assertEqual(temperature_ids, set())
            missing = [
                row
                for row in exclusions
                if row["method_id"] == "m2"
                and row["pareto_id"] in {"temperature_vs_front_expected", "hotspot_vs_temperature"}
            ]
            self.assertEqual(len(missing), 2)
            self.assertTrue(all(row["exclusion_reason"] == "missing_axis" for row in missing))
            self.assertTrue(all("temp_mae_display_c" in row["missing_fields"] for row in missing))

    def test_structural_metadata_rejects_nested_feasibility_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "nested_feasibility.csv"
            write_rows(
                input_path,
                [
                    endpoint_row(
                        "A",
                        "m1",
                        structural_metadata_json=(
                            '{"family":"legacy","audit":{"isFeasible":false}}'
                        ),
                    )
                ],
            )
            with self.assertRaisesRegex(ValueError, "forbidden design marker"):
                load_normalized_csvs([input_path])

    def test_all_metadata_values_including_status_and_extras_reject_design_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ("status.csv", {"evidence_status": "method_infeasible"}),
                ("extra.csv", {"review_note": "declared feasible"}),
                (
                    "structural_value.csv",
                    {"structural_metadata_json": '{"audit":[{"label":"feasibility"}]}'},
                ),
            )
            for name, overrides in cases:
                path = root / name
                write_rows(path, [endpoint_row("A", "m1", **overrides)])
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "forbidden design marker"
                ):
                    load_normalized_csvs([path])

    def test_metric_provenance_and_hotspot_protocol_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_source_cases = (
                "appearance_source_sha256",
                "temperature_source_sha256",
                "hotspot_source_sha256",
                "efficiency_source_sha256",
                "formal_split_sha256",
                "reference_manifest_sha256",
                "depth_source_sha256",
            )
            for field in missing_source_cases:
                path = root / f"missing_{field}.csv"
                write_rows(path, [endpoint_row("Building", "m1", **{field: ""})])
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError, "missing or invalid metric provenance"
                ):
                    load_normalized_csvs([path])

            bad_hotspot_contracts = (
                {"hotspot_threshold_sha256": ""},
                {"hotspot_metric_domain": "direct_temperature_c"},
                {"hotspot_threshold_rule": "test_q95"},
                {"hotspot_threshold_histogram_bins": 4096},
            )
            for index, overrides in enumerate(bad_hotspot_contracts):
                path = root / f"bad_hotspot_{index}.csv"
                write_rows(path, [endpoint_row("Building", "m1", **overrides)])
                with self.subTest(overrides=overrides), self.assertRaisesRegex(
                    ValueError, "missing or invalid metric provenance"
                ):
                    load_normalized_csvs([path])

            good_path = root / "good.csv"
            rows = load_rows(
                good_path,
                [
                    endpoint_row("Building", "m1", hotspot_threshold_sha256=SHA_A),
                    endpoint_row("InternalRoad", "m1", hotspot_threshold_sha256=SHA_B),
                ],
            )
            macro_rows, _ = build_macro_rows(rows, ("Building", "InternalRoad"))
            self.assertEqual(len(macro_rows), 1)
            self.assertAlmostEqual(macro_rows[0]["hotspot_auprc_display"], 0.8)
            memberships, exclusions = compute_memberships(
                macro_rows, level="macro", scene_label="MACRO"
            )
            self.assertTrue(
                any(row["pareto_id"] == "hotspot_vs_temperature" for row in memberships)
            )
            self.assertFalse(
                any(row["pareto_id"] == "hotspot_vs_temperature" for row in exclusions)
            )

            tampered = [dict(row) for row in rows]
            tampered[1]["temperature_source_sha256"] = ""
            macro_rows, _ = build_macro_rows(tampered, ("Building", "InternalRoad"))
            self.assertIsNone(macro_rows[0]["temp_mae_display_c"])

            macro_rows, _ = build_macro_rows(rows, ("Building", "InternalRoad"))
            macro_rows[0]["source_sha256_by_scene_json"] = json.dumps(
                {
                    "Building": {
                        "temperature_source_sha256": SHA_A,
                        "hotspot_source_sha256": SHA_A,
                        "hotspot_threshold_sha256": SHA_A,
                    },
                    "InternalRoad": {
                        "temperature_source_sha256": SHA_B,
                        "hotspot_source_sha256": SHA_B,
                    },
                }
            )
            memberships, exclusions = compute_memberships(
                macro_rows, level="macro", scene_label="MACRO"
            )
            self.assertFalse(
                any(row["pareto_id"] == "hotspot_vs_temperature" for row in memberships)
            )
            hotspot_exclusion = next(
                row
                for row in exclusions
                if row["pareto_id"] == "hotspot_vs_temperature"
            )
            self.assertEqual(hotspot_exclusion["exclusion_reason"], "missing_axis_provenance")
            self.assertIn(
                "hotspot_auprc_display:InternalRoad:hotspot_threshold_sha256",
                hotspot_exclusion["missing_fields"],
            )

            direct = [dict(rows[0])]
            direct[0]["appearance_source_sha256"] = ""
            memberships, exclusions = compute_memberships(direct, level="scene")
            self.assertFalse(
                any(row["pareto_id"].startswith("appearance_vs_front") for row in memberships)
            )
            self.assertTrue(
                all(
                    row["exclusion_reason"] == "missing_axis_provenance"
                    for row in exclusions
                    if row["pareto_id"].startswith("appearance_vs_front")
                )
            )

    def test_scene_visual_encoding_distinguishes_building_and_internalroad(self):
        encodings = scene_visual_encodings(("InternalRoad", "Building"))
        self.assertEqual(encodings["Building"], "filled_method_color")
        self.assertEqual(encodings["InternalRoad"], "hollow_method_color_edge")
        self.assertNotEqual(encodings["Building"], encodings["InternalRoad"])

    def test_membership_is_per_scene_and_separate_for_all_depth_definitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "scenes.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row(
                        "A",
                        "m1",
                        front_auc_expected=0.10,
                        front_auc_median=0.30,
                        front_auc_max_contribution=0.30,
                        thermal_lpips=0.20,
                        temp_mae_display_c=0.8,
                        hotspot_auprc_display=0.90,
                    ),
                    endpoint_row(
                        "A",
                        "m2",
                        front_auc_expected=0.20,
                        front_auc_median=0.10,
                        front_auc_max_contribution=0.10,
                        thermal_lpips=0.30,
                        temp_mae_display_c=1.0,
                        hotspot_auprc_display=0.80,
                    ),
                    endpoint_row(
                        "B",
                        "m1",
                        front_auc_expected=0.40,
                        front_auc_median=0.40,
                        front_auc_max_contribution=0.40,
                        thermal_lpips=0.40,
                        temp_mae_display_c=2.0,
                        hotspot_auprc_display=0.40,
                    ),
                    endpoint_row(
                        "B",
                        "m2",
                        front_auc_expected=0.10,
                        front_auc_median=0.10,
                        front_auc_max_contribution=0.10,
                        thermal_lpips=0.10,
                        temp_mae_display_c=0.5,
                        hotspot_auprc_display=0.95,
                    ),
                ],
            )
            memberships, exclusions = compute_memberships(rows, level="scene")
            self.assertEqual(exclusions, [])

            def members(scene, pareto_id):
                return {
                    row["method_id"]
                    for row in memberships
                    if row["scene"] == scene
                    and row["pareto_id"] == pareto_id
                    and row["is_member"]
                }

            self.assertEqual(members("A", "appearance_vs_front_expected"), {"m1"})
            self.assertEqual(members("B", "appearance_vs_front_expected"), {"m2"})
            self.assertEqual(members("A", "appearance_vs_front_median"), {"m1", "m2"})
            self.assertEqual(members("A", "appearance_vs_front_max_contribution"), {"m1", "m2"})
            self.assertEqual(members("A", "temperature_vs_front_expected"), {"m1"})
            self.assertEqual(members("A", "temperature_vs_front_median"), {"m1", "m2"})
            self.assertEqual(members("A", "temperature_vs_front_max_contribution"), {"m1", "m2"})
            self.assertEqual(members("A", "hotspot_vs_temperature"), {"m1"})
            self.assertEqual(members("B", "hotspot_vs_temperature"), {"m2"})

    def test_macro_is_complete_two_scene_unweighted_and_worst_is_directional(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "macro.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row(
                        "A",
                        "m1",
                        temp_mae_display_c=1.0,
                        hotspot_auprc_display=0.90,
                        front_auc_expected=0.20,
                        gaussian_count=100,
                    ),
                    endpoint_row(
                        "B",
                        "m1",
                        temp_mae_display_c=3.0,
                        hotspot_auprc_display=0.50,
                        front_auc_expected=0.40,
                        gaussian_count=300,
                    ),
                    endpoint_row("A", "m2", hotspot_auprc_display=0.70),
                    endpoint_row(
                        "B",
                        "m2",
                        hotspot_auprc_display="",
                        render_width=1280,
                        render_height=960,
                    ),
                    endpoint_row("A", "m3"),
                ],
            )

            macro_rows, coverage_exclusions = build_macro_rows(rows, ("A", "B"))
            self.assertEqual(resolve_macro_scenes(rows), ("A", "B"))
            third_scene = dict(rows[0], scene="C", method_id="m4", display_name="M4")
            self.assertEqual(resolve_macro_scenes([*rows, third_scene]), ())
            self.assertEqual(
                resolve_macro_scenes([*rows, third_scene], ("A", "B")),
                ("A", "B"),
            )
            self.assertEqual({row["method_id"] for row in macro_rows}, {"m1", "m2"})
            m1 = next(row for row in macro_rows if row["method_id"] == "m1")
            self.assertEqual(m1["scene_count"], 2)
            self.assertAlmostEqual(m1["temp_mae_display_c"], 2.0)
            self.assertAlmostEqual(m1["hotspot_auprc_display"], 0.70)
            self.assertAlmostEqual(m1["front_auc_expected"], 0.30)
            self.assertAlmostEqual(m1["gaussian_count"], 200.0)
            m2 = next(row for row in macro_rows if row["method_id"] == "m2")
            self.assertIsNone(m2["hotspot_auprc_display"])

            m3_exclusions = [row for row in coverage_exclusions if row["method_id"] == "m3"]
            self.assertEqual(len(m3_exclusions), len(PARETO_SPECS))
            self.assertTrue(all(row["exclusion_reason"] == "incomplete_two_scene_coverage" for row in m3_exclusions))
            self.assertTrue(all(row["missing_fields"] == "B" for row in m3_exclusions))

            _, axis_exclusions = compute_memberships(macro_rows, level="macro", scene_label="MACRO")
            hotspot_exclusion = next(
                row
                for row in axis_exclusions
                if row["method_id"] == "m2" and row["pareto_id"] == "hotspot_vs_temperature"
            )
            self.assertEqual(hotspot_exclusion["missing_fields"], "hotspot_auprc_display")
            render_exclusion = next(
                row
                for row in axis_exclusions
                if row["method_id"] == "m2" and row["pareto_id"] == "render_cost"
            )
            self.assertEqual(
                render_exclusion["exclusion_reason"],
                "noncomparable_render_context_across_scenes",
            )
            self.assertIn('"A":{"height":480', render_exclusion["details"])
            self.assertIn('"B":{"height":960', render_exclusion["details"])
            self.assertIn('"render_timing_scope":"pure_render_cuda_event"', render_exclusion["details"])

            worst = build_worst_rows(rows, ("A", "B"))
            temp_worst = next(
                row
                for row in worst
                if row["method_id"] == "m1" and row["metric"] == "temp_mae_display_c"
            )
            hotspot_worst = next(
                row
                for row in worst
                if row["method_id"] == "m1" and row["metric"] == "hotspot_auprc_display"
            )
            self.assertEqual((temp_worst["direction"], temp_worst["worst_scene"], temp_worst["worst_value"]), ("min", "B", 3.0))
            self.assertEqual((hotspot_worst["direction"], hotspot_worst["worst_scene"], hotspot_worst["worst_value"]), ("max", "B", 0.5))

    def test_render_membership_never_crosses_resolution_or_timing_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "render.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row("A", "m1", gaussian_count=100, render_ms_per_view=10, render_width=640, render_height=480, render_timing_scope="pure"),
                    endpoint_row("A", "m2", gaussian_count=200, render_ms_per_view=20, render_width=1280, render_height=960, render_timing_scope="pure"),
                    endpoint_row("A", "m3", gaussian_count=150, render_ms_per_view=15, render_width=640, render_height=480, render_timing_scope="pure"),
                    endpoint_row("A", "m4", gaussian_count=180, render_ms_per_view=18, render_width=640, render_height=480, render_timing_scope="end_to_end"),
                    endpoint_row("A", "m5", gaussian_count=90, render_ms_per_view=9, render_width=640, render_height=480, render_timing_scope=""),
                ],
            )
            memberships, exclusions = compute_memberships(rows, level="scene")
            render = {
                row["method_id"]: row
                for row in memberships
                if row["pareto_id"] == "render_cost"
            }
            self.assertTrue(render["m1"]["is_member"])
            self.assertTrue(render["m2"]["is_member"])
            self.assertFalse(render["m3"]["is_member"])
            self.assertTrue(render["m4"]["is_member"])
            self.assertNotEqual(render["m1"]["comparison_group"], render["m2"]["comparison_group"])
            self.assertNotEqual(render["m1"]["comparison_group"], render["m4"]["comparison_group"])
            missing = next(
                row
                for row in exclusions
                if row["method_id"] == "m5" and row["pareto_id"] == "render_cost"
            )
            self.assertEqual(missing["exclusion_reason"], "missing_render_comparability_metadata")
            self.assertEqual(missing["missing_fields"], "render_timing_scope")

    def test_independent_timing_scopes_and_legacy_render_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical_path = root / "canonical.csv"
            canonical = endpoint_row(
                "A",
                "m1",
                train_timing_scope="rgb_plus_thermal_training",
                render_timing_scope="renderer_only_cuda_event",
            )
            loaded = load_rows(canonical_path, [canonical])
            self.assertEqual(
                loaded[0][TRAIN_TIMING_COLUMN], "rgb_plus_thermal_training"
            )
            self.assertEqual(
                loaded[0][RENDER_TIMING_COLUMN], "renderer_only_cuda_event"
            )
            self.assertNotIn("timing_scope", loaded[0])

            legacy_path = root / "legacy.csv"
            legacy = endpoint_row("A", "m1")
            legacy.pop(RENDER_TIMING_COLUMN)
            legacy["timing_scope"] = "legacy_renderer_only"
            loaded_legacy = load_rows(legacy_path, [legacy])
            self.assertEqual(
                loaded_legacy[0][RENDER_TIMING_COLUMN], "legacy_renderer_only"
            )
            self.assertNotIn("timing_scope", loaded_legacy[0])

            conflict_path = root / "conflict.csv"
            conflict = endpoint_row(
                "A", "m1", render_timing_scope="renderer_only"
            )
            conflict["timing_scope"] = "end_to_end"
            write_rows(conflict_path, [conflict])
            with self.assertRaisesRegex(ValueError, "conflicting render_timing_scope"):
                load_normalized_csvs([conflict_path])

    def test_training_membership_is_stratified_by_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "training.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row(
                        "A",
                        "m1",
                        gaussian_count=100,
                        train_wall_time_s=10,
                        train_timing_scope="full_pipeline",
                    ),
                    endpoint_row(
                        "A",
                        "m2",
                        gaussian_count=50,
                        train_wall_time_s=5,
                        train_timing_scope="thermal_stage_only",
                    ),
                    endpoint_row(
                        "A",
                        "m3",
                        gaussian_count=200,
                        train_wall_time_s=20,
                        train_timing_scope="full_pipeline",
                    ),
                    endpoint_row(
                        "A",
                        "m4",
                        gaussian_count=25,
                        train_wall_time_s=2,
                        train_timing_scope="",
                    ),
                ],
            )
            memberships, exclusions = compute_memberships(rows, level="scene")
            training = {
                row["method_id"]: row
                for row in memberships
                if row["pareto_id"] == "training_cost"
            }
            self.assertTrue(training["m1"]["is_member"])
            self.assertTrue(training["m2"]["is_member"])
            self.assertFalse(training["m3"]["is_member"])
            self.assertNotEqual(
                training["m1"]["comparison_group"],
                training["m2"]["comparison_group"],
            )
            missing = next(
                row
                for row in exclusions
                if row["method_id"] == "m4" and row["pareto_id"] == "training_cost"
            )
            self.assertEqual(
                missing["exclusion_reason"],
                "missing_training_comparability_metadata",
            )
            self.assertEqual(missing["missing_fields"], TRAIN_TIMING_COLUMN)

    def test_macro_training_scope_mismatch_is_explicitly_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "macro_training.csv"
            rows = load_rows(
                input_path,
                [
                    endpoint_row(
                        "Building", "m1", train_timing_scope="full_pipeline"
                    ),
                    endpoint_row(
                        "InternalRoad", "m1", train_timing_scope="thermal_only"
                    ),
                    endpoint_row(
                        "Building", "m2", train_timing_scope="full_pipeline"
                    ),
                    endpoint_row(
                        "InternalRoad", "m2", train_timing_scope="full_pipeline"
                    ),
                ],
            )
            macro_rows, _ = build_macro_rows(rows, ("Building", "InternalRoad"))
            mismatch = next(row for row in macro_rows if row["method_id"] == "m1")
            matched = next(row for row in macro_rows if row["method_id"] == "m2")
            self.assertFalse(mismatch["train_context_consistent"])
            self.assertIsNone(mismatch["train_wall_time_s"])
            self.assertEqual(mismatch[TRAIN_TIMING_COLUMN], "")
            self.assertTrue(matched["train_context_consistent"])
            self.assertEqual(matched[TRAIN_TIMING_COLUMN], "full_pipeline")

            memberships, exclusions = compute_memberships(
                macro_rows, level="macro", scene_label="MACRO"
            )
            self.assertFalse(
                any(
                    row["method_id"] == "m1"
                    and row["pareto_id"] == "training_cost"
                    for row in memberships
                )
            )
            mismatch_exclusion = next(
                row
                for row in exclusions
                if row["method_id"] == "m1"
                and row["pareto_id"] == "training_cost"
            )
            self.assertEqual(
                mismatch_exclusion["exclusion_reason"],
                "noncomparable_training_scope_across_scenes",
            )
            self.assertIn("full_pipeline", mismatch_exclusion["details"])
            self.assertIn("thermal_only", mismatch_exclusion["details"])
            self.assertTrue(
                any(
                    row["method_id"] == "m2"
                    and row["pareto_id"] == "training_cost"
                    for row in memberships
                )
            )

    def test_end_to_end_outputs_hashes_and_python_figure_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "endpoint_summary.csv"
            rows = [
                endpoint_row(scene, method, **overrides)
                for scene, method, overrides in (
                    ("Building", "m1", {}),
                    ("Building", "m2", {"front_auc_expected": 0.14, "thermal_lpips": 0.17, "temp_mae_display_c": 1.2, "hotspot_auprc_display": 0.75}),
                    ("InternalRoad", "m1", {"front_auc_expected": 0.12, "thermal_lpips": 0.22, "temp_mae_display_c": 1.1, "hotspot_auprc_display": 0.78}),
                    ("InternalRoad", "m2", {"front_auc_expected": 0.09, "thermal_lpips": 0.25, "temp_mae_display_c": 0.9, "hotspot_auprc_display": 0.82}),
                )
            ]
            write_rows(input_path, rows)
            output = root / "export"
            returned = export_reassessment([input_path], output)

            expected_names = {
                "normalized.csv",
                "membership.csv",
                "macro.csv",
                "worst.csv",
                "exclusions.csv",
                "reassessment_pareto_4panel.svg",
                "reassessment_pareto_4panel.pdf",
                "reassessment_pareto_4panel.tiff",
                "reassessment_pareto_4panel.png",
                "manifest.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_names)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(returned, manifest)
            self.assertTrue(verify_manifest_self_hash(manifest))
            tampered = copy.deepcopy(manifest)
            tampered["row_counts"]["normalized"] = 999
            self.assertFalse(verify_manifest_self_hash(tampered))
            self.assertEqual(manifest["inputs"][0]["sha256"], sha256_file(input_path))
            self.assertEqual(Path(manifest["inputs"][0]["path"]), input_path.resolve())
            exporter_path = Path(__file__).parents[1] / "tools" / "export_reassessment_pareto.py"
            self.assertEqual(manifest["exporter"]["sha256"], sha256_file(exporter_path))
            self.assertEqual(manifest["exporter"]["size_bytes"], exporter_path.stat().st_size)
            self.assertEqual(manifest["row_counts"], {"normalized": 4, "membership": 54, "macro": 2, "worst": 46, "exclusions": 0})
            self.assertEqual(manifest["metric_directions"], METRIC_DIRECTIONS)
            self.assertTrue(manifest["policy"]["unconstrained"])
            self.assertEqual(
                manifest["policy"]["macro_scenes"], ["Building", "InternalRoad"]
            )
            self.assertEqual(
                manifest["hotspot_contract"],
                {
                    "metric": "hotspot_auprc_display",
                    "domain": HOTSPOT_DISPLAY_DOMAIN,
                    "threshold_rule": HOTSPOT_THRESHOLD_RULE,
                    "histogram_bins": HOTSPOT_THRESHOLD_BINS,
                    "threshold_hash_column": "hotspot_threshold_sha256",
                    "source_hash_column": "hotspot_source_sha256",
                },
            )
            self.assertEqual(
                manifest["figure_contract"]["visual_encodings"]["scene"],
                {
                    "Building": "filled_method_color",
                    "InternalRoad": "hollow_method_color_edge",
                },
            )
            self.assertEqual(
                manifest["figure_contract"]["protocol_noop_alias_rendering"]["scene_level"],
                NOOP_ALIAS_FIGURE_NOTE,
            )
            self.assertIn("train_timing_scope", manifest["policy"]["training_membership"])
            self.assertIn("render_timing_scope", manifest["policy"]["render_membership"])
            assert_no_design_markers(self, manifest)

            for name, receipt in manifest["outputs"].items():
                artifact = output / name
                self.assertEqual(receipt["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
                self.assertEqual(receipt["size_bytes"], artifact.stat().st_size)

            with (output / "normalized.csv").open("r", encoding="utf-8", newline="") as handle:
                normalized = list(csv.DictReader(handle))
            self.assertEqual(len(normalized), 4)
            self.assertIn("hotspot_auprc_display", normalized[0])
            self.assertNotIn("hotspot_auprc", normalized[0])
            self.assertIn(TRAIN_TIMING_COLUMN, normalized[0])
            self.assertIn(RENDER_TIMING_COLUMN, normalized[0])
            self.assertNotIn("timing_scope", normalized[0])
            self.assertFalse(any("feasible" in key.lower() for key in normalized[0]))
            for row in normalized:
                assert_no_design_markers(self, row)
                self.assertEqual(row["hotspot_metric_domain"], HOTSPOT_DISPLAY_DOMAIN)
                self.assertEqual(row["hotspot_threshold_rule"], HOTSPOT_THRESHOLD_RULE)
                self.assertEqual(
                    int(row["hotspot_threshold_histogram_bins"]), HOTSPOT_THRESHOLD_BINS
                )
                self.assertRegex(row["hotspot_threshold_sha256"], r"^[0-9a-f]{64}$")

            svg = (output / "reassessment_pareto_4panel.svg").read_text(encoding="utf-8")
            self.assertIn("<text", svg)
            self.assertIn("font-family", svg)
            self.assertIn("Building", svg)
            self.assertIn("InternalRoad", svg)
            self.assertIn("Training scopes", svg)
            self.assertIn("Render strata", svg)
            self.assertIn("full training wall clock", svg)
            self.assertIn("pure render cuda event", svg)
            self.assertIsNone(re.search(r"[^\x00-\x7f]", svg))
            for token in ("鈫", "掳", "鈥", "脳", "�"):
                self.assertNotIn(token, svg)

            source = (
                Path(__file__).parents[1]
                / "tools"
                / "export_reassessment_pareto.py"
            ).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"[^\x00-\x7f]", source))
            for token in ("鈫", "掳", "鈥", "脳", "�"):
                self.assertNotIn(token, source)
            pdf = (output / "reassessment_pareto_4panel.pdf").read_bytes()
            self.assertTrue(pdf.startswith(b"%PDF"))
            self.assertIn(b"/FontFile2", pdf)

            with Image.open(output / "reassessment_pareto_4panel.png") as image:
                self.assertEqual(image.size, (2160, 1620))
            with Image.open(output / "reassessment_pareto_4panel.tiff") as image:
                self.assertEqual(image.size, (4320, 3240))
                self.assertEqual(image.mode, "RGB")
                dpi = image.info.get("dpi")
                self.assertIsNotNone(dpi)
                self.assertAlmostEqual(float(dpi[0]), 600.0, places=1)
                self.assertAlmostEqual(float(dpi[1]), 600.0, places=1)


if __name__ == "__main__":
    unittest.main()
