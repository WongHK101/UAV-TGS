import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.build_reassessment_pareto_input import (
    DEPTH_DEFINITIONS,
    DEPTH_THRESHOLDS,
    HOTSPOT_BINS,
    HOTSPOT_DOMAIN,
    HOTSPOT_RULE,
    NOOP_CLAIM_BOUNDARY,
    NOOP_COPIED_METRIC_ROLES,
    NOOP_ENDPOINT_COVERAGE_SCHEMA,
    SOURCE_SCHEMA,
    ContractError,
    canonical_json_sha256,
    compile_reassessment_input,
    self_hash,
    sha256_file,
)
from tools.export_reassessment_pareto import load_normalized_csvs


SPLIT_SHA = "1" * 64
REFERENCE_SHA = "2" * 64


def write_json(path: Path, value) -> str:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sha256_file(path)


def identity(path: Path) -> dict:
    return {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def add_self_hash(value: dict, field: str) -> dict:
    value[field] = self_hash(value, field)
    return value


def threshold_payload() -> dict:
    value = {
        "schema": "uav-tgs-oct-train-only-hotspot-threshold-v1",
        "scene_name": "Building",
        "source_split": "train",
        "test_statistics_used": False,
        "quantile": 0.95,
        "histogram_bins": HOTSPOT_BINS,
        "range_c": [10.0, 50.0],
        "threshold_c": 42.0,
        "train_view_ids_sha256": "3" * 64,
        "valid_train_pixels": 1000,
        "source_receipt": {
            "bound_split_sha256": SPLIT_SHA,
            "receipt_sha256": "4" * 64,
        },
    }
    return add_self_hash(value, "threshold_sha256")


def phase_c_payload() -> dict:
    definitions = {}
    for definition_index, definition in enumerate(DEPTH_DEFINITIONS):
        points = []
        for threshold_index, threshold in enumerate(DEPTH_THRESHOLDS):
            points.append(
                {
                    "threshold_m": threshold,
                    "front_rate": 0.10 + definition_index * 0.01 + threshold_index * 0.001,
                    "agreement_rate": 0.80 + definition_index * 0.01 + threshold_index * 0.001,
                }
            )
        definitions[definition] = {
            # Deliberately incompatible with test: the compiler must never use it.
            "overall": {
                "front_curve_auc": {"normalized": 0.99},
                "agreement_curve_auc": {"normalized": 0.01},
            },
            "groups": {
                "split": {
                    "test": {
                        "view_count": 8,
                        "missing_rate": 0.01 + definition_index * 0.001,
                        "mean_abs_error_m": 0.5 + definition_index * 0.1,
                        "median_abs_error_m": 0.2 + definition_index * 0.1,
                        "signed_bias_m": -0.1 + definition_index * 0.01,
                        "threshold_metrics": points,
                        "front_curve_auc": {"normalized": 0.11 + definition_index * 0.01},
                        "agreement_curve_auc": {"normalized": 0.88 + definition_index * 0.01},
                    }
                }
            },
        }
    return {
        "protocol": "formal-multi-depth-definition-and-top1-responsibility-v2",
        "scene_name": "Building",
        "thresholds_m": list(DEPTH_THRESHOLDS),
        "formal_split_manifest_sha256": SPLIT_SHA,
        "reference_manifest_sha256": REFERENCE_SHA,
        "all_split_reference_binding": {
            "formal_split_manifest_sha256": SPLIT_SHA,
            "mesh_or_backend_rebuilt": False,
        },
        "depth_definitions": definitions,
    }


def make_fixture(root: Path) -> tuple[Path, dict[str, Path]]:
    threshold = threshold_payload()

    r1 = {
        "schema": "uav-tgs-formal-baseline-appearance-r1-aggregate-v1",
        "status": "passed",
        "evaluations": [
            {
                "scene": "Building",
                "method": "Raw-F3",
                "endpoint": "ours_60000",
                "split": "test",
                "resolution": {
                    "label": "r1/full-resolution",
                    "width": 1257,
                    "height": 1006,
                },
                "metrics": {"PSNR": 20.1, "SSIM": 0.82, "LPIPS": 0.42},
                "formal_split": {"sha256": SPLIT_SHA},
            }
        ],
    }
    add_self_hash(r1, "aggregate_sha256")
    r1_path = root / "r1.json"
    write_json(r1_path, r1)

    rgb_path = root / "rgb_results.json"
    write_json(rgb_path, {"ours_35000": {"PSNR": 30.0, "SSIM": 0.91, "LPIPS": 0.12}})

    inputs = {"bound_split": {"sha256": SPLIT_SHA}}
    hotspot = {
        "schema": "uav-tgs-formal-baseline-hotspot-evaluation-v1",
        "status": "complete",
        "scene_name": "Building",
        "method_name": "Raw-F3",
        "split": "test",
        "inputs": inputs,
        "inputs_sha256": canonical_json_sha256(inputs),
        "formal_binding_compatibility": {"status": "passed"},
        "display_semantics": {"comparable_to_oct_evaluator_v2": True},
        "hotspot_threshold": threshold,
        "selection_boundary": {
            "threshold_source_split": "train",
            "test_statistics_used_for_threshold": False,
            "quantile": 0.95,
            "threshold_histogram_bins": HOTSPOT_BINS,
            "threshold_c": threshold["threshold_c"],
            "test_role": "final_report_only",
        },
        "metrics": {
            "temperature_error": {
                "mae_c": 1.2,
                "rmse_c": 1.5,
                "signed_bias_c": -0.2,
                "p95_abs_error_c": 2.8,
            },
            "hotspot_auprc_histogram_4096": 0.70,
            "hotspot_iou": 0.40,
            "off_lut_distance_rgb": {"mean": 5.0, "p95": 20.0},
        },
    }
    add_self_hash(hotspot, "report_payload_sha256")
    hotspot_path = root / "hotspot.json"
    write_json(hotspot_path, hotspot)

    depth_path = root / "depth.json"
    write_json(depth_path, phase_c_payload())

    train = {
        "schema_name": "uav-tgs-efficiency",
        "schema_version": 1,
        "kind": "training_stage",
        "status": "completed",
        "stage": "thermal",
        "wall_time_s": 150.0,
        "timing_scope": "training_call_boundary",
        "device": {
            "device_name": "GPU",
            "peak_torch_allocated_bytes": 1000,
            "peak_torch_reserved_bytes": 2000,
        },
        "result": {
            "optimizer_updates_executed": 30000,
            "gaussian_count": 100000,
        },
    }
    train_path = root / "train.json"
    write_json(train_path, train)

    render = {
        "schema_name": "uav-tgs-efficiency",
        "schema_version": 1,
        "kind": "render",
        "status": "completed",
        "gaussian_count": 100000,
        "resolutions": [{"width": 314, "height": 252}],
        "benchmark": {
            "timed_views": 24,
            "timing_scope": "render_only_no_io",
            "wall_total_s": 0.1,
            "cuda_event_ms_per_view": 1.2,
            "cuda_event_fps": 833.333,
            "device": {
                "device_name": "GPU",
                "peak_torch_allocated_bytes": 800,
                "peak_torch_reserved_bytes": 900,
            },
        },
    }
    render_path = root / "render.json"
    write_json(render_path, render)

    oct_report = {
        "schema": "uav-tgs-oct-formal-evaluation-v2",
        "scene_name": "Building",
        "variant": "oct_scalar",
        "split": "test",
        "resolution": "r1/full-resolution",
        "training_evaluation_compatibility": {"status": "passed"},
        "shared_occupancy_invariant": {"exact": True, "topology_count": 100000},
        "hotspot_threshold": threshold,
        "metrics": {
            "formal_full_frame_psnr_db": 21.0,
            "formal_full_frame_ssim": 0.84,
            "formal_full_frame_lpips": 0.39,
            "temperature_mae_c": 0.75,
            "temperature_rmse_c": 1.0,
            "temperature_bias_c": -0.05,
            "temperature_p95_abs_c": 2.0,
            "temperature_semantics": "direct OCT apparent temperature",
            "palette_inverted_comparable": {
                "mae_c": 0.80,
                "rmse_c": 1.05,
                "signed_bias_c": -0.06,
                "p95_abs_error_c": 2.1,
            },
            "hotspot_auprc_histogram_4096": 0.75,
            "hotspot_iou": 0.45,
            "hotspot_primary_semantics": "display-equivalent apparent temperature",
            "off_lut_distance": {"mean_rgb_distance": 0.0, "p95_rgb_distance": 0.0},
        },
        "cost": {
            "verified_training_endpoint": {
                "optimizer_steps": 30000,
                "wall_time_s": 100.0,
                "end_to_end_wall_time_s": 110.0,
                "ms_per_step": 3.3,
                "raster_passes": 30000,
                "device": {
                    "device_name": "GPU",
                    "peak_torch_allocated_bytes": 1100,
                    "peak_torch_reserved_bytes": 2100,
                },
            },
            "pure_render": {
                "views": 8,
                "mean_ms_per_view": 3.7,
                "fps": 270.0,
                "synchronized_cuda_event": True,
            },
        },
    }
    oct_path = root / "oct.json"
    write_json(oct_path, oct_report)

    oct_checkpoint_sha = "5" * 64
    probe_camera_sha = "6" * 64
    exact_alias = {
        "schema": "uav-tgs-oct-exact-raw-anchor-alias-v2",
        "status": "passed",
        "scene": "Building",
        "variant": "oct_scalar",
        "endpoint": {"checkpoint": {"sha256": oct_checkpoint_sha}},
        "formal_probe_camera_manifest": {"sha256": probe_camera_sha},
        "formal_evaluation_v2": {"sha256": sha256_file(oct_path)},
        "inherited_phase_c_depth": {
            "metrics_summary": {"sha256": sha256_file(depth_path)}
        },
        "formal_camera_and_split_binding": {
            "bound_split": {"sha256": SPLIT_SHA}
        },
        "runtime_before_after": {"exact": True},
    }
    add_self_hash(exact_alias, "receipt_sha256")
    exact_alias_path = root / "exact_alias.json"
    write_json(exact_alias_path, exact_alias)

    pair_source_ply_sha = "7" * 64
    pair_alias_ply_sha = "8" * 64
    pair_probe_sha = "9" * 64
    exact_fields = {
        key: {"exact": True, "max_abs_diff": 0.0}
        for key in (
            "x",
            "y",
            "z",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "opacity",
        )
    }
    pair_alias = {
        "schema": "uav-tgs-exact-geometry-occupancy-alias-v2",
        "status": "passed",
        "source_label": "Raw-30000",
        "alias_label": "Raw-F3-60000",
        "source": {"ply": {"sha256": pair_source_ply_sha}},
        "alias": {"ply": {"sha256": pair_alias_ply_sha}},
        "formal_probe_camera_manifest": {"sha256": pair_probe_sha},
        "ordered_vertex_schema_exact": True,
        "ordered_xyz_and_topology_exact": True,
        "geometry_and_occupancy_fields": exact_fields,
    }
    add_self_hash(pair_alias, "receipt_sha256")
    pair_alias_path = root / "pair_alias.json"
    write_json(pair_alias_path, pair_alias)

    noop = {
        "schema": NOOP_ENDPOINT_COVERAGE_SCHEMA,
        "status": "passed",
        "evidence_status": "formal_protocol_noop_alias",
        "independent_endpoint_run": False,
        "independent_performance_claim": False,
        "scene": "Building",
        "source_method_id": "raw_f3",
        "alias_method_id": "scsp_f3",
        "copy_roles": "all",
        "copied_metric_roles": list(NOOP_COPIED_METRIC_ROLES),
        "claim_boundary": NOOP_CLAIM_BOUNDARY,
        "formal_split_sha256": SPLIT_SHA,
    }
    add_self_hash(noop, "receipt_sha256")
    noop_path = root / "noop.json"
    write_json(noop_path, noop)

    artifact_path = root / "fixed.png"
    artifact_path.write_bytes(b"not-an-image-but-hash-bound")

    base_ids = {
        "geometry_endpoint_id": "raw-geometry",
        "rgb_endpoint_id": "raw-rgb",
        "thermal_endpoint_id": "raw-thermal",
        "depth_source_id": "raw-depth",
    }
    raw_endpoint = {
        "scene": "Building",
        "method_id": "raw_f3",
        "display_name": "Raw F3",
        **base_ids,
        "structural_metadata": {"family": "f3"},
        "appearance": [
            {
                "kind": "results",
                "modality": "rgb",
                "source": identity(rgb_path),
                "endpoint": "ours_35000",
                "split": "test",
                "formal_split_sha256": SPLIT_SHA,
                "resolution": {"label": "r4/quarter-resolution", "width": 314, "height": 252},
            },
            {
                "kind": "r1_aggregate",
                "modality": "thermal",
                "source": identity(r1_path),
                "selector": {"scene": "Building", "method": "Raw-F3"},
            },
        ],
        "thermal": {
            "kind": "baseline_hotspot",
            "source": identity(hotspot_path),
            "selector": {"scene": "Building", "method": "Raw-F3"},
        },
        "depth": {"kind": "phase_c_depth", "source": identity(depth_path)},
        "efficiency": [
            {"kind": "uav_tgs_efficiency", "source": identity(train_path)},
            {"kind": "uav_tgs_efficiency", "source": identity(render_path)},
        ],
        "artifacts": [
            {
                "artifact_role": "fixed_thermal",
                "split": "test",
                "view_id": "0008",
                "source": identity(artifact_path),
            }
        ],
    }
    oct_endpoint = {
        "scene": "Building",
        "method_id": "oct_scalar",
        "display_name": "OCT-Scalar",
        "geometry_endpoint_id": "raw-geometry",
        "rgb_endpoint_id": "raw-rgb",
        "thermal_endpoint_id": "oct-scalar-30k",
        "depth_source_id": "raw-depth",
        "structural_metadata": {"family": "oct", "variant": "scalar"},
        "oct": {
            "kind": "oct_evaluation_v2",
            "source": identity(oct_path),
            "variant": "oct_scalar",
            "resolution": {"label": "r1/full-resolution", "width": 1257, "height": 1006},
        },
        "alias": {
            "kind": "exact_geometry",
            "source": identity(exact_alias_path),
            "variant": "oct_scalar",
            "endpoint_checkpoint_sha256": oct_checkpoint_sha,
            "formal_probe_camera_manifest_sha256": probe_camera_sha,
        },
        "depth": {"kind": "phase_c_depth", "source": identity(depth_path)},
    }
    noop_endpoint = {
        "scene": "Building",
        "method_id": "scsp_f3",
        "display_name": "SCSP + F3",
        "evidence_status": "formal_explicit_noop_alias",
        "geometry_endpoint_id": "scsp-noop-geometry",
        "rgb_endpoint_id": "scsp-noop-rgb",
        "thermal_endpoint_id": "scsp-noop-thermal",
        "depth_source_id": "raw-depth",
        "structural_metadata": {
            "family": "scsp",
            "independent_training_run": False,
            "independent_performance_claim": False,
        },
        "alias": {
            "kind": "noop",
            "source_method_id": "raw_f3",
            "alias_method_id": "scsp_f3",
            "source": identity(noop_path),
            "copy_roles": "all",
            "copied_metric_roles": list(NOOP_COPIED_METRIC_ROLES),
            "claim_boundary": NOOP_CLAIM_BOUNDARY,
        },
    }
    manifest = {
        "schema": SOURCE_SCHEMA,
        "scenes": [
            {
                "scene": "Building",
                "formal_split_sha256": SPLIT_SHA,
                "reference_manifest_sha256": REFERENCE_SHA,
                "hotspot_threshold_sha256": threshold["threshold_sha256"],
            }
        ],
        "endpoints": [raw_endpoint, oct_endpoint, noop_endpoint],
    }
    add_self_hash(manifest, "manifest_sha256")
    manifest_path = root / "sources.json"
    write_json(manifest_path, manifest)
    return manifest_path, {
        "r1": r1_path,
        "rgb": rgb_path,
        "hotspot": hotspot_path,
        "depth": depth_path,
        "train": train_path,
        "render": render_path,
        "oct": oct_path,
        "exact_alias": exact_alias_path,
        "pair_alias": pair_alias_path,
        "noop": noop_path,
        "artifact": artifact_path,
    }


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_manifest(path: Path, manifest: dict) -> None:
    manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
    write_json(path, manifest)


def pair_endpoint_from_raw(manifest: dict, pair_receipt: Path) -> dict:
    endpoint = copy.deepcopy(manifest["endpoints"][0])
    endpoint.update(
        {
            "method_id": "pair_f3",
            "display_name": "Pair F3",
            "thermal_endpoint_id": "pair-thermal",
        }
    )
    endpoint["alias"] = {
        "kind": "exact_geometry",
        "source": identity(pair_receipt),
        "source_label": "Raw-30000",
        "alias_label": "Raw-F3-60000",
        "source_ply_sha256": "7" * 64,
        "alias_ply_sha256": "8" * 64,
        "formal_probe_camera_manifest_sha256": "9" * 64,
    }
    return endpoint


class ReassessmentParetoInputCompilerTests(unittest.TestCase):
    def test_end_to_end_compiles_exact_test_depth_and_stratified_costs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            output = root / "compiled"
            returned = compile_reassessment_input(manifest_path, output)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "endpoint_summary.csv",
                    "depth_curve.csv",
                    "efficiency.csv",
                    "artifact_manifest.csv",
                    "source_manifest.json",
                    "declared_source_manifest.json",
                    "manifest.json",
                },
            )
            endpoints = read_csv(output / "endpoint_summary.csv")
            self.assertEqual(
                [(row["scene"], row["method_id"]) for row in endpoints],
                [
                    ("Building", "oct_scalar"),
                    ("Building", "raw_f3"),
                    ("Building", "scsp_f3"),
                ],
            )
            self.assertNotIn("ogs_f3", {row["method_id"] for row in endpoints})
            self.assertNotIn("physir", {row["method_id"] for row in endpoints})
            # The compiler's headline table is the direct, strict input of the
            # unconstrained exporter rather than a merely similar side table.
            loaded = load_normalized_csvs([output / "endpoint_summary.csv"])
            self.assertEqual(len(loaded.rows), 3)

            raw = next(row for row in endpoints if row["method_id"] == "raw_f3")
            self.assertEqual(raw["rgb_psnr_db"], "30")
            self.assertEqual(raw["thermal_psnr_db"], "20.100000000000001")
            self.assertEqual(raw["temp_mae_direct_c"], "")
            self.assertEqual(raw["temp_mae_display_c"], "1.2")
            self.assertEqual(raw["hotspot_metric_domain"], HOTSPOT_DOMAIN)
            self.assertEqual(raw["hotspot_threshold_rule"], HOTSPOT_RULE)
            self.assertEqual(raw["hotspot_threshold_histogram_bins"], str(HOTSPOT_BINS))
            self.assertEqual(raw["render_width"], "314")
            self.assertEqual(raw["render_height"], "252")
            self.assertEqual(raw["rgb_appearance_resolution_label"], "r4/quarter-resolution")
            self.assertEqual(raw["rgb_appearance_width"], "314")
            self.assertEqual(raw["rgb_appearance_height"], "252")
            self.assertEqual(raw["thermal_appearance_resolution_label"], "r1/full-resolution")
            self.assertEqual(raw["thermal_appearance_width"], "1257")
            self.assertEqual(raw["thermal_appearance_height"], "1006")
            self.assertEqual(raw["appearance_resolution_label"], "")
            self.assertEqual(raw["appearance_width"], "")
            self.assertEqual(raw["appearance_height"], "")
            self.assertEqual(raw["train_timing_scope"], "training_call_boundary")
            self.assertEqual(raw["render_timing_scope"], "render_only_no_io")
            self.assertEqual(raw["timing_scope"], raw["render_timing_scope"])
            self.assertNotEqual(
                raw["appearance_source_sha256"], raw["rgb_appearance_source_sha256"]
            )

            oct_row = next(row for row in endpoints if row["method_id"] == "oct_scalar")
            self.assertEqual(oct_row["temp_mae_direct_c"], "0.75")
            self.assertEqual(oct_row["temp_mae_display_c"], "0.80000000000000004")
            self.assertNotEqual(oct_row["temp_mae_direct_c"], oct_row["temp_mae_display_c"])
            self.assertEqual(oct_row["off_lut_mean_rgb"], "0")
            self.assertTrue(oct_row["alias_receipt_sha256"])
            self.assertEqual(oct_row["rgb_appearance_width"], "")
            self.assertEqual(oct_row["thermal_appearance_width"], "1257")
            self.assertEqual(oct_row["appearance_width"], "1257")
            self.assertEqual(
                oct_row["train_timing_scope"],
                "oct_end_to_end_training_including_pretraining_setup",
            )
            self.assertEqual(
                oct_row["render_timing_scope"], "pure_render_synchronized_cuda_event"
            )
            self.assertEqual(oct_row["timing_scope"], oct_row["render_timing_scope"])

            alias = next(row for row in endpoints if row["method_id"] == "scsp_f3")
            self.assertEqual(alias["thermal_psnr_db"], raw["thermal_psnr_db"])
            self.assertEqual(alias["depth_source_sha256"], raw["depth_source_sha256"])
            self.assertNotEqual(alias["alias_receipt_sha256"], raw["alias_receipt_sha256"])
            self.assertEqual(alias["geometry_endpoint_id"], "scsp-noop-geometry")

            raw_text = (output / "endpoint_summary.csv").read_text(encoding="utf-8").lower()
            self.assertNotIn("nan", raw_text)
            self.assertNotIn("physir", raw_text)

            depth = read_csv(output / "depth_curve.csv")
            self.assertEqual(len(depth), 3 * len(DEPTH_DEFINITIONS) * len(DEPTH_THRESHOLDS))
            self.assertEqual({row["group_type"] for row in depth}, {"split"})
            self.assertEqual({row["group_label"] for row in depth}, {"test"})
            expected_first = next(
                row
                for row in depth
                if row["method_id"] == "raw_f3"
                and row["depth_definition"] == "expected"
                and row["threshold_m"] == "0.25"
            )
            self.assertEqual(expected_first["front_rate"], "0.10000000000000001")
            self.assertEqual(expected_first["front_auc_normalized"], "0.11")
            self.assertNotEqual(expected_first["front_auc_normalized"], "0.99")

            efficiency = read_csv(output / "efficiency.csv")
            self.assertEqual(len(efficiency), 6)
            render_groups = {
                row["comparison_group"]
                for row in efficiency
                if row["kind"] == "render"
            }
            self.assertIn("render:314x252:render_only_no_io", render_groups)
            self.assertIn(
                "render:1257x1006:pure_render_synchronized_cuda_event", render_groups
            )

            artifacts = read_csv(output / "artifact_manifest.csv")
            self.assertEqual(len(artifacts), 2)
            self.assertEqual({row["method_id"] for row in artifacts}, {"raw_f3", "scsp_f3"})
            self.assertTrue(all(row["source_manifest_sha256"] for row in artifacts))

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(returned, manifest)
            self.assertEqual(manifest["manifest_sha256"], self_hash(manifest, "manifest_sha256"))
            for name, receipt in manifest["outputs"].items():
                path = output / name
                self.assertEqual(receipt["sha256"], sha256_file(path))
                self.assertEqual(Path(receipt["path"]), path.resolve())
            source = json.loads((output / "source_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(source["manifest_sha256"], self_hash(source, "manifest_sha256"))
            self.assertEqual(len(source["verified_inputs"]), 10)
            self.assertEqual(
                (output / "declared_source_manifest.json").read_bytes(),
                manifest_path.read_bytes(),
            )

    def test_sha_tamper_and_protocol_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            paths["depth"].write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "SHA-256 mismatch"):
                compile_reassessment_input(manifest_path, root / "out_sha")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["scenes"][0]["formal_split_sha256"] = "9" * 64
            manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "formal split differs"):
                compile_reassessment_input(manifest_path, root / "out_protocol")

    def test_hotspot_test_statistics_and_implicit_alias_are_rejected_or_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            report = json.loads(paths["hotspot"].read_text(encoding="utf-8"))
            report["hotspot_threshold"]["test_statistics_used"] = True
            report["hotspot_threshold"]["threshold_sha256"] = self_hash(
                report["hotspot_threshold"], "threshold_sha256"
            )
            report["report_payload_sha256"] = self_hash(report, "report_payload_sha256")
            report_sha = write_json(paths["hotspot"], report)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"][0]["thermal"]["source"]["sha256"] = report_sha
            manifest["endpoints"][0]["thermal"]["source"]["size_bytes"] = paths[
                "hotspot"
            ].stat().st_size
            manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "not frozen train-q95"):
                compile_reassessment_input(manifest_path, root / "out_threshold")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"] = [manifest["endpoints"][0]]
            manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
            write_json(manifest_path, manifest)
            output = root / "out_no_alias"
            compile_reassessment_input(manifest_path, output)
            rows = read_csv(output / "endpoint_summary.csv")
            self.assertEqual([row["method_id"] for row in rows], ["raw_f3"])

    def test_results_selector_and_source_manifest_self_hash_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            results = json.loads(paths["rgb"].read_text(encoding="utf-8"))
            results["ours_99999"] = results["ours_35000"]
            result_sha = write_json(paths["rgb"], results)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest["endpoints"][0]["appearance"][0]["source"]
            source["sha256"] = result_sha
            source["size_bytes"] = paths["rgb"].stat().st_size
            manifest["manifest_sha256"] = self_hash(manifest, "manifest_sha256")
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "must contain exactly"):
                compile_reassessment_input(manifest_path, root / "out_results")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"][0]["display_name"] = "tampered"
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "manifest_sha256 mismatch"):
                compile_reassessment_input(manifest_path, root / "out_manifest")

    def test_pair_alias_identity_is_manifest_bound_and_missing_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            endpoint = pair_endpoint_from_raw(manifest, paths["pair_alias"])
            manifest["endpoints"] = [manifest["endpoints"][0], endpoint]
            write_manifest(manifest_path, manifest)
            compile_reassessment_input(manifest_path, root / "out_valid")

        mutations = {
            "source_label": "Wrong-Source",
            "alias_label": "Wrong-Alias",
            "source_ply_sha256": "a" * 64,
            "alias_ply_sha256": "b" * 64,
            "formal_probe_camera_manifest_sha256": "c" * 64,
        }
        for field, wrong in mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, paths = make_fixture(root)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                endpoint = pair_endpoint_from_raw(manifest, paths["pair_alias"])
                endpoint["alias"][field] = wrong
                manifest["endpoints"] = [manifest["endpoints"][0], endpoint]
                write_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ContractError, "pair alias identity differs"):
                    compile_reassessment_input(manifest_path, root / "out")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            endpoint = pair_endpoint_from_raw(manifest, paths["pair_alias"])
            endpoint["alias"].pop("source_label")
            manifest["endpoints"] = [manifest["endpoints"][0], endpoint]
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "source_label"):
                compile_reassessment_input(manifest_path, root / "out_missing")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            receipt = json.loads(paths["pair_alias"].read_text(encoding="utf-8"))
            receipt["formal_probe_camera_manifest"].pop("sha256")
            add_self_hash(receipt, "receipt_sha256")
            write_json(paths["pair_alias"], receipt)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            endpoint = pair_endpoint_from_raw(manifest, paths["pair_alias"])
            manifest["endpoints"] = [manifest["endpoints"][0], endpoint]
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "formal_probe_camera_manifest.sha256"):
                compile_reassessment_input(manifest_path, root / "out_receipt_missing")

    def test_oct_alias_binds_variant_checkpoint_probe_evaluation_and_depth(self):
        manifest_field_mutations = {
            "variant": "oct_residual",
            "endpoint_checkpoint_sha256": "a" * 64,
            "formal_probe_camera_manifest_sha256": "b" * 64,
        }
        for field, wrong in manifest_field_mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, _ = make_fixture(root)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["endpoints"][1]["alias"][field] = wrong
                write_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ContractError, "OCT runtime alias"):
                    compile_reassessment_input(manifest_path, root / "out")

        receipt_mutations = (
            ("formal_evaluation_v2", "sha256"),
            ("inherited_phase_c_depth", "metrics_summary", "sha256"),
        )
        for path_parts in receipt_mutations:
            with self.subTest(receipt_path=path_parts), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, paths = make_fixture(root)
                receipt = json.loads(paths["exact_alias"].read_text(encoding="utf-8"))
                cursor = receipt
                for part in path_parts[:-1]:
                    cursor = cursor[part]
                cursor[path_parts[-1]] = "d" * 64
                add_self_hash(receipt, "receipt_sha256")
                write_json(paths["exact_alias"], receipt)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["endpoints"][1]["alias"]["source"] = identity(paths["exact_alias"])
                write_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ContractError, "OCT runtime alias"):
                    compile_reassessment_input(manifest_path, root / "out")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"][1]["alias"].pop("endpoint_checkpoint_sha256")
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "endpoint_checkpoint_sha256"):
                compile_reassessment_input(manifest_path, root / "out_missing")

    def test_noop_requires_dedicated_endpoint_coverage_contract(self):
        receipt_mutations = {
            "source_method_id": "not_raw_f3",
            "copy_roles": "metrics_only",
            "copied_metric_roles": ["depth"],
            "claim_boundary": "independent result",
            "evidence_status": "formal",
            "independent_endpoint_run": True,
            "independent_performance_claim": True,
            "schema": "uav-tgs-final-exact-geometry-alias-receipt-set-v3",
        }
        for field, wrong in receipt_mutations.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, paths = make_fixture(root)
                receipt = json.loads(paths["noop"].read_text(encoding="utf-8"))
                receipt[field] = wrong
                add_self_hash(receipt, "receipt_sha256")
                write_json(paths["noop"], receipt)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["endpoints"][2]["alias"]["source"] = identity(paths["noop"])
                write_manifest(manifest_path, manifest)
                expected = "unsupported no-op receipt schema" if field == "schema" else "coverage differs"
                with self.assertRaisesRegex(ContractError, expected):
                    compile_reassessment_input(manifest_path, root / "out")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = make_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"][2]["alias"]["selector"] = "whole_endpoint"
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "does not use a selector"):
                compile_reassessment_input(manifest_path, root / "out_selector")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, paths = make_fixture(root)
            receipt = json.loads(paths["noop"].read_text(encoding="utf-8"))
            receipt.pop("claim_boundary")
            add_self_hash(receipt, "receipt_sha256")
            write_json(paths["noop"], receipt)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["endpoints"][2]["alias"]["source"] = identity(paths["noop"])
            write_manifest(manifest_path, manifest)
            with self.assertRaisesRegex(ContractError, "coverage differs"):
                compile_reassessment_input(manifest_path, root / "out_missing_claim")

    def test_noop_endpoint_declaration_cannot_claim_independent_evidence(self):
        mutations = (
            ("evidence_status", "formal_independent_endpoint", False),
            ("independent_training_run", True, False),
            ("independent_performance_claim", True, False),
            ("independent_training_run", None, True),
            ("independent_performance_claim", None, True),
        )
        for field, wrong, remove in mutations:
            with self.subTest(field=field, remove=remove), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, _ = make_fixture(root)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                endpoint = manifest["endpoints"][2]
                if field == "evidence_status":
                    endpoint[field] = wrong
                elif remove:
                    endpoint["structural_metadata"].pop(field)
                else:
                    endpoint["structural_metadata"][field] = wrong
                write_manifest(manifest_path, manifest)
                expected = (
                    "evidence_status must be"
                    if field == "evidence_status"
                    else "must explicitly deny independent"
                )
                with self.assertRaisesRegex(ContractError, expected):
                    compile_reassessment_input(manifest_path, root / "out")

    def test_phase_c_rates_auc_and_absolute_errors_are_bounded(self):
        mutations = (
            ("front_rate", 1.01),
            ("agreement_rate", -0.01),
            ("missing_rate", 1.01),
            ("front_auc", 1.01),
            ("agreement_auc", -0.01),
            ("mean_abs_error_m", -0.01),
            ("median_abs_error_m", -0.01),
        )
        for field, wrong in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, paths = make_fixture(root)
                payload = json.loads(paths["depth"].read_text(encoding="utf-8"))
                test = payload["depth_definitions"]["expected"]["groups"]["split"]["test"]
                if field == "front_rate":
                    test["threshold_metrics"][0]["front_rate"] = wrong
                elif field == "agreement_rate":
                    test["threshold_metrics"][0]["agreement_rate"] = wrong
                elif field == "front_auc":
                    test["front_curve_auc"]["normalized"] = wrong
                elif field == "agreement_auc":
                    test["agreement_curve_auc"]["normalized"] = wrong
                else:
                    test[field] = wrong
                write_json(paths["depth"], payload)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["endpoints"] = [manifest["endpoints"][0]]
                manifest["endpoints"][0]["depth"]["source"] = identity(paths["depth"])
                write_manifest(manifest_path, manifest)
                with self.assertRaisesRegex(ContractError, "expected value"):
                    compile_reassessment_input(manifest_path, root / "out")


if __name__ == "__main__":
    unittest.main()
