import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.baseline_evaluation_receipts import (
    ADAPTER_SCHEMA,
    ALIAS_SCHEMA,
    EVALUATOR_SCHEMA,
    FAILED,
    ReceiptContractError,
    STUDY_SCHEMA,
    SUCCEEDED,
    UNSUPPORTED,
    classify_reuse,
    finalize_adapter_receipt,
    finalize_alias_receipt,
    finalize_evaluator_receipt,
    finalize_study_manifest,
    make_scoped_provenance,
    metric_pairing_key,
    sha256_file,
    summarize_study,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


COLLECTION_SHA = digest("formal-collection")
COLLECTION_SPLIT_SHA = digest("formal-split")
FORMAL_RULE_SHA = digest("formal-rule")
TEST_METRIC_CATALOG = {
    "appearance": ["psnr_db", "quality", "error"],
    "efficiency": ["training_wall_time_and_scope"],
}
TEST_PROTOCOL = {
    "protocol_id": "formal-protocol-v1",
    "protocol_version": "1.0.0",
    "collection_hash": COLLECTION_SHA,
    "collection": {
        "collection_split_hash": COLLECTION_SPLIT_SHA,
        "formal_rule_hash": FORMAL_RULE_SHA,
    },
    "depth_definitions": [
        "transmittance_median",
        "alpha_weighted_expected",
        "maximum_contribution_surface",
    ],
    "geometry_thresholds_m": [0.25, 0.5, 1, 2, 5, 10, 15, 20],
    "common_metrics": TEST_METRIC_CATALOG,
}
TEST_PROTOCOL_BYTES = (
    json.dumps(TEST_PROTOCOL, indent=2, sort_keys=True, allow_nan=False) + "\n"
).encode("utf-8")
PROTOCOL_SHA = hashlib.sha256(TEST_PROTOCOL_BYTES).hexdigest()


def identity(role: str, label: str, *, logical_id: str | None = None) -> dict:
    result = {"role": role, "sha256": digest(label)}
    if logical_id is not None:
        result["logical_id"] = logical_id
    return result


def provenance(
    scope: str,
    *,
    code_label: str,
    input_label: str = "formal-input",
    repository: dict | None = None,
) -> dict:
    return make_scoped_provenance(
        scope=scope,
        relevant_files=[
            {
                **identity("entrypoint", code_label),
                "path": f"/relocated/{code_label}.py",
            }
        ],
        inputs=[identity("formal_input", input_label, logical_id="formal")],
        settings={"recipe": "formal-v1", "seed": 7},
        environment={"framework": "torch", "major_version": 2},
        repository=repository or {"commit": digest("commit-a")},
    )


def cost_record(
    cost_id: str,
    *,
    stage: str,
    batch_id: str,
    wall_time_s: float,
    gpu_seconds: float,
    peak_allocated_vram_mb: float = 900.0,
    peak_reserved_vram_mb: float = 1024.0,
    disk_bytes: int = 4096,
    cost_class: str = "method_specific",
    accounting: str = "measured",
    source_cost_id: str | None = None,
    batch_wall_time_s: float | None = None,
    batch_gpu_seconds: float | None = None,
    batch_peak_allocated_vram_mb: float | None = None,
    batch_peak_reserved_vram_mb: float | None = None,
    batch_disk_bytes: int | None = None,
    exclusion_reason: str | None = None,
    reported_wall_time_s: float | None = None,
    reported_gpu_seconds: float | None = None,
    reported_peak_allocated_vram_mb: float | None = None,
    reported_peak_reserved_vram_mb: float | None = None,
    reported_disk_bytes: int | None = None,
) -> dict:
    result = {
        "cost_id": cost_id,
        "cost_class": cost_class,
        "stage": stage,
        "execution_environment": {
            "device_id": "cuda:0",
            "gpu_model": "RTX-PRO-6000",
            "gpu_uuid": "GPU-test-uuid",
            "cuda_version": "12.8",
            "driver_version": "570.0",
            "torch_version": "2.7.0",
            "python_version": "3.10.16",
            "runtime_id": "formal-runtime-v1",
        },
        "timing_scope": {
            "scope_id": f"formal-{stage}-scope-v1",
            "included_operations": [f"method-only-{stage}"],
            "excluded_operations": ["shared-cfr", "common-evaluation"],
        },
        "batch_execution": {
            "batch_id": batch_id,
            "wall_time_s": wall_time_s if batch_wall_time_s is None else batch_wall_time_s,
            "gpu_seconds": gpu_seconds if batch_gpu_seconds is None else batch_gpu_seconds,
            "peak_allocated_vram_mb": (
                peak_allocated_vram_mb
                if batch_peak_allocated_vram_mb is None
                else batch_peak_allocated_vram_mb
            ),
            "peak_reserved_vram_mb": (
                peak_reserved_vram_mb
                if batch_peak_reserved_vram_mb is None
                else batch_peak_reserved_vram_mb
            ),
            "disk_bytes": disk_bytes if batch_disk_bytes is None else batch_disk_bytes,
        },
        "reported_method": {
            "accounting": accounting,
            "wall_time_s": wall_time_s if reported_wall_time_s is None else reported_wall_time_s,
            "gpu_seconds": gpu_seconds if reported_gpu_seconds is None else reported_gpu_seconds,
            "peak_allocated_vram_mb": (
                peak_allocated_vram_mb
                if reported_peak_allocated_vram_mb is None
                else reported_peak_allocated_vram_mb
            ),
            "peak_reserved_vram_mb": (
                peak_reserved_vram_mb
                if reported_peak_reserved_vram_mb is None
                else reported_peak_reserved_vram_mb
            ),
            "disk_bytes": disk_bytes if reported_disk_bytes is None else reported_disk_bytes,
        },
    }
    if source_cost_id is not None:
        result["reported_method"]["source_cost_id"] = source_cost_id
    if exclusion_reason is not None:
        result["reported_method"]["exclusion_reason"] = exclusion_reason
    return result


def adapter_receipt(
    scene: str,
    method_id: str,
    *,
    status: str = SUCCEEDED,
    code_label: str = "adapter-code",
    artifact_label: str | None = None,
    repository: dict | None = None,
    costs: list[dict] | None = None,
    run_scope: str = "preliminary",
    method_origin: dict | None = None,
) -> dict:
    result = {
        "schema": ADAPTER_SCHEMA,
        "status": status,
        "scene": scene,
        "reported_method_id": method_id,
        "adapter_id": "official-adapter-v1",
        "run_scope": run_scope,
        "method_origin": method_origin
        or {
            "kind": "internal",
            "repository_url": "https://github.com/WongHK101/UAV-TGS",
            "source_commit": digest("source-commit"),
            "license": "research-code",
            "adapter_commit": digest("adapter-commit"),
            "compatibility_patch_sha256": digest("no-compatibility-patch"),
            "official_recipe_id": "formal-internal-recipe-v1",
            "input_conversion": "native formal collection",
            "camera_mapping": "formal cameras exact",
            "split_mapping": "formal split exact",
            "output_conversion": "native render bundle",
            "semantic_limitations": [],
            "adapter_changes_method": False,
        },
        "training_provenance": provenance(
            "training_endpoint",
            code_label=code_label,
            input_label=f"{scene}-formal-input",
            repository=repository,
        ),
        "cost_records": list(costs or []),
    }
    if status == SUCCEEDED:
        result["endpoint"] = {
            "endpoint_id": f"{scene}-{method_id}-endpoint",
            "endpoint_kind": "renderable_checkpoint",
            "artifacts": [
                identity(
                    "checkpoint",
                    artifact_label or f"{scene}-{method_id}-artifact",
                )
            ],
            "model_profile": {
                "representation": "gaussian",
                "model_size_bytes": 8192,
                "parameter_count": 2048,
                "gaussian_count": 512,
            },
        }
    elif status == FAILED:
        result["failure"] = {"code": "TRAIN_FAILED", "message": "training failed"}
    elif status == UNSUPPORTED:
        result["unsupported"] = {"reason": "adapter cannot represent this scene"}
    return finalize_adapter_receipt(result)


def metric_scope(
    scene: str,
    metric_definition_id: str,
    *,
    aggregation: str = "scene_macro",
    split: str = "test",
) -> dict:
    return {
        "metric_definition_id": metric_definition_id,
        "protocol_sha256": PROTOCOL_SHA,
        "collection_sha256": COLLECTION_SHA,
        "split_sha256": digest("formal-split"),
        "camera_sha256": digest(f"{scene}-formal-camera"),
        "split": split,
        "aggregation": aggregation,
        "support_sha256": digest(f"{scene}-test-support"),
        "view_count": 8,
        "sample_count": 800,
        "resolution": {"width": 640, "height": 512},
    }


def metric_value(
    scene: str,
    metric_definition_id: str,
    value: float,
    *,
    unit: str,
    status: str = SUCCEEDED,
    aggregation: str = "scene_macro",
) -> dict:
    scope = metric_scope(scene, metric_definition_id, aggregation=aggregation)
    result = {
        "status": status,
        "unit": unit,
        "evaluation_scope": scope,
        "pairing_key": metric_pairing_key(scope),
    }
    if status == SUCCEEDED:
        result["value"] = value
    elif status == FAILED:
        result["failure"] = {"code": "METRIC_FAILED", "message": "metric failed"}
    elif status == UNSUPPORTED:
        result["unsupported"] = {"reason": "metric is structurally unavailable"}
    return result


def evaluator_receipt(
    adapter: dict,
    *,
    status: str = SUCCEEDED,
    code_label: str = "evaluator-code",
    metrics: dict | None = None,
    costs: list[dict] | None = None,
) -> dict:
    result = {
        "schema": EVALUATOR_SCHEMA,
        "status": status,
        "scene": adapter["scene"],
        "reported_method_id": adapter["reported_method_id"],
        "evaluator_id": "formal-evaluator-v1",
        "endpoint_reuse_key": adapter["endpoint"]["reuse_key"],
        "evaluation_provenance": provenance(
            "endpoint_evaluation",
            code_label=code_label,
            input_label=f"{adapter['scene']}-evaluation-protocol",
        ),
        "cost_records": list(costs or []),
    }
    if status == SUCCEEDED:
        result["metrics"] = metrics or {
            "psnr_db": metric_value(
                adapter["scene"], "psnr_db-v1", 30.0, unit="dB"
            )
        }
    elif status == FAILED:
        result["failure"] = {"code": "EVAL_FAILED", "message": "evaluation failed"}
    elif status == UNSUPPORTED:
        result["unsupported"] = {"reason": "metric is unavailable for this endpoint"}
    return finalize_evaluator_receipt(result)


def alias_receipt(
    source: dict,
    alias_method_id: str,
    *,
    costs: list[dict] | None = None,
) -> dict:
    return finalize_alias_receipt(
        {
            "schema": ALIAS_SCHEMA,
            "status": SUCCEEDED,
            "scene": source["scene"],
            "source_method_id": source["reported_method_id"],
            "reported_method_id": alias_method_id,
            "source_endpoint_reuse_key": source["endpoint"]["reuse_key"],
            "reuse": {
                "exact_endpoint": True,
                "roles": ["geometry", "appearance", "metrics"],
            },
            "zero_modification": {
                "manifest_sha256": digest(
                    f"{source['scene']}-{alias_method_id}-zero-modification"
                ),
                "modified_count": 0,
                "total_gaussian_count": 512,
                "checked_fields": [
                    "xyz",
                    "scaling",
                    "rotation",
                    "opacity",
                    "topology",
                ],
            },
            "cost_records": list(costs or []),
        }
    )


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def bound(path: Path) -> dict:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def study_summary(
    root: Path,
    *,
    scenes: list[str],
    methods: list[str],
    metric_specs: list[dict],
    comparisons: list[dict],
    adapters: list[dict],
    evaluators: list[dict],
    aliases: list[dict] | None = None,
    expected_scenes_by_method: dict[str, list[str]] | None = None,
) -> dict:
    protocol_path = root / "formal_protocol.json"
    protocol_path.write_bytes(TEST_PROTOCOL_BYTES)
    adapter_paths = []
    for index, receipt in enumerate(adapters):
        path = root / f"adapter_{index}.json"
        write_json(path, receipt)
        adapter_paths.append(path)
    evaluator_paths = []
    for index, receipt in enumerate(evaluators):
        path = root / f"evaluator_{index}.json"
        write_json(path, receipt)
        evaluator_paths.append(path)
    alias_paths = []
    for index, receipt in enumerate(aliases or []):
        path = root / f"alias_{index}.json"
        write_json(path, receipt)
        alias_paths.append(path)
    expected_by_method = expected_scenes_by_method or {
        method: scenes for method in methods
    }
    matrix_scenes = [
        scene
        for scene in scenes
        if all(scene in expected_by_method.get(method, scenes) for method in methods)
    ]
    manifest = finalize_study_manifest(
        {
            "schema": STUDY_SCHEMA,
            "protocol": {
                "path": protocol_path.name,
                "sha256": PROTOCOL_SHA,
                "protocol_id": "formal-protocol-v1",
                "protocol_version": "1.0.0",
                "collection_hash": COLLECTION_SHA,
                "collection_split_hash": COLLECTION_SPLIT_SHA,
                "formal_rule_hash": FORMAL_RULE_SHA,
            },
            "matrix_contract": {
                "matrix_id": "test_matrix",
                "method_ids": methods,
                "scene_ids": matrix_scenes,
                "complete_cross_product": True,
                "expected_cell_count": len(methods) * len(matrix_scenes),
            },
            "geometry_contract": {
                "depth_definitions": TEST_PROTOCOL["depth_definitions"],
                "thresholds_m": TEST_PROTOCOL["geometry_thresholds_m"],
            },
            "required_metric_catalog": TEST_METRIC_CATALOG,
            "scenes": scenes,
            "reported_methods": [
                {
                    "method_id": method,
                    "display_name": method.upper(),
                    "expected_scenes": (
                        expected_by_method
                    ).get(method, scenes),
                }
                for method in methods
            ],
            "metrics": [
                {
                    **spec,
                    "metric_definition_id": spec.get(
                        "metric_definition_id", f"{spec['metric_id']}-v1"
                    ),
                    "unit": spec.get(
                        "unit",
                        {"psnr_db": "dB", "quality": "score", "error": "unit"}.get(
                            spec["metric_id"], "unitless"
                        ),
                    ),
                    "split": spec.get("split", "test"),
                    "aggregation": spec.get("aggregation", "scene_macro"),
                    "macro_aggregation": spec.get("macro_aggregation", "mean"),
                    "tie_tolerance": spec.get("tie_tolerance", 0.0),
                }
                for spec in metric_specs
            ],
            "comparisons": [
                {
                    "comparison_id": comparison["comparison_id"],
                    "method_ids": (
                        comparison["method_ids"]
                        if "method_ids" in comparison
                        else [comparison["left_method_id"], comparison["right_method_id"]]
                    ),
                    "baseline_method_id": (
                        comparison["baseline_method_id"]
                        if "baseline_method_id" in comparison
                        else comparison["left_method_id"]
                    ),
                }
                for comparison in comparisons
            ],
            "adapter_receipts": [bound(path) for path in adapter_paths],
            "evaluator_receipts": [bound(path) for path in evaluator_paths],
            "alias_receipts": [bound(path) for path in alias_paths],
        }
    )
    manifest_path = root / "study.json"
    write_json(manifest_path, manifest)
    return summarize_study(manifest_path)


class BaselineEvaluationReceiptTests(unittest.TestCase):
    def test_exact_alias_has_zero_execution_and_inherits_reported_method_cost(self):
        train_cost = cost_record(
            "source-train",
            stage="training",
            batch_id="shared-train-batch",
            wall_time_s=100.0,
            gpu_seconds=80.0,
        )
        source = adapter_receipt("Building", "source", costs=[train_cost])
        evaluation = evaluator_receipt(
            source,
            costs=[
                cost_record(
                    "source-eval",
                    stage="evaluation",
                    batch_id="source-eval-batch",
                    wall_time_s=10.0,
                    gpu_seconds=2.0,
                )
            ],
        )
        alias = alias_receipt(
            source,
            "source-noop-alias",
            costs=[
                cost_record(
                    "alias-train",
                    stage="training",
                    batch_id="alias-zero-train",
                    wall_time_s=100.0,
                    gpu_seconds=80.0,
                    accounting="inherited_exact_endpoint",
                    source_cost_id="source-train",
                    batch_wall_time_s=0.0,
                    batch_gpu_seconds=0.0,
                    batch_peak_allocated_vram_mb=0.0,
                    batch_peak_reserved_vram_mb=0.0,
                    batch_disk_bytes=0,
                ),
                cost_record(
                    "alias-eval",
                    stage="evaluation",
                    batch_id="alias-zero-eval",
                    wall_time_s=10.0,
                    gpu_seconds=2.0,
                    accounting="inherited_exact_endpoint",
                    source_cost_id="source-eval",
                    batch_wall_time_s=0.0,
                    batch_gpu_seconds=0.0,
                    batch_peak_allocated_vram_mb=0.0,
                    batch_peak_reserved_vram_mb=0.0,
                    batch_disk_bytes=0,
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=["Building"],
                methods=["source", "source-noop-alias"],
                metric_specs=[
                    {
                        "metric_id": "psnr_db",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    }
                ],
                comparisons=[
                    {
                        "comparison_id": "source-vs-alias",
                        "left_method_id": "source",
                        "right_method_id": "source-noop-alias",
                    }
                ],
                adapters=[source],
                evaluators=[evaluation],
                aliases=[alias],
            )

        batch = summary["cost"]["batch_accounting"]
        self.assertEqual(batch["unique_batch_count"], 4)
        self.assertEqual(batch["nonzero_batch_count"], 2)
        self.assertEqual(batch["wall_time_s_sum"], 110.0)
        totals = {
            (row["scene"], row["method_id"]): row
            for row in summary["cost"]["reported_method_accounting"]["totals"]
        }
        self.assertEqual(totals[("Building", "source")]["wall_time_s"], 110.0)
        self.assertEqual(
            totals[("Building", "source-noop-alias")]["wall_time_s"], 110.0
        )
        profiles = {
            (row["scene"], row["method_id"]): row
            for row in summary["cost"]["reported_method_accounting"]["model_profiles"]
        }
        for field in ("representation", "model_size_bytes", "parameter_count", "gaussian_count"):
            self.assertEqual(
                profiles[("Building", "source")][field],
                profiles[("Building", "source-noop-alias")][field],
            )
        macro = summary["primary_paired_macro"][0]
        self.assertEqual(macro["common_scene_count"], 1)
        self.assertEqual(macro["right_minus_left"], 0.0)
        reuse_group = summary["endpoint_reuse"]["groups"][0]
        self.assertEqual(len(reuse_group["reported_endpoints"]), 2)

    def test_unrelated_repository_change_does_not_invalidate_endpoint(self):
        first = adapter_receipt(
            "Building",
            "method",
            artifact_label="stable-endpoint",
            repository={
                "commit": digest("commit-before"),
                "unrelated_files": {"README.md": digest("old-readme")},
            },
        )
        second = adapter_receipt(
            "Building",
            "method",
            artifact_label="stable-endpoint",
            repository={
                "commit": digest("commit-after"),
                "unrelated_files": {"README.md": digest("new-readme")},
            },
        )
        self.assertEqual(
            first["training_provenance"]["scope_sha256"],
            second["training_provenance"]["scope_sha256"],
        )
        self.assertEqual(first["endpoint"]["reuse_key"], second["endpoint"]["reuse_key"])
        self.assertNotEqual(first["receipt_sha256"], second["receipt_sha256"])
        self.assertTrue(classify_reuse(first, second)["endpoint_reusable"])

        relevant_change = adapter_receipt(
            "Building",
            "method",
            code_label="adapter-code-v2",
            artifact_label="stable-endpoint",
        )
        decision = classify_reuse(first, relevant_change)
        self.assertFalse(decision["endpoint_reusable"])
        self.assertEqual(decision["action"], "retrain_and_reevaluate")

    def test_evaluator_only_change_requires_reevaluation_not_retraining(self):
        adapter = adapter_receipt("Building", "method")
        old_evaluator = evaluator_receipt(adapter, code_label="evaluator-v1")
        new_evaluator = evaluator_receipt(adapter, code_label="evaluator-v2")
        self.assertNotEqual(
            old_evaluator["evaluation_reuse_key"],
            new_evaluator["evaluation_reuse_key"],
        )
        decision = classify_reuse(adapter, adapter, old_evaluator, new_evaluator)
        self.assertTrue(decision["endpoint_reusable"])
        self.assertFalse(decision["evaluation_reusable"])
        self.assertEqual(decision["action"], "reevaluate_only")

    def test_failed_and_unsupported_remain_in_coverage_but_not_primary_macro(self):
        scenes = ["A", "B", "C"]
        left_adapters = [adapter_receipt(scene, "left") for scene in scenes]
        left_evaluators = [
            evaluator_receipt(
                adapter,
                metrics={
                    "psnr_db": metric_value(
                        adapter["scene"], "psnr_db-v1", 30.0, unit="dB"
                    )
                },
            )
            for adapter in left_adapters
        ]
        right_a = adapter_receipt("A", "right")
        right_b = adapter_receipt("B", "right", status=FAILED)
        right_c = adapter_receipt("C", "right")
        right_a_eval = evaluator_receipt(
            right_a,
            metrics={
                "psnr_db": metric_value("A", "psnr_db-v1", 31.0, unit="dB")
            },
        )
        right_c_eval = evaluator_receipt(
            right_c,
            metrics={
                "psnr_db": metric_value(
                    "C", "psnr_db-v1", 0.0, unit="dB", status=UNSUPPORTED
                )
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=scenes,
                methods=["left", "right"],
                metric_specs=[
                    {
                        "metric_id": "psnr_db",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    }
                ],
                comparisons=[
                    {
                        "comparison_id": "left-vs-right",
                        "left_method_id": "left",
                        "right_method_id": "right",
                    }
                ],
                adapters=[*left_adapters, right_a, right_b, right_c],
                evaluators=[*left_evaluators, right_a_eval, right_c_eval],
            )

        right_coverage = next(
            row
            for row in summary["coverage"]["method_metric"]
            if row["method_id"] == "right" and row["metric_id"] == "psnr_db"
        )
        self.assertEqual(
            right_coverage["status_counts"],
            {FAILED: 1, SUCCEEDED: 1, UNSUPPORTED: 1},
        )
        macro = summary["primary_paired_macro"][0]
        self.assertEqual(macro["common_scene_count"], 1)
        self.assertEqual(macro["common_scenes"], ["A"])
        comparison = summary["coverage"]["comparison_metric"][0]
        self.assertEqual(len(comparison["excluded"]), 2)
        failures = {
            (row["phase"], row["scene"], row["status"])
            for row in summary["failures_and_unsupported"]
        }
        self.assertIn(("adapter", "B", FAILED), failures)
        self.assertIn(("metric", "C", UNSUPPORTED), failures)
        method_coverage = next(
            row
            for row in summary["coverage"]["endpoint_completion"]
            if row["method_id"] == "right"
        )
        self.assertEqual(method_coverage["valid_scene_count"], 2)
        self.assertEqual(method_coverage["failed_scene_count"], 1)
        self.assertEqual(method_coverage["unsupported_scene_count"], 0)
        self.assertEqual(method_coverage["alias_scene_count"], 0)
        self.assertEqual(len(method_coverage["failure_signatures"]), 1)
        secondary = next(
            row
            for row in summary["secondary_available_scene_macro"]
            if row["method_id"] == "right" and row["metric_id"] == "psnr_db"
        )
        self.assertEqual(secondary["n_valid"], 1)
        self.assertEqual(secondary["scenes"], ["A"])

    def test_paired_macro_scene_intersection_is_metric_specific(self):
        scenes = ["A", "B"]
        adapters = [
            adapter_receipt(scene, method)
            for scene in scenes
            for method in ("left", "right")
        ]
        evaluators = []
        for adapter in adapters:
            scene = adapter["scene"]
            method = adapter["reported_method_id"]
            metrics = {}
            if method == "left" or scene == "A":
                metrics["quality"] = metric_value(
                    scene,
                    "quality-v1",
                    10.0 if method == "left" else 11.0,
                    unit="score",
                )
            if method == "left" or scene == "B":
                metrics["error"] = metric_value(
                    scene,
                    "error-v1",
                    2.0 if method == "left" else 1.0,
                    unit="unit",
                )
            evaluators.append(evaluator_receipt(adapter, metrics=metrics))

        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=scenes,
                methods=["left", "right"],
                metric_specs=[
                    {
                        "metric_id": "quality",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    },
                    {
                        "metric_id": "error",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "min",
                    },
                ],
                comparisons=[
                    {
                        "comparison_id": "left-vs-right",
                        "left_method_id": "left",
                        "right_method_id": "right",
                    }
                ],
                adapters=adapters,
                evaluators=evaluators,
            )

        macros = {row["metric_id"]: row for row in summary["primary_paired_macro"]}
        self.assertEqual(macros["quality"]["common_scenes"], ["A"])
        self.assertEqual(macros["error"]["common_scenes"], ["B"])
        self.assertEqual(macros["quality"]["right_preferred_delta"], 1.0)
        self.assertEqual(macros["error"]["right_preferred_delta"], 1.0)

    def test_three_method_group_reuses_one_common_scene_set_for_every_pair(self):
        scenes = ["A", "B"]
        methods = ["baseline", "second", "third"]
        adapters = [
            adapter_receipt(scene, method) for scene in scenes for method in methods
        ]
        evaluators = []
        for adapter in adapters:
            scene = adapter["scene"]
            method = adapter["reported_method_id"]
            status = FAILED if scene == "B" and method == "third" else SUCCEEDED
            evaluators.append(
                evaluator_receipt(
                    adapter,
                    metrics={
                        "quality": metric_value(
                            scene,
                            "quality-v1",
                            {"baseline": 10.0, "second": 11.0, "third": 12.0}[method],
                            unit="score",
                            status=status,
                        )
                    },
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=scenes,
                methods=methods,
                metric_specs=[
                    {
                        "metric_id": "quality",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    }
                ],
                comparisons=[
                    {
                        "comparison_id": "all-methods",
                        "method_ids": methods,
                        "baseline_method_id": "baseline",
                    }
                ],
                adapters=adapters,
                evaluators=evaluators,
            )
        pairs = summary["primary_paired_macro"]
        self.assertEqual(len(pairs), 2)
        self.assertEqual({tuple(row["common_scenes"]) for row in pairs}, {("A",)})
        self.assertEqual(
            summary["primary_common_scene_macro"][0]["common_scenes"], ["A"]
        )
        coverage = summary["coverage"]["comparison_metric"][0]
        self.assertEqual(coverage["common_scene_count"], 1)
        self.assertEqual(coverage["excluded"][0]["method_status"]["third"], FAILED)

    def test_unknown_fields_nonfinite_values_and_scope_mismatch_fail_closed(self):
        adapter = adapter_receipt("Building", "method")
        unknown = dict(adapter)
        unknown["unregistered_field"] = "must-fail"
        with self.assertRaises(ReceiptContractError):
            finalize_adapter_receipt(unknown)

        with self.assertRaises(ReceiptContractError):
            evaluator_receipt(
                adapter,
                metrics={
                    "psnr_db": metric_value(
                        "Building", "psnr_db-v1", float("nan"), unit="dB"
                    )
                },
            )

        wrong_scope_metric = metric_value(
            "Building",
            "psnr_db-v1",
            30.0,
            unit="dB",
            aggregation="frame_macro",
        )
        evaluator = evaluator_receipt(
            adapter, metrics={"psnr_db": wrong_scope_metric}
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ReceiptContractError):
                study_summary(
                    Path(directory),
                    scenes=["Building"],
                    methods=["method"],
                    metric_specs=[
                        {
                            "metric_id": "psnr_db",
                            "evaluator_id": "formal-evaluator-v1",
                            "direction": "max",
                        }
                    ],
                    comparisons=[],
                    adapters=[adapter],
                    evaluators=[evaluator],
                )

    def test_method_specific_expected_matrix_marks_other_scenes_not_required(self):
        adapter = adapter_receipt("A", "six-scene-method")
        evaluator = evaluator_receipt(adapter)
        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=["A", "B"],
                methods=["six-scene-method"],
                metric_specs=[
                    {
                        "metric_id": "psnr_db",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    }
                ],
                comparisons=[],
                adapters=[adapter],
                evaluators=[evaluator],
                expected_scenes_by_method={"six-scene-method": ["A"]},
            )
        metric_coverage = summary["coverage"]["method_metric"][0]
        self.assertEqual(metric_coverage["expected_scene_count"], 1)
        self.assertEqual(metric_coverage["not_required_scenes"], ["B"])
        self.assertEqual(metric_coverage["status_counts"], {SUCCEEDED: 1})

    def test_formal_training_cost_and_shared_evaluation_accounting_are_separate(self):
        with self.assertRaises(ReceiptContractError):
            adapter_receipt("A", "method", run_scope="formal")
        training = cost_record(
            "formal-training",
            stage="training",
            batch_id="formal-training-batch",
            wall_time_s=100.0,
            gpu_seconds=90.0,
        )
        adapter = adapter_receipt(
            "A", "method", run_scope="formal", costs=[training]
        )
        shared_evaluation = cost_record(
            "shared-evaluation",
            stage="evaluation",
            batch_id="shared-evaluation-batch",
            wall_time_s=5.0,
            gpu_seconds=1.0,
            cost_class="shared_excluded",
            accounting="excluded_shared",
            exclusion_reason="common formal evaluator",
            reported_wall_time_s=0.0,
            reported_gpu_seconds=0.0,
            reported_peak_allocated_vram_mb=0.0,
            reported_peak_reserved_vram_mb=0.0,
            reported_disk_bytes=0,
        )
        evaluator = evaluator_receipt(adapter, costs=[shared_evaluation])
        with tempfile.TemporaryDirectory() as directory:
            summary = study_summary(
                Path(directory),
                scenes=["A"],
                methods=["method"],
                metric_specs=[
                    {
                        "metric_id": "psnr_db",
                        "evaluator_id": "formal-evaluator-v1",
                        "direction": "max",
                    }
                ],
                comparisons=[],
                adapters=[adapter],
                evaluators=[evaluator],
            )
        method_total = summary["cost"]["reported_method_accounting"]["totals"][0]
        self.assertEqual(method_total["wall_time_s"], 100.0)
        self.assertEqual(
            summary["cost"]["shared_excluded_execution"]["wall_time_s_sum"], 5.0
        )
        self.assertFalse(
            summary["cost"]["shared_excluded_execution"]["enters_reported_method_cost"]
        )

    def test_external_fairness_metadata_and_zero_modification_alias_are_fail_closed(self):
        incomplete_external = {
            "kind": "external",
            "repository_url": "https://github.com/example/official",
        }
        with self.assertRaises(ReceiptContractError):
            adapter_receipt("A", "external", method_origin=incomplete_external)

        source = adapter_receipt("A", "source")
        alias = alias_receipt(source, "scsp_refit_f3")
        bad_alias = dict(alias)
        bad_alias["zero_modification"] = dict(alias["zero_modification"])
        bad_alias["zero_modification"]["modified_count"] = 1
        with self.assertRaises(ReceiptContractError):
            finalize_alias_receipt(bad_alias)

    def test_bundled_json_schema_is_valid_json_and_lists_all_document_types(self):
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "schemas"
            / "uav_tgs_baseline_receipts_v1.schema.json"
        )
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(
            set(payload["$defs"]),
            {
                "sha256",
                "terminalStatus",
                "identity",
                "scopedProvenance",
                "renderProfile",
                "timingScope",
                "executionEnvironment",
                "batchExecution",
                "reportedCost",
                "costRecord",
                "failure",
                "unsupported",
                "modelProfile",
                "endpoint",
                "metricScope",
                "metric",
                "methodOrigin",
                "adapterReceipt",
                "evaluatorReceipt",
                "zeroModification",
                "aliasReceipt",
                "boundReceipt",
                "studyManifest",
                "studySummary",
            },
        )
        for definition in (
            "identity",
            "scopedProvenance",
            "costRecord",
            "metricScope",
            "metric",
            "adapterReceipt",
            "evaluatorReceipt",
            "aliasReceipt",
            "studyManifest",
            "studySummary",
        ):
            self.assertFalse(payload["$defs"][definition]["additionalProperties"])


if __name__ == "__main__":
    unittest.main()
