from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys

import pytest

from tools.hold8_minimal_receipts import (
    ALIAS_SCHEMA,
    EVALUATION_SCHEMA,
    FAILED,
    FORMAL_SCENES,
    Hold8ReceiptError,
    METRIC_NA,
    METRIC_VALID,
    PROTOCOL_ID,
    SUCCEEDED,
    TRAINING_SCHEMA,
    UNSUPPORTED,
    finalize_evaluation_receipt,
    finalize_training_receipt,
    make_noop_alias_receipt,
    summarize_six_scene_study,
    validate_noop_alias_receipt,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
COMMIT_A = "1" * 40
COMMIT_B = "2" * 40
COMMIT_C = "3" * 40


def training_draft(
    scene: str,
    method: str,
    *,
    status: str = SUCCEEDED,
    reported_time: float = 100.0,
    host: str = "900",
    command: str | None = None,
    config_sha: str = SHA_B,
    source_scoped_sha: str = SHA_C,
    repository_commit: str = COMMIT_A,
    source_training_commit: str = COMMIT_A,
    current_runner_commit: str = COMMIT_A,
    reused_endpoint: bool = False,
    reuse_reason: str | None = None,
) -> dict:
    succeeded = status == SUCCEEDED
    return {
        "scene": scene,
        "method_id": method,
        "repository": "https://github.com/WongHK101/UAV-TGS",
        "repository_commit": repository_commit,
        "source_training_commit": source_training_commit,
        "current_runner_commit": current_runner_commit,
        "source_scoped_code_sha256": source_scoped_sha,
        "reused_endpoint": reused_endpoint,
        "reuse_reason": reuse_reason,
        "runtime_patch_sha256": None,
        "recipe": method,
        "config_sha256": config_sha,
        "seed": 0,
        "split_manifest_sha256": SHA_C,
        "data_sha256": SHA_D,
        "camera_sha256": SHA_E,
        "range_sha256": SHA_F,
        "lut_sha256": SHA_A,
        "host": host,
        "gpu": "RTX PRO 6000",
        "command": command or f"python train.py --scene {scene} --recipe {method}",
        "endpoint_sha256": SHA_A if succeeded else None,
        "completion_status": status,
        "failure_reason": None if succeeded else "synthetic training failure",
        "batch_execution_wall_time_s": 0.0 if reused_endpoint else 90.0,
        "reported_method_wall_time_s": reported_time if succeeded else None,
        "peak_vram_bytes": 12_000_000_000,
        "model_size_bytes": 50_000_000 if succeeded else None,
        "gaussian_count": 10_000 if succeeded else None,
    }


def evaluation_draft(
    metrics: dict,
    *,
    status: str = SUCCEEDED,
    reference_sha: str | None = SHA_D,
) -> dict:
    return {
        "evaluator_code_sha256": SHA_E,
        "evaluator_config_sha256": SHA_F,
        "reference_sha256": reference_sha,
        "completion_status": status,
        "failure_reason": None if status == SUCCEEDED else "synthetic eval failure",
        "metrics": metrics if status == SUCCEEDED else {},
        "render_fps": 42.0 if status == SUCCEEDED else None,
    }


def valid(value: float) -> dict:
    return {"status": METRIC_VALID, "value": value}


def na(reason: str = "method does not export RGB") -> dict:
    return {"status": METRIC_NA, "reason": reason}


def make_pair(
    scene: str,
    method: str,
    metrics: dict,
    *,
    training_status: str = SUCCEEDED,
) -> tuple[dict, dict | None]:
    training = finalize_training_receipt(
        training_draft(scene, method, status=training_status)
    )
    if training_status != SUCCEEDED:
        return training, None
    evaluation = finalize_evaluation_receipt(evaluation_draft(metrics), training)
    return training, evaluation


def test_flat_signatures_bind_semantics_without_recursive_self_hash() -> None:
    first = finalize_training_receipt(training_draft("Building", "raw_f3"))
    relocated = finalize_training_receipt(
        training_draft("Building", "raw_f3", host="901")
    )
    unrelated_runner_change = finalize_training_receipt(
        training_draft(
            "Building",
            "raw_f3",
            repository_commit=COMMIT_B,
            source_training_commit=COMMIT_B,
            current_runner_commit=COMMIT_C,
        )
    )
    changed_scoped_code = finalize_training_receipt(
        training_draft("Building", "raw_f3", source_scoped_sha=SHA_D)
    )
    changed_config = finalize_training_receipt(
        training_draft("Building", "raw_f3", config_sha=SHA_C)
    )
    changed_command = finalize_training_receipt(
        training_draft(
            "Building",
            "raw_f3",
            command="python train.py --scene Building --recipe raw_f3 --unexpected-override",
        )
    )

    assert first["schema"] == TRAINING_SCHEMA
    assert first["protocol_id"] == PROTOCOL_ID
    assert first["training_signature"] == relocated["training_signature"]
    assert first["training_signature"] == unrelated_runner_change["training_signature"]
    assert first["training_signature"] != changed_scoped_code["training_signature"]
    assert first["training_signature"] != changed_config["training_signature"]
    assert first["training_signature"] != changed_command["training_signature"]
    assert "receipt_sha256" not in first

    evaluation = finalize_evaluation_receipt(
        evaluation_draft({"t_psnr": valid(30.0)}), first
    )
    changed_metrics = finalize_evaluation_receipt(
        evaluation_draft({"t_psnr": valid(31.0)}), first
    )
    changed_reference = finalize_evaluation_receipt(
        evaluation_draft({"t_psnr": valid(30.0)}, reference_sha=SHA_E), first
    )
    assert evaluation["schema"] == EVALUATION_SCHEMA
    # A signature identifies evaluator provenance, not the resulting scalar values.
    assert evaluation["evaluation_signature"] == changed_metrics["evaluation_signature"]
    assert evaluation["evaluation_signature"] != changed_reference["evaluation_signature"]
    assert "receipt_sha256" not in evaluation


def test_reused_endpoint_has_zero_batch_cost_and_preserves_formal_method_cost() -> None:
    reused = finalize_training_receipt(
        training_draft(
            "Building",
            "raw_f3",
            reported_time=321.5,
            source_training_commit=COMMIT_A,
            current_runner_commit=COMMIT_C,
            reused_endpoint=True,
            reuse_reason="scoped provenance matches locked formal endpoint",
        )
    )
    assert reused["batch_execution_wall_time_s"] == 0.0
    assert reused["reported_method_wall_time_s"] == 321.5
    assert reused["reuse_reason"]

    nonzero_batch = training_draft(
        "Building",
        "raw_f3",
        reused_endpoint=True,
        reuse_reason="scoped provenance match",
    )
    nonzero_batch["batch_execution_wall_time_s"] = 1.0
    with pytest.raises(Hold8ReceiptError, match="must be 0"):
        finalize_training_receipt(nonzero_batch)

    zero_reported = training_draft(
        "Building",
        "raw_f3",
        reported_time=0.0,
        reused_endpoint=True,
        reuse_reason="scoped provenance match",
    )
    with pytest.raises(Hold8ReceiptError, match="positive formal cost"):
        finalize_training_receipt(zero_reported)

    fresh_with_reason = training_draft("Building", "raw_f3")
    fresh_with_reason["reuse_reason"] = "should not be present"
    with pytest.raises(Hold8ReceiptError, match="must be null"):
        finalize_training_receipt(fresh_with_reason)

    short_commit = training_draft("Building", "raw_f3")
    short_commit["current_runner_commit"] = "f9a99b1"
    with pytest.raises(Hold8ReceiptError, match="40-hex"):
        finalize_training_receipt(short_commit)


def test_exact_alias_has_zero_batch_cost_and_inherits_nonzero_method_cost() -> None:
    training = finalize_training_receipt(
        training_draft("Building", "raw_f3", reported_time=321.5)
    )
    evaluation = finalize_evaluation_receipt(
        evaluation_draft({"t_psnr": valid(29.5)}), training
    )
    alias = make_noop_alias_receipt(
        training,
        evaluation,
        alias_method_id="scsp_refit_f3",
        proof_sha256=SHA_F,
    )

    assert alias["schema"] == ALIAS_SCHEMA
    assert alias["modified_count"] == 0
    assert alias["independent_endpoint_run"] is False
    assert alias["independent_performance_claim"] is False
    assert alias["batch_execution_wall_time_s"] == 0.0
    assert alias["additional_alias_cost_s"] == 0.0
    assert alias["reported_method_wall_time_s"] == 321.5

    tampered = copy.deepcopy(alias)
    tampered["reported_method_wall_time_s"] = 0.0
    with pytest.raises(Hold8ReceiptError, match="reported_method_wall_time_s"):
        validate_noop_alias_receipt(tampered, training, evaluation)


def test_metric_specific_common_scene_macro_never_zero_fills_failures_or_na() -> None:
    methods = ["raw_f3", "legacy_l", "external"]
    training_receipts: list[dict] = []
    evaluation_receipts: list[dict] = []

    for scene_index, scene in enumerate(FORMAL_SCENES):
        for method_index, method in enumerate(methods):
            if method == "legacy_l" and scene == "Orchard":
                training, evaluation = make_pair(
                    scene,
                    method,
                    {},
                    training_status=FAILED,
                )
            else:
                rgb = na() if method == "external" and scene == "PVpanel" else valid(
                    20.0 + scene_index + method_index
                )
                training, evaluation = make_pair(
                    scene,
                    method,
                    {
                        "rgb_psnr": rgb,
                        "t_psnr": valid(30.0 + scene_index + method_index),
                    },
                )
            training_receipts.append(training)
            if evaluation is not None:
                evaluation_receipts.append(evaluation)

    summary = summarize_six_scene_study(
        methods=methods,
        metrics=["rgb_psnr", "t_psnr"],
        training_receipts=training_receipts,
        evaluation_receipts=evaluation_receipts,
    )

    assert summary["all_methods_complete_all_scenes"] is False
    assert summary["coverage"]["legacy_l"]["completed_scene_count"] == 5
    assert summary["coverage"]["legacy_l"]["failed_scene_count"] == 1

    # Orchard is excluded because legacy failed. PVpanel is excluded from RGB
    # only because the external method declared RGB N/A.
    assert summary["metrics"]["t_psnr"]["common_scenes"] == list(FORMAL_SCENES[:-1])
    assert summary["metrics"]["rgb_psnr"]["common_scenes"] == [
        "Building",
        "InternalRoad",
        "TransmissionTower",
        "Urban20K",
    ]
    assert summary["metrics"]["rgb_psnr"]["method_coverage"]["external"][
        "na_scenes"
    ] == ["PVpanel"]

    # A failed scene is omitted, not silently inserted as zero.
    expected_legacy_t = sum(31.0 + index for index in range(5)) / 5
    assert summary["metrics"]["t_psnr"]["scene_macro"]["legacy_l"] == pytest.approx(
        expected_legacy_t
    )


def test_unsupported_scene_is_coverage_only_and_never_enters_macro() -> None:
    methods = ["raw_f3", "external"]
    training_receipts: list[dict] = []
    evaluation_receipts: list[dict] = []
    for scene_index, scene in enumerate(FORMAL_SCENES):
        raw_training, raw_evaluation = make_pair(
            scene, "raw_f3", {"t_psnr": valid(30.0 + scene_index)}
        )
        training_receipts.append(raw_training)
        assert raw_evaluation is not None
        evaluation_receipts.append(raw_evaluation)

        external_status = UNSUPPORTED if scene == "PVpanel" else SUCCEEDED
        external_training, external_evaluation = make_pair(
            scene,
            "external",
            {"t_psnr": valid(20.0 + scene_index)},
            training_status=external_status,
        )
        training_receipts.append(external_training)
        if external_evaluation is not None:
            evaluation_receipts.append(external_evaluation)

    summary = summarize_six_scene_study(
        methods=methods,
        metrics=["t_psnr"],
        training_receipts=training_receipts,
        evaluation_receipts=evaluation_receipts,
    )

    assert summary["coverage"]["external"]["unsupported_scene_count"] == 1
    assert summary["metrics"]["t_psnr"]["common_scene_count"] == 5
    assert "PVpanel" not in summary["metrics"]["t_psnr"]["common_scenes"]
    # The raw method's macro is paired to the exact same five scenes rather
    # than including the unsupported scene or inserting a synthetic zero.
    expected_raw = statistics.fmean(
        30.0 + index
        for index, scene in enumerate(FORMAL_SCENES)
        if scene != "PVpanel"
    )
    assert summary["metrics"]["t_psnr"]["scene_macro"]["raw_f3"] == pytest.approx(
        expected_raw
    )


def test_alias_is_valid_scene_but_not_an_independent_run_or_zero_cost_method() -> None:
    raw_trainings: list[dict] = []
    raw_evaluations: list[dict] = []
    scsp_trainings: list[dict] = []
    scsp_evaluations: list[dict] = []
    aliases: list[dict] = []

    for index, scene in enumerate(FORMAL_SCENES):
        raw_training, raw_evaluation = make_pair(
            scene, "raw_f3", {"temperature_mae": valid(1.0 + index)}
        )
        assert raw_evaluation is not None
        raw_trainings.append(raw_training)
        raw_evaluations.append(raw_evaluation)
        if scene == "Building":
            aliases.append(
                make_noop_alias_receipt(
                    raw_training,
                    raw_evaluation,
                    alias_method_id="scsp_refit_f3",
                    proof_sha256=SHA_F,
                )
            )
        else:
            scsp_training, scsp_evaluation = make_pair(
                scene,
                "scsp_refit_f3",
                {"temperature_mae": valid(0.9 + index)},
            )
            assert scsp_evaluation is not None
            scsp_trainings.append(scsp_training)
            scsp_evaluations.append(scsp_evaluation)

    summary = summarize_six_scene_study(
        methods=["raw_f3", "scsp_refit_f3"],
        metrics=["temperature_mae"],
        training_receipts=raw_trainings + scsp_trainings,
        evaluation_receipts=raw_evaluations + scsp_evaluations,
        alias_receipts=aliases,
    )

    scsp = summary["coverage"]["scsp_refit_f3"]
    assert summary["all_methods_complete_all_scenes"] is True
    assert scsp["completed_scene_count"] == 6
    assert scsp["independent_completed_scene_count"] == 5
    assert scsp["alias_scene_count"] == 1
    assert scsp["batch_execution_wall_time_s"] == 5 * 90.0
    # The alias inherits Raw-F3's method cost rather than pretending the
    # configuration costs zero from the common RGB anchor.
    assert scsp["reported_method_wall_time_s"] == 6 * 100.0
    assert summary["metrics"]["temperature_mae"]["common_scene_count"] == 6


def test_nonfinite_metrics_and_alias_independent_endpoint_conflicts_fail_closed() -> None:
    training = finalize_training_receipt(training_draft("Building", "raw_f3"))
    with pytest.raises(Hold8ReceiptError, match="finite"):
        finalize_evaluation_receipt(
            evaluation_draft({"temperature_mae": valid(float("nan"))}), training
        )

    evaluation = finalize_evaluation_receipt(
        evaluation_draft({"temperature_mae": valid(1.0)}), training
    )
    alias = make_noop_alias_receipt(
        training,
        evaluation,
        alias_method_id="scsp_refit_f3",
        proof_sha256=SHA_F,
    )
    independent_alias_training = finalize_training_receipt(
        training_draft("Building", "scsp_refit_f3")
    )
    independent_alias_evaluation = finalize_evaluation_receipt(
        evaluation_draft({"temperature_mae": valid(1.0)}),
        independent_alias_training,
    )
    with pytest.raises(Hold8ReceiptError, match="must not also claim"):
        summarize_six_scene_study(
            methods=["raw_f3", "scsp_refit_f3"],
            metrics=["temperature_mae"],
            training_receipts=[training, independent_alias_training],
            evaluation_receipts=[evaluation, independent_alias_evaluation],
            alias_receipts=[alias],
        )


def test_cli_finalizes_flat_training_receipt(tmp_path: Path) -> None:
    draft_path = tmp_path / "draft.json"
    output_path = tmp_path / "receipt.json"
    draft_path.write_text(json.dumps(training_draft("Building", "raw_f3")), encoding="utf-8")
    script = Path(__file__).parents[1] / "tools" / "hold8_minimal_receipts.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "finalize-training",
            "--input",
            str(draft_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=Path(__file__).parents[1],
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == TRAINING_SCHEMA
    assert len(payload["training_signature"]) == 64
    assert "receipt_sha256" not in payload
