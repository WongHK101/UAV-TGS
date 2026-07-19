from __future__ import annotations

import hashlib
import copy
import json
import re
from pathlib import Path

import pytest

from tools.baseline_evaluation_receipts import (
    ReceiptContractError,
    finalize_study_manifest,
    validate_study_manifest,
)

from tools.build_aaai27_final_asset_inventory import (
    EXPECTED_SCENES,
    EXPANSION_SCENES,
    EXTERNAL_METHODS,
    REPRESENTATIVE_SCENES,
    build_inventory,
    canonical_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "protocols" / "aaai27_final_experiment_protocol_v1.json"
MARKDOWN_PATH = REPO_ROOT / "protocols" / "aaai27_final_experiment_protocol_v1.md"
INVENTORY_PATH = REPO_ROOT / "protocols" / "aaai27_final_experiment_asset_inventory_v1.json"
STUDY_PATH = REPO_ROOT / "protocols" / "aaai27_final_study_manifest_v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_study_manifest_binds_protocol_matrix_and_geometry() -> None:
    payload = _load(STUDY_PATH)
    validate_study_manifest(payload, base=STUDY_PATH.parent)

    bad_geometry = copy.deepcopy(payload)
    bad_geometry["geometry_contract"]["thresholds_m"][-1] = 21
    with pytest.raises(ReceiptContractError):
        finalize_study_manifest(bad_geometry)

    shrunken_group = copy.deepcopy(payload)
    shrunken_group["comparisons"][0]["method_ids"] = [
        "raw_f3",
        "scsp_refit_f3",
    ]
    with pytest.raises(ReceiptContractError):
        finalize_study_manifest(shrunken_group)

    wrong_protocol = copy.deepcopy(payload)
    wrong_protocol["protocol"]["sha256"] = hashlib.sha256(
        b"wrong-protocol"
    ).hexdigest()
    wrong_protocol = finalize_study_manifest(wrong_protocol)
    with pytest.raises(ReceiptContractError):
        validate_study_manifest(wrong_protocol, base=STUDY_PATH.parent)


def test_required_protocol_identity_and_file_hashes_are_bound() -> None:
    protocol = _load(PROTOCOL_PATH)
    required = {
        "protocol_id",
        "version",
        "protocol_version",
        "creation_timestamp",
        "collection_hash",
        "representative_scenes",
        "internal_methods",
        "external_methods",
        "geometry_thresholds_m",
        "mandatory_review_gates",
        "execution_phases",
        "prohibited_tasks",
        "protocol_markdown_sha256",
        "code_commit",
    }
    assert required <= set(protocol)
    assert protocol["protocol_id"] == "uav-tgs-aaai27-final-experiment-v1"
    assert protocol["version"] == "1.0.0"
    assert protocol["protocol_version"] == protocol["version"]
    assert protocol["collection_hash"] == "c6f32a1c44f49a725a62beeb105ffb37f5de265c5b513f5d01bd303439d60832"
    assert protocol["protocol_markdown_sha256"] == _sha256(MARKDOWN_PATH)
    assert protocol["collection"]["asset_inventory"]["sha256"] == _sha256(INVENTORY_PATH)
    commit = protocol["code_commit"]
    assert commit == "__PHASE0_CODE_COMMIT__" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_scene_and_method_matrices_are_exact() -> None:
    protocol = _load(PROTOCOL_PATH)
    inventory = _load(INVENTORY_PATH)
    representative = list(REPRESENTATIVE_SCENES)
    expansion = list(EXPANSION_SCENES)
    assert protocol["representative_scenes"] == representative
    assert protocol["expansion_scenes"] == expansion
    assert sorted(protocol["collection"]["scenes"]) == sorted(EXPECTED_SCENES)
    internal = {method["display_name"]: method["scenes"] for method in protocol["internal_methods"]}
    assert internal == inventory["method_scene_matrix"]["internal"]
    assert internal["Raw-F3"] == representative
    assert internal["Adaptive Opacity + Scale-Clamp"] == representative
    assert internal["SCSP-Refit+F3"] == representative + expansion
    external = inventory["method_scene_matrix"]["external"]
    assert list(external) == list(EXTERNAL_METHODS)
    assert all(scenes == representative for scenes in external.values())


def test_geometry_reporting_uses_guard_and_preregistered_fallback() -> None:
    protocol = _load(PROTOCOL_PATH)
    assert protocol["geometry_thresholds_m"] == [0.25, 0.5, 1, 2, 5, 10, 15, 20]
    assert protocol["depth_definitions"] == [
        "transmittance_median",
        "alpha_weighted_expected",
        "maximum_contribution_surface",
    ]
    reporting = protocol["geometry_reporting"]
    assert reporting["main_table_selection_source"] == ["guard"]
    assert "train" not in reporting["main_table_selection_source"]
    assert reporting["selection_must_precede_test_aggregation"] is True
    assert reporting["fallback_if_uniform_guard_reference_cannot_complete_m"] == [1, 5, 10]
    assert reporting["fallback_may_read_external_test"] is False
    assert reporting["all_eight_thresholds_remain_supplementary"] is True
    completeness = reporting["guard_completeness_receipt"]
    assert completeness["required"] is True
    assert completeness["scene_scope"] == list(REPRESENTATIVE_SCENES)
    assert completeness["complete_condition"] == (
        "all_six_scenes_COMPLETE_under_one_registered_reference_protocol"
    )
    assert completeness["complete_action"] == "run_guard_only_selection_algorithm"
    assert completeness["incomplete_action"] == "activate_fixed_fallback_1_5_10m"
    assert completeness["freeze_before_test_aggregation"] is True
    assert completeness["forbidden_inputs"] == [
        "train_metrics",
        "test_metrics",
        "external_method_results",
    ]
    assert protocol["data_use_policy"]["test"] == ["final_reporting_only"]
    assert protocol["data_use_policy"]["test_tuning_forbidden"] is True
    assert "hotspot_threshold_selection" in protocol["data_use_policy"]["train"]
    assert "hotspot_threshold_selection" not in protocol["data_use_policy"]["guard"]
    algorithm = reporting["main_table_selection_algorithm"]
    assert algorithm["local_candidates_m"] == [0.25, 0.5, 1, 2]
    assert algorithm["mid_candidates_m"] == [5, 10]
    assert algorithm["large_candidates_m"] == [15, 20]
    assert algorithm["ties"] == "choose_smaller_threshold"


def test_only_nine_review_gates_exist_and_external_completion_is_nonblocking() -> None:
    protocol = _load(PROTOCOL_PATH)
    gates = protocol["mandatory_review_gates"]
    assert len(gates) == 9
    assert len({gate["gate_id"] for gate in gates}) == 9
    assert all(gate["definition_status"] == "DEFINED" for gate in gates)
    assert all("status" not in gate for gate in gates)
    assert gates[0]["runtime_status_snapshot"] == "WAITING_GPT"
    assert all(
        gate["runtime_status_snapshot"] == "BLOCKED_BY_PREDECESSOR"
        for gate in gates[1:]
    )
    review = protocol["review_policy"]
    assert review["mandatory_review_gates_are_static_definitions"] is True
    assert review["runtime_status_authority"] == "hash_bound_gate_receipt"
    assert review["phase0_runtime_status_snapshot"] == "WAITING_GPT"
    assert review["future_gate_runtime_status_snapshot"] == "BLOCKED_BY_PREDECESSOR"
    assert [phase["phase_id"] for phase in protocol["execution_phases"]] == list(range(9))
    external_phases = protocol["execution_phases"][3:8]
    assert [phase["method"] for phase in external_phases] == [
        method["method_id"] for method in protocol["external_methods"]
    ]
    assert all(phase["six_scene_package_wait"] is False for phase in external_phases)
    contract = protocol["external_method_contract"]
    assert contract["building_approval_required_before_remaining_five"] is True
    assert contract["six_scene_completion_package_requires_waiting_gpt"] is False
    assert protocol["review_policy"]["additional_blocking_gates_forbidden"] is True


def test_prohibited_scope_is_explicit() -> None:
    prohibited = set(_load(PROTOCOL_PATH)["prohibited_tasks"])
    required = {
        "OCT-Scalar",
        "OCT-Residual",
        "surface-aware Resplat",
        "OGS-v1",
        "OGS-v2",
        "new heuristics",
        "multi-seed experiments",
        "test-set tuning",
        "MS-Splatting",
        "paper writing",
        "additional blocking review gates",
    }
    assert required <= prohibited


def test_checked_in_inventory_has_valid_payload_hash_and_collection_counts() -> None:
    inventory = _load(INVENTORY_PATH)
    claimed = inventory.pop("inventory_payload_sha256")
    assert claimed == canonical_sha256(inventory)
    assert inventory["collection"]["counts"] == {
        "total": 8232,
        "train": 6763,
        "guard": 445,
        "test": 1024,
    }
    assert len(inventory["scene_inventory"]) == 11
    assert sum(row["counts"]["total"] for row in inventory["scene_inventory"]) == 8232
    assert inventory["path_policy"]["machine_absolute_paths_recorded"] is False


def test_phase0_availability_and_external_sources_are_explicit() -> None:
    inventory = _load(INVENTORY_PATH)
    snapshot = inventory["phase0_availability_snapshot"]
    scene_assets = snapshot["scene_assets"]
    assert sorted(scene_assets["formal_split_and_dataset_present"]) == sorted(EXPECTED_SCENES)
    assert scene_assets["radiometry_complete"] == ["Building", "InternalRoad"]
    assert scene_assets["radiometry_decoded_temperature_npy_only"] == ["Urban20K"]
    assert scene_assets["guard_and_reference_verified"] == ["Building", "InternalRoad"]
    endpoints = snapshot["internal_formal_endpoints"]
    assert endpoints["Building"]["scsp_refit_f3"] == "present_noop_alias_to_raw_f3"
    assert endpoints["InternalRoad"]["scsp_refit_f3"] == "present"
    phase1_jobs = snapshot["planned_new_training_jobs"]["phase_1"]
    assert phase1_jobs["formula"] == "12+2R"
    assert phase1_jobs["expanded_formula"] == "sum_s(3+2*r_s)=12+2R"
    assert phase1_jobs["candidate_scenes"] == list(REPRESENTATIVE_SCENES[2:])
    assert phase1_jobs["aggregate_definition"] == "R=sum_s r_s"
    assert phase1_jobs["aggregate_range"] == [0, 4]
    assert phase1_jobs["estimated_range"] == [12, 20]
    assert "unresolved in Phase 0" in phase1_jobs["note"]
    assert "without reading guard/test data" in phase1_jobs["note"]
    phase2_jobs = snapshot["planned_new_training_jobs"]["phase_2"]
    assert phase2_jobs["formula"] == "10+M5"
    assert phase2_jobs["expanded_formula"] == "sum_s(2+m_s)=10+M5"
    assert phase2_jobs["candidate_scenes"] == list(EXPANSION_SCENES)
    assert phase2_jobs["aggregate_definition"] == "M5=sum_s m_s"
    assert phase2_jobs["aggregate_range"] == [0, 5]
    assert phase2_jobs["estimated_range"] == [10, 15]

    sources = {row["method"]: row for row in snapshot["external_sources"]}
    assert list(sources) == list(EXTERNAL_METHODS)
    assert sources["MMOne"]["source_commit"] == "f49fc4e7a1fb6d6444ba5a75b176e9b6cbcca901"
    assert sources["MMOne"]["archive_sha256"] == (
        "91b465c2d265b02ebd800179821119ae9f84fe3647361c6293ada7a5c4aeb793"
    )
    assert all(row["source_snapshot_status"] == "available" for row in sources.values())
    assert all(row["formal_adapter_status"] == "missing" for row in sources.values())


def test_conditional_job_matrix_defines_r_and_m5_without_test_data() -> None:
    protocol = _load(PROTOCOL_PATH)
    status = protocol["phase0_asset_status"]

    phase1 = status["phase1_new_jobs"]
    assert phase1["formula"] == "12+2R"
    assert phase1["expanded_formula"] == "sum_s(3+2*r_s)=12+2R"
    assert phase1["candidate_scenes"] == [
        "PVpanel",
        "TransmissionTower",
        "Urban20K",
        "Orchard",
    ]
    assert "modified_gaussian_count>0" in phase1["per_scene_indicator"]
    assert phase1["aggregate_definition"] == "R=sum_s r_s"
    assert phase1["aggregate_range"] == [0, 4]
    assert phase1["noop_branch_jobs"] == [
        "formal_raw_rgb_anchor",
        "raw_f3",
        "legacy_l",
    ]
    assert phase1["modified_branch_jobs"][-2:] == [
        "scsp_rgb_sh_only_refit",
        "scsp_refit_f3",
    ]

    phase2 = status["phase2_new_jobs"]
    assert phase2["formula"] == "10+M5"
    assert phase2["expanded_formula"] == "sum_s(2+m_s)=10+M5"
    assert phase2["candidate_scenes"] == list(EXPANSION_SCENES)
    assert "modified_gaussian_count>0" in phase2["per_scene_indicator"]
    assert phase2["aggregate_definition"] == "M5=sum_s m_s"
    assert phase2["aggregate_range"] == [0, 5]
    assert phase2["noop_branch_jobs"] == [
        "formal_raw_rgb_anchor",
        "raw_f3_exact_noop_alias_source",
    ]
    assert phase2["modified_branch_jobs"] == [
        "formal_raw_rgb_anchor",
        "scsp_rgb_sh_only_refit",
        "scsp_refit_f3",
    ]

    resolution = status["conditional_job_resolution"]
    assert resolution["source"] == "read_only_train_side_SCSP_manifest"
    assert resolution["modified_condition"] == "modified_gaussian_count>0"
    assert resolution["preflight_receipt_required_before_each_scene_training"] is True
    assert resolution["forbidden_inputs"] == ["guard", "test"]
    assert resolution["rule_retuning"] is False

    inventory_jobs = _load(INVENTORY_PATH)["phase0_availability_snapshot"][
        "planned_new_training_jobs"
    ]
    shared_fields = {
        "formula",
        "expanded_formula",
        "candidate_scenes",
        "per_scene_indicator",
        "aggregate_definition",
        "aggregate_range",
        "noop_branch_jobs",
        "modified_branch_jobs",
        "estimated_range",
    }
    for protocol_jobs, inventory_key in ((phase1, "phase_1"), (phase2, "phase_2")):
        inventory_phase = inventory_jobs[inventory_key]
        assert {key: protocol_jobs[key] for key in shared_fields} == {
            key: inventory_phase[key] for key in shared_fields
        }


def test_contract_covers_provenance_macro_cost_resume_and_audit() -> None:
    protocol = _load(PROTOCOL_PATH)
    output = protocol["formal_endpoint_output_contract"]
    assert output["render_once_per_required_modality"] is True
    assert output["single_depth_contribution_bundle_reused_for_all_thresholds_and_definitions"]
    provenance = protocol["provenance_contract"]
    assert provenance["reuse_receipt_required"] is True
    assert provenance["alias_receipt_required"] is True
    aggregation = protocol["aggregation_contract"]
    assert aggregation["primary"] == "metric_specific_comparison_group_common_scene_macro"
    assert "pair_by_pair_intersections_forbidden" in aggregation["common_scene_rule"]
    efficiency = protocol["efficiency_accounting"]
    assert efficiency["scsp_noop_alias_reported_method_cost"] == "equal_to_raw_f3_not_zero"
    assert "CFR" in efficiency["exclude_shared"]
    operations = protocol["execution_operations"]
    assert operations["device_allocation"] == "not_preallocated_to_900_or_901"
    assert operations["parallel_devices_allowed"] is True
    assert operations["device_environment_receipt_required_per_run"] is True
    assert protocol["phase0_asset_status"]["phase0_package_must_mark_conditional_items_unresolved"]
    assert protocol["package_policy"]["one_payload_sha_manifest_per_package"] is True
    assert "repeated_full_repository_tests" in protocol["audit_scope"]["not_required"]


def test_markdown_contains_no_non_ascii_or_known_mojibake() -> None:
    text = MARKDOWN_PATH.read_text(encoding="utf-8")
    assert text.isascii()
    assert not any(token in text for token in ("鈥", "鈫", "锟", "�"))


def test_inventory_generator_is_deterministic_and_fails_on_scene_drift(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    scene_rows = []
    for index, scene in enumerate(EXPECTED_SCENES):
        digest = f"{index + 1:064x}"
        manifest = {
            "scene": scene,
            "counts": {"total": 1, "train": 1, "guard": 0, "test": 0},
            "input_records_hash": digest,
            "selected_test_blocks_hash": digest,
            "split_hash": digest,
        }
        (scene_dir / f"{scene}.split.json").write_text(json.dumps(manifest), encoding="utf-8")
        scene_rows.append({"scene": scene, "record_count": 1, "input_records_hash": digest})
    collection = {
        "expected_scenes": list(EXPECTED_SCENES),
        "scene_count": 11,
        "counts": {"total": 11, "train": 11, "guard": 0, "test": 0},
        "collection_hash": "a" * 64,
        "collection_split_hash": "b" * 64,
        "formal_rule_hash": "c" * 64,
        "validation": {"status": "passed"},
        "collection_hash_basis": {"scenes": scene_rows},
    }
    collection_path = tmp_path / "collection_manifest.json"
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    first = build_inventory(collection_path, creation_timestamp="2026-01-01T00:00:00Z")
    second = build_inventory(collection_path, creation_timestamp="2026-01-01T00:00:00Z")
    assert first == second
    claimed = first.pop("inventory_payload_sha256")
    assert claimed == canonical_sha256(first)

    collection["expected_scenes"] = list(EXPECTED_SCENES[:-1])
    collection_path.write_text(json.dumps(collection), encoding="utf-8")
    with pytest.raises(ValueError, match="scene set mismatch"):
        build_inventory(collection_path, creation_timestamp="2026-01-01T00:00:00Z")
