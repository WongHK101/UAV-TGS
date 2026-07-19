from __future__ import annotations

"""Build the machine-independent Phase-0 collection/asset inventory.

The generator validates the formal split manifests and emits the frozen
Phase-0 availability snapshot declared below.  It never mutates or scans the
large datasets, checkpoints, or experiment outputs.  Runtime preflight must
bind the declared logical assets to concrete files with scoped hashes before a
formal run starts.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROTOCOL_ID = "uav-tgs-aaai27-final-experiment-v1"
PROTOCOL_VERSION = "1.0.0"
INVENTORY_SCHEMA = "uav-tgs-aaai27-final-asset-inventory-v1"
PREFLIGHT_SNAPSHOT_TIMESTAMP = "2026-07-19T10:15:00+08:00"
EXPECTED_SCENES = (
    "Building",
    "Garden",
    "InternalRoad",
    "Orchard",
    "PVpanel",
    "Plaza",
    "Road",
    "TransmissionTower",
    "Urban100K",
    "Urban20K",
    "Urban50K",
)
REPRESENTATIVE_SCENES = (
    "Building",
    "InternalRoad",
    "PVpanel",
    "TransmissionTower",
    "Urban20K",
    "Orchard",
)
EXPANSION_SCENES = ("Garden", "Plaza", "Road", "Urban50K", "Urban100K")
INTERNAL_METHODS = ("Raw-F3", "SCSP-Refit+F3", "Adaptive Opacity + Scale-Clamp")
EXTERNAL_METHODS = (
    "Thermal3D-GS",
    "ThermalGaussian-OMMG",
    "MMOne",
    "ThermoNeRF",
    "PhysIR-Splat",
)

RADIOMETRY_COMPLETE_SCENES = ("Building", "InternalRoad")
RADIOMETRY_DECODED_ONLY_SCENES = ("Urban20K",)
REFERENCE_VERIFIED_SCENES = ("Building", "InternalRoad")

EXTERNAL_SOURCE_INVENTORY: tuple[dict[str, Any], ...] = (
    {
        "method": "Thermal3D-GS",
        "official_repository": "https://github.com/mzzcdf/Thermal3DGS",
        "source_commit": "03366b2a350ac5db6690dfd7fca51a56ba9e89a7",
        "archive_sha256": "16dcf734912b43c5492fa23bfa6c39066f4e4f7a2dc1596b5e7a395d0e51c68",
        "declared_environment": "python=3.7.13,pytorch=1.12.1,cudatoolkit=11.6",
        "dependency_state": "source snapshot materialized",
        "adapter_risk": (
            "legacy CUDA/PyTorch stack requires an isolated Blackwell compatibility build; "
            "formal shared-camera injection and contribution-equivalent depth export remain unimplemented"
        ),
    },
    {
        "method": "ThermalGaussian-OMMG",
        "official_repository": "https://github.com/chen-hangyu/Thermal-Gaussian-main",
        "source_branch": "OMMG",
        "source_commit": "5d3243ac444ba48215d6ea2391a68163b94267f9",
        "archive_sha256": "9a9c6d2b723a600c18e85101dd069eb46c620f69f2c5ee04dda4bdeb2ea03af0",
        "declared_environment": "python=3.7.13,pytorch=1.12.1,cudatoolkit=11.6",
        "dependency_state": (
            "OMMG rasterizer source materialized; must remain isolated from the API-incompatible main branch"
        ),
        "adapter_risk": (
            "legacy stack and custom rasterizer need an isolated compatibility build; modality-Gaussian "
            "contributions must be mapped to the common depth definitions without substituting another branch"
        ),
    },
    {
        "method": "MMOne",
        "official_repository": "https://github.com/Neal2020GitHub/MMOne",
        "source_commit": "f49fc4e7a1fb6d6444ba5a75b176e9b6cbcca901",
        "archive_sha256": "91b465c2d265b02ebd800179821119ae9f84fe3647361c6293ada7a5c4aeb793",
        "declared_environment": "python=3.7.16,pytorch=1.12.1,cudatoolkit=11.6",
        "dependency_state": "repository-owned rasterizers and GLM sources materialized",
        "adapter_risk": (
            "the official stack predates Blackwell; only mechanical compatibility patches are allowed, and "
            "shared versus thermal-density geometry must retain the official decomposition/densification semantics"
        ),
    },
    {
        "method": "ThermoNeRF",
        "official_repository": "https://github.com/Schindler-EPFL-Lab/thermo-nerf",
        "source_commit": "3f163c1454338bdf81af3a629c25c61443f01121",
        "archive_sha256": "842b6d6387d6ec3888f53b5571cd48c5cec763b03d783e906ebc47a824b027e5",
        "declared_environment": "python>=3.10,nerfstudio>=1.1.5,mlflow>=2.11.1",
        "dependency_state": (
            "source snapshot materialized; optional FLIR extractor is absent and is not required for prepared inputs; "
            "tiny-cuda-nn v2.0 recursive snapshot is available"
        ),
        "adapter_risk": (
            "formal camera injection must disable camera optimization/test-pose refinement, and ray weights must "
            "export equivalent median/expected/maximum-contribution depth"
        ),
    },
    {
        "method": "PhysIR-Splat",
        "official_repository": "https://github.com/JingyuanGao0919/physir-splat",
        "source_commit": "f4d440073cb89ec7bc757909080c7b713089f7cc",
        "archive_sha256": "6f3d892f04e940b4e242484dd461f30a9cd0b56058009c6df5b05f6a26f9fc8f",
        "declared_environment": "torch=1.13.1,torchvision=0.14.1 (upstream requirements)",
        "dependency_state": (
            "missing upstream GLM gitlink completed with g-truc/glm@"
            "5c46b9c07008ae65cb81ab79cd677ecc1934b903 and recorded provenance"
        ),
        "adapter_risk": (
            "the full physical renderer and VGGT-IR path must be verified rather than free thermal SH; physical "
            "input semantics and method-specific preprocessing cost require explicit receipts"
        ),
    },
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(label: str, value: Any) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return text


def _require_counts(label: str, value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}.counts must be an object")
    counts = {key: int(value.get(key, -1)) for key in ("total", "train", "guard", "test")}
    if any(number < 0 for number in counts.values()):
        raise ValueError(f"{label}.counts contains a negative/missing value")
    if counts["train"] + counts["guard"] + counts["test"] != counts["total"]:
        raise ValueError(f"{label}.counts do not sum to total")
    return counts


def _scene_manifest_path(collection_manifest: Path, scene: str) -> Path:
    candidates = (
        collection_manifest.parent / "scenes" / f"{scene}.split.json",
        collection_manifest.parent / f"{scene}.split.json",
    )
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one split manifest for {scene}, found={found}")
    return found[0]


def _logical_asset_requirements() -> list[dict[str, Any]]:
    return [
        {
            "asset_class": "formal_split_and_binding",
            "required_for": ["all_formal_runs"],
            "required_items_per_scene": [
                "scene split manifest",
                "bound_split.json",
                "binding_manifest.json",
                "train/guard/test lists and hashes",
            ],
            "mutation_policy": "immutable",
        },
        {
            "asset_class": "canonical_rgb_thermal_inputs",
            "required_for": ["all_methods"],
            "required_items_per_scene": [
                "CFR RGB observations",
                "canonical Hot-Iron lossless PNG observations",
                "float32 Celsius targets",
                "radiometry protocol/range/LUT identities",
            ],
            "mutation_policy": "read_only",
        },
        {
            "asset_class": "formal_cameras_and_reference_depth",
            "required_for": ["all_metrics"],
            "required_items_per_scene": [
                "formal camera manifest",
                "native-camera binding",
                "reference-depth manifest and valid masks",
                "expected/median/max-contribution render adapters",
            ],
            "mutation_policy": "immutable",
        },
        {
            "asset_class": "internal_method_anchors",
            "required_for": ["phase_1", "phase_2"],
            "required_items_per_scene": [
                "formal raw RGB anchor",
                "SCSP manifest/anchor or no-op alias proof",
                "fixed RGB-refit camera sequence",
                "strict F3 recipe and invariant audit",
                "legacy adaptive opacity/scale-clamp recipe for representative scenes",
            ],
            "mutation_policy": "inputs_read_only_outputs_isolated",
        },
        {
            "asset_class": "external_method_adapter",
            "required_for": ["phase_3", "phase_4", "phase_5", "phase_6", "phase_7"],
            "required_items_per_method": [
                "official repository URL and exact commit",
                "license and environment receipt",
                "adapter source/tests/manifest",
                "Building smoke receipt",
                "Building full-metric qualification package",
            ],
            "mutation_policy": "adapter_only_no_method_retuning",
        },
        {
            "asset_class": "formal_run_outputs",
            "required_for": ["phase_1_through_phase_8"],
            "required_items_per_endpoint": [
                "command and provenance manifest",
                "completion or failure receipt",
                "appearance/temperature/geometry/hotspot metrics",
                "efficiency receipt",
                "fixed qualitative views",
            ],
            "mutation_policy": "isolated_append_only",
        },
    ]


def _phase0_availability_snapshot() -> dict[str, Any]:
    """Return the frozen audit facts; this is not a live filesystem probe."""

    incomplete_radiometry = tuple(
        scene
        for scene in EXPECTED_SCENES
        if scene not in RADIOMETRY_COMPLETE_SCENES + RADIOMETRY_DECODED_ONLY_SCENES
    )
    unverified_reference = tuple(
        scene for scene in EXPECTED_SCENES if scene not in REFERENCE_VERIFIED_SCENES
    )
    missing_endpoint_scenes = tuple(
        scene for scene in EXPECTED_SCENES if scene not in ("Building", "InternalRoad")
    )

    external = []
    for row in EXTERNAL_SOURCE_INVENTORY:
        item = dict(row)
        item.update(
            {
                "source_snapshot_status": "available",
                "formal_adapter_status": "missing",
                "formal_building_endpoint_status": "missing",
            }
        )
        external.append(item)

    return {
        "snapshot_timestamp": PREFLIGHT_SNAPSHOT_TIMESTAMP,
        "snapshot_semantics": (
            "versioned declaration from the local/900 Phase-0 audit; runtime assets must still be bound "
            "by scoped hashes before execution"
        ),
        "source_snapshot_manifest": {
            "project_relative_path": "staging_archives/repository_manifest.json",
            "sha256": "53d593a94adc57e114ea442be6ce9ddd4f10c3f2bb12f9e7dea5ccc963adbb3a",
        },
        "scene_assets": {
            "formal_split_and_dataset_present": list(EXPECTED_SCENES),
            "radiometry_complete": list(RADIOMETRY_COMPLETE_SCENES),
            "radiometry_decoded_temperature_npy_only": list(
                RADIOMETRY_DECODED_ONLY_SCENES
            ),
            "radiometry_incomplete": list(incomplete_radiometry),
            "guard_and_reference_verified": list(REFERENCE_VERIFIED_SCENES),
            "guard_and_reference_not_verified": list(unverified_reference),
        },
        "internal_formal_endpoints": {
            "Building": {
                "raw_f3": "present",
                "scsp_refit_f3": "present_noop_alias_to_raw_f3",
                "legacy_l": "present",
            },
            "InternalRoad": {
                "raw_f3": "present",
                "scsp_refit_f3": "present",
                "legacy_l": "present",
            },
            "missing_scenes": list(missing_endpoint_scenes),
        },
        "planned_new_training_jobs": {
            "phase_1": {
                "formula": "12+2R",
                "expanded_formula": "sum_s(3+2*r_s)=12+2R",
                "candidate_scenes": list(REPRESENTATIVE_SCENES[2:]),
                "per_scene_indicator": (
                    "r_s=1 iff read-only train-side SCSP manifest modified_gaussian_count>0; "
                    "otherwise r_s=0"
                ),
                "aggregate_definition": "R=sum_s r_s",
                "aggregate_range": [0, 4],
                "noop_branch_jobs": ["formal_raw_rgb_anchor", "raw_f3", "legacy_l"],
                "modified_branch_jobs": [
                    "formal_raw_rgb_anchor",
                    "raw_f3",
                    "legacy_l",
                    "scsp_rgb_sh_only_refit",
                    "scsp_refit_f3",
                ],
                "estimated_range": [12, 20],
                "note": (
                    "R is the number of the four missing representative scenes whose read-only train-side "
                    "SCSP manifest reports modified_gaussian_count>0; R is unresolved in Phase 0 and each "
                    "indicator is fixed before scene training without reading guard/test data or retuning the rule"
                ),
            },
            "phase_2": {
                "formula": "10+M5",
                "expanded_formula": "sum_s(2+m_s)=10+M5",
                "candidate_scenes": list(EXPANSION_SCENES),
                "per_scene_indicator": (
                    "m_s=1 iff read-only train-side SCSP manifest modified_gaussian_count>0; "
                    "otherwise m_s=0"
                ),
                "aggregate_definition": "M5=sum_s m_s",
                "aggregate_range": [0, 5],
                "noop_branch_jobs": [
                    "formal_raw_rgb_anchor",
                    "raw_f3_exact_noop_alias_source",
                ],
                "modified_branch_jobs": [
                    "formal_raw_rgb_anchor",
                    "scsp_rgb_sh_only_refit",
                    "scsp_refit_f3",
                ],
                "estimated_range": [10, 15],
                "note": (
                    "M5 is the number of the five expansion scenes whose read-only train-side SCSP manifest "
                    "reports modified_gaussian_count>0; M5 is unresolved in Phase 0 and each indicator is fixed "
                    "before scene training without reading guard/test data or retuning the rule"
                ),
            },
        },
        "external_sources": external,
    }


def build_inventory(
    collection_manifest: Path,
    *,
    creation_timestamp: str | None = None,
) -> dict[str, Any]:
    collection_manifest = collection_manifest.resolve()
    collection = load_json(collection_manifest)
    expected = tuple(sorted(str(scene) for scene in collection.get("expected_scenes", [])))
    if expected != tuple(sorted(EXPECTED_SCENES)):
        raise ValueError(f"Formal collection scene set mismatch: {expected}")
    if int(collection.get("scene_count", -1)) != len(EXPECTED_SCENES):
        raise ValueError("Formal collection scene_count mismatch")
    collection_counts = _require_counts("collection", collection.get("counts"))
    collection_hash = _require_sha256("collection_hash", collection.get("collection_hash"))
    collection_split_hash = _require_sha256(
        "collection_split_hash", collection.get("collection_split_hash")
    )
    formal_rule_hash = _require_sha256("formal_rule_hash", collection.get("formal_rule_hash"))
    validation = collection.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "passed":
        raise ValueError("Formal collection validation is not passed")

    basis_by_scene: dict[str, Mapping[str, Any]] = {}
    basis = collection.get("collection_hash_basis")
    if isinstance(basis, dict):
        for row in basis.get("scenes", []):
            if isinstance(row, dict):
                basis_by_scene[str(row.get("scene", ""))] = row

    scenes: list[dict[str, Any]] = []
    summed = {key: 0 for key in collection_counts}
    for scene in EXPECTED_SCENES:
        path = _scene_manifest_path(collection_manifest, scene)
        manifest = load_json(path)
        observed_scene = str(manifest.get("scene", manifest.get("scene_name", "")))
        if observed_scene != scene:
            raise ValueError(f"Scene manifest name mismatch: expected={scene}, observed={observed_scene}")
        counts = _require_counts(scene, manifest.get("counts"))
        for key, value in counts.items():
            summed[key] += value
        input_records_hash = _require_sha256(
            f"{scene}.input_records_hash", manifest.get("input_records_hash")
        )
        split_hash = _require_sha256(f"{scene}.split_hash", manifest.get("split_hash"))
        selected_hash = _require_sha256(
            f"{scene}.selected_test_blocks_hash", manifest.get("selected_test_blocks_hash")
        )
        basis_row = basis_by_scene.get(scene)
        if basis_row:
            if int(basis_row.get("record_count", -1)) != counts["total"]:
                raise ValueError(f"{scene} collection-basis record count mismatch")
            if str(basis_row.get("input_records_hash", "")).lower() != input_records_hash:
                raise ValueError(f"{scene} collection-basis input hash mismatch")
        scenes.append(
            {
                "scene": scene,
                "role": "representative" if scene in REPRESENTATIVE_SCENES else "expansion",
                "counts": counts,
                "input_records_hash": input_records_hash,
                "selected_test_blocks_hash": selected_hash,
                "split_hash": split_hash,
                "manifest": {
                    "relative_path": path.relative_to(collection_manifest.parent).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": sha256_file(path),
                },
            }
        )
    if summed != collection_counts:
        raise ValueError(f"Per-scene counts do not match collection counts: {summed} != {collection_counts}")

    timestamp = creation_timestamp or datetime.now(timezone.utc).isoformat()
    core: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "creation_timestamp": str(timestamp),
        "collection": {
            "collection_hash": collection_hash,
            "collection_split_hash": collection_split_hash,
            "formal_rule_hash": formal_rule_hash,
            "counts": collection_counts,
            "scene_count": len(EXPECTED_SCENES),
            "manifest": {
                "relative_path": collection_manifest.name,
                "size_bytes": int(collection_manifest.stat().st_size),
                "sha256": sha256_file(collection_manifest),
            },
            "validation_status": "passed",
        },
        "scene_inventory": scenes,
        "phase0_availability_snapshot": _phase0_availability_snapshot(),
        "logical_asset_requirements": _logical_asset_requirements(),
        "method_scene_matrix": {
            "internal": {
                "Raw-F3": list(REPRESENTATIVE_SCENES),
                "SCSP-Refit+F3": list(REPRESENTATIVE_SCENES + EXPANSION_SCENES),
                "Adaptive Opacity + Scale-Clamp": list(REPRESENTATIVE_SCENES),
            },
            "external": {method: list(REPRESENTATIVE_SCENES) for method in EXTERNAL_METHODS},
        },
        "path_policy": {
            "machine_absolute_paths_recorded": False,
            "manifest_paths_relative_to_collection_root": True,
            "runtime_inventory_must_bind_absolute_assets_by_sha256": True,
        },
    }
    core["inventory_payload_sha256"] = canonical_sha256(core)
    return core


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite inventory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection_manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--creation_timestamp", default="")
    args = parser.parse_args()
    payload = build_inventory(
        Path(args.collection_manifest),
        creation_timestamp=str(args.creation_timestamp).strip() or None,
    )
    save_json(Path(args.output), payload)
    print(
        f"inventory={Path(args.output).resolve()} "
        f"collection_hash={payload['collection']['collection_hash']} "
        f"inventory_sha256={payload['inventory_payload_sha256']}"
    )


if __name__ == "__main__":
    main()
