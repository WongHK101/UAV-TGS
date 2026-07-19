#!/usr/bin/env python3
"""Build the exact Hold-8 Phase-1 internal training queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.thermal_radiometry.build_hold8_split import (
    EXPECTED_SCENE_COUNTS,
    HOLDOUT_PERIOD,
    PROTOCOL_ID,
    SCHEMA_NAME as SPLIT_SCHEMA_NAME,
    SCHEMA_VERSION as SPLIT_SCHEMA_VERSION,
)


SCHEMA = "uav-tgs-aaai27-hold8-phase1-queue-v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
METHODS = ("raw_f3", "scsp_refit_f3", "adaptive_opacity_scale_clamp")
METHOD_RECIPES = {
    "raw_f3": {
        "dependency": "hold8_rgb_anchor_30k",
        "training": "strict F3, 30k thermal updates, appearance only",
        "alias_allowed": False,
    },
    "scsp_refit_f3": {
        "dependency": "hold8_rgb_anchor_30k",
        "training": "SCSP projection + RGB SH-only refit 5k + strict F3 30k",
        "alias_allowed": True,
        "alias_condition": "new Hold-8 SCSP modified_count == 0",
    },
    "adaptive_opacity_scale_clamp": {
        "dependency": "hold8_rgb_anchor_30k",
        "training": "locked Legacy-L adaptive opacity + scale-clamp Stage 2, 30k",
        "alias_allowed": False,
    },
}
DEFAULT_ASSIGNMENT = {
    "900": ("Building", "InternalRoad", "Urban20K"),
    "901": ("PVpanel", "TransmissionTower", "Orchard"),
}
HISTORICAL_MINUTES = {
    "Building": 12.88,
    "InternalRoad": 11.81,
    "PVpanel": 15.86,
    "TransmissionTower": 14.14,
    "Urban20K": 15.93,
    "Orchard": 14.04,
}


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("collection must be a JSON object")
    return value


def _require_sha(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(token):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return token


def _require_commit(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not _COMMIT_RE.fullmatch(token):
        raise ValueError("code_commit must be a full 40-character Git commit")
    return token


def _expected_split_counts(total: int) -> dict[str, int]:
    test = (total + HOLDOUT_PERIOD - 1) // HOLDOUT_PERIOD
    return {"total": total, "train": total - test, "test": test}


def _validate_formal_collection(collection: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    if collection.get("schema_name") != f"{SPLIT_SCHEMA_NAME}_collection":
        raise ValueError("collection is not the formal Hold-8 collection schema")
    if collection.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("collection schema version is not the formal Hold-8 version")
    if collection.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("collection protocol_id is not AAAI27 Hold-8 v2")

    validation = collection.get("validation")
    required_validation = (
        "scene_set_exact",
        "scene_counts_exact",
        "aggregate_counts_exact",
        "all_scene_validations_passed",
        "labels_exactly_train_test",
    )
    if not isinstance(validation, Mapping) or validation.get("status") != "passed":
        raise ValueError("collection formal validation status is not passed")
    if any(validation.get(field) is not True for field in required_validation):
        raise ValueError("collection formal validation flags are incomplete")

    collection_hash = _require_sha(collection.get("collection_hash"), "collection_hash")
    collection_hash_basis = collection.get("collection_hash_basis")
    if not isinstance(collection_hash_basis, Mapping):
        raise ValueError("collection lacks collection_hash_basis")
    if json_hash(collection_hash_basis) != collection_hash:
        raise ValueError("collection_hash does not authenticate collection_hash_basis")

    collection_split_hash = _require_sha(
        collection.get("collection_split_hash"), "collection_split_hash"
    )
    generator = collection.get("generator")
    if not isinstance(generator, Mapping):
        raise ValueError("collection lacks split generator identity")
    _require_commit(generator.get("code_commit"))
    _require_sha(generator.get("source_sha256"), "split generator source_sha256")
    registered_counts = {scene: int(count) for scene, count in EXPECTED_SCENE_COUNTS.items()}
    if collection.get("expected_scene_counts") != dict(sorted(registered_counts.items())):
        raise ValueError("collection expected_scene_counts do not match the registered collection")
    if collection.get("scene_count") != len(registered_counts):
        raise ValueError("collection scene_count does not match the registered collection")

    expected_aggregate = {key: 0 for key in ("total", "train", "test")}
    expected_by_scene = {
        scene: _expected_split_counts(total) for scene, total in registered_counts.items()
    }
    for counts in expected_by_scene.values():
        for key in expected_aggregate:
            expected_aggregate[key] += counts[key]
    if collection.get("counts") != expected_aggregate:
        raise ValueError("collection aggregate counts do not match the registered Hold-8 counts")

    scenes_raw = collection.get("scenes")
    if not isinstance(scenes_raw, list) or len(scenes_raw) != len(registered_counts):
        raise ValueError("collection lacks the exact eleven formal scene rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in scenes_raw:
        if not isinstance(item, Mapping):
            raise ValueError("collection scene rows must be JSON objects")
        scene = str(item.get("scene", ""))
        if not scene or scene in indexed:
            raise ValueError(f"collection contains a missing/duplicate scene: {scene!r}")
        indexed[scene] = item
    if set(indexed) != set(registered_counts):
        raise ValueError("collection scene set does not match the registered eleven scenes")
    for scene, expected in expected_by_scene.items():
        if indexed[scene].get("counts") != expected:
            raise ValueError(f"formal Hold-8 counts mismatch for {scene}")
        _require_sha(indexed[scene].get("split_hash"), f"{scene} split_hash")

    collection_split_basis = {
        "protocol_id": PROTOCOL_ID,
        "collection_hash": collection_hash,
        "generator": dict(generator),
        "scenes": [
            {"scene": scene, "split_hash": indexed[scene]["split_hash"]}
            for scene in sorted(indexed)
        ],
    }
    if json_hash(collection_split_basis) != collection_split_hash:
        raise ValueError("collection_split_hash does not authenticate formal scene split hashes")

    return {scene: expected_by_scene[scene] for scene in HISTORICAL_MINUTES}


def build_queue(
    collection: Mapping[str, Any],
    *,
    code_commit: str,
    radiometry_plan_sha256: str | None = None,
    host_preflight_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    code_commit = _require_commit(code_commit)
    counts = _validate_formal_collection(collection)
    prerequisite_hashes: dict[str, Any] = {}
    if radiometry_plan_sha256 is not None:
        prerequisite_hashes["radiometry_plan_sha256"] = _require_sha(
            radiometry_plan_sha256, "radiometry_plan_sha256"
        )
    if host_preflight_sha256 is not None:
        if set(host_preflight_sha256) != set(DEFAULT_ASSIGNMENT):
            raise ValueError("host_preflight_sha256 must contain exactly hosts 900 and 901")
        prerequisite_hashes["host_preflight_sha256"] = {
            host: _require_sha(host_preflight_sha256[host], f"host {host} preflight SHA-256")
            for host in sorted(DEFAULT_ASSIGNMENT)
        }
    jobs: list[dict[str, Any]] = []
    endpoint_jobs: list[dict[str, Any]] = []
    for host, scenes in DEFAULT_ASSIGNMENT.items():
        for scene in scenes:
            for method in METHODS:
                endpoint_jobs.append(
                    {
                        "job_id": f"{scene}:{method}",
                        "host": host,
                        "scene": scene,
                        "method": method,
                        **METHOD_RECIPES[method],
                        "status": "PLANNED_NOT_STARTED",
                    }
                )
            jobs.append(
                {
                    "host": host,
                    "scene": scene,
                    "counts": counts[scene],
                    "stages": [
                        "verify_hold8_assets",
                        "build_train_only_range_canonical_hotspot",
                        "build_train_only_openmvs_reference",
                        "train_rgb_anchor_30k",
                        "run_raw_f3_30k",
                        "run_scsp_projection_refit5k_f3_30k_or_new_noop_alias",
                        "run_adaptive_opacity_scale_clamp_30k",
                        "render_once_and_evaluate",
                        "sync_to_900_and_verify_sha",
                    ],
                    "formal_endpoint_rows": list(METHODS),
                    "historical_total_gpu_minutes_estimate": HISTORICAL_MINUTES[scene],
                    "status": "PLANNED_NOT_STARTED",
                }
            )
    host_estimates = {
        host: sum(HISTORICAL_MINUTES[scene] for scene in scenes)
        for host, scenes in DEFAULT_ASSIGNMENT.items()
    }
    payload = {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "code_commit": code_commit,
        "collection_hash": collection.get("collection_hash"),
        "collection_split_hash": collection.get("collection_split_hash"),
        "prerequisite_hashes": prerequisite_hashes,
        "representative_scene_count": 6,
        "rgb_anchor_training_count": 6,
        "formal_internal_endpoint_rows": 18,
        "endpoint_jobs": endpoint_jobs,
        "scsp_alias_policy": "recompute from each new Hold-8 anchor; no guard4 alias reuse",
        "assignment_is_scheduler_default_not_protocol": True,
        "jobs": jobs,
        "host_historical_gpu_minutes": host_estimates,
        "estimated_phase1": {
            "gpu_hours_total": "1.5-2.5",
            "two_host_wall_hours_including_reference_and_evaluation": "6-10",
            "disk_peak_gib": {"900": "180-260", "901": "120-200"},
            "basis": "guard4 receipts scaled conservatively; not a hard limit",
        },
        "formal_training_started": False,
    }
    payload["queue_sha256"] = json_hash(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--radiometry-plan-sha256", required=True)
    parser.add_argument("--host-900-preflight-sha256", required=True)
    parser.add_argument("--host-901-preflight-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    host_preflight = {
        "900": args.host_900_preflight_sha256,
        "901": args.host_901_preflight_sha256,
    }
    payload = build_queue(
        load(args.collection),
        code_commit=args.code_commit,
        radiometry_plan_sha256=args.radiometry_plan_sha256,
        host_preflight_sha256=host_preflight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"jobs": len(payload["jobs"]), "queue_sha256": payload["queue_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
