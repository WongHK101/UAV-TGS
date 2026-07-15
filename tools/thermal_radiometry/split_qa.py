#!/usr/bin/env python3
"""Compare deterministic guard-width candidates for RGB-T data splitting.

The tool deliberately does not select a preferred guard width.  It reuses
``build_split.build_split_manifest`` with a fixed 16-frame block size and
reports guard=2/4 candidates (configurable only as an explicit comparison
set).  Input manifests may be audit JSONL files or protocol JSON manifests;
records are grouped by their scene before each candidate split is built.

For every test frame the report includes the nearest usable training frame by
timestamp, GPS and gimbal orientation.  Missing metadata is represented by
``null`` plus coverage counts rather than silently substituted values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - direct-script import path
    from . import build_split
except ImportError:  # pragma: no cover
    import build_split  # type: ignore


SCHEMA_NAME = "uav_tgs_split_guard_qa"
SCHEMA_VERSION = 2
BLOCK_SIZE = 16
TEST_PERIOD_BLOCKS = 8
DEFAULT_GUARDS = (2, 4)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lookup(record: Mapping[str, Any], dotted_names: Iterable[str]) -> Any:
    for dotted_name in dotted_names:
        value: Any = record
        found = True
        for part in dotted_name.split("."):
            if not isinstance(value, Mapping) or part not in value:
                found = False
                break
            value = value[part]
        if found and value not in (None, ""):
            return value
    return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp_epoch(record: Mapping[str, Any]) -> float | None:
    value = _lookup(
        record,
        (
            "timestamp",
            "timestamp_utc",
            "utc_at_exposure",
            "capture_time",
            "datetime_original",
            "rgb_capture_time",
            "metadata.timestamp",
            "metadata.timestamp_utc",
            "metadata.utc_at_exposure",
            "metadata.capture_time",
            "metadata.datetime_original",
            "audit.capture_time",
        ),
    )
    parsed = build_split._parse_timestamp(value)
    return None if parsed is None else parsed[0]


def _qa_metadata(record: Mapping[str, Any], source_index: int) -> dict[str, Any]:
    """Extract only metadata needed by split QA without inventing fallbacks."""
    latitude = _finite_float(
        _lookup(
            record,
            (
                "gps_latitude",
                "latitude",
                "gps.latitude",
                "metadata.gps_latitude",
                "metadata.latitude",
                "metadata.gps.latitude",
                "audit.gps_latitude",
            ),
        )
    )
    longitude = _finite_float(
        _lookup(
            record,
            (
                "gps_longitude",
                "longitude",
                "gps.longitude",
                "metadata.gps_longitude",
                "metadata.longitude",
                "metadata.gps.longitude",
                "audit.gps_longitude",
            ),
        )
    )
    if latitude is not None and not (-90.0 <= latitude <= 90.0):
        latitude = None
    if longitude is not None and not (-180.0 <= longitude <= 180.0):
        longitude = None

    pitch = _finite_float(
        _lookup(
            record,
            (
                "gimbal_pitch_deg",
                "gimbal_pitch",
                "metadata.gimbal_pitch_deg",
                "metadata.gimbal_pitch",
                "metadata.gimbal.pitch_deg",
                "gimbal.pitch_deg",
                "gimbal.pitch",
                "audit.gimbal_pitch_deg",
            ),
        )
    )
    yaw = _finite_float(
        _lookup(
            record,
            (
                "gimbal_yaw_deg",
                "gimbal_yaw",
                "metadata.gimbal_yaw_deg",
                "metadata.gimbal_yaw",
                "metadata.gimbal.yaw_deg",
                "gimbal.yaw_deg",
                "gimbal.yaw",
                "audit.gimbal_yaw_deg",
            ),
        )
    )
    if pitch is not None and not (-180.0 <= pitch <= 180.0):
        pitch = None

    return {
        "pair_id": build_split._pair_id_for(record, source_index),
        "timestamp_epoch": _timestamp_epoch(record),
        "gps_latitude": latitude,
        "gps_longitude": longitude,
        "gimbal_pitch_deg": pitch,
        "gimbal_yaw_deg": yaw,
    }


def _haversine_m(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    lat1 = first["gps_latitude"]
    lon1 = first["gps_longitude"]
    lat2 = second["gps_latitude"]
    lon2 = second["gps_longitude"]
    if None in (lat1, lon1, lat2, lon2):
        return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    term = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * 6_371_008.8 * math.asin(min(1.0, math.sqrt(term)))


def _pitch_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    if first["gimbal_pitch_deg"] is None or second["gimbal_pitch_deg"] is None:
        return None
    return abs(first["gimbal_pitch_deg"] - second["gimbal_pitch_deg"])


def _yaw_difference(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    if first["gimbal_yaw_deg"] is None or second["gimbal_yaw_deg"] is None:
        return None
    raw = abs(first["gimbal_yaw_deg"] - second["gimbal_yaw_deg"]) % 360.0
    return min(raw, 360.0 - raw)


def _nearest(
    test_record: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
    metric,
) -> tuple[Mapping[str, Any] | None, float | None]:
    best_record: Mapping[str, Any] | None = None
    best_value: float | None = None
    for candidate in train_records:
        value = metric(test_record, candidate)
        if value is None:
            continue
        # Pair ID is the stable deterministic tie-breaker.
        if best_value is None or (value, str(candidate["pair_id"])) < (
            best_value,
            str(best_record["pair_id"]),
        ):
            best_record = candidate
            best_value = value
    return best_record, best_value


def _temporal_gap(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    if first["timestamp_epoch"] is None or second["timestamp_epoch"] is None:
        return None
    return abs(first["timestamp_epoch"] - second["timestamp_epoch"])


def _gimbal_distance(first: Mapping[str, Any], second: Mapping[str, Any]) -> float | None:
    pitch = _pitch_difference(first, second)
    yaw = _yaw_difference(first, second)
    if pitch is None or yaw is None:
        return None
    return math.hypot(pitch, yaw)


def _test_nearest_train(
    test_record: Mapping[str, Any],
    train_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    temporal_record, temporal_gap = _nearest(test_record, train_records, _temporal_gap)
    gps_record, gps_distance = _nearest(test_record, train_records, _haversine_m)
    gimbal_record, gimbal_distance = _nearest(test_record, train_records, _gimbal_distance)

    temporal_details: dict[str, Any] | None = None
    if temporal_record is not None:
        temporal_details = {
            "pair_id": temporal_record["pair_id"],
            "gap_s": temporal_gap,
            "gps_distance_m": _haversine_m(test_record, temporal_record),
            "gimbal_pitch_difference_deg": _pitch_difference(test_record, temporal_record),
            "gimbal_yaw_difference_deg": _yaw_difference(test_record, temporal_record),
        }

    gps_details: dict[str, Any] | None = None
    if gps_record is not None:
        gps_details = {
            "pair_id": gps_record["pair_id"],
            "distance_m": gps_distance,
            "temporal_gap_s": _temporal_gap(test_record, gps_record),
            "gimbal_pitch_difference_deg": _pitch_difference(test_record, gps_record),
            "gimbal_yaw_difference_deg": _yaw_difference(test_record, gps_record),
        }

    gimbal_details: dict[str, Any] | None = None
    if gimbal_record is not None:
        gimbal_details = {
            "pair_id": gimbal_record["pair_id"],
            "combined_difference_deg": gimbal_distance,
            "pitch_difference_deg": _pitch_difference(test_record, gimbal_record),
            "yaw_difference_deg": _yaw_difference(test_record, gimbal_record),
            "temporal_gap_s": _temporal_gap(test_record, gimbal_record),
            "gps_distance_m": _haversine_m(test_record, gimbal_record),
        }

    return {
        "pair_id": test_record["pair_id"],
        "stratum": test_record["stratum"],
        "strip_id": test_record["strip_id"],
        "nearest_train_temporal_gap_s": temporal_gap,
        "nearest_train_gps_distance_m": gps_distance,
        "nearest_train_gimbal_pitch_difference_deg": (
            None if gimbal_details is None else gimbal_details["pitch_difference_deg"]
        ),
        "nearest_train_gimbal_yaw_difference_deg": (
            None if gimbal_details is None else gimbal_details["yaw_difference_deg"]
        ),
        "nearest_train_by_time": temporal_details,
        "nearest_train_by_gps": gps_details,
        "nearest_train_by_gimbal": gimbal_details,
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Iterable[Any], *, expected_count: int) -> dict[str, Any]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return {
            "supported_count": 0,
            "missing_count": expected_count,
            "support_rate": 0.0 if expected_count else None,
            "min": None,
            "mean": None,
            "std": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "supported_count": len(finite),
        "missing_count": expected_count - len(finite),
        "support_rate": len(finite) / expected_count if expected_count else None,
        "min": min(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.pstdev(finite),
        "median": statistics.median(finite),
        "p95": _percentile(finite, 0.95),
        "max": max(finite),
    }


def _group_counts(records: Sequence[Mapping[str, Any]], key_names: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        group_key = tuple(str(record[key]) for key in key_names)
        group = grouped.setdefault(
            group_key,
            {**dict(zip(key_names, group_key)), "total": 0, "train": 0, "test": 0, "guard": 0},
        )
        group["total"] += 1
        group[str(record["split"])] += 1
    return [grouped[key] for key in sorted(grouped)]


def _ratio_summary(counts: Mapping[str, int]) -> dict[str, Any]:
    retained = counts["train"] + counts["test"]
    return {
        "retained_count": retained,
        "retained_ratio": retained / counts["total"] if counts["total"] else None,
        "train_to_test_ratio": counts["train"] / counts["test"] if counts["test"] else None,
        "train_fraction_of_retained": counts["train"] / retained if retained else None,
        "test_fraction_of_retained": counts["test"] / retained if retained else None,
    }


def _nearest_summary(nearest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise independent minima and the single nearest-by-time anchor.

    The GPS minimum and gimbal minimum may refer to different training frames.
    Values in ``nearest_by_time_observation`` all refer to the same training
    frame selected solely by minimum temporal gap.
    """
    expected_count = len(nearest)
    return {
        "semantics": {
            "independent_minima": (
                "temporal_gap_s, gps_distance_m, and gimbal differences are "
                "minimised independently and may come from different training frames"
            ),
            "nearest_by_time_observation": (
                "all fields use the one training frame selected by minimum temporal gap"
            ),
        },
        "per_metric_independent_minima": {
            "temporal_gap_s": _distribution(
                (item["nearest_train_temporal_gap_s"] for item in nearest),
                expected_count=expected_count,
            ),
            "gps_distance_m": _distribution(
                (item["nearest_train_gps_distance_m"] for item in nearest),
                expected_count=expected_count,
            ),
            "gimbal_pitch_difference_deg": _distribution(
                (item["nearest_train_gimbal_pitch_difference_deg"] for item in nearest),
                expected_count=expected_count,
            ),
            "gimbal_yaw_difference_deg": _distribution(
                (item["nearest_train_gimbal_yaw_difference_deg"] for item in nearest),
                expected_count=expected_count,
            ),
        },
        "nearest_by_time_observation": {
            "temporal_gap_s": _distribution(
                (
                    None
                    if item["nearest_train_by_time"] is None
                    else item["nearest_train_by_time"]["gap_s"]
                    for item in nearest
                ),
                expected_count=expected_count,
            ),
            "gps_distance_m": _distribution(
                (
                    None
                    if item["nearest_train_by_time"] is None
                    else item["nearest_train_by_time"]["gps_distance_m"]
                    for item in nearest
                ),
                expected_count=expected_count,
            ),
            "gimbal_pitch_difference_deg": _distribution(
                (
                    None
                    if item["nearest_train_by_time"] is None
                    else item["nearest_train_by_time"]["gimbal_pitch_difference_deg"]
                    for item in nearest
                ),
                expected_count=expected_count,
            ),
            "gimbal_yaw_difference_deg": _distribution(
                (
                    None
                    if item["nearest_train_by_time"] is None
                    else item["nearest_train_by_time"]["gimbal_yaw_difference_deg"]
                    for item in nearest
                ),
                expected_count=expected_count,
            ),
        },
    }


def _scene_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    seed: str,
    guard_frames: int,
    strip_max_gap_s: float,
) -> dict[str, Any]:
    split = build_split.build_split_manifest(
        records,
        scene=scene,
        seed=seed,
        block_size=BLOCK_SIZE,
        test_period_blocks=TEST_PERIOD_BLOCKS,
        guard_frames=guard_frames,
        strip_max_gap_s=strip_max_gap_s,
        fail_on_invalid=False,
    )
    metadata_by_pair: dict[str, dict[str, Any]] = {}
    for index, source_record in enumerate(records):
        extracted = _qa_metadata(source_record, index)
        pair_id = str(extracted["pair_id"])
        if pair_id in metadata_by_pair:
            raise ValueError(f"duplicate pair_id in scene {scene}: {pair_id}")
        metadata_by_pair[pair_id] = extracted

    assigned: list[dict[str, Any]] = []
    for split_record in split["records"]:
        pair_id = str(split_record["pair_id"])
        source_metadata = metadata_by_pair[pair_id]
        assigned.append(
            {
                **source_metadata,
                "stratum": split_record["stratum"],
                "strip_id": split_record["strip_id"],
                "split": split_record["split"],
            }
        )

    train_records = [record for record in assigned if record["split"] == "train"]
    test_records = [record for record in assigned if record["split"] == "test"]
    nearest = [
        _test_nearest_train(record, train_records)
        for record in sorted(test_records, key=lambda item: str(item["pair_id"]))
    ]
    counts = {key: int(split["counts"][key]) for key in ("total", "train", "test", "guard")}
    strip_counts = _group_counts(assigned, ("stratum", "strip_id"))
    stratum_counts = _group_counts(assigned, ("stratum",))
    without_test = [
        {"stratum": item["stratum"], "strip_id": item["strip_id"], "frame_count": item["total"]}
        for item in strip_counts
        if item["test"] == 0
    ]
    strips_without_train = [
        {
            "stratum": item["stratum"],
            "strip_id": item["strip_id"],
            "frame_count": item["total"],
            "test_count": item["test"],
            "guard_count": item["guard"],
        }
        for item in strip_counts
        if item["train"] == 0
    ]
    strata_without_train = [
        {
            "stratum": item["stratum"],
            "frame_count": item["total"],
            "test_count": item["test"],
            "guard_count": item["guard"],
        }
        for item in stratum_counts
        if item["train"] == 0
    ]
    strata_without_test = [
        {
            "stratum": item["stratum"],
            "frame_count": item["total"],
            "train_count": item["train"],
            "guard_count": item["guard"],
        }
        for item in stratum_counts
        if item["test"] == 0
    ]
    warnings: list[dict[str, Any]] = []
    for item in strips_without_train:
        warnings.append(
            {
                "code": "strip_without_train",
                "severity": "error",
                "message": "Invalid split leaves this strip with no training frames.",
                **item,
            }
        )
    for item in strata_without_train:
        warnings.append(
            {
                "code": "stratum_without_train",
                "severity": "error",
                "message": "Invalid split leaves this stratum with no training frames.",
                **item,
            }
        )
    existing_warning_codes = {str(item["code"]) for item in warnings}
    for error in split["validation"]["errors"]:
        if error in existing_warning_codes:
            continue
        warnings.append(
            {
                "code": str(error),
                "severity": "error",
                "message": "Scene-budget split failed a formal validation invariant.",
            }
        )
    metadata_supported = {
        "timestamp": sum(record["timestamp_epoch"] is not None for record in assigned),
        "gps": sum(
            record["gps_latitude"] is not None and record["gps_longitude"] is not None
            for record in assigned
        ),
        "gimbal": sum(
            record["gimbal_pitch_deg"] is not None and record["gimbal_yaw_deg"] is not None
            for record in assigned
        ),
    }
    fallback = split["rule"]["ordering_mode"] == "filename_order_fallback"
    return {
        "scene": scene,
        "split_hash": split["split_hash"],
        "input_records_hash": split["input_records_hash"],
        "allocation_hash": split["allocation_hash"],
        "selected_test_blocks_hash": split["selected_test_blocks_hash"],
        "selected_candidate_hashes": split["selected_candidate_hashes"],
        "test_block_budget": split["test_block_budget"],
        "stratum_allocations": split["stratum_allocations"],
        "validation": split["validation"],
        "ordering_mode": split["rule"]["ordering_mode"],
        "metadata_reliability": split["metadata_reliability"],
        "metadata_fallback_frame_count": counts["total"] if fallback else 0,
        "metadata_fallback_rate": 1.0 if fallback else 0.0,
        "metadata_support": {
            name: {
                "supported_count": supported,
                "missing_count": counts["total"] - supported,
                "support_rate": supported / counts["total"] if counts["total"] else None,
            }
            for name, supported in metadata_supported.items()
        },
        "counts": counts,
        "test_fraction_of_scene": counts["test"] / counts["total"] if counts["total"] else None,
        "retained_train_test": _ratio_summary(counts),
        "stratum_counts": stratum_counts,
        "strip_counts": strip_counts,
        "strips_without_test": without_test,
        "strips_without_train": strips_without_train,
        "strata_without_train": strata_without_train,
        "strata_without_test": strata_without_test,
        "warning_count": len(warnings),
        "warnings": warnings,
        "test_frame_nearest_train": nearest,
        "nearest_train_summary": _nearest_summary(nearest),
    }


def build_split_qa_report(
    manifests: Sequence[Path],
    *,
    seed: str = "uav-tgs-aaai27-v1",
    guards: Sequence[int] = DEFAULT_GUARDS,
    strip_max_gap_s: float = 10.0,
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("at least one manifest is required")
    guard_values = tuple(int(value) for value in guards)
    if not guard_values or any(value < 0 for value in guard_values):
        raise ValueError("guards must be a non-empty collection of non-negative integers")
    if len(set(guard_values)) != len(guard_values):
        raise ValueError("guards must not contain duplicates")

    scene_records: dict[str, list[Mapping[str, Any]]] = {}
    input_descriptors: list[dict[str, Any]] = []
    for manifest in sorted((Path(path).resolve() for path in manifests), key=lambda item: str(item)):
        records, metadata = build_split.load_manifest(manifest)
        top_scene = metadata.get("scene") if isinstance(metadata, Mapping) else None
        scenes_in_file: set[str] = set()
        for record in records:
            scene_value = record.get("scene") or top_scene
            if scene_value in (None, ""):
                raise ValueError(f"scene is absent in manifest record: {manifest}")
            scene = str(scene_value)
            scene_records.setdefault(scene, []).append(record)
            scenes_in_file.add(scene)
        input_descriptors.append(
            {
                "path": str(manifest),
                "sha256": _file_sha256(manifest),
                "record_count": len(records),
                "scenes": sorted(scenes_in_file),
            }
        )

    candidates: list[dict[str, Any]] = []
    for guard in guard_values:
        scenes = [
            _scene_candidate(
                scene_records[scene],
                scene=scene,
                seed=seed,
                guard_frames=guard,
                strip_max_gap_s=strip_max_gap_s,
            )
            for scene in sorted(scene_records)
        ]
        total_counts = {
            key: sum(scene["counts"][key] for scene in scenes)
            for key in ("total", "train", "test", "guard")
        }
        fallback_frames = sum(scene["metadata_fallback_frame_count"] for scene in scenes)
        fallback_scenes = sum(scene["metadata_fallback_rate"] == 1.0 for scene in scenes)
        all_nearest = [
            item
            for scene in scenes
            for item in scene["test_frame_nearest_train"]
        ]
        candidate_warnings = [
            {"scene": scene["scene"], **warning}
            for scene in scenes
            for warning in scene["warnings"]
        ]
        candidates.append(
            {
                "guard_frames_each_side": guard,
                "counts": total_counts,
                "retained_train_test": _ratio_summary(total_counts),
                "metadata_fallback": {
                    "frame_count": fallback_frames,
                    "frame_rate": fallback_frames / total_counts["total"] if total_counts["total"] else None,
                    "scene_count": fallback_scenes,
                    "scene_rate": fallback_scenes / len(scenes) if scenes else None,
                },
                "strips_without_test_count": sum(
                    len(scene["strips_without_test"]) for scene in scenes
                ),
                "strips_without_train_count": sum(
                    len(scene["strips_without_train"]) for scene in scenes
                ),
                "strata_without_train_count": sum(
                    len(scene["strata_without_train"]) for scene in scenes
                ),
                "strata_without_test_count": sum(
                    len(scene["strata_without_test"]) for scene in scenes
                ),
                "valid_scene_count": sum(
                    scene["validation"]["status"] == "passed" for scene in scenes
                ),
                "invalid_scene_count": sum(
                    scene["validation"]["status"] != "passed" for scene in scenes
                ),
                "warning_count": len(candidate_warnings),
                "warnings": candidate_warnings,
                "nearest_train_summary": _nearest_summary(all_nearest),
                "scenes": scenes,
            }
        )

    cross_guard_test_set_comparison: list[dict[str, Any]] = []
    for first_index, first_candidate in enumerate(candidates):
        first_by_scene = {scene["scene"]: scene for scene in first_candidate["scenes"]}
        for second_candidate in candidates[first_index + 1 :]:
            second_by_scene = {
                scene["scene"]: scene for scene in second_candidate["scenes"]
            }
            for scene in sorted(first_by_scene):
                first_scene = first_by_scene[scene]
                second_scene = second_by_scene[scene]
                first_blocks = set(first_scene["selected_candidate_hashes"])
                second_blocks = set(second_scene["selected_candidate_hashes"])
                union = first_blocks | second_blocks
                intersection = first_blocks & second_blocks
                cross_guard_test_set_comparison.append(
                    {
                        "scene": scene,
                        "first_guard_frames_each_side": first_candidate[
                            "guard_frames_each_side"
                        ],
                        "second_guard_frames_each_side": second_candidate[
                            "guard_frames_each_side"
                        ],
                        "first_selected_test_blocks_hash": first_scene[
                            "selected_test_blocks_hash"
                        ],
                        "second_selected_test_blocks_hash": second_scene[
                            "selected_test_blocks_hash"
                        ],
                        "same_selected_test_blocks": first_blocks == second_blocks,
                        "intersection_block_count": len(intersection),
                        "union_block_count": len(union),
                        "jaccard": len(intersection) / len(union) if union else 1.0,
                    }
                )

    result: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "decision_status": "comparison_only_no_guard_selected",
        "nearest_neighbor_semantics": {
            "per_metric_minima": (
                "nearest time, GPS, and gimbal values are independent minima and may "
                "refer to different training observations"
            ),
            "same_observation_diagnostic": (
                "nearest_train_summary.nearest_by_time_observation reports GPS and "
                "gimbal differences for the single observation nearest in time"
            ),
        },
        "seed": seed,
        "fixed_rule": {
            "block_size_frames": BLOCK_SIZE,
            "test_period_blocks": TEST_PERIOD_BLOCKS,
            "scene_test_block_budget": (
                "round_half_up(scene_frame_count / "
                "(block_size_frames * test_period_blocks))"
            ),
            "minimum_train_frames_after_selected_test": (
                build_split.MIN_TRAIN_FRAMES_AFTER_TEST
            ),
            "stratum_allocation": "largest_remainder_over_feasible_capacity",
            "test_fraction_of_scene_range": [
                build_split.MIN_TEST_FRACTION,
                build_split.MAX_TEST_FRACTION,
            ],
            "strip_max_gap_s": strip_max_gap_s,
            "guard_candidates_frames_each_side": list(guard_values),
        },
        "input_manifests": input_descriptors,
        "scene_count": len(scene_records),
        "candidates": candidates,
        "cross_guard_test_set_comparison": cross_guard_test_set_comparison,
    }
    result["qa_hash"] = _hash_json(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Audit/protocol JSON or JSONL; repeat for multiple scenes",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default="uav-tgs-aaai27-v1")
    parser.add_argument("--guards", type=int, nargs="+", default=list(DEFAULT_GUARDS))
    parser.add_argument("--strip-max-gap-s", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"QA output exists; pass --overwrite: {args.output}")
    report = build_split_qa_report(
        args.manifest,
        seed=str(args.seed),
        guards=args.guards,
        strip_max_gap_s=float(args.strip_max_gap_s),
    )
    _write_json(args.output, report)
    print(
        f"scenes={report['scene_count']} guards="
        f"{','.join(str(item['guard_frames_each_side']) for item in report['candidates'])} "
        f"decision={report['decision_status']} qa_hash={report['qa_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
