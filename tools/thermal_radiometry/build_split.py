#!/usr/bin/env python3
"""Build a deterministic, leakage-guarded RGB-T scene split manifest.

The preferred ordering derives contiguous strips from reliable timestamp and
gimbal metadata.  If any record lacks those fields, the complete scene falls
back to a deterministic natural filename order.  A scene-level test-block
budget is allocated across strata by deterministic largest remainder.  Test
data are selected only as complete 16-frame blocks; neighbouring frames are
marked ``guard`` and are never included in training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_NAME = "uav_tgs_deterministic_block_split"
SCHEMA_VERSION = 2
DEFAULT_BLOCK_SIZE = 16
DEFAULT_TEST_PERIOD_BLOCKS = 8
DEFAULT_GUARD_FRAMES = 4
MIN_TRAIN_FRAMES_AFTER_TEST = 16
MIN_TEST_FRACTION = 0.08
MAX_TEST_FRACTION = 0.16


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
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


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value.replace("\\", "/"))
    )


def _parse_timestamp(value: Any) -> tuple[float, str] | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        epoch = float(value)
        rendered = datetime.fromtimestamp(epoch, timezone.utc).isoformat()
        return epoch, rendered
    if not isinstance(value, str) or not value.strip():
        return None

    raw = value.strip()
    candidates = [raw]
    if raw.endswith("Z"):
        candidates.insert(0, raw[:-1] + "+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            parsed = parsed.astimezone(timezone.utc)
            return parsed.timestamp(), parsed.isoformat()
        except ValueError:
            pass

    for pattern in (
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y:%m:%d %H:%M:%S",
        "%Y%m%d%H%M%S",
    ):
        try:
            parsed = datetime.strptime(raw, pattern).replace(tzinfo=timezone.utc)
            return parsed.timestamp(), parsed.isoformat()
        except ValueError:
            pass
    return None


def _as_finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _filename_for(record: Mapping[str, Any], pair_id: str) -> str:
    value = _lookup(
        record,
        (
            "filename",
            "source_filename",
            "original_files.thermal",
            "original_files.rgb",
            "source_files.thermal",
            "source_files.rgb",
            "files.thermal",
            "files.rgb",
            "thermal_path",
            "rgb_path",
            "source_path",
            "path",
        ),
    )
    if value is None:
        return pair_id
    return str(value)


def _pair_id_for(record: Mapping[str, Any], source_index: int) -> str:
    value = _lookup(record, ("pair_id", "pair", "frame_id", "id", "stem"))
    if value is not None:
        return str(value)
    filename = _filename_for(record, "")
    if filename:
        stem = Path(filename).stem
        stem = re.sub(r"_[TW]$", "", stem, flags=re.IGNORECASE)
        if stem:
            return stem
    return f"record-{source_index:06d}"


def _original_files(record: Mapping[str, Any]) -> Any:
    for key in ("original_files", "source_files", "files"):
        if key in record and record[key] not in (None, ""):
            return copy.deepcopy(record[key])
    collected = {
        key: record[key]
        for key in ("rgb_path", "thermal_path", "source_path", "path")
        if key in record and record[key] not in (None, "")
    }
    return collected


def _temperature_npy(record: Mapping[str, Any]) -> str | None:
    value = _lookup(
        record,
        (
            "temperature_npy",
            "temperature_path",
            "npy_path",
            "output_path",
            "outputs.temperature_npy",
            "files.temperature_npy",
            "derived.temperature_npy",
        ),
    )
    return None if value is None else str(value)


def _extract_records(payload: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if isinstance(payload, list):
        records = payload
        metadata: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        records = None
        for key in ("records", "frames", "pairs", "items"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        if records is None:
            raise ValueError("manifest must contain records, frames, pairs, or items")
        metadata = payload
    else:
        raise ValueError("manifest must be a JSON object or array")
    if not records:
        raise ValueError("manifest has no records")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("every manifest record must be a JSON object")
    return list(records), metadata


def load_manifest(path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _extract_records(records)
    return _extract_records(json.loads(path.read_text(encoding="utf-8")))


def _gimbal_stratum(pitch: float, yaw: float) -> str:
    pitch_bin = int(round(pitch / 15.0) * 15)
    if pitch <= -75.0:
        return f"nadir:p{pitch_bin:+04d}"
    yaw_bin = int(round((yaw % 360.0) / 90.0) * 90) % 360
    return f"oblique:p{pitch_bin:+04d}:y{yaw_bin:03d}"


def _normalise_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for source_index, record in enumerate(records):
        pair_id = _pair_id_for(record, source_index)
        timestamp = _parse_timestamp(
            _lookup(
                record,
                (
                    "timestamp",
                    "timestamp_utc",
                    "utc_at_exposure",
                    "datetime_original",
                    "capture_time",
                    "rgb_capture_time",
                    "metadata.timestamp",
                    "metadata.timestamp_utc",
                    "metadata.utc_at_exposure",
                    "metadata.datetime_original",
                ),
            )
        )
        pitch = _as_finite_float(
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
                ),
            )
        )
        yaw = _as_finite_float(
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
                ),
            )
        )
        normalised.append(
            {
                "source_index": source_index,
                "pair_id": pair_id,
                "filename": _filename_for(record, pair_id),
                "original_files": _original_files(record),
                "temperature_npy": _temperature_npy(record),
                "timestamp_epoch": None if timestamp is None else timestamp[0],
                "timestamp_utc": None if timestamp is None else timestamp[1],
                "gimbal_pitch_deg": pitch,
                "gimbal_yaw_deg": yaw,
                "manifest_stratum": _lookup(record, ("stratum", "metadata.stratum")),
                "manifest_strip": _lookup(record, ("strip_id", "strip", "metadata.strip_id")),
                "source_record_hash": _hash_json(record),
            }
        )
    return normalised


def _metadata_reliability(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    missing_timestamp = [r["pair_id"] for r in records if r["timestamp_epoch"] is None]
    missing_gimbal = [
        r["pair_id"]
        for r in records
        if r["gimbal_pitch_deg"] is None or r["gimbal_yaw_deg"] is None
    ]
    out_of_range_gimbal = [
        r["pair_id"]
        for r in records
        if r["gimbal_pitch_deg"] is not None
        and r["gimbal_yaw_deg"] is not None
        and not (-180.0 <= r["gimbal_pitch_deg"] <= 180.0)
    ]
    unique_timestamps = len(
        {r["timestamp_epoch"] for r in records if r["timestamp_epoch"] is not None}
    )
    timestamp_variation_ok = len(records) == 1 or unique_timestamps >= 2
    reliable = not (
        missing_timestamp
        or missing_gimbal
        or out_of_range_gimbal
        or not timestamp_variation_ok
    )
    reasons: list[str] = []
    if missing_timestamp:
        reasons.append("missing_or_unparseable_timestamp")
    if missing_gimbal:
        reasons.append("missing_or_nonfinite_gimbal")
    if out_of_range_gimbal:
        reasons.append("gimbal_pitch_out_of_range")
    if not timestamp_variation_ok:
        reasons.append("timestamps_have_no_variation")
    return {
        "reliable": reliable,
        "record_count": len(records),
        "timestamp_count": len(records) - len(missing_timestamp),
        "unique_timestamp_count": unique_timestamps,
        "gimbal_count": len(records) - len(missing_gimbal),
        "reasons": reasons,
    }


def _build_strips(
    records: list[dict[str, Any]],
    *,
    metadata_reliable: bool,
    max_gap_s: float,
) -> tuple[str, list[dict[str, Any]]]:
    if not metadata_reliable:
        ordered = sorted(
            records,
            key=lambda r: (_natural_key(r["filename"]), _natural_key(r["pair_id"])),
        )
        return "filename_order_fallback", [
            {"strip_id": "filename-0000", "stratum": "filename_order", "records": ordered}
        ]

    ordered = sorted(
        records,
        key=lambda r: (r["timestamp_epoch"], _natural_key(r["filename"]), r["source_index"]),
    )
    explicit_strip_available = all(r["manifest_strip"] not in (None, "") for r in ordered)
    strips: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous_timestamp: float | None = None
    previous_explicit: str | None = None

    for record in ordered:
        derived_stratum = (
            str(record["manifest_stratum"])
            if record["manifest_stratum"] not in (None, "")
            else _gimbal_stratum(record["gimbal_pitch_deg"], record["gimbal_yaw_deg"])
        )
        explicit = str(record["manifest_strip"]) if explicit_strip_available else None
        gap = (
            None
            if previous_timestamp is None
            else record["timestamp_epoch"] - previous_timestamp
        )
        begins_new = current is None
        if current is not None:
            begins_new = (
                derived_stratum != current["stratum"]
                or (gap is not None and gap > max_gap_s)
                or (explicit_strip_available and explicit != previous_explicit)
            )
        if begins_new:
            strip_id = f"tg-{len(strips):04d}"
            current = {"strip_id": strip_id, "stratum": derived_stratum, "records": []}
            strips.append(current)
        current["records"].append(record)
        previous_timestamp = record["timestamp_epoch"]
        previous_explicit = explicit
    return "timestamp_gimbal", strips


class SplitAllocationError(ValueError):
    """Raised when the requested scene-level split cannot satisfy its guards."""


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _selection_partition(
    strip_infos: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, set[int]]], dict[str, int], dict[str, int]]:
    positions = {
        str(info["strip_id"]): {"test": set(), "guard": set()}
        for info in strip_infos
    }
    for candidate in selected:
        entry = positions[str(candidate["strip_id"])]
        entry["test"].update(candidate["test_positions"])
        entry["guard"].update(candidate["guard_positions"])
    for entry in positions.values():
        entry["guard"].difference_update(entry["test"])

    strip_train: dict[str, int] = {}
    stratum_train: dict[str, int] = {}
    for info in strip_infos:
        strip_id = str(info["strip_id"])
        stratum = str(info["stratum"])
        removed = positions[strip_id]["test"] | positions[strip_id]["guard"]
        train_count = int(info["frame_count"]) - len(removed)
        strip_train[strip_id] = train_count
        stratum_train[stratum] = stratum_train.get(stratum, 0) + train_count
    return positions, strip_train, stratum_train


def _can_add_candidate(
    strip_infos: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    minimum_train_frames: int,
) -> bool:
    _, strip_train, stratum_train = _selection_partition(
        strip_infos, [*selected, candidate]
    )
    return (
        strip_train[str(candidate["strip_id"])] >= minimum_train_frames
        and stratum_train[str(candidate["stratum"])] >= minimum_train_frames
    )


def _largest_remainder_quotas(
    eligible_by_stratum: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target: int,
    scene: str,
    seed: str,
) -> dict[str, dict[str, Any]]:
    total_weight = sum(len(items) for items in eligible_by_stratum.values())
    allocatable_target = min(target, total_weight)

    allocation: dict[str, dict[str, Any]] = {}
    for stratum in sorted(eligible_by_stratum):
        weight = len(eligible_by_stratum[stratum])
        ideal = (allocatable_target * weight / total_weight) if total_weight else 0.0
        floor_quota = int(math.floor(ideal))
        allocation[stratum] = {
            "eligible_block_count": weight,
            "ideal_quota": ideal,
            "floor_quota": floor_quota,
            "remainder": ideal - floor_quota,
            "tie_hash": _hash_json(
                {
                    "seed": seed,
                    "scene": scene,
                    "stratum": stratum,
                    "purpose": "largest_remainder_tie_break",
                }
            ),
            "quota": floor_quota,
        }

    remaining = allocatable_target - sum(item["quota"] for item in allocation.values())
    order = sorted(
        allocation,
        key=lambda stratum: (
            -float(allocation[stratum]["remainder"]),
            str(allocation[stratum]["tie_hash"]),
            stratum,
        ),
    )
    for stratum in order[:remaining]:
        allocation[stratum]["quota"] += 1
    return allocation


def build_split_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    seed: str,
    source_manifest: str | None = None,
    source_manifest_sha256: str | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    test_period_blocks: int = DEFAULT_TEST_PERIOD_BLOCKS,
    guard_frames: int = DEFAULT_GUARD_FRAMES,
    strip_max_gap_s: float = 10.0,
    fail_on_invalid: bool = True,
) -> dict[str, Any]:
    if not scene:
        raise ValueError("scene must not be empty")
    if not records:
        raise ValueError("records must not be empty")
    if block_size <= 0 or test_period_blocks <= 0 or guard_frames < 0:
        raise ValueError("block size and period must be positive; guard must be non-negative")
    if strip_max_gap_s <= 0:
        raise ValueError("strip_max_gap_s must be positive")

    normalised = _normalise_records(records)
    pair_ids = [record["pair_id"] for record in normalised]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_id values must be unique within a scene")

    reliability = _metadata_reliability(normalised)
    ordering_mode, strips = _build_strips(
        normalised,
        metadata_reliable=reliability["reliable"],
        max_gap_s=strip_max_gap_s,
    )
    input_records_hash = _hash_json(
        [
            {
                "source_index": record["source_index"],
                "pair_id": record["pair_id"],
                "source_record_hash": record["source_record_hash"],
            }
            for record in normalised
        ]
    )
    rule = {
        "ordering_preference": "timestamp_gimbal_then_filename_order",
        "ordering_mode": ordering_mode,
        "metadata_reliability_policy": "all_records_required",
        "strip_max_gap_s": strip_max_gap_s,
        "gimbal_pitch_bin_deg": 15,
        "gimbal_yaw_bin_deg": 90,
        "nadir_pitch_threshold_deg": -75.0,
        "block_size_frames": block_size,
        "test_period_blocks": test_period_blocks,
        "guard_frames_each_side": guard_frames,
        "partial_tail_block_can_be_test": False,
        "test_block_budget_rule": (
            "round_half_up(scene_frame_count / "
            "(block_size_frames * test_period_blocks))"
        ),
        "candidate_rule": "all complete blocks in every strip",
        "candidate_order_rule": (
            "sha256(seed,scene,stratum,strip_hash,block_index,ordered_block_pairs)"
        ),
        "candidate_hash_guard_invariant": True,
        "minimum_train_frames_after_selected_test": MIN_TRAIN_FRAMES_AFTER_TEST,
        "short_stratum_rule": (
            "a complete block without at least 16 remaining strip and stratum "
            "training frames is ineligible and remains train"
        ),
        "allocation_rule": (
            "largest_remainder_over_deterministic_feasible_capacity_by_stratum"
        ),
        "test_fraction_of_scene_range": [MIN_TEST_FRACTION, MAX_TEST_FRACTION],
        "zero_budget_fraction_exemption": True,
    }
    rule_hash = _hash_json(rule)

    stratum_frame_counts: dict[str, int] = {}
    strip_infos: list[dict[str, Any]] = []
    for strip in strips:
        strip_records = strip["records"]
        stratum = str(strip["stratum"])
        strip_hash = _hash_json(
            {
                "scene": scene,
                "strip_id": strip["strip_id"],
                "stratum": stratum,
                "ordered_pairs": [r["pair_id"] for r in strip_records],
            }
        )
        full_block_count = len(strip_records) // block_size
        info = {
            "strip_id": str(strip["strip_id"]),
            "stratum": stratum,
            "records": strip_records,
            "strip_hash": strip_hash,
            "frame_count": len(strip_records),
            "full_block_count": full_block_count,
            "tail_frame_count": len(strip_records) % block_size,
            "candidates": [],
        }
        strip_infos.append(info)
        stratum_frame_counts[stratum] = stratum_frame_counts.get(stratum, 0) + len(
            strip_records
        )

    all_candidates: list[dict[str, Any]] = []
    for info in strip_infos:
        strip_records = info["records"]
        for block_index in range(int(info["full_block_count"])):
            start = block_index * block_size
            end = start + block_size
            test_positions = tuple(range(start, end))
            guard_positions = tuple(
                [*range(max(0, start - guard_frames), start)]
                + [*range(end, min(len(strip_records), end + guard_frames))]
            )
            removed_count = len(set(test_positions) | set(guard_positions))
            strip_train_if_only = len(strip_records) - removed_count
            stratum_train_if_only = (
                stratum_frame_counts[str(info["stratum"])] - removed_count
            )
            eligibility_reasons: list[str] = []
            if strip_train_if_only < MIN_TRAIN_FRAMES_AFTER_TEST:
                eligibility_reasons.append("strip_would_have_fewer_than_16_train_frames")
            if stratum_train_if_only < MIN_TRAIN_FRAMES_AFTER_TEST:
                eligibility_reasons.append("stratum_would_have_fewer_than_16_train_frames")
            ordered_block_pairs = [
                strip_records[position]["pair_id"] for position in test_positions
            ]
            candidate = {
                "strip_id": str(info["strip_id"]),
                "stratum": str(info["stratum"]),
                "strip_hash": str(info["strip_hash"]),
                "block_index": block_index,
                "test_positions": test_positions,
                "guard_positions": guard_positions,
                "ordered_block_pairs": ordered_block_pairs,
                "candidate_hash": _hash_json(
                    {
                        "seed": seed,
                        "scene": scene,
                        "stratum": info["stratum"],
                        "strip_hash": info["strip_hash"],
                        "block_index": block_index,
                        "ordered_block_pairs": ordered_block_pairs,
                    }
                ),
                "eligible_in_isolation": not eligibility_reasons,
                "eligibility_reasons": eligibility_reasons,
                "strip_train_frames_if_only_selected": strip_train_if_only,
                "stratum_train_frames_if_only_selected": stratum_train_if_only,
            }
            info["candidates"].append(candidate)
            all_candidates.append(candidate)

    feasible_by_stratum: dict[str, list[dict[str, Any]]] = {
        stratum: [] for stratum in sorted(stratum_frame_counts)
    }
    for stratum in sorted(feasible_by_stratum):
        ordered = sorted(
            (
                candidate
                for candidate in all_candidates
                if candidate["stratum"] == stratum
                and candidate["eligible_in_isolation"]
            ),
            key=lambda candidate: (candidate["candidate_hash"], candidate["strip_id"], candidate["block_index"]),
        )
        feasible: list[dict[str, Any]] = []
        for candidate in ordered:
            if _can_add_candidate(
                strip_infos,
                feasible,
                candidate,
                minimum_train_frames=MIN_TRAIN_FRAMES_AFTER_TEST,
            ):
                candidate["feasible_sequence_rank"] = len(feasible)
                feasible.append(candidate)
            else:
                candidate["eligibility_reasons"].append(
                    "cumulative_strip_or_stratum_train_minimum"
                )
        feasible_by_stratum[stratum] = feasible

    target_test_blocks = _round_half_up(
        len(normalised) / float(block_size * test_period_blocks)
    )
    quota_details = _largest_remainder_quotas(
        feasible_by_stratum,
        target=target_test_blocks,
        scene=scene,
        seed=seed,
    )
    selected_candidates: list[dict[str, Any]] = []
    for stratum in sorted(feasible_by_stratum):
        quota = int(quota_details[stratum]["quota"])
        selected_candidates.extend(feasible_by_stratum[stratum][:quota])
        quota_details[stratum]["feasible_capacity"] = len(feasible_by_stratum[stratum])
        quota_details[stratum]["selected_block_count"] = min(
            quota, len(feasible_by_stratum[stratum])
        )

    selected_hashes = {
        str(candidate["candidate_hash"]) for candidate in selected_candidates
    }
    selected_test_blocks_hash = _hash_json(sorted(selected_hashes))
    allocation_hash = _hash_json(
        {
            "scene": scene,
            "seed": seed,
            "rule_hash": rule_hash,
            "input_records_hash": input_records_hash,
            "target_test_blocks": target_test_blocks,
            "stratum_quotas": {
                stratum: int(details["quota"])
                for stratum, details in sorted(quota_details.items())
            },
            "selected_candidate_hashes": sorted(selected_hashes),
        }
    )
    positions_by_strip, strip_train_counts, stratum_train_counts = _selection_partition(
        strip_infos, selected_candidates
    )
    test_candidate_by_position: dict[tuple[str, int], Mapping[str, Any]] = {}
    for candidate in selected_candidates:
        for position in candidate["test_positions"]:
            test_candidate_by_position[(str(candidate["strip_id"]), position)] = candidate

    output_records: list[dict[str, Any]] = []
    strip_summaries: list[dict[str, Any]] = []
    for info in strip_infos:
        strip_id = str(info["strip_id"])
        strip_records = info["records"]
        test_positions = positions_by_strip[strip_id]["test"]
        guard_positions = positions_by_strip[strip_id]["guard"]
        for position, record in enumerate(strip_records):
            block_index = position // block_size
            block_offset = position % block_size
            if position in test_positions:
                split = "test"
                assignment_rule = "scene_budget_stratum_allocated_test_block"
                selected_candidate_hash = str(
                    test_candidate_by_position[(strip_id, position)]["candidate_hash"]
                )
            elif position in guard_positions:
                split = "guard"
                assignment_rule = "adjacent_to_test_block"
                selected_candidate_hash = None
            else:
                split = "train"
                assignment_rule = "non_test_non_guard"
                selected_candidate_hash = None
            stable_assignment = {
                "scene": scene,
                "pair_id": record["pair_id"],
                "stratum": info["stratum"],
                "strip_id": strip_id,
                "strip_hash": info["strip_hash"],
                "position_in_strip": position,
                "block_index": block_index,
                "block_offset": block_offset,
                "split": split,
                "rule": assignment_rule,
                "rule_hash": rule_hash,
                "input_records_hash": input_records_hash,
                "allocation_hash": allocation_hash,
                "selected_candidate_hash": selected_candidate_hash,
            }
            output_record = {
                **stable_assignment,
                "hash": _hash_json(stable_assignment),
                "original_files": record["original_files"],
                "source_record_hash": record["source_record_hash"],
                "filename": record["filename"],
                "timestamp_utc": record["timestamp_utc"],
                "gimbal_pitch_deg": record["gimbal_pitch_deg"],
                "gimbal_yaw_deg": record["gimbal_yaw_deg"],
            }
            if record["temperature_npy"] is not None:
                output_record["temperature_npy"] = record["temperature_npy"]
            output_records.append(output_record)

        strip_summaries.append(
            {
                "strip_id": strip_id,
                "stratum": info["stratum"],
                "strip_hash": info["strip_hash"],
                "frame_count": len(strip_records),
                "full_block_count": info["full_block_count"],
                "tail_frame_count": info["tail_frame_count"],
                "test_block_indices": sorted(
                    candidate["block_index"]
                    for candidate in selected_candidates
                    if candidate["strip_id"] == strip_id
                ),
                "train_frame_count": strip_train_counts[strip_id],
                "test_frame_count": len(test_positions),
                "guard_frame_count": len(guard_positions),
                "candidates": [
                    {
                        "block_index": candidate["block_index"],
                        "candidate_hash": candidate["candidate_hash"],
                        "eligible_in_isolation": candidate["eligible_in_isolation"],
                        "feasible_sequence_rank": candidate.get("feasible_sequence_rank"),
                        "selected": candidate["candidate_hash"] in selected_hashes,
                        "eligibility_reasons": candidate["eligibility_reasons"],
                        "strip_train_frames_if_only_selected": candidate[
                            "strip_train_frames_if_only_selected"
                        ],
                        "stratum_train_frames_if_only_selected": candidate[
                            "stratum_train_frames_if_only_selected"
                        ],
                    }
                    for candidate in info["candidates"]
                ],
            }
        )

    counts = {
        split: sum(record["split"] == split for record in output_records)
        for split in ("train", "test", "guard")
    }
    test_fraction = counts["test"] / len(output_records) if output_records else 0.0
    strips_without_train = sorted(
        strip_id for strip_id, count in strip_train_counts.items() if count == 0
    )
    strata_without_train = sorted(
        stratum for stratum, count in stratum_train_counts.items() if count == 0
    )
    selected_strip_ids = {str(candidate["strip_id"]) for candidate in selected_candidates}
    selected_strata = {str(candidate["stratum"]) for candidate in selected_candidates}
    validation_errors: list[str] = []
    if len(selected_candidates) != target_test_blocks:
        validation_errors.append("test_block_budget_shortfall")
    if strips_without_train:
        validation_errors.append("strip_without_train")
    if strata_without_train:
        validation_errors.append("stratum_without_train")
    if any(
        strip_train_counts[strip_id] < MIN_TRAIN_FRAMES_AFTER_TEST
        for strip_id in selected_strip_ids
    ):
        validation_errors.append("selected_strip_below_16_train")
    if any(
        stratum_train_counts[stratum] < MIN_TRAIN_FRAMES_AFTER_TEST
        for stratum in selected_strata
    ):
        validation_errors.append("selected_stratum_below_16_train")
    fraction_check_required = target_test_blocks > 0
    if fraction_check_required and not (
        MIN_TEST_FRACTION <= test_fraction <= MAX_TEST_FRACTION
    ):
        validation_errors.append("test_fraction_out_of_range")
    validation = {
        "status": "passed" if not validation_errors else "failed",
        "errors": validation_errors,
        "exact_budget": len(selected_candidates) == target_test_blocks,
        "no_strip_without_train": not strips_without_train,
        "no_stratum_without_train": not strata_without_train,
        "strips_without_train": strips_without_train,
        "strata_without_train": strata_without_train,
        "selected_strip_minimum_train_passed": not any(
            strip_train_counts[strip_id] < MIN_TRAIN_FRAMES_AFTER_TEST
            for strip_id in selected_strip_ids
        ),
        "selected_stratum_minimum_train_passed": not any(
            stratum_train_counts[stratum] < MIN_TRAIN_FRAMES_AFTER_TEST
            for stratum in selected_strata
        ),
        "test_fraction_check_required": fraction_check_required,
        "test_fraction_of_scene": test_fraction,
        "test_fraction_range": [MIN_TEST_FRACTION, MAX_TEST_FRACTION],
        "test_fraction_passed": (
            None
            if not fraction_check_required
            else MIN_TEST_FRACTION <= test_fraction <= MAX_TEST_FRACTION
        ),
    }

    stratum_summaries: list[dict[str, Any]] = []
    for stratum in sorted(stratum_frame_counts):
        stratum_records = [record for record in output_records if record["stratum"] == stratum]
        stratum_counts = {
            split_name: sum(record["split"] == split_name for record in stratum_records)
            for split_name in ("train", "test", "guard")
        }
        stratum_summaries.append(
            {
                "stratum": stratum,
                "frame_count": stratum_frame_counts[stratum],
                **quota_details[stratum],
                "counts": stratum_counts,
                "without_test": stratum_counts["test"] == 0,
                "selected_candidate_hashes": sorted(
                    candidate["candidate_hash"]
                    for candidate in selected_candidates
                    if candidate["stratum"] == stratum
                ),
            }
        )

    split_basis = [
        {
            key: record[key]
            for key in (
                "pair_id",
                "stratum",
                "strip_id",
                "block_index",
                "block_offset",
                "split",
                "rule",
                "hash",
                "source_record_hash",
            )
        }
        for record in output_records
    ]
    result = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "seed": seed,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "input_records_hash": input_records_hash,
        "rule": rule,
        "rule_hash": rule_hash,
        "metadata_reliability": reliability,
        "counts": {"total": len(output_records), **counts},
        "test_block_budget": {
            "target": target_test_blocks,
            "selected": len(selected_candidates),
            "shortfall": max(0, target_test_blocks - len(selected_candidates)),
        },
        "allocation_hash": allocation_hash,
        "selected_test_blocks_hash": selected_test_blocks_hash,
        "selected_candidate_hashes": sorted(selected_hashes),
        "stratum_allocations": stratum_summaries,
        "strips": strip_summaries,
        "records": output_records,
        "validation": validation,
    }
    result["split_hash"] = _hash_json(
        {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "seed": seed,
            "rule_hash": rule_hash,
            "source_manifest_sha256": source_manifest_sha256,
            "input_records_hash": input_records_hash,
            "allocation_hash": allocation_hash,
            "records": split_basis,
        }
    )
    if validation_errors and fail_on_invalid:
        raise SplitAllocationError(
            f"split validation failed for scene {scene}: {', '.join(validation_errors)}"
        )
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", help="override the scene stored in the input manifest")
    parser.add_argument("--seed", default="uav-tgs-aaai27-v1")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--test-period-blocks", type=int, default=DEFAULT_TEST_PERIOD_BLOCKS)
    parser.add_argument("--guard-frames", type=int, default=DEFAULT_GUARD_FRAMES)
    parser.add_argument("--strip-max-gap-s", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records, metadata = load_manifest(args.manifest)
    scene = args.scene or metadata.get("scene")
    if scene is None:
        scenes = {str(record.get("scene")) for record in records if record.get("scene")}
        if len(scenes) == 1:
            scene = scenes.pop()
    if not scene:
        raise ValueError("scene is absent; pass --scene")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"split output exists; pass --overwrite: {args.output}")
    result = build_split_manifest(
        records,
        scene=str(scene),
        seed=str(args.seed),
        source_manifest=str(args.manifest.resolve()),
        source_manifest_sha256=_file_sha256(args.manifest),
        block_size=args.block_size,
        test_period_blocks=args.test_period_blocks,
        guard_frames=args.guard_frames,
        strip_max_gap_s=args.strip_max_gap_s,
    )
    _write_json(args.output, result)
    print(
        f"scene={result['scene']} mode={result['rule']['ordering_mode']} "
        f"train={result['counts']['train']} test={result['counts']['test']} "
        f"guard={result['counts']['guard']} split_hash={result['split_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
