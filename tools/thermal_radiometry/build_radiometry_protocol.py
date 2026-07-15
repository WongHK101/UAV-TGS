#!/usr/bin/env python3
"""Resolve an R-JPEG audit manifest into an explicit decode protocol.

The output is JSONL that can be passed directly to
``decode_temperature.py --input-manifest``.  No radiometric parameter is left
implicit.  Distance is resolved once per view/flight strip in this order:

1. robust median of valid strip LRF measurements;
2. robust median of ``relative_altitude / sin(abs(gimbal_pitch))``;
3. an explicitly supplied scene benchmark assumption.

The input R-JPEGs are treated as read-only.  The resolver never writes below a
source image directory and verifies source size/mtime before and after writing
the derived manifest.
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

try:
    from . import build_split
except ImportError:  # pragma: no cover - direct script execution
    import build_split  # type: ignore


SCHEMA_VERSION = "uav-tgs.radiometry-protocol.v1"
DISTANCE_POLICY = (
    "strip_valid_lrf_robust_median",
    "strip_relative_altitude_gimbal_geometry_estimate",
    "scene_benchmark_assumption",
)
CONSTANT_SPECS = {
    "humidity_percent": ("embedded_humidity_percent", 0.0, 100.0),
    "ambient_c": ("embedded_ambient_c", None, None),
    "reflected_c": ("embedded_reflected_c", None, None),
    "emissivity": ("embedded_emissivity", 0.0, 1.0),
}
PASSTHROUGH_FIELDS = (
    "capture_time",
    "rgb_capture_time",
    "timestamp",
    "timestamp_utc",
    "gimbal_pitch_deg",
    "gimbal_yaw_deg",
    "gimbal_roll_deg",
    "gps_latitude",
    "gps_longitude",
    "gps_altitude_m",
    "relative_altitude_m",
    "rgb_path",
    "thermal_path",
    "original_files",
    "stratum",
)
AMBIGUOUS_SOURCES = {
    "embedded",
    "global_cli",
    "manifest_unspecified",
    "unknown",
    "unspecified",
}


class ProtocolConfigurationError(ValueError):
    """Raised before any output is published when the protocol is invalid."""


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


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_float(value: Any) -> float | None:
    result = _finite_float(value)
    return result if result is not None and result > 0.0 else None


def robust_inliers(values: Iterable[float]) -> list[float]:
    """Return the deterministic Hampel inlier values used by the strip estimate."""
    finite_values: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is not None:
            finite_values.append(number)
    ordered = sorted(finite_values)
    if not ordered:
        return []
    center = float(statistics.median(ordered))
    if len(ordered) < 3:
        return ordered
    deviations = [abs(value - center) for value in ordered]
    mad = float(statistics.median(deviations))
    if mad == 0.0:
        inliers = [value for value in ordered if value == center]
        if not inliers:
            inliers = ordered
    else:
        threshold = 3.0 * 1.4826 * mad
        inliers = [value for value in ordered if abs(value - center) <= threshold]
    if not inliers:  # defensive; the median itself should always survive
        inliers = ordered
    return inliers


def robust_median(values: Iterable[float]) -> tuple[float | None, int, int]:
    """Return a deterministic Hampel-filtered median and input/inlier counts."""
    finite_values = [number for value in values if (number := _finite_float(value)) is not None]
    inliers = robust_inliers(finite_values)
    if not inliers:
        return None, 0, 0
    return float(statistics.median(inliers)), len(finite_values), len(inliers)


def _source_path(record: Mapping[str, Any], base: Path) -> Path:
    raw = record.get("source_path", record.get("thermal_path"))
    if raw in (None, ""):
        raise ProtocolConfigurationError("Every audit record needs source_path/thermal_path")
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _validate_audit_records(
    records: Sequence[Mapping[str, Any]], *, manifest_path: Path, scene_override: str | None
) -> tuple[list[dict[str, Any]], str, dict[str, tuple[int, int]]]:
    if not records:
        raise ProtocolConfigurationError("Audit manifest contains no records")
    base = manifest_path.resolve().parent
    normalized: list[dict[str, Any]] = []
    source_snapshots: dict[str, tuple[int, int]] = {}
    seen_pairs: set[str] = set()
    seen_sources: set[str] = set()
    scenes: set[str] = set()
    for index, original in enumerate(records):
        if not isinstance(original, Mapping):
            raise ProtocolConfigurationError(f"Audit record {index} is not a JSON object")
        record = dict(original)
        source = _source_path(record, base)
        if not source.is_file():
            raise FileNotFoundError(f"R-JPEG source does not exist: {source}")
        if record.get("rjpeg_detected") is not True:
            raise ProtocolConfigurationError(
                f"Audit record is not a confirmed R-JPEG: {record.get('pair_id', source)}"
            )
        pair_id = str(record.get("pair_id") or record.get("frame_id") or source.stem)
        if pair_id in seen_pairs:
            raise ProtocolConfigurationError(f"Duplicate pair_id in audit manifest: {pair_id}")
        source_key = os.path.normcase(str(source))
        if source_key in seen_sources:
            raise ProtocolConfigurationError(f"Duplicate source path in audit manifest: {source}")
        seen_pairs.add(pair_id)
        seen_sources.add(source_key)
        scene = str(scene_override or record.get("scene") or "").strip()
        if not scene:
            raise ProtocolConfigurationError("Scene is absent; pass --scene")
        scenes.add(scene)
        stat = source.stat()
        audited_size = record.get("source_size_bytes")
        audited_mtime = record.get("source_mtime_ns")
        if audited_size is not None and int(audited_size) != stat.st_size:
            raise ProtocolConfigurationError(f"Source size changed since audit: {source}")
        if audited_mtime is not None and int(audited_mtime) != stat.st_mtime_ns:
            raise ProtocolConfigurationError(f"Source mtime changed since audit: {source}")
        source_snapshots[source_key] = (stat.st_size, stat.st_mtime_ns)
        record["source_path"] = str(source)
        record["scene"] = scene
        record["pair_id"] = pair_id
        record["frame_id"] = str(record.get("frame_id") or pair_id)
        normalized.append(record)
    if len(scenes) != 1:
        raise ProtocolConfigurationError(
            f"A protocol manifest must contain exactly one scene, got: {sorted(scenes)}"
        )
    return normalized, next(iter(scenes)), source_snapshots


def _assign_strips(
    records: Sequence[Mapping[str, Any]], *, strip_max_gap_s: float
) -> tuple[str, list[dict[str, Any]], Mapping[str, Any]]:
    if strip_max_gap_s <= 0:
        raise ProtocolConfigurationError("strip_max_gap_s must be positive")
    explicit = [record.get("strip_id") not in (None, "") for record in records]
    if any(explicit) and not all(explicit):
        raise ProtocolConfigurationError("strip_id must be present for every record or no record")

    normalised = build_split._normalise_records(records)
    if all(explicit):
        groups: dict[str, list[dict[str, Any]]] = {}
        for item in normalised:
            strip_id = str(item["manifest_strip"])
            groups.setdefault(strip_id, []).append(item)
        strips: list[dict[str, Any]] = []
        for strip_id in sorted(groups, key=build_split._natural_key):
            members = sorted(
                groups[strip_id],
                key=lambda item: (
                    item["timestamp_epoch"] is None,
                    item["timestamp_epoch"] if item["timestamp_epoch"] is not None else 0.0,
                    build_split._natural_key(item["filename"]),
                    item["source_index"],
                ),
            )
            strata = {
                str(item["manifest_stratum"])
                for item in members
                if item["manifest_stratum"] not in (None, "")
            }
            stratum = next(iter(strata)) if len(strata) == 1 else "explicit_strip"
            strips.append({"strip_id": strip_id, "stratum": stratum, "records": members})
        return "explicit_strip", strips, {
            "reliable": True,
            "record_count": len(records),
            "reasons": [],
        }

    reliability = build_split._metadata_reliability(normalised)
    mode, strips = build_split._build_strips(
        normalised,
        metadata_reliable=bool(reliability["reliable"]),
        max_gap_s=strip_max_gap_s,
    )
    return mode, strips, reliability


def _validate_constant(
    name: str,
    value: Any,
    source: Any,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    number = _finite_float(value)
    if number is None:
        raise ProtocolConfigurationError(f"Scene constant {name} must be a finite number")
    embedded_field, minimum, maximum = CONSTANT_SPECS[name]
    if name == "emissivity" and number <= 0:
        raise ProtocolConfigurationError("emissivity must be in (0, 1]")
    if minimum is not None and name != "emissivity" and number < minimum:
        raise ProtocolConfigurationError(f"{name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ProtocolConfigurationError(f"{name} must be <= {maximum}")
    label = str(source or "").strip()
    if not label:
        raise ProtocolConfigurationError(f"Scene constant {name} needs an explicit source")
    if label.casefold() in AMBIGUOUS_SOURCES:
        raise ProtocolConfigurationError(f"Ambiguous provenance for {name}: {label!r}")
    if "embedded" in label.casefold():
        traceable: list[float] = []
        for record in records:
            embedded = _finite_float(record.get(embedded_field))
            metadata_sources = record.get("metadata_sources")
            source_tag = (
                metadata_sources.get(embedded_field)
                if isinstance(metadata_sources, Mapping)
                else None
            )
            if embedded is not None and source_tag not in (None, ""):
                traceable.append(embedded)
        if not traceable:
            raise ProtocolConfigurationError(
                f"{name} is labelled embedded but the audit has no traceable {embedded_field}"
            )
        if label.casefold() == "scene_embedded_median":
            expected = float(statistics.median(traceable))
            if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-9):
                raise ProtocolConfigurationError(
                    f"{name}={number} does not equal audited scene embedded median {expected}"
                )
    return {"value": number, "source": label, "scope": "scene"}


def _geometry_distance(record: Mapping[str, Any], minimum_abs_pitch_deg: float) -> float | None:
    altitude = _positive_float(record.get("relative_altitude_m"))
    pitch = _finite_float(record.get("gimbal_pitch_deg"))
    if altitude is None or pitch is None or abs(pitch) < minimum_abs_pitch_deg:
        return None
    denominator = abs(math.sin(math.radians(pitch)))
    if denominator <= 0.0:
        return None
    distance = altitude / denominator
    return distance if math.isfinite(distance) and distance > 0.0 else None


def _source_snapshot_unchanged(
    records: Sequence[Mapping[str, Any]], snapshots: Mapping[str, tuple[int, int]]
) -> None:
    for record in records:
        source = Path(str(record["source_path"]))
        stat = source.stat()
        key = os.path.normcase(str(source))
        if snapshots[key] != (stat.st_size, stat.st_mtime_ns):
            raise RuntimeError(f"Raw source changed while resolving protocol: {source}")


def _assert_output_outside_sources(output: Path, records: Sequence[Mapping[str, Any]]) -> None:
    output_parent = output.resolve(strict=False).parent
    for record in records:
        source_parent = Path(str(record["source_path"])).resolve().parent
        if (
            output_parent == source_parent
            or source_parent in output_parent.parents
            or output_parent in source_parent.parents
        ):
            raise ProtocolConfigurationError(
                f"Refusing to write protocol output inside raw source directory: {output_parent}"
            )


def build_protocol(
    records: Sequence[Mapping[str, Any]],
    *,
    scene: str,
    source_manifest: str,
    source_manifest_sha256: str,
    scene_constants: Mapping[str, Mapping[str, Any]],
    scene_distance_m: float | None,
    scene_distance_provenance: str,
    strip_max_gap_s: float = 10.0,
    minimum_abs_pitch_deg: float = 10.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build decode-ready records and a deterministic protocol summary."""
    if minimum_abs_pitch_deg <= 0.0 or minimum_abs_pitch_deg >= 90.0:
        raise ProtocolConfigurationError("minimum_abs_pitch_deg must be in (0, 90)")
    missing_constants = [name for name in CONSTANT_SPECS if name not in scene_constants]
    if missing_constants:
        raise ProtocolConfigurationError(
            "Missing explicit scene constants: " + ", ".join(sorted(missing_constants))
        )
    constants = {
        name: _validate_constant(
            name,
            scene_constants[name].get("value"),
            scene_constants[name].get("source"),
            records,
        )
        for name in CONSTANT_SPECS
    }
    scene_distance = _positive_float(scene_distance_m)
    if scene_distance_m is not None and scene_distance is None:
        raise ProtocolConfigurationError("scene_distance_m must be finite and > 0")
    assumption_provenance = str(scene_distance_provenance or "").strip()
    if scene_distance is not None and not assumption_provenance:
        raise ProtocolConfigurationError("scene distance assumption needs explicit provenance")

    mode, strips, metadata_reliability = _assign_strips(
        records, strip_max_gap_s=strip_max_gap_s
    )
    by_source_index = {index: dict(record) for index, record in enumerate(records)}
    output_records: list[dict[str, Any]] = []
    strip_summaries: list[dict[str, Any]] = []

    for strip in strips:
        members = [by_source_index[item["source_index"]] for item in strip["records"]]
        lrf_values = [
            distance
            for record in members
            if record.get("lrf_distance_valid") is True
            for distance in [_positive_float(record.get("lrf_distance_m"))]
            if distance is not None
        ]
        lrf_median, lrf_count, lrf_inlier_count = robust_median(lrf_values)
        lrf_inlier_values = robust_inliers(lrf_values)
        geometry_values = [
            distance
            for record in members
            for distance in [_geometry_distance(record, minimum_abs_pitch_deg)]
            if distance is not None
        ]
        geometry_median, geometry_count, geometry_inlier_count = robust_median(geometry_values)
        # A filename-order fallback is not a reliable flight/view strip, so a
        # geometry value pooled across it must not be presented as strip-level.
        strip_is_view_flight_resolved = mode != "filename_order_fallback"
        lrf_usable = strip_is_view_flight_resolved and lrf_median is not None
        geometry_usable = strip_is_view_flight_resolved and geometry_median is not None

        if lrf_usable:
            used_distance = lrf_median
            distance_source = "strip_valid_lrf_robust_median"
        elif geometry_usable:
            used_distance = float(geometry_median)
            distance_source = "strip_relative_altitude_gimbal_geometry_estimate"
        elif scene_distance is not None:
            used_distance = scene_distance
            distance_source = "scene_benchmark_assumption"
        else:
            raise ProtocolConfigurationError(
                f"Unable to resolve distance for strip {strip['strip_id']}; "
                "provide --scene-distance-m"
            )

        strip_summary = {
            "strip_id": str(strip["strip_id"]),
            "stratum": str(strip["stratum"]),
            "frame_count": len(members),
            "distance_m": used_distance,
            "distance_source": distance_source,
            "valid_lrf_count": lrf_count,
            "robust_lrf_inlier_count": lrf_inlier_count,
            "robust_lrf_outlier_count": lrf_count - lrf_inlier_count,
            "robust_lrf_median_m": lrf_median,
            "lrf_estimate_usable": lrf_usable,
            "geometry_candidate_count": geometry_count,
            "robust_geometry_inlier_count": geometry_inlier_count,
            "geometry_estimate_m": geometry_median,
            "geometry_estimate_usable": geometry_usable,
        }
        strip_summaries.append(strip_summary)

        for record in members:
            raw_lrf = _finite_float(record.get("lrf_distance_m"))
            raw_lrf_valid = bool(
                record.get("lrf_distance_valid") is True
                and raw_lrf is not None
                and raw_lrf > 0.0
            )
            raw_lrf_robust_inlier = (
                raw_lrf in lrf_inlier_values if raw_lrf_valid and raw_lrf is not None else None
            )
            if distance_source == "strip_valid_lrf_robust_median":
                if not raw_lrf_valid:
                    fallback_reason = "frame_lrf_invalid_or_missing;using_strip_valid_lrf_median"
                elif raw_lrf_robust_inlier is False:
                    fallback_reason = "frame_lrf_robust_outlier;using_strip_valid_lrf_median"
                else:
                    fallback_reason = "none"
            elif distance_source == "strip_relative_altitude_gimbal_geometry_estimate":
                fallback_reason = "strip_has_no_valid_lrf"
            else:
                if mode == "filename_order_fallback":
                    reason = "view_flight_strip_unavailable"
                    if lrf_median is not None:
                        reason += ";lrf_pool_rejected_without_reliable_view_flight_strip"
                    if geometry_median is not None:
                        reason += ";geometry_rejected_without_reliable_view_flight_strip"
                else:
                    reason = "strip_has_no_valid_lrf;strip_geometry_estimate_unavailable"
                fallback_reason = reason

            distance_resolution = {
                "strip_id": str(strip["strip_id"]),
                "raw_lrf_distance_m": raw_lrf,
                "raw_lrf_status": record.get("lrf_status"),
                "raw_lrf_valid": raw_lrf_valid,
                "raw_lrf_robust_inlier": raw_lrf_robust_inlier,
                "raw_lrf_robust_outlier": raw_lrf_robust_inlier is False,
                "raw_lrf_source_tag": (
                    record.get("metadata_sources", {}).get("lrf_distance_m")
                    if isinstance(record.get("metadata_sources"), Mapping)
                    else None
                ),
                "used_distance_m": used_distance,
                "used_distance_source": distance_source,
                "fallback_reason": fallback_reason,
                "raw_to_used_delta_m": (
                    raw_lrf - used_distance if raw_lrf_valid and raw_lrf is not None else None
                ),
                "strip_valid_lrf_count": lrf_count,
                "strip_lrf_inlier_count": lrf_inlier_count,
                "strip_geometry_candidate_count": geometry_count,
                "strip_geometry_inlier_count": geometry_inlier_count,
                "scene_assumption_provenance": (
                    assumption_provenance
                    if distance_source == "scene_benchmark_assumption"
                    else None
                ),
            }
            decode_parameters = {
                "distance_m": {"value": used_distance, "source": distance_source},
                **{
                    name: {"value": item["value"], "source": item["source"]}
                    for name, item in constants.items()
                },
            }
            output: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "frame_id": str(record["frame_id"]),
                "pair_id": str(record["pair_id"]),
                "source_path": str(record["source_path"]),
                "strip_id": str(strip["strip_id"]),
                "decode_parameters": decode_parameters,
                "raw_lrf_distance_m": raw_lrf,
                "raw_lrf_status": record.get("lrf_status"),
                "raw_lrf_valid": raw_lrf_valid,
                "raw_lrf_robust_inlier": raw_lrf_robust_inlier,
                "raw_lrf_robust_outlier": raw_lrf_robust_inlier is False,
                "used_distance_m": used_distance,
                "used_distance_source": distance_source,
                "distance_fallback_reason": fallback_reason,
                "source_audit_record_hash": _hash_json(record),
                "source_manifest": source_manifest,
                "source_manifest_sha256": source_manifest_sha256,
                "metadata": {"radiometry_protocol": distance_resolution},
            }
            for field in PASSTHROUGH_FIELDS:
                if field in record:
                    output[field] = record[field]
            output["protocol_record_hash"] = _hash_json(output)
            output_records.append(output)

    protocol_basis = [
        {
            key: record[key]
            for key in (
                "scene",
                "frame_id",
                "pair_id",
                "source_path",
                "strip_id",
                "decode_parameters",
                "raw_lrf_distance_m",
                "raw_lrf_status",
                "raw_lrf_valid",
                "raw_lrf_robust_inlier",
                "raw_lrf_robust_outlier",
                "used_distance_m",
                "used_distance_source",
                "distance_fallback_reason",
                "source_audit_record_hash",
                "protocol_record_hash",
            )
        }
        for record in output_records
    ]
    protocol_hash = _hash_json(protocol_basis)
    for record in output_records:
        record["protocol_hash"] = protocol_hash

    source_counts = {
        source: sum(record["used_distance_source"] == source for record in output_records)
        for source in DISTANCE_POLICY
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scene": scene,
        "record_count": len(output_records),
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "protocol_hash": protocol_hash,
        "distance_policy": list(DISTANCE_POLICY),
        "distance_source_frame_counts": source_counts,
        "scene_constants": constants,
        "scene_distance_assumption_m": scene_distance,
        "scene_distance_assumption_provenance": assumption_provenance,
        "strip_assignment_mode": mode,
        "strip_max_gap_s": strip_max_gap_s,
        "minimum_abs_pitch_deg": minimum_abs_pitch_deg,
        "metadata_reliability": metadata_reliability,
        "strips": strip_summaries,
    }
    return output_records, summary


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--strip-max-gap-s", type=float, default=10.0)
    parser.add_argument("--minimum-abs-pitch-deg", type=float, default=10.0)
    parser.add_argument("--scene-distance-m", type=float)
    parser.add_argument(
        "--scene-distance-provenance",
        default="benchmark_assumption",
        help="Traceable label for the scene distance fallback",
    )
    parser.add_argument("--humidity-percent", type=float, required=True)
    parser.add_argument("--humidity-percent-source", required=True)
    parser.add_argument("--ambient-c", type=float, required=True)
    parser.add_argument("--ambient-c-source", required=True)
    parser.add_argument("--reflected-c", type=float, required=True)
    parser.add_argument("--reflected-c-source", required=True)
    parser.add_argument("--emissivity", type=float, default=0.95)
    parser.add_argument("--emissivity-source", default="benchmark_assumption")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.audit_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Audit manifest does not exist: {manifest_path}")
    output = args.output.expanduser().resolve(strict=False)
    summary_out = (
        args.summary_out.expanduser().resolve(strict=False)
        if args.summary_out is not None
        else output.with_suffix(output.suffix + ".summary.json")
    )
    if output == summary_out:
        raise ProtocolConfigurationError("Protocol output and summary output must differ")
    for path in (output, summary_out):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {path}")

    raw_records, _ = build_split.load_manifest(manifest_path)
    records, scene, snapshots = _validate_audit_records(
        raw_records, manifest_path=manifest_path, scene_override=args.scene
    )
    _assert_output_outside_sources(output, records)
    _assert_output_outside_sources(summary_out, records)
    scene_constants = {
        "humidity_percent": {
            "value": args.humidity_percent,
            "source": args.humidity_percent_source,
        },
        "ambient_c": {"value": args.ambient_c, "source": args.ambient_c_source},
        "reflected_c": {"value": args.reflected_c, "source": args.reflected_c_source},
        "emissivity": {"value": args.emissivity, "source": args.emissivity_source},
    }
    output_records, summary = build_protocol(
        records,
        scene=scene,
        source_manifest=str(manifest_path),
        source_manifest_sha256=_file_sha256(manifest_path),
        scene_constants=scene_constants,
        scene_distance_m=args.scene_distance_m,
        scene_distance_provenance=args.scene_distance_provenance,
        strip_max_gap_s=args.strip_max_gap_s,
        minimum_abs_pitch_deg=args.minimum_abs_pitch_deg,
    )
    _source_snapshot_unchanged(records, snapshots)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, output_records)
    _write_json(summary_out, summary)
    _source_snapshot_unchanged(records, snapshots)
    return {**summary, "output": str(output), "summary_out": str(summary_out)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run(args)
    counts = summary["distance_source_frame_counts"]
    print(
        f"scene={summary['scene']} records={summary['record_count']} "
        f"lrf={counts['strip_valid_lrf_robust_median']} "
        f"geometry={counts['strip_relative_altitude_gimbal_geometry_estimate']} "
        f"assumption={counts['scene_benchmark_assumption']} "
        f"protocol_hash={summary['protocol_hash']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
