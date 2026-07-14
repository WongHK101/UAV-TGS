#!/usr/bin/env python3
"""Read-only audit of DJI radiometric JPEG metadata.

The auditor deliberately does not decode temperatures.  It records the metadata
needed to choose a later radiometry protocol and can optionally join an explicit
RGB/thermal pair manifest to measure timestamp offsets.  ExifTool is invoked in
read-only JSON mode; no metadata is written back to source images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "uav-tgs.rjpeg-audit.v1"
DEFAULT_SUFFIXES = (".jpg", ".jpeg", ".rjpeg")


def _json_value(value: Any) -> Any:
    """Convert ExifTool values into strict JSON-compatible scalar containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _tag_name(key: str) -> str:
    """Return a normalized ExifTool tag name without its optional group prefix."""
    tail = key.rsplit(":", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", tail.lower())


def _tag_index(metadata: Mapping[str, Any]) -> Dict[str, List[Tuple[str, Any]]]:
    index: Dict[str, List[Tuple[str, Any]]] = {}
    for key, value in metadata.items():
        index.setdefault(_tag_name(str(key)), []).append((str(key), value))
    return index


def _first_tag(
    index: Mapping[str, Sequence[Tuple[str, Any]]], aliases: Sequence[str]
) -> Tuple[Any, Optional[str]]:
    for alias in aliases:
        candidates = index.get(_tag_name(alias), ())
        if candidates:
            key, value = candidates[0]
            return _json_value(value), key
    return None, None


def _as_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        result = float(match.group(0))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> Optional[int]:
    number = _as_float(value)
    return int(number) if number is not None else None


def _parse_exif_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    match = re.match(
        r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(\.\d+)?(?:([+-]\d{2}:?\d{2})|Z)?",
        text,
    )
    if not match:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    fraction = match.group(7) or ""
    offset = match.group(8) or ""
    iso = (
        f"{match.group(1)}-{match.group(2)}-{match.group(3)}T"
        f"{match.group(4)}:{match.group(5)}:{match.group(6)}{fraction}{offset}"
    )
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def _timestamp_delta_ms(thermal_value: Any, rgb_value: Any) -> Optional[float]:
    thermal = _parse_exif_datetime(thermal_value)
    rgb = _parse_exif_datetime(rgb_value)
    if thermal is None or rgb is None:
        return None
    if (thermal.tzinfo is None) != (rgb.tzinfo is None):
        return None
    return (thermal - rgb).total_seconds() * 1000.0


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str, base: Optional[Path] = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve(strict=False)


def _read_pair_manifest(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    base = path.resolve().parent
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            thermal_value = record.get("thermal_path", record.get("source_path"))
            if not thermal_value:
                raise ValueError(f"Missing thermal_path/source_path at {path}:{line_number}")
            item = dict(record)
            item["thermal_path"] = str(_resolve_path(str(thermal_value), base))
            if record.get("rgb_path"):
                item["rgb_path"] = str(_resolve_path(str(record["rgb_path"]), base))
            records.append(item)
    return records


def _discover_inputs(inputs: Sequence[str], recursive: bool) -> List[Dict[str, Any]]:
    found: Dict[str, Dict[str, Any]] = {}
    for raw_value in inputs:
        path = _resolve_path(raw_value)
        if path.is_file():
            candidates: Iterable[Path] = (path,)
            root = path.parent
        elif path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
            root = path
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in DEFAULT_SUFFIXES:
                continue
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            relative = resolved.relative_to(root) if resolved.is_relative_to(root) else Path(resolved.name)
            scene = relative.parts[0] if len(relative.parts) > 1 else root.name
            found[key] = {
                "thermal_path": str(resolved),
                "scene": scene,
                "frame_id": resolved.stem,
                "pair_id": resolved.stem,
            }
    return [found[key] for key in sorted(found)]


def _find_exiftool(requested: str) -> str:
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    located = shutil.which(requested)
    if located:
        return located
    raise FileNotFoundError(
        f"ExifTool executable not found: {requested!r}. Install ExifTool or pass --exiftool PATH."
    )


def _run_exiftool(executable: str, paths: Sequence[Path], chunk_size: int = 128) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for offset in range(0, len(paths), chunk_size):
        chunk = paths[offset : offset + chunk_size]
        command = [executable, "-j", "-G1", "-a", "-u", "-s", "-n", *map(str, chunk)]
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ExifTool failed with exit code {completed.returncode}: {message[:1000]}")
        try:
            records = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ExifTool returned invalid JSON: {exc}") from exc
        if len(records) != len(chunk):
            raise RuntimeError(f"ExifTool returned {len(records)} records for {len(chunk)} files")
        for requested_path, metadata in zip(chunk, records):
            source = metadata.get("SourceFile")
            if source is None:
                source, _ = _first_tag(_tag_index(metadata), ("SourceFile",))
            source_path = _resolve_path(str(source)) if source else requested_path.resolve()
            result[os.path.normcase(str(source_path))] = metadata
            # ExifTool may normalize path spelling differently; preserve an order-safe alias.
            result.setdefault(os.path.normcase(str(requested_path.resolve())), metadata)
    return result


def _selected_metadata(metadata: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    index = _tag_index(metadata)
    fields: Dict[str, Sequence[str]] = {
        "camera_model": ("Model", "CameraModelName"),
        "camera_serial": ("SerialNumber", "CameraSerialNumber"),
        "image_width": ("ImageWidth", "ExifImageWidth"),
        "image_height": ("ImageHeight", "ExifImageHeight"),
        "capture_time": ("SubSecDateTimeOriginal", "DateTimeOriginal", "CreateDate"),
        "gps_latitude": ("GPSLatitude",),
        "gps_longitude": ("GPSLongitude",),
        "gps_altitude_m": ("GPSAltitude", "AbsoluteAltitude"),
        "relative_altitude_m": ("RelativeAltitude",),
        "gimbal_pitch_deg": ("GimbalPitchDegree", "GimbalPitch"),
        "gimbal_yaw_deg": ("GimbalYawDegree", "GimbalYaw"),
        "gimbal_roll_deg": ("GimbalRollDegree", "GimbalRoll"),
        "lrf_status": ("LRFStatus", "LaserRangingStatus", "LaserRangeFinderStatus"),
        "lrf_distance_m": (
            "LRFTargetDistance",
            "LaserRangingDistance",
            "LaserRangeFinderDistance",
            "LaserDistance",
        ),
        "embedded_distance_m": ("ObjectDistance", "SubjectDistance", "ThermalObjectDistance"),
        "embedded_humidity_percent": ("RelativeHumidity", "Humidity", "ThermalHumidity"),
        "embedded_emissivity": ("Emissivity", "ThermalEmissivity"),
        "embedded_ambient_c": ("AmbientTemperature", "AtmosphericTemperature"),
        "embedded_reflected_c": ("ReflectedApparentTemperature", "ReflectedTemperature"),
        "native_palette": ("ThermalPalette", "Palette", "ColorPalette", "PaletteType"),
        "rjpeg_version": ("RJPEGVersion", "ThermalRJPEGVersion"),
        "dirp_version": ("DIRPVersion", "ThermalSDKVersion"),
        "thermal_data": ("ThermalData", "RawThermalImage", "TemperatureData"),
        "thermal_calibration": ("ThermalCalibration", "ThermalCalibrationData"),
    }
    selected: Dict[str, Dict[str, Any]] = {}
    for name, aliases in fields.items():
        value, source_tag = _first_tag(index, aliases)
        selected[name] = {"value": value, "source_tag": source_tag}
    return selected


def _coerce_selected(selected: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    result = {name: item.get("value") for name, item in selected.items()}
    for field in ("image_width", "image_height"):
        result[field] = _as_int(result[field])
    for field in (
        "gps_latitude",
        "gps_longitude",
        "gps_altitude_m",
        "relative_altitude_m",
        "gimbal_pitch_deg",
        "gimbal_yaw_deg",
        "gimbal_roll_deg",
        "lrf_distance_m",
        "embedded_distance_m",
        "embedded_humidity_percent",
        "embedded_emissivity",
        "embedded_ambient_c",
        "embedded_reflected_c",
    ):
        result[field] = _as_float(result[field])
    return result


def build_audit_record(
    item: Mapping[str, Any],
    thermal_metadata: Mapping[str, Any],
    rgb_metadata: Optional[Mapping[str, Any]] = None,
    hash_source: bool = False,
) -> Dict[str, Any]:
    """Build one stable audit record from ExifTool metadata (public for tests)."""
    source_path = Path(str(item["thermal_path"])).resolve()
    thermal_selected = _selected_metadata(thermal_metadata)
    values = _coerce_selected(thermal_selected)
    thermal_marker = values.pop("thermal_data")
    calibration_marker = values.pop("thermal_calibration")
    thermal_time = values.get("capture_time")
    rgb_time = None
    if rgb_metadata:
        rgb_selected = _coerce_selected(_selected_metadata(rgb_metadata))
        rgb_time = rgb_selected.get("capture_time")
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scene": str(item.get("scene") or source_path.parent.name),
        "frame_id": str(item.get("frame_id") or source_path.stem),
        "pair_id": str(item.get("pair_id") or item.get("frame_id") or source_path.stem),
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        **values,
        "rgb_path": str(Path(str(item["rgb_path"])).resolve()) if item.get("rgb_path") else None,
        "rgb_capture_time": rgb_time,
        "rgb_thermal_timestamp_delta_ms": _timestamp_delta_ms(thermal_time, rgb_time),
        "thermal_data_present": thermal_marker is not None,
        "thermal_calibration_present": calibration_marker is not None,
        "rjpeg_detected": thermal_marker is not None and calibration_marker is not None,
        "tsdk_decode_success": None,
        "metadata_sources": {
            name: details.get("source_tag") for name, details in thermal_selected.items()
        },
    }
    lrf_status = str(record["lrf_status"] or "").strip().lower()
    record["lrf_distance_valid"] = bool(
        record["lrf_distance_m"] is not None
        and record["lrf_distance_m"] > 0
        and lrf_status == "normal"
    )
    if hash_source:
        record["source_sha256"] = _sha256(source_path)
    warnings: List[str] = []
    if not record["thermal_data_present"]:
        warnings.append("thermal_data_tag_missing")
    if not record["thermal_calibration_present"]:
        warnings.append("thermal_calibration_tag_missing")
    if record["native_palette"] is None:
        warnings.append("native_palette_unavailable")
    if record["lrf_distance_m"] is None:
        warnings.append("lrf_distance_unavailable")
    elif not record["lrf_distance_valid"]:
        warnings.append("lrf_distance_invalid")
    if item.get("rgb_path") and rgb_time is None:
        warnings.append("rgb_timestamp_unavailable")
    record["warnings"] = warnings
    return record


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _ensure_output_is_not_in_input_tree(output: Path, input_roots: Sequence[Path]) -> None:
    resolved = output.resolve(strict=False)
    for root in input_roots:
        source_root = root.resolve(strict=False) if root.is_dir() else root.resolve(strict=False).parent
        if resolved == source_root or source_root in resolved.parents:
            raise ValueError(
                f"Refusing to write audit output inside source tree {source_root}: {resolved}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", help="R-JPEG files or directories")
    parser.add_argument(
        "--pairs-jsonl",
        type=Path,
        help="Explicit JSONL pairs with thermal_path and optional rgb_path/scene/frame_id/pair_id",
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--exiftool", default="exiftool")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--hash-source", action="store_true", help="Compute full source SHA-256")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(recursive=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.inputs) == bool(args.pairs_jsonl):
        raise SystemExit("Provide either positional inputs or --pairs-jsonl, but not both.")
    items = _read_pair_manifest(args.pairs_jsonl.resolve()) if args.pairs_jsonl else _discover_inputs(args.inputs, args.recursive)
    if not items:
        raise SystemExit("No R-JPEG candidates found.")
    source_roots = [Path(value) for value in args.inputs]
    if args.pairs_jsonl:
        source_roots = [Path(str(item["thermal_path"])).parent for item in items]
    summary_path = args.summary_json or args.output_jsonl.with_suffix(".summary.json")
    _ensure_output_is_not_in_input_tree(args.output_jsonl, source_roots)
    _ensure_output_is_not_in_input_tree(summary_path, source_roots)
    if not args.overwrite:
        existing = [path for path in (args.output_jsonl, summary_path) if path.exists()]
        if existing:
            raise SystemExit(f"Output exists; pass --overwrite: {existing[0]}")

    exiftool = _find_exiftool(args.exiftool)
    all_paths: Dict[str, Path] = {}
    for item in items:
        for field in ("thermal_path", "rgb_path"):
            if not item.get(field):
                continue
            path = Path(str(item[field])).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing {field}: {path}")
            all_paths[os.path.normcase(str(path))] = path
    metadata = _run_exiftool(exiftool, [all_paths[key] for key in sorted(all_paths)])

    records: List[Dict[str, Any]] = []
    for item in items:
        thermal_path = Path(str(item["thermal_path"])).resolve()
        thermal_metadata = metadata.get(os.path.normcase(str(thermal_path)))
        if thermal_metadata is None:
            raise RuntimeError(f"ExifTool metadata missing for {thermal_path}")
        rgb_metadata = None
        if item.get("rgb_path"):
            rgb_path = Path(str(item["rgb_path"])).resolve()
            rgb_metadata = metadata.get(os.path.normcase(str(rgb_path)))
        records.append(build_audit_record(item, thermal_metadata, rgb_metadata, args.hash_source))

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(args.output_jsonl.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.output_jsonl)

    palette_counts = Counter(str(record["native_palette"] or "<unavailable>") for record in records)
    lrf_available = sum(record["lrf_distance_m"] is not None for record in records)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "record_count": len(records),
        "rjpeg_detected_count": sum(bool(record["rjpeg_detected"]) for record in records),
        "lrf_distance_available_count": lrf_available,
        "lrf_distance_valid_count": sum(bool(record["lrf_distance_valid"]) for record in records),
        "rgb_pair_count": sum(record["rgb_path"] is not None for record in records),
        "timestamp_delta_available_count": sum(
            record["rgb_thermal_timestamp_delta_ms"] is not None for record in records
        ),
        "native_palette_counts": dict(sorted(palette_counts.items())),
        "source_hashing": bool(args.hash_source),
        "exiftool": exiftool,
        "output_jsonl": str(args.output_jsonl.resolve()),
    }
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
