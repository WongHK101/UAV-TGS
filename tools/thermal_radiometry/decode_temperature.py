#!/usr/bin/env python3
"""Traceable wrapper for DJI TSDK R-JPEG temperature decoding.

This repository does not vendor DJI libraries.  By default it uses the official
``dji_irp`` utility found below ``--tsdk-root``/``DJI_TSDK_ROOT``.  A custom
Python adapter may instead be supplied with ``--adapter``; its contract is::

    def decode_rjpeg(*, input_path: Path, output_path: Path,
                     tsdk_root: Path, parameters: dict) -> dict | None:
        # Decode ``input_path`` and write a 2-D float32 Celsius NPY.

Every adapter must treat ``input_path`` as read-only, use exactly the supplied
radiometry parameters, and write the requested ``output_path``.  It may return
JSON-serializable SDK/version diagnostics.  The adapter is selected with
``--adapter path/to/adapter.py:decode_rjpeg`` or ``--adapter module:function``.
The built-in official-utility adapter is ``builtin:dji_irp``.  The proprietary
SDK is located only through ``--tsdk-root`` or ``DJI_TSDK_ROOT``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "uav-tgs.temperature-decode.v1"
PARAMETER_NAMES = (
    "distance_m",
    "humidity_percent",
    "emissivity",
    "ambient_c",
    "reflected_c",
)


class DecodeConfigurationError(ValueError):
    """Raised before invoking an adapter when the protocol is underspecified."""


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve_tsdk_root(cli_value: Optional[Path], environ: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if environ is None else environ
    raw_value = str(cli_value) if cli_value is not None else env.get("DJI_TSDK_ROOT")
    if not raw_value:
        raise DecodeConfigurationError(
            "DJI TSDK root is not configured. Pass --tsdk-root PATH or set DJI_TSDK_ROOT."
        )
    root = Path(raw_value).expanduser().resolve(strict=False)
    if not root.is_dir():
        raise DecodeConfigurationError(f"DJI TSDK root does not exist or is not a directory: {root}")
    return root


def _split_adapter_spec(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise DecodeConfigurationError(
            "--adapter must be PATH.py:function or importable.module:function"
        )
    module_spec, function_name = spec.rsplit(":", 1)
    if not module_spec or not function_name:
        raise DecodeConfigurationError(
            "--adapter must be PATH.py:function or importable.module:function"
        )
    return module_spec, function_name


def load_adapter(spec: str, tsdk_root: Path) -> Callable[..., Any]:
    """Load the explicit repository-external adapter without assuming a DJI ABI."""
    if spec == "builtin:dji_irp":
        try:
            from dji_irp_adapter import decode_rjpeg
        except ImportError:  # pragma: no cover - module execution path
            from .dji_irp_adapter import decode_rjpeg
        return decode_rjpeg
    module_spec, function_name = _split_adapter_spec(spec)
    candidate = Path(module_spec).expanduser()
    if candidate.suffix.lower() == ".py" or candidate.is_file():
        candidate = candidate.resolve(strict=False)
        if not candidate.is_file():
            raise DecodeConfigurationError(f"Adapter file does not exist: {candidate}")
        module_name = f"uav_tgs_tsdk_adapter_{hashlib.sha256(str(candidate).encode()).hexdigest()[:12]}"
        import_spec = importlib.util.spec_from_file_location(module_name, candidate)
        if import_spec is None or import_spec.loader is None:
            raise DecodeConfigurationError(f"Unable to load adapter file: {candidate}")
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[module_name] = module
        import_spec.loader.exec_module(module)
    else:
        root_text = str(tsdk_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        try:
            module = importlib.import_module(module_spec)
        except ImportError as exc:
            raise DecodeConfigurationError(f"Unable to import adapter module {module_spec!r}: {exc}") from exc
    function = getattr(module, function_name, None)
    if not callable(function):
        raise DecodeConfigurationError(f"Adapter function is not callable: {spec}")
    return function


def _strict_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise TypeError("Adapter diagnostics contain a non-finite number")
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    raise TypeError(f"Adapter diagnostics are not JSON serializable: {type(value).__name__}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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
                raise DecodeConfigurationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            source = record.get("source_path", record.get("thermal_path"))
            if not source:
                raise DecodeConfigurationError(
                    f"Missing source_path/thermal_path at {path}:{line_number}"
                )
            source_path = Path(str(source)).expanduser()
            if not source_path.is_absolute():
                source_path = base / source_path
            item = dict(record)
            item["source_path"] = str(source_path.resolve(strict=False))
            records.append(item)
    return records


def _discover_inputs(inputs: Sequence[str], recursive: bool, scene: Optional[str]) -> List[Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for value in inputs:
        path = Path(value).expanduser().resolve(strict=False)
        if path.is_file():
            candidates: Iterable[Path] = (path,)
            root = path.parent
        elif path.is_dir():
            candidates = path.rglob("*") if recursive else path.glob("*")
            root = path
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in (".jpg", ".jpeg", ".rjpeg"):
                continue
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            relative = resolved.relative_to(root) if resolved.is_relative_to(root) else Path(resolved.name)
            inferred_scene = scene or (relative.parts[0] if len(relative.parts) > 1 else root.name)
            records[key] = {
                "source_path": str(resolved),
                "scene": inferred_scene,
                "frame_id": resolved.stem,
                "pair_id": resolved.stem,
            }
    return [records[key] for key in sorted(records)]


def _normalize_parameter(name: str, raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        value = raw.get("value")
        source = raw.get("source")
    else:
        value = raw
        source = None
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DecodeConfigurationError(f"Radiometry parameter {name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise DecodeConfigurationError(f"Radiometry parameter {name} must be finite")
    if name == "distance_m" and number <= 0:
        raise DecodeConfigurationError("distance_m must be > 0")
    if name == "humidity_percent" and not 0 <= number <= 100:
        raise DecodeConfigurationError("humidity_percent must be in [0, 100]")
    if name == "emissivity" and not 0 < number <= 1:
        raise DecodeConfigurationError("emissivity must be in (0, 1]")
    return {"value": number, "source": str(source or "manifest_unspecified")}


def resolve_parameters(record: Mapping[str, Any], global_parameters: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Resolve per-frame parameters, using explicit globals only as fallbacks."""
    raw_parameters = record.get("decode_parameters", record.get("radiometry_parameters", {}))
    if raw_parameters is None:
        raw_parameters = {}
    if not isinstance(raw_parameters, Mapping):
        raise DecodeConfigurationError("decode_parameters must be a JSON object")
    resolved: Dict[str, Dict[str, Any]] = {}
    for name in PARAMETER_NAMES:
        item = _normalize_parameter(name, raw_parameters.get(name))
        if item is None:
            item = _normalize_parameter(name, global_parameters.get(name))
        if item is not None:
            resolved[name] = item
    missing = [name for name in PARAMETER_NAMES if name not in resolved]
    if missing:
        identifier = record.get("pair_id", record.get("source_path", "<unknown>"))
        raise DecodeConfigurationError(
            f"Missing radiometry parameters for {identifier}: {', '.join(missing)}. "
            "Provide them in decode_parameters or as explicit CLI fallbacks."
        )
    return resolved


def _safe_component(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or fallback)).strip("._")
    return text or fallback


def _assert_output_outside_sources(output_root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    resolved_output = output_root.resolve(strict=False)
    for record in records:
        source_parent = Path(str(record["source_path"])).resolve(strict=False).parent
        if resolved_output == source_parent or source_parent in resolved_output.parents:
            raise DecodeConfigurationError(
                f"Refusing to write derived output inside source directory {source_parent}: {resolved_output}"
            )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _validate_temperature_map(
    path: Path, expected_width: Optional[int], expected_height: Optional[int]
) -> Tuple[Tuple[int, int], float, float]:
    if not path.is_file():
        raise RuntimeError(f"Adapter did not create the requested NPY: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.dtype("float32"):
        raise RuntimeError(f"Adapter output must be float32, got {array.dtype}: {path}")
    if array.ndim != 2:
        raise RuntimeError(f"Adapter output must be a 2-D Celsius map, got shape {array.shape}: {path}")
    height, width = map(int, array.shape)
    if expected_width is not None and width != expected_width:
        raise RuntimeError(f"Expected width {expected_width}, got {width}: {path}")
    if expected_height is not None and height != expected_height:
        raise RuntimeError(f"Expected height {expected_height}, got {height}: {path}")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Adapter output contains NaN or infinity: {path}")
    return (height, width), float(array.min()), float(array.max())


def _parameter_cli_fallbacks(args: argparse.Namespace) -> Dict[str, Optional[Dict[str, Any]]]:
    values: Dict[str, Optional[Dict[str, Any]]] = {}
    for name in PARAMETER_NAMES:
        value = getattr(args, name)
        if value is None:
            values[name] = None
            continue
        source = getattr(args, f"{name}_source") or "global_cli"
        values[name] = {"value": value, "source": source}
    if values["emissivity"] is None:
        values["emissivity"] = {"value": 0.95, "source": "benchmark_assumption"}
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="*", help="R-JPEG files/directories (alternative to --input-manifest)")
    parser.add_argument("--input-manifest", type=Path, help="Audit/protocol JSONL with source_path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Derived protocol root")
    parser.add_argument("--manifest-out", type=Path, help="Default: OUTPUT/manifests/decode_manifest.jsonl")
    parser.add_argument("--tsdk-root", type=Path, help="Overrides DJI_TSDK_ROOT")
    parser.add_argument(
        "--adapter",
        default="builtin:dji_irp",
        help="Decoder adapter (default: builtin:dji_irp; or PATH.py:function/module:function)",
    )
    parser.add_argument("--scene", help="Scene label for positional inputs")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--probe", action="store_true", help="Decode exactly the first frame as an SDK compatibility probe")
    parser.add_argument("--keep-going", action="store_true", help="Record failures and continue batch decoding")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--no-source-sha256", dest="hash_source", action="store_false")
    for name, help_text in (
        ("distance_m", "Object distance in metres"),
        ("humidity_percent", "Relative humidity in percent"),
        ("emissivity", "Emissivity; defaults to explicit benchmark assumption 0.95"),
        ("ambient_c", "Ambient/atmospheric temperature in Celsius"),
        ("reflected_c", "Reflected apparent temperature in Celsius"),
    ):
        option = "--" + name.replace("_", "-")
        parser.add_argument(option, dest=name, type=float, help=help_text)
        parser.add_argument(option + "-source", dest=f"{name}_source", help=f"Provenance label for {name}")
    parser.set_defaults(recursive=True, hash_source=True)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if bool(args.inputs) == bool(args.input_manifest):
        raise DecodeConfigurationError(
            "Provide either positional inputs or --input-manifest, but not both."
        )
    records = _read_jsonl(args.input_manifest.resolve()) if args.input_manifest else _discover_inputs(args.inputs, args.recursive, args.scene)
    if not records:
        raise DecodeConfigurationError("No R-JPEG candidates found.")
    if args.probe:
        records = records[:1]
    for record in records:
        source = Path(str(record["source_path"]))
        if not source.is_file():
            raise FileNotFoundError(f"Source R-JPEG does not exist: {source}")

    tsdk_root = resolve_tsdk_root(args.tsdk_root)
    os.environ["DJI_TSDK_ROOT"] = str(tsdk_root)
    dll_handle = None
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        # This is only a loader search path; no SDK filenames or proprietary ABI are assumed.
        dll_handle = os.add_dll_directory(str(tsdk_root))
    adapter = load_adapter(args.adapter, tsdk_root)
    output_root = args.output_dir.resolve(strict=False)
    _assert_output_outside_sources(output_root, records)
    manifest_path = (args.manifest_out or output_root / "manifests" / "decode_manifest.jsonl").resolve(strict=False)
    _assert_output_outside_sources(manifest_path.parent, records)
    if manifest_path.exists() and not args.overwrite:
        raise DecodeConfigurationError(f"Manifest exists; pass --overwrite: {manifest_path}")

    global_parameters = _parameter_cli_fallbacks(args)
    seen_outputs: set[str] = set()
    output_records: List[Dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    try:
        for record in records:
            source = Path(str(record["source_path"])).resolve()
            scene = _safe_component(record.get("scene"), source.parent.name or "scene")
            frame_id = _safe_component(record.get("frame_id") or record.get("pair_id"), source.stem)
            output_path = output_root / "temperature_c" / scene / f"{frame_id}.npy"
            normalized_output = os.path.normcase(str(output_path.resolve(strict=False)))
            if normalized_output in seen_outputs:
                raise DecodeConfigurationError(f"Duplicate derived output path: {output_path}")
            seen_outputs.add(normalized_output)
            if output_path.exists() and not args.overwrite:
                raise DecodeConfigurationError(f"Temperature output exists; pass --overwrite: {output_path}")
            parameters = resolve_parameters(record, global_parameters)
            input_stat = source.stat()
            request = {
                "schema_version": SCHEMA_VERSION,
                "scene": scene,
                "frame_id": frame_id,
                "pair_id": str(record.get("pair_id") or frame_id),
                "source_path": str(source),
                "output_path": str(output_path.resolve(strict=False)),
                "tsdk_root": str(tsdk_root),
                "adapter": args.adapter,
                "parameters": parameters,
            }
            request_path = output_root / "manifests" / "decode_requests" / f"{scene}--{frame_id}.json"
            if request_path.exists() and not args.overwrite:
                raise DecodeConfigurationError(f"Decode request exists; pass --overwrite: {request_path}")
            _write_json(request_path, request)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            partial = output_path.with_name(f".{output_path.stem}.partial.npy")
            if partial.exists():
                partial.unlink()
            result_record: Dict[str, Any] = dict(request)
            result_record["request_path"] = str(request_path.resolve())
            try:
                source_sha256 = _sha256(source) if args.hash_source else None
                started = time.perf_counter()
                diagnostics = adapter(
                    input_path=source,
                    output_path=partial,
                    tsdk_root=tsdk_root,
                    parameters={name: dict(value) for name, value in parameters.items()},
                )
                elapsed = time.perf_counter() - started
                if diagnostics is not None and not isinstance(diagnostics, Mapping):
                    raise RuntimeError("Adapter return value must be a mapping or None")
                shape, minimum, maximum = _validate_temperature_map(
                    partial, args.expected_width, args.expected_height
                )
                after_stat = source.stat()
                if (input_stat.st_size, input_stat.st_mtime_ns) != (
                    after_stat.st_size,
                    after_stat.st_mtime_ns,
                ):
                    raise RuntimeError(f"Source file changed during decode: {source}")
                os.replace(partial, output_path)
                result_record.update(
                    {
                        "success": True,
                        "shape_hw": list(shape),
                        "dtype": "float32",
                        "temperature_min_c": minimum,
                        "temperature_max_c": maximum,
                        "source_size_bytes": input_stat.st_size,
                        "source_mtime_ns": input_stat.st_mtime_ns,
                        "source_sha256": source_sha256,
                        "output_sha256": _sha256(output_path),
                        "adapter_diagnostics": _strict_json_value(diagnostics or {}),
                        "elapsed_seconds": elapsed,
                    }
                )
                success_count += 1
            except Exception as exc:  # Preserve a per-frame failure record before fail-fast.
                if partial.exists():
                    partial.unlink()
                result_record.update(
                    {
                        "success": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                failure_count += 1
                output_records.append(result_record)
                if not args.keep_going:
                    break
                continue
            output_records.append(result_record)
    finally:
        if dll_handle is not None:
            dll_handle.close()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary_manifest, manifest_path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "probe": bool(args.probe),
        "requested_count": len(records),
        "success_count": success_count,
        "failure_count": failure_count,
        "manifest_path": str(manifest_path),
        "tsdk_root": str(tsdk_root),
        "adapter": args.adapter,
    }
    _write_json(manifest_path.with_suffix(".summary.json"), summary)
    if failure_count:
        first_failure = next(record for record in output_records if not record["success"])
        raise RuntimeError(
            f"Temperature decode failed for {failure_count} frame(s); first: "
            f"{first_failure.get('source_path')}: {first_failure.get('error')}"
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DecodeConfigurationError, FileNotFoundError, RuntimeError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
