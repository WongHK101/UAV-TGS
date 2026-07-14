#!/usr/bin/env python3
"""Adapter for the official ``dji_irp`` utility shipped with DJI TSDK.

No SDK library or executable is vendored by this repository.  The caller
passes a local SDK root and this adapter invokes the documented ``measure``
action with ``--measurefmt float32``.  The raw output is converted to a
two-dimensional float32 Celsius NPY after checking the resolution printed by
the utility.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_dji_irp(tsdk_root: Path) -> Path:
    root = Path(tsdk_root).resolve()
    override = os.environ.get("DJI_TSDK_DJI_IRP")
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    if os.name == "nt":
        candidates.extend(
            [
                root / "utility" / "bin" / "windows" / "release_x64" / "dji_irp.exe",
                root / "utility" / "bin" / "windows" / "release_x86" / "dji_irp.exe",
            ]
        )
    else:
        candidates.extend(
            [
                root / "utility" / "bin" / "linux" / "release_x64" / "dji_irp",
                root / "utility" / "bin" / "linux" / "release_x86" / "dji_irp",
            ]
        )
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved.is_file():
            return resolved
    searched = "\n  ".join(str(item.resolve(strict=False)) for item in candidates)
    raise FileNotFoundError(f"Official dji_irp utility was not found. Searched:\n  {searched}")


def _value(parameters: Mapping[str, Any], name: str) -> float:
    item = parameters[name]
    raw = item.get("value") if isinstance(item, Mapping) else item
    return float(raw)


def _parse_resolution(stdout: str) -> Tuple[int, int]:
    width_match = re.search(r"image\s+width\s*:\s*(\d+)", stdout, flags=re.IGNORECASE)
    height_match = re.search(r"image\s+height\s*:\s*(\d+)", stdout, flags=re.IGNORECASE)
    if width_match is None or height_match is None:
        raise RuntimeError("dji_irp output did not report an R-JPEG resolution")
    return int(width_match.group(1)), int(height_match.group(1))


def _parse_parameter_ranges(stdout: str) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    aliases = {
        "distance": "distance_m",
        "humidity": "humidity_percent",
        "emissivity": "emissivity",
        "ambienttemp": "ambient_c",
        "ambient_temp": "ambient_c",
        "reflection": "reflected_c",
    }
    pattern = re.compile(
        r"^\s*([A-Za-z_]+)\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(stdout):
        normalized = match.group(1).lower()
        name = aliases.get(normalized)
        if name:
            result[name] = {"min": float(match.group(2)), "max": float(match.group(3))}
    return result


def decode_rjpeg(
    *,
    input_path: Path,
    output_path: Path,
    tsdk_root: Path,
    parameters: Mapping[str, Any],
) -> Dict[str, Any]:
    executable = resolve_dji_irp(tsdk_root)
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_path = destination.with_name(destination.name + ".dji_irp_float32.raw")
    if raw_path.exists():
        raw_path.unlink()
    command = [
        str(executable),
        "-s",
        str(source),
        "-a",
        "measure",
        "-o",
        str(raw_path),
        "--measurefmt",
        "float32",
        "--distance",
        format(_value(parameters, "distance_m"), ".9g"),
        "--humidity",
        format(_value(parameters, "humidity_percent"), ".9g"),
        "--emissivity",
        format(_value(parameters, "emissivity"), ".9g"),
        "--ambient",
        format(_value(parameters, "ambient_c"), ".9g"),
        "--reflection",
        format(_value(parameters, "reflected_c"), ".9g"),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(executable.parent),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        output_text = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        parameter_ranges = _parse_parameter_ranges(output_text)
        if completed.returncode != 0 or not raw_path.is_file():
            excerpt = output_text.strip()[-4000:]
            raise RuntimeError(
                f"dji_irp measure failed with exit code {completed.returncode}:\n{excerpt}"
            )
        width, height = _parse_resolution(output_text)
        expected_bytes = width * height * np.dtype("<f4").itemsize
        actual_bytes = raw_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"dji_irp raw size mismatch: expected {expected_bytes} bytes for "
                f"{width}x{height}, got {actual_bytes}"
            )
        temperature = np.fromfile(raw_path, dtype="<f4").reshape(height, width)
        np.save(destination, temperature.astype(np.float32, copy=False), allow_pickle=False)
        version_match = re.search(r"DIRP API version number\s*:\s*([^\r\n]+)", output_text)
        rjpeg_match = re.search(r"R-JPEG version\s*:\s*([^\r\n]+)", output_text)
        return {
            "backend": "official-dji-irp",
            "executable": str(executable),
            "executable_sha256": _sha256(executable),
            "command": command,
            "resolution": {"width": width, "height": height},
            "dirp_api_version": version_match.group(1).strip() if version_match else None,
            "rjpeg_version": rjpeg_match.group(1).strip() if rjpeg_match else None,
            "measurement_parameter_ranges": parameter_ranges,
            "parameters_applied": {
                name: _value(parameters, name)
                for name in (
                    "distance_m",
                    "humidity_percent",
                    "emissivity",
                    "ambient_c",
                    "reflected_c",
                )
            },
            "stdout_tail": completed.stdout[-4000:],
        }
    finally:
        if raw_path.exists():
            raw_path.unlink()
