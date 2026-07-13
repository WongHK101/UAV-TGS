#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_uavfgs.py

Prepare COLMAP- and 3DGS-compatible inputs for UAV-FGS.
This script converts standardized RGB-T data into the camera, database, and
layout artifacts expected by the downstream reconstruction pipeline and exports
pose-prior records for GPS-enabled runs.
"""

import argparse
import hashlib
import json
import os
import math
import re
import signal
import sqlite3
import struct
import subprocess
import sys
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import numpy as np


COMPLETION_MANIFEST_SCHEMA_VERSION = 1
COMPLETION_MANIFEST_NAME = "conversion_completion_manifest.json"
DATABASE_PROVENANCE_SCHEMA_VERSION = 1
DATABASE_PROVENANCE_NAME = "database_provenance.json"
TRANSACTION_JOURNAL_NAME = "transaction.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
_OWNED_TEMP_PATHS: List[Path] = []
GLOBAL_MAPPER_FORBIDDEN_OUTPUT_PATTERNS = [
    r"\bTermination:\s*(?:FAILURE|USER_FAILURE)\b",
    r"Number of consecutive invalid steps",
    r"Ceres was compiled without (?:CUDA|cuDSS) support",
    r"Falling back to CPU-based",
    r"(?:CUDA|cuDSS).*\b(?:falling back|fallback)\b.*\b(?:CPU|Eigen)\b",
    r"\b(?:falling back|fallback)\b.*(?:CUDA|cuDSS).*\b(?:CPU|Eigen)\b",
    r"\b(?:Using|Switching to)\s+(?:an?\s+)?CPU-based\s+(?:linear\s+)?solver\b",
]
ENU_GEOMETRY_SCALE_BASIS = "centered RMS radius over ENU inliers"
ENU_GEOMETRY_SCALE_RATIO_MIN = 0.5
ENU_GEOMETRY_SCALE_RATIO_MAX = 2.0


def log_info(msg: str) -> None:
    print(f"INFO: {msg}", flush=True)


def log_warn(msg: str) -> None:
    print(f"WARNING: {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"ERROR: {msg}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows virus scanners/indexers can briefly open the old
            # journal between writes. Keep the operation atomic and retry the
            # same replace rather than falling back to an in-place write.
            if os.name != "nt" or attempt == 9:
                raise
            time.sleep(0.02 * (attempt + 1))


def build_file_inventory(root: Path, *, image_files_only: bool = False) -> Dict[str, Any]:
    """Return a content-addressed, relative-path inventory for a directory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Inventory root is missing or not a directory: {root}")
    entries: List[Dict[str, Any]] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        if image_files_only and path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        entries.append(
            {
                "name": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    names = [str(entry["name"]) for entry in entries]
    return {
        "count": len(entries),
        "names": names,
        "entries_sha256": _canonical_sha256(entries),
        "entries": entries,
    }


def _resolve_executable(exe: str) -> str:
    exe_expanded = os.path.expandvars(exe)
    if os.path.isabs(exe_expanded) and os.path.exists(exe_expanded):
        return exe_expanded

    candidates = [exe_expanded]
    if os.name == "nt":
        base = exe_expanded
        if not base.lower().endswith((".exe", ".cmd", ".bat")):
            candidates = [base, base + ".exe", base + ".cmd", base + ".bat"]

    for v in candidates:
        p = shutil.which(v)
        if p:
            return p
    return exe_expanded


def _should_use_shell(resolved_exe: str) -> bool:
    if os.name != "nt":
        return False
    low = resolved_exe.lower()
    return low.endswith(".bat") or low.endswith(".cmd")


def _terminate_process_group(proc: subprocess.Popen, timeout_s: float = 5.0) -> None:
    """Terminate a protocol-critical command and every child it spawned."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(1.0, timeout_s),
            )
        except Exception:
            proc.kill()
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    deadline = time.monotonic() + max(0.1, timeout_s)
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            proc.kill()


def run_cmd(cmd_list, cwd=None, forbidden_output_patterns: Optional[List[str]] = None):
    if not cmd_list:
        raise ValueError("Empty command list")

    resolved0 = _resolve_executable(cmd_list[0])
    use_shell = _should_use_shell(resolved0)

    cmd = [resolved0] + cmd_list[1:]
    if use_shell:
        invocation = " ".join(
            [f'"{x}"' if (" " in str(x)) else str(x) for x in cmd]
        )
        log_info("Running (shell): " + invocation)
    else:
        invocation = cmd
        log_info("Running: " + " ".join([str(x) for x in cmd]))

    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (forbidden_output_patterns or [])]
    if not patterns:
        subprocess.run(invocation, cwd=cwd, check=True, shell=use_shell)
        return

    # Some COLMAP commands can return zero even when an internal Ceres solve
    # terminates with FAILURE. Stream output to the parent log while enforcing
    # fail-closed diagnostics for protocol-critical commands.
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        invocation,
        cwd=cwd,
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        **popen_kwargs,
    )
    matched_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if any(pattern.search(line) for pattern in patterns):
            matched_lines.append(line.strip())
            # A CUDA/Ceres fallback is a protocol violation. Stop immediately
            # instead of allowing an expensive CPU solve to finish first.
            _terminate_process_group(proc)
            try:
                remainder, _ = proc.communicate(timeout=5)
                if remainder:
                    print(remainder, end="", flush=True)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc, timeout_s=0.1)
                proc.communicate()
            unique_matches = list(dict.fromkeys(matched_lines))
            raise RuntimeError(
                "Command emitted a forbidden solver/GPU fallback diagnostic and was terminated:\n"
                + "\n".join(unique_matches[:20])
            )
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, invocation)


def split_args(s: str):
    return s.strip().split() if s else []


def option_value(tokens: List[str], name: str, default=None):
    """Return the last value supplied for a ``--name`` style CLI option."""
    value = default
    for idx, token in enumerate(tokens):
        if token.startswith(name + "="):
            value = token.split("=", 1)[1]
        elif token == name and idx + 1 < len(tokens):
            value = tokens[idx + 1]
    return value


def reject_cli_overrides(tokens: List[str], reserved: List[str], source: str) -> None:
    """Reject exact or Boost.Program_options-guessed prefixes of locked options."""
    for token in tokens:
        token_name = token.split("=", 1)[0]
        if not token_name.startswith("--"):
            continue
        for option in reserved:
            # COLMAP's Boost parser accepts a unique long-option prefix (for
            # example --project_pat as --project_path).  Conservatively reject
            # every prefix of a locked option, including ambiguous ones.
            if option.startswith(token_name):
                raise ValueError(
                    f"{source} must not override protocol-locked option {option}; "
                    "abbreviated long options are also forbidden; use the dedicated "
                    "command-line option instead."
                )


def reject_cli_abbreviations(tokens: List[str], controlled: List[str], source: str) -> None:
    """Require full spellings for controlled options that may otherwise be normalized."""
    for token in tokens:
        token_name = token.split("=", 1)[0]
        if not token_name.startswith("--"):
            continue
        for option in controlled:
            if token_name != option and option.startswith(token_name):
                raise ValueError(
                    f"{source} must spell controlled option {option} in full; "
                    f"abbreviated option {token_name} is forbidden."
                )


def remove_cli_options(tokens: List[str], names: List[str], source: str) -> List[str]:
    """Remove parsed scalar options so normalized locked commands contain no duplicates."""
    output: List[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        matched = next((name for name in names if token == name or token.startswith(name + "=")), None)
        if matched is None:
            output.append(token)
            index += 1
            continue
        if token == matched:
            if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                raise ValueError(f"{source} option {matched} requires a value")
            index += 2
        else:
            index += 1
    return output


def verify_colmap_runtime(
    colmap_exe: str,
    required_version: str,
    require_cuda: bool,
    *,
    required_sha256: str = "",
    require_cudss: bool = False,
) -> Dict[str, Any]:
    """Fail closed unless the selected runtime satisfies the formal GPU contract."""
    resolved = _resolve_executable(colmap_exe)
    resolved_path = Path(resolved)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"COLMAP executable is missing: {resolved}")
    executable_sha256 = _sha256_file(resolved_path)
    required_sha256 = str(required_sha256 or "").strip().lower()
    if required_sha256:
        if not re.fullmatch(r"[0-9a-f]{64}", required_sha256):
            raise ValueError("--required_colmap_sha256 must be exactly 64 lowercase/uppercase hex characters")
        if executable_sha256 != required_sha256:
            raise RuntimeError(
                "COLMAP executable SHA256 mismatch: "
                f"expected={required_sha256} actual={executable_sha256} path={resolved_path}"
            )
    proc = subprocess.run(
        [resolved, "-h"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    text = "\n".join(part for part in (proc.stdout, proc.stderr) if part)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to query COLMAP executable {resolved!r} (exit={proc.returncode}).\n{text}"
        )
    if required_version and f"COLMAP {required_version}" not in text:
        raise RuntimeError(
            f"Expected COLMAP {required_version}, but {resolved!r} reported:\n{text[:1000]}"
        )
    if require_cuda and "with CUDA" not in text:
        raise RuntimeError(
            f"CUDA-enabled COLMAP is required, but {resolved!r} did not report 'with CUDA'.\n"
            f"{text[:1000]}"
        )

    ldd_checked = False
    ldd_has_cudss = False
    ldd_output_sha256 = ""
    if require_cudss and sys.platform.startswith("linux"):
        ldd_checked = True
        ldd_proc = subprocess.run(
            ["ldd", resolved],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            check=False,
        )
        ldd_text = ldd_proc.stdout or ""
        ldd_output_sha256 = hashlib.sha256(ldd_text.encode("utf-8", errors="replace")).hexdigest()
        if ldd_proc.returncode != 0:
            raise RuntimeError(f"ldd failed for COLMAP runtime {resolved!r}:\n{ldd_text[:2000]}")
        missing_lines = [line.strip() for line in ldd_text.splitlines() if "not found" in line.lower()]
        if missing_lines:
            raise RuntimeError(
                f"COLMAP runtime has unresolved shared libraries: {resolved}\n"
                + "\n".join(missing_lines[:20])
            )
        ldd_has_cudss = re.search(r"\blibcudss\.so(?:\.|\s|$)", ldd_text, re.IGNORECASE) is not None
        if not ldd_has_cudss:
            raise RuntimeError(
                "GlobalMapper formal runs require a COLMAP/Ceres runtime linked to libcudss on Linux, "
                f"but ldd did not report libcudss: {resolved}"
            )
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    log_info(
        f"COLMAP runtime verified: {first_line} | sha256={executable_sha256}"
        + (" | ldd_libcudss=present" if ldd_checked else "")
    )
    return {
        "resolved_path": str(resolved_path.resolve()),
        "executable_sha256": executable_sha256,
        "required_sha256": required_sha256,
        "version_help_first_line": first_line,
        "version_help_sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        "reported_with_cuda": "with CUDA" in text,
        "ldd_checked": ldd_checked,
        "ldd_has_cudss": ldd_has_cudss,
        "ldd_output_sha256": ldd_output_sha256,
    }


def build_global_mapper_command(
    colmap_exe: str,
    database_path: Path,
    image_path: Path,
    output_path: Path,
    gpu_index: int,
    random_seed: int,
    extra_args: str = "",
) -> List[str]:
    """Build the protocol-locked COLMAP 4.1 global SfM command."""
    extra_tokens = split_args(extra_args)
    reject_cli_overrides(
        extra_tokens,
        [
            "--project_path",
            "--database_path",
            "--image_path",
            "--output_path",
            "--GlobalMapper.gp_use_gpu",
            "--GlobalMapper.gp_gpu_index",
            "--GlobalMapper.ba_ceres_use_gpu",
            "--GlobalMapper.ba_ceres_gpu_index",
            "--GlobalMapper.random_seed",
        ],
        "--global_mapper_args",
    )
    gpu = str(gpu_index)
    return [
        colmap_exe,
        "global_mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_path),
        "--output_path",
        str(output_path),
        "--GlobalMapper.gp_use_gpu",
        "1",
        "--GlobalMapper.gp_gpu_index",
        gpu,
        "--GlobalMapper.ba_ceres_use_gpu",
        "1",
        "--GlobalMapper.ba_ceres_gpu_index",
        gpu,
        "--GlobalMapper.random_seed",
        str(random_seed),
    ] + extra_tokens


def build_matcher_command(
    colmap_exe: str,
    matching: str,
    database_path: Path,
    gpu_index: int,
    extra_args: str = "",
) -> List[str]:
    """Build a GPU-locked matcher command without COLMAP project-file bypasses."""
    matcher_tokens = split_args(extra_args)
    reject_cli_overrides(
        matcher_tokens,
        [
            "--project_path",
            "--database_path",
            "--FeatureMatching.use_gpu",
            "--FeatureMatching.gpu_index",
        ],
        "--matcher_args",
    )
    subcommands = {
        "spatial": "spatial_matcher",
        "exhaustive": "exhaustive_matcher",
        "sequential": "sequential_matcher",
        "vocab_tree": "vocab_tree_matcher",
    }
    if matching not in subcommands:
        raise ValueError(f"Unsupported matching mode: {matching!r}")
    return [
        colmap_exe,
        subcommands[matching],
        "--database_path",
        str(database_path),
        "--FeatureMatching.use_gpu",
        "1",
        "--FeatureMatching.gpu_index",
        str(gpu_index),
    ] + matcher_tokens


def build_model_aligner_command(
    colmap_exe: str,
    input_path: Path,
    output_path: Path,
    database_path: Path,
    transform_path: Path,
    extra_args: str,
) -> Tuple[List[str], float, int]:
    """Build the GPS->ENU command with deterministic, non-overridable controls."""
    user_tokens = split_args(extra_args)
    reject_cli_overrides(
        user_tokens,
        [
            "--project_path",
            "--input_path",
            "--output_path",
            "--database_path",
            "--ref_images_path",
            "--ref_model_path",
            "--transform_path",
            "--default_random_seed",
        ],
        "--model_aligner_args",
    )
    controlled_options = [
        "--ref_is_gps",
        "--alignment_type",
        "--alignment_max_error",
        "--min_common_images",
    ]
    reject_cli_abbreviations(user_tokens, controlled_options, "--model_aligner_args")
    ref_is_gps = str(option_value(user_tokens, "--ref_is_gps", "1")).strip().lower()
    alignment_type = str(option_value(user_tokens, "--alignment_type", "enu")).strip().lower()
    if ref_is_gps not in ("1", "true") or alignment_type != "enu":
        raise ValueError(
            "UAV-FGS model alignment is protocol-locked to image-embedded WGS84-like values -> local ENU; "
            "--model_aligner_args must retain --ref_is_gps=1 --alignment_type=enu."
        )
    alignment_max_error = float(option_value(user_tokens, "--alignment_max_error", "30.0"))
    min_common_images = int(option_value(user_tokens, "--min_common_images", "3"))
    if not math.isfinite(alignment_max_error) or alignment_max_error <= 0 or min_common_images < 3:
        raise ValueError(
            "model_aligner requires a finite alignment_max_error > 0 and min_common_images >= 3"
        )
    remaining_tokens = remove_cli_options(
        user_tokens,
        controlled_options,
        "--model_aligner_args",
    )
    command = [
        colmap_exe,
        "model_aligner",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--database_path",
        str(database_path),
        "--transform_path",
        str(transform_path),
        "--ref_is_gps",
        "1",
        "--alignment_type",
        "enu",
        "--alignment_max_error",
        str(alignment_max_error),
        "--min_common_images",
        str(min_common_images),
        "--default_random_seed",
        "0",
    ] + remaining_tokens
    return command, alignment_max_error, min_common_images



def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def _read_c_string(f) -> str:
    chars = []
    while True:
        c = f.read(1)
        if not c or c == b"\x00":
            break
        chars.append(c)
    return b"".join(chars).decode("utf-8", errors="replace")


def read_num_registered_images(model_dir: Path) -> int:
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"
    if images_bin.exists():
        with images_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if images_txt.exists():
        n = 0
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    n += 1
        return n
    return -1


def read_num_points3d(model_dir: Path) -> int:
    points_bin = model_dir / "points3D.bin"
    points_txt = model_dir / "points3D.txt"
    if points_bin.exists():
        with points_bin.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
        return int(n)
    if points_txt.exists():
        n = 0
        with points_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                n += 1
        return n
    return -1


def read_image_names(model_dir: Path):
    names = []
    images_txt = model_dir / "images.txt"
    images_bin = model_dir / "images.bin"

    if images_bin.exists():
        with images_bin.open("rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                f.read(4)
                f.read(8 * 7)
                f.read(4)
                name = _read_c_string(f)
                names.append(name)
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.read(num_points2d * (8 + 8 + 8))
        return names

    if images_txt.exists():
        expect_image = True
        with images_txt.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line.startswith("#"):
                    continue
                if not expect_image:
                    expect_image = True
                    continue
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 10 and parts[0].isdigit():
                    names.append(" ".join(parts[9:]))
                    expect_image = False
        return names

    return names


def select_best_sparse_model(sparse_root: Path) -> Path:
    if not sparse_root.exists():
        raise FileNotFoundError(f"sparse root not found: {sparse_root}")

    candidates = []
    for p in sparse_root.iterdir():
        if not p.is_dir():
            continue
        if not re.fullmatch(r"\d+", p.name):
            continue
        reg = read_num_registered_images(p)
        pts = read_num_points3d(p)
        if reg < 0 and pts < 0:
            continue
        candidates.append((reg, pts, p))

    if not candidates:
        raise RuntimeError(f"No valid sparse models under: {sparse_root}")

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0][2]

    log_info(f"Sparse models found under {sparse_root}:")
    for reg, pts, p in candidates[:30]:
        log_info(f"  model={p.name} | registered={reg} | points3D={pts}")
    log_info(f"Selected best sparse model: {best}")
    return best


def validate_sparse_model(model_dir: Path, min_registered_images: int, mapper_mode: str) -> Tuple[int, int]:
    """Reject empty/undersized models even when COLMAP exits successfully."""
    registered = read_num_registered_images(model_dir)
    points3d = read_num_points3d(model_dir)
    required = max(3, int(min_registered_images))
    if registered < required or points3d <= 0:
        raise RuntimeError(
            f"COLMAP {mapper_mode} did not produce a usable sparse model: "
            f"registered_images={registered} (required>={required}), points3D={points3d}, "
            f"model={model_dir}. No incremental/fallback mapper will be attempted."
        )
    log_info(
        f"SfM model postcondition passed: mapper={mapper_mode}, "
        f"registered_images={registered}, points3D={points3d}"
    )
    return registered, points3d


def _parse_exiftool_gps_records(records: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float, float]]:
    """Parse ExifTool ``-n -G -json`` records without correcting their values."""
    gps: Dict[str, Tuple[float, float, float]] = {}

    def pick(rec: Dict[str, Any], keys: List[str]):
        casefolded = {str(key).casefold(): value for key, value in rec.items()}
        for key in keys:
            value = casefolded.get(key.casefold())
            if value is not None:
                return value
        # Retain compatibility with group names returned by other ExifTool
        # versions, but only after the explicit EXIF/XMP/Composite priority.
        for key in keys:
            suffix = ":" + key.split(":")[-1].casefold()
            for candidate, value in casefolded.items():
                if value is not None and candidate.endswith(suffix):
                    return value
        return None

    for record in records:
        source = str(record.get("SourceFile", ""))
        basename = os.path.basename(source)
        if not basename:
            continue
        lat = pick(
            record,
            ["EXIF:GPSLatitude", "XMP:GPSLatitude", "Composite:GPSLatitude", "GPSLatitude"],
        )
        lon = pick(
            record,
            ["EXIF:GPSLongitude", "XMP:GPSLongitude", "Composite:GPSLongitude", "GPSLongitude"],
        )
        alt = pick(
            record,
            [
                "EXIF:GPSAltitude",
                "XMP:GPSAltitude",
                "Composite:GPSAltitude",
                "XMP-drone-dji:AbsoluteAltitude",
                "AbsoluteAltitude",
                "RelativeAltitude",
                "GPSAltitude",
            ],
        )
        if not (is_number(lat) and is_number(lon) and is_number(alt)):
            continue
        values = (float(lat), float(lon), float(alt))
        if not all(math.isfinite(value) for value in values):
            continue
        if basename in gps and gps[basename] != values:
            raise RuntimeError(f"Conflicting ExifTool GPS records for basename {basename!r}")
        gps[basename] = values
    return gps


def exiftool_extract_gps(input_dir: Path, exiftool_exe: str):
    if not input_dir.exists():
        raise FileNotFoundError(f"input dir not found: {input_dir}")

    exiftool_path = _resolve_executable(exiftool_exe)

    cmd = [
        exiftool_path,
        "-q", "-q",
        "-json",
        "-n",
        # Ask exiftool for all groups so PNG EXIF/XMP GPS tags are not dropped.
        "-G",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-EXIF:GPSLatitude",
        "-EXIF:GPSLongitude",
        "-EXIF:GPSAltitude",
        "-XMP:GPSLatitude",
        "-XMP:GPSLongitude",
        "-XMP:GPSAltitude",
        "-Composite:GPSLatitude",
        "-Composite:GPSLongitude",
        "-Composite:GPSAltitude",
        "-XMP-drone-dji:AbsoluteAltitude",
        "-r",
        "-ext", "jpg",
        "-ext", "jpeg",
        "-ext", "JPG",
        "-ext", "JPEG",
        "-ext", "png",
        "-ext", "PNG",
        str(input_dir),
    ]

    log_info("Extracting GPS from images via exiftool (may take a bit)...")
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    text = p.stdout.decode("utf-8", errors="replace")

    records = []
    if text.strip():
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('[')
            end = text.rfind(']')
            if start >= 0 and end > start:
                try:
                    records = json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    log_warn("Exiftool output is not valid JSON after bracket recovery; treat as no GPS.")
                    records = []
            else:
                log_warn("Exiftool output has no JSON payload; treat as no GPS.")
                records = []
    else:
        log_warn("Exiftool returned empty output; treat as no GPS.")

    gps = _parse_exiftool_gps_records(records)
    log_info(f"EXIF GPS entries found: {len(gps)} (keyed by basename)")
    return gps


def ensure_pose_priors_table(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pose_priors'")
    if cur.fetchone():
        return

    # A database created by COLMAP 4.x has frame_data and uses generic
    # data/sensor associations for pose priors. Keep the legacy schema only for
    # old databases so this helper remains backwards compatible.
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frame_data'")
    sensor_schema = cur.fetchone() is not None
    if sensor_schema:
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS pose_priors (
                pose_prior_id INTEGER PRIMARY KEY NOT NULL,
                corr_data_id INTEGER NOT NULL,
                corr_sensor_id INTEGER NOT NULL,
                corr_sensor_type INTEGER NOT NULL,
                position BLOB,
                position_covariance BLOB,
                gravity BLOB,
                coordinate_system INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS pose_prior_data_assignment ON
                pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type);
            """
        )
    else:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS pose_priors (
                image_id INTEGER PRIMARY KEY NOT NULL,
                position BLOB,
                coordinate_system INTEGER NOT NULL,
                position_covariance BLOB,
                FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
            )
            """
        )
    con.commit()


def get_pose_priors_schema(con: sqlite3.Connection):
    cur = con.cursor()
    cur.execute("PRAGMA table_info(pose_priors)")
    rows = cur.fetchall()
    return [r[1] for r in rows]


def populate_pose_priors_from_exif(
    database_path: Path,
    input_dir: Path,
    exiftool_exe: str,
    wgs84_code: int,
    prior_position_std_m: Optional[float] = None,
    swap_latlon: bool = False,
):
    """
    Populate COLMAP pose_priors from EXIF GPS.

    Notes:
    - Newer COLMAP versions store pose priors in a dedicated `pose_priors` table.
    - For GPS, COLMAP expects WGS84 coordinate system code (in your build it's 0).
    - `position_covariance` should be NULL or finite values. Writing NaNs can make
      downstream alignment fail.

    prior_position_std_m:
      If provided, we write a diagonal covariance:
        [std_lat_deg^2, std_lon_deg^2, std_alt_m^2]
      where std_lat_deg/std_lon_deg are meters converted to degrees at that latitude.
      If None, position_covariance is left NULL.
    """
    gps = exiftool_extract_gps(input_dir, exiftool_exe=exiftool_exe)
    if swap_latlon:
        # Keep the returned/audited values identical to the values written to
        # the database. The previous implementation swapped only inside the
        # insert loop, making every later EXIF-vs-DB audit fail by construction.
        gps = {base: (lon, lat, alt) for base, (lat, lon, alt) in gps.items()}

    con = sqlite3.connect(str(database_path))
    try:
        ensure_pose_priors_table(con)
        cols = get_pose_priors_schema(con)
        legacy_expected = ["image_id", "position", "coordinate_system", "position_covariance"]
        sensor_expected = [
            "pose_prior_id",
            "corr_data_id",
            "corr_sensor_id",
            "corr_sensor_type",
            "position",
            "position_covariance",
            "gravity",
            "coordinate_system",
        ]
        if cols not in (legacy_expected, sensor_expected):
            raise RuntimeError(f"pose_priors columns unexpected: {cols}")
        sensor_schema = cols == sensor_expected
        log_info(f"pose_priors schema: {'sensor-associated (COLMAP 4.x)' if sensor_schema else 'legacy'}")

        cur = con.cursor()
        # A rerun must never retain priors from an older input set.
        cur.execute("DELETE FROM pose_priors")
        if sensor_schema:
            cur.execute("SELECT image_id, name, camera_id FROM images")
        else:
            cur.execute("SELECT image_id, name FROM images")
        rows = cur.fetchall()

        # Map by basename (works if basenames are unique). Also keep a duplicate check.
        name2entry = {}
        dup_bases = set()
        for row in rows:
            image_id, name = row[:2]
            camera_id = row[2] if sensor_schema else None
            base = os.path.basename(name)
            if base in name2entry:
                dup_bases.add(base)
            name2entry[base] = (image_id, camera_id)
        if dup_bases:
            raise RuntimeError(
                f"Duplicate basenames detected in DB images table (showing up to 5): {list(sorted(dup_bases))[:5]}. "
                "Cannot associate image EXIF GPS priors unambiguously."
            )

        inserted = 0
        matched = 0

        # Precompute some sanity stats.
        lat_list, lon_list, alt_list = [], [], []

        for base, (image_id, camera_id) in name2entry.items():
            if base not in gps:
                continue
            matched += 1
            lat, lon, alt = gps[base]

            # position is stored as 3 float64 values (lat_deg, lon_deg, alt_m)
            pos = np.asarray([lat, lon, alt], dtype=np.float64)

            # Optional covariance (NULL by default).
            cov_blob = None
            if prior_position_std_m is not None and prior_position_std_m > 0:
                # Convert meters -> degrees at this latitude (rough WGS84 approximation).
                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * max(1e-6, math.cos(math.radians(lat)))
                std_lat_deg = prior_position_std_m / meters_per_deg_lat
                std_lon_deg = prior_position_std_m / meters_per_deg_lon
                cov = np.diag([std_lat_deg**2, std_lon_deg**2, float(prior_position_std_m) ** 2]).astype(np.float64)
                cov_blob = cov.tobytes()

            if sensor_schema:
                # COLMAP 4.x associates priors with a generic data/sensor ID.
                # For an image, corr_data_id=image_id, corr_sensor_id=camera_id,
                # and SensorType::CAMERA is integer 0. Missing covariance and
                # gravity use the same explicit NaN blobs as PosePrior defaults.
                if cov_blob is None:
                    cov_blob = np.full((3, 3), np.nan, dtype=np.float64).tobytes()
                gravity_blob = np.full(3, np.nan, dtype=np.float64).tobytes()
                cur.execute(
                    "INSERT OR REPLACE INTO pose_priors("
                    "pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
                    "position, position_covariance, gravity, coordinate_system) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        int(image_id),
                        int(image_id),
                        int(camera_id),
                        0,
                        pos.tobytes(),
                        cov_blob,
                        gravity_blob,
                        int(wgs84_code),
                    ),
                )
            else:
                cur.execute(
                    "INSERT OR REPLACE INTO pose_priors(image_id, position, coordinate_system, position_covariance) "
                    "VALUES (?, ?, ?, ?)",
                    (int(image_id), pos.tobytes(), int(wgs84_code), cov_blob),
                )
            inserted += 1

            lat_list.append(pos[0])
            lon_list.append(pos[1])
            alt_list.append(pos[2])

        con.commit()
        log_info(f"Pose priors populated: inserted={inserted}, matched_images_in_db={matched}, db_images_total={len(rows)}")

        cur.execute("SELECT COUNT(*) FROM pose_priors")
        cnt = cur.fetchone()[0]
        log_info(f"pose_priors rows in DB: {cnt}")

        cur.execute("SELECT MIN(coordinate_system), MAX(coordinate_system) FROM pose_priors")
        mn, mx = cur.fetchone()
        log_info(f"pose_priors.coordinate_system range: {mn}..{mx}")

        if lat_list and lon_list and alt_list:
            log_info(
                "pose_priors position ranges (from inserted rows): "
                f"lat[{min(lat_list):.8f},{max(lat_list):.8f}] "
                f"lon[{min(lon_list):.8f},{max(lon_list):.8f}] "
                f"alt[{min(alt_list):.3f},{max(alt_list):.3f}]"
            )
            # Basic plausibility checks (WGS84 degrees)
            if not (-90 <= min(lat_list) <= 90 and -90 <= max(lat_list) <= 90):
                log_warn("Latitude range looks suspicious (outside [-90, 90]).")
            if not (-180 <= min(lon_list) <= 180 and -180 <= max(lon_list) <= 180):
                log_warn("Longitude range looks suspicious (outside [-180, 180]).")
            if min(alt_list) < -500.0 or max(alt_list) > 6000.0:
                log_warn(
                    "Image-embedded altitude is outside the conservative plausibility band [-500m, 6000m]."
                )
            log_warn(
                "Image-embedded WGS84-like values are used verbatim for relative local ENU only; "
                "their true absolute location/elevation is not validated."
            )

    finally:
        con.close()

    return gps


def sanity_check_overlap(model_dir: Path, gps_by_basename: dict):
    model_names = read_image_names(model_dir)
    if not model_names:
        log_warn("Could not read image names from selected model; skip overlap check.")
        return 0
    bases = [os.path.basename(n) for n in model_names]
    overlap = sum(1 for b in bases if b in gps_by_basename)
    log_info(f"Selected model images: {len(bases)}; images-with-GPS(overlap by basename): {overlap}")
    if overlap < 3:
        log_warn("Overlap < 3; model_aligner likely fails (min_common_images default is 3).")
    return overlap


def _qvec_to_rotmat(qvec) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm == 0:
        raise ValueError(f"Invalid COLMAP quaternion: {qvec}")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def read_registered_camera_centers(model_dir: Path) -> Dict[str, np.ndarray]:
    """Read world-space camera centers from a COLMAP binary or text model."""
    centers: Dict[str, np.ndarray] = {}
    images_bin = model_dir / "images.bin"
    images_txt = model_dir / "images.txt"

    if images_bin.exists():
        with images_bin.open("rb") as f:
            header = f.read(8)
            if len(header) != 8:
                raise RuntimeError(f"Truncated COLMAP images.bin: {images_bin}")
            num_images = struct.unpack("<Q", header)[0]
            for _ in range(num_images):
                image_id_raw = f.read(4)
                pose_raw = f.read(8 * 7)
                camera_id_raw = f.read(4)
                if len(image_id_raw) != 4 or len(pose_raw) != 56 or len(camera_id_raw) != 4:
                    raise RuntimeError(f"Truncated COLMAP image record: {images_bin}")
                qvec_tvec = struct.unpack("<7d", pose_raw)
                name = _read_c_string(f)
                npoints_raw = f.read(8)
                if len(npoints_raw) != 8:
                    raise RuntimeError(f"Truncated COLMAP points2D count: {images_bin}")
                num_points2d = struct.unpack("<Q", npoints_raw)[0]
                f.seek(num_points2d * 24, os.SEEK_CUR)
                rot = _qvec_to_rotmat(qvec_tvec[:4])
                tvec = np.asarray(qvec_tvec[4:], dtype=np.float64)
                centers[name] = -rot.T @ tvec
        return centers

    if images_txt.exists():
        expect_image = True
        with images_txt.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped.startswith("#"):
                    continue
                if not expect_image:
                    expect_image = True
                    continue
                if not stripped:
                    continue
                parts = stripped.split()
                if len(parts) < 10:
                    raise RuntimeError(f"Malformed COLMAP images.txt record: {stripped[:200]}")
                qvec = [float(v) for v in parts[1:5]]
                tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float64)
                name = " ".join(parts[9:])
                rot = _qvec_to_rotmat(qvec)
                centers[name] = -rot.T @ tvec
                expect_image = False
        return centers

    raise FileNotFoundError(f"No images.bin or images.txt under aligned model: {model_dir}")


def wgs84_to_enu(lat_lon_alt: np.ndarray, origin_lat_lon_alt: np.ndarray) -> np.ndarray:
    """Match COLMAP 4.1 GPSTransform::EllipsoidToENU for WGS84."""
    coords = np.asarray(lat_lon_alt, dtype=np.float64)
    origin = np.asarray(origin_lat_lon_alt, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or origin.shape != (3,):
        raise ValueError("WGS84 coordinates must have shape (N, 3) and origin shape (3,)")

    semimajor = 6378137.0
    flattening = 1.0 / 298.257223563
    eccentricity_sq = flattening * (2.0 - flattening)

    def to_ecef(values: np.ndarray) -> np.ndarray:
        lat = np.deg2rad(values[:, 0])
        lon = np.deg2rad(values[:, 1])
        alt = values[:, 2]
        sin_lat = np.sin(lat)
        cos_lat = np.cos(lat)
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        radius = semimajor / np.sqrt(1.0 - eccentricity_sq * sin_lat * sin_lat)
        return np.column_stack(
            (
                (radius + alt) * cos_lat * cos_lon,
                (radius + alt) * cos_lat * sin_lon,
                (radius * (1.0 - eccentricity_sq) + alt) * sin_lat,
            )
        )

    xyz = to_ecef(coords)
    ref_xyz = to_ecef(origin.reshape(1, 3))[0]
    ref_lat = math.radians(float(origin[0]))
    ref_lon = math.radians(float(origin[1]))
    sin_lat, cos_lat = math.sin(ref_lat), math.cos(ref_lat)
    sin_lon, cos_lon = math.sin(ref_lon), math.cos(ref_lon)
    ecef_to_enu = np.asarray(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return (ecef_to_enu @ (xyz - ref_xyz).T).T


def write_enu_alignment_audit(
    database_path: Path,
    aligned_model: Path,
    gps_by_basename: dict,
    transform_path: Path,
    report_path: Path,
    alignment_max_error: float,
    min_common_images: int,
    expected_image_names: Optional[List[str]] = None,
) -> dict:
    """Verify local ENU agreement without claiming true absolute geolocation."""
    if not transform_path.is_file() or transform_path.stat().st_size == 0:
        raise RuntimeError(f"model_aligner did not write a non-empty transform: {transform_path}")
    try:
        transform_values = [float(value) for value in transform_path.read_text(encoding="utf-8").split()]
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"Cannot parse model_aligner Sim3 transform: {transform_path}") from exc
    # COLMAP 4.x Sim3d::ToFile stores scale, quaternion(wxyz), translation(xyz).
    if len(transform_values) != 8 or not np.all(np.isfinite(transform_values)):
        raise RuntimeError(
            "COLMAP 4.x model_aligner transform must contain exactly 8 finite values "
            f"(scale, quaternion wxyz, translation xyz): {transform_path}"
        )
    transform_scale = float(transform_values[0])
    transform_quaternion = np.asarray(transform_values[1:5], dtype=np.float64)
    if transform_scale <= 0 or not np.isclose(
        np.linalg.norm(transform_quaternion), 1.0, rtol=1e-6, atol=1e-8
    ):
        raise RuntimeError(
            f"Invalid COLMAP Sim3 scale/quaternion in model_aligner transform: {transform_path}"
        )

    con = _connect_immutable_sqlite(database_path)
    try:
        columns = get_pose_priors_schema(con)
        if "corr_data_id" in columns:
            prior_rows = con.execute(
                "SELECT images.image_id, images.name, pose_priors.position, pose_priors.coordinate_system "
                "FROM images JOIN pose_priors ON pose_priors.corr_data_id = images.image_id "
                "AND pose_priors.corr_sensor_type = 0 ORDER BY images.image_id"
            ).fetchall()
        else:
            prior_rows = con.execute(
                "SELECT images.image_id, images.name, pose_priors.position, pose_priors.coordinate_system "
                "FROM images JOIN pose_priors ON pose_priors.image_id = images.image_id "
                "ORDER BY images.image_id"
            ).fetchall()
    finally:
        con.close()

    refs = []
    exif_reference_matches = 0
    for _, name, position_blob, coordinate_system in prior_rows:
        if int(coordinate_system) != 0:
            raise RuntimeError(
                f"Pose prior for {name!r} is not WGS84 (coordinate_system={coordinate_system})."
            )
        if position_blob is None or len(position_blob) != 24:
            raise RuntimeError(f"Pose prior for {name!r} does not contain 3 float64 WGS84 values.")
        position = struct.unpack("3d", position_blob)
        refs.append((name, position))
        exif_value = gps_by_basename.get(os.path.basename(name))
        if exif_value is not None and np.allclose(position, exif_value, rtol=0.0, atol=1e-12):
            exif_reference_matches += 1
    if len(refs) < min_common_images:
        raise RuntimeError(
            f"Only {len(refs)} image GPS references are available; model_aligner requires at least {min_common_images}."
        )
    expected_names = None
    if expected_image_names is not None:
        expected_names = sorted(str(name) for name in expected_image_names)
        if len(expected_names) != len(set(expected_names)):
            raise RuntimeError("Expected formal image inventory contains duplicate names")
        reference_names = sorted(str(name) for name, _ in refs)
        if reference_names != expected_names:
            raise RuntimeError(
                "Formal ENU audit requires an exact pose-prior name set: "
                f"priors={len(reference_names)} expected={len(expected_names)}"
            )

    origin_name, origin_gps = refs[0]
    enu_values = wgs84_to_enu(
        np.asarray([gps for _, gps in refs], dtype=np.float64),
        np.asarray(origin_gps, dtype=np.float64),
    )
    enu_by_name = {name: enu for (name, _), enu in zip(refs, enu_values)}
    centers = read_registered_camera_centers(aligned_model)
    if expected_names is not None and sorted(centers) != expected_names:
        raise RuntimeError(
            "Formal ENU audit requires an exact aligned-camera name set: "
            f"aligned={len(centers)} expected={len(expected_names)}"
        )

    errors = []
    aligned_common = []
    reference_common = []
    for name, center in centers.items():
        ref = enu_by_name.get(name)
        if ref is None:
            # Normal UAV-FGS inputs use basenames. Keep a unique-basename
            # fallback for models that preserved an image subdirectory.
            matching = [value for ref_name, value in enu_by_name.items() if os.path.basename(ref_name) == os.path.basename(name)]
            if len(matching) == 1:
                ref = matching[0]
        if ref is not None:
            center_arr = np.asarray(center, dtype=np.float64)
            ref_arr = np.asarray(ref, dtype=np.float64)
            errors.append(float(np.linalg.norm(center_arr - ref_arr)))
            aligned_common.append(center_arr)
            reference_common.append(ref_arr)

    if len(errors) < min_common_images:
        raise RuntimeError(
            f"Aligned model has only {len(errors)} registered cameras with GPS; need at least {min_common_images}."
        )
    errors_arr = np.asarray(errors, dtype=np.float64)
    aligned_common_arr = np.asarray(aligned_common, dtype=np.float64)
    reference_common_arr = np.asarray(reference_common, dtype=np.float64)
    if not np.all(np.isfinite(errors_arr)):
        raise RuntimeError("Non-finite camera-to-GPS ENU alignment error detected.")

    inlier_count = int(np.count_nonzero(errors_arr <= alignment_max_error))
    inlier_mask = errors_arr <= alignment_max_error
    aligned_inliers = aligned_common_arr[inlier_mask]
    reference_inliers = reference_common_arr[inlier_mask]
    if inlier_count > 0:
        aligned_centered = aligned_inliers - np.mean(aligned_inliers, axis=0, keepdims=True)
        reference_centered = reference_inliers - np.mean(reference_inliers, axis=0, keepdims=True)
        aligned_spread = float(
            np.sqrt(np.mean(np.sum(np.square(aligned_centered), axis=1)))
        )
        reference_spread = float(
            np.sqrt(np.mean(np.sum(np.square(reference_centered), axis=1)))
        )
        reference_coordinate_scale = max(
            1.0,
            float(np.max(np.abs(reference_inliers))),
        )
    else:
        aligned_spread = 0.0
        reference_spread = 0.0
        reference_coordinate_scale = 1.0
    reference_observability_tolerance = (
        np.finfo(np.float64).eps * reference_coordinate_scale * 1024.0
    )
    reference_scale_observable = bool(
        inlier_count >= min_common_images
        and np.isfinite(reference_spread)
        and reference_spread > reference_observability_tolerance
    )
    spread_ratio = (
        float(aligned_spread / reference_spread)
        if reference_scale_observable
        else None
    )
    geometry_scale_verified = bool(
        reference_scale_observable
        and spread_ratio is not None
        and np.isfinite(aligned_spread)
        and np.isfinite(spread_ratio)
        and ENU_GEOMETRY_SCALE_RATIO_MIN
        <= spread_ratio
        <= ENU_GEOMETRY_SCALE_RATIO_MAX
    )
    median_error = float(np.median(errors_arr))
    verified = (
        inlier_count >= min_common_images
        and median_error <= alignment_max_error
        and exif_reference_matches == len(refs)
        and geometry_scale_verified
    )
    if expected_names is not None:
        verified = verified and (
            len(refs) == len(expected_names)
            and len(errors) == len(expected_names)
            and inlier_count == len(expected_names)
        )
    gps_value_records = [
        [str(name), float(position[0]), float(position[1]), float(position[2])]
        for name, position in refs
    ]
    payload = {
        "status": "verified" if verified else "failed",
        "coordinate_frame": "local ENU from image-embedded WGS84-like values (COLMAP 4.1 convention)",
        "coordinate_scope": "relative_local_enu_only",
        "absolute_geolocation_validated": False,
        "absolute_geolocation_plausibility_warning": (
            "Image-embedded coordinates are used verbatim and were not validated as the true city location "
            "or absolute elevation; this audit supports only relative local-ENU alignment."
        ),
        "image_embedded_gps_values_sha256": _canonical_sha256(gps_value_records),
        "enu_origin_image": origin_name,
        "enu_origin_wgs84_lat_lon_alt": [float(v) for v in origin_gps],
        "database_image_gps_references": len(refs),
        "database_priors_matching_current_exif": exif_reference_matches,
        "registered_common_images": len(errors),
        "alignment_max_error_m": float(alignment_max_error),
        "min_common_images": int(min_common_images),
        "within_threshold_images": inlier_count,
        "formal_expected_images": None if expected_names is None else len(expected_names),
        "model_aligner_default_random_seed": 0,
        "error_m": {
            "mean": float(np.mean(errors_arr)),
            "median": median_error,
            "max": float(np.max(errors_arr)),
        },
        "geometry_scale_check": {
            "basis": ENU_GEOMETRY_SCALE_BASIS,
            "inlier_images": inlier_count,
            "aligned_spread_m": aligned_spread,
            "reference_spread_m": reference_spread,
            "reference_observability_tolerance_m": float(reference_observability_tolerance),
            "reference_scale_observable": reference_scale_observable,
            "aligned_to_reference_ratio": spread_ratio,
            "accepted_ratio": [
                ENU_GEOMETRY_SCALE_RATIO_MIN,
                ENU_GEOMETRY_SCALE_RATIO_MAX,
            ],
            "verified": geometry_scale_verified,
        },
        "aligned_camera_extent_xyz_m": [float(v) for v in np.ptp(aligned_common_arr, axis=0)],
        "reference_enu_extent_xyz_m": [float(v) for v in np.ptp(reference_common_arr, axis=0)],
        "transform_path": str(transform_path),
        "transform_sim3": {
            "scale": transform_scale,
            "quaternion_wxyz": [float(value) for value in transform_quaternion],
            "translation_xyz": [float(value) for value in transform_values[5:8]],
        },
        "aligned_model_path": str(aligned_model),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(report_path.name + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, report_path)

    if not verified:
        spread_ratio_text = (
            "unobservable" if spread_ratio is None else f"{spread_ratio:.6g}"
        )
        raise RuntimeError(
            "model_aligner returned success, but the aligned camera centers failed the ENU audit: "
            f"median={median_error:.6f}m, inliers={inlier_count}/{len(errors)}, "
            f"threshold={alignment_max_error:.6f}m, "
            f"centered_rms_scale_ratio={spread_ratio_text} "
            f"(required {ENU_GEOMETRY_SCALE_RATIO_MIN:g}..{ENU_GEOMETRY_SCALE_RATIO_MAX:g}), "
            f"database_priors_matching_current_exif={exif_reference_matches}/{len(refs)}. "
            f"Report: {report_path}"
        )
    log_info(
        "ENU alignment verified: "
        f"common={len(errors)}, within_threshold={inlier_count}, "
        f"mean={payload['error_m']['mean']:.6f}m, median={median_error:.6f}m"
    )
    log_info(f"Alignment audit: {report_path}")
    return payload


def _geometry_scale_check_payload_verified(value: Any) -> bool:
    """Validate the content-addressed ENU non-collapse evidence in an audit."""
    if not isinstance(value, dict) or value.get("verified") is not True:
        return False
    if value.get("basis") != ENU_GEOMETRY_SCALE_BASIS:
        return False
    accepted = value.get("accepted_ratio")
    if accepted != [ENU_GEOMETRY_SCALE_RATIO_MIN, ENU_GEOMETRY_SCALE_RATIO_MAX]:
        return False
    if value.get("reference_scale_observable") is not True:
        return False
    try:
        inlier_images = int(value["inlier_images"])
        aligned_spread = float(value["aligned_spread_m"])
        reference_spread = float(value["reference_spread_m"])
        observability_tolerance = float(value["reference_observability_tolerance_m"])
        ratio = float(value["aligned_to_reference_ratio"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    if (
        inlier_images < 3
        or not np.all(
            np.isfinite(
                [aligned_spread, reference_spread, observability_tolerance, ratio]
            )
        )
        or aligned_spread < 0.0
        or observability_tolerance < 0.0
        or reference_spread <= observability_tolerance
        or not np.isclose(
            ratio,
            aligned_spread / reference_spread,
            rtol=1e-12,
            atol=1e-15,
        )
    ):
        return False
    return ENU_GEOMETRY_SCALE_RATIO_MIN <= ratio <= ENU_GEOMETRY_SCALE_RATIO_MAX


def ensure_3dgs_sparse_layout(root: Path) -> Path:
    """Normalize COLMAP output layout for 3DGS.

    3DGS loaders typically expect:
      <root>/images
      <root>/sparse/0/{cameras,images,points3D}.{bin|txt}

    COLMAP may instead write the model directly under <root>/sparse (no /0)
    or under a different numeric subfolder. This function makes sure that
    <root>/sparse/0 exists and contains the model files.
    """

    sparse = root / "sparse"
    if not sparse.exists():
        raise FileNotFoundError(f"COLMAP undistorted sparse folder not found: {sparse}")

    model0 = sparse / "0"

    def has_any_model_files(p: Path) -> bool:
        for fn in (
            "cameras.bin",
            "images.bin",
            "points3D.bin",
            "cameras.txt",
            "images.txt",
            "points3D.txt",
            "rigs.bin",
            "frames.bin",
        ):
            if (p / fn).exists():
                return True
        return False

    # Case 1: already in sparse/0
    if model0.is_dir() and has_any_model_files(model0):
        return model0

    # Case 2: model files are directly under sparse/
    if has_any_model_files(sparse):
        model0.mkdir(parents=True, exist_ok=True)
        for child in list(sparse.iterdir()):
            if child.name == "0":
                continue
            if child.is_file():
                dst = model0 / child.name
                if dst.exists():
                    dst.unlink()
                shutil.move(str(child), str(dst))
        return model0

    # Case 3: numeric subfolders (but no /0)
    subdirs = [p for p in sparse.iterdir() if p.is_dir() and re.fullmatch(r"\d+", p.name)]
    if subdirs:
        candidates = [p for p in subdirs if has_any_model_files(p)] or subdirs

        def score(p: Path) -> int:
            s = 0
            pts_bin = p / "points3D.bin"
            imgs_bin = p / "images.bin"
            pts_txt = p / "points3D.txt"
            if pts_bin.exists():
                s += int(pts_bin.stat().st_size)
            if imgs_bin.exists():
                s += int(imgs_bin.stat().st_size) // 10
            if pts_txt.exists():
                s += int(pts_txt.stat().st_size) // 50
            return s

        best = max(candidates, key=score)
        if best.name != "0":
            if model0.exists():
                shutil.rmtree(model0)
            shutil.copytree(best, model0)
        return model0

    raise RuntimeError(f"Could not locate any COLMAP model under: {sparse}")


def export_model_as_txt(colmap_exe: str, model_dir: Path) -> None:
    """Export cameras/images/points to TXT for 3DGS fallback readers."""
    if (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists() and (model_dir / "points3D.txt").exists():
        return

    log_info(f"Exporting COLMAP model as TXT for 3DGS compatibility: {model_dir}")
    try:
        run_cmd([
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(model_dir),
            "--output_type",
            "TXT",
        ])
    except Exception as e:
        log_warn(f"model_converter failed (will continue): {e}")

def ensure_camera_models_supported_for_3dgs(colmap_exe: str, model_dir: Path) -> None:
    """3DGS loaders often only accept PINHOLE / SIMPLE_PINHOLE camera models.
    Some COLMAP outputs (or previous conversions) may keep e.g. SIMPLE_RADIAL/OPENCV models.
    This function:
      1) Ensures cameras.txt exists (via model_converter -> TXT).
      2) Rewrites cameras.txt to PINHOLE or SIMPLE_PINHOLE when possible by dropping distortion params.
      3) Regenerates cameras.bin/images.bin/points3D.bin from the (possibly) edited TXT model.
    """
    export_model_as_txt(colmap_exe, model_dir)
    cam_txt = model_dir / "cameras.txt"
    if not cam_txt.exists():
        log_warn(f"cameras.txt not found under {model_dir}; cannot enforce camera model.")
        return

    txt = cam_txt.read_text(encoding="utf-8", errors="replace").splitlines()
    changed = False
    new_lines: List[str] = []
    conversions = []

    def fmt_params(ps: List[float]) -> List[str]:
        # use repr to preserve enough precision without trailing '+'
        return [repr(float(p)) for p in ps]

    for line in txt:
        s = line.strip()
        if not s or s.startswith("#"):
            new_lines.append(line)
            continue

        parts = s.split()
        if len(parts) < 5:
            new_lines.append(line)
            continue

        cam_id, model, w, h = parts[0], parts[1], parts[2], parts[3]
        # params may include leading '+'; float() handles it
        try:
            params = [float(x) for x in parts[4:]]
        except Exception:
            new_lines.append(line)
            continue

        if model in ("PINHOLE", "SIMPLE_PINHOLE"):
            new_lines.append(line)
            continue

        model_new = None
        params_new: List[float] = []

        # Most models start with f cx cy (SIMPLE_*) or fx fy cx cy (OPENCV-like)
        if model in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"):
            if len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]
        elif model in ("OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
        else:
            # Fallback heuristic:
            # - if >=4 params, treat as fx fy cx cy -> PINHOLE
            # - else if >=3 params, treat as f cx cy -> SIMPLE_PINHOLE
            if len(params) >= 4:
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                model_new = "PINHOLE"
                params_new = [fx, fy, cx, cy]
            elif len(params) >= 3:
                f, cx, cy = params[0], params[1], params[2]
                model_new = "SIMPLE_PINHOLE"
                params_new = [f, cx, cy]

        if model_new is None:
            # keep original line if we can't safely convert
            new_lines.append(line)
            continue

        changed = True
        conversions.append(f"{cam_id}:{model}->{model_new}")
        new_line = " ".join([cam_id, model_new, w, h] + fmt_params(params_new))
        new_lines.append(new_line)

    if changed:
        cam_txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        log_info(f"Adjusted camera models for 3DGS: {', '.join(conversions[:8])}{' ...' if len(conversions)>8 else ''}")

        # Regenerate BIN from TXT so 3DGS (which prefers .bin) reads the supported model.
        try:
            run_cmd([
                colmap_exe,
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(model_dir),
                "--output_type",
                "BIN",
            ])
        except Exception as e:
            log_warn(f"Failed to regenerate BIN model after camera-model fix (will keep TXT): {e}")
    else:
        log_info("Camera models already supported (PINHOLE/SIMPLE_PINHOLE); no conversion needed.")


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def _sqlite_sidecar_paths(database_path: Path) -> List[Path]:
    return [
        Path(str(database_path) + suffix)
        for suffix in ("-wal", "-shm", "-journal")
        if Path(str(database_path) + suffix).exists()
    ]


def _assert_closed_sqlite_database(database_path: Path) -> None:
    sidecars = _sqlite_sidecar_paths(database_path)
    if sidecars:
        raise RuntimeError(
            "Refusing to inspect/copy a SQLite database with WAL/journal sidecars; "
            f"checkpoint and close its writer first: {sidecars}"
        )


def _database_image_names(database_path: Path, *, immutable: bool = False) -> List[str]:
    if immutable:
        con = _connect_immutable_sqlite(database_path)
    else:
        con = sqlite3.connect(str(database_path))
    try:
        return sorted(str(row[0]) for row in con.execute("SELECT name FROM images").fetchall())
    finally:
        con.close()


def _connect_immutable_sqlite(database_path: Path) -> sqlite3.Connection:
    _assert_closed_sqlite_database(database_path)
    uri = Path(database_path).resolve().as_uri() + "?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _finalize_private_sqlite_database(database_path: Path) -> str:
    """Checkpoint private WAL state and make the main DB file the sole identity."""
    con = sqlite3.connect(str(database_path))
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        mode_row = con.execute("PRAGMA journal_mode=DELETE").fetchone()
        mode = "" if mode_row is None else str(mode_row[0]).lower()
        if mode != "delete":
            raise RuntimeError(
                f"Could not finalize private SQLite database in DELETE journal mode: {database_path} ({mode})"
            )
        con.commit()
    finally:
        con.close()
    _assert_closed_sqlite_database(database_path)
    # Read through an immutable connection as a final integrity/schema check;
    # unlike an ordinary read-only connection this cannot create WAL/SHM files.
    _database_image_names(database_path, immutable=True)
    return _sha256_file(database_path)


def _alignment_database_semantic_summary(database_path: Path) -> Dict[str, Any]:
    """Digest aligner-relevant DB semantics without trusting its physical file hash."""
    con = _connect_immutable_sqlite(database_path)
    try:
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise RuntimeError(f"SQLite quick_check failed for {database_path}: {quick_check}")
        tables = {
            str(row[0])
            for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }

        def normalized_rows(table: str) -> List[List[Any]]:
            if table not in tables:
                return []
            rows = con.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            normalized: List[List[Any]] = []
            for row in rows:
                values: List[Any] = []
                for value in row:
                    if isinstance(value, bytes):
                        values.append(
                            {
                                "size_bytes": len(value),
                                "sha256": hashlib.sha256(value).hexdigest(),
                            }
                        )
                    else:
                        values.append(value)
                normalized.append(values)
            return normalized

        image_rows = normalized_rows("images")
        pose_prior_rows = normalized_rows("pose_priors")
        table_counts = {
            table: int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            for table in sorted(
                tables.intersection(
                    {
                        "cameras",
                        "images",
                        "pose_priors",
                        "keypoints",
                        "descriptors",
                        "matches",
                        "two_view_geometries",
                        "rigs",
                        "frames",
                        "frame_data",
                    }
                )
            )
        }
    finally:
        con.close()
    core = {
        "quick_check": quick_check,
        "image_rows_sha256": _canonical_sha256(image_rows),
        "pose_prior_rows_sha256": _canonical_sha256(pose_prior_rows),
        "table_counts": table_counts,
    }
    return {**core, "semantic_sha256": _canonical_sha256(core)}


def _remove_sqlite_database_family(database_path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(database_path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _copy_closed_sqlite_database(source: Path, destination: Path) -> None:
    _assert_closed_sqlite_database(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _database_provenance_core(
    *,
    input_inventory: Dict[str, Any],
    colmap_runtime: Dict[str, Any],
    feature_command: List[str],
    matcher_command: List[str],
    camera_model: str,
    prior_position_std_m: Optional[float],
    wgs84_code: int,
    swap_latlon: bool,
) -> Dict[str, Any]:
    return {
        "schema_version": DATABASE_PROVENANCE_SCHEMA_VERSION,
        "input_inventory": input_inventory,
        "colmap_executable_sha256": colmap_runtime["executable_sha256"],
        "fresh_build_feature_command": [str(value) for value in feature_command],
        "fresh_build_matcher_command": [str(value) for value in matcher_command],
        "camera_model": str(camera_model),
        "prior_position_std_m": None if prior_position_std_m is None else float(prior_position_std_m),
        "wgs84_code": int(wgs84_code),
        "swap_latlon": bool(swap_latlon),
    }


def database_policy_contract(policy: str) -> Dict[str, Any]:
    contracts = {
        "reset": {
            "run_feature_extractor": True,
            "run_matcher": True,
            "feature_match_origin_kind": "fresh_current_run",
            "reuse_verified_eligible": True,
        },
        "reuse_verified": {
            "run_feature_extractor": False,
            "run_matcher": False,
            "feature_match_origin_kind": "reused_verified_snapshot",
            "reuse_verified_eligible": True,
        },
        "adopt_legacy": {
            "run_feature_extractor": False,
            "run_matcher": False,
            "feature_match_origin_kind": "adopted_legacy_pinned_unverified",
            "reuse_verified_eligible": False,
        },
    }
    if policy not in contracts:
        raise ValueError(f"Unknown database policy: {policy}")
    return dict(contracts[policy])


def _validate_verified_database_reuse(
    *,
    database_path: Path,
    provenance_path: Path,
    expected_core: Dict[str, Any],
) -> Dict[str, Any]:
    if not database_path.is_file() or not provenance_path.is_file():
        raise RuntimeError(
            "--database_policy=reuse_verified requires both database.db and its provenance manifest. "
            "Use --database_policy=reset for a clean rebuild or adopt_legacy for an explicit legacy adoption."
        )
    _assert_closed_sqlite_database(database_path)
    provenance = _load_json(provenance_path)
    if provenance.get("status") != "complete":
        raise RuntimeError(f"Database provenance is not complete: {provenance_path}")
    if provenance.get("reuse_verified_eligible") is not True:
        raise RuntimeError(
            "Existing database provenance is explicitly ineligible for reuse_verified; "
            "an adopted legacy database must be adopted again by pinned SHA or rebuilt with reset."
        )
    if provenance.get("core") != expected_core:
        raise RuntimeError(
            "Existing COLMAP database provenance does not match the current input/runtime/arguments. "
            "Use --database_policy=reset."
        )
    expected_sha = str(provenance.get("database_sha256", ""))
    actual_sha = _sha256_file(database_path)
    if not expected_sha or expected_sha != actual_sha:
        raise RuntimeError(
            f"Existing COLMAP database SHA mismatch: expected={expected_sha} actual={actual_sha}. "
            "Use --database_policy=reset."
        )
    expected_names = sorted(str(name) for name in expected_core["input_inventory"]["names"])
    actual_names = _database_image_names(database_path, immutable=True)
    if actual_names != expected_names:
        raise RuntimeError(
            "Existing COLMAP database image inventory differs from input/. "
            "Use --database_policy=reset."
        )
    return provenance


def _write_database_provenance(
    *,
    database_path: Path,
    provenance_path: Path,
    core: Dict[str, Any],
    adopted_legacy: bool,
    hash_lifecycle: Dict[str, Any],
    feature_match_origin: Dict[str, Any],
    reuse_verified_eligible: bool,
) -> Dict[str, Any]:
    payload = {
        "schema_version": DATABASE_PROVENANCE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "core": core,
        "database_sha256": _sha256_file(database_path),
        "adopted_legacy_database": bool(adopted_legacy),
        "hash_lifecycle": hash_lifecycle,
        "feature_match_origin": feature_match_origin,
        "reuse_verified_eligible": bool(reuse_verified_eligible),
    }
    _atomic_write_json(provenance_path, payload)
    return payload


def _model_summary(model_dir: Path) -> Dict[str, Any]:
    registered_names = sorted(read_image_names(model_dir))
    registered_count = read_num_registered_images(model_dir)
    points3d_count = read_num_points3d(model_dir)
    if registered_count != len(registered_names):
        raise RuntimeError(
            f"COLMAP model count/name mismatch: header={registered_count} names={len(registered_names)} "
            f"model={model_dir}"
        )
    return {
        "registered_images": registered_count,
        "registered_image_names": registered_names,
        "registered_image_names_sha256": _canonical_sha256(registered_names),
        "points3D": points3d_count,
        "model_files": build_file_inventory(model_dir),
    }


def validate_undistorted_layout(layout_root: Path, *, min_registered_images: int) -> Dict[str, Any]:
    images_dir = layout_root / "images"
    model_dir = layout_root / "sparse" / "0"
    stereo_dir = layout_root / "stereo"
    image_inventory = build_file_inventory(images_dir, image_files_only=True)
    model_summary = _model_summary(model_dir)
    required = max(3, int(min_registered_images))
    if model_summary["registered_images"] < required or model_summary["points3D"] <= 0:
        raise RuntimeError(
            "Undistorted COLMAP layout is incomplete: "
            f"registered={model_summary['registered_images']} required>={required}, "
            f"points3D={model_summary['points3D']}"
        )
    image_names = sorted(str(name) for name in image_inventory["names"])
    registered_names = sorted(str(name) for name in model_summary["registered_image_names"])
    if image_names != registered_names:
        missing_images = sorted(set(registered_names) - set(image_names))[:10]
        extra_images = sorted(set(image_names) - set(registered_names))[:10]
        raise RuntimeError(
            "Undistorted image inventory does not exactly match registered COLMAP views: "
            f"images={len(image_names)} registered={len(registered_names)} "
            f"missing={missing_images} extra={extra_images}"
        )
    if not stereo_dir.is_dir():
        raise RuntimeError(f"COLMAP image_undistorter did not produce stereo/: {stereo_dir}")
    return {
        "images": image_inventory,
        "sparse_model": model_summary,
        "stereo_directory_present": True,
    }


def _completion_manifest_path(source_path: Path) -> Path:
    return source_path / "distorted" / COMPLETION_MANIFEST_NAME


def validate_conversion_completion_manifest(
    source_path: Path,
    *,
    expected_arguments: Optional[List[str]] = None,
    expected_colmap_sha256: str = "",
    expected_min_registered_images: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Validate every artifact needed to safely resume after conversion."""
    root = Path(source_path)
    manifest_path = _completion_manifest_path(root)
    try:
        manifest = _load_json(manifest_path)
        if manifest.get("schema_version") != COMPLETION_MANIFEST_SCHEMA_VERSION:
            return False, "completion manifest schema mismatch", manifest
        if manifest.get("status") != "complete":
            return False, "completion manifest status is not complete", manifest
        arguments = [str(value) for value in manifest.get("arguments", [])]
        if manifest.get("arguments_sha256") != _canonical_sha256(arguments):
            return False, "completion manifest argument digest mismatch", manifest
        if expected_arguments is not None and arguments != [str(value) for value in expected_arguments]:
            return False, "converter argv differs from completion manifest", manifest
        runtime = manifest.get("colmap_runtime", {})
        manifest_colmap_sha = str(runtime.get("executable_sha256", ""))
        if expected_colmap_sha256 and manifest_colmap_sha != expected_colmap_sha256:
            return False, "COLMAP executable SHA differs from completion manifest", manifest
        mapper_mode = str(manifest.get("mapper_mode", ""))
        if mapper_mode not in {"global", "incremental"}:
            return False, "completion manifest mapper mode is missing or invalid", manifest
        runtime_requirements = manifest.get("runtime_requirements", {})
        if not isinstance(runtime_requirements, dict):
            return False, "completion manifest runtime requirements are missing or invalid", manifest
        require_cuda = runtime_requirements.get("require_cuda_colmap")
        require_cudss_check = runtime_requirements.get("cudss_runtime_check_required")
        if not isinstance(require_cuda, bool) or not isinstance(require_cudss_check, bool):
            return False, "completion manifest runtime requirement flags are missing or invalid", manifest
        if require_cuda and runtime.get("reported_with_cuda") is not True:
            return False, "completion manifest does not prove the required CUDA COLMAP runtime", manifest
        if require_cudss_check and (
            mapper_mode != "global"
            or runtime.get("ldd_checked") is not True
            or runtime.get("ldd_has_cudss") is not True
        ):
            return False, "completion manifest does not prove the required GlobalMapper cuDSS runtime", manifest

        current_input = build_file_inventory(root / "input", image_files_only=True)
        if current_input != manifest.get("input_inventory"):
            return False, "input image inventory differs from completion manifest", manifest

        database_meta = manifest.get("database", {})
        database_path = root / "distorted" / "database.db"
        provenance_path = root / "distorted" / DATABASE_PROVENANCE_NAME
        if not database_path.is_file() or not provenance_path.is_file():
            return False, "database or database provenance manifest is missing", manifest
        _assert_closed_sqlite_database(database_path)
        actual_database_sha = _sha256_file(database_path)
        if actual_database_sha != database_meta.get("database_sha256"):
            return False, "database SHA differs from completion manifest", manifest
        if _sha256_file(provenance_path) != database_meta.get("provenance_sha256"):
            return False, "database provenance SHA differs from completion manifest", manifest
        provenance = _load_json(provenance_path)
        if provenance.get("status") != "complete" or provenance.get("database_sha256") != actual_database_sha:
            return False, "database provenance content is incomplete or stale", manifest
        if provenance.get("hash_lifecycle") != database_meta.get("hash_lifecycle"):
            return False, "database hash lifecycle differs between provenance and completion manifest", manifest
        if provenance.get("feature_match_origin") != database_meta.get("feature_match_origin"):
            return False, "database feature/match origin differs between provenance and completion manifest", manifest
        if provenance.get("reuse_verified_eligible") is not database_meta.get("reuse_verified_eligible"):
            return False, "database reuse eligibility differs between provenance and completion manifest", manifest
        lifecycle = database_meta.get("hash_lifecycle", {})
        if lifecycle.get("published_database_sha256") != actual_database_sha:
            return False, "published database SHA is inconsistent with its lifecycle", manifest
        isolated_aligner_db = lifecycle.get("isolated_model_aligner_database", {})
        if isolated_aligner_db.get("published") is not False:
            return False, "model_aligner database was not recorded as an isolated non-published copy", manifest
        database_names = _database_image_names(database_path, immutable=True)
        if database_names != sorted(str(name) for name in current_input["names"]):
            return False, "published database image inventory differs from current input", manifest

        try:
            manifest_min_registered = int(manifest["min_registered_images"])
        except (KeyError, TypeError, ValueError):
            return False, "completion manifest minimum registered-image count is missing or invalid", manifest
        if manifest_min_registered < 3:
            return False, "completion manifest minimum registered-image count is below 3", manifest
        if (
            expected_min_registered_images is not None
            and manifest_min_registered != int(expected_min_registered_images)
        ):
            return False, "minimum registered-image count differs from completion manifest", manifest
        min_registered = manifest_min_registered
        current_outputs = validate_undistorted_layout(root, min_registered_images=min_registered)
        if current_outputs != manifest.get("outputs"):
            return False, "final images/sparse output inventory differs from completion manifest", manifest
        input_names = set(str(name) for name in current_input["names"])
        registered_names = set(
            str(name) for name in current_outputs["sparse_model"]["registered_image_names"]
        )
        if not registered_names.issubset(input_names):
            return False, "registered sparse-model views are not a subset of the current input inventory", manifest
        if min_registered == int(current_input["count"]) and registered_names != input_names:
            return False, "formal all-input registration count is not exact", manifest

        alignment = manifest.get("alignment", {})
        if bool(alignment.get("enabled", False)):
            audit_path = root / "distorted" / "model_alignment_audit.json"
            transform_path = root / "distorted" / "model_alignment_transform.txt"
            aligned_model = root / "distorted" / "sparse_aligned"
            if not audit_path.is_file() or not transform_path.is_file() or not aligned_model.is_dir():
                return False, "published alignment artifacts are missing", manifest
            audit = _load_json(audit_path)
            if audit.get("status") != "verified":
                return False, "alignment audit is not verified", manifest
            if not _geometry_scale_check_payload_verified(audit.get("geometry_scale_check")):
                return False, "alignment audit ENU geometry-scale evidence is missing or invalid", manifest
            if isolated_aligner_db.get("enabled") is not True or isolated_aligner_db.get("semantics_unchanged") is not True:
                return False, "isolated model_aligner database lifecycle is incomplete", manifest
            if isolated_aligner_db.get("pre_semantic_sha256") != isolated_aligner_db.get("post_semantic_sha256"):
                return False, "model_aligner database semantic digests differ", manifest
            if _sha256_file(audit_path) != alignment.get("audit_sha256"):
                return False, "alignment audit SHA differs from completion manifest", manifest
            if _sha256_file(transform_path) != alignment.get("transform_sha256"):
                return False, "alignment transform SHA differs from completion manifest", manifest
            if build_file_inventory(aligned_model) != alignment.get("aligned_model_inventory"):
                return False, "aligned sparse model differs from completion manifest", manifest
            aligned_summary = _model_summary(aligned_model)
            if aligned_summary["registered_images"] < min_registered or aligned_summary["points3D"] <= 0:
                return False, "aligned sparse model is undersized or empty", manifest
            if alignment.get("coordinate_scope") != "relative_local_enu_only":
                return False, "alignment coordinate scope is not relative local ENU", manifest
            if alignment.get("absolute_geolocation_validated") is not False:
                return False, "alignment incorrectly claims validated absolute geolocation", manifest
            if audit.get("coordinate_scope") != "relative_local_enu_only" or audit.get("absolute_geolocation_validated") is not False:
                return False, "alignment audit geolocation scope is invalid", manifest
            if audit.get("model_aligner_default_random_seed") != 0:
                return False, "model_aligner deterministic seed evidence is missing", manifest
            if audit.get("image_embedded_gps_values_sha256") != alignment.get("image_embedded_gps_values_sha256"):
                return False, "image-embedded GPS digest differs between audit and manifest", manifest
            if min_registered == int(current_input["count"]):
                expected_count = int(current_input["count"])
                if sorted(aligned_summary["registered_image_names"]) != sorted(input_names):
                    return False, "formal aligned-camera name set is not exact", manifest
                for key in (
                    "database_image_gps_references",
                    "database_priors_matching_current_exif",
                    "registered_common_images",
                    "within_threshold_images",
                    "formal_expected_images",
                ):
                    if int(audit.get(key, -1)) != expected_count:
                        return False, f"formal alignment audit count is not exact: {key}", manifest
                if float(audit.get("error_m", {}).get("max", float("inf"))) > float(
                    audit.get("alignment_max_error_m", -1.0)
                ):
                    return False, "formal alignment audit has a camera outside the threshold", manifest
        return True, "verified", manifest
    except Exception as exc:
        return False, f"completion manifest validation error: {exc}", None


def verify_existing_output_ownership(
    root: Path,
    *,
    allow_unverified: bool,
    include_resized_outputs: bool,
) -> None:
    root = Path(root)
    destinations = [
        root / "images",
        root / "sparse",
        root / "stereo",
        root / "distorted" / "sparse",
        root / "distorted" / "sparse_aligned",
        root / "distorted" / "model_alignment_transform.txt",
        root / "distorted" / "model_alignment_audit.json",
    ]
    if include_resized_outputs:
        destinations.extend(root / f"images_{scale}" for scale in (2, 4, 8))
    existing = [path for path in destinations if _path_exists(path)]
    if not existing:
        return
    verified, reason, _ = validate_conversion_completion_manifest(root)
    if verified:
        log_info("Existing conversion outputs are owned by a verified completion manifest")
        return
    if not allow_unverified:
        raise RuntimeError(
            "Refusing to replace existing images/sparse/stereo or alignment artifacts without a verified "
            f"conversion completion manifest ({reason}). Existing paths: {[str(path) for path in existing]}. "
            "Use a fresh workspace or explicitly pass --allow_replace_unverified_outputs after auditing them."
        )
    log_warn(
        "Explicitly replacing unverified existing derived outputs after user/launcher opt-in: "
        + ", ".join(str(path) for path in existing)
    )


def _path_exists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _remove_owned_path(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_conversion_lock(root: Path) -> Path:
    lock_path = Path(root) / ".convert_uavfgs.lock"
    payload = {
        "pid": os.getpid(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for _ in range(2):
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing = _load_json(lock_path)
                existing_pid = int(existing.get("pid", -1))
            except Exception:
                existing_pid = -1
                try:
                    if time.time() - lock_path.stat().st_mtime < 60.0:
                        raise RuntimeError(
                            f"A newly-created conversion lock is not readable yet; refusing a concurrent run: {lock_path}"
                        )
                except FileNotFoundError:
                    continue
            if _pid_is_alive(existing_pid):
                raise RuntimeError(
                    f"Another convert_uavfgs process holds the workspace lock: {lock_path} pid={existing_pid}"
                )
            log_warn(f"Removing stale owned conversion lock: {lock_path}")
            lock_path.unlink()
            continue
        try:
            os.write(fd, (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return lock_path
    raise RuntimeError(f"Could not acquire conversion workspace lock: {lock_path}")


def _transaction_json(transaction: Dict[str, Any], status: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "status": status,
        "records": [
            {
                "source": None if record["source"] is None else str(record["source"]),
                "destination": str(record["destination"]),
                "backup": str(record["backup"]),
                "had_existing": bool(record["had_existing"]),
                "installed": bool(record["installed"]),
            }
            for record in transaction["records"]
        ],
        "cleanup_roots": [str(path) for path in transaction.get("cleanup_roots", [])],
    }


def _write_transaction_journal(transaction: Dict[str, Any], status: str) -> None:
    _atomic_write_json(
        transaction["backup_root"] / TRANSACTION_JOURNAL_NAME,
        _transaction_json(transaction, status),
    )


def _lexically_within(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def recover_incomplete_output_transactions(root: Path) -> None:
    """Rollback an interrupted multi-directory publish before starting a new run."""
    root = Path(root)
    for backup_root in sorted(root.glob(".convert_uavfgs_backup_*")):
        journal_path = backup_root / TRANSACTION_JOURNAL_NAME
        if not journal_path.is_file():
            raise RuntimeError(
                f"Found an owned conversion backup without a transaction journal: {backup_root}. "
                "Refusing automatic deletion."
            )
        journal = _load_json(journal_path)
        if journal.get("schema_version") != 1 or journal.get("status") not in {
            "prepared", "installing", "installed", "committed"
        }:
            raise RuntimeError(f"Invalid conversion transaction journal: {journal_path}")
        records = []
        for raw in journal.get("records", []):
            source = None if raw.get("source") is None else Path(str(raw["source"]))
            destination = Path(str(raw["destination"]))
            backup = Path(str(raw["backup"]))
            if not _lexically_within(destination, root) or not _lexically_within(backup, backup_root):
                raise RuntimeError(f"Unsafe path in conversion transaction journal: {journal_path}")
            if source is not None and not _lexically_within(source, root):
                raise RuntimeError(f"Unsafe staging path in conversion transaction journal: {journal_path}")
            installed = bool(raw.get("installed", False))
            if source is not None and not _path_exists(source) and _path_exists(destination):
                # The process may have died after the atomic source->destination
                # rename but before the next journal fsync/replace.
                installed = True
            records.append(
                {
                    "source": source,
                    "destination": destination,
                    "backup": backup,
                    "had_existing": bool(raw.get("had_existing", False)),
                    "installed": installed,
                }
            )
        cleanup_roots = [Path(str(value)) for value in journal.get("cleanup_roots", [])]
        if any(not _lexically_within(path, root) for path in cleanup_roots):
            raise RuntimeError(f"Unsafe cleanup path in conversion transaction journal: {journal_path}")
        if journal["status"] == "committed":
            log_warn(f"Finishing cleanup for committed conversion transaction: {backup_root}")
            for path in cleanup_roots:
                _remove_owned_path(path)
            _remove_owned_path(backup_root)
            continue

        log_warn(f"Rolling back interrupted conversion transaction: {backup_root}")
        transaction = {
            "backup_root": backup_root,
            "records": records,
            "cleanup_roots": cleanup_roots,
        }
        _rollback_staged_items(transaction)
        for path in cleanup_roots:
            _remove_owned_path(path)


def _swap_staged_items(
    items: List[Tuple[Optional[Path], Path]],
    *,
    backup_root: Path,
    cleanup_roots: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Install staged files/directories with exception-safe rollback."""
    if _path_exists(backup_root):
        raise FileExistsError(f"Refusing existing conversion backup directory: {backup_root}")
    backup_root.mkdir(parents=True, exist_ok=False)
    records: List[Dict[str, Any]] = []
    for index, (source, destination) in enumerate(items):
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = backup_root / f"{index:02d}_{destination.name}"
        if source is not None and not _path_exists(source):
            _remove_owned_path(backup_root)
            raise FileNotFoundError(f"Staged conversion output is missing: {source}")
        records.append(
            {
                "source": source,
                "destination": destination,
                "backup": backup,
                "had_existing": _path_exists(destination),
                "installed": False,
            }
        )
    transaction = {
        "backup_root": backup_root,
        "records": records,
        "cleanup_roots": list(cleanup_roots or []),
    }
    _write_transaction_journal(transaction, "prepared")
    try:
        for record in records:
            _write_transaction_journal(transaction, "installing")
            if record["had_existing"]:
                os.replace(record["destination"], record["backup"])
            if record["source"] is not None:
                os.replace(record["source"], record["destination"])
                record["installed"] = True
            _write_transaction_journal(transaction, "installing")
        _write_transaction_journal(transaction, "installed")
    except Exception:
        for record in reversed(records):
            destination = record["destination"]
            source = record["source"]
            backup = record["backup"]
            if record["installed"] and _path_exists(destination):
                assert source is not None
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
            if record["had_existing"] and _path_exists(backup):
                os.replace(backup, destination)
        _remove_owned_path(backup_root)
        raise
    return transaction


def _rollback_staged_items(transaction: Dict[str, Any]) -> None:
    for record in reversed(transaction["records"]):
        destination: Path = record["destination"]
        source: Optional[Path] = record["source"]
        backup: Path = record["backup"]
        if record["installed"] and _path_exists(destination):
            assert source is not None
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, source)
        if record["had_existing"] and _path_exists(backup):
            os.replace(backup, destination)
    _remove_owned_path(transaction["backup_root"])


def _commit_staged_items(transaction: Dict[str, Any], cleanup_roots: List[Path]) -> None:
    transaction["cleanup_roots"] = list(cleanup_roots)
    _write_transaction_journal(transaction, "committed")
    cleanup_failed = False
    for path in cleanup_roots:
        try:
            _remove_owned_path(path)
        except Exception as exc:
            cleanup_failed = True
            log_warn(f"Could not remove owned conversion staging/backup path {path}: {exc}")
    if cleanup_failed:
        log_warn(
            f"Keeping committed transaction journal for cleanup on the next run: {transaction['backup_root']}"
        )
        return
    try:
        _remove_owned_path(transaction["backup_root"])
    except Exception as exc:
        log_warn(f"Could not remove owned conversion backup path {transaction['backup_root']}: {exc}")


def _build_resized_images(layout_root: Path) -> List[str]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("--resize requires Pillow before conversion outputs can be committed") from exc
    images_dir = layout_root / "images"
    image_files = [path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    derived_names: List[str] = []
    for scale in (2, 4, 8):
        derived_name = f"images_{scale}"
        derived_dir = layout_root / derived_name
        derived_dir.mkdir(parents=True, exist_ok=False)
        derived_names.append(derived_name)
        for source in image_files:
            relative = source.relative_to(images_dir)
            destination = derived_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                width, height = image.size
                resized = image.resize((max(1, width // scale), max(1, height // scale)), Image.LANCZOS)
                resized.save(destination)
    return derived_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source_path", required=True, help="Dataset root (contains input/)")
    parser.add_argument("--colmap_executable", default="colmap", help="Path/name of colmap executable")
    parser.add_argument("--exiftool_executable", default="exiftool", help="Path/name of exiftool executable")
    parser.add_argument(
        "--required_colmap_version",
        default="4.1.0",
        help="Required COLMAP version string (default: 4.1.0; empty disables the version check).",
    )
    parser.add_argument(
        "--required_colmap_sha256",
        default="",
        help="Optional exact SHA256 pin for the COLMAP executable (recommended for formal runs).",
    )
    parser.add_argument(
        "--require_cuda_colmap", dest="require_cuda_colmap", action="store_true",
        help="Require the selected COLMAP executable to report CUDA support (default: on).",
    )
    parser.add_argument(
        "--allow_non_cuda_colmap", dest="require_cuda_colmap", action="store_false",
        help="Explicitly allow a non-CUDA COLMAP executable (not permitted for formal UAV-FGS runs).",
    )
    parser.set_defaults(require_cuda_colmap=True)
    parser.add_argument(
        "--require_global_mapper_cudss",
        dest="require_global_mapper_cudss",
        action="store_true",
        help="On Linux, require GlobalMapper's runtime to resolve libcudss (default: on).",
    )
    parser.add_argument(
        "--allow_global_mapper_without_cudss",
        dest="require_global_mapper_cudss",
        action="store_false",
        help="Allow a GlobalMapper runtime without libcudss (not for formal runs).",
    )
    parser.set_defaults(require_global_mapper_cudss=True)
    parser.add_argument(
        "--database_policy",
        choices=["reset", "reuse_verified", "adopt_legacy"],
        default="reset",
        help=(
            "COLMAP database handling: reset builds a fresh DB (formal default); reuse_verified requires an "
            "exact provenance match; adopt_legacy explicitly adopts an unproven DB after inventory checks."
        ),
    )
    parser.add_argument(
        "--expected_legacy_database_sha256",
        default="",
        help="Required exact pre-adoption database.db SHA256 when --database_policy=adopt_legacy.",
    )
    parser.add_argument(
        "--allow_replace_unverified_outputs",
        action="store_true",
        help="Explicitly allow replacing pre-existing derived output paths without a valid completion manifest.",
    )
    parser.add_argument(
        "--wgs84_code", type=int, default=0,
        help="Integer code for PosePrior coordinate_system=WGS84. Your COLMAP error showed WGS84==0."
    )

    parser.add_argument(
        "--prior_position_std_m", type=float, default=1.0,
        help="Write finite diagonal pose-prior covariance from this GPS std-dev in meters (default: 1.0)."
    )
    parser.add_argument(
        "--swap_latlon", action="store_true",
        help="Swap lat/lon when writing pose priors (debug option if alignment fails due to ordering)."
    )


    parser.add_argument("--camera", default="SIMPLE_RADIAL")
    parser.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    parser.add_argument("--matcher_args", default="")
    parser.add_argument("--colmap_gpu_index", type=int, default=0, help="GPU index locked for COLMAP extraction, matching, and global SfM (default: 0).")
    parser.add_argument("--sfm_mapper", default="global", choices=["global", "incremental"], help="SfM mapper mode (default: global; incremental is explicit opt-in only).")
    parser.add_argument("--global_mapper_args", default="", help="Additional COLMAP global_mapper options; protocol-locked GPU/random-seed options cannot be overridden here.")
    parser.add_argument("--global_mapper_random_seed", type=int, default=0, help="Deterministic COLMAP GlobalMapper seed (default: 0).")
    parser.add_argument("--mapper_multiple_models", type=int, default=1)
    parser.add_argument("--min_model_size", type=int, default=10)
    parser.add_argument("--init_min_num_inliers", type=int, default=100)
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=30)

    parser.add_argument("--use_model_aligner", dest="use_model_aligner", action="store_true")
    parser.add_argument("--no_use_model_aligner", dest="use_model_aligner", action="store_false")
    parser.set_defaults(use_model_aligner=True)
    parser.add_argument(
        "--model_aligner_args",
        default="--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0",
        help="COLMAP model_aligner options (default: GPS -> local ENU, 30m RANSAC threshold).",
    )

    parser.add_argument("--resize", action="store_true")

    args = parser.parse_args()

    root = Path(args.source_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {root}")
    conversion_lock = acquire_conversion_lock(root)
    _OWNED_TEMP_PATHS.append(conversion_lock)
    recover_incomplete_output_transactions(root)
    input_dir = root / "input"
    distorted_dir = root / "distorted"
    stable_db_path = distorted_dir / "database.db"
    stable_database_provenance_path = distorted_dir / DATABASE_PROVENANCE_NAME

    if not input_dir.is_dir():
        raise FileNotFoundError(f"COLMAP input directory is missing: {input_dir}")
    if int(args.min_model_size) < 3:
        raise ValueError("--min_model_size must be >= 3")
    input_inventory = build_file_inventory(input_dir, image_files_only=True)
    if input_inventory["count"] < int(args.min_model_size):
        raise RuntimeError(
            f"input/ has only {input_inventory['count']} images, fewer than --min_model_size={args.min_model_size}"
        )

    colmap_exe = args.colmap_executable
    colmap_runtime = verify_colmap_runtime(
        colmap_exe,
        required_version=str(args.required_colmap_version).strip(),
        require_cuda=bool(args.require_cuda_colmap),
        required_sha256=str(args.required_colmap_sha256),
        require_cudss=(
            args.sfm_mapper == "global"
            and bool(args.require_global_mapper_cudss)
            and bool(args.require_cuda_colmap)
        ),
    )
    # Execute the exact absolute binary whose bytes/version/runtime linkage were
    # just verified.  This prevents a later PATH lookup from selecting another
    # executable between preflight and the protocol-critical commands.
    colmap_exe = str(colmap_runtime["resolved_path"])

    verify_existing_output_ownership(
        root,
        allow_unverified=bool(args.allow_replace_unverified_outputs),
        include_resized_outputs=bool(args.resize),
    )

    distorted_dir.mkdir(parents=True, exist_ok=True)
    run_token = f"{os.getpid()}_{int(time.time() * 1000)}"
    run_workspace = distorted_dir / f".convert_uavfgs_run_{run_token}"
    layout_staging = root / f".convert_uavfgs_layout_{run_token}"
    backup_root = root / f".convert_uavfgs_backup_{run_token}"
    # Staging paths are disposable on an ordinary failure.  The backup path is
    # deliberately excluded: once publication begins it is governed only by
    # the transaction journal, so a secondary rollback error cannot cause the
    # generic exception cleanup to delete the last copy of prior outputs.
    _OWNED_TEMP_PATHS.extend([run_workspace, layout_staging])
    run_workspace.mkdir(parents=True, exist_ok=False)
    layout_staging.mkdir(parents=True, exist_ok=False)
    db_path = run_workspace / "database.db"
    database_provenance_path = run_workspace / DATABASE_PROVENANCE_NAME

    gpu_index = str(args.colmap_gpu_index)
    matcher_cmd = build_matcher_command(
        colmap_exe,
        str(args.matching),
        db_path,
        int(args.colmap_gpu_index),
        str(args.matcher_args),
    )

    feature_cmd = [
        colmap_exe, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(input_dir),
        "--ImageReader.camera_model", str(args.camera),
        "--ImageReader.single_camera", "1",
        "--FeatureExtraction.use_gpu", "1",
        "--FeatureExtraction.gpu_index", gpu_index,
    ]
    canonical_feature_cmd = [
        str(stable_db_path) if str(value) == str(db_path) else str(value)
        for value in feature_cmd
    ]
    canonical_matcher_cmd = [
        str(stable_db_path) if str(value) == str(db_path) else str(value)
        for value in matcher_cmd
    ]
    database_core = _database_provenance_core(
        input_inventory=input_inventory,
        colmap_runtime=colmap_runtime,
        feature_command=canonical_feature_cmd,
        matcher_command=canonical_matcher_cmd,
        camera_model=args.camera,
        prior_position_std_m=args.prior_position_std_m,
        wgs84_code=args.wgs84_code,
        swap_latlon=args.swap_latlon,
    )
    database_contract = database_policy_contract(str(args.database_policy))

    reuse_verified_database = False
    adopted_legacy_database = False
    stable_source_database_sha256 = ""
    if args.database_policy == "reset":
        log_info("Building a fresh COLMAP database in the private run workspace")
    elif args.database_policy == "reuse_verified":
        reused_provenance = _validate_verified_database_reuse(
            database_path=stable_db_path,
            provenance_path=stable_database_provenance_path,
            expected_core=database_core,
        )
        stable_source_database_sha256 = str(reused_provenance["database_sha256"])
        _copy_closed_sqlite_database(stable_db_path, db_path)
        reuse_verified_database = True
        log_info("Reusing a private snapshot of the COLMAP database after exact provenance/SHA verification")
    else:
        if not stable_db_path.is_file():
            raise RuntimeError("--database_policy=adopt_legacy requires an existing distorted/database.db")
        _assert_closed_sqlite_database(stable_db_path)
        expected_legacy_sha = str(args.expected_legacy_database_sha256 or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_legacy_sha):
            raise ValueError(
                "--database_policy=adopt_legacy requires --expected_legacy_database_sha256 with 64 hex characters"
            )
        actual_legacy_sha = _sha256_file(stable_db_path)
        if actual_legacy_sha != expected_legacy_sha:
            raise RuntimeError(
                "Legacy COLMAP database SHA256 mismatch before adoption: "
                f"expected={expected_legacy_sha} actual={actual_legacy_sha} path={stable_db_path}"
            )
        stable_source_database_sha256 = actual_legacy_sha
        legacy_names = _database_image_names(stable_db_path, immutable=True)
        expected_input_names = sorted(str(name) for name in input_inventory["names"])
        if legacy_names != expected_input_names:
            raise RuntimeError(
                "Legacy COLMAP database image inventory differs from input/ before adoption; "
                "use --database_policy=reset."
            )
        _copy_closed_sqlite_database(stable_db_path, db_path)
        adopted_legacy_database = True
        log_warn(
            "Explicitly adopting a SHA-pinned legacy COLMAP database; provenance will record the adoption"
        )

    if database_contract["run_feature_extractor"]:
        run_cmd(
            feature_cmd,
            forbidden_output_patterns=[
                r"Creating SIFT CPU feature extractor",
                r"Falling back to CPU",
            ],
        )

    gps_map = populate_pose_priors_from_exif(
        db_path,
        input_dir,
        exiftool_exe=args.exiftool_executable,
        wgs84_code=args.wgs84_code,
        prior_position_std_m=args.prior_position_std_m,
        swap_latlon=args.swap_latlon,
    )

    if database_contract["run_matcher"]:
        run_cmd(
            matcher_cmd,
            forbidden_output_patterns=[
                r"Creating SIFT CPU feature matcher",
                r"Falling back to CPU",
            ],
        )

    input_names = sorted(str(name) for name in input_inventory["names"])
    database_hash_lifecycle: Dict[str, Any] = {
        "source_policy": str(args.database_policy),
        "stable_source_database_sha256": stable_source_database_sha256,
    }
    database_hash_lifecycle["after_database_selection_and_pose_priors_sha256"] = (
        _finalize_private_sqlite_database(db_path)
    )
    mapping_database_summary = _alignment_database_semantic_summary(db_path)
    mapping_counts = mapping_database_summary["table_counts"]
    for table in ("keypoints", "descriptors"):
        if int(mapping_counts.get(table, 0)) < int(args.min_model_size):
            raise RuntimeError(
                f"COLMAP database has too few {table} rows for mapping: "
                f"{mapping_counts.get(table, 0)} < {args.min_model_size}"
            )
    if int(mapping_counts.get("two_view_geometries", 0)) <= 0:
        raise RuntimeError("COLMAP database has no verified two-view geometries")
    database_names = _database_image_names(db_path, immutable=True)
    if database_names != input_names:
        missing = sorted(set(input_names) - set(database_names))[:10]
        extra = sorted(set(database_names) - set(input_names))[:10]
        raise RuntimeError(
            "COLMAP database image inventory does not exactly match input/: "
            f"input={len(input_names)} database={len(database_names)} missing={missing} extra={extra}. "
            "Use --database_policy=reset."
        )
    sparse_root = run_workspace / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=False)
    sparse_aligned = run_workspace / "sparse_aligned"

    if args.sfm_mapper == "global":
        global_cmd = build_global_mapper_command(
            colmap_exe=colmap_exe,
            database_path=db_path,
            image_path=input_dir,
            output_path=sparse_root,
            gpu_index=args.colmap_gpu_index,
            random_seed=args.global_mapper_random_seed,
            extra_args=args.global_mapper_args,
        )
        log_info("SfM mapper mode: COLMAP global_mapper (incremental fallback is disabled)")
        run_cmd(
            global_cmd,
            forbidden_output_patterns=GLOBAL_MAPPER_FORBIDDEN_OUTPUT_PATTERNS,
        )
    else:
        log_warn("SfM mapper mode: incremental mapper was explicitly requested by --sfm_mapper=incremental")
        run_cmd([
            colmap_exe, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(input_dir),
            "--output_path", str(sparse_root),
            "--Mapper.multiple_models", str(args.mapper_multiple_models),
            "--Mapper.min_model_size", str(args.min_model_size),
            "--Mapper.init_min_num_inliers", str(args.init_min_num_inliers),
            "--Mapper.abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
        ])

    database_hash_lifecycle["after_mapper_before_model_aligner_sha256"] = (
        _finalize_private_sqlite_database(db_path)
    )
    published_database_semantics = _alignment_database_semantic_summary(db_path)

    best_model = select_best_sparse_model(sparse_root)
    registered_images, _ = validate_sparse_model(
        best_model,
        min_registered_images=args.min_model_size,
        mapper_mode=args.sfm_mapper,
    )
    best_model_names = sorted(read_image_names(best_model))
    if len(best_model_names) != registered_images or not set(best_model_names).issubset(set(input_names)):
        raise RuntimeError(
            "Selected SfM model names are inconsistent with input/: "
            f"header={registered_images} names={len(best_model_names)} input={len(input_names)}"
        )
    formal_all_inputs = int(args.min_model_size) == len(input_names)
    if formal_all_inputs and best_model_names != input_names:
        raise RuntimeError(
            "Formal all-input registration is not exact in the source SfM model: "
            f"registered={len(best_model_names)} input={len(input_names)}"
        )

    sanity_check_overlap(best_model, gps_map)

    aligned_model_for_undistort = best_model
    transform_path = run_workspace / "model_alignment_transform.txt"
    audit_path = run_workspace / "model_alignment_audit.json"
    alignment_payload: Optional[Dict[str, Any]] = None

    if args.use_model_aligner:
        sparse_aligned.mkdir(parents=True, exist_ok=False)
        aligner_database_path = run_workspace / "model_aligner_database.db"
        _copy_closed_sqlite_database(db_path, aligner_database_path)
        aligner_pre_sha = _finalize_private_sqlite_database(aligner_database_path)
        aligner_pre_semantics = _alignment_database_semantic_summary(aligner_database_path)
        aligner_cmd, alignment_max_error, min_common_images = build_model_aligner_command(
            colmap_exe=colmap_exe,
            input_path=best_model,
            output_path=sparse_aligned,
            database_path=aligner_database_path,
            transform_path=transform_path,
            extra_args=args.model_aligner_args,
        )
        log_info("Running model_aligner once (fail-closed; no threshold relaxation):")
        run_cmd(aligner_cmd)
        aligner_post_sha = _finalize_private_sqlite_database(aligner_database_path)
        aligner_post_semantics = _alignment_database_semantic_summary(aligner_database_path)
        if aligner_post_semantics != aligner_pre_semantics:
            raise RuntimeError(
                "model_aligner changed image/pose-prior/table-count semantics in its isolated database copy"
            )
        database_hash_lifecycle["isolated_model_aligner_database"] = {
            "enabled": True,
            "published": False,
            "pre_sha256": aligner_pre_sha,
            "post_sha256": aligner_post_sha,
            "physical_sha256_changed": bool(aligner_pre_sha != aligner_post_sha),
            "pre_semantic_sha256": aligner_pre_semantics["semantic_sha256"],
            "post_semantic_sha256": aligner_post_semantics["semantic_sha256"],
            "semantics_unchanged": True,
        }
        alignment_payload = write_enu_alignment_audit(
            database_path=db_path,
            aligned_model=sparse_aligned,
            gps_by_basename=gps_map,
            transform_path=transform_path,
            report_path=audit_path,
            alignment_max_error=alignment_max_error,
            min_common_images=min_common_images,
            expected_image_names=input_names if formal_all_inputs else None,
        )
        # Publish-stable paths belong in the audit even though it is built in a
        # private run workspace and moved only after every output validates.
        alignment_payload["transform_path"] = str(distorted_dir / "model_alignment_transform.txt")
        alignment_payload["aligned_model_path"] = str(distorted_dir / "sparse_aligned")
        _atomic_write_json(audit_path, alignment_payload)
        aligned_model_for_undistort = sparse_aligned
    else:
        database_hash_lifecycle["isolated_model_aligner_database"] = {
            "published": False,
            "enabled": False,
        }

    run_cmd([
        colmap_exe, "image_undistorter",
        "--image_path", str(input_dir),
        "--input_path", str(aligned_model_for_undistort),
        "--output_path", str(layout_staging),
        "--output_type", "COLMAP",
    ])

    model0 = ensure_3dgs_sparse_layout(layout_staging)
    export_model_as_txt(colmap_exe, model0)
    ensure_camera_models_supported_for_3dgs(colmap_exe, model0)
    export_model_as_txt(colmap_exe, model0)  # keep TXT after possible BIN regen
    staged_outputs = validate_undistorted_layout(
        layout_staging,
        min_registered_images=args.min_model_size,
    )
    resized_names = _build_resized_images(layout_staging) if args.resize else []

    # The input directory is external to this transaction. Refuse to publish a
    # model if it changed while SfM was running.
    if build_file_inventory(input_dir, image_files_only=True) != input_inventory:
        raise RuntimeError("input/ changed during COLMAP reconstruction; staged outputs will not be published")

    published_database_sha256 = _finalize_private_sqlite_database(db_path)
    if published_database_sha256 != database_hash_lifecycle["after_mapper_before_model_aligner_sha256"]:
        raise RuntimeError("The publishable COLMAP database changed after its pre-aligner identity was frozen")
    if _alignment_database_semantic_summary(db_path) != published_database_semantics:
        raise RuntimeError("The publishable COLMAP database semantics changed after mapping")
    database_hash_lifecycle["published_database_sha256"] = published_database_sha256
    if _database_image_names(db_path, immutable=True) != input_names:
        raise RuntimeError("COLMAP database image inventory changed during mapping/alignment")
    feature_match_origin = {
        "kind": database_contract["feature_match_origin_kind"],
        "current_feature_command_executed": bool(database_contract["run_feature_extractor"]),
        "current_matcher_command_executed": bool(database_contract["run_matcher"]),
    }
    if stable_source_database_sha256:
        feature_match_origin["source_database_sha256"] = stable_source_database_sha256
    reuse_verified_eligible = bool(database_contract["reuse_verified_eligible"])
    database_provenance = _write_database_provenance(
        database_path=db_path,
        provenance_path=database_provenance_path,
        core=database_core,
        adopted_legacy=adopted_legacy_database,
        hash_lifecycle=database_hash_lifecycle,
        feature_match_origin=feature_match_origin,
        reuse_verified_eligible=reuse_verified_eligible,
    )

    alignment_manifest: Dict[str, Any] = {"enabled": bool(args.use_model_aligner)}
    if args.use_model_aligner:
        alignment_manifest.update(
            {
                "audit_sha256": _sha256_file(audit_path),
                "transform_sha256": _sha256_file(transform_path),
                "aligned_model_inventory": build_file_inventory(sparse_aligned),
                "coordinate_scope": "relative_local_enu_only",
                "absolute_geolocation_validated": False,
                "image_embedded_gps_values_sha256": alignment_payload[
                    "image_embedded_gps_values_sha256"
                ],
            }
        )
    arguments = [str(value) for value in sys.argv[1:]]
    completion_payload: Dict[str, Any] = {
        "schema_version": COMPLETION_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(root.resolve()),
        "arguments": arguments,
        "arguments_sha256": _canonical_sha256(arguments),
        "colmap_runtime": colmap_runtime,
        "runtime_requirements": {
            "require_cuda_colmap": bool(args.require_cuda_colmap),
            "require_global_mapper_cudss": bool(args.require_global_mapper_cudss),
            "cudss_runtime_check_required": bool(
                args.sfm_mapper == "global"
                and args.require_cuda_colmap
                and args.require_global_mapper_cudss
                and sys.platform.startswith("linux")
            ),
        },
        "input_inventory": input_inventory,
        "database": {
            "policy": str(args.database_policy),
            "database_sha256": database_provenance["database_sha256"],
            "provenance_sha256": _sha256_file(database_provenance_path),
            "adopted_legacy_database": bool(adopted_legacy_database),
            "hash_lifecycle": database_hash_lifecycle,
            "feature_match_origin": feature_match_origin,
            "reuse_verified_eligible": reuse_verified_eligible,
        },
        "mapper_mode": str(args.sfm_mapper),
        "min_registered_images": int(args.min_model_size),
        "alignment": alignment_manifest,
        "outputs": staged_outputs,
        "resized_output_names": resized_names,
    }
    staged_completion_manifest = run_workspace / COMPLETION_MANIFEST_NAME
    _atomic_write_json(staged_completion_manifest, completion_payload)

    swap_items: List[Tuple[Optional[Path], Path]] = [
        (layout_staging / "images", root / "images"),
        (layout_staging / "sparse", root / "sparse"),
        (layout_staging / "stereo", root / "stereo"),
    ]
    swap_items.extend((layout_staging / name, root / name) for name in resized_names)
    swap_items.append((sparse_root, distorted_dir / "sparse"))
    if args.use_model_aligner:
        swap_items.extend(
            [
                (sparse_aligned, distorted_dir / "sparse_aligned"),
                (transform_path, distorted_dir / "model_alignment_transform.txt"),
                (audit_path, distorted_dir / "model_alignment_audit.json"),
            ]
        )
    else:
        swap_items.extend(
            [
                (None, distorted_dir / "sparse_aligned"),
                (None, distorted_dir / "model_alignment_transform.txt"),
                (None, distorted_dir / "model_alignment_audit.json"),
            ]
        )
    swap_items.extend(
        [
            (db_path, stable_db_path),
            (database_provenance_path, stable_database_provenance_path),
            (None, Path(str(stable_db_path) + "-wal")),
            (None, Path(str(stable_db_path) + "-shm")),
            (None, Path(str(stable_db_path) + "-journal")),
        ]
    )
    swap_items.append((staged_completion_manifest, _completion_manifest_path(root)))

    transaction = _swap_staged_items(
        swap_items,
        backup_root=backup_root,
        cleanup_roots=[run_workspace, layout_staging],
    )
    try:
        verified, reason, _ = validate_conversion_completion_manifest(
            root,
            expected_arguments=arguments,
            expected_colmap_sha256=colmap_runtime["executable_sha256"],
            expected_min_registered_images=args.min_model_size,
        )
        if not verified:
            raise RuntimeError(f"Published conversion failed completion-manifest validation: {reason}")
    except Exception:
        _rollback_staged_items(transaction)
        raise
    _commit_staged_items(transaction, [run_workspace, layout_staging])
    log_info(
        f"3DGS conversion committed atomically: images={root/'images'} | sparse_model={root/'sparse'/'0'} | "
        f"manifest={_completion_manifest_path(root)}"
    )
    conversion_lock.unlink(missing_ok=True)
    log_info("Done.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        for owned_path in reversed(_OWNED_TEMP_PATHS):
            try:
                _remove_owned_path(owned_path)
            except Exception as cleanup_error:
                log_warn(f"Could not clean owned temporary path {owned_path}: {cleanup_error}")
        log_err(f"Command failed: {e}")
        sys.exit(e.returncode)
    except Exception as e:
        for owned_path in reversed(_OWNED_TEMP_PATHS):
            try:
                _remove_owned_path(owned_path)
            except Exception as cleanup_error:
                log_warn(f"Could not clean owned temporary path {owned_path}: {cleanup_error}")
        log_err(str(e))
        raise

