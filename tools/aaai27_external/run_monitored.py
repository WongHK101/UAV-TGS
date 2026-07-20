#!/usr/bin/env python3
"""Run one external-baseline process with lightweight wall/VRAM accounting."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Sequence


SCHEMA = "uav-tgs-aaai27-external-process-receipt-v1"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_text(command: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def gpu_identity() -> dict[str, object] | None:
    raw = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if not raw:
        return None
    fields = [field.strip() for field in raw.splitlines()[0].split(",")]
    if len(fields) != 4:
        return {"raw": raw}
    return {
        "index": int(fields[0]),
        "name": fields[1],
        "driver_version": fields[2],
        "memory_total_mib": float(fields[3]),
    }


def process_gpu_memory_mib(pid: int) -> float | None:
    raw = _run_text(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        return None
    total = 0.0
    found = False
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            process_id = int(fields[0])
            memory = float(fields[1])
        except ValueError:
            continue
        if process_id == pid:
            found = True
            total += memory
    return total if found else 0.0


def run(
    *,
    command: Sequence[str],
    cwd: Path,
    receipt_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    poll_seconds: float,
) -> dict[str, object]:
    if not command:
        raise ValueError("command is empty")
    cwd = cwd.resolve()
    if not cwd.is_dir():
        raise FileNotFoundError(cwd)
    for path in (receipt_path, stdout_path, stderr_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    started_at = now()
    started = time.perf_counter()
    identity = gpu_identity()
    peak_mib: float | None = None
    samples = 0
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(command), cwd=cwd, stdout=stdout, stderr=stderr, env=os.environ.copy()
        )
        while process.poll() is None:
            memory = process_gpu_memory_mib(process.pid)
            if memory is not None:
                peak_mib = memory if peak_mib is None else max(peak_mib, memory)
                samples += 1
            time.sleep(poll_seconds)
        return_code = int(process.returncode)
        memory = process_gpu_memory_mib(process.pid)
        if memory is not None:
            peak_mib = memory if peak_mib is None else max(peak_mib, memory)
            samples += 1
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "status": "SUCCEEDED" if return_code == 0 else "FAILED",
        "return_code": return_code,
        "started_at": started_at,
        "completed_at": now(),
        "wall_time_s": time.perf_counter() - started,
        "host": socket.gethostname(),
        "cwd": str(cwd),
        "command": list(command),
        "gpu": identity,
        "peak_process_vram_mib": peak_mib,
        "gpu_sample_count": samples,
        "stdout": {
            "path": str(stdout_path),
            "size_bytes": stdout_path.stat().st_size,
            "sha256": sha256_file(stdout_path),
        },
        "stderr": {
            "path": str(stderr_path),
            "size_bytes": stderr_path.stat().st_size,
            "sha256": sha256_file(stderr_path),
        },
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if args.poll_seconds <= 0:
        raise ValueError("poll interval must be positive")
    receipt = run(
        command=command,
        cwd=args.cwd,
        receipt_path=args.receipt,
        stdout_path=args.stdout,
        stderr_path=args.stderr,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "SUCCEEDED" else int(receipt["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())

