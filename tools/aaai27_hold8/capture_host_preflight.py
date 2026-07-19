#!/usr/bin/env python3
"""Capture a read-only 900/901 Hold-8 host preflight receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "uav-tgs-aaai27-hold8-host-preflight-v2"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def run(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def capture(*, host_id: str, project_root: Path, code_root: Path, expected_commit: str) -> dict[str, Any]:
    project_root = project_root.resolve()
    code_root = code_root.resolve()
    if host_id not in {"900", "901"}:
        raise ValueError("host-id must be 900 or 901")
    expected_root = Path(
        "/root/autodl-tmp/UAV-TGS" if host_id == "900" else "/root/autodl-tmp/UAV-TGS-901"
    )
    if project_root != expected_root:
        raise ValueError(f"project root mismatch for host {host_id}: {project_root}")
    expected_commit = str(expected_commit).lower()
    if COMMIT_RE.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a full lowercase 40-hex Git commit")
    if not project_root.is_dir() or not code_root.is_dir():
        raise FileNotFoundError(project_root if not project_root.is_dir() else code_root)
    try:
        code_root.relative_to(project_root)
    except ValueError as error:
        raise ValueError("code_root must remain inside the host's isolated project root") from error
    disk = shutil.disk_usage(project_root)
    commit = run(["git", "-C", str(code_root), "rev-parse", "HEAD"])
    dirty = run(["git", "-C", str(code_root), "status", "--porcelain=v1", "--untracked-files=all"])
    actual_commit = commit.get("stdout", "")
    python_probe = run(
        [
            str(project_root / "environments" / "uav-tgs" / "bin" / "python"),
            "-c",
            (
                "import json,torch; print(json.dumps({'torch':torch.__version__,"
                "'cuda_available':torch.cuda.is_available(),"
                "'cuda_version':torch.version.cuda,'device_count':torch.cuda.device_count()}))"
            ),
        ]
    )
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    tools = {}
    openmvs_bin = (
        project_root
        / "tools"
        / "openmvs-2.4.0-refine-fail-closed"
        / "bin"
        / "OpenMVS"
    )
    for name, path in {
        "colmap": project_root
        / "tools"
        / "colmap-4.1.0-cuda12.8-ceres2.3dev-cudss0.8-sm120"
        / "bin"
        / "colmap",
        "openmvs_interface_colmap": openmvs_bin / "InterfaceCOLMAP",
        "openmvs_densify": openmvs_bin / "DensifyPointCloud",
        "openmvs_reconstruct": openmvs_bin / "ReconstructMesh",
        "openmvs_refine": openmvs_bin / "RefineMesh",
    }.items():
        tools[name] = {"path": str(path), "exists": path.is_file(), "executable": os.access(path, os.X_OK)}
    status = "ready_no_gpu" if not gpu["available"] else "ready_gpu_visible"
    blockers: list[str] = []
    if actual_commit != expected_commit:
        blockers.append("code_commit_mismatch")
    if dirty.get("stdout"):
        blockers.append("dirty_worktree")
    if not python_probe["available"]:
        blockers.append("python_environment_unavailable")
    if not all(item["exists"] and item["executable"] for item in tools.values()):
        blockers.append("required_tool_unavailable")
    payload = {
        "schema": SCHEMA,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host_id": host_id,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "code": {"path": str(code_root), "commit": actual_commit, "expected_commit": expected_commit, "dirty": bool(dirty.get("stdout"))},
        "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "gpu_probe": gpu,
        "python_probe": python_probe,
        "tools": tools,
        "status": status if not blockers else "blocked",
        "blockers": blockers,
        "phase0_only": True,
        "formal_training_started": False,
    }
    basis = dict(payload)
    basis.pop("captured_at")
    payload["receipt_sha256"] = json_hash(basis)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-id", required=True, choices=("900", "901"))
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = capture(
        host_id=args.host_id,
        project_root=args.project_root,
        code_root=args.code_root,
        expected_commit=args.expected_commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "blockers": payload["blockers"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
