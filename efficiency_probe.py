"""Lightweight, opt-in efficiency measurements shared by UAV-FGS tools.

The module is intentionally importable without PyTorch or CUDA.  It provides:

* a boundary-only PyTorch training probe;
* a render-only benchmark helper with warm-up and CUDA events;
* a command wrapper usable by other repositories;
* offline artifact size and PLY Gaussian-count helpers.

No probe is active unless its caller explicitly enables it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
SCHEMA_NAME = "uav-tgs-efficiency"
MIB = 1024 * 1024


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def read_json_if_present(path: os.PathLike[str] | str) -> Optional[Dict[str, Any]]:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def ply_vertex_count(path: os.PathLike[str] | str) -> Optional[int]:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        with candidate.open("rb") as stream:
            for raw_line in stream:
                line = raw_line.decode("ascii", errors="ignore").strip()
                if line.startswith("element vertex "):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except Exception:
        return None
    return None


def artifact_record(path: os.PathLike[str] | str, label: Optional[str] = None) -> Dict[str, Any]:
    candidate = Path(path)
    record: Dict[str, Any] = {
        "label": label,
        "path": str(candidate),
        "exists": candidate.is_file(),
        "serialized_bytes": None,
        "gaussian_count": None,
    }
    if not candidate.is_file():
        return record
    try:
        record["serialized_bytes"] = int(candidate.stat().st_size)
    except Exception:
        pass
    if candidate.suffix.lower() == ".ply":
        record["gaussian_count"] = ply_vertex_count(candidate)
    return record


def timing_summary(total_seconds: Optional[float], count: int) -> Dict[str, Optional[float]]:
    if total_seconds is None or count <= 0:
        return {"total_s": total_seconds, "ms_per_item": None, "items_per_s": None}
    total = float(total_seconds)
    if (not math.isfinite(total)) or total <= 0.0:
        return {"total_s": total_seconds, "ms_per_item": None, "items_per_s": None}
    return {
        "total_s": total,
        "ms_per_item": total * 1000.0 / float(count),
        "items_per_s": float(count) / total,
    }


def _cuda_info(torch_module: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_index": None,
        "device_name": None,
        "peak_torch_allocated_bytes": None,
        "peak_torch_reserved_bytes": None,
    }
    try:
        cuda = torch_module.cuda
        if not bool(cuda.is_available()):
            return info
        device_index = int(cuda.current_device())
        info.update(
            {
                "cuda_available": True,
                "device_index": device_index,
                "device_name": str(cuda.get_device_name(device_index)),
            }
        )
    except Exception:
        pass
    return info


def _reset_torch_peak(torch_module: Any) -> bool:
    try:
        if not bool(torch_module.cuda.is_available()):
            return False
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()
        return True
    except Exception:
        return False


def _read_torch_peak(torch_module: Any, synchronize: bool) -> Dict[str, Any]:
    info = _cuda_info(torch_module)
    if not info["cuda_available"]:
        return info
    try:
        if synchronize:
            torch_module.cuda.synchronize()
        info["peak_torch_allocated_bytes"] = int(torch_module.cuda.max_memory_allocated())
        info["peak_torch_reserved_bytes"] = int(torch_module.cuda.max_memory_reserved())
    except Exception:
        pass
    return info


class TorchStageProbe:
    """Measure one PyTorch stage without touching its iteration hot path."""

    def __init__(
        self,
        enabled: bool,
        output_path: os.PathLike[str] | str,
        stage: str,
        torch_module: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_path = Path(output_path)
        self.stage = str(stage)
        self.torch = torch_module
        self.metadata = dict(metadata or {})
        self.started_at: Optional[str] = None
        self._start_perf: Optional[float] = None
        self._peak_reset = False

    def start(self) -> None:
        if not self.enabled:
            return
        self.started_at = now_iso()
        self._peak_reset = _reset_torch_peak(self.torch)
        self._start_perf = time.perf_counter()

    def finish(
        self,
        status: str,
        result: Optional[Mapping[str, Any]] = None,
        error: Optional[BaseException] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        ended_at = now_iso()
        device = _read_torch_peak(self.torch, synchronize=(status == "completed"))
        elapsed = None if self._start_perf is None else max(0.0, time.perf_counter() - self._start_perf)
        payload: Dict[str, Any] = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "kind": "training_stage",
            "status": str(status),
            "stage": self.stage,
            "started_at": self.started_at,
            "ended_at": ended_at,
            "wall_time_s": elapsed,
            "timing_scope": "training_call_boundary_including_model_setup_eval_and_save",
            "peak_memory_scope": "torch_caching_allocator",
            "peak_memory_reset_succeeded": bool(self._peak_reset),
            "device": device,
            "metadata": dict(self.metadata),
            "result": dict(result or {}),
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
        }
        try:
            atomic_write_json(self.output_path, payload)
        except Exception as write_error:
            print(f"[WARN] Efficiency probe could not write {self.output_path}: {write_error}", file=sys.stderr)
        return payload


def benchmark_render_calls(
    render_once: Callable[[Any], Any],
    views: Iterable[Any],
    repeats: int,
    warmup_views: int,
    torch_module: Any,
) -> Dict[str, Any]:
    """Benchmark pure render calls; the callback must not perform I/O."""

    view_list = list(views)
    if not view_list:
        raise ValueError("render benchmark requires at least one view")
    repeat_count = int(repeats)
    if repeat_count <= 0:
        raise ValueError("render benchmark repeats must be positive")
    warmup_count = max(0, int(warmup_views))

    cuda_available = False
    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception:
        cuda_available = False

    peak_reset = _reset_torch_peak(torch_module) if cuda_available else False
    for index in range(warmup_count):
        render_once(view_list[index % len(view_list)])
    if cuda_available:
        torch_module.cuda.synchronize()

    start_event = end_event = None
    if cuda_available:
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    if start_event is not None:
        start_event.record()
    for _ in range(repeat_count):
        for view in view_list:
            render_once(view)
    if end_event is not None:
        end_event.record()
        torch_module.cuda.synchronize()
    wall_total = max(0.0, time.perf_counter() - wall_start)

    cuda_total_ms: Optional[float] = None
    if start_event is not None and end_event is not None:
        try:
            cuda_total_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            cuda_total_ms = None

    timed_views = len(view_list) * repeat_count
    wall = timing_summary(wall_total, timed_views)
    cuda = timing_summary(None if cuda_total_ms is None else cuda_total_ms / 1000.0, timed_views)
    device = _read_torch_peak(torch_module, synchronize=False)
    return {
        "warmup_views": warmup_count,
        "num_unique_views": len(view_list),
        "repeats": repeat_count,
        "timed_views": timed_views,
        "timing_scope": "render_only_no_gt_no_cpu_transfer_no_io",
        "warmup_excluded": True,
        "io_excluded": True,
        "wall_total_s": wall["total_s"],
        "wall_ms_per_view": wall["ms_per_item"],
        "wall_fps": wall["items_per_s"],
        "cuda_event_total_ms": cuda_total_ms,
        "cuda_event_ms_per_view": cuda["ms_per_item"],
        "cuda_event_fps": cuda["items_per_s"],
        "peak_memory_scope": "torch_caching_allocator_after_model_load",
        "peak_memory_reset_succeeded": bool(peak_reset),
        "device": device,
    }


def _query_nvidia_smi_process_memory(pid: int) -> Optional[int]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    total_mib = 0.0
    found = False
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mib = float(parts[1].replace("MiB", "").strip())
        except Exception:
            continue
        if row_pid == int(pid):
            total_mib += used_mib
            found = True
    return int(total_mib * MIB) if found else None


def run_command_probe(
    command: Sequence[str],
    output_path: os.PathLike[str] | str,
    artifacts: Optional[Mapping[str, os.PathLike[str] | str]] = None,
    poll_gpu: bool = True,
    poll_interval_s: float = 1.0,
) -> int:
    if not command:
        raise ValueError("command probe requires a command")
    started_at = now_iso()
    start_perf = time.perf_counter()
    try:
        process = subprocess.Popen(list(command))
    except OSError as error:
        payload = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "kind": "command",
            "status": "launch_failed",
            "started_at": started_at,
            "ended_at": now_iso(),
            "wall_time_s": max(0.0, time.perf_counter() - start_perf),
            "command": list(command),
            "pid": None,
            "return_code": 127,
            "error": {"type": type(error).__name__, "message": str(error)},
            "peak_process_gpu_memory_sampled_bytes": None,
            "gpu_memory_sampling": {
                "backend": "nvidia-smi_direct_process_poll" if poll_gpu else "disabled",
                "interval_s": max(0.1, float(poll_interval_s)) if poll_gpu else None,
                "samples": 0,
                "note": "direct process PID only; launcher child processes are not included",
            },
            "artifacts": [artifact_record(path, label) for label, path in (artifacts or {}).items()],
        }
        try:
            atomic_write_json(output_path, payload)
        except Exception as write_error:
            print(f"[WARN] Efficiency probe could not write {output_path}: {write_error}", file=sys.stderr)
        return 127
    peak_process_gpu_bytes: Optional[int] = None
    sample_count = 0
    interval = max(0.1, float(poll_interval_s))
    try:
        while True:
            if poll_gpu:
                sampled = _query_nvidia_smi_process_memory(process.pid)
                if sampled is not None:
                    peak_process_gpu_bytes = sampled if peak_process_gpu_bytes is None else max(peak_process_gpu_bytes, sampled)
                    sample_count += 1
            try:
                return_code = int(process.wait(timeout=interval))
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        process.terminate()
        return_code = int(process.wait())
        raise
    finally:
        elapsed = max(0.0, time.perf_counter() - start_perf)
        payload = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "kind": "command",
            "status": "completed" if process.returncode == 0 else "failed",
            "started_at": started_at,
            "ended_at": now_iso(),
            "wall_time_s": elapsed,
            "command": list(command),
            "pid": int(process.pid),
            "return_code": process.returncode,
            "peak_process_gpu_memory_sampled_bytes": peak_process_gpu_bytes,
            "gpu_memory_sampling": {
                "backend": "nvidia-smi_direct_process_poll" if poll_gpu else "disabled",
                "interval_s": interval if poll_gpu else None,
                "samples": sample_count,
                "note": "direct process PID only; launcher children and short peaks between samples may be missed",
            },
            "artifacts": [artifact_record(path, label) for label, path in (artifacts or {}).items()],
        }
        try:
            atomic_write_json(output_path, payload)
        except Exception as write_error:
            print(f"[WARN] Efficiency probe could not write {output_path}: {write_error}", file=sys.stderr)
    return return_code


def _parse_artifacts(values: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"artifact must use LABEL=PATH: {value}")
        label, path = value.split("=", 1)
        if not label.strip() or not path.strip():
            raise ValueError(f"artifact must use LABEL=PATH: {value}")
        result[label.strip()] = path.strip()
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lightweight efficiency probe reusable across reconstruction methods.")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    run_parser = subparsers.add_parser("run", help="Measure an external command and optional artifacts.")
    run_parser.add_argument("--output", required=True, help="Output JSON path.")
    run_parser.add_argument("--artifact", action="append", default=[], metavar="LABEL=PATH")
    run_parser.add_argument("--gpu_poll_interval", type=float, default=1.0)
    run_parser.add_argument("--no_gpu_poll", action="store_true", default=False)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.subcommand == "run":
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("run requires a command after --")
        try:
            artifacts = _parse_artifacts(args.artifact)
        except ValueError as error:
            parser.error(str(error))
        return run_command_probe(
            command,
            args.output,
            artifacts=artifacts,
            poll_gpu=not args.no_gpu_poll,
            poll_interval_s=args.gpu_poll_interval,
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
