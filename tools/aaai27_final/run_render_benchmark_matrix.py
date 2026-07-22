#!/usr/bin/env python3
"""Prepare and sequentially run the frozen 8x6 render-only matrix on host 900."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from PIL import Image


SCENES = {
    "Building": 77,
    "InternalRoad": 70,
    "PVpanel": 37,
    "TransmissionTower": 85,
    "Urban20K": 94,
    "Orchard": 74,
}
METHODS = (
    "raw_f3",
    "scsp_refit_f3",
    "adaptive_opacity_scale_clamp",
    "thermalgaussian_ommg",
    "mmone",
    "thermal3dgs",
    "thermonerf",
    "physir_splat_sh",
)
DISPLAY = {
    "raw_f3": "Raw-F3",
    "scsp_refit_f3": "SCSP-Refit+F3",
    "adaptive_opacity_scale_clamp": "Adaptive Opacity+Scale-Clamp",
    "thermalgaussian_ommg": "ThermalGaussian-OMMG",
    "mmone": "MMOne",
    "thermal3dgs": "Thermal3D-GS",
    "thermonerf": "ThermoNeRF",
    "physir_splat_sh": "PhysIR-Splat-SH†",
}
ADAPTER_METHOD = {
    "thermalgaussian_ommg": "thermalgaussian_ommg",
    "mmone": "mmone",
    "thermal3dgs": "thermal3dgs",
    "thermonerf": "thermonerf",
    "physir_splat_sh": "physir_splat",
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for value in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(value)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{time.time_ns()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _model_root(project: Path, benchmark: Path, scene: str, method: str) -> tuple[Path, int]:
    if method in {"raw_f3", "adaptive_opacity_scale_clamp"}:
        return project / "experiments/aaai27_hold8_v2" / scene / "methods" / method / "model", 60000
    if method == "scsp_refit_f3":
        return project / "experiments/aaai27_hold8_v2" / scene / "methods/scsp_refit_f3/model", 65000
    if scene == "Building":
        base = project / "external_phase_a/experiments/Building"
        mapping = {
            "thermalgaussian_ommg": base / "ThermalGaussian-OMMG/formal/model",
            "mmone": base / "MMOne/formal/model",
            "thermal3dgs": base / "Thermal3D-GS/formal_stream_sync/model",
            "thermonerf": base / "ThermoNeRF/formal/model",
            "physir_splat_sh": base / "PhysIR-Splat/formal_default_sh_no_opacity_reset/model",
        }
        return mapping[method], 30000
    if method in {"thermalgaussian_ommg", "physir_splat_sh"}:
        base = project / "external_phase_b/experiments" / scene
        mapping = {
            "thermalgaussian_ommg": base / "ThermalGaussian-OMMG/formal/model",
            "physir_splat_sh": base / "PhysIR-Splat/formal_default_sh_no_opacity_reset/model",
        }
        return mapping[method], 30000
    base = benchmark / "benchmark_assets/host_901/external_phase_b/experiments" / scene
    mapping = {
        "mmone": base / "MMOne/formal/model",
        "thermal3dgs": base / "Thermal3D-GS/formal_stream_sync/model",
        "thermonerf": base / "ThermoNeRF/formal/model",
    }
    return mapping[method], 30000


def _source_repo(project: Path, method: str) -> Path:
    mapping = {
        "raw_f3": project / "code",
        "scsp_refit_f3": project / "code",
        "adaptive_opacity_scale_clamp": project / "code",
        "thermalgaussian_ommg": project / "external_phase_a/sources/ThermalGaussian-OMMG",
        "mmone": project / "external_phase_a/sources/MMOne",
        "thermal3dgs": project / "external_phase_a/sources/Thermal3DGS",
        "thermonerf": project / "external_phase_a/sources/thermo-nerf",
        "physir_splat_sh": project / "external_phase_a/sources/physir-splat",
    }
    return mapping[method]


def _runtime_pythonpath(project: Path, method: str) -> list[Path]:
    """Return the frozen method-specific extension path used by formal runs.

    Several GS baselines expose the same Python module name while requiring
    different CUDA rasterizer ABIs.  Keeping the path on the clean child
    process avoids mutating the shared UAV-TGS environment or importing a
    different method's rasterizer.
    """
    base = project / "external_phase_a/sources"
    mapping = {
        "thermalgaussian_ommg": [
            base / "ThermalGaussian-OMMG/submodules/diff-gaussian-rasterization",
        ],
        "mmone": [
            base / "MMOne/submodules/diff-gaussian-rasterization",
        ],
        "thermal3dgs": [
            base / "Thermal3DGS/submodules/depth-diff-gaussian-rasterization",
        ],
        "physir_splat_sh": [
            project / "external_phase_a/environments/physir_default_sh/site",
        ],
    }
    return mapping.get(method, [])


def _building_reference(project: Path, scene: str, method: str, iteration: int, first: str) -> Path | None:
    if scene != "Building":
        return None
    if method in {"raw_f3", "scsp_refit_f3", "adaptive_opacity_scale_clamp"}:
        return (
            project
            / "experiments/aaai27_hold8_v2/Building/methods"
            / method
            / "model/test"
            / f"ours_{iteration}/renders"
            / first
        )
    base = project / "external_phase_a/experiments/Building"
    if method == "thermalgaussian_ommg":
        return base / "ThermalGaussian-OMMG/formal/normalized_thermal/test/ours_formal/renders" / first
    if method == "mmone":
        return base / "MMOne/formal/normalized_thermal/test/ours_formal/renders" / first
    if method == "thermal3dgs":
        return (
            base
            / "Thermal3D-GS/from_901_stream_sync/extracted/normalized_thermal/test/ours_formal/renders"
            / first
        )
    if method == "thermonerf":
        return base / "ThermoNeRF/from_901/extracted/normalized_thermal/test/ours_formal/renders" / first
    if method == "physir_splat_sh":
        return (
            base
            / "PhysIR-Splat/formal_default_sh_no_opacity_reset/model/test"
            / f"ours_{iteration}/renders/00000.png"
        )
    raise ValueError(method)


def _prepare_adapters(project: Path, benchmark: Path, python: Path) -> None:
    for scene in SCENES:
        for method, adapter in ADAPTER_METHOD.items():
            target = benchmark / "datasets" / scene / adapter
            manifest = target / "adapter_manifest.json"
            if manifest.is_file():
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if (
                    payload.get("scene") == scene
                    and payload.get("method") == adapter
                    and payload.get("test_count") == SCENES[scene]
                    and payload.get("protocol") == "aaai27_hold8_v2"
                ):
                    continue
                raise ValueError(f"stale benchmark adapter: {manifest}")
            if target.exists():
                raise FileExistsError(f"refusing incomplete adapter directory: {target}")
            command = [
                str(python),
                str(project / "code/tools/aaai27_external/building_adapter.py"),
                "--project-root",
                str(project),
                "--scene",
                scene,
                "--method",
                adapter,
                "--output",
                str(target),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)


def _matrix(project: Path, benchmark: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    uav_python = project / "environments/uav-tgs/bin/python"
    thermo_python = project / "external_phase_a/environments/thermonerf/bin/python"
    # Scene-major order completes all eight Building correctness checks before
    # any non-Building timing job starts.
    for scene, count in SCENES.items():
        for method in METHODS:
            model_root, iteration = _model_root(project, benchmark, scene, method)
            source = _source_repo(project, method)
            runtime_pythonpath = _runtime_pythonpath(project, method)
            test_list = project / "derived/aaai27_hold8_v2" / scene / "runtime_lists/thermal_test_list.txt"
            first = next(line.strip() for line in test_list.read_text(encoding="utf-8").splitlines() if line.strip())
            thermal_root = project / "derived/aaai27_hold8_v2" / scene / "thermal_benchmark"
            if method in INTERNAL_METHODS:
                dataset = thermal_root
                adapter_manifest = None
                cfg_text = (model_root / "cfg_args").read_text(encoding="utf-8")
                if "resolution=4" not in cfg_text:
                    raise ValueError(f"internal formal endpoint is not frozen at resolution=4: {model_root}")
                first_gt = next((thermal_root / "images").glob("*"))
                with Image.open(first_gt) as image:
                    width, height = round(image.width / 4), round(image.height / 4)
            else:
                dataset = benchmark / "datasets" / scene / ADAPTER_METHOD[method]
                adapter_manifest = dataset / "adapter_manifest.json"
                first_gt = next((thermal_root / "images").glob("*"))
                with Image.open(first_gt) as image:
                    width, height = image.size
            reference = _building_reference(project, scene, method, iteration, first)
            for required in (model_root, source, dataset, test_list):
                if not required.exists():
                    raise FileNotFoundError(required)
            for required in runtime_pythonpath:
                if not required.is_dir():
                    raise FileNotFoundError(required)
            if adapter_manifest is not None and not adapter_manifest.is_file():
                raise FileNotFoundError(adapter_manifest)
            if reference is not None and not reference.is_file():
                raise FileNotFoundError(reference)
            output = benchmark / "runtime/results" / scene / f"{method}.json"
            log = benchmark / "runtime/logs" / f"{scene}_{method}.log"
            command = [
                str(thermo_python if method == "thermonerf" else uav_python),
                str(benchmark / "runtime/benchmark_render_only.py"),
                "--method",
                method,
                "--display-name",
                DISPLAY[method],
                "--scene",
                scene,
                "--source-repo",
                str(source),
                "--model-root",
                str(model_root),
                "--dataset-path",
                str(dataset),
                "--iteration",
                str(iteration),
                "--expected-view-count",
                str(count),
                "--formal-width",
                str(width),
                "--formal-height",
                str(height),
                "--passes",
                "3",
                "--output",
                str(output),
            ]
            if adapter_manifest is None:
                command += ["--test-list", str(test_list)]
            else:
                command += ["--adapter-manifest", str(adapter_manifest)]
            if reference is not None:
                command += ["--correctness-reference", str(reference)]
                if method == "thermonerf":
                    command += ["--correctness-max-abs-u8", "24", "--correctness-mean-abs-u8", "1.5"]
                elif method in {"thermalgaussian_ommg", "mmone"}:
                    command += ["--correctness-max-abs-u8", "12", "--correctness-mean-abs-u8", "0.75"]
                else:
                    command += ["--correctness-max-abs-u8", "1", "--correctness-mean-abs-u8", "0.01"]
            jobs.append(
                {
                    "method": method,
                    "scene": scene,
                    "view_count": count,
                    "formal_resolution_wh": [width, height],
                    "model_root": str(model_root),
                    "source_repo": str(source),
                    "runtime_pythonpath": [str(path) for path in runtime_pythonpath],
                    "dataset": str(dataset),
                    "test_list_sha256": _sha256(test_list),
                    "adapter_manifest_sha256": _sha256(adapter_manifest) if adapter_manifest else None,
                    "correctness_reference": str(reference) if reference else None,
                    "output": str(output),
                    "log": str(log),
                    "command": command,
                }
            )
    return jobs


# Deliberately mirrors the frozen matrix definition without importing the
# aggregation module, so a matrix-preparation failure cannot be hidden by an
# import-path accident.
INTERNAL_METHODS = {
    "raw_f3",
    "scsp_refit_f3",
    "adaptive_opacity_scale_clamp",
}


def _valid_receipt(path: Path, job: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    wrapper = Path(__file__).resolve().with_name("benchmark_render_only.py")
    return (
        payload.get("status") == "completed"
        and payload.get("method") == job["method"]
        and payload.get("scene") == job["scene"]
        and payload.get("view_count") == job["view_count"]
        and payload.get("output_resolution_wh") == job["formal_resolution_wh"]
        and len(payload.get("passes", [])) == 3
        and payload.get("benchmark_wrapper", {}).get("sha256") == _sha256(wrapper)
    )


def run(args: argparse.Namespace) -> int:
    project = args.project_root.resolve()
    benchmark = args.benchmark_root.resolve()
    runtime = benchmark / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "results").mkdir(exist_ok=True)
    (runtime / "logs").mkdir(exist_ok=True)
    sibling_wrapper = Path(__file__).resolve().with_name("benchmark_render_only.py")
    wrapper_source = (
        sibling_wrapper
        if sibling_wrapper.is_file()
        else project / "code/tools/aaai27_final/benchmark_render_only.py"
    )
    wrapper_runtime = runtime / "benchmark_render_only.py"
    if not wrapper_runtime.is_file() or _sha256(wrapper_runtime) != _sha256(wrapper_source):
        wrapper_runtime.write_bytes(wrapper_source.read_bytes())

    uav_python = project / "environments/uav-tgs/bin/python"
    _prepare_adapters(project, benchmark, uav_python)
    jobs = _matrix(project, benchmark)
    if len(jobs) != 48:
        raise ValueError(f"expected 48 jobs, found {len(jobs)}")
    manifest = {
        "schema": "uav-tgs-aaai27-render-only-matrix-v1",
        "created_at": _now(),
        "host_policy": "all jobs on host 900, one clean process at a time",
        "gpu_policy": "one RTX PRO 6000; no concurrent GPU jobs",
        "wrapper_sha256": _sha256(wrapper_runtime),
        "jobs": jobs,
    }
    _atomic_json(runtime / "matrix_manifest.json", manifest)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "jobs": len(jobs)}, sort_keys=True))
        return 0

    completed = 0
    for index, job in enumerate(jobs, start=1):
        output = Path(job["output"])
        if _valid_receipt(output, job):
            completed += 1
            print(f"[{index}/48] reuse {job['scene']} {job['method']}", flush=True)
            continue
        if output.exists():
            raise ValueError(f"invalid existing receipt; refusing overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        log = Path(job["log"])
        print(f"[{index}/48] run {job['scene']} {job['method']}", flush=True)
        started = time.monotonic()
        with log.open("w", encoding="utf-8") as handle:
            child_env = os.environ.copy()
            runtime_pythonpath = job.get("runtime_pythonpath", [])
            if runtime_pythonpath:
                inherited = child_env.get("PYTHONPATH")
                child_env["PYTHONPATH"] = os.pathsep.join(
                    [*runtime_pythonpath, *([inherited] if inherited else [])]
                )
            result = subprocess.run(
                job["command"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=child_env,
            )
        if result.returncode:
            raise RuntimeError(
                f"benchmark failed rc={result.returncode}: {job['scene']} {job['method']} log={log}"
            )
        if not _valid_receipt(output, job):
            raise RuntimeError(f"benchmark did not produce a valid receipt: {output}")
        completed += 1
        print(
            f"[{index}/48] completed {job['scene']} {job['method']} in {time.monotonic()-started:.1f}s",
            flush=True,
        )
    _atomic_json(
        runtime / "MATRIX_COMPLETION.json",
        {"schema": "uav-tgs-aaai27-render-only-matrix-completion-v1", "status": "completed", "completed_at": _now(), "job_count": completed},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("/root/autodl-tmp/UAV-TGS"))
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("/root/autodl-tmp/UAV-TGS/final_results_aggregation_v1"),
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
