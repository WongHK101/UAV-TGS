#!/usr/bin/env python3
"""Evaluate one completed External Phase-B endpoint with the frozen sidecars."""

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
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aaai27_external.normalize_sequence_renders import bind


METHODS = (
    "ThermalGaussian-OMMG",
    "PhysIR-Splat",
    "MMOne",
    "Thermal3D-GS",
    "ThermoNeRF",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def run_command(
    command: Sequence[str], *, cwd: Path, log_root: Path, label: str, env: dict | None = None
) -> dict:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{label}.stdout.log"
    stderr_path = log_root / f"{label}.stderr.log"
    started_at = now()
    started = time.perf_counter()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(
            list(command), cwd=cwd, stdout=stdout, stderr=stderr, env=env, check=False
        )
    receipt = {
        "status": "SUCCEEDED" if result.returncode == 0 else "FAILED",
        "return_code": int(result.returncode),
        "started_at": started_at,
        "completed_at": now(),
        "wall_time_s": time.perf_counter() - started,
        "cwd": str(cwd.resolve()),
        "command": list(command),
        "stdout": {"path": str(stdout_path), "sha256": sha256_file(stdout_path)},
        "stderr": {"path": str(stderr_path), "sha256": sha256_file(stderr_path)},
    }
    receipt_path = log_root / f"{label}.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {stderr_path}")
    return receipt


def method_layout(root: Path, scene: str, method: str) -> tuple[Path, Path, str]:
    base = root / "external_phase_b"
    if method == "ThermalGaussian-OMMG":
        return (
            base / "experiments" / scene / method / "formal",
            base / "datasets" / scene / "thermalgaussian_ommg",
            "thermalgaussian_ommg",
        )
    if method == "PhysIR-Splat":
        return (
            base / "experiments" / scene / method / "formal_default_sh_no_opacity_reset",
            base / "datasets" / scene / "physir_splat",
            "physir_splat",
        )
    if method == "MMOne":
        return (
            base / "experiments" / scene / method / "formal",
            base / "datasets" / scene / "mmone",
            "mmone",
        )
    if method == "Thermal3D-GS":
        return (
            base / "experiments" / scene / method / "formal_stream_sync",
            base / "datasets" / scene / "thermal3dgs",
            "thermal3dgs",
        )
    return (
        base / "experiments" / scene / method / "formal",
        base / "datasets" / scene / "thermonerf",
        "thermonerf",
    )


def normalize(root: Path, scene: str, method: str, run_root: Path) -> dict[str, Path]:
    formal = root / "derived" / "aaai27_hold8_v2" / scene
    binding = root / "derived" / "thermal_radiometry" / "aaai27_hold8_v2" / scene / "binding"
    test_list = binding / "test_list.txt"
    model = run_root / "model"
    specs: dict[str, dict[str, object]] = {}
    if method == "ThermalGaussian-OMMG":
        raw = model / "test" / "ours_30000"
        specs = {
            "rgb": {"render": raw / "renders_color", "gt": raw / "gt_color", "policy": "exact", "gt_policy": "exact"},
            "thermal": {"render": raw / "renders_thermal", "gt": raw / "gt_thermal", "policy": "pil-default-resize-to-formal", "gt_policy": "pil-default-resize-to-raw"},
        }
    elif method == "PhysIR-Splat":
        raw = model / "test" / "ours_30000"
        specs = {
            "rgb": {"render": raw / "renders_rgb", "gt": raw / "gt_rgb", "policy": "pil-default-resize-to-formal", "gt_policy": "pil-default-resize-to-raw"},
            "thermal": {"render": raw / "renders", "gt": raw / "gt", "policy": "exact", "gt_policy": "exact"},
        }
    elif method == "MMOne":
        specs = {
            "rgb": {"render": model / "rgb/test/ours_30000/renders", "gt": model / "rgb/test/ours_30000/gt", "policy": "exact", "gt_policy": "exact"},
            "thermal": {"render": model / "thermal/test/ours_30000/renders", "gt": model / "thermal/test/ours_30000/gt", "policy": "pil-default-resize-to-formal", "gt_policy": "pil-default-resize-to-raw"},
        }
    elif method == "Thermal3D-GS":
        specs = {
            "thermal": {"render": model / "test/ours_30000/renders", "gt": model / "test/ours_30000/gt", "policy": "exact", "gt_policy": "exact"},
        }
    else:
        specs = {
            "rgb": {"render": run_root / "metrics", "glob": "img_*.jpg", "policy": "right-half-to-formal"},
            "thermal": {"render": run_root / "metrics", "glob": "thermal_[0-9][0-9][0-9][0-9][0-9].jpg", "policy": "hotiron-grayscale-inverse-to-formal"},
        }

    outputs: dict[str, Path] = {}
    for modality, spec in specs.items():
        output = run_root / f"normalized_{modality}"
        if not (output / "render_binding_manifest.json").is_file():
            bind(
                raw_render_root=Path(spec["render"]),
                raw_render_glob=str(spec.get("glob", "*.png")),
                raw_gt_root=Path(spec["gt"]) if "gt" in spec else None,
                raw_gt_policy=str(spec.get("gt_policy", "exact")),
                render_policy=str(spec["policy"]),
                formal_gt_root=(formal / "workspace/images" if modality == "rgb" else formal / "thermal_benchmark/images"),
                test_list=test_list,
                output_model_root=output,
                method=("PhysIR-Splat-default-thermal-SH-no-opacity-reset" if method == "PhysIR-Splat" else method),
                modality=modality,
                replace=False,
            )
        outputs[modality] = output
    return outputs


def evaluate(args: argparse.Namespace) -> Path:
    root = args.project_root.resolve()
    code = root / "code"
    python = root / "environments/uav-tgs/bin/python"
    thermo_python = root / "external_phase_a/environments/thermonerf/bin/python"
    run_root, dataset, _ = method_layout(root, args.scene, args.method)
    completion = run_root / "COMPLETION.json"
    if load_json(completion).get("status") != "SUCCEEDED":
        raise ValueError(f"endpoint is not completed: {completion}")
    evaluation = run_root / "evaluation"
    logs = evaluation / "logs"
    outputs = normalize(root, args.scene, args.method, run_root)

    for modality, model_root in outputs.items():
        if not (model_root / "results.json").is_file():
            run_command(
                [str(python), "metrics.py", "-m", str(model_root)],
                cwd=code,
                log_root=logs,
                label=f"appearance_{modality}",
            )

    formal = root / "derived/aaai27_hold8_v2" / args.scene
    binding_root = root / "derived/thermal_radiometry/aaai27_hold8_v2" / args.scene / "binding"
    collection = root / "derived/thermal_radiometry/aaai27_hold8_v2/splits/hold8/collection_manifest.json"
    protocol_binding = root / "experiments/aaai27_hold8_v2" / args.scene / "protocol/formal_radiometry_evaluation_binding.json"
    binding_payload = load_json(protocol_binding)
    formal_binding = binding_payload.get("formal_binding")
    tsdk = formal_binding.get("tsdk_target") if isinstance(formal_binding, dict) else None
    decode_sha = tsdk.get("decode_manifest_sha256") if isinstance(tsdk, dict) else None
    if not isinstance(decode_sha, str):
        raise ValueError("formal radiometry binding lacks decode manifest SHA")
    decode_candidates = sorted(
        (root / "derived/thermal_radiometry").glob(
            f"*/{args.scene}/manifests/decode_manifest.jsonl"
        )
    )
    decode_manifests = [
        path for path in decode_candidates if sha256_file(path) == decode_sha
    ]
    if len(decode_manifests) != 1:
        raise ValueError(
            f"cannot uniquely resolve frozen decode manifest: {decode_manifests}"
        )
    decode_manifest = decode_manifests[0]
    decode_protocol = decode_manifest.with_name("decode_protocol_used_v1.jsonl")
    temperature_root = formal / "thermal_undistorted/temperature_c"
    support = formal / "formal_support"
    range_manifest = formal / "radiometry/range_manifest.json"
    thermal_renders = outputs["thermal"] / "test/ours_formal/renders"
    temperature_report = evaluation / "temperature/test.json"
    hotspot_report = evaluation / "hotspot/test.json"
    temperature_report.parent.mkdir(parents=True, exist_ok=True)
    hotspot_report.parent.mkdir(parents=True, exist_ok=True)
    if not temperature_report.is_file():
        run_command(
            [str(python), "tools/thermal_radiometry/evaluate_temperature.py", "--ground-truth-root", str(temperature_root), "--render-root", str(thermal_renders), "--report", str(temperature_report), "--range-manifest", str(range_manifest), "--split-manifest", str(binding_root / "bound_split.json"), "--subset", "test", "--mask-root", str(support / "bool"), "--alpha-threshold", "0", "--require-support"],
            cwd=code, log_root=logs, label="temperature",
        )
    if not hotspot_report.is_file():
        run_command(
            [str(python), "tools/evaluate_formal_baseline_hotspots.py", "--method-name", args.method, "--scene-name", args.scene, "--formal-radiometry-binding-manifest", str(protocol_binding), "--bound-split", str(binding_root / "bound_split.json"), "--decode-manifest", str(decode_manifest), "--decode-protocol", str(decode_protocol), "--range-manifest", str(range_manifest), "--canonical-manifest", str(formal / "radiometry/canonical_manifest.json"), "--optimization-support-manifest", str(formal / "thermal_undistorted/manifest.json"), "--evaluation-support-manifest", str(support / "manifest.json"), "--hotspot-threshold-manifest", str(formal / "radiometry/hotspot_threshold_train_q95.json"), "--temperature-root", str(temperature_root), "--evaluation-support-root", str(support), "--render-root", str(thermal_renders), "--output", str(hotspot_report)],
            cwd=code, log_root=logs, label="hotspot",
        )

    geometry = evaluation / "geometry"
    model_manifest = geometry / "expected_depth/manifest.json"
    geometry.mkdir(parents=True, exist_ok=True)
    if not model_manifest.is_file():
        common = [
            "--scene-name", args.scene,
            "--adapter-manifest", str(dataset / "adapter_manifest.json"),
            "--render-binding-manifest", str(outputs["thermal"] / "render_binding_manifest.json"),
            "--collection-manifest", str(collection),
            "--scene-split-manifest", str(binding_root / "bound_split.json"),
            "--output-root", str(geometry / "expected_depth"),
        ]
        if args.method == "ThermoNeRF":
            candidates = sorted((run_root / "model" / f"{args.scene}_formal/thermal-nerf").glob("*"))
            candidates = [path for path in candidates if path.is_dir()]
            if len(candidates) != 1:
                raise ValueError(f"expected exactly one ThermoNeRF endpoint, found {candidates}")
            command = [str(thermo_python), str(code / "tools/aaai27_external/export_thermonerf_expected_depth.py"), "--method-name", args.method, "--source-repo", str(root / "external_phase_a/sources/thermo-nerf"), "--model-root", str(candidates[0]), "--dataset-path", str(dataset), "--camera-source-path", str(formal / "workspace"), "--train-list", str(binding_root / "train_list.txt"), "--test-list", str(binding_root / "test_list.txt"), "--reference-manifest", str(formal / "reference_openmvs_hold8_v2/bound_expected_depth/manifest.json"), *common]
        else:
            model = run_root / "model"
            command = [str(python), str(code / "tools/aaai27_external/export_gaussian_expected_depth.py"), "--method-name", args.method, "--model-ply", str(model / "point_cloud/iteration_30000/point_cloud.ply"), "--camera-source-path", str(formal / "thermal_benchmark"), "--train-list", str(binding_root / "train_list.txt"), "--test-list", str(binding_root / "test_list.txt"), "--camera-train-list", str(formal / "runtime_lists/thermal_train_list.txt"), "--camera-test-list", str(formal / "runtime_lists/thermal_test_list.txt"), "--formal-sparse-root", str(formal / "workspace/sparse/0"), *common]
            if args.method == "MMOne":
                command[command.index("--camera-source-path"):command.index("--camera-source-path")] = ["--mmone-thermal-checkpoint", str(model / "thermal_chkpnt30000.pth")]
        run_command(command, cwd=code, log_root=logs, label="expected_depth_export")

    geometry_output = geometry / "evaluation"
    metrics = geometry_output / "geometry_metrics.json"
    if not metrics.is_file():
        run_command(
            [str(python), "tools/hold8_expected_depth_evaluator.py", "evaluate", "--reference-manifest", str(formal / "reference_openmvs_hold8_v2/bound_expected_depth/manifest.json"), "--model-manifest", str(model_manifest), "--collection-manifest", str(collection), "--scene-split-manifest", str(binding_root / "bound_split.json"), "--expected-collection-manifest-sha256", sha256_file(collection), "--expected-scene-split-manifest-sha256", sha256_file(binding_root / "bound_split.json"), "--out-dir", str(geometry_output)],
            cwd=code, log_root=logs, label="expected_depth_evaluate",
        )

    artifacts = {
        "temperature": temperature_report,
        "hotspot": hotspot_report,
        "geometry": metrics,
        **{f"appearance_{name}": path / "results.json" for name, path in outputs.items()},
    }
    receipt = {
        "schema": "uav-tgs-aaai27-external-phase-b-endpoint-evaluation-v1",
        "status": "SUCCEEDED",
        "scene": args.scene,
        "method": args.method,
        "completed_at": now(),
        "endpoint_completion_sha256": sha256_file(completion),
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in artifacts.items()
        },
    }
    output = evaluation / "COMPLETION.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = evaluate(args)
    print(json.dumps({"status": "SUCCEEDED", "completion": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
