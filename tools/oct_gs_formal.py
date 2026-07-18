#!/usr/bin/env python3
"""Standalone fail-closed formal runner for OCT-Scalar/OCT-Residual.

This runner never changes ``train.py``.  It binds one immutable RGB anchor to
the formal split, float32 TSDK target, support, fixed Hot-Iron display LUT, and
fixed camera sequence before calibration, training, checkpointing, or report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from oct_gs.field import OCTConfig, OCTGaussianField, OCT_VARIANTS, build_oct_optimizer
from oct_gs.formal import (
    FORMAL_EXPERIMENT_RECIPE,
    FormalOCTBinding,
    FormalOCTTargetStore,
    build_formal_binding,
    load_json_object,
    sha256_file,
    sha256_json,
)
from oct_gs.losses import OCTLossWeights, oct_rendering_loss
from oct_gs.protocol import (
    BuildingGradientCalibrator,
    OCTStageCostTracker,
    capture_occupancy_snapshot,
    load_frozen_calibration,
    load_oct_checkpoint,
    load_oct_protocol_manifest,
    restore_oct_optimizer_state,
    save_oct_checkpoint,
    validate_training_source_provenance,
    verify_oct_field_finite,
    verify_oct_post_step_finite,
    verify_occupancy_snapshot,
    write_oct_protocol_manifest,
)
from oct_gs.radiance import BandRadianceProxy
from oct_gs.rendering import OCTRendererContext
from utils.camera_sequence import (
    build_sequence_manifest,
    camera_lookup,
    camera_parameters_hash,
    load_sequence_manifest,
    save_sequence_manifest,
)


FORMAL_STEPS = int(FORMAL_EXPERIMENT_RECIPE["steps"])
ENDPOINTS = tuple(int(value) for value in FORMAL_EXPERIMENT_RECIPE["endpoints"])
DEFAULT_TEMPERATURE_LR = float(FORMAL_EXPERIMENT_RECIPE["temperature_lr"])
DEFAULT_RESIDUAL_LR = float(FORMAL_EXPERIMENT_RECIPE["residual_lr"])
DEFAULT_ADAM_EPS = float(FORMAL_EXPERIMENT_RECIPE["adam_eps"])
HOTSPOT_SCHEMA = "uav-tgs-oct-train-only-hotspot-threshold-v1"
ENDPOINT_SCHEMA = "uav-tgs-oct-endpoint-v2"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    if path.exists():
        raise FileExistsError(f"formal OCT JSON already exists; refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale formal OCT JSON temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _require_absent(path: str | Path, label: str) -> Path:
    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"{label} already exists; formal OCT never overwrites: {destination}")
    return destination


def _require_fresh_directory(path: str | Path, label: str) -> Path:
    destination = Path(path).resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"{label} exists and is not a directory: {destination}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"{label} is nonempty; use a new traceable formal output root: {destination}"
        )
    return destination


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _require_isolated_output(
    candidate: str | Path,
    args: argparse.Namespace,
    *,
    label: str,
) -> Path:
    """Reject any output tree that contains, or is contained by, an input tree."""

    output = Path(candidate).resolve()
    input_roots = {
        "source dataset": Path(args.source_path).resolve(),
        "anchor model": Path(args.model_path).resolve(),
        "temperature targets": Path(args.temperature_root).resolve(),
        "canonical targets": Path(args.canonical_root).resolve(),
        "optimization support": Path(args.support_root).resolve(),
        "evaluation support": Path(args.evaluation_support_root).resolve(),
    }
    for input_label, root in input_roots.items():
        if output == root or _is_within(output, root) or _is_within(root, output):
            raise ValueError(
                f"{label} must be isolated from the {input_label} tree: "
                f"output={output}, input={root}"
            )
    return output


def _validate_recipe_args(args: argparse.Namespace) -> None:
    expected = FORMAL_EXPERIMENT_RECIPE
    checks = {
        "resolution": int,
        "residual_bound_fraction": float,
        "thermometric_domain": str,
        "temperature_lr": float,
        "residual_lr": float,
        "sequence_seed": int,
    }
    for name, caster in checks.items():
        if not hasattr(args, name):
            continue
        actual = caster(getattr(args, name))
        expected_value = caster(expected[name])
        if actual != expected_value:
            raise ValueError(
                f"formal OCT paired recipe fixes {name}={expected_value!r}; got {actual!r}"
            )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_bytes(arguments: list[str]) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git provenance command failed: git {' '.join(arguments)}: {message}")
    return bytes(result.stdout)


def _formal_source_paths() -> list[Path]:
    relative_paths = {
        Path("tools/oct_gs_formal.py"),
        Path("gaussian_renderer/__init__.py"),
        Path("utils/camera_sequence.py"),
        Path("tools/thermal_radiometry/palette_lut.py"),
        Path("scene/__init__.py"),
        Path("scene/cameras.py"),
        Path("scene/dataset_readers.py"),
        Path("scene/gaussian_model.py"),
    }
    relative_paths.update(
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "oct_gs").glob("*.py")
        if path.is_file()
    )
    rasterizer_root = REPO_ROOT / "submodules" / "diff-gaussian-rasterization"
    native_suffixes = {".py", ".cu", ".cuh", ".cpp", ".h", ".hpp"}
    relative_paths.update(
        path.relative_to(REPO_ROOT)
        for path in rasterizer_root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() in native_suffixes
        and "third_party" not in path.relative_to(rasterizer_root).parts
    )
    return sorted((REPO_ROOT / path).resolve() for path in relative_paths)


def _formal_source_provenance() -> dict[str, Any]:
    """Bind formal execution to a clean commit and explicit runtime source bytes."""

    commit = _git_bytes(["rev-parse", "HEAD"]).decode("ascii", errors="strict").strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("formal OCT requires a full lowercase 40-hex Git commit")
    status = _git_bytes(["status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        preview = status.decode("utf-8", errors="replace").splitlines()[:10]
        raise RuntimeError(
            "formal OCT execution refuses a dirty Git worktree: " + "; ".join(preview)
        )
    tracked = {
        item.decode("utf-8", errors="strict")
        for item in _git_bytes(["ls-files", "-z"]).split(b"\0")
        if item
    }
    files: list[dict[str, Any]] = []
    for path in _formal_source_paths():
        if not path.is_file():
            raise FileNotFoundError(f"formal OCT source file is missing: {path}")
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative not in tracked:
            raise RuntimeError(f"formal OCT source is not Git-tracked: {relative}")
        data = path.read_bytes()
        files.append(
            {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )
    files.sort(key=lambda record: record["path"])
    payload = {
        "schema": "uav-tgs-oct-training-source-v1",
        "git_commit": commit,
        "git_clean": not bool(status),
        "git_status_porcelain_sha256": hashlib.sha256(status).hexdigest(),
        "files": files,
        "files_sha256": sha256_json(files),
    }
    return validate_training_source_provenance(payload)


def _require_matching_source_provenance(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    """Require an artifact's source inventory to equal the current clean source."""

    try:
        stored_payload = validate_training_source_provenance(stored)
        current_payload = validate_training_source_provenance(current)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} source provenance is invalid") from exc
    if stored_payload != current_payload:
        raise RuntimeError(
            f"{label} source/commit differs from the current formal OCT source"
        )
    return current_payload


def _read_bound(path: str | Path, scene: str) -> dict[str, Any]:
    payload = load_json_object(path, "bound split")
    if payload.get("scene") != scene or not isinstance(payload.get("records"), list):
        raise ValueError("bound split scene/records mismatch")
    return payload


def _write_camera_lists(bound: Mapping[str, Any], root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for split in ("train", "test"):
        names = [
            str(record["thermal_camera_name"])
            for record in bound["records"]
            if record.get("split") == split
        ]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"invalid formal {split} camera membership")
        path = root / f"{split}.txt"
        data = "".join(f"{name}\n" for name in names).encode("utf-8")
        if path.is_file():
            if path.read_bytes() != data:
                raise RuntimeError(f"existing formal camera list differs: {path}")
        else:
            path.write_bytes(data)
        paths.append(path)
    return paths[0], paths[1]


class Runtime:
    def __init__(self, args: argparse.Namespace) -> None:
        if int(args.anchor_iteration) <= 0:
            raise ValueError("formal OCT requires anchor_iteration > 0")
        if not torch.cuda.is_available():
            raise RuntimeError("formal OCT execution requires CUDA")
        if int(args.resolution) != -1:
            raise ValueError("formal OCT v1 is fixed to native r1/full-resolution")
        # Lazy imports keep protocol inspection/CLI help usable on machines
        # where the CUDA extensions are intentionally not built.
        from scene import Scene
        from scene.gaussian_model import GaussianModel

        self.args = args
        self.scene_name = str(args.scene)
        self.bound_payload = _read_bound(args.bound_split, self.scene_name)
        output_root = _require_isolated_output(
            args.output_root, args, label="formal OCT output root"
        )
        list_root = output_root / self.scene_name / "protocol_lists"
        train_list, test_list = _write_camera_lists(self.bound_payload, list_root)
        model_path = Path(args.model_path).resolve()
        expected_anchor_artifact = (
            model_path
            / "point_cloud"
            / f"iteration_{int(args.anchor_iteration)}"
            / "point_cloud.ply"
        ).resolve()
        supplied_anchor_artifact = Path(args.anchor_artifact).resolve()
        if supplied_anchor_artifact != expected_anchor_artifact:
            raise ValueError(
                "--anchor-artifact must be the exact PLY loaded by model-path/anchor-iteration"
            )
        if not supplied_anchor_artifact.is_file():
            raise FileNotFoundError(supplied_anchor_artifact)
        dataset = SimpleNamespace(
            sh_degree=3,
            source_path=str(Path(args.source_path).resolve()),
            model_path=str(model_path),
            images=str(args.images),
            depths="",
            resolution=int(args.resolution),
            white_background=False,
            train_test_exp=False,
            # OCT never consumes Scene's RGB/T image tensors.  Keeping them on
            # CPU avoids retaining every formal image on the GPU; the native
            # float32 target for the selected camera is transferred on demand.
            data_device="cpu",
            eval=True,
            train_list=str(train_list),
            test_list=str(test_list),
            train_list_sha256=sha256_file(train_list),
            test_list_sha256=sha256_file(test_list),
        )
        self.anchor = GaussianModel(3)
        self.scene = Scene(
            dataset,
            self.anchor,
            load_iteration=int(args.anchor_iteration),
            shuffle=False,
        )
        self.train_cameras = sorted(
            self.scene.getTrainCameras(), key=lambda camera: str(camera.image_name)
        )
        self.test_cameras = sorted(
            self.scene.getTestCameras(), key=lambda camera: str(camera.image_name)
        )
        declared_train = sorted(
            str(record["thermal_camera_name"])
            for record in self.bound_payload["records"]
            if record["split"] == "train"
        )
        declared_test = sorted(
            str(record["thermal_camera_name"])
            for record in self.bound_payload["records"]
            if record["split"] == "test"
        )
        if [camera.image_name for camera in self.train_cameras] != declared_train:
            raise ValueError("loaded train cameras differ from formal membership")
        if [camera.image_name for camera in self.test_cameras] != declared_test:
            raise ValueError("loaded test cameras differ from formal membership")
        self.camera_parameters_sha256 = camera_parameters_hash(
            self.train_cameras + self.test_cameras
        )
        self.anchor_snapshot = capture_occupancy_snapshot(self.anchor)
        range_payload = load_json_object(args.range_manifest, "range manifest")
        self.tmin_c = float(range_payload["Tmin"])
        self.tmax_c = float(range_payload["Tmax"])
        self.proxy = BandRadianceProxy(self.tmin_c, self.tmax_c).cuda()
        count = int(self.anchor_snapshot["topology_count"])
        self.configs = {
            variant: OCTConfig(
                num_gaussians=count,
                tmin_c=self.tmin_c,
                tmax_c=self.tmax_c,
                variant=variant,
                residual_bound_fraction=float(args.residual_bound_fraction),
                learn_uncertainty=False,
            )
            for variant in OCT_VARIANTS
        }
        self.pipe = SimpleNamespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=False,
        )

    def binding(self) -> FormalOCTBinding:
        binding = build_formal_binding(
            scene_name=self.scene_name,
            bound_split_path=self.args.bound_split,
            decode_manifest_path=self.args.decode_manifest,
            decode_protocol_path=self.args.decode_protocol,
            range_manifest_path=self.args.range_manifest,
            canonical_manifest_path=self.args.canonical_manifest,
            temperature_root=self.args.temperature_root,
            canonical_root=self.args.canonical_root,
            support_manifest_path=self.args.support_manifest,
            support_root=self.args.support_root,
            evaluation_support_manifest_path=self.args.evaluation_support_manifest,
            evaluation_support_root=self.args.evaluation_support_root,
            camera_sequence_path=self.args.camera_sequence,
            camera_parameters_sha256=self.camera_parameters_sha256,
            anchor_artifact_path=self.args.anchor_artifact,
            anchor_snapshot=self.anchor_snapshot,
            field_configs=self.configs,
            radiance_proxy=self.proxy,
            verify_payload_files=True,
        )
        for camera in self.train_cameras + self.test_cameras:
            record = binding.by_camera[str(camera.image_name)]
            camera_shape = (int(camera.image_height), int(camera.image_width))
            if camera_shape != record.shape_hw:
                raise ValueError(
                    f"native camera/temperature dimensions differ for {camera.image_name}: "
                    f"camera={camera_shape}, target={record.shape_hw}"
                )
        return binding


def _sequence_metadata(runtime: Runtime) -> dict[str, Any]:
    args = runtime.args
    return {
        "scene": runtime.scene_name,
        "bound_split_sha256": sha256_file(args.bound_split),
        "decode_manifest_sha256": sha256_file(args.decode_manifest),
        "decode_protocol_sha256": sha256_file(args.decode_protocol),
        "range_manifest_sha256": sha256_file(args.range_manifest),
        "canonical_manifest_sha256": sha256_file(args.canonical_manifest),
        "support_manifest_sha256": sha256_file(args.support_manifest),
        "evaluation_support_manifest_sha256": sha256_file(args.evaluation_support_manifest),
        "anchor_artifact_sha256": sha256_file(args.anchor_artifact),
        "anchor_occupancy_sha256": runtime.anchor_snapshot["overall_sha256"],
        "camera_parameters_sha256": runtime.camera_parameters_sha256,
        "purpose": "OCT formal 30k paired Scalar/Residual camera schedule",
        "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
    }


def command_prepare(args: argparse.Namespace) -> int:
    _require_absent(args.camera_sequence, "camera sequence")
    runtime = Runtime(args)
    names = [str(camera.image_name) for camera in runtime.train_cameras]
    sequence = build_sequence_manifest(
        names,
        steps=FORMAL_STEPS,
        seed=int(args.sequence_seed),
        metadata=_sequence_metadata(runtime),
    )
    save_sequence_manifest(args.camera_sequence, sequence)
    print(
        json.dumps(
            {
                "camera_sequence": str(Path(args.camera_sequence).resolve()),
                "sequence_sha256": sequence["sequence_sha256"],
                "steps": sequence["steps"],
            },
            indent=2,
        )
    )
    return 0


def command_build_binding(args: argparse.Namespace) -> int:
    output = _require_absent(args.binding_output, "formal binding")
    receipt_output = output.with_name("calibration_source_receipt.json")
    if str(args.scene) == "Building":
        _require_absent(receipt_output, "calibration source receipt")
    runtime = Runtime(args)
    binding = runtime.binding()
    binding.write(output)
    if runtime.scene_name == "Building":
        _atomic_json(receipt_output, binding.calibration_receipt())
    print(json.dumps({"binding": str(output), "formal_protocol_sha256": binding.formal_protocol_sha256}, indent=2))
    return 0


def _load_sequence(runtime: Runtime, binding: FormalOCTBinding) -> dict[str, Any]:
    payload = load_sequence_manifest(
        runtime.args.camera_sequence,
        camera_names=binding.names("train"),
        expected_steps=FORMAL_STEPS,
    )
    expected_seed = int(FORMAL_EXPERIMENT_RECIPE["sequence_seed"])
    if type(payload.get("seed")) is not int or payload["seed"] != expected_seed:
        raise ValueError(
            f"formal OCT camera sequence must use the frozen seed={expected_seed}"
        )
    return payload


def _remaining_sequence(sequence: Mapping[str, Any], start_step: int) -> list[str]:
    values = sequence.get("sequence")
    if not isinstance(values, list) or len(values) != FORMAL_STEPS:
        raise ValueError("formal OCT sequence payload/length is invalid")
    if type(start_step) is not int or start_step < 0 or start_step >= FORMAL_STEPS:
        raise ValueError("formal OCT resume sequence offset is outside [0, 30000)")
    return [str(value) for value in values[start_step:]]


def _loss_for_view(
    output: Mapping[str, Any],
    targets: FormalOCTTargetStore,
    camera: Any,
    proxy: BandRadianceProxy,
    weights: OCTLossWeights,
    domain: str,
) -> dict[str, Any]:
    height, width = output["temperature_c"].shape[-2:]
    temperature, color, support = targets.get(
        str(camera.image_name), height, width, output["temperature_c"].device
    )
    return oct_rendering_loss(
        output["temperature_c"],
        temperature,
        output["hot_iron"],
        radiance_proxy=proxy,
        target_hot_iron=color,
        mask=support,
        weights=weights,
        thermometric_domain=domain,
    )


def command_calibrate(args: argparse.Namespace) -> int:
    # Calibration chooses the only loss weights used by both formal scenes, so
    # it is subject to the same clean-source contract as training.
    source_provenance = _formal_source_provenance()
    output_path = _require_absent(args.calibration_manifest, "Building calibration manifest")
    runtime = Runtime(args)
    binding = runtime.binding()
    if binding.scene_name != "Building":
        raise ValueError("loss calibration is Building/train only")
    _load_sequence(runtime, binding)
    targets = FormalOCTTargetStore(binding)
    targets.preload(binding.names("train"))
    cameras = camera_lookup(runtime.train_cameras)
    calibrator = BuildingGradientCalibrator(
        binding.calibration_receipt(), thermometric_domain=args.thermometric_domain
    )
    raster_passes = 0
    calibration_cost = OCTStageCostTracker(
        {
            "scene": "Building",
            "mode": "train-only-gradient-calibration",
            "formal_protocol_sha256": binding.formal_protocol_sha256,
        }
    )
    calibration_cost.start()
    for variant in OCT_VARIANTS:
        field = OCTGaussianField(runtime.configs[variant]).cuda()
        context = OCTRendererContext(runtime.anchor, runtime.proxy)
        for camera_name in binding.names("train"):
            output = context.render(cameras[camera_name], field, runtime.pipe)
            components = _loss_for_view(
                output,
                targets,
                cameras[camera_name],
                runtime.proxy,
                OCTLossWeights(1.0, 1.0, 1.0, 0.0),
                args.thermometric_domain,
            )
            calibrator.observe(camera_name, variant, components, field.parameters())
            raster_passes += int(output["raster_passes"])
            calibration_cost.record_step(raster_passes=int(output["raster_passes"]))
        context.verify_anchor_unchanged()
    payload = calibrator.freeze(
        output_path,
        metadata={
            "formal_protocol_sha256": binding.formal_protocol_sha256,
            "raster_passes": raster_passes,
            "calibration_cost": calibration_cost.finish(),
            "experiment_recipe": dict(FORMAL_EXPERIMENT_RECIPE),
            "experiment_recipe_sha256": sha256_json(FORMAL_EXPERIMENT_RECIPE),
            "source_provenance": source_provenance,
            "uncertainty_calibration": "N/A (OCT-GS v1 disables sigma/NLL)",
        },
    )
    print(json.dumps({"calibration": str(output_path), "sha256": payload["calibration_sha256"], "weights": payload["weights"]}, indent=2))
    return 0


def _optimizer_config(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    return {
        "type": "Adam",
        "field_only": True,
        "temperature_lr": float(args.temperature_lr),
        "residual_lr": float(args.residual_lr) if variant == "oct_residual" else None,
        "adam_eps": DEFAULT_ADAM_EPS,
        "schedule": "constant",
        "steps": FORMAL_STEPS,
    }


def _build_optimizer(args: argparse.Namespace, field: OCTGaussianField) -> torch.optim.Adam:
    return build_oct_optimizer(
        field,
        temperature_lr=float(args.temperature_lr),
        residual_lr=float(args.residual_lr) if field.config.variant == "oct_residual" else None,
    )


def _endpoint_receipt_payload(
    *,
    step: int,
    checkpoint: Path,
    protocol: Mapping[str, Any],
    binding: FormalOCTBinding,
    anchor_snapshot: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    recent_loss_mean: float,
    cost: Mapping[str, Any],
    resumed_from_step: int | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": ENDPOINT_SCHEMA,
        "step": int(step),
        "sequence_offset": int(step),
        "checkpoint_sha256": sha256_file(checkpoint),
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "formal_protocol_sha256": binding.formal_protocol_sha256,
        "anchor_occupancy_sha256": anchor_snapshot["overall_sha256"],
        "source_files_sha256": source_provenance["files_sha256"],
        "recent_loss_mean": float(recent_loss_mean),
        "resumed_from_step": resumed_from_step,
        "cost": dict(cost),
    }
    payload["endpoint_sha256"] = sha256_json(payload)
    return payload


def _validate_resume_layout(
    *,
    run_root: Path,
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    protocol: Mapping[str, Any],
    binding: FormalOCTBinding,
    anchor_snapshot: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> int:
    if (run_root / "STATUS.json").exists():
        raise FileExistsError("completed formal OCT runs cannot be resumed")
    step = checkpoint.get("step")
    if type(step) is not int or step not in ENDPOINTS:
        raise ValueError("resume checkpoint is not a formal OCT endpoint")
    if checkpoint.get("sequence_offset") != step:
        raise ValueError("resume checkpoint sequence offset/global step mismatch")
    if step >= FORMAL_STEPS:
        raise ValueError("the final 30k endpoint cannot be resumed")
    expected_checkpoint = (run_root / "checkpoints" / f"step_{step}.pt").resolve()
    if checkpoint_path.resolve() != expected_checkpoint:
        raise ValueError(
            "resume checkpoint must be the endpoint inside the selected formal run root"
        )
    expected_steps = {value for value in ENDPOINTS if value <= step}
    checkpoint_dir = run_root / "checkpoints"
    endpoint_dir = run_root / "endpoints"
    actual_checkpoints = (
        {path.name for path in checkpoint_dir.iterdir()} if checkpoint_dir.is_dir() else set()
    )
    actual_endpoints = (
        {path.name for path in endpoint_dir.iterdir()} if endpoint_dir.is_dir() else set()
    )
    expected_checkpoint_names = {f"step_{value}.pt" for value in expected_steps}
    expected_endpoint_names = {f"step_{value}.json" for value in expected_steps}
    if actual_checkpoints != expected_checkpoint_names:
        raise RuntimeError(
            "resume checkpoint directory is incomplete/contaminated or has later artifacts"
        )
    if actual_endpoints != expected_endpoint_names:
        raise RuntimeError(
            "resume endpoint directory is incomplete/contaminated or has later artifacts"
        )
    receipt_path = endpoint_dir / f"step_{step}.json"
    receipt = load_json_object(receipt_path, "OCT resume endpoint")
    supplied_hash = receipt.get("endpoint_sha256")
    basis = dict(receipt)
    basis.pop("endpoint_sha256", None)
    if receipt.get("schema") != ENDPOINT_SCHEMA or supplied_hash != sha256_json(basis):
        raise ValueError("resume endpoint receipt schema/hash mismatch")
    expected_fields = {
        "step": step,
        "sequence_offset": step,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "protocol_manifest_sha256": protocol["manifest_sha256"],
        "formal_protocol_sha256": binding.formal_protocol_sha256,
        "anchor_occupancy_sha256": anchor_snapshot["overall_sha256"],
        "source_files_sha256": source_provenance["files_sha256"],
    }
    for key, expected in expected_fields.items():
        if receipt.get(key) != expected:
            raise ValueError(f"resume endpoint receipt mismatch for {key}")
    if receipt.get("cost") != checkpoint.get("cost_summary"):
        raise ValueError("resume endpoint/checkpoint cost receipt mismatch")
    return step


def command_train(args: argparse.Namespace) -> int:
    # Provenance is checked before Runtime writes protocol camera lists or
    # allocates/loads any formal training state.
    source_provenance = _formal_source_provenance()
    runtime = Runtime(args)
    binding = runtime.binding()
    sequence = _load_sequence(runtime, binding)
    variant = str(args.variant)
    calibration, weights = load_frozen_calibration(
        args.calibration_manifest,
        expected_domain=args.thermometric_domain,
        expected_variant=variant,
        expected_calibration_sha256=args.calibration_sha256,
        building_receipt=binding.calibration_receipt() if binding.scene_name == "Building" else None,
        consumer_scene=binding.scene_name,
    )
    calibration_metadata = calibration.get("metadata")
    if not isinstance(calibration_metadata, Mapping):
        raise RuntimeError("formal OCT calibration metadata is missing")
    _require_matching_source_provenance(
        calibration_metadata.get("source_provenance", {}),
        source_provenance,
        label="formal OCT calibration",
    )
    run_root = Path(args.output_root).resolve() / binding.scene_name / variant
    protocol_path = run_root / "protocol.json"
    resume_argument = getattr(args, "resume_checkpoint", None)
    resumed_from_step: int | None = None
    resume_checkpoint_sha256: str | None = None
    if resume_argument is None:
        _require_fresh_directory(run_root, "OCT training run")
        field = OCTGaussianField(runtime.configs[variant]).cuda()
        optimizer = _build_optimizer(args, field)
        protocol = write_oct_protocol_manifest(
            protocol_path,
            field=field,
            anchor_snapshot=runtime.anchor_snapshot,
            radiance_proxy=runtime.proxy,
            calibration_manifest=args.calibration_manifest,
            formal_binding=binding,
            expected_calibration_sha256=calibration["calibration_sha256"],
            thermometric_domain=args.thermometric_domain,
            optimizer_config=_optimizer_config(args, variant),
            source_provenance=source_provenance,
            git_commit=source_provenance["git_commit"],
        )
    else:
        if not run_root.is_dir():
            raise FileNotFoundError(f"formal OCT resume run root is missing: {run_root}")
        if not protocol_path.is_file():
            raise FileNotFoundError(f"formal OCT resume protocol is missing: {protocol_path}")
        protocol = load_oct_protocol_manifest(protocol_path)
        if protocol.get("source_provenance") != source_provenance:
            raise RuntimeError(
                "formal OCT resume source/commit differs from the immutable training protocol"
            )
        if protocol.get("scene_name") != binding.scene_name or protocol.get("variant") != variant:
            raise ValueError("formal OCT resume scene/variant differs from protocol")
        if protocol.get("loss", {}).get("calibration_sha256") != calibration[
            "calibration_sha256"
        ]:
            raise ValueError("formal OCT resume calibration differs from protocol")
        checkpoint_path = Path(resume_argument).resolve()
        field, checkpoint = load_oct_checkpoint(
            checkpoint_path,
            anchor=runtime.anchor,
            protocol_manifest=protocol_path,
            formal_binding=binding,
            map_location="cpu",
        )
        if field.config.variant != variant:
            raise ValueError("formal OCT resume checkpoint variant differs from request")
        field = field.cuda()
        optimizer = _build_optimizer(args, field)
        restored_step = restore_oct_optimizer_state(
            field=field,
            optimizer=optimizer,
            checkpoint_metadata=checkpoint,
            anchor=runtime.anchor,
            protocol_manifest=protocol_path,
        )
        resumed_from_step = _validate_resume_layout(
            run_root=run_root,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            protocol=protocol,
            binding=binding,
            anchor_snapshot=runtime.anchor_snapshot,
            source_provenance=source_provenance,
        )
        if restored_step != resumed_from_step:
            raise RuntimeError("restored Adam step differs from resume sequence offset")
        resume_checkpoint_sha256 = checkpoint["checkpoint_sha256"]
        # The large serialized optimizer mapping is no longer needed once Adam
        # has copied it onto the live field parameters.
        checkpoint.pop("optimizer_state_dict", None)
    start_step = 0 if resumed_from_step is None else resumed_from_step
    targets = FormalOCTTargetStore(binding)
    targets.preload(binding.names("train"))
    cameras = camera_lookup(runtime.train_cameras)
    context = OCTRendererContext(runtime.anchor, runtime.proxy)
    tracker = OCTStageCostTracker(
        {
            "scene": binding.scene_name,
            "variant": variant,
            "formal_protocol_sha256": binding.formal_protocol_sha256,
            "segment_start_step": start_step,
            "resumed_from_checkpoint_sha256": resume_checkpoint_sha256,
        }
    )
    tracker.start()
    losses: list[float] = []
    for step, camera_name in enumerate(
        _remaining_sequence(sequence, start_step), start=start_step + 1
    ):
        optimizer.zero_grad(set_to_none=True)
        output = context.render(cameras[camera_name], field, runtime.pipe)
        loss = _loss_for_view(
            output, targets, cameras[camera_name], runtime.proxy, weights, args.thermometric_domain
        )
        total = loss["total"]
        if not bool(torch.isfinite(total).item()):
            raise FloatingPointError(f"non-finite OCT loss at step {step}")
        total.backward()
        for parameter in field.parameters():
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all().item()):
                raise FloatingPointError(f"missing/non-finite OCT gradient at step {step}")
        optimizer.step()
        verify_oct_post_step_finite(field, optimizer, step)
        tracker.record_step(raster_passes=int(output["raster_passes"]))
        losses.append(float(total.detach().item()))
        if step in ENDPOINTS:
            context.verify_anchor_unchanged()
            cost = tracker.snapshot(status=f"endpoint_{step}")
            cost["segment_start_step"] = start_step
            cost["segment_optimizer_steps"] = int(step - start_step)
            cost["cumulative_optimizer_steps"] = step
            cost["resumed_from_checkpoint_sha256"] = resume_checkpoint_sha256
            checkpoint = run_root / "checkpoints" / f"step_{step}.pt"
            endpoint_path = run_root / "endpoints" / f"step_{step}.json"
            _require_absent(checkpoint, f"formal OCT checkpoint step {step}")
            _require_absent(endpoint_path, f"formal OCT endpoint receipt step {step}")
            save_oct_checkpoint(
                checkpoint,
                field=field,
                anchor=runtime.anchor,
                anchor_snapshot=runtime.anchor_snapshot,
                optimizer=optimizer,
                step=step,
                protocol_manifest=protocol_path,
                cost_summary=cost,
            )
            endpoint_receipt = _endpoint_receipt_payload(
                step=step,
                checkpoint=checkpoint,
                protocol=protocol,
                binding=binding,
                anchor_snapshot=runtime.anchor_snapshot,
                source_provenance=source_provenance,
                recent_loss_mean=float(np.mean(losses[-min(100, len(losses)) :])),
                cost=cost,
                resumed_from_step=resumed_from_step,
            )
            _atomic_json(endpoint_path, endpoint_receipt)
    verify_occupancy_snapshot(runtime.anchor, runtime.anchor_snapshot)
    status_path = _require_absent(run_root / "STATUS.json", "formal OCT completion status")
    final_cost = tracker.finish()
    final_cost["segment_start_step"] = start_step
    final_cost["segment_optimizer_steps"] = FORMAL_STEPS - start_step
    final_cost["cumulative_optimizer_steps"] = FORMAL_STEPS
    final_cost["resumed_from_checkpoint_sha256"] = resume_checkpoint_sha256
    _atomic_json(
        status_path,
        {
            "status": "completed",
            "steps": FORMAL_STEPS,
            "endpoints": list(ENDPOINTS),
            "protocol_manifest_sha256": protocol["manifest_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "source_provenance": source_provenance,
            "resumed_from_step": resumed_from_step,
            "resumed_from_checkpoint_sha256": resume_checkpoint_sha256,
            "alpha_backend": context.alpha_backend,
            "cost": final_cost,
        },
    )
    return 0


def command_freeze_hotspot(args: argparse.Namespace) -> int:
    _require_absent(args.hotspot_threshold_manifest, "hotspot threshold manifest")
    runtime = Runtime(args)
    binding = runtime.binding()
    # A deterministic fixed-bin train-only estimate avoids loading hundreds of
    # full-resolution frames into RAM.  The test split is never read here.
    bins = int(args.hotspot_bins)
    if bins < 256:
        raise ValueError("hotspot histogram requires at least 256 bins")
    edges = np.linspace(binding.tmin_c, binding.tmax_c, bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    train_records = [record for record in binding.records if record.split == "train"]
    for record in train_records:
        temperature = np.load(record.temperature_path, mmap_mode="r", allow_pickle=False)
        if record.support_path.suffix.casefold() == ".npy":
            support = np.load(record.support_path, mmap_mode="r", allow_pickle=False)
        else:
            with Image.open(record.support_path) as image:
                support = np.asarray(image.convert("L"), dtype=np.uint8) == 255
        values = np.asarray(temperature)[np.asarray(support)]
        values = np.clip(values, binding.tmin_c, binding.tmax_c)
        counts += np.histogram(values, bins=edges)[0]
    total = int(counts.sum())
    if total <= 0:
        raise ValueError("Building/InternalRoad train support contains no valid pixels")
    target_rank = int(math.ceil(float(args.hotspot_quantile) * total))
    index = int(np.searchsorted(np.cumsum(counts), target_rank, side="left"))
    threshold = float(edges[min(index + 1, bins)])
    payload: dict[str, Any] = {
        "schema": HOTSPOT_SCHEMA,
        "scene_name": binding.scene_name,
        "source_receipt": binding.hotspot_receipt(),
        "source_split": "train",
        "test_statistics_used": False,
        "quantile": float(args.hotspot_quantile),
        "histogram_bins": bins,
        "valid_train_pixels": total,
        "threshold_c": threshold,
        "range_c": [binding.tmin_c, binding.tmax_c],
        "train_view_ids_sha256": sha256_json(binding.names("train")),
    }
    payload["threshold_sha256"] = sha256_json(payload)
    _atomic_json(Path(args.hotspot_threshold_manifest), payload)
    return 0


def _load_hotspot_threshold(path: str | Path, binding: FormalOCTBinding) -> dict[str, Any]:
    payload = load_json_object(path, "hotspot threshold")
    supplied = payload.get("threshold_sha256")
    basis = dict(payload)
    basis.pop("threshold_sha256", None)
    if payload.get("schema") != HOTSPOT_SCHEMA or supplied != sha256_json(basis):
        raise ValueError("hotspot threshold manifest/hash mismatch")
    if payload.get("scene_name") != binding.scene_name:
        raise ValueError("hotspot threshold scene mismatch")
    if payload.get("source_receipt") != binding.hotspot_receipt():
        raise ValueError("hotspot threshold radiometry/split receipt mismatch")
    if payload.get("source_split") != "train" or payload.get("test_statistics_used") is not False:
        raise ValueError("hotspot threshold is not train-only")
    if payload.get("train_view_ids_sha256") != sha256_json(binding.names("train")):
        raise ValueError("hotspot threshold train membership mismatch")
    return payload


def _save_exact_png(path: Path, color_chw: torch.Tensor) -> None:
    value = (
        color_chw.detach().movedim(0, -1).clamp(0.0, 1.0).mul(255.0).round()
        .to(torch.uint8).cpu().numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(value, mode="RGB").save(path, format="PNG", compress_level=9)


def _copy_immutable_reference(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    # Never hard-link formal GT into a writable evaluator tree: a later tool
    # that opens the destination in-place must not be able to mutate the
    # canonical benchmark inode.
    shutil.copy2(source, destination)


def _run_formal_appearance_evaluator(
    appearance_root: Path,
    *,
    method_name: str,
    expected_names: list[str],
) -> dict[str, Any]:
    evaluator = (REPO_ROOT / "metrics.py").resolve()
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(evaluator), "-m", str(appearance_root)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (appearance_root / "metrics.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (appearance_root / "metrics.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"formal metrics.py failed with exit code {result.returncode}")
    aggregate_path = appearance_root / "results.json"
    per_view_path = appearance_root / "per_view.json"
    aggregate = load_json_object(aggregate_path, "formal appearance aggregate")
    per_view = load_json_object(per_view_path, "formal appearance per-view")
    if set(aggregate) != {method_name} or set(per_view) != {method_name}:
        raise ValueError("formal metrics.py method set mismatch")
    metrics = aggregate[method_name]
    if set(metrics) != {"SSIM", "PSNR", "LPIPS"} or not all(
        np.isfinite(float(metrics[name])) for name in ("SSIM", "PSNR", "LPIPS")
    ):
        raise ValueError("formal metrics.py aggregate is incomplete/non-finite")
    expected = set(expected_names)
    for metric_name in ("SSIM", "PSNR", "LPIPS"):
        values = per_view[method_name].get(metric_name)
        if not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError(f"formal metrics.py per-view coverage mismatch for {metric_name}")
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError(f"formal metrics.py per-view {metric_name} is non-finite")
    return {
        "protocol": "repository_metrics_py_full_frame_v1",
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "git_commit": _git_commit(),
        "wall_time_s": float(elapsed),
        "results_path": str(aggregate_path),
        "results_sha256": sha256_file(aggregate_path),
        "per_view_path": str(per_view_path),
        "per_view_sha256": sha256_file(per_view_path),
        "metrics": {name: float(metrics[name]) for name in ("SSIM", "PSNR", "LPIPS")},
        "per_view": per_view[method_name],
    }


def _run_formal_temperature_evaluator(
    output_root: Path,
    *,
    render_root: Path,
    ground_truth_root: Path,
    evaluation_support_root: Path,
    range_manifest: Path,
    split_manifest: Path,
    expected_count: int,
) -> dict[str, Any]:
    evaluator = (REPO_ROOT / "tools" / "thermal_radiometry" / "evaluate_temperature.py").resolve()
    report_path = output_root / "temperature_palette_inverted_comparable.json"
    command = [
        sys.executable,
        str(evaluator),
        "--ground-truth-root",
        str(ground_truth_root),
        "--render-root",
        str(render_root),
        "--report",
        str(report_path),
        "--range-manifest",
        str(range_manifest),
        "--split-manifest",
        str(split_manifest),
        "--subset",
        "test",
        "--mask-root",
        str(evaluation_support_root),
        "--alpha-threshold",
        "0",
        "--require-support",
    ]
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    (output_root / "temperature_evaluator.stdout.txt").write_text(
        result.stdout, encoding="utf-8"
    )
    (output_root / "temperature_evaluator.stderr.txt").write_text(
        result.stderr, encoding="utf-8"
    )
    if result.returncode != 0 or not report_path.is_file():
        raise RuntimeError(
            f"formal palette-inverted temperature evaluator failed with exit code {result.returncode}"
        )
    report = load_json_object(report_path, "formal palette-inverted temperature report")
    if report.get("schema") != "uav-tgs-apparent-temperature-evaluation-v2":
        raise ValueError("formal palette-inverted temperature report schema mismatch")
    if report.get("status") != "complete" or report.get("primary_metric_valid") is not True:
        raise ValueError("formal palette-inverted temperature report is incomplete")
    summary = report.get("summary")
    if not isinstance(summary, Mapping) or int(summary.get("evaluated_file_count", -1)) != int(
        expected_count
    ):
        raise ValueError("formal palette-inverted temperature coverage mismatch")
    primary = summary.get("temperature_error_supported_pixels")
    if not isinstance(primary, Mapping):
        raise ValueError("formal palette-inverted primary temperature metric is missing")
    for name in ("mae_c", "rmse_c", "signed_bias_c", "p95_abs_error_c"):
        if not np.isfinite(float(primary.get(name))):
            raise ValueError(f"formal palette-inverted temperature metric {name} is invalid")
    expected_roots = {
        "ground_truth_root": ground_truth_root.resolve(),
        "render_root": render_root.resolve(),
    }
    for key, expected in expected_roots.items():
        actual = report.get(key)
        if not isinstance(actual, str) or Path(actual).resolve() != expected:
            raise ValueError(f"formal palette-inverted report {key} provenance mismatch")
    split = report.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("formal palette-inverted split provenance is missing")
    if (
        Path(str(split.get("path", ""))).resolve() != split_manifest.resolve()
        or split.get("sha256") != sha256_file(split_manifest)
        or split.get("subset") != "test"
    ):
        raise ValueError("formal palette-inverted split provenance mismatch")
    temperature_range = report.get("temperature_range")
    range_source = (
        temperature_range.get("source")
        if isinstance(temperature_range, Mapping)
        else None
    )
    if not isinstance(range_source, Mapping) or (
        Path(str(range_source.get("path", ""))).resolve() != range_manifest.resolve()
        or range_source.get("sha256") != sha256_file(range_manifest)
    ):
        raise ValueError("formal palette-inverted range provenance mismatch")
    support_policy = report.get("support_policy")
    if not isinstance(support_policy, Mapping) or (
        Path(str(support_policy.get("mask_root", ""))).resolve()
        != evaluation_support_root.resolve()
        or support_policy.get("require_support") is not True
        or support_policy.get("support_is_explicit") is not True
        or float(support_policy.get("alpha_threshold", math.nan)) != 0.0
    ):
        raise ValueError("formal palette-inverted support provenance/policy mismatch")
    coverage = summary.get("support_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("formal palette-inverted support coverage is missing")
    exact_counts = {
        "expected_frames": expected_count,
        "render_available_frames": expected_count,
        "support_available_frames": expected_count,
        "explicit_support_frames": expected_count,
        "missing_render_frames": 0,
        "missing_mask_frames": 0,
        "frames_without_supported_pixels": 0,
    }
    for key, expected in exact_counts.items():
        if type(coverage.get(key)) is not int or int(coverage[key]) != int(expected):
            raise ValueError(f"formal palette-inverted support coverage mismatch for {key}")
    off_lut = summary.get("off_lut_distance_aggregates")
    if not isinstance(off_lut, Mapping):
        raise ValueError("formal palette-inverted off-LUT aggregate is missing")
    for domain in ("supported_pixels", "all_pixels_diagnostic"):
        values = off_lut.get(domain)
        if not isinstance(values, Mapping) or float(values.get("max_rgb_distance", math.nan)) != 0.0:
            raise ValueError(f"formal OCT exact-display render is off the fixed LUT in {domain}")
    return {
        "protocol": "repository_evaluate_temperature_v2_palette_inverse",
        "evaluator_path": str(evaluator),
        "evaluator_sha256": sha256_file(evaluator),
        "git_commit": _git_commit(),
        "wall_time_s": float(elapsed),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "primary_supported_pixels": dict(primary),
        "off_lut_distance": {key: dict(value) for key, value in off_lut.items()},
    }


def command_eval(args: argparse.Namespace) -> int:
    if int(args.resolution) != -1:
        raise ValueError("formal OCT test evaluation is fixed to r1/full-resolution")
    # Reject dirty or drifted rendering code before creating any evaluation
    # output.  The immutable training protocol is the source-of-truth receipt.
    source_provenance = _formal_source_provenance()
    protocol_path = Path(args.protocol_manifest).resolve()
    training_protocol = load_oct_protocol_manifest(protocol_path)
    _require_matching_source_provenance(
        training_protocol.get("source_provenance", {}),
        source_provenance,
        label="formal OCT evaluation/training protocol",
    )
    _require_isolated_output(args.eval_output, args, label="formal OCT evaluation output")
    output_root = _require_fresh_directory(args.eval_output, "OCT evaluation output")
    runtime = Runtime(args)
    binding = runtime.binding()
    variant = str(args.variant)
    field, checkpoint = load_oct_checkpoint(
        args.checkpoint,
        anchor=runtime.anchor,
        protocol_manifest=protocol_path,
        formal_binding=binding,
        map_location="cuda",
    )
    if field.config.variant != variant:
        raise ValueError("checkpoint variant differs from requested evaluation variant")
    field = field.cuda().eval()
    threshold_manifest = _load_hotspot_threshold(args.hotspot_threshold_manifest, binding)
    hotspot_threshold = float(threshold_manifest["threshold_c"])
    targets = FormalOCTTargetStore(binding)
    targets.preload(binding.names("test"), evaluation_support=True)
    context = OCTRendererContext(runtime.anchor, runtime.proxy)
    tracker = OCTStageCostTracker({"scene": binding.scene_name, "variant": variant, "mode": "test-r1"})
    tracker.start()
    rows: list[dict[str, Any]] = []
    absolute_errors: list[np.ndarray] = []
    signed_sum = squared_sum = 0.0
    valid_count = 0
    hotspot_intersection = hotspot_union = 0
    hotspot_bins = 4096
    hotspot_pos = np.zeros(hotspot_bins, dtype=np.int64)
    hotspot_neg = np.zeros(hotspot_bins, dtype=np.int64)
    gaussian_sum = torch.zeros(field.num_gaussians, device="cuda", dtype=torch.float32)
    gaussian_square_sum = torch.zeros_like(gaussian_sum)
    gaussian_visible_count = torch.zeros(
        field.num_gaussians, device="cuda", dtype=torch.int32
    )
    render_time_ms: list[float] = []
    appearance_method = f"{variant}_step_{int(checkpoint['step'])}"
    appearance_root = output_root / "formal_appearance"
    appearance_renders = appearance_root / "test" / appearance_method / "renders"
    appearance_gt = appearance_root / "test" / appearance_method / "gt"
    expected_appearance_names: list[str] = []

    with torch.no_grad():
        for camera in runtime.test_cameras:
            render_start = torch.cuda.Event(enable_timing=True)
            render_end = torch.cuda.Event(enable_timing=True)
            render_start.record()
            output = context.render(camera, field, runtime.pipe, exact_display=True)
            render_end.record()
            render_end.synchronize()
            render_time_ms.append(float(render_start.elapsed_time(render_end)))
            height, width = output["temperature_c"].shape[-2:]
            record = binding.by_camera[str(camera.image_name)]
            if (height, width) != record.shape_hw:
                raise ValueError("formal test render is not native target resolution")
            target_t, target_rgb, support = targets.get(
                camera.image_name,
                height,
                width,
                "cuda",
                evaluation_support=True,
            )
            valid = support[0]
            signed = (output["temperature_c"] - target_t[0])[valid]
            if not bool(torch.isfinite(signed).all().item()):
                raise FloatingPointError("non-finite formal test temperature")
            abs_cpu = signed.abs().float().cpu().numpy()
            absolute_errors.append(abs_cpu)
            signed_sum += float(signed.double().sum().item())
            squared_sum += float(signed.double().square().sum().item())
            valid_count += int(signed.numel())
            color_error = (output["hot_iron"] - target_rgb)[:, valid]
            mse = float(color_error.double().square().mean().item())
            metric_loss = oct_rendering_loss(
                output["temperature_c"],
                target_t,
                output["hot_iron"],
                radiance_proxy=runtime.proxy,
                target_hot_iron=target_rgb,
                mask=support,
                weights=OCTLossWeights(1.0, 0.0, 1.0, 0.0),
                thermometric_domain="celsius",
            )
            prediction_hot = output["temperature_c"][valid] >= hotspot_threshold
            target_hot = target_t[0][valid] >= hotspot_threshold
            hotspot_intersection += int((prediction_hot & target_hot).sum().item())
            hotspot_union += int((prediction_hot | target_hot).sum().item())
            score = ((output["temperature_c"][valid] - binding.tmin_c) / (binding.tmax_c - binding.tmin_c)).clamp(0, 1)
            indices = torch.clamp((score * (hotspot_bins - 1)).long(), 0, hotspot_bins - 1)
            hotspot_pos += torch.bincount(indices[target_hot], minlength=hotspot_bins).cpu().numpy()
            hotspot_neg += torch.bincount(indices[~target_hot], minlength=hotspot_bins).cpu().numpy()
            gaussian_t = output["gaussian_temperature_c"][:, 0]
            radii = output.get("radii")
            if not isinstance(radii, torch.Tensor) or radii.shape != gaussian_t.shape:
                raise RuntimeError("formal OCT evaluation requires per-Gaussian radii")
            visible = radii > 0
            gaussian_sum[visible] += gaussian_t[visible]
            gaussian_square_sum[visible] += gaussian_t[visible].square()
            gaussian_visible_count[visible] += 1
            tracker.record_step(raster_passes=int(output["raster_passes"]))
            stem = Path(camera.image_name).stem
            render_t = output["temperature_c"].detach().cpu().numpy().astype(np.float32, copy=False)
            (output_root / "temperature_c").mkdir(parents=True, exist_ok=True)
            np.save(output_root / "temperature_c" / f"{stem}.npy", render_t, allow_pickle=False)
            appearance_name = f"{stem}.png"
            render_path = appearance_renders / appearance_name
            _save_exact_png(render_path, output["hot_iron"])
            _copy_immutable_reference(record.canonical_path, appearance_gt / appearance_name)
            expected_appearance_names.append(appearance_name)
            rows.append(
                {
                    "camera_name": camera.image_name,
                    "mae_c": float(np.mean(abs_cpu)),
                    "rmse_c": float(np.sqrt(np.mean(np.square(abs_cpu, dtype=np.float64)))),
                    "bias_c": float(signed.double().mean().item()),
                    "p95_abs_c": float(np.quantile(abs_cpu, 0.95)),
                    "support_masked_psnr_db": (
                        None if mse == 0.0 else float(-10.0 * math.log10(mse))
                    ),
                    "support_masked_ssim": float(
                        1.0 - metric_loss["color_dssim"].item()
                    ),
                    "valid_pixels": int(signed.numel()),
                }
            )
    if valid_count <= 0:
        raise ValueError("formal test support contains no valid pixels")
    all_abs = np.concatenate(absolute_errors)
    cumulative_tp = np.cumsum(hotspot_pos[::-1], dtype=np.int64)
    cumulative_fp = np.cumsum(hotspot_neg[::-1], dtype=np.int64)
    positives = int(hotspot_pos.sum())
    if positives > 0:
        recall = cumulative_tp / positives
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        previous = np.concatenate(([0.0], recall[:-1]))
        auprc = float(np.sum((recall - previous) * precision))
    else:
        auprc = None
    visibility_denominator = gaussian_visible_count.clamp_min(1).to(torch.float32)
    variance = (
        gaussian_square_sum / visibility_denominator
        - (gaussian_sum / visibility_denominator).square()
    ).clamp_min(0.0)
    variance_valid = gaussian_visible_count >= 2
    measured_variance = variance[variance_valid]
    if measured_variance.numel() == 0:
        raise ValueError("no Gaussian is visible in at least two formal test views")
    if variant == "oct_scalar":
        if float(measured_variance.max().item()) > 1e-6:
            raise RuntimeError("OCT-Scalar cross-view temperature variance is not zero")
    context.verify_anchor_unchanged()
    appearance_receipt = _run_formal_appearance_evaluator(
        appearance_root,
        method_name=appearance_method,
        expected_names=expected_appearance_names,
    )
    palette_temperature_receipt = _run_formal_temperature_evaluator(
        output_root,
        render_root=appearance_renders,
        ground_truth_root=Path(args.temperature_root).resolve(),
        evaluation_support_root=(Path(args.evaluation_support_root).resolve() / "bool"),
        range_manifest=Path(args.range_manifest).resolve(),
        split_manifest=Path(args.bound_split).resolve(),
        expected_count=len(runtime.test_cameras),
    )
    training_formal_sha = checkpoint["protocol_receipt"]["formal_protocol_sha256"]
    if training_formal_sha != binding.formal_protocol_sha256:
        raise RuntimeError("training/evaluation formal binding receipt mismatch")
    report: dict[str, Any] = {
        "schema": "uav-tgs-oct-formal-evaluation-v1",
        "scene_name": binding.scene_name,
        "variant": variant,
        "split": "test",
        "resolution": "r1/full-resolution",
        "formal_protocol_sha256": binding.formal_protocol_sha256,
        "training_formal_protocol_sha256": training_formal_sha,
        "evaluation_formal_protocol_sha256": binding.formal_protocol_sha256,
        "training_protocol_manifest_sha256": checkpoint["protocol_receipt"]["manifest_sha256"],
        "binding_verified_exact": True,
        "source_provenance": source_provenance,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "metrics": {
            "temperature_mae_c": float(all_abs.mean()),
            "temperature_rmse_c": float(math.sqrt(squared_sum / valid_count)),
            "temperature_bias_c": float(signed_sum / valid_count),
            "temperature_p95_abs_c": float(np.quantile(all_abs, 0.95)),
            "temperature_semantics": (
                "direct OCT rendered apparent-temperature vs float32 TSDK target; "
                "not palette-inverted"
            ),
            "palette_inverted_comparable": palette_temperature_receipt[
                "primary_supported_pixels"
            ],
            "support_masked_psnr_db_mean_per_view": (
                None
                if any(row["support_masked_psnr_db"] is None for row in rows)
                else float(np.mean([row["support_masked_psnr_db"] for row in rows]))
            ),
            "support_masked_ssim_mean_per_view": float(
                np.mean([row["support_masked_ssim"] for row in rows])
            ),
            "formal_full_frame_psnr_db": appearance_receipt["metrics"]["PSNR"],
            "formal_full_frame_ssim": appearance_receipt["metrics"]["SSIM"],
            "formal_full_frame_lpips": appearance_receipt["metrics"]["LPIPS"],
            "hotspot_iou": float(hotspot_intersection / max(hotspot_union, 1)),
            "hotspot_auprc_histogram_4096": auprc,
            "off_lut_distance": palette_temperature_receipt["off_lut_distance"][
                "supported_pixels"
            ],
            "off_lut_note": (
                "verified exact fixed-LUT display property; not treated as an accuracy win"
            ),
        },
        "hotspot_threshold": threshold_manifest,
        "cross_view_gaussian_temperature_variance_c2": {
            "status": "zero_by_construction" if variant == "oct_scalar" else "measured_visible_views",
            "population": "Gaussians with radii>0 in at least two formal test views",
            "population_count": int(measured_variance.numel()),
            "visible_view_count_p50": float(
                torch.quantile(gaussian_visible_count[variance_valid].float(), 0.50).item()
            ),
            "visible_view_count_p95": float(
                torch.quantile(gaussian_visible_count[variance_valid].float(), 0.95).item()
            ),
            "mean": float(measured_variance.mean().item()),
            "p95": float(torch.quantile(measured_variance, 0.95).item()),
            "max": float(measured_variance.max().item()),
        },
        "uncertainty_calibration": "N/A (OCT-GS v1 disables sigma/NLL)",
        "shared_occupancy_invariant": "passed",
        "alpha_backend": context.alpha_backend,
        "cost": {
            "end_to_end_evaluation": tracker.finish(),
            "pure_render": {
                "synchronized_cuda_event": True,
                "views": len(render_time_ms),
                "total_ms": float(np.sum(render_time_ms)),
                "mean_ms_per_view": float(np.mean(render_time_ms)),
                "fps": float(1000.0 / np.mean(render_time_ms)),
            },
            "formal_appearance_evaluator_wall_time_s": appearance_receipt["wall_time_s"],
            "formal_temperature_evaluator_wall_time_s": palette_temperature_receipt[
                "wall_time_s"
            ],
        },
        "appearance_evaluator": appearance_receipt,
        "palette_inverted_temperature_evaluator": palette_temperature_receipt,
        "per_view": rows,
    }
    _atomic_json(output_root / "evaluation.json", report)
    return 0


def command_cuda_smoke(args: argparse.Namespace) -> int:
    _require_absent(args.smoke_output, "CUDA smoke receipt")
    runtime = Runtime(args)
    binding = runtime.binding()
    field = OCTGaussianField(runtime.configs[args.variant]).cuda()
    optimizer = _build_optimizer(args, field)
    targets = FormalOCTTargetStore(binding, max_cache_items=2)
    camera = runtime.train_cameras[0]
    context = OCTRendererContext(runtime.anchor, runtime.proxy)
    optimizer.zero_grad(set_to_none=True)
    output = context.render(camera, field, runtime.pipe)
    loss = _loss_for_view(
        output,
        targets,
        camera,
        runtime.proxy,
        OCTLossWeights(1.0, 1.0, 1.0, 0.0),
        args.thermometric_domain,
    )
    loss["total"].backward()
    gradients = {
        name: float(parameter.grad.norm().item())
        for name, parameter in field.named_parameters()
        if parameter.grad is not None
    }
    if set(gradients) != {name for name, _ in field.named_parameters()} or not all(
        np.isfinite(value) and value > 0.0 for value in gradients.values()
    ):
        raise RuntimeError("OCT CUDA smoke has missing/non-finite/zero field gradients")
    optimizer.step()
    context.verify_anchor_unchanged()
    payload = {
        "schema": "uav-tgs-oct-cuda-smoke-v1",
        "status": "passed",
        "scene_name": binding.scene_name,
        "variant": args.variant,
        "formal_protocol_sha256": binding.formal_protocol_sha256,
        "camera_name": camera.image_name,
        "loss": float(loss["total"].detach().item()),
        "gradient_norms": gradients,
        "raster_passes": output["raster_passes"],
        "alpha_backend": output["alpha_backend"],
        "anchor_invariant": "passed",
        "uncertainty": "disabled",
    }
    _atomic_json(Path(args.smoke_output), payload)
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene", choices=("Building", "InternalRoad"), required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--anchor-iteration", type=int, required=True)
    parser.add_argument("--anchor-artifact", required=True)
    parser.add_argument("--bound-split", required=True)
    parser.add_argument("--decode-manifest", required=True)
    parser.add_argument("--decode-protocol", required=True)
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--temperature-root", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--support-manifest", required=True)
    parser.add_argument("--support-root", required=True)
    parser.add_argument("--evaluation-support-manifest", required=True)
    parser.add_argument("--evaluation-support-root", required=True)
    parser.add_argument("--camera-sequence", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--residual-bound-fraction", type=float, default=0.05)


def _add_training(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=OCT_VARIANTS, required=True)
    parser.add_argument("--thermometric-domain", choices=("celsius", "radiance"), default="celsius")
    parser.add_argument("--temperature-lr", type=float, default=DEFAULT_TEMPERATURE_LR)
    parser.add_argument("--residual-lr", type=float, default=DEFAULT_RESIDUAL_LR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-sequence")
    _add_common(prepare)
    prepare.add_argument("--sequence-seed", type=int, default=0)
    prepare.set_defaults(function=command_prepare)
    binding = commands.add_parser("build-binding")
    _add_common(binding)
    binding.add_argument("--binding-output", required=True)
    binding.set_defaults(function=command_build_binding)
    calibrate = commands.add_parser("calibrate-building")
    _add_common(calibrate)
    calibrate.add_argument("--thermometric-domain", choices=("celsius", "radiance"), default="celsius")
    calibrate.add_argument("--calibration-manifest", required=True)
    calibrate.set_defaults(function=command_calibrate)
    train = commands.add_parser("train")
    _add_common(train)
    _add_training(train)
    train.add_argument("--calibration-manifest", required=True)
    train.add_argument("--calibration-sha256", required=True)
    train.add_argument(
        "--resume-checkpoint",
        default=None,
        help="resume only from an existing formal 10k/20k endpoint in this run root",
    )
    train.set_defaults(function=command_train)
    hotspot = commands.add_parser("freeze-hotspot-threshold")
    _add_common(hotspot)
    hotspot.add_argument("--hotspot-threshold-manifest", required=True)
    hotspot.add_argument("--hotspot-quantile", type=float, default=0.95, choices=None)
    hotspot.add_argument("--hotspot-bins", type=int, default=65_536)
    hotspot.set_defaults(function=command_freeze_hotspot)
    evaluate = commands.add_parser("eval")
    _add_common(evaluate)
    _add_training(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--protocol-manifest", required=True)
    evaluate.add_argument("--hotspot-threshold-manifest", required=True)
    evaluate.add_argument("--eval-output", required=True)
    evaluate.add_argument("--compute-lpips", action="store_true")
    evaluate.set_defaults(function=command_eval)
    smoke = commands.add_parser("cuda-smoke")
    _add_common(smoke)
    _add_training(smoke)
    smoke.add_argument("--smoke-output", required=True)
    smoke.set_defaults(function=command_cuda_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _validate_recipe_args(args)
    if hasattr(args, "hotspot_quantile") and not (0.5 < args.hotspot_quantile < 1.0):
        raise ValueError("hotspot quantile must lie in (0.5,1)")
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
