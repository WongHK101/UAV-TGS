"""Fail-closed calibration, invariants, manifests, checkpoints, and costs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .field import OCTConfig, OCTGaussianField, OCT_VARIANTS, verify_field_only_optimizer
from .formal import (
    FORMAL_EXPERIMENT_RECIPE,
    FormalOCTBinding,
    sha256_file,
    sha256_json,
)
from .losses import OCTLossWeights, OCT_LOSS_FORMULA_VERSION
from .radiance import (
    BandRadianceProxy,
    METHOD_SEMANTICS,
    TARGET_SEMANTICS,
    display_metadata,
)


OCCUPANCY_SCHEMA = "uav-tgs-shared-occupancy-snapshot-v1"
CHECKPOINT_SCHEMA = "uav-tgs-oct-formal-checkpoint-v3"
PROTOCOL_SCHEMA = "uav-tgs-oct-formal-protocol-v3"
TRAINING_SOURCE_SCHEMA = "uav-tgs-oct-training-source-v1"
CALIBRATION_SCHEMA = "uav-tgs-oct-building-gradient-calibration-v2"
CALIBRATION_RECEIPT_SCHEMA = "uav-tgs-oct-calibration-source-receipt-v1"
CALIBRATION_COMPONENTS = ("thermometric", "color_l1", "color_dssim")
_OCCUPANCY_RAW_FIELDS = {
    "xyz": "_xyz",
    "scaling": "_scaling",
    "rotation": "_rotation",
    "opacity": "_opacity",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CLEAN_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
FORMAL_ENDPOINTS = tuple(int(value) for value in FORMAL_EXPERIMENT_RECIPE["endpoints"])


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _formal_endpoint(step: Any) -> int:
    if type(step) is not int or step not in FORMAL_ENDPOINTS:
        raise ValueError(
            f"formal OCT checkpoints are restricted to endpoints {FORMAL_ENDPOINTS}; "
            f"got {step!r}"
        )
    return step


def validate_training_source_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the clean-commit/source digest embedded in a formal run.

    The producer is responsible for obtaining these values from Git and the
    filesystem.  This validator deliberately rejects dirty or partial
    provenance so protocol/checkpoint consumers cannot silently downgrade it.
    """

    if not isinstance(provenance, Mapping):
        raise TypeError("formal OCT training source provenance must be a mapping")
    value = dict(provenance)
    if value.get("schema") != TRAINING_SOURCE_SCHEMA:
        raise ValueError("formal OCT training source schema mismatch")
    commit = value.get("git_commit")
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("formal OCT requires a full lowercase Git commit SHA")
    if value.get("git_clean") is not True:
        raise ValueError("formal OCT training requires a clean Git worktree")
    if value.get("git_status_porcelain_sha256") != _CLEAN_STATUS_SHA256:
        raise ValueError("formal OCT clean-status digest is invalid")
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("formal OCT training source file inventory is empty")
    paths: list[str] = []
    normalized_files: list[dict[str, Any]] = []
    for record in files:
        if not isinstance(record, Mapping):
            raise ValueError("formal OCT training source record must be a mapping")
        path = record.get("path")
        digest = record.get("sha256")
        size = record.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise ValueError(f"invalid formal OCT source path: {path!r}")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"invalid formal OCT source SHA for {path}")
        if type(size) is not int or size < 0:
            raise ValueError(f"invalid formal OCT source size for {path}")
        paths.append(path)
        normalized_files.append({"path": path, "sha256": digest, "bytes": size})
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("formal OCT source paths must be unique and sorted")
    if value.get("files_sha256") != sha256_json(normalized_files):
        raise ValueError("formal OCT source inventory hash mismatch")
    expected_keys = {
        "schema",
        "git_commit",
        "git_clean",
        "git_status_porcelain_sha256",
        "files",
        "files_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("formal OCT source provenance contains unknown/missing fields")
    return value


def verify_oct_field_finite(field: OCTGaussianField, *, label: str) -> None:
    """Fail after one device synchronization if any learned field value is non-finite."""

    parameters = tuple(field.parameters())
    if not parameters:
        raise RuntimeError("OCT field has no learned parameters")
    checks = torch.stack([torch.isfinite(parameter).all() for parameter in parameters])
    if not bool(checks.all().item()):
        bad = [
            name
            for (name, parameter), finite in zip(field.named_parameters(), checks.tolist())
            if not finite
        ]
        raise FloatingPointError(f"non-finite OCT field after {label}: {bad}")


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    byte_view = value.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(byte_view).hexdigest(),
    }


def capture_occupancy_snapshot(anchor: Any) -> dict[str, Any]:
    """Hash ordered raw RGB anchor fields defining occupancy and topology."""

    records: dict[str, Any] = {}
    counts: set[int] = set()
    for public_name, raw_name in _OCCUPANCY_RAW_FIELDS.items():
        tensor = getattr(anchor, raw_name, None)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"anchor does not expose tensor {raw_name}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"formal OCT anchor {raw_name} must be float32")
        records[public_name] = _tensor_record(tensor)
        counts.add(int(tensor.shape[0]))
    if len(counts) != 1:
        raise ValueError("anchor occupancy tensors have inconsistent topology")
    payload = {
        "schema": OCCUPANCY_SCHEMA,
        "topology_count": next(iter(counts)),
        "ordered_fields": list(_OCCUPANCY_RAW_FIELDS),
        "fields": records,
    }
    payload["overall_sha256"] = sha256_json(payload)
    return payload


def verify_occupancy_snapshot(anchor: Any, expected: Mapping[str, Any]) -> dict[str, Any]:
    if expected.get("schema") != OCCUPANCY_SCHEMA:
        raise ValueError("shared occupancy snapshot schema mismatch")
    current = capture_occupancy_snapshot(anchor)
    if current["overall_sha256"] != expected.get("overall_sha256"):
        differences: list[str] = []
        if current["topology_count"] != expected.get("topology_count"):
            differences.append("topology_count")
        expected_fields = expected.get("fields", {})
        for name in _OCCUPANCY_RAW_FIELDS:
            if current["fields"].get(name) != expected_fields.get(name):
                differences.append(name)
        raise RuntimeError(
            "shared RGB occupancy invariant failed: "
            + ", ".join(differences or ["digest"])
        )
    return current


def _validate_source_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(receipt)
    if value.get("schema") != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError("invalid OCT calibration source receipt schema")
    supplied_hash = value.get("receipt_sha256")
    basis = dict(value)
    basis.pop("receipt_sha256", None)
    if supplied_hash != sha256_json(basis):
        raise ValueError("OCT calibration source receipt hash mismatch")
    if value.get("scene_name") != "Building":
        raise ValueError("OCT calibration source must be Building")
    train_ids = value.get("train_view_ids")
    if not isinstance(train_ids, list) or not train_ids:
        raise ValueError("OCT calibration receipt has no Building train views")
    if len(train_ids) != len(set(train_ids)) or not all(
        isinstance(item, str) and item for item in train_ids
    ):
        raise ValueError("OCT calibration train view IDs are invalid")
    if value.get("train_view_ids_sha256") != sha256_json(train_ids):
        raise ValueError("OCT calibration train view hash mismatch")
    configs = value.get("field_configs")
    if not isinstance(configs, Mapping) or set(configs) != set(OCT_VARIANTS):
        raise ValueError("OCT calibration receipt must pin both v1 variants")
    for name in OCT_VARIANTS:
        config = OCTConfig.from_dict(dict(configs[name]))
        if config.variant != name:
            raise ValueError("OCT calibration receipt config/variant mismatch")
    if value.get("field_configs_sha256") != sha256_json(configs):
        raise ValueError("OCT calibration field config hash mismatch")
    return value


class BuildingGradientCalibrator:
    """One immutable calibration batch over validated Building train views."""

    def __init__(
        self,
        formal_receipt: Mapping[str, Any],
        *,
        thermometric_domain: str = "celsius",
    ) -> None:
        if thermometric_domain not in ("celsius", "radiance"):
            raise ValueError("thermometric_domain must be 'celsius' or 'radiance'")
        self.receipt = _validate_source_receipt(formal_receipt)
        self.thermometric_domain = thermometric_domain
        self.allowed_views = set(self.receipt["train_view_ids"])
        self._records: list[dict[str, Any]] = []
        self._keys: set[tuple[str, str]] = set()

    def add_gradient_norms(
        self,
        view_id: str,
        variant: str,
        norms: Mapping[str, float],
    ) -> None:
        identifier = str(view_id)
        if identifier not in self.allowed_views:
            raise ValueError(f"calibration view is not in Building train: {identifier}")
        if variant not in OCT_VARIANTS:
            raise ValueError(f"calibration variant must be one of {OCT_VARIANTS}")
        key = (variant, identifier)
        if key in self._keys:
            raise ValueError("duplicate calibration variant/view record")
        record: dict[str, Any] = {"view_id": identifier, "variant": variant}
        for component_index, name in enumerate(CALIBRATION_COMPONENTS):
            value = float(norms[name])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"gradient norm {name} must be finite and positive")
            record[name] = value
        self._records.append(record)
        self._keys.add(key)

    def observe(
        self,
        view_id: str,
        variant: str,
        component_losses: Mapping[str, torch.Tensor],
        parameters: Iterable[torch.Tensor],
    ) -> dict[str, float]:
        params = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not params:
            raise ValueError("calibration requires trainable OCT parameters")
        norms: dict[str, float] = {}
        for component_index, name in enumerate(CALIBRATION_COMPONENTS):
            loss = component_losses.get(name)
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError(f"missing scalar calibration loss {name}")
            gradients = torch.autograd.grad(
                loss,
                params,
                retain_graph=component_index < len(CALIBRATION_COMPONENTS) - 1,
                create_graph=False,
                allow_unused=True,
            )
            squared_sum = 0.0
            element_count = 0
            for parameter, gradient in zip(params, gradients):
                element_count += int(parameter.numel())
                if gradient is not None:
                    squared_sum += float(gradient.detach().double().square().sum().item())
            if element_count <= 0 or squared_sum <= 0.0:
                raise RuntimeError(f"zero/absent OCT gradient for {name}")
            norms[name] = float(np.sqrt(squared_sum / float(element_count)))
        self.add_gradient_norms(view_id, variant, norms)
        return norms

    def payload(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        observed_variants = {record["variant"] for record in self._records}
        if observed_variants != set(OCT_VARIANTS):
            raise RuntimeError("calibration must contain both OCT-Scalar and OCT-Residual")
        for variant in OCT_VARIANTS:
            observed_views = {
                record["view_id"]
                for record in self._records
                if record["variant"] == variant
            }
            if observed_views != self.allowed_views:
                missing = sorted(self.allowed_views - observed_views)
                extra = sorted(observed_views - self.allowed_views)
                raise RuntimeError(
                    f"calibration must cover every Building train view for {variant}; "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
        medians = {
            name: float(np.median([record[name] for record in self._records]))
            for name in CALIBRATION_COMPONENTS
        }
        reference = medians["thermometric"]
        weights = OCTLossWeights(
            thermometric=1.0,
            color_l1=reference / medians["color_l1"],
            color_dssim=reference / medians["color_dssim"],
            uncertainty_nll=0.0,
        )
        metadata_payload = dict(metadata or {})
        if metadata_payload.get("experiment_recipe") != FORMAL_EXPERIMENT_RECIPE:
            raise ValueError("calibration must pin the frozen paired OCT recipe")
        if metadata_payload.get("experiment_recipe_sha256") != sha256_json(
            FORMAL_EXPERIMENT_RECIPE
        ):
            raise ValueError("calibration paired-recipe hash mismatch")
        payload: dict[str, Any] = {
            "schema": CALIBRATION_SCHEMA,
            "status": "frozen",
            "created_at": _now_iso(),
            "scene_name": "Building",
            "split": "train",
            "test_statistics_used": False,
            "thermometric_domain": self.thermometric_domain,
            "variants": list(OCT_VARIANTS),
            "source_receipt": self.receipt,
            "rule": "equalize pooled median RMS field gradient to thermometric component",
            "sample_count": len(self._records),
            "median_gradient_rms": medians,
            "weights": weights.to_dict(),
            "records": self._records,
            "loss_formula_version": OCT_LOSS_FORMULA_VERSION,
            "transfer_policy": "same calibration SHA/domain for InternalRoad",
            "metadata": metadata_payload,
        }
        payload["calibration_sha256"] = sha256_json(payload)
        return payload

    def freeze(
        self,
        output_path: str | Path,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.payload(metadata)
        _atomic_json(Path(output_path), payload)
        return payload


def load_frozen_calibration(
    path: str | Path,
    *,
    expected_domain: str | None = None,
    expected_variant: str | None = None,
    expected_calibration_sha256: str | None = None,
    building_receipt: Mapping[str, Any] | None = None,
    consumer_scene: str | None = None,
) -> tuple[dict[str, Any], OCTLossWeights]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != CALIBRATION_SCHEMA or payload.get("status") != "frozen":
        raise ValueError("invalid OCT calibration manifest")
    if payload.get("scene_name") != "Building" or payload.get("split") != "train":
        raise ValueError("OCT calibration must originate from Building train")
    if payload.get("test_statistics_used") is not False:
        raise ValueError("OCT calibration does not prove test exclusion")
    if payload.get("loss_formula_version") != OCT_LOSS_FORMULA_VERSION:
        raise ValueError("OCT calibration loss formula version mismatch")
    calibration_metadata = payload.get("metadata")
    if not isinstance(calibration_metadata, Mapping):
        raise ValueError("OCT calibration metadata is missing")
    if calibration_metadata.get("experiment_recipe") != FORMAL_EXPERIMENT_RECIPE:
        raise ValueError("OCT calibration paired recipe mismatch")
    if calibration_metadata.get("experiment_recipe_sha256") != sha256_json(
        FORMAL_EXPERIMENT_RECIPE
    ):
        raise ValueError("OCT calibration paired-recipe hash mismatch")
    supplied_hash = payload.get("calibration_sha256")
    basis = dict(payload)
    basis.pop("calibration_sha256", None)
    if supplied_hash != sha256_json(basis):
        raise ValueError("OCT calibration manifest hash mismatch")
    if expected_calibration_sha256 is not None and supplied_hash != expected_calibration_sha256:
        raise ValueError("OCT calibration SHA differs from the expected frozen receipt")
    if consumer_scene == "InternalRoad" and expected_calibration_sha256 is None:
        raise ValueError("InternalRoad requires an explicit expected Building calibration SHA")
    receipt = _validate_source_receipt(payload.get("source_receipt", {}))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("OCT calibration has no gradient records")
    expected_keys = {
        (variant, view_id)
        for variant in OCT_VARIANTS
        for view_id in receipt["train_view_ids"]
    }
    actual_keys = {
        (record.get("variant"), record.get("view_id"))
        for record in records
        if isinstance(record, Mapping)
    }
    if len(records) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError("OCT calibration does not exactly cover both variants/Building train")
    medians: dict[str, float] = {}
    for name in CALIBRATION_COMPONENTS:
        values = np.asarray([float(record[name]) for record in records], dtype=np.float64)
        if not bool(np.isfinite(values).all()) or bool((values <= 0.0).any()):
            raise ValueError(f"invalid OCT calibration gradient record {name}")
        medians[name] = float(np.median(values))
    if payload.get("median_gradient_rms") != medians:
        raise ValueError("OCT calibration median gradients are inconsistent")
    expected_weights = OCTLossWeights(
        thermometric=1.0,
        color_l1=medians["thermometric"] / medians["color_l1"],
        color_dssim=medians["thermometric"] / medians["color_dssim"],
        uncertainty_nll=0.0,
    )
    if payload.get("weights") != expected_weights.to_dict():
        raise ValueError("OCT calibration weights do not follow the frozen rule")
    if building_receipt is not None:
        expected_receipt = _validate_source_receipt(building_receipt)
        if receipt["receipt_sha256"] != expected_receipt["receipt_sha256"]:
            raise ValueError("Building calibration/source formal protocol mismatch")
    if expected_domain is not None and payload.get("thermometric_domain") != expected_domain:
        raise ValueError("OCT calibration thermometric domain mismatch")
    if expected_variant is not None and expected_variant not in payload.get("variants", []):
        raise ValueError("OCT calibration does not cover requested variant")
    weights = expected_weights
    weights.validate()
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_file_sha256"] = sha256_file(manifest_path)
    return payload, weights


class OCTStageCostTracker:
    """Boundary wall-time/VRAM plus explicit raster-pass accounting."""

    def __init__(self, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata = dict(metadata or {})
        self.started_at: str | None = None
        self._start_perf: float | None = None
        self.steps = 0
        self.views = 0
        self.raster_passes = 0
        self._peak_reset = False

    def start(self) -> None:
        if self._start_perf is not None:
            raise RuntimeError("cost tracker already started")
        self.started_at = _now_iso()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._peak_reset = True
        self._start_perf = time.perf_counter()

    def record_step(self, *, rendered_views: int = 1, raster_passes: int = 1) -> None:
        if self._start_perf is None:
            raise RuntimeError("cost tracker has not started")
        if int(rendered_views) <= 0 or int(raster_passes) <= 0:
            raise ValueError("rendered_views and raster_passes must be positive")
        self.steps += 1
        self.views += int(rendered_views)
        self.raster_passes += int(raster_passes)

    def snapshot(self, status: str = "running") -> dict[str, Any]:
        if self._start_perf is None:
            raise RuntimeError("cost tracker has not started")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = max(0.0, time.perf_counter() - self._start_perf)
        device = {
            "cuda_available": bool(torch.cuda.is_available()),
            "device_name": None,
            "peak_torch_allocated_bytes": None,
            "peak_torch_reserved_bytes": None,
        }
        if torch.cuda.is_available():
            device.update(
                {
                    "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
                    "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            )
        return {
            "schema": "uav-tgs-oct-cost-v1",
            "status": status,
            "started_at": self.started_at,
            "captured_at": _now_iso(),
            "wall_time_s": elapsed,
            "optimizer_steps": self.steps,
            "rendered_views": self.views,
            "raster_passes": self.raster_passes,
            "ms_per_step": None if self.steps == 0 else elapsed * 1000.0 / self.steps,
            "raster_passes_per_view": None if self.views == 0 else self.raster_passes / self.views,
            "peak_memory_reset_succeeded": self._peak_reset,
            "device": device,
            "metadata": self.metadata,
        }

    def finish(self, status: str = "completed") -> dict[str, Any]:
        return self.snapshot(status=status)


def _validate_protocol_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("OCT formal protocol schema mismatch")
    supplied_hash = payload.get("manifest_sha256")
    basis = dict(payload)
    basis.pop("manifest_sha256", None)
    if supplied_hash != sha256_json(basis):
        raise ValueError("OCT formal protocol manifest hash mismatch")
    formal = dict(payload.get("formal_binding", {}))
    formal_hash = formal.get("formal_protocol_sha256")
    formal_basis = dict(formal)
    formal_basis.pop("formal_protocol_sha256", None)
    if formal_hash != sha256_json(formal_basis):
        raise ValueError("embedded formal binding hash mismatch")
    source = validate_training_source_provenance(payload.get("source_provenance", {}))
    if payload.get("git_commit") != source["git_commit"]:
        raise ValueError("formal protocol Git commit/source provenance mismatch")
    payload["manifest_path"] = str(manifest_path)
    payload["manifest_file_sha256"] = sha256_file(manifest_path)
    return payload


def load_oct_protocol_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fully validate an immutable formal OCT protocol manifest."""

    return _validate_protocol_manifest(path)


def write_oct_protocol_manifest(
    output_path: str | Path,
    *,
    field: OCTGaussianField,
    anchor_snapshot: Mapping[str, Any],
    radiance_proxy: BandRadianceProxy,
    calibration_manifest: str | Path,
    formal_binding: FormalOCTBinding,
    expected_calibration_sha256: str | None,
    thermometric_domain: str,
    optimizer_config: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    scene = formal_binding.scene_name
    variant = field.config.variant
    expected_config = formal_binding.payload["field_configs"]["payload"][variant]
    if field.config.to_dict() != expected_config:
        raise ValueError("field config differs from immutable formal binding")
    if anchor_snapshot.get("overall_sha256") != formal_binding.payload["anchor"]["occupancy_sha256"]:
        raise ValueError("anchor snapshot differs from immutable formal binding")
    if radiance_proxy.metadata() != formal_binding.payload["radiance_proxy"]:
        raise ValueError("radiance proxy differs from immutable formal binding")
    target_contract = formal_binding.payload.get("tsdk_target", {})
    if target_contract.get("target_semantics") != TARGET_SEMANTICS:
        raise ValueError("formal target is not TSDK-referenced apparent temperature")
    if target_contract.get("method_semantics") != METHOD_SEMANTICS:
        raise ValueError("formal target/radiance-proxy semantics mismatch")
    if target_contract.get("environmental_correction_reapplied_by_oct") is not False:
        raise ValueError("OCT must not repeat the TSDK environmental correction")
    building_receipt = (
        formal_binding.calibration_receipt() if scene == "Building" else None
    )
    calibration, weights = load_frozen_calibration(
        calibration_manifest,
        expected_domain=thermometric_domain,
        expected_variant=variant,
        expected_calibration_sha256=expected_calibration_sha256,
        building_receipt=building_receipt,
        consumer_scene=scene,
    )
    optimizer_payload = dict(optimizer_config)
    if optimizer_payload.get("type") != "Adam" or optimizer_payload.get("field_only") is not True:
        raise ValueError("formal OCT optimizer must be field-only Adam")
    for key in ("temperature_lr", "adam_eps"):
        value = float(optimizer_payload.get(key))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid OCT optimizer setting {key}")
    residual_lr = optimizer_payload.get("residual_lr")
    if variant == "oct_residual":
        if residual_lr is None or not np.isfinite(float(residual_lr)) or float(residual_lr) <= 0.0:
            raise ValueError("OCT-Residual protocol requires residual_lr")
    elif residual_lr is not None:
        raise ValueError("OCT-Scalar protocol must not declare residual_lr")
    paired_recipe = formal_binding.payload.get("experiment_recipe", {})
    if paired_recipe.get("payload") != FORMAL_EXPERIMENT_RECIPE or paired_recipe.get(
        "sha256"
    ) != sha256_json(FORMAL_EXPERIMENT_RECIPE):
        raise ValueError("formal binding paired recipe mismatch")
    if thermometric_domain != FORMAL_EXPERIMENT_RECIPE["thermometric_domain"]:
        raise ValueError("thermometric domain differs from the paired recipe")
    expected_optimizer = {
        "temperature_lr": FORMAL_EXPERIMENT_RECIPE["temperature_lr"],
        "adam_eps": FORMAL_EXPERIMENT_RECIPE["adam_eps"],
        "schedule": FORMAL_EXPERIMENT_RECIPE["schedule"],
        "steps": FORMAL_EXPERIMENT_RECIPE["steps"],
    }
    for key, expected in expected_optimizer.items():
        if optimizer_payload.get(key) != expected:
            raise ValueError(f"optimizer {key} differs from the frozen paired recipe")
    expected_residual_lr = (
        FORMAL_EXPERIMENT_RECIPE["residual_lr"] if variant == "oct_residual" else None
    )
    if optimizer_payload.get("residual_lr") != expected_residual_lr:
        raise ValueError("optimizer residual_lr differs from the frozen paired recipe")
    source_payload = validate_training_source_provenance(source_provenance)
    if git_commit is not None and git_commit != source_payload["git_commit"]:
        raise ValueError("explicit Git commit differs from training source provenance")
    payload: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "created_at": _now_iso(),
        "scene_name": scene,
        "variant": variant,
        "git_commit": source_payload["git_commit"],
        "source_provenance": source_payload,
        "formal_binding": formal_binding.immutable_summary(),
        "field": field.protocol_metadata(),
        "anchor_snapshot": dict(anchor_snapshot),
        "loss": {
            "formula_version": OCT_LOSS_FORMULA_VERSION,
            "thermometric_domain": thermometric_domain,
            "weights": weights.to_dict(),
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_manifest_file_sha256": calibration["manifest_file_sha256"],
            "calibration_source_receipt_sha256": calibration["source_receipt"]["receipt_sha256"],
        },
        "optimizer": optimizer_payload,
        "display": display_metadata(),
        "invariants": {
            "shared_xyz_scaling_rotation_opacity_topology": True,
            "thermal_opacity_exists": False,
            "thermal_geometry_exists": False,
            "uncertainty_nll_enabled": False,
            "densification": False,
            "pruning": False,
        },
        "claim_boundary": {
            "measurement_conditioned_apparent_radiance": True,
            "target_semantics": TARGET_SEMANTICS,
            "renderer_semantics": METHOD_SEMANTICS,
            "absolute_thermometry": False,
            "tsdk_correction_reapplied_in_renderer": False,
            "emissivity_reflection_atmosphere_are_protocol_metadata_only": True,
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)
    _atomic_json(Path(output_path), payload)
    return payload


def _protocol_receipt(protocol: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    return {
        "manifest_file_sha256": sha256_file(path),
        "manifest_sha256": protocol["manifest_sha256"],
        "formal_protocol_sha256": protocol["formal_binding"]["formal_protocol_sha256"],
        "anchor_occupancy_sha256": protocol["anchor_snapshot"]["overall_sha256"],
        "field_config_sha256": sha256_json(protocol["field"]["config"]),
        "calibration_sha256": protocol["loss"]["calibration_sha256"],
        "thermometric_domain": protocol["loss"]["thermometric_domain"],
    }


def _verify_optimizer_matches_protocol(
    optimizer: torch.optim.Optimizer,
    protocol: Mapping[str, Any],
) -> None:
    recipe = protocol["optimizer"]
    expected_lrs = {"apparent_temperature": float(recipe["temperature_lr"])}
    if protocol["variant"] == "oct_residual":
        expected_lrs["bounded_view_residual"] = float(recipe["residual_lr"])
    actual_groups = {str(group.get("name")): group for group in optimizer.param_groups}
    if set(actual_groups) != set(expected_lrs):
        raise RuntimeError("checkpoint optimizer groups differ from formal protocol")
    for name, expected_lr in expected_lrs.items():
        group = actual_groups[name]
        if float(group.get("lr")) != expected_lr:
            raise RuntimeError(f"checkpoint optimizer LR differs for {name}")
        if float(group.get("eps")) != float(recipe["adam_eps"]):
            raise RuntimeError(f"checkpoint optimizer Adam eps differs for {name}")


def _verify_adam_state_at_step(
    field: OCTGaussianField,
    optimizer: torch.optim.Optimizer,
    expected_step: int,
) -> None:
    """Require complete, finite Adam moments at the exact global step."""

    if type(expected_step) is not int or expected_step <= 0:
        raise ValueError("restored OCT optimizer step must be a positive integer")
    expected_group_parameters = {
        "apparent_temperature": field.raw_base_temperature,
    }
    if field.raw_residual_amplitude is not None:
        expected_group_parameters["bounded_view_residual"] = field.raw_residual_amplitude
    actual_groups = {str(group.get("name")): group for group in optimizer.param_groups}
    if set(actual_groups) != set(expected_group_parameters):
        raise RuntimeError("Adam parameter groups do not match the OCT variant")
    for group_name, expected_parameter in expected_group_parameters.items():
        parameters = actual_groups[group_name].get("params", [])
        if len(parameters) != 1 or parameters[0] is not expected_parameter:
            raise RuntimeError(f"Adam group {group_name} is bound to the wrong OCT parameter")
    finite_checks: list[torch.Tensor] = []
    for name, parameter in field.named_parameters():
        finite_checks.append(torch.isfinite(parameter).all())
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise RuntimeError(f"missing Adam state for OCT parameter {name}")
        step_value = state.get("step")
        if isinstance(step_value, torch.Tensor):
            if step_value.numel() != 1 or not bool(torch.isfinite(step_value).item()):
                raise RuntimeError(f"invalid Adam step state for OCT parameter {name}")
            actual_step = int(step_value.item())
            if float(step_value.item()) != float(actual_step):
                raise RuntimeError(f"non-integral Adam step for OCT parameter {name}")
        elif type(step_value) in (int, float):
            if not np.isfinite(float(step_value)) or float(step_value) != int(step_value):
                raise RuntimeError(f"invalid Adam step state for OCT parameter {name}")
            actual_step = int(step_value)
        else:
            raise RuntimeError(f"missing Adam step state for OCT parameter {name}")
        if actual_step != expected_step:
            raise RuntimeError(
                f"Adam/global step mismatch for OCT parameter {name}: "
                f"state={actual_step} checkpoint={expected_step}"
            )
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = state.get(moment_name)
            if not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape:
                raise RuntimeError(
                    f"missing/malformed Adam {moment_name} for OCT parameter {name}"
                )
            if moment.device != parameter.device or moment.dtype != parameter.dtype:
                raise RuntimeError(
                    f"Adam {moment_name} device/dtype mismatch for OCT parameter {name}"
                )
            finite_checks.append(torch.isfinite(moment).all())
    if not finite_checks or not bool(torch.stack(finite_checks).all().item()):
        raise FloatingPointError(
            f"non-finite OCT field/Adam state at global step {expected_step}"
        )


def verify_oct_post_step_finite(
    field: OCTGaussianField,
    optimizer: torch.optim.Optimizer,
    step: int,
) -> None:
    """Per-update formal safety check for parameters, moments, and Adam step."""

    _verify_adam_state_at_step(field, optimizer, step)


def restore_oct_optimizer_state(
    *,
    field: OCTGaussianField,
    optimizer: torch.optim.Optimizer,
    checkpoint_metadata: Mapping[str, Any],
    anchor: Any,
    protocol_manifest: str | Path,
) -> int:
    """Restore Adam exactly and verify its state/global-step/ownership contract."""

    protocol = _validate_protocol_manifest(protocol_manifest)
    step = _formal_endpoint(checkpoint_metadata.get("step"))
    if checkpoint_metadata.get("sequence_offset") != step:
        raise ValueError("checkpoint sequence offset differs from its global step")
    if checkpoint_metadata.get("protocol_receipt") != _protocol_receipt(
        protocol, protocol_manifest
    ):
        raise ValueError("optimizer checkpoint/protocol receipt mismatch")
    state_dict = checkpoint_metadata.get("optimizer_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("OCT checkpoint has no Adam state")
    optimizer.load_state_dict(dict(state_dict))
    forbidden = [getattr(anchor, name) for name in _OCCUPANCY_RAW_FIELDS.values()]
    verify_field_only_optimizer(field, optimizer, forbidden)
    _verify_optimizer_matches_protocol(optimizer, protocol)
    _verify_adam_state_at_step(field, optimizer, step)
    verify_oct_field_finite(field, label=f"checkpoint step {step} restore")
    return step


def save_oct_checkpoint(
    output_path: str | Path,
    *,
    field: OCTGaussianField,
    anchor: Any,
    anchor_snapshot: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    step: int,
    protocol_manifest: str | Path,
    cost_summary: Mapping[str, Any],
) -> Path:
    checkpoint_step = _formal_endpoint(step)
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"formal OCT checkpoint already exists: {target.resolve()}")
    protocol = _validate_protocol_manifest(protocol_manifest)
    verify_occupancy_snapshot(anchor, anchor_snapshot)
    if anchor_snapshot.get("overall_sha256") != protocol["anchor_snapshot"]["overall_sha256"]:
        raise RuntimeError("checkpoint anchor differs from protocol anchor")
    if field.config.to_dict() != protocol["field"]["config"]:
        raise RuntimeError("checkpoint field config differs from protocol")
    forbidden = [getattr(anchor, name) for name in _OCCUPANCY_RAW_FIELDS.values()]
    verify_field_only_optimizer(field, optimizer, forbidden)
    _verify_optimizer_matches_protocol(optimizer, protocol)
    verify_oct_field_finite(field, label=f"checkpoint step {checkpoint_step}")
    _verify_adam_state_at_step(field, optimizer, checkpoint_step)
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "created_at": _now_iso(),
        "step": checkpoint_step,
        "sequence_offset": checkpoint_step,
        "field_config": field.config.to_dict(),
        "field_state_dict": field.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "anchor_snapshot": dict(anchor_snapshot),
        "protocol_receipt": _protocol_receipt(protocol, protocol_manifest),
        "cost_summary": dict(cost_summary),
        "contains_anchor_tensors": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale formal OCT checkpoint temporary exists: {temporary}")
    torch.save(payload, temporary)
    if target.exists():
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"formal OCT checkpoint appeared during save: {target.resolve()}")
    try:
        # Hard-link publication is atomic and fails if another writer created
        # the endpoint after our preflight check; unlike os.replace it never
        # overwrites an immutable formal artifact.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _torch_load(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("OCT checkpoint must contain a mapping")
    return payload


def inspect_oct_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    payload = _torch_load(path, "cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("OCT checkpoint schema mismatch")
    step = _formal_endpoint(payload.get("step"))
    if payload.get("sequence_offset") != step:
        raise ValueError("OCT checkpoint sequence offset/global step mismatch")
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "schema": payload["schema"],
        "step": step,
        "sequence_offset": payload["sequence_offset"],
        "field_config": payload["field_config"],
        "protocol_receipt": payload["protocol_receipt"],
        "anchor_snapshot": payload["anchor_snapshot"],
        "contains_anchor_tensors": payload["contains_anchor_tensors"],
        "has_optimizer_state": payload.get("optimizer_state_dict") is not None,
    }


def load_oct_checkpoint(
    checkpoint_path: str | Path,
    *,
    anchor: Any,
    protocol_manifest: str | Path,
    formal_binding: FormalOCTBinding,
    map_location: str | torch.device = "cpu",
) -> tuple[OCTGaussianField, dict[str, Any]]:
    path = Path(checkpoint_path).resolve()
    payload = _torch_load(path, map_location)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("OCT checkpoint schema mismatch")
    step = _formal_endpoint(payload.get("step"))
    if payload.get("sequence_offset") != step:
        raise ValueError("OCT checkpoint sequence offset/global step mismatch")
    if payload.get("contains_anchor_tensors") is not False:
        raise ValueError("OCT checkpoint violates the sidecar-only contract")
    protocol = _validate_protocol_manifest(protocol_manifest)
    current_binding = formal_binding.immutable_summary()
    if protocol.get("formal_binding") != current_binding:
        raise ValueError(
            "checkpoint training binding differs from the requested evaluation binding"
        )
    if payload.get("protocol_receipt") != _protocol_receipt(protocol, protocol_manifest):
        raise ValueError("checkpoint/protocol immutable receipt mismatch")
    snapshot = payload.get("anchor_snapshot", {})
    verify_occupancy_snapshot(anchor, snapshot)
    if snapshot.get("overall_sha256") != protocol["anchor_snapshot"]["overall_sha256"]:
        raise ValueError("checkpoint anchor is not the protocol anchor")
    config = OCTConfig.from_dict(payload["field_config"])
    if config.to_dict() != protocol["field"]["config"]:
        raise ValueError("checkpoint field config is not the protocol field config")
    expected_config = current_binding["field_configs"]["payload"].get(config.variant)
    if config.to_dict() != expected_config:
        raise ValueError("checkpoint field config differs from the evaluation binding")
    field = OCTGaussianField(config)
    field.load_state_dict(payload["field_state_dict"], strict=True)
    verify_oct_field_finite(field, label=f"checkpoint step {step} load")
    metadata = {key: value for key, value in payload.items() if key != "field_state_dict"}
    metadata["checkpoint_path"] = str(path)
    metadata["checkpoint_sha256"] = sha256_file(path)
    return field, metadata
