"""Pure PyTorch utilities for the fixed-anchor OGS-v1 protocol.

This module deliberately has no dependency on the training loop or renderer.  It
contains the numerically sensitive observability audit, the differentiable OGS
loss, the compact training-cache contract, and deterministic scale matching.

The public loss accepts *activated* Gaussian scales (positive axis standard
deviations) and raw quaternions in the repository's ``(w, x, y, z)`` order.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


OGS_CACHE_SCHEMA_VERSION = "ogs_v1_anchor_cache_v1"
OGS_V1_FORMULA_CONTRACT_SCHEMA = "uav-tgs-ogs-v1-formula-contract-v1"
OGS_V1_MIN_VISIBLE_VIEWS = 8
OGS_V1_MIN_ACTIVATED_OPACITY = 0.01
OGS_V1_RHO = 3.0
OGS_V1_LOSS_EPS = 1e-8
OGS_V1_VARIANCE_EPS = 1e-16
OGS_V1_RECALIBRATED_LAMBDA_MODE = (
    "train_only_gradient_probe_recalibrated_1_1"
)
OGS_V1_INITIAL_LAMBDA_MODE = "initial_1e-3"
OGS_V1_RECALIBRATION = {
    "calibration_source": "train_only_gradient_probe",
    "previous_unweighted_lr_scaled_ratio_median": 0.01832437506695896,
    "target_lower_ratio": 0.02,
    "derived_lambda": 1.0914424053708873,
    "deployed_lambda": 1.1,
}
# This is intentionally a literal pin rather than a value computed at import.
# ``verify_ogs_v1_formula_hash`` recomputes the versioned semantic contract and
# normalized implementation AST, then fails closed if either drifts.
OGS_V1_FORMULA_SHA256 = (
    "110c632eb36d19a1a76882ff4c91f50f90825ff01a85fb75685396603be344e2"
)
_CACHE_TENSOR_KEYS = (
    "observability",
    "weakest_direction",
    "perpendicular_thickness",
    "eligible_mask",
    "visible_count",
)


def ogs_v1_formula_contract() -> Dict[str, Any]:
    """Return the versioned semantic contract hashed by the OGS-v1 pilot.

    Expressions are deliberately explicit rather than prose summaries.  The
    hash additionally covers normalized AST for the implementation functions
    listed in ``_ogs_v1_formula_implementation`` below.
    """

    return {
        "schema": OGS_V1_FORMULA_CONTRACT_SCHEMA,
        "observability": {
            "visibility": "projected_radii_strict_gt_0",
            "bearing": "unit(camera_center-gaussian_xyz)",
            "moment_accumulation_dtype": "float64",
            "M": "mean(v*v^T)",
            "H": "I-M",
            "symmetrize_M_and_H": True,
            "weakest_direction": "eigenvector_of_lambda_min(H)",
            "normalized_observability": (
                "clip(3*lambda_min(H)/(trace(H)+eps),0,1)"
            ),
        },
        "eligibility": {
            "fixed_at_anchor": True,
            "minimum_projected_visible_views_inclusive": (
                OGS_V1_MIN_VISIBLE_VIEWS
            ),
            "minimum_activated_opacity_strict_gt": (
                OGS_V1_MIN_ACTIVATED_OPACITY
            ),
        },
        "anchor_thickness": {
            "t_parallel0_squared": "n^T*Sigma0*n",
            "t_perp0_squared": (
                "max((trace(Sigma0)-t_parallel0_squared)/2,variance_eps)"
            ),
            "detached": ["observability", "weakest_direction", "t_perp0"],
        },
        "dynamic_thickness": {
            "Sigma": "R*diag(activated_scale^2)*R^T",
            "t_parallel": "sqrt(max(n^T*Sigma*n,variance_eps))",
            "dynamic_fields": ["scaling", "rotation"],
        },
        "loss": {
            "rho": OGS_V1_RHO,
            "eps": OGS_V1_LOSS_EPS,
            "penalty": (
                "relu(log(t_parallel+eps)-log(rho*t_perp0+eps))^2"
            ),
            "risk": "(1-observability)^2*penalty",
            "reduction": "sum(eligible*risk)/fixed_eligible_count",
            "zero_eligible": "fail_closed",
            "gradient_scope": ["scaling", "rotation"],
        },
        "topology": {
            "fixed": True,
            "densification": False,
            "pruning": False,
        },
        "numeric": {
            "loss_eps": OGS_V1_LOSS_EPS,
            "variance_eps": OGS_V1_VARIANCE_EPS,
        },
    }


def build_ogs_eligibility_mask(
    visible_count: torch.Tensor,
    activated_opacity: torch.Tensor,
    min_visible_views: int = OGS_V1_MIN_VISIBLE_VIEWS,
    min_activated_opacity: float = OGS_V1_MIN_ACTIVATED_OPACITY,
) -> torch.Tensor:
    """Return the fixed anchor-side OGS-v1 eligibility mask."""

    if visible_count.ndim != 1:
        raise ValueError("visible_count must have shape (N,)")
    opacity = activated_opacity.reshape(-1)
    if opacity.shape != visible_count.shape:
        raise ValueError("activated_opacity and visible_count must have the same length")
    if min_visible_views != OGS_V1_MIN_VISIBLE_VIEWS:
        raise ValueError(
            "OGS-v1 pins minimum visible views to "
            f"{OGS_V1_MIN_VISIBLE_VIEWS}"
        )
    if float(min_activated_opacity) != OGS_V1_MIN_ACTIVATED_OPACITY:
        raise ValueError(
            "OGS-v1 pins the strict opacity threshold to "
            f"{OGS_V1_MIN_ACTIVATED_OPACITY}"
        )
    return (
        visible_count >= OGS_V1_MIN_VISIBLE_VIEWS
    ) & (opacity.to(device=visible_count.device) > OGS_V1_MIN_ACTIVATED_OPACITY)


def _require_shape(tensor: torch.Tensor, shape_tail: Tuple[int, ...], name: str) -> None:
    if tensor.ndim != len(shape_tail) + 1 or tuple(tensor.shape[1:]) != shape_tail:
        expected = "(N{})".format(
            "".join(", {}".format(value) for value in shape_tail)
        )
        raise ValueError("{} must have shape {}; got {}".format(name, expected, tuple(tensor.shape)))


def quaternion_to_rotation(rotations: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert ``(w, x, y, z)`` quaternions to device-aware rotation matrices."""

    _require_shape(rotations, (4,), "rotations")
    norms = torch.linalg.vector_norm(rotations, dim=-1, keepdim=True)
    if not bool(torch.isfinite(norms).all()):
        raise ValueError("rotations contain NaN or Inf")
    if bool((norms <= eps).any()):
        raise ValueError("rotations contain a zero-norm quaternion")
    q = rotations / norms
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def covariance_thickness(
    scales: torch.Tensor,
    rotations: torch.Tensor,
    directions: torch.Tensor,
    variance_eps: float = 1e-16,
) -> Dict[str, torch.Tensor]:
    """Compute thickness along ``directions`` and in its perpendicular plane.

    ``scales`` are activated Gaussian axis standard deviations.  The covariance
    is ``R diag(scales**2) R^T``.  The full covariance is not materialized.
    """

    _require_shape(scales, (3,), "scales")
    _require_shape(rotations, (4,), "rotations")
    _require_shape(directions, (3,), "directions")
    if not (scales.shape[0] == rotations.shape[0] == directions.shape[0]):
        raise ValueError("scales, rotations, and directions must have the same length")
    if bool((scales <= 0).any()) or not bool(torch.isfinite(scales).all()):
        raise ValueError("activated scales must be finite and strictly positive")

    directions = directions.to(device=scales.device, dtype=scales.dtype)
    direction_norm = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if bool((direction_norm <= variance_eps).any()) or not bool(
        torch.isfinite(direction_norm).all()
    ):
        raise ValueError("directions must be finite and non-zero")
    directions = directions / direction_norm
    rotation_matrices = quaternion_to_rotation(
        rotations.to(device=scales.device, dtype=scales.dtype)
    )
    local_directions = torch.bmm(
        rotation_matrices.transpose(1, 2), directions.unsqueeze(-1)
    ).squeeze(-1)
    scale_squared = scales.square()
    parallel_squared = (local_directions.square() * scale_squared).sum(dim=-1)
    trace = scale_squared.sum(dim=-1)
    perpendicular_squared = torch.clamp(
        (trace - parallel_squared) * 0.5, min=variance_eps
    )
    return {
        "parallel_squared": parallel_squared,
        "parallel": torch.sqrt(torch.clamp(parallel_squared, min=variance_eps)),
        "perpendicular_squared": perpendicular_squared,
        "perpendicular": torch.sqrt(perpendicular_squared),
        "trace": trace,
    }


def initialize_bearing_moment_accumulators(
    gaussian_count: int,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Allocate the float64 streaming second-moment and count accumulators."""

    if gaussian_count <= 0:
        raise ValueError("gaussian_count must be positive")
    return (
        torch.zeros((gaussian_count, 3, 3), dtype=torch.float64, device=device),
        torch.zeros((gaussian_count,), dtype=torch.int64, device=device),
    )


@torch.no_grad()
def accumulate_camera_bearings(
    moment_sum: torch.Tensor,
    visible_count: torch.Tensor,
    xyz: torch.Tensor,
    camera_center: torch.Tensor,
    visibility: torch.Tensor,
    eps: float = 1e-15,
) -> int:
    """Accumulate unit camera-bearing outer products for one camera.

    ``visibility`` may be an ``(N,)`` boolean mask or a one-dimensional index
    tensor.  The direction sign is immaterial because the accumulated statistic
    is ``v v^T``.
    """

    _require_shape(moment_sum, (3, 3), "moment_sum")
    _require_shape(xyz, (3,), "xyz")
    if visible_count.ndim != 1 or visible_count.shape[0] != moment_sum.shape[0]:
        raise ValueError("visible_count must have shape (N,)")
    if xyz.shape[0] != moment_sum.shape[0]:
        raise ValueError("xyz and accumulators must have the same length")
    if moment_sum.dtype != torch.float64:
        raise ValueError("moment_sum must use float64")
    if visible_count.dtype not in (torch.int32, torch.int64):
        raise ValueError("visible_count must use an integer dtype")

    if visibility.dtype == torch.bool:
        if visibility.ndim != 1 or visibility.shape[0] != xyz.shape[0]:
            raise ValueError("boolean visibility must have shape (N,)")
        indices = torch.nonzero(visibility, as_tuple=False).squeeze(-1)
    else:
        if visibility.ndim != 1:
            raise ValueError("visibility indices must be one-dimensional")
        indices = visibility.to(dtype=torch.int64)
    indices = indices.to(device=moment_sum.device)
    if indices.numel() == 0:
        return 0
    if int(indices.min()) < 0 or int(indices.max()) >= xyz.shape[0]:
        raise IndexError("visibility contains an out-of-range Gaussian index")

    selected_xyz = xyz.index_select(0, indices.to(device=xyz.device)).to(
        device=moment_sum.device, dtype=torch.float64
    )
    center = torch.as_tensor(
        camera_center, device=moment_sum.device, dtype=torch.float64
    ).reshape(-1)
    if center.numel() != 3:
        raise ValueError("camera_center must contain three values")
    bearings = center.unsqueeze(0) - selected_xyz
    norms = torch.linalg.vector_norm(bearings, dim=-1, keepdim=True)
    valid = torch.isfinite(norms.squeeze(-1)) & (norms.squeeze(-1) > eps)
    indices = indices[valid]
    bearings = bearings[valid] / norms[valid]
    if indices.numel() == 0:
        return 0
    outer = bearings.unsqueeze(-1) * bearings.unsqueeze(-2)
    moment_sum.index_add_(0, indices, outer)
    visible_count.index_add_(
        0, indices, torch.ones_like(indices, dtype=visible_count.dtype)
    )
    return int(indices.numel())


def _canonicalize_vector_sign(vectors: torch.Tensor) -> torch.Tensor:
    """Choose a deterministic representative of the ``n``/``-n`` eigenvector."""

    largest_component = vectors.abs().argmax(dim=-1, keepdim=True)
    pivot = vectors.gather(-1, largest_component)
    sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
    return vectors * sign


def compute_observability_from_moments(
    moment_sum: torch.Tensor,
    visible_count: torch.Tensor,
    chunk_size: Optional[int] = 131072,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Compute float64 OGS observability and eigendiagnostics in chunks."""

    _require_shape(moment_sum, (3, 3), "moment_sum")
    if visible_count.ndim != 1 or visible_count.shape[0] != moment_sum.shape[0]:
        raise ValueError("visible_count must have shape (N,)")
    if moment_sum.dtype != torch.float64:
        raise ValueError("moment_sum must use float64")
    if bool((visible_count < 0).any()):
        raise ValueError("visible_count cannot be negative")
    count = moment_sum.shape[0]
    if chunk_size is None:
        chunk_size = count
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    device = moment_sum.device
    observability = torch.zeros(count, dtype=torch.float64, device=device)
    weakest_direction = torch.zeros((count, 3), dtype=torch.float64, device=device)
    weakest_direction[:, 0] = 1.0
    eigenvalues = torch.full((count, 3), torch.nan, dtype=torch.float64, device=device)
    weak_eigengap = torch.full((count,), torch.nan, dtype=torch.float64, device=device)
    min_eigengap = torch.full((count,), torch.nan, dtype=torch.float64, device=device)
    trace_m_residual = torch.full((count,), torch.nan, dtype=torch.float64, device=device)
    trace_h_residual = torch.full((count,), torch.nan, dtype=torch.float64, device=device)
    eigen_finite = torch.zeros(count, dtype=torch.bool, device=device)
    eigen_in_range = torch.zeros(count, dtype=torch.bool, device=device)
    valid_observation = visible_count > 0
    identity = torch.eye(3, dtype=torch.float64, device=device)

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        local_valid = valid_observation[start:end]
        if not bool(local_valid.any()):
            continue
        valid_indices = torch.nonzero(local_valid, as_tuple=False).squeeze(-1)
        sums = moment_sum[start:end].index_select(0, valid_indices)
        counts = visible_count[start:end].index_select(0, valid_indices).to(torch.float64)
        mean_moment = sums / counts[:, None, None]
        mean_moment = 0.5 * (mean_moment + mean_moment.transpose(1, 2))
        hessian = identity.unsqueeze(0) - mean_moment
        hessian = 0.5 * (hessian + hessian.transpose(1, 2))
        local_eigenvalues, local_eigenvectors = torch.linalg.eigh(hessian)
        local_weakest = _canonicalize_vector_sign(local_eigenvectors[:, :, 0])
        trace_h = hessian.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        local_observability = torch.clamp(
            3.0 * local_eigenvalues[:, 0] / (trace_h + eps), min=0.0, max=1.0
        )
        global_indices = valid_indices + start
        observability[global_indices] = local_observability
        weakest_direction[global_indices] = local_weakest
        eigenvalues[global_indices] = local_eigenvalues
        adjacent_gaps = local_eigenvalues[:, 1:] - local_eigenvalues[:, :-1]
        weak_eigengap[global_indices] = adjacent_gaps[:, 0]
        min_eigengap[global_indices] = adjacent_gaps.min(dim=-1).values
        trace_m = mean_moment.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        trace_m_residual[global_indices] = (trace_m - 1.0).abs()
        trace_h_residual[global_indices] = (trace_h - 2.0).abs()
        finite = torch.isfinite(local_eigenvalues).all(dim=-1)
        eigen_finite[global_indices] = finite
        eigen_in_range[global_indices] = finite & (
            (local_eigenvalues >= -1e-10) & (local_eigenvalues <= 1.0 + 1e-10)
        ).all(dim=-1)

    return {
        "observability": observability,
        "weakest_direction": weakest_direction,
        "eigenvalues": eigenvalues,
        "weak_eigengap": weak_eigengap,
        "min_eigengap": min_eigengap,
        "trace_m_residual": trace_m_residual,
        "trace_h_residual": trace_h_residual,
        "eigen_finite": eigen_finite,
        "eigen_in_range": eigen_in_range,
        "valid_observation": valid_observation,
    }


def moments_from_bearings(bearings: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """Return the float64 sum of unit-bearing second moments."""

    _require_shape(bearings, (3,), "bearings")
    values = bearings.to(torch.float64)
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if bool((norms <= eps).any()) or not bool(torch.isfinite(norms).all()):
        raise ValueError("bearings must be finite and non-zero")
    values = values / norms
    return (values.unsqueeze(-1) * values.unsqueeze(-2)).sum(dim=0)


def build_ogs_cache(
    observability: torch.Tensor,
    weakest_direction: torch.Tensor,
    visible_count: torch.Tensor,
    activated_opacity: torch.Tensor,
    anchor_scales: torch.Tensor,
    anchor_rotations: torch.Tensor,
    metadata: Optional[Mapping[str, Any]] = None,
    min_visible_views: int = OGS_V1_MIN_VISIBLE_VIEWS,
    min_activated_opacity: float = OGS_V1_MIN_ACTIVATED_OPACITY,
    eps: float = OGS_V1_VARIANCE_EPS,
) -> Dict[str, Any]:
    """Build the detached, compact float32 cache used during OGS training."""

    gaussian_count = int(observability.shape[0])
    if observability.ndim != 1:
        raise ValueError("observability must have shape (N,)")
    _require_shape(weakest_direction, (3,), "weakest_direction")
    _require_shape(anchor_scales, (3,), "anchor_scales")
    _require_shape(anchor_rotations, (4,), "anchor_rotations")
    opacity = activated_opacity.reshape(-1)
    if visible_count.ndim != 1 or opacity.ndim != 1:
        raise ValueError("visible_count and activated_opacity must have shape (N,)")
    lengths = (
        weakest_direction.shape[0],
        visible_count.shape[0],
        opacity.shape[0],
        anchor_scales.shape[0],
        anchor_rotations.shape[0],
    )
    if any(length != gaussian_count for length in lengths):
        raise ValueError("all anchor tensors must have the same Gaussian count")
    cache_device = anchor_scales.device
    cache_dtype = anchor_scales.dtype
    directions = weakest_direction.to(device=cache_device, dtype=cache_dtype).detach()
    direction_norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if not bool(torch.isfinite(direction_norms).all()) or bool(
        (direction_norms <= eps).any()
    ):
        raise ValueError("weakest_direction must be finite and non-zero")
    directions = directions / direction_norms
    anchor_thickness = covariance_thickness(
        anchor_scales.detach(),
        anchor_rotations.detach(),
        directions,
        variance_eps=eps,
    )
    eligible_mask = build_ogs_eligibility_mask(
        visible_count.to(device=cache_device),
        opacity.to(device=cache_device),
        min_visible_views=min_visible_views,
        min_activated_opacity=min_activated_opacity,
    )
    cache: Dict[str, Any] = {
        "schema_version": OGS_CACHE_SCHEMA_VERSION,
        "gaussian_count": gaussian_count,
        "observability": observability.detach().to(device="cpu", dtype=torch.float32),
        "weakest_direction": directions.detach().to(device="cpu", dtype=torch.float32),
        "perpendicular_thickness": anchor_thickness["perpendicular"]
        .detach()
        .to(device="cpu", dtype=torch.float32),
        "eligible_mask": eligible_mask.detach().to(device="cpu", dtype=torch.bool),
        "visible_count": visible_count.detach().to(device="cpu", dtype=torch.int32),
        "metadata": dict(metadata or {}),
    }
    cache["cache_sha256"] = compute_ogs_cache_hash(cache)
    validate_ogs_cache(cache)
    return cache


def _update_hash_with_tensor(digest: "hashlib._Hash", key: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(key.encode("utf-8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))


def compute_ogs_cache_hash(cache: Mapping[str, Any]) -> str:
    """Hash the semantic cache content independently of ``torch.save`` bytes."""

    digest = hashlib.sha256()
    digest.update(str(cache.get("schema_version", "")).encode("utf-8"))
    digest.update(str(cache.get("gaussian_count", "")).encode("ascii"))
    for key in _CACHE_TENSOR_KEYS:
        if key not in cache or not isinstance(cache[key], torch.Tensor):
            raise ValueError("cache is missing tensor '{}'".format(key))
        _update_hash_with_tensor(digest, key, cache[key])
    metadata_json = json.dumps(
        cache.get("metadata", {}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest.update(metadata_json.encode("utf-8"))
    return digest.hexdigest()


def validate_ogs_cache(
    cache: Mapping[str, Any],
    expected_gaussian_count: Optional[int] = None,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Fail closed on a malformed, stale, or tampered OGS anchor cache."""

    if cache.get("schema_version") != OGS_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported OGS cache schema")
    gaussian_count = cache.get("gaussian_count")
    if not isinstance(gaussian_count, int) or gaussian_count <= 0:
        raise ValueError("cache gaussian_count must be a positive integer")
    if expected_gaussian_count is not None and gaussian_count != expected_gaussian_count:
        raise ValueError(
            "OGS cache Gaussian count mismatch: expected {}, got {}".format(
                expected_gaussian_count, gaussian_count
            )
        )
    for key in _CACHE_TENSOR_KEYS:
        if key not in cache or not isinstance(cache[key], torch.Tensor):
            raise ValueError("cache is missing tensor '{}'".format(key))

    observability = cache["observability"]
    directions = cache["weakest_direction"]
    perpendicular = cache["perpendicular_thickness"]
    eligible = cache["eligible_mask"]
    counts = cache["visible_count"]
    if observability.shape != (gaussian_count,):
        raise ValueError("observability shape does not match gaussian_count")
    if directions.shape != (gaussian_count, 3):
        raise ValueError("weakest_direction shape does not match gaussian_count")
    if perpendicular.shape != (gaussian_count,):
        raise ValueError("perpendicular_thickness shape does not match gaussian_count")
    if eligible.shape != (gaussian_count,) or eligible.dtype != torch.bool:
        raise ValueError("eligible_mask must be bool with shape (N,)")
    if counts.shape != (gaussian_count,) or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError("visible_count must be integer with shape (N,)")
    if observability.dtype != torch.float32 or directions.dtype != torch.float32:
        raise ValueError("observability and weakest_direction must be float32")
    if perpendicular.dtype != torch.float32:
        raise ValueError("perpendicular_thickness must be float32")
    if not bool(torch.isfinite(observability).all()) or bool(
        ((observability < 0) | (observability > 1)).any()
    ):
        raise ValueError("observability must be finite and within [0, 1]")
    if not bool(torch.isfinite(directions).all()):
        raise ValueError("weakest_direction contains NaN or Inf")
    direction_norms = torch.linalg.vector_norm(directions, dim=-1)
    if not bool(torch.allclose(direction_norms, torch.ones_like(direction_norms), atol=2e-5)):
        raise ValueError("weakest_direction must contain unit vectors")
    if not bool(torch.isfinite(perpendicular).all()) or bool((perpendicular <= 0).any()):
        raise ValueError("perpendicular_thickness must be finite and positive")
    if bool((counts < 0).any()):
        raise ValueError("visible_count cannot be negative")
    if bool((eligible & (counts < OGS_V1_MIN_VISIBLE_VIEWS)).any()):
        raise ValueError(
            "eligible_mask contains a Gaussian with fewer than "
            f"{OGS_V1_MIN_VISIBLE_VIEWS} views"
        )

    metadata = cache.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("cache metadata must be a mapping")
    if expected_metadata is not None:
        for key, expected_value in expected_metadata.items():
            if key not in metadata or metadata[key] != expected_value:
                raise ValueError(
                    "OGS cache metadata mismatch for '{}': expected {!r}, got {!r}".format(
                        key, expected_value, metadata.get(key)
                    )
                )
    stored_hash = cache.get("cache_sha256")
    if not isinstance(stored_hash, str) or stored_hash != compute_ogs_cache_hash(cache):
        raise ValueError("OGS cache SHA-256 mismatch")


def save_ogs_cache(path: Union[str, Path], cache: Mapping[str, Any]) -> str:
    """Validate and save a cache.  Returns its semantic SHA-256."""

    validate_ogs_cache(cache)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(cache), destination)
    return str(cache["cache_sha256"])


def load_ogs_cache(
    path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    expected_gaussian_count: Optional[int] = None,
    expected_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Load, validate, and detach the compact OGS cache."""

    try:
        raw_cache = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        raw_cache = torch.load(Path(path), map_location="cpu")
    if not isinstance(raw_cache, Mapping):
        raise ValueError("OGS cache file does not contain a mapping")
    cache = dict(raw_cache)
    validate_ogs_cache(cache, expected_gaussian_count, expected_metadata)
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    for key in _CACHE_TENSOR_KEYS:
        cache[key] = cache[key].detach().to(device=target_device)
        cache[key].requires_grad_(False)
    return cache


def ogs_v1_loss(
    scales: torch.Tensor,
    rotations: torch.Tensor,
    cache: Mapping[str, Any],
    rho: float = OGS_V1_RHO,
    eps: float = OGS_V1_LOSS_EPS,
) -> Dict[str, Any]:
    """Evaluate the fixed-anchor OGS-v1 loss and diagnostics.

    The denominator is the fixed eligible count, never the current active count.
    """

    if float(rho) != OGS_V1_RHO:
        raise ValueError(f"OGS-v1 pins rho to {OGS_V1_RHO}")
    if float(eps) != OGS_V1_LOSS_EPS:
        raise ValueError(f"OGS-v1 pins loss eps to {OGS_V1_LOSS_EPS}")
    gaussian_count = scales.shape[0]
    if int(cache.get("gaussian_count", -1)) != gaussian_count:
        raise ValueError("OGS cache and current Gaussian count differ")
    for key in _CACHE_TENSOR_KEYS:
        if key not in cache:
            raise ValueError("OGS cache is missing '{}'".format(key))

    eligible = cache["eligible_mask"].to(device=scales.device, dtype=torch.bool).detach()
    eligible_count = int(eligible.sum().item())
    if eligible_count == 0:
        raise RuntimeError("OGS-v1 has zero fixed eligible Gaussians")
    observability = cache["observability"].to(
        device=scales.device, dtype=scales.dtype
    ).detach()
    directions = cache["weakest_direction"].to(
        device=scales.device, dtype=scales.dtype
    ).detach()
    perpendicular_anchor = cache["perpendicular_thickness"].to(
        device=scales.device, dtype=scales.dtype
    ).detach()

    thickness = covariance_thickness(scales, rotations, directions)
    parallel = thickness["parallel"]
    penalty = torch.relu(
        torch.log(parallel + eps)
        - torch.log(rho * perpendicular_anchor + eps)
    ).square()
    risk = (1.0 - observability).square() * penalty
    selected_penalty = penalty[eligible]
    selected_risk = risk[eligible]
    loss = selected_risk.sum() / eligible_count
    active = eligible & (parallel > rho * perpendicular_anchor)
    q_current = parallel / (thickness["perpendicular"] + eps)
    return {
        "loss": loss,
        "weighted_loss": loss,
        "penalty_mean": selected_penalty.sum() / eligible_count,
        "active_count": int(active.sum().item()),
        "eligible_count": eligible_count,
        "q_current": q_current,
        "t_parallel_current": parallel,
        "t_perp_current": thickness["perpendicular"],
        "penalty": penalty,
        "risk": risk,
        "active_mask": active,
    }


def _normalized_ast_payload(value: Any) -> Any:
    """Return a Python-version-stable representation of an AST value."""

    if isinstance(value, ast.AST):
        ignored_fields = {"type_comment", "type_ignores", "type_params"}
        return {
            "node": value.__class__.__name__,
            "fields": {
                name: _normalized_ast_payload(field_value)
                for name, field_value in ast.iter_fields(value)
                if name not in ignored_fields
            },
        }
    if isinstance(value, list):
        return [_normalized_ast_payload(item) for item in value]
    return value


def _normalized_function_ast(function: Any) -> Any:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    return _normalized_ast_payload(tree)


def _ogs_v1_formula_implementation() -> Dict[str, Any]:
    """Return normalized AST for every implementation component in the pin."""

    functions = (
        quaternion_to_rotation,
        covariance_thickness,
        compute_observability_from_moments,
        build_ogs_eligibility_mask,
        ogs_v1_loss,
    )
    return {
        function.__name__: _normalized_function_ast(function)
        for function in functions
    }


def compute_ogs_v1_formula_hash() -> str:
    """Hash the semantic contract plus normalized formula implementation AST."""

    payload = {
        "contract": ogs_v1_formula_contract(),
        "implementation_ast": _ogs_v1_formula_implementation(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_ogs_v1_formula_hash(expected_sha256: Optional[str] = None) -> str:
    """Fail closed on semantic/implementation drift or an external hash mismatch."""

    actual = compute_ogs_v1_formula_hash()
    if actual != OGS_V1_FORMULA_SHA256:
        raise RuntimeError(
            "OGS-v1 formula implementation drifted from the repository pin: "
            f"pinned={OGS_V1_FORMULA_SHA256} actual={actual}"
        )
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if expected != actual:
            raise RuntimeError(
                "OGS-v1 formula SHA-256 mismatch: "
                f"expected={expected} actual={actual}"
            )
    return actual


def ogs_v1_recalibration_manifest() -> Dict[str, Any]:
    """Return an isolated copy of the approved train-only calibration record."""

    calibration = dict(OGS_V1_RECALIBRATION)
    derived = (
        calibration["target_lower_ratio"]
        / calibration["previous_unweighted_lr_scaled_ratio_median"]
    )
    if abs(float(derived) - float(calibration["derived_lambda"])) > 1e-15:
        raise RuntimeError("OGS-v1 recalibration derivation no longer matches its pin")
    return calibration


def _percentiles(values: torch.Tensor, points: Sequence[float]) -> Dict[str, float]:
    finite_values = values.detach().to(torch.float64).reshape(-1)
    finite_values = finite_values[torch.isfinite(finite_values)]
    if finite_values.numel() == 0:
        return {"p{:g}".format(point): float("nan") for point in points}
    quantiles = torch.tensor(
        [point / 100.0 for point in points],
        dtype=torch.float64,
        device=finite_values.device,
    )
    result = torch.quantile(finite_values, quantiles).cpu().tolist()
    return {
        "p{:g}".format(point): float(value) for point, value in zip(points, result)
    }


@torch.no_grad()
def summarize_scale_safety(
    anchor_scales: torch.Tensor,
    current_scales: torch.Tensor,
    current_rotations: torch.Tensor,
    cache: Mapping[str, Any],
    rho: float = 3.0,
    eps: float = 1e-8,
    extreme_axis_low: float = 0.1,
    extreme_axis_high: float = 10.0,
    extreme_q: float = 100.0,
) -> Dict[str, Any]:
    """Return the scale-safety statistics without imposing a training policy."""

    _require_shape(anchor_scales, (3,), "anchor_scales")
    _require_shape(current_scales, (3,), "current_scales")
    if anchor_scales.shape != current_scales.shape:
        raise ValueError("anchor_scales and current_scales must have the same shape")
    if bool((anchor_scales <= 0).any()) or bool((current_scales <= 0).any()):
        raise ValueError("activated scales must be positive")
    ratios = current_scales / anchor_scales
    trace_ratio = current_scales.square().sum(-1) / anchor_scales.square().sum(-1)
    log_volume_ratio = 2.0 * torch.log(ratios).sum(-1)
    volume_ratio = torch.exp(
        torch.clamp(log_volume_ratio, min=-700.0, max=700.0)
    )
    loss_diagnostics = ogs_v1_loss(
        current_scales, current_rotations, cache, rho=rho, eps=eps
    )
    eligible = cache["eligible_mask"].to(device=current_scales.device, dtype=torch.bool)
    percentile_points = (1, 5, 50, 95, 99, 100)
    axis_statistics = {
        "axis_{}".format(axis): _percentiles(ratios[:, axis], percentile_points)
        for axis in range(3)
    }
    extreme_mask = (
        (ratios < extreme_axis_low).any(dim=-1)
        | (ratios > extreme_axis_high).any(dim=-1)
        | (loss_diagnostics["q_current"] > extreme_q)
    )
    finite = (
        torch.isfinite(current_scales).all()
        & torch.isfinite(current_rotations).all()
        & torch.isfinite(trace_ratio).all()
        & torch.isfinite(volume_ratio).all()
        & torch.isfinite(loss_diagnostics["q_current"]).all()
    )
    return {
        "axis_ratio_percentiles": axis_statistics,
        "covariance_trace_ratio_percentiles": _percentiles(
            trace_ratio, percentile_points
        ),
        "covariance_volume_ratio_percentiles": _percentiles(
            volume_ratio, percentile_points
        ),
        "fraction_any_axis_gt_2": float((ratios > 2.0).any(dim=-1).float().mean()),
        "fraction_any_axis_lt_0_5": float((ratios < 0.5).any(dim=-1).float().mean()),
        "q_current_percentiles_eligible": _percentiles(
            loss_diagnostics["q_current"][eligible], (50, 95, 99, 100)
        ),
        "active_count": loss_diagnostics["active_count"],
        "eligible_count": loss_diagnostics["eligible_count"],
        "finite": bool(finite),
        "extreme_ellipsoid_count": int(extreme_mask.sum()),
        "extreme_definition": {
            "axis_ratio_below": extreme_axis_low,
            "axis_ratio_above": extreme_axis_high,
            "q_above": extreme_q,
        },
    }


def _robust_scale_features(
    activated_scales: np.ndarray,
    reference_mask: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(~np.isfinite(activated_scales)) or np.any(activated_scales <= 0):
        raise ValueError("activated scales must be finite and positive")
    sorted_log_scales = np.sort(np.log(activated_scales), axis=1)
    reference = sorted_log_scales[reference_mask]
    if reference.shape[0] == 0:
        raise ValueError("robust scale standardization has an empty reference set")
    center = np.median(reference, axis=0)
    mad = np.median(np.abs(reference - center), axis=0)
    robust_scale = 1.4826 * mad
    robust_scale = np.where(robust_scale > eps, robust_scale, 1.0)
    standardized = (sorted_log_scales - center) / robust_scale
    return standardized, center, robust_scale


def match_scale_controls(
    activated_scales: Union[torch.Tensor, np.ndarray],
    eligible_mask: Union[torch.Tensor, np.ndarray],
    clamp_indices: Iterable[int],
    controls_per_target: int = 5,
    standardization_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Greedily match deterministic scale controls without outcome leakage.

    Matching uses only the three sorted activated log-scales, standardized by
    robust MAD.  Candidate controls are fixed-eligible and all clamp indices are
    excluded.  Global no-replacement matching is preferred; reuse is recorded
    only when the remaining unused pool is insufficient.
    """

    if controls_per_target <= 0:
        raise ValueError("controls_per_target must be positive")
    if isinstance(activated_scales, torch.Tensor):
        scale_array = activated_scales.detach().cpu().numpy()
    else:
        scale_array = np.asarray(activated_scales)
    scale_array = np.asarray(scale_array, dtype=np.float64)
    if scale_array.ndim != 2 or scale_array.shape[1] != 3:
        raise ValueError("activated_scales must have shape (N, 3)")
    gaussian_count = scale_array.shape[0]
    if isinstance(eligible_mask, torch.Tensor):
        eligible_array = eligible_mask.detach().cpu().numpy()
    else:
        eligible_array = np.asarray(eligible_mask)
    eligible_array = np.asarray(eligible_array, dtype=bool).reshape(-1)
    if eligible_array.shape != (gaussian_count,):
        raise ValueError("eligible_mask must have shape (N,)")

    targets = sorted({int(index) for index in clamp_indices})
    if not targets:
        raise ValueError("clamp_indices cannot be empty")
    if targets[0] < 0 or targets[-1] >= gaussian_count:
        raise IndexError("clamp_indices contain an out-of-range Gaussian index")
    target_set = set(targets)
    candidate_indices = np.asarray(
        [
            index
            for index in range(gaussian_count)
            if eligible_array[index] and index not in target_set
        ],
        dtype=np.int64,
    )
    if candidate_indices.size < controls_per_target:
        raise ValueError(
            "fewer than {} distinct eligible non-clamp controls".format(
                controls_per_target
            )
        )

    if standardization_mask is None:
        reference_mask = np.ones(gaussian_count, dtype=bool)
    else:
        if isinstance(standardization_mask, torch.Tensor):
            reference_mask = standardization_mask.detach().cpu().numpy()
        else:
            reference_mask = np.asarray(standardization_mask)
        reference_mask = np.asarray(reference_mask, dtype=bool).reshape(-1)
        if reference_mask.shape != (gaussian_count,):
            raise ValueError("standardization_mask must have shape (N,)")
    standardized, center, robust_scale = _robust_scale_features(
        scale_array, reference_mask, eps
    )

    used = set()
    records = []
    reuse_count = 0
    for target in targets:
        squared_distance = np.square(
            standardized[candidate_indices] - standardized[target]
        ).sum(axis=1)
        order = np.lexsort((candidate_indices, squared_distance))
        ordered_indices = candidate_indices[order]
        ordered_distances = squared_distance[order]
        distance_by_index = {
            int(index): float(distance)
            for index, distance in zip(ordered_indices, ordered_distances)
        }
        available = [int(index) for index in ordered_indices if int(index) not in used]
        selected = available[:controls_per_target]
        selected_reuse = [False] * len(selected)
        if len(selected) < controls_per_target:
            selected_set = set(selected)
            for index in ordered_indices:
                index = int(index)
                if index in selected_set:
                    continue
                selected.append(index)
                selected_reuse.append(index in used)
                selected_set.add(index)
                if len(selected) == controls_per_target:
                    break
        reuse_count += sum(selected_reuse)
        used.update(selected)
        records.append(
            {
                "target_index": target,
                "control_indices": selected,
                "squared_distances": [distance_by_index[index] for index in selected],
                "reused": selected_reuse,
            }
        )
    return {
        "records": records,
        "controls_per_target": controls_per_target,
        "candidate_count": int(candidate_indices.size),
        "unique_control_count": len(used),
        "reuse_count": int(reuse_count),
        "reuse_required": bool(reuse_count),
        "standardization": {
            "feature": "three_sorted_activated_log_scales",
            "center": center.tolist(),
            "robust_scale_1_4826_mad": robust_scale.tolist(),
            "reference_count": int(reference_mask.sum()),
        },
    }
