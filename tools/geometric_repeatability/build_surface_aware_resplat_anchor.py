#!/usr/bin/env python3
"""Build one deterministic surface-aware resplat checkpoint anchor.

The tool is deliberately checkpoint-only: it never writes a PLY and never
optimizes a model.  Candidate selection is consumed from the locked v2
train/guard receipt, while the local support distance is consumed from the
locked SCSP manifest.  Test data cannot enter this sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.build_shared_clamped_anchor import MODEL_FIELDS
from utils.read_write_model import read_points3D_binary, read_points3D_text
from utils.sparse_support import VoxelHashNN


PROTOCOL = "uav-tgs-surface-aware-resplat-anchor-v1"
CANDIDATE_PROTOCOL = "uav-tgs-conditional-resplat-decision-v2"
NEIGHBOR_COUNT = 16
MAX_COUNT_GROWTH_FRACTION = 0.05
INT32_MAX = int(np.iinfo(np.int32).max)
EPS64 = float(np.finfo(np.float64).eps)


class SurfaceAwareResplatError(RuntimeError):
    """Fail-closed resplat-anchor construction error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise SurfaceAwareResplatError(f"required input is missing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SurfaceAwareResplatError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SurfaceAwareResplatError(f"JSON root must be an object: {path}")
    return payload


def _require_declared_sha(
    actual: str, expected: str | None, label: str
) -> None:
    if expected and actual.lower() != str(expected).strip().lower():
        raise SurfaceAwareResplatError(
            f"{label} SHA-256 mismatch: expected={str(expected).lower()} actual={actual}"
        )


def _as_finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise SurfaceAwareResplatError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SurfaceAwareResplatError(f"{label} is not finite")
    return result


def canonicalize_direction(vector: np.ndarray) -> np.ndarray:
    """Normalize an eigenvector and choose a deterministic sign."""

    value = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(value)):
        raise SurfaceAwareResplatError("direction contains NaN/Inf")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SurfaceAwareResplatError("direction has zero norm")
    value = value / norm
    pivot = int(np.argmax(np.abs(value)))
    if value[pivot] < 0.0:
        value = -value
    return value


def pca_surface_frame(neighbors: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a deterministic right-handed [t1,t2,n] PCA frame in float64."""

    points = np.asarray(neighbors, dtype=np.float64)
    if points.ndim != 2 or points.shape != (NEIGHBOR_COUNT, 3):
        raise SurfaceAwareResplatError(
            f"PCA requires exactly {NEIGHBOR_COUNT} distinct 3D centroids"
        )
    if not np.all(np.isfinite(points)):
        raise SurfaceAwareResplatError("PCA neighbors contain NaN/Inf")
    if np.unique(points, axis=0).shape[0] != NEIGHBOR_COUNT:
        raise SurfaceAwareResplatError("PCA neighbors are not distinct centroids")
    centered = points - np.mean(points, axis=0, dtype=np.float64)
    covariance = centered.T @ centered / float(points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise SurfaceAwareResplatError("PCA eigendecomposition contains NaN/Inf")
    scale = max(float(eigenvalues[-1]), 1.0)
    tolerance = scale * max(points.shape) * EPS64
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    if rank < 2:
        raise SurfaceAwareResplatError(
            f"PCA neighborhood rank must be at least 2, observed {rank}"
        )

    normal = canonicalize_direction(eigenvectors[:, 0])
    tangent_1 = canonicalize_direction(eigenvectors[:, 2])
    # Deriving t2 makes right-handedness exact and removes the arbitrary sign
    # of the middle eigenvector.
    tangent_2 = np.cross(normal, tangent_1)
    tangent_2 /= np.linalg.norm(tangent_2)
    frame = np.column_stack((tangent_1, tangent_2, normal))
    if not np.all(np.isfinite(frame)) or not np.allclose(
        frame.T @ frame, np.eye(3), rtol=0.0, atol=5e-12
    ):
        raise SurfaceAwareResplatError("PCA frame is not finite and orthonormal")
    if float(np.linalg.det(frame)) <= 0.0:
        raise SurfaceAwareResplatError("PCA frame is not right-handed")
    return frame, eigenvalues.astype(np.float64, copy=False), rank


def quaternion_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Convert the repository's scalar-first quaternion to a rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(q)):
        raise SurfaceAwareResplatError("quaternion contains NaN/Inf")
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not math.isfinite(norm):
        raise SurfaceAwareResplatError("quaternion has zero norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a deterministic scalar-first quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise SurfaceAwareResplatError("rotation contains NaN/Inf")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=5e-10):
        raise SurfaceAwareResplatError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, rel_tol=0.0, abs_tol=5e-10):
        raise SurfaceAwareResplatError("rotation is not proper")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * root,
                (matrix[2, 1] - matrix[1, 2]) / root,
                (matrix[0, 2] - matrix[2, 0]) / root,
                (matrix[1, 0] - matrix[0, 1]) / root,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            root = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            q = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / root,
                    0.25 * root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                ]
            )
        elif axis == 1:
            root = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            q = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    0.25 * root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                ]
            )
        else:
            root = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            q = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                    0.25 * root,
                ]
            )
    q /= np.linalg.norm(q)
    # q and -q are the same rotation.  Prefer positive w; at w==0 use the
    # first largest-magnitude vector component as the deterministic pivot.
    if q[0] < 0.0:
        q = -q
    elif q[0] == 0.0:
        pivot = 1 + int(np.argmax(np.abs(q[1:])))
        if q[pivot] < 0.0:
            q = -q
    return q


def moment_matched_axis(parent_std: float, child_count: int) -> tuple[float, float]:
    """Return child sigma and centered-lattice spacing preserving variance."""

    std = _as_finite_float(parent_std, "parent tangent standard deviation")
    count = int(child_count)
    if std <= 0.0 or count <= 0:
        raise SurfaceAwareResplatError("moment matching requires std>0 and count>0")
    if count == 1:
        return std, 0.0
    child_std = std / float(count)
    spacing = math.sqrt(12.0) * std / float(count)
    lattice_variance = spacing * spacing * (count * count - 1.0) / 12.0
    if not math.isclose(
        child_std * child_std + lattice_variance,
        std * std,
        rel_tol=2e-14,
        abs_tol=2e-14,
    ):
        raise SurfaceAwareResplatError("tangent moment matching failed")
    return child_std, spacing


def coincident_child_logit(parent_logit: float, child_count: int) -> float:
    """Preserve parent transmittance for coincident equal-alpha children."""

    raw = _as_finite_float(parent_logit, "parent opacity logit")
    count = int(child_count)
    if count <= 0:
        raise SurfaceAwareResplatError("child count must be positive")
    # log(1-sigmoid(raw)) = -softplus(raw), evaluated without overflow.
    log_parent_transmittance = -float(np.logaddexp(0.0, raw))
    log_child_transmittance = log_parent_transmittance / float(count)
    child_alpha = -math.expm1(log_child_transmittance)
    if not (0.0 < child_alpha < 1.0):
        raise SurfaceAwareResplatError("child opacity is outside (0,1)")
    child_raw = math.log(child_alpha) - log_child_transmittance
    if not math.isfinite(child_raw):
        raise SurfaceAwareResplatError("child opacity logit is not finite")
    return child_raw


def build_voxel_centroids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Build the exact established SCSP voxel-centroid table on CPU."""

    xyz = np.asarray(points, dtype=np.float32)
    size = _as_finite_float(voxel_size, "voxel size")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise SurfaceAwareResplatError("sparse points must have shape (N,3) with N>0")
    if not np.all(np.isfinite(xyz)) or size <= 0.0:
        raise SurfaceAwareResplatError("sparse points/voxel size are invalid")
    # SCSP itself uses VoxelHashNN float32 metric centroids.  Reuse that exact
    # implementation, on CPU, then promote only the PCA/query arithmetic to
    # float64 as required by this protocol.
    index = VoxelHashNN(torch.as_tensor(xyz, dtype=torch.float32), voxel_size=size)
    centroids = (
        index._centroids_sorted.detach().cpu().numpy().astype(np.float64, copy=True)
    )
    if centroids.shape[0] < NEIGHBOR_COUNT or not np.all(np.isfinite(centroids)):
        raise SurfaceAwareResplatError(
            f"fewer than {NEIGHBOR_COUNT} finite voxel centroids are available"
        )
    return centroids


def nearest_distinct_centroids(
    center: Sequence[float], centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select 16 global nearest centroids with deterministic tie breaking."""

    query = np.asarray(center, dtype=np.float64).reshape(3)
    support = np.asarray(centroids, dtype=np.float64)
    if support.ndim != 2 or support.shape[1] != 3 or support.shape[0] < NEIGHBOR_COUNT:
        raise SurfaceAwareResplatError("voxel-centroid table has invalid shape")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(support)):
        raise SurfaceAwareResplatError("center/voxel centroids contain NaN/Inf")
    squared = np.sum((support - query[None, :]) ** 2, axis=1, dtype=np.float64)
    indices = np.arange(support.shape[0], dtype=np.int64)
    order = np.lexsort((indices, support[:, 2], support[:, 1], support[:, 0], squared))
    selected = order[:NEIGHBOR_COUNT]
    neighbors = support[selected]
    if np.unique(neighbors, axis=0).shape[0] != NEIGHBOR_COUNT:
        raise SurfaceAwareResplatError("nearest voxel-centroid set is not distinct")
    return neighbors, selected.astype(np.int64, copy=False)


def _parent_plan(
    parent_index: int,
    center: np.ndarray,
    raw_scaling: np.ndarray,
    raw_rotation: np.ndarray,
    parent_opacity_logit: float,
    support_distance: float,
    centroids: np.ndarray,
) -> dict[str, Any]:
    d = _as_finite_float(support_distance, "SCSP local support distance")
    if d <= 0.0:
        raise SurfaceAwareResplatError("SCSP local support distance must be >0")
    neighbors, neighbor_indices = nearest_distinct_centroids(center, centroids)
    frame, eigenvalues, rank = pca_surface_frame(neighbors)

    activated = np.exp(np.asarray(raw_scaling, dtype=np.float64).reshape(3))
    if not np.all(np.isfinite(activated)) or np.any(activated <= 0.0):
        raise SurfaceAwareResplatError("parent activated scale is invalid")
    rotation = quaternion_to_rotation(raw_rotation)
    covariance = rotation @ np.diag(activated * activated) @ rotation.T
    projected_variances = np.diag(frame.T @ covariance @ frame)
    if not np.all(np.isfinite(projected_variances)) or np.any(projected_variances <= 0.0):
        raise SurfaceAwareResplatError("projected parent covariance is invalid")
    parent_std = np.sqrt(projected_variances)
    count_1 = max(1, int(math.ceil(float(parent_std[0]) / d)))
    count_2 = max(1, int(math.ceil(float(parent_std[1]) / d)))
    child_count = int(count_1 * count_2)
    child_std_1, spacing_1 = moment_matched_axis(float(parent_std[0]), count_1)
    child_std_2, spacing_2 = moment_matched_axis(float(parent_std[1]), count_2)
    child_std_n = min(float(parent_std[2]), d)
    if child_std_n <= 0.0 or not math.isfinite(child_std_n):
        raise SurfaceAwareResplatError("child normal standard deviation is invalid")

    offsets_1 = (np.arange(count_1, dtype=np.float64) - (count_1 - 1.0) / 2.0) * spacing_1
    offsets_2 = (np.arange(count_2, dtype=np.float64) - (count_2 - 1.0) / 2.0) * spacing_2
    child_centers = np.asarray(
        [
            np.asarray(center, dtype=np.float64)
            + offset_1 * frame[:, 0]
            + offset_2 * frame[:, 1]
            for offset_1 in offsets_1
            for offset_2 in offsets_2
        ],
        dtype=np.float64,
    )
    # Lattice is centered at the original Gaussian, tangent-only.  In
    # particular, no point is projected onto the fitted plane.
    if not np.allclose(np.mean(child_centers, axis=0), center, rtol=0.0, atol=2e-12):
        raise SurfaceAwareResplatError("child lattice is not centered at the parent")
    normal_offsets = (child_centers - np.asarray(center, dtype=np.float64)) @ frame[:, 2]
    if not np.allclose(normal_offsets, 0.0, rtol=0.0, atol=2e-12):
        raise SurfaceAwareResplatError("child lattice contains a normal-plane projection")

    child_scale = np.asarray([child_std_1, child_std_2, child_std_n], dtype=np.float64)
    child_log_scale = np.log(child_scale)
    child_quaternion = rotation_to_quaternion(frame)
    child_opacity_logit = coincident_child_logit(parent_opacity_logit, child_count)
    return {
        "parent_index": int(parent_index),
        "support_distance_m": d,
        "neighbor_centroid_indices": neighbor_indices,
        "neighbors": neighbors,
        "pca_eigenvalues": eigenvalues,
        "pca_rank": rank,
        "frame": frame,
        "parent_projected_std": parent_std,
        "lattice_shape": (count_1, count_2),
        "spacing": (spacing_1, spacing_2),
        "child_count": child_count,
        "child_centers": child_centers,
        "child_log_scale": child_log_scale,
        "child_quaternion": child_quaternion,
        "child_opacity_logit": child_opacity_logit,
    }


def validate_checkpoint_schema(params: Sequence[Any]) -> int:
    """Validate the raw Gaussian capture schema and all per-row tensor fields."""

    if not isinstance(params, (tuple, list)) or len(params) != 12:
        raise SurfaceAwareResplatError("checkpoint does not match GaussianModel.capture()")
    xyz = params[MODEL_FIELDS["xyz"]]
    if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SurfaceAwareResplatError("checkpoint xyz has invalid schema")
    count = int(xyz.shape[0])
    shapes = {
        "features_dc": (count, 1, 3),
        "scaling": (count, 3),
        "rotation": (count, 4),
        "opacity": (count, 1),
        "max_radii2D": (count,),
        "xyz_gradient_accum": (count, 1),
        "denom": (count, 1),
    }
    for name, expected in shapes.items():
        value = params[MODEL_FIELDS[name]]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise SurfaceAwareResplatError(
                f"checkpoint {name} shape mismatch: expected={expected} "
                f"actual={getattr(value, 'shape', None)}"
            )
    rest = params[MODEL_FIELDS["features_rest"]]
    if (
        not isinstance(rest, torch.Tensor)
        or rest.ndim != 3
        or rest.shape[0] != count
        or rest.shape[2] != 3
    ):
        raise SurfaceAwareResplatError("checkpoint features_rest has invalid schema")
    for name, index in MODEL_FIELDS.items():
        value = params[index]
        if not bool(torch.isfinite(value.detach()).all()):
            raise SurfaceAwareResplatError(f"checkpoint {name} contains NaN/Inf")
    return count


def _load_checkpoint_cpu(path: Path, iteration: int) -> tuple[list[Any], int]:
    """Load a trusted project checkpoint without requiring a visible GPU."""

    try:
        payload = torch.load(
            str(path), map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) != 2:
        raise SurfaceAwareResplatError(
            "checkpoint must be a (model_params, iteration) pair"
        )
    model_params, actual_iteration = payload
    if int(actual_iteration) != int(iteration):
        raise SurfaceAwareResplatError(
            f"checkpoint iteration mismatch: expected={iteration} actual={actual_iteration}"
        )
    params = list(model_params) if isinstance(model_params, (tuple, list)) else model_params
    count = validate_checkpoint_schema(params)
    return list(params), count


def build_resplat_parameters(
    params: Sequence[Any],
    candidate_indices: Sequence[int],
    support_distances: Mapping[int, float],
    centroids: np.ndarray,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Pure checkpoint transformation used by both the CLI and unit tests."""

    source = list(params)
    gaussian_count = validate_checkpoint_schema(source)
    candidates = [int(value) for value in candidate_indices]
    if candidates != sorted(candidates) or len(candidates) != len(set(candidates)):
        raise SurfaceAwareResplatError("candidate indices must be unique and sorted")
    if not candidates:
        raise SurfaceAwareResplatError("candidate receipt contains no candidates")
    if any(index < 0 or index >= gaussian_count for index in candidates):
        raise SurfaceAwareResplatError("candidate index is outside the checkpoint")
    if set(candidates) != set(int(key) for key in support_distances):
        raise SurfaceAwareResplatError("candidate/support-distance index sets differ")

    arrays = {
        name: source[index].detach().clone()
        for name, index in MODEL_FIELDS.items()
    }
    plans: list[dict[str, Any]] = []
    for index in candidates:
        plans.append(
            _parent_plan(
                index,
                arrays["xyz"][index].cpu().numpy(),
                arrays["scaling"][index].cpu().numpy(),
                arrays["rotation"][index].cpu().numpy(),
                float(arrays["opacity"][index, 0].item()),
                support_distances[index],
                centroids,
            )
        )

    final_count = gaussian_count + sum(plan["child_count"] - 1 for plan in plans)
    growth = (final_count - gaussian_count) / float(gaussian_count)
    if final_count > INT32_MAX:
        raise SurfaceAwareResplatError("resplat output exceeds signed int32 index space")
    if growth > MAX_COUNT_GROWTH_FRACTION:
        raise SurfaceAwareResplatError(
            f"resplat count growth exceeds 5%: {growth:.9f}"
        )

    appended: dict[str, list[torch.Tensor]] = {name: [] for name in MODEL_FIELDS}
    append_cursor = gaussian_count
    for plan in plans:
        parent = int(plan["parent_index"])
        child_count = int(plan["child_count"])
        output_indices = [parent] + list(range(append_cursor, append_cursor + child_count - 1))
        append_cursor += child_count - 1
        plan["output_indices"] = output_indices

        centers = torch.as_tensor(
            plan["child_centers"], dtype=arrays["xyz"].dtype, device=arrays["xyz"].device
        )
        scaling = torch.as_tensor(
            np.repeat(plan["child_log_scale"][None, :], child_count, axis=0),
            dtype=arrays["scaling"].dtype,
            device=arrays["scaling"].device,
        )
        rotation = torch.as_tensor(
            np.repeat(plan["child_quaternion"][None, :], child_count, axis=0),
            dtype=arrays["rotation"].dtype,
            device=arrays["rotation"].device,
        )
        opacity = torch.full(
            (child_count, 1),
            float(plan["child_opacity_logit"]),
            dtype=arrays["opacity"].dtype,
            device=arrays["opacity"].device,
        )
        children = {
            "xyz": centers,
            "features_dc": arrays["features_dc"][parent : parent + 1].repeat(child_count, 1, 1),
            "features_rest": arrays["features_rest"][parent : parent + 1].repeat(child_count, 1, 1),
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "max_radii2D": torch.zeros(
                child_count,
                dtype=arrays["max_radii2D"].dtype,
                device=arrays["max_radii2D"].device,
            ),
            "xyz_gradient_accum": torch.zeros(
                (child_count, 1),
                dtype=arrays["xyz_gradient_accum"].dtype,
                device=arrays["xyz_gradient_accum"].device,
            ),
            "denom": torch.zeros(
                (child_count, 1),
                dtype=arrays["denom"].dtype,
                device=arrays["denom"].device,
            ),
        }
        for name in MODEL_FIELDS:
            arrays[name][parent] = children[name][0]
            if child_count > 1:
                appended[name].append(children[name][1:])

    if append_cursor != final_count:
        raise SurfaceAwareResplatError("internal child-index accounting mismatch")
    for name in MODEL_FIELDS:
        if appended[name]:
            arrays[name] = torch.cat([arrays[name], *appended[name]], dim=0)
        if int(arrays[name].shape[0]) != final_count:
            raise SurfaceAwareResplatError(f"output {name} count mismatch")
        if not bool(torch.isfinite(arrays[name]).all()):
            raise SurfaceAwareResplatError(f"output {name} contains NaN/Inf")

    candidate_set = set(candidates)
    noncandidate = [i for i in range(gaussian_count) if i not in candidate_set]
    index_tensor = torch.as_tensor(noncandidate, dtype=torch.long)
    for name in MODEL_FIELDS:
        before = source[MODEL_FIELDS[name]].detach().cpu().index_select(0, index_tensor)
        after = arrays[name].detach().cpu().index_select(0, index_tensor)
        if not torch.equal(before, after):
            raise SurfaceAwareResplatError(
                f"noncandidate rows changed or moved in field {name}"
            )

    output = list(source)
    for name, index in MODEL_FIELDS.items():
        tensor = arrays[name]
        if isinstance(source[index], nn.Parameter):
            output[index] = nn.Parameter(tensor, requires_grad=True)
        else:
            output[index] = tensor
    output[10] = {
        "state": {},
        "param_groups": [],
        "fresh_optimizer_required": True,
        "reason": "surface_aware_resplat_changed_topology",
    }
    validate_checkpoint_schema(output)
    return output, plans


def _load_sparse_points(path: Path) -> np.ndarray:
    if path.name == "points3D.bin":
        records = read_points3D_binary(str(path))
    elif path.name == "points3D.txt":
        records = read_points3D_text(str(path))
    else:
        raise SurfaceAwareResplatError("sparse points must be points3D.bin or points3D.txt")
    if not records:
        raise SurfaceAwareResplatError("locked sparse point file is empty")
    # COLMAP IDs are sorted explicitly; dictionary order is not an input to PCA.
    xyz = np.stack([records[key].xyz for key in sorted(records)], axis=0).astype(np.float32)
    if not np.all(np.isfinite(xyz)):
        raise SurfaceAwareResplatError("locked sparse points contain NaN/Inf")
    return xyz


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise SurfaceAwareResplatError(f"missing receipt field: {'.'.join(keys)}")
        value = value[key]
    return value


def _load_contract(
    candidate_receipt: Mapping[str, Any],
    scsp_manifest: Mapping[str, Any],
    *,
    scsp_identity: Mapping[str, Any],
    sparse_identity: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    gaussian_count: int,
) -> tuple[list[int], dict[int, float], float]:
    if candidate_receipt.get("protocol") != CANDIDATE_PROTOCOL or candidate_receipt.get("status") != "passed":
        raise SurfaceAwareResplatError("candidate receipt is not a passed v2 receipt")
    decision = _nested(candidate_receipt, "decision")
    if (
        decision.get("triggered") is not True
        or decision.get("decision") != "execute_one_deterministic_resplat"
    ):
        raise SurfaceAwareResplatError("candidate receipt does not authorize resplatting")
    selection = _nested(candidate_receipt, "selection_policy")
    if (
        selection.get("formal_test_metrics_used") is not False
        or int(selection.get("formal_test_npz_open_count_before_receipt", -1)) != 0
    ):
        raise SurfaceAwareResplatError("candidate receipt is not train/guard-only")
    rule = _nested(decision, "candidate_rule")
    raw_candidates = rule.get("candidate_indices")
    if not isinstance(raw_candidates, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_candidates
    ):
        raise SurfaceAwareResplatError("candidate_rule.candidate_indices is invalid")
    candidates = [int(value) for value in raw_candidates]
    if candidates != sorted(candidates) or len(candidates) != len(set(candidates)) or not candidates:
        raise SurfaceAwareResplatError("candidate indices must be nonempty, unique, and sorted")
    if rule.get("candidate_count") != len(candidates):
        raise SurfaceAwareResplatError("candidate count disagrees with candidate list")
    if str(rule.get("candidate_indices_sha256", "")).lower() != _canonical_json_sha256(candidates):
        raise SurfaceAwareResplatError("candidate index-list hash mismatch")
    alias = decision.get("eligible_candidate_indices")
    if alias != candidates:
        raise SurfaceAwareResplatError("candidate alias disagrees with authoritative v2 rule")
    receipt_count = int(_nested(candidate_receipt, "inputs", "gaussian_count"))
    if receipt_count != gaussian_count or any(i < 0 or i >= gaussian_count for i in candidates):
        raise SurfaceAwareResplatError("candidate receipt/checkpoint Gaussian index spaces differ")

    declared_scsp_sha = str(_nested(candidate_receipt, "inputs", "scsp_manifest", "sha256")).lower()
    if declared_scsp_sha != scsp_identity["sha256"]:
        raise SurfaceAwareResplatError("candidate receipt/SCSP manifest identity mismatch")
    locked_sha = str(
        _nested(candidate_receipt, "decision_contract", "constants", "locked_scsp_manifest_sha256")
    ).lower()
    if locked_sha != scsp_identity["sha256"]:
        raise SurfaceAwareResplatError("candidate receipt does not bind this locked SCSP manifest")
    if (
        scsp_manifest.get("status") != "passed"
        or scsp_manifest.get("method") != "scsp"
        or _nested(scsp_manifest, "invariants", "no_training") is not True
    ):
        raise SurfaceAwareResplatError("SCSP manifest is not a passed no-training anchor")
    if str(_nested(scsp_manifest, "input", "checkpoint_sha256")).lower() != checkpoint_identity["sha256"]:
        raise SurfaceAwareResplatError("raw checkpoint is not the SCSP input anchor")
    anchor_sha = str(_nested(candidate_receipt, "inputs", "gaussian_index_anchor_sha256")).lower()
    if anchor_sha != str(_nested(scsp_manifest, "input", "ply_sha256")).lower():
        raise SurfaceAwareResplatError("candidate index anchor is not the SCSP raw PLY anchor")

    scsp_support = _nested(scsp_manifest, "sparse_support")
    candidate_support = _nested(candidate_receipt, "inputs", "sparse_support")
    for support, label in ((scsp_support, "SCSP"), (candidate_support, "candidate")):
        if str(support.get("points3d_sha256", "")).lower() != sparse_identity["sha256"]:
            raise SurfaceAwareResplatError(f"{label} receipt/sparse-point identity mismatch")
    voxel_size = _as_finite_float(scsp_support.get("voxel_size"), "SCSP voxel size")
    if voxel_size <= 0.0 or not math.isclose(
        voxel_size,
        _as_finite_float(candidate_support.get("voxel_size"), "candidate voxel size"),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise SurfaceAwareResplatError("candidate and SCSP voxel sizes differ")

    modified_rows = scsp_manifest.get("modified_rows")
    if not isinstance(modified_rows, list):
        raise SurfaceAwareResplatError("SCSP manifest lacks modified_rows")
    row_map: dict[int, Mapping[str, Any]] = {}
    for row in modified_rows:
        if not isinstance(row, Mapping) or isinstance(row.get("gaussian_index"), bool):
            raise SurfaceAwareResplatError("SCSP modified_rows has invalid schema")
        index = int(row["gaussian_index"])
        if index in row_map:
            raise SurfaceAwareResplatError("SCSP modified_rows contains duplicate indices")
        row_map[index] = row
    diagnostics = rule.get("candidate_diagnostics")
    if not isinstance(diagnostics, list):
        raise SurfaceAwareResplatError("candidate receipt lacks candidate_diagnostics")
    diagnostic_map = {
        int(row["gaussian_index"]): row
        for row in diagnostics
        if isinstance(row, Mapping) and isinstance(row.get("gaussian_index"), int)
    }
    if set(diagnostic_map) != set(candidates):
        raise SurfaceAwareResplatError("candidate diagnostics/index list mismatch")
    distances: dict[int, float] = {}
    for index in candidates:
        if index not in row_map:
            raise SurfaceAwareResplatError(f"candidate {index} is not an SCSP modified row")
        distance = _as_finite_float(
            row_map[index].get("local_support_distance"),
            f"SCSP local support distance for {index}",
        )
        if distance <= 0.0:
            raise SurfaceAwareResplatError(f"SCSP support distance for {index} is not positive")
        candidate_distance = _as_finite_float(
            diagnostic_map[index].get("local_support_distance_m"),
            f"candidate local support distance for {index}",
        )
        # The locked SCSP receipt stores the authoritative float32 distance.
        # Candidate-v2 recomputes the same diagnostic through a separate
        # centroid reduction whose float32 accumulation can differ by a few
        # parts per million.  Keep a tight identity check without mistaking
        # that benign reduction-order noise for a protocol mismatch.
        if not math.isclose(distance, candidate_distance, rel_tol=1e-5, abs_tol=1e-6):
            raise SurfaceAwareResplatError(f"candidate/SCSP support distance differs for {index}")
        distances[index] = distance
    return candidates, distances, voxel_size


def _json_parent(plan: Mapping[str, Any]) -> dict[str, Any]:
    count_1, count_2 = plan["lattice_shape"]
    return {
        "parent_index": int(plan["parent_index"]),
        "output_child_indices": [int(value) for value in plan["output_indices"]],
        "child_count": int(plan["child_count"]),
        "support_distance_m": float(plan["support_distance_m"]),
        "neighbor_centroid_indices": [int(value) for value in plan["neighbor_centroid_indices"]],
        "pca_eigenvalues": [float(value) for value in plan["pca_eigenvalues"]],
        "pca_rank": int(plan["pca_rank"]),
        "right_handed_frame_columns_t1_t2_n": np.asarray(plan["frame"]).tolist(),
        "parent_projected_std_t1_t2_n": [float(value) for value in plan["parent_projected_std"]],
        "lattice_shape": [int(count_1), int(count_2)],
        "lattice_spacing_t1_t2": [float(value) for value in plan["spacing"]],
        "child_activated_scale_t1_t2_n": [float(value) for value in np.exp(plan["child_log_scale"])],
        "child_rotation_wxyz": [float(value) for value in plan["child_quaternion"]],
        "child_opacity_logit": float(plan["child_opacity_logit"]),
        "child_centers": np.asarray(plan["child_centers"]).tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_checkpoint = Path(args.input_checkpoint).resolve()
    candidate_path = Path(args.candidate_receipt).resolve()
    scsp_path = Path(args.scsp_manifest).resolve()
    sparse_path = Path(args.sparse_points).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise SurfaceAwareResplatError(f"refusing to overwrite output: {output_dir}")
    identities = {
        "input_checkpoint": _identity(input_checkpoint),
        "candidate_receipt": _identity(candidate_path),
        "scsp_manifest": _identity(scsp_path),
        "sparse_points": _identity(sparse_path),
    }
    _require_declared_sha(
        identities["input_checkpoint"]["sha256"], args.expected_checkpoint_sha256, "checkpoint"
    )
    _require_declared_sha(
        identities["candidate_receipt"]["sha256"], args.expected_candidate_receipt_sha256, "candidate receipt"
    )
    _require_declared_sha(
        identities["scsp_manifest"]["sha256"], args.expected_scsp_manifest_sha256, "SCSP manifest"
    )
    _require_declared_sha(
        identities["sparse_points"]["sha256"], args.expected_sparse_points_sha256, "sparse points"
    )

    iteration = int(args.anchor_iteration)
    params, loaded_count = _load_checkpoint_cpu(input_checkpoint, iteration)
    gaussian_count = validate_checkpoint_schema(params)
    if loaded_count != gaussian_count:
        raise SurfaceAwareResplatError("checkpoint loader/schema Gaussian counts differ")
    candidate_receipt = _read_json(candidate_path)
    scsp_manifest = _read_json(scsp_path)
    candidates, support_distances, voxel_size = _load_contract(
        candidate_receipt,
        scsp_manifest,
        scsp_identity=identities["scsp_manifest"],
        sparse_identity=identities["sparse_points"],
        checkpoint_identity=identities["input_checkpoint"],
        gaussian_count=gaussian_count,
    )
    sparse_points = _load_sparse_points(sparse_path)
    centroids = build_voxel_centroids(sparse_points, voxel_size)
    declared_raw_count = int(_nested(scsp_manifest, "sparse_support", "raw_point_count"))
    declared_centroid_count = int(_nested(scsp_manifest, "sparse_support", "voxel_centroid_count"))
    if sparse_points.shape[0] != declared_raw_count or centroids.shape[0] != declared_centroid_count:
        raise SurfaceAwareResplatError(
            "locked sparse-point/voxel-centroid counts differ from SCSP manifest"
        )

    output_params, plans = build_resplat_parameters(
        params, candidates, support_distances, centroids
    )
    final_count = validate_checkpoint_schema(output_params)
    growth_fraction = (final_count - gaussian_count) / float(gaussian_count)
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "status": "passed",
        "mode": str(args.mode),
        "scene_name": str(candidate_receipt.get("scene_name", "")),
        "anchor_iteration": iteration,
        "producer": {
            "tool": "tools/geometric_repeatability/build_surface_aware_resplat_anchor.py",
            "tool_identity": _identity(Path(__file__).resolve()),
            "code_commit": str(args.code_commit),
        },
        "inputs": identities,
        "candidate_contract": {
            "protocol": CANDIDATE_PROTOCOL,
            "indices": candidates,
            "indices_sha256": _canonical_json_sha256(candidates),
            "selection_source": "locked train/guard-only Candidate-v2 receipt",
            "test_used": False,
        },
        "sparse_support": {
            "voxel_size": voxel_size,
            "raw_point_count": int(sparse_points.shape[0]),
            "voxel_centroid_count": int(centroids.shape[0]),
            "global_nearest_distinct_centroid_count": NEIGHBOR_COUNT,
            "pca_dtype": "float64",
            "support_distance_source": "SCSP modified-row second-nearest voxel-centroid distance",
        },
        "formula": {
            "lattice": "center + a*t1 + b*t2; no fitted-plane projection",
            "axis_child_count": "max(1,ceil(parent_projected_std/d))",
            "tangent_child_std": "parent_projected_std/axis_child_count",
            "tangent_spacing": "sqrt(12)*parent_projected_std/axis_child_count",
            "normal_child_std": "min(parent_projected_normal_std,d)",
            "opacity": "alpha_child=1-(1-alpha_parent)^(1/child_count)",
            "appearance": "inherit parent f_dc and f_rest exactly",
        },
        "counts": {
            "gaussian_count_before": gaussian_count,
            "gaussian_count_after": final_count,
            "candidate_parent_count": len(candidates),
            "total_children": int(sum(plan["child_count"] for plan in plans)),
            "net_growth": final_count - gaussian_count,
            "growth_fraction": growth_fraction,
            "maximum_growth_fraction": MAX_COUNT_GROWTH_FRACTION,
            "within_signed_int32": final_count <= INT32_MAX,
        },
        "parents": [_json_parent(plan) for plan in plans],
        "row_order": {
            "noncandidate_original_rows_remain_at_exact_original_positions": True,
            "first_child_replaces_parent_row": True,
            "remaining_children_appended_in_parent_then_lattice_order": True,
        },
        "invariants": {
            "finite": True,
            "pca_rank_at_least_2": True,
            "right_handed_frames": True,
            "tangent_only_centered_lattices": True,
            "noncandidate_row_position_exact": True,
            "shared_parent_sh_inherited": True,
            "no_ply_written": True,
            "test_data_used": False,
            "fresh_optimizer_required": True,
        },
        "output": {
            "checkpoint": None,
            "fresh_optimizer_restore_mode_required": "fresh",
            "ply": None,
        },
    }

    partial = output_dir.with_name(f".{output_dir.name}.partial-{uuid.uuid4().hex}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == "build":
            checkpoint_name = f"chkpnt{iteration}.pth"
            checkpoint_path = partial / checkpoint_name
            torch.save((tuple(output_params), iteration), checkpoint_path)
            reloaded, reloaded_count = _load_checkpoint_cpu(checkpoint_path, iteration)
            if reloaded_count != final_count or validate_checkpoint_schema(reloaded) != final_count:
                raise SurfaceAwareResplatError("saved checkpoint failed schema/count reload")
            optimizer = reloaded[10]
            if (
                not isinstance(optimizer, Mapping)
                or optimizer.get("fresh_optimizer_required") is not True
                or optimizer.get("state") != {}
                or optimizer.get("param_groups") != []
            ):
                raise SurfaceAwareResplatError("saved checkpoint inherited optimizer state")
            manifest["output"]["checkpoint"] = {
                "file": checkpoint_name,
                "sha256": _sha256(checkpoint_path),
                "size_bytes": int(checkpoint_path.stat().st_size),
            }
        elif args.mode != "preflight":
            raise SurfaceAwareResplatError(f"unsupported mode: {args.mode}")
        manifest_name = (
            "surface_aware_resplat_manifest.json"
            if args.mode == "build"
            else "surface_aware_resplat_preflight.json"
        )
        (partial / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if any(path.suffix.lower() == ".ply" for path in partial.rglob("*")):
            raise SurfaceAwareResplatError("checkpoint-only sidecar unexpectedly wrote a PLY")
        if _sha256(input_checkpoint) != identities["input_checkpoint"]["sha256"]:
            raise SurfaceAwareResplatError("input checkpoint changed during construction")
        os.replace(partial, output_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "build"), required=True)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--candidate-receipt", required=True)
    parser.add_argument("--scsp-manifest", required=True)
    parser.add_argument("--sparse-points", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-iteration", type=int, default=30000)
    parser.add_argument("--expected-checkpoint-sha256", default="")
    parser.add_argument("--expected-candidate-receipt-sha256", default="")
    parser.add_argument("--expected-scsp-manifest-sha256", default="")
    parser.add_argument("--expected-sparse-points-sha256", default="")
    parser.add_argument("--code-commit", default="")
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
