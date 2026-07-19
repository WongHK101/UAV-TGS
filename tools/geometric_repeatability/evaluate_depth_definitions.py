"""Evaluate the three formal depth definitions and optional responsibilities.

This diagnostic is deliberately separate from the legacy reference-depth
evaluator.  It consumes diagnostic NPZ bundles produced by
``export_gaussian_probe_bundle.py --depth_diagnostics`` and never changes the
formal split or reference backend.

``--metric_only`` accepts a non-Gaussian adapter bundle containing only the
three registered depth maps and uses finite positive depth as the fixed model
support rule. It does not require Gaussian indices, opacity, top contributors,
or responsibility attribution.

Responsibility attribution on the Gaussian path is intentionally narrow. Front error is assigned
only for max-contribution Gaussian-center depth, whose depth and Gaussian index
refer to the same compositing event. RGB or Celsius residual attribution, when
allowed by explicit modality/provenance manifests, is a top-1 shared-occupancy
approximation rather than a top-k or exact causal decomposition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.thermal_radiometry.palette_lut import hot_iron_lut, lut_sha256


DEFAULT_THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
FRONT_AUC_LOG_MIN_M = 0.25
FRONT_AUC_LOG_MAX_M = 20.0
FORMAL_OPACITY_THRESHOLD = 0.5
FORMAL_DEPTH_MIN_M = 1.0e-6
FORMAL_SPLIT_LABELS = frozenset({"train", "guard", "test"})
REFERENCE_DEPTH_SEMANTICS = "metric_camera_z_reference_mesh"
TEMPERATURE_RESPONSIBILITY_PROTOCOL = "uav-tgs-temperature-responsibility-v1"
TEMPERATURE_SEMANTICS = "TSDK-referenced apparent-temperature consistency"
TEMPERATURE_RESOLUTION_NATIVE_EXACT = "native_exact"
TEMPERATURE_RESOLUTION_BILINEAR_NEAREST = (
    "bilinear_temperature_nearest_support"
)
DIAGNOSTIC_DEPTH_SEMANTICS = {
    "depth_expected_alpha_normalized": "metric camera-z; sum(alpha*T*z)/sum(alpha*T)",
    "depth_transmittance_median": "metric camera-z at first accepted contributor where transmittance <= 0.5; zero if absent",
    "depth_max_contribution": "metric camera-z of Gaussian maximizing alpha*T",
    "top_contributor_index": "zero-based Gaussian index maximizing alpha*T; -1 if absent",
    "top_contributor_weight": "unnormalized compositing weight alpha*T",
    "accumulated_opacity": "sum of accepted alpha*T weights",
}
OPACITY_BINS = (
    ("[0,0.1)", 0.0, 0.1),
    ("[0.1,0.25)", 0.1, 0.25),
    ("[0.25,0.5)", 0.25, 0.5),
    ("[0.5,0.75)", 0.5, 0.75),
    ("[0.75,1.01]", 0.75, 1.01),
)
DEPTH_DEFINITIONS = {
    "expected": "depth_expected_alpha_normalized",
    "median": "depth_transmittance_median",
    "max_contribution": "depth_max_contribution",
}
FORMAL_DEPTH_SEMANTICS = {
    key: DIAGNOSTIC_DEPTH_SEMANTICS[value]
    for key, value in DEPTH_DEFINITIONS.items()
}
FORMAL_SUPPORT_POLICY = {
    "reference_support": "reference_valid and finite positive reference depth",
    "gaussian_model_support": (
        "finite positive rendered depth and finite accumulated_opacity >= 0.5"
    ),
    "metric_only_model_support": "finite positive rendered depth",
    "missing": "reference support and not model support",
    "depth_min_m": FORMAL_DEPTH_MIN_M,
    "opacity_threshold": FORMAL_OPACITY_THRESHOLD,
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


FORMAL_SUPPORT_POLICY_SHA256 = _canonical_sha256(FORMAL_SUPPORT_POLICY)
FORMAL_GEOMETRY_PROTOCOL = "uav-tgs-aaai27-formal-geometry-metrics-v1"
FORMAL_GEOMETRY_PROTOCOL_SPEC = {
    "protocol": FORMAL_GEOMETRY_PROTOCOL,
    "thresholds_m": list(DEFAULT_THRESHOLDS_M),
    "depth_definitions": FORMAL_DEPTH_SEMANTICS,
    "support_policy_sha256": FORMAL_SUPPORT_POLICY_SHA256,
    "formal_auc": "front_auc_log_0p25_20m; trapezoid over all eight log-threshold samples",
}
FORMAL_GEOMETRY_PROTOCOL_SHA256 = _canonical_sha256(
    FORMAL_GEOMETRY_PROTOCOL_SPEC
)
ALIGNMENT_MAX_TRANSLATION_ERROR_M = 1.0e-4
ALIGNMENT_MAX_ROTATION_ERROR_DEG = 5.0e-2
PRINCIPAL_POINT_CENTER_TOLERANCE_PX = 1.0e-6


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_view_npz(manifest_path: Path, view: Mapping[str, Any]) -> Path:
    root = manifest_path.parent.resolve()
    path = (root / str(view.get("npz_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"View NPZ escapes manifest root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = view.get("npz_size_bytes")
    expected_sha = str(view.get("npz_sha256", "")).lower()
    if expected_size is not None and int(expected_size) != int(path.stat().st_size):
        raise RuntimeError(f"NPZ size mismatch: {path}")
    if expected_sha and expected_sha != _sha256(path):
        raise RuntimeError(f"NPZ SHA-256 mismatch: {path}")
    return path


def _parse_thresholds(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(float(item) for item in value)
    if not parsed or any((not np.isfinite(x)) or x <= 0.0 for x in parsed):
        raise ValueError("Depth thresholds must be finite positive values")
    if tuple(sorted(set(parsed))) != parsed:
        raise ValueError("Depth thresholds must be unique and strictly increasing")
    return parsed


def _validate_formal_geometry_contract(
    model_manifest: Mapping[str, Any],
    *,
    metric_only: bool,
) -> Dict[str, Any]:
    contract = model_manifest.get("formal_geometry_metric_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Model manifest is missing formal_geometry_metric_contract")
    expected = {
        "protocol": FORMAL_GEOMETRY_PROTOCOL,
        "protocol_sha256": FORMAL_GEOMETRY_PROTOCOL_SHA256,
        "support_policy_sha256": FORMAL_SUPPORT_POLICY_SHA256,
        "thresholds_m": list(DEFAULT_THRESHOLDS_M),
        "depth_definitions": FORMAL_DEPTH_SEMANTICS,
        "opacity_threshold": FORMAL_OPACITY_THRESHOLD,
        "depth_min_m": FORMAL_DEPTH_MIN_M,
        "support_mode": (
            "finite_positive_depth_metric_only"
            if metric_only
            else "gaussian_accumulated_opacity"
        ),
    }
    for key, expected_value in expected.items():
        observed = contract.get(key)
        if isinstance(expected_value, float):
            try:
                matches = math.isclose(
                    float(observed),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(
                f"Formal geometry metric contract mismatch for {key}: "
                f"observed={observed!r} expected={expected_value!r}"
            )
    return dict(contract)


def _as_hw(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{label} must have shape HxW or 1xHxW, got {array.shape}")
    return array


def _image_abs_residual(rendered: np.ndarray, target: np.ndarray) -> np.ndarray:
    rendered = np.asarray(rendered, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if rendered.shape != target.shape:
        raise ValueError(f"Image residual shape mismatch: {rendered.shape} vs {target.shape}")
    if rendered.ndim == 2:
        return np.abs(rendered - target)
    if rendered.ndim != 3:
        raise ValueError(f"Image arrays must be HxW, CxHxW, or HxWxC; got {rendered.shape}")
    if rendered.shape[0] in (1, 3, 4):
        return np.mean(np.abs(rendered - target), axis=0)
    if rendered.shape[-1] in (1, 3, 4):
        return np.mean(np.abs(rendered - target), axis=-1)
    raise ValueError(f"Cannot identify image channel axis: {rendered.shape}")


def _first_present(npz: Mapping[str, np.ndarray], names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in npz:
            return np.asarray(npz[name])
    return None


def _native_camera_map(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json_or_list(path)
    if not isinstance(payload, list):
        raise ValueError("Native model cameras.json must contain a list")
    result: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Native model camera entries must be objects")
        stem = _normalized_stem(item.get("img_name", ""))
        if not stem or stem in result:
            raise ValueError(f"Native model cameras.json has missing/duplicate image stem {stem!r}")
        position = np.asarray(item.get("position"), dtype=np.float64)
        rotation = np.asarray(item.get("rotation"), dtype=np.float64)
        if position.shape != (3,) or rotation.shape != (3, 3) or not np.all(
            np.isfinite(np.concatenate([position, rotation.reshape(-1)]))
        ):
            raise ValueError(f"Native model camera {stem!r} has invalid pose")
        result[stem] = item
    return result


def _load_json_or_list(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _native_camera_matrix(item: Mapping[str, Any]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(item["rotation"], dtype=np.float64)
    matrix[:3, 3] = np.asarray(item["position"], dtype=np.float64)
    return matrix


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first, dtype=np.float64) @ np.asarray(second, dtype=np.float64).T
    cosine = np.clip((float(np.trace(relative)) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _view_metadata(
    reference_view: Mapping[str, Any],
    model_view: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    supplemental_view: Mapping[str, Any] | None = None,
) -> Dict[str, str]:
    def pick(*keys: str, default: str = "unknown") -> str:
        for source in (supplemental_view or {}, model_view, reference_view, model_manifest):
            for key in keys:
                value = source.get(key)
                if value is not None and str(value).strip():
                    return str(value)
        return default

    raw_orientation = pick("orientation", "view_class", "view_type", "stratum")
    orientation_lower = raw_orientation.lower()
    if "oblique" in orientation_lower:
        orientation = "oblique"
    elif "nadir" in orientation_lower:
        orientation = "nadir"
    else:
        pitch: float | None = None
        for source in (supplemental_view or {}, model_view, reference_view, model_manifest):
            for key in ("gimbal_pitch_deg", "pitch_deg"):
                if source.get(key) is not None:
                    try:
                        pitch = float(source[key])
                    except (TypeError, ValueError):
                        pitch = None
                    if pitch is not None:
                        break
            if pitch is not None:
                break
        if pitch is None:
            pitch_match = re.search(r"(?:pitch|gimbal)[^0-9+\-]*([+\-]?\d+(?:\.\d+)?)", orientation_lower)
            if pitch_match:
                pitch = float(pitch_match.group(1))
        if pitch is not None and np.isfinite(pitch):
            orientation = "nadir" if abs(pitch + 90.0) <= 15.0 else "oblique"
        else:
            orientation = raw_orientation
    block = ""
    if supplemental_view:
        for key in ("block_id", "block", "route_block"):
            value = supplemental_view.get(key)
            if value is not None and str(value).strip():
                block = str(value)
                break
        if not block and supplemental_view.get("block_index") is not None:
            block_index = str(supplemental_view["block_index"])
            strip_id = str(supplemental_view.get("strip_id", "")).strip()
            block = f"{strip_id}:{block_index}" if strip_id else block_index
    if not block:
        block = pick("block_id", "block", "route_block", default="")
    if not block:
        block_index = pick("block_index", default="")
        strip_id = pick("strip_id", default="")
        if block_index:
            block = f"{strip_id}:{block_index}" if strip_id else block_index
        else:
            block = "unknown"
    return {
        "split": pick("split", "partition", "split_label"),
        "block": block,
        "orientation": orientation,
    }


def _normalized_stem(value: Any) -> str:
    normalized = str(value).strip().replace("\\", "/")
    return Path(normalized).stem.lower()


def _formal_split_records_by_stem(split_manifest: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    if split_manifest is None:
        raise ValueError("A formal split manifest is required")
    records = split_manifest.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Formal split manifest 'records' must be a list")
    by_stem: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Formal split records must be JSON objects")
        keys = {
            _normalized_stem(record.get("pair_id", "")),
            _normalized_stem(record.get("filename", "")),
            _normalized_stem(record.get("camera_name", "")),
            _normalized_stem(record.get("thermal_camera_name", "")),
        }
        original_files = record.get("original_files")
        if isinstance(original_files, dict):
            keys.update(_normalized_stem(value) for value in original_files.values())
        keys.discard("")
        for key in keys:
            previous = by_stem.get(key)
            if previous is not None and previous is not record:
                raise ValueError(f"Duplicate formal split stem {key!r}")
            by_stem[key] = record
    labels = {str(record.get("split", "")).strip().lower() for record in records}
    if labels != FORMAL_SPLIT_LABELS:
        raise ValueError(
            "Formal split manifest labels must be exactly train, guard, and test; "
            f"observed={sorted(labels)}"
        )
    if not by_stem:
        raise ValueError("Formal split manifest contains no addressable records")
    return by_stem


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value).strip()))


def _manifest_scene(manifest: Mapping[str, Any], *, label: str) -> str:
    value = manifest.get("scene_name", manifest.get("scene", ""))
    scene = str(value).strip()
    if not scene:
        records = manifest.get("records")
        if isinstance(records, list):
            scenes = {
                str(record.get("scene_name", record.get("scene", ""))).strip()
                for record in records
                if isinstance(record, dict)
            }
            scenes.discard("")
            if len(scenes) == 1:
                scene = next(iter(scenes))
            elif len(scenes) > 1:
                raise ValueError(f"{label} contains multiple scenes: {sorted(scenes)}")
    if not scene:
        raise ValueError(f"{label} is missing scene_name/scene")
    return scene


def _identity_sha(manifest: Mapping[str, Any], key: str, *, label: str) -> str:
    identity = manifest.get(key)
    if not isinstance(identity, dict):
        raise ValueError(f"{label} is missing {key}")
    value = str(identity.get("sha256", "")).lower()
    if not _is_sha256(value):
        raise ValueError(f"{label}.{key}.sha256 is invalid")
    return value


def _view_map(manifest: Mapping[str, Any], *, label: str) -> Dict[str, Dict[str, Any]]:
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{label}.views must be a non-empty list")
    mapped: Dict[str, Dict[str, Any]] = {}
    for view in views:
        if not isinstance(view, dict):
            raise ValueError(f"{label}.views entries must be objects")
        name = str(view.get("image_name", "")).strip()
        if not name or name in mapped:
            raise ValueError(f"{label} has missing/duplicate image_name {name!r}")
        mapped[name] = view
    return mapped


def _camera_fields(view: Mapping[str, Any], *, label: str) -> Dict[str, Any]:
    required = ("width", "height", "fx", "fy", "cx", "cy", "camera_to_world")
    missing = [key for key in required if key not in view]
    if missing:
        raise ValueError(f"{label} missing camera fields: {missing}")
    width, height = int(view["width"]), int(view["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} has invalid dimensions {width}x{height}")
    intrinsics = np.asarray([view["fx"], view["fy"], view["cx"], view["cy"]], dtype=np.float64)
    if not np.all(np.isfinite(intrinsics)) or np.any(intrinsics[:2] <= 0.0):
        raise ValueError(f"{label} has invalid intrinsics")
    c2w = np.asarray(view["camera_to_world"], dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise ValueError(f"{label}.camera_to_world must be finite 4x4")
    if not np.allclose(c2w[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9):
        raise ValueError(f"{label}.camera_to_world has invalid homogeneous row")
    return {"width": width, "height": height, "intrinsics": intrinsics, "camera_to_world": c2w}


def _validate_camera_equivalence(
    image_name: str,
    reference_view: Mapping[str, Any],
    model_view: Mapping[str, Any],
    probe_view: Mapping[str, Any],
) -> tuple[int, int]:
    camera_sets = {
        "reference": _camera_fields(reference_view, label=f"reference[{image_name}]"),
        "model": _camera_fields(model_view, label=f"model[{image_name}]"),
        "probe": _camera_fields(probe_view, label=f"probe[{image_name}]"),
    }
    baseline = camera_sets["probe"]
    for label, camera in camera_sets.items():
        if (camera["width"], camera["height"]) != (baseline["width"], baseline["height"]):
            raise ValueError(f"{image_name}: {label}/probe dimensions mismatch")
        if not np.allclose(camera["intrinsics"], baseline["intrinsics"], rtol=1e-9, atol=1e-6):
            raise ValueError(f"{image_name}: {label}/probe intrinsics mismatch")
        if not np.allclose(camera["camera_to_world"], baseline["camera_to_world"], rtol=1e-9, atol=1e-8):
            raise ValueError(f"{image_name}: {label}/probe camera_to_world mismatch")
    return int(baseline["height"]), int(baseline["width"])


def _camera_sha256_for_matrix(view: Mapping[str, Any], camera_to_world: Any) -> str:
    camera = _camera_fields(view, label="camera hash input")
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise ValueError("Render camera_to_world must be finite 4x4")
    payload = {
        "width": camera["width"],
        "height": camera["height"],
        "fx_fy_cx_cy": [float(value) for value in camera["intrinsics"]],
        "camera_to_world": c2w.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _camera_sha256(view: Mapping[str, Any]) -> str:
    return _camera_sha256_for_matrix(view, view["camera_to_world"])


def _camera_set_sha256(views: Mapping[str, Mapping[str, Any]]) -> str:
    payload = [(name, _camera_sha256(views[name])) for name in sorted(views)]
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _render_camera_set_sha256(views: Mapping[str, Mapping[str, Any]]) -> str:
    payload = []
    for name in sorted(views):
        view = views[name]
        payload.append((name, _camera_sha256_for_matrix(view, view.get("native_camera_to_world"))))
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_group_label(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "empty"
    suffix = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:10]
    return f"{base[:80]}__{suffix}"


@dataclass
class ViewDepthStats:
    image_name: str
    metadata: Dict[str, str]
    reference_count: int
    valid_count: int
    missing_count: int
    signed_residual: np.ndarray
    front_counts: Dict[float, int]
    agreement_counts: Dict[float, int]
    behind_counts: Dict[float, int]


def _curve_auc(thresholds: Sequence[float], values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(thresholds, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.size == 1:
        raw = 0.0
        normalized = float(y[0])
    else:
        trapezoid = getattr(np, "trapezoid", None)
        raw = float(trapezoid(y, x) if trapezoid is not None else np.trapz(y, x))
        normalized = raw / float(x[-1] - x[0])
    return {"raw_rate_m": raw, "normalized": normalized, "tau_min_m": float(x[0]), "tau_max_m": float(x[-1])}


def _log_curve_auc(
    thresholds: Sequence[float],
    values: Sequence[float],
    *,
    tau_min_m: float = FRONT_AUC_LOG_MIN_M,
    tau_max_m: float = FRONT_AUC_LOG_MAX_M,
) -> Dict[str, Any] | None:
    """Integrate a threshold curve uniformly in log-distance.

    The named 0.25--20 m score is emitted only for the exact formal eight-point
    grid. Merely including both endpoints is insufficient. Samples are
    integrated with the trapezoid rule in ``log(tau)`` and the result is
    normalized by the log-interval width.
    """

    x = np.asarray(thresholds, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size == 0:
        raise ValueError("Log-AUC thresholds and values must be non-empty 1-D arrays of equal length")
    if np.any(~np.isfinite(x)) or np.any(x <= 0.0) or np.any(~np.isfinite(y)):
        raise ValueError("Log-AUC thresholds must be positive and all inputs finite")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("Log-AUC thresholds must be strictly increasing")
    expected_grid = np.asarray(DEFAULT_THRESHOLDS_M, dtype=np.float64)
    if (
        x.size != expected_grid.size
        or not np.allclose(x, expected_grid, rtol=0.0, atol=1.0e-12)
        or not math.isclose(float(x[0]), tau_min_m, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(float(x[-1]), tau_max_m, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        return None
    selected_x = x
    selected_y = y
    log_x = np.log(selected_x)
    trapezoid = getattr(np, "trapezoid", None)
    raw = float(
        trapezoid(selected_y, log_x) if trapezoid is not None else np.trapz(selected_y, log_x)
    )
    normalized = raw / float(math.log(tau_max_m / tau_min_m))
    return {
        "raw_rate_log_interval": raw,
        "normalized": normalized,
        "tau_min_m": float(tau_min_m),
        "tau_max_m": float(tau_max_m),
        "integration_domain": "natural_log_threshold_m",
        "sample_thresholds_m": [float(value) for value in selected_x],
    }


def compute_depth_metrics(
    reference_depth: np.ndarray,
    rendered_depth: np.ndarray,
    *,
    reference_valid: np.ndarray | None = None,
    accumulated_opacity: np.ndarray | None = None,
    opacity_threshold: float = 0.5,
    model_selection_mask: np.ndarray | None = None,
    depth_min: float = 1e-6,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
) -> Dict[str, Any]:
    """Compute one-view metrics; exposed separately for unit tests and reuse."""

    thresholds = _parse_thresholds(thresholds_m)
    reference_depth = _as_hw(np.asarray(reference_depth, dtype=np.float64), label="reference_depth")
    rendered_depth = _as_hw(np.asarray(rendered_depth, dtype=np.float64), label="rendered_depth")
    if reference_depth.shape != rendered_depth.shape:
        raise ValueError(f"Depth shape mismatch: {reference_depth.shape} vs {rendered_depth.shape}")
    if reference_valid is None:
        reference_mask = np.isfinite(reference_depth) & (reference_depth > depth_min)
    else:
        reference_mask = _as_hw(np.asarray(reference_valid), label="reference_valid").astype(bool)
        reference_mask &= np.isfinite(reference_depth) & (reference_depth > depth_min)
    model_mask = np.isfinite(rendered_depth) & (rendered_depth > depth_min)
    if model_selection_mask is not None:
        selection = _as_hw(np.asarray(model_selection_mask), label="model_selection_mask").astype(bool)
        if selection.shape != reference_depth.shape:
            raise ValueError("Selection/depth shape mismatch")
        model_mask &= selection
    if accumulated_opacity is not None:
        opacity = _as_hw(np.asarray(accumulated_opacity, dtype=np.float64), label="accumulated_opacity")
        if opacity.shape != reference_depth.shape:
            raise ValueError("Opacity/depth shape mismatch")
        model_mask &= np.isfinite(opacity) & (opacity >= float(opacity_threshold))
    joint = reference_mask & model_mask
    missing = reference_mask & ~model_mask
    signed = np.asarray(rendered_depth[joint] - reference_depth[joint], dtype=np.float32)
    front_counts = {
        tau: int(np.count_nonzero(joint & (rendered_depth < reference_depth - tau))) for tau in thresholds
    }
    agreement_counts = {
        tau: int(np.count_nonzero(joint & (np.abs(rendered_depth - reference_depth) <= tau))) for tau in thresholds
    }
    behind_counts = {
        tau: int(np.count_nonzero(joint & (rendered_depth > reference_depth + tau))) for tau in thresholds
    }
    return {
        "reference_count": int(np.count_nonzero(reference_mask)),
        "valid_count": int(np.count_nonzero(joint)),
        "missing_count": int(np.count_nonzero(missing)),
        "signed_residual": signed,
        "front_counts": front_counts,
        "agreement_counts": agreement_counts,
        "behind_counts": behind_counts,
        "joint_valid_mask": joint,
    }


def _summarize_view_stats(
    stats: Sequence[ViewDepthStats],
    thresholds: Sequence[float],
    *,
    evaluation_mode: str = "formal",
) -> Dict[str, Any]:
    if evaluation_mode not in {"formal", "legacy_custom"}:
        raise ValueError(f"Unsupported evaluation mode: {evaluation_mode!r}")
    reference_count = sum(item.reference_count for item in stats)
    valid_count = sum(item.valid_count for item in stats)
    missing_count = sum(item.missing_count for item in stats)
    if reference_count <= 0:
        empty_payload = {
            "view_count": len(stats),
            "reference_valid_pixels": 0,
            "model_valid_pixels": 0,
            "missing_pixels": 0,
            "missing_rate": None,
            "mean_abs_error_m": None,
            "median_abs_error_m": None,
            "signed_bias_m": None,
            "threshold_metrics": [],
            "front_curve_auc_log": None,
            "front_auc_log_0p25_20m": None,
            "evaluation_mode": evaluation_mode,
        }
        if evaluation_mode == "legacy_custom":
            empty_payload["front_curve_auc_linear_legacy"] = None
            empty_payload["agreement_curve_auc_linear_legacy"] = None
        return empty_payload
    residual_arrays = [item.signed_residual for item in stats if item.signed_residual.size]
    residual = np.concatenate(residual_arrays) if residual_arrays else np.zeros((0,), dtype=np.float32)
    front_rates: List[float] = []
    agreement_rates: List[float] = []
    threshold_metrics: List[Dict[str, Any]] = []
    for tau in thresholds:
        front_count = sum(item.front_counts[tau] for item in stats)
        agreement_count = sum(item.agreement_counts[tau] for item in stats)
        behind_count = sum(item.behind_counts[tau] for item in stats)
        front_rate = float(front_count) / float(reference_count)
        agreement_rate = float(agreement_count) / float(reference_count)
        behind_rate = float(behind_count) / float(reference_count)
        front_rates.append(front_rate)
        agreement_rates.append(agreement_rate)
        threshold_metrics.append(
            {
                "threshold_m": float(tau),
                "front_count": int(front_count),
                "front_rate": front_rate,
                "agreement_count": int(agreement_count),
                "agreement_rate": agreement_rate,
                "behind_count": int(behind_count),
                "behind_rate": behind_rate,
            }
        )
    front_curve_auc_log = (
        _log_curve_auc(thresholds, front_rates)
        if evaluation_mode == "formal"
        else None
    )
    if evaluation_mode == "formal" and front_curve_auc_log is None:
        raise ValueError(
            "Formal front_auc_log_0p25_20m requires the exact complete eight-point threshold grid"
        )
    payload = {
        "view_count": len(stats),
        "reference_valid_pixels": int(reference_count),
        "model_valid_pixels": int(valid_count),
        "missing_pixels": int(missing_count),
        "missing_rate": float(missing_count) / float(reference_count),
        "mean_abs_error_m": float(np.mean(np.abs(residual))) if residual.size else None,
        "median_abs_error_m": float(np.median(np.abs(residual))) if residual.size else None,
        "signed_bias_m": float(np.mean(residual)) if residual.size else None,
        "threshold_metrics": threshold_metrics,
        "front_curve_auc_log": front_curve_auc_log,
        "front_auc_log_0p25_20m": (
            front_curve_auc_log["normalized"] if front_curve_auc_log is not None else None
        ),
        "evaluation_mode": evaluation_mode,
    }
    if evaluation_mode == "legacy_custom":
        payload["front_curve_auc_linear_legacy"] = _curve_auc(
            thresholds, front_rates
        )
        payload["agreement_curve_auc_linear_legacy"] = _curve_auc(
            thresholds, agreement_rates
        )
    return payload


def _write_cdf(path: Path, residual_arrays: Sequence[np.ndarray], max_points: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonempty = [array for array in residual_arrays if array.size]
    residual = np.sort(np.concatenate(nonempty).astype(np.float32, copy=False)) if nonempty else np.zeros((0,), np.float32)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["signed_residual_m", "empirical_cdf", "sample_rank", "sample_count"])
        if residual.size == 0:
            return
        point_count = min(int(max_points), int(residual.size))
        indices = np.unique(np.linspace(0, residual.size - 1, point_count, dtype=np.int64))
        for index in indices:
            writer.writerow(
                [
                    f"{float(residual[index]):.9f}",
                    f"{float(index + 1) / float(residual.size):.12f}",
                    int(index + 1),
                    int(residual.size),
                ]
            )


@dataclass(frozen=True)
class BoundIndexSet:
    label: str
    indices: frozenset[int]
    path: str
    file_sha256: str
    anchor_sha256: str


def _nested_value(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for keys in paths:
        value: Any = payload
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None and str(value).strip():
            return value
    return None


def _load_bound_index_set(
    path: Path | None,
    *,
    label: str,
    explicit_anchor_sha256: str,
    model_index_anchor_sha256: str,
    gaussian_count: int,
) -> BoundIndexSet:
    if path is None:
        if str(explicit_anchor_sha256).strip():
            raise ValueError(f"{label} anchor SHA was supplied without an index set")
        return BoundIndexSet(label, frozenset(), "", "", "")
    if not path.is_file():
        raise FileNotFoundError(path)
    path = path.resolve()
    suffix = path.suffix.lower()
    embedded_anchor = ""
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as payload:
            keys = list(payload.keys())
            if not keys:
                values = np.zeros((0,), dtype=np.int64)
            else:
                values = np.asarray(payload[keys[0]])
    elif suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict):
            values = payload.get(
                "indices",
                payload.get(
                    "selected_indices",
                    payload.get("modified_indices", payload.get("gaussian_indices", [])),
                ),
            )
            embedded = _nested_value(
                payload,
                (
                    ("anchor_ply_sha256",),
                    ("gaussian_index_anchor_sha256",),
                    ("input", "anchor_sha256"),
                    ("input", "ply_sha256"),
                    ("anchor", "sha256"),
                ),
            )
            embedded_anchor = str(embedded or "").lower()
        else:
            raise ValueError(f"Unsupported JSON index-set payload: {path}")
    else:
        tokens: List[str] = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            token = line.split(",", 1)[0].strip()
            if token and not token.lower().startswith(("index", "gaussian")):
                tokens.append(token)
        values = tokens
    raw_values = np.asarray(values).reshape(-1)
    parsed: set[int] = set()
    for value in raw_values.tolist():
        if isinstance(value, (float, np.floating)) and (not np.isfinite(value) or not float(value).is_integer()):
            raise ValueError(f"{label} contains a non-integer index: {value!r}")
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} contains a non-integer index: {value!r}") from exc
        if index < 0 or index >= int(gaussian_count):
            raise ValueError(f"{label} index {index} is outside [0, {gaussian_count})")
        parsed.add(index)

    explicit_anchor = str(explicit_anchor_sha256).strip().lower()
    if explicit_anchor and not _is_sha256(explicit_anchor):
        raise ValueError(f"{label} explicit anchor SHA-256 is invalid")
    if embedded_anchor and not _is_sha256(embedded_anchor):
        raise ValueError(f"{label} embedded anchor SHA-256 is invalid")
    if explicit_anchor and embedded_anchor and explicit_anchor != embedded_anchor:
        raise ValueError(f"{label} explicit/embedded anchor SHA-256 mismatch")
    bound_anchor = explicit_anchor or embedded_anchor
    if not bound_anchor:
        raise ValueError(f"{label} must bind its Gaussian index space to an anchor SHA-256")
    if bound_anchor != model_index_anchor_sha256:
        raise ValueError(
            f"{label} anchor SHA-256 does not match the model Gaussian index anchor: "
            f"{bound_anchor} != {model_index_anchor_sha256}"
        )
    return BoundIndexSet(
        label=label,
        indices=frozenset(parsed),
        path=str(path),
        file_sha256=_sha256(path),
        anchor_sha256=bound_anchor,
    )


def _overlap(candidate: set[int], reference: set[int]) -> Dict[str, Any] | None:
    if not reference:
        return None
    intersection = candidate & reference
    union = candidate | reference
    return {
        "intersection_count": len(intersection),
        "candidate_count": len(candidate),
        "reference_count": len(reference),
        "candidate_precision": float(len(intersection)) / float(len(candidate)) if candidate else 0.0,
        "reference_recall": float(len(intersection)) / float(len(reference)),
        "jaccard": float(len(intersection)) / float(len(union)) if union else 1.0,
    }


def summarize_responsibility(
    mass: np.ndarray,
    *,
    scsp_indices: set[int] | None = None,
    clamp20_indices: set[int] | None = None,
) -> Dict[str, Any]:
    mass = np.asarray(mass, dtype=np.float64).reshape(-1)
    mass = np.where(np.isfinite(mass) & (mass > 0.0), mass, 0.0)
    positive_indices = np.flatnonzero(mass > 0.0)
    order = positive_indices[np.argsort(-mass[positive_indices], kind="stable")]
    total = float(np.sum(mass, dtype=np.float64))

    def set_mass(indices: set[int] | None) -> Dict[str, Any] | None:
        if not indices:
            return None
        valid = np.fromiter(sorted(indices), dtype=np.int64)
        carried = float(np.sum(mass[valid], dtype=np.float64))
        return {
            "set_count": int(valid.size),
            "positive_mass_count": int(np.count_nonzero(mass[valid] > 0.0)),
            "mass": carried,
            "mass_share": carried / total if total > 0.0 else 0.0,
        }

    def top_summary(requested_count: int) -> Dict[str, Any]:
        requested_count = min(max(0, int(requested_count)), int(mass.size))
        realized_count = min(requested_count, int(order.size))
        indices = order[:realized_count]
        selected = {int(index) for index in indices.tolist()}
        selected_mass = float(np.sum(mass[indices], dtype=np.float64)) if realized_count else 0.0
        return {
            "requested_count": requested_count,
            "realized_count": realized_count,
            "count": realized_count,
            "mass": selected_mass,
            "mass_share": selected_mass / total if total > 0.0 else 0.0,
            "scsp_overlap": _overlap(selected, scsp_indices or set()),
            "clamp20_overlap": _overlap(selected, clamp20_indices or set()),
        }

    fractions = {"top_0.01pct": 0.0001, "top_0.1pct": 0.001, "top_1pct": 0.01}
    payload: Dict[str, Any] = {
        "gaussian_count": int(mass.size),
        "nonzero_gaussian_count": int(np.count_nonzero(mass)),
        "total_mass": total,
        "top20": top_summary(min(20, mass.size)),
        "scsp_set_mass": set_mass(scsp_indices),
        "clamp20_set_mass": set_mass(clamp20_indices),
    }
    for label, fraction in fractions.items():
        payload[label] = top_summary(int(math.ceil(float(mass.size) * fraction)))
    payload["top20_entries"] = [
        {
            "rank": rank + 1,
            "gaussian_index": int(index),
            "mass": float(mass[index]),
            "mass_share": float(mass[index]) / total if total > 0.0 else 0.0,
        }
        for rank, index in enumerate(order[: min(20, order.size)])
    ]
    return payload


def _concentration(entries: Mapping[str, float]) -> Dict[str, Any]:
    ordered = sorted(((str(key), float(value)) for key, value in entries.items()), key=lambda item: (-item[1], item[0]))
    total = sum(value for _, value in ordered)
    shares = [value / total for _, value in ordered] if total > 0.0 else []
    return {
        "entry_count": len(ordered),
        "total_mass": total,
        "top1_share": shares[0] if shares else 0.0,
        "top5_share": sum(shares[:5]),
        "herfindahl_index": sum(value * value for value in shares),
        "top_entries": [{"label": key, "mass": value, "share": value / total if total > 0.0 else 0.0} for key, value in ordered[:20]],
    }


def _scatter_add(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = (indices >= 0) & (indices < target.size) & np.isfinite(values) & (values > 0.0)
    if np.any(valid):
        target += np.bincount(indices[valid], weights=values[valid], minlength=target.size)


def _assignment_coverage(indices: np.ndarray, values: np.ndarray, gaussian_count: int) -> Dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if indices.shape != values.shape:
        raise ValueError(f"Responsibility index/value shape mismatch: {indices.shape} vs {values.shape}")
    positive = np.isfinite(values) & (values > 0.0)
    assigned = positive & (indices >= 0) & (indices < int(gaussian_count))
    raw_mass = float(np.sum(values[positive], dtype=np.float64))
    assigned_mass = float(np.sum(values[assigned], dtype=np.float64))
    unassigned_mass = raw_mass - assigned_mass
    raw_pixels = int(np.count_nonzero(positive))
    assigned_pixels = int(np.count_nonzero(assigned))
    return {
        "raw_mass": raw_mass,
        "assigned_mass": assigned_mass,
        "unassigned_mass": unassigned_mass,
        "assigned_mass_fraction": assigned_mass / raw_mass if raw_mass > 0.0 else 1.0,
        "raw_positive_pixel_count": raw_pixels,
        "assigned_positive_pixel_count": assigned_pixels,
        "unassigned_positive_pixel_count": raw_pixels - assigned_pixels,
        "assigned_positive_pixel_fraction": assigned_pixels / raw_pixels if raw_pixels > 0 else 1.0,
    }


def _accumulate_assignment_coverage(target: MutableMapping[str, Any], update: Mapping[str, Any]) -> None:
    for key in (
        "raw_mass",
        "assigned_mass",
        "unassigned_mass",
        "raw_positive_pixel_count",
        "assigned_positive_pixel_count",
        "unassigned_positive_pixel_count",
    ):
        target[key] = target.get(key, 0) + update[key]


def _finalize_assignment_coverage(value: Mapping[str, Any]) -> Dict[str, Any]:
    raw_mass = float(value.get("raw_mass", 0.0))
    assigned_mass = float(value.get("assigned_mass", 0.0))
    raw_pixels = int(value.get("raw_positive_pixel_count", 0))
    assigned_pixels = int(value.get("assigned_positive_pixel_count", 0))
    return {
        "raw_mass": raw_mass,
        "assigned_mass": assigned_mass,
        "unassigned_mass": float(value.get("unassigned_mass", raw_mass - assigned_mass)),
        "assigned_mass_fraction": assigned_mass / raw_mass if raw_mass > 0.0 else 1.0,
        "raw_positive_pixel_count": raw_pixels,
        "assigned_positive_pixel_count": assigned_pixels,
        "unassigned_positive_pixel_count": int(value.get("unassigned_positive_pixel_count", raw_pixels - assigned_pixels)),
        "assigned_positive_pixel_fraction": assigned_pixels / raw_pixels if raw_pixels > 0 else 1.0,
    }


def _group_assignment_coverage(entries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    finalized = {str(label): _finalize_assignment_coverage(value) for label, value in entries.items()}
    aggregate: Dict[str, Any] = {}
    for value in finalized.values():
        _accumulate_assignment_coverage(aggregate, value)
    ordered = sorted(finalized.items(), key=lambda item: item[0])
    return {
        "aggregate": _finalize_assignment_coverage(aggregate),
        "entry_count": len(ordered),
        "minimum_assigned_mass_fraction": (
            min(value["assigned_mass_fraction"] for _, value in ordered) if ordered else 1.0
        ),
        "entries": [{"label": label, **value} for label, value in ordered],
        "top_unassigned_mass_entries": [
            {"label": label, **value}
            for label, value in sorted(
                ordered,
                key=lambda item: (-float(item[1]["unassigned_mass"]), item[0]),
            )[:20]
        ],
    }


def _validate_top_contributor_arrays(
    index_array: np.ndarray,
    weight_array: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    gaussian_count: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_index = _as_hw(np.asarray(index_array), label=f"{label} top_contributor_index")
    if raw_index.shape != expected_shape:
        raise ValueError(f"{label}: top-contributor index/image shape mismatch")
    if not np.issubdtype(raw_index.dtype, np.integer):
        raise TypeError(f"{label}: top_contributor_index must have integer dtype, got {raw_index.dtype}")
    indices = raw_index.astype(np.int64, copy=False)
    invalid = (indices < -1) | (indices >= int(gaussian_count))
    if np.any(invalid):
        bad = int(indices[invalid][0])
        raise ValueError(f"{label}: top contributor index {bad} is outside -1/[0,{gaussian_count})")

    raw_weight = _as_hw(np.asarray(weight_array), label=f"{label} top_contributor_weight")
    if raw_weight.shape != expected_shape:
        raise ValueError(f"{label}: top-contributor weight/image shape mismatch")
    if not np.issubdtype(raw_weight.dtype, np.floating):
        raise TypeError(f"{label}: top_contributor_weight must be floating point, got {raw_weight.dtype}")
    weights = raw_weight.astype(np.float64, copy=False)
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{label}: top_contributor_weight contains NaN/Inf")
    if np.any(weights < 0.0):
        raise ValueError(f"{label}: top_contributor_weight contains negative values")
    if np.any((indices == -1) & (weights != 0.0)):
        raise ValueError(f"{label}: absent top contributor must have zero weight")
    return indices, weights


def _validate_joint_diagnostic_arrays(
    model_npz: Mapping[str, np.ndarray],
    *,
    expected_shape: tuple[int, int],
    gaussian_count: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices, weights = _validate_top_contributor_arrays(
        model_npz["top_contributor_index"],
        model_npz["top_contributor_weight"],
        expected_shape=expected_shape,
        gaussian_count=gaussian_count,
        label=label,
    )
    opacity = _as_hw(np.asarray(model_npz["accumulated_opacity"], dtype=np.float64), label=f"{label} opacity")
    expected = _as_hw(
        np.asarray(model_npz[DEPTH_DEFINITIONS["expected"]], dtype=np.float64),
        label=f"{label} expected depth",
    )
    median = _as_hw(
        np.asarray(model_npz[DEPTH_DEFINITIONS["median"]], dtype=np.float64),
        label=f"{label} median depth",
    )
    maximum = _as_hw(
        np.asarray(model_npz[DEPTH_DEFINITIONS["max_contribution"]], dtype=np.float64),
        label=f"{label} max-contribution depth",
    )
    for name, value in (("opacity", opacity), ("expected", expected), ("median", median), ("maximum", maximum)):
        if value.shape != expected_shape or not np.all(np.isfinite(value)):
            raise ValueError(f"{label}: {name} diagnostic must be finite and match camera dimensions")
    eps = 1.0e-5
    if np.any(opacity < -eps) or np.any(opacity > 1.0 + eps):
        raise ValueError(f"{label}: accumulated opacity is outside [0,1] tolerance")
    if np.any(weights > opacity + eps):
        raise ValueError(f"{label}: top-contributor weight exceeds accumulated opacity")
    has_top = indices >= 0
    if not np.array_equal(has_top, weights > 0.0) or not np.array_equal(has_top, maximum > 0.0):
        raise ValueError(f"{label}: top index/weight/max-depth absence semantics disagree")
    if np.any(expected < 0.0) or np.any(median < 0.0) or np.any(maximum < 0.0):
        raise ValueError(f"{label}: diagnostic depth contains negative values")
    if not np.array_equal(expected > 0.0, opacity > 0.0):
        raise ValueError(f"{label}: expected-depth/opacity support semantics disagree")
    if np.any((opacity >= 0.5 + eps) & (median <= 0.0)) or np.any((opacity < 0.5 - eps) & (median != 0.0)):
        raise ValueError(f"{label}: transmittance-median depth/opacity threshold semantics disagree")
    return indices, weights


def _contract_shape_hw(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element H,W array")
    try:
        shape = (int(value[0]), int(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid dimensions") from exc
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(f"{label} dimensions must be positive")
    return shape


def _validate_temperature_resolution_contract(
    *,
    payload: Mapping[str, Any],
    derivation: Mapping[str, Any],
    target_provenance: Mapping[str, Any],
    model_views: Mapping[str, Mapping[str, Any]],
    contract_views: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate native-exact or the sole explicit direct-Celsius resize recipe."""

    exact = target_provenance.get(
        "source_training_target_forward_colorization_exact_on_valid_support"
    )
    resolution = payload.get("resolution_policy")
    if resolution is None:
        # Backward-compatible native receipts predate the explicit resolution
        # block.  Their old evidence boundary remains strict and unchanged.
        if exact is not True:
            raise ValueError(
                "Legacy/native temperature provenance must prove exact training colorization"
            )
        if target_provenance.get("direct_float_temperature_resized", False) is not False:
            raise ValueError("Legacy/native temperature provenance cannot claim resizing")
        return
    if not isinstance(resolution, Mapping):
        raise ValueError("Temperature resolution policy must be an object")
    policy = str(resolution.get("requested_policy", "")).strip()
    if policy not in {
        TEMPERATURE_RESOLUTION_NATIVE_EXACT,
        TEMPERATURE_RESOLUTION_BILINEAR_NEAREST,
    }:
        raise ValueError(f"Unsupported temperature resolution policy {policy!r}")
    if resolution.get("native_exact_is_default") is not True:
        raise ValueError("Temperature resolution contract must preserve native_exact default")
    if str(resolution.get("temperature_interpolation_when_resized", "")) != (
        "bilinear; align_corners=false"
    ):
        raise ValueError("Temperature resolution contract does not pin bilinear interpolation")
    if str(resolution.get("support_interpolation_when_resized", "")) != "nearest":
        raise ValueError("Temperature resolution contract does not pin nearest support")
    try:
        declared_view_count = int(resolution.get("view_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Temperature resolution view count is invalid") from exc
    if declared_view_count != len(model_views):
        raise ValueError("Temperature resolution contract view count mismatch")
    try:
        declared_resized_count = int(resolution.get("resized_view_count", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Temperature resolution resized-view count is invalid") from exc
    if not 0 <= declared_resized_count <= len(model_views):
        raise ValueError("Temperature resolution resized-view count is out of range")
    derivation_resolution = derivation.get("resolution_policy")
    if not isinstance(derivation_resolution, Mapping) or dict(
        derivation_resolution
    ) != dict(resolution):
        raise ValueError("Temperature derivation/contract resolution policy mismatch")
    if str(target_provenance.get("domain", "")) != "float32 Celsius":
        raise ValueError("Temperature resolution target domain is not direct float32 Celsius")

    forward_receipt = derivation.get("source_target_forward_validation_receipt")
    if not isinstance(forward_receipt, Mapping):
        raise ValueError("Temperature resolution derivation lacks forward-validation receipt")

    if policy == TEMPERATURE_RESOLUTION_NATIVE_EXACT:
        if exact is not True or declared_resized_count != 0:
            raise ValueError(
                "native_exact temperature provenance must retain exact forward colorization"
            )
        if target_provenance.get("direct_float_temperature_resized") is not False:
            raise ValueError("native_exact temperature provenance cannot claim resizing")
        if str(
            target_provenance.get(
                "source_training_target_forward_colorization_status", ""
            )
        ) != "exact_on_valid_support":
            raise ValueError("native_exact temperature forward-validation status mismatch")
        if str(target_provenance.get("source", "")) != (
            "TSDK-referenced undistorted NPY"
        ):
            raise ValueError("native_exact temperature target is not direct TSDK float32")
        if (
            forward_receipt.get("exact") is not True
            or str(forward_receipt.get("status", "")) != "exact_on_valid_support"
        ):
            raise ValueError("native_exact derivation forward-validation receipt mismatch")
    else:
        if exact is not False or declared_resized_count <= 0:
            raise ValueError(
                "Explicit temperature resize must record forward exact=false and resized views"
            )
        if target_provenance.get("direct_float_temperature_resized") is not True:
            raise ValueError("Explicit temperature resize lacks direct-float resize provenance")
        if str(
            target_provenance.get(
                "source_training_target_forward_colorization_status", ""
            )
        ) != "not_applicable_after_direct_float_temperature_resize":
            raise ValueError("Explicit temperature resize forward-validation status mismatch")
        if str(target_provenance.get("source", "")) != (
            "image-domain bilinear resampling of direct float32 TSDK-referenced "
            "undistorted NPY"
        ):
            raise ValueError("Explicit temperature resize target is not direct float32 Celsius")
        if (
            forward_receipt.get("exact") is not False
            or str(forward_receipt.get("status", ""))
            != "not_applicable_after_direct_float_temperature_resize"
        ):
            raise ValueError("Explicit temperature resize derivation claim is incomplete")

    observed_resized_count = 0
    for name, model_view in model_views.items():
        contract_view = contract_views[name]
        model_receipt = model_view.get("temperature_resolution")
        contract_receipt = contract_view.get("temperature_resolution")
        if (
            not isinstance(model_receipt, Mapping)
            or not isinstance(contract_receipt, Mapping)
            or dict(model_receipt) != dict(contract_receipt)
        ):
            raise ValueError(
                f"Temperature resolution model/contract receipt mismatch for {name!r}"
            )
        receipt_policy = str(contract_receipt.get("policy", "")).strip()
        requested_policy = str(
            contract_receipt.get("requested_policy", "")
        ).strip()
        if receipt_policy != policy or requested_policy != policy:
            raise ValueError(f"Temperature resolution per-view policy mismatch for {name!r}")
        original_shape = _contract_shape_hw(
            contract_receipt.get("original_shape_hw"),
            label=f"{name} original temperature shape",
        )
        output_shape = _contract_shape_hw(
            contract_receipt.get("output_shape_hw"),
            label=f"{name} output temperature shape",
        )
        camera_shape = (int(model_view.get("height", -1)), int(model_view.get("width", -1)))
        if output_shape != camera_shape:
            raise ValueError(f"Temperature resolution/camera shape mismatch for {name!r}")
        resized = contract_receipt.get("resized")
        if not isinstance(resized, bool):
            raise ValueError(f"Temperature resolution resized flag is invalid for {name!r}")
        observed_resized_count += int(resized)
        if contract_receipt.get("png_or_palette_inverse_used_for_target") is not False:
            raise ValueError(f"Temperature target provenance is not direct Celsius for {name!r}")
        if str(contract_receipt.get("target_domain", "")) != "direct float32 Celsius":
            raise ValueError(f"Temperature target domain mismatch for {name!r}")
        expected_forward = not resized
        if (
            contract_receipt.get("direct_float_temperature_resized") is not resized
            or contract_receipt.get(
                "source_training_target_forward_colorization_exact_on_valid_support"
            )
            is not expected_forward
        ):
            raise ValueError(f"Temperature per-view forward provenance mismatch for {name!r}")
        expected_status = (
            "not_applicable_after_direct_float_temperature_resize"
            if resized
            else "exact_on_valid_support"
        )
        if str(
            contract_receipt.get(
                "source_training_target_forward_colorization_status", ""
            )
        ) != expected_status:
            raise ValueError(f"Temperature per-view forward status mismatch for {name!r}")
        if resized:
            if policy != TEMPERATURE_RESOLUTION_BILINEAR_NEAREST:
                raise ValueError("Only the explicit bilinear/nearest policy may resize")
            if original_shape == output_shape:
                raise ValueError(f"Resized temperature shape did not change for {name!r}")
            if (
                str(contract_receipt.get("temperature_interpolation", ""))
                != "bilinear"
                or str(contract_receipt.get("support_interpolation", ""))
                != "nearest"
                or contract_receipt.get("align_corners") is not False
                or str(contract_receipt.get("target_semantics", ""))
                != "image-domain bilinear resampling of the direct float32 "
                "TSDK-referenced undistorted temperature NPY"
            ):
                raise ValueError(f"Temperature resize recipe mismatch for {name!r}")
        else:
            if original_shape != output_shape:
                raise ValueError(f"Unresized temperature shape changed for {name!r}")
            if (
                str(contract_receipt.get("temperature_interpolation", "")) != "none"
                or str(contract_receipt.get("support_interpolation", "")) != "none"
                or contract_receipt.get("align_corners") is not None
                or str(contract_receipt.get("target_semantics", ""))
                != "direct float32 TSDK-referenced undistorted temperature NPY"
            ):
                raise ValueError(f"Native temperature recipe mismatch for {name!r}")
    if observed_resized_count != declared_resized_count:
        raise ValueError("Temperature resolution resized-view count does not match views")


def _temperature_contract(
    manifest_path: Path | None,
    *,
    scene_name: str,
    model_manifest_path: Path,
    model_manifest: Mapping[str, Any],
    model_views: Mapping[str, Mapping[str, Any]],
    temperature_fields_present: bool,
    formal_split_sha256: str,
    formal_split_counts: Mapping[str, int],
) -> Dict[str, Any] | None:
    if manifest_path is None:
        if temperature_fields_present:
            raise ValueError(
                "Temperature arrays are present but no explicit temperature-responsibility manifest was supplied"
            )
        return None
    manifest_path = manifest_path.resolve()
    payload = _load_json(manifest_path)
    if str(payload.get("protocol", "")) != TEMPERATURE_RESPONSIBILITY_PROTOCOL:
        raise ValueError("Unsupported temperature-responsibility manifest protocol")
    if _manifest_scene(payload, label="temperature responsibility manifest") != scene_name:
        raise ValueError("Temperature-responsibility scene mismatch")
    if str(payload.get("semantics", "")) != TEMPERATURE_SEMANTICS:
        raise ValueError("Temperature-responsibility semantics mismatch")
    if str(payload.get("units", "")).strip().lower() not in {"celsius", "degc", "degree_celsius"}:
        raise ValueError("Temperature-responsibility units must be Celsius")
    if str(payload.get("dtype", "")).strip().lower() != "float32":
        raise ValueError("Temperature-responsibility dtype must be float32")
    tsdk_sha = str(payload.get("tsdk_protocol_sha256", "")).lower()
    if not _is_sha256(tsdk_sha):
        raise ValueError("Temperature-responsibility TSDK protocol SHA-256 is invalid")
    expected_model_sha = str(payload.get("model_manifest_sha256", "")).lower()
    if expected_model_sha != _sha256(model_manifest_path):
        raise ValueError("Temperature-responsibility/model manifest SHA-256 mismatch")
    if str(payload.get("formal_split_manifest_sha256", "")).lower() != str(formal_split_sha256).lower():
        raise ValueError("Temperature-responsibility/current formal split SHA-256 mismatch")
    derivation = model_manifest.get("temperature_responsibility_derivation")
    if not isinstance(derivation, Mapping) or str(derivation.get("protocol", "")) != TEMPERATURE_RESPONSIBILITY_PROTOCOL:
        raise ValueError("Derived model manifest is missing the temperature-responsibility derivation contract")
    lineage_keys = (
        "source_model_manifest_sha256",
        "source_render_binding_manifest_sha256",
        "formal_split_manifest_sha256",
        "tsdk_binding_manifest_sha256",
        "tsdk_protocol_sha256",
        "temperature_manifest_sha256",
        "valid_support_manifest_sha256",
        "range_manifest_sha256",
        "canonical_manifest_sha256",
        "lut_sha256_uint8_rgb",
    )
    for key in lineage_keys:
        contract_value = str(payload.get(key, "")).strip().lower()
        derived_value = str(derivation.get(key, "")).strip().lower()
        if not _is_sha256(contract_value) or contract_value != derived_value:
            raise ValueError(f"Temperature-responsibility derivation/contract mismatch for {key}")
    if str(payload.get("lut_sha256_uint8_rgb", "")).lower() != lut_sha256(hot_iron_lut()):
        raise ValueError("Temperature-responsibility contract does not use the repository fixed Hot-Iron LUT")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or coverage.get("all_formal_views_exactly_once") is not True:
        raise ValueError("Temperature-responsibility contract has incomplete formal coverage")
    contract_counts = coverage.get("split_counts")
    if not isinstance(contract_counts, Mapping) or {
        split: int(contract_counts.get(split, -1)) for split in FORMAL_SPLIT_LABELS
    } != {split: int(formal_split_counts.get(split, -1)) for split in FORMAL_SPLIT_LABELS}:
        raise ValueError("Temperature-responsibility/current formal split counts mismatch")
    target_provenance = payload.get("target_provenance")
    if not isinstance(target_provenance, Mapping):
        raise ValueError("Temperature target provenance must be an object")
    if target_provenance.get("png_or_palette_inverse_used_for_target") is not False:
        raise ValueError("Temperature target provenance does not prove direct Celsius")
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        raise ValueError("Temperature-responsibility manifest is missing keys")
    required_keys = ("rendered", "target", "valid_mask")
    if any(not str(keys.get(key, "")).strip() for key in required_keys):
        raise ValueError("Temperature-responsibility manifest must define rendered/target/valid_mask keys")
    contract_views = payload.get("views")
    if not isinstance(contract_views, list):
        raise ValueError("Temperature-responsibility manifest views must be a list")
    bound: Dict[str, Dict[str, Any]] = {}
    for item in contract_views:
        if not isinstance(item, dict):
            raise ValueError("Temperature-responsibility view entries must be objects")
        name = str(item.get("image_name", "")).strip()
        if not name or name in bound:
            raise ValueError(f"Temperature-responsibility duplicate/missing image_name {name!r}")
        sha = str(item.get("npz_sha256", "")).lower()
        if not _is_sha256(sha):
            raise ValueError(f"Temperature-responsibility view {name!r} has invalid NPZ SHA-256")
        bound[name] = item
    if set(bound) != set(model_views):
        raise ValueError("Temperature-responsibility manifest must cover every model view exactly")
    _validate_temperature_resolution_contract(
        payload=payload,
        derivation=derivation,
        target_provenance=target_provenance,
        model_views=model_views,
        contract_views=bound,
    )
    for name, model_view in model_views.items():
        if str(bound[name]["npz_sha256"]).lower() != str(model_view.get("npz_sha256", "")).lower():
            raise ValueError(f"Temperature-responsibility NPZ binding mismatch for {name}")
        if str(bound[name].get("split", "")).strip().lower() != str(
            model_view.get("bound_split", model_view.get("split", ""))
        ).strip().lower():
            raise ValueError(f"Temperature-responsibility split binding mismatch for {name}")
        for key in ("temperature_target_sha256", "valid_support_sha256", "source_npz_sha256"):
            contract_sha = str(bound[name].get(key, "")).strip().lower()
            model_sha = str(model_view.get(key, "")).strip().lower()
            if not _is_sha256(contract_sha) or not _is_sha256(model_sha):
                raise ValueError(f"Temperature-responsibility view {name!r} is missing {key}")
            if contract_sha != model_sha:
                raise ValueError(
                    f"Temperature-responsibility view {name!r} {key} exact binding mismatch"
                )
    if not temperature_fields_present:
        raise ValueError("Temperature-responsibility manifest supplied but model arrays contain no temperature fields")
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "tsdk_protocol_sha256": tsdk_sha,
        "keys": {key: str(keys[key]) for key in required_keys},
        "formal_split_manifest_sha256": str(formal_split_sha256).lower(),
        "range_manifest_sha256": str(payload["range_manifest_sha256"]).lower(),
        "canonical_manifest_sha256": str(payload["canonical_manifest_sha256"]).lower(),
        "valid_support_manifest_sha256": str(payload["valid_support_manifest_sha256"]).lower(),
        "lut_sha256_uint8_rgb": str(payload["lut_sha256_uint8_rgb"]).lower(),
        "resolution_policy": (
            dict(payload["resolution_policy"])
            if isinstance(payload.get("resolution_policy"), Mapping)
            else {
                "requested_policy": TEMPERATURE_RESOLUTION_NATIVE_EXACT,
                "legacy_receipt_without_explicit_resolution_block": True,
            }
        ),
        "payload": payload,
    }


def _load_temperature_arrays(
    model_npz: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
    *,
    expected_shape: tuple[int, int],
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = contract["keys"]
    missing = [str(keys[name]) for name in ("rendered", "target", "valid_mask") if str(keys[name]) not in model_npz]
    if missing:
        raise KeyError(f"{label}: missing temperature responsibility arrays {missing}")
    rendered_raw = _as_hw(np.asarray(model_npz[str(keys["rendered"])]), label=f"{label} rendered temperature")
    target_raw = _as_hw(np.asarray(model_npz[str(keys["target"])]), label=f"{label} target temperature")
    valid_raw = _as_hw(np.asarray(model_npz[str(keys["valid_mask"])]), label=f"{label} temperature valid mask")
    if rendered_raw.shape != expected_shape or target_raw.shape != expected_shape or valid_raw.shape != expected_shape:
        raise ValueError(f"{label}: temperature arrays do not match the camera dimensions")
    if rendered_raw.dtype != np.float32 or target_raw.dtype != np.float32:
        raise TypeError(f"{label}: rendered/target Celsius arrays must be stored as float32")
    if valid_raw.dtype != np.bool_ and not np.issubdtype(valid_raw.dtype, np.integer):
        raise TypeError(f"{label}: temperature valid mask must be bool/integer")
    valid = valid_raw.astype(bool)
    if not np.all(np.isfinite(rendered_raw[valid])) or not np.all(np.isfinite(target_raw[valid])):
        raise ValueError(f"{label}: valid temperature pixels contain NaN/Inf")
    return rendered_raw, target_raw, valid


def evaluate(
    *,
    reference_manifest_path: Path,
    model_manifest_path: Path,
    out_dir: Path,
    thresholds_m: Sequence[float] = DEFAULT_THRESHOLDS_M,
    opacity_threshold: float = 0.5,
    depth_min: float = 1e-6,
    gaussian_count: int | None = None,
    scsp_indices_path: Path | None = None,
    clamp20_indices_path: Path | None = None,
    scsp_anchor_sha256: str = "",
    clamp20_anchor_sha256: str = "",
    probe_camera_manifest_path: Path | None = None,
    formal_split_manifest_path: Path | None = None,
    temperature_responsibility_manifest_path: Path | None = None,
    cdf_max_points: int = 4096,
    evaluation_mode: str = "formal",
    metric_only: bool = False,
) -> Dict[str, Any]:
    thresholds = _parse_thresholds(thresholds_m)
    evaluation_mode = str(evaluation_mode).strip().lower()
    if evaluation_mode not in {"formal", "legacy_custom"}:
        raise ValueError("evaluation_mode must be 'formal' or 'legacy_custom'")
    if evaluation_mode == "formal":
        if thresholds != DEFAULT_THRESHOLDS_M:
            raise ValueError(
                "Formal mode requires exactly thresholds_m="
                f"{list(DEFAULT_THRESHOLDS_M)}; custom thresholds are legacy_custom only"
            )
        if not math.isclose(
            float(opacity_threshold),
            FORMAL_OPACITY_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(
                f"Formal mode requires opacity_threshold={FORMAL_OPACITY_THRESHOLD}"
            )
        if not math.isclose(
            float(depth_min),
            FORMAL_DEPTH_MIN_M,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise ValueError(f"Formal mode requires depth_min={FORMAL_DEPTH_MIN_M}")
    if metric_only and any(
        value is not None
        for value in (
            scsp_indices_path,
            clamp20_indices_path,
            temperature_responsibility_manifest_path,
        )
    ):
        raise ValueError(
            "metric_only evaluation cannot request Gaussian responsibility sidecars"
        )
    if metric_only and any(
        str(value).strip() for value in (scsp_anchor_sha256, clamp20_anchor_sha256)
    ):
        raise ValueError("metric_only evaluation cannot bind Gaussian index anchors")
    if int(cdf_max_points) <= 0:
        raise ValueError("cdf_max_points must be positive")
    if probe_camera_manifest_path is None:
        raise ValueError("A probe camera manifest is required for formal evaluation")
    if formal_split_manifest_path is None:
        raise ValueError("A formal split manifest is required for formal evaluation")
    reference_manifest_path = reference_manifest_path.resolve()
    model_manifest_path = model_manifest_path.resolve()
    probe_camera_manifest_path = probe_camera_manifest_path.resolve()
    formal_split_manifest_path = formal_split_manifest_path.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to write formal evaluation into non-empty directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reference_manifest = _load_json(reference_manifest_path)
    model_manifest = _load_json(model_manifest_path)
    probe_manifest = _load_json(probe_camera_manifest_path)
    formal_split_manifest = _load_json(formal_split_manifest_path)
    formal_metric_contract = (
        _validate_formal_geometry_contract(
            model_manifest,
            metric_only=bool(metric_only),
        )
        if evaluation_mode == "formal"
        else None
    )
    formal_records = _formal_split_records_by_stem(formal_split_manifest)
    scene_names = {
        _manifest_scene(reference_manifest, label="reference manifest"),
        _manifest_scene(model_manifest, label="model manifest"),
        _manifest_scene(probe_manifest, label="probe camera manifest"),
        _manifest_scene(formal_split_manifest, label="formal split manifest"),
    }
    if len(scene_names) != 1:
        raise ValueError(f"Scene mismatch across formal inputs: {sorted(scene_names)}")
    scene_name = next(iter(scene_names))
    if str(reference_manifest.get("depth_semantics", "")) != REFERENCE_DEPTH_SEMANTICS:
        raise ValueError(f"Reference depth semantics must be {REFERENCE_DEPTH_SEMANTICS!r}")
    reference_mesh_sha = str(reference_manifest.get("reference_mesh_sha256", "")).lower()
    reference_mesh_backend = str(reference_manifest.get("reference_mesh_backend", "")).strip()
    if not _is_sha256(reference_mesh_sha) or not reference_mesh_backend.startswith("openmvs_"):
        raise ValueError("Reference manifest must bind a hashed OpenMVS reference mesh/backend")
    all_split_binding = reference_manifest.get("all_split_reference_binding")
    if not isinstance(all_split_binding, dict):
        raise ValueError(
            "Reference manifest must be an all-split extension with all_split_reference_binding; "
            "a test-only reference manifest is not sufficient for train/guard responsibility"
        )
    if str(all_split_binding.get("reference_mesh_sha256", "")).lower() != reference_mesh_sha:
        raise ValueError("All-split reference/base mesh SHA-256 mismatch")
    if str(all_split_binding.get("reference_mesh_backend", "")) != reference_mesh_backend:
        raise ValueError("All-split reference/base mesh backend mismatch")
    if not _is_sha256(all_split_binding.get("base_reference_manifest_sha256", "")):
        raise ValueError("All-split reference is missing its base reference manifest hash")
    base_reference_identity = reference_manifest.get("base_reference_manifest_identity")
    if not isinstance(base_reference_identity, Mapping):
        raise ValueError("All-split reference is missing its base reference manifest identity")
    base_reference_path = Path(str(base_reference_identity.get("path", ""))).resolve()
    base_reference_sha = str(base_reference_identity.get("sha256", "")).lower()
    try:
        base_reference_size = int(base_reference_identity.get("size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Base reference manifest identity has an invalid size") from exc
    if (
        not base_reference_path.is_file()
        or not _is_sha256(base_reference_sha)
        or base_reference_size < 0
        or _sha256(base_reference_path) != base_reference_sha
        or int(base_reference_path.stat().st_size) != base_reference_size
    ):
        raise ValueError("Base reference manifest file identity mismatch")
    if base_reference_sha != str(all_split_binding.get("base_reference_manifest_sha256", "")).lower():
        raise ValueError("Top-level base reference identity/all-split binding SHA-256 mismatch")
    if str(all_split_binding.get("extension_protocol", "")) != "fixed-openmvs-mesh-all-formal-splits-v1":
        raise ValueError("All-split reference extension protocol mismatch")
    if all_split_binding.get("mesh_or_backend_rebuilt") is not False:
        raise ValueError("All-split reference must explicitly declare mesh_or_backend_rebuilt=false")
    if set(all_split_binding.get("bound_split_labels", [])) != FORMAL_SPLIT_LABELS:
        raise ValueError("All-split reference must bind exactly train/guard/test")
    if str(all_split_binding.get("operator_pinned_base_reference_sha256", "")).lower() != str(
        all_split_binding.get("base_reference_manifest_sha256", "")
    ).lower():
        raise ValueError("All-split reference is missing the operator-pinned canonical base identity")
    if str(all_split_binding.get("operator_pinned_reference_mesh_sha256", "")).lower() != reference_mesh_sha:
        raise ValueError("All-split reference is missing the operator-pinned canonical mesh identity")
    lock_identity = reference_manifest.get("formal_reference_lock_identity")
    if not isinstance(lock_identity, Mapping):
        raise ValueError("All-split reference is missing an external formal reference lock identity")
    lock_path = Path(str(lock_identity.get("path", ""))).resolve()
    if not lock_path.is_file() or _sha256(lock_path) != str(lock_identity.get("sha256", "")).lower() or int(
        lock_path.stat().st_size
    ) != int(lock_identity.get("size_bytes", -1)):
        raise ValueError("Formal reference lock file identity mismatch")
    if str(all_split_binding.get("formal_reference_lock_sha256", "")).lower() != _sha256(lock_path):
        raise ValueError("All-split reference/formal reference lock SHA-256 mismatch")
    lock = _load_json(lock_path)
    if str(lock.get("protocol", "")) != "uav-tgs-formal-reference-lock-v1" or str(
        lock.get("status", "")
    ).lower() != "approved":
        raise ValueError("Formal reference lock is not an approved protocol receipt")
    expected_lock_values = {
        "scene_name": scene_name,
        "base_reference_manifest_sha256": str(all_split_binding["base_reference_manifest_sha256"]).lower(),
        "reference_mesh_sha256": reference_mesh_sha,
    }
    for key, expected in expected_lock_values.items():
        if str(lock.get(key, lock.get("scene", "") if key == "scene_name" else "")).strip().lower() != str(
            expected
        ).lower():
            raise ValueError(f"Formal reference lock mismatch for {key}")
    diagnostic_semantics = model_manifest.get("depth_diagnostics")
    if not isinstance(diagnostic_semantics, dict) or diagnostic_semantics.get("enabled") is not True:
        raise ValueError("Model manifest does not declare enabled depth diagnostics")
    required_diagnostic_semantics = (
        {
            array_key: DIAGNOSTIC_DEPTH_SEMANTICS[array_key]
            for array_key in DEPTH_DEFINITIONS.values()
        }
        if metric_only
        else DIAGNOSTIC_DEPTH_SEMANTICS
    )
    for key, expected in required_diagnostic_semantics.items():
        if str(diagnostic_semantics.get(key, "")) != expected:
            raise ValueError(f"Model diagnostic semantics mismatch for {key}")
    appearance_modality = str(model_manifest.get("appearance_modality", "")).strip().lower()
    if appearance_modality not in {"rgb", "thermal_canonical", "none"}:
        raise ValueError("Model manifest must explicitly declare appearance_modality=rgb/thermal_canonical/none")

    probe_sha = _sha256(probe_camera_manifest_path)
    split_sha = _sha256(formal_split_manifest_path)
    if str(lock.get("formal_split_manifest_sha256", "")).lower() != split_sha:
        raise ValueError("Formal reference lock/current formal split SHA-256 mismatch")
    if str(probe_manifest.get("camera_manifest_type", "")) != "formal_all_split_probe_camera_manifest_v1":
        raise ValueError("Probe camera manifest is not a formal all-split binding")
    if _identity_sha(probe_manifest, "bound_split_manifest_identity", label="probe") != split_sha:
        raise ValueError("Probe/formal split manifest hash mismatch")
    if str(all_split_binding.get("probe_camera_manifest_sha256", "")).lower() != probe_sha:
        raise ValueError("All-split reference/probe manifest hash mismatch")
    if str(all_split_binding.get("formal_split_manifest_sha256", "")).lower() != split_sha:
        raise ValueError("All-split reference/formal split manifest hash mismatch")
    for manifest, label in ((reference_manifest, "reference"), (model_manifest, "model")):
        if _identity_sha(manifest, "probe_camera_manifest_identity", label=label) != probe_sha:
            raise ValueError(f"{label}/probe camera manifest hash mismatch")
        if _identity_sha(manifest, "formal_split_manifest_identity", label=label) != split_sha:
            raise ValueError(f"{label}/formal split manifest hash mismatch")
    probe_model_camera_identity = probe_manifest.get("model_cameras_json_identity")
    model_native_camera_identity = model_manifest.get("native_cameras_json_identity")
    if not isinstance(probe_model_camera_identity, dict) or not isinstance(model_native_camera_identity, dict):
        raise ValueError("Probe/model manifests must bind model cameras.json identities")
    for key in ("sha256", "size_bytes"):
        if str(probe_model_camera_identity.get(key, "")) != str(model_native_camera_identity.get(key, "")):
            raise ValueError("Model diagnostic render is not bound to the probe-approved cameras.json")
    native_camera_path = Path(str(model_manifest.get("native_cameras_json", ""))).resolve()
    if not native_camera_path.is_file():
        raise FileNotFoundError(f"Model-bound native cameras.json is missing: {native_camera_path}")
    if _sha256(native_camera_path) != str(model_native_camera_identity.get("sha256", "")).lower() or int(
        native_camera_path.stat().st_size
    ) != int(model_native_camera_identity.get("size_bytes", -1)):
        raise ValueError("Model-bound native cameras.json on disk has changed")
    native_cameras = _native_camera_map(native_camera_path)
    alignment = model_manifest.get("strict_to_native_alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("Model manifest is missing strict-to-native camera alignment")
    strict_to_native = np.asarray(alignment.get("strict_to_native_transform"), dtype=np.float64)
    if strict_to_native.shape != (4, 4) or not np.all(np.isfinite(strict_to_native)):
        raise ValueError("Strict-to-native camera alignment transform must be finite 4x4")
    if not np.allclose(strict_to_native[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-12):
        raise ValueError("Strict-to-native camera alignment is not an affine rigid transform")
    rotation = strict_to_native[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-8) or not np.isclose(
        np.linalg.det(rotation), 1.0, rtol=0.0, atol=1e-8
    ):
        raise ValueError("Strict-to-native camera alignment rotation is not proper orthonormal")
    alignment_validation = alignment.get("revalidated_against_bound_native_cameras")
    if not isinstance(alignment_validation, Mapping) or alignment_validation.get("status") != "passed":
        raise ValueError("Strict-to-native camera alignment lacks a passed native-camera revalidation")
    if float(alignment_validation.get("maximum_allowed_translation_error_m", np.nan)) != ALIGNMENT_MAX_TRANSLATION_ERROR_M:
        raise ValueError("Strict/native translation integrity limit mismatch")
    if float(alignment_validation.get("maximum_allowed_rotation_error_deg", np.nan)) != ALIGNMENT_MAX_ROTATION_ERROR_DEG:
        raise ValueError("Strict/native rotation integrity limit mismatch")

    ref_by_name = _view_map(reference_manifest, label="reference manifest")
    model_by_name = _view_map(model_manifest, label="model manifest")
    probe_by_name = _view_map(probe_manifest, label="probe camera manifest")
    if set(ref_by_name) != set(model_by_name) or set(ref_by_name) != set(probe_by_name):
        raise ValueError(
            "Reference/model/probe view sets must match exactly; "
            f"reference={len(ref_by_name)} model={len(model_by_name)} probe={len(probe_by_name)}"
        )
    computed_camera_set_sha = _camera_set_sha256(probe_by_name)
    if str(probe_manifest.get("camera_set_sha256", "")).lower() != computed_camera_set_sha:
        raise ValueError("Probe camera_set_sha256 mismatch")
    for manifest, label in ((reference_manifest, "reference"), (model_manifest, "model")):
        if str(manifest.get("camera_set_sha256", "")).lower() != computed_camera_set_sha:
            raise ValueError(f"{label} camera_set_sha256 mismatch")
    computed_render_camera_set_sha = _render_camera_set_sha256(model_by_name)
    if str(model_manifest.get("render_camera_set_sha256", "")).lower() != computed_render_camera_set_sha:
        raise ValueError("Model render-camera set SHA-256 mismatch")
    coverage = model_manifest.get("native_camera_coverage")
    if not isinstance(coverage, dict) or coverage.get("probe_bound_model_cameras_json_identity_verified") is not True:
        raise ValueError("Model manifest does not prove the probe-bound native camera identity")
    if str(coverage.get("render_camera_set_sha256", "")).lower() != computed_render_camera_set_sha:
        raise ValueError("Native-camera coverage/render-camera set hash mismatch")

    declared_count = 0
    model_index_anchor_sha = ""
    if metric_only:
        if gaussian_count is not None:
            raise ValueError("metric_only evaluation does not accept gaussian_count")
    else:
        manifest_count = model_manifest.get("gaussian_count")
        if manifest_count is None or int(manifest_count) <= 0:
            raise ValueError("Formal model manifest must declare a positive gaussian_count")
        declared_count = int(manifest_count)
        if gaussian_count is not None and int(gaussian_count) != declared_count:
            raise ValueError("CLI/model gaussian_count mismatch")
        index_anchor = model_manifest.get("gaussian_index_anchor")
        if not isinstance(index_anchor, dict) or not _is_sha256(index_anchor.get("sha256", "")):
            raise ValueError("Model manifest must declare gaussian_index_anchor.sha256")
        model_index_anchor_sha = str(index_anchor["sha256"]).lower()
        index_binding = model_manifest.get("gaussian_index_binding")
        if not isinstance(index_binding, dict) or index_binding.get("status") != "verified":
            raise ValueError("Model manifest must include a verified Gaussian index-space binding")
        if int(index_binding.get("gaussian_count", -1)) != declared_count:
            raise ValueError("Gaussian index-space binding count mismatch")
        bound_anchor_identity = index_binding.get("gaussian_index_anchor")
        rendered_ply_identity = index_binding.get("rendered_model_point_cloud")
        model_ply_identity = model_manifest.get("model_point_cloud")
        if not all(
            isinstance(value, dict)
            for value in (
                bound_anchor_identity,
                rendered_ply_identity,
                model_ply_identity,
            )
        ):
            raise ValueError("Gaussian index-space binding is missing artifact identities")
        if str(bound_anchor_identity.get("sha256", "")).lower() != model_index_anchor_sha:
            raise ValueError("Gaussian index-space binding anchor mismatch")
        if str(rendered_ply_identity.get("sha256", "")).lower() != str(
            model_ply_identity.get("sha256", "")
        ).lower():
            raise ValueError("Gaussian index-space binding rendered PLY mismatch")
        proof = str(index_binding.get("proof", ""))
        if proof not in {
            "identical_ply_sha256",
            "exact_ordered_xyz_sequence",
            "fixed_topology_invariant_audit_receipt",
        }:
            raise ValueError("Unsupported Gaussian index-space proof")
        if proof == "exact_ordered_xyz_sequence":
            rendered_xyz = index_binding.get("rendered_ordered_xyz")
            anchor_xyz = index_binding.get("anchor_ordered_xyz")
            if (
                not isinstance(rendered_xyz, dict)
                or not isinstance(anchor_xyz, dict)
                or str(rendered_xyz.get("sequence_sha256", "")).lower()
                != str(anchor_xyz.get("sequence_sha256", "")).lower()
            ):
                raise ValueError("Ordered-XYZ Gaussian index proof is inconsistent")
        if proof == "fixed_topology_invariant_audit_receipt":
            receipt = index_binding.get("binding_receipt_identity")
            if not isinstance(receipt, dict) or not _is_sha256(receipt.get("sha256", "")):
                raise ValueError("Fixed-topology Gaussian index proof is missing its receipt identity")
            receipt_path = Path(str(receipt.get("path", ""))).resolve()
            if (
                not receipt_path.is_file()
                or _sha256(receipt_path) != str(receipt["sha256"]).lower()
                or int(receipt_path.stat().st_size)
                != int(receipt.get("size_bytes", -1))
            ):
                raise ValueError("Fixed-topology Gaussian index receipt identity mismatch")

    cached_paths: Dict[str, tuple[Path, Path, Dict[str, str], tuple[int, int]]] = {}
    temperature_field_names = {
        "render_temperature_c",
        "temperature_render_c",
        "rendered_temperature_c",
        "target_temperature_c",
        "temperature_target_c",
        "ground_truth_temperature_c",
        "temperature_valid_mask",
    }
    temperature_fields_present = False
    matched_formal_records: set[int] = set()
    formal_record_count = len(formal_split_manifest.get("records", []))
    expected_direct_native_stems = {
        _normalized_stem(name)
        for name, view in model_by_name.items()
        if str(view.get("bound_split", view.get("split", ""))).strip().lower() in {"train", "test"}
    }
    if set(native_cameras) != expected_direct_native_stems:
        raise ValueError(
            "Native cameras.json must be exactly the formal train+test set; "
            f"missing={sorted(expected_direct_native_stems - set(native_cameras))[:8]} "
            f"extra={sorted(set(native_cameras) - expected_direct_native_stems)[:8]}"
        )
    observed_alignment_center_errors: List[float] = []
    observed_alignment_rotation_errors: List[float] = []
    for image_name in sorted(ref_by_name):
        ref_view = ref_by_name[image_name]
        model_view = model_by_name[image_name]
        expected_shape = _validate_camera_equivalence(image_name, ref_view, model_view, probe_by_name[image_name])
        if not np.isclose(float(model_view["cx"]), float(model_view["width"]) / 2.0, rtol=0.0, atol=PRINCIPAL_POINT_CENTER_TOLERANCE_PX) or not np.isclose(
            float(model_view["cy"]),
            float(model_view["height"]) / 2.0,
            rtol=0.0,
            atol=PRINCIPAL_POINT_CENTER_TOLERANCE_PX,
        ):
            raise ValueError(f"{image_name}: repository camera path requires a centered principal point")
        expected_camera_sha = _camera_sha256(probe_by_name[image_name])
        for view, label in ((ref_view, "reference"), (model_view, "model")):
            if str(view.get("camera_sha256", "")).lower() != expected_camera_sha:
                raise ValueError(f"{image_name}: {label} camera_sha256 mismatch")
        expected_render_camera_sha = _camera_sha256_for_matrix(
            model_view,
            model_view.get("native_camera_to_world"),
        )
        if str(model_view.get("render_camera_sha256", "")).lower() != expected_render_camera_sha:
            raise ValueError(f"{image_name}: model render_camera_sha256 mismatch")
        strict_c2w = np.asarray(model_view["camera_to_world"], dtype=np.float64)
        render_c2w = np.asarray(model_view["native_camera_to_world"], dtype=np.float64)
        transformed_c2w = strict_to_native @ strict_c2w
        if not np.allclose(render_c2w, transformed_c2w, rtol=0.0, atol=1e-11):
            raise ValueError(f"{image_name}: stored render camera is not strict_to_native @ strict camera")
        native_entry = native_cameras.get(_normalized_stem(image_name))
        if native_entry is None:
            if str(model_view.get("bound_split", "")).lower() != "guard":
                raise ValueError(f"{image_name}: only formal guard cameras may use alignment extrapolation")
            if model_view.get("render_camera_is_alignment_extrapolation") is not True:
                raise ValueError(f"{image_name}: missing guard-camera alignment-extrapolation marker")
            if str(model_view.get("bound_native_camera_sha256", "")):
                raise ValueError(f"{image_name}: guard camera unexpectedly claims a direct native-camera hash")
        else:
            if model_view.get("render_camera_is_alignment_extrapolation") is not False:
                raise ValueError(f"{image_name}: train/test camera must be directly checked against cameras.json")
            direct_native = _native_camera_matrix(native_entry)
            direct_sha = _camera_sha256_for_matrix(model_view, direct_native)
            if str(model_view.get("bound_native_camera_sha256", "")).lower() != direct_sha:
                raise ValueError(f"{image_name}: bound native-camera hash mismatch")
            center_error = float(np.linalg.norm(render_c2w[:3, 3] - direct_native[:3, 3]))
            rotation_error = _rotation_error_deg(render_c2w[:3, :3], direct_native[:3, :3])
            observed_alignment_center_errors.append(center_error)
            observed_alignment_rotation_errors.append(rotation_error)
            if center_error > ALIGNMENT_MAX_TRANSLATION_ERROR_M or rotation_error > ALIGNMENT_MAX_ROTATION_ERROR_DEG:
                raise ValueError(f"{image_name}: strict/native camera alignment residual exceeds integrity limits")
            if not np.isclose(
                center_error,
                float(model_view.get("alignment_center_error_m", np.nan)),
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                rotation_error,
                float(model_view.get("alignment_rotation_error_deg", np.nan)),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(f"{image_name}: stored strict/native residual evidence mismatch")
        ref_path = _resolve_view_npz(reference_manifest_path, ref_view)
        model_path = _resolve_view_npz(model_manifest_path, model_view)
        supplemental_view = formal_records.get(_normalized_stem(image_name))
        if supplemental_view is None:
            raise KeyError(f"View {image_name!r} is absent from the supplied formal split manifest")
        matched_formal_records.add(id(supplemental_view))
        bound_split = str(supplemental_view.get("split", "")).strip().lower()
        if bound_split not in FORMAL_SPLIT_LABELS:
            raise ValueError(f"{image_name}: invalid formal split label {bound_split!r}")
        if str(ref_view.get("bound_split", "")).strip().lower() != bound_split:
            raise ValueError(f"{image_name}: reference bound_split mismatch")
        if str(model_view.get("bound_split", "")).strip().lower() != bound_split:
            raise ValueError(f"{image_name}: model bound_split mismatch")
        if str(probe_by_name[image_name].get("bound_split", "")).strip().lower() != bound_split:
            raise ValueError(f"{image_name}: probe bound_split mismatch")
        if str(probe_by_name[image_name].get("camera_sha256", "")).lower() != expected_camera_sha:
            raise ValueError(f"{image_name}: probe camera_sha256 mismatch")
        cached_paths[image_name] = (
            ref_path,
            model_path,
            _view_metadata(ref_view, model_view, model_manifest, supplemental_view=supplemental_view),
            expected_shape,
        )
        with np.load(ref_path, allow_pickle=False) as ref_npz, np.load(model_path, allow_pickle=False) as model_npz:
            for key in ("depth", "valid_mask"):
                if key not in ref_npz:
                    raise KeyError(f"{ref_path} is missing {key!r}")
            reference_depth = _as_hw(np.asarray(ref_npz["depth"]), label=f"{image_name} reference depth")
            reference_valid = _as_hw(np.asarray(ref_npz["valid_mask"]), label=f"{image_name} reference valid")
            if reference_depth.shape != expected_shape or reference_valid.shape != expected_shape:
                raise ValueError(f"{image_name}: reference arrays/camera dimensions mismatch")
            required_model_keys = list(DEPTH_DEFINITIONS.values())
            if not metric_only:
                required_model_keys.extend(
                    (
                        "accumulated_opacity",
                        "top_contributor_index",
                        "top_contributor_weight",
                    )
                )
            for key in required_model_keys:
                if key not in model_npz:
                    raise KeyError(f"{model_path} is missing diagnostic key {key!r}")
            shape_keys = list(DEPTH_DEFINITIONS.values())
            if not metric_only:
                shape_keys.append("accumulated_opacity")
            for key in shape_keys:
                if _as_hw(np.asarray(model_npz[key]), label=f"{image_name} {key}").shape != expected_shape:
                    raise ValueError(f"{image_name}: {key}/camera dimensions mismatch")
            if not metric_only:
                _validate_joint_diagnostic_arrays(
                    model_npz,
                    expected_shape=expected_shape,
                    gaussian_count=declared_count,
                    label=image_name,
                )
            temperature_fields_present |= bool(set(model_npz.files) & temperature_field_names)
            has_rgb_arrays = "render_rgb" in model_npz or "target_rgb" in model_npz
            if appearance_modality != "rgb" and has_rgb_arrays:
                raise ValueError(
                    f"{image_name}: non-RGB bundle contains render_rgb/target_rgb arrays and could masquerade as RGB"
                )

    if len(ref_by_name) != formal_record_count or len(matched_formal_records) != formal_record_count:
        raise ValueError(
            "Reference/model/probe views must cover every formal split record exactly once; "
            f"views={len(ref_by_name)} records={formal_record_count} matched_records={len(matched_formal_records)}"
        )
    if int(alignment_validation.get("count", -1)) != len(observed_alignment_center_errors):
        raise ValueError("Strict/native alignment revalidation count mismatch")
    alignment_observed = {
        "translation_error_mean_m": float(np.mean(observed_alignment_center_errors)),
        "translation_error_max_m": float(np.max(observed_alignment_center_errors)),
        "rotation_error_mean_deg": float(np.mean(observed_alignment_rotation_errors)),
        "rotation_error_max_deg": float(np.max(observed_alignment_rotation_errors)),
    }
    for key, value in alignment_observed.items():
        if not np.isclose(value, float(alignment_validation.get(key, np.nan)), rtol=0.0, atol=1e-10):
            raise ValueError(f"Strict/native aggregate alignment evidence mismatch for {key}")

    temperature_fields_present = temperature_fields_present or temperature_responsibility_manifest_path is not None
    temperature_contract = _temperature_contract(
        temperature_responsibility_manifest_path,
        scene_name=scene_name,
        model_manifest_path=model_manifest_path,
        model_manifest=model_manifest,
        model_views=model_by_name,
        temperature_fields_present=temperature_fields_present,
        formal_split_sha256=split_sha,
        formal_split_counts={
            split: sum(
                1
                for record in formal_split_manifest.get("records", [])
                if isinstance(record, dict) and str(record.get("split", "")).strip().lower() == split
            )
            for split in FORMAL_SPLIT_LABELS
        },
    )
    if metric_only:
        scsp_set = BoundIndexSet("SCSP set", frozenset(), "", "", "")
        clamp20_set = BoundIndexSet("clamp20 set", frozenset(), "", "", "")
    else:
        scsp_set = _load_bound_index_set(
            scsp_indices_path,
            label="SCSP set",
            explicit_anchor_sha256=scsp_anchor_sha256,
            model_index_anchor_sha256=model_index_anchor_sha,
            gaussian_count=declared_count,
        )
        clamp20_set = _load_bound_index_set(
            clamp20_indices_path,
            label="clamp20 set",
            explicit_anchor_sha256=clamp20_anchor_sha256,
            model_index_anchor_sha256=model_index_anchor_sha,
            gaussian_count=declared_count,
        )

    depth_payload: Dict[str, Any] = {}
    depth_csv_rows: List[List[Any]] = []
    group_csv_rows: List[List[Any]] = []
    for depth_label, depth_key in DEPTH_DEFINITIONS.items():
        view_stats: List[ViewDepthStats] = []
        no_opacity_stats: List[ViewDepthStats] = []
        opacity_bin_stats: Dict[str, List[ViewDepthStats]] = {label: [] for label, _, _ in OPACITY_BINS}

        def append_stats(target: List[ViewDepthStats], image_name: str, metadata: Dict[str, str], metrics: Mapping[str, Any]) -> None:
            target.append(
                ViewDepthStats(
                    image_name=image_name,
                    metadata=metadata,
                    reference_count=int(metrics["reference_count"]),
                    valid_count=int(metrics["valid_count"]),
                    missing_count=int(metrics["missing_count"]),
                    signed_residual=np.asarray(metrics["signed_residual"]),
                    front_counts=dict(metrics["front_counts"]),
                    agreement_counts=dict(metrics["agreement_counts"]),
                    behind_counts=dict(metrics["behind_counts"]),
                )
            )

        for image_name, (ref_path, model_path, metadata, _expected_shape) in cached_paths.items():
            with np.load(ref_path, allow_pickle=False) as ref_npz, np.load(model_path, allow_pickle=False) as model_npz:
                reference_depth = _as_hw(np.asarray(ref_npz["depth"], dtype=np.float64), label="reference depth")
                reference_valid = _as_hw(np.asarray(ref_npz["valid_mask"]), label="reference valid").astype(bool)
                opacity = (
                    None
                    if metric_only
                    else _as_hw(
                        np.asarray(model_npz["accumulated_opacity"], dtype=np.float64),
                        label="opacity",
                    )
                )
                rendered_depth = _as_hw(np.asarray(model_npz[depth_key], dtype=np.float64), label=depth_key)
                metrics = compute_depth_metrics(
                    reference_depth,
                    rendered_depth,
                    reference_valid=reference_valid,
                    accumulated_opacity=opacity,
                    opacity_threshold=opacity_threshold,
                    depth_min=depth_min,
                    thresholds_m=thresholds,
                )
                append_stats(view_stats, image_name, metadata, metrics)
                if not metric_only:
                    append_stats(
                        no_opacity_stats,
                        image_name,
                        metadata,
                        compute_depth_metrics(
                            reference_depth,
                            rendered_depth,
                            reference_valid=reference_valid,
                            depth_min=depth_min,
                            thresholds_m=thresholds,
                        ),
                    )
                    assert opacity is not None
                    for bin_label, low, high in OPACITY_BINS:
                        bin_mask = np.isfinite(opacity) & (opacity >= low) & (opacity < high)
                        append_stats(
                            opacity_bin_stats[bin_label],
                            image_name,
                            metadata,
                            compute_depth_metrics(
                                reference_depth,
                                rendered_depth,
                                reference_valid=reference_valid & bin_mask,
                                depth_min=depth_min,
                                thresholds_m=thresholds,
                            ),
                        )
        overall = _summarize_view_stats(
            view_stats,
            thresholds,
            evaluation_mode=evaluation_mode,
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        cdf_outputs: Dict[str, Any] = {"groups": {}}
        for group_type in ("split", "block", "orientation"):
            labels = sorted({item.metadata[group_type] for item in view_stats})
            grouped[group_type] = {
                label: _summarize_view_stats(
                    [item for item in view_stats if item.metadata[group_type] == label],
                    thresholds,
                    evaluation_mode=evaluation_mode,
                )
                for label in labels
            }
            cdf_outputs["groups"][group_type] = {}
            for label in labels:
                relative = Path("signed_residual_cdf") / depth_label / group_type / f"{_safe_group_label(label)}.csv"
                selected = [item.signed_residual for item in view_stats if item.metadata[group_type] == label]
                _write_cdf(out_dir / relative, selected, cdf_max_points)
                cdf_outputs["groups"][group_type][label] = str(relative).replace("\\", "/")
        overall_cdf = Path("signed_residual_cdf") / depth_label / "overall.csv"
        _write_cdf(out_dir / overall_cdf, [item.signed_residual for item in view_stats], cdf_max_points)
        cdf_outputs["overall"] = str(overall_cdf).replace("\\", "/")
        if metric_only:
            opacity_sensitivity = {
                "status": "not_applicable_metric_only",
                "support_mode": "finite_positive_depth_metric_only",
            }
        else:
            opacity_sensitivity = {
                "diagnostic_only": True,
                "main_opacity_threshold": float(opacity_threshold),
                "no_opacity_mask": _summarize_view_stats(
                    no_opacity_stats,
                    thresholds,
                    evaluation_mode=evaluation_mode,
                ),
                "opacity_bins": {},
            }
            main_reference_count = int(overall["reference_valid_pixels"])
            for bin_label, low, high in OPACITY_BINS:
                summary = _summarize_view_stats(
                    opacity_bin_stats[bin_label],
                    thresholds,
                    evaluation_mode=evaluation_mode,
                )
                summary["opacity_min_inclusive"] = low
                summary["opacity_max_exclusive"] = high
                summary["reference_pixel_share"] = (
                    float(summary["reference_valid_pixels"])
                    / float(main_reference_count)
                    if main_reference_count > 0
                    else 0.0
                )
                opacity_sensitivity["opacity_bins"][bin_label] = summary
        depth_payload[depth_label] = {
            "npz_key": depth_key,
            "overall": overall,
            "groups": grouped,
            "signed_residual_cdf": cdf_outputs,
            "opacity_sensitivity": opacity_sensitivity,
        }
        for metric in overall["threshold_metrics"]:
            depth_csv_rows.append(
                [
                    depth_label,
                    metric["threshold_m"],
                    metric["front_rate"],
                    metric["agreement_rate"],
                    overall["missing_rate"],
                    overall["mean_abs_error_m"],
                    overall["median_abs_error_m"],
                    overall["signed_bias_m"],
                    metric["behind_rate"],
                    overall["front_auc_log_0p25_20m"],
                    (
                        overall["front_curve_auc_linear_legacy"]["normalized"]
                        if overall.get("front_curve_auc_linear_legacy") is not None
                        else None
                    ),
                    (
                        overall["agreement_curve_auc_linear_legacy"]["normalized"]
                        if overall.get("agreement_curve_auc_linear_legacy") is not None
                        else None
                    ),
                ]
            )
        for group_type, groups in grouped.items():
            for group_label, summary in groups.items():
                for metric in summary["threshold_metrics"]:
                    group_csv_rows.append(
                        [
                            depth_label,
                            group_type,
                            group_label,
                            metric["threshold_m"],
                            metric["front_rate"],
                            metric["agreement_rate"],
                            summary["missing_rate"],
                            summary["mean_abs_error_m"],
                            summary["median_abs_error_m"],
                            summary["signed_bias_m"],
                            metric["behind_rate"],
                            summary["front_auc_log_0p25_20m"],
                            (
                                summary["front_curve_auc_linear_legacy"]["normalized"]
                                if summary.get("front_curve_auc_linear_legacy") is not None
                                else None
                            ),
                            (
                                summary["agreement_curve_auc_linear_legacy"]["normalized"]
                                if summary.get("agreement_curve_auc_linear_legacy") is not None
                                else None
                            ),
                        ]
                    )
    masses: Dict[str, Dict[str, np.ndarray]] = {}
    view_mass: Dict[str, Dict[str, MutableMapping[str, float]]] = {}
    block_mass: Dict[str, Dict[str, MutableMapping[str, float]]] = {}
    assignment_coverage: Dict[str, Dict[str, MutableMapping[str, Any]]] = {}
    view_assignment_coverage: Dict[str, Dict[str, Dict[str, MutableMapping[str, Any]]]] = {}
    block_assignment_coverage: Dict[str, Dict[str, Dict[str, MutableMapping[str, Any]]]] = {}
    split_view_counts: Dict[str, int] = {}
    evaluated_splits = {metadata["split"].strip().lower() for _, _, metadata, _ in cached_paths.values()}
    if evaluated_splits != FORMAL_SPLIT_LABELS:
        raise ValueError(
            "Formal metric evaluation requires actual train, guard, and test reference/model views; "
            f"observed={sorted(evaluated_splits)}. A test-only reference must first be extended from the same bound mesh."
        )
    responsibility_items = () if metric_only else cached_paths.items()
    for image_name, (ref_path, model_path, metadata, expected_shape) in responsibility_items:
        split = metadata["split"].strip().lower()
        masses.setdefault(split, {})
        view_mass.setdefault(split, {})
        block_mass.setdefault(split, {})
        assignment_coverage.setdefault(split, {})
        view_assignment_coverage.setdefault(split, {})
        block_assignment_coverage.setdefault(split, {})
        split_view_counts[split] = split_view_counts.get(split, 0) + 1
        with np.load(ref_path, allow_pickle=False) as ref_npz, np.load(model_path, allow_pickle=False) as model_npz:
            top_index, _top_weight = _validate_joint_diagnostic_arrays(
                model_npz,
                expected_shape=expected_shape,
                gaussian_count=declared_count,
                label=image_name,
            )
            opacity = _as_hw(np.asarray(model_npz["accumulated_opacity"], dtype=np.float64), label="opacity")
            reference_depth = _as_hw(np.asarray(ref_npz["depth"], dtype=np.float64), label="reference depth")
            reference_valid = _as_hw(np.asarray(ref_npz["valid_mask"]), label="reference valid").astype(bool)

            # A top-contributor index describes only max-contribution Gaussian-center
            # depth. Expected/median depth curves are reported above without
            # fabricating per-Gaussian attribution for those estimators.
            rendered_max_depth = _as_hw(
                np.asarray(model_npz[DEPTH_DEFINITIONS["max_contribution"]], dtype=np.float64),
                label="max-contribution depth",
            )
            model_valid = (
                np.isfinite(rendered_max_depth)
                & (rendered_max_depth > depth_min)
                & np.isfinite(opacity)
                & (opacity >= opacity_threshold)
            )
            front_mass = np.where(
                reference_valid & model_valid,
                np.maximum(reference_depth - rendered_max_depth, 0.0),
                0.0,
            )
            signal = "front_max_contribution"
            masses[split].setdefault(signal, np.zeros((declared_count,), dtype=np.float64))
            _scatter_add(masses[split][signal], top_index, front_mass)
            coverage_update = _assignment_coverage(top_index, front_mass, declared_count)
            _accumulate_assignment_coverage(
                assignment_coverage[split].setdefault(signal, {}),
                coverage_update,
            )
            view_assignment_coverage[split].setdefault(signal, {})[image_name] = dict(coverage_update)
            _accumulate_assignment_coverage(
                block_assignment_coverage[split].setdefault(signal, {}).setdefault(metadata["block"], {}),
                coverage_update,
            )
            total_signal = float(coverage_update["assigned_mass"])
            view_mass[split].setdefault(signal, {})[image_name] = total_signal
            block_mass[split].setdefault(signal, {})[metadata["block"]] = (
                block_mass[split].setdefault(signal, {}).get(metadata["block"], 0.0) + total_signal
            )

            if appearance_modality == "rgb":
                rendered_rgb = _first_present(model_npz, ("render_rgb",))
                target_rgb = _first_present(model_npz, ("target_rgb",))
                if rendered_rgb is None or target_rgb is None:
                    raise KeyError(f"{model_path}: RGB bundle requires render_rgb and target_rgb")
                rgb_mass = _image_abs_residual(rendered_rgb, target_rgb)
                if rgb_mass.shape != expected_shape:
                    raise ValueError(f"{image_name}: RGB residual/camera dimensions mismatch")
                masses[split].setdefault("rgb_abs_top1_occupancy_approx", np.zeros((declared_count,), dtype=np.float64))
                _scatter_add(masses[split]["rgb_abs_top1_occupancy_approx"], top_index, rgb_mass)
                coverage_update = _assignment_coverage(top_index, rgb_mass, declared_count)
                _accumulate_assignment_coverage(
                    assignment_coverage[split].setdefault("rgb_abs_top1_occupancy_approx", {}),
                    coverage_update,
                )
                view_assignment_coverage[split].setdefault("rgb_abs_top1_occupancy_approx", {})[
                    image_name
                ] = dict(coverage_update)
                _accumulate_assignment_coverage(
                    block_assignment_coverage[split]
                    .setdefault("rgb_abs_top1_occupancy_approx", {})
                    .setdefault(metadata["block"], {}),
                    coverage_update,
                )
                total_signal = float(coverage_update["assigned_mass"])
                view_mass[split].setdefault("rgb_abs_top1_occupancy_approx", {})[image_name] = total_signal
                block_mass[split].setdefault("rgb_abs_top1_occupancy_approx", {})[metadata["block"]] = (
                    block_mass[split].setdefault("rgb_abs_top1_occupancy_approx", {}).get(metadata["block"], 0.0)
                    + total_signal
                )

            if temperature_contract is not None:
                rendered_temperature, target_temperature, temperature_valid = _load_temperature_arrays(
                    model_npz,
                    temperature_contract,
                    expected_shape=expected_shape,
                    label=image_name,
                )
                temperature_mass = np.where(
                    temperature_valid,
                    np.abs(rendered_temperature.astype(np.float64) - target_temperature.astype(np.float64)),
                    0.0,
                )
                temp_signal = "temperature_abs_top1_occupancy_approx"
                masses[split].setdefault(temp_signal, np.zeros((declared_count,), dtype=np.float64))
                _scatter_add(masses[split][temp_signal], top_index, temperature_mass)
                coverage_update = _assignment_coverage(top_index, temperature_mass, declared_count)
                _accumulate_assignment_coverage(
                    assignment_coverage[split].setdefault(temp_signal, {}),
                    coverage_update,
                )
                view_assignment_coverage[split].setdefault(temp_signal, {})[image_name] = dict(coverage_update)
                _accumulate_assignment_coverage(
                    block_assignment_coverage[split].setdefault(temp_signal, {}).setdefault(metadata["block"], {}),
                    coverage_update,
                )
                total_signal = float(coverage_update["assigned_mass"])
                view_mass[split].setdefault(temp_signal, {})[image_name] = total_signal
                block_mass[split].setdefault(temp_signal, {})[metadata["block"]] = (
                    block_mass[split].setdefault(temp_signal, {}).get(metadata["block"], 0.0) + total_signal
                )

    responsibility_payload: Dict[str, Any] = {
        "status": (
            "not_requested_metric_only"
            if metric_only
            else "completed_optional_gaussian_sidecar"
        ),
        "attribution_semantics": (
            None
            if metric_only
            else "front error is attributed only for max-contribution depth; RGB and temperature residuals use "
            "top-1 shared-occupancy assignment approximations with explicit assigned/unassigned residual coverage "
            "(not top-k or exact causal decomposition)"
        ),
        "signals": (
            {}
            if metric_only
            else {
                "front_max_contribution": "top contributor and max-contribution depth are the same compositing event",
                "rgb_abs_top1_occupancy_approx": "diagnostic top-1 shared-occupancy approximation",
                "temperature_abs_top1_occupancy_approx": (
                    "diagnostic top-1 shared-occupancy approximation; generated only under explicit TSDK/Celsius manifest contract"
                ),
            }
        ),
        "method_selection_policy": "train/guard only; test is final-report-only",
        "gaussian_count": int(declared_count) if not metric_only else None,
        "appearance_modality": appearance_modality,
        "rgb_responsibility_status": "available_top1_approximation" if appearance_modality == "rgb" else "not_applicable",
        "gaussian_count_source": (
            "model_manifest (CLI equality checked when supplied)"
            if not metric_only
            else None
        ),
        "gaussian_index_anchor_sha256": model_index_anchor_sha or None,
        "scsp_set": {
            "count": len(scsp_set.indices),
            "path": scsp_set.path,
            "file_sha256": scsp_set.file_sha256,
            "anchor_sha256": scsp_set.anchor_sha256,
        },
        "clamp20_set": {
            "count": len(clamp20_set.indices),
            "path": clamp20_set.path,
            "file_sha256": clamp20_set.file_sha256,
            "anchor_sha256": clamp20_set.anchor_sha256,
        },
        "temperature_contract": (
            {key: value for key, value in temperature_contract.items() if key != "payload"}
            if temperature_contract is not None
            else None
        ),
        "split_policy": {},
        "splits": {},
    }
    npz_arrays: Dict[str, np.ndarray] = {}
    responsibility_dir = out_dir / "responsibility"
    for split, signals in sorted(masses.items()):
        normalized_split = split.strip().lower()
        responsibility_payload["split_policy"][split] = {
            "method_selection_eligible": normalized_split in {"train", "guard"},
            "final_report_only": normalized_split == "test",
            "view_count": int(split_view_counts.get(split, 0)),
        }
        split_payload: Dict[str, Any] = {}
        for signal, mass in sorted(signals.items()):
            summary = summarize_responsibility(
                mass,
                scsp_indices=set(scsp_set.indices),
                clamp20_indices=set(clamp20_set.indices),
            )
            coverage_summary = _finalize_assignment_coverage(assignment_coverage[split].get(signal, {}))
            if not np.isclose(
                float(summary["total_mass"]),
                float(coverage_summary["assigned_mass"]),
                rtol=1e-12,
                atol=1e-12,
            ):
                raise RuntimeError(f"Responsibility assigned-mass accounting mismatch for {split}/{signal}")
            summary["assignment_coverage"] = coverage_summary
            summary["view_assignment_coverage"] = _group_assignment_coverage(
                view_assignment_coverage[split].get(signal, {})
            )
            summary["block_assignment_coverage"] = _group_assignment_coverage(
                block_assignment_coverage[split].get(signal, {})
            )
            summary["view_concentration"] = _concentration(view_mass[split].get(signal, {}))
            summary["block_concentration"] = _concentration(block_mass[split].get(signal, {}))
            summary["view_coverage_count"] = len(view_mass[split].get(signal, {}))
            summary["view_coverage_fraction"] = (
                float(summary["view_coverage_count"]) / float(split_view_counts[split])
                if split_view_counts.get(split, 0) > 0
                else 0.0
            )
            if summary["view_coverage_fraction"] != 1.0:
                raise ValueError(f"Responsibility signal {split}/{signal} does not cover every view in its split")
            split_payload[signal] = summary
            npz_arrays[f"{split}__{signal}"] = mass
            top_path = responsibility_dir / split / f"{signal}_top20.csv"
            top_path.parent.mkdir(parents=True, exist_ok=True)
            with top_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["rank", "gaussian_index", "mass", "mass_share"])
                writer.writeheader()
                writer.writerows(summary["top20_entries"])
        responsibility_payload["splits"][split] = split_payload
    if npz_arrays:
        np.savez_compressed(out_dir / "responsibility_arrays.npz", **npz_arrays)

    csv_header = [
        "depth_definition",
        "threshold_m",
        "front_rate",
        "agreement_rate",
        "missing_rate",
        "mean_abs_error_m",
        "median_abs_error_m",
        "signed_bias_m",
        "behind_rate",
        "front_auc_log_0p25_20m",
        "front_auc_linear_legacy",
        "agreement_auc_linear_legacy",
    ]
    with (out_dir / "depth_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_header)
        writer.writerows(depth_csv_rows)
    with (out_dir / "group_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "depth_definition",
                "group_type",
                "group_label",
                "threshold_m",
                "front_rate",
                "agreement_rate",
                "missing_rate",
                "mean_abs_error_m",
                "median_abs_error_m",
                "signed_bias_m",
                "behind_rate",
                "front_auc_log_0p25_20m",
                "front_auc_linear_legacy",
                "agreement_auc_linear_legacy",
            ]
        )
        writer.writerows(group_csv_rows)

    summary_payload = {
        "protocol": (
            FORMAL_GEOMETRY_PROTOCOL
            if evaluation_mode == "formal"
            else "legacy-custom-multi-depth-definition-v1"
        ),
        "protocol_sha256": (
            FORMAL_GEOMETRY_PROTOCOL_SHA256
            if evaluation_mode == "formal"
            else None
        ),
        "support_policy_sha256": (
            FORMAL_SUPPORT_POLICY_SHA256
            if evaluation_mode == "formal"
            else None
        ),
        "formal_geometry_metric_contract": formal_metric_contract,
        "evaluation_mode": evaluation_mode,
        "metric_only": bool(metric_only),
        "support_mode": (
            "finite_positive_depth_metric_only"
            if metric_only
            else "gaussian_accumulated_opacity"
        ),
        "scene_name": scene_name,
        "thresholds_m": list(thresholds),
        "opacity_threshold": float(opacity_threshold),
        "depth_min": float(depth_min),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": _sha256(reference_manifest_path),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": _sha256(model_manifest_path),
        "probe_camera_manifest": str(probe_camera_manifest_path),
        "probe_camera_manifest_sha256": probe_sha,
        "formal_split_manifest": str(formal_split_manifest_path),
        "formal_split_manifest_sha256": split_sha,
        "camera_set_sha256": computed_camera_set_sha,
        "reference_mesh_sha256": reference_mesh_sha,
        "reference_mesh_backend": reference_mesh_backend,
        "all_split_reference_binding": all_split_binding,
        "depth_definitions": depth_payload,
        "responsibility": responsibility_payload,
    }
    _save_json(out_dir / "metrics_summary.json", summary_payload)
    _save_json(out_dir / "responsibility_summary.json", responsibility_payload)
    return summary_payload


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--model_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--evaluation_mode",
        choices=("formal", "legacy_custom"),
        default="formal",
        help="Formal is fail-closed; custom thresholds/linear AUC require legacy_custom.",
    )
    parser.add_argument(
        "--metric_only",
        action="store_true",
        help="Evaluate three supplied depth maps without Gaussian responsibility arrays.",
    )
    parser.add_argument("--thresholds_m", default=",".join(str(x) for x in DEFAULT_THRESHOLDS_M))
    parser.add_argument("--opacity_threshold", type=float, default=FORMAL_OPACITY_THRESHOLD)
    parser.add_argument("--depth_min", type=float, default=FORMAL_DEPTH_MIN_M)
    parser.add_argument("--gaussian_count", type=int, default=None)
    parser.add_argument("--scsp_indices", default="")
    parser.add_argument("--clamp20_indices", default="")
    parser.add_argument("--scsp_anchor_sha256", default="")
    parser.add_argument("--clamp20_anchor_sha256", default="")
    parser.add_argument("--probe_camera_manifest", required=True)
    parser.add_argument("--formal_split_manifest", required=True)
    parser.add_argument("--temperature_responsibility_manifest", default="")
    parser.add_argument("--cdf_max_points", type=int, default=4096)
    return parser


def main() -> None:
    args = _argparser().parse_args()
    summary = evaluate(
        reference_manifest_path=Path(args.reference_manifest),
        model_manifest_path=Path(args.model_manifest),
        out_dir=Path(args.out_dir),
        thresholds_m=_parse_thresholds(args.thresholds_m),
        opacity_threshold=float(args.opacity_threshold),
        depth_min=float(args.depth_min),
        gaussian_count=args.gaussian_count,
        scsp_indices_path=Path(args.scsp_indices).resolve() if str(args.scsp_indices).strip() else None,
        clamp20_indices_path=Path(args.clamp20_indices).resolve() if str(args.clamp20_indices).strip() else None,
        scsp_anchor_sha256=str(args.scsp_anchor_sha256),
        clamp20_anchor_sha256=str(args.clamp20_anchor_sha256),
        probe_camera_manifest_path=Path(args.probe_camera_manifest).resolve(),
        formal_split_manifest_path=Path(args.formal_split_manifest).resolve(),
        temperature_responsibility_manifest_path=(
            Path(args.temperature_responsibility_manifest).resolve()
            if str(args.temperature_responsibility_manifest).strip()
            else None
        ),
        cdf_max_points=int(args.cdf_max_points),
        evaluation_mode=str(args.evaluation_mode),
        metric_only=bool(args.metric_only),
    )
    print(f"DEPTH_DEFINITION_METRICS {Path(args.out_dir).resolve() / 'metrics_summary.json'}")
    print(f"DEPTH_DEFINITION_COUNT {len(summary['depth_definitions'])}")


if __name__ == "__main__":
    main()
