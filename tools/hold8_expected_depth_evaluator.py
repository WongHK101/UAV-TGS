"""Minimal Hold-8 expected-depth geometry evaluator.

This module is intentionally independent from the legacy three-depth
diagnostic evaluator.  It evaluates only alpha/volume-weighted expected
camera-z depth and does not emit behind, AUC, guard, block, or responsibility
statistics.

Input bundles contain one NPZ per test view.  Existing UAV-TGS renderer output
(``depth_expected_alpha_normalized`` + ``accumulated_opacity``) is accepted,
as are the canonical aliases ``expected_depth_camera_z`` + ``weight_sum``.
An adapter may instead provide ``weighted_depth_sum`` + ``weight_sum``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "uav-tgs-aaai27-hold8-expected-depth-v1"
NA_SCHEMA_VERSION = "uav-tgs-aaai27-hold8-geometry-na-v1"
EXPECTED_DEPTH_EPSILON = 1.0e-8
MAIN_THRESHOLDS_M = (1.0, 2.0, 5.0)
SUPPLEMENTAL_THRESHOLDS_M = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)

_NORMALIZED_EXPECTED_KEYS = (
    "expected_depth_camera_z",
    "depth_expected_alpha_normalized",
)
_WEIGHT_SUM_KEYS = ("weight_sum", "accumulated_opacity")
_WEIGHTED_DEPTH_SUM_KEY = "weighted_depth_sum"
_POSITIVE_SAMPLE_KEY = "has_finite_positive_depth_sample"
_SPLIT_BINDING_FIELDS = (
    "collection_hash",
    "collection_split_hash",
    "scene_split_hash",
    "test_list_sha256",
    "scene_split_manifest_sha256",
    "collection_manifest_sha256",
)
_MODEL_DEPTH_CONTRACT = {
    "name": "alpha_volume_weighted_expected_camera_z",
    "z_semantics": "metric_camera_z",
    "weight_semantics": "official_renderer_alpha_or_volume_contribution",
    "weight_epsilon": EXPECTED_DEPTH_EPSILON,
    "positive_sample_evidence": _POSITIVE_SAMPLE_KEY,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require_sha256(value: Any, label: str) -> str:
    token = str(value or "").strip().lower()
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return token


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _first_present(payload: Mapping[str, np.ndarray], keys: Sequence[str]) -> str | None:
    present = [key for key in keys if key in payload]
    if len(present) > 1:
        first = np.asarray(payload[present[0]])
        for key in present[1:]:
            if not np.array_equal(first, np.asarray(payload[key]), equal_nan=True):
                raise ValueError(f"Conflicting equivalent arrays: {present}")
    return present[0] if present else None


def expected_depth_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return normalized expected camera-z depth, weight sum, representation.

    Validity is deliberately not hidden in the division: callers must require
    a finite weight sum strictly greater than :data:`EXPECTED_DEPTH_EPSILON`
    and a finite positive normalized depth.
    """

    weight_key = _first_present(arrays, _WEIGHT_SUM_KEYS)
    if weight_key is None:
        raise ValueError(
            "Expected one weight-sum array: weight_sum or accumulated_opacity"
        )
    weight_sum = np.asarray(arrays[weight_key], dtype=np.float64)
    if _POSITIVE_SAMPLE_KEY not in arrays:
        raise ValueError(
            f"Expected {_POSITIVE_SAMPLE_KEY} to certify a finite positive camera-z sample"
        )
    positive_sample = np.asarray(arrays[_POSITIVE_SAMPLE_KEY], dtype=bool)

    expected_key = _first_present(arrays, _NORMALIZED_EXPECTED_KEYS)
    has_numerator = _WEIGHTED_DEPTH_SUM_KEY in arrays
    if expected_key is not None and has_numerator:
        raise ValueError(
            "Ambiguous expected-depth representation: provide either normalized "
            "expected depth or weighted_depth_sum, not both"
        )
    if expected_key is not None:
        expected = np.asarray(arrays[expected_key], dtype=np.float64)
        representation = "normalized_expected_depth"
    elif has_numerator:
        numerator = np.asarray(arrays[_WEIGHTED_DEPTH_SUM_KEY], dtype=np.float64)
        if numerator.shape != weight_sum.shape:
            raise ValueError(
                f"weighted_depth_sum/weight_sum shape mismatch: "
                f"{numerator.shape} vs {weight_sum.shape}"
            )
        expected = np.full(weight_sum.shape, np.nan, dtype=np.float64)
        denominator_valid = np.isfinite(weight_sum) & (
            weight_sum > EXPECTED_DEPTH_EPSILON
        )
        np.divide(numerator, weight_sum, out=expected, where=denominator_valid)
        representation = "weighted_depth_sum"
    else:
        raise ValueError(
            "Expected expected_depth_camera_z, depth_expected_alpha_normalized, "
            "or weighted_depth_sum"
        )

    if expected.shape != weight_sum.shape:
        raise ValueError(
            f"Expected-depth/weight-sum shape mismatch: {expected.shape} vs "
            f"{weight_sum.shape}"
        )
    if positive_sample.shape != weight_sum.shape:
        raise ValueError(
            f"Positive-sample/weight-sum shape mismatch: {positive_sample.shape} vs "
            f"{weight_sum.shape}"
        )
    return expected, weight_sum, positive_sample, representation


def compute_expected_depth_metrics(
    reference_depth: np.ndarray,
    reference_valid_mask: np.ndarray,
    expected_depth: np.ndarray,
    weight_sum: np.ndarray,
    has_finite_positive_depth_sample: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute Hold-8 expected-depth metrics for one or more arrays.

    Front/agreement/median-error denominators contain only pixels where both
    reference and model are valid.  Missing rate alone uses reference-valid
    pixels as its denominator.
    """

    reference_depth = np.asarray(reference_depth, dtype=np.float64)
    reference_valid_mask = np.asarray(reference_valid_mask, dtype=bool)
    expected_depth = np.asarray(expected_depth, dtype=np.float64)
    weight_sum = np.asarray(weight_sum, dtype=np.float64)
    if has_finite_positive_depth_sample is None:
        positive_sample = np.isfinite(expected_depth) & (expected_depth > 0.0)
    else:
        positive_sample = np.asarray(has_finite_positive_depth_sample, dtype=bool)
    shapes = {
        reference_depth.shape,
        reference_valid_mask.shape,
        expected_depth.shape,
        weight_sum.shape,
        positive_sample.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "Reference depth, valid mask, expected depth, and weight sum must "
            "have identical shapes"
        )

    reference_valid = (
        reference_valid_mask
        & np.isfinite(reference_depth)
        & (reference_depth > 0.0)
    )
    model_valid = (
        np.isfinite(weight_sum)
        & (weight_sum > EXPECTED_DEPTH_EPSILON)
        & np.isfinite(expected_depth)
        & (expected_depth > 0.0)
        & positive_sample
    )
    joint_valid = reference_valid & model_valid
    missing = reference_valid & ~model_valid

    reference_count = int(np.count_nonzero(reference_valid))
    joint_count = int(np.count_nonzero(joint_valid))
    missing_count = int(np.count_nonzero(missing))
    if reference_count <= 0:
        raise ValueError("Reference-valid set is empty")

    signed_error = expected_depth[joint_valid] - reference_depth[joint_valid]
    abs_error = np.abs(signed_error)
    median_abs_error = float(np.median(abs_error)) if joint_count else None

    threshold_metrics: list[dict[str, Any]] = []
    for threshold in SUPPLEMENTAL_THRESHOLDS_M:
        if joint_count:
            front_count = int(np.count_nonzero(signed_error < -threshold))
            agreement_count = int(np.count_nonzero(abs_error <= threshold))
            front_rate: float | None = front_count / joint_count
            agreement_rate: float | None = agreement_count / joint_count
        else:
            front_count = 0
            agreement_count = 0
            front_rate = None
            agreement_rate = None
        threshold_metrics.append(
            {
                "threshold_m": threshold,
                "is_main_table_threshold": threshold in MAIN_THRESHOLDS_M,
                "joint_valid_pixels": joint_count,
                "front_count": front_count,
                "front_rate": front_rate,
                "agreement_count": agreement_count,
                "agreement_rate": agreement_rate,
            }
        )

    return {
        "counts": {
            "reference_valid_pixels": reference_count,
            "joint_valid_pixels": joint_count,
            "missing_pixels": missing_count,
        },
        "missing_rate": missing_count / reference_count,
        "median_absolute_depth_error_m": median_abs_error,
        "threshold_metrics": threshold_metrics,
    }


def _resolve_npz(manifest_path: Path, view: Mapping[str, Any], label: str) -> Path:
    root = manifest_path.parent.resolve()
    path = (root / str(view.get("npz_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} NPZ escapes manifest root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_size = view.get("npz_size_bytes")
    if expected_size is None:
        raise ValueError(f"{label} NPZ requires npz_size_bytes: {path}")
    expected_sha = _require_sha256(view.get("npz_sha256"), f"{label} NPZ SHA-256")
    if int(expected_size) != path.stat().st_size:
        raise RuntimeError(f"{label} NPZ size mismatch: {path}")
    if expected_sha != _sha256(path):
        raise RuntimeError(f"{label} NPZ SHA-256 mismatch: {path}")
    return path


def _index_views(manifest: Mapping[str, Any], label: str) -> dict[str, Mapping[str, Any]]:
    views = manifest.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{label} manifest must contain a non-empty views list")
    indexed: dict[str, Mapping[str, Any]] = {}
    image_names: set[str] = set()
    for view in views:
        if not isinstance(view, Mapping):
            raise ValueError(f"{label} view entries must be objects")
        pair_id = str(view.get("pair_id", "")).strip()
        image_name = str(view.get("image_name", "")).strip()
        if not pair_id or pair_id in indexed:
            raise ValueError(f"{label} manifest has empty/duplicate pair_id: {pair_id!r}")
        if not image_name or image_name in image_names:
            raise ValueError(
                f"{label} manifest has empty/duplicate image_name for pair {pair_id!r}"
            )
        indexed[pair_id] = view
        image_names.add(image_name)
    return indexed


def _list_sha256(values: Sequence[str]) -> str:
    raw = ("\n".join(values) + ("\n" if values else "")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_authoritative_split(
    *,
    collection_manifest_path: Path,
    scene_split_manifest_path: Path,
    scene_name: str,
) -> tuple[dict[str, str], list[str]]:
    collection = _load_json(collection_manifest_path)
    scene_split = _load_json(scene_split_manifest_path)
    if collection.get("protocol_id") != "uav-tgs-aaai27-hold8-v2":
        raise ValueError("Collection manifest is not AAAI27 Hold-8 v2")
    if scene_split.get("protocol_id") != "uav-tgs-aaai27-hold8-v2":
        raise ValueError("Scene split manifest is not AAAI27 Hold-8 v2")
    if scene_split.get("scene") != scene_name:
        raise ValueError("Scene split manifest/geometry scene mismatch")
    records = scene_split.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Scene split manifest contains no records")
    test_pair_ids: list[str] = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError("Scene split record is not an object")
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id or int(row.get("zero_based_sorted_index", -1)) != index:
            raise ValueError("Scene split pair/index contract failed")
        expected_label = "test" if index % 8 == 0 else "train"
        if row.get("split") != expected_label:
            raise ValueError("Scene split modulo-8 membership failed")
        if expected_label == "test":
            test_pair_ids.append(pair_id)
    if len(test_pair_ids) != len(set(test_pair_ids)):
        raise ValueError("Scene split contains duplicate test pair IDs")
    hashes = scene_split.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("Scene split manifest lacks hashes")
    test_list_sha = _list_sha256(test_pair_ids)
    if test_list_sha != _require_sha256(
        hashes.get("test_list_sha256"), "Scene split test_list_sha256"
    ):
        raise ValueError("Scene split test-list hash mismatch")
    scene_entries = [
        row
        for row in collection.get("scenes", [])
        if isinstance(row, Mapping) and row.get("scene") == scene_name
    ]
    if len(scene_entries) != 1:
        raise ValueError("Collection must contain exactly one matching scene")
    scene_entry = scene_entries[0]
    scene_manifest_sha = _sha256(scene_split_manifest_path)
    if scene_manifest_sha != _require_sha256(
        scene_entry.get("manifest_sha256"), "Collection scene manifest_sha256"
    ):
        raise ValueError("Collection/scene manifest SHA mismatch")
    if scene_entry.get("split_hash") != scene_split.get("split_hash"):
        raise ValueError("Collection/scene split_hash mismatch")
    collection_hash = _require_sha256(
        collection.get("collection_hash"), "Collection collection_hash"
    )
    if scene_split.get("collection_hash") != collection_hash:
        raise ValueError("Collection/scene collection_hash mismatch")
    return (
        {
            "collection_hash": collection_hash,
            "collection_split_hash": _require_sha256(
                collection.get("collection_split_hash"),
                "Collection collection_split_hash",
            ),
            "scene_split_hash": _require_sha256(
                scene_split.get("split_hash"), "Scene split_hash"
            ),
            "test_list_sha256": test_list_sha,
            "scene_split_manifest_sha256": scene_manifest_sha,
            "collection_manifest_sha256": _sha256(collection_manifest_path),
        },
        test_pair_ids,
    )


def evaluate_manifests(
    reference_manifest_path: Path,
    model_manifest_path: Path,
    collection_manifest_path: Path,
    scene_split_manifest_path: Path,
    *,
    expected_collection_manifest_sha256: str,
    expected_scene_split_manifest_sha256: str,
) -> dict[str, Any]:
    reference_manifest_path = Path(reference_manifest_path).resolve()
    model_manifest_path = Path(model_manifest_path).resolve()
    collection_manifest_path = Path(collection_manifest_path).resolve()
    scene_split_manifest_path = Path(scene_split_manifest_path).resolve()
    if _sha256(collection_manifest_path) != _require_sha256(
        expected_collection_manifest_sha256,
        "Expected collection manifest SHA-256",
    ):
        raise ValueError("Collection manifest differs from the frozen protocol SHA-256")
    if _sha256(scene_split_manifest_path) != _require_sha256(
        expected_scene_split_manifest_sha256,
        "Expected scene split manifest SHA-256",
    ):
        raise ValueError("Scene split manifest differs from the frozen protocol SHA-256")
    reference = _load_json(reference_manifest_path)
    model = _load_json(model_manifest_path)

    if str(reference.get("split", "")) != "test" or str(model.get("split", "")) != "test":
        raise ValueError("Hold-8 formal geometry evaluator accepts test split only")
    if str(reference.get("scene_name", "")) != str(model.get("scene_name", "")):
        raise ValueError("Reference/model scene_name mismatch")
    scene_name = str(reference.get("scene_name", ""))
    authoritative_binding, test_pair_ids = _validate_authoritative_split(
        collection_manifest_path=collection_manifest_path,
        scene_split_manifest_path=scene_split_manifest_path,
        scene_name=scene_name,
    )
    split_binding: dict[str, str] = {}
    for field in _SPLIT_BINDING_FIELDS:
        reference_value = _require_sha256(
            reference.get(field), f"Reference {field}"
        )
        model_value = _require_sha256(model.get(field), f"Model {field}")
        if reference_value != model_value:
            raise ValueError(f"Reference/model {field} mismatch")
        if reference_value != authoritative_binding[field]:
            raise ValueError(f"Reference/model {field} is not authoritative")
        split_binding[field] = reference_value

    if reference.get("depth_semantics") != "metric_camera_z":
        raise ValueError("Reference depth_semantics must be metric_camera_z")
    if model.get("depth_contract") != _MODEL_DEPTH_CONTRACT:
        raise ValueError("Model depth_contract is not the formal camera-z contract")

    reference_views = _index_views(reference, "Reference")
    model_views = _index_views(model, "Model")
    if set(reference_views) != set(model_views):
        missing = sorted(set(reference_views) - set(model_views))
        extra = sorted(set(model_views) - set(reference_views))
        raise ValueError(
            f"Reference/model view-set mismatch; missing={missing[:8]}, extra={extra[:8]}"
        )
    if set(reference_views) != set(test_pair_ids) or len(reference_views) != len(test_pair_ids):
        raise ValueError("Geometry view set does not equal the authoritative test list")

    all_ref_depth: list[np.ndarray] = []
    all_ref_valid: list[np.ndarray] = []
    all_expected: list[np.ndarray] = []
    all_weight: list[np.ndarray] = []
    all_positive_sample: list[np.ndarray] = []
    representations: set[str] = set()
    for image_name in sorted(reference_views):
        ref_path = _resolve_npz(
            reference_manifest_path, reference_views[image_name], "Reference"
        )
        model_path = _resolve_npz(
            model_manifest_path, model_views[image_name], "Model"
        )
        with np.load(ref_path, allow_pickle=False) as ref_arrays:
            if "depth" not in ref_arrays or "valid_mask" not in ref_arrays:
                raise ValueError(f"Reference NPZ lacks depth/valid_mask: {ref_path}")
            ref_depth = np.asarray(ref_arrays["depth"], dtype=np.float64)
            ref_valid = np.asarray(ref_arrays["valid_mask"], dtype=bool)
        with np.load(model_path, allow_pickle=False) as model_arrays:
            expected, weight, positive_sample, representation = expected_depth_from_arrays(
                model_arrays
            )
        if ref_depth.shape != expected.shape or ref_valid.shape != expected.shape:
            raise ValueError(
                f"Shape mismatch for {image_name}: reference={ref_depth.shape}, "
                f"valid={ref_valid.shape}, model={expected.shape}"
            )
        all_ref_depth.append(ref_depth.reshape(-1))
        all_ref_valid.append(ref_valid.reshape(-1))
        all_expected.append(expected.reshape(-1))
        all_weight.append(weight.reshape(-1))
        all_positive_sample.append(positive_sample.reshape(-1))
        representations.add(representation)

    metrics = compute_expected_depth_metrics(
        np.concatenate(all_ref_depth),
        np.concatenate(all_ref_valid),
        np.concatenate(all_expected),
        np.concatenate(all_weight),
        np.concatenate(all_positive_sample),
    )
    method_name = str(model.get("method_name", ""))
    if not method_name:
        raise ValueError("Model manifest is missing method_name")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "geometry_status": "available",
        "scene_name": str(reference["scene_name"]),
        "method_name": method_name,
        "split": "test",
        "depth_definition": {
            "name": "alpha_volume_weighted_expected_camera_z",
            "formula": "sum(w_i * z_i) / sum(w_i)",
            "z_semantics": "metric camera-z",
            "weight_semantics": "official renderer alpha/volume contribution weights",
            "weight_epsilon": EXPECTED_DEPTH_EPSILON,
            "validity": (
                "finite weight_sum > epsilon and finite positive normalized depth"
            ),
            "front_agreement_median_denominator": "joint_valid_pixels",
            "missing_denominator": "reference_valid_pixels",
        },
        "thresholds": {
            "main_m": list(MAIN_THRESHOLDS_M),
            "supplemental_m": list(SUPPLEMENTAL_THRESHOLDS_M),
        },
        "input_representation": sorted(representations),
        "inputs": {
            "reference_manifest_sha256": _sha256(reference_manifest_path),
            "model_manifest_sha256": _sha256(model_manifest_path),
            "collection_manifest_sha256": _sha256(collection_manifest_path),
            "scene_split_manifest_sha256": _sha256(scene_split_manifest_path),
            "split_binding": split_binding,
        },
        "metrics": metrics,
    }


def geometry_na_receipt(
    *, scene_name: str, method_name: str, technical_reason: str
) -> dict[str, Any]:
    scene_name = scene_name.strip()
    method_name = method_name.strip()
    technical_reason = technical_reason.strip()
    if not scene_name or not method_name or not technical_reason:
        raise ValueError("scene_name, method_name, and technical_reason must be non-empty")
    return {
        "schema_version": NA_SCHEMA_VERSION,
        "status": "not_applicable",
        "geometry_status": "not_available",
        "scene_name": scene_name,
        "method_name": method_name,
        "split": "test",
        "requested_depth_definition": "alpha_volume_weighted_expected_camera_z",
        "technical_reason": technical_reason,
        "metrics": None,
    }


def _write_threshold_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = payload["metrics"]["threshold_metrics"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate AAAI27 Hold-8 alpha/volume-weighted expected depth"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--reference-manifest", required=True)
    evaluate_parser.add_argument("--model-manifest", required=True)
    evaluate_parser.add_argument("--collection-manifest", required=True)
    evaluate_parser.add_argument("--scene-split-manifest", required=True)
    evaluate_parser.add_argument("--expected-collection-manifest-sha256", required=True)
    evaluate_parser.add_argument("--expected-scene-split-manifest-sha256", required=True)
    evaluate_parser.add_argument("--out-dir", required=True)
    na_parser = subparsers.add_parser("geometry-na")
    na_parser.add_argument("--scene", required=True)
    na_parser.add_argument("--method", required=True)
    na_parser.add_argument("--technical-reason", required=True)
    na_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "evaluate":
        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = evaluate_manifests(
            Path(args.reference_manifest),
            Path(args.model_manifest),
            Path(args.collection_manifest),
            Path(args.scene_split_manifest),
            expected_collection_manifest_sha256=args.expected_collection_manifest_sha256,
            expected_scene_split_manifest_sha256=args.expected_scene_split_manifest_sha256,
        )
        _write_json(out_dir / "geometry_metrics.json", payload)
        _write_threshold_csv(out_dir / "front_agreement_curve.csv", payload)
        print(f"HOLD8_EXPECTED_DEPTH_METRICS {out_dir / 'geometry_metrics.json'}")
    else:
        output = Path(args.output).resolve()
        _write_json(
            output,
            geometry_na_receipt(
                scene_name=args.scene,
                method_name=args.method,
                technical_reason=args.technical_reason,
            ),
        )
        print(f"HOLD8_GEOMETRY_NA {output}")


if __name__ == "__main__":
    main()
