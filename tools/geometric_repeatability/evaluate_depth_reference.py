from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import load_json, save_json, write_simple_csv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_identity() -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        git_error = ""
    except Exception as exc:
        commit = ""
        status = "git-identity-unavailable"
        git_error = f"{type(exc).__name__}: {exc}"
    return {
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
        "repo_root": str(REPO_ROOT.resolve()),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_error": git_error,
    }


def _verified_view_npz(manifest_path: Path, view: Dict[str, Any], *, label: str) -> Path:
    manifest_root = manifest_path.parent.resolve()
    npz_path = (manifest_root / str(view.get("npz_file", ""))).resolve()
    try:
        npz_path.relative_to(manifest_root)
    except ValueError as exc:
        raise ValueError(f"{label} view escapes its manifest root: {npz_path}") from exc
    if not npz_path.is_file():
        raise FileNotFoundError(f"{label} view NPZ is missing: {npz_path}")
    expected_size = int(view.get("npz_size_bytes", -1))
    expected_sha = str(view.get("npz_sha256", "")).lower()
    if expected_size != int(npz_path.stat().st_size) or expected_sha != _sha256_file(npz_path):
        raise RuntimeError(f"{label} view NPZ identity mismatch: {npz_path}")
    return npz_path


def _raw_depth_to_metric_camera_z(raw_depth: np.ndarray, depth_semantics: str) -> np.ndarray:
    raw_depth = np.asarray(raw_depth, dtype=np.float64)
    if depth_semantics == "metric_camera_z_from_renderer":
        return raw_depth
    if depth_semantics == "inverse_camera_z_from_renderer":
        metric = np.full(raw_depth.shape, np.nan, dtype=np.float64)
        positive = np.isfinite(raw_depth) & (raw_depth > 0.0)
        metric[positive] = 1.0 / raw_depth[positive]
        return metric
    raise ValueError(f"Unsupported depth semantics: {depth_semantics!r}")


def _make_model_valid_mask(metric_depth: np.ndarray, opacity: np.ndarray, depth_min: float, opacity_threshold: float) -> np.ndarray:
    return (
        np.isfinite(metric_depth)
        & np.isfinite(opacity)
        & (metric_depth > float(depth_min))
        & (opacity >= float(opacity_threshold))
    )


def _load_runtime_flags(out_dir: Path) -> Dict[str, Any]:
    for candidate_dir in [out_dir] + list(out_dir.parents):
        flags_path = candidate_dir / "depth_reference_runtime_flags.json"
        if not flags_path.exists():
            continue
        with flags_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Runtime flags file must contain a JSON object: {flags_path}")
        return data
    return {}


def _argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a rendered model depth bundle against a training-only reference depth bundle")
    parser.add_argument("--reference_manifest", required=True)
    parser.add_argument("--model_manifest", required=True)
    parser.add_argument("--adapter_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--enable_agreement_metrics",
        action="store_true",
        help="Also compute symmetric depth-agreement statistics; default OFF for backward compatibility.",
    )
    return parser


def main() -> None:
    args = _argparser().parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_manifest_path = Path(args.reference_manifest).resolve()
    model_manifest_path = Path(args.model_manifest).resolve()
    adapter_manifest_path = Path(args.adapter_manifest).resolve()

    ref_manifest = load_json(ref_manifest_path)
    model_manifest = load_json(model_manifest_path)
    adapter_manifest = load_json(adapter_manifest_path)
    runtime_flags = _load_runtime_flags(out_dir)
    enable_agreement_metrics = bool(args.enable_agreement_metrics or runtime_flags.get("enable_agreement_metrics", False))

    thresholds_m = [float(x) for x in ref_manifest["thresholds_m"]]
    adapter_semantics = str(adapter_manifest["depth_semantics"])
    validity_rule = adapter_manifest["validity_rule"]
    depth_min = float(validity_rule.get("depth_min", 1e-6))
    opacity_threshold = float(validity_rule.get("opacity_threshold", 0.5))

    ref_views = list(ref_manifest["views"])
    model_views = list(model_manifest["views"])
    ref_by_name = {str(v["image_name"]): v for v in ref_views}
    model_by_name = {str(v["image_name"]): v for v in model_views}
    if len(ref_by_name) != len(ref_views) or len(model_by_name) != len(model_views):
        raise ValueError("Reference/model manifests contain duplicate image_name entries")
    missing_in_model = sorted(set(ref_by_name) - set(model_by_name))
    extra_in_model = sorted(set(model_by_name) - set(ref_by_name))
    if missing_in_model:
        sample = ", ".join(missing_in_model[:8])
        raise ValueError(f"Model bundle is missing reference views: {sample}")
    if extra_in_model:
        sample = ", ".join(extra_in_model[:8])
        raise ValueError(f"Model bundle has extra views not in reference: {sample}")

    total_ref_valid = 0
    total_model_valid_on_ref = 0
    total_missing = 0
    intrusion_counts = {d: 0 for d in thresholds_m}
    too_deep_counts = {d: 0 for d in thresholds_m}
    agreement_counts = {d: 0 for d in thresholds_m}
    intrusion_sum = {d: 0.0 for d in thresholds_m}
    abs_errors: List[np.ndarray] = []
    signed_errors: List[np.ndarray] = []
    per_view_rows: List[List[Any]] = []

    for image_name in sorted(ref_by_name):
        ref_view = ref_by_name[image_name]
        model_view = model_by_name[image_name]
        ref_npz_path = _verified_view_npz(ref_manifest_path, ref_view, label="Reference")
        model_npz_path = _verified_view_npz(model_manifest_path, model_view, label="Model")
        ref_npz = np.load(ref_npz_path)
        model_npz = np.load(model_npz_path)

        ref_depth = np.asarray(ref_npz["depth"], dtype=np.float64)
        ref_valid = np.asarray(ref_npz["valid_mask"], dtype=np.uint8).astype(bool)
        raw_model_depth = np.asarray(model_npz["depth"], dtype=np.float64)
        model_opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
        model_depth = _raw_depth_to_metric_camera_z(raw_model_depth, depth_semantics=adapter_semantics)
        model_valid = _make_model_valid_mask(model_depth, model_opacity, depth_min=depth_min, opacity_threshold=opacity_threshold)

        if ref_depth.shape != model_depth.shape:
            raise ValueError(f"Shape mismatch for {image_name}: ref {ref_depth.shape} vs model {model_depth.shape}")
        if ref_depth.shape != model_opacity.shape:
            raise ValueError(f"Opacity shape mismatch for {image_name}: ref {ref_depth.shape} vs opacity {model_opacity.shape}")

        eval_mask = ref_valid
        valid_joint = eval_mask & model_valid
        total_ref_valid += int(np.count_nonzero(eval_mask))
        total_model_valid_on_ref += int(np.count_nonzero(valid_joint))
        missing_mask = eval_mask & (~model_valid)
        total_missing += int(np.count_nonzero(missing_mask))

        if np.any(valid_joint):
            signed = model_depth[valid_joint] - ref_depth[valid_joint]
            signed_errors.append(np.asarray(signed, dtype=np.float64))
            abs_errors.append(np.asarray(np.abs(signed), dtype=np.float64))
        else:
            signed = np.zeros((0,), dtype=np.float64)

        row: List[Any] = [
            image_name,
            int(np.count_nonzero(eval_mask)),
            int(np.count_nonzero(valid_joint)),
            int(np.count_nonzero(missing_mask)),
        ]

        for delta in thresholds_m:
            intrusion = valid_joint & (model_depth < (ref_depth - delta))
            too_deep = valid_joint & (model_depth > (ref_depth + delta))
            intrusion_count = int(np.count_nonzero(intrusion))
            too_deep_count = int(np.count_nonzero(too_deep))
            if enable_agreement_metrics:
                agreement = valid_joint & (np.abs(model_depth - ref_depth) <= delta)
                agreement_count = int(np.count_nonzero(agreement))
                agreement_counts[delta] += agreement_count
            intrusion_counts[delta] += intrusion_count
            too_deep_counts[delta] += too_deep_count
            if intrusion_count > 0:
                intrusion_sum[delta] += float(np.sum(ref_depth[intrusion] - model_depth[intrusion]))
            row.extend([intrusion_count, too_deep_count])
            if enable_agreement_metrics:
                row.append(agreement_count)
        per_view_rows.append(row)

    if total_ref_valid <= 0:
        raise ValueError("Reference valid set is empty")

    if abs_errors:
        abs_concat = np.concatenate(abs_errors, axis=0)
        signed_concat = np.concatenate(signed_errors, axis=0)
        abs_mean = float(np.mean(abs_concat))
        abs_median = float(np.median(abs_concat))
        signed_mean = float(np.mean(signed_concat))
    else:
        abs_mean = float("nan")
        abs_median = float("nan")
        signed_mean = float("nan")

    curve_rows: List[List[Any]] = []
    summary_rows: List[List[Any]] = []
    for delta in thresholds_m:
        intrusion_count = intrusion_counts[delta]
        too_deep_count = too_deep_counts[delta]
        intrusion_rate = float(intrusion_count) / float(total_ref_valid)
        too_deep_rate = float(too_deep_count) / float(total_ref_valid)
        agreement_rate = float(agreement_counts[delta]) / float(total_ref_valid) if enable_agreement_metrics else None
        intrusion_magnitude = float(intrusion_sum[delta] / intrusion_count) if intrusion_count > 0 else 0.0
        curve_row = [
            adapter_manifest["method_name"],
            f"{delta:.2f}",
            f"{intrusion_rate:.12f}",
            f"{intrusion_magnitude:.12f}",
            f"{too_deep_rate:.12f}",
        ]
        if enable_agreement_metrics:
            curve_row.append(f"{agreement_rate:.12f}")
        curve_rows.append(curve_row)
        summary_row = [
            adapter_manifest["method_name"],
            f"{delta:.2f}",
            f"{intrusion_rate:.12f}",
            f"{intrusion_magnitude:.12f}",
            f"{too_deep_rate:.12f}",
            f"{(float(total_missing) / float(total_ref_valid)):.12f}",
            f"{abs_mean:.12f}" if np.isfinite(abs_mean) else "nan",
            f"{abs_median:.12f}" if np.isfinite(abs_median) else "nan",
            f"{signed_mean:.12f}" if np.isfinite(signed_mean) else "nan",
        ]
        if enable_agreement_metrics:
            summary_row.append(f"{agreement_rate:.12f}")
        summary_rows.append(summary_row)

    summary_payload = {
        "protocol_name": "reference-depth-based-geometric-evaluation-v1",
        "producer_identity": _producer_identity(),
        "scene_name": ref_manifest["scene_name"],
        "method_name": adapter_manifest["method_name"],
        "reference_manifest": str(ref_manifest_path),
        "reference_manifest_sha256": _sha256_file(ref_manifest_path),
        "model_manifest": str(model_manifest_path),
        "model_manifest_sha256": _sha256_file(model_manifest_path),
        "adapter_manifest": str(adapter_manifest_path),
        "adapter_manifest_sha256": _sha256_file(adapter_manifest_path),
        "depth_semantics": adapter_semantics,
        "validity_rule": validity_rule,
        "evaluation_options": {
            "enable_agreement_metrics": enable_agreement_metrics,
        },
        "counts": {
            "reference_valid_pixels": int(total_ref_valid),
            "model_valid_on_reference_pixels": int(total_model_valid_on_ref),
            "missing_pixels": int(total_missing),
        },
        "secondary_metrics": {
            "ModelValidOnReferenceRate": float(total_model_valid_on_ref) / float(total_ref_valid),
            "MissingRate": float(total_missing) / float(total_ref_valid),
            "AbsDepthError_Mean": abs_mean,
            "AbsDepthError_Median": abs_median,
            "SignedDepthBias_Mean": signed_mean,
        },
        "threshold_metrics": [
            {
                "threshold_m": float(delta),
                "FrontIntrusionRate": float(intrusion_counts[delta]) / float(total_ref_valid),
                "FrontIntrusionMagnitude": float(intrusion_sum[delta] / intrusion_counts[delta]) if intrusion_counts[delta] > 0 else 0.0,
                "TooDeepRate": float(too_deep_counts[delta]) / float(total_ref_valid),
                **(
                    {"DepthAgreementRate": float(agreement_counts[delta]) / float(total_ref_valid)}
                    if enable_agreement_metrics
                    else {}
                ),
            }
            for delta in thresholds_m
        ],
    }
    save_json(out_dir / "metrics_summary.json", summary_payload)
    write_simple_csv(
        out_dir / "metrics_summary.csv",
        [
            "method_name",
            "threshold_m",
            "FrontIntrusionRate",
            "FrontIntrusionMagnitude",
            "TooDeepRate",
            "MissingRate",
            "AbsDepthError_Mean",
            "AbsDepthError_Median",
            "SignedDepthBias_Mean",
            *(
                ["DepthAgreementRate"]
                if enable_agreement_metrics
                else []
            ),
        ],
        summary_rows,
    )
    write_simple_csv(
        out_dir / "front_intrusion_curve.csv",
        [
            "method_name",
            "threshold_m",
            "FrontIntrusionRate",
            "FrontIntrusionMagnitude",
            "TooDeepRate",
            *(
                ["DepthAgreementRate"]
                if enable_agreement_metrics
                else []
            ),
        ],
        curve_rows,
    )
    per_view_header = ["image_name", "reference_valid_pixels", "model_valid_pixels", "missing_pixels"]
    for delta in thresholds_m:
        per_view_header.extend([f"FrontIntrusionCount@{delta:.2f}m", f"TooDeepCount@{delta:.2f}m"])
        if enable_agreement_metrics:
            per_view_header.append(f"DepthAgreementCount@{delta:.2f}m")
    write_simple_csv(out_dir / "per_view_counts.csv", per_view_header, per_view_rows)
    print(f"DEPTH_REFERENCE_METRICS {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
