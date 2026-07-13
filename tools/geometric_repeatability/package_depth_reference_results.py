from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_DIR = Path(__file__).resolve().parent

SCENE_ORDER = ["Building", "PVpanel", "Orchard", "Road", "TransmissionTower"]
METHOD_ORDER = [
    "Ours_M00_full",
    "Ours_G01_full",
    "Ours_G02_full",
    "Ours_M01_full",
    "Thermal3D_GS_full",
    "ThermalGaussian_OMMG_full",
    "ThermalGaussian_MSMG_full",
    "ThermalGaussian_MFTG_full",
]
METHOD_GROUPS = {
    "ablation_4method": [
        "Ours_M00_full",
        "Ours_G01_full",
        "Ours_G02_full",
        "Ours_M01_full",
    ],
    "sota_5method": [
        "Ours_M01_full",
        "Thermal3D_GS_full",
        "ThermalGaussian_OMMG_full",
        "ThermalGaussian_MSMG_full",
        "ThermalGaussian_MFTG_full",
    ],
}
THRESHOLD_METRICS = [
    "FrontIntrusionRate",
    "FrontIntrusionMagnitude",
    "TooDeepRate",
    "DepthAgreementRate",
]
SECONDARY_METRICS = [
    "ModelValidOnReferenceRate",
    "MissingRate",
    "AbsDepthError_Mean",
    "AbsDepthError_Median",
    "SignedDepthBias_Mean",
]
OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER = (
    "CUDA mesh refinement path completed; CPU fallback disabled"
)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Package 5-scene x 8-method depth-reference evaluation results.")
    ap.add_argument(
        "--building_root",
        required=True,
    )
    ap.add_argument(
        "--formal4_root",
        required=True,
    )
    ap.add_argument(
        "--repo_root",
        default=str(Path(__file__).resolve().parents[2]),
    )
    ap.add_argument(
        "--out_root",
        required=True,
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=220,
    )
    return ap


def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def _current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdefABCDEF" for char in commit):
        raise RuntimeError(f"Invalid repository commit: {commit!r}")
    return commit


def _assert_producer_identity(identity: object, *, script_path: Path, label: str) -> None:
    if not isinstance(identity, dict):
        raise RuntimeError(f"Missing producer identity for {label}")
    if (
        not _same_resolved_path(str(identity.get("script_path", "")), script_path)
        or str(identity.get("script_sha256", "")) != _sha256_file(script_path)
        or str(identity.get("git_commit", "")) != _current_git_commit()
        or identity.get("git_dirty") is not False
        or str(identity.get("git_error", ""))
    ):
        raise RuntimeError(f"Producer identity does not match the current clean-code contract for {label}")


def _ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        _copy_file(src, dst)


def _method_rank(series: pd.Series) -> pd.Series:
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    return series.map(lambda x: order.get(x, len(order)))


def _scene_rank(series: pd.Series) -> pd.Series:
    order = {name: idx for idx, name in enumerate(SCENE_ORDER)}
    return series.map(lambda x: order.get(x, len(order)))


def _collect_scene_roots(building_root: Path, formal4_root: Path) -> Dict[str, Path]:
    scene_roots = {"Building": building_root}
    for scene_name in SCENE_ORDER[1:]:
        scene_roots[scene_name] = formal4_root / scene_name
    return scene_roots


def _validate_openmvs_reference(scene_name: str, scene_root: Path) -> Path:
    reference_root = scene_root / "reference_openmvs_v1"
    build_path = reference_root / "reference_build_manifest.json"
    reference_path = reference_root / "reference_depth_manifest.json"
    if not build_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError(
            f"Missing isolated OpenMVS reference manifests for {scene_name}: {reference_root}"
        )
    build = _load_json(build_path)
    overrides = build.get("reference_construction_overrides", {})
    if str(build.get("scene_name", "")) != scene_name:
        raise RuntimeError(f"OpenMVS reference scene mismatch for {scene_name}: {build_path}")
    if str(build.get("reference_construction_protocol", "")) != "openmvs-reference-mesh-v1":
        raise RuntimeError(f"Non-OpenMVS reference protocol for {scene_name}: {build_path}")
    if str(build.get("reference_dense_backend", "")) != "openmvs_densify_point_cloud":
        raise RuntimeError(f"Unexpected dense backend for {scene_name}: {build_path}")
    if str(build.get("reference_mesh_backend", "")) not in {
        "openmvs_reconstruct_mesh",
        "openmvs_refine_mesh",
    }:
        raise RuntimeError(f"Unexpected mesh backend for {scene_name}: {build_path}")
    if str(overrides.get("reference_geometry_backend", "")) != "openmvs":
        raise RuntimeError(f"OpenMVS backend declaration missing for {scene_name}: {build_path}")
    if bool(overrides.get("colmap_mvs_fallback_allowed", True)):
        raise RuntimeError(f"COLMAP-MVS fallback is not explicitly forbidden for {scene_name}")
    if int(overrides.get("openmvs_archive_type", 0)) != -1:
        raise RuntimeError(f"OpenMVS interface-archive contract is not frozen for {scene_name}")
    if bool(overrides.get("openmvs_interface_normalize", True)):
        raise RuntimeError(f"OpenMVS coordinate normalization must remain disabled for {scene_name}")
    if not bool(overrides.get("openmvs_cuda_log_evidence_required", False)):
        raise RuntimeError(f"Verified OpenMVS CUDA evidence is not required for {scene_name}")
    refine_enabled = bool(overrides.get("openmvs_refine_mesh", False))
    if refine_enabled and (
        not bool(overrides.get("openmvs_refine_cuda_fail_closed_required", False))
        or str(overrides.get("openmvs_refine_cuda_fail_closed_marker", ""))
        != OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER
    ):
        raise RuntimeError(f"Fail-closed CUDA RefineMesh is not required for {scene_name}")
    evidence = build.get("openmvs_cuda_evidence", {})
    stages = evidence.get("stages", {})
    required_stages = {"densify_point_cloud", "reconstruct_mesh"}
    if refine_enabled:
        required_stages.add("refine_mesh")
    if evidence.get("status") != "verified" or not isinstance(stages, dict):
        raise RuntimeError(f"OpenMVS CUDA evidence is not verified for {scene_name}")
    missing_stages = sorted(required_stages - set(stages))
    if missing_stages:
        raise RuntimeError(f"OpenMVS CUDA evidence is incomplete for {scene_name}: {missing_stages}")
    expected_device = int(overrides.get("openmvs_cuda_device", -1))
    for stage_name in required_stages:
        row = stages[stage_name]
        if int(row.get("expected_cuda_device", -1)) != expected_device:
            raise RuntimeError(f"OpenMVS CUDA-device evidence mismatch for {scene_name}/{stage_name}")
        log_path = Path(str(row.get("log_path", "")))
        expected_sha = str(row.get("log_sha256", ""))
        expected_size = int(row.get("log_size_bytes", -1))
        if not log_path.is_file() or not expected_sha:
            raise FileNotFoundError(f"OpenMVS CUDA evidence log is missing: {scene_name}/{stage_name}")
        if _sha256_file(log_path) != expected_sha or int(log_path.stat().st_size) != expected_size:
            raise RuntimeError(f"OpenMVS CUDA evidence changed for {scene_name}/{stage_name}")
        if stage_name == "refine_mesh":
            if row.get("cuda_fallback_fail_closed") is not True:
                raise RuntimeError(f"RefineMesh CUDA fallback is not fail-closed for {scene_name}")
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER not in log_text:
                raise RuntimeError(f"RefineMesh fail-closed completion marker is missing for {scene_name}")
    mesh_path = Path(str(build.get("reference_mesh_path", "")))
    if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
        raise FileNotFoundError(f"OpenMVS reference mesh is missing or empty for {scene_name}: {mesh_path}")
    if (
        str(build.get("reference_mesh_sha256", "")) != _sha256_file(mesh_path)
        or int(build.get("reference_mesh_size_bytes", -1)) != int(mesh_path.stat().st_size)
    ):
        raise RuntimeError(f"OpenMVS reference mesh identity changed for {scene_name}: {mesh_path}")
    dense_path = Path(str(build.get("reference_dense_ply", "")))
    if (
        not dense_path.is_file()
        or str(build.get("reference_dense_ply_sha256", "")) != _sha256_file(dense_path)
        or int(build.get("reference_dense_ply_size_bytes", -1)) != int(dense_path.stat().st_size)
    ):
        raise RuntimeError(f"OpenMVS dense reference identity changed for {scene_name}: {dense_path}")
    plan_path = Path(str(build.get("openmvs_command_plan", "")))
    if not plan_path.is_file():
        raise FileNotFoundError(f"OpenMVS command plan is missing for {scene_name}: {plan_path}")
    if str(build.get("openmvs_command_plan_sha256", "")) != _sha256_file(plan_path):
        raise RuntimeError(f"OpenMVS command plan identity changed for {scene_name}: {plan_path}")
    receipt_rows = build.get("openmvs_stage_receipts", {})
    required_receipts = {"interface_colmap", "densify_point_cloud", "reconstruct_mesh"}
    if bool(overrides.get("openmvs_refine_mesh", False)):
        required_receipts.add("refine_mesh")
    if not isinstance(receipt_rows, dict) or not required_receipts.issubset(set(receipt_rows)):
        raise RuntimeError(f"OpenMVS stage receipts are incomplete for {scene_name}")
    for stage_name, row in receipt_rows.items():
        receipt_path = Path(str(row.get("path", "")))
        if (
            not receipt_path.is_file()
            or str(row.get("sha256", "")) != _sha256_file(receipt_path)
            or int(row.get("size_bytes", -1)) != int(receipt_path.stat().st_size)
        ):
            raise RuntimeError(f"OpenMVS stage receipt changed for {scene_name}/{stage_name}")
    reference = _load_json(reference_path)
    if str(reference.get("reference_construction_protocol", "")) != "openmvs-reference-mesh-v1":
        raise RuntimeError(f"Reference-depth manifest is not OpenMVS-backed for {scene_name}")
    if not _same_resolved_path(reference.get("reference_mesh_path", ""), mesh_path):
        raise RuntimeError(f"Reference-depth/build manifests disagree on the mesh for {scene_name}")
    if str(reference.get("reference_mesh_sha256", "")) != str(build.get("reference_mesh_sha256", "")):
        raise RuntimeError(f"Reference-depth/build manifests disagree on mesh SHA for {scene_name}")
    for view in reference.get("views", []):
        view_path = reference_root / str(view.get("npz_file", ""))
        if (
            not view_path.is_file()
            or str(view.get("npz_sha256", "")) != _sha256_file(view_path)
            or int(view.get("npz_size_bytes", -1)) != int(view_path.stat().st_size)
        ):
            raise RuntimeError(f"Reference view identity changed for {scene_name}: {view_path}")
    return reference_path.resolve()


def _same_resolved_path(left: str | Path, right: str | Path) -> bool:
    return str(Path(left).resolve()).replace("\\", "/").casefold() == str(Path(right).resolve()).replace("\\", "/").casefold()


def _collect_metrics(
    scene_roots: Dict[str, Path],
    reference_manifests: Dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: List[Dict] = []
    secondary_rows: List[Dict] = []

    for scene_name in SCENE_ORDER:
        scene_root = scene_roots[scene_name]
        for method_name in METHOD_ORDER:
            metrics_path = scene_root / method_name / "evaluation" / "metrics_summary.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics_summary.json: {metrics_path}")
            payload = _load_json(metrics_path)
            if str(payload.get("scene_name", "")) != scene_name:
                raise RuntimeError(f"Metric scene mismatch: {metrics_path}")
            if str(payload.get("method_name", "")) != method_name:
                raise RuntimeError(f"Metric method mismatch: {metrics_path}")
            if str(payload.get("protocol_name", "")) != "reference-depth-based-geometric-evaluation-v1":
                raise RuntimeError(f"Metric protocol mismatch: {metrics_path}")
            _assert_producer_identity(
                payload.get("producer_identity"),
                script_path=TOOL_DIR / "evaluate_depth_reference.py",
                label=f"metrics {scene_name}/{method_name}",
            )
            model_manifest_path = scene_root / method_name / "bundle" / "split_manifest.json"
            model_manifest_payload = _load_json(model_manifest_path)
            _assert_producer_identity(
                model_manifest_payload.get("producer_identity"),
                script_path=TOOL_DIR / "export_gaussian_probe_bundle.py",
                label=f"model bundle {scene_name}/{method_name}",
            )
            expected_inputs = (
                (
                    "reference_manifest",
                    "reference_manifest_sha256",
                    reference_manifests[scene_name],
                ),
                (
                    "model_manifest",
                    "model_manifest_sha256",
                    model_manifest_path,
                ),
                (
                    "adapter_manifest",
                    "adapter_manifest_sha256",
                    scene_root / method_name / "depth_adapter_manifest.json",
                ),
            )
            for path_field, sha_field, expected_path in expected_inputs:
                expected_path = expected_path.resolve()
                if not expected_path.is_file():
                    raise FileNotFoundError(
                        f"Current {path_field} is missing for {scene_name}/{method_name}: {expected_path}"
                    )
                actual_path = str(payload.get(path_field, ""))
                if not actual_path or not _same_resolved_path(actual_path, expected_path):
                    raise RuntimeError(
                        f"Metrics were not produced from the current {path_field}: "
                        f"{metrics_path} -> {actual_path!r}"
                    )
                if str(payload.get(sha_field, "")).lower() != _sha256_file(expected_path):
                    raise RuntimeError(
                        f"Metrics {sha_field} does not match the current input for "
                        f"{scene_name}/{method_name}: {metrics_path}"
                    )
            if payload.get("evaluation_options", {}).get("enable_agreement_metrics") is not True:
                raise RuntimeError(f"Agreement metrics are not enabled in {metrics_path}")
            counts = payload["counts"]
            secondary = payload["secondary_metrics"]
            secondary_row = {
                "scene_name": scene_name,
                "method_name": method_name,
                "source_scene_root": str(scene_root),
                "reference_valid_pixels": int(counts["reference_valid_pixels"]),
                "model_valid_on_reference_pixels": int(counts["model_valid_on_reference_pixels"]),
                "missing_pixels": int(counts["missing_pixels"]),
                "metrics_summary_json": str(metrics_path),
            }
            for metric_name in SECONDARY_METRICS:
                secondary_row[metric_name] = float(secondary[metric_name])
            secondary_rows.append(secondary_row)

            threshold_metrics = payload["threshold_metrics"]
            for record in threshold_metrics:
                row = {
                    "scene_name": scene_name,
                    "method_name": method_name,
                    "source_scene_root": str(scene_root),
                    "threshold_m": float(record["threshold_m"]),
                    "reference_valid_pixels": int(counts["reference_valid_pixels"]),
                    "model_valid_on_reference_pixels": int(counts["model_valid_on_reference_pixels"]),
                    "missing_pixels": int(counts["missing_pixels"]),
                    "metrics_summary_json": str(metrics_path),
                }
                for metric_name in THRESHOLD_METRICS:
                    row[metric_name] = float(record.get(metric_name, math.nan))
                threshold_rows.append(row)

    threshold_df = pd.DataFrame(threshold_rows)
    secondary_df = pd.DataFrame(secondary_rows)
    threshold_df["scene_name"] = pd.Categorical(threshold_df["scene_name"], categories=SCENE_ORDER, ordered=True)
    threshold_df["method_name"] = pd.Categorical(threshold_df["method_name"], categories=METHOD_ORDER, ordered=True)
    secondary_df["scene_name"] = pd.Categorical(secondary_df["scene_name"], categories=SCENE_ORDER, ordered=True)
    secondary_df["method_name"] = pd.Categorical(secondary_df["method_name"], categories=METHOD_ORDER, ordered=True)
    threshold_df = threshold_df.sort_values(["scene_name", "method_name", "threshold_m"]).reset_index(drop=True)
    secondary_df = secondary_df.sort_values(["scene_name", "method_name"]).reset_index(drop=True)
    return threshold_df, secondary_df


def _macro_average_thresholds(threshold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = THRESHOLD_METRICS + ["reference_valid_pixels", "model_valid_on_reference_pixels", "missing_pixels"]
    grouped = threshold_df.groupby(["method_name", "threshold_m"], observed=True)[metric_cols].mean().reset_index()
    grouped["scene_count"] = threshold_df.groupby(["method_name", "threshold_m"], observed=True).size().reset_index(drop=True)
    grouped["method_name"] = pd.Categorical(grouped["method_name"], categories=METHOD_ORDER, ordered=True)
    return grouped.sort_values(["method_name", "threshold_m"]).reset_index(drop=True)


def _macro_average_secondary(secondary_df: pd.DataFrame) -> pd.DataFrame:
    mean_cols = SECONDARY_METRICS + ["reference_valid_pixels", "model_valid_on_reference_pixels", "missing_pixels"]
    mean_df = secondary_df.groupby(["method_name"], observed=True)[mean_cols].mean().reset_index()
    sum_df = secondary_df.groupby(["method_name"], observed=True)[["reference_valid_pixels", "model_valid_on_reference_pixels", "missing_pixels"]].sum().reset_index()
    out = mean_df.merge(
        sum_df.rename(
            columns={
                "reference_valid_pixels": "reference_valid_pixels_sum",
                "model_valid_on_reference_pixels": "model_valid_on_reference_pixels_sum",
                "missing_pixels": "missing_pixels_sum",
            }
        ),
        on="method_name",
        how="left",
    )
    out["scene_count"] = secondary_df.groupby(["method_name"], observed=True).size().reset_index(drop=True)
    out["method_name"] = pd.Categorical(out["method_name"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["method_name"]).reset_index(drop=True)


def _make_rankings(
    threshold_df: pd.DataFrame,
    macro_threshold_df: pd.DataFrame,
    rank_threshold: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    scene_at_rank = threshold_df[threshold_df["threshold_m"] == rank_threshold].copy()
    scene_front = scene_at_rank.sort_values(["scene_name", "FrontIntrusionRate", "method_name"]).reset_index(drop=True)
    scene_front["rank_in_scene"] = scene_front.groupby("scene_name", observed=True)["FrontIntrusionRate"].rank(method="min", ascending=True).astype(int)
    scene_agreement = scene_at_rank.sort_values(["scene_name", "DepthAgreementRate", "method_name"], ascending=[True, False, True]).reset_index(drop=True)
    scene_agreement["rank_in_scene"] = scene_agreement.groupby("scene_name", observed=True)["DepthAgreementRate"].rank(method="min", ascending=False).astype(int)

    macro_at_rank = macro_threshold_df[macro_threshold_df["threshold_m"] == rank_threshold].copy()
    macro_front = macro_at_rank.sort_values(["FrontIntrusionRate", "method_name"]).reset_index(drop=True)
    macro_front["rank_macro"] = range(1, len(macro_front) + 1)
    macro_agreement = macro_at_rank.sort_values(["DepthAgreementRate", "method_name"], ascending=[False, True]).reset_index(drop=True)
    macro_agreement["rank_macro"] = range(1, len(macro_agreement) + 1)

    all_threshold_front = threshold_df.sort_values(["scene_name", "threshold_m", "FrontIntrusionRate", "method_name"]).reset_index(drop=True)
    all_threshold_front["rank_in_scene_threshold"] = all_threshold_front.groupby(["scene_name", "threshold_m"], observed=True)["FrontIntrusionRate"].rank(method="min", ascending=True).astype(int)
    all_threshold_agreement = threshold_df.sort_values(["scene_name", "threshold_m", "DepthAgreementRate", "method_name"], ascending=[True, True, False, True]).reset_index(drop=True)
    all_threshold_agreement["rank_in_scene_threshold"] = all_threshold_agreement.groupby(["scene_name", "threshold_m"], observed=True)["DepthAgreementRate"].rank(method="min", ascending=False).astype(int)

    macro_all_front = macro_threshold_df.sort_values(["threshold_m", "FrontIntrusionRate", "method_name"]).reset_index(drop=True)
    macro_all_front["rank_macro_threshold"] = macro_all_front.groupby(["threshold_m"], observed=True)["FrontIntrusionRate"].rank(method="min", ascending=True).astype(int)
    macro_all_agreement = macro_threshold_df.sort_values(["threshold_m", "DepthAgreementRate", "method_name"], ascending=[True, False, True]).reset_index(drop=True)
    macro_all_agreement["rank_macro_threshold"] = macro_all_agreement.groupby(["threshold_m"], observed=True)["DepthAgreementRate"].rank(method="min", ascending=False).astype(int)

    return {
        "scene_front_1m": scene_front,
        "scene_agreement_1m": scene_agreement,
        "macro_front_1m": macro_front,
        "macro_agreement_1m": macro_agreement,
        "scene_front_all_thresholds": all_threshold_front,
        "scene_agreement_all_thresholds": all_threshold_agreement,
        "macro_front_all_thresholds": macro_all_front,
        "macro_agreement_all_thresholds": macro_all_agreement,
    }


def _plot_scene_panels(threshold_df: pd.DataFrame, metric: str, ylabel: str, out_png: Path, dpi: int) -> None:
    raise NotImplementedError


def _prepare_equal_spaced_axis(df: pd.DataFrame) -> tuple[List[str], Dict[float, int]]:
    thresholds = sorted(float(x) for x in df["threshold_m"].unique().tolist())
    threshold_labels = [f"{threshold:.2f}" for threshold in thresholds]
    position_map = {threshold: idx for idx, threshold in enumerate(thresholds)}
    return threshold_labels, position_map


def _plot_scene_panels_grouped(
    threshold_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    methods: Sequence[str],
    group_name: str,
    out_png: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), squeeze=False)
    axes_flat = axes.flatten()
    threshold_labels, position_map = _prepare_equal_spaced_axis(threshold_df)
    for ax, scene_name in zip(axes_flat, SCENE_ORDER):
        scene_df = threshold_df[threshold_df["scene_name"] == scene_name].copy()
        scene_df = scene_df.sort_values(["method_name", "threshold_m"])
        for method_name in methods:
            method_df = scene_df[scene_df["method_name"] == method_name]
            ax.plot(
                [position_map[float(x)] for x in method_df["threshold_m"].tolist()],
                method_df[metric],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=method_name,
            )
        ax.set_title(scene_name)
        ax.set_xlabel("Threshold (m, equally spaced)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(len(threshold_labels))))
        ax.set_xticklabels(threshold_labels)
        ax.grid(True, alpha=0.25)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].axis("off")
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False)
    fig.suptitle(f"{metric} Across 5 Scenes ({group_name}, 9 thresholds)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_macro_curve_grouped(
    macro_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    methods: Sequence[str],
    group_name: str,
    out_png: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    macro_df = macro_df.sort_values(["method_name", "threshold_m"])
    threshold_labels, position_map = _prepare_equal_spaced_axis(macro_df)
    for method_name in methods:
        method_df = macro_df[macro_df["method_name"] == method_name]
        ax.plot(
            [position_map[float(x)] for x in method_df["threshold_m"].tolist()],
            method_df[metric],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            label=method_name,
        )
    ax.set_xlabel("Threshold (m, equally spaced)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Macro-average {metric} Across 5 Scenes ({group_name})")
    ax.set_xticks(list(range(len(threshold_labels))))
    ax.set_xticklabels(threshold_labels)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_scene_metric_grid_grouped(
    scene_name: str,
    threshold_df: pd.DataFrame,
    methods: Sequence[str],
    group_name: str,
    out_png: Path,
    dpi: int,
) -> None:
    metric_specs = [
        ("FrontIntrusionRate", "FrontIntrusionRate"),
        ("FrontIntrusionMagnitude", "FrontIntrusionMagnitude (m)"),
        ("TooDeepRate", "TooDeepRate"),
        ("DepthAgreementRate", "DepthAgreementRate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
    scene_df = threshold_df[threshold_df["scene_name"] == scene_name].copy()
    scene_df = scene_df.sort_values(["method_name", "threshold_m"])
    threshold_labels, position_map = _prepare_equal_spaced_axis(scene_df)
    for ax, (metric, ylabel) in zip(axes.flatten(), metric_specs):
        for method_name in methods:
            method_df = scene_df[scene_df["method_name"] == method_name]
            ax.plot(
                [position_map[float(x)] for x in method_df["threshold_m"].tolist()],
                method_df[metric],
                marker="o",
                linewidth=2.0,
                markersize=4.0,
                label=method_name,
            )
        ax.set_title(metric)
        ax.set_xlabel("Threshold (m, equally spaced)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(range(len(threshold_labels))))
        ax.set_xticklabels(threshold_labels)
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"{scene_name}: All Threshold-dependent Metrics ({group_name})")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _copy_scene_artifacts(scene_name: str, scene_root: Path, dst_root: Path) -> None:
    target_scene_root = dst_root / "scene_artifacts" / scene_name
    target_scene_root.mkdir(parents=True, exist_ok=True)
    for scene_file in [
        "run_manifest.json",
        "reuse_manifest.csv",
        "scene_run_manifest.json",
    ]:
        _copy_if_exists(scene_root / scene_file, target_scene_root / scene_file)

    for summary_file in scene_root.glob("summary/*"):
        if summary_file.is_file():
            _copy_file(summary_file, target_scene_root / "summary" / summary_file.name)

    reference_root = scene_root / "reference_openmvs_v1"
    for reference_file in [
        reference_root / "reference_depth_manifest.json",
        reference_root / "reference_build_manifest.json",
        reference_root / "probe_camera_manifest.json",
        reference_root / "reference_roi.json",
        reference_root / "openmvs_command_plan.json",
        reference_root / "openmvs_cuda_evidence.json",
    ]:
        if reference_file.exists():
            _copy_file(reference_file, target_scene_root / "reference_openmvs_v1" / reference_file.name)
    build_manifest_path = reference_root / "reference_build_manifest.json"
    if build_manifest_path.is_file():
        build_manifest = _load_json(build_manifest_path)
        stages = build_manifest.get("openmvs_cuda_evidence", {}).get("stages", {})
        for stage_name, row in stages.items():
            source_log = Path(str(row.get("log_path", "")))
            expected_sha = str(row.get("log_sha256", ""))
            expected_size = int(row.get("log_size_bytes", -1))
            if (
                not source_log.is_file()
                or _sha256_file(source_log) != expected_sha
                or int(source_log.stat().st_size) != expected_size
            ):
                raise RuntimeError(f"Cannot package verified CUDA log for {scene_name}/{stage_name}")
            _copy_file(
                source_log,
                target_scene_root
                / "reference_openmvs_v1"
                / "openmvs_cuda_logs"
                / f"{stage_name}.log",
            )

    for method_name in METHOD_ORDER:
        method_root = scene_root / method_name
        if not method_root.exists():
            continue
        for rel_path in [
            Path("depth_adapter_manifest.json"),
            Path("bundle") / "split_manifest.json",
            Path("evaluation") / "metrics_summary.json",
            Path("evaluation") / "metrics_summary.csv",
            Path("evaluation") / "front_intrusion_curve.csv",
            Path("evaluation") / "per_view_counts.csv",
        ]:
            src_path = method_root / rel_path
            if src_path.exists():
                _copy_file(src_path, target_scene_root / method_name / rel_path)


def _copy_root_artifacts(building_root: Path, formal4_root: Path, dst_root: Path) -> None:
    targets = [
        (
            building_root,
            dst_root / "source_root_artifacts" / "Building_reuse_root",
            ["run_manifest.json", "reuse_manifest.csv"],
        ),
        (
            formal4_root,
            dst_root / "source_root_artifacts" / "Formal_4scene_root",
            [
                "batch_run_manifest.json",
                "depth_reference_runtime_flags.json",
                "status.json",
                "model_inventory_snapshot.csv",
                "launch_info.json",
                "resume_launch_info.json",
                "resume2_launch_info.json",
                "resume3_launch_info.json",
                "resume_after_reboot_launch_info.json",
                "resume_after_mesherfix_launch_info.json",
            ],
        ),
    ]
    for src_root, dst_subroot, names in targets:
        for name in names:
            _copy_if_exists(src_root / name, dst_subroot / name)


def _copy_code_snapshot(repo_root: Path, dst_root: Path) -> None:
    file_list = [
        repo_root / "tools" / "geometric_repeatability" / "build_depth_reference.py",
        repo_root / "tools" / "geometric_repeatability" / "openmvs_backend_sanity.py",
        repo_root / "tools" / "geometric_repeatability" / "evaluate_depth_reference.py",
        repo_root / "tools" / "geometric_repeatability" / "summarize_depth_reference_methods.py",
        repo_root / "tools" / "geometric_repeatability" / "depth_reference_common.py",
        repo_root / "tools" / "geometric_repeatability" / "export_gaussian_probe_bundle.py",
        repo_root / "tools" / "geometric_repeatability" / "run_depth_reference_formal_5scene_8method.ps1",
        repo_root / "tools" / "geometric_repeatability" / "DEPTH_REFERENCE_PROTOCOL.md",
        repo_root / "tools" / "geometric_repeatability" / "README.md",
        repo_root / "arguments" / "__init__.py",
    ]
    for src_path in file_list:
        relative = src_path.relative_to(repo_root)
        _copy_file(src_path, dst_root / "code_snapshot" / relative)


def _write_readme(out_root: Path, building_root: Path, formal4_root: Path) -> None:
    readme_text = f"""# Depth-Reference Evaluation Result Pack

This package contains the non-large-file deliverables for the 5-scene x 8-method
reference-depth-based geometric evaluation. It is intended for researchers to
inspect the full numeric outputs, threshold curves, protocol files, and code
snapshots without having to scan the large fused point clouds, meshes, or per-view
NPZ tensors stored in the original experiment roots.

## 1. Scope

- Scenes: Building, PVpanel, Orchard, Road, TransmissionTower
- Methods:
  - Ours_M00_full
  - Ours_G01_full
  - Ours_G02_full
  - Ours_M01_full
  - Thermal3D_GS_full
  - ThermalGaussian_OMMG_full
  - ThermalGaussian_MSMG_full
  - ThermalGaussian_MFTG_full
- Thresholds (meters): 0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 30.00

## 2. What Is Being Evaluated

The protocol measures held-out thermal-view geometry consistency against a
training-only RGB MVS reference. The evaluation does **not** claim absolute
ground-truth geometry accuracy. Instead, it asks:

1. Given a training-only RGB dense reconstruction, what depth should a held-out
   thermal camera observe?
2. When a method renders its own held-out thermal depth for that same camera,
   how often is the rendered surface too close, too far, or within tolerance?

The thresholds are **absolute physical tolerances in meters**. They are not
normalized by scene diagonal.

## 3. Symbols

For each held-out probe pixel `p`:

- `D_ref(p)`: reference metric camera-z depth from the training-only RGB mesh
- `D_model(p)`: rendered metric camera-z depth from the evaluated thermal model
- `M_ref(p)`: binary reference-valid mask
- `M_model(p)`: binary model-valid mask
- `delta`: a depth tolerance in meters

Reference-valid pixels are the denominator for the main rates:

- `V_ref = {{ p | M_ref(p) = 1 }}`
- `V_joint = {{ p | M_ref(p) = 1 and M_model(p) = 1 }}`

## 4. Reference Construction

The reference is built from training-side RGB data only:

1. Use the strict protocol manifest for each scene.
2. Use the aligned training-union RGB COLMAP model to define the shared frame.
3. Import that model with OpenMVS `InterfaceCOLMAP` using `normalize=0`.
4. Run OpenMVS `DensifyPointCloud`, `ReconstructMesh`, and `RefineMesh`.
5. Render held-out probe-view reference depths from the refined OpenMVS mesh.

COLMAP MVS and COLMAP meshers are not fallback backends. `TextureMesh` is not
needed because the evaluator consumes geometry rather than mesh texture.

### ROI construction

The region of interest is derived from the **training-only fused dense points**
using a robust quantile AABB:

- lower quantile = 0.01
- upper quantile = 0.99
- padding ratio = 0.02 of the robust diagonal

### Reference support rule

A reference pixel is considered valid only if all of the following hold:

1. mesh depth is finite and positive
2. the back-projected 3D point lies inside the ROI
3. projected support count from the fused dense cloud is at least 1

Support rendering uses:

- support radius = 1 pixel
- support depth tolerance = 0.10 m

## 5. Model Depth Extraction

For each method, held-out thermal views are rendered from the model using the
same probe cameras as the reference.

### Depth semantics

The Gaussian renderer outputs `inverse_camera_z_from_renderer`. The evaluator
converts it into metric depth by:

- `D_model(p) = 1 / raw_depth(p)` for positive finite raw depth
- otherwise mark as invalid

### Model validity rule

A model pixel is valid if:

1. converted metric depth is finite
2. converted metric depth > `1e-6`
3. opacity >= `0.5`

This rule is frozen in each `depth_adapter_manifest.json`.

## 6. Metric Definitions

All rates below use `|V_ref|` as the denominator.

### 6.1 FrontIntrusionRate at threshold `delta`

Pixels where the rendered surface is too close to the camera:

`FrontIntrusionRate(delta) = |{{ p in V_joint : D_model(p) < D_ref(p) - delta }}| / |V_ref|`

Smaller is better.

### 6.2 FrontIntrusionMagnitude at threshold `delta`

Mean severity of the front-intruding pixels:

`FrontIntrusionMagnitude(delta) = mean( D_ref(p) - D_model(p) )`

computed only over pixels that satisfy
`D_model(p) < D_ref(p) - delta`.

Smaller is better.

### 6.3 TooDeepRate at threshold `delta`

Pixels where the rendered surface is too far from the camera:

`TooDeepRate(delta) = |{{ p in V_joint : D_model(p) > D_ref(p) + delta }}| / |V_ref|`

Smaller is better.

### 6.4 DepthAgreementRate at threshold `delta`

Pixels whose rendered depth falls within the tolerance band:

`DepthAgreementRate(delta) = |{{ p in V_joint : |D_model(p) - D_ref(p)| <= delta }}| / |V_ref|`

Larger is better.

### 6.5 MissingRate

Reference-valid pixels where the model has no valid rendered depth:

`MissingRate = |{{ p in V_ref : M_model(p) = 0 }}| / |V_ref|`

Smaller is better.

### 6.6 ModelValidOnReferenceRate

Reference-valid pixels where the model does produce a valid depth:

`ModelValidOnReferenceRate = |V_joint| / |V_ref|`

Larger is better.

### 6.7 AbsDepthError and SignedDepthBias

These are computed over `V_joint` only.

- `AbsDepthError_Mean = mean( |D_model(p) - D_ref(p)| )`
- `AbsDepthError_Median = median( |D_model(p) - D_ref(p)| )`
- `SignedDepthBias_Mean = mean( D_model(p) - D_ref(p) )`

Interpretation:

- positive signed bias: the model tends to be farther than the reference
- negative signed bias: the model tends to be closer than the reference

## 7. Package Layout

- `tables/`
  - merged CSV tables across scenes, methods, and thresholds
  - macro-average tables
  - threshold-specific ranking tables
- `plots/`
  - 9-threshold curves for each metric
  - per-scene panels and macro-average plots
- `scene_artifacts/`
  - per-scene summary CSV files
  - per-method `metrics_summary.json/csv`, `front_intrusion_curve.csv`,
    `per_view_counts.csv`, `depth_adapter_manifest.json`,
    and bundle/reference manifests
- `code_snapshot/`
  - code files used to produce the evaluation

## 8. Key Tables

- `tables/scene_method_threshold_metrics_long.csv`
  - one row per scene x method x threshold
- `tables/scene_method_secondary_metrics.csv`
  - one row per scene x method with non-thresholded metrics
- `tables/macro_average_threshold_metrics.csv`
  - macro-average across the 5 scenes for each method x threshold
- `tables/macro_average_secondary_metrics.csv`
  - macro-average across scenes for the secondary metrics
- `tables/scene_front_ranking_at_1.00m.csv`
  - per-scene ranking by FrontIntrusionRate at 1.00 m
- `tables/scene_agreement_ranking_at_1.00m.csv`
  - per-scene ranking by DepthAgreementRate at 1.00 m
- `tables/macro_front_ranking_at_1.00m.csv`
  - macro-average ranking by FrontIntrusionRate at 1.00 m
- `tables/macro_agreement_ranking_at_1.00m.csv`
  - macro-average ranking by DepthAgreementRate at 1.00 m

## 9. Key Plots

- `plots/ablation_4method/macro_average/`
  - macro-average curves for the 4 ablation methods
- `plots/ablation_4method/per_scene_panels/`
  - 5-scene panels for the 4 ablation methods
- `plots/ablation_4method/per_scene_metric_grids/`
  - one 2x2 metric grid per scene for the 4 ablation methods
- `plots/sota_5method/macro_average/`
  - macro-average curves for the SOTA-style 5-method comparison
- `plots/sota_5method/per_scene_panels/`
  - 5-scene panels for the SOTA-style 5-method comparison
- `plots/sota_5method/per_scene_metric_grids/`
  - one 2x2 metric grid per scene for the SOTA-style 5-method comparison

### Plotting note

The x-axis is rendered using **equally spaced threshold categories** rather than
true metric spacing, so that small thresholds (`0.10`, `0.25`, `0.50`, `1.00`)
remain visually distinguishable from the large thresholds (`10`, `20`, `30`).
The tick labels still show the true meter values.

## 10. Source Experiment Roots

The packaged files were copied from:

- Building reuse root:
  `{building_root}`
- Formal 4-scene root:
  `{formal4_root}`

## 11. Deliberately Excluded Large Files

The following were intentionally not copied into this result pack:

- fused dense point clouds (`*.ply`, `*.ply.vis`) used as heavy intermediate files
- mesh geometry files (`reference_mesh_*.ply`)
- per-view rendered NPZ tensors under `reference/views/` and `bundle/views/`
- large COLMAP workspaces, dense maps, and raw run logs not needed for paper writing

If those raw heavy artifacts are needed, they should be
retrieved from the original experiment roots listed above.

## 12. Recommended Reading Order For Writing

1. Start with `tables/macro_average_threshold_metrics.csv`.
2. Check `plots/macro_average/` to inspect trend consistency across the 9 thresholds.
3. Inspect `tables/scene_front_ranking_at_1.00m.csv` and
   `tables/scene_agreement_ranking_at_1.00m.csv`.
4. Use `plots/per_scene_metric_grids/` to understand scene-specific behavior.
5. If needed, drill down into `scene_artifacts/<scene>/<method>/evaluation/`.
"""
    (out_root / "README.md").write_text(readme_text, encoding="utf-8")


def _write_tables(
    out_root: Path,
    threshold_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    macro_threshold_df: pd.DataFrame,
    macro_secondary_df: pd.DataFrame,
    rankings: Dict[str, pd.DataFrame],
) -> None:
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(tables_dir / "scene_method_threshold_metrics_long.csv", index=False)
    secondary_df.to_csv(tables_dir / "scene_method_secondary_metrics.csv", index=False)
    macro_threshold_df.to_csv(tables_dir / "macro_average_threshold_metrics.csv", index=False)
    macro_secondary_df.to_csv(tables_dir / "macro_average_secondary_metrics.csv", index=False)
    rankings["scene_front_1m"].to_csv(tables_dir / "scene_front_ranking_at_1.00m.csv", index=False)
    rankings["scene_agreement_1m"].to_csv(tables_dir / "scene_agreement_ranking_at_1.00m.csv", index=False)
    rankings["macro_front_1m"].to_csv(tables_dir / "macro_front_ranking_at_1.00m.csv", index=False)
    rankings["macro_agreement_1m"].to_csv(tables_dir / "macro_agreement_ranking_at_1.00m.csv", index=False)
    rankings["scene_front_all_thresholds"].to_csv(tables_dir / "scene_front_rankings_all_thresholds.csv", index=False)
    rankings["scene_agreement_all_thresholds"].to_csv(tables_dir / "scene_agreement_rankings_all_thresholds.csv", index=False)
    rankings["macro_front_all_thresholds"].to_csv(tables_dir / "macro_front_rankings_all_thresholds.csv", index=False)
    rankings["macro_agreement_all_thresholds"].to_csv(tables_dir / "macro_agreement_rankings_all_thresholds.csv", index=False)
    (
        secondary_df[
            [
                "scene_name",
                "method_name",
                "source_scene_root",
                "metrics_summary_json",
            ]
        ]
        .sort_values(["scene_name", "method_name"])
        .reset_index(drop=True)
        .to_csv(tables_dir / "method_scene_source_index.csv", index=False)
    )
    for group_name, methods in METHOD_GROUPS.items():
        group_threshold = threshold_df[threshold_df["method_name"].isin(methods)].copy()
        group_secondary = secondary_df[secondary_df["method_name"].isin(methods)].copy()
        group_macro_threshold = macro_threshold_df[macro_threshold_df["method_name"].isin(methods)].copy()
        group_macro_secondary = macro_secondary_df[macro_secondary_df["method_name"].isin(methods)].copy()
        group_threshold.to_csv(tables_dir / f"scene_method_threshold_metrics_{group_name}.csv", index=False)
        group_secondary.to_csv(tables_dir / f"scene_method_secondary_metrics_{group_name}.csv", index=False)
        group_macro_threshold.to_csv(tables_dir / f"macro_average_threshold_metrics_{group_name}.csv", index=False)
        group_macro_secondary.to_csv(tables_dir / f"macro_average_secondary_metrics_{group_name}.csv", index=False)


def _write_plots(out_root: Path, threshold_df: pd.DataFrame, macro_threshold_df: pd.DataFrame, dpi: int) -> None:
    plot_specs = [
        ("FrontIntrusionRate", "FrontIntrusionRate"),
        ("FrontIntrusionMagnitude", "FrontIntrusionMagnitude (m)"),
        ("TooDeepRate", "TooDeepRate"),
        ("DepthAgreementRate", "DepthAgreementRate"),
    ]
    for group_name, methods in METHOD_GROUPS.items():
        group_threshold = threshold_df[threshold_df["method_name"].isin(methods)].copy()
        group_macro = macro_threshold_df[macro_threshold_df["method_name"].isin(methods)].copy()
        for metric, ylabel in plot_specs:
            _plot_scene_panels_grouped(
                threshold_df=group_threshold,
                metric=metric,
                ylabel=ylabel,
                methods=methods,
                group_name=group_name,
                out_png=out_root / "plots" / group_name / "per_scene_panels" / f"{metric}_5scene_panel.png",
                dpi=dpi,
            )
            _plot_macro_curve_grouped(
                macro_df=group_macro,
                metric=metric,
                ylabel=ylabel,
                methods=methods,
                group_name=group_name,
                out_png=out_root / "plots" / group_name / "macro_average" / f"{metric}_macro_average.png",
                dpi=dpi,
            )

        for scene_name in SCENE_ORDER:
            _plot_scene_metric_grid_grouped(
                scene_name=scene_name,
                threshold_df=group_threshold,
                methods=methods,
                group_name=group_name,
                out_png=out_root / "plots" / group_name / "per_scene_metric_grids" / f"{scene_name}_all_metrics.png",
                dpi=dpi,
            )


def _write_summary_json(out_root: Path, threshold_df: pd.DataFrame, secondary_df: pd.DataFrame) -> None:
    summary = {
        "protocol_name": "reference-depth-based-geometric-evaluation-v1",
        "scene_count": int(secondary_df["scene_name"].nunique()),
        "method_count": int(secondary_df["method_name"].nunique()),
        "threshold_count": int(threshold_df["threshold_m"].nunique()),
        "scene_names": SCENE_ORDER,
        "method_names": METHOD_ORDER,
        "thresholds_m": sorted(float(x) for x in threshold_df["threshold_m"].unique().tolist()),
        "row_counts": {
            "threshold_rows": int(len(threshold_df)),
            "secondary_rows": int(len(secondary_df)),
        },
    }
    (out_root / "package_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _zip_out_root(out_root: Path) -> Path:
    zip_base = out_root.with_suffix("")
    zip_path = Path(str(zip_base) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    archive = shutil.make_archive(str(zip_base), "zip", root_dir=out_root.parent, base_dir=out_root.name)
    return Path(archive)


def main() -> None:
    args = _build_argparser().parse_args()
    building_root = Path(args.building_root).resolve()
    formal4_root = Path(args.formal4_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out_root).resolve()

    _ensure_clean_dir(out_root)
    scene_roots = _collect_scene_roots(building_root=building_root, formal4_root=formal4_root)
    reference_manifests = {
        scene_name: _validate_openmvs_reference(scene_name, scene_root)
        for scene_name, scene_root in scene_roots.items()
    }
    threshold_df, secondary_df = _collect_metrics(
        scene_roots=scene_roots,
        reference_manifests=reference_manifests,
    )
    macro_threshold_df = _macro_average_thresholds(threshold_df=threshold_df)
    macro_secondary_df = _macro_average_secondary(secondary_df=secondary_df)
    rankings = _make_rankings(
        threshold_df=threshold_df,
        macro_threshold_df=macro_threshold_df,
        rank_threshold=1.0,
    )

    _write_tables(
        out_root=out_root,
        threshold_df=threshold_df,
        secondary_df=secondary_df,
        macro_threshold_df=macro_threshold_df,
        macro_secondary_df=macro_secondary_df,
        rankings=rankings,
    )
    _write_plots(
        out_root=out_root,
        threshold_df=threshold_df,
        macro_threshold_df=macro_threshold_df,
        dpi=int(args.dpi),
    )
    for scene_name, scene_root in scene_roots.items():
        _copy_scene_artifacts(scene_name=scene_name, scene_root=scene_root, dst_root=out_root)
    _copy_root_artifacts(building_root=building_root, formal4_root=formal4_root, dst_root=out_root)
    _copy_code_snapshot(repo_root=repo_root, dst_root=out_root)
    _write_summary_json(out_root=out_root, threshold_df=threshold_df, secondary_df=secondary_df)
    _write_readme(out_root=out_root, building_root=building_root, formal4_root=formal4_root)
    zip_path = _zip_out_root(out_root=out_root)
    print(f"RESULT_PACK_READY {out_root}")
    print(f"RESULT_PACK_ZIP {zip_path}")


if __name__ == "__main__":
    main()
