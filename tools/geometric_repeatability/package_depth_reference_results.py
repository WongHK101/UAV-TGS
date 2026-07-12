from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


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


def _collect_metrics(scene_roots: Dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: List[Dict] = []
    secondary_rows: List[Dict] = []

    for scene_name in SCENE_ORDER:
        scene_root = scene_roots[scene_name]
        for method_name in METHOD_ORDER:
            metrics_path = scene_root / method_name / "evaluation" / "metrics_summary.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics_summary.json: {metrics_path}")
            payload = _load_json(metrics_path)
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

    for reference_file in [
        scene_root / "reference" / "reference_depth_manifest.json",
        scene_root / "reference" / "probe_camera_manifest.json",
        scene_root / "reference" / "reference_roi.json",
    ]:
        if reference_file.exists():
            _copy_file(reference_file, target_scene_root / "reference" / reference_file.name)

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
2. Use the training-union RGB views to define the shared frame and dense stereo
   workspace.
3. Run COLMAP dense stereo and fusion.
4. Build a reference mesh:
   - try `delaunay_mesher` first
   - fall back to `poisson_mesher` if Delaunay is unavailable or fails
5. Render held-out probe-view reference depths from that mesh.

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
    threshold_df, secondary_df = _collect_metrics(scene_roots=scene_roots)
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
