from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def _parse_series(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected label=path format, got: {spec}")
    label, path_text = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in spec: {spec}")
    path = Path(path_text).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {path}")
    return label, path


def _load_frame(label: str, path: Path, metric: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"threshold_abs", metric}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")
    if "scene_name" not in frame.columns:
        frame["scene_name"] = path.parent.parent.name
    out = frame.copy()
    out["series_label"] = label
    out["metrics_csv"] = str(path)
    return out.sort_values("threshold_abs").reset_index(drop=True)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Plot repeatability-vs-threshold curves from metrics.csv files.")
    ap.add_argument(
        "--input",
        action="append",
        required=True,
        help="Series spec in label=path_to_metrics.csv format. Repeat for multiple methods.",
    )
    ap.add_argument("--metric", default="fscore", choices=["precision", "recall", "fscore"])
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--dpi", type=int, default=200)
    return ap


def main() -> None:
    args = _build_argparser().parse_args()

    frames: List[pd.DataFrame] = []
    for spec in args.input:
        label, path = _parse_series(spec)
        frames.append(_load_frame(label=label, path=path, metric=args.metric))

    merged = pd.concat(frames, ignore_index=True)
    scene_names = list(dict.fromkeys(merged["scene_name"].tolist()))
    if not scene_names:
        raise RuntimeError("No scenes found in the provided CSV files.")

    fig, axes = plt.subplots(
        1,
        len(scene_names),
        figsize=(6 * len(scene_names), 4.5),
        squeeze=False,
    )
    axes_row = axes[0]

    for ax, scene_name in zip(axes_row, scene_names):
        scene_df = merged[merged["scene_name"] == scene_name]
        labels = list(dict.fromkeys(scene_df["series_label"].tolist()))
        for label in labels:
            sub = scene_df[scene_df["series_label"] == label].sort_values("threshold_abs")
            ax.plot(
                sub["threshold_abs"],
                sub[args.metric],
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                label=label,
            )
        ax.set_title(scene_name)
        ax.set_xlabel("Threshold (m)")
        ax.set_ylabel(args.metric.capitalize())
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend()

    if args.title:
        fig.suptitle(args.title)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout()

    out_png = Path(args.out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    if args.out_csv:
        out_csv = Path(args.out_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)

    print(f"CURVE_PLOT_SAVED {out_png}")
    if args.out_csv:
        print(f"CURVE_TABLE_SAVED {Path(args.out_csv).resolve()}")


if __name__ == "__main__":
    main()
