#!/usr/bin/env python3
"""Export unweighted constrained Pareto tables and publication plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REQUIRED = (
    "scene",
    "method",
    "t_lpips",
    "temperature_mae_c",
    "front_1m",
    "gaussian_count",
    "feasible",
)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y"):
        return True
    if normalized in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in REQUIRED if name not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} is missing columns: {missing}")
            for source in reader:
                row = {"scene": source["scene"], "method": source["method"]}
                for name in ("t_lpips", "temperature_mae_c", "front_1m", "gaussian_count"):
                    row[name] = float(source[name])
                    if not np.isfinite(row[name]):
                        raise ValueError(f"non-finite {name} in {path}")
                row["feasible"] = parse_bool(source["feasible"])
                rows.append(row)
    if not rows:
        raise ValueError("no Pareto rows were loaded")
    keys = [(row["scene"], row["method"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate scene/method Pareto row")
    return rows


def pareto_members(rows: list[dict], y_name: str) -> set[tuple[str, str]]:
    """Return nondominated feasible points when both axes are minimized."""

    feasible = [row for row in rows if row["feasible"]]
    members = set()
    for candidate in feasible:
        dominated = any(
            other["front_1m"] <= candidate["front_1m"]
            and other[y_name] <= candidate[y_name]
            and (
                other["front_1m"] < candidate["front_1m"]
                or other[y_name] < candidate[y_name]
            )
            for other in feasible
        )
        if not dominated:
            members.add((candidate["scene"], candidate["method"]))
    return members


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    methods = sorted({row["method"] for row in rows})
    macro = []
    worst = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        macro.append(
            {
                "scene": "MACRO",
                "method": method,
                "t_lpips": float(np.mean([row["t_lpips"] for row in selected])),
                "temperature_mae_c": float(np.mean([row["temperature_mae_c"] for row in selected])),
                "front_1m": float(np.mean([row["front_1m"] for row in selected])),
                "gaussian_count": float(np.mean([row["gaussian_count"] for row in selected])),
                "feasible": all(row["feasible"] for row in selected),
            }
        )
        worst.append(
            {
                "method": method,
                "worst_scene_t_lpips": max(selected, key=lambda row: row["t_lpips"])["scene"],
                "worst_t_lpips": max(row["t_lpips"] for row in selected),
                "worst_scene_temperature_mae_c": max(selected, key=lambda row: row["temperature_mae_c"])["scene"],
                "worst_temperature_mae_c": max(row["temperature_mae_c"] for row in selected),
                "worst_scene_front_1m": max(selected, key=lambda row: row["front_1m"])["scene"],
                "worst_front_1m": max(row["front_1m"] for row in selected),
                "all_scenes_feasible": all(row["feasible"] for row in selected),
            }
        )
    return macro, worst


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], macro: list[dict], output_png: Path, output_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in rows})
    palette = plt.get_cmap("tab10")
    colors = {method: palette(index % 10) for index, method in enumerate(methods)}
    counts = np.asarray([row["gaussian_count"] for row in rows + macro], dtype=np.float64)
    low, high = float(counts.min()), float(counts.max())

    def bubble(count):
        if high <= low:
            return 90.0
        return 35.0 + 150.0 * (np.log1p(count) - np.log1p(low)) / (np.log1p(high) - np.log1p(low))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for axis, y_name, y_label in (
        (axes[0], "t_lpips", "Thermal LPIPS ↓"),
        (axes[1], "temperature_mae_c", "Temperature MAE (°C) ↓"),
    ):
        members = pareto_members(rows + macro, y_name)
        for row in rows:
            marker = "o" if row["feasible"] else "x"
            axis.scatter(
                row["front_1m"], row[y_name], s=0.45 * bubble(row["gaussian_count"]),
                marker=marker, color=colors[row["method"]], alpha=0.32, linewidths=1.0,
            )
        for row in macro:
            marker = "o" if row["feasible"] else "X"
            edge = "black" if (row["scene"], row["method"]) in members else "none"
            axis.scatter(
                row["front_1m"], row[y_name], s=bubble(row["gaussian_count"]),
                marker=marker, color=colors[row["method"]], edgecolors=edge,
                linewidths=1.5, label=row["method"], zorder=3,
            )
        axis.set_xlabel("Front intrusion @1 m ↓")
        axis.set_ylabel(y_label)
        axis.grid(True, linewidth=0.5, alpha=0.3)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(5, len(labels)),
        frameon=False,
    )
    fig.suptitle("Constrained Pareto comparison (small: scene; large: macro; ×: infeasible)")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        paths = [Path(value).resolve() for value in args.input]
        rows = load_rows(paths)
        macro, worst = aggregate(rows)
        for row in rows:
            row["pareto_t_lpips"] = (row["scene"], row["method"]) in pareto_members(rows, "t_lpips")
            row["pareto_temperature_mae"] = (row["scene"], row["method"]) in pareto_members(rows, "temperature_mae_c")
        for row in macro:
            row["pareto_t_lpips"] = (row["scene"], row["method"]) in pareto_members(macro, "t_lpips")
            row["pareto_temperature_mae"] = (row["scene"], row["method"]) in pareto_members(macro, "temperature_mae_c")
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        fields = list(REQUIRED) + ["pareto_t_lpips", "pareto_temperature_mae"]
        write_csv(output / "pareto_per_scene.csv", rows, fields)
        write_csv(output / "pareto_macro.csv", macro, fields)
        write_csv(output / "pareto_worst_scene.csv", worst, list(worst[0]))
        plot(rows, macro, output / "pareto.png", output / "pareto.pdf")
        manifest = {
            "schema": "uav-tgs-constrained-pareto-export-v1",
            "status": "passed",
            "inputs": [str(path) for path in paths],
            "weighted_score": False,
            "axes": ["t_lpips vs front_1m", "temperature_mae_c vs front_1m"],
            "bubble": "gaussian_count",
            "small_points": "per-scene",
            "large_points": "macro mean",
            "infeasible_marker": "x",
            "worst_scene_table": True,
        }
        (output / "pareto_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"Pareto export failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
