#!/usr/bin/env python3
"""Held-out validation interface for rendered hotspot gate models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_file(root: Path, stem: str, suffixes: tuple[str, ...]) -> Path:
    matches = []
    for suffix in suffixes:
        direct = root / f"{stem}{suffix}"
        if direct.is_file():
            matches.append(direct)
        matches.extend(root.glob(f"*/{stem}{suffix}"))
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise RuntimeError(f"expected one file for {stem} under {root}, found {len(unique)}")
    return unique[0]


def load_gate(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False).astype(np.float64)
        if value.ndim == 3:
            value = value.mean(axis=2) if value.shape[2] in (3, 4) else value.squeeze()
    else:
        value = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).mean(axis=2) / 255.0
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise RuntimeError(f"invalid rendered gate: {path}")
    return np.clip(value, 0.0, 1.0)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    truth = labels[order].astype(np.float64)
    positives = truth.sum()
    if positives <= 0:
        return float("nan")
    precision = np.cumsum(truth) / np.arange(1, len(truth) + 1)
    return float(np.sum(precision * truth) / positives)


def average_tie_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(scores: np.ndarray, temperature: np.ndarray) -> float:
    left = average_tie_ranks(scores)
    right = average_tie_ranks(temperature)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else float("nan")


def evaluate(args: argparse.Namespace) -> dict:
    render_root = Path(args.render_root).resolve()
    temperature_root = Path(args.temperature_root).resolve()
    support_root = None if not args.support_root else Path(args.support_root).resolve()
    heldout_list = Path(args.heldout_list).resolve()
    names = [Path(line.strip()).stem for line in heldout_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise RuntimeError("held-out list is empty")
    aggregate_scores, aggregate_labels, aggregate_temperature = [], [], []
    per_view = []
    for name in names:
        render_path = resolve_file(render_root, name, (".npy", ".png", ".jpg", ".JPG"))
        temperature_path = resolve_file(temperature_root, name, (".npy",))
        gate = load_gate(render_path)
        temperature = np.load(temperature_path, allow_pickle=False).astype(np.float64)
        if temperature.ndim != 2 or not np.all(np.isfinite(temperature)):
            raise RuntimeError(f"invalid temperature target: {temperature_path}")
        if gate.shape != temperature.shape:
            gate = np.asarray(
                Image.fromarray(gate.astype(np.float32), mode="F").resize(
                    (temperature.shape[1], temperature.shape[0]), Image.Resampling.BILINEAR
                ),
                dtype=np.float64,
            )
        valid = np.ones(temperature.shape, dtype=bool)
        if support_root is not None:
            support_path = resolve_file(support_root, name, (".npy",))
            support = np.load(support_path, allow_pickle=False)
            if support.dtype != np.bool_ or support.shape != temperature.shape:
                raise RuntimeError(f"invalid support mask: {support_path}")
            valid &= support
        scores = gate[valid]
        temperatures = temperature[valid]
        labels = temperatures >= float(args.threshold_c)
        prediction = scores >= 0.5
        tp = int(np.sum(prediction & labels))
        fp = int(np.sum(prediction & ~labels))
        fn = int(np.sum(~prediction & labels))
        per_view.append(
            {
                "image_name": name,
                "valid_pixels": int(valid.sum()),
                "hot_prevalence": float(labels.mean()),
                "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "auprc": average_precision(scores, labels),
            }
        )
        aggregate_scores.append(scores)
        aggregate_labels.append(labels)
        aggregate_temperature.append(temperatures)
    scores = np.concatenate(aggregate_scores)
    labels = np.concatenate(aggregate_labels)
    temperatures = np.concatenate(aggregate_temperature)
    prediction = scores >= 0.5
    tp = int(np.sum(prediction & labels))
    fp = int(np.sum(prediction & ~labels))
    fn = int(np.sum(~prediction & labels))
    # Deterministic uniform stride bounds the rank-correlation memory/runtime.
    max_rank = int(args.max_rank_pixels)
    stride = max(1, len(scores) // max_rank)
    rank_scores = scores[::stride][:max_rank]
    rank_temperature = temperatures[::stride][:max_rank]
    prevalence = float(labels.mean())
    auprc = average_precision(scores, labels)
    rank_correlation = spearman(rank_scores, rank_temperature)
    validated = bool(
        np.isfinite(auprc)
        and np.isfinite(rank_correlation)
        and auprc >= max(0.10, 1.25 * prevalence)
        and rank_correlation >= 0.20
    )
    return {
        "schema": "uav-tgs-heldout-hotspot-gate-evaluation-v1",
        "status": "passed",
        "heldout_only": True,
        "threshold_c": float(args.threshold_c),
        "gate_binary_threshold": 0.5,
        "heldout_list": str(heldout_list),
        "heldout_list_sha256": sha256(heldout_list),
        "global": {
            "valid_pixels": int(len(scores)),
            "hot_prevalence": prevalence,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "auprc": auprc,
            "rank_correlation_spearman": rank_correlation,
            "rank_sample_pixels": int(len(rank_scores)),
        },
        "semantic_rule": {
            "auprc_min": "max(0.10, 1.25*positive_prevalence)",
            "spearman_min": 0.20,
            "fixed_before_evaluation": True,
        },
        "semantic_status": (
            "temperature_gated_fusion_validated"
            if validated
            else "downgraded_to_high_thermal_response_visualization"
        ),
        "per_view": per_view,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--temperature-root", required=True)
    parser.add_argument("--support-root", default="")
    parser.add_argument("--heldout-list", required=True)
    parser.add_argument("--threshold-c", type=float, required=True)
    parser.add_argument("--max-rank-pixels", type=int, default=1_000_000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_rank_pixels <= 0:
        raise SystemExit("--max-rank-pixels must be positive")
    try:
        result = evaluate(args)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"hotspot gate evaluation failed: {error}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
