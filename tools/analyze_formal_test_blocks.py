#!/usr/bin/env python3
"""Fail-closed 60k paired analysis for the five formal Building test blocks.

Inputs are the frozen test list and bound split plus L/C3/F3/A3 ``per_view.json``
and formal temperature reports.  Every input SHA-256 must be declared on the
command line.  The analysis is deliberately limited to 60k; it does not replace
the required A3 40k/50k/60k aggregate evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "uav-tgs-formal-test-block-analysis-v1"
GROUPS = ("L", "C3", "F3", "A3")
BASELINES = ("C3", "F3", "L")
METHOD_KEY = "ours_60000"
EXPECTED_TEST_VIEWS = 80
EXPECTED_BLOCKS = 5
EXPECTED_VIEWS_PER_BLOCK = 16
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# These tolerances are frozen in code so the direction counts cannot be tuned
# after seeing A3.  Off-LUT values remain directional diagnostics (zero
# tolerance) and are intentionally excluded from the combined judgment.
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "PSNR": {"direction": "higher", "tolerance": 0.05, "combined": True, "unit": "dB"},
    "SSIM": {"direction": "higher", "tolerance": 0.003, "combined": True, "unit": "unitless"},
    "LPIPS": {"direction": "lower", "tolerance": 0.003, "combined": True, "unit": "unitless"},
    "temperature_mae_c": {"direction": "lower", "tolerance": 0.1, "combined": True, "unit": "C"},
    "temperature_rmse_c": {"direction": "lower", "tolerance": 0.1, "combined": True, "unit": "C"},
    "temperature_abs_bias_c": {"direction": "lower", "tolerance": 0.1, "combined": True, "unit": "C"},
    "temperature_p95_c": {"direction": "lower", "tolerance": 0.1, "combined": False, "unit": "C"},
    "off_lut_mean_rgb_distance": {
        "direction": "lower",
        "tolerance": 0.0,
        "combined": False,
        "unit": "RGB Euclidean distance",
        "directional_only": True,
    },
    "off_lut_p95_rgb_distance": {
        "direction": "lower",
        "tolerance": 0.0,
        "combined": False,
        "unit": "RGB Euclidean distance",
        "directional_only": True,
    },
}


class BlockAnalysisError(RuntimeError):
    """Raised when an input violates the formal paired-analysis contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockAnalysisError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise BlockAnalysisError(f"JSON root must be an object: {path}")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BlockAnalysisError(f"{label} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise BlockAnalysisError(f"{label} is non-finite: {result!r}")
    return result


def _parse_assignments(values: Sequence[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, assigned = value.partition("=")
        if not separator or not key or not assigned:
            raise BlockAnalysisError(f"{label} must use KEY=VALUE, got {value!r}")
        if key in result:
            raise BlockAnalysisError(f"duplicate {label} key: {key}")
        result[key] = assigned
    return result


def _group_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    parsed = _parse_assignments(values, label)
    if set(parsed) != set(GROUPS):
        raise BlockAnalysisError(
            f"{label} groups must be exactly {list(GROUPS)}; got {sorted(parsed)}"
        )
    return {group: Path(parsed[group]).resolve() for group in GROUPS}


def _verify_all_hashes(
    *,
    test_list: Path,
    bound_split: Path,
    per_view_paths: Mapping[str, Path],
    temperature_paths: Mapping[str, Path],
    declarations: Sequence[str],
) -> dict[str, dict[str, str]]:
    expected = _parse_assignments(declarations, "expected SHA-256")
    paths: dict[str, Path] = {
        "test_list": test_list,
        "bound_split": bound_split,
        **{f"per_view:{group}": path for group, path in per_view_paths.items()},
        **{f"temperature:{group}": path for group, path in temperature_paths.items()},
    }
    if set(expected) != set(paths):
        raise BlockAnalysisError(
            "expected SHA-256 keys must exactly match inputs; "
            f"missing={sorted(set(paths) - set(expected))} "
            f"extra={sorted(set(expected) - set(paths))}"
        )
    result: dict[str, dict[str, str]] = {}
    for label, path in paths.items():
        declared = expected[label]
        if not SHA256_RE.fullmatch(declared):
            raise BlockAnalysisError(f"invalid SHA-256 declaration for {label}: {declared!r}")
        declared = declared.lower()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != declared:
            raise BlockAnalysisError(
                f"SHA-256 mismatch for {label}: expected={declared} actual={actual} path={path}"
            )
        result[label] = {"path": str(path), "sha256": actual}
    return result


def _read_test_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() or line != line.strip() for line in lines):
        raise BlockAnalysisError("test list contains blank or whitespace-padded entries")
    if len(lines) != EXPECTED_TEST_VIEWS or len(set(lines)) != EXPECTED_TEST_VIEWS:
        raise BlockAnalysisError(
            f"test list must contain exactly {EXPECTED_TEST_VIEWS} unique entries"
        )
    for name in lines:
        if Path(name).name != name or Path(name).suffix.lower() != ".png":
            raise BlockAnalysisError(f"test-list entry must be a plain PNG filename: {name!r}")
    return lines


def _load_blocks(
    bound_split_path: Path,
    test_names: Sequence[str],
) -> tuple[list[dict[str, Any]], str, dict[str, tuple[Any, ...]]]:
    payload = _load_json(bound_split_path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise BlockAnalysisError("bound split has no records list")
    test_records = [record for record in records if isinstance(record, dict) and record.get("split") == "test"]
    if len(test_records) != EXPECTED_TEST_VIEWS:
        raise BlockAnalysisError(
            f"bound split must contain exactly {EXPECTED_TEST_VIEWS} test records"
        )
    record_names = [str(record.get("thermal_camera_name", "")) for record in test_records]
    if record_names != list(test_names):
        raise BlockAnalysisError("test-list order/content differs from bound-split test records")

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    expected_assignments: dict[str, tuple[Any, ...]] = {}
    for record in test_records:
        strip_id = record.get("strip_id")
        block_index = record.get("block_index")
        block_offset = record.get("block_offset")
        if not isinstance(strip_id, str) or not strip_id:
            raise BlockAnalysisError("test record has invalid strip_id")
        if not isinstance(block_index, int) or not isinstance(block_offset, int):
            raise BlockAnalysisError("test record has non-integer block_index/block_offset")
        stem = Path(str(record["thermal_camera_name"])).stem
        if stem in expected_assignments:
            raise BlockAnalysisError(f"duplicate bound-split test stem: {stem}")
        expected_assignments[stem] = (
            strip_id,
            block_index,
            record.get("position_in_strip"),
            record.get("pair_id"),
            record.get("stratum"),
            record.get("hash"),
        )
        grouped[(strip_id, block_index)].append(record)
    if len(grouped) != EXPECTED_BLOCKS:
        raise BlockAnalysisError(f"expected exactly {EXPECTED_BLOCKS} test blocks, got {len(grouped)}")

    blocks: list[dict[str, Any]] = []
    test_position = {name: index for index, name in enumerate(test_names)}
    for (strip_id, block_index), members in grouped.items():
        if len(members) != EXPECTED_VIEWS_PER_BLOCK:
            raise BlockAnalysisError(
                f"block {(strip_id, block_index)} has {len(members)} views; "
                f"expected {EXPECTED_VIEWS_PER_BLOCK}"
            )
        ordered = sorted(members, key=lambda item: item["block_offset"])
        offsets = [member["block_offset"] for member in ordered]
        if offsets != list(range(EXPECTED_VIEWS_PER_BLOCK)):
            raise BlockAnalysisError(
                f"block {(strip_id, block_index)} offsets are not 0..15: {offsets}"
            )
        strata = {str(member.get("stratum", "")) for member in ordered}
        if len(strata) != 1 or "" in strata:
            raise BlockAnalysisError(f"block {(strip_id, block_index)} does not have one valid stratum")
        names = [str(member["thermal_camera_name"]) for member in ordered]
        blocks.append({
            "strip_id": strip_id,
            "block_index": block_index,
            "stratum": next(iter(strata)),
            "size": len(names),
            "views": names,
            "first_test_list_position": min(test_position[name] for name in names),
        })
    blocks.sort(key=lambda item: item["first_test_list_position"])
    for block in blocks:
        block.pop("first_test_list_position")
    selected_blocks_hash = payload.get("selected_test_blocks_hash")
    if not isinstance(selected_blocks_hash, str) or not SHA256_RE.fullmatch(selected_blocks_hash):
        raise BlockAnalysisError("bound split lacks a valid selected_test_blocks_hash")
    return blocks, selected_blocks_hash.lower(), expected_assignments


def _metric_map(method: Mapping[str, Any], metric: str, group: str) -> dict[str, float]:
    raw = method.get(metric)
    if not isinstance(raw, dict):
        raise BlockAnalysisError(f"per_view {group}/{METHOD_KEY} lacks metric map {metric}")
    result: dict[str, float] = {}
    for name, value in raw.items():
        if Path(str(name)).name != str(name):
            raise BlockAnalysisError(f"per_view {group}/{metric} has unsafe view name: {name!r}")
        if name in result:
            raise BlockAnalysisError(f"per_view {group}/{metric} has duplicate view: {name}")
        result[str(name)] = _finite_float(value, f"per_view {group}/{metric}/{name}")
    return result


def _load_per_view(path: Path, group: str, test_names: Sequence[str]) -> dict[str, dict[str, float]]:
    payload = _load_json(path)
    method = payload.get(METHOD_KEY)
    if not isinstance(method, dict):
        raise BlockAnalysisError(f"per_view {group} lacks fixed endpoint {METHOD_KEY}")
    maps = {metric: _metric_map(method, metric, group) for metric in ("PSNR", "SSIM", "LPIPS")}
    expected = set(test_names)
    for metric, values in maps.items():
        if set(values) != expected:
            raise BlockAnalysisError(
                f"per_view set mismatch for {group}/{metric}: "
                f"missing={sorted(expected - set(values))} extra={sorted(set(values) - expected)}"
            )
    return {name: {metric: maps[metric][name] for metric in maps} for name in test_names}


def _temperature_view_metrics(item: Mapping[str, Any], group: str, stem: str) -> dict[str, float]:
    temperature = item.get("supported_pixel_temperature_error")
    off_lut = item.get("supported_pixel_off_lut_distance")
    if not isinstance(temperature, dict) or not isinstance(off_lut, dict):
        raise BlockAnalysisError(f"temperature report {group}/{stem} lacks supported metrics")
    signed_bias = _finite_float(
        temperature.get("signed_bias_c"), f"temperature {group}/{stem}/signed_bias_c"
    )
    result = {
        "temperature_mae_c": _finite_float(
            temperature.get("mae_c"), f"temperature {group}/{stem}/mae_c"
        ),
        "temperature_rmse_c": _finite_float(
            temperature.get("rmse_c"), f"temperature {group}/{stem}/rmse_c"
        ),
        "temperature_signed_bias_c": signed_bias,
        "temperature_p95_c": _finite_float(
            temperature.get("p95_abs_error_c"), f"temperature {group}/{stem}/p95"
        ),
        "off_lut_mean_rgb_distance": _finite_float(
            off_lut.get("mean_rgb_distance"), f"temperature {group}/{stem}/off_lut_mean"
        ),
        "off_lut_p95_rgb_distance": _finite_float(
            off_lut.get("p95_rgb_distance"), f"temperature {group}/{stem}/off_lut_p95"
        ),
    }
    for metric, value in result.items():
        if metric != "temperature_signed_bias_c" and value < 0:
            raise BlockAnalysisError(f"temperature {group}/{stem}/{metric} is negative")
    return result


def _load_temperature(
    path: Path,
    group: str,
    test_names: Sequence[str],
    bound_split_sha256: str,
    expected_assignments: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, dict[str, float]], dict[str, tuple[Any, ...]]]:
    payload = _load_json(path)
    if payload.get("status") != "complete" or payload.get("completed_with_missing") is not False:
        raise BlockAnalysisError(f"temperature report is not complete for {group}")
    split = payload.get("split")
    if not isinstance(split, dict) or split.get("subset") != "test":
        raise BlockAnalysisError(f"temperature report has no fixed test split for {group}")
    if split.get("sha256") != bound_split_sha256:
        raise BlockAnalysisError(
            f"temperature report bound-split SHA differs for {group}: {split.get('sha256')!r}"
        )
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_TEST_VIEWS:
        raise BlockAnalysisError(f"temperature report {group} must contain exactly 80 file records")
    summary = payload.get("summary")
    if not isinstance(summary, dict) or summary.get("evaluated_file_count") != EXPECTED_TEST_VIEWS:
        raise BlockAnalysisError(f"temperature summary count differs for {group}")

    expected_stems = {Path(name).stem for name in test_names}
    metrics: dict[str, dict[str, float]] = {}
    fairness: dict[str, tuple[Any, ...]] = {}
    for item in files:
        if not isinstance(item, dict):
            raise BlockAnalysisError(f"temperature report {group} contains a non-object file record")
        stem = str(item.get("relative_id", ""))
        if not stem or stem in metrics:
            raise BlockAnalysisError(f"temperature report {group} has duplicate/empty stem: {stem!r}")
        if item.get("status") != "complete" or int(item.get("missing_pixels", -1)) != 0:
            raise BlockAnalysisError(f"temperature view is incomplete for {group}/{stem}")
        assignment = item.get("split_assignment")
        if not isinstance(assignment, dict) or assignment.get("split") != "test":
            raise BlockAnalysisError(f"temperature view lacks test assignment for {group}/{stem}")
        assignment_tuple = (
            assignment.get("strip_id"),
            assignment.get("block_index"),
            assignment.get("position_in_strip"),
            assignment.get("pair_id"),
            assignment.get("stratum"),
            assignment.get("hash"),
        )
        if expected_assignments.get(stem) != assignment_tuple:
            raise BlockAnalysisError(
                f"temperature split assignment differs from bound split for {group}/{stem}"
            )
        metrics[stem] = _temperature_view_metrics(item, group, stem)
        fairness[stem] = (
            item.get("ground_truth_sha256"),
            item.get("mask_sha256"),
            int(item.get("supported_pixels", -1)),
            assignment.get("strip_id"),
            assignment.get("block_index"),
            assignment.get("position_in_strip"),
            assignment.get("pair_id"),
        )
    if set(metrics) != expected_stems:
        raise BlockAnalysisError(
            f"temperature view set mismatch for {group}: "
            f"missing={sorted(expected_stems - set(metrics))} extra={sorted(set(metrics) - expected_stems)}"
        )
    return metrics, fairness


def _check_temperature_fairness(
    fairness_by_group: Mapping[str, Mapping[str, tuple[Any, ...]]],
    stems: Sequence[str],
) -> None:
    for stem in stems:
        reference = fairness_by_group["L"][stem]
        if not isinstance(reference[0], str) or not isinstance(reference[1], str) or reference[2] <= 0:
            raise BlockAnalysisError(f"temperature fairness metadata is incomplete for L/{stem}")
        for group in GROUPS[1:]:
            if fairness_by_group[group][stem] != reference:
                raise BlockAnalysisError(
                    f"ground truth/support/split metadata differs for {group}/{stem}"
                )


def _block_group_means(
    block: Mapping[str, Any],
    per_view: Mapping[str, Mapping[str, Mapping[str, float]]],
    temperature: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, float]]:
    names = list(block["views"])
    stems = [Path(name).stem for name in names]
    output: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        metrics: dict[str, float] = {}
        for metric in ("PSNR", "SSIM", "LPIPS"):
            metrics[metric] = float(np.mean([per_view[group][name][metric] for name in names]))
        for metric in (
            "temperature_mae_c",
            "temperature_rmse_c",
            "temperature_signed_bias_c",
            "temperature_p95_c",
            "off_lut_mean_rgb_distance",
            "off_lut_p95_rgb_distance",
        ):
            metrics[metric] = float(np.mean([temperature[group][stem][metric] for stem in stems]))
        # Match the formal gate semantics: absolute value of the block's signed
        # bias, not the mean of per-frame absolute biases.
        metrics["temperature_abs_bias_c"] = abs(metrics["temperature_signed_bias_c"])
        output[group] = metrics
    return output


def _classify(metric: str, a3: float, baseline: float) -> dict[str, Any]:
    spec = METRIC_SPECS[metric]
    raw_delta = a3 - baseline
    improvement_delta = raw_delta if spec["direction"] == "higher" else -raw_delta
    tolerance = float(spec["tolerance"])
    epsilon = 1e-12 * max(1.0, abs(a3), abs(baseline), tolerance)
    if improvement_delta > tolerance + epsilon:
        classification = "improved"
    elif improvement_delta < -tolerance - epsilon:
        classification = "declined"
    else:
        classification = "tied"
    return {
        "a3": a3,
        "baseline": baseline,
        "raw_a3_minus_baseline": raw_delta,
        "improvement_delta": improvement_delta,
        "classification": classification,
        "non_degraded": classification != "declined",
    }


def analyze_blocks(
    *,
    blocks: Sequence[Mapping[str, Any]],
    per_view: Mapping[str, Mapping[str, Mapping[str, float]]],
    temperature: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts: dict[str, dict[str, Counter[str]]] = {
        baseline: {metric: Counter() for metric in METRIC_SPECS} for baseline in BASELINES
    }
    combined_count = 0
    combined_metrics = [metric for metric, spec in METRIC_SPECS.items() if spec["combined"]]
    for block in blocks:
        group_means = _block_group_means(block, per_view, temperature)
        comparisons: dict[str, Any] = {}
        for baseline in BASELINES:
            metric_results: dict[str, Any] = {}
            for metric in METRIC_SPECS:
                result = _classify(metric, group_means["A3"][metric], group_means[baseline][metric])
                metric_results[metric] = result
                counts[baseline][metric][result["classification"]] += 1
            combined_non_degraded = all(
                metric_results[metric]["non_degraded"] for metric in combined_metrics
            )
            signed_bias_a3 = group_means["A3"]["temperature_signed_bias_c"]
            signed_bias_baseline = group_means[baseline]["temperature_signed_bias_c"]
            comparisons[f"A3_minus_{baseline}"] = {
                "metrics": metric_results,
                "diagnostic_deltas": {
                    "temperature_signed_bias_c": {
                        "a3": signed_bias_a3,
                        "baseline": signed_bias_baseline,
                        "raw_a3_minus_baseline": signed_bias_a3 - signed_bias_baseline,
                        "classification": "diagnostic_only",
                    }
                },
                "combined_non_degraded": combined_non_degraded,
                "combined_metrics": combined_metrics,
            }
            if baseline == "F3" and combined_non_degraded:
                combined_count += 1
        records.append({
            "block": dict(block),
            "group_means": group_means,
            "paired_comparisons": comparisons,
        })

    count_payload: dict[str, Any] = {}
    for baseline in BASELINES:
        count_payload[f"A3_minus_{baseline}"] = {
            metric: {
                "improved": counts[baseline][metric]["improved"],
                "tied": counts[baseline][metric]["tied"],
                "declined": counts[baseline][metric]["declined"],
            }
            for metric in METRIC_SPECS
        }
    return {
        "blocks": records,
        "classification_counts": count_payload,
        "a3_vs_f3_combined_judgment": {
            "rule": "at least 3 of 5 blocks are non-degraded on every combined metric",
            "combined_metrics": combined_metrics,
            "non_degraded_blocks": combined_count,
            "required_blocks": 3,
            "total_blocks": len(blocks),
            "passed": combined_count >= 3,
        },
    }


CSV_FIELDS = (
    "row_type",
    "strip_id",
    "block_index",
    "stratum",
    "block_size",
    "group",
    "baseline",
    "metric",
    "direction",
    "tolerance",
    "group_mean",
    "baseline_mean",
    "raw_a3_minus_baseline",
    "improvement_delta",
    "classification",
    "combined_metric",
    "combined_block_non_degraded",
)


def _csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in analysis["blocks"]:
        block = record["block"]
        common = {
            "strip_id": block["strip_id"],
            "block_index": block["block_index"],
            "stratum": block["stratum"],
            "block_size": block["size"],
        }
        for group, metrics in record["group_means"].items():
            for metric, value in metrics.items():
                rows.append({
                    "row_type": "group_mean",
                    **common,
                    "group": group,
                    "metric": metric,
                    "group_mean": value,
                })
        for comparison_name, comparison in record["paired_comparisons"].items():
            baseline = comparison_name.removeprefix("A3_minus_")
            for metric, result in comparison["metrics"].items():
                spec = METRIC_SPECS[metric]
                rows.append({
                    "row_type": "paired_delta",
                    **common,
                    "group": "A3",
                    "baseline": baseline,
                    "metric": metric,
                    "direction": spec["direction"],
                    "tolerance": spec["tolerance"],
                    "group_mean": result["a3"],
                    "baseline_mean": result["baseline"],
                    "raw_a3_minus_baseline": result["raw_a3_minus_baseline"],
                    "improvement_delta": result["improvement_delta"],
                    "classification": result["classification"],
                    "combined_metric": spec["combined"],
                    "combined_block_non_degraded": comparison["combined_non_degraded"],
                })
            for metric, result in comparison["diagnostic_deltas"].items():
                rows.append({
                    "row_type": "paired_delta",
                    **common,
                    "group": "A3",
                    "baseline": baseline,
                    "metric": metric,
                    "direction": "diagnostic",
                    "group_mean": result["a3"],
                    "baseline_mean": result["baseline"],
                    "raw_a3_minus_baseline": result["raw_a3_minus_baseline"],
                    "classification": result["classification"],
                    "combined_metric": False,
                    "combined_block_non_degraded": comparison["combined_non_degraded"],
                })
    return rows


def _write_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], overwrite: bool) -> None:
    target = path.resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def run_analysis(
    *,
    test_list_path: Path,
    bound_split_path: Path,
    per_view_paths: Mapping[str, Path],
    temperature_paths: Mapping[str, Path],
    expected_sha256: Sequence[str],
) -> dict[str, Any]:
    for label, paths in (("per_view", per_view_paths), ("temperature", temperature_paths)):
        if set(paths) != set(GROUPS):
            raise BlockAnalysisError(
                f"{label} groups must be exactly {list(GROUPS)}; got {sorted(paths)}"
            )
    test_list_path = test_list_path.resolve()
    bound_split_path = bound_split_path.resolve()
    inputs = _verify_all_hashes(
        test_list=test_list_path,
        bound_split=bound_split_path,
        per_view_paths=per_view_paths,
        temperature_paths=temperature_paths,
        declarations=expected_sha256,
    )
    test_names = _read_test_list(test_list_path)
    blocks, selected_blocks_hash, expected_assignments = _load_blocks(
        bound_split_path, test_names
    )
    bound_sha = inputs["bound_split"]["sha256"]

    per_view = {
        group: _load_per_view(per_view_paths[group], group, test_names) for group in GROUPS
    }
    temperature: dict[str, dict[str, dict[str, float]]] = {}
    fairness: dict[str, dict[str, tuple[Any, ...]]] = {}
    for group in GROUPS:
        temperature[group], fairness[group] = _load_temperature(
            temperature_paths[group],
            group,
            test_names,
            bound_sha,
            expected_assignments,
        )
    _check_temperature_fairness(fairness, [Path(name).stem for name in test_names])
    analysis = analyze_blocks(blocks=blocks, per_view=per_view, temperature=temperature)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "iteration": 60000,
        "claim_boundary": (
            "Paired analysis of the fixed five 16-view Building test blocks at 60k only; "
            "this does not replace required A3 aggregate evaluation at 40k, 50k, and 60k."
        ),
        "aggregation": "unweighted frame-macro mean within each fixed 16-view block",
        "inputs": inputs,
        "protocol": {
            "groups": list(GROUPS),
            "method_key": METHOD_KEY,
            "test_views": EXPECTED_TEST_VIEWS,
            "block_count": EXPECTED_BLOCKS,
            "views_per_block": EXPECTED_VIEWS_PER_BLOCK,
            "selected_test_blocks_hash": selected_blocks_hash,
            "metric_specs": METRIC_SPECS,
        },
        **analysis,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-list", required=True, type=Path)
    parser.add_argument("--bound-split", required=True, type=Path)
    parser.add_argument(
        "--per-view",
        action="append",
        default=[],
        metavar="GROUP=PATH",
        help="Repeat exactly once for L, C3, F3, and A3.",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        default=[],
        metavar="GROUP=PATH",
        help="Repeat exactly once for L, C3, F3, and A3.",
    )
    parser.add_argument(
        "--expected-sha256",
        action="append",
        default=[],
        metavar="LABEL=SHA256",
        help=(
            "Required labels: test_list, bound_split, per_view:<GROUP>, and "
            "temperature:<GROUP>."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    per_view_paths = _group_paths(args.per_view, "per-view")
    temperature_paths = _group_paths(args.temperature, "temperature")
    payload = run_analysis(
        test_list_path=args.test_list,
        bound_split_path=args.bound_split,
        per_view_paths=per_view_paths,
        temperature_paths=temperature_paths,
        expected_sha256=args.expected_sha256,
    )
    rows = _csv_rows(payload)
    _write_json(args.output, payload, args.overwrite)
    _write_csv(args.csv, rows, args.overwrite)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "combined_passed": payload["a3_vs_f3_combined_judgment"]["passed"],
                "output": str(args.output),
                "csv": str(args.csv),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
