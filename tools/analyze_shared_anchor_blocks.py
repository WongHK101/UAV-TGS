#!/usr/bin/env python3
"""Per-test-block RGB/depth summary for raw and shared-clamped anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.analyze_formal_test_blocks import _load_blocks, _read_test_list
except ModuleNotFoundError:
    # Direct ``python tools/analyze_shared_anchor_blocks.py`` execution places
    # the tools directory, rather than the repository root, on sys.path.
    from analyze_formal_test_blocks import _load_blocks, _read_test_list


class SharedAnchorBlockError(RuntimeError):
    pass


THRESHOLDS = (1.0, 2.0, 5.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SharedAnchorBlockError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SharedAnchorBlockError(f"JSON root must be an object: {path}")
    return payload


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SharedAnchorBlockError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise SharedAnchorBlockError(f"{label} is non-finite")
    return result


def _parse_group_paths(
    values: list[str], label: str, expected_groups: tuple[str, ...] | None = None
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        group, separator, raw_path = value.partition("=")
        if not separator or not group or not raw_path:
            raise SharedAnchorBlockError(f"{label} must use GROUP=PATH")
        if group in result:
            raise SharedAnchorBlockError(f"duplicate {label} group: {group}")
        result[group] = Path(raw_path).resolve()
    if not result:
        raise SharedAnchorBlockError(f"{label} requires at least one group")
    if expected_groups is not None and tuple(result) != expected_groups:
        raise SharedAnchorBlockError(
            f"{label} groups/order must be {list(expected_groups)}, got {list(result)}"
        )
    for path in result.values():
        if not path.exists():
            raise SharedAnchorBlockError(f"{label} input is missing: {path}")
    return result


def _appearance(path: Path, iteration: int, test_names: list[str]) -> dict[str, dict[str, float]]:
    payload = _load_json(path)
    key = f"ours_{iteration}"
    method = payload.get(key)
    if not isinstance(method, dict):
        raise SharedAnchorBlockError(f"{path} is missing {key}")
    output: dict[str, dict[str, float]] = {name: {} for name in test_names}
    for metric in ("PSNR", "SSIM", "LPIPS"):
        values = method.get(metric)
        if not isinstance(values, dict) or set(values) != set(test_names):
            raise SharedAnchorBlockError(
                f"{path}:{key}:{metric} view set does not match the test list"
            )
        for name in test_names:
            output[name][metric] = _finite(
                values[name], f"{path}:{key}:{metric}:{name}"
            )
    return output


def _verified_npz(manifest_path: Path, view: dict[str, Any], label: str) -> Path:
    root = manifest_path.parent.resolve()
    path = (root / str(view.get("npz_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SharedAnchorBlockError(f"{label} NPZ escapes manifest root") from exc
    if not path.is_file():
        raise SharedAnchorBlockError(f"{label} NPZ is missing: {path}")
    if (
        int(view.get("npz_size_bytes", -1)) != path.stat().st_size
        or str(view.get("npz_sha256", "")).lower() != _sha256(path)
    ):
        raise SharedAnchorBlockError(f"{label} NPZ identity mismatch: {path}")
    return path


def _metric_depth(raw_depth: np.ndarray, semantics: str) -> np.ndarray:
    raw = np.asarray(raw_depth, dtype=np.float64)
    if semantics == "metric_camera_z_from_renderer":
        return raw
    if semantics == "inverse_camera_z_from_renderer":
        output = np.full(raw.shape, np.nan, dtype=np.float64)
        valid = np.isfinite(raw) & (raw > 0)
        output[valid] = 1.0 / raw[valid]
        return output
    raise SharedAnchorBlockError(f"unsupported depth semantics: {semantics}")


def _depth_inputs(endpoint: Path, test_names: list[str]) -> dict[str, Any]:
    summary_path = endpoint / "metrics" / "metrics_summary.json"
    summary = _load_json(summary_path)
    reference_path = Path(str(summary.get("reference_manifest", ""))).resolve()
    model_path = Path(str(summary.get("model_manifest", ""))).resolve()
    adapter_path = Path(str(summary.get("adapter_manifest", ""))).resolve()
    for label, path, declared in (
        ("reference", reference_path, summary.get("reference_manifest_sha256")),
        ("model", model_path, summary.get("model_manifest_sha256")),
        ("adapter", adapter_path, summary.get("adapter_manifest_sha256")),
    ):
        if not path.is_file() or _sha256(path) != str(declared).lower():
            raise SharedAnchorBlockError(f"{label} manifest identity mismatch: {path}")
    reference = _load_json(reference_path)
    model = _load_json(model_path)
    adapter = _load_json(adapter_path)
    reference_views = reference.get("views")
    model_views = model.get("views")
    if not isinstance(reference_views, list) or not isinstance(model_views, list):
        raise SharedAnchorBlockError("depth manifests must contain view lists")
    reference_by_name = {
        str(view.get("image_name")): view
        for view in reference_views
        if isinstance(view, dict)
    }
    model_by_name = {
        str(view.get("image_name")): view
        for view in model_views
        if isinstance(view, dict)
    }
    if set(reference_by_name) != set(test_names) or set(model_by_name) != set(test_names):
        raise SharedAnchorBlockError("depth manifest view set differs from test list")
    validity = adapter.get("validity_rule")
    if not isinstance(validity, dict):
        raise SharedAnchorBlockError("adapter has no validity rule")
    return {
        "summary_path": summary_path,
        "summary_sha256": _sha256(summary_path),
        "reference_path": reference_path,
        "reference_sha256": _sha256(reference_path),
        "model_path": model_path,
        "model_sha256": _sha256(model_path),
        "adapter_path": adapter_path,
        "adapter_sha256": _sha256(adapter_path),
        "reference_by_name": reference_by_name,
        "model_by_name": model_by_name,
        "depth_semantics": str(adapter.get("depth_semantics")),
        "depth_min": _finite(validity.get("depth_min", 1e-6), "depth_min"),
        "opacity_threshold": _finite(
            validity.get("opacity_threshold", 0.5), "opacity_threshold"
        ),
    }


def _block_depth(names: list[str], inputs: dict[str, Any]) -> dict[str, Any]:
    ref_valid_total = 0
    model_valid_total = 0
    missing_total = 0
    signed_arrays: list[np.ndarray] = []
    counts = {
        threshold: {"front": 0, "behind": 0, "agreement": 0}
        for threshold in THRESHOLDS
    }
    for name in names:
        reference_view = inputs["reference_by_name"][name]
        model_view = inputs["model_by_name"][name]
        reference_npz = np.load(
            _verified_npz(inputs["reference_path"], reference_view, f"reference/{name}")
        )
        model_npz = np.load(
            _verified_npz(inputs["model_path"], model_view, f"model/{name}")
        )
        reference_depth = np.asarray(reference_npz["depth"], dtype=np.float64)
        reference_valid = np.asarray(
            reference_npz["valid_mask"], dtype=np.uint8
        ).astype(bool)
        model_depth = _metric_depth(
            np.asarray(model_npz["depth"], dtype=np.float64),
            inputs["depth_semantics"],
        )
        opacity = np.asarray(model_npz["opacity"], dtype=np.float64)
        if reference_depth.shape != model_depth.shape or model_depth.shape != opacity.shape:
            raise SharedAnchorBlockError(f"depth shape mismatch for {name}")
        model_valid = (
            np.isfinite(model_depth)
            & np.isfinite(opacity)
            & (model_depth > inputs["depth_min"])
            & (opacity >= inputs["opacity_threshold"])
        )
        joint = reference_valid & model_valid
        ref_valid_total += int(np.count_nonzero(reference_valid))
        model_valid_total += int(np.count_nonzero(joint))
        missing_total += int(np.count_nonzero(reference_valid & ~model_valid))
        if np.any(joint):
            signed_arrays.append(model_depth[joint] - reference_depth[joint])
        for threshold in THRESHOLDS:
            counts[threshold]["front"] += int(
                np.count_nonzero(
                    joint & (model_depth < reference_depth - threshold)
                )
            )
            counts[threshold]["behind"] += int(
                np.count_nonzero(
                    joint & (model_depth > reference_depth + threshold)
                )
            )
            counts[threshold]["agreement"] += int(
                np.count_nonzero(
                    joint & (np.abs(model_depth - reference_depth) <= threshold)
                )
            )
    if ref_valid_total <= 0 or not signed_arrays:
        raise SharedAnchorBlockError("block has no valid reference/joint depth pixels")
    signed = np.concatenate(signed_arrays)
    absolute = np.abs(signed)
    return {
        "reference_valid_pixels": ref_valid_total,
        "model_valid_pixels": model_valid_total,
        "missing_pixels": missing_total,
        "missing_rate": missing_total / ref_valid_total,
        "abs_depth_error_mean": float(np.mean(absolute)),
        "abs_depth_error_median": float(np.median(absolute)),
        "signed_depth_bias_mean": float(np.mean(signed)),
        "thresholds": {
            str(int(threshold)): {
                "front": counts[threshold]["front"] / ref_valid_total,
                "behind": counts[threshold]["behind"] / ref_valid_total,
                "agreement": counts[threshold]["agreement"] / ref_valid_total,
            }
            for threshold in THRESHOLDS
        },
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    test_list = Path(args.test_list).resolve()
    bound_split = Path(args.bound_split).resolve()
    test_names = _read_test_list(test_list)
    blocks, split_protocol, _ = _load_blocks(bound_split, test_names)
    appearance_paths = _parse_group_paths(args.appearance, "appearance")
    group_order = tuple(appearance_paths)
    depth_paths = _parse_group_paths(
        args.depth_endpoint, "depth endpoint", expected_groups=group_order
    )
    if not {"Anchor", "S"}.issubset(group_order):
        raise SharedAnchorBlockError(
            "block comparison requires Anchor and S reference groups"
        )
    appearances = {
        group: _appearance(path, args.anchor_iteration, test_names)
        for group, path in appearance_paths.items()
    }
    depth_inputs = {
        group: _depth_inputs(path, test_names)
        for group, path in depth_paths.items()
    }
    if len({value["reference_sha256"] for value in depth_inputs.values()}) != 1:
        raise SharedAnchorBlockError("Anchor and S use different depth references")

    records: list[dict[str, Any]] = []
    for block in blocks:
        names = list(block["views"])
        block_groups: dict[str, Any] = {}
        for group in group_order:
            block_groups[group] = {
                "appearance_frame_macro": {
                    metric: float(
                        np.mean(
                            [appearances[group][name][metric] for name in names]
                        )
                    )
                    for metric in ("PSNR", "SSIM", "LPIPS")
                },
                "depth_pixel_micro": _block_depth(names, depth_inputs[group]),
            }
        records.append(
            {
                "block": dict(block),
                "groups": block_groups,
                "S_minus_Anchor": {
                    "PSNR": (
                        block_groups["S"]["appearance_frame_macro"]["PSNR"]
                        - block_groups["Anchor"]["appearance_frame_macro"]["PSNR"]
                    ),
                    "SSIM": (
                        block_groups["S"]["appearance_frame_macro"]["SSIM"]
                        - block_groups["Anchor"]["appearance_frame_macro"]["SSIM"]
                    ),
                    "LPIPS": (
                        block_groups["S"]["appearance_frame_macro"]["LPIPS"]
                        - block_groups["Anchor"]["appearance_frame_macro"]["LPIPS"]
                    ),
                    "front_at_1m": (
                        block_groups["S"]["depth_pixel_micro"]["thresholds"]["1"]["front"]
                        - block_groups["Anchor"]["depth_pixel_micro"]["thresholds"]["1"]["front"]
                    ),
                    "mean_error_m": (
                        block_groups["S"]["depth_pixel_micro"]["abs_depth_error_mean"]
                        - block_groups["Anchor"]["depth_pixel_micro"]["abs_depth_error_mean"]
                    ),
                    "missing_rate": (
                        block_groups["S"]["depth_pixel_micro"]["missing_rate"]
                        - block_groups["Anchor"]["depth_pixel_micro"]["missing_rate"]
                    ),
                },
            }
        )
    return {
        "schema": "uav-tgs-shared-anchor-test-block-analysis-v1",
        "status": "complete",
        "scene": str(args.scene),
        "anchor_iteration": int(args.anchor_iteration),
        "groups": list(group_order),
        "protocol": {
            "split_protocol": split_protocol,
            "views_per_block": 16,
            "appearance_aggregation": "unweighted frame-macro mean",
            "depth_aggregation": "pixel-micro within each fixed test block",
            "thresholds_m": list(THRESHOLDS),
        },
        "inputs": {
            "test_list": {"path": str(test_list), "sha256": _sha256(test_list)},
            "bound_split": {
                "path": str(bound_split),
                "sha256": _sha256(bound_split),
            },
            "appearance": {
                group: {"path": str(path), "sha256": _sha256(path)}
                for group, path in appearance_paths.items()
            },
            "depth": {
                group: {
                    key: str(value) if key.endswith("_path") else value
                    for key, value in inputs.items()
                    if key.endswith("_path") or key.endswith("_sha256")
                }
                for group, inputs in depth_inputs.items()
            },
        },
        "blocks": records,
    }


def _write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    for record in payload["blocks"]:
        block = record["block"]
        for group, values in record["groups"].items():
            appearance = values["appearance_frame_macro"]
            depth = values["depth_pixel_micro"]
            rows.append(
                {
                    "strip_id": block["strip_id"],
                    "block_index": block["block_index"],
                    "stratum": block["stratum"],
                    "views": block["size"],
                    "group": group,
                    "PSNR": appearance["PSNR"],
                    "SSIM": appearance["SSIM"],
                    "LPIPS": appearance["LPIPS"],
                    "front_1m": depth["thresholds"]["1"]["front"],
                    "agreement_1m": depth["thresholds"]["1"]["agreement"],
                    "behind_1m": depth["thresholds"]["1"]["behind"],
                    "front_2m": depth["thresholds"]["2"]["front"],
                    "agreement_2m": depth["thresholds"]["2"]["agreement"],
                    "behind_2m": depth["thresholds"]["2"]["behind"],
                    "front_5m": depth["thresholds"]["5"]["front"],
                    "agreement_5m": depth["thresholds"]["5"]["agreement"],
                    "behind_5m": depth["thresholds"]["5"]["behind"],
                    "mean_error_m": depth["abs_depth_error_mean"],
                    "median_error_m": depth["abs_depth_error_median"],
                    "missing_rate": depth["missing_rate"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--anchor-iteration", type=int, default=30000)
    parser.add_argument("--test-list", required=True)
    parser.add_argument("--bound-split", required=True)
    parser.add_argument("--appearance", action="append", required=True)
    parser.add_argument("--depth-endpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    try:
        payload = analyze(args)
    except (SharedAnchorBlockError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    output = Path(args.output).resolve()
    csv_path = Path(args.csv).resolve()
    if output.exists() or csv_path.exists():
        parser.error("refusing to overwrite block-analysis output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
