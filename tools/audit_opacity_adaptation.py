#!/usr/bin/env python3
"""Fail-closed audit for geometry-frozen, opacity-adaptive Stage-2 outputs.

The PLY ``opacity`` property is a raw logit.  This tool applies a sigmoid
before reporting opacity statistics and deltas.  It also proves that every
spatial field is byte-exact between the RGB anchor and A3.  Optionally, it
compares the 80 named ``opacity_proxy`` NPY outputs declared by render mapping
manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from plyfile import PlyData


SCHEMA = "uav-tgs-opacity-adaptation-audit-v1"
STRUCTURAL_FIELDS = (
    "x",
    "y",
    "z",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
DELTA_THRESHOLDS = (0.01, 0.05, 0.10)
EXPECTED_PROXY_VIEWS = 80
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class OpacityAuditError(RuntimeError):
    """Raised when an audit input or invariant fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_expected_sha256(value: str, label: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise OpacityAuditError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _verify_sha256(path: Path, expected: str, label: str) -> str:
    expected_normalized = _validate_expected_sha256(expected, label)
    actual = sha256_file(path)
    if actual != expected_normalized:
        raise OpacityAuditError(
            f"{label} SHA-256 mismatch: expected={expected_normalized} actual={actual} path={path}"
        )
    return actual


def _array_sha256(field_name: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(field_name.encode("utf-8"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _read_vertices(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        vertices = PlyData.read(str(path))["vertex"].data
    except (KeyError, ValueError) as exc:
        raise OpacityAuditError(f"PLY has no readable vertex element: {path}") from exc
    if vertices.dtype.names is None:
        raise OpacityAuditError(f"PLY vertex element has no named fields: {path}")
    return vertices


def _require_finite(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise OpacityAuditError(f"{label} contains non-finite values")
    return array


def _sigmoid(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise OpacityAuditError("cannot summarize an empty array")
    _require_finite(array, "summary array")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _delta_summary(anchor: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if anchor.shape != candidate.shape:
        raise OpacityAuditError(
            f"opacity shape mismatch: anchor={anchor.shape} candidate={candidate.shape}"
        )
    signed = np.asarray(candidate, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    absolute = np.abs(signed)
    result = {
        "signed": _distribution(signed),
        "absolute": _distribution(absolute),
        "rmse": float(math.sqrt(float(np.mean(np.square(signed))))),
        "fractions_abs_delta_gt": {},
    }
    for threshold in DELTA_THRESHOLDS:
        result["fractions_abs_delta_gt"][f"{threshold:.2f}"] = float(
            np.mean(absolute > threshold)
        )
    return result


def audit_ply_pair(anchor_ply: Path, a3_ply: Path) -> dict[str, Any]:
    anchor = _read_vertices(anchor_ply)
    candidate = _read_vertices(a3_ply)
    anchor_names = set(anchor.dtype.names or ())
    candidate_names = set(candidate.dtype.names or ())
    required = set(STRUCTURAL_FIELDS) | {"opacity"}
    for label, names in (("anchor", anchor_names), ("A3", candidate_names)):
        missing = sorted(required - names)
        if missing:
            raise OpacityAuditError(f"{label} PLY is missing required fields: {missing}")
    if len(anchor) != len(candidate):
        raise OpacityAuditError(
            f"Gaussian count mismatch: anchor={len(anchor)} A3={len(candidate)}"
        )

    structural: dict[str, Any] = {}
    for field in STRUCTURAL_FIELDS:
        before = _require_finite(np.asarray(anchor[field]), f"anchor field {field}")
        after = _require_finite(np.asarray(candidate[field]), f"A3 field {field}")
        same_dtype = before.dtype == after.dtype
        same_shape = before.shape == after.shape
        before_hash = _array_sha256(field, before)
        after_hash = _array_sha256(field, after)
        exact = bool(
            same_dtype
            and same_shape
            and before_hash == after_hash
            and np.array_equal(before, after)
        )
        structural[field] = {
            "dtype_anchor": before.dtype.str,
            "dtype_a3": after.dtype.str,
            "shape": list(before.shape),
            "anchor_sha256": before_hash,
            "a3_sha256": after_hash,
            "exact": exact,
        }
        if not exact:
            raise OpacityAuditError(f"structural field is not exact: {field}")

    raw_anchor = _require_finite(np.asarray(anchor["opacity"]), "anchor raw opacity")
    raw_a3 = _require_finite(np.asarray(candidate["opacity"]), "A3 raw opacity")
    if raw_anchor.shape != raw_a3.shape:
        raise OpacityAuditError(
            f"raw opacity shape mismatch: anchor={raw_anchor.shape} A3={raw_a3.shape}"
        )
    activated_anchor = _sigmoid(raw_anchor)
    activated_a3 = _sigmoid(raw_a3)
    return {
        "gaussian_count": int(len(anchor)),
        "structural_fields": structural,
        "all_structural_fields_exact": True,
        "activated_opacity": {
            "semantics": "sigmoid(raw PLY opacity logit)",
            "anchor": _distribution(activated_anchor),
            "a3": _distribution(activated_a3),
            "a3_minus_anchor": _delta_summary(activated_anchor, activated_a3),
            "raw_array_sha256": {
                "anchor": _array_sha256("opacity", raw_anchor),
                "a3": _array_sha256("opacity", raw_a3),
            },
        },
    }


def _safe_manifest_output(manifest_path: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise OpacityAuditError(f"unsafe opacity proxy path in manifest: {relative!r}")
    base = manifest_path.resolve().parent
    target = base.joinpath(*pure.parts).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise OpacityAuditError(f"opacity proxy escapes manifest root: {relative!r}") from exc
    return target


def _manifest_proxy_index(manifest_path: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpacityAuditError(f"cannot read render mapping manifest: {manifest_path}") from exc
    if manifest.get("opacity_proxy_saved") is not True:
        raise OpacityAuditError(f"manifest does not declare saved opacity proxies: {manifest_path}")
    expected_semantics = "black_bg_plus_white_override_color_render"
    if manifest.get("opacity_proxy_semantics") != expected_semantics:
        raise OpacityAuditError(
            f"unexpected opacity proxy semantics in {manifest_path}: "
            f"{manifest.get('opacity_proxy_semantics')!r}"
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_PROXY_VIEWS:
        raise OpacityAuditError(
            f"manifest must contain exactly {EXPECTED_PROXY_VIEWS} entries: {manifest_path}"
        )

    index: dict[str, Path] = {}
    declared_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise OpacityAuditError(f"non-object mapping entry in {manifest_path}")
        source_name = str(entry.get("source_image_name", ""))
        source_stem = Path(source_name).stem
        if not source_stem or source_stem in index:
            raise OpacityAuditError(f"duplicate/empty source stem in {manifest_path}: {source_stem!r}")
        output = entry.get("output")
        hashes = entry.get("output_sha256")
        if not isinstance(output, dict) or not isinstance(hashes, dict):
            raise OpacityAuditError(f"mapping entry lacks output/hash objects: {source_name}")
        relative = output.get("opacity_proxy")
        declared = hashes.get("opacity_proxy")
        if not isinstance(relative, str) or not isinstance(declared, str):
            raise OpacityAuditError(f"mapping entry lacks opacity proxy/hash: {source_name}")
        path = _safe_manifest_output(manifest_path, relative)
        if path.suffix.lower() != ".npy" or not path.is_file():
            raise OpacityAuditError(f"opacity proxy is missing or not NPY: {path}")
        actual = sha256_file(path)
        expected = _validate_expected_sha256(declared, f"opacity proxy {source_name}")
        if actual != expected:
            raise OpacityAuditError(
                f"opacity proxy SHA-256 mismatch for {source_name}: expected={expected} actual={actual}"
            )
        index[source_stem] = path
        declared_hashes[source_stem] = expected
    return index, {
        "path": str(manifest_path.resolve()),
        "sha256": sha256_file(manifest_path),
        "semantics": expected_semantics,
        "view_count": len(index),
        "opacity_proxy_sha256": declared_hashes,
    }


def _proxy_row(scope: str, view_id: str, anchor: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if anchor.dtype != candidate.dtype:
        raise OpacityAuditError(
            f"opacity proxy dtype mismatch for {view_id}: anchor={anchor.dtype} A3={candidate.dtype}"
        )
    if anchor.shape != candidate.shape:
        raise OpacityAuditError(
            f"opacity proxy shape mismatch for {view_id}: anchor={anchor.shape} A3={candidate.shape}"
        )
    before = _require_finite(anchor, f"anchor opacity proxy {view_id}").astype(np.float64, copy=False)
    after = _require_finite(candidate, f"A3 opacity proxy {view_id}").astype(np.float64, copy=False)
    delta = after - before
    absolute = np.abs(delta)
    row: dict[str, Any] = {
        "scope": scope,
        "view_id": view_id,
        "count": int(before.size),
        "anchor_mean": float(np.mean(before)),
        "a3_mean": float(np.mean(after)),
        "signed_delta_mean": float(np.mean(delta)),
        "abs_delta_mean": float(np.mean(absolute)),
        "abs_delta_median": float(np.median(absolute)),
        "abs_delta_p95": float(np.percentile(absolute, 95)),
        "abs_delta_p99": float(np.percentile(absolute, 99)),
        "abs_delta_max": float(np.max(absolute)),
        "rmse": float(math.sqrt(float(np.mean(np.square(delta))))),
    }
    for threshold in DELTA_THRESHOLDS:
        row[f"fraction_abs_delta_gt_{threshold:.2f}"] = float(np.mean(absolute > threshold))
    return row


def compare_opacity_proxy_manifests(
    anchor_manifest: Path,
    a3_manifest: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_index, anchor_meta = _manifest_proxy_index(anchor_manifest)
    a3_index, a3_meta = _manifest_proxy_index(a3_manifest)
    if set(anchor_index) != set(a3_index):
        missing = sorted(set(anchor_index) - set(a3_index))
        extra = sorted(set(a3_index) - set(anchor_index))
        raise OpacityAuditError(f"opacity proxy view sets differ: missing={missing} extra={extra}")
    if len(anchor_index) != EXPECTED_PROXY_VIEWS:
        raise OpacityAuditError(f"opacity proxy comparison requires exactly {EXPECTED_PROXY_VIEWS} views")

    rows: list[dict[str, Any]] = []
    all_anchor: list[np.ndarray] = []
    all_a3: list[np.ndarray] = []
    for stem in sorted(anchor_index):
        before = np.load(anchor_index[stem], allow_pickle=False)
        after = np.load(a3_index[stem], allow_pickle=False)
        rows.append(_proxy_row("opacity_proxy_view", stem, before, after))
        all_anchor.append(np.asarray(before).reshape(-1))
        all_a3.append(np.asarray(after).reshape(-1))
    aggregate = _proxy_row(
        "opacity_proxy_pixel_micro",
        "ALL",
        np.concatenate(all_anchor),
        np.concatenate(all_a3),
    )
    frame_macro = {
        "view_count": len(rows),
        "mean_abs_delta_mean": float(np.mean([row["abs_delta_mean"] for row in rows])),
        "mean_rmse": float(np.mean([row["rmse"] for row in rows])),
        "max_abs_delta": float(max(row["abs_delta_max"] for row in rows)),
    }
    return {
        "anchor_manifest": anchor_meta,
        "a3_manifest": a3_meta,
        "view_count": len(rows),
        "pixel_micro": aggregate,
        "frame_macro": frame_macro,
    }, rows


CSV_FIELDS = (
    "scope",
    "view_id",
    "count",
    "anchor_mean",
    "a3_mean",
    "signed_delta_mean",
    "abs_delta_mean",
    "abs_delta_median",
    "abs_delta_p95",
    "abs_delta_p99",
    "abs_delta_max",
    "rmse",
    "fraction_abs_delta_gt_0.01",
    "fraction_abs_delta_gt_0.05",
    "fraction_abs_delta_gt_0.10",
)


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


def run_audit(
    *,
    anchor_ply: Path,
    a3_ply: Path,
    anchor_ply_sha256: str,
    a3_ply_sha256: str,
    anchor_opacity_manifest: Path | None = None,
    a3_opacity_manifest: Path | None = None,
    anchor_opacity_manifest_sha256: str | None = None,
    a3_opacity_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    anchor_ply = anchor_ply.resolve()
    a3_ply = a3_ply.resolve()
    anchor_sha = _verify_sha256(anchor_ply, anchor_ply_sha256, "anchor PLY")
    a3_sha = _verify_sha256(a3_ply, a3_ply_sha256, "A3 PLY")
    ply_audit = audit_ply_pair(anchor_ply, a3_ply)
    activated = ply_audit["activated_opacity"]
    ply_delta = activated["a3_minus_anchor"]
    csv_rows: list[dict[str, Any]] = [{
        "scope": "ply_activated_opacity",
        "view_id": "ALL",
        "count": activated["anchor"]["count"],
        "anchor_mean": activated["anchor"]["mean"],
        "a3_mean": activated["a3"]["mean"],
        "signed_delta_mean": ply_delta["signed"]["mean"],
        "abs_delta_mean": ply_delta["absolute"]["mean"],
        "abs_delta_median": ply_delta["absolute"]["median"],
        "abs_delta_p95": ply_delta["absolute"]["p95"],
        "abs_delta_p99": ply_delta["absolute"]["p99"],
        "abs_delta_max": ply_delta["absolute"]["max"],
        "rmse": ply_delta["rmse"],
        **{
            f"fraction_abs_delta_gt_{threshold:.2f}": ply_delta[
                "fractions_abs_delta_gt"
            ][f"{threshold:.2f}"]
            for threshold in DELTA_THRESHOLDS
        },
    }]

    optional_values = (
        anchor_opacity_manifest,
        a3_opacity_manifest,
        anchor_opacity_manifest_sha256,
        a3_opacity_manifest_sha256,
    )
    if any(value is not None for value in optional_values) and not all(
        value is not None for value in optional_values
    ):
        raise OpacityAuditError(
            "anchor/A3 opacity manifests and both expected manifest SHA-256 values "
            "must all be supplied or all be omitted"
        )
    proxy_audit = None
    if anchor_opacity_manifest is not None and a3_opacity_manifest is not None:
        assert anchor_opacity_manifest_sha256 is not None
        assert a3_opacity_manifest_sha256 is not None
        anchor_manifest_sha = _verify_sha256(
            anchor_opacity_manifest.resolve(),
            anchor_opacity_manifest_sha256,
            "anchor opacity manifest",
        )
        a3_manifest_sha = _verify_sha256(
            a3_opacity_manifest.resolve(),
            a3_opacity_manifest_sha256,
            "A3 opacity manifest",
        )
        proxy_audit, proxy_rows = compare_opacity_proxy_manifests(
            anchor_opacity_manifest.resolve(),
            a3_opacity_manifest.resolve(),
        )
        if proxy_audit["anchor_manifest"]["sha256"] != anchor_manifest_sha:
            raise OpacityAuditError("anchor opacity manifest changed during audit")
        if proxy_audit["a3_manifest"]["sha256"] != a3_manifest_sha:
            raise OpacityAuditError("A3 opacity manifest changed during audit")
        csv_rows.extend(proxy_rows)
        csv_rows.append(proxy_audit["pixel_micro"])

    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "claim": (
            "A3 preserves RGB-anchor xyz/scale/rotation arrays exactly while allowing "
            "activated opacity to adapt"
        ),
        "inputs": {
            "anchor_ply": {"path": str(anchor_ply), "sha256": anchor_sha},
            "a3_ply": {"path": str(a3_ply), "sha256": a3_sha},
        },
        "ply_audit": ply_audit,
        "opacity_proxy_audit": proxy_audit,
    }
    return payload, csv_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-ply", required=True, type=Path)
    parser.add_argument("--a3-ply", required=True, type=Path)
    parser.add_argument("--anchor-ply-sha256", required=True)
    parser.add_argument("--a3-ply-sha256", required=True)
    parser.add_argument("--anchor-opacity-manifest", type=Path)
    parser.add_argument("--a3-opacity-manifest", type=Path)
    parser.add_argument("--anchor-opacity-manifest-sha256")
    parser.add_argument("--a3-opacity-manifest-sha256")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload, rows = run_audit(
        anchor_ply=args.anchor_ply,
        a3_ply=args.a3_ply,
        anchor_ply_sha256=args.anchor_ply_sha256,
        a3_ply_sha256=args.a3_ply_sha256,
        anchor_opacity_manifest=args.anchor_opacity_manifest,
        a3_opacity_manifest=args.a3_opacity_manifest,
        anchor_opacity_manifest_sha256=args.anchor_opacity_manifest_sha256,
        a3_opacity_manifest_sha256=args.a3_opacity_manifest_sha256,
    )
    _write_json(args.report, payload, args.overwrite)
    _write_csv(args.csv, rows, args.overwrite)
    print(json.dumps({"status": "passed", "report": str(args.report), "csv": str(args.csv)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
