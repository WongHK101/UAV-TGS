#!/usr/bin/env python3
"""Build the formal temperature-evaluation support domain.

The output is the intersection of the float-temperature remap validity mask
and the shared RGB-anchor opacity proxy.  This tool does not change evaluator
semantics and never applies a method-specific thermal support estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np


SCHEMA_NAME = "uav-tgs-formal-temperature-support"
SCHEMA_VERSION = 1
OPACITY_PROXY_SEMANTICS = "black_bg_plus_white_override_color_render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _portable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_object(path: Path, label: str) -> Dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return value


def _name_stem(value: Any, label: str) -> str:
    text = str(value or "").replace("\\", "/")
    stem = PurePosixPath(text).stem
    if not stem or stem in {".", ".."}:
        raise ValueError(f"{label} has no valid image stem: {value!r}")
    return stem


def _unique_by_name(records: Iterable[Mapping[str, Any]], field: str, label: str) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    casefolded: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} contains a non-object record")
        name = _name_stem(record.get(field), f"{label}.{field}")
        collision = name.casefold()
        if collision in casefolded:
            raise ValueError(
                f"{label} has duplicate image stem {name!r}: also used by "
                f"{casefolded[collision]!r}"
            )
        casefolded[collision] = name
        indexed[name] = record
    return indexed


def _valid_sha(value: Any, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} is not a SHA-256 digest: {value!r}")
    return text


def _portable_relative(value: Any, prefix: str, label: str) -> str:
    text = str(value or "").replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
        raise ValueError(f"{label} must be a portable relative path: {value!r}")
    parts = list(path.parts)
    if parts and parts[0] == prefix:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"{label} does not identify a file: {value!r}")
    return PurePosixPath(*parts).as_posix()


def _index_npy(root: Path, label: str) -> Dict[str, tuple[Path, str]]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} root does not exist: {resolved}")
    indexed: Dict[str, tuple[Path, str]] = {}
    casefolded: Dict[str, str] = {}
    for path in sorted(resolved.rglob("*.npy"), key=lambda item: item.as_posix()):
        name = path.stem
        collision = name.casefold()
        if collision in casefolded:
            raise ValueError(
                f"{label} root has duplicate basename stem {name!r}: also used by "
                f"{casefolded[collision]!r}"
            )
        casefolded[collision] = name
        indexed[name] = (path, path.relative_to(resolved).as_posix())
    if not indexed:
        raise FileNotFoundError(f"{label} root contains no NPY files: {resolved}")
    return indexed


def _load_valid_support(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.bool_:
        raise TypeError(f"valid_support must be boolean NPY: {path} ({values.dtype})")
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"valid_support must be a non-empty 2D array: {path} ({values.shape})")
    return values


def _load_opacity_proxy(path: Path) -> np.ndarray:
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32:
        raise TypeError(f"opacity_proxy must be float32 NPY: {path} ({values.dtype})")
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"opacity_proxy must be a non-empty 2D array: {path} ({values.shape})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"opacity_proxy contains NaN or infinity: {path}")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"opacity_proxy must lie in [0,1]: {path} (min={minimum}, max={maximum})"
        )
    return values


def _split_test_records(payload: Mapping[str, Any], expected_count: int) -> Dict[str, Mapping[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("split manifest must contain a records list")
    selected = [record for record in records if isinstance(record, Mapping) and record.get("split") == "test"]
    indexed = _unique_by_name(selected, "pair_id", "split test records")
    if len(indexed) != expected_count:
        raise ValueError(
            f"split manifest must select exactly {expected_count} test names, got {len(indexed)}"
        )
    counts = payload.get("counts")
    if not isinstance(counts, Mapping) or int(counts.get("test", -1)) != expected_count:
        raise ValueError(
            f"split manifest counts.test must equal {expected_count}: {counts!r}"
        )
    for name, record in indexed.items():
        _valid_sha(record.get("hash"), f"split record hash for {name}")
    return indexed


def combine_formal_support(
    *,
    split_manifest: Path,
    valid_support_root: Path,
    valid_support_manifest: Path,
    opacity_proxy_root: Path,
    opacity_proxy_manifest: Path,
    output_root: Path,
    opacity_threshold: float,
    expected_test_count: int = 80,
) -> Dict[str, Any]:
    if expected_test_count <= 0:
        raise ValueError("expected_test_count must be positive")
    if not math.isfinite(opacity_threshold) or not 0.0 <= opacity_threshold <= 1.0:
        raise ValueError("opacity_threshold must be finite and lie in [0,1]")

    split_path = Path(split_manifest).resolve()
    valid_manifest_path = Path(valid_support_manifest).resolve()
    opacity_manifest_path = Path(opacity_proxy_manifest).resolve()
    split_payload = _load_object(split_path, "split manifest")
    valid_payload = _load_object(valid_manifest_path, "valid-support manifest")
    opacity_payload = _load_object(opacity_manifest_path, "opacity-proxy manifest")
    split_manifest_sha = _sha256(split_path)
    valid_manifest_sha = _sha256(valid_manifest_path)
    opacity_manifest_sha = _sha256(opacity_manifest_path)
    split_records = _split_test_records(split_payload, expected_test_count)
    expected_names = set(split_records)

    valid_files = valid_payload.get("files")
    if not isinstance(valid_files, list):
        raise ValueError("valid-support manifest must contain a files list")
    valid_records = _unique_by_name(valid_files, "image_name", "valid-support manifest")
    missing_valid_records = sorted(expected_names - set(valid_records))
    if missing_valid_records:
        raise ValueError(
            "valid-support manifest is missing test names: " + ", ".join(missing_valid_records)
        )

    if opacity_payload.get("split") != "test":
        raise ValueError("opacity-proxy manifest must describe the test split")
    if opacity_payload.get("opacity_proxy_saved") is not True:
        raise ValueError("opacity-proxy manifest does not declare saved opacity proxies")
    if opacity_payload.get("opacity_proxy_semantics") != OPACITY_PROXY_SEMANTICS:
        raise ValueError("opacity-proxy manifest has unexpected proxy semantics")
    if opacity_payload.get("support_threshold_applied") is not False:
        raise ValueError("opacity-proxy rendering must not apply a support threshold")
    opacity_entries = opacity_payload.get("entries")
    if not isinstance(opacity_entries, list):
        raise ValueError("opacity-proxy manifest must contain an entries list")
    opacity_records = _unique_by_name(
        opacity_entries,
        "source_image_name",
        "opacity-proxy manifest",
    )
    if set(opacity_records) != expected_names:
        missing = sorted(expected_names - set(opacity_records))
        extra = sorted(set(opacity_records) - expected_names)
        raise ValueError(
            f"opacity-proxy manifest names differ from the {expected_test_count} test names; "
            f"missing={missing}, extra={extra}"
        )

    valid_index = _index_npy(Path(valid_support_root), "valid-support")
    opacity_index = _index_npy(Path(opacity_proxy_root), "opacity-proxy")
    if set(opacity_index) != expected_names:
        missing = sorted(expected_names - set(opacity_index))
        extra = sorted(set(opacity_index) - expected_names)
        raise ValueError(
            f"opacity-proxy files differ from the {expected_test_count} test names; "
            f"missing={missing}, extra={extra}"
        )

    jobs = []
    for name in sorted(expected_names):
        if name not in valid_index:
            raise FileNotFoundError(f"valid-support file missing for test name {name}")
        valid_record = valid_records[name]
        valid_metadata = valid_record.get("valid_support")
        if not isinstance(valid_metadata, Mapping):
            raise ValueError(f"valid-support metadata missing for {name}")
        valid_path, valid_relative = valid_index[name]
        declared_valid_relative = _portable_relative(
            valid_metadata.get("relative_path"),
            "valid_support",
            f"valid-support relative_path for {name}",
        )
        if declared_valid_relative != valid_relative:
            raise ValueError(
                f"valid-support path mismatch for {name}: manifest={declared_valid_relative}, "
                f"root={valid_relative}"
            )
        valid_sha = _sha256(valid_path)
        if valid_sha != _valid_sha(valid_metadata.get("sha256"), f"valid-support SHA for {name}"):
            raise ValueError(f"valid-support source hash mismatch for {name}")

        opacity_record = opacity_records[name]
        output = opacity_record.get("output")
        output_sha = opacity_record.get("output_sha256")
        if not isinstance(output, Mapping) or not isinstance(output_sha, Mapping):
            raise ValueError(f"opacity-proxy output metadata missing for {name}")
        declared_opacity_relative = _portable_relative(
            output.get("opacity_proxy"),
            "opacity_proxy",
            f"opacity-proxy relative path for {name}",
        )
        opacity_path, opacity_relative = opacity_index[name]
        if declared_opacity_relative != opacity_relative:
            raise ValueError(
                f"opacity-proxy path mismatch for {name}: manifest={declared_opacity_relative}, "
                f"root={opacity_relative}"
            )
        opacity_sha = _sha256(opacity_path)
        if opacity_sha != _valid_sha(output_sha.get("opacity_proxy"), f"opacity-proxy SHA for {name}"):
            raise ValueError(f"opacity-proxy source hash mismatch for {name}")

        valid = _load_valid_support(valid_path)
        opacity = _load_opacity_proxy(opacity_path)
        declared_valid_shape = valid_metadata.get("shape")
        if declared_valid_shape != list(valid.shape):
            raise ValueError(
                f"valid-support manifest shape mismatch for {name}: "
                f"declared={declared_valid_shape}, actual={list(valid.shape)}"
            )
        if valid_metadata.get("dtype") != "bool":
            raise ValueError(
                f"valid-support manifest dtype must be 'bool' for {name}: "
                f"{valid_metadata.get('dtype')!r}"
            )
        if valid.shape != opacity.shape:
            raise ValueError(
                f"support shape mismatch for {name}: valid={valid.shape}, opacity={opacity.shape}"
            )
        combined = valid & (opacity > opacity_threshold)
        supported_pixels = int(np.count_nonzero(combined))
        if supported_pixels == 0:
            raise ValueError(f"combined formal support is empty for test name {name}")
        jobs.append(
            {
                "name": name,
                "split_record": split_records[name],
                "valid_path": valid_path,
                "valid_relative": valid_relative,
                "valid_sha256": valid_sha,
                "opacity_path": opacity_path,
                "opacity_relative": opacity_relative,
                "opacity_sha256": opacity_sha,
                "shape": list(valid.shape),
                "valid_pixels": int(np.count_nonzero(valid)),
                "opacity_supported_pixels": int(np.count_nonzero(opacity > opacity_threshold)),
                "supported_pixels": supported_pixels,
                "total_pixels": int(valid.size),
            }
        )

    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        output_records = []
        for job in jobs:
            if _sha256(job["valid_path"]) != job["valid_sha256"]:
                raise RuntimeError(f"valid-support source changed during build: {job['name']}")
            if _sha256(job["opacity_path"]) != job["opacity_sha256"]:
                raise RuntimeError(f"opacity-proxy source changed during build: {job['name']}")
            valid = _load_valid_support(job["valid_path"])
            opacity = _load_opacity_proxy(job["opacity_path"])
            if list(valid.shape) != job["shape"] or opacity.shape != valid.shape:
                raise RuntimeError(f"support source shape changed during build: {job['name']}")
            combined_bool = valid & (opacity > opacity_threshold)
            combined_float = combined_bool.astype(np.float32)
            bool_relative = (Path("bool") / f"{job['name']}.npy").as_posix()
            float_relative = (Path("float") / f"{job['name']}.npy").as_posix()
            bool_path = temporary / bool_relative
            float_path = temporary / float_relative
            bool_path.parent.mkdir(parents=True, exist_ok=True)
            float_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(bool_path, combined_bool, allow_pickle=False)
            np.save(float_path, combined_float, allow_pickle=False)
            output_records.append(
                {
                    "name": job["name"],
                    "split_record_hash": _valid_sha(
                        job["split_record"].get("hash"),
                        f"split record hash for {job['name']}",
                    ),
                    "shape": job["shape"],
                    "sources": {
                        "valid_support": {
                            "relative_path": job["valid_relative"],
                            "sha256": job["valid_sha256"],
                            "dtype": "bool",
                        },
                        "opacity_proxy": {
                            "relative_path": job["opacity_relative"],
                            "sha256": job["opacity_sha256"],
                            "dtype": "float32",
                            "semantics": OPACITY_PROXY_SEMANTICS,
                        },
                    },
                    "outputs": {
                        "bool": {
                            "relative_path": bool_relative,
                            "sha256": _sha256(bool_path),
                            "dtype": "bool",
                        },
                        "float": {
                            "relative_path": float_relative,
                            "sha256": _sha256(float_path),
                            "dtype": "float32",
                        },
                    },
                    "pixels": {
                        "valid_support": job["valid_pixels"],
                        "opacity_above_threshold": job["opacity_supported_pixels"],
                        "combined_supported": job["supported_pixels"],
                        "total": job["total_pixels"],
                    },
                }
            )

        if _sha256(split_path) != split_manifest_sha:
            raise RuntimeError("split manifest changed during build")
        if _sha256(valid_manifest_path) != valid_manifest_sha:
            raise RuntimeError("valid-support manifest changed during build")
        if _sha256(opacity_manifest_path) != opacity_manifest_sha:
            raise RuntimeError("opacity-proxy manifest changed during build")

        total_pixels = sum(record["pixels"]["total"] for record in output_records)
        supported_pixels = sum(
            record["pixels"]["combined_supported"] for record in output_records
        )
        core = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "split": "test",
            "expected_test_count": int(expected_test_count),
            "source_manifests": {
                "split": {
                    "sha256": split_manifest_sha,
                    "split_hash": split_payload.get("split_hash"),
                },
                "valid_support": {
                    "sha256": valid_manifest_sha,
                    "schema": valid_payload.get("schema"),
                },
                "opacity_proxy": {
                    "sha256": opacity_manifest_sha,
                    "schema_name": opacity_payload.get("schema_name"),
                    "schema_version": opacity_payload.get("schema_version"),
                    "iteration": opacity_payload.get("iteration"),
                },
            },
            "policy": {
                "expression": "valid_support AND (opacity_proxy > opacity_threshold)",
                "opacity_threshold": float(opacity_threshold),
                "comparison": "strict_greater_than",
                "opacity_proxy_semantics": OPACITY_PROXY_SEMANTICS,
                "threshold_applied_only_by_this_combiner": True,
            },
            "summary": {
                "file_count": len(output_records),
                "supported_pixels": supported_pixels,
                "total_pixels": total_pixels,
                "supported_ratio": float(supported_pixels / total_pixels),
            },
            "records": output_records,
        }
        payload = {
            **core,
            "portable_content_sha256": _portable_hash(core),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        manifest_sha = _sha256(manifest_path)
        (temporary / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n",
            encoding="ascii",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "output_root": str(destination),
        "manifest_path": str(destination / "manifest.json"),
        "manifest_sha256": manifest_sha,
        "portable_content_sha256": payload["portable_content_sha256"],
        "file_count": len(output_records),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--valid-support-root", required=True, type=Path)
    parser.add_argument("--valid-support-manifest", required=True, type=Path)
    parser.add_argument("--opacity-proxy-root", required=True, type=Path)
    parser.add_argument("--opacity-proxy-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--opacity-threshold",
        required=True,
        type=float,
        help="Explicit proxy threshold; the formal Building example uses 0.01",
    )
    parser.add_argument("--expected-test-count", type=int, default=80)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = combine_formal_support(
        split_manifest=args.split_manifest,
        valid_support_root=args.valid_support_root,
        valid_support_manifest=args.valid_support_manifest,
        opacity_proxy_root=args.opacity_proxy_root,
        opacity_proxy_manifest=args.opacity_proxy_manifest,
        output_root=args.output_root,
        opacity_threshold=args.opacity_threshold,
        expected_test_count=args.expected_test_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
