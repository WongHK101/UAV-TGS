#!/usr/bin/env python3
"""Build an explicit-Celsius bundle for Gaussian responsibility diagnostics.

The source is a formal all-split thermal-canonical diagnostic bundle emitted
by ``export_gaussian_probe_bundle.py``.  This tool does not mutate that bundle.
It publishes a compact derived model bundle containing the depth/top-
contributor arrays required by ``evaluate_depth_definitions.py`` plus three
explicit temperature arrays:

* ``render_temperature_c`` is the nearest fixed repository Hot-Iron inverse of
  the rendered (possibly off-LUT) canonical RGB;
* ``target_temperature_c`` is copied directly from the bound float32 TSDK-
  referenced undistorted NPY (never reconstructed from a PNG); and
* ``temperature_valid_mask`` is the bound boolean undistortion support.

Every association is fail-closed and stem-based only after uniqueness has
been proved.  The output contract is ``uav-tgs-temperature-responsibility-v1``
and is directly consumable by ``evaluate_depth_definitions.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.thermal_radiometry.palette_lut import (
    PALETTE_NAME,
    PALETTE_SIZE,
    hot_iron_lut,
    lut_sha256,
    resolve_temperature_range,
    rgb_to_temperature,
    temperature_to_rgb,
)
from tools.thermal_radiometry.bind_formal_scene import (
    validate_binding_manifest_self_hash,
)


PROTOCOL = "uav-tgs-temperature-responsibility-v1"
SOURCE_RENDER_BINDING_PROTOCOL = "uav-tgs-formal-thermal-render-binding-v1"
SEMANTICS = "TSDK-referenced apparent-temperature consistency"
UNDISTORTED_SCHEMA = "uav-tgs-undistorted-temperature-v1"
CANONICAL_SCHEMA = "uav-tgs-canonical-hot-iron-v1"
FORMAL_SPLITS = frozenset({"train", "guard", "test"})
OUTPUT_MODEL_MANIFEST = "split_manifest.json"
OUTPUT_CONTRACT_MANIFEST = "temperature_responsibility_manifest.json"
TEMPERATURE_KEYS = {
    "rendered": "render_temperature_c",
    "target": "target_temperature_c",
    "valid_mask": "temperature_valid_mask",
}
DIAGNOSTIC_KEYS = (
    "depth_expected_alpha_normalized",
    "depth_transmittance_median",
    "depth_max_contribution",
    "accumulated_opacity",
    "top_contributor_index",
    "top_contributor_weight",
)
DIAGNOSTIC_SEMANTICS = {
    "depth_expected_alpha_normalized": "metric camera-z; sum(alpha*T*z)/sum(alpha*T)",
    "depth_transmittance_median": "metric camera-z at first accepted contributor where transmittance <= 0.5; zero if absent",
    "depth_max_contribution": "metric camera-z of Gaussian maximizing alpha*T",
    "top_contributor_index": "zero-based Gaussian index maximizing alpha*T; -1 if absent",
    "top_contributor_weight": "unnormalized compositing weight alpha*T",
    "accumulated_opacity": "sum of accepted alpha*T weights",
}
FORBIDDEN_RGB_KEYS = frozenset({"render_rgb", "target_rgb"})
SOURCE_TEMPERATURE_KEYS = frozenset(
    {
        "render_temperature_c",
        "temperature_render_c",
        "rendered_temperature_c",
        "target_temperature_c",
        "temperature_target_c",
        "ground_truth_temperature_c",
        "temperature_valid_mask",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(str(value).strip().lower()))


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    with resolved.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _producer_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        error = ""
    except Exception as exc:  # pragma: no cover - only for non-git deployments
        commit = ""
        status = "git-identity-unavailable"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "script_path": str(script),
        "script_sha256": _sha256(script),
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "git_error": error,
    }


def _normalised_stem(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return Path(text).stem.lower() if text else ""


def _scene(payload: Mapping[str, Any], *, label: str) -> str:
    value = str(payload.get("scene_name", payload.get("scene", ""))).strip()
    if not value:
        raise ValueError(f"{label} is missing scene/scene_name")
    return value


def _as_hw(value: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{label} must have shape HxW or 1xHxW, got {array.shape}")
    return array


def _as_hwc_rgb(
    value: np.ndarray, *, expected_shape: tuple[int, int], label: str
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(
            f"{label} must be floating-point canonical RGB, got {array.dtype}"
        )
    if array.ndim != 3:
        raise ValueError(f"{label} must be CxHxW or HxWx3, got {array.shape}")
    if array.shape == (3, *expected_shape):
        array = np.moveaxis(array, 0, -1)
    elif array.shape != (*expected_shape, 3):
        raise ValueError(
            f"{label} shape {array.shape} does not match camera shape {expected_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN/Inf")
    # Gaussian renders are normalized floating RGB.  Do not clip: nearest-LUT
    # projection itself supplies the well-defined endpoint behavior, while the
    # distance preserves evidence that a render is off the display manifold.
    return np.asarray(array, dtype=np.float32) * np.float32(255.0)


def _record_aliases(record: Mapping[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for key in (
        "pair_id",
        "image_name",
        "camera_name",
        "thermal_camera_name",
        "filename",
        "name",
        "relative_id",
        "relative_input",
        "relative_output",
    ):
        stem = _normalised_stem(record.get(key))
        if stem:
            aliases.add(stem)
    original = record.get("original_files")
    if isinstance(original, Mapping):
        for value in original.values():
            stem = _normalised_stem(value)
            if stem:
                aliases.add(stem)
    for key in ("output_temperature", "valid_support", "input_temperature"):
        value = record.get(key)
        if isinstance(value, Mapping):
            for path_key in ("relative_path", "path"):
                stem = _normalised_stem(value.get(path_key))
                if stem:
                    aliases.add(stem)
    return aliases


def _unique_record_index(
    records: Any,
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} records/files must be a non-empty list")
    typed: list[dict[str, Any]] = []
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{label} record {position} must be an object")
        aliases = _record_aliases(record)
        if not aliases:
            raise ValueError(f"{label} record {position} has no usable identity")
        for alias in aliases:
            previous = index.get(alias)
            if previous is not None and previous is not record:
                raise ValueError(f"{label} has ambiguous stem {alias!r}")
            index[alias] = record
        typed.append(record)
    return typed, index


def _formal_records(
    payload: Mapping[str, Any],
    *,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    records, aliases = _unique_record_index(
        payload.get("records"), label="formal split"
    )
    if len(records) != int(expected_count):
        raise ValueError(
            f"Formal split must contain exactly {expected_count} records, observed {len(records)}"
        )
    primary: dict[str, dict[str, Any]] = {}
    counts = {label: 0 for label in sorted(FORMAL_SPLITS)}
    for record in records:
        pair_id = _normalised_stem(record.get("pair_id"))
        if not pair_id or pair_id in primary:
            raise ValueError(f"Formal split has duplicate/missing pair_id {pair_id!r}")
        split = str(record.get("split", "")).strip().lower()
        if split not in FORMAL_SPLITS:
            raise ValueError(
                f"Formal split record {pair_id!r} has invalid split {split!r}"
            )
        primary[pair_id] = record
        counts[split] += 1
    if any(counts[label] <= 0 for label in FORMAL_SPLITS):
        raise ValueError(
            f"Formal split must include train/guard/test, observed {counts}"
        )
    declared = payload.get("counts")
    if isinstance(declared, Mapping):
        for split, count in counts.items():
            if split in declared and int(declared[split]) != count:
                raise ValueError(f"Formal split declared {split} count mismatch")
        if "total" in declared and int(declared["total"]) != len(records):
            raise ValueError("Formal split declared total count mismatch")
    return records, aliases, counts


def _view_index(
    payload: Mapping[str, Any], *, expected_count: int
) -> dict[str, dict[str, Any]]:
    views = payload.get("views")
    if not isinstance(views, list) or len(views) != int(expected_count):
        raise ValueError(
            f"Source model manifest must contain exactly {expected_count} views"
        )
    result: dict[str, dict[str, Any]] = {}
    for item in views:
        if not isinstance(item, dict):
            raise ValueError("Source model views must be objects")
        name = str(item.get("image_name", "")).strip()
        stem = _normalised_stem(name)
        if not name or not stem or stem in result:
            raise ValueError(f"Source model has duplicate/missing image_name {name!r}")
        result[stem] = item
    return result


def _resolve_source_npz(manifest_path: Path, view: Mapping[str, Any]) -> Path:
    root = manifest_path.parent.resolve()
    path = (root / str(view.get("npz_file", ""))).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source view NPZ escapes manifest root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = str(view.get("npz_sha256", "")).strip().lower()
    if not _is_sha256(expected_sha) or _sha256(path) != expected_sha:
        raise RuntimeError(f"Source view NPZ SHA-256 mismatch: {path}")
    if "npz_size_bytes" in view and int(view["npz_size_bytes"]) != int(
        path.stat().st_size
    ):
        raise RuntimeError(f"Source view NPZ size mismatch: {path}")
    return path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _require_output_tree_isolated(
    destination: Path, protected_roots: Mapping[str, Path]
) -> None:
    destination = destination.resolve()
    for label, root in protected_roots.items():
        root = Path(root).resolve()
        if destination == root or _is_within(destination, root) or _is_within(root, destination):
            raise ValueError(
                "Temperature responsibility output must be tree-isolated from "
                f"{label}: output={destination}, protected={root}"
            )


def _index_npy_root(root: Path, *, label: str) -> dict[str, Path]:
    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} root is missing: {resolved}")
    result: dict[str, Path] = {}
    for path in sorted(resolved.rglob("*.npy"), key=lambda item: item.as_posix()):
        stem = path.stem.lower()
        if stem in result:
            raise ValueError(f"{label} root has ambiguous NPY stem {stem!r}")
        result[stem] = path.resolve()
    if not result:
        raise FileNotFoundError(f"{label} root contains no NPY files: {resolved}")
    return result


def _mapping(record: Mapping[str, Any], key: str, *, label: str) -> Mapping[str, Any]:
    value = record.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} record is missing {key!r} metadata")
    return value


def _bound_npy(
    *,
    stem: str,
    root_index: Mapping[str, Path],
    metadata: Mapping[str, Any],
    label: str,
) -> tuple[Path, str]:
    path = root_index.get(stem)
    if path is None:
        raise FileNotFoundError(f"{label} NPY is missing for {stem!r}")
    metadata_stem = ""
    for key in ("relative_path", "path"):
        candidate = _normalised_stem(metadata.get(key))
        if candidate:
            metadata_stem = candidate
            break
    if metadata_stem != stem:
        raise ValueError(
            f"{label} manifest/path stem mismatch for {stem!r}: {metadata_stem!r}"
        )
    expected_sha = str(metadata.get("sha256", "")).strip().lower()
    if not _is_sha256(expected_sha):
        raise ValueError(f"{label} manifest has invalid SHA-256 for {stem!r}")
    observed_sha = _sha256(path)
    if observed_sha != expected_sha:
        raise RuntimeError(f"{label} NPY SHA-256 mismatch for {stem!r}: {path}")
    return path, observed_sha


def _validate_array_metadata(
    metadata: Mapping[str, Any],
    array: np.ndarray,
    *,
    expected_dtype: str,
    label: str,
) -> None:
    if "dtype" in metadata and str(metadata["dtype"]).strip().lower() != expected_dtype:
        raise ValueError(f"{label} manifest dtype must be {expected_dtype}")
    if "shape" in metadata:
        try:
            declared_shape = tuple(int(value) for value in metadata["shape"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} manifest shape is invalid") from exc
        if declared_shape != tuple(array.shape):
            raise ValueError(f"{label} manifest/array shape mismatch")


def _save_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a deterministic, pickle-free compressed NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w") as archive:
        for key in sorted(arrays):
            if not key or "/" in key or "\\" in key:
                raise ValueError(f"Unsafe NPZ key {key!r}")
            array = np.asarray(arrays[key])
            if array.dtype.hasobject:
                raise TypeError(f"Object arrays are forbidden in diagnostic NPZ: {key}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            )


def _split_for_record(record: Mapping[str, Any]) -> str:
    value = str(record.get("split", "")).strip().lower()
    if value not in FORMAL_SPLITS:
        raise ValueError(f"Invalid formal split label {value!r}")
    return value


def _off_lut_summary(distances: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = np.asarray(distances, dtype=np.float64)[valid]
    if selected.size == 0:
        raise ValueError("Temperature valid-support mask contains no valid pixels")
    return {
        "distance_units": "RGB-byte Euclidean",
        "valid_pixels": int(selected.size),
        "mean": float(np.mean(selected)),
        "p95": float(np.percentile(selected, 95.0)),
        "maximum": float(np.max(selected)),
        "exact_lut_fraction": float(np.count_nonzero(selected == 0.0) / selected.size),
    }


def build_temperature_responsibility_bundle(
    *,
    source_model_manifest_path: Path,
    source_render_binding_manifest_path: Path,
    formal_split_manifest_path: Path,
    tsdk_binding_manifest_path: Path,
    temperature_root: Path,
    temperature_manifest_path: Path,
    valid_support_root: Path,
    valid_support_manifest_path: Path,
    range_manifest_path: Path,
    canonical_manifest_path: Path,
    output_root: Path,
    expected_view_count: int = 559,
    chunk_pixels: int = 32768,
) -> dict[str, Any]:
    if int(expected_view_count) <= 0:
        raise ValueError("expected_view_count must be positive")
    if int(chunk_pixels) <= 0:
        raise ValueError("chunk_pixels must be positive")

    source_manifest_path = Path(source_model_manifest_path).resolve()
    render_binding_path = Path(source_render_binding_manifest_path).resolve()
    split_path = Path(formal_split_manifest_path).resolve()
    binding_path = Path(tsdk_binding_manifest_path).resolve()
    target_manifest_path = Path(temperature_manifest_path).resolve()
    support_manifest_path = Path(valid_support_manifest_path).resolve()
    range_path = Path(range_manifest_path).resolve()
    canonical_path = Path(canonical_manifest_path).resolve()
    destination = Path(output_root).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output root: {destination}")
    _require_output_tree_isolated(
        destination,
        {
            "source model bundle": source_manifest_path.parent,
            "source render binding tree": render_binding_path.parent,
            "formal split manifest tree": split_path.parent,
            "TSDK binding manifest tree": binding_path.parent,
            "temperature root": Path(temperature_root).resolve(),
            "temperature manifest tree": target_manifest_path.parent,
            "valid-support root": Path(valid_support_root).resolve(),
            "valid-support manifest tree": support_manifest_path.parent,
            "range manifest tree": range_path.parent,
            "canonical manifest tree": canonical_path.parent,
        },
    )

    source = _load_json(source_manifest_path, label="source model manifest")
    render_binding = _load_json(
        render_binding_path, label="source thermal render binding manifest"
    )
    formal = _load_json(split_path, label="formal split manifest")
    binding = _load_json(binding_path, label="TSDK binding manifest")
    target_manifest = _load_json(
        target_manifest_path, label="temperature target manifest"
    )
    support_manifest = _load_json(support_manifest_path, label="valid-support manifest")
    range_manifest = _load_json(range_path, label="formal range manifest")
    canonical_manifest = _load_json(canonical_path, label="canonical Hot-Iron manifest")
    input_paths = {
        "source_model_manifest": source_manifest_path,
        "source_render_binding_manifest": render_binding_path,
        "formal_split_manifest": split_path,
        "tsdk_binding_manifest": binding_path,
        "temperature_manifest": target_manifest_path,
        "valid_support_manifest": support_manifest_path,
        "range_manifest": range_path,
        "canonical_manifest": canonical_path,
    }
    input_hashes = {key: _sha256(path) for key, path in input_paths.items()}

    scene_name = _scene(source, label="source model manifest")
    for payload, label in (
        (formal, "formal split manifest"),
        (binding, "TSDK binding manifest"),
        (range_manifest, "formal range manifest"),
        (render_binding, "source thermal render binding manifest"),
    ):
        if _scene(payload, label=label) != scene_name:
            raise ValueError(
                f"{label} scene does not match source model scene {scene_name!r}"
            )

    if (
        str(source.get("appearance_modality", "")).strip().lower()
        != "thermal_canonical"
    ):
        raise ValueError(
            "Temperature responsibility requires source appearance_modality=thermal_canonical; "
            "RGB bundles must never be relabeled as temperature"
        )
    diagnostics = source.get("depth_diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("enabled") is not True:
        raise ValueError("Source model must declare enabled formal depth diagnostics")
    for key, expected in DIAGNOSTIC_SEMANTICS.items():
        if str(diagnostics.get(key, "")) != expected:
            raise ValueError(
                f"Source model depth diagnostic semantics mismatch for {key}"
            )
    split_identity = source.get("formal_split_manifest_identity")
    if (
        not isinstance(split_identity, Mapping)
        or str(split_identity.get("sha256", "")).lower()
        != input_hashes["formal_split_manifest"]
    ):
        raise ValueError("Source model/formal split SHA-256 binding mismatch")

    formal_records, formal_aliases, split_counts = _formal_records(
        formal, expected_count=int(expected_view_count)
    )
    source_views = _view_index(source, expected_count=int(expected_view_count))
    if set(source_views) != {
        _normalised_stem(record["pair_id"]) for record in formal_records
    }:
        raise ValueError(
            "Source model views must cover every formal split pair_id exactly"
        )

    if str(canonical_manifest.get("schema", "")) != CANONICAL_SCHEMA:
        raise ValueError(f"Canonical manifest must use schema {CANONICAL_SCHEMA!r}")
    canonical_records, canonical_aliases = _unique_record_index(
        canonical_manifest.get("files"), label="canonical manifest"
    )
    if len(canonical_records) != int(expected_view_count):
        raise ValueError("Canonical manifest must cover every formal view exactly")
    canonical_matches = {
        id(canonical_aliases[stem])
        for stem in source_views
        if stem in canonical_aliases
    }
    if len(canonical_matches) != int(expected_view_count):
        raise ValueError(
            "Canonical manifest identities must match every formal view exactly"
        )
    canonical_palette = canonical_manifest.get("palette")
    if not isinstance(canonical_palette, Mapping):
        raise ValueError("Canonical manifest is missing palette metadata")
    fixed_lut_sha = lut_sha256(hot_iron_lut())
    if (
        str(canonical_palette.get("name", "")) != PALETTE_NAME
        or str(canonical_palette.get("sha256_uint8_rgb", "")).strip().lower()
        != fixed_lut_sha
    ):
        raise ValueError(
            "Canonical manifest does not bind the repository fixed Hot-Iron LUT"
        )
    canonical_range = canonical_manifest.get("temperature_range")
    if not isinstance(canonical_range, Mapping):
        raise ValueError("Canonical manifest is missing temperature_range")
    canonical_range_source = canonical_range.get("source")
    if not isinstance(canonical_range_source, Mapping):
        raise ValueError("Canonical manifest is missing range source provenance")
    if (
        str(canonical_range_source.get("sha256", "")).strip().lower()
        != input_hashes["range_manifest"]
    ):
        raise ValueError("Canonical manifest/formal range SHA-256 binding mismatch")

    if str(render_binding.get("protocol", "")) != SOURCE_RENDER_BINDING_PROTOCOL:
        raise ValueError("Unsupported source thermal render binding protocol")
    if str(render_binding.get("status", "")).strip().lower() != "complete":
        raise ValueError("Source thermal render binding must declare status=complete")
    render_receipt_sha = str(render_binding.get("receipt_sha256", "")).strip().lower()
    if not _is_sha256(render_receipt_sha):
        raise ValueError("Source thermal render binding is missing receipt_sha256")
    render_receipt_basis = dict(render_binding)
    render_receipt_basis.pop("receipt_sha256")
    computed_render_receipt_sha = hashlib.sha256(
        json.dumps(
            render_receipt_basis,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if computed_render_receipt_sha != render_receipt_sha:
        raise ValueError("Source thermal render binding receipt_sha256 mismatch")
    required_render_bindings = {
        "source_model_manifest_sha256": input_hashes["source_model_manifest"],
        "formal_split_manifest_sha256": input_hashes["formal_split_manifest"],
        "range_manifest_sha256": input_hashes["range_manifest"],
        "canonical_manifest_sha256": input_hashes["canonical_manifest"],
        "lut_sha256_uint8_rgb": fixed_lut_sha,
    }
    for key, expected in required_render_bindings.items():
        if str(render_binding.get(key, "")).strip().lower() != expected:
            raise ValueError(f"Source thermal render binding mismatch for {key}")
    if (
        str(render_binding.get("appearance_modality", "")).strip().lower()
        != "thermal_canonical"
    ):
        raise ValueError("Source thermal render binding appearance modality mismatch")
    source_model_artifact = source.get("model_point_cloud")
    if not isinstance(source_model_artifact, Mapping) or not _is_sha256(
        source_model_artifact.get("sha256", "")
    ):
        raise ValueError("Source model manifest is missing model_point_cloud.sha256")
    if (
        str(render_binding.get("model_point_cloud_sha256", "")).strip().lower()
        != str(source_model_artifact["sha256"]).strip().lower()
    ):
        raise ValueError("Source thermal render/model point-cloud binding mismatch")

    validate_binding_manifest_self_hash(binding)
    if str(binding.get("adapter_backend", "")).strip() != "official-dji-irp":
        raise ValueError("TSDK binding must declare adapter_backend=official-dji-irp")
    for key in (
        "binding_hash",
        "collection_hash",
        "collection_manifest_sha256",
        "collection_split_hash",
        "decode_manifest_sha256",
        "decode_protocol_hash",
        "decode_protocol_sha256",
        "formal_rule_hash",
        "scene_manifest_sha256",
        "scene_rule_hash",
        "scene_split_hash",
    ):
        if not _is_sha256(binding.get(key, "")):
            raise ValueError(f"TSDK binding has invalid {key}")
    binding_counts = binding.get("counts")
    if not isinstance(binding_counts, Mapping) or {
        split: int(binding_counts.get(split, -1)) for split in FORMAL_SPLITS
    } != {split: int(split_counts[split]) for split in FORMAL_SPLITS} or int(
        binding_counts.get("total", -1)
    ) != int(expected_view_count):
        raise ValueError("TSDK binding counts do not match the formal split")
    tsdk_protocol_sha = str(binding.get("decode_protocol_sha256", "")).strip().lower()
    if not _is_sha256(tsdk_protocol_sha):
        raise ValueError("TSDK binding has invalid decode_protocol_sha256")
    binding_records, binding_aliases = _unique_record_index(
        binding.get("files"), label="TSDK binding"
    )
    if len(binding_records) != int(expected_view_count):
        raise ValueError("TSDK binding must cover every formal split record exactly")

    if str(target_manifest.get("schema", "")) != UNDISTORTED_SCHEMA:
        raise ValueError(
            f"Temperature target manifest must use schema {UNDISTORTED_SCHEMA!r}"
        )
    if str(support_manifest.get("schema", "")) != UNDISTORTED_SCHEMA:
        raise ValueError(
            f"Valid-support manifest must use schema {UNDISTORTED_SCHEMA!r}"
        )
    if str(target_manifest.get("status", "")).strip().lower() != "complete":
        raise ValueError("Temperature target manifest must declare status=complete")
    if str(support_manifest.get("status", "")).strip().lower() != "complete":
        raise ValueError("Valid-support manifest must declare status=complete")
    target_records, target_aliases = _unique_record_index(
        target_manifest.get("files"), label="temperature target manifest"
    )
    support_records, support_aliases = _unique_record_index(
        support_manifest.get("files"), label="valid-support manifest"
    )
    if len(target_records) != int(expected_view_count) or len(support_records) != int(
        expected_view_count
    ):
        raise ValueError(
            "Temperature and valid-support manifests must cover every formal split view"
        )
    target_files = _index_npy_root(temperature_root, label="temperature target")
    support_files = _index_npy_root(valid_support_root, label="valid-support")
    if set(target_files) != set(source_views) or set(support_files) != set(
        source_views
    ):
        raise ValueError(
            "Temperature/support roots must cover every source model view exactly"
        )

    low_c, high_c, range_provenance = resolve_temperature_range(
        range_manifest=range_path
    )
    range_split_sha = (
        str(range_manifest.get("source_split_manifest_sha256", "")).strip().lower()
    )
    if range_split_sha != input_hashes["formal_split_manifest"]:
        raise ValueError("Formal range/formal split SHA-256 binding mismatch")
    if (
        float(canonical_range.get("tmin_c", np.nan)) != low_c
        or float(canonical_range.get("tmax_c", np.nan)) != high_c
    ):
        raise ValueError("Canonical manifest/formal range numeric bounds mismatch")
    palette = hot_iron_lut()
    palette_sha = lut_sha256(palette)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    output_views: list[dict[str, Any]] = []
    contract_views: list[dict[str, Any]] = []
    seen_records: dict[str, set[int]] = {
        "formal": set(),
        "binding": set(),
        "target": set(),
        "support": set(),
    }
    try:
        for ordinal, stem in enumerate(sorted(source_views)):
            source_view = source_views[stem]
            formal_record = formal_aliases.get(stem)
            binding_record = binding_aliases.get(stem)
            target_record = target_aliases.get(stem)
            support_record = support_aliases.get(stem)
            if any(
                item is None
                for item in (
                    formal_record,
                    binding_record,
                    target_record,
                    support_record,
                )
            ):
                raise KeyError(
                    f"Incomplete formal/TSDK/target/support association for {stem!r}"
                )
            assert formal_record is not None
            assert binding_record is not None
            assert target_record is not None
            assert support_record is not None
            seen_records["formal"].add(id(formal_record))
            seen_records["binding"].add(id(binding_record))
            seen_records["target"].add(id(target_record))
            seen_records["support"].add(id(support_record))
            split = _split_for_record(formal_record)
            if (
                str(source_view.get("bound_split", source_view.get("split", "")))
                .strip()
                .lower()
                != split
            ):
                raise ValueError(
                    f"Source model/formal split label mismatch for {stem!r}"
                )
            if _split_for_record(binding_record) != split:
                raise ValueError(
                    f"TSDK binding/formal split label mismatch for {stem!r}"
                )
            raw_lineage = binding_record.get("raw_thermal")
            record_backend = str(binding_record.get("adapter_backend", "")).strip()
            if not record_backend and isinstance(raw_lineage, Mapping):
                record_backend = str(raw_lineage.get("adapter_backend", "")).strip()
            if record_backend != "official-dji-irp":
                raise ValueError(f"TSDK binding file backend mismatch for {stem!r}")

            binding_temperature_sha = (
                str(
                    binding_record.get(
                        "temperature_sha256", binding_record.get("verified_sha256", "")
                    )
                )
                .strip()
                .lower()
            )
            if not _is_sha256(binding_temperature_sha):
                raise ValueError(
                    f"TSDK binding temperature SHA-256 is invalid for {stem!r}"
                )
            input_temperature = _mapping(
                target_record, "input_temperature", label="temperature target manifest"
            )
            if (
                str(input_temperature.get("sha256", "")).strip().lower()
                != binding_temperature_sha
            ):
                raise ValueError(
                    f"Undistorted target/TSDK source lineage mismatch for {stem!r}"
                )

            support_input_temperature = _mapping(
                support_record, "input_temperature", label="valid-support manifest"
            )
            if (
                str(support_input_temperature.get("sha256", "")).strip().lower()
                != binding_temperature_sha
            ):
                raise ValueError(
                    f"Valid-support/TSDK source lineage mismatch for {stem!r}"
                )

            target_metadata = _mapping(
                target_record, "output_temperature", label="temperature target manifest"
            )
            target_path, target_sha = _bound_npy(
                stem=stem,
                root_index=target_files,
                metadata=target_metadata,
                label="temperature target",
            )
            support_target_metadata = _mapping(
                support_record, "output_temperature", label="valid-support manifest"
            )
            if (
                str(support_target_metadata.get("sha256", "")).strip().lower()
                != target_sha
            ):
                raise ValueError(
                    f"Valid-support/temperature target lineage mismatch for {stem!r}"
                )
            support_metadata = _mapping(
                support_record, "valid_support", label="valid-support manifest"
            )
            support_path, support_sha = _bound_npy(
                stem=stem,
                root_index=support_files,
                metadata=support_metadata,
                label="valid-support",
            )
            target = np.load(target_path, allow_pickle=False)
            support = np.load(support_path, allow_pickle=False)
            if target.dtype != np.float32:
                raise TypeError(
                    f"TSDK target must be float32 Celsius for {stem!r}, got {target.dtype}"
                )
            if support.dtype != np.bool_:
                raise TypeError(
                    f"Valid-support must be boolean for {stem!r}, got {support.dtype}"
                )
            _validate_array_metadata(
                target_metadata,
                target,
                expected_dtype="float32",
                label=f"temperature target {stem!r}",
            )
            _validate_array_metadata(
                support_metadata,
                support,
                expected_dtype="bool",
                label=f"valid-support {stem!r}",
            )
            if target.ndim != 2 or support.ndim != 2 or target.shape != support.shape:
                raise ValueError(f"Temperature/support shape mismatch for {stem!r}")
            if not np.all(np.isfinite(target[support])):
                raise ValueError(
                    f"Valid TSDK target pixels contain NaN/Inf for {stem!r}"
                )

            source_path = _resolve_source_npz(source_manifest_path, source_view)
            with np.load(source_path, allow_pickle=False) as source_npz:
                source_keys = set(source_npz.files)
                if source_keys & FORBIDDEN_RGB_KEYS:
                    raise ValueError(
                        f"Thermal source {stem!r} contains render_rgb/target_rgb and could be misreported as RGB"
                    )
                if source_keys & SOURCE_TEMPERATURE_KEYS:
                    raise ValueError(
                        f"Source {stem!r} already contains temperature fields; refusing ambiguous overwrite"
                    )
                missing = [
                    key
                    for key in (
                        *DIAGNOSTIC_KEYS,
                        "render_thermal_canonical",
                        "target_thermal_canonical",
                    )
                    if key not in source_npz
                ]
                if missing:
                    raise KeyError(
                        f"Source {stem!r} is missing diagnostic arrays {missing}"
                    )
                expected_shape = (int(source_view["height"]), int(source_view["width"]))
                if target.shape != expected_shape:
                    raise ValueError(
                        f"TSDK target/camera shape mismatch for {stem!r}: {target.shape} vs {expected_shape}"
                    )
                compact: dict[str, np.ndarray] = {}
                for key in DIAGNOSTIC_KEYS:
                    value = np.asarray(source_npz[key])
                    if _as_hw(value, label=f"{stem} {key}").shape != expected_shape:
                        raise ValueError(
                            f"Diagnostic/camera shape mismatch for {stem!r}: {key}"
                        )
                    compact[key] = value
                if not np.issubdtype(
                    compact["top_contributor_index"].dtype, np.integer
                ):
                    raise TypeError(
                        f"top_contributor_index must be integer for {stem!r}"
                    )
                for key in (
                    "depth_expected_alpha_normalized",
                    "depth_transmittance_median",
                    "depth_max_contribution",
                    "accumulated_opacity",
                    "top_contributor_weight",
                ):
                    if not np.issubdtype(compact[key].dtype, np.floating):
                        raise TypeError(f"{key} must be floating point for {stem!r}")
                render_rgb_bytes = _as_hwc_rgb(
                    source_npz["render_thermal_canonical"],
                    expected_shape=expected_shape,
                    label=f"{stem} render_thermal_canonical",
                )
                target_rgb_bytes = _as_hwc_rgb(
                    source_npz["target_thermal_canonical"],
                    expected_shape=expected_shape,
                    label=f"{stem} target_thermal_canonical",
                )

            safe_target = np.where(np.isfinite(target), target, np.float32(low_c)).astype(
                np.float32,
                copy=False,
            )
            expected_target_rgb, _target_clipping = temperature_to_rgb(
                safe_target,
                low_c,
                high_c,
                lut=palette,
            )
            if np.any(target_rgb_bytes < -1.0e-5) or np.any(target_rgb_bytes > 255.0 + 1.0e-5) or np.any(
                np.abs(target_rgb_bytes - np.rint(target_rgb_bytes)) > 1.0e-4
            ):
                raise ValueError(
                    f"Source target_thermal_canonical is not byte-exact canonical PNG data for {stem!r}"
                )
            observed_target_rgb = np.rint(target_rgb_bytes).astype(np.int16)
            expected_target_rgb_i16 = expected_target_rgb.astype(np.int16)
            if np.any(support):
                canonical_abs_diff = np.abs(
                    observed_target_rgb[support] - expected_target_rgb_i16[support]
                )
                if np.any(canonical_abs_diff != 0):
                    raise ValueError(
                        f"Source target_thermal_canonical is not the exact fixed-LUT/formal-range "
                        f"forward colorization on valid support for {stem!r}; "
                        f"max_byte_error={int(np.max(canonical_abs_diff))}"
                    )

            render_temperature, off_lut_distance, _ = rgb_to_temperature(
                render_rgb_bytes,
                low_c,
                high_c,
                lut=palette,
                chunk_pixels=int(chunk_pixels),
            )
            compact[TEMPERATURE_KEYS["rendered"]] = np.asarray(
                render_temperature, dtype=np.float32
            )
            # Direct copy from the float32 TSDK-referenced map: no PNG/LUT round-trip.
            compact[TEMPERATURE_KEYS["target"]] = np.asarray(target, dtype=np.float32)
            compact[TEMPERATURE_KEYS["valid_mask"]] = np.asarray(
                support, dtype=np.bool_
            )

            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
            relative = Path("views") / f"{ordinal:06d}_{safe_stem}.npz"
            output_path = temporary / relative
            _save_npz_deterministic(output_path, compact)
            output_sha = _sha256(output_path)
            view_entry = dict(source_view)
            view_entry.update(
                {
                    "npz_file": relative.as_posix(),
                    "npz_size_bytes": int(output_path.stat().st_size),
                    "npz_sha256": output_sha,
                    "source_npz_sha256": str(source_view["npz_sha256"]).lower(),
                    "temperature_target_sha256": target_sha,
                    "valid_support_sha256": support_sha,
                }
            )
            output_views.append(view_entry)
            contract_views.append(
                {
                    "image_name": str(source_view["image_name"]),
                    "pair_id": str(formal_record.get("pair_id", stem)),
                    "split": split,
                    "npz_file": relative.as_posix(),
                    "npz_size_bytes": int(output_path.stat().st_size),
                    "npz_sha256": output_sha,
                    "source_npz_sha256": str(source_view["npz_sha256"]).lower(),
                    "temperature_target_sha256": target_sha,
                    "valid_support_sha256": support_sha,
                    "off_lut": _off_lut_summary(off_lut_distance, support),
                }
            )

        expected_unique = int(expected_view_count)
        for label, values in seen_records.items():
            if len(values) != expected_unique:
                raise ValueError(
                    f"{label} association did not cover every view exactly once: {len(values)}/{expected_unique}"
                )

        producer_identity = _producer_identity()
        derived_manifest = dict(source)
        derived_manifest["views"] = output_views
        derived_manifest["temperature_responsibility_derivation"] = {
            "protocol": PROTOCOL,
            "semantics": SEMANTICS,
            "producer_identity": producer_identity,
            "source_model_manifest_sha256": input_hashes["source_model_manifest"],
            "source_render_binding_manifest_sha256": input_hashes[
                "source_render_binding_manifest"
            ],
            "formal_split_manifest_sha256": input_hashes["formal_split_manifest"],
            "tsdk_binding_manifest_sha256": input_hashes["tsdk_binding_manifest"],
            "tsdk_protocol_sha256": tsdk_protocol_sha,
            "temperature_manifest_sha256": input_hashes["temperature_manifest"],
            "valid_support_manifest_sha256": input_hashes["valid_support_manifest"],
            "range_manifest_sha256": input_hashes["range_manifest"],
            "canonical_manifest_sha256": input_hashes["canonical_manifest"],
            "lut_sha256_uint8_rgb": palette_sha,
            "keys": dict(TEMPERATURE_KEYS),
            "target_source": "direct float32 TSDK-referenced undistorted NPY; never PNG inversion",
            "render_inverse": "nearest repository-owned fixed Hot-Iron LUT",
            "source_target_forward_validation": (
                "target_thermal_canonical equals direct TSDK float32 target forward-colorized with the "
                "same formal range and fixed Hot-Iron LUT on every valid-support pixel"
            ),
        }
        model_manifest_path = temporary / OUTPUT_MODEL_MANIFEST
        _write_json(model_manifest_path, derived_manifest)
        model_manifest_sha = _sha256(model_manifest_path)

        all_off_lut = [item["off_lut"] for item in contract_views]
        contract = {
            "protocol": PROTOCOL,
            "producer_identity": producer_identity,
            "scene_name": scene_name,
            "semantics": SEMANTICS,
            "units": "Celsius",
            "dtype": "float32",
            "keys": dict(TEMPERATURE_KEYS),
            "model_manifest_sha256": model_manifest_sha,
            "source_model_manifest_sha256": input_hashes["source_model_manifest"],
            "source_render_binding_manifest_sha256": input_hashes[
                "source_render_binding_manifest"
            ],
            "formal_split_manifest_sha256": input_hashes["formal_split_manifest"],
            "tsdk_binding_manifest_sha256": input_hashes["tsdk_binding_manifest"],
            "tsdk_protocol_sha256": tsdk_protocol_sha,
            "temperature_manifest_sha256": input_hashes["temperature_manifest"],
            "valid_support_manifest_sha256": input_hashes["valid_support_manifest"],
            "range_manifest_sha256": input_hashes["range_manifest"],
            "canonical_manifest_sha256": input_hashes["canonical_manifest"],
            "lut_sha256_uint8_rgb": palette_sha,
            "formal_range": {
                "tmin_c": float(low_c),
                "tmax_c": float(high_c),
                "manifest_sha256": input_hashes["range_manifest"],
                "provenance": range_provenance,
            },
            "palette": {
                "name": PALETTE_NAME,
                "entries": PALETTE_SIZE,
                "sha256_uint8_rgb": palette_sha,
                "inverse": "nearest RGB-byte Euclidean projection; off-LUT render accepted and measured",
            },
            "coverage": {
                "view_count": int(expected_view_count),
                "split_counts": split_counts,
                "split_labels": sorted(FORMAL_SPLITS),
                "all_formal_views_exactly_once": True,
            },
            "target_provenance": {
                "domain": "float32 Celsius",
                "source": "TSDK-referenced undistorted NPY",
                "png_or_palette_inverse_used_for_target": False,
                "source_training_target_forward_colorization_exact_on_valid_support": True,
            },
            "render_provenance": {
                "source_key": "render_thermal_canonical",
                "source_units": "normalized RGB",
                "inverse": "fixed Hot-Iron nearest LUT",
                "off_lut_distance_units": "RGB-byte Euclidean",
            },
            "off_lut_summary": {
                "view_count": len(all_off_lut),
                "mean_of_view_means": float(
                    np.mean([item["mean"] for item in all_off_lut])
                ),
                "max_over_views": float(max(item["maximum"] for item in all_off_lut)),
            },
            "views": contract_views,
        }
        contract_path = temporary / OUTPUT_CONTRACT_MANIFEST
        _write_json(contract_path, contract)

        # Detect concurrent/tampering changes before atomic publication.
        for key, path in input_paths.items():
            if _sha256(path) != input_hashes[key]:
                raise RuntimeError(f"Input manifest changed during build: {key}")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    result_model = destination / OUTPUT_MODEL_MANIFEST
    result_contract = destination / OUTPUT_CONTRACT_MANIFEST
    return {
        "status": "complete",
        "scene_name": scene_name,
        "view_count": int(expected_view_count),
        "model_manifest": str(result_model),
        "model_manifest_sha256": _sha256(result_model),
        "temperature_responsibility_manifest": str(result_contract),
        "temperature_responsibility_manifest_sha256": _sha256(result_contract),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-manifest", required=True, type=Path)
    parser.add_argument("--source-render-binding-manifest", required=True, type=Path)
    parser.add_argument("--formal-split-manifest", required=True, type=Path)
    parser.add_argument("--tsdk-binding-manifest", required=True, type=Path)
    parser.add_argument("--temperature-root", required=True, type=Path)
    parser.add_argument("--temperature-manifest", required=True, type=Path)
    parser.add_argument("--valid-support-root", required=True, type=Path)
    parser.add_argument("--valid-support-manifest", required=True, type=Path)
    parser.add_argument("--range-manifest", required=True, type=Path)
    parser.add_argument("--canonical-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-view-count", type=int, default=559)
    parser.add_argument("--chunk-pixels", type=int, default=32768)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_temperature_responsibility_bundle(
        source_model_manifest_path=args.source_model_manifest,
        source_render_binding_manifest_path=args.source_render_binding_manifest,
        formal_split_manifest_path=args.formal_split_manifest,
        tsdk_binding_manifest_path=args.tsdk_binding_manifest,
        temperature_root=args.temperature_root,
        temperature_manifest_path=args.temperature_manifest,
        valid_support_root=args.valid_support_root,
        valid_support_manifest_path=args.valid_support_manifest,
        range_manifest_path=args.range_manifest,
        canonical_manifest_path=args.canonical_manifest,
        output_root=args.output_root,
        expected_view_count=args.expected_view_count,
        chunk_pixels=args.chunk_pixels,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
