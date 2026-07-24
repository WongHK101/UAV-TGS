#!/usr/bin/env python3
"""Export palette-neutral thermal sidecars and fixed-range DJI TSDK displays.

This tool does not modify a trained model or its source renders.  It projects
canonical RGB renders onto the repository-owned 256-entry canonical LUT,
stores the resulting apparent-temperature index/temperature sidecars, and
renders any requested DJI TSDK pseudo-colour palette with one fixed scene
temperature range.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

try:
    from palette_lut import (
        hot_iron_lut,
        indices_to_temperature,
        lut_sha256,
        nearest_lut_indices,
        palette_metadata,
        resolve_temperature_range,
    )
except ImportError:  # pragma: no cover - module execution path
    from .palette_lut import (
        hot_iron_lut,
        indices_to_temperature,
        lut_sha256,
        nearest_lut_indices,
        palette_metadata,
        resolve_temperature_range,
    )


SCHEMA = "uav-tgs-tsdk-fixed-range-display-v1"
PALETTE_NAMES = (
    "white_hot",
    "fulgurite",
    "iron_red",
    "hot_iron",
    "medical",
    "arctic",
    "rainbow1",
    "rainbow2",
    "tint",
    "black_hot",
)
PALETTE_INDEX = {name: index for index, name in enumerate(PALETTE_NAMES)}
PALETTE_DEPTH = 256
PALETTE_COUNT = len(PALETTE_NAMES)
_DLL_DIRECTORY_HANDLES: list[Any] = []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_png(path: Path, rgb: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.png")
    try:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            temporary,
            format="PNG",
            compress_level=6,
            optimize=False,
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


class _DirpColorBar(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("manual_enable", ctypes.c_bool),
        ("high", ctypes.c_float),
        ("low", ctypes.c_float),
    ]


class _DirpEnhancement(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("brightness", ctypes.c_int32)]


class _DirpPseudoColorLut(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("red", (ctypes.c_uint8 * PALETTE_DEPTH) * PALETTE_COUNT),
        ("green", (ctypes.c_uint8 * PALETTE_DEPTH) * PALETTE_COUNT),
        ("blue", (ctypes.c_uint8 * PALETTE_DEPTH) * PALETTE_COUNT),
    ]


@dataclass(frozen=True)
class TSDKPaletteBundle:
    """DJI palette data and the original R-JPEG display metadata."""

    luts_rgb: np.ndarray
    original_palette_index: int
    original_color_bar_manual: bool
    original_color_bar_low_c: float
    original_color_bar_high_c: float
    original_brightness: int
    reference_rjpeg: Path
    reference_rjpeg_sha256: str
    library_path: Path
    library_sha256: str

    def __post_init__(self) -> None:
        luts = np.asarray(self.luts_rgb)
        if luts.shape != (PALETTE_COUNT, PALETTE_DEPTH, 3):
            raise ValueError(
                "TSDK LUT array must have shape "
                f"{(PALETTE_COUNT, PALETTE_DEPTH, 3)}, got {luts.shape}"
            )
        if luts.dtype != np.uint8:
            raise ValueError(f"TSDK LUT array must be uint8, got {luts.dtype}")
        if not 0 <= int(self.original_palette_index) < PALETTE_COUNT:
            raise ValueError(
                f"Invalid original TSDK palette index: {self.original_palette_index}"
            )
        for index, name in enumerate(PALETTE_NAMES):
            unique = np.unique(luts[index], axis=0).shape[0]
            if unique != PALETTE_DEPTH:
                raise ValueError(
                    f"TSDK palette {name!r} is not one-to-one: "
                    f"{unique}/{PALETTE_DEPTH} unique RGB entries"
                )

    def palette(self, name: str) -> np.ndarray:
        try:
            index = PALETTE_INDEX[name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported TSDK palette {name!r}; choose from {PALETTE_NAMES}"
            ) from exc
        return self.luts_rgb[index]

    def metadata(self) -> Dict[str, Any]:
        return {
            "source": "DJI Thermal SDK dirp_get_pseudo_color_lut",
            "library_path": str(self.library_path),
            "library_sha256": self.library_sha256,
            "reference_rjpeg": str(self.reference_rjpeg),
            "reference_rjpeg_sha256": self.reference_rjpeg_sha256,
            "original_palette_index": int(self.original_palette_index),
            "original_palette_name": PALETTE_NAMES[self.original_palette_index],
            "original_color_bar": {
                "manual_enable": bool(self.original_color_bar_manual),
                "low_c": float(self.original_color_bar_low_c),
                "high_c": float(self.original_color_bar_high_c),
                "note": (
                    "In automatic mode these stored values are not the "
                    "per-frame adaptive display range."
                ),
            },
            "original_brightness": int(self.original_brightness),
            "all_luts_sha256_uint8_rgb": _array_sha256(self.luts_rgb),
            "palettes": {
                name: {
                    "index": index,
                    "entries": PALETTE_DEPTH,
                    "unique_rgb_entries": int(
                        np.unique(self.luts_rgb[index], axis=0).shape[0]
                    ),
                    "sha256_uint8_rgb": _array_sha256(self.luts_rgb[index]),
                }
                for index, name in enumerate(PALETTE_NAMES)
            },
        }


def _resolve_tsdk_library(tsdk_root: Path) -> Path:
    root = Path(tsdk_root).expanduser().resolve()
    if root.is_file():
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"TSDK root does not exist: {root}")
    if os.name == "nt":
        candidates = (
            root / "utility/bin/windows/release_x64/libdirp.dll",
            root / "tsdk-core/lib/windows/release_x64/libdirp.dll",
            root / "libdirp.dll",
        )
    else:
        candidates = (
            root / "utility/bin/linux/release_x64/libdirp.so",
            root / "tsdk-core/lib/linux/release_x64/libdirp.so",
            root / "libdirp.so",
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate libdirp under {root}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def _load_dirp_library(library_path: Path) -> ctypes.CDLL:
    library = Path(library_path).resolve()
    dependency_dir = library.parent
    if os.name == "nt":
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(dependency_dir)))
    else:  # pragma: no cover - exercised on the Linux experiment hosts
        for pattern in (
            "libexif.so*",
            "libMicroIA*.so",
            "libMicroJPEG*.so",
            "libMicroTA*.so",
            "libv_*.so",
        ):
            for dependency in sorted(dependency_dir.glob(pattern)):
                try:
                    ctypes.CDLL(str(dependency), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
    return ctypes.CDLL(str(library))


def load_tsdk_palette_bundle(
    tsdk_root: Path,
    reference_rjpeg: Path,
) -> TSDKPaletteBundle:
    """Read exact DJI pseudo-colour LUTs through the official DIRP API."""
    library_path = _resolve_tsdk_library(tsdk_root)
    source = Path(reference_rjpeg).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference R-JPEG does not exist: {source}")
    payload = source.read_bytes()
    if not payload:
        raise ValueError(f"Reference R-JPEG is empty: {source}")
    data = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    handle = ctypes.c_void_p()
    library = _load_dirp_library(library_path)
    library.dirp_create_from_rjpeg.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.dirp_create_from_rjpeg.restype = ctypes.c_int32
    library.dirp_destroy.argtypes = [ctypes.c_void_p]
    library.dirp_destroy.restype = ctypes.c_int32
    library.dirp_get_pseudo_color.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.dirp_get_pseudo_color.restype = ctypes.c_int32
    library.dirp_get_color_bar.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DirpColorBar),
    ]
    library.dirp_get_color_bar.restype = ctypes.c_int32
    library.dirp_get_enhancement_params.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DirpEnhancement),
    ]
    library.dirp_get_enhancement_params.restype = ctypes.c_int32
    library.dirp_get_pseudo_color_lut.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_DirpPseudoColorLut),
    ]
    library.dirp_get_pseudo_color_lut.restype = ctypes.c_int32

    create_code = library.dirp_create_from_rjpeg(data, len(payload), ctypes.byref(handle))
    if create_code != 0 or not handle.value:
        raise RuntimeError(
            f"dirp_create_from_rjpeg failed with code {create_code}: {source}"
        )
    try:
        original_palette = ctypes.c_int32(-1)
        color_bar = _DirpColorBar()
        enhancement = _DirpEnhancement()
        raw_luts = _DirpPseudoColorLut()
        calls = (
            (
                "dirp_get_pseudo_color",
                library.dirp_get_pseudo_color(
                    handle, ctypes.byref(original_palette)
                ),
            ),
            (
                "dirp_get_color_bar",
                library.dirp_get_color_bar(handle, ctypes.byref(color_bar)),
            ),
            (
                "dirp_get_enhancement_params",
                library.dirp_get_enhancement_params(
                    handle, ctypes.byref(enhancement)
                ),
            ),
            (
                "dirp_get_pseudo_color_lut",
                library.dirp_get_pseudo_color_lut(
                    handle, ctypes.byref(raw_luts)
                ),
            ),
        )
        for name, code in calls:
            if code != 0:
                raise RuntimeError(f"{name} failed with DIRP code {code}")
        luts = np.empty((PALETTE_COUNT, PALETTE_DEPTH, 3), dtype=np.uint8)
        for palette_index in range(PALETTE_COUNT):
            for value_index in range(PALETTE_DEPTH):
                luts[palette_index, value_index] = (
                    raw_luts.red[palette_index][value_index],
                    raw_luts.green[palette_index][value_index],
                    raw_luts.blue[palette_index][value_index],
                )
    finally:
        destroy_code = library.dirp_destroy(handle)
        if destroy_code != 0:
            raise RuntimeError(f"dirp_destroy failed with DIRP code {destroy_code}")

    return TSDKPaletteBundle(
        luts_rgb=luts,
        original_palette_index=int(original_palette.value),
        original_color_bar_manual=bool(color_bar.manual_enable),
        original_color_bar_low_c=float(color_bar.low),
        original_color_bar_high_c=float(color_bar.high),
        original_brightness=int(enhancement.brightness),
        reference_rjpeg=source,
        reference_rjpeg_sha256=_sha256(source),
        library_path=library_path,
        library_sha256=_sha256(library_path),
    )


def discover_render_images(
    input_root: Path,
    *,
    pattern: str = "*.png",
    recursive: bool = False,
) -> Sequence[Path]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Render root is not a directory: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    images = sorted(
        (path for path in iterator if path.is_file()),
        key=lambda path: path.as_posix(),
    )
    if not images:
        raise FileNotFoundError(
            f"No render images matching {pattern!r} under {root}"
        )
    return images


def _validate_palette_names(names: Sequence[str]) -> tuple[str, ...]:
    expanded = PALETTE_NAMES if tuple(names) == ("all",) else tuple(names)
    if not expanded:
        raise ValueError("At least one TSDK palette is required")
    unknown = sorted(set(expanded) - set(PALETTE_NAMES))
    if unknown:
        raise ValueError(
            f"Unsupported TSDK palettes {unknown}; choose from {PALETTE_NAMES}"
        )
    if len(set(expanded)) != len(expanded):
        raise ValueError(f"Duplicate TSDK palettes are not allowed: {expanded}")
    return expanded


def export_render_file(
    source_path: Path,
    *,
    relative_path: Path,
    output_root: Path,
    tmin_c: float,
    tmax_c: float,
    bundle: TSDKPaletteBundle,
    palette_names: Sequence[str],
    save_off_lut_map: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Export scalar sidecars and palette PNGs for one canonical render."""
    source = Path(source_path).resolve()
    destination_root = Path(output_root).resolve()
    palettes = _validate_palette_names(palette_names)
    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    indices, off_lut = nearest_lut_indices(rgb)
    temperature_c = indices_to_temperature(indices, tmin_c, tmax_c)
    canonical_projection = hot_iron_lut()[indices]
    reconstruction_abs = np.abs(
        canonical_projection.astype(np.int16) - rgb.astype(np.int16)
    )
    relative_npy = Path(relative_path).with_suffix(".npy")
    index_path = destination_root / "temperature_index" / relative_npy
    temperature_path = (
        destination_root / "apparent_temperature_c" / relative_npy
    )
    off_lut_path = destination_root / "off_lut_distance_rgb" / relative_npy
    outputs = [index_path, temperature_path]
    if save_off_lut_map:
        outputs.append(off_lut_path)
    palette_paths = {
        name: destination_root / "palettes" / name / relative_path
        for name in palettes
    }
    outputs.extend(palette_paths.values())
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite palette-neutral outputs: "
            + ", ".join(str(path) for path in existing[:5])
        )

    _atomic_save_npy(index_path, indices.astype(np.uint8, copy=False))
    _atomic_save_npy(
        temperature_path,
        temperature_c.astype(np.float32, copy=False),
    )
    if save_off_lut_map:
        _atomic_save_npy(
            off_lut_path,
            off_lut.astype(np.float32, copy=False),
        )
    palette_records: Dict[str, Any] = {}
    for name, destination in palette_paths.items():
        remapped = bundle.palette(name)[indices]
        _atomic_save_png(destination, remapped)
        with Image.open(destination) as saved_image:
            saved = np.asarray(saved_image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(saved, remapped):
            raise RuntimeError(f"Lossless PNG verification failed: {destination}")
        palette_records[name] = {
            "output_path": str(destination),
            "output_sha256": _sha256(destination),
            "palette_sha256_uint8_rgb": _array_sha256(bundle.palette(name)),
            "index_roundtrip_exact": True,
        }

    return {
        "relative_path": Path(relative_path).as_posix(),
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
        "pixels": int(indices.size),
        "temperature_index": {
            "path": str(index_path),
            "sha256": _sha256(index_path),
            "dtype": "uint8",
            "minimum": int(indices.min()),
            "maximum": int(indices.max()),
        },
        "apparent_temperature_c": {
            "path": str(temperature_path),
            "sha256": _sha256(temperature_path),
            "dtype": "float32",
            "minimum": float(temperature_c.min()),
            "maximum": float(temperature_c.max()),
            "qualification": (
                "palette-inverted TSDK-referenced apparent temperature; "
                "not absolute thermometry"
            ),
        },
        "off_lut_distance_rgb": {
            "map_path": str(off_lut_path) if save_off_lut_map else None,
            "map_sha256": _sha256(off_lut_path) if save_off_lut_map else None,
            "mean": float(np.mean(off_lut)),
            "p95": float(np.percentile(off_lut, 95)),
            "maximum": float(np.max(off_lut)),
        },
        "canonical_projection_reconstruction": {
            "mean_abs_rgb": float(np.mean(reconstruction_abs)),
            "maximum_abs_rgb": int(np.max(reconstruction_abs)),
            "exact_source_pixels": int(np.count_nonzero(off_lut == 0.0)),
            "exact_source_fraction": float(np.mean(off_lut == 0.0)),
        },
        "palettes": palette_records,
    }


def export_render_tree(
    input_root: Path,
    output_root: Path,
    *,
    tmin_c: float,
    tmax_c: float,
    bundle: TSDKPaletteBundle,
    palette_names: Sequence[str] = ("hot_iron",),
    range_provenance: Optional[Mapping[str, Any]] = None,
    pattern: str = "*.png",
    recursive: bool = False,
    save_off_lut_map: bool = False,
    manifest_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    source_root = Path(input_root).resolve()
    destination_root = Path(output_root).resolve()
    if source_root == destination_root or _is_within(destination_root, source_root):
        raise ValueError("Palette output must be outside the render source tree")
    images = discover_render_images(
        source_root,
        pattern=pattern,
        recursive=recursive,
    )
    palettes = _validate_palette_names(palette_names)
    records = [
        export_render_file(
            source,
            relative_path=source.relative_to(source_root),
            output_root=destination_root,
            tmin_c=tmin_c,
            tmax_c=tmax_c,
            bundle=bundle,
            palette_names=palettes,
            save_off_lut_map=save_off_lut_map,
            overwrite=overwrite,
        )
        for source in images
    ]
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source_root),
        "output_root": str(destination_root),
        "input_pattern": pattern,
        "recursive": bool(recursive),
        "fixed_scene_temperature_range": {
            "tmin_c": float(tmin_c),
            "tmax_c": float(tmax_c),
            "bin_width_c": float((tmax_c - tmin_c) / 255.0),
            "source": dict(range_provenance or {"source": "function-argument"}),
        },
        "source_palette": palette_metadata(),
        "source_palette_lut_sha256": lut_sha256(),
        "tsdk": bundle.metadata(),
        "requested_palettes": list(palettes),
        "conversion": {
            "input": "canonical RGB render",
            "scalar_recovery": (
                "nearest projection onto the repository canonical 256-entry LUT"
            ),
            "display_mapping": (
                "recovered uint8 index mapped through the selected exact DJI "
                "TSDK 256-entry RGB LUT"
            ),
            "range_mode": "fixed_scene",
            "model_modified": False,
            "formal_metrics_modified": False,
            "lossless_boundary": (
                "palette PNG/index roundtrip is exact after scalar projection; "
                "off-LUT source RGB projection is deterministic but not lossless"
            ),
        },
        "files": records,
        "summary": {
            "file_count": len(records),
            "pixels": int(sum(record["pixels"] for record in records)),
            "maximum_off_lut_distance_rgb": float(
                max(record["off_lut_distance_rgb"]["maximum"] for record in records)
            ),
            "maximum_palette_roundtrip_failures": 0,
        },
    }
    target = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else destination_root / "palette_display_manifest.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    payload["manifest_path"] = str(target)
    payload["manifest_sha256"] = _sha256(target)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--tmin-c", type=float)
    parser.add_argument("--tmax-c", type=float)
    parser.add_argument("--range-manifest", type=Path)
    parser.add_argument(
        "--tsdk-root",
        type=Path,
        help="DJI TSDK root; defaults to DJI_TSDK_ROOT.",
    )
    parser.add_argument("--reference-rjpeg", required=True, type=Path)
    parser.add_argument(
        "--palette",
        action="append",
        choices=(*PALETTE_NAMES, "all"),
        default=None,
        help="Repeat to export multiple palettes; default: hot_iron.",
    )
    parser.add_argument("--save-off-lut-map", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    tsdk_root = args.tsdk_root
    if tsdk_root is None:
        environment_root = os.environ.get("DJI_TSDK_ROOT")
        if not environment_root:
            raise ValueError(
                "Provide --tsdk-root or set the DJI_TSDK_ROOT environment variable"
            )
        tsdk_root = Path(environment_root)
    tmin_c, tmax_c, range_provenance = resolve_temperature_range(
        tmin_c=args.tmin_c,
        tmax_c=args.tmax_c,
        range_manifest=args.range_manifest,
    )
    bundle = load_tsdk_palette_bundle(tsdk_root, args.reference_rjpeg)
    result = export_render_tree(
        args.input_root,
        args.output_root,
        tmin_c=tmin_c,
        tmax_c=tmax_c,
        bundle=bundle,
        palette_names=tuple(args.palette or ("hot_iron",)),
        range_provenance=range_provenance,
        pattern=args.pattern,
        recursive=args.recursive,
        save_off_lut_map=args.save_off_lut_map,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "file_count": result["summary"]["file_count"],
                "manifest": result["manifest_path"],
                "manifest_sha256": result["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
