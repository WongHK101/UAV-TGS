# -*- coding: utf-8 -*-
"""
run_uavfgs_pipeline.py

A resumable, one-command CLI runner for the end-to-end RGB-T Gaussian
reconstruction and fusion pipeline used in this work:

1) CFR crop/align (cfr.py)
2) Crop+EXIF evaluation (eval_crop_metrics.py) and (optionally) auto-pick best candidate
3) Prepare COLMAP input/ directory from chosen candidate
4) COLMAP pipeline with GPS priors + alignment (convert_uavfgs.py)
5) Stage-1 3DGS train (RGB), render, metrics
6) Undistort thermal images using aligned sparse model (colmap image_undistorter)
7) Normalize thermal_UD/sparse layout (move files into sparse/0 if needed)
8) Stage-2 3DGS train (Thermal), render, metrics
9) Blend RGB+Thermal models (blend_model_strict_endpoints.py)
10) Evaluate sweep (eval_blend_sweep.py, with optional auto_render)

Key features vs the previous version:
- **Resumable by default**: If a step's expected outputs already exist, it will be skipped on rerun.
- Writes per-step markers under: <data_root>/_pipeline_state/*.json
- More robust COLMAP executable handling on Windows:
  - resolves "colmap" via PATH
  - supports colmap.bat / colmap.cmd (wraps with cmd.exe /c)
  - optionally tries PowerShell Get-Command to resolve "colmap" when PATH lookup fails

Assumptions:
- Put this file in the repository root,
  next to: train.py, render.py, metrics.py, and the helper scripts:
  cfr.py / eval_crop_metrics.py / convert_uavfgs.py / blend_model_strict_endpoints.py / eval_blend_sweep.py
- Run this script using the same Python environment you use for 3DGS.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import csv
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from efficiency_probe import artifact_record, atomic_write_json, read_json_if_present


ORIG_3DGS_OPT_DEFAULTS: Dict[str, float] = {
    "position_lr_init": 0.00016,
    "position_lr_final": 0.0000016,
    "position_lr_delay_mult": 0.01,
    "position_lr_max_steps": 30000,
    "feature_lr": 0.0025,
    "opacity_lr": 0.025,
    "scaling_lr": 0.005,
    "rotation_lr": 0.001,
    "lambda_dssim": 0.2,
    "densification_interval": 100,
    "opacity_reset_interval": 3000,
    "densify_from_iter": 500,
    "densify_until_iter": 15000,
    "densify_grad_threshold": 0.0002,
}

# ----------------------------
# Small utilities
# ----------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def exists_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())

def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return total
    if p.is_file():
        try:
            return p.stat().st_size
        except Exception:
            return 0
    for root, _, files in os.walk(p):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except Exception:
                continue
    return total

def _count_images_recursive(p: Path) -> int:
    if not p.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    count = 0
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            count += 1
    return count

def _ply_vertex_count(p: Path) -> Optional[int]:
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("element vertex"):
                    parts = line.strip().split()
                    return int(parts[-1])
                if line.strip() == "end_header":
                    break
    except Exception:
        return None
    return None


def _json_number(path: Path, key: str) -> Optional[float]:
    if (not path.exists()) or (not path.is_file()):
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        v = obj.get(key, None)
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float)):
            fv = float(v)
            return fv if math.isfinite(fv) else None
    except Exception:
        return None
    return None


def _find_prune_before_thermal_stats(log_path: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Parse one-line prune log, e.g.:
    [INFO] SparseSupport prune_before_thermal: before=123 after=100 removed=23 keep_ratio=0.8130
    """
    if (not log_path.exists()) or (not log_path.is_file()):
        return None, None, None, None
    pat = re.compile(
        r"SparseSupport prune_before_thermal:\s*before=(\d+)\s+after=(\d+)\s+removed=(\d+)\s+keep_ratio=([0-9]*\.?[0-9]+)"
    )
    last = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    last = m
    except Exception:
        return None, None, None, None
    if last is None:
        return None, None, None, None
    try:
        before = float(int(last.group(1)))
        after = float(int(last.group(2)))
        removed = float(int(last.group(3)))
        keep = float(last.group(4))
        return before, after, removed, keep
    except Exception:
        return None, None, None, None


def _validate_finite_float(
    ap: argparse.ArgumentParser,
    name: str,
    v: Optional[float],
    *,
    min_value: Optional[float] = None,
    strict_positive: bool = False,
) -> None:
    """Argparse-time validation for optional floats.

    Notes:
      - Only called when the corresponding feature is enabled, so defaults remain unchanged.
      - When strict_positive=True, requires v > 0.
      - When min_value is set, requires v >= min_value.
    """
    if v is None:
        return
    try:
        fv = float(v)
    except Exception:
        ap.error(f"{name} must be a float, got {v!r}")
        return
    if not math.isfinite(fv):
        ap.error(f"{name} must be a finite float, got {v!r}")
    if strict_positive and fv <= 0.0:
        ap.error(f"{name} must be > 0, got {v!r}")
    if (min_value is not None) and fv < float(min_value):
        ap.error(f"{name} must be >= {min_value}, got {v!r}")


def _str2bool(v: str) -> bool:
    """Argparse-friendly bool parser."""
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


def _cli_option_was_supplied(argv: Sequence[str], option: str) -> bool:
    """Return whether an argparse option was explicitly supplied."""
    return any(token == option or token.startswith(option + "=") for token in argv)


def _resolve_artifact_save_semantics(
    thermal_recipe: str,
    requested_semantics: Optional[str],
) -> str:
    """Resolve Stage-2 endpoint-save ordering without changing legacy defaults."""
    recipe = str(thermal_recipe)
    requested = None if requested_semantics is None else str(requested_semantics)
    if requested not in (None, "legacy", "aligned"):
        raise ValueError(f"unsupported artifact save semantics: {requested!r}")
    if recipe in ("aaai_strict", "geometry_frozen_opacity_adaptive"):
        if requested == "legacy":
            raise ValueError(
                f"--thermal_recipe {recipe} requires "
                "--artifact_save_semantics aligned"
            )
        return "aligned"
    return "legacy" if requested is None else requested


def _apply_thermal_recipe_defaults(
    args: argparse.Namespace,
    argv: Sequence[str],
) -> List[str]:
    """Apply recipe defaults and reject controls incompatible with the recipe.

    Parser defaults remain the established legacy values. Formal recipes fill
    unspecified controls while retaining their explicit optimizer/freeze/clamp
    contracts. Legacy commands remain byte-for-byte unchanged.
    """
    recipe = str(args.thermal_recipe)
    if recipe not in ("aaai_strict", "geometry_frozen_opacity_adaptive"):
        return []

    applied: List[str] = []
    defaults = (
        (
            ("thermal_max_sh_degree", "--thermal_max_sh_degree", 1),
            ("thermal_optimizer_state", "--thermal_optimizer_state", "fresh"),
            ("thermal_freeze_mode", "--thermal_freeze_mode", "strict"),
            ("thermal_scale_clamp", "--thermal_scale_clamp", "off"),
        )
        if recipe == "aaai_strict"
        else (
            ("thermal_max_sh_degree", "--thermal_max_sh_degree", 3),
            ("thermal_optimizer_state", "--thermal_optimizer_state", "fresh"),
            (
                "thermal_freeze_mode", "--thermal_freeze_mode",
                "geometry_frozen_opacity_adaptive",
            ),
            ("thermal_scale_clamp", "--thermal_scale_clamp", "off"),
        )
    )
    for attribute, option, value in defaults:
        supplied = _cli_option_was_supplied(argv, option)
        allow_strict_sh_override = recipe == "aaai_strict" and attribute == "thermal_max_sh_degree"
        if supplied and not allow_strict_sh_override and getattr(args, attribute) != value:
            raise ValueError(
                f"--thermal_recipe {recipe} requires {option} {value}"
            )
        if not supplied:
            setattr(args, attribute, value)
            applied.append(attribute)
    return applied


def _resolve_thermal_checkpoint_iterations(
    *,
    rgb_iter: int,
    t_iter: int,
    offsets: Sequence[int],
    use_offsets: bool,
) -> List[int]:
    """Resolve stage-2-relative offsets to absolute train.py iterations."""
    if not use_offsets:
        return [int(t_iter)]
    if int(t_iter) <= int(rgb_iter):
        raise ValueError("thermal checkpoint offsets require --t_iter > --rgb_iter")

    normalized = [int(offset) for offset in offsets]
    if not normalized:
        raise ValueError("--thermal_checkpoint_offsets must not be empty")
    if any(offset <= 0 for offset in normalized):
        raise ValueError("--thermal_checkpoint_offsets values must be > 0")
    if normalized != sorted(set(normalized)):
        raise ValueError("--thermal_checkpoint_offsets must be unique and strictly increasing")

    absolute = [
        int(rgb_iter) + offset
        for offset in normalized
        if int(rgb_iter) + offset <= int(t_iter)
    ]
    if int(t_iter) not in absolute:
        absolute.append(int(t_iter))
    return sorted(set(absolute))


def _thermal_learning_rate_tokens(args: argparse.Namespace) -> Dict[str, str]:
    """Return explicit stage-2 LR tokens for the resolved freeze protocol."""
    mode = str(args.thermal_freeze_mode)
    if mode == "continuous_unfrozen":
        return {
            "position_lr_init": str(args.t_unfrozen_position_lr),
            "position_lr_final": str(args.t_unfrozen_position_lr),
            "scaling_lr": str(args.t_unfrozen_scaling_lr),
            "rotation_lr": str(args.t_unfrozen_rotation_lr),
            "opacity_lr": str(args.t_opacity_lr),
            "feature_lr": str(args.t_feature_lr),
        }
    if mode == "strict":
        opacity_lr = "0"
    elif mode == "geometry_frozen_opacity_adaptive":
        opacity_lr = str(args.t_opacity_lr)
    elif mode == "legacy":
        opacity_lr = str(args.t_opacity_lr)
    else:
        raise ValueError(f"unsupported thermal freeze mode: {mode}")
    return {
        "position_lr_init": "0",
        "position_lr_final": "0",
        "scaling_lr": "0",
        "rotation_lr": "0",
        "opacity_lr": opacity_lr,
        "feature_lr": str(args.t_feature_lr),
    }


def _thermal_optimizer_contract(args: argparse.Namespace) -> Dict[str, object]:
    mode = str(args.thermal_freeze_mode)
    cap = args.thermal_max_sh_degree
    appearance_groups = ["f_dc"] if cap == 0 else ["f_dc", "f_rest"]
    if mode == "strict":
        return {
            "optimizer_groups": appearance_groups,
            "trainable_fields": appearance_groups,
            "frozen_fields": ["xyz", "scaling", "rotation", "opacity", "exposure"],
        }
    if mode == "geometry_frozen_opacity_adaptive":
        return {
            "optimizer_groups": [*appearance_groups, "opacity"],
            "trainable_fields": [*appearance_groups, "opacity"],
            "frozen_fields": ["xyz", "scaling", "rotation", "exposure"],
        }
    if mode == "continuous_unfrozen":
        all_groups = ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]
        return {
            "optimizer_groups": all_groups,
            "trainable_fields": all_groups,
            "frozen_fields": ["exposure"],
        }
    return {
        "optimizer_groups": ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"],
        "trainable_fields": ["f_dc", "f_rest", "opacity"],
        "frozen_fields": ["xyz", "scaling", "rotation"],
    }


def _build_tstruct_train_args(args: argparse.Namespace) -> List[str]:
    """Build extra args forwarded to *thermal* train.py for pseudo-color structure loss."""
    if float(getattr(args, "t_struct_grad_w", 0.0)) <= 0.0:
        return []
    norm_raw = getattr(args, "t_struct_grad_norm", True)
    if isinstance(norm_raw, str):
        norm = norm_raw.strip().lower() in {"1","true","yes","y","t","on"}
    else:
        norm = bool(norm_raw)
    return [
        "--t_struct_grad_w",
        str(float(getattr(args, "t_struct_grad_w", 0.0))),
        "--t_struct_grad_norm",
        "true" if norm else "false",
    ]


def _build_ss_train_args(*pos, **kw) -> List[str]:
    """Build train.py args for Sparse Support.

    Accepts either:
      - _build_ss_train_args(ap, args)
      - _build_ss_train_args(args, ap=ap)  (preferred)
      - _build_ss_train_args(args)         (errors are raised as RuntimeError)
    """
    ap: Optional[argparse.ArgumentParser] = kw.get("ap", None)
    args: Optional[argparse.Namespace] = None

    if len(pos) == 1 and isinstance(pos[0], argparse.Namespace):
        args = pos[0]
    elif len(pos) == 2 and isinstance(pos[0], argparse.ArgumentParser) and isinstance(pos[1], argparse.Namespace):
        ap, args = pos[0], pos[1]
    elif len(pos) == 2 and isinstance(pos[1], argparse.ArgumentParser) and isinstance(pos[0], argparse.Namespace):
        args, ap = pos[0], pos[1]
    else:
        raise TypeError("_build_ss_train_args expects (args) or (ap, args)")

    if args is None:
        raise RuntimeError("internal: args is None in _build_ss_train_args")

    # Validation (only when SS is enabled)
    if getattr(args, "ss_enable", False):
        # If ap is missing, fall back to RuntimeError with clear messages.
        def _err(msg: str) -> None:
            if ap is not None:
                ap.error(msg)
            raise RuntimeError(msg)

        # source is already validated by argparse choices
        try:
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_aabb_margin", args.ss_aabb_margin, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_voxel_size", args.ss_voxel_size, strict_positive=True)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_nn_dist_thr", args.ss_nn_dist_thr, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_adaptive_alpha", args.ss_adaptive_alpha, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_adaptive_beta", args.ss_adaptive_beta, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_adaptive_max_scale", args.ss_adaptive_max_scale, min_value=1.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_trim_tail_pct", args.ss_trim_tail_pct, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_island_radius", args.ss_island_radius, strict_positive=True)
            if int(args.ss_drop_small_islands) < 0:
                _err("--ss_drop_small_islands must be >= 0")
            if float(args.ss_trim_tail_pct) >= 100.0:
                _err("--ss_trim_tail_pct must be < 100")
        except SystemExit:
            # argparse.error triggers SystemExit; just re-raise
            raise
        except Exception as e:
            _err(str(e))

    if not getattr(args, "ss_enable", False):
        return []

    out: List[str] = ["--ss_enable"]

    # Always forward explicit SS params once enabled.
    out += ["--ss_source", str(args.ss_source)]
    out += ["--ss_use_aabb", "true" if bool(getattr(args, "ss_use_aabb", True)) else "false"]
    out += ["--ss_aabb_margin", str(args.ss_aabb_margin)]

    if args.ss_voxel_size is not None:
        out += ["--ss_voxel_size", str(args.ss_voxel_size)]
    if args.ss_nn_dist_thr is not None:
        out += ["--ss_nn_dist_thr", str(args.ss_nn_dist_thr)]
    if bool(getattr(args, "ss_adaptive_nn", False)):
        out += ["--ss_adaptive_nn"]
    if float(getattr(args, "ss_adaptive_alpha", 1.0)) != 1.0:
        out += ["--ss_adaptive_alpha", str(args.ss_adaptive_alpha)]
    if float(getattr(args, "ss_adaptive_beta", 0.0)) != 0.0:
        out += ["--ss_adaptive_beta", str(args.ss_adaptive_beta)]
    if float(getattr(args, "ss_adaptive_max_scale", 1.5)) != 1.5:
        out += ["--ss_adaptive_max_scale", str(args.ss_adaptive_max_scale)]
    if float(getattr(args, "ss_trim_tail_pct", 0.0)) > 0.0:
        out += ["--ss_trim_tail_pct", str(args.ss_trim_tail_pct)]
    if int(getattr(args, "ss_drop_small_islands", 0)) > 0:
        out += ["--ss_drop_small_islands", str(args.ss_drop_small_islands)]
    if getattr(args, "ss_island_radius", None) is not None:
        out += ["--ss_island_radius", str(args.ss_island_radius)]
    return out


def list_images(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if not dir_path.exists():
        return []
    out: List[Path] = []
    for fp in dir_path.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts:
            out.append(fp)
    out.sort()
    return out


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_image_inventory(dir_path: Path) -> Dict[str, object]:
    entries = [
        {
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
        for path in list_images(dir_path)
    ]
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "count": len(entries),
        "entries_sha256": hashlib.sha256(encoded).hexdigest(),
        "entries": entries,
    }


def _read_colmap_image_names(model_dir: Path) -> List[str]:
    """
    Return image names recorded in COLMAP model (images.bin/txt), or [] on failure.
    """
    ext: Optional[str] = None
    if (model_dir / "images.bin").exists():
        ext = ".bin"
    elif (model_dir / "images.txt").exists():
        ext = ".txt"
    if ext is None:
        return []
    try:
        from utils.read_write_model import read_model  # local import to keep startup robust
        _, images, _ = read_model(str(model_dir), ext=ext)
        names = [str(im.name) for im in images.values() if getattr(im, "name", None)]
        return sorted(set(names))
    except Exception as e:
        eprint(f"[WARN] Failed to read COLMAP image names from {model_dir}: {e}")
        return []


def _build_image_name_alias_dir(src_dir: Path, required_names: List[str], alias_dir: Path) -> Tuple[int, int]:
    """
    Build alias dir where files are named exactly as COLMAP expects.
    Mapping uses filename stem (case-insensitive), so JPG/PNG mismatch can be bridged.
    Returns: (linked_count, missing_count)
    """
    src_imgs = list_images(src_dir)
    by_stem_ci: Dict[str, Path] = {}
    for fp in src_imgs:
        key = fp.stem.lower()
        if key not in by_stem_ci:
            by_stem_ci[key] = fp

    if alias_dir.exists():
        shutil.rmtree(alias_dir)
    ensure_dir(alias_dir)

    linked = 0
    missing = 0
    for req in required_names:
        req_rel = Path(req)
        src = by_stem_ci.get(req_rel.stem.lower())
        if src is None:
            missing += 1
            continue
        dst = alias_dir / req_rel
        ensure_dir(dst.parent)
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        linked += 1
    return linked, missing


def _resolve_explicit_camera_lists(train_list: str, test_list: str) -> Tuple[str, str]:
    """Validate and normalise an explicit train/test camera split.

    Supplying only one list is never allowed: doing so would let the dataset
    loader silently fill the other side from the legacy LLFF hold-out rule.
    The file contents are checked here as an early pipeline preflight; the
    dataset reader still performs the authoritative COLMAP-name checks.
    """
    raw_train = str(train_list or "").strip()
    raw_test = str(test_list or "").strip()
    if bool(raw_train) != bool(raw_test):
        raise ValueError("--train_list and --test_list must be supplied together")
    if not raw_train:
        return "", ""

    resolved: List[str] = []
    name_sets: List[set[str]] = []
    for label, value in (("train", raw_train), ("test", raw_test)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit {label} camera list does not exist: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not names:
            raise ValueError(f"Explicit {label} camera list is empty: {path}")
        if len(names) != len(set(names)):
            raise ValueError(f"Explicit {label} camera list contains duplicate camera names: {path}")
        resolved.append(str(path))
        name_sets.append(set(names))

    overlap = sorted(name_sets[0] & name_sets[1])
    if overlap:
        raise ValueError(f"Explicit train/test camera lists overlap: {overlap[:8]}")
    return resolved[0], resolved[1]


def _append_explicit_camera_lists(
    command: List[str], train_list: str, test_list: str,
    train_list_sha256: str = "", test_list_sha256: str = "",
) -> None:
    if train_list:
        command.extend(["--train_list", train_list, "--test_list", test_list])
        command.extend([
            "--train_list_sha256", train_list_sha256,
            "--test_list_sha256", test_list_sha256,
        ])


def contains_any_file(root: Path, names: Tuple[str, ...], max_depth: int = 2) -> bool:
    """
    Returns True if any file with basename in `names` exists within `root` up to `max_depth`.
    """
    if not root.exists():
        return False
    root = root.resolve()
    # BFS with depth
    queue: List[Tuple[Path, int]] = [(root, 0)]
    while queue:
        cur, d = queue.pop(0)
        try:
            for p in cur.iterdir():
                if p.is_file() and p.name in names:
                    return True
                if p.is_dir() and d < max_depth:
                    queue.append((p, d + 1))
        except Exception:
            continue
    return False

def hardlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    mode:
      - copy: shutil.copy2
      - hardlink: os.link when possible (same filesystem); fallback to copy
      - symlink: os.symlink when possible; fallback to copy
    """
    # Never copy through an old symlink or overwrite an old hardlink in place.
    # Either case could modify the previous source image rather than the
    # derived input/ entry.  input/ is owned by this pipeline, so replacing the
    # directory entry first is both safer and gives every mode the same exact-
    # mirror semantics.
    if os.path.lexists(str(dst)):
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refusing to replace a directory as an input image: {dst}")
        dst.unlink()
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Unknown link mode: {mode}")

def prepare_input_dir(src_images_dir: Path, dataset_root: Path, clean: bool, link_mode: str) -> Path:
    """
    Copy/link chosen aligned RGB images into <dataset_root>/input for COLMAP.
    """
    input_dir = dataset_root / "input"
    if clean and input_dir.exists():
        eprint(f"[INFO] Cleaning existing input dir: {input_dir}")
        shutil.rmtree(input_dir)
    ensure_dir(input_dir)

    src_imgs = list_images(src_images_dir)
    if not src_imgs:
        raise FileNotFoundError(f"No images found under: {src_images_dir}")

    src_names = [fp.name for fp in src_imgs]
    if len(src_names) != len(set(src_names)):
        raise RuntimeError(f"Source image basenames are not unique: {src_images_dir}")
    expected_names = set(src_names)
    stale_files = 0
    for existing in list(input_dir.iterdir()):
        if existing.name in expected_names:
            continue
        if existing.is_dir() and not existing.is_symlink():
            raise RuntimeError(
                f"Unexpected directory in derived input/: {existing}. Use --clean_input for an explicit reset."
            )
        existing.unlink()
        stale_files += 1
    if stale_files:
        eprint(f"[INFO] Removed {stale_files} stale files from COLMAP input/ exact-sync")

    # input/ is a derived, exact mirror: overwrite current names and reject any
    # stale/unowned directory instead of silently mixing source generations.
    eprint(f"[INFO] Preparing COLMAP input/ from: {src_images_dir}  (count={len(src_imgs)}, mode={link_mode})")
    for fp in src_imgs:
        hardlink_or_copy(fp, input_dir / fp.name, mode=link_mode)
    actual_names = {fp.name for fp in list_images(input_dir)}
    if actual_names != expected_names:
        raise RuntimeError(
            f"COLMAP input exact-sync failed: expected={len(expected_names)} actual={len(actual_names)}"
        )
    return input_dir


# ----------------------------
# Windows executable resolution
# ----------------------------

def _ps_resolve_command(name: str) -> Optional[str]:
    """
    Try to resolve an executable/script name via PowerShell Get-Command.
    Helpful when the user can run "colmap" in PowerShell but Python cannot find it.
    """
    if os.name != "nt":
        return None
    ps = shutil.which("pwsh") or shutil.which("powershell")
    if not ps:
        return None
    try:
        # Not using -NoProfile: allow user profiles where aliases/functions might exist.
        out = subprocess.check_output(
            [ps, "-Command", f"(Get-Command {name} -ErrorAction SilentlyContinue).Source"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        out = out.strip()
        if not out:
            return None
        # first line is enough
        return out.splitlines()[0].strip() or None
    except Exception:
        return None

def _normalize_cmd_for_windows(cmd: List[str]) -> List[str]:
    if os.name != "nt" or not cmd:
        return cmd

    exe0 = cmd[0]
    resolved: Optional[str] = None

    p0 = Path(exe0)
    if p0.exists():
        resolved = str(p0.resolve())
    else:
        resolved = shutil.which(exe0)
        if resolved is None and (("\\" not in exe0) and ("/" not in exe0)):
            resolved = _ps_resolve_command(exe0)

    if resolved:
        cmd = [resolved] + cmd[1:]

    suffix = Path(cmd[0]).suffix.lower()
    # Batch scripts cannot be launched directly by CreateProcess; wrap with cmd.exe /c
    if suffix in (".bat", ".cmd"):
        cmd = ["cmd.exe", "/c"] + cmd

    return cmd

def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stdout=None,
    stderr=None,
) -> None:
    cmd2 = _normalize_cmd_for_windows(cmd)
    cwd_str = str(cwd) if cwd else None
    pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
    eprint(f"\n[RUN] {pretty}")
    subprocess.run(cmd2, cwd=cwd_str, env=env, check=True, stdout=stdout, stderr=stderr)


# ----------------------------
# Resumable step markers
# ----------------------------

def marker_path(state_dir: Path, step_name: str) -> Path:
    return state_dir / f"{step_name}.json"

def write_marker(marker: Path, step_name: str, cmd: List[str], cwd: Optional[Path], note: str = "") -> None:
    ensure_dir(marker.parent)
    payload = {
        "step": step_name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(cwd) if cwd else "",
        "cmd": cmd,
        "note": note,
    }
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def marker_matches(marker: Path, cmd: List[str]) -> bool:
    if not marker.exists():
        return False
    try:
        obj = json.loads(marker.read_text(encoding="utf-8"))
        return obj.get("cmd", None) == cmd
    except Exception:
        return False

def should_skip_step(state_dir: Path, step_name: str, cmd: List[str], outputs_ok: bool, force: bool) -> bool:
    """
    Skip when:
      - not forced, and
      - outputs look OK, and either:
          (a) marker exists and matches cmd, or
          (b) marker missing but outputs exist (auto-mark)
    """
    if force:
        return False
    m = marker_path(state_dir, step_name)
    if outputs_ok and marker_matches(m, cmd):
        eprint(f"[SKIP] {step_name}  (marker matched + outputs exist)")
        return True
    if outputs_ok and not m.exists():
        eprint(f"[SKIP] {step_name}  (outputs exist; auto-marking)")
        write_marker(m, step_name, cmd, cwd=None, note="auto-marked from existing outputs")
        return True
    return False


# ----------------------------
# Domain helpers
# ----------------------------

def ensure_sparse_0(sparse_dir: Path) -> Path:
    """
    Some tools expect sparse/0/ with cameras/images/points3D.
    If sparse_dir contains model files directly, move them into sparse/0/.
    """
    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse dir not found: {sparse_dir}")

    subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if any(d.name == "0" for d in subdirs):
        model0 = sparse_dir / "0"
    elif len(subdirs) == 1 and subdirs[0].name.isdigit():
        model0 = subdirs[0]
    else:
        model0 = sparse_dir / "0"
        ensure_dir(model0)

    moved = 0
    for fp in sparse_dir.iterdir():
        if fp.is_file():
            dst = model0 / fp.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(fp), str(dst))
            moved += 1

    if moved:
        eprint(f"[INFO] Moved {moved} sparse model files into: {model0}")

    return model0

@dataclass
class CropCandidate:
    tag: str
    rgb_dir: Path
    count: int
    mean: Dict[str, Optional[float]]
    std: Dict[str, Optional[float]]

def _to_float_or_none(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f

def pick_best_candidate(
    summary_all_json: Path,
    prefer: Tuple[str, ...] = ("edge_f1", "grad_ncc", "nmi", "edge_dice", "mi"),
    mode: str = "legacy",
    edge_f1_eps: float = 0.002,
    allowed_tags: Optional[Tuple[str, ...]] = None,
) -> CropCandidate:
    """
    Auto-pick best crop candidate from eval_crop_metrics summary_all.json.

    Modes:
      - legacy: sort by preferred metrics (descending), same as old behavior.
      - robust: shortlist by edge_f1 margin, then rank with weighted normalized score.
    """
    obj = json.loads(summary_all_json.read_text(encoding="utf-8"))
    candidates = obj.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates found in: {summary_all_json}")

    parsed: List[CropCandidate] = []
    for c in candidates:
        parsed.append(
            CropCandidate(
                tag=str(c.get("tag", "")),
                rgb_dir=Path(str(c.get("rgb_dir", ""))),
                count=int(c.get("count", 0)),
                mean=dict(c.get("mean", {})),
                std=dict(c.get("std", {})),
            )
        )

    parsed = [c for c in parsed if c.count > 0 and c.rgb_dir.exists()]
    if allowed_tags is not None:
        allow = set(allowed_tags)
        parsed = [c for c in parsed if c.tag in allow]
    if not parsed:
        raise RuntimeError("All candidates have count==0 or rgb_dir missing; cannot auto pick.")

    def key_fn(c: CropCandidate):
        ks = []
        for k in prefer:
            v = _to_float_or_none(c.mean.get(k, None))
            ks.append(-1e30 if v is None else float(v))
        return tuple(ks)

    if mode == "legacy":
        parsed.sort(key=key_fn, reverse=True)
        return parsed[0]

    if mode != "robust":
        raise ValueError(f"Unknown auto-pick mode: {mode}")

    # Robust mode:
    # 1) keep candidates close to best edge_f1
    # 2) combine multiple metrics by weighted normalized score
    edge_vals = [_to_float_or_none(c.mean.get("edge_f1")) for c in parsed]
    edge_vals = [v for v in edge_vals if v is not None]
    shortlist = parsed
    if edge_vals:
        edge_best = max(edge_vals)
        shortlist = []
        for c in parsed:
            v = _to_float_or_none(c.mean.get("edge_f1"))
            if v is not None and v >= (edge_best - float(edge_f1_eps)):
                shortlist.append(c)
        if not shortlist:
            shortlist = parsed

    weights = {
        "grad_ncc": 0.30,
        "grad_ssim": 0.25,
        "edge_f1": 0.20,
        "nmi": 0.15,
        "mi": 0.10,
    }
    metric_stats: Dict[str, Tuple[float, float]] = {}
    for k in weights.keys():
        vals = []
        for c in shortlist:
            v = _to_float_or_none(c.mean.get(k))
            if v is not None:
                vals.append(v)
        if vals:
            metric_stats[k] = (min(vals), max(vals))

    def robust_key_fn(c: CropCandidate):
        score = 0.0
        total_w = 0.0
        for k, w in weights.items():
            v = _to_float_or_none(c.mean.get(k))
            if v is None:
                continue
            if k not in metric_stats:
                continue
            lo, hi = metric_stats[k]
            if hi - lo <= 1e-12:
                vn = 0.5
            else:
                vn = (v - lo) / (hi - lo)
            score += float(w) * float(vn)
            total_w += float(w)
        if total_w > 1e-12:
            score /= total_w
        else:
            score = -1e30
        # tie-break by legacy ordering
        return (score,) + key_fn(c)

    shortlist.sort(key=robust_key_fn, reverse=True)
    return shortlist[0]

def _summary_tags(path: Path) -> set:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    tags = set()
    for c in obj.get("candidates", []):
        if isinstance(c, dict):
            t = c.get("tag")
            if isinstance(t, str) and t:
                tags.add(t)
    return tags


def _resolve_default_subdir(root: Path, preferred_names: tuple[str, ...], fallback_name: str) -> Path:
    try:
        child_dirs = {p.name: p for p in root.iterdir() if p.is_dir()}
    except FileNotFoundError:
        return root / fallback_name

    for name in preferred_names:
        path = child_dirs.get(name)
        if path is not None:
            return path

    folded = {name.casefold(): path for name, path in child_dirs.items()}
    for name in preferred_names:
        path = folded.get(name.casefold())
        if path is not None:
            return path

    return root / fallback_name


# ----------------------------
# Pipeline
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full CFR->COLMAP->3DGS(RGB)->ThermalUD->3DGS(T)->Blend->Eval pipeline with one command (resumable)."
    )
    ap.add_argument("--data_root", required=True, help="Dataset root, e.g. <DATA_ROOT>")
    ap.add_argument("--out_root", required=True, help="Output root, e.g. <OUT_ROOT>")

    ap.add_argument("--rgb_dir", default="", help="RGB directory. Default: <data_root>/RGB, or <data_root>/rgb if RGB is absent")
    ap.add_argument("--th_dir", default="", help="Thermal directory. Default: <data_root>/thermal")
    ap.add_argument(
        "--train_list", default="",
        help="Explicit COLMAP camera-name list for training. Must be supplied with --test_list.",
    )
    ap.add_argument(
        "--test_list", default="",
        help="Explicit COLMAP camera-name list for evaluation. Must be supplied with --train_list.",
    )
    ap.add_argument(
        "--thermal_train_list", default="",
        help="Optional Stage-2 camera-name list (for example canonical .png names). Must be paired.",
    )
    ap.add_argument(
        "--thermal_test_list", default="",
        help="Optional Stage-2 evaluation list. Defaults to the RGB lists when both are omitted.",
    )

    ap.add_argument("--colmap", default="colmap", help="COLMAP executable (default: colmap). Can be colmap.exe / colmap.bat / full path.")
    ap.add_argument("--exiftool", default="exiftool", help="ExifTool executable (default: exiftool)")
    ap.add_argument("--required_colmap_version", default="4.1.0",
                    help="Required COLMAP version for SfM (default: 4.1.0).")
    ap.add_argument("--required_colmap_sha256", default="",
                    help="Optional exact COLMAP executable SHA256 pin (recommended for formal runs).")
    ap.add_argument("--require_cuda_colmap", dest="require_cuda_colmap", action="store_true",
                    help="Require CUDA-enabled COLMAP (default: on).")
    ap.add_argument("--allow_non_cuda_colmap", dest="require_cuda_colmap", action="store_false",
                    help="Explicitly allow non-CUDA COLMAP (not for formal experiments).")
    ap.set_defaults(require_cuda_colmap=True)
    ap.add_argument("--require_global_mapper_cudss", dest="require_global_mapper_cudss", action="store_true",
                    help="On Linux, require the GlobalMapper runtime to resolve libcudss (default: on).")
    ap.add_argument("--allow_global_mapper_without_cudss", dest="require_global_mapper_cudss", action="store_false",
                    help="Allow GlobalMapper without libcudss (not for formal runs).")
    ap.set_defaults(require_global_mapper_cudss=True)
    ap.add_argument("--database_policy", choices=["reset", "reuse_verified", "adopt_legacy"], default="reset",
                    help="COLMAP DB policy (formal default: reset; verified reuse and explicit legacy adoption are opt-in).")
    ap.add_argument("--expected_legacy_database_sha256", default="",
                    help="Required pre-adoption database.db SHA256 for --database_policy=adopt_legacy.")
    ap.add_argument("--allow_replace_unverified_outputs", action="store_true",
                    help="Explicitly replace existing derived COLMAP outputs without completion-manifest ownership.")

    ap.add_argument("--align", default="fit", choices=["auto", "fit", "exif", "ecc", "dual", "raw"],
                    help="Which RGB source to use for COLMAP: aligned candidate or raw RGB (default: fit)")
    ap.add_argument("--cfr_fit_k_mode", default="full", choices=["full", "no_kupdate", "naive_k"],
                    help="Forward to cfr.py --fit_k_mode (default: full)")
    ap.add_argument("--cfr_fit_agg_mode", default="median", choices=["median", "mean", "per_pair"],
                    help="Forward to cfr.py --fit_agg_mode (default: median)")
    ap.add_argument("--cfr_exif_noise_pct", type=float, default=0.0,
                    help="Forward to cfr.py --exif_noise_pct (default: 0.0)")
    ap.add_argument("--cfr_exif_missing", action="store_true", default=False,
                    help="Forward to cfr.py --exif_missing (default: off)")
    ap.add_argument("--cfr_exif_noise_seed", type=int, default=0,
                    help="Forward to cfr.py --exif_noise_seed (default: 0)")
    ap.add_argument("--auto_pick_mode", default="robust", choices=["legacy", "robust"],
                    help="Auto-pick strategy when --align auto (default: robust)")
    ap.add_argument("--auto_pick_edge_f1_eps", type=float, default=0.002,
                    help="Robust auto-pick: keep candidates within this edge_f1 margin (default: 0.002)")
    ap.add_argument("--comparison", dest="comparison", action="store_true", default=True,
                    help="Enable cfr.py --comparison (writes side-by-side visuals; slower) (default: on)")
    ap.add_argument("--no_comparison", dest="comparison", action="store_false",
                    help="Disable cfr.py --comparison (default: off)")
    ap.add_argument("--link_mode", default="copy", choices=["copy", "hardlink", "symlink"],
                    help="How to put aligned images into data_root/input (default: copy). hardlink is fastest if same disk.")
    ap.add_argument("--clean_input", action="store_true", help="Clean data_root/input before preparing it")
    ap.add_argument("--clean_fit", action="store_true", help="Clean data_root/fit before running cfr.py")
    ap.add_argument("--clean_thermal_ud", action="store_true", help="Clean data_root/thermal_UD before thermal undistort")
    ap.add_argument("--clean_blend_out", action="store_true", help="Clean out_root/Model_F before blending (otherwise resumable)")

    ap.add_argument("--force", action="store_true", help="Force rerun steps even if outputs exist (disables resume skipping)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not execute")
    ap.add_argument("--skip_train", action="store_true", help="Skip BOTH training stages + their render/metrics (for debugging earlier steps)")
    ap.add_argument("--skip_blend", action="store_true", help="Skip blending + sweep eval")
    ap.add_argument("--save_cmds", action="store_true", default=False,
                    help="Save stage commands to out_root/cmd_*.txt (default: off)")
    ap.add_argument("--run_metrics_plus", action="store_true", default=True,
                    help="Run metrics_plus.py after metrics.py (default: on)")
    ap.add_argument("--metrics_plus_K", type=int, default=8,
                    help="AlignedPSNR search window for metrics_plus.py (default: 8)")
    ap.add_argument("--metrics_plus_bg", type=int, default=0, choices=[0, 1],
                    help="Background color for metrics_plus.py (default: 0)")
    ap.add_argument("--metrics_plus_extra_iqa", type=str,
                    default="flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3",
                    help="Extra IQA set for metrics_plus.py (default: enabled common set)")
    ap.add_argument("--metrics_plus_extra_iqa_space", type=str, default="y", choices=["y", "rgb"],
                    help="IQA input space for metrics_plus.py (default: y)")
    ap.add_argument("--metrics_plus_extra_iqa_device", type=str, default="cuda", choices=["cpu", "cuda", "auto"],
                    help="IQA backend device for metrics_plus.py (default: cuda)")
    ap.add_argument("--run_novel_view_metrics", action="store_true", default=False,
                    help="Run novel_view_metrics.py after thermal metrics (default: off)")
    ap.add_argument("--novel_view_mode", type=str, default="grid72",
                    choices=["orbit", "test_offset", "grid72"],
                    help="Novel-view mode for novel_view_metrics.py (default: grid72)")
    ap.add_argument("--novel_view_N", type=int, default=60,
                    help="Number of novel views for novel_view_metrics.py (used by orbit/test_offset; default: 60)")
    ap.add_argument("--novel_bg", type=int, default=0, choices=[0, 1],
                    help="Background color for novel_view_metrics.py (default: 0)")
    ap.add_argument("--novel_view_device", type=str, default="cuda", choices=["cpu", "cuda"],
                    help="Render device for novel_view_metrics.py (default: cuda)")
    ap.add_argument("--novel_grid_azimuth_count", type=int, default=8,
                    help="grid72 azimuth count for novel_view_metrics.py (default: 8)")
    ap.add_argument("--novel_grid_pitch_list", type=str, default="15,30,60",
                    help="grid72 pitch list, comma-separated (default: 15,30,60)")
    ap.add_argument("--novel_grid_distance_factors", type=str, default="0.5,1,1.5",
                    help="grid72 distance factors, comma-separated (default: 0.5,1,1.5)")
    ap.add_argument("--novel_grid_no_topdown", action="store_true", default=False,
                    help="Disable top-down frame in novel_view_metrics.py grid72 mode (default: off)")
    ap.add_argument("--novel_dump_ellipsoid_proxy", type=_str2bool, nargs="?", const=True, default=False,
                    help="Dump same-camera ellipsoid proxy images in novel_view_metrics.py (default: off)")
    ap.add_argument("--novel_ellipsoid_proxy_dir", type=str, default="",
                    help="Optional output dir for ellipsoid proxy images (default: empty -> script default)")
    ap.add_argument("--novel_dump_sibr_ellipsoid", type=_str2bool, nargs="?", const=True, default=False,
                    help="Dump same-camera SIBR offline ellipsoid images in novel_view_metrics.py (default: off)")
    ap.add_argument("--novel_sibr_exe", type=str, default="",
                    help="Optional path to SIBR_gaussianViewer_app(.exe) for SIBR ellipsoid dump")
    ap.add_argument("--novel_sibr_out_dir", type=str, default="",
                    help="Optional output directory for SIBR ellipsoid images")
    ap.add_argument("--novel_sibr_device", type=int, default=0,
                    help="CUDA device index for SIBR viewer when dumping ellipsoid images (default: 0)")
    ap.add_argument("--novel_sibr_mode", type=str, default="ellipsoids", choices=["ellipsoids", "splats", "points"],
                    help="SIBR gaussian mode for offline dump (default: ellipsoids)")
    ap.add_argument("--novel_sibr_keep_path_file", action="store_true", default=False,
                    help="Keep generated SIBR lookat path file (default: off)")
    ap.add_argument("--debug_dump", action="store_true", default=False,
                    help="Write pipeline debug JSON (default: off)")
    ap.add_argument("--debug_dump_path", type=str, default=None,
                    help="Debug JSON path (default: out_root/pipeline_debug.json)")
    ap.add_argument("--profile_pipeline", action="store_true", default=False,
                    help="Enable pipeline profiling (durations/metadata; default: off)")
    ap.add_argument("--profile_out", type=str, default=None,
                    help="Profiling JSON path (default: out_root/pipeline_profile.json)")
    ap.add_argument("--profile_collect_sizes", action="store_true", default=False,
                    help="Collect file/dir sizes in profile (default: off)")
    ap.add_argument("--profile_collect_counts", action="store_true", default=False,
                    help="Collect counts (PLY vertices / image counts) in profile (default: off)")
    ap.add_argument("--profile_save_logs", action="store_true", default=False,
                    help="Redirect each run step stdout+stderr to out_root/log_<step>.txt (default: off)")
    ap.add_argument("--benchmark_efficiency", action="store_true", default=False,
                    help="Opt in to training/render efficiency sidecars and a final summary (default: off).")
    ap.add_argument("--efficiency_out", type=str, default=None,
                    help="Efficiency summary JSON path (default: out_root/efficiency_benchmark.json).")
    ap.add_argument("--efficiency_render_warmup_views", type=int, default=10,
                    help="Render benchmark warm-up calls, excluded from timing (default: 10).")
    ap.add_argument("--efficiency_render_repeats", type=int, default=3,
                    help="Timed passes over the test views (default: 3).")

    # Step range control (for partial runs / resume)
    ap.add_argument("--from_step", type=int, default=1, help="Execute steps starting from this number (1-14).")
    ap.add_argument("--to_step", type=int, default=14, help="Execute steps up to this number (1-14).")

    # COLMAP defaults copied from your example
    ap.add_argument("--camera", default="SIMPLE_RADIAL")
    ap.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    ap.add_argument("--matcher_args", default="--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500")
    ap.add_argument("--colmap_gpu_index", type=int, default=0,
                    help="GPU index for COLMAP extraction, matching, and global SfM (default: 0).")
    ap.add_argument("--sfm_mapper", default="global", choices=["global", "incremental"],
                    help="COLMAP SfM mode (default: global; incremental requires explicit opt-in).")
    ap.add_argument("--global_mapper_args", default="",
                    help="Additional global_mapper options; locked GPU/seed options are configured separately.")
    ap.add_argument("--global_mapper_random_seed", type=int, default=0,
                    help="Deterministic COLMAP GlobalMapper seed (default: 0).")
    ap.add_argument("--mapper_multiple_models", type=int, default=1)
    ap.add_argument("--min_model_size", type=int, default=5)
    ap.add_argument("--init_min_num_inliers", type=int, default=50)
    ap.add_argument("--abs_pose_min_num_inliers", type=int, default=20)
    ap.add_argument("--use_model_aligner", dest="use_model_aligner", action="store_true")
    ap.add_argument("--no_use_model_aligner", dest="use_model_aligner", action="store_false")
    ap.set_defaults(use_model_aligner=True)
    ap.add_argument("--model_aligner_args", default="--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0")
    ap.add_argument("--prior_position_std_m", type=float, default=1.0)
    ap.add_argument("--wgs84_code", type=int, default=0)
    ap.add_argument("--swap_latlon", action="store_true",
                    help="Swap latitude/longitude consistently in pose priors and the ENU audit (debug only).")

    # Stage 1 training defaults (RGB)
    ap.add_argument("--rgb_iter", type=int, default=30000)
    ap.add_argument(
        "--rgb_res",
        type=int,
        default=-1,
        help=(
            "RGB training/render resolution argument (default: -1, native input "
            "resolution with the standard 3DGS 1600-pixel auto cap). Use 4 only "
            "for explicit legacy quarter-resolution reproduction."
        ),
    )
    ap.add_argument("--rgb_densify_from", type=int, default=1500)
    ap.add_argument("--rgb_densify_until", type=int, default=10000)
    ap.add_argument("--rgb_densify_interval", type=int, default=300)
    ap.add_argument("--rgb_densify_grad", type=float, default=0.001)
    ap.add_argument("--rgb_lambda_dssim", type=float, default=0.3)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument(
        "--baseline_modules_off",
        action="store_true",
        default=False,
        help="Run full-pipeline baseline: disable repo improvements while keeping the current M01 recipe (except thermal opacity falls back to the original 3DGS value; default: off)",
    )
    ap.add_argument(
        "--grouped_ablation_mode",
        default="none",
        choices=["none", "m00_plus_ssp", "m00_plus_stt"],
        help="Selective grouped-ablation restore on top of --baseline_modules_off (default: none).",
    )

    # Sparse Support (Improvement 1) - forwarded to train.py only when enabled
    ap.add_argument("--ss_enable", action="store_true", help="Enable sparse support gating (default: off)")
    ap.add_argument("--ss_enable_rgb", action="store_true", default=True,
                    help="Enable sparse support for RGB stage only (overrides --ss_enable when set; default: on)")
    ap.add_argument("--no_ss_enable_rgb", dest="ss_enable_rgb", action="store_false",
                    help="Disable sparse support for RGB stage.")
    ap.add_argument("--ss_enable_t", action="store_true", default=False,
                    help="Enable sparse support for Thermal stage only (overrides --ss_enable when set; default: off)")
    ap.add_argument("--no_ss_enable_t", dest="ss_enable_t", action="store_false",
                    help="Disable sparse support for Thermal stage.")
    ap.add_argument(
        "--ss_source",
        default="colmap_sparse",
        choices=["colmap_sparse", "init_pcd"],
        help="Sparse support source (default: colmap_sparse)",
    )
    ap.add_argument(
        "--ss_use_aabb",
        type=_str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use AABB gate before NN (default: false). Set true to enable AABB+NN gating.",
    )
    ap.add_argument(
        "--ss_aabb_margin",
        type=float,
        default=0.0,
        help="AABB margin for sparse support (default: 0.0)",
    )
    ap.add_argument(
        "--ss_voxel_size",
        type=float,
        default=1.5,
        help="Voxel size for VoxelHashNN index (default: 1.5)",
    )
    ap.add_argument(
        "--ss_nn_dist_thr",
        type=float,
        default=3.5,
        help="Reserved NN distance threshold for gating (default: 3.5)",
    )
    ap.add_argument(
        "--ss_adaptive_nn",
        action="store_true",
        default=True,
        help="Enable adaptive NN threshold using local spacing proxy (default: on).",
    )
    ap.add_argument("--no_ss_adaptive_nn", dest="ss_adaptive_nn", action="store_false",
                    help="Disable adaptive NN threshold.")
    ap.add_argument(
        "--ss_adaptive_alpha",
        type=float,
        default=1.2,
        help="Adaptive NN alpha for local spacing scale (default: 1.2).",
    )
    ap.add_argument(
        "--ss_adaptive_beta",
        type=float,
        default=0.2,
        help="Adaptive NN beta additive margin in world units (default: 0.2).",
    )
    ap.add_argument(
        "--ss_adaptive_max_scale",
        type=float,
        default=2.0,
        help="Adaptive NN max multiplier over base threshold (default: 2.0).",
    )
    ap.add_argument(
        "--ss_trim_tail_pct",
        type=float,
        default=0.0,
        help="Drop farthest kept NN tail percent after gating (default: 0.0, disabled).",
    )
    ap.add_argument(
        "--ss_drop_small_islands",
        type=int,
        default=10,
        help="Drop tiny disconnected SS islands smaller than this many points (default: 10).",
    )
    ap.add_argument(
        "--ss_island_radius",
        type=float,
        default=10.0,
        help="Island grouping voxel radius (default: 10.0).",
    )

    # Stage 2 training defaults (Thermal)
    ap.add_argument("--t_iter", type=int, default=60000)
    ap.add_argument(
        "--t_res",
        type=int,
        default=-1,
        help=(
            "Thermal training/render resolution argument (default: -1, native input "
            "resolution with the standard 3DGS 1600-pixel auto cap). Use 4 only "
            "for explicit legacy quarter-resolution reproduction."
        ),
    )
    ap.add_argument("--t_feature_lr", type=float, default=0.001)
    ap.add_argument("--t_opacity_lr", type=float, default=2e-4,
                    help="Thermal-only opacity lr (default: 2e-4)")
    ap.add_argument(
        "--t_unfrozen_position_lr", type=float, default=1.6e-6,
        help="continuous_unfrozen xyz LR, used as both init and final (default: 1.6e-6)."
    )
    ap.add_argument(
        "--t_unfrozen_scaling_lr", type=float, default=0.005,
        help="continuous_unfrozen scale LR (default: 0.005)."
    )
    ap.add_argument(
        "--t_unfrozen_rotation_lr", type=float, default=0.001,
        help="continuous_unfrozen rotation LR (default: 0.001)."
    )
    ap.add_argument("--t_lambda_dssim", type=float, default=0.05)
    ap.add_argument("--ss_prune_before_thermal", action="store_true", default=False,
                    help="Thermal-only: prune outside sparse support after restore (default: off)")
    ap.add_argument("--no_ss_prune_before_thermal", dest="ss_prune_before_thermal", action="store_false",
                    help="Disable thermal pre-train sparse-support prune.")
    ap.add_argument("--ss_prune_after_rgb", action="store_true", default=True,
                    help="RGB-only: prune outside sparse support once after stage-1 training (default: on)")
    ap.add_argument("--no_ss_prune_after_rgb", dest="ss_prune_after_rgb", action="store_false",
                    help="Disable RGB post-train sparse-support prune.")
    ap.add_argument("--clamp_scale_max", type=float, default=None,
                    help="Thermal-only: clamp max gaussian scale after restore/prune (default: None)")
    ap.add_argument("--clamp_scale_max_rgb", type=float, default=None,
                    help="RGB-only: clamp max gaussian scale after densify (default: None)")
    ap.add_argument("--clamp_scale_after_rgb_final", action="store_true", default=False,
                    help="RGB-only: clamp once after stage-1 training finishes (default: off)")
    ap.add_argument("--clamp_scale_max_t", type=float, default=10.0,
                    help="Thermal-only: clamp max gaussian scale after restore/prune (default: 10.0)")
    ap.add_argument("--thermal_reset_features", action="store_true", default=True,
                    help="Thermal-only: reset SH features after restore/prune/clamp (default: on)")
    ap.add_argument("--no_thermal_reset_features", dest="thermal_reset_features", action="store_false",
                    help="Disable thermal SH feature reset.")
    ap.add_argument(
        "--thermal_recipe",
        choices=["legacy", "aaai_strict", "geometry_frozen_opacity_adaptive"],
        default="legacy",
        help="Stage-2 protocol preset. Legacy preserves the established command exactly (default: legacy)."
    )
    ap.add_argument(
        "--artifact_save_semantics",
        choices=["legacy", "aligned"],
        default=None,
        help=(
            "Stage-2 PLY/checkpoint save ordering. Unset preserves legacy "
            "ordering; formal Stage-2 recipes resolve to aligned endpoints."
        ),
    )
    ap.add_argument(
        "--thermal_freeze_mode",
        choices=[
            "legacy", "strict", "continuous_unfrozen",
            "geometry_frozen_opacity_adaptive",
        ],
        default="legacy",
        help=(
            "Stage-2 parameter policy: legacy freezes xyz/scale/rotation; strict also freezes opacity; "
            "geometry_frozen_opacity_adaptive trains only SH/opacity; continuous_unfrozen keeps "
            "nonzero geometry LRs while topology stays fixed (default: legacy)."
        ),
    )
    ap.add_argument(
        "--thermal_scale_clamp", choices=["legacy", "off"], default="legacy",
        help="Stage-2 scale clamp policy: established clamp behavior or no clamp (default: legacy)."
    )
    ap.add_argument(
        "--thermal_checkpoint_offsets", nargs="+", type=int,
        default=[10000, 20000, 30000],
        help=(
            "Stage-2 checkpoint offsets from --rgb_iter. Applied by formal recipes, or when explicitly "
            "provided; the final --t_iter checkpoint is always retained."
        ),
    )
    ap.add_argument(
        "--thermal_max_sh_degree", type=int, choices=[0, 1, 3], default=None,
        help="Thermal-only SH cap. When feature reset is enabled, cold-restart from SH degree 0."
    )
    ap.add_argument(
        "--thermal_optimizer_state", choices=["restore", "fresh"], default="restore",
        help="Thermal checkpoint optimizer state (default: restore)."
    )
    ap.add_argument("--debug_gaussian_stats", action="store_true", default=False,
                    help="Thermal-only: log gaussian stats after restore/before save (default: off)")
    ap.add_argument("--sgf_disable", action="store_true", default=False,
                    help="Thermal-only: disable SGF (reapply LRs after restore) (default: off)")

    # Improvement 4: thermal pseudo-color structure gradient loss (default: disabled)
    ap.add_argument("--t_struct_grad_w", type=float, default=0.006,
                    help="Thermal pseudo-color structure gradient loss weight (default: 0.006; set 0 to disable).")
    ap.add_argument("--t_struct_grad_norm", type=_str2bool, nargs="?", const=True, default=True,
                    help="Whether to normalize structure grad loss (default: True). Use --t_struct_grad_norm false to disable.")

    # Blend defaults
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--methods", nargs="+", default=[
        "sh_only", "sh_opacity", "sh_opacity_scale", "sh_opacity_geom",
        "dc_ycc_only", "sh_opacity_dc_ycc"
    ])
    ap.add_argument("--blend_dc_y_from", default="lerp", choices=["rgb", "t", "lerp"],
                    help="Forward to blend_model_strict_endpoints.py --dc_y_from (default: lerp)")
    ap.add_argument("--verify_endpoints", action="store_true", default=True)

    # Eval sweep
    ap.add_argument("--auto_render", action="store_true", default=True)
    ap.add_argument("--eval_thermal_scalar", default="hue_y", choices=["hue", "sat", "val", "hue_y"],
                    help="Forward to eval_blend_sweep.py --thermal_scalar (default: hue_y)")
    ap.add_argument("--eval_thermal_align", default="linear", choices=["none", "linear", "rank"],
                    help="Forward to eval_blend_sweep.py --thermal_align (default: linear)")
    ap.add_argument("--eval_sample_frames", type=int, default=0,
                    help="Forward to eval_blend_sweep.py --sample_frames (default: 0=all)")
    ap.add_argument("--eval_seed", type=int, default=0,
                    help="Forward to eval_blend_sweep.py --seed (default: 0)")
    ap.add_argument("--eval_no_montage", action="store_true", default=False,
                    help="Forward to eval_blend_sweep.py --no_montage (default: off)")
    ap.add_argument("--eval_montage_cols", type=int, default=9999,
                    help="Forward to eval_blend_sweep.py --montage_cols (default: 9999)")
    ap.add_argument("--eval_montage_samples", type=int, default=5,
                    help="Forward to eval_blend_sweep.py --montage_samples (default: 5)")
    ap.add_argument("--eval_render_iter", type=int, default=None,
                    help="Forward to eval_blend_sweep.py --render_iter (default: None)")
    ap.add_argument("--eval_render_source", type=str, default=None,
                    help="Forward to eval_blend_sweep.py --render_source (default: None)")
    ap.add_argument("--eval_render_images", type=str, default=None,
                    help="Forward to eval_blend_sweep.py --render_images (default: None)")
    ap.add_argument("--eval_render_resolution", type=int, default=None,
                    help="Forward to eval_blend_sweep.py --render_resolution (default: None)")
    ap.add_argument("--eval_render_extra", type=str, default="",
                    help="Forward to eval_blend_sweep.py --render_extra (default: empty)")
    ap.add_argument("--eval_gt_mode", default="link", choices=["keep", "delete", "link"],
                    help="Forward to eval_blend_sweep.py --gt_mode (default: link)")

    args = ap.parse_args()
    cli_args = sys.argv[1:]
    try:
        args.train_list, args.test_list = _resolve_explicit_camera_lists(
            args.train_list, args.test_list
        )
        args.thermal_train_list, args.thermal_test_list = _resolve_explicit_camera_lists(
            args.thermal_train_list, args.thermal_test_list
        )
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    if args.thermal_train_list and not args.train_list:
        ap.error("thermal camera lists require the explicit RGB train/test lists")
    if not args.thermal_train_list:
        args.thermal_train_list, args.thermal_test_list = args.train_list, args.test_list
    args.train_list_sha256 = _sha256_file(Path(args.train_list)) if args.train_list else ""
    args.test_list_sha256 = _sha256_file(Path(args.test_list)) if args.test_list else ""
    args.thermal_train_list_sha256 = (
        _sha256_file(Path(args.thermal_train_list)) if args.thermal_train_list else ""
    )
    args.thermal_test_list_sha256 = (
        _sha256_file(Path(args.thermal_test_list)) if args.thermal_test_list else ""
    )
    thermal_protocol_requested = {
        "thermal_recipe": args.thermal_recipe,
        "artifact_save_semantics": args.artifact_save_semantics,
        "thermal_freeze_mode": args.thermal_freeze_mode,
        "thermal_scale_clamp": args.thermal_scale_clamp,
        "thermal_max_sh_degree": args.thermal_max_sh_degree,
        "thermal_optimizer_state": args.thermal_optimizer_state,
        "thermal_checkpoint_offsets": list(args.thermal_checkpoint_offsets),
        "t_unfrozen_position_lr": args.t_unfrozen_position_lr,
        "t_unfrozen_scaling_lr": args.t_unfrozen_scaling_lr,
        "t_unfrozen_rotation_lr": args.t_unfrozen_rotation_lr,
    }
    thermal_checkpoint_offsets_explicit = _cli_option_was_supplied(
        cli_args, "--thermal_checkpoint_offsets"
    )
    artifact_save_semantics_explicit = _cli_option_was_supplied(
        cli_args, "--artifact_save_semantics"
    )

    # Validate step range
    if args.from_step < 1 or args.to_step > 14 or args.from_step > args.to_step:
        ap.error("--from_step/--to_step must satisfy 1 <= from_step <= to_step <= 14")
    if args.eval_sample_frames < 0:
        ap.error("--eval_sample_frames must be >= 0")
    if args.eval_montage_cols <= 0:
        ap.error("--eval_montage_cols must be > 0")
    if args.eval_montage_samples < 0:
        ap.error("--eval_montage_samples must be >= 0")
    if float(args.cfr_exif_noise_pct) < 0.0:
        ap.error("--cfr_exif_noise_pct must be >= 0")
    if args.efficiency_render_warmup_views < 0:
        ap.error("--efficiency_render_warmup_views must be >= 0")
    if args.efficiency_render_repeats <= 0:
        ap.error("--efficiency_render_repeats must be > 0")
    if args.benchmark_efficiency and args.dry_run:
        ap.error("--benchmark_efficiency cannot be combined with --dry_run")
    if args.grouped_ablation_mode != "none" and (not bool(getattr(args, "baseline_modules_off", False))):
        ap.error("--grouped_ablation_mode requires --baseline_modules_off")

    if bool(getattr(args, "baseline_modules_off", False)):
        # Preserve the current M01 recipe / protocol, but disable repository-specific
        # algorithmic modules. Thermal opacity falls back to the original 3DGS
        # value to remove the conservative-opacity improvement.
        args.ss_enable = False
        args.ss_enable_rgb = False
        args.ss_enable_t = False
        args.ss_adaptive_nn = False
        args.ss_trim_tail_pct = 0.0
        args.ss_drop_small_islands = 0
        args.ss_prune_before_thermal = False
        args.ss_prune_after_rgb = False
        args.clamp_scale_max = None
        args.clamp_scale_max_rgb = None
        args.clamp_scale_after_rgb_final = False
        args.clamp_scale_max_t = None
        args.thermal_reset_features = False
        args.t_struct_grad_w = 0.0
        args.t_struct_grad_norm = True
        args.sgf_disable = True
        args.t_opacity_lr = float(ORIG_3DGS_OPT_DEFAULTS["opacity_lr"])

        if args.grouped_ablation_mode == "m00_plus_ssp":
            # Restore only the stage-1 SSP package on top of the baseline transfer recipe.
            args.ss_enable = False
            args.ss_enable_rgb = True
            args.ss_enable_t = False
            args.ss_source = "colmap_sparse"
            args.ss_use_aabb = False
            args.ss_aabb_margin = 0.0
            args.ss_voxel_size = 1.5
            args.ss_nn_dist_thr = 3.5
            args.ss_adaptive_nn = True
            args.ss_adaptive_alpha = 1.2
            args.ss_adaptive_beta = 0.2
            args.ss_adaptive_max_scale = 2.0
            args.ss_trim_tail_pct = 0.0
            args.ss_drop_small_islands = 10
            args.ss_island_radius = 10.0
            args.ss_prune_before_thermal = False
            args.ss_prune_after_rgb = True
        elif args.grouped_ablation_mode == "m00_plus_stt":
            # Restore only the stage-2 STT package on top of the baseline transfer recipe.
            args.ss_enable = False
            args.ss_enable_rgb = False
            args.ss_enable_t = False
            args.ss_prune_before_thermal = False
            args.ss_prune_after_rgb = False
            args.clamp_scale_max_t = 10.0
            args.thermal_reset_features = True
            args.t_struct_grad_w = 0.006
            args.t_struct_grad_norm = True
            args.sgf_disable = False
            args.t_opacity_lr = 2e-4

    formal_stage2_recipe = args.thermal_recipe in (
        "aaai_strict", "geometry_frozen_opacity_adaptive"
    )
    try:
        if formal_stage2_recipe and (
            args.baseline_modules_off or args.grouped_ablation_mode != "none"
        ):
            raise ValueError(
                f"--thermal_recipe {args.thermal_recipe} cannot be combined with baseline recipe flags"
            )
        if formal_stage2_recipe and not args.thermal_reset_features:
            raise ValueError(
                f"--thermal_recipe {args.thermal_recipe} requires thermal feature reset"
            )
        if formal_stage2_recipe and args.sgf_disable:
            raise ValueError(
                f"--thermal_recipe {args.thermal_recipe} does not use --sgf_disable"
            )
        if args.thermal_recipe == "geometry_frozen_opacity_adaptive":
            if _cli_option_was_supplied(cli_args, "--t_opacity_lr") and not math.isclose(
                float(args.t_opacity_lr), 2e-4, rel_tol=0.0, abs_tol=0.0
            ):
                raise ValueError(
                    "--thermal_recipe geometry_frozen_opacity_adaptive requires "
                    "--t_opacity_lr 0.0002"
                )
            args.t_opacity_lr = 2e-4
        thermal_recipe_defaults_applied = _apply_thermal_recipe_defaults(args, cli_args)
        args.artifact_save_semantics = _resolve_artifact_save_semantics(
            args.thermal_recipe, args.artifact_save_semantics
        )
        if args.thermal_freeze_mode == "strict" and args.thermal_scale_clamp != "off":
            raise ValueError(
                "--thermal_freeze_mode strict requires --thermal_scale_clamp off"
            )
        if (
            args.thermal_freeze_mode == "geometry_frozen_opacity_adaptive"
            and args.thermal_scale_clamp != "off"
        ):
            raise ValueError(
                "--thermal_freeze_mode geometry_frozen_opacity_adaptive requires "
                "--thermal_scale_clamp off"
            )
        thermal_checkpoint_offsets_applied = (
            formal_stage2_recipe or thermal_checkpoint_offsets_explicit
        )
        thermal_checkpoint_iterations = _resolve_thermal_checkpoint_iterations(
            rgb_iter=args.rgb_iter,
            t_iter=args.t_iter,
            offsets=args.thermal_checkpoint_offsets,
            use_offsets=thermal_checkpoint_offsets_applied,
        )
        thermal_lr_tokens = _thermal_learning_rate_tokens(args)
    except ValueError as exc:
        ap.error(str(exc))

    # Validate improvement-4 params (always validated; only forwarded when enabled)
    if not math.isfinite(float(getattr(args, "t_struct_grad_w", 0.0))) or float(getattr(args, "t_struct_grad_w", 0.0)) < 0.0:
        ap.error("--t_struct_grad_w must be a finite float >= 0")
    if not math.isfinite(float(getattr(args, "t_opacity_lr", 0.0))) or float(getattr(args, "t_opacity_lr", 0.0)) < 0.0:
        ap.error("--t_opacity_lr must be a finite float >= 0")
    if args.thermal_freeze_mode == "continuous_unfrozen":
        continuous_lrs = {
            "--t_unfrozen_position_lr": args.t_unfrozen_position_lr,
            "--t_unfrozen_scaling_lr": args.t_unfrozen_scaling_lr,
            "--t_unfrozen_rotation_lr": args.t_unfrozen_rotation_lr,
            "--t_opacity_lr": args.t_opacity_lr,
            "--t_feature_lr": args.t_feature_lr,
        }
        invalid_lrs = [
            name
            for name, value in continuous_lrs.items()
            if not math.isfinite(float(value)) or float(value) <= 0.0
        ]
        if invalid_lrs:
            ap.error(
                "--thermal_freeze_mode continuous_unfrozen requires finite nonzero LRs: "
                + ", ".join(invalid_lrs)
            )
    # Sparse Support argument sanity (only when enabled for any stage)
    ss_any_enabled = bool(args.ss_enable or args.ss_enable_rgb or args.ss_enable_t)
    if ss_any_enabled:
        _validate_finite_float(ap, "--ss_aabb_margin", args.ss_aabb_margin, min_value=0.0)
        _validate_finite_float(ap, "--ss_voxel_size", args.ss_voxel_size, strict_positive=True)
        _validate_finite_float(ap, "--ss_nn_dist_thr", args.ss_nn_dist_thr, min_value=0.0)
        _validate_finite_float(ap, "--ss_adaptive_alpha", args.ss_adaptive_alpha, min_value=0.0)
        _validate_finite_float(ap, "--ss_adaptive_beta", args.ss_adaptive_beta, min_value=0.0)
        _validate_finite_float(ap, "--ss_adaptive_max_scale", args.ss_adaptive_max_scale, min_value=1.0)
        _validate_finite_float(ap, "--ss_trim_tail_pct", args.ss_trim_tail_pct, min_value=0.0)
        _validate_finite_float(ap, "--ss_island_radius", args.ss_island_radius, strict_positive=True)
        if int(args.ss_drop_small_islands) < 0:
            ap.error("--ss_drop_small_islands must be >= 0")
        if float(args.ss_trim_tail_pct) >= 100.0:
            ap.error("--ss_trim_tail_pct must be < 100")

    ss_stage_override = bool(args.ss_enable_rgb) or bool(args.ss_enable_t)
    if ss_stage_override:
        ss_enable_rgb = bool(args.ss_enable_rgb)
        ss_enable_t = bool(args.ss_enable_t)
    else:
        ss_enable_rgb = bool(args.ss_enable)
        ss_enable_t = bool(args.ss_enable)

    if formal_stage2_recipe and (
        ss_enable_t or bool(args.ss_prune_before_thermal)
    ):
        ap.error(
            f"--thermal_recipe {args.thermal_recipe} does not permit "
            "Stage-2 sparse-support pruning"
        )

    def _build_ss_args_for_stage(enable_flag: bool) -> List[str]:
        if not enable_flag:
            return []
        tmp = argparse.Namespace(**vars(args))
        tmp.ss_enable = True
        try:
            return _build_ss_train_args(tmp, ap=ap)
        except ValueError as e:
            ap.error(str(e))
        return []

    ss_train_extra: List[str] = _build_ss_args_for_stage(ss_enable_rgb)
    ss_train2_extra: List[str] = _build_ss_args_for_stage(ss_enable_t)

    clamp_effective_rgb = getattr(args, "clamp_scale_max_rgb", None)
    clamp_requested_t = args.clamp_scale_max_t if args.clamp_scale_max_t is not None else args.clamp_scale_max
    clamp_effective_t = None if args.thermal_scale_clamp == "off" else clamp_requested_t
    optimizer_contract = _thermal_optimizer_contract(args)
    thermal_protocol = {
        "schema_name": "uav-tgs-stage2-protocol",
        "schema_version": 1,
        "thermal_recipe": args.thermal_recipe,
        "artifact_save_semantics": args.artifact_save_semantics,
        "artifact_save_semantics_explicit": artifact_save_semantics_explicit,
        "thermal_freeze_mode": args.thermal_freeze_mode,
        "thermal_scale_clamp": args.thermal_scale_clamp,
        "thermal_max_sh_degree": args.thermal_max_sh_degree,
        "thermal_optimizer_state": args.thermal_optimizer_state,
        "thermal_checkpoint_offsets": (
            list(args.thermal_checkpoint_offsets)
            if thermal_checkpoint_offsets_applied
            else None
        ),
        "thermal_checkpoint_offsets_configured": list(args.thermal_checkpoint_offsets),
        "thermal_checkpoint_offsets_applied": thermal_checkpoint_offsets_applied,
        "thermal_checkpoint_iterations": list(thermal_checkpoint_iterations),
        "t_unfrozen_position_lr": args.t_unfrozen_position_lr,
        "t_unfrozen_scaling_lr": args.t_unfrozen_scaling_lr,
        "t_unfrozen_rotation_lr": args.t_unfrozen_rotation_lr,
        "recipe_defaults_applied": list(thermal_recipe_defaults_applied),
        "requested": thermal_protocol_requested,
        "topology_fixed": True,
        "topology_controls": {
            "densify_from_iter": 999999,
            "densify_until_iter": 0,
            "densification_interval": 999999,
            "opacity_reset_interval": 999999,
        },
        "learning_rates": {
            name: float(value) for name, value in thermal_lr_tokens.items()
        },
        "optimizer_groups": list(optimizer_contract["optimizer_groups"]),
        "trainable_fields": list(optimizer_contract["trainable_fields"]),
        "frozen_fields": list(optimizer_contract["frozen_fields"]),
        "scale_clamp_requested": clamp_requested_t,
        "scale_clamp_effective": clamp_effective_t,
    }

    if (
        (not getattr(args, "sgf_disable", False))
        and ss_enable_t
        and (not getattr(args, "ss_prune_before_thermal", False))
        and (not getattr(args, "ss_prune_after_rgb", False))
    ):
        eprint("[WARN] SGF on + ss_enable_t detected, but both ss_prune_before_thermal and ss_prune_after_rgb are off. "
               "SS gating in thermal may not take effect; consider enabling at least one prune path.")
    # Improvement-4 forwarding args (only forwarded when enabled; safe no-op otherwise)
    tstruct_train_extra: List[str] = _build_tstruct_train_args(args)
    def _maybe_raise_file_not_found(msg: str) -> None:
        if args.dry_run:
            eprint(f"[WARN] {msg} (dry_run)")
            return
        raise FileNotFoundError(msg)



    gs_root = Path(__file__).resolve().parent  # gaussian-splatting repo root
    py = sys.executable
    metrics_plus_path = gs_root / "metrics_plus.py"
    if args.run_metrics_plus and not metrics_plus_path.exists():
        raise FileNotFoundError(f"metrics_plus.py not found: {metrics_plus_path}")
    novel_view_metrics_path = gs_root / "novel_view_metrics.py"
    if args.run_novel_view_metrics and not novel_view_metrics_path.exists():
        raise FileNotFoundError(f"novel_view_metrics.py not found: {novel_view_metrics_path}")

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()

    if args.rgb_dir:
        rgb_dir = Path(args.rgb_dir).resolve()
    else:
        rgb_dir = _resolve_default_subdir(data_root, ("RGB", "rgb"), "RGB")
    th_dir = Path(args.th_dir).resolve() if args.th_dir else (data_root / "thermal")

    fit_dir = data_root / "fit"
    metrics_out = fit_dir / "metrics"
    input_dir = data_root / "input"
    thermal_ud = data_root / "thermal_UD"

    model_rgb = out_root / "Model_RGB"
    model_t = out_root / "Model_T"
    model_f = out_root / "Model_F"
    eval_out = out_root / "eval"
    train_eff_rgb = model_rgb / "train_efficiency.json"
    train_eff_t = model_t / "train_efficiency.json"
    render_eff_rgb = model_rgb / "render_efficiency.json"
    render_eff_t = model_t / "render_efficiency.json"

    ensure_dir(out_root)
    train1_cmd = None
    train2_cmd = None
    render1_cmd = None
    render2_cmd = None
    metrics1_cmd = None
    metrics2_cmd = None
    blend_cmd = None
    sweep_cmd = None
    ckpt_rgb = None
    ckpt_t = None

    # State dir for resumable markers
    state_dir = data_root / "_pipeline_state"
    ensure_dir(state_dir)

    def _in_step_range(n: int) -> bool:
        return args.from_step <= n <= args.to_step

    def _efficiency_sidecar_completed(
        path: Path,
        kind: str,
        *,
        stage: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        payload = read_json_if_present(path)
        if not payload:
            return False
        if payload.get("schema_name") != "uav-tgs-efficiency" or payload.get("schema_version") != 1:
            return False
        if payload.get("kind") != kind or payload.get("status") != "completed":
            return False
        if stage is not None and payload.get("stage") != stage:
            return False
        if kind == "training_stage" and iteration is not None:
            result = payload.get("result")
            if not isinstance(result, dict) or result.get("final_iteration") != int(iteration):
                return False
        if kind == "render":
            if iteration is not None and payload.get("iteration") != int(iteration):
                return False
            benchmark = payload.get("benchmark")
            if not isinstance(benchmark, dict):
                return False
            if benchmark.get("warmup_views") != int(args.efficiency_render_warmup_views):
                return False
            if benchmark.get("repeats") != int(args.efficiency_render_repeats):
                return False
        return True

    def _sidecar_fingerprint(path: Path) -> Optional[Tuple[int, int, str]]:
        if not path.is_file():
            return None
        try:
            stat = path.stat()
            return int(stat.st_size), int(stat.st_mtime_ns), _sha256_file(path)
        except Exception:
            return None

    debug_enabled = bool(args.debug_dump)
    debug_dump_written = False
    step_order = [
        "01_cfr",
        "02_eval_crop",
        "03_prepare_input",
        "04_convert_uavfgs",
        "05_train_rgb",
        "06_render_rgb",
        "07_metrics_rgb",
        "08_undistort_thermal",
        "09_normalize_sparse_ud",
        "10_train_thermal",
        "11_render_thermal",
        "12_metrics_thermal",
        "13_blend",
        "14_eval_sweep",
    ]
    debug_steps: Dict[str, Dict[str, object]] = {}
    if debug_enabled:
        for name in step_order:
            debug_steps[name] = {
                "decision": "not_reached",
                "marker": str(marker_path(state_dir, name)),
            }

    # Efficiency summaries reuse the existing boundary-only stage timer.  The
    # expensive recursive profile size/count collectors remain opt-in.
    profile_enabled = bool(args.profile_pipeline or args.benchmark_efficiency)
    profile_dump_written = False
    efficiency_dump_written = False
    profile_start_iso = datetime.now().isoformat(timespec="seconds")
    profile_start_perf = time.perf_counter()
    profile_steps: Dict[str, Dict[str, object]] = {}
    profile_run_meta: Dict[str, Dict[str, object]] = {}
    if profile_enabled:
        for name in step_order:
            profile_steps[name] = {"decision": "not_reached"}

    def _format_cmd_for_debug(cmd: List[str]) -> str:
        cmd2 = _normalize_cmd_for_windows(cmd)
        return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])

    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _json_safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {str(k): _json_safe(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    def _collect_artifacts(step_name: str) -> Dict[str, object]:
        rgb_ply = model_rgb / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
        t_ply = model_t / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
        novel_dir = model_t / "novel_views"
        novel_grid_dir = model_t / "novel_views_grid"
        novel_grid_ellip_dir = model_t / "novel_views_grid_ellip_sibr"
        novel_grid_json = novel_grid_dir / "novel_view_metrics_grid.json"
        novel_legacy_json = model_t / "novel_view_metrics.json"
        metrics_plus_rgb = model_rgb / "results_plus.json"
        metrics_plus_t = model_t / "results_plus.json"
        paths: Dict[str, Optional[str]] = {
            "out_root": str(out_root),
            "model_rgb": str(model_rgb),
            "model_t": str(model_t),
            "model_f": str(model_f),
            "ckpt_rgb": str(ckpt_rgb) if ckpt_rgb is not None else None,
            "ckpt_t": str(ckpt_t) if ckpt_t is not None else None,
            "rgb_ply": str(rgb_ply),
            "t_ply": str(t_ply),
            "thermal_ud": str(thermal_ud),
            "render_rgb_dir": str(model_rgb / "test"),
            "render_t_dir": str(model_t / "test"),
            "novel_views_dir": str(novel_dir),
            "novel_views_grid_dir": str(novel_grid_dir),
            "novel_views_grid_ellip_sibr_dir": str(novel_grid_ellip_dir),
            "novel_view_metrics_grid": str(novel_grid_json),
            "novel_view_metrics_legacy": str(novel_legacy_json),
            "metrics_rgb": str(model_rgb / "results.json"),
            "metrics_t": str(model_t / "results.json"),
            "metrics_plus_rgb": str(metrics_plus_rgb),
            "metrics_plus_t": str(metrics_plus_t),
        }

        exists = {}
        for k, v in paths.items():
            if v is None:
                exists[k] = False
            else:
                exists[k] = Path(v).exists()

        artifacts: Dict[str, object] = {"paths": paths, "exists": exists}

        if args.profile_collect_sizes:
            sizes: Dict[str, Optional[int]] = {}
            for k in ("ckpt_rgb", "ckpt_t", "rgb_ply", "t_ply", "out_root"):
                v = paths.get(k, None)
                if v is None:
                    sizes[k] = None
                else:
                    sizes[k] = _dir_size_bytes(Path(v))
            artifacts["sizes_bytes"] = sizes

        if args.profile_collect_counts:
            counts: Dict[str, Optional[float]] = {}
            counts["rgb_ply_vertices"] = _ply_vertex_count(rgb_ply)
            counts["t_ply_vertices"] = _ply_vertex_count(t_ply)
            counts["render_rgb_images"] = _count_images_recursive(model_rgb / "test")
            counts["render_t_images"] = _count_images_recursive(model_t / "test")
            counts["novel_views_images"] = _count_images_recursive(novel_dir)
            counts["novel_views_grid_images"] = _count_images_recursive(novel_grid_dir)
            counts["novel_views_grid_ellip_sibr_images"] = _count_images_recursive(novel_grid_ellip_dir)
            # Optional metric extraction (if producers already write these keys).
            counts["gaussian_count_novel_grid"] = _json_number(novel_grid_json, "gaussian_count")
            if counts["gaussian_count_novel_grid"] is None:
                counts["gaussian_count_novel_grid"] = _json_number(novel_legacy_json, "gaussian_count")
            if counts["gaussian_count_novel_grid"] is None:
                vtx = _ply_vertex_count(t_ply)
                counts["gaussian_count_novel_grid"] = float(vtx) if vtx is not None else None
            counts["scale_outlier_ratio"] = _json_number(metrics_plus_t, "ScaleOutlierRatio")
            if counts["scale_outlier_ratio"] is None:
                counts["scale_outlier_ratio"] = _json_number(metrics_plus_t, "scale_outlier_ratio")
            if counts["scale_outlier_ratio"] is None:
                counts["scale_outlier_ratio"] = _json_number(novel_grid_json, "ScaleOutlierRatio")
            if counts["scale_outlier_ratio"] is None:
                counts["scale_outlier_ratio"] = _json_number(novel_grid_json, "scale_outlier_ratio")
            if counts["scale_outlier_ratio"] is None:
                counts["scale_outlier_ratio"] = _json_number(novel_legacy_json, "ScaleOutlierRatio")
            if counts["scale_outlier_ratio"] is None:
                counts["scale_outlier_ratio"] = _json_number(novel_legacy_json, "scale_outlier_ratio")
            counts["opacity_low_coverage_ratio"] = _json_number(metrics_plus_t, "OpacityLowCoverageRatio")
            if counts["opacity_low_coverage_ratio"] is None:
                counts["opacity_low_coverage_ratio"] = _json_number(metrics_plus_t, "opacity_low_coverage_ratio")
            if counts["opacity_low_coverage_ratio"] is None:
                counts["opacity_low_coverage_ratio"] = _json_number(novel_grid_json, "OpacityLowCoverageRatio")
            if counts["opacity_low_coverage_ratio"] is None:
                counts["opacity_low_coverage_ratio"] = _json_number(novel_grid_json, "opacity_low_coverage_ratio")
            if counts["opacity_low_coverage_ratio"] is None:
                counts["opacity_low_coverage_ratio"] = _json_number(novel_legacy_json, "OpacityLowCoverageRatio")
            if counts["opacity_low_coverage_ratio"] is None:
                counts["opacity_low_coverage_ratio"] = _json_number(novel_legacy_json, "opacity_low_coverage_ratio")
            counts["render_time_per_frame_s"] = _json_number(novel_grid_json, "RenderTimePerFrame_s")
            if counts["render_time_per_frame_s"] is None:
                counts["render_time_per_frame_s"] = _json_number(novel_grid_json, "render_time_per_frame_s")
            if counts["render_time_per_frame_s"] is None:
                counts["render_time_per_frame_s"] = _json_number(novel_legacy_json, "RenderTimePerFrame_s")
            if counts["render_time_per_frame_s"] is None:
                counts["render_time_per_frame_s"] = _json_number(novel_legacy_json, "render_time_per_frame_s")
            prune_before, prune_after, prune_removed, prune_keep = _find_prune_before_thermal_stats(out_root / "log_10_train_thermal.txt")
            counts["prune_before_thermal_before"] = prune_before
            counts["prune_before_thermal_after"] = prune_after
            counts["prune_before_thermal_removed"] = prune_removed
            counts["prune_before_thermal_keep_ratio"] = prune_keep
            artifacts["counts"] = counts

        return artifacts

    def _profile_update_step(
        step_name: str,
        decision: Optional[str] = None,
        cmd: Optional[List[str]] = None,
        outputs_ok: Optional[bool] = None,
        note: str = "",
    ) -> None:
        if not profile_enabled:
            return
        entry = profile_steps.get(step_name, {})
        if decision is not None:
            entry["decision"] = decision
        if cmd is not None:
            entry["cmd"] = _format_cmd_for_debug(cmd)
        if outputs_ok is not None:
            entry["outputs_ok"] = bool(outputs_ok)
        if note:
            entry["note"] = note
        if step_name in profile_run_meta:
            entry.update(profile_run_meta.get(step_name, {}))
        if "start" not in entry:
            now = _now_iso()
            entry["start"] = now
            entry["end"] = now
            entry["duration_s"] = 0.0
            entry["returncode"] = None
        entry["artifacts"] = _collect_artifacts(step_name)
        profile_steps[step_name] = entry

    def _profile_step_start(step_name: Optional[str], cmd: Optional[List[str]] = None) -> Optional[float]:
        if not profile_enabled or not step_name:
            return None
        entry = profile_run_meta.get(step_name, {})
        if "start" not in entry:
            entry["start"] = _now_iso()
            entry["duration_s"] = 0.0
        if cmd is not None:
            entry["cmd"] = _format_cmd_for_debug(cmd)
        profile_run_meta[step_name] = entry
        return time.perf_counter()

    def _profile_step_end(step_name: Optional[str], t0: Optional[float], returncode: Optional[int] = None, log_path: Optional[Path] = None) -> None:
        if not profile_enabled or not step_name or t0 is None:
            return
        entry = profile_run_meta.get(step_name, {})
        entry["end"] = _now_iso()
        entry["duration_s"] = float(entry.get("duration_s", 0.0)) + float(time.perf_counter() - t0)
        if returncode is not None:
            entry["returncode"] = returncode
        else:
            entry.setdefault("returncode", None)
        if log_path is not None:
            entry["log_path"] = str(log_path)
        profile_run_meta[step_name] = entry
        _profile_update_step(step_name, decision="run")

    def _record_step(step_name: str, decision: str, cmd: Optional[List[str]] = None,
                     outputs_ok: Optional[bool] = None, note: str = "") -> None:
        if not debug_enabled:
            pass
        else:
            entry = debug_steps.get(step_name, {})
            entry["decision"] = decision
            if cmd is not None:
                entry["cmd"] = _format_cmd_for_debug(cmd)
            if outputs_ok is not None:
                entry["outputs_ok"] = bool(outputs_ok)
            if note:
                entry["note"] = note
            if "marker" not in entry:
                entry["marker"] = str(marker_path(state_dir, step_name))
            debug_steps[step_name] = entry
        _profile_update_step(step_name, decision=decision, cmd=cmd, outputs_ok=outputs_ok, note=note)

    def _write_debug_dump(reason: str = "") -> None:
        nonlocal debug_dump_written
        if (not debug_enabled) or debug_dump_written:
            return
        debug_dump_written = True
        dump_path = Path(args.debug_dump_path).expanduser() if args.debug_dump_path else (out_root / "pipeline_debug.json")
        ensure_dir(dump_path.parent)

        render_cmd = None
        if render2_cmd is not None and _in_step_range(11):
            render_cmd = render2_cmd
        elif render1_cmd is not None and _in_step_range(6):
            render_cmd = render1_cmd

        metrics_cmd = None
        if metrics2_cmd is not None and _in_step_range(12):
            metrics_cmd = metrics2_cmd
        elif metrics1_cmd is not None and _in_step_range(7):
            metrics_cmd = metrics1_cmd

        payload = {
            "reason": reason,
            "args": _json_safe(vars(args)),
            "protocol": {"thermal_stage2": _json_safe(thermal_protocol)},
            "cmds": {
                "train1_cmd": _format_cmd_for_debug(train1_cmd) if train1_cmd else None,
                "train2_cmd": _format_cmd_for_debug(train2_cmd) if train2_cmd else None,
                "render_cmd": _format_cmd_for_debug(render_cmd) if render_cmd else None,
                "metrics_cmd": _format_cmd_for_debug(metrics_cmd) if metrics_cmd else None,
                "render1_cmd": _format_cmd_for_debug(render1_cmd) if render1_cmd else None,
                "render2_cmd": _format_cmd_for_debug(render2_cmd) if render2_cmd else None,
                "metrics1_cmd": _format_cmd_for_debug(metrics1_cmd) if metrics1_cmd else None,
                "metrics2_cmd": _format_cmd_for_debug(metrics2_cmd) if metrics2_cmd else None,
            },
            "flags": {
                "ss_enable": bool(getattr(args, "ss_enable", False)),
                "ss_enable_rgb": bool(getattr(args, "ss_enable_rgb", False)),
                "ss_enable_t": bool(getattr(args, "ss_enable_t", False)),
                "ss_source": getattr(args, "ss_source", None),
                "ss_use_aabb": bool(getattr(args, "ss_use_aabb", True)),
                "ss_aabb_margin": getattr(args, "ss_aabb_margin", None),
                "ss_voxel_size": getattr(args, "ss_voxel_size", None),
                "ss_nn_dist_thr": getattr(args, "ss_nn_dist_thr", None),
                "ss_adaptive_nn": bool(getattr(args, "ss_adaptive_nn", False)),
                "ss_adaptive_alpha": getattr(args, "ss_adaptive_alpha", None),
                "ss_adaptive_beta": getattr(args, "ss_adaptive_beta", None),
                "ss_adaptive_max_scale": getattr(args, "ss_adaptive_max_scale", None),
                "ss_trim_tail_pct": getattr(args, "ss_trim_tail_pct", None),
                "ss_drop_small_islands": getattr(args, "ss_drop_small_islands", None),
                "ss_island_radius": getattr(args, "ss_island_radius", None),
                "ss_prune_before_thermal": bool(getattr(args, "ss_prune_before_thermal", False)),
                "ss_prune_after_rgb": bool(getattr(args, "ss_prune_after_rgb", False)),
                "baseline_modules_off": bool(getattr(args, "baseline_modules_off", False)),
                "grouped_ablation_mode": str(getattr(args, "grouped_ablation_mode", "none")),
                "clamp_scale_max": getattr(args, "clamp_scale_max", None),
                "clamp_scale_max_rgb": getattr(args, "clamp_scale_max_rgb", None),
                "clamp_scale_after_rgb_final": bool(getattr(args, "clamp_scale_after_rgb_final", False)),
                "clamp_scale_max_t": getattr(args, "clamp_scale_max_t", None),
                "clamp_effective_rgb": clamp_effective_rgb,
                "clamp_effective_t": clamp_effective_t,
                "thermal_reset_features": bool(getattr(args, "thermal_reset_features", False)),
                "thermal_recipe": getattr(args, "thermal_recipe", "legacy"),
                "thermal_freeze_mode": getattr(args, "thermal_freeze_mode", "legacy"),
                "thermal_scale_clamp": getattr(args, "thermal_scale_clamp", "legacy"),
                "thermal_checkpoint_offsets": list(getattr(args, "thermal_checkpoint_offsets", [])),
                "thermal_checkpoint_iterations": list(thermal_checkpoint_iterations),
                "t_unfrozen_position_lr": getattr(args, "t_unfrozen_position_lr", None),
                "t_unfrozen_scaling_lr": getattr(args, "t_unfrozen_scaling_lr", None),
                "t_unfrozen_rotation_lr": getattr(args, "t_unfrozen_rotation_lr", None),
                "thermal_max_sh_degree": getattr(args, "thermal_max_sh_degree", None),
                "thermal_optimizer_state": getattr(args, "thermal_optimizer_state", "restore"),
                "sgf_disable": bool(getattr(args, "sgf_disable", False)),
                "t_opacity_lr": getattr(args, "t_opacity_lr", None),
                "debug_gaussian_stats": bool(getattr(args, "debug_gaussian_stats", False)),
                "save_cmds": bool(getattr(args, "save_cmds", False)),
                "run_metrics_plus": bool(getattr(args, "run_metrics_plus", False)),
                "metrics_plus_K": getattr(args, "metrics_plus_K", None),
                "metrics_plus_bg": getattr(args, "metrics_plus_bg", None),
                "metrics_plus_extra_iqa": getattr(args, "metrics_plus_extra_iqa", None),
                "metrics_plus_extra_iqa_space": getattr(args, "metrics_plus_extra_iqa_space", None),
                "metrics_plus_extra_iqa_device": getattr(args, "metrics_plus_extra_iqa_device", None),
                "run_novel_view_metrics": bool(getattr(args, "run_novel_view_metrics", False)),
                "novel_view_mode": getattr(args, "novel_view_mode", None),
                "novel_view_N": getattr(args, "novel_view_N", None),
                "novel_bg": getattr(args, "novel_bg", None),
                "novel_view_device": getattr(args, "novel_view_device", None),
                "novel_grid_azimuth_count": getattr(args, "novel_grid_azimuth_count", None),
                "novel_grid_pitch_list": getattr(args, "novel_grid_pitch_list", None),
                "novel_grid_distance_factors": getattr(args, "novel_grid_distance_factors", None),
                "novel_grid_no_topdown": bool(getattr(args, "novel_grid_no_topdown", False)),
                "novel_dump_ellipsoid_proxy": bool(getattr(args, "novel_dump_ellipsoid_proxy", False)),
                "novel_ellipsoid_proxy_dir": getattr(args, "novel_ellipsoid_proxy_dir", None),
                "novel_dump_sibr_ellipsoid": bool(getattr(args, "novel_dump_sibr_ellipsoid", False)),
                "novel_sibr_exe": getattr(args, "novel_sibr_exe", None),
                "novel_sibr_out_dir": getattr(args, "novel_sibr_out_dir", None),
                "novel_sibr_device": getattr(args, "novel_sibr_device", None),
                "novel_sibr_mode": getattr(args, "novel_sibr_mode", None),
                "novel_sibr_keep_path_file": bool(getattr(args, "novel_sibr_keep_path_file", False)),
                "eval_thermal_scalar": getattr(args, "eval_thermal_scalar", None),
                "eval_thermal_align": getattr(args, "eval_thermal_align", None),
                "eval_sample_frames": getattr(args, "eval_sample_frames", None),
                "eval_seed": getattr(args, "eval_seed", None),
                "eval_no_montage": bool(getattr(args, "eval_no_montage", False)),
                "eval_montage_cols": getattr(args, "eval_montage_cols", None),
                "eval_montage_samples": getattr(args, "eval_montage_samples", None),
                "eval_render_iter": getattr(args, "eval_render_iter", None),
                "eval_render_source": getattr(args, "eval_render_source", None),
                "eval_render_images": getattr(args, "eval_render_images", None),
                "eval_render_resolution": getattr(args, "eval_render_resolution", None),
                "eval_render_extra": getattr(args, "eval_render_extra", None),
                "eval_gt_mode": getattr(args, "eval_gt_mode", None),
                "cfr_fit_k_mode": getattr(args, "cfr_fit_k_mode", None),
                "cfr_fit_agg_mode": getattr(args, "cfr_fit_agg_mode", None),
                "cfr_exif_noise_pct": getattr(args, "cfr_exif_noise_pct", None),
                "cfr_exif_missing": bool(getattr(args, "cfr_exif_missing", False)),
                "cfr_exif_noise_seed": getattr(args, "cfr_exif_noise_seed", None),
                "dry_run": bool(getattr(args, "dry_run", False)),
            },
            "paths": {
                "data_root": str(data_root),
                "out_root": str(out_root),
                "model_rgb": str(model_rgb),
                "model_t": str(model_t),
                "model_f": str(model_f),
                "ckpt_rgb": str(ckpt_rgb) if ckpt_rgb is not None else None,
                "ckpt_t": str(ckpt_t) if ckpt_t is not None else None,
                "thermal_ud": str(thermal_ud),
                "state_dir": str(state_dir),
            },
            "steps": [_json_safe(debug_steps.get(name, {})) for name in step_order],
        }
        dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        eprint(f"[INFO] DebugDump: {dump_path}")

    def _write_efficiency_dump(reason: str = "") -> None:
        nonlocal efficiency_dump_written
        if (not args.benchmark_efficiency) or efficiency_dump_written:
            return
        dump_path = Path(args.efficiency_out).expanduser() if args.efficiency_out else (out_root / "efficiency_benchmark.json")

        stages: Dict[str, Dict[str, object]] = {}
        for name in step_order:
            entry = profile_steps.get(name, {})
            run_meta = profile_run_meta.get(name, {})
            stages[name] = {
                "decision": entry.get("decision", "not_reached"),
                # Only an actually entered step has a meaningful duration.
                # Skipped/not-reached stages stay null instead of becoming 0 s.
                "wall_time_s": run_meta.get("duration_s"),
                "return_code": run_meta.get("returncode"),
            }

        rgb_ply = model_rgb / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
        t_ply = model_t / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
        fused_models = []
        if model_f.is_dir():
            try:
                fused_models = [
                    artifact_record(path, str(path.relative_to(model_f)))
                    for path in sorted(model_f.rglob("point_cloud.ply"))
                ]
            except Exception as scan_error:
                eprint(f"[WARN] Efficiency fused-model scan failed: {scan_error}")

        final_reason = reason or "incomplete"
        if final_reason.startswith("error:"):
            run_status = "failed"
        elif final_reason == "done":
            run_status = "completed"
        else:
            run_status = "partial"
        payload = {
            "schema_name": "uav-tgs-efficiency",
            "schema_version": 1,
            "kind": "pipeline",
            "run": {
                "status": run_status,
                "reason": final_reason,
                "started_at": profile_start_iso,
                "ended_at": _now_iso(),
                "total_wall_time_s": float(time.perf_counter() - profile_start_perf),
                "data_root": str(data_root),
                "out_root": str(out_root),
            },
            "measurement_scope": {
                "training_time": "train.py entry-to-return; includes model setup, evaluation, and saves",
                "training_peak_memory": "PyTorch caching allocator, reset once at training boundary",
                "render_time": "test-view render calls only; warm-up, GT handling, CPU transfer, and image I/O excluded",
                "stage_time": "blocking subprocess wall time",
            },
            "protocol": {"thermal_stage2": _json_safe(thermal_protocol)},
            "stages": stages,
            "training": {
                "rgb": read_json_if_present(train_eff_rgb),
                "thermal": read_json_if_present(train_eff_t),
            },
            "rendering": {
                "rgb": read_json_if_present(render_eff_rgb),
                "thermal": read_json_if_present(render_eff_t),
            },
            "models": {
                "rgb": artifact_record(rgb_ply, "rgb"),
                "thermal": artifact_record(t_ply, "thermal"),
                "fused": fused_models,
            },
        }
        try:
            atomic_write_json(dump_path, payload)
            efficiency_dump_written = True
            eprint(f"[INFO] EfficiencyBenchmark: {dump_path}")
        except Exception as write_error:
            eprint(f"[WARN] Efficiency summary could not be written: {write_error}")

    def _write_profile_dump(reason: str = "") -> None:
        nonlocal profile_dump_written
        _write_efficiency_dump(reason)
        if (not args.profile_pipeline) or profile_dump_written:
            return
        profile_dump_written = True
        dump_path = Path(args.profile_out).expanduser() if args.profile_out else (out_root / "pipeline_profile.json")
        ensure_dir(dump_path.parent)

        time_end = _now_iso()
        total_s = time.perf_counter() - profile_start_perf

        payload = {
            "meta": {
                "time_start": profile_start_iso,
                "time_end": time_end,
                "total_s": float(total_s),
                "cwd": str(Path.cwd()),
                "python": sys.executable,
                "reason": reason or "done",
                "args": _json_safe(vars(args)),
                "protocol": {"thermal_stage2": _json_safe(thermal_protocol)},
                "flags": {
                    "ss_enable": bool(getattr(args, "ss_enable", False)),
                    "ss_enable_rgb": bool(getattr(args, "ss_enable_rgb", False)),
                    "ss_enable_t": bool(getattr(args, "ss_enable_t", False)),
                    "ss_use_aabb": bool(getattr(args, "ss_use_aabb", True)),
                    "ss_adaptive_nn": bool(getattr(args, "ss_adaptive_nn", False)),
                    "ss_adaptive_alpha": getattr(args, "ss_adaptive_alpha", None),
                    "ss_adaptive_beta": getattr(args, "ss_adaptive_beta", None),
                    "ss_adaptive_max_scale": getattr(args, "ss_adaptive_max_scale", None),
                    "ss_trim_tail_pct": getattr(args, "ss_trim_tail_pct", None),
                    "ss_drop_small_islands": getattr(args, "ss_drop_small_islands", None),
                    "ss_island_radius": getattr(args, "ss_island_radius", None),
                    "ss_prune_after_rgb": bool(getattr(args, "ss_prune_after_rgb", False)),
                    "baseline_modules_off": bool(getattr(args, "baseline_modules_off", False)),
                    "grouped_ablation_mode": str(getattr(args, "grouped_ablation_mode", "none")),
                    "clamp_scale_max": getattr(args, "clamp_scale_max", None),
                    "clamp_scale_max_rgb": getattr(args, "clamp_scale_max_rgb", None),
                    "clamp_scale_after_rgb_final": bool(getattr(args, "clamp_scale_after_rgb_final", False)),
                    "clamp_scale_max_t": getattr(args, "clamp_scale_max_t", None),
                    "clamp_effective_rgb": clamp_effective_rgb,
                    "clamp_effective_t": clamp_effective_t,
                    "thermal_reset_features": bool(getattr(args, "thermal_reset_features", False)),
                    "thermal_recipe": getattr(args, "thermal_recipe", "legacy"),
                    "thermal_freeze_mode": getattr(args, "thermal_freeze_mode", "legacy"),
                    "thermal_scale_clamp": getattr(args, "thermal_scale_clamp", "legacy"),
                    "thermal_checkpoint_offsets": list(getattr(args, "thermal_checkpoint_offsets", [])),
                    "thermal_checkpoint_iterations": list(thermal_checkpoint_iterations),
                    "t_unfrozen_position_lr": getattr(args, "t_unfrozen_position_lr", None),
                    "t_unfrozen_scaling_lr": getattr(args, "t_unfrozen_scaling_lr", None),
                    "t_unfrozen_rotation_lr": getattr(args, "t_unfrozen_rotation_lr", None),
                    "thermal_max_sh_degree": getattr(args, "thermal_max_sh_degree", None),
                    "thermal_optimizer_state": getattr(args, "thermal_optimizer_state", "restore"),
                    "t_opacity_lr": getattr(args, "t_opacity_lr", None),
                    "eval_thermal_scalar": getattr(args, "eval_thermal_scalar", None),
                    "eval_thermal_align": getattr(args, "eval_thermal_align", None),
                    "eval_sample_frames": getattr(args, "eval_sample_frames", None),
                    "eval_seed": getattr(args, "eval_seed", None),
                    "eval_no_montage": bool(getattr(args, "eval_no_montage", False)),
                    "eval_montage_cols": getattr(args, "eval_montage_cols", None),
                    "eval_montage_samples": getattr(args, "eval_montage_samples", None),
                    "eval_render_iter": getattr(args, "eval_render_iter", None),
                    "eval_render_source": getattr(args, "eval_render_source", None),
                    "cfr_fit_k_mode": getattr(args, "cfr_fit_k_mode", None),
                    "cfr_fit_agg_mode": getattr(args, "cfr_fit_agg_mode", None),
                    "cfr_exif_noise_pct": getattr(args, "cfr_exif_noise_pct", None),
                    "cfr_exif_missing": bool(getattr(args, "cfr_exif_missing", False)),
                    "cfr_exif_noise_seed": getattr(args, "cfr_exif_noise_seed", None),
                    "novel_dump_sibr_ellipsoid": bool(getattr(args, "novel_dump_sibr_ellipsoid", False)),
                    "novel_sibr_exe": getattr(args, "novel_sibr_exe", None),
                    "novel_sibr_out_dir": getattr(args, "novel_sibr_out_dir", None),
                    "novel_sibr_device": getattr(args, "novel_sibr_device", None),
                    "novel_sibr_mode": getattr(args, "novel_sibr_mode", None),
                    "novel_sibr_keep_path_file": bool(getattr(args, "novel_sibr_keep_path_file", False)),
                    "eval_render_images": getattr(args, "eval_render_images", None),
                    "eval_render_resolution": getattr(args, "eval_render_resolution", None),
                    "eval_render_extra": getattr(args, "eval_render_extra", None),
                    "eval_gt_mode": getattr(args, "eval_gt_mode", None),
                },
            },
            "steps": _json_safe(profile_steps),
        }
        dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        eprint(f"[INFO] ProfileDump: {dump_path}")

    profile_reason = {"value": ""}
    if profile_enabled:
        import atexit

        def _profile_excepthook(exctype, value, tb):
            profile_reason["value"] = f"error:{getattr(exctype, '__name__', 'Exception')}"
            _write_profile_dump(profile_reason["value"])
            sys.__excepthook__(exctype, value, tb)

        sys.excepthook = _profile_excepthook
        atexit.register(lambda: _write_profile_dump(profile_reason["value"]))


    def maybe_run(cmd: List[str], cwd: Optional[Path] = None, step_name: Optional[str] = None):
        t0 = _profile_step_start(step_name, cmd)
        log_path: Optional[Path] = None
        if args.profile_save_logs and step_name:
            log_path = out_root / f"log_{step_name}.txt"
        if args.dry_run:
            cmd2 = _normalize_cmd_for_windows(cmd)
            pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
            eprint(f"\n[DRY] {pretty}")
            _profile_step_end(step_name, t0, returncode=None, log_path=log_path)
            return
        rc = None
        try:
            if log_path is not None:
                ensure_dir(log_path.parent)
                with log_path.open("a", encoding="utf-8") as f:
                    run_cmd(cmd, cwd=cwd, stdout=f, stderr=f)
            else:
                run_cmd(cmd, cwd=cwd)
            rc = 0
        except subprocess.CalledProcessError as e:
            rc = e.returncode
            raise
        finally:
            _profile_step_end(step_name, t0, returncode=rc, log_path=log_path)

    # Expected CFR outputs
    cand_fit = fit_dir / "image" / "image-fit"
    cand_exif = fit_dir / "image" / "image-exif"
    cand_ecc = fit_dir / "image" / "image-ecc"
    cand_dual = fit_dir / "image" / "image-dual"
    th_dual = fit_dir / "thermal" / "thermal-dual"
    need_ecc = (args.align == "ecc")
    need_dual = (args.align == "dual")
    need_raw = (args.align == "raw")
    stress_fit_missing = (args.align == "fit" and bool(getattr(args, "cfr_exif_missing", False)))

    # -------- 1) CFR
    if args.clean_fit and fit_dir.exists():
        eprint(f"[INFO] Cleaning fit dir: {fit_dir}")
        shutil.rmtree(fit_dir)

    cfr_align = "both"
    if need_ecc:
        cfr_align = "all"
    elif stress_fit_missing:
        cfr_align = "fit"
    cfr_cmd = [py, "cfr.py", "--rgb_dir", str(rgb_dir), "--th_dir", str(th_dir), "--out_dir", str(fit_dir),
               "--align", cfr_align, "--stage", "both"]
    cfr_cmd += ["--fit_k_mode", str(args.cfr_fit_k_mode)]
    cfr_cmd += ["--fit_agg_mode", str(args.cfr_fit_agg_mode)]
    if float(args.cfr_exif_noise_pct) > 0.0:
        cfr_cmd += ["--exif_noise_pct", str(args.cfr_exif_noise_pct)]
    if bool(args.cfr_exif_missing):
        cfr_cmd += ["--exif_missing"]
    if int(args.cfr_exif_noise_seed) != 0:
        cfr_cmd += ["--exif_noise_seed", str(args.cfr_exif_noise_seed)]
    if need_dual:
        cfr_cmd.append("--dual")
    if args.comparison:
        cfr_cmd.append("--comparison")

    if stress_fit_missing:
        cfr_outputs_ok = cand_fit.exists() and (len(list_images(cand_fit)) > 0)
    else:
        cfr_outputs_ok = (
            cand_fit.exists() and cand_exif.exists() and
            (len(list_images(cand_fit)) > 0) and (len(list_images(cand_exif)) > 0)
        )
    if need_dual:
        cfr_outputs_ok = cfr_outputs_ok and cand_dual.exists() and th_dual.exists() and (
            (len(list_images(cand_dual)) > 0) and (len(list_images(th_dual)) > 0)
        )
    if need_ecc:
        cfr_outputs_ok = cfr_outputs_ok and cand_ecc.exists() and (len(list_images(cand_ecc)) > 0)
    if need_raw:
        eprint("[SKIP] 01_cfr (--align raw uses original RGB directly)")
        _record_step("01_cfr", "skip_raw", cfr_cmd, outputs_ok=True, note=f"align=raw; src={rgb_dir}")
    elif not _in_step_range(1):
        eprint("[SKIP] 01_cfr (outside selected step range)")
        _record_step("01_cfr", "skip_range", cfr_cmd, outputs_ok=cfr_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "01_cfr", cfr_cmd, outputs_ok=cfr_outputs_ok, force=args.force)
        if skip:
            _record_step("01_cfr", "skip", cfr_cmd, outputs_ok=cfr_outputs_ok)
        else:
            ensure_dir(fit_dir)
            maybe_run(cfr_cmd, cwd=gs_root, step_name="01_cfr")
            # re-evaluate
            if stress_fit_missing:
                cfr_outputs_ok = cand_fit.exists() and (len(list_images(cand_fit)) > 0)
            else:
                cfr_outputs_ok = (
                    cand_fit.exists() and cand_exif.exists() and
                    (len(list_images(cand_fit)) > 0) and (len(list_images(cand_exif)) > 0)
                )
            if need_dual:
                cfr_outputs_ok = cfr_outputs_ok and cand_dual.exists() and th_dual.exists() and (
                    (len(list_images(cand_dual)) > 0) and (len(list_images(th_dual)) > 0)
                )
            if need_ecc:
                cfr_outputs_ok = cfr_outputs_ok and cand_ecc.exists() and (len(list_images(cand_ecc)) > 0)
            if not cfr_outputs_ok:
                if stress_fit_missing:
                    need_msg = f"Expected cfr outputs missing. Need:\n  {cand_fit}"
                else:
                    need_msg = f"Expected cfr outputs missing. Need:\n  {cand_fit}\n  {cand_exif}"
                if need_dual:
                    need_msg += f"\n  {cand_dual}\n  {th_dual}"
                if need_ecc:
                    need_msg += f"\n  {cand_ecc}"
                _maybe_raise_file_not_found(need_msg)
            _record_step("01_cfr", "run", cfr_cmd, outputs_ok=cfr_outputs_ok)
            write_marker(marker_path(state_dir, "01_cfr"), "01_cfr", cfr_cmd, cwd=gs_root)

    # -------- 2) Evaluate crop candidates (default: fit + exif)
    ensure_dir(metrics_out)
    summary_all = metrics_out / "summary_all.json"
    metrics_rgb = metrics_out / "crop_rgb"
    ensure_dir(metrics_rgb)
    metrics_dual = metrics_out / "crop_dual"
    eval_dual = need_dual
    if eval_dual:
        ensure_dir(metrics_dual)
    eval_cmd_base = [py, "eval_crop_metrics.py", "--th_dir", str(th_dir),
                     "--rgb_dir", str(cand_fit), "--rgb_dir", str(cand_exif),
                     "--tag", "fit", "--tag", "exif",
                     "--out_dir", str(metrics_rgb)]
    eval_cmd_dual = [py, "eval_crop_metrics.py", "--th_dir", str(th_dual),
                     "--rgb_dir", str(cand_dual),
                     "--tag", "dual",
                     "--out_dir", str(metrics_dual)]
    eval_outputs_ok = summary_all.exists() and summary_all.stat().st_size > 50
    if eval_outputs_ok:
        tags = _summary_tags(summary_all)
        if not {"fit", "exif"}.issubset(tags):
            eval_outputs_ok = False
        if eval_dual:
            eval_outputs_ok = eval_outputs_ok and ("dual" in tags)
        else:
            # Default path should not auto-pick dual from stale summaries.
            eval_outputs_ok = eval_outputs_ok and ("dual" not in tags)

    if need_raw:
        eprint("[SKIP] 02_eval_crop (--align raw bypasses CFR candidate evaluation)")
        _record_step("02_eval_crop", "skip_raw", eval_cmd_base, outputs_ok=True, note=f"align=raw; src={rgb_dir}")
    elif stress_fit_missing:
        eprint("[SKIP] 02_eval_crop (stress_fit_missing skips fit/exif candidate evaluation)")
        _record_step(
            "02_eval_crop",
            "skip_stress_fit_missing",
            eval_cmd_base,
            outputs_ok=True,
            note="stress_fit_missing=True; skip step2 to avoid EXIF gate dependency",
        )
    elif not _in_step_range(2):

        eprint("[SKIP] 02_eval_crop (outside selected step range)")

        _record_step("02_eval_crop", "skip_range", eval_cmd_base, outputs_ok=eval_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "02_eval_crop", eval_cmd_base, outputs_ok=eval_outputs_ok, force=args.force)
        if skip:
            _record_step("02_eval_crop", "skip", eval_cmd_base, outputs_ok=eval_outputs_ok)
        else:
            # Try with --ssim first (if available), then fallback without it.
            try:
                maybe_run(eval_cmd_base + ["--ssim"], cwd=gs_root, step_name="02_eval_crop_rgb")
            except subprocess.CalledProcessError:
                eprint("[WARN] eval_crop_metrics.py failed with --ssim. Retrying without --ssim ...")
                maybe_run(eval_cmd_base, cwd=gs_root, step_name="02_eval_crop_rgb")

            if eval_dual:
                try:
                    maybe_run(eval_cmd_dual + ["--ssim"], cwd=gs_root, step_name="02_eval_crop_dual")
                except subprocess.CalledProcessError:
                    eprint("[WARN] eval_crop_metrics.py (dual) failed with --ssim. Retrying without --ssim ...")
                    maybe_run(eval_cmd_dual, cwd=gs_root, step_name="02_eval_crop_dual")

            # merge summaries into metrics_out/summary_all.json (+ csv)
            try:
                rgb_all = json.loads((metrics_rgb / "summary_all.json").read_text(encoding="utf-8"))
                objs = [rgb_all]
                if eval_dual:
                    dual_all = json.loads((metrics_dual / "summary_all.json").read_text(encoding="utf-8"))
                    objs.append(dual_all)
                cands = []
                for obj in objs:
                    for c in obj.get("candidates", []):
                        if c and isinstance(c, dict):
                            cands.append(c)
                combined = {"candidates": cands}
                summary_all.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

                # write summary_all.csv
                metrics_keys = ["mi", "nmi", "grad_ncc", "edge_dice", "edge_f1"]
                has_ssim = any(("grad_ssim" in c.get("mean", {})) for c in cands if isinstance(c, dict))
                if has_ssim:
                    metrics_keys.append("grad_ssim")
                rows = []
                for c in cands:
                    row = {"tag": c.get("tag"), "count": c.get("count"), "rgb_dir": c.get("rgb_dir")}
                    mean = c.get("mean", {}) if isinstance(c, dict) else {}
                    std = c.get("std", {}) if isinstance(c, dict) else {}
                    for k in metrics_keys:
                        row[f"{k}_mean"] = mean.get(k)
                        row[f"{k}_std"] = std.get(k)
                    rows.append(row)
                out_all_csv = metrics_out / "summary_all.csv"
                write_csv(str(out_all_csv), rows)
            except Exception as e:
                eprint(f"[WARN] Failed to merge crop metrics: {e}")

            eval_outputs_ok = summary_all.exists() and summary_all.stat().st_size > 50
            if not eval_outputs_ok:
                _maybe_raise_file_not_found(f"summary_all.json not found or empty: {summary_all}")
            _record_step("02_eval_crop", "run", eval_cmd_base, outputs_ok=eval_outputs_ok)
            write_marker(marker_path(state_dir, "02_eval_crop"), "02_eval_crop", eval_cmd_base, cwd=gs_root)

    # -------- 3) Decide which candidate to use, then prepare input/
    if args.align == "raw":
        chosen_tag = "raw"
        chosen_dir = rgb_dir
    elif args.align == "fit":
        chosen_tag = "fit"
        chosen_dir = cand_fit
    elif args.align == "exif":
        chosen_tag = "exif"
        chosen_dir = cand_exif
    elif args.align == "ecc":
        chosen_tag = "ecc"
        chosen_dir = cand_ecc
    elif args.align == "dual":
        chosen_tag = "dual"
        chosen_dir = cand_dual
    else:
        best = pick_best_candidate(
            summary_all,
            mode=args.auto_pick_mode,
            edge_f1_eps=args.auto_pick_edge_f1_eps,
            allowed_tags=("fit", "exif") if (not need_dual and not need_ecc) else None,
        )
        chosen_tag = best.tag
        chosen_dir = best.rgb_dir
        eprint(f"[INFO] Auto-picked candidate: {chosen_tag}  (mode={args.auto_pick_mode}, from {summary_all})")

    th_dir_for_ud = th_dir
    if chosen_tag == "dual":
        if th_dual.exists() and (len(list_images(th_dual)) > 0):
            th_dir_for_ud = th_dual
            eprint(f"[INFO] Using dual thermal for undistort: {th_dir_for_ud}")
        else:
            eprint(f"[WARN] dual selected but thermal-dual missing: {th_dual}. Fallback to original thermal.")

    # Prepare input dir
    if args.clean_input and input_dir.exists():
        eprint(f"[INFO] Cleaning input dir: {input_dir}")
        shutil.rmtree(input_dir)

    chosen_inventory = _flat_image_inventory(chosen_dir)
    input_inventory_before = _flat_image_inventory(input_dir) if input_dir.is_dir() else None
    prep_cmd = [
        py,
        "-c",
        f"print('prepare_input: {chosen_tag} -> {input_dir}')",
        f"source_inventory_sha256={chosen_inventory['entries_sha256']}",
        f"source_image_count={chosen_inventory['count']}",
        f"link_mode={args.link_mode}",
    ]
    prep_outputs_ok = input_inventory_before == chosen_inventory

    if not _in_step_range(3):

        eprint("[SKIP] 03_prepare_input (outside selected step range)")

        _record_step("03_prepare_input", "skip_range", prep_cmd, outputs_ok=prep_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "03_prepare_input", prep_cmd, outputs_ok=prep_outputs_ok, force=args.force)
        if skip:
            _record_step("03_prepare_input", "skip", prep_cmd, outputs_ok=prep_outputs_ok)
        else:
            t0 = _profile_step_start("03_prepare_input", prep_cmd)
            rc = 0
            try:
                prepare_input_dir(chosen_dir, data_root, clean=False, link_mode=args.link_mode)
            except FileNotFoundError as e:
                rc = 1
                _maybe_raise_file_not_found(str(e))
            finally:
                _profile_step_end("03_prepare_input", t0, returncode=rc)
            prep_outputs_ok = input_dir.is_dir() and _flat_image_inventory(input_dir) == chosen_inventory
            if not prep_outputs_ok:
                _maybe_raise_file_not_found(f"input dir has no images: {input_dir}")
            _record_step("03_prepare_input", "run", prep_cmd, outputs_ok=prep_outputs_ok)
            write_marker(marker_path(state_dir, "03_prepare_input"), "03_prepare_input", prep_cmd, cwd=gs_root, note=f"chosen={chosen_tag}; src={chosen_dir}")

    # -------- 4) COLMAP (convert_uavfgs.py)
    sparse_aligned = data_root / "distorted" / "sparse_aligned"
    sparse_fallback = data_root / "sparse" / "0"
    conversion_manifest = data_root / "distorted" / "conversion_completion_manifest.json"
    convert_cmd = [
        py, "convert_uavfgs.py",
        "-s", str(data_root),
        "--colmap_executable", str(args.colmap),
        "--exiftool_executable", str(args.exiftool),
        "--required_colmap_version", str(args.required_colmap_version),
        "--required_colmap_sha256", str(args.required_colmap_sha256),
        "--database_policy", str(args.database_policy),
        "--expected_legacy_database_sha256", str(args.expected_legacy_database_sha256),
        "--wgs84_code", str(args.wgs84_code),
        "--prior_position_std_m", str(args.prior_position_std_m),
        "--camera", str(args.camera),
        "--matching", str(args.matching),
        "--matcher_args", str(args.matcher_args),
        "--colmap_gpu_index", str(args.colmap_gpu_index),
        "--sfm_mapper", str(args.sfm_mapper),
        "--global_mapper_args", str(args.global_mapper_args),
        "--global_mapper_random_seed", str(args.global_mapper_random_seed),
        "--mapper_multiple_models", str(args.mapper_multiple_models),
        "--min_model_size", str(args.min_model_size),
        "--init_min_num_inliers", str(args.init_min_num_inliers),
        "--abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
    ]
    if bool(args.swap_latlon):
        convert_cmd.append("--swap_latlon")
    convert_cmd.append("--require_cuda_colmap" if bool(args.require_cuda_colmap) else "--allow_non_cuda_colmap")
    convert_cmd.append(
        "--require_global_mapper_cudss"
        if bool(args.require_global_mapper_cudss)
        else "--allow_global_mapper_without_cudss"
    )
    if bool(args.allow_replace_unverified_outputs):
        convert_cmd.append("--allow_replace_unverified_outputs")
    if bool(args.use_model_aligner):
        convert_cmd.extend(["--use_model_aligner", "--model_aligner_args", str(args.model_aligner_args)])
    else:
        convert_cmd.append("--no_use_model_aligner")

    conversion_validation_cache: Optional[Tuple[bool, str, Optional[Dict[str, object]], str]] = None

    def _pick_sparse_model_for_ud(*, refresh: bool = False) -> Optional[Path]:
        """Accept step 4 only through the atomic completion-manifest contract."""
        nonlocal conversion_validation_cache
        if refresh:
            conversion_validation_cache = None
        if conversion_validation_cache is None:
            try:
                import convert_uavfgs as converter

                resolved_colmap = Path(converter._resolve_executable(str(args.colmap)))
                if not resolved_colmap.is_file():
                    raise FileNotFoundError(f"COLMAP executable is missing: {resolved_colmap}")
                current_colmap_sha = converter._sha256_file(resolved_colmap)
                verified, reason, manifest = converter.validate_conversion_completion_manifest(
                    data_root,
                    expected_arguments=[str(value) for value in convert_cmd[2:]],
                    expected_colmap_sha256=current_colmap_sha,
                    expected_min_registered_images=int(args.min_model_size),
                )
                conversion_validation_cache = (verified, reason, manifest, current_colmap_sha)
            except Exception as exc:
                conversion_validation_cache = (False, str(exc), None, "")
        verified, reason, manifest, _ = conversion_validation_cache
        if not verified or manifest is None:
            if conversion_manifest.exists():
                eprint(f"[WARN] Step 04 completion manifest rejected: {reason}")
            return None
        alignment = manifest.get("alignment", {})
        if isinstance(alignment, dict) and bool(alignment.get("enabled", False)):
            return sparse_aligned
        return sparse_fallback

    sparse_model_for_ud = _pick_sparse_model_for_ud()
    convert_outputs_ok = sparse_model_for_ud is not None

    if not _in_step_range(4):

        eprint("[SKIP] 04_convert_uavfgs (outside selected step range)")

        _record_step("04_convert_uavfgs", "skip_range", convert_cmd, outputs_ok=convert_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "04_convert_uavfgs", convert_cmd, outputs_ok=convert_outputs_ok, force=args.force)
        if skip:
            _record_step("04_convert_uavfgs", "skip", convert_cmd, outputs_ok=convert_outputs_ok)
        else:
            maybe_run(convert_cmd, cwd=gs_root, step_name="04_convert_uavfgs")
            sparse_model_for_ud = _pick_sparse_model_for_ud(refresh=True)
            convert_outputs_ok = sparse_model_for_ud is not None
            if not convert_outputs_ok:
                _maybe_raise_file_not_found(
                    "Atomic conversion completion manifest is missing or invalid after convert step.\n"
                    f"Manifest: {conversion_manifest}\n"
                    f"Checked aligned: {sparse_aligned}\n"
                    f"Checked fallback: {sparse_fallback}"
                )
            elif sparse_model_for_ud != sparse_aligned:
                eprint(f"[WARN] model alignment was explicitly disabled; using sparse model: {sparse_model_for_ud}")
            _record_step("04_convert_uavfgs", "run", convert_cmd, outputs_ok=convert_outputs_ok)
            write_marker(marker_path(state_dir, "04_convert_uavfgs"), "04_convert_uavfgs", convert_cmd, cwd=gs_root)

    # Optional early stop after COLMAP
    if args.to_step <= 4:
        eprint("[INFO] Step range ends at 04_convert_uavfgs. Done.")
        profile_reason["value"] = "end_at_step4"
        _write_debug_dump("end_at_step4")
        return

    if args.skip_train:
        if args.from_step > 4:
            eprint("[INFO] --skip_train ignored because --from_step > 4 (you selected later steps).")
        else:
            eprint("[INFO] --skip_train set. Stopping after COLMAP.")
            profile_reason["value"] = "skip_train"
            _write_debug_dump("skip_train")
            return

    # -------- 5) Stage-1 training (RGB)
    ensure_dir(model_rgb)
    ckpt_rgb = model_rgb / f"chkpnt{args.rgb_iter}.pth"

    train1_cmd = [
        py, "train.py",
        "-s", str(data_root),
        "--images", "images",
        "-m", str(model_rgb),
        "-r", str(args.rgb_res),
        "--iterations", str(args.rgb_iter),
        "--checkpoint_iterations", str(args.rgb_iter),
        "--data_device", str(args.device),
        "--eval",
        "--densify_from_iter", str(args.rgb_densify_from),
        "--densify_until_iter", str(args.rgb_densify_until),
        "--densification_interval", str(args.rgb_densify_interval),
        "--densify_grad_threshold", str(args.rgb_densify_grad),
        "--lambda_dssim", str(args.rgb_lambda_dssim),
    ]
    _append_explicit_camera_lists(
        train1_cmd, args.train_list, args.test_list,
        args.train_list_sha256, args.test_list_sha256,
    )
    if bool(getattr(args, "baseline_modules_off", False)):
        train1_cmd.append("--baseline_modules_off")
    if args.grouped_ablation_mode == "m00_plus_ssp":
        train1_cmd.append("--baseline_restore_ssp")

    # Forward sparse support opts only when enabled for RGB stage.
    if ss_train_extra:
        train1_cmd.extend(ss_train_extra)
    if args.ss_prune_after_rgb:
        train1_cmd.append("--ss_prune_after_rgb")
    if clamp_effective_rgb is not None:
        train1_cmd.extend(["--clamp_scale_max", str(clamp_effective_rgb), "--clamp_scale_after_densify"])
    if args.clamp_scale_after_rgb_final:
        train1_cmd.append("--clamp_scale_after_rgb_final")
    if args.benchmark_efficiency:
        train1_cmd.extend([
            "--benchmark_efficiency",
            "--efficiency_output", str(train_eff_rgb),
            "--efficiency_stage", "rgb",
        ])

    train1_outputs_ok = ckpt_rgb.exists() and (
        (not args.benchmark_efficiency)
        or _efficiency_sidecar_completed(train_eff_rgb, "training_stage", stage="rgb", iteration=args.rgb_iter)
    )
    if not _in_step_range(5):
        eprint("[SKIP] 05_train_rgb (outside selected step range)")
        _record_step("05_train_rgb", "skip_range", train1_cmd, outputs_ok=train1_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "05_train_rgb", train1_cmd, outputs_ok=train1_outputs_ok, force=args.force)
        if skip:
            _record_step("05_train_rgb", "skip", train1_cmd, outputs_ok=train1_outputs_ok)
        else:
            train1_sidecar_before = _sidecar_fingerprint(train_eff_rgb) if args.benchmark_efficiency else None
            maybe_run(train1_cmd, cwd=gs_root, step_name="05_train_rgb")
            train1_sidecar_fresh = (
                (not args.benchmark_efficiency)
                or (
                    _efficiency_sidecar_completed(train_eff_rgb, "training_stage", stage="rgb", iteration=args.rgb_iter)
                    and _sidecar_fingerprint(train_eff_rgb) != train1_sidecar_before
                )
            )
            train1_outputs_ok = ckpt_rgb.exists() and train1_sidecar_fresh
            if not train1_outputs_ok:
                _maybe_raise_file_not_found(
                    f"RGB checkpoint or fresh completed efficiency sidecar missing after training: {model_rgb}"
                )
            _record_step("05_train_rgb", "run", train1_cmd, outputs_ok=train1_outputs_ok)
            write_marker(marker_path(state_dir, "05_train_rgb"), "05_train_rgb", train1_cmd, cwd=gs_root)

    # Render RGB (keep -r consistent with training to avoid mismatched intrinsics/resolution)
    render1_cmd = [py, "render.py", "-m", str(model_rgb), "-s", str(data_root), "-r", str(args.rgb_res)]
    _append_explicit_camera_lists(
        render1_cmd, args.train_list, args.test_list,
        args.train_list_sha256, args.test_list_sha256,
    )
    if args.benchmark_efficiency:
        render1_cmd.extend([
            "--benchmark_efficiency",
            "--benchmark_warmup_views", str(args.efficiency_render_warmup_views),
            "--benchmark_repeats", str(args.efficiency_render_repeats),
            "--benchmark_output", str(render_eff_rgb),
        ])
    render1_outputs_ok = (model_rgb / "test").exists() and contains_any_file(model_rgb / "test", ("00000.png",), max_depth=5)  # weak check
    # Better: any image in test dir
    if (model_rgb / "test").exists():
        try:
            render1_outputs_ok = any(p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in (model_rgb / "test").rglob("*"))
        except Exception:
            pass
    render1_outputs_ok = render1_outputs_ok and (
        (not args.benchmark_efficiency)
        or _efficiency_sidecar_completed(render_eff_rgb, "render", iteration=args.rgb_iter)
    )

    if not _in_step_range(6):

        eprint("[SKIP] 06_render_rgb (outside selected step range)")

        _record_step("06_render_rgb", "skip_range", render1_cmd, outputs_ok=render1_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "06_render_rgb", render1_cmd, outputs_ok=render1_outputs_ok, force=args.force)
        if skip:
            _record_step("06_render_rgb", "skip", render1_cmd, outputs_ok=render1_outputs_ok)
        else:
            render1_sidecar_before = _sidecar_fingerprint(render_eff_rgb) if args.benchmark_efficiency else None
            maybe_run(render1_cmd, cwd=gs_root, step_name="06_render_rgb")
            render1_outputs_ok = (model_rgb / "test").exists()
            if render1_outputs_ok:
                try:
                    render1_outputs_ok = any(
                        p.suffix.lower() in (".png", ".jpg", ".jpeg")
                        for p in (model_rgb / "test").rglob("*")
                    )
                except Exception:
                    render1_outputs_ok = False
            render1_sidecar_fresh = (
                (not args.benchmark_efficiency)
                or (
                    _efficiency_sidecar_completed(render_eff_rgb, "render", iteration=args.rgb_iter)
                    and _sidecar_fingerprint(render_eff_rgb) != render1_sidecar_before
                )
            )
            render1_outputs_ok = render1_outputs_ok and render1_sidecar_fresh
            if not render1_outputs_ok:
                _maybe_raise_file_not_found(f"RGB render outputs or efficiency sidecar missing under: {model_rgb}")
            _record_step("06_render_rgb", "run", render1_cmd, outputs_ok=render1_outputs_ok)
            write_marker(marker_path(state_dir, "06_render_rgb"), "06_render_rgb", render1_cmd, cwd=gs_root)

    metrics1_cmd = [py, "metrics.py", "-m", str(model_rgb)]
    metrics1_outputs_ok = (model_rgb / "results.json").exists() or (model_rgb / "results.txt").exists()
    if not _in_step_range(7):
        eprint("[SKIP] 07_metrics_rgb (outside selected step range)")
        _record_step("07_metrics_rgb", "skip_range", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "07_metrics_rgb", metrics1_cmd, outputs_ok=metrics1_outputs_ok, force=args.force)
        if skip:
            _record_step("07_metrics_rgb", "skip", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
        else:
            maybe_run(metrics1_cmd, cwd=gs_root, step_name="07_metrics_rgb")
            if args.run_metrics_plus:
                metrics_plus_cmd = [py, "metrics_plus.py", "-m", str(model_rgb),
                                    "--K", str(args.metrics_plus_K), "--bg", str(args.metrics_plus_bg),
                                    "--extra_iqa", str(args.metrics_plus_extra_iqa),
                                    "--extra_iqa_space", str(args.metrics_plus_extra_iqa_space),
                                    "--extra_iqa_device", str(args.metrics_plus_extra_iqa_device),
                                    "--save_json"]
                maybe_run(metrics_plus_cmd, cwd=gs_root, step_name="07_metrics_rgb")
            _record_step("07_metrics_rgb", "run", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
            # even if we can't detect output, write marker so reruns can skip
            write_marker(marker_path(state_dir, "07_metrics_rgb"), "07_metrics_rgb", metrics1_cmd, cwd=gs_root)

    # -------- 6) Undistort thermal using aligned sparse model
    if args.clean_thermal_ud and thermal_ud.exists():
        eprint(f"[INFO] Cleaning existing thermal_UD: {thermal_ud}")
        shutil.rmtree(thermal_ud)

    th_dir_for_ud_effective = th_dir_for_ud
    if _in_step_range(8):
        sparse_model_for_ud = _pick_sparse_model_for_ud()
        model_names = _read_colmap_image_names(sparse_model_for_ud) if sparse_model_for_ud is not None else []
        if model_names:
            thermal_names = {p.name for p in list_images(th_dir_for_ud)}
            missing_exact = [n for n in model_names if n not in thermal_names]
            if missing_exact:
                thermal_stems = {p.stem.lower() for p in list_images(th_dir_for_ud)}
                missing_by_stem = [n for n in missing_exact if Path(n).stem.lower() not in thermal_stems]
                if not missing_by_stem:
                    alias_dir = data_root / "_pipeline_tmp" / "thermal_ud_alias"
                    linked, missing = _build_image_name_alias_dir(th_dir_for_ud, model_names, alias_dir)
                    if missing == 0 and linked > 0:
                        th_dir_for_ud_effective = alias_dir
                        eprint(
                            f"[INFO] Thermal image name alias enabled: src={th_dir_for_ud} -> alias={alias_dir} "
                            f"(linked={linked})"
                        )
                    else:
                        eprint(
                            f"[WARN] Thermal alias build incomplete (linked={linked}, missing={missing}); "
                            f"using original thermal dir: {th_dir_for_ud}"
                        )
                else:
                    eprint(
                        f"[WARN] Thermal names do not match sparse model (missing exact={len(missing_exact)}, "
                        f"missing by stem={len(missing_by_stem)})."
                    )

    undistort_cmd = [
        str(args.colmap), "image_undistorter",
        "--image_path", str(th_dir_for_ud_effective),
        "--input_path", str(sparse_model_for_ud if sparse_model_for_ud is not None else sparse_aligned),
        "--output_path", str(thermal_ud),
        "--output_type", "COLMAP",
    ]
    undistort_outputs_ok = thermal_ud.exists() and (thermal_ud / "images").exists() and (len(list_images(thermal_ud / "images")) > 0) and (thermal_ud / "sparse").exists()
    # Preflight: step 08 requires sparse_aligned from step 04
    if _in_step_range(8):
        sparse_model_for_ud = _pick_sparse_model_for_ud()
        if sparse_model_for_ud is None:
            _maybe_raise_file_not_found(
                "Missing COLMAP sparse model required for thermal undistort.\n"
                f"Checked aligned: {sparse_aligned} (with cameras.bin/txt).\n"
                f"Checked fallback: {sparse_fallback} (with cameras.bin/txt).\n"
                "Run with --from_step 4 (or earlier), or fix your COLMAP outputs."
            )
        if not th_dir_for_ud_effective.exists():
            _maybe_raise_file_not_found(f"Thermal directory not found: {th_dir_for_ud_effective}")

    if not _in_step_range(8):
        eprint("[SKIP] 08_undistort_thermal (outside selected step range)")
        _record_step("08_undistort_thermal", "skip_range", undistort_cmd, outputs_ok=undistort_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "08_undistort_thermal", undistort_cmd, outputs_ok=undistort_outputs_ok, force=args.force)
        if skip:
            _record_step("08_undistort_thermal", "skip", undistort_cmd, outputs_ok=undistort_outputs_ok)
        else:
            maybe_run(undistort_cmd, cwd=gs_root, step_name="08_undistort_thermal")
            undistort_outputs_ok = thermal_ud.exists() and (thermal_ud / "images").exists() and (len(list_images(thermal_ud / "images")) > 0) and (thermal_ud / "sparse").exists()
            if not undistort_outputs_ok:
                _maybe_raise_file_not_found(f"thermal_UD seems incomplete: {thermal_ud}")
            _record_step("08_undistort_thermal", "run", undistort_cmd, outputs_ok=undistort_outputs_ok)
            write_marker(marker_path(state_dir, "08_undistort_thermal"), "08_undistort_thermal", undistort_cmd, cwd=gs_root)

    # -------- 7) Normalize sparse layout for thermal_UD
    sparse_dir_ud = thermal_ud / "sparse"
    norm_cmd = [py, "-c", f"print('ensure_sparse_0: {sparse_dir_ud}')"]  # marker cmd
    norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)

    if not _in_step_range(9):

        eprint("[SKIP] 09_normalize_sparse_ud (outside selected step range)")

        _record_step("09_normalize_sparse_ud", "skip_range", norm_cmd, outputs_ok=norm_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "09_normalize_sparse_ud", norm_cmd, outputs_ok=norm_outputs_ok, force=args.force)
        if skip:
            _record_step("09_normalize_sparse_ud", "skip", norm_cmd, outputs_ok=norm_outputs_ok)
        else:
            t0 = _profile_step_start("09_normalize_sparse_ud", norm_cmd)
            rc = 0
            try:
                ensure_sparse_0(sparse_dir_ud)
            except FileNotFoundError as e:
                rc = 1
                _maybe_raise_file_not_found(str(e))
            finally:
                _profile_step_end("09_normalize_sparse_ud", t0, returncode=rc)
        norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)
        if not norm_outputs_ok:
            _maybe_raise_file_not_found(f"thermal_UD sparse/0 not found or missing cameras.*: {sparse_dir_ud / '0'}")
        _record_step("09_normalize_sparse_ud", "run", norm_cmd, outputs_ok=norm_outputs_ok)
        write_marker(marker_path(state_dir, "09_normalize_sparse_ud"), "09_normalize_sparse_ud", norm_cmd, cwd=gs_root)

    # -------- 8) Stage-2 training (Thermal)
    ensure_dir(model_t)
    ckpt_t = model_t / f"chkpnt{args.t_iter}.pth"
    if not ckpt_rgb.exists():
        _maybe_raise_file_not_found(f"Start checkpoint not found: {ckpt_rgb}")

    train2_cmd = [
        py, "train.py",
        "-s", str(thermal_ud),
        "--images", "images",
        "-m", str(model_t),
        "--start_checkpoint", str(ckpt_rgb),
        "-r", str(args.t_res),
        "--iterations", str(args.t_iter),
        "--checkpoint_iterations", *[str(iteration) for iteration in thermal_checkpoint_iterations],
    ]
    _append_explicit_camera_lists(
        train2_cmd, args.thermal_train_list, args.thermal_test_list,
        args.thermal_train_list_sha256, args.thermal_test_list_sha256,
    )
    if thermal_checkpoint_offsets_applied:
        train2_cmd.extend([
            "--save_iterations",
            *[str(iteration) for iteration in thermal_checkpoint_iterations],
        ])
    if args.thermal_recipe != "legacy":
        train2_cmd.extend(["--thermal_recipe", str(args.thermal_recipe)])
    if formal_stage2_recipe or artifact_save_semantics_explicit:
        train2_cmd.extend([
            "--artifact_save_semantics", str(args.artifact_save_semantics)
        ])
    if args.thermal_freeze_mode != "legacy":
        train2_cmd.extend(["--thermal_freeze_mode", str(args.thermal_freeze_mode)])
    if args.thermal_scale_clamp != "legacy":
        train2_cmd.extend(["--thermal_scale_clamp", str(args.thermal_scale_clamp)])
    if bool(getattr(args, "baseline_modules_off", False)):
        train2_cmd.append("--baseline_modules_off")
    if args.grouped_ablation_mode == "m00_plus_stt":
        train2_cmd.append("--baseline_restore_stt")

    train2_cmd.extend([
        # The resolved protocol always spells out LRs; topology remains fixed.
        "--position_lr_init", thermal_lr_tokens["position_lr_init"],
        "--position_lr_final", thermal_lr_tokens["position_lr_final"],
        "--scaling_lr", thermal_lr_tokens["scaling_lr"],
        "--rotation_lr", thermal_lr_tokens["rotation_lr"],
        "--opacity_lr", thermal_lr_tokens["opacity_lr"],
        "--feature_lr", thermal_lr_tokens["feature_lr"],
        "--densify_from_iter", "999999",
        "--densify_until_iter", "0",
        "--densification_interval", "999999",
        "--opacity_reset_interval", "999999",
        "--lambda_dssim", str(args.t_lambda_dssim),
        "--eval",
    ])

    if ss_train2_extra:
        train2_cmd.extend(ss_train2_extra)
    if args.ss_prune_before_thermal:
        train2_cmd.append("--ss_prune_before_thermal")
    if clamp_effective_t is not None:
        train2_cmd.extend(["--clamp_scale_max", str(clamp_effective_t)])
    if args.thermal_reset_features:
        train2_cmd.append("--thermal_reset_features")
    if args.thermal_max_sh_degree is not None:
        train2_cmd.extend(["--thermal_max_sh_degree", str(args.thermal_max_sh_degree)])
    # Preserve the byte-for-byte legacy command by relying on train.py's
    # restore default unless fresh state is explicitly requested.
    if args.thermal_optimizer_state != "restore":
        train2_cmd.extend(["--thermal_optimizer_state", str(args.thermal_optimizer_state)])
    if args.debug_gaussian_stats:
        train2_cmd.append("--debug_gaussian_stats")
    if args.sgf_disable:
        train2_cmd.append("--sgf_disable")

    train2_cmd.extend(tstruct_train_extra)
    if args.benchmark_efficiency:
        train2_cmd.extend([
            "--benchmark_efficiency",
            "--efficiency_output", str(train_eff_t),
            "--efficiency_stage", "thermal",
        ])
    train2_outputs_ok = ckpt_t.exists() and (
        (not args.benchmark_efficiency)
        or _efficiency_sidecar_completed(train_eff_t, "training_stage", stage="thermal", iteration=args.t_iter)
    )
    # Preflight: step 10 requires stage-1 checkpoint and thermal_UD dataset
    if _in_step_range(10):
        if not ckpt_rgb.exists():
            _maybe_raise_file_not_found(
                f"Stage-1 RGB checkpoint missing: {ckpt_rgb}\n"
                "Run with --from_step 5 (or earlier) to train RGB first, or set --rgb_iter to match an existing checkpoint."
            )
        if not thermal_ud.exists() or not (thermal_ud / "images").exists():
            _maybe_raise_file_not_found(f"thermal_UD dataset missing: {thermal_ud} (need images/). Run step 08 first.")

    if not _in_step_range(10):
        eprint("[SKIP] 10_train_thermal (outside selected step range)")
        _record_step("10_train_thermal", "skip_range", train2_cmd, outputs_ok=train2_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "10_train_thermal", train2_cmd, outputs_ok=train2_outputs_ok, force=args.force)
        if skip:
            _record_step("10_train_thermal", "skip", train2_cmd, outputs_ok=train2_outputs_ok)
        else:
            train2_sidecar_before = _sidecar_fingerprint(train_eff_t) if args.benchmark_efficiency else None
            maybe_run(train2_cmd, cwd=gs_root, step_name="10_train_thermal")
            train2_sidecar_fresh = (
                (not args.benchmark_efficiency)
                or (
                    _efficiency_sidecar_completed(train_eff_t, "training_stage", stage="thermal", iteration=args.t_iter)
                    and _sidecar_fingerprint(train_eff_t) != train2_sidecar_before
                )
            )
            train2_outputs_ok = ckpt_t.exists() and train2_sidecar_fresh
            if not train2_outputs_ok:
                _maybe_raise_file_not_found(
                    f"Thermal checkpoint or fresh completed efficiency sidecar missing after training: {model_t}"
                )
            _record_step("10_train_thermal", "run", train2_cmd, outputs_ok=train2_outputs_ok)
            write_marker(marker_path(state_dir, "10_train_thermal"), "10_train_thermal", train2_cmd, cwd=gs_root)

    render2_cmd = [py, "render.py", "-m", str(model_t), "-s", str(thermal_ud), "-r", str(args.t_res)]
    _append_explicit_camera_lists(
        render2_cmd, args.thermal_train_list, args.thermal_test_list,
        args.thermal_train_list_sha256, args.thermal_test_list_sha256,
    )
    if args.benchmark_efficiency:
        render2_cmd.extend([
            "--benchmark_efficiency",
            "--benchmark_warmup_views", str(args.efficiency_render_warmup_views),
            "--benchmark_repeats", str(args.efficiency_render_repeats),
            "--benchmark_output", str(render_eff_t),
        ])
    render2_outputs_ok = (model_t / "test").exists()
    if (model_t / "test").exists():
        try:
            render2_outputs_ok = any(p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in (model_t / "test").rglob("*"))
        except Exception:
            pass
    render2_outputs_ok = render2_outputs_ok and (
        (not args.benchmark_efficiency)
        or _efficiency_sidecar_completed(render_eff_t, "render", iteration=args.t_iter)
    )

    if not _in_step_range(11):

        eprint("[SKIP] 11_render_thermal (outside selected step range)")

        _record_step("11_render_thermal", "skip_range", render2_cmd, outputs_ok=render2_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "11_render_thermal", render2_cmd, outputs_ok=render2_outputs_ok, force=args.force)
        if skip:
            _record_step("11_render_thermal", "skip", render2_cmd, outputs_ok=render2_outputs_ok)
        else:
            render2_sidecar_before = _sidecar_fingerprint(render_eff_t) if args.benchmark_efficiency else None
            maybe_run(render2_cmd, cwd=gs_root, step_name="11_render_thermal")
            render2_outputs_ok = (model_t / "test").exists()
            if render2_outputs_ok:
                try:
                    render2_outputs_ok = any(
                        p.suffix.lower() in (".png", ".jpg", ".jpeg")
                        for p in (model_t / "test").rglob("*")
                    )
                except Exception:
                    render2_outputs_ok = False
            render2_sidecar_fresh = (
                (not args.benchmark_efficiency)
                or (
                    _efficiency_sidecar_completed(render_eff_t, "render", iteration=args.t_iter)
                    and _sidecar_fingerprint(render_eff_t) != render2_sidecar_before
                )
            )
            render2_outputs_ok = render2_outputs_ok and render2_sidecar_fresh
            if not render2_outputs_ok:
                _maybe_raise_file_not_found(f"Thermal render outputs or efficiency sidecar missing under: {model_t}")
            _record_step("11_render_thermal", "run", render2_cmd, outputs_ok=render2_outputs_ok)
            write_marker(marker_path(state_dir, "11_render_thermal"), "11_render_thermal", render2_cmd, cwd=gs_root)

    metrics2_cmd = [py, "metrics.py", "-m", str(model_t)]
    metrics2_outputs_ok = (model_t / "results.json").exists() or (model_t / "results.txt").exists()
    if args.save_cmds:
        def _format_cmd_for_save(cmd: List[str]) -> str:
            cmd2 = _normalize_cmd_for_windows(cmd)
            return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])

        def _save_cmd(path: Path, cmd: List[str]) -> None:
            path.write_text(_format_cmd_for_save(cmd) + "\n", encoding="utf-8")

        render_cmd = render2_cmd if _in_step_range(11) else render1_cmd
        metrics_cmd = metrics2_cmd if _in_step_range(12) else metrics1_cmd
        _save_cmd(out_root / "cmd_train1.txt", train1_cmd)
        _save_cmd(out_root / "cmd_train2.txt", train2_cmd)
        _save_cmd(out_root / "cmd_render.txt", render_cmd)
        _save_cmd(out_root / "cmd_metrics.txt", metrics_cmd)
        eprint(f"[INFO] SavedCmds: {out_root}")
    if not _in_step_range(12):
        eprint("[SKIP] 12_metrics_thermal (outside selected step range)")
        _record_step("12_metrics_thermal", "skip_range", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "12_metrics_thermal", metrics2_cmd, outputs_ok=metrics2_outputs_ok, force=args.force)
        if skip:
            _record_step("12_metrics_thermal", "skip", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
        else:
            maybe_run(metrics2_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            if args.run_metrics_plus:
                metrics_plus_cmd = [py, "metrics_plus.py", "-m", str(model_t),
                                    "--K", str(args.metrics_plus_K), "--bg", str(args.metrics_plus_bg),
                                    "--extra_iqa", str(args.metrics_plus_extra_iqa),
                                    "--extra_iqa_space", str(args.metrics_plus_extra_iqa_space),
                                    "--extra_iqa_device", str(args.metrics_plus_extra_iqa_device),
                                    "--save_json"]
                maybe_run(metrics_plus_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            if args.run_novel_view_metrics:
                novel_cmd = [py, "novel_view_metrics.py", "-m", str(model_t),
                             "--mode", str(args.novel_view_mode),
                             "--N", str(args.novel_view_N), "--bg", str(args.novel_bg),
                             "--device", str(args.novel_view_device)]
                if str(args.novel_view_mode) == "grid72":
                    novel_cmd.extend([
                        "--grid_azimuth_count", str(args.novel_grid_azimuth_count),
                        "--grid_pitch_list", str(args.novel_grid_pitch_list),
                        "--grid_distance_factors", str(args.novel_grid_distance_factors),
                    ])
                    if args.novel_grid_no_topdown:
                        novel_cmd.append("--grid_no_topdown")
                if bool(getattr(args, "novel_dump_ellipsoid_proxy", False)):
                    novel_cmd.append("--dump_ellipsoid_proxy")
                    if str(getattr(args, "novel_ellipsoid_proxy_dir", "")).strip():
                        novel_cmd.extend(["--ellipsoid_proxy_dir", str(args.novel_ellipsoid_proxy_dir)])
                sibr_ellipsoid_enabled = bool(getattr(args, "novel_dump_sibr_ellipsoid", False))
                # Always pass explicit on/off to avoid novel_view_metrics.py default overriding pipeline intent.
                novel_cmd.extend(["--dump_sibr_ellipsoid", "true" if sibr_ellipsoid_enabled else "false"])
                if sibr_ellipsoid_enabled:
                    if str(getattr(args, "novel_sibr_exe", "")).strip():
                        novel_cmd.extend(["--sibr_exe", str(args.novel_sibr_exe)])
                    if str(getattr(args, "novel_sibr_out_dir", "")).strip():
                        novel_cmd.extend(["--sibr_out_dir", str(args.novel_sibr_out_dir)])
                    novel_cmd.extend(["--sibr_device", str(int(getattr(args, "novel_sibr_device", 0)))])
                    novel_cmd.extend(["--sibr_gaussian_mode", str(getattr(args, "novel_sibr_mode", "ellipsoids"))])
                    if bool(getattr(args, "novel_sibr_keep_path_file", False)):
                        novel_cmd.append("--sibr_keep_path_file")
                maybe_run(novel_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            _record_step("12_metrics_thermal", "run", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
            write_marker(marker_path(state_dir, "12_metrics_thermal"), "12_metrics_thermal", metrics2_cmd, cwd=gs_root)

    # Optional early stop after stage-2
    if args.to_step <= 12:
        eprint("[INFO] Step range ends at 12_metrics_thermal. Done.")
        profile_reason["value"] = "end_at_step12"
        _write_debug_dump("end_at_step12")
        return

    if args.skip_blend:
        if args.from_step > 12:
            eprint("[INFO] --skip_blend ignored because --from_step > 12 (you selected later steps).")
        else:
            eprint("[INFO] --skip_blend set. Stopping after stage-2 training.")
            profile_reason["value"] = "skip_blend"
            _write_debug_dump("skip_blend")
            return

    # -------- 9) Blend models
    if args.clean_blend_out and model_f.exists():
        eprint(f"[INFO] Cleaning existing blend output: {model_f}")
        shutil.rmtree(model_f)

    blend_cmd = [
        py, "blend_model_strict_endpoints.py",
        "--rgb_model_dir", str(model_rgb), "--rgb_iter", str(args.rgb_iter),
        "--t_model_dir", str(model_t), "--t_iter", str(args.t_iter),
        "--alphas", str(args.alphas),
        "--out_root", str(model_f),
        "--out_iter", str(args.t_iter),
        "--dc_y_from", str(args.blend_dc_y_from),
        "--methods",
    ] + list(args.methods)

    if args.verify_endpoints:
        blend_cmd.append("--verify_endpoints")
    # only add clean_out when explicitly requested (so resume works)
    if args.clean_blend_out:
        blend_cmd.append("--clean_out")

    blend_outputs_ok = model_f.exists() and any(p.is_dir() for p in model_f.iterdir())
    # Preflight: step 13 requires RGB/T trained point clouds at requested iterations
    if _in_step_range(13):
        rgb_ply = model_rgb / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
        t_ply = model_t / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
        if not rgb_ply.exists():
            _maybe_raise_file_not_found(
                f"RGB point cloud missing for blend: {rgb_ply}\n"
                "Run step 05 (RGB train) first, or set --rgb_iter to an existing iteration."
            )
        if not t_ply.exists():
            _maybe_raise_file_not_found(
                f"Thermal point cloud missing for blend: {t_ply}\n"
                "Run step 10 (thermal train) first, or set --t_iter to an existing iteration."
            )

    if not _in_step_range(13):
        eprint("[SKIP] 13_blend (outside selected step range)")
        _record_step("13_blend", "skip_range", blend_cmd, outputs_ok=blend_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "13_blend", blend_cmd, outputs_ok=blend_outputs_ok, force=args.force)
        if skip:
            _record_step("13_blend", "skip", blend_cmd, outputs_ok=blend_outputs_ok)
        else:
            maybe_run(blend_cmd, cwd=gs_root, step_name="13_blend")
            blend_outputs_ok = model_f.exists() and any(p.is_dir() for p in model_f.iterdir())
            if not blend_outputs_ok:
                _maybe_raise_file_not_found(f"Blend output looks empty: {model_f}")
            _record_step("13_blend", "run", blend_cmd, outputs_ok=blend_outputs_ok)
            write_marker(marker_path(state_dir, "13_blend"), "13_blend", blend_cmd, cwd=gs_root)

    # -------- 10) Evaluate sweep
    ensure_dir(eval_out)
    sweep_cmd = [
        py, "eval_blend_sweep.py",
        "--sweep_root", str(model_f),
        "--rgb_render", str(model_rgb),
        "--t_render", str(model_t),
        "--out_dir", str(eval_out),
        "--thermal_scalar", str(args.eval_thermal_scalar),
        "--thermal_align", str(args.eval_thermal_align),
        "--sample_frames", str(args.eval_sample_frames),
        "--seed", str(args.eval_seed),
        "--montage_cols", str(args.eval_montage_cols),
        "--montage_samples", str(args.eval_montage_samples),
        "--gt_mode", str(args.eval_gt_mode),
    ]
    if args.auto_render:
        sweep_cmd.append("--auto_render")
    if args.eval_no_montage:
        sweep_cmd.append("--no_montage")
    if args.eval_render_iter is not None:
        sweep_cmd += ["--render_iter", str(args.eval_render_iter)]
    if args.eval_render_source:
        sweep_cmd += ["--render_source", str(args.eval_render_source)]
    if args.eval_render_images:
        sweep_cmd += ["--render_images", str(args.eval_render_images)]
    if args.eval_render_resolution is not None:
        sweep_cmd += ["--render_resolution", str(args.eval_render_resolution)]
    if args.eval_render_extra:
        sweep_cmd += ["--render_extra", str(args.eval_render_extra)]

    sweep_outputs_ok = (eval_out / "summary.csv").exists() and (eval_out / "summary.csv").stat().st_size > 50
    if not _in_step_range(14):
        eprint("[SKIP] 14_eval_sweep (outside selected step range)")
        _record_step("14_eval_sweep", "skip_range", sweep_cmd, outputs_ok=sweep_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "14_eval_sweep", sweep_cmd, outputs_ok=sweep_outputs_ok, force=args.force)
        if skip:
            _record_step("14_eval_sweep", "skip", sweep_cmd, outputs_ok=sweep_outputs_ok)
        else:
            maybe_run(sweep_cmd, cwd=gs_root, step_name="14_eval_sweep")
            sweep_outputs_ok = (eval_out / "summary.csv").exists() and (eval_out / "summary.csv").stat().st_size > 50
            if not sweep_outputs_ok:
                eprint("[WARN] eval_blend_sweep finished but summary.csv not found; please check logs.")
            _record_step("14_eval_sweep", "run", sweep_cmd, outputs_ok=sweep_outputs_ok)
            write_marker(marker_path(state_dir, "14_eval_sweep"), "14_eval_sweep", sweep_cmd, cwd=gs_root)

    profile_reason["value"] = "done"
    _write_debug_dump("done")
    eprint("\n[DONE] Full pipeline finished.")


if __name__ == "__main__":
    main()
