from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_images_binary, read_images_text

from depth_reference_common import (
    build_probe_view_manifest,
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    load_json,
    load_ply_mesh,
    load_ply_points_xyz,
    parse_thresholds_m,
    render_mesh_depth_for_view,
    render_support_count_for_view,
    save_json,
)


OPENMVS_CUDA_NEGATIVE_PATTERNS = (
    r"fall(?:ing)?\s+back\s+to\s+cpu",
    r"\bcuda\s+error\b",
    r"cuda[^\r\n]*(?:failed|unavailable)",
    r"cpu[- ]only",
)
OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER = (
    "CUDA mesh refinement path completed; CPU fallback disabled"
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build training-only OpenMVS reference-mesh depth artifacts for "
            "held-out geometry diagnostics"
        )
    )
    parser.add_argument("--strict_protocol_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--openmvs_interface_colmap_cmd", default="InterfaceCOLMAP")
    parser.add_argument("--openmvs_densify_cmd", default="DensifyPointCloud")
    parser.add_argument("--openmvs_reconstruct_mesh_cmd", default="ReconstructMesh")
    parser.add_argument("--openmvs_refine_mesh_cmd", default="RefineMesh")
    parser.add_argument("--openmvs_cuda_device", type=int, default=0)
    parser.add_argument("--openmvs_resolution_level", type=int, default=1)
    parser.add_argument("--openmvs_max_resolution", type=int, default=2000)
    parser.add_argument("--openmvs_min_resolution", type=int, default=640)
    parser.add_argument("--openmvs_number_views", type=int, default=8)
    parser.add_argument("--openmvs_number_views_fuse", type=int, default=3)
    parser.add_argument("--openmvs_iterations", type=int, default=4)
    parser.add_argument("--openmvs_refine_resolution_level", type=int, default=1)
    parser.add_argument("--openmvs_refine_scales", type=int, default=2)
    parser.add_argument(
        "--skip_openmvs_refine_mesh",
        action="store_true",
        help=(
            "Use ReconstructMesh output directly. RefineMesh is enabled by default "
            "and is never skipped implicitly."
        ),
    )
    parser.add_argument("--resolution_arg", type=int, default=4)
    parser.add_argument("--thresholds_m", default="0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00")
    parser.add_argument("--bbox_lower_quantile", type=float, default=0.01)
    parser.add_argument("--bbox_upper_quantile", type=float, default=0.99)
    parser.add_argument("--bbox_padding_ratio", type=float, default=0.02)
    parser.add_argument("--support_min_count", type=int, default=1)
    parser.add_argument("--support_radius_px", type=int, default=1)
    parser.add_argument("--support_depth_tolerance_m", type=float, default=0.10)
    parser.add_argument("--force_openmvs_interface", action="store_true")
    parser.add_argument("--force_openmvs_densify", action="store_true")
    parser.add_argument("--force_openmvs_mesh", action="store_true")
    parser.add_argument("--force_openmvs_refine", action="store_true")
    parser.add_argument("--force_views", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate source inputs and OpenMVS executables, then print the exact "
            "command plan without creating or modifying artifacts."
        ),
    )
    return parser


def _is_reparse_point(path: Path) -> bool:
    try:
        attrs = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _remove_tree_or_link(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    if _is_reparse_point(path):
        completed = subprocess.run(
            ["cmd", "/c", "rmdir", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 and path.exists():
            raise RuntimeError(f"Failed to remove junction {path}: {completed.stdout}\n{completed.stderr}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _ensure_dir_link(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        try:
            if link_path.resolve() == target_path.resolve():
                return
        except OSError:
            pass
        raise RuntimeError(f"Refusing to replace existing OpenMVS input path: {link_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        link_path.symlink_to(target_path, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 and not link_path.exists():
        raise RuntimeError(
            f"Failed to create junction {link_path} -> {target_path}: "
            f"{completed.stdout}\n{completed.stderr}"
        )


def _validate_colmap_source_workspace(source_workspace_root: Path) -> tuple[Path, Path, bool]:
    images_src = source_workspace_root / "images"
    sparse_src = source_workspace_root / "sparse" / "0"
    if not images_src.is_dir() or not sparse_src.is_dir():
        raise FileNotFoundError(
            "Expected the training-only COLMAP source to contain images/ and sparse/0; "
            f"got images={images_src.is_dir()} sparse0={sparse_src.is_dir()}"
        )
    image_count = len([path for path in images_src.iterdir() if path.is_file()])
    if image_count <= 0:
        raise RuntimeError(f"Training-only COLMAP image directory is empty: {images_src}")
    binary_names = ("cameras.bin", "images.bin", "points3D.bin")
    text_names = ("cameras.txt", "images.txt", "points3D.txt")
    has_binary_model = all((sparse_src / name).is_file() for name in binary_names)
    has_text_model = all((sparse_src / name).is_file() for name in text_names)
    if not (has_binary_model or has_text_model):
        raise FileNotFoundError(
            "OpenMVS InterfaceCOLMAP requires a complete COLMAP model in sparse/0 "
            f"(BIN or TXT): {sparse_src}"
        )
    return images_src, sparse_src, has_binary_model


def _load_name_list(path: Path) -> List[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Image-name list is empty: {path}")
    return names


def _normalized_image_name(name: str) -> str:
    return Path(str(name).replace("\\", "/")).as_posix().lstrip("./").casefold()


def _partition_stem_map(path: Path, label: str) -> Dict[str, str]:
    by_stem: Dict[str, str] = {}
    for name in _load_name_list(path):
        normalized = _normalized_image_name(name)
        suffix = Path(normalized).suffix
        if not suffix:
            raise ValueError(f"{label} entry has no image extension: {name!r}")
        stem = normalized[: -len(suffix)]
        if stem in by_stem:
            raise ValueError(
                f"{label} contains duplicate image stems: "
                f"{by_stem[stem]!r} and {name!r}"
            )
        by_stem[stem] = normalized
    return by_stem


def _validate_probe_camera_partition_stems(
    reference_train_list: Path,
    reference_probe_exclusion_list: Path,
    probe_camera_train_list: Path,
    probe_camera_test_list: Path,
) -> Dict[str, Any]:
    reference_train = _partition_stem_map(reference_train_list, "reference train list")
    reference_probe = _partition_stem_map(
        reference_probe_exclusion_list, "reference probe-exclusion list"
    )
    camera_train = _partition_stem_map(probe_camera_train_list, "probe-camera train list")
    camera_test = _partition_stem_map(probe_camera_test_list, "probe-camera test list")

    for partition, reference, camera in (
        ("train", reference_train, camera_train),
        ("probe", reference_probe, camera_test),
    ):
        missing_camera = sorted(reference.keys() - camera.keys())
        extra_camera = sorted(camera.keys() - reference.keys())
        if missing_camera or extra_camera:
            raise RuntimeError(
                "RGB reference and probe-camera partitions differ beyond image "
                f"extensions for {partition}: missing_camera_stems={missing_camera[:10]} "
                f"extra_camera_stems={extra_camera[:10]}"
            )

    reference_overlap = sorted(reference_train.keys() & reference_probe.keys())
    camera_overlap = sorted(camera_train.keys() & camera_test.keys())
    if reference_overlap or camera_overlap:
        raise RuntimeError(
            "Train/probe stem partitions overlap: "
            f"reference={reference_overlap[:10]} probe_camera={camera_overlap[:10]}"
        )

    return {
        "status": "passed",
        "comparison_rule": "normalized_relative_path_without_final_extension",
        "train_stem_count": len(reference_train),
        "probe_stem_count": len(reference_probe),
        "reference_train_extensions": sorted(
            {Path(name).suffix for name in reference_train.values()}
        ),
        "reference_probe_extensions": sorted(
            {Path(name).suffix for name in reference_probe.values()}
        ),
        "probe_camera_train_extensions": sorted(
            {Path(name).suffix for name in camera_train.values()}
        ),
        "probe_camera_test_extensions": sorted(
            {Path(name).suffix for name in camera_test.values()}
        ),
    }


def _resolve_reference_protocol_paths(strict: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = strict["artifacts"]
    lists = strict["lists"]
    return {
        "workspace_root": Path(artifacts["train_union_source_root"]).resolve(),
        "strict_thermal_root": Path(artifacts["strict_thermal_root"]).resolve(),
        "probe_camera_root": Path(
            artifacts.get("probe_camera_root", artifacts["strict_thermal_root"])
        ).resolve(),
        "train_union_list": Path(lists["train_union"]).resolve(),
        "probe_camera_train_list": Path(
            lists.get("probe_camera_train", lists["train_union"])
        ).resolve(),
        "probe_list": Path(lists["probe_test"]).resolve(),
        "reference_probe_exclusion_list": Path(
            lists.get("reference_probe_exclusion", lists["probe_test"])
        ).resolve(),
        "extended_probe_camera_interface": (
            "probe_camera_root" in artifacts
            or "probe_camera_train" in lists
            or "reference_probe_exclusion" in lists
        ),
    }


def _validate_training_only_partition(
    source_workspace_root: Path,
    train_union_list: Path,
    probe_list: Path,
) -> Dict[str, int]:
    images_src, sparse_src, has_binary_model = _validate_colmap_source_workspace(source_workspace_root)
    train_names = {_normalized_image_name(name) for name in _load_name_list(train_union_list)}
    probe_names = {_normalized_image_name(name) for name in _load_name_list(probe_list)}
    overlap = sorted(train_names & probe_names)
    if overlap:
        raise RuntimeError(f"Train/probe lists overlap; refusing reference construction: {overlap[:10]}")
    source_names = {
        _normalized_image_name(path.relative_to(images_src).as_posix())
        for path in images_src.rglob("*")
        if path.is_file()
    }
    missing_train = sorted(train_names - source_names)
    nontraining_source = sorted(source_names - train_names)
    if missing_train or nontraining_source:
        raise RuntimeError(
            "Training-only OpenMVS image workspace does not exactly match train_union: "
            f"missing_train={missing_train[:10]} nontraining_source={nontraining_source[:10]}"
        )
    if source_names & probe_names:
        raise RuntimeError("Probe images are present in the OpenMVS source workspace")
    model_images = (
        read_images_binary(str(sparse_src / "images.bin"))
        if has_binary_model
        else read_images_text(str(sparse_src / "images.txt"))
    )
    model_names = {_normalized_image_name(image.name) for image in model_images.values()}
    if not model_names:
        raise RuntimeError("Training-only OpenMVS sparse model has no registered images")
    nontraining_model_names = sorted(model_names - train_names)
    if nontraining_model_names:
        raise RuntimeError(
            "OpenMVS sparse model contains cameras outside train_union; refusing "
            f"reference construction: {nontraining_model_names[:10]}"
        )
    if model_names & probe_names:
        raise RuntimeError("Probe cameras are present in the OpenMVS sparse model")
    return {
        "train_union_count": len(train_names),
        "probe_count": len(probe_names),
        "openmvs_source_image_count": len(source_names),
        "openmvs_sparse_registered_count": len(model_names),
        "unregistered_train_count": len(train_names - model_names),
    }


def _prepare_openmvs_input_workspace(source_workspace_root: Path, prepared_root: Path) -> Path:
    images_src, sparse_src, has_binary_model = _validate_colmap_source_workspace(source_workspace_root)
    prepared_root.mkdir(parents=True, exist_ok=True)
    _ensure_dir_link(prepared_root / "images", images_src)
    sparse_dst = prepared_root / "sparse"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    selected_suffix = ".bin" if has_binary_model else ".txt"
    selected_names = {f"cameras{selected_suffix}", f"images{selected_suffix}", f"points3D{selected_suffix}"}
    for name in selected_names:
        src_file = sparse_src / name
        dst_file = sparse_dst / name
        needs_copy = (
            not dst_file.exists()
            or src_file.stat().st_size != dst_file.stat().st_size
            or src_file.stat().st_mtime > dst_file.stat().st_mtime
            or _sha256_file(src_file) != _sha256_file(dst_file)
        )
        if needs_copy:
            shutil.copy2(src_file, dst_file)
    for stale_suffix in ({".txt"} if has_binary_model else {".bin"}):
        for stem in ("cameras", "images", "points3D"):
            stale_file = sparse_dst / f"{stem}{stale_suffix}"
            if stale_file.exists():
                stale_file.unlink()
    return prepared_root


def _has_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_workspace_fingerprint(source_workspace_root: Path) -> Dict[str, Any]:
    images_src, sparse_src, has_binary_model = _validate_colmap_source_workspace(source_workspace_root)
    image_records = []
    for image_path in sorted(path for path in images_src.rglob("*") if path.is_file()):
        stat_result = image_path.stat()
        image_records.append(
            {
                "relative_path": image_path.relative_to(images_src).as_posix(),
                "size_bytes": int(stat_result.st_size),
                "mtime_ns": int(stat_result.st_mtime_ns),
                "sha256": _sha256_file(image_path),
            }
        )
    image_inventory_bytes = json.dumps(
        image_records,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    suffix = ".bin" if has_binary_model else ".txt"
    sparse_hashes = {
        f"{stem}{suffix}": _sha256_file(sparse_src / f"{stem}{suffix}")
        for stem in ("cameras", "images", "points3D")
    }
    return {
        "image_count": len(image_records),
        "image_inventory_sha256": hashlib.sha256(image_inventory_bytes).hexdigest(),
        "sparse_model_format": "bin" if has_binary_model else "txt",
        "sparse_model_sha256": sparse_hashes,
    }


def _resolve_executable(command: str) -> str:
    path = Path(command).expanduser()
    if path.is_file():
        if os.name != "nt" and not os.access(path, os.X_OK):
            raise PermissionError(f"OpenMVS command is not executable: {path}")
        return str(path.resolve())
    resolved = shutil.which(command)
    if resolved is None:
        raise FileNotFoundError(
            f"Required OpenMVS executable not found: {command!r}. "
            "Install a CUDA-enabled OpenMVS build or pass its full executable path."
        )
    return str(Path(resolved).resolve())


def _binary_contains_marker(path: Path, marker: str) -> bool:
    needle = marker.encode("utf-8")
    carry = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            data = carry + chunk
            if needle in data:
                return True
            carry = data[-max(0, len(needle) - 1) :]
    return False


def _resolve_openmvs_executables(args: argparse.Namespace) -> Dict[str, str]:
    executables = {
        "interface_colmap": _resolve_executable(args.openmvs_interface_colmap_cmd),
        "densify": _resolve_executable(args.openmvs_densify_cmd),
        "reconstruct_mesh": _resolve_executable(args.openmvs_reconstruct_mesh_cmd),
    }
    if args.skip_openmvs_refine_mesh:
        executables["refine_mesh"] = str(args.openmvs_refine_mesh_cmd)
    else:
        executables["refine_mesh"] = _resolve_executable(args.openmvs_refine_mesh_cmd)
        refine_path = Path(executables["refine_mesh"])
        if not _binary_contains_marker(refine_path, OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER):
            raise RuntimeError(
                "RefineMesh lacks the required CUDA fail-closed marker. Build OpenMVS 2.4.0 "
                "with tools/geometric_repeatability/openmvs-2.4.0-refine-cuda-fail-closed.patch; "
                "the stock binary may silently fall back to CPU after CUDA initialization."
            )
    return executables


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "openmvs_max_resolution",
        "openmvs_min_resolution",
        "openmvs_number_views",
        "openmvs_number_views_fuse",
        "openmvs_iterations",
        "openmvs_refine_scales",
        "resolution_arg",
        "support_min_count",
    )
    for field in positive_fields:
        value = int(getattr(args, field))
        if value <= 0:
            raise ValueError(f"{field} must be positive, got {value}")
    nonnegative_fields = (
        "openmvs_resolution_level",
        "openmvs_refine_resolution_level",
        "support_radius_px",
    )
    for field in nonnegative_fields:
        value = int(getattr(args, field))
        if value < 0:
            raise ValueError(f"{field} must be non-negative, got {value}")
    if int(args.openmvs_min_resolution) > int(args.openmvs_max_resolution):
        raise ValueError("openmvs_min_resolution cannot exceed openmvs_max_resolution")
    if not (0.0 <= float(args.bbox_lower_quantile) < float(args.bbox_upper_quantile) <= 1.0):
        raise ValueError("bbox quantiles must satisfy 0 <= lower < upper <= 1")
    if float(args.support_depth_tolerance_m) <= 0.0:
        raise ValueError("support_depth_tolerance_m must be positive")
    if int(args.openmvs_cuda_device) < 0:
        raise ValueError(
            "openmvs_cuda_device must be an explicit non-negative CUDA device index; "
            "CPU (-2) and automatic/fallback-capable (-1) modes are forbidden"
        )


def _openmvs_paths(out_dir: Path) -> Dict[str, Path]:
    workspace = out_dir / "_openmvs_workspace"
    return {
        "input_workspace": out_dir / "_openmvs_input",
        "workspace": workspace,
        "interface_mvs": workspace / "scene.mvs",
        "dense_mvs": workspace / "reference_openmvs_dense.mvs",
        "dense_ply": workspace / "reference_openmvs_dense.ply",
        "mesh_ply": workspace / "reference_openmvs_mesh.ply",
        "refined_ply": workspace / "reference_openmvs_mesh_refined.ply",
    }


def _build_openmvs_command_plan(
    args: argparse.Namespace,
    *,
    paths: Dict[str, Path],
    executables: Dict[str, str],
    colmap_binary_model: bool,
) -> List[Dict[str, Any]]:
    workspace = paths["workspace"]
    archive_args = ["--archive-type", "-1"]
    return [
        {
            "stage": "interface_colmap",
            "enabled": True,
            "command": [
                executables["interface_colmap"],
                "--input-file",
                str(paths["input_workspace"]),
                "--output-file",
                str(paths["interface_mvs"]),
                "--working-folder",
                str(workspace),
                "--image-folder",
                "images",
                "--binary",
                "1" if colmap_binary_model else "0",
                "--normalize",
                "0",
                *archive_args,
            ],
            "required_outputs": [str(paths["interface_mvs"])],
            "cuda_evidence_device": None,
        },
        {
            "stage": "densify_point_cloud",
            "enabled": True,
            "command": [
                executables["densify"],
                "--input-file",
                str(paths["interface_mvs"]),
                "--output-file",
                str(paths["dense_mvs"]),
                "--working-folder",
                str(workspace),
                "--resolution-level",
                str(int(args.openmvs_resolution_level)),
                "--max-resolution",
                str(int(args.openmvs_max_resolution)),
                "--min-resolution",
                str(int(args.openmvs_min_resolution)),
                "--number-views",
                str(int(args.openmvs_number_views)),
                "--number-views-fuse",
                str(int(args.openmvs_number_views_fuse)),
                "--iters",
                str(int(args.openmvs_iterations)),
                "--estimate-roi",
                "0",
                "--crop-to-roi",
                "0",
                "--cuda-device",
                str(int(args.openmvs_cuda_device)),
                *archive_args,
            ],
            "required_outputs": [str(paths["dense_mvs"]), str(paths["dense_ply"])],
            "cuda_evidence_device": int(args.openmvs_cuda_device),
            "cuda_evidence_scope": (
                "CUDA PatchMatch depth-map estimation; process-level fail-closed guard "
                "streams CUDA errors and terminates before CPU fallback"
            ),
            "cuda_fallback_fail_closed": True,
            # OpenMVS 2.4 may reuse depth*.dmap files from its working folder.
            # They are disposable caches, not evidence-bearing outputs, and
            # must never survive a failed/invalidated densification attempt.
            "cache_cleanup_root": str(workspace),
            "cache_cleanup_globs": ["*.dmap"],
        },
        {
            "stage": "reconstruct_mesh",
            "enabled": True,
            "command": [
                executables["reconstruct_mesh"],
                "--input-file",
                str(paths["dense_mvs"]),
                "--output-file",
                str(paths["mesh_ply"]),
                "--working-folder",
                str(workspace),
                "--crop-to-roi",
                "0",
                "--cuda-device",
                str(int(args.openmvs_cuda_device)),
                *archive_args,
            ],
            "required_outputs": [str(paths["mesh_ply"])],
            "cuda_evidence_device": int(args.openmvs_cuda_device),
            "cuda_evidence_scope": (
                "CUDA-enabled available kernels; OpenMVS Delaunay/graph-cut reconstruction "
                "contains CPU work by algorithm design"
            ),
            "cuda_fallback_fail_closed": False,
        },
        {
            "stage": "refine_mesh",
            "enabled": not bool(args.skip_openmvs_refine_mesh),
            "command": [
                executables["refine_mesh"],
                "--input-file",
                str(paths["dense_mvs"]),
                "--mesh-file",
                str(paths["mesh_ply"]),
                "--output-file",
                str(paths["refined_ply"]),
                "--working-folder",
                str(workspace),
                "--resolution-level",
                str(int(args.openmvs_refine_resolution_level)),
                "--scales",
                str(int(args.openmvs_refine_scales)),
                "--cuda-device",
                str(int(args.openmvs_cuda_device)),
                *archive_args,
            ],
            "required_outputs": [str(paths["refined_ply"])],
            "cuda_evidence_device": int(args.openmvs_cuda_device),
            "cuda_evidence_scope": "fail-closed CUDA photometric mesh refinement",
            "cuda_fallback_fail_closed": True,
            "cuda_required_log_patterns": [
                re.escape(OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER),
            ],
        },
    ]


def _validate_openmvs_cuda_log(stage: Dict[str, Any], log_text: str, *, log_path: Path) -> None:
    device = stage.get("cuda_evidence_device")
    if device is None:
        return
    device = int(device)
    for pattern in OPENMVS_CUDA_NEGATIVE_PATTERNS:
        if re.search(pattern, log_text, flags=re.IGNORECASE):
            raise RuntimeError(
                f"OpenMVS stage {stage['stage']} reported CUDA failure/CPU fallback; see {log_path}"
            )
    positive_pattern = rf"CUDA\s+device\s+{device}\s+initialized\s*:"
    if re.search(positive_pattern, log_text, flags=re.IGNORECASE) is None:
        raise RuntimeError(
            f"OpenMVS stage {stage['stage']} has no proof that CUDA device {device} initialized; "
            f"see {log_path}"
        )
    for required_pattern in stage.get("cuda_required_log_patterns", []):
        if re.search(str(required_pattern), log_text, flags=re.IGNORECASE) is None:
            raise RuntimeError(
                f"OpenMVS stage {stage['stage']} is missing required fail-closed CUDA completion "
                f"evidence {required_pattern!r}; see {log_path}"
            )


def _terminate_process_group(process: subprocess.Popen, timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, timeout_s),
            )
        except Exception:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        process.terminate()
    deadline = time.monotonic() + max(0.1, timeout_s)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()


def _run_openmvs_stage(stage: Dict[str, Any], *, cwd: Path, log_path: Path) -> None:
    cmd = [str(value) for value in stage["command"]]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process: subprocess.Popen | None = None
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
            log_file.write("COMMAND " + json.dumps(cmd, ensure_ascii=False) + "\n")
            log_file.flush()
            popen_kwargs: Dict[str, Any] = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                **popen_kwargs,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                if stage.get("cuda_evidence_device") is not None and any(
                    re.search(pattern, line, flags=re.IGNORECASE)
                    for pattern in OPENMVS_CUDA_NEGATIVE_PATTERNS
                ):
                    _terminate_process_group(process)
                    raise RuntimeError(
                        f"OpenMVS stage {stage['stage']} reported CUDA failure/CPU fallback "
                        f"and was terminated immediately; see {log_path}"
                    )
            process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"OpenMVS stage {stage['stage']} failed with exit code {returncode}; see {log_path}"
            )
        _validate_openmvs_cuda_log(
            stage,
            log_path.read_text(encoding="utf-8", errors="replace"),
            log_path=log_path,
        )
        missing = [path for path in stage["required_outputs"] if not _has_nonempty_file(Path(path))]
        if missing:
            raise RuntimeError(
                f"OpenMVS stage {stage['stage']} reported success but required outputs are missing/empty: {missing}"
            )
    except BaseException:
        if process is not None and process.poll() is None:
            _terminate_process_group(process)
        _remove_outputs(stage)
        raise


def _remove_outputs(stage: Dict[str, Any]) -> None:
    for output in stage["required_outputs"]:
        _remove_tree_or_link(Path(output))
    cache_root_value = str(stage.get("cache_cleanup_root", "")).strip()
    if cache_root_value:
        cache_root = Path(cache_root_value)
        if cache_root.is_dir():
            for pattern in stage.get("cache_cleanup_globs", []):
                for cache_path in cache_root.rglob(str(pattern)):
                    _remove_tree_or_link(cache_path)


def _stage_receipt_path(out_dir: Path, stage: Dict[str, Any]) -> Path:
    return out_dir / "_openmvs_state" / f"{stage['stage']}.success.json"


def _stage_contract_sha256(stage: Dict[str, Any]) -> str:
    return _canonical_sha256(
        {
            "stage": str(stage["stage"]),
            "enabled": bool(stage["enabled"]),
            "command": [str(value) for value in stage["command"]],
            "required_outputs": [str(value) for value in stage["required_outputs"]],
            "cuda_evidence_device": stage.get("cuda_evidence_device"),
            "cuda_evidence_scope": str(stage.get("cuda_evidence_scope", "")),
            "cuda_fallback_fail_closed": bool(stage.get("cuda_fallback_fail_closed", False)),
            "cuda_required_log_patterns": [
                str(value) for value in stage.get("cuda_required_log_patterns", [])
            ],
            "cache_cleanup_root": str(stage.get("cache_cleanup_root", "")),
            "cache_cleanup_globs": [str(value) for value in stage.get("cache_cleanup_globs", [])],
        }
    )


def _stage_output_records(stage: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for value in stage["required_outputs"]:
        path = Path(str(value))
        if not _has_nonempty_file(path):
            raise FileNotFoundError(f"Cannot record missing OpenMVS stage output: {path}")
        records.append(
            {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
        )
    return records


def _write_stage_receipt(
    stage: Dict[str, Any],
    *,
    out_dir: Path,
    plan_sha256: str,
    log_path: Path,
) -> None:
    receipt_path = _stage_receipt_path(out_dir, stage)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "stage": str(stage["stage"]),
        "plan_sha256": str(plan_sha256),
        "stage_contract_sha256": _stage_contract_sha256(stage),
        "outputs": _stage_output_records(stage),
        "log": {
            "path": str(log_path),
            "size_bytes": int(log_path.stat().st_size),
            "sha256": _sha256_file(log_path),
        },
    }
    save_json(receipt_path, payload)


def _validate_stage_receipt(
    stage: Dict[str, Any],
    *,
    out_dir: Path,
    plan_sha256: str,
    log_path: Path,
) -> bool:
    receipt_path = _stage_receipt_path(out_dir, stage)
    try:
        receipt = load_json(receipt_path)
        if (
            receipt.get("schema_version") != 1
            or receipt.get("status") != "complete"
            or receipt.get("stage") != str(stage["stage"])
            or receipt.get("plan_sha256") != str(plan_sha256)
            or receipt.get("stage_contract_sha256") != _stage_contract_sha256(stage)
        ):
            return False
        expected_outputs = _stage_output_records(stage)
        if receipt.get("outputs") != expected_outputs:
            return False
        log_record = receipt.get("log", {})
        if (
            str(log_record.get("path", "")) != str(log_path)
            or not _has_nonempty_file(log_path)
            or int(log_record.get("size_bytes", -1)) != int(log_path.stat().st_size)
            or str(log_record.get("sha256", "")) != _sha256_file(log_path)
        ):
            return False
        _validate_cached_openmvs_cuda_evidence(stage, log_path=log_path)
        return True
    except Exception:
        return False


def _validate_cached_openmvs_cuda_evidence(stage: Dict[str, Any], *, log_path: Path) -> None:
    if stage.get("cuda_evidence_device") is None:
        return
    if not _has_nonempty_file(log_path):
        raise RuntimeError(
            f"Cannot reuse cached OpenMVS stage {stage['stage']}: CUDA evidence log is missing/empty: "
            f"{log_path}. Force this stage (or an upstream stage) to rebuild it."
        )
    _validate_openmvs_cuda_log(
        stage,
        log_path.read_text(encoding="utf-8", errors="replace"),
        log_path=log_path,
    )


def _collect_openmvs_cuda_evidence(
    command_plan: List[Dict[str, Any]],
    *,
    out_dir: Path,
) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}
    for stage in command_plan:
        if not bool(stage["enabled"]) or stage.get("cuda_evidence_device") is None:
            continue
        log_path = out_dir / "logs" / f"openmvs_{stage['stage']}.log"
        _validate_cached_openmvs_cuda_evidence(stage, log_path=log_path)
        stages[str(stage["stage"])] = {
            "expected_cuda_device": int(stage["cuda_evidence_device"]),
            "log_path": str(log_path),
            "log_sha256": _sha256_file(log_path),
            "log_size_bytes": int(log_path.stat().st_size),
            "required_positive_evidence": [
                f"CUDA device {int(stage['cuda_evidence_device'])} initialized:",
                *[str(value) for value in stage.get("cuda_required_log_patterns", [])],
            ],
            "cuda_evidence_scope": str(stage.get("cuda_evidence_scope", "")),
            "cuda_fallback_fail_closed": bool(stage.get("cuda_fallback_fail_closed", False)),
            "algorithm_cpu_components_expected": stage["stage"] == "reconstruct_mesh",
        }
    return {
        "status": "verified",
        "stages": stages,
    }


def _run_openmvs_pipeline(
    args: argparse.Namespace,
    *,
    paths: Dict[str, Path],
    command_plan: List[Dict[str, Any]],
    out_dir: Path,
    plan_sha256: str,
) -> tuple[Path, Path, str]:
    force_by_stage = {
        "interface_colmap": bool(args.force_openmvs_interface),
        "densify_point_cloud": bool(args.force_openmvs_densify),
        "reconstruct_mesh": bool(args.force_openmvs_mesh),
        "refine_mesh": bool(args.force_openmvs_refine),
    }
    invalidate_downstream = False
    enabled_stages = [stage for stage in command_plan if bool(stage["enabled"])]
    for stage_index, stage in enumerate(enabled_stages):
        outputs = [Path(path) for path in stage["required_outputs"]]
        log_path = out_dir / "logs" / f"openmvs_{stage['stage']}.log"
        stage_complete = all(_has_nonempty_file(path) for path in outputs) and _validate_stage_receipt(
            stage,
            out_dir=out_dir,
            plan_sha256=plan_sha256,
            log_path=log_path,
        )
        must_rerun = invalidate_downstream or force_by_stage[stage["stage"]] or not stage_complete
        if not must_rerun:
            continue
        # Invalidate the entire dependency suffix *before* replacing any
        # upstream output.  If the process or host dies immediately after this
        # stage succeeds, a later invocation must not be able to adopt stale
        # downstream receipts produced from the previous upstream artifact.
        downstream_stages = enabled_stages[stage_index:]
        for downstream_stage in downstream_stages:
            _remove_tree_or_link(_stage_receipt_path(out_dir, downstream_stage))
        for downstream_stage in downstream_stages:
            _remove_outputs(downstream_stage)
        try:
            _run_openmvs_stage(
                stage,
                cwd=paths["workspace"],
                log_path=log_path,
            )
            _write_stage_receipt(
                stage,
                out_dir=out_dir,
                plan_sha256=plan_sha256,
                log_path=log_path,
            )
        except BaseException:
            _remove_tree_or_link(_stage_receipt_path(out_dir, stage))
            _remove_outputs(stage)
            raise
        invalidate_downstream = True

    dense_ply = paths["dense_ply"]
    if not _has_nonempty_file(dense_ply):
        raise RuntimeError(f"OpenMVS did not produce a valid dense point cloud: {dense_ply}")
    if bool(args.skip_openmvs_refine_mesh):
        mesh_path = paths["mesh_ply"]
        mesh_backend = "openmvs_reconstruct_mesh"
    else:
        mesh_path = paths["refined_ply"]
        mesh_backend = "openmvs_refine_mesh"
    if not _has_nonempty_file(mesh_path):
        raise RuntimeError(
            f"Required OpenMVS reference mesh is missing/empty: {mesh_path}. "
            "There is no COLMAP-MVS or unrefined fallback."
        )
    return dense_ply, mesh_path, mesh_backend


def _construction_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "reference_geometry_backend": "openmvs",
        "openmvs_archive_type": -1,
        "openmvs_interface_normalize": False,
        "openmvs_cuda_device": int(args.openmvs_cuda_device),
        "openmvs_cuda_log_evidence_required": True,
        "openmvs_refine_cuda_fail_closed_required": not bool(args.skip_openmvs_refine_mesh),
        "openmvs_refine_cuda_fail_closed_marker": OPENMVS_REFINE_CUDA_FAIL_CLOSED_MARKER,
        "openmvs_resolution_level": int(args.openmvs_resolution_level),
        "openmvs_max_resolution": int(args.openmvs_max_resolution),
        "openmvs_min_resolution": int(args.openmvs_min_resolution),
        "openmvs_number_views": int(args.openmvs_number_views),
        "openmvs_number_views_fuse": int(args.openmvs_number_views_fuse),
        "openmvs_iterations": int(args.openmvs_iterations),
        "openmvs_estimate_roi": False,
        "openmvs_crop_to_roi": False,
        "openmvs_refine_mesh": not bool(args.skip_openmvs_refine_mesh),
        "openmvs_refine_resolution_level": int(args.openmvs_refine_resolution_level),
        "openmvs_refine_scales": int(args.openmvs_refine_scales),
        "texture_mesh_used": False,
        "colmap_mvs_fallback_allowed": False,
    }


def _load_and_validate_geometry(
    dense_ply: Path,
    mesh_path: Path,
) -> tuple[Dict[str, int], np.ndarray, np.ndarray, np.ndarray]:
    dense_points = load_ply_points_xyz(dense_ply)
    if dense_points.ndim != 2 or dense_points.shape[0] <= 0 or dense_points.shape[1] != 3:
        raise RuntimeError(f"OpenMVS dense point cloud is invalid: {dense_ply}")
    if not np.isfinite(dense_points).all():
        raise RuntimeError(f"OpenMVS dense point cloud contains non-finite XYZ values: {dense_ply}")
    vertices, faces = load_ply_mesh(mesh_path)
    if vertices.shape[0] <= 0 or faces.shape[0] <= 0:
        raise RuntimeError(f"OpenMVS mesh has no vertices/faces: {mesh_path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"OpenMVS mesh contains non-finite vertices: {mesh_path}")
    stats = {
        "dense_point_count": int(dense_points.shape[0]),
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
    }
    return stats, dense_points, vertices, faces


def main() -> None:
    args = _build_argparser().parse_args()
    _validate_args(args)
    strict_manifest_path = Path(args.strict_protocol_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    strict = load_json(strict_manifest_path)
    scene_name = str(strict["scene_name"])
    resolved = _resolve_reference_protocol_paths(strict)
    workspace_root = resolved["workspace_root"]
    strict_thermal_root = resolved["strict_thermal_root"]
    probe_camera_root = resolved["probe_camera_root"]
    train_union_list = resolved["train_union_list"]
    probe_camera_train_list = resolved["probe_camera_train_list"]
    probe_list = resolved["probe_list"]
    reference_probe_exclusion_list = resolved["reference_probe_exclusion_list"]
    extended_probe_camera_interface = bool(resolved["extended_probe_camera_interface"])

    _, _, colmap_binary_model = _validate_colmap_source_workspace(workspace_root)
    for required_path, label in (
        (strict_thermal_root, "strict thermal root"),
        (probe_camera_root, "probe-camera root"),
        (train_union_list, "train-union list"),
        (probe_camera_train_list, "probe-camera train list"),
        (probe_list, "probe list"),
        (reference_probe_exclusion_list, "reference probe-exclusion list"),
    ):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label}: {required_path}")
    partition_audit = _validate_training_only_partition(
        workspace_root,
        train_union_list,
        reference_probe_exclusion_list,
    )
    probe_camera_partition_stem_audit = _validate_probe_camera_partition_stems(
        train_union_list,
        reference_probe_exclusion_list,
        probe_camera_train_list,
        probe_list,
    )

    paths = _openmvs_paths(out_dir)
    executables = _resolve_openmvs_executables(args)
    command_plan = _build_openmvs_command_plan(
        args,
        paths=paths,
        executables=executables,
        colmap_binary_model=colmap_binary_model,
    )
    plan_payload = {
        "reference_construction_protocol": "openmvs-reference-mesh-v1",
        "scene_name": scene_name,
        "source_workspace_root": str(workspace_root),
        "source_workspace_fingerprint": _source_workspace_fingerprint(workspace_root),
        "training_only_partition_audit": partition_audit,
        "strict_protocol_manifest_sha256": _sha256_file(strict_manifest_path),
        "train_union_list_sha256": _sha256_file(train_union_list),
        "probe_list_sha256": _sha256_file(probe_list),
        "output_root": str(out_dir),
        "openmvs_executable_sha256": {
            name: _sha256_file(Path(path))
            for name, path in executables.items()
            if name != "refine_mesh" or not bool(args.skip_openmvs_refine_mesh)
        },
        "commands": command_plan,
        "construction_overrides": _construction_overrides(args),
    }
    if extended_probe_camera_interface:
        plan_payload.update(
            {
                "probe_camera_root": str(probe_camera_root),
                "probe_camera_train_list_sha256": _sha256_file(probe_camera_train_list),
                "reference_probe_exclusion_list_sha256": _sha256_file(
                    reference_probe_exclusion_list
                ),
                "probe_camera_partition_stem_audit": probe_camera_partition_stem_audit,
            }
        )
    plan_sha256 = _canonical_sha256(plan_payload)
    if args.dry_run:
        print(json.dumps(plan_payload, indent=2, ensure_ascii=False))
        print("OPENMVS_REFERENCE_DRY_RUN_OK")
        return

    plan_path = out_dir / "openmvs_command_plan.json"
    owned_reset_paths = [
        paths["input_workspace"],
        paths["workspace"],
        out_dir / "_openmvs_state",
        out_dir / "logs",
        out_dir / "views",
        out_dir / "probe_camera_manifest.json",
        out_dir / "reference_roi.json",
        out_dir / "openmvs_cuda_evidence.json",
        out_dir / "reference_depth_manifest.json",
        out_dir / "reference_build_manifest.json",
        plan_path,
    ]
    force_full_rebuild = bool(args.force_openmvs_interface)
    if plan_path.exists():
        try:
            previous_plan = load_json(plan_path)
        except Exception as exc:
            if not force_full_rebuild:
                raise RuntimeError(
                    f"Existing OpenMVS ownership plan is unreadable: {plan_path}. "
                    "Use a new --out_dir or pass --force_openmvs_interface to reset owned artifacts."
                ) from exc
            previous_plan = None
        if previous_plan != plan_payload and not force_full_rebuild:
            raise RuntimeError(
                "OpenMVS command/source plan differs from the existing isolated output. "
                "Use a new --out_dir, or pass --force_openmvs_interface to invalidate and "
                "rebuild every downstream OpenMVS artifact."
            )
    elif out_dir.is_dir() and any(out_dir.iterdir()):
        known_names = {path.name for path in owned_reset_paths}
        unknown = sorted(path.name for path in out_dir.iterdir() if path.name not in known_names)
        if not force_full_rebuild or unknown:
            raise RuntimeError(
                "OpenMVS output contains artifacts without its ownership plan; refusing cache adoption. "
                f"Use a new --out_dir, or pass --force_openmvs_interface when only known owned artifacts "
                f"are present. unknown={unknown} out_dir={out_dir}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    if force_full_rebuild:
        # Interface-level invalidation means the imported scene and every
        # downstream receipt/log/view/manifest must be regenerated together.
        for owned_path in owned_reset_paths:
            _remove_tree_or_link(owned_path)
    paths["workspace"].mkdir(parents=True, exist_ok=True)
    prepared_workspace = _prepare_openmvs_input_workspace(workspace_root, paths["input_workspace"])
    save_json(plan_path, plan_payload)

    dense_ply, mesh_path, mesh_backend = _run_openmvs_pipeline(
        args,
        paths=paths,
        command_plan=command_plan,
        out_dir=out_dir,
        plan_sha256=plan_sha256,
    )
    cuda_evidence = _collect_openmvs_cuda_evidence(command_plan, out_dir=out_dir)
    cuda_evidence_path = out_dir / "openmvs_cuda_evidence.json"
    save_json(cuda_evidence_path, cuda_evidence)
    geometry_stats, fused_points, vertices_world, faces = _load_and_validate_geometry(dense_ply, mesh_path)
    geometry_artifacts = {
        "dense_ply": {
            "path": str(dense_ply),
            "size_bytes": int(dense_ply.stat().st_size),
            "sha256": _sha256_file(dense_ply),
        },
        "mesh": {
            "path": str(mesh_path),
            "size_bytes": int(mesh_path.stat().st_size),
            "sha256": _sha256_file(mesh_path),
        },
    }
    stage_receipts = {
        str(stage["stage"]): {
            "path": str(_stage_receipt_path(out_dir, stage)),
            "size_bytes": int(_stage_receipt_path(out_dir, stage).stat().st_size),
            "sha256": _sha256_file(_stage_receipt_path(out_dir, stage)),
        }
        for stage in command_plan
        if bool(stage["enabled"])
    }
    roi = compute_quantile_bbox(
        fused_points,
        lower_quantile=float(args.bbox_lower_quantile),
        upper_quantile=float(args.bbox_upper_quantile),
        padding_ratio_of_robust_diagonal=float(args.bbox_padding_ratio),
    )
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)

    roi_path = out_dir / "reference_roi.json"
    save_json(
        roi_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "reference_construction_protocol": "openmvs-reference-mesh-v1",
            "scene_name": scene_name,
            "roi_rule": {
                "type": "training_reference_dense_quantile_aabb",
                "lower_quantile": float(args.bbox_lower_quantile),
                "upper_quantile": float(args.bbox_upper_quantile),
                "padding_ratio_of_robust_diagonal": float(args.bbox_padding_ratio),
            },
            "bbox_min": roi["bbox_min"].tolist(),
            "bbox_max": roi["bbox_max"].tolist(),
            "scene_diagonal": float(roi["scene_diagonal"]),
            "source_points_path": str(dense_ply),
            "source_points_backend": "openmvs_densify_point_cloud",
            "source_points_sha256": geometry_artifacts["dense_ply"]["sha256"],
        },
    )

    camera_manifest_path = out_dir / "probe_camera_manifest.json"
    # Rebuild camera evidence on every entry.  Its cost is small relative to
    # OpenMVS and it prevents an unrelated/stale manifest from being adopted by
    # mere path existence.
    camera_manifest = build_probe_view_manifest(
        source_path=probe_camera_root,
        images_dir_name="images",
        resolution_arg=int(args.resolution_arg),
        train_list=probe_camera_train_list,
        test_list=probe_list,
        scene_name=scene_name,
    )
    save_json(camera_manifest_path, camera_manifest)

    thresholds_m = parse_thresholds_m(args.thresholds_m)
    views_dir = out_dir / "views"
    _remove_tree_or_link(views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    manifest_views: List[Dict[str, Any]] = []
    for view in camera_manifest["views"]:
        depth = render_mesh_depth_for_view(vertices_world, faces, view)
        support_count = render_support_count_for_view(
            fused_points,
            view,
            depth_tolerance_m=float(args.support_depth_tolerance_m),
            support_radius_px=int(args.support_radius_px),
        )
        finite = np.isfinite(depth) & (depth > 0.0)
        inside_roi = (
            compute_inside_bbox_mask(depth, view, bbox_min=bbox_min, bbox_max=bbox_max)
            if np.any(finite)
            else np.zeros_like(finite, dtype=bool)
        )
        valid_mask = finite & inside_roi & (support_count >= int(args.support_min_count))
        view_rel = Path("views") / f"{view['view_id']}.npz"
        view_path = out_dir / view_rel
        np.savez_compressed(
            view_path,
            depth=np.asarray(depth, dtype=np.float64),
            support_count=np.asarray(support_count, dtype=np.int32),
            valid_mask=np.asarray(valid_mask, dtype=np.uint8),
            inside_roi=np.asarray(inside_roi, dtype=np.uint8),
        )
        manifest_views.append(
            {
                "view_id": str(view["view_id"]),
                "image_name": str(view["image_name"]),
                "width": int(view["width"]),
                "height": int(view["height"]),
                "fx": float(view["fx"]),
                "fy": float(view["fy"]),
                "cx": float(view["cx"]),
                "cy": float(view["cy"]),
                "camera_to_world": view["camera_to_world"],
                "npz_file": str(view_rel).replace("\\", "/"),
                "npz_size_bytes": int(view_path.stat().st_size),
                "npz_sha256": _sha256_file(view_path),
            }
        )

    support_rule = {
        "type": "training_dense_projected_support_count",
        "source_backend": "openmvs_densify_point_cloud",
        "min_support_count": int(args.support_min_count),
        "support_radius_px": int(args.support_radius_px),
        "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
    }
    construction_overrides = _construction_overrides(args)
    common_manifest = {
        "protocol_name": "reference-depth-based-geometric-evaluation-v1",
        "reference_construction_protocol": "openmvs-reference-mesh-v1",
        "scene_name": scene_name,
        "strict_protocol_manifest": str(strict_manifest_path),
        "reference_workspace_root": str(workspace_root),
        "openmvs_input_workspace": str(prepared_workspace),
        "openmvs_workspace": str(paths["workspace"]),
        "openmvs_command_plan": str(plan_path),
        "openmvs_command_plan_sha256": _sha256_file(plan_path),
        "openmvs_plan_semantic_sha256": plan_sha256,
        "openmvs_stage_receipts": stage_receipts,
        "openmvs_cuda_evidence_path": str(cuda_evidence_path),
        "openmvs_cuda_evidence": cuda_evidence,
        "reference_dense_backend": "openmvs_densify_point_cloud",
        "reference_dense_ply": str(dense_ply),
        "reference_fused_ply": str(dense_ply),
        "reference_mesher_input_ply": str(dense_ply),
        "reference_mesh_path": str(mesh_path),
        "reference_mesh_sha256": geometry_artifacts["mesh"]["sha256"],
        "reference_mesh_size_bytes": geometry_artifacts["mesh"]["size_bytes"],
        "reference_dense_ply_sha256": geometry_artifacts["dense_ply"]["sha256"],
        "reference_dense_ply_size_bytes": geometry_artifacts["dense_ply"]["size_bytes"],
        "reference_geometry_artifacts": geometry_artifacts,
        "reference_mesh_backend": mesh_backend,
        "coordinate_frame_rule": (
            "inherit_aligned_training_sfm_via_interfacecolmap_normalize_0_no_transform"
        ),
        "geometry_stats": geometry_stats,
        "roi_path": str(roi_path),
        "depth_semantics": "metric_camera_z_reference_mesh",
        "distance_unit": "meters",
        "thresholds_m": thresholds_m,
        "support_rule": support_rule,
        "reference_construction_overrides": construction_overrides,
    }

    ref_manifest_path = out_dir / "reference_depth_manifest.json"
    save_json(
        ref_manifest_path,
        {
            **common_manifest,
            "camera_manifest_path": str(camera_manifest_path),
            "views": manifest_views,
        },
    )
    build_manifest_path = out_dir / "reference_build_manifest.json"
    build_manifest = {
        **common_manifest,
        "source_workspace_root": str(workspace_root),
        "strict_thermal_root": str(strict_thermal_root),
        "train_union_list": str(train_union_list),
        "probe_list": str(probe_list),
        "reference_depth_manifest": str(ref_manifest_path),
        "camera_manifest_path": str(camera_manifest_path),
    }
    if extended_probe_camera_interface:
        build_manifest.update(
            {
                "probe_camera_root": str(probe_camera_root),
                "probe_camera_train_list": str(probe_camera_train_list),
                "reference_probe_exclusion_list": str(reference_probe_exclusion_list),
                "probe_camera_partition_stem_audit": probe_camera_partition_stem_audit,
            }
        )
    save_json(build_manifest_path, build_manifest)
    print(f"REFERENCE_DEPTH_MANIFEST {ref_manifest_path}")
    print(f"REFERENCE_BUILD_MANIFEST {build_manifest_path}")


if __name__ == "__main__":
    main()
