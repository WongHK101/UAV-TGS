from __future__ import annotations

import argparse
import os
import re
import shutil
import stat
import struct
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from depth_reference_common import (
    build_probe_view_manifest,
    compute_inside_bbox_mask,
    compute_quantile_bbox,
    load_json,
    load_ply_mesh,
    load_ply_points_xyz,
    parse_thresholds_m,
    relative_or_abs,
    render_mesh_depth_for_view,
    render_support_count_for_view,
    run_colmap,
    save_json,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build training-only reference depth artifacts for held-out geometry evaluation")
    parser.add_argument("--strict_protocol_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--colmap_cmd", default="colmap")
    parser.add_argument("--resolution_arg", type=int, default=4)
    parser.add_argument("--thresholds_m", default="0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00")
    parser.add_argument("--bbox_lower_quantile", type=float, default=0.01)
    parser.add_argument("--bbox_upper_quantile", type=float, default=0.99)
    parser.add_argument("--bbox_padding_ratio", type=float, default=0.02)
    parser.add_argument("--support_min_count", type=int, default=1)
    parser.add_argument("--support_radius_px", type=int, default=1)
    parser.add_argument("--support_depth_tolerance_m", type=float, default=0.10)
    parser.add_argument("--patch_match_max_image_size", type=int, default=2000)
    parser.add_argument("--patch_match_auto_source_count", type=int, default=None)
    parser.add_argument("--patch_match_window_radius", type=int, default=None)
    parser.add_argument("--patch_match_num_iterations", type=int, default=None)
    parser.add_argument("--patch_match_geom_consistency", type=int, choices=[0, 1], default=None)
    parser.add_argument("--patch_match_filter", type=int, choices=[0, 1], default=None)
    parser.add_argument("--patch_match_min_triangulation_angle", type=float, default=None)
    parser.add_argument("--patch_match_filter_min_triangulation_angle", type=float, default=None)
    parser.add_argument("--patch_match_filter_min_num_consistent", type=int, default=None)
    parser.add_argument("--patch_match_filter_min_ncc", type=float, default=None)
    parser.add_argument("--patch_match_filter_geom_consistency_max_cost", type=float, default=None)
    parser.add_argument("--stereo_fusion_max_image_size", type=int, default=None)
    parser.add_argument("--stereo_fusion_min_num_pixels", type=int, default=None)
    parser.add_argument("--stereo_fusion_max_reproj_error", type=float, default=None)
    parser.add_argument("--stereo_fusion_max_depth_error", type=float, default=None)
    parser.add_argument("--stereo_fusion_max_normal_error", type=float, default=None)
    parser.add_argument("--mesh_backend_preference", choices=["auto", "delaunay", "poisson"], default="auto")
    parser.add_argument("--poisson_depth", type=int, default=None)
    parser.add_argument("--poisson_trim", type=float, default=None)
    parser.add_argument("--poisson_point_weight", type=float, default=None)
    parser.add_argument(
        "--mesh_crop_fused_to_roi",
        action="store_true",
        help=(
            "Crop the fused dense point cloud to the training-side robust ROI before "
            "Poisson meshing. Default OFF to preserve baseline behavior."
        ),
    )
    parser.add_argument("--force_patch_match", action="store_true")
    parser.add_argument("--force_fusion", action="store_true")
    parser.add_argument("--force_mesh", action="store_true")
    parser.add_argument("--force_views", action="store_true")
    return parser


def _is_reparse_point(path: Path) -> bool:
    try:
        attrs = os.lstat(path).st_file_attributes
    except (AttributeError, FileNotFoundError):
        return False
    return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _remove_tree_or_link(path: Path) -> None:
    if not path.exists():
        return
    if _is_reparse_point(path):
        completed = subprocess.run(["cmd", "/c", "rmdir", str(path)], check=False, capture_output=True, text=True)
        if completed.returncode != 0 and path.exists():
            raise RuntimeError(f"Failed to remove junction {path}: {completed.stdout}\n{completed.stderr}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _ensure_dir_junction(link_path: Path, target_path: Path) -> None:
    if link_path.exists():
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0 and (not link_path.exists()):
        raise RuntimeError(
            f"Failed to create junction {link_path} -> {target_path}: "
            f"{completed.stdout}\n{completed.stderr}"
        )


def _prepare_flat_colmap_workspace(source_workspace_root: Path, prepared_root: Path) -> Path:
    images_src = source_workspace_root / "images"
    input_src = source_workspace_root / "input"
    distorted_src = source_workspace_root / "distorted"
    stereo_src = source_workspace_root / "stereo"
    sparse_src = source_workspace_root / "sparse" / "0"
    if not images_src.exists() or not stereo_src.exists() or not sparse_src.exists():
        raise FileNotFoundError(
            "Expected source workspace to contain images/, stereo/, and sparse/0; "
            f"got images={images_src.exists()} stereo={stereo_src.exists()} sparse0={sparse_src.exists()}"
        )
    prepared_root.mkdir(parents=True, exist_ok=True)
    _ensure_dir_junction(prepared_root / "images", images_src)
    if input_src.exists():
        _ensure_dir_junction(prepared_root / "input", input_src)
    if distorted_src.exists():
        _ensure_dir_junction(prepared_root / "distorted", distorted_src)
    stereo_dst = prepared_root / "stereo"
    if stereo_dst.exists() and _is_reparse_point(stereo_dst):
        _remove_tree_or_link(stereo_dst)
    stereo_dst.mkdir(parents=True, exist_ok=True)
    for cfg_name in ("patch-match.cfg", "fusion.cfg"):
        cfg_src = stereo_src / cfg_name
        if cfg_src.exists():
            shutil.copy2(cfg_src, stereo_dst / cfg_name)
    (stereo_dst / "depth_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "normal_maps").mkdir(parents=True, exist_ok=True)
    (stereo_dst / "consistency_graphs").mkdir(parents=True, exist_ok=True)
    sparse_dst = prepared_root / "sparse"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for src_file in sorted(sparse_src.iterdir()):
        if not src_file.is_file():
            continue
        dst_file = sparse_dst / src_file.name
        if not dst_file.exists():
            shutil.copy2(src_file, dst_file)
    return prepared_root


def _rewrite_patch_match_auto_source_count(prepared_workspace: Path, source_count: int | None) -> None:
    if source_count is None:
        return
    if source_count <= 0:
        raise ValueError(f"patch_match_auto_source_count must be positive, got {source_count}")
    cfg_path = prepared_workspace / "stereo" / "patch-match.cfg"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot rewrite missing patch-match config: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    rewritten = re.sub(r"(?m)^__auto__,\s*\d+\s*$", f"__auto__, {int(source_count)}", text)
    cfg_path.write_text(rewritten, encoding="ascii")


def _has_dense_outputs(prepared_workspace: Path, *, require_geometric: bool = False) -> bool:
    depth_maps = prepared_workspace / "stereo" / "depth_maps"
    if not depth_maps.exists():
        return False
    images_dir = prepared_workspace / "images"
    expected_count = len([p for p in images_dir.iterdir() if p.is_file()]) if images_dir.exists() else 0
    if expected_count <= 0:
        return any(depth_maps.rglob("*.bin"))
    suffix = "*.geometric.bin" if require_geometric else "*.photometric.bin"
    return len(list(depth_maps.rglob(suffix))) >= expected_count


def _has_nonempty_file(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _sync_fused_point_cloud_for_mesher(prepared_workspace: Path, fused_ply: Path) -> Path:
    workspace_fused = prepared_workspace / "fused.ply"
    if _has_nonempty_file(fused_ply):
        needs_copy = (
            (not _has_nonempty_file(workspace_fused))
            or workspace_fused.stat().st_size != fused_ply.stat().st_size
            or workspace_fused.stat().st_mtime < fused_ply.stat().st_mtime
        )
        if needs_copy:
            shutil.copy2(fused_ply, workspace_fused)
    fused_vis = fused_ply.with_suffix(fused_ply.suffix + ".vis")
    workspace_fused_vis = prepared_workspace / "fused.ply.vis"
    if _has_nonempty_file(fused_vis):
        needs_copy_vis = (
            (not _has_nonempty_file(workspace_fused_vis))
            or workspace_fused_vis.stat().st_size != fused_vis.stat().st_size
            or workspace_fused_vis.stat().st_mtime < fused_vis.stat().st_mtime
        )
        if needs_copy_vis:
            shutil.copy2(fused_vis, workspace_fused_vis)
    return workspace_fused


def _append_colmap_option(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    cmd.extend([key, str(value)])


def _run_delaunay_mesher(args: argparse.Namespace, prepared_workspace: Path, output_path: Path) -> None:
    run_colmap(
        args.colmap_cmd,
        [
            "delaunay_mesher",
            "--input_path",
            str(prepared_workspace),
            "--input_type",
            "dense",
            "--output_path",
            str(output_path),
        ],
        cwd=prepared_workspace,
    )
    if not _has_nonempty_file(output_path):
        raise RuntimeError(f"Delaunay mesher produced an empty mesh: {output_path}")


def _run_poisson_mesher(args: argparse.Namespace, prepared_workspace: Path, fused_ply: Path, output_path: Path) -> None:
    cmd: List[str] = [
        "poisson_mesher",
        "--input_path",
        str(fused_ply),
        "--output_path",
        str(output_path),
    ]
    _append_colmap_option(cmd, "--PoissonMeshing.depth", args.poisson_depth)
    _append_colmap_option(cmd, "--PoissonMeshing.trim", args.poisson_trim)
    _append_colmap_option(cmd, "--PoissonMeshing.point_weight", args.poisson_point_weight)
    run_colmap(
        args.colmap_cmd,
        cmd,
        cwd=prepared_workspace,
    )
    if not _has_nonempty_file(output_path):
        raise RuntimeError(f"Poisson mesher produced an empty mesh: {output_path}")


def _parse_binary_vertex_ply(path: Path) -> tuple[int, int, np.dtype, List[str]]:
    """Return data offset, vertex count, structured dtype, and header lines.

    This helper intentionally supports the binary-little-endian vertex-only PLY
    files produced by COLMAP stereo_fusion. It keeps the rest of the pipeline
    dependency-free and avoids loading very large fused clouds fully into RAM.
    """
    dtype_map = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "<i2",
        "int16": "<i2",
        "ushort": "<u2",
        "uint16": "<u2",
        "int": "<i4",
        "int32": "<i4",
        "uint": "<u4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    header_lines: List[str] = []
    vertex_count: int | None = None
    vertex_props: List[tuple[str, str]] = []
    current_element = ""
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                raise ValueError(f"PLY header has no end_header: {path}")
            line = raw.decode("ascii", errors="replace").rstrip("\r\n")
            header_lines.append(line)
            if line.startswith("format ") and line != "format binary_little_endian 1.0":
                raise ValueError(f"Only binary_little_endian PLY is supported, got {line!r} in {path}")
            if line.startswith("element "):
                parts = line.split()
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif line.startswith("property ") and current_element == "vertex":
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Unsupported vertex property line in {path}: {line}")
                prop_type, prop_name = parts[1], parts[2]
                if prop_type not in dtype_map:
                    raise ValueError(f"Unsupported PLY property type {prop_type!r} in {path}")
                vertex_props.append((prop_name, dtype_map[prop_type]))
            elif line == "end_header":
                data_offset = f.tell()
                break
    if vertex_count is None:
        raise ValueError(f"PLY has no vertex element: {path}")
    dtype = np.dtype(vertex_props)
    return data_offset, vertex_count, dtype, header_lines


def _write_cropped_binary_vertex_ply(
    input_path: Path,
    output_path: Path,
    *,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    chunk_size: int = 2_000_000,
) -> Dict[str, Any]:
    data_offset, vertex_count, dtype, header_lines = _parse_binary_vertex_ply(input_path)
    if not all(name in dtype.names for name in ("x", "y", "z")):
        raise ValueError(f"PLY vertex element must contain x/y/z fields: {input_path}")

    vertices = np.memmap(input_path, dtype=dtype, mode="r", offset=data_offset, shape=(vertex_count,))
    keep_mask = np.zeros(vertex_count, dtype=bool)
    for start in range(0, vertex_count, int(chunk_size)):
        chunk = vertices[start : start + int(chunk_size)]
        mask = (
            np.isfinite(chunk["x"])
            & np.isfinite(chunk["y"])
            & np.isfinite(chunk["z"])
            & (chunk["x"] >= float(bbox_min[0]))
            & (chunk["x"] <= float(bbox_max[0]))
            & (chunk["y"] >= float(bbox_min[1]))
            & (chunk["y"] <= float(bbox_max[1]))
            & (chunk["z"] >= float(bbox_min[2]))
            & (chunk["z"] <= float(bbox_max[2]))
        )
        keep_mask[start : start + int(chunk_size)] = mask
    kept_count = int(keep_mask.sum())
    if kept_count <= 0:
        raise RuntimeError(f"ROI crop removed all fused points from {input_path}")

    rewritten_header: List[str] = []
    for line in header_lines:
        if line.startswith("element vertex "):
            rewritten_header.append(f"element vertex {kept_count}")
        else:
            rewritten_header.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(("\n".join(rewritten_header) + "\n").encode("ascii"))
        for start in range(0, vertex_count, int(chunk_size)):
            chunk = vertices[start : start + int(chunk_size)]
            mask = keep_mask[start : start + int(chunk_size)]
            if np.any(mask):
                np.asarray(chunk[mask]).tofile(f)

    input_vis = input_path.with_suffix(input_path.suffix + ".vis")
    output_vis = output_path.with_suffix(output_path.suffix + ".vis")
    wrote_vis = False
    if _has_nonempty_file(input_vis):
        _write_cropped_colmap_vis(input_vis, output_vis, keep_mask=keep_mask)
        wrote_vis = True

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_vis_path": str(input_vis) if input_vis.exists() else "",
        "output_vis_path": str(output_vis) if wrote_vis else "",
        "input_vertex_count": int(vertex_count),
        "kept_vertex_count": int(kept_count),
        "kept_ratio": float(kept_count / max(1, vertex_count)),
        "bbox_min": np.asarray(bbox_min, dtype=np.float64).tolist(),
        "bbox_max": np.asarray(bbox_max, dtype=np.float64).tolist(),
    }


def _write_cropped_colmap_vis(input_vis: Path, output_vis: Path, *, keep_mask: np.ndarray) -> None:
    """Crop COLMAP fused.ply.vis in lockstep with a vertex keep mask.

    COLMAP writes this file as a uint64 point count followed by one record per
    point: uint32 visible_count and visible_count uint32 image ids.
    """
    u64 = struct.Struct("<Q")
    u32 = struct.Struct("<I")
    output_vis.parent.mkdir(parents=True, exist_ok=True)
    with input_vis.open("rb") as fin, output_vis.open("wb") as fout:
        header = fin.read(u64.size)
        if len(header) != u64.size:
            raise ValueError(f"Invalid COLMAP vis file header: {input_vis}")
        (vis_point_count,) = u64.unpack(header)
        if int(vis_point_count) != int(keep_mask.shape[0]):
            raise ValueError(
                f"VIS point count mismatch for {input_vis}: "
                f"vis={vis_point_count} mask={keep_mask.shape[0]}"
            )
        fout.write(u64.pack(int(keep_mask.sum())))
        for keep in keep_mask:
            count_raw = fin.read(u32.size)
            if len(count_raw) != u32.size:
                raise ValueError(f"Unexpected EOF while reading visible-count record from {input_vis}")
            (visible_count,) = u32.unpack(count_raw)
            payload_size = int(visible_count) * u32.size
            payload = fin.read(payload_size)
            if len(payload) != payload_size:
                raise ValueError(f"Unexpected EOF while reading visible image ids from {input_vis}")
            if bool(keep):
                fout.write(count_raw)
                fout.write(payload)


def main() -> None:
    args = _build_argparser().parse_args()
    strict_manifest_path = Path(args.strict_protocol_manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    strict = load_json(strict_manifest_path)
    scene_name = str(strict["scene_name"])
    artifacts = strict["artifacts"]
    lists = strict["lists"]
    workspace_root = Path(artifacts["train_union_source_root"]).resolve()
    prepared_workspace = _prepare_flat_colmap_workspace(workspace_root, out_dir / "_colmap_workspace_flat")
    _rewrite_patch_match_auto_source_count(prepared_workspace, args.patch_match_auto_source_count)
    strict_thermal_root = Path(artifacts["strict_thermal_root"]).resolve()
    train_union_list = Path(lists["train_union"]).resolve()
    probe_list = Path(lists["probe_test"]).resolve()

    stereo_root = workspace_root / "stereo"
    if not stereo_root.exists():
        raise FileNotFoundError(f"Expected COLMAP stereo workspace at {stereo_root}")

    fused_ply = out_dir / "reference_fused_geometric.ply"
    delaunay_mesh = out_dir / "reference_mesh_delaunay.ply"
    poisson_mesh = out_dir / "reference_mesh_poisson.ply"
    mesh_path = delaunay_mesh
    mesh_backend = "delaunay_mesher"

    require_geometric_outputs = bool(args.patch_match_geom_consistency == 1)
    can_reuse_fused_without_dense = _has_nonempty_file(fused_ply) and (not args.force_patch_match) and (not args.force_fusion)
    if (not can_reuse_fused_without_dense) and (
        args.force_patch_match or (not _has_dense_outputs(prepared_workspace, require_geometric=require_geometric_outputs))
    ):
        patch_match_cmd: List[str] = [
            "patch_match_stereo",
            "--workspace_path",
            str(prepared_workspace),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.max_image_size",
            str(int(args.patch_match_max_image_size)),
        ]
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.geom_consistency",
            args.patch_match_geom_consistency,
        )
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.filter",
            args.patch_match_filter,
        )
        _append_colmap_option(patch_match_cmd, "--PatchMatchStereo.window_radius", args.patch_match_window_radius)
        _append_colmap_option(patch_match_cmd, "--PatchMatchStereo.num_iterations", args.patch_match_num_iterations)
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.min_triangulation_angle",
            args.patch_match_min_triangulation_angle,
        )
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.filter_min_triangulation_angle",
            args.patch_match_filter_min_triangulation_angle,
        )
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.filter_min_num_consistent",
            args.patch_match_filter_min_num_consistent,
        )
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.filter_min_ncc",
            args.patch_match_filter_min_ncc,
        )
        _append_colmap_option(
            patch_match_cmd,
            "--PatchMatchStereo.filter_geom_consistency_max_cost",
            args.patch_match_filter_geom_consistency_max_cost,
        )
        run_colmap(
            args.colmap_cmd,
            patch_match_cmd,
            cwd=prepared_workspace,
        )

    if args.force_fusion or (not _has_nonempty_file(fused_ply)):
        stereo_fusion_cmd: List[str] = [
            "stereo_fusion",
            "--workspace_path",
            str(prepared_workspace),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_ply),
        ]
        _append_colmap_option(stereo_fusion_cmd, "--StereoFusion.max_image_size", args.stereo_fusion_max_image_size)
        _append_colmap_option(stereo_fusion_cmd, "--StereoFusion.min_num_pixels", args.stereo_fusion_min_num_pixels)
        _append_colmap_option(stereo_fusion_cmd, "--StereoFusion.max_reproj_error", args.stereo_fusion_max_reproj_error)
        _append_colmap_option(stereo_fusion_cmd, "--StereoFusion.max_depth_error", args.stereo_fusion_max_depth_error)
        _append_colmap_option(stereo_fusion_cmd, "--StereoFusion.max_normal_error", args.stereo_fusion_max_normal_error)
        run_colmap(
            args.colmap_cmd,
            stereo_fusion_cmd,
            cwd=prepared_workspace,
        )
    if not _has_nonempty_file(fused_ply):
        raise RuntimeError(f"COLMAP stereo_fusion did not produce a valid fused point cloud at {fused_ply}")

    fused_points = load_ply_points_xyz(fused_ply)
    roi = compute_quantile_bbox(
        fused_points,
        lower_quantile=float(args.bbox_lower_quantile),
        upper_quantile=float(args.bbox_upper_quantile),
        padding_ratio_of_robust_diagonal=float(args.bbox_padding_ratio),
    )
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)

    mesher_fused_ply = fused_ply
    mesh_crop_manifest: Dict[str, Any] | None = None
    if bool(args.mesh_crop_fused_to_roi):
        cropped_fused_ply = out_dir / "reference_fused_geometric_roi_crop.ply"
        cropped_fused_vis = cropped_fused_ply.with_suffix(cropped_fused_ply.suffix + ".vis")
        full_fused_vis = fused_ply.with_suffix(fused_ply.suffix + ".vis")
        crop_vis_missing = _has_nonempty_file(full_fused_vis) and (not _has_nonempty_file(cropped_fused_vis))
        if args.force_mesh or (not _has_nonempty_file(cropped_fused_ply)) or crop_vis_missing:
            mesh_crop_manifest = _write_cropped_binary_vertex_ply(
                fused_ply,
                cropped_fused_ply,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            )
            save_json(out_dir / "reference_fused_geometric_roi_crop_manifest.json", mesh_crop_manifest)
        else:
            mesh_crop_manifest = load_json(out_dir / "reference_fused_geometric_roi_crop_manifest.json")
        mesher_fused_ply = cropped_fused_ply

    for stale_mesh in (delaunay_mesh, poisson_mesh):
        if stale_mesh.exists() and (not _has_nonempty_file(stale_mesh)):
            stale_mesh.unlink()

    if args.force_mesh or (not _has_nonempty_file(delaunay_mesh) and not _has_nonempty_file(poisson_mesh)):
        _sync_fused_point_cloud_for_mesher(prepared_workspace, mesher_fused_ply)
        if args.mesh_backend_preference == "poisson":
            try:
                _run_poisson_mesher(args, prepared_workspace, mesher_fused_ply, poisson_mesh)
                mesh_path = poisson_mesh
                mesh_backend = "poisson_mesher"
            except Exception:
                _run_delaunay_mesher(args, prepared_workspace, delaunay_mesh)
                mesh_path = delaunay_mesh
                mesh_backend = "delaunay_mesher"
        else:
            try:
                _run_delaunay_mesher(args, prepared_workspace, delaunay_mesh)
                mesh_path = delaunay_mesh
                mesh_backend = "delaunay_mesher"
            except Exception:
                _run_poisson_mesher(args, prepared_workspace, mesher_fused_ply, poisson_mesh)
                mesh_path = poisson_mesh
                mesh_backend = "poisson_mesher"
    elif _has_nonempty_file(delaunay_mesh):
        mesh_path = delaunay_mesh
        mesh_backend = "delaunay_mesher"
    elif _has_nonempty_file(poisson_mesh):
        mesh_path = poisson_mesh
        mesh_backend = "poisson_mesher"
    else:
        raise FileNotFoundError("No reference mesh was produced")

    roi_path = out_dir / "reference_roi.json"
    save_json(
        roi_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
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
            "source_points_path": str(fused_ply),
        },
    )

    camera_manifest_path = out_dir / "probe_camera_manifest.json"
    if args.force_views or (not camera_manifest_path.exists()):
        camera_manifest = build_probe_view_manifest(
            source_path=strict_thermal_root,
            images_dir_name="images",
            resolution_arg=int(args.resolution_arg),
            train_list=train_union_list,
            test_list=probe_list,
            scene_name=scene_name,
        )
        save_json(camera_manifest_path, camera_manifest)
    else:
        camera_manifest = load_json(camera_manifest_path)

    vertices_world, faces = load_ply_mesh(mesh_path)
    bbox_min = np.asarray(roi["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(roi["bbox_max"], dtype=np.float64)
    thresholds_m = parse_thresholds_m(args.thresholds_m)

    views_dir = out_dir / "views"
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
        inside_roi = compute_inside_bbox_mask(depth, view, bbox_min=bbox_min, bbox_max=bbox_max) if np.any(finite) else np.zeros_like(finite, dtype=bool)
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
            }
        )

    ref_manifest_path = out_dir / "reference_depth_manifest.json"
    save_json(
        ref_manifest_path,
        {
            "protocol_name": "reference-depth-based-geometric-evaluation-v1",
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "camera_manifest_path": str(camera_manifest_path),
            "reference_workspace_root": str(workspace_root),
            "reference_fused_ply": str(fused_ply),
            "reference_mesher_input_ply": str(mesher_fused_ply),
            "reference_mesh_path": str(mesh_path),
            "reference_mesh_backend": mesh_backend,
            "roi_path": str(roi_path),
            "depth_semantics": "metric_camera_z_reference_mesh",
            "distance_unit": "meters",
            "thresholds_m": thresholds_m,
            "support_rule": {
                "type": "training_dense_projected_support_count",
                "min_support_count": int(args.support_min_count),
                "support_radius_px": int(args.support_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
            "reference_construction_overrides": {
                "patch_match_max_image_size": int(args.patch_match_max_image_size),
                "patch_match_auto_source_count": args.patch_match_auto_source_count,
                "patch_match_window_radius": args.patch_match_window_radius,
                "patch_match_num_iterations": args.patch_match_num_iterations,
                "patch_match_geom_consistency": args.patch_match_geom_consistency,
                "patch_match_filter": args.patch_match_filter,
                "patch_match_min_triangulation_angle": args.patch_match_min_triangulation_angle,
                "patch_match_filter_min_triangulation_angle": args.patch_match_filter_min_triangulation_angle,
                "patch_match_filter_min_num_consistent": args.patch_match_filter_min_num_consistent,
                "patch_match_filter_min_ncc": args.patch_match_filter_min_ncc,
                "patch_match_filter_geom_consistency_max_cost": args.patch_match_filter_geom_consistency_max_cost,
                "stereo_fusion_max_image_size": args.stereo_fusion_max_image_size,
                "stereo_fusion_min_num_pixels": args.stereo_fusion_min_num_pixels,
                "stereo_fusion_max_reproj_error": args.stereo_fusion_max_reproj_error,
                "stereo_fusion_max_depth_error": args.stereo_fusion_max_depth_error,
                "stereo_fusion_max_normal_error": args.stereo_fusion_max_normal_error,
                "mesh_backend_preference": str(args.mesh_backend_preference),
                "poisson_depth": args.poisson_depth,
                "poisson_trim": args.poisson_trim,
                "poisson_point_weight": args.poisson_point_weight,
                "mesh_crop_fused_to_roi": bool(args.mesh_crop_fused_to_roi),
                "mesh_crop_manifest": mesh_crop_manifest,
            },
            "views": manifest_views,
        },
    )

    build_manifest_path = out_dir / "reference_build_manifest.json"
    save_json(
        build_manifest_path,
        {
            "scene_name": scene_name,
            "strict_protocol_manifest": str(strict_manifest_path),
            "source_workspace_root": str(workspace_root),
            "prepared_workspace_root": str(prepared_workspace),
            "strict_thermal_root": str(strict_thermal_root),
            "train_union_list": str(train_union_list),
            "probe_list": str(probe_list),
            "reference_fused_ply": str(fused_ply),
            "reference_mesher_input_ply": str(mesher_fused_ply),
            "reference_mesh_path": str(mesh_path),
            "reference_mesh_backend": mesh_backend,
            "reference_depth_manifest": str(ref_manifest_path),
            "roi_path": str(roi_path),
            "camera_manifest_path": str(camera_manifest_path),
            "thresholds_m": thresholds_m,
            "support_rule": {
                "min_support_count": int(args.support_min_count),
                "support_radius_px": int(args.support_radius_px),
                "support_depth_tolerance_m": float(args.support_depth_tolerance_m),
            },
            "reference_construction_overrides": {
                "patch_match_max_image_size": int(args.patch_match_max_image_size),
                "patch_match_auto_source_count": args.patch_match_auto_source_count,
                "patch_match_window_radius": args.patch_match_window_radius,
                "patch_match_num_iterations": args.patch_match_num_iterations,
                "patch_match_geom_consistency": args.patch_match_geom_consistency,
                "patch_match_filter": args.patch_match_filter,
                "patch_match_min_triangulation_angle": args.patch_match_min_triangulation_angle,
                "patch_match_filter_min_triangulation_angle": args.patch_match_filter_min_triangulation_angle,
                "patch_match_filter_min_num_consistent": args.patch_match_filter_min_num_consistent,
                "patch_match_filter_min_ncc": args.patch_match_filter_min_ncc,
                "patch_match_filter_geom_consistency_max_cost": args.patch_match_filter_geom_consistency_max_cost,
                "stereo_fusion_max_image_size": args.stereo_fusion_max_image_size,
                "stereo_fusion_min_num_pixels": args.stereo_fusion_min_num_pixels,
                "stereo_fusion_max_reproj_error": args.stereo_fusion_max_reproj_error,
                "stereo_fusion_max_depth_error": args.stereo_fusion_max_depth_error,
                "stereo_fusion_max_normal_error": args.stereo_fusion_max_normal_error,
                "mesh_backend_preference": str(args.mesh_backend_preference),
                "poisson_depth": args.poisson_depth,
                "poisson_trim": args.poisson_trim,
                "poisson_point_weight": args.poisson_point_weight,
                "mesh_crop_fused_to_roi": bool(args.mesh_crop_fused_to_roi),
                "mesh_crop_manifest": mesh_crop_manifest,
            },
        },
    )
    print(f"REFERENCE_DEPTH_MANIFEST {ref_manifest_path}")
    print(f"REFERENCE_BUILD_MANIFEST {build_manifest_path}")


if __name__ == "__main__":
    main()
