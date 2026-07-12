from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.read_write_model import read_model


PROTOCOL_NAME = "pose-controlled-cross-subset-geometric-repeatability-v1"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, default=_json_default)
        f.write("\n")


def _load_name_list(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            name = raw_line.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"List file is empty: {path}")
    return names


def _write_name_list(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")


def _resolve_executable(exe: str) -> str:
    expanded = os.path.expandvars(exe)
    if os.path.isabs(expanded) and os.path.exists(expanded):
        return expanded
    candidates = [expanded]
    if os.name == "nt" and not expanded.lower().endswith((".exe", ".cmd", ".bat")):
        candidates = [expanded, expanded + ".exe", expanded + ".cmd", expanded + ".bat"]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    return expanded


def _should_use_shell(resolved_exe: str) -> bool:
    if os.name != "nt":
        return False
    low = resolved_exe.lower()
    return low.endswith(".bat") or low.endswith(".cmd")


def _run_cmd(cmd: Sequence[str], cwd: Path | None = None) -> None:
    if not cmd:
        raise ValueError("Empty command")
    resolved0 = _resolve_executable(str(cmd[0]))
    use_shell = _should_use_shell(resolved0)
    if use_shell:
        cmd_text = " ".join(f'"{token}"' if " " in str(token) else str(token) for token in [resolved0, *cmd[1:]])
        print(f"RUN(shell): {cmd_text}", flush=True)
        subprocess.run(cmd_text, cwd=str(cwd) if cwd else None, check=True, shell=True)
    else:
        resolved_cmd = [resolved0, *[str(token) for token in cmd[1:]]]
        print("RUN:", " ".join(resolved_cmd), flush=True)
        subprocess.run(resolved_cmd, cwd=str(cwd) if cwd else None, check=True)


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

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
    raise ValueError(f"Unsupported link mode: {mode}")


def _build_named_view_dir(source_dir: Path, required_names: Sequence[str], out_dir: Path, link_mode: str) -> Dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = [p for p in source_dir.iterdir() if p.is_file()]
    by_exact = {p.name: p for p in src_files}
    by_stem_ci: Dict[str, Path] = {}
    for p in src_files:
        key = p.stem.lower()
        if key not in by_stem_ci:
            by_stem_ci[key] = p

    missing: List[str] = []
    linked = 0
    for required_name in required_names:
        rel = Path(required_name)
        src = by_exact.get(required_name)
        if src is None:
            src = by_stem_ci.get(rel.stem.lower())
        if src is None:
            missing.append(required_name)
            continue
        dst = out_dir / rel
        _link_or_copy(src, dst, mode=link_mode)
        linked += 1
    return {
        "out_dir": str(out_dir),
        "requested_count": len(required_names),
        "linked_count": linked,
        "missing_count": len(missing),
        "missing_names": missing,
        "link_mode": link_mode,
    }


def _detect_model_ext(model_dir: Path) -> str:
    if (model_dir / "images.bin").exists() and (model_dir / "cameras.bin").exists():
        return ".bin"
    if (model_dir / "images.txt").exists() and (model_dir / "cameras.txt").exists():
        return ".txt"
    raise FileNotFoundError(f"Could not detect COLMAP model format under {model_dir}")


def _read_model_image_names(model_dir: Path) -> List[str]:
    _, images, _ = read_model(str(model_dir), ext=_detect_model_ext(model_dir))
    return sorted(str(image.name) for image in images.values())


def _count_model_points(model_dir: Path) -> int:
    _, _, points3d = read_model(str(model_dir), ext=_detect_model_ext(model_dir))
    return int(len(points3d))


def _select_best_sparse_model(sparse_root: Path) -> Path:
    candidates: List[Tuple[int, int, Path]] = []
    for child in sparse_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        try:
            image_count = len(_read_model_image_names(child))
            point_count = _count_model_points(child)
        except Exception:
            continue
        candidates.append((image_count, point_count, child))
    if not candidates:
        raise RuntimeError(f"No readable sparse model under {sparse_root}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _ensure_sparse_0(sparse_dir: Path) -> Path:
    model0 = sparse_dir / "0"
    if model0.is_dir() and ((model0 / "cameras.bin").exists() or (model0 / "cameras.txt").exists()):
        return model0

    direct_files = [p for p in sparse_dir.iterdir() if p.is_file()]
    if direct_files:
        model0.mkdir(parents=True, exist_ok=True)
        for fp in direct_files:
            shutil.move(str(fp), str(model0 / fp.name))
        return model0

    subdirs = [p for p in sparse_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if len(subdirs) == 1:
        return subdirs[0]
    raise RuntimeError(f"Could not normalize sparse/0 under {sparse_dir}")


def _query_single_camera_id(database_path: Path) -> int:
    con = sqlite3.connect(str(database_path))
    try:
        cur = con.cursor()
        cur.execute("SELECT camera_id FROM cameras ORDER BY camera_id ASC")
        rows = [int(row[0]) for row in cur.fetchall()]
    finally:
        con.close()
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one camera in DB {database_path}, got {rows}")
    return rows[0]


def _run_matcher(colmap_executable: str, database_path: Path, matching: str, matcher_args: str) -> None:
    cmd: List[str]
    if matching == "spatial":
        cmd = [str(colmap_executable), "spatial_matcher", "--database_path", str(database_path)]
    elif matching == "exhaustive":
        cmd = [str(colmap_executable), "exhaustive_matcher", "--database_path", str(database_path)]
    elif matching == "sequential":
        cmd = [str(colmap_executable), "sequential_matcher", "--database_path", str(database_path)]
    elif matching == "vocab_tree":
        cmd = [str(colmap_executable), "vocab_tree_matcher", "--database_path", str(database_path)]
    else:
        raise ValueError(f"Unsupported matching mode: {matching}")
    if matcher_args:
        cmd.extend(str(matcher_args).split())
    _run_cmd(cmd, cwd=REPO_ROOT)


def _try_read_registered_names(model_dir: Path) -> List[str] | None:
    try:
        return _read_model_image_names(model_dir)
    except Exception:
        return None


def _validate_exact_name_set(required_names: Sequence[str], actual_names: Sequence[str], label: str) -> None:
    required = list(required_names)
    actual = list(actual_names)
    required_set = set(required)
    actual_set = set(actual)
    missing = sorted(required_set - actual_set)
    extra = sorted(actual_set - required_set)
    if missing or extra:
        sample_missing = ", ".join(missing[:8]) if missing else "-"
        sample_extra = ", ".join(extra[:8]) if extra else "-"
        raise RuntimeError(
            f"{label} mismatch. missing={len(missing)} sample_missing=[{sample_missing}] "
            f"extra={len(extra)} sample_extra=[{sample_extra}]"
        )


def _has_nonempty_dataset_root(root: Path) -> bool:
    return (
        root.exists()
        and (root / "images").is_dir()
        and any((root / "images").iterdir())
        and (root / "sparse").is_dir()
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Materialize a strict train-union-only shared frame and final strict RGB/T datasets")
    ap.add_argument("--scene_name", required=True)
    ap.add_argument("--source_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--colmap_executable", default="colmap")
    ap.add_argument("--exiftool_executable", default="exiftool")
    ap.add_argument("--camera", default="SIMPLE_RADIAL")
    ap.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    ap.add_argument("--matcher_args", default="--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500")
    ap.add_argument("--mapper_multiple_models", type=int, default=1)
    ap.add_argument("--min_model_size", type=int, default=5)
    ap.add_argument("--init_min_num_inliers", type=int, default=50)
    ap.add_argument("--abs_pose_min_num_inliers", type=int, default=20)
    ap.add_argument("--use_model_aligner", action="store_true")
    ap.add_argument("--model_aligner_args", default="--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0")
    ap.add_argument("--wgs84_code", type=int, default=0)
    ap.add_argument("--prior_position_std_m", type=float, default=1.0)
    ap.add_argument("--rgb_link_mode", default="hardlink", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--thermal_link_mode", default="symlink", choices=["copy", "hardlink", "symlink"])
    ap.add_argument("--thermal_source_dir", default="")
    ap.add_argument("--force_refresh", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    source_root = Path(args.source_root).resolve()
    split_dir = Path(args.split_dir).resolve()
    out_root = Path(args.out_root).resolve()
    status_json = out_root / "status.json"
    manifest_json = out_root / "strict_protocol_manifest.json"

    rgb_source_input = source_root / "input"
    thermal_source_dir = Path(args.thermal_source_dir).resolve() if args.thermal_source_dir else (source_root / "thermal").resolve()

    probe_list_path = split_dir / "probe_test.txt"
    train_union_list_path = split_dir / "train_union.txt"
    train_odd_list_path = split_dir / "train_odd.txt"
    train_even_list_path = split_dir / "train_even.txt"

    if not rgb_source_input.exists():
        raise FileNotFoundError(f"RGB input dir not found: {rgb_source_input}")
    if not thermal_source_dir.exists():
        raise FileNotFoundError(f"Thermal source dir not found: {thermal_source_dir}")

    train_union_names = _load_name_list(train_union_list_path)
    probe_names = _load_name_list(probe_list_path)

    work_root = out_root / "workspace"
    union_root = work_root / "rgb_train_union_source"
    union_input_dir = union_root / "input"
    all_rgb_input_dir = work_root / "rgb_union_plus_probe_input"
    thermal_alias_dir = work_root / "thermal_union_plus_probe_alias"
    train_union_aligned_model = union_root / "distorted" / "sparse_aligned"
    train_union_sparse_root = union_root / "distorted" / "sparse"
    registered_model_dir = work_root / "registered_probe_model"
    strict_rgb_root = out_root / "rgb_strict"
    strict_thermal_root = out_root / "thermal_strict"
    artifact_dir = out_root / "artifacts"
    linked_probe_list_path = work_root / "probe_test.txt"
    linked_train_union_list_path = work_root / "train_union.txt"

    out_root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    status: Dict[str, Any] = {
        "protocol_name": PROTOCOL_NAME,
        "scene_name": args.scene_name,
        "stage": "starting",
        "updated_at": None,
        "out_root": str(out_root),
        "source_root": str(source_root),
    }

    def write_status(stage: str, extra: Dict[str, Any] | None = None) -> None:
        status["stage"] = stage
        status["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if extra:
            status.update(extra)
        _save_json(status_json, status)

    try:
        if args.force_refresh:
            for path in (all_rgb_input_dir, thermal_alias_dir, registered_model_dir, strict_rgb_root, strict_thermal_root):
                if path.exists():
                    shutil.rmtree(path)

        write_status("build_train_union_input")
        expected_union_input = union_input_dir.exists() and len(list(union_input_dir.rglob("*.*"))) == len(train_union_names)
        if not expected_union_input or args.force_refresh:
            _build_named_view_dir(
                source_dir=rgb_source_input,
                required_names=train_union_names,
                out_dir=union_input_dir,
                link_mode=args.rgb_link_mode,
            )
        _write_name_list(linked_train_union_list_path, train_union_names)
        _write_name_list(linked_probe_list_path, probe_names)

        write_status("convert_gtgs_train_union")
        union_model0 = union_root / "sparse" / "0"
        if args.force_refresh and union_root.exists():
            for child in ("distorted", "images", "sparse"):
                target = union_root / child
                if target.exists():
                    shutil.rmtree(target)
        if not ((union_model0 / "images.bin").exists() or (union_model0 / "images.txt").exists()):
            convert_cmd = [
                sys.executable,
                str((REPO_ROOT / "convert-gtgs.py").resolve()),
                "-s", str(union_root),
                "--colmap_executable", str(args.colmap_executable),
                "--exiftool_executable", str(args.exiftool_executable),
                "--wgs84_code", str(args.wgs84_code),
                "--prior_position_std_m", str(args.prior_position_std_m),
                "--camera", str(args.camera),
                "--matching", str(args.matching),
                "--matcher_args", str(args.matcher_args),
                "--mapper_multiple_models", str(args.mapper_multiple_models),
                "--min_model_size", str(args.min_model_size),
                "--init_min_num_inliers", str(args.init_min_num_inliers),
                "--abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
            ]
            if args.use_model_aligner:
                convert_cmd.extend(["--use_model_aligner", "--model_aligner_args", str(args.model_aligner_args)])
            _run_cmd(convert_cmd, cwd=REPO_ROOT)

        if train_union_aligned_model.exists() and ((train_union_aligned_model / "images.bin").exists() or (train_union_aligned_model / "images.txt").exists()):
            shared_frame_model_dir = train_union_aligned_model
        else:
            shared_frame_model_dir = _select_best_sparse_model(train_union_sparse_root)
        train_union_registered_names = _read_model_image_names(shared_frame_model_dir)
        _validate_exact_name_set(train_union_names, train_union_registered_names, "train_union shared frame")

        write_status("build_all_rgb_input")
        all_required_names = train_union_names + probe_names
        expected_all_rgb = all_rgb_input_dir.exists() and len(list(all_rgb_input_dir.rglob("*.*"))) == len(all_required_names)
        if not expected_all_rgb or args.force_refresh:
            rgb_link_result = _build_named_view_dir(
                source_dir=rgb_source_input,
                required_names=all_required_names,
                out_dir=all_rgb_input_dir,
                link_mode=args.rgb_link_mode,
            )
            if rgb_link_result["missing_count"] != 0:
                raise RuntimeError(f"Missing RGB images when building all-view input: {rgb_link_result['missing_names'][:8]}")

        write_status("register_probe_views")
        registered_names = None if args.force_refresh else _try_read_registered_names(registered_model_dir)
        registered_images_ready = registered_names is not None and set(registered_names) == set(all_required_names)
        if not registered_images_ready:
            database_path = union_root / "distorted" / "database.db"
            if not database_path.exists():
                raise FileNotFoundError(f"COLMAP database not found: {database_path}")
            camera_id = _query_single_camera_id(database_path)
            _run_cmd(
                [
                    str(args.colmap_executable), "feature_extractor",
                    "--database_path", str(database_path),
                    "--image_path", str(all_rgb_input_dir),
                    "--image_list_path", str(linked_probe_list_path),
                    "--ImageReader.existing_camera_id", str(camera_id),
                    "--ImageReader.single_camera", "1",
                ],
                cwd=REPO_ROOT,
            )
            write_status("match_probe_views")
            _run_matcher(
                colmap_executable=str(args.colmap_executable),
                database_path=database_path,
                matching=str(args.matching),
                matcher_args=str(args.matcher_args),
            )
            write_status("register_probe_views")
            if registered_model_dir.exists():
                shutil.rmtree(registered_model_dir)
            registered_model_dir.mkdir(parents=True, exist_ok=True)
            _run_cmd(
                [
                    str(args.colmap_executable), "image_registrator",
                    "--database_path", str(database_path),
                    "--input_path", str(shared_frame_model_dir),
                    "--output_path", str(registered_model_dir),
                    "--Mapper.fix_existing_frames", "1",
                    "--Mapper.ba_refine_focal_length", "0",
                    "--Mapper.ba_refine_principal_point", "0",
                    "--Mapper.ba_refine_extra_params", "0",
                    "--Mapper.abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
                ],
                cwd=REPO_ROOT,
            )
            registered_names = _read_model_image_names(registered_model_dir)
        else:
            print("REUSE: registered probe model already contains full train-union+probe set", flush=True)

        if registered_names is None:
            registered_names = _read_model_image_names(registered_model_dir)
        _validate_exact_name_set(all_required_names, registered_names, "registered train_union+probe model")

        write_status("undistort_rgb_strict")
        if not _has_nonempty_dataset_root(strict_rgb_root) or args.force_refresh:
            if strict_rgb_root.exists():
                shutil.rmtree(strict_rgb_root)
            _run_cmd(
                [
                    str(args.colmap_executable), "image_undistorter",
                    "--image_path", str(all_rgb_input_dir),
                    "--input_path", str(registered_model_dir),
                    "--output_path", str(strict_rgb_root),
                    "--output_type", "COLMAP",
                ],
                cwd=REPO_ROOT,
            )
            _ensure_sparse_0(strict_rgb_root / "sparse")

        write_status("build_thermal_alias")
        expected_thermal_alias = thermal_alias_dir.exists() and len(list(thermal_alias_dir.rglob("*.*"))) == len(all_required_names)
        if not expected_thermal_alias or args.force_refresh:
            thermal_link_result = _build_named_view_dir(
                source_dir=thermal_source_dir,
                required_names=all_required_names,
                out_dir=thermal_alias_dir,
                link_mode=args.thermal_link_mode,
            )
            if thermal_link_result["missing_count"] != 0:
                raise RuntimeError(f"Missing thermal images when building alias dir: {thermal_link_result['missing_names'][:8]}")

        write_status("undistort_thermal_strict")
        if not _has_nonempty_dataset_root(strict_thermal_root) or args.force_refresh:
            if strict_thermal_root.exists():
                shutil.rmtree(strict_thermal_root)
            _run_cmd(
                [
                    str(args.colmap_executable), "image_undistorter",
                    "--image_path", str(thermal_alias_dir),
                    "--input_path", str(registered_model_dir),
                    "--output_path", str(strict_thermal_root),
                    "--output_type", "COLMAP",
                ],
                cwd=REPO_ROOT,
            )
            _ensure_sparse_0(strict_thermal_root / "sparse")

        train_union_points_path = shared_frame_model_dir / "points3D.bin"
        if not train_union_points_path.exists():
            train_union_points_path = shared_frame_model_dir / "points3D.txt"
        if not train_union_points_path.exists():
            raise FileNotFoundError(f"Could not find train-union points3D file under {shared_frame_model_dir}")

        protocol_manifest = {
            "protocol_name": PROTOCOL_NAME,
            "scene_name": args.scene_name,
            "source_root": str(source_root),
            "source_rgb_input": str(rgb_source_input),
            "source_thermal_dir": str(thermal_source_dir),
            "split_dir": str(split_dir),
            "lists": {
                "probe_test": str(probe_list_path),
                "train_union": str(train_union_list_path),
                "train_odd": str(train_odd_list_path),
                "train_even": str(train_even_list_path),
            },
            "shared_frame_rule": {
                "type": "train_union_only_convert_gtgs_reconstruction",
                "convert_gtgs_camera": str(args.camera),
                "matching": str(args.matching),
                "matcher_args": str(args.matcher_args),
                "use_model_aligner": bool(args.use_model_aligner),
                "model_aligner_args": str(args.model_aligner_args),
            },
            "probe_pose_assignment_rule": {
                "type": "post_registration_into_frozen_train_union_frame",
                "tool": "COLMAP image_registrator",
                "fix_existing_frames": True,
                "ba_refine_focal_length": False,
                "ba_refine_principal_point": False,
                "ba_refine_extra_params": False,
            },
            "counts": {
                "train_union_expected": len(train_union_names),
                "probe_expected": len(probe_names),
                "registered_total": len(registered_names),
            },
            "artifacts": {
                "workspace_root": str(work_root),
                "train_union_source_root": str(union_root),
                "shared_frame_model_dir": str(shared_frame_model_dir),
                "registered_model_dir": str(registered_model_dir),
                "strict_rgb_root": str(strict_rgb_root),
                "strict_thermal_root": str(strict_thermal_root),
                "roi_source_points_path": str(train_union_points_path),
                "rgb_all_views_input_dir": str(all_rgb_input_dir),
                "thermal_alias_dir": str(thermal_alias_dir),
            },
        }
        _save_json(manifest_json, protocol_manifest)

        write_status(
            "completed",
            {
                "completed": True,
                "manifest": str(manifest_json),
                "strict_rgb_root": str(strict_rgb_root),
                "strict_thermal_root": str(strict_thermal_root),
                "roi_source_points_path": str(train_union_points_path),
            },
        )
        print(f"STRICT_PROTOCOL_DATASET_READY {manifest_json}", flush=True)
    except Exception as exc:
        write_status("failed", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
