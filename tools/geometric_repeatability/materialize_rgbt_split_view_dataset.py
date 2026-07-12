from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List


def _read_name_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path) -> str:
    _ensure_dir(dst.parent)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _materialize_subset(
    names: Iterable[str],
    rgb_src_dir: Path,
    thermal_src_dir: Path,
    rgb_dst_dir: Path,
    thermal_dst_dir: Path,
    stats: Dict[str, int],
) -> None:
    for name in names:
        rgb_src = rgb_src_dir / name
        thermal_src = thermal_src_dir / name
        if not rgb_src.exists():
            raise FileNotFoundError(f"RGB source file missing: {rgb_src}")
        if not thermal_src.exists():
            raise FileNotFoundError(f"Thermal source file missing: {thermal_src}")
        for src, dst, prefix in (
            (rgb_src, rgb_dst_dir / name, "rgb"),
            (thermal_src, thermal_dst_dir / name, "thermal"),
        ):
            mode = _link_or_copy(src, dst)
            if mode == "hardlink":
                stats[f"{prefix}_hardlinks"] += 1
            elif mode == "copy":
                stats[f"{prefix}_copies"] += 1
            else:
                stats[f"{prefix}_existing"] += 1


def _copy_sparse_tree(src_root: Path, dst_root: Path, stats: Dict[str, int]) -> None:
    if not src_root.exists():
        raise FileNotFoundError(f"Sparse source root missing: {src_root}")
    for src in src_root.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(src_root)
        mode = _link_or_copy(src, dst_root / rel)
        if mode == "hardlink":
            stats["sparse_hardlinks"] += 1
        elif mode == "copy":
            stats["sparse_copies"] += 1
        else:
            stats["sparse_existing"] += 1


def _write_json(path: Path, payload: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Materialize a train/test RGB+thermal split-view dataset root for methods that expect folder-based splits."
    )
    ap.add_argument("--rgb_src_dir", required=True)
    ap.add_argument("--thermal_src_dir", required=True)
    ap.add_argument("--sparse_src_root", required=True)
    ap.add_argument("--train_list", required=True)
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    rgb_src_dir = Path(args.rgb_src_dir).resolve()
    thermal_src_dir = Path(args.thermal_src_dir).resolve()
    sparse_src_root = Path(args.sparse_src_root).resolve()
    train_list = Path(args.train_list).resolve()
    test_list = Path(args.test_list).resolve()
    out_root = Path(args.out_root).resolve()

    if not rgb_src_dir.exists():
        raise FileNotFoundError(f"RGB source dir missing: {rgb_src_dir}")
    if not thermal_src_dir.exists():
        raise FileNotFoundError(f"Thermal source dir missing: {thermal_src_dir}")
    if not train_list.exists():
        raise FileNotFoundError(f"Train list missing: {train_list}")
    if not test_list.exists():
        raise FileNotFoundError(f"Test list missing: {test_list}")

    train_names = _read_name_list(train_list)
    test_names = _read_name_list(test_list)
    overlap = sorted(set(train_names) & set(test_names))
    if overlap:
        sample = ", ".join(overlap[:8])
        raise ValueError(f"Train/test lists overlap: {sample}")

    stats = {
        "rgb_hardlinks": 0,
        "rgb_copies": 0,
        "rgb_existing": 0,
        "thermal_hardlinks": 0,
        "thermal_copies": 0,
        "thermal_existing": 0,
        "sparse_hardlinks": 0,
        "sparse_copies": 0,
        "sparse_existing": 0,
    }

    _materialize_subset(
        names=train_names,
        rgb_src_dir=rgb_src_dir,
        thermal_src_dir=thermal_src_dir,
        rgb_dst_dir=out_root / "rgb" / "train",
        thermal_dst_dir=out_root / "thermal" / "train",
        stats=stats,
    )
    _materialize_subset(
        names=test_names,
        rgb_src_dir=rgb_src_dir,
        thermal_src_dir=thermal_src_dir,
        rgb_dst_dir=out_root / "rgb" / "test",
        thermal_dst_dir=out_root / "thermal" / "test",
        stats=stats,
    )
    _copy_sparse_tree(sparse_src_root, out_root / "sparse", stats=stats)

    manifest = {
        "dataset_type": "rgbt_folder_split_v1",
        "out_root": str(out_root),
        "rgb_src_dir": str(rgb_src_dir),
        "thermal_src_dir": str(thermal_src_dir),
        "sparse_src_root": str(sparse_src_root),
        "train_list": str(train_list),
        "test_list": str(test_list),
        "train_count": len(train_names),
        "test_count": len(test_names),
        "stats": stats,
    }
    manifest_path = out_root / "split_dataset_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"SPLIT_DATASET_READY {manifest_path}")


if __name__ == "__main__":
    main()
