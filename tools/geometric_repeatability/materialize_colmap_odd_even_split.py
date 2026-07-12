from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text


def _load_colmap_image_names(source_path: Path) -> List[str]:
    images_bin = source_path / "sparse" / "0" / "images.bin"
    images_txt = source_path / "sparse" / "0" / "images.txt"
    if images_bin.exists():
        cam_extrinsics = read_extrinsics_binary(str(images_bin))
    elif images_txt.exists():
        cam_extrinsics = read_extrinsics_text(str(images_txt))
    else:
        raise FileNotFoundError(f"Could not find COLMAP images.bin/images.txt under {source_path}")
    return sorted(extr.name for extr in cam_extrinsics.values())


def _resolve_optional_path(base_dir: Path, path_text: str | None) -> Path | None:
    if not path_text:
        return None
    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_name_list(list_path: Path) -> List[str]:
    if not list_path.exists():
        raise FileNotFoundError(f"List file not found: {list_path}")
    names: List[str] = []
    with list_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            name = raw_line.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"List file is empty: {list_path}")
    return names


def _write_name_list(path: Path, names: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _infer_scene_name(source_path: Path) -> str:
    if source_path.name.lower() in {"thermal_ud", "rgb_ud", "thermal", "rgb", "images"} and source_path.parent.name:
        return source_path.parent.name
    return source_path.name


def build_split(source_path: Path, llffhold: int, explicit_test_file: Path | None) -> Dict:
    sorted_names = _load_colmap_image_names(source_path)
    if explicit_test_file is not None:
        probe_names = _load_name_list(explicit_test_file)
        probe_rule = {
            "mode": "explicit_test_file",
            "path": str(explicit_test_file),
        }
    elif llffhold > 0:
        probe_names = [name for idx, name in enumerate(sorted_names) if idx % llffhold == 0]
        probe_rule = {
            "mode": "llffhold",
            "llffhold": int(llffhold),
            "indexing": "0-based over sorted COLMAP image names",
        }
    else:
        fallback_test = source_path / "sparse" / "0" / "test.txt"
        probe_names = _load_name_list(fallback_test)
        probe_rule = {
            "mode": "fallback_test_txt",
            "path": str(fallback_test),
        }

    known_names = set(sorted_names)
    unknown_probe = sorted(set(probe_names) - known_names)
    if unknown_probe:
        sample = ", ".join(unknown_probe[:8])
        raise ValueError(f"Probe split contains unknown camera names: {sample}")

    probe_set = set(probe_names)
    train_union = [name for name in sorted_names if name not in probe_set]
    train_odd = train_union[0::2]
    train_even = train_union[1::2]

    if set(train_odd) & set(train_even):
        raise AssertionError("Odd/even train splits overlap unexpectedly")
    if (set(train_odd) | set(train_even)) != set(train_union):
        raise AssertionError("Odd/even train splits do not cover the union train set")

    return {
        "source_path": str(source_path),
        "all_sorted_names": sorted_names,
        "probe_rule": probe_rule,
        "probe_test": probe_names,
        "train_union": train_union,
        "train_odd": train_odd,
        "train_even": train_even,
        "odd_even_rule": "1-based odd/even positions over sorted train_union after removing fixed probe_test views",
        "counts": {
            "all_views": len(sorted_names),
            "probe_test": len(probe_names),
            "train_union": len(train_union),
            "train_odd": len(train_odd),
            "train_even": len(train_even),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize explicit train/probe odd-even split files for a COLMAP scene")
    ap.add_argument("--source_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--llffhold", type=int, default=8)
    ap.add_argument("--test_file", default="")
    args = ap.parse_args()

    source_path = Path(args.source_path).resolve()
    out_dir = Path(args.out_dir).resolve()
    explicit_test_file = _resolve_optional_path(source_path, args.test_file)
    split = build_split(source_path=source_path, llffhold=int(args.llffhold), explicit_test_file=explicit_test_file)

    manifest = {
        "protocol_name": "pose-controlled-cross-subset-geometric-repeatability-v1",
        "scene_name": _infer_scene_name(source_path),
        "source_path": str(source_path),
        "probe_rule": split["probe_rule"],
        "odd_even_rule": split["odd_even_rule"],
        "counts": split["counts"],
        "files": {
            "probe_test": "probe_test.txt",
            "train_union": "train_union.txt",
            "train_odd": "train_odd.txt",
            "train_even": "train_even.txt",
        },
    }

    _write_name_list(out_dir / "probe_test.txt", split["probe_test"])
    _write_name_list(out_dir / "train_union.txt", split["train_union"])
    _write_name_list(out_dir / "train_odd.txt", split["train_odd"])
    _write_name_list(out_dir / "train_even.txt", split["train_even"])
    _save_json(out_dir / "split_manifest.json", manifest)

    print(f"SPLIT_SAVED {out_dir}")
    for key, value in split["counts"].items():
        print(f"{key.upper()} {value}")


if __name__ == "__main__":
    main()
