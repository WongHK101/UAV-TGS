from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


PROTOCOL_NAME = "pose-controlled-cross-subset-geometric-repeatability-v1"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def _relpath_or_abs(target: Path, anchor: Path) -> str:
    try:
        return str(target.resolve().relative_to(anchor.resolve())).replace("\\", "/")
    except Exception:
        return str(target.resolve())


def _assert_same_view_geometry(a: Dict[str, Any], b: Dict[str, Any], idx: int) -> None:
    keys = ("view_id", "width", "height", "fx", "fy", "cx", "cy", "camera_to_world")
    for key in keys:
        if a[key] != b[key]:
            raise ValueError(f"Odd/even manifest mismatch for view index {idx}, key {key!r}")


def build_scene_manifest(odd_manifest_path: Path, even_manifest_path: Path, roi_path: Path, out_path: Path) -> Path:
    odd = _load_json(odd_manifest_path)
    even = _load_json(even_manifest_path)
    if odd.get("bundle_type") != "gaussian_probe_split_bundle_v1":
        raise ValueError("Unexpected odd bundle type")
    if even.get("bundle_type") != "gaussian_probe_split_bundle_v1":
        raise ValueError("Unexpected even bundle type")
    odd_views = odd.get("views", [])
    even_views = even.get("views", [])
    odd_depth_semantics = odd.get("depth_semantics")
    even_depth_semantics = even.get("depth_semantics")
    if odd_depth_semantics != even_depth_semantics:
        raise ValueError(
            f"Odd/even depth semantics mismatch: {odd_depth_semantics!r} vs {even_depth_semantics!r}"
        )
    if len(odd_views) != len(even_views):
        raise ValueError(f"Odd/even view count mismatch: {len(odd_views)} vs {len(even_views)}")
    scene_name = odd.get("scene_name") or even.get("scene_name") or "UnknownScene"
    out_dir = out_path.parent.resolve()
    manifest_views: List[Dict[str, Any]] = []
    for idx, (ov, ev) in enumerate(zip(odd_views, even_views)):
        _assert_same_view_geometry(ov, ev, idx)
        odd_npz = (odd_manifest_path.parent / ov["npz_file"]).resolve()
        even_npz = (even_manifest_path.parent / ev["npz_file"]).resolve()
        manifest_views.append(
            {
                "view_id": ov["view_id"],
                "width": ov["width"],
                "height": ov["height"],
                "fx": ov["fx"],
                "fy": ov["fy"],
                "cx": ov["cx"],
                "cy": ov["cy"],
                "camera_to_world": ov["camera_to_world"],
                "odd_file": _relpath_or_abs(odd_npz, out_dir),
                "even_file": _relpath_or_abs(even_npz, out_dir),
            }
        )

    scene_manifest = {
        "protocol_name": PROTOCOL_NAME,
        "scene_name": scene_name,
        "roi_path": _relpath_or_abs(roi_path.resolve(), out_dir),
        "depth_semantics": odd_depth_semantics,
        "distance_domain": "after_roi_crop_and_after_voxel_downsampling",
        "validity_rule": {
            "mode": "opacity_threshold",
            "opacity_threshold": 0.5,
            "depth_min": 1e-6,
        },
        "odd_bundle": str(odd_manifest_path.resolve()),
        "even_bundle": str(even_manifest_path.resolve()),
        "views": manifest_views,
    }
    _save_json(out_path, scene_manifest)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build evaluator scene manifest from odd/even split bundles")
    ap.add_argument("--odd_manifest", required=True)
    ap.add_argument("--even_manifest", required=True)
    ap.add_argument("--roi_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = build_scene_manifest(
        odd_manifest_path=Path(args.odd_manifest).resolve(),
        even_manifest_path=Path(args.even_manifest).resolve(),
        roi_path=Path(args.roi_path).resolve(),
        out_path=Path(args.out).resolve(),
    )
    print(f"SCENE_MANIFEST_SAVED {out_path}")


if __name__ == "__main__":
    main()
