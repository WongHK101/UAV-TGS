from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from numba import njit
from plyfile import PlyData
from scipy.ndimage import maximum_filter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def parse_thresholds_m(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if str(x).strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    if any(v <= 0 for v in values):
        raise ValueError(f"Thresholds must be positive, got {values!r}")
    return values


def compute_scaled_resolution(orig_w: int, orig_h: int, resolution_arg: int) -> Tuple[int, int]:
    if resolution_arg in (1, 2, 4, 8):
        return int(round(orig_w / resolution_arg)), int(round(orig_h / resolution_arg))
    if resolution_arg == -1:
        if orig_w > 1600:
            scale = float(orig_w) / 1600.0
            return int(orig_w / scale), int(orig_h / scale)
        return int(orig_w), int(orig_h)
    scale = float(orig_w) / float(resolution_arg)
    return int(orig_w / scale), int(orig_h / scale)


def _camera_to_world_from_caminfo(cam_info) -> np.ndarray:
    from utils.graphics_utils import getWorld2View2

    w2c = getWorld2View2(cam_info.R, cam_info.T, np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0).astype(np.float64)
    return np.linalg.inv(w2c)


def build_probe_view_manifest(
    *,
    source_path: Path,
    images_dir_name: str,
    resolution_arg: int,
    train_list: Path,
    test_list: Path,
    scene_name: str,
) -> Dict[str, Any]:
    # Keep heavy training/runtime imports lazy so --help and --dry_run do not
    # require compiled Gaussian-splatting CUDA extensions.
    from scene.dataset_readers import readColmapSceneInfo
    from utils.graphics_utils import fov2focal

    # The repo reader prints per-camera progress; suppress it so long dense runs can
    # safely continue even when stdout is redirected or detached.
    with contextlib.redirect_stdout(io.StringIO()):
        scene_info = readColmapSceneInfo(
            path=str(source_path),
            images=images_dir_name,
            depths="",
            eval=True,
            train_test_exp=False,
            train_list=str(train_list),
            test_list=str(test_list),
        )
    views: List[Dict[str, Any]] = []
    for idx, cam in enumerate(sorted(scene_info.test_cameras, key=lambda c: c.image_name)):
        width, height = compute_scaled_resolution(int(cam.width), int(cam.height), resolution_arg=resolution_arg)
        views.append(
            {
                "view_id": f"{idx:05d}",
                "image_name": str(cam.image_name),
                "width": int(width),
                "height": int(height),
                "fx": float(fov2focal(cam.FovX, width)),
                "fy": float(fov2focal(cam.FovY, height)),
                "cx": float(width / 2.0),
                "cy": float(height / 2.0),
                "camera_to_world": _camera_to_world_from_caminfo(cam).tolist(),
            }
        )
    return {
        "camera_manifest_type": "heldout_probe_camera_manifest_v1",
        "scene_name": str(scene_name),
        "source_path": str(source_path.resolve()),
        "images_dir_name": str(images_dir_name),
        "resolution_arg": int(resolution_arg),
        "train_list": str(train_list.resolve()),
        "test_list": str(test_list.resolve()),
        "views": views,
    }


def image_stem_key(name: str) -> str:
    return Path(str(name)).stem.lower()


def load_native_camera_entries(cameras_json_path: Path) -> Dict[str, Dict[str, Any]]:
    payload = load_json(cameras_json_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected cameras.json to contain a list, got {type(payload)!r}: {cameras_json_path}")
    entries: Dict[str, Dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("img_name", ""))
        if not raw_name:
            continue
        key = image_stem_key(raw_name)
        if key not in entries:
            entries[key] = item
    if not entries:
        raise ValueError(f"No camera entries with img_name found in {cameras_json_path}")
    return entries


def _rotation_delta_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    rel = np.asarray(rot_a, dtype=np.float64) @ np.asarray(rot_b, dtype=np.float64).T
    cos_angle = max(-1.0, min(1.0, (float(np.trace(rel)) - 1.0) / 2.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _estimate_rigid_transform(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if src_points.shape != dst_points.shape or src_points.ndim != 2 or src_points.shape[1] != 3:
        raise ValueError(f"Expected matched Nx3 point sets, got {src_points.shape} and {dst_points.shape}")
    src_mean = np.mean(src_points, axis=0)
    dst_mean = np.mean(dst_points, axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean
    cov = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    trans = dst_mean - rot @ src_mean
    return rot.astype(np.float64), trans.astype(np.float64)


def estimate_strict_to_native_world_transform(
    strict_views: Sequence[Dict[str, Any]],
    native_cameras_by_stem: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    strict_centers: List[np.ndarray] = []
    native_centers: List[np.ndarray] = []
    strict_rotations: List[np.ndarray] = []
    native_rotations: List[np.ndarray] = []
    common_names: List[str] = []

    for view in strict_views:
        image_name = str(view["image_name"])
        key = image_stem_key(image_name)
        native_entry = native_cameras_by_stem.get(key)
        if native_entry is None:
            continue
        strict_c2w = np.asarray(view["camera_to_world"], dtype=np.float64)
        strict_centers.append(strict_c2w[:3, 3])
        strict_rotations.append(strict_c2w[:3, :3])
        native_centers.append(np.asarray(native_entry["position"], dtype=np.float64))
        native_rotations.append(np.asarray(native_entry["rotation"], dtype=np.float64))
        common_names.append(image_name)

    if len(common_names) < 3:
        raise ValueError(f"Need at least 3 common cameras to estimate rigid transform, got {len(common_names)}")

    strict_centers_arr = np.asarray(strict_centers, dtype=np.float64)
    native_centers_arr = np.asarray(native_centers, dtype=np.float64)
    rot, trans = _estimate_rigid_transform(strict_centers_arr, native_centers_arr)

    transformed_centers = (rot @ strict_centers_arr.T).T + trans[None, :]
    center_residuals = transformed_centers - native_centers_arr
    center_errors = np.linalg.norm(center_residuals, axis=1)

    rotation_errors: List[float] = []
    for strict_rot, native_rot in zip(strict_rotations, native_rotations):
        predicted_rot = rot @ strict_rot
        rotation_errors.append(_rotation_delta_deg(predicted_rot, native_rot))

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = trans
    return {
        "common_count": int(len(common_names)),
        "common_image_names": list(common_names),
        "strict_to_native_transform": transform.tolist(),
        "translation_mean_xyz": np.mean(center_residuals, axis=0).tolist(),
        "translation_std_xyz": np.std(center_residuals, axis=0).tolist(),
        "translation_error_mean_m": float(np.mean(center_errors)),
        "translation_error_max_m": float(np.max(center_errors)),
        "rotation_error_mean_deg": float(np.mean(rotation_errors)),
        "rotation_error_max_deg": float(np.max(rotation_errors)),
    }


def apply_world_transform_to_camera_to_world(camera_to_world: np.ndarray, world_transform: np.ndarray) -> np.ndarray:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    transform = np.asarray(world_transform, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"camera_to_world must be 4x4, got {c2w.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"world_transform must be 4x4, got {transform.shape}")
    return transform @ c2w


def load_ply_points_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    return np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64, copy=False)


def load_ply_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertices = np.vstack([ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]]).T.astype(np.float64, copy=False)
    if "face" not in ply:
        raise ValueError(f"{path} has no face element")
    face_element = ply["face"]
    if "vertex_indices" in face_element.data.dtype.names:
        raw_faces = face_element.data["vertex_indices"]
    elif "vertex_index" in face_element.data.dtype.names:
        raw_faces = face_element.data["vertex_index"]
    else:
        raise ValueError(f"{path} face element has no vertex_indices field")
    faces = np.asarray([np.asarray(face, dtype=np.int64) for face in raw_faces], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"{path} must contain triangular faces, got shape {faces.shape}")
    return vertices, faces


def compute_quantile_bbox(
    points: np.ndarray,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    padding_ratio_of_robust_diagonal: float = 0.02,
) -> Dict[str, Any]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}")
    robust_min = np.quantile(points, lower_quantile, axis=0)
    robust_max = np.quantile(points, upper_quantile, axis=0)
    robust_diag = float(np.linalg.norm(robust_max - robust_min))
    padding = robust_diag * float(padding_ratio_of_robust_diagonal)
    bbox_min = robust_min - padding
    bbox_max = robust_max + padding
    scene_diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    return {
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "padding_ratio_of_robust_diagonal": float(padding_ratio_of_robust_diagonal),
        "bbox_min": bbox_min.astype(np.float64),
        "bbox_max": bbox_max.astype(np.float64),
        "scene_diagonal": scene_diagonal,
    }


def transform_world_to_camera(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    c2w = np.asarray(camera_to_world, dtype=np.float64)
    w2c = np.linalg.inv(c2w)
    return points_world @ w2c[:3, :3].T + w2c[:3, 3]


@njit(cache=True)
def _rasterize_depth_numba(
    vertices_cam: np.ndarray,
    faces: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> np.ndarray:
    depth = np.full((height, width), np.nan, dtype=np.float64)
    eps = 1e-12
    for face_idx in range(faces.shape[0]):
        i0 = faces[face_idx, 0]
        i1 = faces[face_idx, 1]
        i2 = faces[face_idx, 2]
        v0 = vertices_cam[i0]
        v1 = vertices_cam[i1]
        v2 = vertices_cam[i2]
        z0 = v0[2]
        z1 = v1[2]
        z2 = v2[2]
        if z0 <= 1e-8 or z1 <= 1e-8 or z2 <= 1e-8:
            continue

        x0 = fx * (v0[0] / z0) + cx
        y0 = fy * (v0[1] / z0) + cy
        x1 = fx * (v1[0] / z1) + cx
        y1 = fy * (v1[1] / z1) + cy
        x2 = fx * (v2[0] / z2) + cx
        y2 = fy * (v2[1] / z2) + cy

        tri_min_x = max(0, int(math.floor(min(x0, x1, x2))))
        tri_max_x = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        tri_min_y = max(0, int(math.floor(min(y0, y1, y2))))
        tri_max_y = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        if tri_min_x > tri_max_x or tri_min_y > tri_max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < eps:
            continue

        for py in range(tri_min_y, tri_max_y + 1):
            sample_y = py + 0.5
            for px in range(tri_min_x, tri_max_x + 1):
                sample_x = px + 0.5
                w0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denom
                w1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denom
                w2 = 1.0 - w0 - w1
                if w0 < -eps or w1 < -eps or w2 < -eps:
                    continue
                inv_z = w0 / z0 + w1 / z1 + w2 / z2
                if inv_z <= eps:
                    continue
                z = 1.0 / inv_z
                current = depth[py, px]
                if np.isnan(current) or z < current:
                    depth[py, px] = z
    return depth


def render_mesh_depth_for_view(vertices_world: np.ndarray, faces: np.ndarray, view: Dict[str, Any]) -> np.ndarray:
    vertices_cam = transform_world_to_camera(vertices_world, np.asarray(view["camera_to_world"], dtype=np.float64))
    return _rasterize_depth_numba(
        vertices_cam=vertices_cam,
        faces=np.asarray(faces, dtype=np.int64),
        fx=float(view["fx"]),
        fy=float(view["fy"]),
        cx=float(view["cx"]),
        cy=float(view["cy"]),
        width=int(view["width"]),
        height=int(view["height"]),
    )


@njit(cache=True)
def _project_support_points_numba(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth_tolerance_m: float,
) -> np.ndarray:
    nearest = np.full((height, width), np.inf, dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.int32)
    for idx in range(points_cam.shape[0]):
        x = points_cam[idx, 0]
        y = points_cam[idx, 1]
        z = points_cam[idx, 2]
        if z <= 1e-8:
            continue
        px = int(math.floor(fx * (x / z) + cx + 0.5))
        py = int(math.floor(fy * (y / z) + cy + 0.5))
        if px < 0 or px >= width or py < 0 or py >= height:
            continue
        current = nearest[py, px]
        if z < current - depth_tolerance_m:
            nearest[py, px] = z
            counts[py, px] = 1
        elif abs(z - current) <= depth_tolerance_m:
            counts[py, px] += 1
    return counts


def render_support_count_for_view(
    points_world: np.ndarray,
    view: Dict[str, Any],
    *,
    depth_tolerance_m: float,
    support_radius_px: int,
) -> np.ndarray:
    points_cam = transform_world_to_camera(points_world, np.asarray(view["camera_to_world"], dtype=np.float64))
    counts = _project_support_points_numba(
        points_cam=points_cam,
        fx=float(view["fx"]),
        fy=float(view["fy"]),
        cx=float(view["cx"]),
        cy=float(view["cy"]),
        width=int(view["width"]),
        height=int(view["height"]),
        depth_tolerance_m=float(depth_tolerance_m),
    )
    if support_radius_px > 0:
        size = int(2 * support_radius_px + 1)
        counts = maximum_filter(counts, size=size, mode="nearest")
    return np.asarray(counts, dtype=np.int32)


def backproject_depth_to_world(depth: np.ndarray, view: Dict[str, Any]) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    z = depth[ys, xs].astype(np.float64, copy=False)
    fx = float(view["fx"])
    fy = float(view["fy"])
    cx = float(view["cx"])
    cy = float(view["cy"])
    x_cam = ((xs.astype(np.float64) + 0.5) - cx) * z / fx
    y_cam = ((ys.astype(np.float64) + 0.5) - cy) * z / fy
    ones = np.ones_like(z)
    cam_h = np.stack([x_cam, y_cam, z, ones], axis=1)
    c2w = np.asarray(view["camera_to_world"], dtype=np.float64)
    world_h = cam_h @ c2w.T
    out = np.full((h, w, 3), np.nan, dtype=np.float64)
    out[ys, xs] = world_h[:, :3]
    return out


def compute_inside_bbox_mask(depth: np.ndarray, view: Dict[str, Any], bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    world = backproject_depth_to_world(depth, view)
    finite = np.isfinite(world).all(axis=2)
    inside = finite & np.all(world >= bbox_min[None, None, :], axis=2) & np.all(world <= bbox_max[None, None, :], axis=2)
    return np.asarray(inside, dtype=bool)


def relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def write_simple_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(str(x) for x in header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
