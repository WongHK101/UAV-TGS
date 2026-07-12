# Geometric Repeatability Tools

This folder contains the v1 tooling for:

- pose-controlled cross-subset geometric repeatability
- method-agnostic ROI construction from training-side sparse points
- deterministic point-cloud evaluation using `NumPy + SciPy cKDTree`

## Files

- `PROTOCOL.md`: fixed paper-facing protocol definition
- `evaluator.py`: ROI builder and scene evaluator
- `sanity_tests.py`: synthetic checks for determinism and metric behavior
- `materialize_colmap_odd_even_split.py`: generate explicit `train_odd/train_even/probe_test` lists from a COLMAP scene
- `export_gaussian_probe_bundle.py`: export probe-view `depth + opacity` bundles from Gaussian models in this repo
- `build_scene_manifest.py`: merge odd/even split bundles into one evaluator scene manifest

## Expected Scene Manifest Format

The evaluator consumes one JSON manifest per scene. Relative paths are resolved relative to the manifest file.

Top-level fields:

```json
{
  "protocol_name": "pose-controlled-cross-subset-geometric-repeatability-v1",
  "scene_name": "Building",
  "roi_path": "roi.json",
  "depth_semantics": "inverse_camera_z_from_renderer",
  "distance_domain": "after_roi_crop_and_after_voxel_downsampling",
  "validity_rule": {
    "mode": "opacity_threshold",
    "opacity_threshold": 0.5,
    "depth_min": 1e-6
  },
  "views": [
    {
      "view_id": "00000",
      "width": 1280,
      "height": 720,
      "fx": 1024.0,
      "fy": 1024.0,
      "cx": 640.0,
      "cy": 360.0,
      "camera_to_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "odd_file": "odd/00000.npz",
      "even_file": "even/00000.npz"
    }
  ]
}
```

Each per-view `.npz` file must contain:

- `depth`: `HxW` inverse camera-z depth for the repo Gaussian renderer export path
- `opacity`: `HxW` accumulated opacity in `[0, 1]`

The evaluator applies the fixed v1 rule from `PROTOCOL.md`:

- convert inverse depth to metric camera-z, then valid pixel = finite positive metric depth AND opacity >= 0.5

## ROI Builder

Build a method-agnostic shared ROI from a training-side sparse point cloud:

```powershell
python tools\geometric_repeatability\evaluator.py build-roi `
  --scene_name Building `
  --points_path <SCENE_ROOT>\sparse\0\points3D.ply `
  --out tools\geometric_repeatability\artifacts\building_roi.json
```

## Explicit Odd/Even Split Lists

Materialize protocol-fixed `probe_test`, `train_union`, `train_odd`, and `train_even` lists from a COLMAP scene:

```powershell
python tools\geometric_repeatability\materialize_colmap_odd_even_split.py `
  --source_path <SCENE_ROOT> `
  --out_dir tools\geometric_repeatability\artifacts\building_protocol_split `
  --llffhold 8
```

This writes:

- `probe_test.txt`
- `train_union.txt`
- `train_odd.txt`
- `train_even.txt`
- `split_manifest.json`

The repo loader now accepts explicit split lists via:

- `--train_list path\to\train_odd.txt`
- `--test_list path\to\probe_test.txt`

When these are omitted, the original loader behavior remains unchanged.

## Gaussian Probe Export

Export one split bundle from a trained Gaussian model directory:

```powershell
python tools\geometric_repeatability\export_gaussian_probe_bundle.py `
  --model_path <MODEL_ROOT> `
  --iteration 60000 `
  --split_label odd `
  --out_dir tools\geometric_repeatability\artifacts\building_odd_bundle `
  --max_views 4
```

This writes:

- per-view `.npz` files containing `depth` and `opacity`
- `split_manifest.json`

## Scene Manifest Builder

Merge odd/even split bundles into the evaluator-facing scene manifest:

```powershell
python tools\geometric_repeatability\build_scene_manifest.py `
  --odd_manifest tools\geometric_repeatability\artifacts\building_odd_bundle\split_manifest.json `
  --even_manifest tools\geometric_repeatability\artifacts\building_even_bundle\split_manifest.json `
  --roi_path tools\geometric_repeatability\artifacts\building_roi.json `
  --out tools\geometric_repeatability\artifacts\building_manifest.json
```

## Scene Evaluation

Evaluate one scene bundle:

```powershell
python tools\geometric_repeatability\evaluator.py evaluate-scene `
  --manifest tools\geometric_repeatability\artifacts\building_manifest.json `
  --out_dir tools\geometric_repeatability\artifacts\building_eval
```

The evaluator saves:

- ROI snapshot
- threshold snapshot
- odd/even point clouds after ROI crop
- odd/even point clouds after voxel downsampling
- `metrics.json`
- `metrics.csv`

## Sanity Tests

```powershell
python tools\geometric_repeatability\sanity_tests.py
```
