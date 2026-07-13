# Reference-Depth Geometry Evaluation Code Package

Package date: 2026-05-05

This package contains the depth-reference geometry evaluation code used for
our current frozen protocol. It is intended for users who want to evaluate
3DGS-style models by comparing rendered model depth against a training-only
RGB MVS reference rendered from the same held-out probe cameras.

The protocol is useful, but it is not a ground-truth depth benchmark. Before
using the numbers in a paper or report, manually verify both camera alignment
and reference mesh quality.

## 1. What This Evaluation Measures

The evaluator compares two depth maps at each held-out/probe camera:

- `D_ref`: depth rendered from a training-only RGB MVS reference mesh
- `D_model`: depth rendered from the evaluated model at the same probe camera

It measures whether the model surface is:

- in front of the reference surface
- behind the reference surface
- close to the reference surface
- missing at reference-valid pixels

Recommended wording:

- reference-depth-based geometric evaluation
- held-out geometry consistency against a training-only MVS reference
- front/behind/agreement depth analysis

Do not describe it as:

- ground-truth depth accuracy
- absolute geometry accuracy
- a standard benchmark with external 3D ground truth

## 2. Most Important Safety Checks

Two issues can invalidate the metrics.

### 2.1 Camera/Viewpoint Mismatch

The model depth must be rendered from exactly the same probe camera as the
reference depth. A previous bug produced plausible-looking metric curves while
model depth was rendered from a shifted camera.

Required checks:

- render a panel containing probe GT, reference depth, model RGB, and model depth
- confirm the model RGB aligns with the probe GT before trusting depth metrics
- for our Gaussian exporter, use `--camera_frame_mode probe_manifest_native_align`
- provide both `--probe_camera_manifest` and `--native_cameras_json`

Warning signs:

- model RGB/depth shows a different side or corner of the object than probe GT
- roof/facade edges in the model render are shifted relative to GT
- depth maps look clean, but visual alignment is clearly wrong

### 2.2 Reference Mesh Holes Or Soft Surfaces

The reference mesh is a depth carrier, not true geometry. If it has large holes,
floating fragments, or rounded building edges, the metric may reward or punish
methods for reference artifacts.

Required checks:

- inspect the OpenMVS dense point cloud
- inspect the OpenMVS refined mesh in CloudCompare or MeshLab
- render unmasked reference depth for several probe cameras
- render masked reference depth and confirm the mask is not hiding failure cases

Warning signs:

- large facade holes
- straight roof/building edges become curved in rendered reference depth
- big blank or invalid regions in important evaluated areas
- many floating mesh fragments in front of surfaces

## 3. Frozen Parameters Used In Our Current Run

The current frozen reference build uses:

```text
Reference backend: OpenMVS only (no COLMAP-MVS fallback)
OpenMVS archive_type: -1
InterfaceCOLMAP normalize: 0
DensifyPointCloud resolution_level: 1
DensifyPointCloud max_resolution: 2000
DensifyPointCloud min_resolution: 640
DensifyPointCloud number_views: 8
DensifyPointCloud number_views_fuse: 3
DensifyPointCloud iters: 4
OpenMVS estimate_roi/crop_to_roi: 0/0
RefineMesh resolution_level: 1
RefineMesh scales: 2
TextureMesh: not used
CUDA evidence: Densify/Reconstruct/Refine logs must contain `CUDA device 0 initialized:`;
patched RefineMesh must also contain `CUDA mesh refinement path completed; CPU fallback disabled`
The runner streams stage output and immediately terminates on `CUDA error`,
CUDA-unavailable, or CPU-fallback text; partial outputs and depth-map caches are removed.

Evaluation thresholds:
0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 30.00 meters
```

These parameters should be treated as fixed before comparing methods. If any
reference-construction parameter changes, rerun all methods for that scene on
the same rebuilt reference.

## 4. Package Layout

```text
tools/geometric_repeatability/
  build_depth_reference.py
  export_gaussian_probe_bundle.py
  evaluate_depth_reference.py
  summarize_depth_reference_methods.py
  visualize_depth_reference_method_comparison.py
  visualize_strict_probe_method_rgb_depth_comparison.py
  visualize_reference_validity_debug.py
  depth_reference_common.py
  DEPTH_REFERENCE_PROTOCOL.md
  REFERENCE_DEPTH_EVAL_WARNINGS.md
  run_depth_reference_formal_5scene_8method.ps1
```

Main scripts:

- `build_depth_reference.py`: builds the training-only OpenMVS reference and renders `D_ref`
- `export_gaussian_probe_bundle.py`: renders model RGB/depth/opacity from probe cameras
- `evaluate_depth_reference.py`: computes metrics between `D_ref` and `D_model`
- `summarize_depth_reference_methods.py`: aggregates method-level metric CSVs
- `visualize_strict_probe_method_rgb_depth_comparison.py`: builds alignment/debug panels
- `visualize_reference_validity_debug.py`: checks reference mask and valid regions

## 5. Dependencies

Expected environment:

- Windows + PowerShell
- Python environment compatible with the 3DGS repository
- CUDA-enabled OpenMVS executables: `InterfaceCOLMAP`, `DensifyPointCloud`,
  `ReconstructMesh`, and a fail-closed `RefineMesh` built with
  `openmvs-2.4.0-refine-cuda-fail-closed.patch`
- a clean Git worktree; bundle and metric manifests bind the commit plus the
  exporter/evaluator SHA256 and are invalidated when code changes
- CUDA/PyTorch environment for Gaussian model rendering

Python packages used by the packaged tools include:

- `numpy`
- `scipy`
- `numba`
- `plyfile`
- `Pillow`
- `matplotlib`
- `torch`

The Gaussian export and visualization scripts also import this repository's
local modules:

- `arguments`
- `gaussian_renderer`
- `scene`
- `utils`

So the scripts should be run from the 3DGS/FGS repository root, or with the
repository root on `PYTHONPATH`.

## 6. Input Requirements

For each scene, prepare:

- a strict/probe protocol manifest
- a training RGB image set for MVS reference construction
- held-out/probe camera definitions
- one trained model directory per method
- a valid model `cfg_args`
- a native model camera JSON used by the probe alignment exporter

The strict/probe manifest must ensure:

- reference geometry uses training views only
- held-out/probe views are not used to build the reference mesh
- ROI/support masks are derived from training-side reference data only

## 7. Step-by-Step Usage

All commands below are templates. Replace paths with your local paths.

### Step 1: Build the training-only reference

```powershell
$py = "python"
$repo = "<REPO_ROOT>"
$openmvs = "<OPENMVS_BIN>"

& $py "$repo\tools\geometric_repeatability\build_depth_reference.py" `
  --strict_protocol_manifest "<WORK_ROOT>\strict_protocol_manifest.json" `
  --out_dir "<WORK_ROOT>\DepthReference\SceneName\reference_openmvs_v1" `
  --openmvs_interface_colmap_cmd "$openmvs\InterfaceCOLMAP.exe" `
  --openmvs_densify_cmd "$openmvs\DensifyPointCloud.exe" `
  --openmvs_reconstruct_mesh_cmd "$openmvs\ReconstructMesh.exe" `
  --openmvs_refine_mesh_cmd "$openmvs\RefineMesh.exe" `
  --openmvs_cuda_device 0 `
  --openmvs_resolution_level 1 `
  --openmvs_max_resolution 2000 `
  --openmvs_min_resolution 640 `
  --openmvs_number_views 8 `
  --openmvs_number_views_fuse 3 `
  --openmvs_iterations 4 `
  --openmvs_refine_resolution_level 1 `
  --openmvs_refine_scales 2 `
  --resolution_arg 4 `
  --thresholds_m "0.10,0.25,0.50,1.00,2.00,5.00,10.00,20.00,30.00" `
  --support_min_count 1 `
  --support_radius_px 1 `
  --support_depth_tolerance_m 0.10
```

Important outputs:

- `reference_depth_manifest.json`
- `probe_camera_manifest.json`
- `views/`
- `openmvs_command_plan.json`
- `_openmvs_workspace/reference_openmvs_dense.ply`
- `_openmvs_workspace/reference_openmvs_mesh_refined.ply`

### Step 2: Render each model at the same probe cameras

```powershell
& $py "$repo\tools\geometric_repeatability\export_gaussian_probe_bundle.py" `
  -s "<SCENE_ROOT>" `
  -m "<MODEL_ROOT>" `
  -r 4 `
  --iteration 30000 `
  --out_dir "<WORK_ROOT>\DepthReference\SceneName\MethodName\probe_bundle" `
  --split_label "MethodName" `
  --scene_name_override "SceneName" `
  --camera_frame_mode probe_manifest_native_align `
  --probe_camera_manifest "<WORK_ROOT>\DepthReference\SceneName\reference_openmvs_v1\probe_camera_manifest.json" `
  --native_cameras_json "<MODEL_ROOT>\cameras.json"
```

For a fair comparison, all methods must use the same `probe_camera_manifest`
and the same reference for the scene.

Create the frozen depth-semantics adapter consumed by the evaluator:

```powershell
$adapter = [ordered]@{
  protocol_name = "reference-depth-based-geometric-evaluation-v1"
  method_name = "MethodName"
  model_path = "<MODEL_ROOT>"
  source_path = "<SCENE_ROOT>"
  iteration = 30000
  depth_semantics = "inverse_camera_z_from_renderer"
  validity_rule = [ordered]@{
    mode = "opacity_threshold"
    opacity_threshold = 0.5
    depth_min = 1e-6
  }
}
$adapterPath = "<WORK_ROOT>\DepthReference\SceneName\MethodName\probe_bundle\depth_adapter_manifest.json"
$adapterJson = $adapter | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText(
  $adapterPath,
  $adapterJson + [Environment]::NewLine,
  (New-Object System.Text.UTF8Encoding($false))
)
```

### Step 3: Evaluate model depth against reference depth

```powershell
& $py "$repo\tools\geometric_repeatability\evaluate_depth_reference.py" `
  --reference_manifest "<WORK_ROOT>\DepthReference\SceneName\reference_openmvs_v1\reference_depth_manifest.json" `
  --model_manifest "<WORK_ROOT>\DepthReference\SceneName\MethodName\probe_bundle\split_manifest.json" `
  --adapter_manifest "<WORK_ROOT>\DepthReference\SceneName\MethodName\probe_bundle\depth_adapter_manifest.json" `
  --out_dir "<WORK_ROOT>\DepthReference\SceneName\MethodName\metrics" `
  --enable_agreement_metrics
```

Important outputs:

- `metrics_summary.json`
- `metrics_summary.csv`
- `front_intrusion_curve.csv`
- `per_view_counts.csv`

### Step 4: Visualize alignment and depth behavior

```powershell
& $py "$repo\tools\geometric_repeatability\visualize_strict_probe_method_rgb_depth_comparison.py" `
  --reference_manifest "<WORK_ROOT>\DepthReference\SceneName\reference_openmvs_v1\reference_depth_manifest.json" `
  --out_dir "<WORK_ROOT>\DepthReference\SceneName\visual_checks" `
  --scene_name "SceneName" `
  --gt_images_root "<STRICT_DATASET_ROOT>" `
  --gt_images_dir_name "images" `
  --random_n 10 `
  --random_seed 1 `
  --include_unmasked_reference_depth `
  --method "Ours=<WORK_ROOT>\Ours\probe_bundle" `
  --method "Baseline=<WORK_ROOT>\Baseline\probe_bundle"
```

Use these panels before trusting metrics.

## 8. Metric Definitions

Let `V_ref` be pixels where the reference is valid:

```text
V_ref = { p | M_ref(p) = 1 }
N_ref = |V_ref|
```

Let `V_model` be pixels where the model depth is valid, and define:

```text
e(p) = D_model(p) - D_ref(p)
```

Sign convention:

- `e(p) < 0`: model surface is closer to the camera than the reference
- `e(p) > 0`: model surface is farther from the camera than the reference

For threshold `delta`:

```text
FrontIntrusionRate@delta =
  count(p in V_ref and p in V_model and e(p) < -delta) / N_ref

FrontIntrusionMagnitude@delta =
  mean(D_ref(p) - D_model(p)) over p where e(p) < -delta

TooDeepRate@delta =
  count(p in V_ref and p in V_model and e(p) > delta) / N_ref

DepthAgreementRate@delta =
  count(p in V_ref and p in V_model and abs(e(p)) <= delta) / N_ref

MissingRate =
  count(p in V_ref and p not in V_model) / N_ref
```

Lower is better for:

- `FrontIntrusionRate`
- `FrontIntrusionMagnitude`
- `TooDeepRate`
- `MissingRate`
- `AbsDepthError_Median`
- `AbsDepthError_Mean`

Higher is better for:

- `DepthAgreementRate`

`SignedDepthBias_Mean` is diagnostic:

- negative means the model is biased toward the camera
- positive means the model is biased away from the camera

## 9. Recommended Reporting

Do not rely on one threshold only. Report the full threshold curve:

```text
0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 30.00 m
```

For paper material, include:

- per-scene curves
- macro-average curves
- per-method tables for all thresholds
- visual panels showing GT/probe, reference depth, model RGB, model depth, and error classes

## 10. Checklist Before Using Results

Use the numbers only after all checks pass:

- same probe cameras are used for reference and all methods
- model RGB render aligns with probe GT
- unmasked reference depth looks plausible
- masked reference valid region does not hide main structures
- fused point cloud and mesh have no severe holes in evaluated regions
- all methods use the same scene reference
- depth adapter manifests are frozen before comparison
- thresholds are fixed before looking at comparative results

## 11. Common Failure Cases

Camera mismatch:

- symptom: model RGB/depth is a different view than GT/reference
- fix: use `probe_manifest_native_align` and verify native camera transform

Mesh holes:

- symptom: invalid reference depth on facades or roofs
- fix: rebuild the reference, inspect the OpenMVS dense cloud and refined mesh,
  and audit the frozen `DensifyPointCloud`/`ReconstructMesh`/`RefineMesh`
  parameters before deciding whether a protocol-wide change is justified

Overly soft mesh:

- symptom: straight edges become rounded or bulged
- fix: inspect the OpenMVS dense cloud and the pre-/post-refinement meshes;
  do not tune refinement settings per scene or silently replace the backend

Stale metrics:

- symptom: metrics do not correspond to current visualizations
- fix: delete or isolate old metric directories and rerun all methods on the same reference

## 12. Notes On Compatibility

The packaged scripts preserve backward-compatible defaults where new behavior
was added:

- `evaluate_depth_reference.py --enable_agreement_metrics` is opt-in
- `export_gaussian_probe_bundle.py --camera_frame_mode scene_test` remains the default
- strict probe/native alignment is enabled only when explicitly requested
- unmasked reference visualization is opt-in

For this protocol, however, the recommended formal setting is to enable:

- `--enable_agreement_metrics`
- `--camera_frame_mode probe_manifest_native_align`
- `--include_unmasked_reference_depth` for visual checks
