# Reference Depth Evaluation Warnings

This note is for anyone reusing the reference/depth evaluation tools in
`tools/geometric_repeatability`.

The reference-depth protocol is useful for diagnosing geometry, but it is not a
plug-and-play ground-truth evaluator. In our debugging, two issues were severe
enough to invalidate metrics if left unchecked:

- camera/viewpoint mismatch between the probe camera and the evaluated model
- unstable training-only OpenMVS reference geometry, especially facade holes,
  false surfaces, and floating mesh fragments

Treat the evaluator as reliable only after both issues have been ruled out for
the scene and method being evaluated.

## 1. What This Protocol Measures

The intended protocol is:

1. Build a training-only RGB MVS reference.
2. Convert the MVS reference into a mesh.
3. Render reference depth from held-out/probe cameras.
4. Render each method's model depth from the same probe cameras.
5. Compare the two depth maps with fixed meter-scale thresholds.

This should be described as:

- reference-depth-based geometric evaluation
- held-out geometry consistency against a training-only MVS reference
- front/behind/agreement depth analysis under a fixed reference

Do not describe it as:

- absolute ground-truth depth accuracy
- true geometric accuracy
- a standard benchmark with external 3D ground truth

If the reference mesh is wrong, the metric is wrong. If the model depth is
rendered from the wrong camera, the metric is wrong.

## 2. Frozen Protocol Used In Our Current Run

The current fixed reference-construction protocol is:

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
The runner must stream and immediately terminate on `CUDA error`, CUDA-unavailable,
or CPU-fallback text, then delete partial outputs and depth-map caches.

Evaluation thresholds:
0.10, 0.25, 0.50, 1.00, 2.00, 5.00, 10.00, 20.00, 30.00 meters
```

`InterfaceCOLMAP.normalize=0` is required so the reference remains in the
already-aligned SfM frame. OpenMVS automatic ROI estimation/cropping is disabled
because the evaluator's existing robust training-only ROI is the authoritative
and auditable validity boundary.

Important: these settings must be treated as a fixed protocol. Do not tune them
per scene after looking at comparative metrics.

## 3. No Manual Scene-Specific Cleanup

The final protocol is intended to be end-to-end and reproducible. It should not
include manual per-scene mesh editing.

Avoid the following for paper-facing metrics:

- manually deleting mesh islands in CloudCompare
- hand-cropping only one scene's dense point cloud
- manually filling facade holes
- changing OpenMVS reconstruction parameters scene by scene
- changing masks after viewing which method benefits

Allowed only if frozen and documented for all scenes:

- deterministic train-side ROI rules
- fixed quantile-based outlier clipping, if explicitly added to the protocol
- fixed connected-component or density cleanup, if applied identically to all
  scenes before comparative results are inspected

At the time of the current run, no manual clipping or scene-specific mesh
cleanup is part of the frozen protocol.

## 4. Critical Bug: Camera / Viewpoint Mismatch

Before trusting any metric, verify that the rendered model depth is from the
same physical probe camera as:

- the probe GT image
- the reference mesh depth
- the model RGB render
- the model depth render

In our debugging, a real failure mode was:

- probe GT and reference depth used the strict probe camera
- model RGB/depth was rendered in the model's native training frame without the
  proper strict-to-native alignment

This produced visually plausible but wrong comparisons. The GT/reference showed
one roof/facade layout, while the model RGB/depth showed a nearby but different
view. Metrics computed from this mismatch are invalid, even if the numbers look
smooth or monotonic.

Minimum checks:

- For several probe views, lay out the probe GT, reference depth, model RGB
  render at the same probe view, and model depth.
- Check roof edges, facade corners, windows, roads, and other high-frequency
  structures.
- Overlay image edges from GT and model RGB/depth visualizations when possible.
- If the model RGB/depth shows a different field of view, different parallax, or
  different visible building side than the probe GT, stop and fix camera
  alignment first.

Likely fixes:

- Confirm whether the model is trained in the strict probe frame or in the
  native/original method frame.
- If the model is in a native frame, apply the strict-to-native world transform
  before rendering model RGB/depth.
- Save the camera transform used by the exporter in a manifest and verify it
  against a manually inspected same-view RGB render.
- Do not use neighboring held-out images as visual proxies for the evaluated
  camera. The displayed GT image must correspond to the exact camera used for
  reference and model rendering.

Relevant code paths:

- `export_gaussian_probe_bundle.py`
- `visualize_strict_probe_method_rgb_depth_comparison.py`
- `depth_reference_common.py`

## 5. Critical Risk: Reference Mesh Quality

Do not assume the reference mesh is reliable just because OpenMVS produced a
mesh.

In our Building pilot, we observed several reference-geometry failures:

- large facade holes, even when roof regions looked mostly complete
- straight building edges becoming rounded or curved after meshing
- many floating mesh fragments around roofs and facades
- low-support facade regions becoming soft or locally bulged in reconstructed meshes
- meshes collapsing or becoming unusable when a small number of dense outlier
  points expanded the global bounding box

These failures directly affect front/behind/agreement metrics. For example:

- Mesh holes reduce the valid evaluation region or make the reference depth
  discontinuous.
- Rounded or bridged edges can mark correct model geometry as wrong.
- Floating reference fragments can create false front/behind penalties.
- Excessive masking can hide exactly the regions where methods differ.
- Smoothed or false OpenMVS surfaces can understate or overstate model depth error near facades.

Minimum checks:

- Inspect both `_openmvs_workspace/reference_openmvs_dense.ply` and
  `_openmvs_workspace/reference_openmvs_mesh_refined.ply`.
- Open the dense point cloud and mesh side by side in CloudCompare or another
  3D viewer.
- Inspect roofs, facade planes, vertical edges, balconies, railings, and ground
  contact regions.
- Render unmasked reference depth and masked reference depth. If the unmasked
  depth is already discontinuous or visibly distorted, the mesh is not yet a
  stable evaluation reference.
- Do not rely only on aggregate valid-pixel ratios. A mesh can have a high valid
  ratio while still being wrong around important facade edges.

Useful CloudCompare examples:

```powershell
Start-Process -FilePath 'CloudCompare.exe' `
  -ArgumentList @('PATH_TO_REFERENCE_MESH.ply')
```

```powershell
Start-Process -FilePath 'CloudCompare.exe' `
  -ArgumentList @('PATH_TO_REFERENCE_MESH.ply', 'PATH_TO_REFERENCE_FUSED_POINT_CLOUD.ply')
```

## 6. Observed Mesh Failure Modes and Practical Fixes

### 6.1 Facade Holes

Symptoms:

- side walls have large missing regions
- roofs are mostly present but vertical surfaces are sparse
- reference depth has blank or jagged facade regions

Likely causes:

- insufficient OpenMVS point support on vertical surfaces
- oblique views not providing stable matches for facade pixels
- insufficient view agreement during dense fusion

Practical fix in the frozen protocol:

- Keep the frozen OpenMVS parameters from Section 2 for every scene.
- Do not manually fill holes.
- If a parameter change is justified, rebuild a new isolated reference for all
  compared methods rather than tuning one scene in place.

### 6.2 Floating Mesh Fragments

Symptoms:

- small blobs float around roofs/facades
- speckled mesh fragments appear in empty space
- OpenMVS mesh output contains many low-confidence surface islands

Likely causes:

- OpenMVS depth outliers survive fusion
- view-agreement settings are too permissive
- mesh reconstruction wraps sparse/noisy point support into surfaces

Potential fixes if this becomes a blocker:

- Increase `DensifyPointCloud.number_views_fuse` only as a new globally frozen
  protocol variant.
- Add a fixed, deterministic outlier-removal step before meshing.
- Add fixed connected-component cleanup for very small isolated mesh islands.

These fixes must be frozen and applied uniformly. They are not part of the
current fixed protocol unless explicitly added and all metrics are rerun.

### 6.3 Mesh Collapse From Extreme Outliers

Symptoms:

- dense point cloud looks mostly plausible, but the mesh has only a tiny
  number of vertices/faces or appears completely collapsed
- mesh file size is unexpectedly tiny compared with previous runs
- point cloud coordinate range contains extreme values far from the scene

Observed example:

- most Building dense points were near the scene, but a small fraction of
  outliers expanded the coordinate range to thousands of meters
- reconstruction operated over this huge box, causing the actual building to
  be under-resolved

Potential fixes:

- Inspect robust quantiles and full coordinate min/max of the dense point cloud.
- If clipping is introduced, apply a fixed documented rule before meshing and
  rebuild every reference with that rule.
- Save clipping statistics, including input count, kept count, removed count,
  raw coordinate range, robust bbox, and padding.
- Do not silently change clipping after looking at comparative metrics.

### 6.4 Rounded / Non-Sharp Building Edges

Symptoms:

- straight roof/facade boundaries become rounded or wavy
- depth edges are not aligned with visible image edges
- small mesh holes make originally straight lines appear curved

Likely causes:

- incomplete point support around the edge
- mesh refinement smoothing over missing data
- OpenMVS depth noise near occlusion boundaries

Practical fix in the frozen protocol:

- Use the fixed `RefineMesh` settings from Section 2.
- Accept that the reference mesh is a depth carrier, not a perfect visual mesh.
- Verify unmasked reference depth before trusting metrics.

## 7. Metric Validity Rules

Depth metrics should not be reported unless all of the following are true:

- model RGB/depth render and probe GT are confirmed to be the same camera
- reference mesh is visually acceptable in the evaluated regions
- unmasked reference depth is not severely broken
- masked valid region is not hiding the main failure cases
- all compared methods use the same frozen reference for the same scene
- all depth adapters and validity-mask rules are frozen before comparing methods

If a reference rebuild changes:

- OpenMVS dense-reconstruction parameters
- source image count
- view-fusion requirements
- outlier clipping
- mesh backend
- density filtering
- probe camera alignment behavior

then previously computed metrics are not directly comparable. At minimum, rerun
all methods for the affected scene on the same rebuilt reference.

For paper-facing comparisons, use one frozen reference-construction
configuration for all methods and all reported scenes, or clearly report that a
scene-specific reference configuration was used and justify why.

## 8. Recommended Sanity Workflow

Before large batch evaluation:

1. Build the training-only reference for one pilot scene.
2. Inspect the dense point cloud manually.
3. Inspect the mesh manually.
4. Render unmasked and masked reference depth for several probe cameras.
5. Render a probe-view visualization panel with probe GT, reference depth, model
   RGB at the same probe view, and model depth.
6. Confirm camera agreement first.
7. Confirm reference geometry quality second.
8. Only then run full metrics.

When debugging, prioritize failures in this order:

1. camera/viewpoint mismatch
2. unstable or incomplete reference geometry
3. method-specific depth adapter problems
4. threshold/reporting choices

## 9. If Results Look Too Strange

Typical warning signs:

- model depth seems to show a different side of the building than GT
- model RGB render does not align with the probe image
- reference depth has large blank facade regions
- unmasked reference depth has jagged or discontinuous building edges
- straight roof edges become obviously curved in the reference
- all methods score unexpectedly poorly, including visually strong methods
- metric curves look smooth but contradict obvious visual alignment checks

When this happens, do not trust the numbers until the following issues have
been ruled out:

- camera/viewpoint mismatch
- reference mesh holes or false surfaces
- OpenMVS dense point-cloud outliers
- excessive reference masking
- stale metrics computed on an older reference version

## 10. Current Status Of Our Protocol

Earlier COLMAP-MVS/Poisson references and their derived metrics are superseded
diagnostic artifacts. The current frozen protocol is the OpenMVS setting listed
in Section 2. Paper-facing metrics must be recomputed across all scenes and
methods from the isolated `reference_openmvs_v1` outputs.

Do not distribute or cite intermediate depth-reference metrics without also
disclosing which reference version was used and whether the reference mesh
passed visual quality checks.
