# Reference-Depth Geometry Evaluation Protocol

Version: `v1_frozen_20260505_d12_trim6`

## 1. Goal

This protocol is designed to quantify the kind of geometric failure that is repeatedly observed by inspection in our thermal reconstructions, especially:

- roof or facade geometry that expands toward the camera
- floating structure above the true surface
- unstable first-surface depth under held-out viewpoints

The protocol is intentionally designed to **avoid retraining odd/even models** for every baseline. Instead, it reuses already trained models and compares their rendered held-out depth against an **external training-only reference depth**.

## 2. What This Protocol Measures

This protocol measures:

- held-out geometric consistency against an external reference
- front-surface intrusion and depth bias
- surface stability of already trained models without rerunning the full training pipeline

This protocol does **not** measure:

- absolute geometric ground-truth accuracy
- full end-to-end reconstruction repeatability
- photometric quality

Paper wording should therefore use terms such as:

- `reference-depth-based geometric evaluation`
- `held-out geometry consistency against a training-only MVS reference`
- `front-intrusion depth analysis`

Paper wording should **not** call this:

- `ground-truth depth accuracy`
- `absolute geometry accuracy`

## 3. Fixed Data Split

For each scene:

- keep the existing benchmark train/test split unchanged
- use **all training views only** to build the external reference geometry
- use the original **held-out test views only** as evaluation cameras

Important constraints:

- held-out views must not be used in reference-geometry construction
- held-out views must not be used in threshold tuning
- held-out views must not be used in ROI construction
- method outputs must not influence the reference geometry

## 4. Reference Geometry

### 4.1 Modality

Use **RGB training views** to build the external reference geometry.

Reason:

- RGB images are more suitable than thermal images for classical dense multi-view stereo
- the purpose of the reference is geometric support, not thermal appearance fidelity

This protocol assumes RGB and thermal cameras are already expressed in the same calibrated scene frame used by the benchmark. If that shared frame is not trustworthy, the protocol should not be executed until the cross-modal frame is verified.

### 4.2 Construction Rule

Build one external reference geometry `G_ref` per scene from **training RGB views only**.

Recommended pipeline:

1. training-only aligned sparse reconstruction / training-only camera subset
2. import the aligned COLMAP sparse model with OpenMVS `InterfaceCOLMAP`
3. OpenMVS dense reconstruction with `DensifyPointCloud`
4. OpenMVS mesh reconstruction and refinement with `ReconstructMesh` and `RefineMesh`

Rationale for using a mesh:

- the final evaluation target is a per-view first-surface depth map
- a mesh provides a single-valued z-bufferable surface for held-out rendering
- this is more stable for held-out depth rasterization than directly comparing raw Gaussian centers

The fused dense point cloud should still be saved as an audit artifact, but the primary reference carrier for depth rendering is the **reference mesh**.

### 4.2.1 Frozen Reference Reconstruction Parameters

The final fixed reference reconstruction settings are:

- backend: `OpenMVS` only; no COLMAP MVS/mesher fallback
- all four OpenMVS stages use interface archive mode: `archive_type = -1`
- `InterfaceCOLMAP.normalize = 0` so the aligned SfM coordinate frame is preserved
- `DensifyPointCloud.resolution_level = 1`
- `DensifyPointCloud.max_resolution = 2000`
- `DensifyPointCloud.min_resolution = 640`
- `DensifyPointCloud.number_views = 8`
- `DensifyPointCloud.number_views_fuse = 3`
- `DensifyPointCloud.iters = 4`
- automatic OpenMVS ROI estimation/cropping disabled; the evaluator's existing robust training-only ROI rule remains authoritative
- `RefineMesh.resolution_level = 1`
- `RefineMesh.scales = 2`
- CUDA device: `0` by default
- `DensifyPointCloud`, `ReconstructMesh`, and `RefineMesh` logs must prove
  `CUDA device 0 initialized:` and must not report CUDA failure; OpenMVS's
  Delaunay/graph-cut reconstruction contains CPU work by algorithm design
- the runner streams OpenMVS output and terminates the whole stage immediately
  on `CUDA error`, CUDA-unavailable, or CPU-fallback text; partial outputs and
  depth-map caches are removed before control returns
- `RefineMesh` must be built with
  `openmvs-2.4.0-refine-cuda-fail-closed.patch`; its binary is checked before
  execution and its log must contain
  `CUDA mesh refinement path completed; CPU fallback disabled`
- `TextureMesh` is not run because reference depth requires geometry, not texture

OpenMVS v2.4 artifact contract:

- `InterfaceCOLMAP` produces `scene.mvs`
- `DensifyPointCloud` produces `reference_openmvs_dense.mvs` and `.ply`
- `ReconstructMesh` produces only `reference_openmvs_mesh.ply`
- `RefineMesh` receives the dense `.mvs` plus `--mesh-file` and produces only
  `reference_openmvs_mesh_refined.ply`
- successful stage receipts bind the plan, command contract, required outputs,
  logs, and SHA256 identities; densification invalidation also deletes `.dmap`
  sidecar caches before rerun
- bundle/metric reuse additionally requires the same clean Git commit and the
  current exporter/evaluator script SHA256

These settings must be frozen before comparing methods. Any change requires a new isolated reference output and a rerun of every method evaluated against it.

### 4.3 Reference ROI

Construct one scene ROI from the **training-side reference geometry only**.

The ROI must:

- be fixed before cross-method comparison
- be reused for all methods in the scene
- not depend on model outputs
- not depend on held-out views

### 4.4 Reference Support / Confidence

The reference validity mask must not be defined by mesh coverage alone. It must also require training-side geometric support.

In `v1`, the support source is fixed as:

- support derived from the **training-only dense reconstruction artifacts**
- support attached to the rendered reference surface through a frozen projection rule

The exact implementation may use per-face or per-pixel support, but it must satisfy all of the following:

- it is derived only from training-side dense reconstruction outputs
- it is frozen before any cross-method comparative result is inspected
- it is identical for all compared methods in the same scene
- it does not use held-out images or model outputs

The support threshold must be recorded explicitly in the run manifest. It must be treated as a **fixed protocol parameter**, not as a post-hoc tuning knob.

## 5. Reference Depth

For each held-out thermal evaluation camera `v`, render the reference mesh into that camera to obtain:

- `D_ref(v)`: reference metric depth
- `M_ref(v)`: reference validity mask

Definition:

- depth is positive metric camera-z depth
- depth corresponds to the first visible surface along each camera ray

Reference validity mask `M_ref(v)` is `1` only where all of the following hold:

- the reference mesh covers the pixel after rasterization
- the pixel lies inside the fixed scene ROI
- the reference support/confidence for that surface is above a fixed threshold

The support/confidence source must come from the training-side dense reconstruction only, for example:

- fused-point support count
- dense stereo consistency score
- mesh-face support inherited from fused points

Exact support threshold must be fixed before comparative results are inspected and saved in a manifest.

## 6. Model Depth

For each already trained method and held-out thermal camera `v`, render:

- `D_model(v)`: model metric depth
- `M_model(v)`: model validity mask

Rules:

- use the **thermal model** for all compared methods
- use the exact same held-out thermal camera and image resolution as the reference render
- export positive metric camera-z depth
- use the method's standard held-out rendering pipeline, with a frozen adapter if conversion is needed

If a method natively outputs inverse depth or another depth-like quantity, convert it once through a method-specific adapter that is frozen before any comparative study. That adapter must then remain identical for all scenes.

Each method must have its own frozen adapter record, for example in `depth_adapter_manifest.json`, containing:

- method name
- native depth semantic
- conversion to metric camera-z depth
- validity-mask source
- any threshold or unit conversion used

The adapter is allowed to differ across methods, but once defined for a method it must remain unchanged across all scenes.

## 7. Pixel Set Used For Evaluation

Per held-out view, evaluate only on pixels inside the fixed reference-valid region:

- primary valid set: `M_ref(v) = 1`

Within that region, keep the following bookkeeping separate:

- pixels where the model also has valid depth
- pixels where the model has no valid depth

This avoids rewarding methods for simply failing to place geometry.

In reporting, at least the following categories must remain distinguishable:

- front intrusion
- valid but too deep
- invalid or missing prediction

## 8. Metrics

Let `e(v, p) = D_model(v, p) - D_ref(v, p)` on valid pixels.

Use **absolute meter thresholds**. The fixed threshold sweep is:

- `0.10 m`
- `0.25 m`
- `0.50 m`
- `1.00 m`
- `2.00 m`
- `5.00 m`
- `10.00 m`
- `20.00 m`
- `30.00 m`

These thresholds are intended to capture physically meaningful tolerances on large outdoor scenes.

Important wording:

- this is a `fixed physical tolerance in meters` protocol
- it is not a scale-normalized threshold protocol

### 8.1 Primary Metric: Front-Intrusion Rate

For each threshold `delta`:

- `FrontIntrusionRate@delta`

Definition:

- fraction of reference-valid pixels where the model predicts a surface that is too close to the camera:
- `D_model < D_ref - delta`

This is the key metric for the observed failure mode where a roof or facade grows outward toward the camera.

### 8.2 Primary Companion Metric: Front-Intrusion Magnitude

For each threshold `delta`:

- `FrontIntrusionMagnitude@delta`

Definition:

- mean of `(D_ref - D_model)` over pixels satisfying `D_model < D_ref - delta`

This measures not just how often intrusion occurs, but how severe it is.

### 8.3 Secondary Metrics

Report the following as secondary diagnostics:

- `DepthAgreementRate@delta`
- `AbsDepthError_Median`
- `AbsDepthError_Mean`
- `SignedDepthBias_Mean`
- `MissingRate`
- `TooDeepRate@delta`

Where:

- `DepthAgreementRate@delta` counts reference-valid pixels where the model is valid and `|D_model - D_ref| <= delta`
- `SignedDepthBias_Mean < 0` indicates systematic front bias
- `MissingRate` counts reference-valid pixels where the model does not provide valid depth
- `TooDeepRate@delta` counts reference-valid pixels where the model is valid but deeper than the reference by more than `delta`

## 9. Reporting

Recommended paper-facing reporting:

- main text: one compact table of primary depth metrics on the evaluated scenes
- main text or supplementary: threshold curve over `0.10 / 0.25 / 0.50 / 1.00 / 2.00 / 5.00 / 10.00 / 20.00 / 30.00 m`
- supplementary: per-scene metric tables and visual overlays

The primary narrative should emphasize:

- lower front-intrusion rate
- lower front-intrusion magnitude
- better held-out depth consistency against the same training-only reference

## 10. Audit Artifacts

The evaluator should save:

- the fixed protocol file used for the run
- the training-only reference construction manifest
- the support/confidence threshold manifest
- the `depth_adapter_manifest.json`
- the training-only ROI definition
- the dense fused point cloud summary
- the reference mesh path and metadata
- per-view `D_ref` and `M_ref`
- per-method per-view `D_model` and `M_model`
- per-scene metric CSV files
- curve CSV files
- a small set of visual overlays for manual inspection

## 11. Initial Execution Scope

If this protocol is approved, the first pilot should be:

- scene: `Building`
- methods: `Ours`, `ThermalGaussian_OMMG`, `ThermalGaussian_MFTG`, `ThermalGaussian_MSMG`, `Thermal3D_GS`
- modality: thermal `T` model

Only after the pilot is judged stable should the protocol expand to more scenes.

## 12. Why This Protocol Was Chosen

This protocol is chosen because it is a better fit than the two weaker alternatives below:

1. comparing dense point clouds directly to Gaussian centers
   - rejected because Gaussian centers are not equivalent to a reconstructed surface

2. retraining every baseline multiple times with new seeds or new odd/even splits
   - rejected as too expensive and operationally fragile for the current setting

The chosen protocol keeps the comparison focused on the actual observed failure mode:

- whether the model places its visible surface in front of the externally supported scene surface under held-out views
