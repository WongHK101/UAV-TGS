# Pose-Controlled Cross-Subset Geometric Repeatability (v1)

This document fixes the paper-facing evaluation protocol before any comparative results are inspected.

## Claim Scope

This protocol measures:

- `pose-controlled cross-subset geometric repeatability`
- `pose-controlled reconstruction self-consistency`
- `geometry stability under disjoint-view training`

This protocol does **not** measure:

- absolute geometric accuracy
- end-to-end pipeline repeatability
- superiority to external pseudo-ground-truth pipelines

## Data Split

For each evaluated scene:

- keep the original train/test partition unchanged
- split the original training views into `Train-O` and `Train-E` by odd/even ordering
- keep the original held-out test views as the shared `probe-view` set
- probe views must **not** participate in shared-frame reconstruction, ROI construction, threshold tuning, or protocol adjustment

## Shared Camera Frame

- estimate one shared external camera frame from the union of training views only
- keep intrinsics/extrinsics fixed for all odd/even runs of all methods
- any method-specific pose refinement must be disabled for this protocol, or the protocol must be explicitly described as evaluating only the fixed-pose reconstruction stage
- held-out probe views may be post-registered into this frozen frame for evaluation only, but that registration must not update the train-union frame, ROI, thresholds, or training split definition

Paper wording:

> This protocol isolates reconstruction-stage geometry repeatability from pose-estimation variability rather than measuring end-to-end pipeline repeatability.

## Probe Views

- use the original held-out test views as the common probe-view set
- all methods render depth/opacity on exactly the same probe views
- no training views are used for the main repeatability metric
- when probe-view poses are not natively present in the train-union model, they must be assigned afterward by post-registration into the frozen shared frame with existing training frames fixed

In one line:

> Probe views do not participate in shared-frame reconstruction or ROI construction; their poses are assigned afterward by registration into the frozen train-union frame for evaluation only.

## Exact Depth Semantics

For the Gaussian renderer used in this repo, the exported `depth` array is **inverse camera-z depth as returned by the renderer**:

- `depth[v, u]` is `1 / z_camera` for valid pixels
- the evaluator converts it back to metric camera-z depth using `z_camera = 1 / depth`
- camera coordinates then follow the saved `camera_to_world` transform and the usual pinhole back-projection rule
- invalid pixels must not be encoded using signed sentinel values; they are filtered by the validity-mask rule below

The manifest string for this rule is:

- `inverse_camera_z_from_renderer`

In one line:

> Exported depth is inverse probe-view camera-z depth from the renderer and is converted back to metric camera-z before world-space back-projection.

## Exact Validity-Mask Rule

The paper-facing v1 protocol uses the following rule:

- render an `opacity` map in `[0, 1]` on the same probe view as the depth map
- a pixel is valid iff:
  - recovered metric camera-z depth is finite
  - recovered metric camera-z depth `> 1e-6`
  - `opacity` is finite
  - `opacity >= 0.5`

In one line:

> Validity mask = finite positive recovered metric depth AND probe-view opacity >= 0.5.

## Exact ROI Construction Rule

The shared ROI must be method-agnostic and training-side only.

Per scene:

1. Load the training-side sparse point cloud in the shared pose-controlled frame.
2. Remove non-finite points.
3. Compute an axis-aligned robust bounding box using per-axis quantiles:
   - lower quantile = `1%`
   - upper quantile = `99%`
4. Expand the resulting box on every axis by `2%` of the robust-box diagonal.
5. Save the final expanded box as the scene ROI.

This ROI:

- is identical for all methods
- is identical for odd/even runs
- does not depend on model outputs

In one line:

> Shared ROI = training-side sparse-point robust AABB (1%-99% per-axis) expanded by 2% of its diagonal.

## Exact Scene-Diagonal Definition

The scene diagonal is defined from the final saved ROI:

- `scene_diagonal = ||roi_bbox_max - roi_bbox_min||_2`

All distance thresholds and the default voxel size are derived from this exact value.

## Exact Voxel Size

The evaluator uses one deterministic voxel size per scene:

- `voxel_size = 0.001 * scene_diagonal`

That is:

- voxel size = `0.1%` of scene diagonal

## Exact Thresholds

The paper-facing thresholds are:

- `0.5%` of scene diagonal
- `1.0%` of scene diagonal
- `2.0%` of scene diagonal

Main-text reporting target:

- `Precision / Recall / F-score @ 1% scene diagonal`

Supplementary reporting target:

- `Precision / Recall / F-score @ 0.5%, 1%, 2% scene diagonal`

## Exact Distance Domain

Distances are computed:

- after depth back-projection into world coordinates
- after shared ROI crop
- after deterministic voxel downsampling

In one line:

> All nearest-neighbor distances are measured on ROI-cropped, voxel-downsampled world-space point clouds.

## Exact Probe-View Rendering Resolution Rule

- all methods must use the same probe-view rendering resolution for a given scene
- the saved manifest must record the width/height of every probe view
- the evaluator validates that all odd/even files match the manifest resolution

## Exact Point-Cloud Aggregation Rule

For each split (`odd` or `even`):

1. sort probe views deterministically by `view_id`
2. back-project valid pixels in row-major image order
3. concatenate all world-space points in that deterministic order
4. crop to the shared ROI
5. voxel-downsample deterministically

## Deterministic Voxel Downsampling

Voxel downsampling is deterministic:

- voxel index = `floor((point - roi_bbox_min) / voxel_size)`
- points are grouped by voxel index
- within each occupied voxel, keep the first point in the deterministic concatenation order above

No random sampling is used.

## Saved Artifacts

For every evaluated scene/method pair, save at least:

- shared ROI JSON
- scene diagonal and threshold JSON
- odd point cloud after ROI crop and after voxel downsampling
- even point cloud after ROI crop and after voxel downsampling
- per-threshold `P/R/F` tables
- manifest snapshot used for the run

## Minimal Paper-Facing Statement

Recommended wording:

> Under a shared pose-controlled probe-view protocol, our method produces more repeatable and self-consistent geometry across disjoint-view training splits, as evidenced by higher bidirectional cross-subset 3D precision/recall/F-scores on the evaluated scenes.

Mandatory caveats:

- this is **not** an absolute geometric accuracy test
- this is **not** end-to-end pipeline repeatability
- it quantifies repeatability/self-consistency under a shared pose-controlled evaluation frame
