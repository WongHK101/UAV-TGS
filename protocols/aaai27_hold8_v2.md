# UAV-TGS AAAI27 Hold-8 v2 Formal Experiment Protocol

Protocol ID: `uav-tgs-aaai27-hold8-v2`

Status: `WAITING_GPT_PHASE0_V2`; Phase 0-v2 is complete, but formal training
is not authorized until the GPT review gate is approved. This protocol
supersedes the formal-table role of `aaai27_final_experiment_protocol_v1`;
the earlier guard4 evidence remains archived under the boundary below.

## 1. Evidence boundary and archived guard4 work

The superseded guard4 v1 protocol is retained only as
`exploratory_internal_ablation_only`. Its 18 endpoint appearance/cost
summaries, 18 guard-geometry summaries, five completed formal evaluations,
invariants, SCSP counts, and fixed qualitative results may be used for internal
mechanism analysis. They must not be mixed with Hold-8 formal tables or macros,
used to infer a Hold-8 SCSP alias, or represented as a complete six-scene
temperature/hotspot comparison. No remaining guard4 formal evaluations will be
run.

Hold-8 is described as a **community-standard hold-8 view-interpolation
split**. It is not a spatially isolated split and does not establish
out-of-region generalization.

## 2. Collection and split

The frozen collection contains 11 scenes: Building, Garden, InternalRoad,
Orchard, PVpanel, Plaza, Road, TransmissionTower, Urban20K, Urban50K, and
Urban100K. For each scene, canonical RGB-thermal pair IDs are sorted with a
numeric-aware natural sort. The zero-based sorted position defines membership:

- `index % 8 == 0`: test;
- every other index: train.

There is no guard or validation subset. Missing numeric IDs do not create
phantom positions and do not change this position-based rule. All internal and
external methods must consume the same explicit train/test lists and hashes;
repository-specific implicit directory ordering is prohibited.

Each scene manifest records the pair-ordering hash, train-list hash,
test-list hash, source collection hash, split hash, and split-generator source
identity. The generated six-scene and 11-scene counts are fail-closed against
the approved Phase 0-v2 expected counts.

| Scene | Total | Train | Test |
|---|---:|---:|---:|
| Building | 614 | 537 | 77 |
| Garden | 656 | 574 | 82 |
| InternalRoad | 559 | 489 | 70 |
| Orchard | 588 | 514 | 74 |
| PVpanel | 289 | 252 | 37 |
| Plaza | 668 | 584 | 84 |
| Road | 467 | 408 | 59 |
| TransmissionTower | 673 | 588 | 85 |
| Urban100K | 1671 | 1462 | 209 |
| Urban20K | 748 | 654 | 94 |
| Urban50K | 1299 | 1136 | 163 |
| **All 11** | **8232** | **7198** | **1034** |

The six representative scenes contain 3471 pairs: 3034 train and 437 test.

## 3. Common preprocessing and radiometry

The following assets are split-independent and may be reused only after their
identity is verified: CFR observations, full-frame TSDK float32 Celsius maps,
the fixed Hot-Iron LUT, per-frame valid masks, and full-collection SfM
cameras/sparse models. Pose reconstruction is common preprocessing;
photometric model training reads only Hold-8 train images.

Every scene receives new Hold-8-bound assets:

1. decoded-temperature binding and explicit train/test lists;
2. `Tmin/Tmax` estimated from train temperature only, with test used only for
   post-estimation clipping QA;
3. lossless canonical Hot-Iron PNGs rendered with that frozen scene range;
4. a hotspot threshold estimated from train temperature and split-independent
   valid masks only;
5. an OpenMVS reconstruction built from Hold-8 train images and used to render
   reference depth at test cameras;
6. a new RGB 30k anchor, SCSP projection/refit, and all three internal Stage-2
   endpoints.

Guard4 range, canonical observations, hotspot thresholds, references, RGB
anchors, endpoints, and modified counts are not formal Hold-8 inputs.

## 4. Internal and external matrix

The six representative scenes are Building, InternalRoad, PVpanel,
TransmissionTower, Urban20K, and Orchard. Phase 1 runs:

- Raw-F3;
- SCSP-Refit + strict F3;
- Adaptive Opacity + Scale-Clamp (`legacy_l` remains an internal code ID).

SCSP no-op status is recomputed from each new Hold-8 anchor. A genuine no-op
may alias Raw-F3 without duplicate training, but the reported method cost equals
Raw-F3 rather than zero. Phase 2 expands only SCSP-Refit + strict F3 to all 11
scenes.

External methods remain Thermal3D-GS, ThermalGaussian-OMMG, MMOne,
ThermoNeRF, and PhysIR-Splat. Each method must pass a Building implementation
and fairness qualification before the other five representative scenes run.
Qualification is not a performance gate.

## 5. Formal metrics

### Appearance, temperature, and hotspots

Main metrics are RGB PSNR/SSIM/LPIPS when supported, thermal PSNR/SSIM/LPIPS,
TSDK-referenced apparent-temperature MAE/RMSE, and hotspot AUPRC. Temperature
bias/P95/off-LUT and hotspot IoU/precision/recall are supplementary.

### Geometry

The only mandatory common depth is camera-z alpha/volume-weighted expected
depth:

`D = sum(w_i * z_i) / sum(w_i)`.

The implementation uses the fixed constant `epsilon=1e-8`; it is never tuned by
method, scene, or result. A model pixel is missing when the weight sum is at or
below epsilon, there is no finite positive sample, or the final depth is not
finite. Missing pixels are excluded from the front, agreement, and median
absolute-error denominators. Missing rate uses the reference-valid denominator.
Reference and model manifests must bind the same collection, scene split, and
explicit test-list hashes before any metric is computed.

Main geometry reports Front and Agreement at 1/2/5 m, median absolute depth
error, and missing rate. Supplementary CSVs contain unsmoothed Front and
Agreement values at `[0.25, 0.5, 1, 2, 5, 10, 15, 20]` m. There is no formal
front AUC, behind table, guard threshold selection, mandatory three-depth
adapter, or all-method block/orientation aggregation. Existing median and
maximum-contribution depth remain InternalRoad mechanism diagnostics only.

An external method that cannot expose equivalent renderer weights records
geometry as `N/A` with a technical reason. Its core renderer is not rewritten,
and a semantically different native depth is not relabelled as common depth.

### Efficiency

Report method-specific total training time, peak VRAM, model size or Gaussian
count, and render FPS. Common CFR, TSDK decode, shared SfM, common RGB anchor,
OpenMVS reference construction, and common evaluation are excluded from
method-specific cost. Method-exclusive preprocessing is included.

## 6. Minimal receipts, aliases, failures, and aggregation

Each endpoint has a flat receipt containing method/repository/commit/runtime
patch, recipe/config/seed, split/data/camera/range/LUT hashes, host/GPU,
command, endpoint hash, completion state, training time, VRAM, and model size.
Training and evaluation each have one non-recursive scoped signature. Whole
repository changes unrelated to the scoped inputs do not invalidate an
endpoint.

An alias records its source endpoint, exact no-op evidence,
`independent_endpoint_run=false`, zero additional batch cost, and the effective
reported Raw-F3 method cost. It never creates an independent performance claim.

With complete coverage, report a six-scene macro. With failures or unsupported
outputs, report `completed x/6` and one per-metric common-scene macro with its
exact scene list. `N/A` RGB does not shrink thermal or temperature macros.
Failed/unsupported cells are never imputed.

## 7. Hosts, authority, and scheduling

Host 900 is the authoritative AutoDL archive. Host 901 is a temporary isolated
compute node rooted at `/root/autodl-tmp/UAV-TGS-901`. The Phase 1 scheduler
defaults to:

- 900: Building, InternalRoad, Urban20K;
- 901: PVpanel, TransmissionTower, Orchard.

This assignment is operational, not a protocol identity. A not-yet-started
scene may move after an asset/GPU/disk/runtime preflight. A started scene should
remain on one host through Stage 1 and all internal variants. Both hosts use
identical scoped code, split, recipe, and environment semantics. Every 901
result is returned to 900 and verified by file count, bytes, and SHA-256 before
any scratch cleanup. No formal result may exist only on 901.

## 8. Review gates and prohibited work

Blocking review gates remain Phase 0-v2, Phase 1, Phase 2, every external
method's Building qualification, and Phase 8 final audit. A completed external
six-scene report does not add another blocking gate before the next method's
Building work.

Phase 0-v2 does not authorize RGB 30k training, thermal 30k training, external
method training, or paper performance claims. Test data may not select a
method, recipe, checkpoint, threshold, seed, or fallback.

## 9. Guard4 archive cleanup

Cleanup is optional and cannot block Hold-8 preparation. Before deletion, the
inventory records relative path, size, and SHA-256. Final PLY, configuration,
cameras, exposure, recipes, code identity, scalar results, refined mesh,
compact reference bundles, and all receipts are retained. A checkpoint may be
deleted only after its fixed-view render and a reloaded final-PLY render agree
within the separately locked tolerance; any state not recoverable from PLY is
retained. The cleanup receipt is generated after deletion. Phase 0-v2 only
creates the inventory and readiness gates; it does not authorize deletion.
