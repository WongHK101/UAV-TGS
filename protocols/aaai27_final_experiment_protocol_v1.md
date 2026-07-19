# UAV-TGS AAAI-27 Final Experiment Protocol v1

Protocol ID: `uav-tgs-aaai27-final-experiment-v1`  
Version: `1.0.0`  
Status: preregistered and fail-closed  
Formal collection hash: `c6f32a1c44f49a725a62beeb105ffb37f5de265c5b513f5d01bd303439d60832`

This document is the complete execution contract for the final UAV-TGS AAAI-27
experiment batch. It supersedes the earlier pilot protocol fixture for this
batch only. Legacy behavior remains available for historical reproduction, but
no earlier pilot gate, result, split, or method-selection rule may silently
replace a rule below.

## 1. Frozen evidence boundary

The formal collection contains exactly 8,232 observations across 11 scenes:
6,763 train, 445 guard, and 1,024 test. The formal split uses seed
`uav-tgs-aaai27-v1`, 16-frame blocks, scene-level test-block budgets, and four
guard frames around each selected test block where available. Its identities
are:

- collection hash: `c6f32a1c44f49a725a62beeb105ffb37f5de265c5b513f5d01bd303439d60832`;
- collection split hash: `f23259152b46e75db32770c1bea815d0e168d8a5654c5139876bbe7ebcc3041d`;
- formal rule hash: `7017ec53e5699504208f6393415c1e97fd768b924a7325d3ab9a83b1e760b82c`.

The exact scenes are Building, Garden, InternalRoad, Orchard, PVpanel, Plaza,
Road, TransmissionTower, Urban20K, Urban50K, and Urban100K.

The six representative scenes are, in frozen order:

1. Building: built structures and the existing formal anchor/reference audit;
2. InternalRoad: oblique road views and the strongest known front-intrusion case;
3. PVpanel: compact repetitive panels and hotspot-bearing thermal content;
4. TransmissionTower: thin structures and difficult background separation;
5. Urban20K: a mid-large urban coverage/scale stress case;
6. Orchard: vegetation and repetitive agricultural texture.

The five expansion scenes are Garden, Plaza, Road, Urban50K, and Urban100K.

Original data are read-only. Every derived asset and run must use an isolated,
traceable directory. The formal collection, CFR output, canonical Hot-Iron
observations, float32 Celsius targets, camera protocol, and reference-depth
backend are immutable during this batch. A missing or mismatched asset fails
closed; it is not regenerated with a different rule.

Train data may be used for fitting and calibration. Checkpoint/model selection
may use train and guard. Preregistered main-table geometry-threshold selection
uses the six-scene internal guard evidence only. Hotspot thresholds come from
the train-only temperature distribution.
Test data are report-only. Test results must never select a method, adapter,
checkpoint, threshold, seed, fallback, or hyperparameter. Every such choice
must be frozen in a receipt before the corresponding test aggregation starts.

## 2. Frozen methods

### 2.1 Internal methods

`Raw-F3` is the fidelity-oriented internal method. It starts from the frozen
formal raw RGB geometry anchor and performs strict F3 thermal transfer: SH3
cold restart, fresh Adam, only `f_dc/f_rest` trainable, and exact freeze of
xyz/scaling/rotation/opacity/topology. It applies no thermal scale clamp,
densification, pruning, or opacity adaptation.

`SCSP-Refit+F3` is the primary geometry-stable/inspection-oriented method. It
starts from the frozen SCSP anchor, performs the preregistered RGB SH-only
appearance refit with fresh optimizer and fixed camera sequence, then applies
the same strict F3 transfer as Raw-F3. Geometry, opacity, and topology must be
exact across both appearance-only stages. SCSP parameters are frozen by the
implementation commit and may not be retuned per scene. When the SCSP manifest
proves that zero Gaussians were changed, an explicit no-op alias receipt may
bind the result to Raw-F3; duplicate training must not be fabricated.

`Adaptive Opacity + Scale-Clamp` is the adaptive baseline. Its code ID remains
`legacy_l`, but publication-facing data must use the display name in this
sentence rather than calling it legacy. It keeps the frozen recipe: the declared
legacy scale clamp and thermal opacity adaptation are enabled and fully
reported. It is not described as strict occupancy conservation and is not the
primary method.

Phase 1 runs all three internal methods on all six representative scenes.
Phase 2 runs only SCSP-Refit+F3 on the five expansion scenes, completing the
primary method on all 11 scenes. Raw-F3 and Adaptive Opacity + Scale-Clamp are
not expanded beyond the six representative scenes in this protocol.

### 2.2 External methods and order

External methods are qualified and executed in this exact order:

1. Thermal3D-GS
2. ThermalGaussian-OMMG
3. MMOne
4. ThermoNeRF
5. PhysIR-Splat

ThermoSplat, MrGS, unavailable/non-published-code methods, and non-LWIR
methods are outside this matrix. ThermalGaussian MFTG/MSMG may remain only as
older or supplemental evidence; neither may substitute for the frozen OMMG
six-scene row. MS-Splatting is explicitly prohibited in Section 6.

Each method must have an isolated adapter that records the official repository
URL, exact source commit, environment, input conversion, camera mapping, split
mapping, output conversion, and any unavoidable semantic limitation. Adapters
may only perform the minimum interface conversion needed for the frozen common
protocol. They may not change the method to improve a result.

The unified adapter contract has four explicit boundaries:

- input: frozen scene ID, train/guard/test lists, formal cameras, canonical
  RGB/thermal observations, radiometry/range/LUT identities, and method-owned
  preprocessing declarations;
- execution: official recipe plus an immutable compatibility patch, with exact
  command, seed, optimizer/scheduler, iteration/resolution, environment and
  resume lineage;
- endpoint: checkpoint/model identity, supported render branches, alpha or ray
  support, contribution semantics, native temperature/radiance semantics, and
  completion/failure status;
- evaluation: formal RGB when supported, formal thermal, alpha/support mask,
  three equivalent depth definitions or `UNSUPPORTED`, native temperature or
  radiance when supported, and complete provenance/cost receipts.

Training provenance and evaluation provenance are separate scoped records.
The former binds source/patch, environment, formal inputs, recipe, random
state, command, resume parent, endpoint and training costs. The latter binds
endpoint, adapter/evaluator version, cameras/split, LUT/range/support mask,
reference backend, metric schema, output hashes, valid counts and finite-value
checks. A training receipt cannot silently stand in for an evaluation receipt.

For each external method, execution is strictly:

1. implement and test the adapter;
2. run a Building smoke test;
3. run Building formal evaluation with the full metric set;
4. create the Building qualification package and enter `WAITING_GPT`;
5. only after explicit GPT approval, run InternalRoad, PVpanel,
   TransmissionTower, Urban20K, and Orchard;
6. create the six-scene completion package.

The six-scene completion package is not a blocking review gate. Once it is
generated and locally verified, work proceeds directly to the next external
method's adapter and Building qualification. No extra `WAITING_GPT` state is
inserted after external six-scene completion.

Old PhysIR hold-8/r4 or legacy-display results remain preliminary and cannot be
promoted into this matrix. Every external result must be generated under the
same frozen collection, cameras, CFR inputs, reference protocol, and metric
definitions used by the internal methods.

Building smoke is technical only and never enters formal statistics. It may be
shortened, but it must prove formal data loading, camera/split injection,
forward and backward execution, checkpoint saving, test-view rendering, finite
outputs, and evaluator-adapter readability. Building formal training then
starts from scratch with the frozen official recipe; it may not resume the
smoke checkpoint or use test results to choose a checkpoint or parameter.

The Building qualification gate is an implementation/fairness gate, not a
performance gate. Its package must show official source/environment/patch
provenance, formal configuration, smoke and training receipts, every supported
common metric, eight-threshold geometry CSVs, representative qualitative
outputs, unsupported-metric explanations, and actual GPU/wall-clock/disk cost.
Poor Building performance alone cannot reject a method. After approval, the
Building endpoint is reused and is not retrained.

Method-specific qualification checks are frozen as follows:

- Thermal3D-GS: shared formal camera injection, contribution-based depth
  export, thermal-only versus RGB output boundary, and unchanged official
  training semantics;
- ThermalGaussian-OMMG: the OMMG branch only, its isolated rasterizer
  environment, no substitution of another ThermalGaussian variant, and a
  common depth meaning for modality Gaussians;
- MMOne: the official RGB-thermal branch, Blackwell/modern-PyTorch patches
  limited to mechanical compatibility, explicit shared versus modality
  geometry, and no change to decomposition or densification logic;
- ThermoNeRF: fixed formal cameras, disabled camera optimizer and test-pose
  refinement, equivalent ray-weight median/expected/max depth, and frozen
  training iterations/resolution;
- PhysIR-Splat: the full official physical renderer rather than free thermal
  SH, fair shared-camera/VGGT-IR handling, explicit apparent-radiance or
  temperature semantics, emissivity/environment/atmosphere inputs, and all
  method-specific VGGT/feature preprocessing cost.

If a common metric is structurally unavailable, the adapter records
`UNSUPPORTED` and the technical reason. It may not substitute a differently
defined native metric. A normal single-scene failure is recorded as `FAILED`
and independent scenes continue. Only the same irreparable failure signature
across multiple scenes, or proof that a fair adapter is impossible, may stop
the remaining scenes for that method.

## 3. Mandatory outputs and metrics

Every formal method-scene endpoint must provide a provenance manifest, command,
source and adapter commits, input/split/camera identities, completion or failure
status, checkpoint/endpoint identity, and efficiency scope. Failed runs remain
in the matrix as failures; they are not silently omitted or replaced.

Each endpoint is rendered once per required modality and the render bundle is
reused across appearance, temperature, hotspot and qualitative consumers. One
depth/contribution bundle is reused to compute all eight thresholds and all
three depth definitions. Evaluators must not trigger hidden retraining or a
second semantically different render.

The common metrics are:

- RGB and thermal appearance: PSNR, SSIM, and LPIPS;
- TSDK-referenced apparent-temperature consistency: MAE, RMSE, signed bias,
  P95 absolute error, clipping, and off-LUT distance;
- geometry: missing rate, mean/median/signed depth residual, front/agreement/
  behind curves, signed residual CDF, per-block and nadir/oblique aggregates;
- hotspot reporting: train-only-frozen threshold, AUPRC, IoU, precision, and
  recall where the method produces a compatible output;
- efficiency: training wall time and scope, peak VRAM, render time/FPS and
  resolution, Gaussian/model count or an explicitly marked non-Gaussian model
  size, and failure cost where applicable;
- exact geometry/opacity/topology invariants for methods that claim shared or
  frozen occupancy.

Geometry is evaluated with all three registered definitions:

1. transmittance median depth;
2. alpha-weighted expected depth;
3. maximum-contribution surface depth.

Every definition must export curves at exactly
`[0.25, 0.5, 1, 2, 5, 10, 15, 20]` metres. All eight thresholds are retained
for curves and supplementary reporting.

For valid reference pixels, the common definitions are:

- `front(tau) = P(D_render < D_ref - tau)`;
- `agreement(tau) = P(abs(D_render - D_ref) <= tau)`;
- `behind(tau) = P(D_render > D_ref + tau)`.

Missing is reported separately. GS and NeRF adapters may compute ray weights
differently internally, but the exported depth definitions and masks must be
equivalent. A method that cannot expose equivalent weights records the affected
geometry diagnostics as `UNSUPPORTED`; native depth with another meaning cannot
be relabelled as a common metric.

The finite-positive-depth `metric-only` support path is retained only for
adapter development, smoke tests, or a method that genuinely cannot export
equivalent contribution support. Whenever a method can export alpha,
accumulated opacity, ray weights, or Gaussian contributions, its formal
geometry evaluation must use that contribution-weighted path. ThermoNeRF must
prefer ray weights and accumulated opacity; selected GS methods must prefer
Gaussian alpha/contribution. An adapter may not silently select `metric-only`
to reduce implementation work. If equivalent contribution support is truly
unavailable, the Building qualification records the technical reason and marks
the affected formal geometry metrics `UNSUPPORTED` or
`secondary_metric_only`. Such secondary results are excluded from the primary
contribution-based paired macro. This clarification does not alter the three
internal Gaussian methods in Phase 1.

All curve CSVs contain the eight unsmoothed sample points. Figures use threshold
in metres, preferably a logarithmic x-axis, and add no hidden samples or
interpolation. The scalar geometry summary, when required for Pareto analysis,
is named `front_auc_log_0p25_20m`: trapezoidal integration in log-threshold
coordinates over exactly the eight points, normalized to `[0,1]`. It must not
be mixed with any earlier 0.25-to-5 m or 1/2/5 m AUC.

Main-table geometry threshold selection is made from the six representative
scenes' guard results for the three internal methods and frozen before any
external formal test begins. For transmittance-median front and agreement, the
separation score at each threshold is the mean of the two signals' scene-median
pairwise absolute differences among the three internal methods. Select the
highest-score local threshold from `[0.25, 0.5, 1, 2]` and the highest-score
mid-scale threshold from `[5, 10]`. Optionally add the highest-score large
threshold from `[15, 20]` only if its score is at least half the maximum selected
score and its guard metric vector has absolute Pearson correlation below 0.95
with both selected vectors. Ties choose the smaller threshold. This yields two
or three thresholds without using external results. If uniform guard reference-
depth cannot be completed across all six scenes, the preregistered fallback is
the fixed set `[1, 5, 10]` metres. Neither selection nor fallback may inspect
external test rankings. All eight thresholds remain mandatory in curves and
supplementary CSVs.

The selection/fallback decision requires one frozen guard-completeness receipt.
For every representative scene, that receipt binds the guard split, camera,
reference-depth manifest, reference backend and support-mask hashes, the
expected and validated guard-view counts, and a `COMPLETE` or `INCOMPLETE`
status with a failure reason. The guard-only selection algorithm is used if and
only if all six rows are `COMPLETE` under the same registered reference
protocol. Otherwise the receipt deterministically activates `[1, 5, 10]`.
The receipt is frozen before test aggregation and may not read train metrics,
test metrics, or any external-method result.

Temperature evaluation uses native apparent-temperature/radiance through a
fixed benchmark conversion where available. Pseudocolor outputs use the common
canonical LUT and train-only scene range. No method-specific test calibration
is permitted; an ambiguous conversion is `N/A`. Temperature output includes
MAE, RMSE, bias, P95 absolute error, and off-LUT mean/P95. RGB metrics are `N/A`
for a thermal-only method; an RGB image is never synthesized solely to fill the
table.

### 3.1 Paired macro, coverage, and failures

The primary cross-method statistic is metric-specific and uses one comparison-
group common-scene set: the intersection valid for every method in that named
comparison group and metric. All rows in that group use exactly the same scene
set; pair-by-pair intersections are forbidden. Every aggregate records group,
metric, `common_scene_count`, and the exact scene list; win/tie/loss is computed
only on those paired scenes. Coverage records valid, failed,
unsupported, and alias scene counts plus failure signatures. An available-
scene macro may be supplied only as a secondary result with metric name,
`n_valid`, and its scene list. Failed/unsupported cells remain explicit and are
never imputed. A method missing RGB, geometry, or temperature output therefore
changes only the corresponding metric's paired set, not unrelated metrics.

### 3.2 Cost accounting

Method-specific cost includes post-anchor training, method-only preprocessing,
peak PyTorch allocated/reserved memory, parameter/Gaussian count, model size,
render time/FPS with resolution, and completed-scene count. Shared CFR, TSDK
decode, common COLMAP, common RGB Stage-1 anchor, OpenMVS reference, and common
evaluation are excluded from method-specific cost. VGGT-IR, independent thermal
camera recovery, or proprietary feature preprocessing are included when unique
to an external method.

SCSP-Refit+F3 cost is SCSP projection plus 5k RGB SH-only refit plus 30k strict
F3. A no-op alias may have zero incremental batch execution, but its reported
method cost equals Raw-F3 and must never be reported as zero.

### 3.3 Frozen table and figure data products

Phase outputs are data products only; this protocol does not authorize paper
writing.

- Table 1 covers SCSP-Refit+F3 on all 11 scenes. Panel A contains scene-level
  RGB/T PSNR, SSIM, LPIPS, temperature MAE/RMSE, and hotspot AUPRC. Panel B
  contains the frozen median front/agreement thresholds, median absolute depth
  error, missing, Gaussian count, post-anchor method time, and completion/alias.
  It also exports scene macro, worst scene, completion fraction, and SCSP no-op
  ratio.
- Table 2 compares all internal and external methods on the paired/common six-
  scene set. It exports applicable RGB LPIPS, T-PSNR/SSIM/LPIPS, temperature
  MAE/RMSE, hotspot AUPRC, frozen median front/agreement thresholds, median
  absolute depth error, missing, training time, VRAM, model size/count, and
  completed/6.
- Table 3 is the simplified six-scene internal comparison of Raw-F3,
  SCSP-Refit+F3, and Adaptive Opacity + Scale-Clamp. It includes RGB/T LPIPS,
  temperature MAE, AUPRC, one or two frozen median front thresholds, median
  depth error, and post-anchor method time. Raw anchor, SCSP zero-refit, and
  SCSP-Refit are optional existing-anchor supplemental rows, not new training
  routes.
- The main geometry figure is the paired six-scene macro median front and
  agreement curves for all methods. Supplementary data contain behind curves,
  all eight values, per-scene curves, expected/max-contribution curves, and
  nadir/oblique curves. Every plotted series has a source CSV.

The machine-readable result tree separates `endpoints/`, `metrics/appearance`,
`metrics/temperature`, `metrics/hotspot`, `metrics/geometry/{definition}`,
`efficiency/`, `coverage/`, `tables/`, `curves/`, `qualitative/`, and
`receipts/`. Aggregators consume manifests and scoped hashes, never directory
name guesses.

### 3.4 Scoped endpoint reuse

An existing endpoint is reusable only when its scoped provenance matches the
formal run: training core, renderer, recipe, optimizer/scheduler, split/camera/
range, LUT/support mask, evaluator/reference protocol, and input anchor/data.
Changes limited to reports, ZIP tooling, Resplat, matrix runners, or
documentation do not invalidate an endpoint. Reuse and alias decisions require
receipts; unrelated repository commit changes cannot force retraining.

## 4. Mandatory review gates

There are exactly nine blocking review gates:

1. Phase 0 protocol and asset freeze;
2. Phase 1 six-scene internal matrix;
3. Phase 2 all-11-scene primary-method completion;
4. Thermal3D-GS Building qualification;
5. ThermalGaussian-OMMG Building qualification;
6. MMOne Building qualification;
7. ThermoNeRF Building qualification;
8. PhysIR-Splat Building qualification;
9. Phase 8 final audit.

This list is the static gate definition, not an approval receipt. Every listed
gate has definition state `DEFINED`. Runtime state is authoritative only in a
hash-bound gate receipt. GPT approval of Phase 0 is recorded in
`protocols/receipts/phase0_gpt_approval_receipt.json`; the current snapshots are
Phase 0 `APPROVED`, Phase 1 `READY`, and every later gate
`BLOCKED_BY_PREDECESSOR`. A later gate cannot become `WAITING_GPT` until its own
package is complete and all predecessor approvals or locally verified
non-blocking completions have been recorded.

At each gate, Codex creates the specified compact review package, verifies its
manifest, records `WAITING_GPT`, and performs no downstream gated work until an
explicit GPT approval receipt is recorded. Ordinary implementation fixes and
non-gated external six-scene completion packages do not create new review
gates. A gate package must expose failures and missing assets rather than
manufacture a passing result.

## 5. Execution phases

### Phase 0 - Protocol, collection, and asset freeze

Version this Markdown contract, its machine-readable JSON, and a collection/
asset inventory. Verify code, collection, formal scene manifests, radiometry,
cameras, reference assets, internal anchors, external source availability, GPU
environment, storage, and output roots. No formal training is allowed. Produce
the Phase 0 package and enter `WAITING_GPT`.

The frozen Phase-0 availability snapshot is intentionally honest about missing
work:

- formal split/data entries exist for all 11 scenes;
- complete formal radiometry currently exists only for Building and
  InternalRoad; Urban20K has decoded temperature NPY only; the other eight
  scenes are incomplete;
- guard/reference binding is verified only for Building and InternalRoad;
- Building has Adaptive Opacity + Scale-Clamp and Raw-F3 endpoints plus a
  SCSP no-op alias; InternalRoad has all three formal internal endpoints;
  other scene endpoints are missing;
- all five external immutable source snapshots exist, but formal adapters and
  formal Building endpoints do not;
- for each of the four missing representative scenes (PVpanel,
  TransmissionTower, Urban20K, and Orchard), let `r_s=1` exactly when its
  read-only train-side SCSP manifest reports `modified_gaussian_count > 0`, and
  `r_s=0` otherwise. Define `R=sum_s r_s`, so `R` is an integer in `[0,4]`.
  A no-op scene requires three new training jobs: formal raw RGB anchor,
  Raw-F3, and Adaptive Opacity + Scale-Clamp. A modified scene requires those
  three plus SCSP RGB SH-only refit and SCSP-Refit+F3. Hence Phase 1 is
  `sum_s(3+2r_s)=12+2R`, or 12 to 20 new training jobs;
- for each of the five expansion scenes, let `m_s=1` under the same
  `modified_gaussian_count > 0` rule and `m_s=0` otherwise. Define
  `M5=sum_s m_s`, so `M5` is an integer in `[0,5]`. A no-op scene requires a
  formal raw RGB anchor plus Raw-F3 used by the exact no-op alias. A modified
  scene requires the formal raw RGB anchor, SCSP RGB SH-only refit, and
  SCSP-Refit+F3. Hence Phase 2 is `sum_s(2+m_s)=10+M5`, or 10 to 15 new
  training jobs.

Because radiometry/reference assets are incomplete, Phase 0 freezes only these
conditional matrices, formulas and ranges. Before any scene training, a
preflight receipt binds the read-only train-side SCSP manifest path and SHA,
`modified_gaussian_count`, the binary indicator, and the resulting job branch.
It reads neither guard nor test data and does not retune the rule. The Phase-0
package marks `R` and `M5` unresolved rather than inventing exact counts.

The versioned asset inventory carries the exact scene statuses, repository
commits, archive hashes, dependency state, and adapter risks. Availability is a
snapshot, not permission to skip runtime hash binding.

### Phase 1 - Internal representative-scene matrix

After Phase 0 approval, run Raw-F3, SCSP-Refit+F3, and Adaptive Opacity +
Scale-Clamp on the six representative scenes. Produce all common metrics,
invariants, cost records, fixed qualitative views, and explicit failures.
Complete/verify the six-scene guard reference and freeze the main-table
threshold-selection (or fallback) receipt before any external formal test.
Produce the Phase 1 package and enter `WAITING_GPT`.

### Phase 2 - Primary method on the expansion scenes

After Phase 1 approval, run only SCSP-Refit+F3 on Garden, Plaza, Road,
Urban50K, and Urban100K. Aggregate the primary method over all 11 scenes,
produce the Phase 2 package, and enter `WAITING_GPT`.

### Phases 3-7 - External methods

After Phase 2 approval, execute one phase per external method in the frozen
order: Thermal3D-GS, ThermalGaussian-OMMG, MMOne, ThermoNeRF, PhysIR-Splat.
Each phase applies the adapter -> Building smoke -> Building formal -> blocking
Building qualification -> approved remaining-five execution -> non-blocking
six-scene package sequence in Section 2.2.

### Phase 8 - Final aggregation and audit

After all five external phases, validate the complete internal and external
matrices, provenance, fairness, missing/failure rows, metric domains, geometry
curves, cost scopes, fixed qualitative views, and package hashes. Create the
final compact review package and enter `WAITING_GPT`. Phase 8 does not include
paper drafting or claim writing.

### 5.1 Scheduling, resume, and failure handling

The normative dual-host policy is
`protocols/aaai27_compute_resource_policy_v1.json` (file SHA-256
`395848efb1450bd778e26946b24baccd3b16580b139c23fa2cba45138b19424b`;
canonical payload SHA-256
`958ae2c8d861b30e7922fd851cd1464fef1e913af2dd50acc3ecfd5802823691`).
Host 900 is authoritative storage. Host 901 is temporary scratch compute under
`/root/autodl-tmp/UAV-TGS-901`; formal outputs produced there must return to
900 and pass count, size, and SHA-256 verification before related 901 data are
removed. The resource policy changes scheduling only and cannot change formal
inputs or metrics.

This protocol does not pre-allocate formal work to server 900 or 901. Before
each downstream phase, the phase plan submitted to GPT identifies the concrete
device, GPU model/configuration, runtime stack and expected resource envelope.
Independent jobs may run in parallel across approved devices only when the
frozen protocol and hardware-comparability policy remain unchanged. Every run
records a device/environment receipt; cross-device efficiency comparisons must
either use matched hardware/runtime or be reported as non-comparable. Local CPU
radiometry, asset preparation, packaging and eligible evaluation may overlap
GPU training; GPU evaluation may overlap only when memory isolation is safe and
recorded. Queue order, cache, CLI/API, adapter structure, checkpoint retention
and minor bug fixes are implementation choices within the frozen semantics.

Every run uses an isolated output root and an append-only receipt. Resume is
allowed only from an endpoint with matching training-scoped hashes and records
the parent checkpoint and already-consumed iterations; it must not repeat a
completed formal training stage. A failed scene records its elapsed GPU/wall
time, disk use, failure signature and last valid artifact, then independent
scenes continue. Cleanup may remove reproducible caches and redundant
checkpoints only after their retained endpoint/receipt/hash is verified; raw
data and formal evidence are never cleaned.

Intermediate execution pauses are limited to: discovered data leakage; a
required change to formal split/CFR/reference; the same systemic failure over
multiple scenes; proof that fair adaptation is impossible; resources materially
exceeding the Phase-0 budget; or a fact overturning a frozen method definition.
Ordinary poor performance, a small regression, or one scene failure is not a
pause condition.

### 5.2 Packages and audit scope

Generate one compact package for each of Phase 0, Phase 1 and Phase 2; one for
each external Building qualification; one non-blocking completion package for
each external six-scene phase; and one Phase 8 package. Every package has one
payload-SHA manifest. Only the nine gates in Section 4 enter `WAITING_GPT`.
Packages do not contain checkpoints, datasets, SDKs or large render dumps.

Required audit evidence is limited to scoped split/data/radiometry/config/code
hashes; train/guard/test scope; schema/index/count checks; NaN/Inf checks;
claimed frozen-field invariants; endpoint/evaluation provenance; and original-
data read-only verification. The batch does not require multiple unzip tools,
per-scene ZIPs, per-commit status reports, cross-platform duplicate file hashes,
repeated full-repository tests, or stopping on a small performance regression.

## 6. Prohibited work

The following are prohibited for this batch:

- OCT-Scalar, OCT-Residual, or any OCT-GS expansion;
- surface-aware Resplat or any resplat variant;
- OGS-v1, OGS-v2, or any OGS variant;
- new heuristics, adaptive thresholds, method variants, or parameter grids;
- multi-seed or seed-search experiments;
- any use of test results for tuning, selection, fallback, or debugging choices;
- MS-Splatting implementation or evaluation;
- paper, rebuttal, abstract, claim, or manuscript writing;
- changing the formal split, CFR, radiometry target, camera protocol,
  reference-depth backend, or original data;
- unapproved external-method scene expansion before its Building gate;
- adding blocking review gates beyond the nine listed above.

Unexpected incompatibility or catastrophic failure must be preserved as an
auditable failure and handled at the next applicable mandatory gate. It does
not authorize a new heuristic, threshold search, protocol rewrite, or hidden
substitution.

## 7. Completion condition

The required matrix contains Raw-F3 on six scenes, SCSP-Refit+F3 on 11 scenes,
Adaptive Opacity + Scale-Clamp on six scenes, and each of five external methods
on six scenes. Aliases, failures and unsupported metrics remain explicit rows.
The batch is complete only when Phase 8 has produced a hash-verified package
covering every required method-scene row, including explicit failure or
not-applicable receipts, and has entered `WAITING_GPT`. GPT approval of Phase 8
is an external review outcome; it does not retroactively authorize paper
writing within this protocol.
