# Thermal radiometry tools

These CLIs build a traceable, derived thermal benchmark without writing to the
raw R-JPEG tree.  DJI TSDK is an external dependency: do not copy its DLL/SO
files into Git.  Point the tools at a local installation with either
`--tsdk-root` or `DJI_TSDK_ROOT`.

## H30T compatibility probe

Audit a raw frame first:

```powershell
python tools/thermal_radiometry/audit_rjpeg.py RAW_T.JPG `
  --output-jsonl DERIVED/manifests/rjpeg_audit.jsonl `
  --hash-source
```

Decode one frame with the official `dji_irp` utility bundled in TSDK:

```powershell
python tools/thermal_radiometry/decode_temperature.py RAW_T.JPG `
  --output-dir DERIVED `
  --tsdk-root PATH_TO_DJI_TSDK `
  --probe --scene SCENE `
  --distance-m 5 --distance-m-source compatibility_probe `
  --humidity-percent 70 --humidity-percent-source benchmark_assumption `
  --emissivity 0.95 --emissivity-source benchmark_assumption `
  --ambient-c 25 --ambient-c-source benchmark_assumption `
  --reflected-c 23 --reflected-c-source benchmark_assumption
```

The decoder writes a two-dimensional float32 Celsius NPY plus per-frame
request/receipt manifests.  The default adapter is `builtin:dji_irp`.  A custom
adapter can be selected as `PATH.py:decode_rjpeg` or `module:decode_rjpeg`; its
call contract is documented at the top of `decode_temperature.py`.

Parameter values are never silently labeled as embedded.  Every value carries
an explicit source such as `per_frame_lrf`, `scene_embedded_median`,
`geometry_estimate`, or `benchmark_assumption`.

## Explicit per-scene decode protocol

Resolve an audit manifest into the complete, decode-ready protocol before a
batch run:

```powershell
python tools/thermal_radiometry/build_radiometry_protocol.py `
  --audit-manifest DERIVED/manifests/rjpeg_audit.jsonl `
  --output DERIVED/manifests/decode_protocol.jsonl `
  --summary-out DERIVED/qa/decode_protocol_summary.json `
  --scene SCENE `
  --scene-distance-m 30 `
  --scene-distance-provenance benchmark_assumption:scene_nominal_altitude `
  --humidity-percent 70 `
  --humidity-percent-source benchmark_assumption:aaai_pilot_v1 `
  --ambient-c 25 `
  --ambient-c-source benchmark_assumption:aaai_pilot_v1 `
  --reflected-c 23 `
  --reflected-c-source benchmark_assumption:aaai_pilot_v1
```

The default emissivity is the explicit benchmark assumption `0.95`.  Distance
is fixed per view/flight strip in this order: robust median of valid LRF
measurements, robust median of `relative altitude / sin(abs(gimbal pitch))`,
then the explicitly supplied scene assumption.  Each frame retains its raw
LRF fields, resolved strip ID, used distance and source, and fallback reason.
No missing environmental value is presented as embedded metadata.

Pass the result directly to the decoder:

```powershell
python tools/thermal_radiometry/decode_temperature.py `
  --input-manifest DERIVED/manifests/decode_protocol.jsonl `
  --output-dir DERIVED `
  --tsdk-root PATH_TO_DJI_TSDK
```

## Leakage-guarded split and train-only range

Build a deterministic split from the audit/protocol manifest:

```powershell
python tools/thermal_radiometry/build_split.py `
  --manifest DERIVED/manifests/audit_or_decode.jsonl `
  --output DERIVED/splits/split_manifest.json `
  --scene SCENE
```

The preferred route uses timestamp and gimbal strata.  If those fields are not
reliable for every frame, the complete scene uses natural filename order.  The
scene receives `round_half_up(scene_frames / (16 * 8))` complete test blocks.
Eligible blocks must leave at least 16 training frames in both their strip and
stratum; a largest-remainder allocation distributes the scene budget across
strata, and stable hashes select blocks within each stratum.  Short unsupported
strata therefore remain entirely in training instead of being forced into test.
Adjacent frames on each side of a selected block are `guard` and never train.
The frozen AAAI protocol uses four guard frames on each side; this is also the
default of `build_split.py`.

The earlier side-by-side guard=2/4 comparison remains available as development
QA only; guard=2 is not part of the formal protocol:

```powershell
python tools/thermal_radiometry/split_qa.py `
  --manifest DERIVED/manifests/all_scenes_audit.jsonl `
  --output DERIVED/qa/split_guard_2_4.json
```

The report includes per-scene budget and test fraction, scene/stratum/strip
counts, strata without a test block, fail-closed validation, cross-guard test
set overlap, metadata coverage/fallback, and for every test frame its nearest
usable train observation by time, GPS, and gimbal angle.

Freeze the complete 11-scene protocol, including a collection hash and one
manifest per scene, with:

```powershell
python tools/thermal_radiometry/build_formal_split.py `
  --manifest DERIVED/manifests/all_scenes_audit.jsonl `
  --output-root DERIVED/splits/aaai27_guard4
```

This command is intentionally fixed to block size 16, period 8, guard 4, and a
minimum of 16 training frames in every related strip and stratum.  A stratum may
legitimately receive no test block; leaving any strip or stratum without train
frames fails closed.

After decoded NPY paths have been attached to the split records, estimate a
fixed range using training frames only:

```powershell
python tools/thermal_radiometry/estimate_scene_range.py `
  --split-manifest DERIVED/splits/split_manifest.json `
  --output DERIVED/qa/range_manifest.json
```

The default range is the envelope of per-training-frame p0.1/p99.9 estimates
plus a 2% span margin.  Test maps are read only after the range is fixed and
contribute only clipping QA; guard maps are not read.

## Canonical palette, round-trip QA, and temperature evaluation

Render every float32 map with the same repository-owned, 256-entry Hot-Iron
LUT, linear scene range, gamma 1, and lossless PNG:

```powershell
python tools/thermal_radiometry/render_canonical_palette.py `
  --temperature-root DERIVED/temperature_c `
  --output-root DERIVED/canonical_hotiron `
  --range-manifest DERIVED/qa/range_manifest.json
```

The LUT is named `uav-tgs-hot-iron-v1`; its RGB-byte SHA and uniqueness count
are written to every manifest.  It is a repository-owned canonical palette,
not a claim that native DJI previews use the same color table.

Validate the canonical encoding before training:

```powershell
python tools/thermal_radiometry/validate_roundtrip.py `
  --temperature-root DERIVED/temperature_c `
  --canonical-root DERIVED/canonical_hotiron `
  --range-manifest DERIVED/qa/range_manifest.json `
  --report DERIVED/qa/roundtrip_report.json
```

The check fails if an in-range pixel exceeds half a temperature bin, a
canonical pixel is off the LUT, or clipping exceeds the configured limit.

Evaluate a thermal render against the original float32 maps with:

```powershell
python tools/thermal_radiometry/evaluate_temperature.py `
  --ground-truth-root DERIVED/temperature_c `
  --render-root RENDERED_THERMAL `
  --range-manifest DERIVED/qa/range_manifest.json `
  --split-manifest DERIVED/splits/split_manifest.json `
  --subset test `
  --mask-root RENDERED_ALPHA `
  --alpha-threshold 0.01 `
  --require-support `
  --report OUTPUT/temperature_metrics.json
```

`--subset` accepts `train`, `test`, or `guard` and requires the split manifest.
Masks mirror the render-relative paths and may be grayscale/RGBA images or
two-dimensional NPY arrays.  Mask values and RGBA render alpha are normalised
to `[0,1]`; a pixel is supported only when its value is strictly greater than
`--alpha-threshold`.  An external mask takes precedence over render alpha.  If
neither exists, an RGB render is treated as fully supported for legacy CLI
compatibility, but `support_is_explicit=false` and a warning are written to the
report.  Use `--require-support` for formal evaluation: it fails closed on a
missing render/mask, implicit RGB full-frame fallback, or a frame without any
supported pixels.

For a real model render, the primary reported quantity is
**palette-inverted TSDK-referenced apparent-temperature error** on supported
pixels.  The umbrella evaluation name remains **TSDK-referenced
apparent-temperature consistency**.  The report separately includes:

- supported-pixel and all-pixel-diagnostic MAE, RMSE, signed bias, and P95;
- pixel-micro aggregates and frame-macro mean/standard deviation;
- unsupported, missing-render, missing-mask, and combined ratios;
- supported/all-pixel clipping and off-LUT distance.

Missing renders or requested masks are counted rather than silently dropped.
`primary_metric_valid` is false and the report status is
`invalid_no_supported_pixels` when the supported domain is empty.  A valid
primary result with missing inputs has status `completed_with_missing`; it is
never labeled simply `complete`.
The all-pixel result is diagnostic only.  These quantities are not absolute
thermometry, true surface temperature, or physical ground truth.

## Float-temperature undistortion

Do not infer temperature by inverting a palette image after COLMAP
undistortion.  Remap decoded float32 Celsius maps directly with the input and
output sparse models:

```powershell
python tools/thermal_radiometry/undistort_temperature.py `
  --temperature-root DERIVED/temperature_c/Building `
  --input-model DATASET/distorted/sparse/0 `
  --output-model DATASET/undistorted/sparse/0 `
  --output-root DERIVED/undistorted_temperature/Building
```

The input temperature shape must exactly match its distorted COLMAP camera.
The supported path is `SIMPLE_RADIAL`, `RADIAL`, `SIMPLE_PINHOLE`, or `PINHOLE`
input to `PINHOLE`/`SIMPLE_PINHOLE` output.  All image associations, camera
models, poses, float32 dtypes, finite values, and shapes are preflighted before
an output directory is atomically published.  Each output map has a mirrored
boolean `valid_support` NPY; border-filled values outside that mask must not be
used for evaluation.  `manifest.json` pins model parameters and hashes of the
model, source maps, remapped maps, and masks.

## Formal shared support domain

Formal temperature evaluation uses one support domain shared by every Stage-2
method.  First render the frozen RGB anchor with image-name outputs and the
existing depth-evaluation opacity proxy:

```powershell
python render.py `
  -m OUTPUT/RGB_ANCHOR `
  --iteration 30000 `
  --skip_train `
  --save_by_image_name `
  --save_opacity_proxy
```

The proxy is explicitly
`black_bg_plus_white_override_color_render`; it is not described as native
rasterizer alpha.  Combine it once with the float-remap validity masks:

```powershell
python tools/thermal_radiometry/combine_formal_support.py `
  --split-manifest DERIVED/splits/guard4/scenes/Building.split.json `
  --valid-support-root DERIVED/undistorted_temperature/Building/valid_support `
  --valid-support-manifest DERIVED/undistorted_temperature/Building/manifest.json `
  --opacity-proxy-root OUTPUT/RGB_ANCHOR/test/ours_30000/opacity_proxy `
  --opacity-proxy-manifest OUTPUT/RGB_ANCHOR/test/ours_30000/render_mapping_manifest.json `
  --opacity-threshold 0.01 `
  --expected-test-count 80 `
  --output-root DERIVED/formal_support/Building
```

`--opacity-threshold` is deliberately explicit; `0.01` is the formal Building
example rather than a hidden evaluator default.  The command requires exactly
the 80 frozen test names, verifies both source manifests and every selected
source-file SHA-256, checks boolean/float32 dtypes and identical shapes, and
fails if any frame has empty combined support.  It atomically writes equivalent
boolean and float32 NPY trees plus a path-portable deterministic manifest,
portable content hash, and `manifest.sha256`.

Use the same generated float (or boolean) tree for L, C3, and F3.  Because the
threshold has already been applied by the combiner, formal evaluator calls use
`--mask-root DERIVED/formal_support/Building/float --alpha-threshold 0` with
`--require-support`; do not rebuild support from each thermal model.

## Palette-neutral display export

Formal models and metrics remain in the fixed canonical representation.  To
change only the visualization palette, project a canonical render onto the
repository's 256-entry canonical LUT, retain the resulting scalar index and
apparent-temperature sidecars, and map that index through an exact DJI TSDK
palette LUT:

```powershell
python tools/thermal_radiometry/tsdk_palette_display.py `
  --input-root OUTPUT/test/ours_60000/renders `
  --output-root OUTPUT/palette_display `
  --range-manifest DERIVED/radiometry/range_manifest.json `
  --tsdk-root D:/path/to/dji_thermal_sdk `
  --reference-rjpeg DATASET/raw/frame.JPG `
  --palette hot_iron `
  --palette rainbow2 `
  --save-off-lut-map
```

The output is always outside the source-render tree and contains:

- `temperature_index/*.npy`, the nearest canonical LUT indices;
- `apparent_temperature_c/*.npy`, fixed-scene-range apparent temperatures;
- `palettes/<name>/*.png`, lossless displays using exact TSDK LUT entries;
- an optional off-LUT distance map and a provenance manifest.

The scene `Tmin/Tmax` are fixed and shared by all methods.  A TSDK palette
change therefore does not alter the model, temperature range, or formal
metrics.  The palette output is exactly reversible to the projected scalar
index because every accepted TSDK LUT must contain 256 unique RGB entries.
Projection of an off-LUT model render onto the canonical LUT is deterministic
but is not described as lossless; the manifest records its mean, P95, maximum,
and exact-pixel fraction.
