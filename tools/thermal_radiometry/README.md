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

## Leakage-guarded split and train-only range

Build a deterministic split from the audit/protocol manifest:

```powershell
python tools/thermal_radiometry/build_split.py `
  --manifest DERIVED/manifests/audit_or_decode.jsonl `
  --output DERIVED/splits/split_manifest.json `
  --scene SCENE
```

The preferred route uses timestamp and gimbal strata.  If those fields are not
reliable for every frame, the complete scene uses natural filename order.  A
stable scene/strip hash selects complete 16-frame test blocks every eight
blocks; two adjacent frames on each side are `guard` and never enter training.

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
  --report OUTPUT/temperature_metrics.json
```

The reported quantity is **TSDK-referenced apparent-temperature consistency**
(MAE, RMSE, signed bias, P95 error, clipping, and off-LUT distance).  It is not
absolute thermometry, true surface temperature, or physical ground truth.
