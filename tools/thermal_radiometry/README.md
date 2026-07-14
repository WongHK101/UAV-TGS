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
