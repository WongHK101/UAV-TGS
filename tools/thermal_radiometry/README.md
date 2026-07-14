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
