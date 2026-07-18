# OCT-GS sidecar

`oct_gs` implements the first, deliberately narrow OCT-GS branch without
changing the legacy `train.py` path. It does not require an OCT-specific CUDA
kernel or the optional depth/alpha diagnostics ABI.

## Claim boundary

OCT-GS is **measurement-conditioned apparent-radiance rendering**.  Each
Gaussian owns a direct apparent-temperature scalar while its `xyz`, scale,
rotation, opacity, ordering, and topology are taken exactly from a frozen RGB
anchor.  The TSDK float Celsius target has already been decoded under the fixed
benchmark assumptions, so the renderer does not apply emissivity, reflected
temperature, atmosphere, or distance correction a second time.  It does not
claim absolute thermometry.

The fixed 7.5--13.5 um Planck-band integral is a monotonic compositing proxy, not
an asserted H30T spectral-response calibration.  Canonical Hot-Iron is only a
forward display mapping; no differentiable palette inversion is used.

## Two representations

- `oct_scalar`: one view-independent apparent-temperature parameter per
  Gaussian.
- `oct_residual`: the same base plus one bounded scalar coefficient multiplying
  `dot(frozen weak covariance axis, point-to-camera direction)`.  This first-order
  one-dimensional basis has zero mean over the sphere.  Its amplitude contracts
  automatically near the scene temperature endpoints, so the actual Celsius
  residual remains odd, bounded, and in range; it cannot become a free RGB SH
  representation.

OCT-GS v1 deliberately disables per-Gaussian uncertainty and NLL.  Reports mark
uncertainty calibration as N/A rather than silently introducing a third branch.

## Runtime integration

1. Load the formal RGB anchor as usual and capture
   `capture_occupancy_snapshot(anchor)` before OCT initialization.
2. Create `OCTGaussianField(OCTConfig(...))` with exactly the anchor Gaussian
   count and `build_oct_optimizer(...)`.  The optimizer verifier rejects any
   anchor parameter.
3. Reuse one `OCTRendererContext`. It supplies the normalized formal background
   radiance directly to the legacy rasterizer, so shared-occupancy composition is
   exact in one differentiable pass without an alpha side channel.
4. Train with `oct_rendering_loss(...)` against the float Celsius map and its
   forward canonical Hot-Iron observation.
5. On Building train views only, call
   `BuildingGradientCalibrator.observe(...)`, freeze the weights once, and reuse
   that manifest unchanged for InternalRoad.
6. Save with `save_oct_checkpoint(...)`.  The checkpoint is a thermometric
   sidecar and contains no RGB anchor tensors.  Saving fails if the raw
   xyz/scaling/rotation/opacity fingerprint changed.

`OCTStageCostTracker` records boundary wall time, peak PyTorch VRAM, steps,
rendered views, and raster-pass cost.  `write_oct_protocol_manifest` pins the
formal train/test/guard split, train-only range, float32 TSDK decode and parameter
receipts, canonical LUT, support, 30k camera sequence, Building calibration,
optimizer recipe, anchor hashes, method semantics, and claim boundaries.

`tools/oct_gs_formal.py` is the standalone formal entry point.  Its sequence is:

```text
prepare-sequence
build-binding                       # validates every immutable receipt/hash
calibrate-building                 # once; both variants, Building train only
freeze-hotspot-threshold           # separately for each scene, train only
cuda-smoke                         # before a formal run on the target GPU
train --variant oct_scalar         # fixed endpoints 10k/20k/30k
train --variant oct_residual
eval                               # test is forced to r1/full resolution
```

Every subcommand requires the same explicit anchor/split/decode/range/LUT/support
arguments.  This repetition is intentional: a changed file hash fails before an
experiment can resume or be evaluated.  Training reads temperature only from
the bound float32 `.npy`; it never inverts the canonical PNG.  Support-aware
resize is deterministic at camera resolution, while formal test rejects any
resolution other than native r1.

## Non-training CLI

```text
python tools/oct_gs_cli.py self-check
python tools/oct_gs_cli.py calibrate --gradient-records building_train_gradients.json --source-receipt building_calibration_source_receipt.json --output oct_loss_calibration.json
python tools/oct_gs_cli.py inspect-checkpoint --checkpoint oct_step30000.pt
```

The small CLI intentionally does not launch experiments.  Run
`tools/oct_gs_formal.py cuda-smoke ...` on the target GPU to verify
`override_color` gradients, exact one-pass native-background composition,
anchor invariants, and direct native-resolution float32 target loading.
