# UAV-FGS Research Code

This repository contains the research code for the UAV-FGS RGB-T UAV 3D Gaussian Splatting pipeline. It is a self-contained adaptation of the original 3D Gaussian Splatting codebase, with the additional scripts required for:

- CFR-based raw RGB-T standardization from cross-FoV image pairs
- RGB-stage 3DGS reconstruction
- thermal-stage 3DGS transfer/training
- controllable RGB-T fusion and sweep evaluation
- GT-view and auxiliary evaluation used by the paper

The repository intentionally omits non-essential viewer/build artifacts, generated outputs, and local experiment records.

## Quick Start

This repository is derived from the original 3D Gaussian Splatting codebase. The setup steps and checks needed for reproduction are listed here.

### 1) System prerequisites

The CUDA extensions in `submodules/` require both:

- a system CUDA toolkit with `nvcc`
- a system C/C++ compiler

The Conda environment below installs the PyTorch CUDA runtime, but it does not replace the system compiler toolchain needed to build the extensions.

Windows prerequisites:

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` available on `PATH` (CUDA 11.8 is the intended match here)
- Visual Studio 2019 or 2022 Build Tools with the MSVC x64 C/C++ toolchain
- a shell where `cl` is available
- COLMAP 4.1.0 built with CUDA on `PATH` (or pass `--colmap <path>`); GlobalMapper runs additionally require a CUDA-enabled Ceres build with cuDSS
- ExifTool on `PATH` (or pass `--exiftool <path>`)

Recommended Windows shell:

- Start from `x64 Native Tools Command Prompt for VS 2019/2022`, or any shell where `cl` is already available before installing the CUDA extensions.

Windows preflight checks:

```powershell
where.exe nvcc
where.exe cl
where.exe colmap
where.exe exiftool
colmap -h
```

Linux prerequisites:

- NVIDIA GPU and driver
- CUDA toolkit with `nvcc` available on `PATH`
- GCC/G++ toolchain
- COLMAP 4.1.0 built with CUDA on `PATH`; GlobalMapper runs additionally require `ldd` to resolve `libcudss`
- ExifTool on `PATH`

Linux example (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y build-essential gcc g++ cmake ninja-build libimage-exiftool-perl
```

Linux preflight checks:

```bash
which nvcc
which g++
which colmap
which exiftool
colmap -h
ldd "$(readlink -f "$(which colmap)")" | grep libcudss
```

The Ubuntu distribution `colmap` package is not a valid substitute for this
CUDA/cuDSS runtime. The converter rejects a CPU-only or non-4.1.0 binary, and
formal runs pin the accepted executable by SHA-256.

### 2) Create environment

Recommended explicit setup:

```bash
conda create -n uav-fgs python=3.10.18 pip=25.2 numpy=1.26.4 -y
conda activate uav-fgs
conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia -c defaults
python -m pip install -r requirements.txt
```

Equivalent one-shot alternative:

```bash
conda env create -f environment.yml
conda activate uav-fgs
```

### 3) Install CUDA extensions (required)

Run these commands from the repository root, in the same shell that already sees `nvcc` and the system compiler:

```bash
python -m pip install --no-build-isolation submodules/simple-knn
python -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
python -m pip install --no-build-isolation submodules/fused-ssim
```

Standard non-editable installs are intentional here. They are sufficient for reproduction and are more robust than editable installs across different `pip` and `setuptools` versions.

### 4) Sanity checks

Do not rely only on the `conda` or `pip` exit text. Always run the checks below after installation, especially if package manager output included warnings such as cache, clobber, or safety messages.

Toolchain checks:

Windows:

```powershell
where.exe nvcc
where.exe cl
where.exe colmap
where.exe exiftool
```

Linux:

```bash
which nvcc
which g++
which colmap
which exiftool
```

Python/runtime checks:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import cv2, diff_gaussian_rasterization, simple_knn, fused_ssim; print('extensions OK')"
python train.py -h
python render.py -h
python metrics.py -h
python cfr.py -h
python run_uavfgs_pipeline.py -h
```

### 5) Smoke tests

Dataset layout:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

If `--rgb_dir` is omitted, the pipeline first looks for `<DATA_ROOT>/RGB` and then falls back to `<DATA_ROOT>/rgb`.

Lightweight CFR smoke test on a real scene:

```bash
python cfr.py --rgb_dir <DATA_ROOT>/rgb --th_dir <DATA_ROOT>/thermal --out_dir <TMP_CFR_OUT> --samples 3 --align fit --stage both --comparison
```

Lightweight pipeline smoke test:

1. Prepare a tiny paired subset at:

```text
<TINY_ROOT>/
  rgb/
  thermal/
```

2. Copy a few paired RGB/T images from one scene into the two folders above.

3. Run the first two pipeline steps:

```bash
python run_uavfgs_pipeline.py --data_root <TINY_ROOT> --out_root <TINY_OUT> --rgb_dir <TINY_ROOT>/rgb --to_step 2
```

### 6) Run full pipeline

Full dataset layout:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

Main command:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT>
```

Equivalent explicit lowercase-RGB form:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT> --rgb_dir <DATA_ROOT>/rgb
```

The pipeline is resumable by default. To rerun from scratch, use cleaning flags such as:

`--clean_fit --clean_input --clean_thermal_ud --clean_blend_out --force`

## Package Layout

The package keeps the same root-level structure expected by the original 3DGS codebase:

- `train.py`, `render.py`, `metrics.py`: inherited 3DGS training/rendering/metric entry points
- `run_uavfgs_pipeline.py`: one-command end-to-end pipeline used in this work
- `cfr.py`: crop/FoV/resolution standardization for raw RGB-T pairs
- `convert_uavfgs.py`: COLMAP + input conversion helper
- `eval_crop_metrics.py`: crop/alignment evaluation helper
- `metrics_plus.py`: extended GT-view metrics and auxiliary diagnostics
- `novel_view_metrics.py`: optional no-reference novel-view evaluation
- `blend_model_strict_endpoints.py`: RGB-T Gaussian fusion/export
- `eval_blend_sweep.py`: fusion-sweep evaluation
- `arguments/`, `gaussian_renderer/`, `scene/`, `utils/`, `lpipsPyTorch/`: required runtime code
- `submodules/`: vendored CUDA extensions required by the optimizer/renderer

## Environment

This repository is tested with:

- Python `3.10`
- PyTorch `2.0.1` + TorchVision `0.15.2`
- CUDA runtime `11.8` via `pytorch-cuda=11.8`
- Conda-based setup

The environment spec is:

- `environment.yml` for Conda + PyTorch/CUDA
- `requirements.txt` for the remaining Python packages

`requirements.txt` intentionally relies on the default PyPI index for portable installation.

The repository uses `opencv-python-headless` because the pipeline does not require OpenCV GUI windows. `opencv-python` is also acceptable if preferred locally.

Important note for extension builds:

- PyTorch CUDA runtime from Conda is not enough by itself
- the extension build additionally requires system `nvcc`
- the extension build additionally requires a visible system compiler (`cl` on Windows, `g++` on Linux)

## External Tools

The full raw-pair pipeline expects:

- CUDA-enabled COLMAP 4.1.0 with Ceres/cuDSS on `PATH`, or pass `--colmap <path>`
- ExifTool on `PATH`, or pass `--exiftool <path>`

`cfr.py` has fallback behavior when ExifTool is unavailable, but the full pipeline should still be configured with COLMAP and ExifTool available.

The default SfM command is COLMAP 4.1.0 `global_mapper`, with global positioning and bundle adjustment locked to GPU 0 and random seed 0. It does not fall back to incremental mapping. On Linux, the default preflight requires `ldd` to resolve `libcudss` with no missing libraries; formal runs should additionally pin the executable with `--required_colmap_sha256`. Internal Ceres failure or a CUDA/cuDSS-to-CPU solver fallback terminates the process group immediately. The default `model_aligner` then uses deterministic seed 0 and aligns registered camera centers from image-embedded WGS84-like values into a local ENU frame (`--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0`). The audit also requires the centered-RMS camera-cloud scale ratio to remain in `[0.5, 2.0]`, preventing a compact or collapsed model from passing only because every camera lies within the absolute 30 m threshold. These embedded values are used verbatim: the audit verifies relative local-ENU consistency only and must not be cited as proof of the true city location or absolute elevation. Alignment failure is fatal; the pipeline does not relax the threshold automatically.

`input/` is maintained as an exact content mirror. The formal database default is `--database_policy reset`; `reuse_verified` requires an exact input/runtime/argument/database provenance match. `adopt_legacy` requires `--expected_legacy_database_sha256`, skips feature extraction/matching instead of pretending to regenerate them, records their pinned but unverified origin, and remains ineligible for later `reuse_verified`. `model_aligner` receives a separate closed SQLite copy; its pre/post physical and semantic digests are recorded, the copy is discarded, and the publishable database keeps its pre-aligner identity. Immutable read-only database audits do not create WAL/SHM sidecars. Undistorted outputs are built in private staging directories, validated, and transactionally installed so a failed or interrupted rerun restores the previous complete outputs. Existing output paths without completion-manifest ownership are not replaced unless the caller explicitly supplies `--allow_replace_unverified_outputs`. Successful runs write:

```text
<DATA_ROOT>/distorted/model_alignment_transform.txt
<DATA_ROOT>/distorted/model_alignment_audit.json
<DATA_ROOT>/distorted/database_provenance.json
<DATA_ROOT>/distorted/conversion_completion_manifest.json
```

Step 4 resume accepts only a verified completion manifest whose converter argv, COLMAP SHA, input content inventory, alignment evidence, final image inventory, registered sparse-model views, and point count still match. For formal scene runs, set `--min_model_size` to the protocol-required image count (for example, `614` for Building); the smaller generic default is only a usability floor.

## Data Assumptions

The end-to-end pipeline expects a dataset root of the form:

```text
<DATA_ROOT>/
  rgb/
  thermal/
```

Benchmark data is distributed separately from this source repository.

## Main Entry Point

The primary entry point used in this project is:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT>
```

This orchestrates the pipeline in order:

1. CFR standardization from raw RGB-T pairs
2. crop/alignment evaluation
3. COLMAP conversion and reconstruction
4. stage-1 RGB 3DGS training/rendering/metrics
5. thermal undistortion and layout normalization
6. stage-2 thermal 3DGS training/rendering/metrics
7. RGB-T Gaussian blending
8. fusion-sweep evaluation

The script is resumable by default and writes per-step state files under:

```text
<DATA_ROOT>/_pipeline_state/
```

## Optional Efficiency Measurements

Efficiency probing is opt-in and disabled by default. For a full UAV-FGS run, enable it with:

```bash
python run_uavfgs_pipeline.py --data_root <DATA_ROOT> --out_root <OUT_ROOT> --benchmark_efficiency
```

The pipeline writes `<OUT_ROOT>/efficiency_benchmark.json` and stage sidecars next to the measured RGB/thermal models. A resumable step is considered efficiency-complete only when its expected sidecar is present; an older model without that sidecar is not silently reported as measured.

`efficiency_probe.py` can also wrap an arbitrary command from this or another method repository:

```bash
python efficiency_probe.py run \
  --output <METHOD_OUTPUT>/efficiency.json \
  --artifact final_model=<METHOD_OUTPUT>/point_cloud.ply \
  -- python <OTHER_METHOD>/train.py <METHOD_ARGS>
```

The shared JSON schema records command or stage wall time, explicit artifact sizes, and PLY vertex count when available. UAV-FGS training peak memory is measured at the training-call boundary with the PyTorch caching allocator; it does not include driver or non-PyTorch allocations. Rendering FPS is render-only: controlled warm-up views, ground-truth work, CPU transfer, and file I/O are excluded. The generic command wrapper samples only the directly launched PID through `nvidia-smi`; launcher children and very short peaks may be missed. These scopes should remain explicit when comparing different algorithms.

## Depth-Reference Geometry Evaluation

The reference-depth geometry consistency and front-intrusion tools are under
`tools/geometric_repeatability/`. They support explicit train/test camera lists,
training-side OpenMVS dense/mesh references, native-camera alignment, per-view
metrics, plots, visualizations, and result packaging. Reference geometry uses
`InterfaceCOLMAP -> DensifyPointCloud -> ReconstructMesh -> RefineMesh`; COLMAP
MVS/meshers are not fallback backends. Formal runs require the supplied
OpenMVS 2.4 RefineMesh fail-closed CUDA patch and bind cached bundles/metrics to
the clean Git commit plus exporter/evaluator SHA256 identities.

```bash
python tools/geometric_repeatability/sanity_tests.py
python tools/geometric_repeatability/openmvs_backend_sanity.py
python tools/geometric_repeatability/evaluate_depth_reference.py -h
python tools/geometric_repeatability/export_gaussian_probe_bundle.py -h
```

See `tools/geometric_repeatability/README.md` and
`tools/geometric_repeatability/DEPTH_REFERENCE_PROTOCOL.md` for the protocol and
full workflow. These metrics measure held-out geometry consistency against a
training-side reference surface; they are not absolute ground-truth 3D accuracy.

## Important Defaults

The repository follows the current code defaults, not older README snapshots.

- `run_metrics_plus=true`
- `run_novel_view_metrics=false`
- fusion `dc_y_from=lerp`
- blend `endpoint_mode=blend`

Please treat `run_uavfgs_pipeline.py -h` as the authoritative source of current CLI defaults.

## Minimal Direct Usage

If you want to run steps separately, the main scripts are:

```bash
python cfr.py -h
python convert_uavfgs.py -h
python train.py -h
python render.py -h
python metrics.py -h
python metrics_plus.py -h
python novel_view_metrics.py -h
python blend_model_strict_endpoints.py -h
python eval_blend_sweep.py -h
```

## Troubleshooting

- If `cl` is missing on Windows, reopen the session from a Visual Studio developer prompt and reinstall the CUDA extensions from that shell.
- If `nvcc` is missing, install a system CUDA toolkit and reopen the shell before building the extensions.
- If `conda env create` or `conda install` prints cache, safety, or clobber warnings, complete the installation, then run the sanity checks above. If the sanity checks fail, clean the Conda cache and recreate the environment.
- If extension installation fails, first confirm that `python -c "import torch"` works in the active environment, then confirm that `nvcc` and the system compiler are visible in the same shell.

## Notes on Scope

- This repository is prepared for academic research and reproducibility.
- It intentionally excludes non-core viewers, build caches, and local result tables.
- It preserves third-party license headers and upstream attribution where required by inherited components.
