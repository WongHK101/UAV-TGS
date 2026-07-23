#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL images currently export OMP_NUM_THREADS/MKL_NUM_THREADS=0, which is
# invalid for libgomp and makes CPU-side metric aggregation unnecessarily slow.
if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=16; fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export MKL_NUM_THREADS=16; fi

# Frozen Hold-8 v2 ablation ladder used to isolate three questions:
#   t_only_sfm_3dgs     : Can thermal imagery recover its own SfM + vanilla 3DGS?
#   rgb_sfm_t_3dgs      : What is gained by RGB cameras/sparse points alone?
#   naive_two_pass_3dgs : What is gained by transferring the dense RGB model when
#                         the second pass is an ordinary, fully-unfrozen 3DGS run?
#
# All outputs are isolated from the frozen AAAI27 experiment root.  The formal
# recipes use native resolution and the official 3DGS optimization defaults.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 Scene smoke|formal|evaluate t_only_sfm_3dgs|rgb_sfm_t_3dgs|naive_two_pass_3dgs" >&2
  exit 2
fi

SCENE="$1"
STAGE="$2"
METHOD="$3"
ROOT="${UAV_TGS_ROOT:-/root/autodl-tmp/UAV-TGS}"
CODE="$ROOT/code"
PY="$ROOT/environments/uav-tgs/bin/python"
FORMAL_EXPERIMENT=aaai27_hold8_v2_native
ABLATION_EXPERIMENT=aaai27_hold8_v2_native_vanilla_ablation
PROTOCOL_ID=uav-tgs-aaai27-hold8-v2-vanilla-ablation-v1
RESOLUTION=-1

case "$SCENE" in
  Building) SLUG=building; TOTAL=614; TRAIN=537; TEST=77; DECODE_PROTO=aaai_second_review_v1 ;;
  InternalRoad) SLUG=internalroad; TOTAL=559; TRAIN=489; TEST=70; DECODE_PROTO=aaai_second_review_v1 ;;
  PVpanel) SLUG=pvpanel; TOTAL=289; TRAIN=252; TEST=37; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  TransmissionTower) SLUG=transmissiontower; TOTAL=673; TRAIN=588; TEST=85; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  Urban20K) SLUG=urban20k; TOTAL=748; TRAIN=654; TEST=94; DECODE_PROTO=aaai27_a3_three_scene_v1 ;;
  Orchard) SLUG=orchard; TOTAL=588; TRAIN=514; TEST=74; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  *) echo "unsupported representative scene: $SCENE" >&2; exit 2 ;;
esac
case "$STAGE" in smoke|formal|evaluate) ;; *) echo "unsupported stage: $STAGE" >&2; exit 2 ;; esac
case "$METHOD" in
  t_only_sfm_3dgs|rgb_sfm_t_3dgs|naive_two_pass_3dgs) ;;
  *) echo "unsupported method: $METHOD" >&2; exit 2 ;;
esac

DERIVED="$ROOT/derived/aaai27_hold8_v2/$SCENE"
DECODE="$ROOT/derived/thermal_radiometry/$DECODE_PROTO/$SCENE"
TEMP_SOURCE="$DECODE/temperature_c/$SCENE"
THERMAL="$DERIVED/thermal_benchmark"
TEMP_FORMAL="$DERIVED/thermal_undistorted"
RUNTIME="$DERIVED/runtime_lists"
RANGE="$DERIVED/radiometry/range_manifest.json"
FORMAL_EXP="$ROOT/experiments/$FORMAL_EXPERIMENT/$SCENE"
RGB_MODEL="$FORMAL_EXP/rgb_anchor/Model_RGB"
RGB_CKPT="$RGB_MODEL/chkpnt30000.pth"

if [[ "$STAGE" == smoke ]]; then
  EXP="$ROOT/experiments/${ABLATION_EXPERIMENT}_smoke/$SCENE"
  ITERATIONS=1000
  DENSIFY_FROM=100
  DENSIFY_UNTIL=900
  DENSIFY_INTERVAL=100
  OPACITY_RESET=500
else
  EXP="$ROOT/experiments/$ABLATION_EXPERIMENT/$SCENE"
  ITERATIONS=30000
  DENSIFY_FROM=500
  DENSIFY_UNTIL=15000
  DENSIFY_INTERVAL=100
  OPACITY_RESET=3000
fi
METHOD_ROOT="$EXP/$METHOD"
MODEL="$METHOD_ROOT/model"
LOG_ROOT="$ROOT/logs/experiments/$(basename "$(dirname "$EXP")")/$SCENE/$METHOD"
STATUS="$METHOD_ROOT/STATUS"
COLMAP="$ROOT/tools/colmap-4.1.0-cuda12.8-ceres2.3dev-cudss0.8-sm120/bin/colmap"
COLMAP_SHA=fd0d5597820afd3212215f61d979313fa7671a34dbdc64ef3d8398d5589cfc63
COLMAP_RUNTIME_LIB="$ROOT/tools/runtime_dependencies/colmap_4.1_cuda/lib"
if [[ -d "$COLMAP_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$COLMAP_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

sha() { sha256sum "$1" | awk '{print $1}'; }
require_file() { [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 1; }; }
require_clean_code() {
  test -x "$PY"
  test -z "$(git -C "$CODE" status --porcelain=v1 --untracked-files=all)"
}

write_receipt() {
  local output="$1" kind="$2" status="$3"
  shift 3
  "$PY" - "$output" "$kind" "$status" "$SCENE" "$METHOD" "$PROTOCOL_ID" \
    "$(git -C "$CODE" rev-parse HEAD)" "$@" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
out,kind,status,scene,method,protocol,commit,*items=sys.argv[1:]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
inputs={}
for item in items:
    key,path=item.split('=',1); p=Path(path)
    inputs[key]={'path':str(p.resolve()),'sha256':sha(p),'size_bytes':p.stat().st_size}
value={'schema':'uav-tgs-vanilla-ablation-receipt-v1','kind':kind,'status':status,
       'scene':scene,'method':method,'protocol_id':protocol,'code_commit':commit,
       'created_at_utc':datetime.now(timezone.utc).isoformat(),'inputs':inputs}
payload=json.dumps(value,sort_keys=True,separators=(',',':')).encode()
value['payload_sha256']=hashlib.sha256(payload).hexdigest()
p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
PY
}

write_failure() {
  local rc="$1" line="$2"
  set +e
  mkdir -p "$METHOD_ROOT"
  printf 'failed\n' > "$STATUS"
  "$PY" - "$METHOD_ROOT/failure.json" "$rc" "$line" "$SCENE" "$METHOD" "$STAGE" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
out,rc,line,scene,method,stage=sys.argv[1:]
value={'schema':'uav-tgs-vanilla-ablation-failure-v1','status':'failed','scene':scene,
       'method':method,'stage':stage,'exit_code':int(rc),'shell_line':int(line),
       'created_at_utc':datetime.now(timezone.utc).isoformat()}
p=Path(out);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
PY
}
trap 'rc=$?; line=$LINENO; write_failure "$rc" "$line"; exit "$rc"' ERR

preflight() {
  require_clean_code
  require_file "$RANGE"
  require_file "$RUNTIME/thermal_train_list.txt"
  require_file "$RUNTIME/thermal_test_list.txt"
  require_file "$THERMAL/sparse/0/cameras.bin"
  require_file "$THERMAL/sparse/0/images.bin"
  require_file "$THERMAL/sparse/0/points3D.bin"
  test "$(wc -l < "$RUNTIME/thermal_train_list.txt")" -eq "$TRAIN"
  test "$(wc -l < "$RUNTIME/thermal_test_list.txt")" -eq "$TEST"
  test "$(find "$TEMP_SOURCE" -maxdepth 1 -type f -name '*.npy' | wc -l)" -eq "$TOTAL"
  if [[ "$METHOD" == naive_two_pass_3dgs ]]; then require_file "$RGB_CKPT"; fi
}

prepare_t_only_dataset() {
  local source="$METHOD_ROOT/t_only_source" ud="$METHOD_ROOT/t_only_temperature_ud"
  local dataset="$METHOD_ROOT/t_only_dataset"
  local cache="${UAV_TGS_T_ONLY_CACHE:-}"
  local distorted_model
  if [[ -f "$dataset/PREPARED" ]]; then
    test "$(tr -d '\r\n' < "$dataset/PREPARED")" = passed
    printf '%s\n' "$dataset"
    return
  fi
  if [[ -n "$cache" ]]; then
    test -f "$cache/t_only_dataset/PREPARED"
    test -f "$cache/t_only_dataset_receipt.json"
    test -f "$cache/t_only_temperature_ud/manifest.json"
    ln -s "$cache/t_only_dataset" "$dataset"
    ln -s "$cache/t_only_temperature_ud" "$ud"
    ln -s "$cache/t_only_dataset_receipt.json" "$METHOD_ROOT/t_only_dataset_receipt.json"
    write_receipt "$METHOD_ROOT/t_only_cache_binding.json" t_only_cache_binding passed \
      "cache_receipt=$cache/t_only_dataset_receipt.json"
    printf '%s\n' "$dataset"
    return
  fi
  if [[ ! -f "$source/distorted/conversion_completion_manifest.json" ]]; then
    test ! -e "$source" -a ! -e "$ud" -a ! -e "$dataset"
    mkdir -p "$source/input" "$dataset/images" "$dataset/sparse/0" "$LOG_ROOT"
    "$PY" "$CODE/tools/thermal_radiometry/render_canonical_palette.py" \
      --temperature-root "$TEMP_SOURCE" --output-root "$source/input" \
      --range-manifest "$RANGE" --manifest "$METHOD_ROOT/t_only_source_canonical.json"
    test "$(find "$source/input" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$TOTAL"
    "$PY" "$CODE/convert_uavfgs.py" -s "$source" \
      --colmap_executable "$COLMAP" --exiftool_executable exiftool \
      --required_colmap_version 4.1.0 --required_colmap_sha256 "$COLMAP_SHA" \
      --database_policy reset --camera SIMPLE_RADIAL --matching exhaustive \
      --colmap_gpu_index 0 --sfm_mapper incremental --mapper_multiple_models 1 \
      --min_model_size "$TOTAL" --init_min_num_inliers 100 --abs_pose_min_num_inliers 30 \
      --require_cuda_colmap --no_use_model_aligner \
      2>&1 | tee "$LOG_ROOT/t_only_colmap.log"
  else
    echo "Reusing completed T-only SfM conversion: $source"
    test "$(find "$source/input" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$TOTAL"
  fi
  require_file "$source/distorted/sparse/0/cameras.bin"
  require_file "$source/sparse/0/cameras.bin"
  distorted_model="$("$PY" - "$CODE" "$source/distorted/sparse" "$source/sparse/0" <<'PY'
import sys
from pathlib import Path

code, candidates_root, output_model = map(Path, sys.argv[1:])
sys.path.insert(0, str(code))
from scene.colmap_loader import read_extrinsics_binary

expected = {
    image.name
    for image in read_extrinsics_binary(str(output_model / "images.bin")).values()
}
matches = []
for candidate in sorted(candidates_root.iterdir(), key=lambda path: path.name):
    images_bin = candidate / "images.bin"
    if not images_bin.is_file():
        continue
    names = {
        image.name
        for image in read_extrinsics_binary(str(images_bin)).values()
    }
    if names == expected:
        matches.append(candidate)
if len(matches) != 1:
    raise SystemExit(
        f"Expected exactly one distorted COLMAP model matching the undistorted "
        f"image set, found {len(matches)}: {[str(path) for path in matches]}"
    )
print(matches[0])
PY
)"
  if [[ -e "$ud" && ! -f "$ud/manifest.json" ]]; then
    rm -rf -- "$ud"
  fi
  mkdir -p "$dataset/images" "$dataset/sparse/0" "$LOG_ROOT"
  "$PY" "$CODE/tools/thermal_radiometry/undistort_temperature.py" \
    --temperature-root "$TEMP_SOURCE" --input-model "$distorted_model" \
    --output-model "$source/sparse/0" --input-model-format bin --output-model-format bin \
    --output-root "$ud"
  "$PY" "$CODE/tools/thermal_radiometry/render_canonical_palette.py" \
    --temperature-root "$ud/temperature_c" --output-root "$dataset/images" \
    --range-manifest "$RANGE" --manifest "$METHOD_ROOT/t_only_undistorted_canonical.json"
  cp -a "$source/sparse/0/." "$dataset/sparse/0/"
  test "$(find "$dataset/images" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$TOTAL"
  write_receipt "$METHOD_ROOT/t_only_dataset_receipt.json" t_only_dataset passed \
    "range=$RANGE" "conversion=$source/distorted/conversion_completion_manifest.json" \
    "temperature_manifest=$ud/manifest.json" \
    "canonical_manifest=$METHOD_ROOT/t_only_undistorted_canonical.json"
  printf 'passed\n' > "$dataset/PREPARED"
  printf '%s\n' "$dataset"
}

render_and_measure_appearance() {
  local source="$1" train_list="$2" test_list="$3"
  local train_sha test_sha
  train_sha="$(sha "$train_list")"; test_sha="$(sha "$test_list")"
  cd "$CODE"
  "$PY" render.py -s "$source" --images images -m "$MODEL" -r 1 --eval \
    --train_list "$train_list" --test_list "$test_list" \
    --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" \
    --iteration "$ITERATIONS" --skip_train --save_by_image_name \
    --benchmark_efficiency --benchmark_warmup_views 10 --benchmark_repeats 3 \
    --benchmark_output "$METHOD_ROOT/render_efficiency.json" \
    2>&1 | tee "$LOG_ROOT/render.log"
  "$PY" metrics.py -m "$MODEL" 2>&1 | tee "$LOG_ROOT/metrics.log"
  require_file "$MODEL/results.json"
  require_file "$MODEL/per_view.json"
}

run_training() {
  preflight
  if [[ -f "$STATUS" ]]; then
    test "$(tr -d '\r\n' < "$STATUS")" = passed
    echo "$SCENE/$METHOD/$STAGE already passed"
    return
  fi
  test ! -e "$METHOD_ROOT"
  mkdir -p "$METHOD_ROOT" "$LOG_ROOT"
  local source train_list test_list train_sha test_sha
  if [[ "$METHOD" == t_only_sfm_3dgs ]]; then
    prepare_t_only_dataset
    source="$METHOD_ROOT/t_only_dataset"
  else
    source="$THERMAL"
  fi
  train_list="$RUNTIME/thermal_train_list.txt"
  test_list="$RUNTIME/thermal_test_list.txt"
  train_sha="$(sha "$train_list")"; test_sha="$(sha "$test_list")"

  local common=(
    -s "$source" --images images -r "$RESOLUTION" -m "$MODEL"
    --train_list "$train_list" --test_list "$test_list"
    --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha"
    --iterations "$ITERATIONS" --checkpoint_iterations "$ITERATIONS"
    --save_iterations "$ITERATIONS" --test_iterations "$ITERATIONS"
    --position_lr_init 0.00016 --position_lr_final 0.0000016
    --position_lr_delay_mult 0.01 --position_lr_max_steps 30000
    --feature_lr 0.0025 --opacity_lr 0.025 --scaling_lr 0.005 --rotation_lr 0.001
    --lambda_dssim 0.2 --densify_from_iter "$DENSIFY_FROM"
    --densify_until_iter "$DENSIFY_UNTIL" --densification_interval "$DENSIFY_INTERVAL"
    --densify_grad_threshold 0.0002 --opacity_reset_interval "$OPACITY_RESET"
    --percent_dense 0.01 --temperature_loss_mode none --eval --disable_viewer
    --artifact_save_semantics aligned --benchmark_efficiency
    --efficiency_output "$METHOD_ROOT/train_efficiency.json" --efficiency_stage thermal_vanilla
  )
  cd "$CODE"
  if [[ "$METHOD" == naive_two_pass_3dgs ]]; then
    "$PY" train.py "${common[@]}" --start_checkpoint "$RGB_CKPT" \
      --checkpoint_restart_mode vanilla_full --thermal_optimizer_state fresh \
      2>&1 | tee "$LOG_ROOT/train.log"
  else
    "$PY" train.py "${common[@]}" --baseline_modules_off \
      2>&1 | tee "$LOG_ROOT/train.log"
  fi
  render_and_measure_appearance "$source" "$train_list" "$test_list"
  require_file "$MODEL/chkpnt${ITERATIONS}.pth"
  require_file "$MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
  local receipt_inputs=(
    "checkpoint=$MODEL/chkpnt${ITERATIONS}.pth"
    "ply=$MODEL/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"
    "results=$MODEL/results.json" "per_view=$MODEL/per_view.json"
    "train_efficiency=$METHOD_ROOT/train_efficiency.json"
    "render_efficiency=$METHOD_ROOT/render_efficiency.json"
    "train_list=$train_list" "test_list=$test_list"
  )
  if [[ "$METHOD" == naive_two_pass_3dgs ]]; then
    receipt_inputs+=("restart_protocol=$MODEL/vanilla_checkpoint_restart_protocol.json" "rgb_anchor=$RGB_CKPT")
  elif [[ "$METHOD" == t_only_sfm_3dgs ]]; then
    receipt_inputs+=("t_only_dataset=$METHOD_ROOT/t_only_dataset_receipt.json")
  fi
  write_receipt "$METHOD_ROOT/endpoint.json" "$STAGE" passed "${receipt_inputs[@]}"
  printf 'passed\n' > "$STATUS"
  echo "$SCENE/$METHOD/$STAGE training passed"
}

case "$STAGE" in
  smoke|formal) run_training ;;
  evaluate)
    exec "$CODE/tools/aaai27_hold8/evaluate_vanilla_ablation_method.sh" "$SCENE" "$METHOD"
    ;;
esac
