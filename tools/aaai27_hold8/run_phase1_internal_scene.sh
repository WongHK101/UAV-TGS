#!/usr/bin/env bash
set -Eeuo pipefail

# Hold-8 v2 Phase-1 runner.  Each invocation is isolated to one scene and one
# stage so two hosts can schedule scenes independently without changing recipes.

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 Scene prepare|rgb|thermal|method|all [raw_f3|scsp_refit_f3|adaptive_opacity_scale_clamp]" >&2
  exit 2
fi

SCENE="$1"
STAGE="$2"
METHOD="${3:-}"
ROOT="${UAV_TGS_ROOT:-/root/autodl-tmp/UAV-TGS}"
CODE="$ROOT/code"
PY="$ROOT/environments/uav-tgs/bin/python"
PROTOCOL_ID=uav-tgs-aaai27-hold8-v2

case "$SCENE" in
  Building) SLUG=building; TOTAL=614; TRAIN=537; TEST=77; DECODE_PROTO=aaai_second_review_v1 ;;
  InternalRoad) SLUG=internalroad; TOTAL=559; TRAIN=489; TEST=70; DECODE_PROTO=aaai_second_review_v1 ;;
  PVpanel) SLUG=pvpanel; TOTAL=289; TRAIN=252; TEST=37; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  TransmissionTower) SLUG=transmissiontower; TOTAL=673; TRAIN=588; TEST=85; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  Urban20K) SLUG=urban20k; TOTAL=748; TRAIN=654; TEST=94; DECODE_PROTO=aaai27_a3_three_scene_v1 ;;
  Orchard) SLUG=orchard; TOTAL=588; TRAIN=514; TEST=74; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac

BIND="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/$SCENE/binding"
DECODE="$ROOT/derived/thermal_radiometry/$DECODE_PROTO/$SCENE"
TEMP_SOURCE="$DECODE/temperature_c/$SCENE"
SOURCE_CANON="$ROOT/derived/canonical_${SLUG}_stage2_formal_v1"
if [[ -d "$ROOT/benchmarks/aaai27_cfr_fit_v1_20260719/$SCENE/rgb" ]]; then
  CFR="$ROOT/benchmarks/aaai27_cfr_fit_v1_20260719/$SCENE"
else
  CFR="$ROOT/datasets/cfr_fit_v1/$SCENE"
fi

DERIVED="$ROOT/derived/aaai27_hold8_v2/$SCENE"
WORK="$DERIVED/workspace"
RANGE="$DERIVED/radiometry/range_manifest.json"
TEMP_UD="$DERIVED/thermal_undistorted"
THERMAL="$DERIVED/thermal_benchmark"
FORMAL_SUPPORT="$DERIVED/formal_support"
HOTSPOT="$DERIVED/radiometry/hotspot_threshold_train_q95.json"
EXP="$ROOT/experiments/aaai27_hold8_v2/$SCENE"
RGB_ROOT="$EXP/rgb_anchor"
RGB_MODEL="$RGB_ROOT/Model_RGB"
RGB_CKPT="$RGB_MODEL/chkpnt30000.pth"
RGB_PLY="$RGB_MODEL/point_cloud/iteration_30000/point_cloud.ply"
LOG_ROOT="$ROOT/logs/experiments/aaai27_hold8_v2/$SCENE"
HEAD="$(git -C "$CODE" rev-parse HEAD)"

sha() { sha256sum "$1" | awk '{print $1}'; }
require_clean_code() {
  test -x "$PY"
  test -z "$(git -C "$CODE" status --porcelain=v1 --untracked-files=all)"
}
write_json_receipt() {
  local output="$1" kind="$2" status="$3"
  shift 3
  "$PY" - "$output" "$kind" "$status" "$SCENE" "$HEAD" "$@" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
out,kind,status,scene,commit,*items=sys.argv[1:]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
inputs={}
for item in items:
 key,path=item.split('=',1); p=Path(path)
 inputs[key]={"path":str(p.resolve()),"sha256":sha(p),"size_bytes":p.stat().st_size}
value={"schema":"uav-tgs-hold8-phase1-flat-receipt-v1","kind":kind,"status":status,
       "scene":scene,"code_commit":commit,"created_at_utc":datetime.now(timezone.utc).isoformat(),
       "inputs":inputs}
raw=json.dumps(value,sort_keys=True,separators=(',',':')).encode();value['payload_sha256']=hashlib.sha256(raw).hexdigest()
p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.tmp-{os.getpid()}')
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8');os.replace(tmp,p)
PY
}

verify_binding() {
  require_clean_code
  test -f "$BIND/binding_manifest.json" -a -f "$BIND/bound_split.json"
  test "$(wc -l < "$BIND/train_list.txt")" -eq "$TRAIN"
  test "$(wc -l < "$BIND/test_list.txt")" -eq "$TEST"
  test "$(wc -l < "$BIND/thermal_train_list.txt")" -eq "$TRAIN"
  test "$(wc -l < "$BIND/thermal_test_list.txt")" -eq "$TEST"
  test ! -e "$BIND/guard_list.txt" -a ! -e "$BIND/thermal_guard_list.txt"
  test "$(find "$CFR/rgb" -maxdepth 1 -type f | wc -l)" -eq "$TOTAL"
  test "$(find "$TEMP_SOURCE" -maxdepth 1 -type f -name '*.npy' | wc -l)" -eq "$TOTAL"
}

prepare_scene() {
  verify_binding
  if [[ -f "$DERIVED/PREPARE_STATUS" ]]; then
    test "$(tr -d '\r\n' < "$DERIVED/PREPARE_STATUS")" = passed
    return
  fi
  test ! -e "$WORK" -a ! -e "$RANGE"
  mkdir -p "$DERIVED/radiometry" "$WORK/images" "$WORK/sparse/0" "$WORK/distorted/sparse_aligned" "$EXP/protocol" "$LOG_ROOT"
  "$PY" "$CODE/tools/thermal_radiometry/estimate_scene_range.py" \
    --split-manifest "$BIND/bound_split.json" --npy-root "$DECODE/temperature_c" --output "$RANGE"
  cp -a "$SOURCE_CANON/workspace/sparse/0/." "$WORK/sparse/0/"
  cp -a "$SOURCE_CANON/workspace/distorted/sparse_aligned/." "$WORK/distorted/sparse_aligned/"
  for image in "$CFR"/rgb/*; do ln -s "$image" "$WORK/images/$(basename "$image")"; done
  test "$(find "$WORK/images" -maxdepth 1 -type l | wc -l)" -eq "$TOTAL"
  write_json_receipt "$EXP/protocol/prepare.json" prepare passed \
    "binding=$BIND/binding_manifest.json" "split=$BIND/bound_split.json" "range=$RANGE" \
    "cameras=$WORK/sparse/0/cameras.bin" "images=$WORK/sparse/0/images.bin"
  printf 'passed\n' > "$DERIVED/PREPARE_STATUS"
}

run_rgb() {
  prepare_scene
  if [[ -f "$RGB_ROOT/STATUS" ]]; then test "$(tr -d '\r\n' < "$RGB_ROOT/STATUS")" = passed; return; fi
  test ! -e "$RGB_ROOT"
  mkdir -p "$RGB_ROOT" "$LOG_ROOT"
  cd "$CODE"
  "$PY" run_uavfgs_pipeline.py \
    --data_root "$WORK" --out_root "$RGB_ROOT" --rgb_dir "$CFR/rgb" --th_dir "$CFR/thermal" \
    --train_list "$BIND/train_list.txt" --test_list "$BIND/test_list.txt" \
    --from_step 5 --to_step 7 --align fit --no_comparison --rgb_iter 30000 --rgb_res 4 \
    --rgb_densify_from 1500 --rgb_densify_until 10000 --rgb_densify_interval 300 \
    --rgb_densify_grad 0.001 --rgb_lambda_dssim 0.3 \
    --ss_enable_rgb --ss_source colmap_sparse --ss_use_aabb false --ss_voxel_size 1.5 \
    --ss_nn_dist_thr 3.5 --ss_adaptive_nn --ss_adaptive_alpha 1.2 --ss_adaptive_beta 0.2 \
    --ss_adaptive_max_scale 2.0 --ss_trim_tail_pct 0 --ss_drop_small_islands 10 \
    --ss_island_radius 10 --ss_prune_after_rgb --benchmark_efficiency \
    --efficiency_render_warmup_views 10 --efficiency_render_repeats 3 --profile_pipeline \
    --profile_save_logs --metrics_plus_extra_iqa "" --save_cmds \
    2>&1 | tee "$LOG_ROOT/rgb_anchor.log"
  test -f "$RGB_CKPT" -a -f "$RGB_PLY" -a -f "$RGB_MODEL/results.json" -a -f "$RGB_MODEL/per_view.json"
  write_json_receipt "$RGB_ROOT/completion.json" rgb_anchor passed \
    "checkpoint=$RGB_CKPT" "ply=$RGB_PLY" "train_list=$BIND/train_list.txt" "test_list=$BIND/test_list.txt" \
    "results=$RGB_MODEL/results.json" "per_view=$RGB_MODEL/per_view.json"
  printf 'passed\n' > "$RGB_ROOT/STATUS"
}

prepare_thermal() {
  run_rgb
  if [[ -f "$DERIVED/THERMAL_STATUS" ]]; then test "$(tr -d '\r\n' < "$DERIVED/THERMAL_STATUS")" = passed; return; fi
  for path in "$TEMP_UD" "$THERMAL" "$FORMAL_SUPPORT" "$HOTSPOT"; do test ! -e "$path"; done
  mkdir -p "$DERIVED/radiometry" "$THERMAL/sparse/0" "$EXP/protocol" "$LOG_ROOT"
  local train_sha test_sha anchor_native sparse_bin
  train_sha="$(sha "$BIND/thermal_train_list.txt")"; test_sha="$(sha "$BIND/thermal_test_list.txt")"
  anchor_native="$DERIVED/support_anchor_native"; sparse_bin="$DERIVED/thermal_sparse_source"
  cd "$CODE"
  "$PY" tools/thermal_radiometry/undistort_temperature.py \
    --temperature-root "$TEMP_SOURCE" --input-model "$WORK/distorted/sparse_aligned" \
    --output-model "$WORK/sparse/0" --input-model-format bin --output-model-format bin --output-root "$TEMP_UD"
  "$PY" tools/thermal_radiometry/render_canonical_palette.py \
    --temperature-root "$TEMP_UD/temperature_c" --output-root "$THERMAL/images" \
    --range-manifest "$RANGE" --manifest "$DERIVED/radiometry/canonical_manifest.json"
  "$PY" tools/thermal_radiometry/validate_roundtrip.py \
    --temperature-root "$TEMP_UD/temperature_c" --canonical-root "$THERMAL/images" \
    --range-manifest "$RANGE" --report "$DERIVED/radiometry/roundtrip.json"
  mkdir -p "$sparse_bin"; cp "$WORK/sparse/0/"{cameras.bin,images.bin,points3D.bin} "$sparse_bin/"
  "$PY" -m tools.thermal_radiometry.rename_colmap_image_extension \
    --source-model "$sparse_bin" --output-model "$THERMAL/sparse/0" --target-extension .png \
    --output-format .bin --manifest "$DERIVED/radiometry/thermal_sparse_png.json"
  "$PY" tools/thermal_radiometry/freeze_train_hotspot_threshold.py \
    --scene "$SCENE" --bound-split "$BIND/bound_split.json" \
    --decode-manifest "$DECODE/manifests/decode_manifest.jsonl" \
    --decode-protocol "$DECODE/manifests/decode_protocol_used_v1.jsonl" \
    --range-manifest "$RANGE" --temperature-root "$TEMP_UD" \
    --optimization-support-manifest "$TEMP_UD/manifest.json" --optimization-support-root "$TEMP_UD" \
    --output "$HOTSPOT"
  mkdir -p "$anchor_native/point_cloud"; cp -al "$RGB_MODEL/point_cloud/iteration_30000" "$anchor_native/point_cloud/"
  cp -a "$RGB_MODEL/cfg_args" "$anchor_native/cfg_args"; cp -a "$RGB_MODEL/cameras.json" "$anchor_native/cameras.json"
  "$PY" render.py -s "$THERMAL" --images images -m "$anchor_native" -r 1 --eval \
    --train_list "$BIND/thermal_train_list.txt" --test_list "$BIND/thermal_test_list.txt" \
    --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" --iteration 30000 \
    --skip_train --save_by_image_name --save_opacity_proxy 2>&1 | tee "$LOG_ROOT/support_render.log"
  "$PY" tools/thermal_radiometry/combine_formal_support.py \
    --split-manifest "$BIND/bound_split.json" --valid-support-root "$TEMP_UD/valid_support" \
    --valid-support-manifest "$TEMP_UD/manifest.json" --opacity-proxy-root "$anchor_native/test/ours_30000/opacity_proxy" \
    --opacity-proxy-manifest "$anchor_native/test/ours_30000/render_mapping_manifest.json" \
    --output-root "$FORMAL_SUPPORT" --opacity-threshold 0.01 --expected-test-count "$TEST"
  test "$(find "$THERMAL/images" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$TOTAL"
  test "$(find "$FORMAL_SUPPORT/bool" -maxdepth 1 -type f -name '*.npy' | wc -l)" -eq "$TEST"
  write_json_receipt "$EXP/protocol/thermal_benchmark.json" thermal_benchmark passed \
    "range=$RANGE" "canonical=$DERIVED/radiometry/canonical_manifest.json" \
    "hotspot=$HOTSPOT" "support=$FORMAL_SUPPORT/manifest.json"
  printf 'passed\n' > "$DERIVED/THERMAL_STATUS"
}

render_thermal() {
  local model="$1" iteration="$2" out="$3"
  local train_sha test_sha
  train_sha="$(sha "$BIND/thermal_train_list.txt")"; test_sha="$(sha "$BIND/thermal_test_list.txt")"
  cd "$CODE"
  "$PY" render.py -s "$THERMAL" --images images -m "$model" -r 4 --eval \
    --train_list "$BIND/thermal_train_list.txt" --test_list "$BIND/thermal_test_list.txt" \
    --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" --iteration "$iteration" \
    --skip_train --save_by_image_name --benchmark_efficiency --benchmark_warmup_views 10 \
    --benchmark_repeats 3 --benchmark_output "$out/render_efficiency.json"
  "$PY" metrics.py -m "$model"; "$PY" metrics_plus.py -m "$model" --extra_iqa "" --save_json
}

run_raw_or_adaptive() {
  local method="$1" method_root model iteration train_eff recipe
  method_root="$EXP/methods/$method"; model="$method_root/model"; iteration=60000
  if [[ -f "$method_root/STATUS" ]]; then test "$(tr -d '\r\n' < "$method_root/STATUS")" = passed; return; fi
  test ! -e "$method_root"; mkdir -p "$method_root/logs" "$method_root/efficiency" "$method_root/protocol"
  local train_sha test_sha; train_sha="$(sha "$BIND/thermal_train_list.txt")"; test_sha="$(sha "$BIND/thermal_test_list.txt")"
  local extra=()
  if [[ "$method" == raw_f3 ]]; then
    recipe=raw_f3_strict_sh3_v1
    extra=(--opacity_lr 0 --thermal_recipe aaai_strict --thermal_max_sh_degree 3 --thermal_optimizer_state fresh --thermal_freeze_mode strict --thermal_scale_clamp off --temperature_loss_mode none)
  else
    recipe=adaptive_opacity_scale_clamp_v1
    extra=(--opacity_lr 0.0002 --clamp_scale_max 10.0 --thermal_recipe legacy --thermal_optimizer_state restore --thermal_freeze_mode legacy --thermal_scale_clamp legacy --temperature_loss_mode none)
  fi
  cd "$CODE"
  "$PY" train.py -s "$THERMAL" --images images -r 4 -m "$model" \
    --train_list "$BIND/thermal_train_list.txt" --test_list "$BIND/thermal_test_list.txt" \
    --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" --start_checkpoint "$RGB_CKPT" \
    --iterations "$iteration" --checkpoint_iterations "$iteration" --save_iterations "$iteration" --test_iterations "$iteration" \
    --position_lr_init 0 --position_lr_final 0 --scaling_lr 0 --rotation_lr 0 --feature_lr 0.001 \
    --densify_from_iter 999999 --densify_until_iter 0 --densification_interval 999999 --opacity_reset_interval 999999 \
    --lambda_dssim 0.05 --t_struct_grad_w 0.006 --t_struct_grad_norm true --thermal_reset_features \
    --eval --disable_viewer --artifact_save_semantics aligned --benchmark_efficiency \
    --efficiency_output "$method_root/efficiency/train.json" --efficiency_stage thermal "${extra[@]}" \
    2>&1 | tee "$method_root/logs/train.log"
  render_thermal "$model" "$iteration" "$method_root/efficiency" 2>&1 | tee "$method_root/logs/render_eval.log"
  local audit_args=()
  if [[ "$method" == raw_f3 ]]; then audit_args+=(--strict-geometry); fi
  "$PY" tools/stage2_endpoint_audit.py --rgb-ply "$RGB_PLY" --model-root "$model" --group "$method" \
    --iterations "$iteration" --thermal-max-sh-degree 3 "${audit_args[@]}" \
    --report "$method_root/endpoint_audit.json"
  test -f "$model/chkpnt${iteration}.pth" -a -f "$model/point_cloud/iteration_${iteration}/point_cloud.ply" -a -f "$model/results.json"
  write_json_receipt "$method_root/endpoint.json" "$recipe" passed \
    "checkpoint=$model/chkpnt${iteration}.pth" "ply=$model/point_cloud/iteration_${iteration}/point_cloud.ply" \
    "results=$model/results.json" "per_view=$model/per_view.json" "audit=$method_root/endpoint_audit.json" \
    "train_efficiency=$method_root/efficiency/train.json" "render_efficiency=$method_root/efficiency/render_efficiency.json"
  printf 'passed\n' > "$method_root/STATUS"
}

run_scsp() {
  local method_root="$EXP/methods/scsp_refit_f3" anchor="$EXP/methods/scsp_refit_f3/scsp_anchor"
  if [[ -f "$method_root/STATUS" ]]; then test "$(tr -d '\r\n' < "$method_root/STATUS")" = passed; return; fi
  test ! -e "$method_root"; mkdir -p "$method_root/logs" "$method_root/efficiency" "$method_root/protocol"
  cd "$CODE"
  "$PY" tools/build_adaptive_scale_anchor.py --scene-name "$SCENE" --method scsp \
    --input-model-dir "$RGB_MODEL" --output-model-dir "$anchor" --anchor-iteration 30000 \
    --sparse-root "$WORK/sparse/0" --support-voxel-size 1.5 --support-max-voxel-radius 2 \
    --expected-checkpoint-sha256 "$(sha "$RGB_CKPT")" --expected-ply-sha256 "$(sha "$RGB_PLY")" \
    --code-commit "$HEAD" > "$method_root/logs/scsp_projection.jsonl"
  local modified; modified="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["counts"]["modified_gaussians"])' "$anchor/adaptive_scale_manifest.json")"
  if [[ "$modified" -eq 0 ]]; then
    test -f "$EXP/methods/raw_f3/endpoint.json"
    write_json_receipt "$method_root/alias_to_raw_f3.json" scsp_noop_alias passed \
      "scsp_manifest=$anchor/adaptive_scale_manifest.json" "raw_endpoint=$EXP/methods/raw_f3/endpoint.json"
    printf 'passed\n' > "$method_root/STATUS"; return
  fi
  local refit="$method_root/rgb_refit" sequence="$method_root/protocol/camera_sequence.json"
  local train_sha test_sha anchor_ckpt anchor_ply
  train_sha="$(sha "$BIND/train_list.txt")"; test_sha="$(sha "$BIND/test_list.txt")"
  anchor_ckpt="$anchor/chkpnt30000.pth"; anchor_ply="$anchor/point_cloud/iteration_30000/point_cloud.ply"
  "$PY" tools/build_fixed_camera_sequence.py --camera-names "$BIND/train_list.txt" --output "$sequence" \
    --seed 0 --steps 5000 --scene "$SCENE" --split-sha256 "$train_sha" --anchor-sha256 "$(sha "$anchor_ckpt")"
  "$PY" train.py -s "$WORK" --images images -r 4 --eval -m "$refit" \
    --train_list "$BIND/train_list.txt" --test_list "$BIND/test_list.txt" --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" \
    --start_checkpoint "$anchor_ckpt" --iterations 35000 --save_iterations 35000 --checkpoint_iterations 35000 --test_iterations 35000 \
    --position_lr_max_steps 30000 --rgb_continuation_anchor_iteration 30000 --rgb_continuation_scheduler_horizon 30000 \
    --rgb_continuation_updates 5000 --rgb_continuation_recipe appearance_only --rgb_optimizer_state fresh \
    --fixed_camera_sequence "$sequence" --artifact_save_semantics aligned --optimizer_step_at_final_iteration --disable_viewer \
    --benchmark_efficiency --efficiency_output "$method_root/efficiency/rgb_refit.json" --efficiency_stage rgb_sh_only_refit \
    2>&1 | tee "$method_root/logs/rgb_refit.log"
  local model="$method_root/model" refit_ckpt="$refit/chkpnt35000.pth" refit_ply="$refit/point_cloud/iteration_35000/point_cloud.ply"
  local ttrain ttest; ttrain="$(sha "$BIND/thermal_train_list.txt")"; ttest="$(sha "$BIND/thermal_test_list.txt")"
  "$PY" train.py -s "$THERMAL" --images images -r 4 -m "$model" \
    --train_list "$BIND/thermal_train_list.txt" --test_list "$BIND/thermal_test_list.txt" --train_list_sha256 "$ttrain" --test_list_sha256 "$ttest" \
    --start_checkpoint "$refit_ckpt" --iterations 65000 --checkpoint_iterations 65000 --save_iterations 65000 --test_iterations 65000 \
    --position_lr_init 0 --position_lr_final 0 --scaling_lr 0 --rotation_lr 0 --opacity_lr 0 --feature_lr 0.001 \
    --densify_from_iter 999999 --densify_until_iter 0 --densification_interval 999999 --opacity_reset_interval 999999 \
    --lambda_dssim 0.05 --t_struct_grad_w 0.006 --t_struct_grad_norm true --thermal_reset_features --eval --disable_viewer \
    --thermal_recipe aaai_strict --thermal_max_sh_degree 3 --thermal_optimizer_state fresh --thermal_freeze_mode strict \
    --thermal_scale_clamp off --artifact_save_semantics aligned --temperature_loss_mode none --benchmark_efficiency \
    --efficiency_output "$method_root/efficiency/f3_train.json" --efficiency_stage thermal 2>&1 | tee "$method_root/logs/f3_train.log"
  render_thermal "$model" 65000 "$method_root/efficiency" 2>&1 | tee "$method_root/logs/render_eval.log"
  "$PY" tools/stage2_endpoint_audit.py --rgb-ply "$refit_ply" --model-root "$model" --group scsp_refit_f3 \
    --iterations 65000 --thermal-max-sh-degree 3 --strict-geometry --report "$method_root/endpoint_audit.json"
  write_json_receipt "$method_root/endpoint.json" scsp_refit_f3 passed \
    "checkpoint=$model/chkpnt65000.pth" "ply=$model/point_cloud/iteration_65000/point_cloud.ply" \
    "scsp_manifest=$anchor/adaptive_scale_manifest.json" "refit_audit=$refit/rgb_appearance_refit_freeze_audit.json" \
    "results=$model/results.json" "per_view=$model/per_view.json" "audit=$method_root/endpoint_audit.json" \
    "refit_efficiency=$method_root/efficiency/rgb_refit.json" "f3_efficiency=$method_root/efficiency/f3_train.json"
  printf 'passed\n' > "$method_root/STATUS"
}

run_method() {
  prepare_thermal
  case "$METHOD" in
    raw_f3) run_raw_or_adaptive raw_f3 ;;
    adaptive_opacity_scale_clamp) run_raw_or_adaptive adaptive_opacity_scale_clamp ;;
    scsp_refit_f3) run_scsp ;;
    *) echo "unsupported/missing method: $METHOD" >&2; exit 2 ;;
  esac
}

case "$STAGE" in
  prepare) prepare_scene ;;
  rgb) run_rgb ;;
  thermal) prepare_thermal ;;
  method) run_method ;;
  all)
    prepare_thermal
    METHOD=raw_f3; run_method
    METHOD=scsp_refit_f3; run_method
    METHOD=adaptive_opacity_scale_clamp; run_method
    ;;
  *) echo "unsupported stage: $STAGE" >&2; exit 2 ;;
esac
