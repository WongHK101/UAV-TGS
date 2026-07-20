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
  Garden) SLUG=garden; TOTAL=656; TRAIN=574; TEST=82; DECODE_PROTO=aaai27_phase2_formal_v1 ;;
  Plaza) SLUG=plaza; TOTAL=668; TRAIN=584; TEST=84; DECODE_PROTO=aaai27_phase2_formal_v1 ;;
  Road) SLUG=road; TOTAL=467; TRAIN=408; TEST=59; DECODE_PROTO=aaai27_phase2_formal_v1 ;;
  Urban50K) SLUG=urban50k; TOTAL=1299; TRAIN=1136; TEST=163; DECODE_PROTO=aaai27_phase2_formal_v1 ;;
  Urban100K) SLUG=urban100k; TOTAL=1671; TRAIN=1462; TEST=209; DECODE_PROTO=aaai27_phase2_formal_v1 ;;
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
RUNTIME_LISTS="$DERIVED/runtime_lists"
RGB_TRAIN_LIST="$RUNTIME_LISTS/train_list.txt"
RGB_TEST_LIST="$RUNTIME_LISTS/test_list.txt"
THERMAL_TRAIN_LIST="$RUNTIME_LISTS/thermal_train_list.txt"
THERMAL_TEST_LIST="$RUNTIME_LISTS/thermal_test_list.txt"
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
COLMAP="$ROOT/tools/colmap-4.1.0-cuda12.8-ceres2.3dev-cudss0.8-sm120/bin/colmap"
COLMAP_SHA=fd0d5597820afd3212215f61d979313fa7671a34dbdc64ef3d8398d5589cfc63
COLMAP_RUNTIME_LIB="$ROOT/tools/runtime_dependencies/colmap_4.1_cuda/lib"
if [[ -d "$COLMAP_RUNTIME_LIB" ]]; then
  export LD_LIBRARY_PATH="$COLMAP_RUNTIME_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

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

prepare_shared_sfm() {
  local source_work="$SOURCE_CANON/workspace"
  local completion="$source_work/distorted/conversion_completion_manifest.json"
  if [[ -f "$completion" ]]; then
    test -f "$source_work/sparse/0/cameras.bin" \
      -a -f "$source_work/sparse/0/images.bin" \
      -a -f "$source_work/sparse/0/points3D.bin" \
      -a -f "$source_work/distorted/sparse_aligned/cameras.bin" \
      -a -f "$source_work/distorted/sparse_aligned/images.bin" \
      -a -f "$source_work/distorted/sparse_aligned/points3D.bin"
    return
  fi
  test ! -e "$SOURCE_CANON"
  test -x "$COLMAP"
  test "$(sha "$COLMAP")" = "$COLMAP_SHA"
  mkdir -p "$source_work/input" "$LOG_ROOT"
  while IFS= read -r image; do
    ln -s "$image" "$source_work/input/$(basename "$image")"
  done < <(find "$CFR/rgb" -maxdepth 1 -type f | sort)
  test "$(find "$source_work/input" -maxdepth 1 -type l | wc -l)" -eq "$TOTAL"
  cd "$CODE"
  "$PY" convert_uavfgs.py -s "$source_work" \
    --colmap_executable "$COLMAP" --exiftool_executable exiftool \
    --required_colmap_version 4.1.0 --required_colmap_sha256 "$COLMAP_SHA" \
    --database_policy reset --expected_legacy_database_sha256 "" \
    --wgs84_code 0 --prior_position_std_m 1.0 --camera SIMPLE_RADIAL \
    --matching spatial --matcher_args "--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500" \
    --colmap_gpu_index 0 --sfm_mapper global --global_mapper_args "" \
    --global_mapper_random_seed 0 --mapper_multiple_models 1 --min_model_size "$TOTAL" \
    --init_min_num_inliers 50 --abs_pose_min_num_inliers 20 \
    --require_cuda_colmap --require_global_mapper_cudss --use_model_aligner \
    --model_aligner_args "--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0" \
    2>&1 | tee "$LOG_ROOT/shared_sfm.log"
  test -f "$completion" \
    -a -f "$source_work/sparse/0/cameras.bin" \
    -a -f "$source_work/sparse/0/images.bin" \
    -a -f "$source_work/sparse/0/points3D.bin"
}

materialize_runtime_inputs() {
  mkdir -p "$RUNTIME_LISTS" "$WORK/images"
  "$PY" - "$CODE" "$WORK/sparse/0/images.bin" "$CFR/rgb" \
    "$BIND/train_list.txt" "$BIND/test_list.txt" "$WORK/images" \
    "$RGB_TRAIN_LIST" "$RGB_TEST_LIST" "$THERMAL_TRAIN_LIST" "$THERMAL_TEST_LIST" \
    "$RUNTIME_LISTS/manifest.json" "$SCENE" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

(code, images_bin, source_root, train_in, test_in, images_out,
 train_out, test_out, thermal_train_out, thermal_test_out, manifest_out, scene) = sys.argv[1:]
sys.path.insert(0, code)
from scene.colmap_loader import read_extrinsics_binary

def stem(value):
    return Path(str(value).replace("\\", "/")).stem.casefold()

def read_list(path):
    return [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

cameras = read_extrinsics_binary(images_bin)
actual = {}
for item in cameras.values():
    key = stem(item.name)
    if key in actual:
        raise RuntimeError(f"duplicate COLMAP camera stem: {key}")
    actual[key] = item.name

sources = {}
for item in Path(source_root).iterdir():
    if not item.is_file():
        continue
    key = stem(item.name)
    if key in sources:
        raise RuntimeError(f"duplicate CFR image stem: {key}")
    sources[key] = item.resolve()
if set(actual) != set(sources):
    raise RuntimeError(
        f"COLMAP/CFR stem mismatch: missing={sorted(set(actual)-set(sources))[:8]} "
        f"extra={sorted(set(sources)-set(actual))[:8]}"
    )

def map_list(path):
    values = read_list(path)
    keys = [stem(value) for value in values]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"duplicate split member in {path}")
    missing = [key for key in keys if key not in actual]
    if missing:
        raise RuntimeError(f"split/COLMAP mismatch in {path}: {missing[:8]}")
    return [actual[key] for key in keys]

train = map_list(train_in)
test = map_list(test_in)
if set(map(stem, train)) & set(map(stem, test)):
    raise RuntimeError("runtime train/test overlap")
if len(train) + len(test) != len(actual):
    raise RuntimeError("runtime train/test lists do not cover the camera collection")

image_dir = Path(images_out)
for item in image_dir.iterdir():
    if not item.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink runtime image: {item}")
    item.unlink()
for name in actual.values():
    os.symlink(sources[stem(name)], image_dir / name)

def atomic_lines(path, values):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
    temporary.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
    os.replace(temporary, target)

atomic_lines(train_out, train)
atomic_lines(test_out, test)
atomic_lines(thermal_train_out, [str(Path(value).with_suffix(".png")) for value in train])
atomic_lines(thermal_test_out, [str(Path(value).with_suffix(".png")) for value in test])
payload = {
    "schema": "uav-tgs-hold8-runtime-camera-binding-v1",
    "scene": scene,
    "mapping_rule": "case-insensitive pair stem to exact shared-COLMAP camera name",
    "membership_changed": False,
    "counts": {"total": len(actual), "train": len(train), "test": len(test)},
    "inputs": {"train_list_sha256": sha(train_in), "test_list_sha256": sha(test_in)},
    "outputs": {
        "train_list_sha256": sha(train_out), "test_list_sha256": sha(test_out),
        "thermal_train_list_sha256": sha(thermal_train_out),
        "thermal_test_list_sha256": sha(thermal_test_out),
    },
}
target = Path(manifest_out); temporary = target.with_name(target.name + f".tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
  test "$(find "$WORK/images" -maxdepth 1 -type l | wc -l)" -eq "$TOTAL"
  test "$(wc -l < "$RGB_TRAIN_LIST")" -eq "$TRAIN"
  test "$(wc -l < "$RGB_TEST_LIST")" -eq "$TEST"
}

prepare_scene() {
  verify_binding
  prepare_shared_sfm
  if [[ -f "$DERIVED/PREPARE_STATUS" ]]; then
    test "$(tr -d '\r\n' < "$DERIVED/PREPARE_STATUS")" = passed
    materialize_runtime_inputs
    return
  fi
  test ! -e "$WORK" -a ! -e "$RANGE"
  mkdir -p "$DERIVED/radiometry" "$WORK/images" "$WORK/sparse/0" "$WORK/distorted/sparse_aligned" "$EXP/protocol" "$LOG_ROOT"
  "$PY" "$CODE/tools/thermal_radiometry/estimate_scene_range.py" \
    --split-manifest "$BIND/bound_split.json" --npy-root "$DECODE/temperature_c" --output "$RANGE"
  cp -a "$SOURCE_CANON/workspace/sparse/0/." "$WORK/sparse/0/"
  cp -a "$SOURCE_CANON/workspace/distorted/sparse_aligned/." "$WORK/distorted/sparse_aligned/"
  materialize_runtime_inputs
  write_json_receipt "$EXP/protocol/prepare.json" prepare passed \
    "binding=$BIND/binding_manifest.json" "split=$BIND/bound_split.json" "range=$RANGE" \
    "cameras=$WORK/sparse/0/cameras.bin" "images=$WORK/sparse/0/images.bin" \
    "runtime_lists=$RUNTIME_LISTS/manifest.json"
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
    --train_list "$RGB_TRAIN_LIST" --test_list "$RGB_TEST_LIST" \
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
    "checkpoint=$RGB_CKPT" "ply=$RGB_PLY" "train_list=$RGB_TRAIN_LIST" "test_list=$RGB_TEST_LIST" \
    "results=$RGB_MODEL/results.json" "per_view=$RGB_MODEL/per_view.json"
  printf 'passed\n' > "$RGB_ROOT/STATUS"
}

prepare_thermal() {
  run_rgb
  if [[ -f "$DERIVED/THERMAL_STATUS" ]]; then test "$(tr -d '\r\n' < "$DERIVED/THERMAL_STATUS")" = passed; return; fi
  for path in "$TEMP_UD" "$THERMAL" "$FORMAL_SUPPORT" "$HOTSPOT"; do test ! -e "$path"; done
  mkdir -p "$DERIVED/radiometry" "$THERMAL" "$EXP/protocol" "$LOG_ROOT"
  local train_sha test_sha anchor_native sparse_bin
  train_sha="$(sha "$THERMAL_TRAIN_LIST")"; test_sha="$(sha "$THERMAL_TEST_LIST")"
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
    --train_list "$THERMAL_TRAIN_LIST" --test_list "$THERMAL_TEST_LIST" \
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
  train_sha="$(sha "$THERMAL_TRAIN_LIST")"; test_sha="$(sha "$THERMAL_TEST_LIST")"
  cd "$CODE"
  "$PY" render.py -s "$THERMAL" --images images -m "$model" -r 4 --eval \
    --train_list "$THERMAL_TRAIN_LIST" --test_list "$THERMAL_TEST_LIST" \
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
  local train_sha test_sha; train_sha="$(sha "$THERMAL_TRAIN_LIST")"; test_sha="$(sha "$THERMAL_TEST_LIST")"
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
    --train_list "$THERMAL_TRAIN_LIST" --test_list "$THERMAL_TEST_LIST" \
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
  local projection_start projection_end
  projection_start="$(date +%s.%N)"
  "$PY" tools/build_adaptive_scale_anchor.py --scene-name "$SCENE" --method scsp \
    --input-model-dir "$RGB_MODEL" --output-model-dir "$anchor" --anchor-iteration 30000 \
    --sparse-root "$WORK/sparse/0" --support-voxel-size 1.5 --support-max-voxel-radius 2 \
    --expected-checkpoint-sha256 "$(sha "$RGB_CKPT")" --expected-ply-sha256 "$(sha "$RGB_PLY")" \
    --code-commit "$HEAD" > "$method_root/logs/scsp_projection.jsonl"
  projection_end="$(date +%s.%N)"
  "$PY" - "$method_root/efficiency/scsp_projection.json" "$projection_start" "$projection_end" \
    "$anchor/adaptive_scale_manifest.json" <<'PY'
import hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
output,start,end,manifest=sys.argv[1:]
manifest_path=Path(manifest)
value={
    'schema':'uav-tgs-scsp-projection-efficiency-v1',
    'stage':'scsp_projection',
    'wall_time_s':float(end)-float(start),
    'started_epoch_s':float(start),
    'finished_epoch_s':float(end),
    'created_at_utc':datetime.now(timezone.utc).isoformat(),
    'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
}
target=Path(output); temporary=target.with_name(target.name+f'.tmp-{os.getpid()}')
temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(temporary,target)
PY
  local modified; modified="$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["counts"]["modified_gaussians"])' "$anchor/adaptive_scale_manifest.json")"
  if [[ "$modified" -eq 0 ]]; then
    if [[ ! -f "$EXP/methods/raw_f3/endpoint.json" ]]; then
      run_raw_or_adaptive raw_f3
    fi
    test -f "$EXP/methods/raw_f3/endpoint.json"
    write_json_receipt "$method_root/alias_to_raw_f3.json" scsp_noop_alias passed \
      "scsp_manifest=$anchor/adaptive_scale_manifest.json" "raw_endpoint=$EXP/methods/raw_f3/endpoint.json" \
      "projection_efficiency=$method_root/efficiency/scsp_projection.json"
    printf 'passed\n' > "$method_root/STATUS"; return
  fi
  local refit="$method_root/rgb_refit" sequence="$method_root/protocol/camera_sequence.json"
  local train_sha test_sha anchor_ckpt anchor_ply
  train_sha="$(sha "$RGB_TRAIN_LIST")"; test_sha="$(sha "$RGB_TEST_LIST")"
  anchor_ckpt="$anchor/chkpnt30000.pth"; anchor_ply="$anchor/point_cloud/iteration_30000/point_cloud.ply"
  "$PY" tools/build_fixed_camera_sequence.py --camera-names "$RGB_TRAIN_LIST" --output "$sequence" \
    --seed 0 --steps 5000 --scene "$SCENE" --split-sha256 "$train_sha" --anchor-sha256 "$(sha "$anchor_ckpt")"
  "$PY" train.py -s "$WORK" --images images -r 4 --eval -m "$refit" \
    --train_list "$RGB_TRAIN_LIST" --test_list "$RGB_TEST_LIST" --train_list_sha256 "$train_sha" --test_list_sha256 "$test_sha" \
    --start_checkpoint "$anchor_ckpt" --iterations 35000 --save_iterations 35000 --checkpoint_iterations 35000 --test_iterations 35000 \
    --position_lr_max_steps 30000 --rgb_continuation_anchor_iteration 30000 --rgb_continuation_scheduler_horizon 30000 \
    --rgb_continuation_updates 5000 --rgb_continuation_recipe appearance_only --rgb_optimizer_state fresh \
    --fixed_camera_sequence "$sequence" --artifact_save_semantics aligned --optimizer_step_at_final_iteration --disable_viewer \
    --benchmark_efficiency --efficiency_output "$method_root/efficiency/rgb_refit.json" --efficiency_stage rgb_sh_only_refit \
    2>&1 | tee "$method_root/logs/rgb_refit.log"
  local model="$method_root/model" refit_ckpt="$refit/chkpnt35000.pth" refit_ply="$refit/point_cloud/iteration_35000/point_cloud.ply"
  local ttrain ttest; ttrain="$(sha "$THERMAL_TRAIN_LIST")"; ttest="$(sha "$THERMAL_TEST_LIST")"
  "$PY" train.py -s "$THERMAL" --images images -r 4 -m "$model" \
    --train_list "$THERMAL_TRAIN_LIST" --test_list "$THERMAL_TEST_LIST" --train_list_sha256 "$ttrain" --test_list_sha256 "$ttest" \
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
    "projection_efficiency=$method_root/efficiency/scsp_projection.json" \
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
