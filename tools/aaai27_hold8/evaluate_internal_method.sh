#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 Scene raw_f3|scsp_refit_f3|adaptive_opacity_scale_clamp" >&2
  exit 2
fi

SCENE="$1"; METHOD="$2"
ROOT="${UAV_TGS_ROOT:-/root/autodl-tmp/UAV-TGS}"
CODE="$ROOT/code"; PY="$ROOT/environments/uav-tgs/bin/python"
case "$SCENE" in
  Building) SLUG=building; DECODE_PROTO=aaai_second_review_v1 ;;
  InternalRoad) SLUG=internalroad; DECODE_PROTO=aaai_second_review_v1 ;;
  PVpanel) SLUG=pvpanel; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  TransmissionTower) SLUG=transmissiontower; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  Urban20K) SLUG=urban20k; DECODE_PROTO=aaai27_a3_three_scene_v1 ;;
  Orchard) SLUG=orchard; DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  *) echo "unsupported scene: $SCENE" >&2; exit 2 ;;
esac
case "$METHOD" in raw_f3|scsp_refit_f3|adaptive_opacity_scale_clamp) ;; *) exit 2 ;; esac

BIND="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/$SCENE/binding"
COLLECTION="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/splits/hold8/collection_manifest.json"
DECODE="$ROOT/derived/thermal_radiometry/$DECODE_PROTO/$SCENE"
DERIVED="$ROOT/derived/aaai27_hold8_v2/$SCENE"
WORK="$DERIVED/workspace"; THERMAL="$DERIVED/thermal_benchmark"
TEMP_UD="$DERIVED/thermal_undistorted"; SUPPORT="$DERIVED/formal_support"
RANGE="$DERIVED/radiometry/range_manifest.json"
CANONICAL="$DERIVED/radiometry/canonical_manifest.json"
HOTSPOT_THRESHOLD="$DERIVED/radiometry/hotspot_threshold_train_q95.json"
RUNTIME="$DERIVED/runtime_lists"
EXP="$ROOT/experiments/aaai27_hold8_v2/$SCENE"
METHOD_ROOT="$EXP/methods/$METHOD"
REFERENCE="$DERIVED/reference_openmvs_hold8_v2/bound_expected_depth/manifest.json"
EVAL_ROOT="$EXP/evaluation/$METHOD"
BINDING="$EXP/protocol/formal_radiometry_evaluation_binding.json"
LOG_ROOT="$ROOT/logs/experiments/aaai27_hold8_v2/$SCENE/evaluation/$METHOD"
RGB_ANCHOR="$EXP/rgb_anchor/Model_RGB"

sha() { sha256sum "$1" | awk '{print $1}'; }
test -x "$PY"; test -z "$(git -C "$CODE" status --porcelain=v1 --untracked-files=all)"
test "$(tr -d '\r\n' < "$METHOD_ROOT/STATUS")" = passed
test -f "$REFERENCE" -a -f "$COLLECTION" -a -f "$BIND/bound_split.json"

if [[ -f "$EVAL_ROOT/STATUS" ]]; then
  test "$(tr -d '\r\n' < "$EVAL_ROOT/STATUS")" = passed
  echo "$SCENE/$METHOD Hold-8 evaluation already passed"
  exit 0
fi
test ! -e "$EVAL_ROOT"

if [[ "$METHOD" == scsp_refit_f3 && -f "$METHOD_ROOT/alias_to_raw_f3.json" ]]; then
  RAW="$EXP/evaluation/raw_f3"
  test "$(tr -d '\r\n' < "$RAW/STATUS")" = passed
  mkdir -p "$EVAL_ROOT"
  "$PY" - "$SCENE" "$METHOD_ROOT/alias_to_raw_f3.json" "$RAW/completion.json" "$EVAL_ROOT/alias.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
scene,training,raw,output=sys.argv[1:]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
value={'schema':'uav-tgs-hold8-evaluation-alias-v1','status':'passed','scene':scene,
       'method':'scsp_refit_f3','aliased_endpoint':'raw_f3','independent_endpoint_run':False,
       'independent_performance_claim':False,'additional_alias_cost_s':0.0,
       'training_alias_sha256':sha(training),'raw_evaluation_completion_sha256':sha(raw)}
Path(output).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
  printf 'passed\n' > "$EVAL_ROOT/STATUS"
  echo "$SCENE/$METHOD evaluation aliased to Raw-F3"
  exit 0
fi

case "$METHOD" in
  raw_f3|adaptive_opacity_scale_clamp) MODEL="$METHOD_ROOT/model"; ITERATION=60000; RGB_MODEL="$RGB_ANCHOR"; RGB_ITER=30000 ;;
  scsp_refit_f3) MODEL="$METHOD_ROOT/model"; ITERATION=65000; RGB_MODEL="$METHOD_ROOT/rgb_refit"; RGB_ITER=35000 ;;
esac
test -f "$MODEL/point_cloud/iteration_${ITERATION}/point_cloud.ply" -a -f "$METHOD_ROOT/endpoint.json"
mkdir -p "$EVAL_ROOT"/{appearance,temperature,hotspot,geometry,efficiency,protocol} "$LOG_ROOT" "$EXP/protocol"
printf 'running\n' > "$EVAL_ROOT/STATUS"
on_error() { local status=$?; printf 'failed\n' > "$EVAL_ROOT/STATUS"; exit "$status"; }
trap on_error ERR

cd "$CODE"
if [[ ! -f "$RGB_MODEL/results.json" ]]; then
  "$PY" render.py -s "$WORK" --images images -m "$RGB_MODEL" -r 4 --eval \
    --train_list "$RUNTIME/train_list.txt" --test_list "$RUNTIME/test_list.txt" \
    --train_list_sha256 "$(sha "$RUNTIME/train_list.txt")" --test_list_sha256 "$(sha "$RUNTIME/test_list.txt")" \
    --iteration "$RGB_ITER" --skip_train --save_by_image_name > "$LOG_ROOT/rgb_render.log" 2>&1
  "$PY" metrics.py -m "$RGB_MODEL" > "$LOG_ROOT/rgb_metrics.log" 2>&1
  "$PY" metrics_plus.py -m "$RGB_MODEL" --extra_iqa "" --save_json >> "$LOG_ROOT/rgb_metrics.log" 2>&1
fi
cp -a "$RGB_MODEL/results.json" "$EVAL_ROOT/appearance/rgb_results.json"
cp -a "$RGB_MODEL/per_view.json" "$EVAL_ROOT/appearance/rgb_per_view.json"
cp -a "$MODEL/results.json" "$EVAL_ROOT/appearance/thermal_results.json"
cp -a "$MODEL/per_view.json" "$EVAL_ROOT/appearance/thermal_per_view.json"
cp -a "$METHOD_ROOT/efficiency/." "$EVAL_ROOT/efficiency/"

if [[ ! -f "$BINDING" ]]; then
  "$PY" tools/thermal_radiometry/build_formal_evaluation_binding.py \
    --scene-name "$SCENE" --bound-split "$BIND/bound_split.json" \
    --decode-manifest "$DECODE/manifests/decode_manifest.jsonl" \
    --decode-protocol "$DECODE/manifests/decode_protocol_used_v1.jsonl" \
    --range-manifest "$RANGE" --canonical-manifest "$CANONICAL" \
    --optimization-support-manifest "$TEMP_UD/manifest.json" \
    --evaluation-support-manifest "$SUPPORT/manifest.json" \
    --temperature-root "$TEMP_UD/temperature_c" --output "$BINDING" \
    > "$LOG_ROOT/radiometry_binding.log" 2>&1
fi

RENDERS="$MODEL/test/ours_${ITERATION}/renders"
test -d "$RENDERS"
"$PY" tools/thermal_radiometry/evaluate_temperature.py \
  --ground-truth-root "$TEMP_UD/temperature_c" --render-root "$RENDERS" \
  --report "$EVAL_ROOT/temperature/test.json" --range-manifest "$RANGE" \
  --split-manifest "$BIND/bound_split.json" --subset test --mask-root "$SUPPORT/bool" \
  --alpha-threshold 0 --require-support > "$LOG_ROOT/temperature.log" 2>&1

"$PY" tools/evaluate_formal_baseline_hotspots.py --method-name "$METHOD" --scene-name "$SCENE" \
  --formal-radiometry-binding-manifest "$BINDING" --bound-split "$BIND/bound_split.json" \
  --decode-manifest "$DECODE/manifests/decode_manifest.jsonl" \
  --decode-protocol "$DECODE/manifests/decode_protocol_used_v1.jsonl" \
  --range-manifest "$RANGE" --canonical-manifest "$CANONICAL" \
  --optimization-support-manifest "$TEMP_UD/manifest.json" \
  --evaluation-support-manifest "$SUPPORT/manifest.json" \
  --hotspot-threshold-manifest "$HOTSPOT_THRESHOLD" --temperature-root "$TEMP_UD/temperature_c" \
  --evaluation-support-root "$SUPPORT" --render-root "$RENDERS" \
  --output "$EVAL_ROOT/hotspot/test.json" > "$LOG_ROOT/hotspot.log" 2>&1

RAW_DEPTH="$EVAL_ROOT/geometry/renderer_bundle"
"$PY" tools/geometric_repeatability/export_gaussian_probe_bundle.py \
  -s "$THERMAL" --images images -m "$MODEL" -r 4 --eval \
  --train_list "$RUNTIME/thermal_train_list.txt" --test_list "$RUNTIME/thermal_test_list.txt" \
  --train_list_sha256 "$(sha "$RUNTIME/thermal_train_list.txt")" \
  --test_list_sha256 "$(sha "$RUNTIME/thermal_test_list.txt")" \
  --iteration "$ITERATION" --out_dir "$RAW_DEPTH" --split_label test \
  --scene_name_override "$SCENE" --camera_frame_mode scene_test --depth_diagnostics \
  --appearance_modality none > "$LOG_ROOT/depth_export.log" 2>&1

BOUND_DEPTH="$EVAL_ROOT/geometry/expected_depth_bundle"
"$PY" -m tools.aaai27_hold8.bind_expected_depth_bundle --kind model \
  --source-manifest "$RAW_DEPTH/split_manifest.json" --collection-manifest "$COLLECTION" \
  --scene-split-manifest "$BIND/bound_split.json" --output-root "$BOUND_DEPTH" \
  --method-name "$METHOD" > "$LOG_ROOT/depth_binding.log" 2>&1

"$PY" tools/hold8_expected_depth_evaluator.py evaluate \
  --reference-manifest "$REFERENCE" --model-manifest "$BOUND_DEPTH/manifest.json" \
  --collection-manifest "$COLLECTION" --scene-split-manifest "$BIND/bound_split.json" \
  --expected-collection-manifest-sha256 "$(sha "$COLLECTION")" \
  --expected-scene-split-manifest-sha256 "$(sha "$BIND/bound_split.json")" \
  --out-dir "$EVAL_ROOT/geometry/metrics" > "$LOG_ROOT/depth_metrics.log" 2>&1

"$PY" - "$SCENE" "$METHOD" "$ITERATION" "$METHOD_ROOT/endpoint.json" \
  "$EVAL_ROOT/appearance/rgb_results.json" "$EVAL_ROOT/appearance/thermal_results.json" \
  "$EVAL_ROOT/temperature/test.json" "$EVAL_ROOT/hotspot/test.json" \
  "$EVAL_ROOT/geometry/metrics/geometry_metrics.json" "$EVAL_ROOT/completion.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
scene,method,iteration,*paths,output=sys.argv[1:]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
artifacts={Path(path).stem+f'_{i}':{'path':str(Path(path).resolve()),'sha256':sha(path),'size_bytes':Path(path).stat().st_size}
           for i,path in enumerate(paths)}
value={'schema':'uav-tgs-hold8-internal-evaluation-completion-v1','status':'passed',
       'scene':scene,'method':method,'iteration':int(iteration),'artifacts':artifacts,
       'completed_at_utc':datetime.now(timezone.utc).isoformat()}
value['payload_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
Path(output).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
printf 'passed\n' > "$EVAL_ROOT/STATUS"
trap - ERR
echo "$SCENE/$METHOD Hold-8 evaluation passed: $EVAL_ROOT"
