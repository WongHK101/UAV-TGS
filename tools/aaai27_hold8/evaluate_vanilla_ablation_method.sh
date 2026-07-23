#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ! "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=16; fi
if [[ ! "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export MKL_NUM_THREADS=16; fi

if [[ $# -ne 2 ]]; then
  echo "usage: $0 Scene t_only_sfm_3dgs|rgb_sfm_t_3dgs|naive_two_pass_3dgs" >&2
  exit 2
fi
SCENE="$1"; METHOD="$2"
ROOT="${UAV_TGS_ROOT:-/root/autodl-tmp/UAV-TGS}"
CODE="$ROOT/code"; PY="$ROOT/environments/uav-tgs/bin/python"
FORMAL_EXPERIMENT=aaai27_hold8_v2_native
ABLATION_EXPERIMENT=aaai27_hold8_v2_native_vanilla_ablation
case "$SCENE" in
  Building) DECODE_PROTO=aaai_second_review_v1 ;;
  InternalRoad) DECODE_PROTO=aaai_second_review_v1 ;;
  PVpanel) DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  TransmissionTower) DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  Urban20K) DECODE_PROTO=aaai27_a3_three_scene_v1 ;;
  Orchard) DECODE_PROTO=aaai27_phase1_formal_v1 ;;
  *) echo "unsupported representative scene: $SCENE" >&2; exit 2 ;;
esac
case "$METHOD" in t_only_sfm_3dgs|rgb_sfm_t_3dgs|naive_two_pass_3dgs) ;; *) exit 2 ;; esac

BIND="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/$SCENE/binding"
COLLECTION="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/splits/hold8/collection_manifest.json"
DECODE="$ROOT/derived/thermal_radiometry/$DECODE_PROTO/$SCENE"
DERIVED="$ROOT/derived/aaai27_hold8_v2/$SCENE"
THERMAL="$DERIVED/thermal_benchmark"; TEMP_FORMAL="$DERIVED/thermal_undistorted"
SUPPORT="$DERIVED/formal_support"; RANGE="$DERIVED/radiometry/range_manifest.json"
CANONICAL="$DERIVED/radiometry/canonical_manifest.json"
HOTSPOT_THRESHOLD="$DERIVED/radiometry/hotspot_threshold_train_q95.json"
RUNTIME="$DERIVED/runtime_lists"
REFERENCE="$DERIVED/reference_openmvs_hold8_v2/bound_expected_depth/manifest.json"
FORMAL_EXP="$ROOT/experiments/$FORMAL_EXPERIMENT/$SCENE"
EXP="$ROOT/experiments/$ABLATION_EXPERIMENT/$SCENE"
METHOD_ROOT="$EXP/$METHOD"; MODEL="$METHOD_ROOT/model"; ITERATION=30000
EVAL_ROOT="$EXP/evaluation/$METHOD"
LOG_ROOT="$ROOT/logs/experiments/$ABLATION_EXPERIMENT/$SCENE/evaluation/$METHOD"
BINDING="$FORMAL_EXP/protocol/formal_radiometry_evaluation_binding.json"

sha() { sha256sum "$1" | awk '{print $1}'; }
test -x "$PY"; test -z "$(git -C "$CODE" status --porcelain=v1 --untracked-files=all)"
test "$(tr -d '\r\n' < "$METHOD_ROOT/STATUS")" = passed
test -f "$MODEL/point_cloud/iteration_${ITERATION}/point_cloud.ply" -a -f "$METHOD_ROOT/endpoint.json"
if [[ -f "$EVAL_ROOT/STATUS" ]]; then
  test "$(tr -d '\r\n' < "$EVAL_ROOT/STATUS")" = passed
  echo "$SCENE/$METHOD evaluation already passed"; exit 0
fi
test ! -e "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"/{appearance,temperature,hotspot,geometry,efficiency,protocol} "$LOG_ROOT"
printf 'running\n' > "$EVAL_ROOT/STATUS"
on_error() { local rc=$?; printf 'failed\n' > "$EVAL_ROOT/STATUS"; exit "$rc"; }
trap on_error ERR

cp -a "$MODEL/results.json" "$EVAL_ROOT/appearance/thermal_results.json"
cp -a "$MODEL/per_view.json" "$EVAL_ROOT/appearance/thermal_per_view.json"
cp -a "$METHOD_ROOT/train_efficiency.json" "$EVAL_ROOT/efficiency/"
cp -a "$METHOD_ROOT/render_efficiency.json" "$EVAL_ROOT/efficiency/"
if [[ "$METHOD" == naive_two_pass_3dgs ]]; then
  cp -a "$FORMAL_EXP/rgb_anchor/Model_RGB/results.json" "$EVAL_ROOT/appearance/rgb_results.json"
  cp -a "$FORMAL_EXP/rgb_anchor/Model_RGB/per_view.json" "$EVAL_ROOT/appearance/rgb_per_view.json"
else
  "$PY" - "$EVAL_ROOT/appearance/rgb_not_applicable.json" "$METHOD" <<'PY'
import json,sys
from pathlib import Path
out,method=sys.argv[1:]
Path(out).write_text(json.dumps({'status':'not_applicable','method':method,
    'reason':'thermal-only model has no RGB appearance branch'},indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
fi

RENDERS="$MODEL/test/ours_${ITERATION}/renders"
test "$(find "$RENDERS" -maxdepth 1 -type f -name '*.png' | wc -l)" -eq "$(wc -l < "$RUNTIME/thermal_test_list.txt")"
if [[ "$METHOD" == t_only_sfm_3dgs ]]; then
  TEMPERATURE_ROOT="$METHOD_ROOT/t_only_temperature_ud/temperature_c"
  MASK_ROOT="$METHOD_ROOT/t_only_temperature_ud/valid_support"
else
  TEMPERATURE_ROOT="$TEMP_FORMAL/temperature_c"
  MASK_ROOT="$SUPPORT/bool"
fi
"$PY" "$CODE/tools/thermal_radiometry/evaluate_temperature.py" \
  --ground-truth-root "$TEMPERATURE_ROOT" --render-root "$RENDERS" \
  --report "$EVAL_ROOT/temperature/test.json" --range-manifest "$RANGE" \
  --split-manifest "$BIND/bound_split.json" --subset test --mask-root "$MASK_ROOT" \
  --alpha-threshold 0 --require-support > "$LOG_ROOT/temperature.log" 2>&1

if [[ "$METHOD" == t_only_sfm_3dgs ]]; then
  "$PY" - "$EVAL_ROOT/hotspot/not_applicable.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'not_applicable',
  'reason':'independent T-only SfM cameras do not share the frozen RGB-camera hotspot support binding'},
  indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
  "$PY" - "$EVAL_ROOT/geometry/not_applicable.json" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'not_applicable',
  'reason':'independent T-only SfM has arbitrary similarity gauge and is not evaluated against the frozen RGB-camera OpenMVS reference'},
  indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
else
  if [[ ! -f "$BINDING" ]]; then
    echo "missing frozen formal radiometry binding: $BINDING" >&2; exit 1
  fi
  "$PY" "$CODE/tools/evaluate_formal_baseline_hotspots.py" \
    --method-name "$METHOD" --scene-name "$SCENE" \
    --formal-radiometry-binding-manifest "$BINDING" --bound-split "$BIND/bound_split.json" \
    --decode-manifest "$DECODE/manifests/decode_manifest.jsonl" \
    --decode-protocol "$DECODE/manifests/decode_protocol_used_v1.jsonl" \
    --range-manifest "$RANGE" --canonical-manifest "$CANONICAL" \
    --optimization-support-manifest "$TEMP_FORMAL/manifest.json" \
    --evaluation-support-manifest "$SUPPORT/manifest.json" \
    --hotspot-threshold-manifest "$HOTSPOT_THRESHOLD" --temperature-root "$TEMP_FORMAL/temperature_c" \
    --evaluation-support-root "$SUPPORT" --render-root "$RENDERS" \
    --output "$EVAL_ROOT/hotspot/test.json" > "$LOG_ROOT/hotspot.log" 2>&1

  RAW_DEPTH="$EVAL_ROOT/geometry/renderer_bundle"
  "$PY" "$CODE/tools/geometric_repeatability/export_gaussian_probe_bundle.py" \
    -s "$THERMAL" --images images -m "$MODEL" -r 4 --eval \
    --train_list "$RUNTIME/thermal_train_list.txt" --test_list "$RUNTIME/thermal_test_list.txt" \
    --train_list_sha256 "$(sha "$RUNTIME/thermal_train_list.txt")" \
    --test_list_sha256 "$(sha "$RUNTIME/thermal_test_list.txt")" \
    --iteration "$ITERATION" --out_dir "$RAW_DEPTH" --split_label test \
    --scene_name_override "$SCENE" --camera_frame_mode scene_test_bound \
    --native_cameras_json "$MODEL/cameras.json" --formal_split_manifest "$BIND/bound_split.json" \
    --depth_diagnostics --appearance_modality none > "$LOG_ROOT/depth_export.log" 2>&1
  BOUND_DEPTH="$EVAL_ROOT/geometry/expected_depth_bundle"
  "$PY" -m tools.aaai27_hold8.bind_expected_depth_bundle --kind model \
    --source-manifest "$RAW_DEPTH/split_manifest.json" --collection-manifest "$COLLECTION" \
    --scene-split-manifest "$BIND/bound_split.json" --output-root "$BOUND_DEPTH" \
    --method-name "$METHOD" > "$LOG_ROOT/depth_binding.log" 2>&1
  "$PY" "$CODE/tools/hold8_expected_depth_evaluator.py" evaluate \
    --reference-manifest "$REFERENCE" --model-manifest "$BOUND_DEPTH/manifest.json" \
    --collection-manifest "$COLLECTION" --scene-split-manifest "$BIND/bound_split.json" \
    --expected-collection-manifest-sha256 "$(sha "$COLLECTION")" \
    --expected-scene-split-manifest-sha256 "$(sha "$BIND/bound_split.json")" \
    --out-dir "$EVAL_ROOT/geometry/metrics" > "$LOG_ROOT/depth_metrics.log" 2>&1
fi

"$PY" - "$EVAL_ROOT/completion.json" "$SCENE" "$METHOD" "$METHOD_ROOT/endpoint.json" \
  "$EVAL_ROOT/appearance/thermal_results.json" "$EVAL_ROOT/temperature/test.json" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
out,scene,method,*paths=sys.argv[1:]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
value={'schema':'uav-tgs-vanilla-ablation-evaluation-v1','status':'passed','scene':scene,
       'method':method,'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'artifacts':{Path(p).name:{'path':str(Path(p).resolve()),'sha256':sha(p),'size_bytes':Path(p).stat().st_size} for p in paths}}
value['payload_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
printf 'passed\n' > "$EVAL_ROOT/STATUS"
trap - ERR
echo "$SCENE/$METHOD evaluation passed"
