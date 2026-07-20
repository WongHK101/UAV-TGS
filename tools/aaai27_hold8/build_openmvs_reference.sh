#!/usr/bin/env bash
set -Eeuo pipefail

# Build one shared, RGB-train-only OpenMVS reference for a Hold-8 scene.  The
# result is evaluation-only and excluded from every method-specific cost.

if [[ $# -ne 1 ]]; then
  echo "usage: $0 Building|InternalRoad|PVpanel|TransmissionTower|Urban20K|Orchard" >&2
  exit 2
fi

SCENE="$1"
ROOT="${UAV_TGS_ROOT:-/root/autodl-tmp/UAV-TGS}"
CODE="$ROOT/code"
PY="$ROOT/environments/uav-tgs/bin/python"
OPENMVS="${OPENMVS_ROOT:-$ROOT/tools/openmvs-2.4.0-refine-fail-closed/bin/OpenMVS}"
BIND="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/$SCENE/binding"
COLLECTION="$ROOT/derived/thermal_radiometry/aaai27_hold8_v2/splits/hold8/collection_manifest.json"
DERIVED="$ROOT/derived/aaai27_hold8_v2/$SCENE"
WORK="$DERIVED/workspace"
THERMAL="$DERIVED/thermal_benchmark"
RUNTIME="$DERIVED/runtime_lists"
RGB_MODEL="$ROOT/experiments/aaai27_hold8_v2/$SCENE/rgb_anchor/Model_RGB"
OUT="$DERIVED/reference_openmvs_hold8_v2"
TRAIN_ONLY="$OUT/train_only_workspace"
BASE="$OUT/base_reference"
BOUND="$OUT/bound_expected_depth"
PROTOCOL="$OUT/protocol.json"
COMPAT_BINDING="$OUT/materializer_binding.json"
EMPTY_GUARD="$OUT/empty_guard_list.txt"
LOG="$ROOT/logs/experiments/aaai27_hold8_v2/$SCENE/openmvs_reference.log"

sha() { sha256sum "$1" | awk '{print $1}'; }

test -x "$PY"
test -z "$(git -C "$CODE" status --porcelain=v1 --untracked-files=all)"
test "$(tr -d '\r\n' < "$DERIVED/THERMAL_STATUS")" = passed
test -f "$COLLECTION" -a -f "$BIND/bound_split.json" -a -f "$BIND/binding_manifest.json"
test -f "$WORK/sparse/0/cameras.bin" -a -d "$WORK/images" -a -d "$THERMAL/sparse/0"
test -f "$RGB_MODEL/cameras.json"
for tool in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh; do test -x "$OPENMVS/$tool"; done

if [[ -f "$OUT/STATUS" ]]; then
  test "$(tr -d '\r\n' < "$OUT/STATUS")" = passed
  test -f "$BOUND/manifest.json"
  echo "$SCENE Hold-8 OpenMVS reference already passed: $BOUND/manifest.json"
  exit 0
fi

test ! -e "$OUT"
available_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
test "$available_kib" -ge 41943040
mkdir -p "$OUT" "$(dirname "$LOG")"
: > "$EMPTY_GUARD"

"$PY" - "$SCENE" "$BIND/binding_manifest.json" "$RUNTIME/train_list.txt" \
  "$RUNTIME/test_list.txt" "$EMPTY_GUARD" "$COMPAT_BINDING" "$PROTOCOL" \
  "$TRAIN_ONLY" "$THERMAL" "$RUNTIME/thermal_train_list.txt" \
  "$RUNTIME/thermal_test_list.txt" <<'PY'
import hashlib, json, os, sys
from pathlib import Path

(scene, source_binding, train, test, guard, compat, protocol, train_only,
 thermal, thermal_train, thermal_test) = sys.argv[1:]
paths = [Path(value).resolve() for value in (source_binding, train, test, guard)]
source_binding, train, test, guard = paths
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(path): return [row.strip() for row in path.read_text(encoding='utf-8-sig').splitlines() if row.strip()]
def normalized_name(value): return Path(str(value).replace('\\','/')).name.casefold()
source = json.loads(source_binding.read_text(encoding='utf-8'))
counts = {'total': len(lines(train)) + len(lines(test)), 'train': len(lines(train)),
          'test': len(lines(test)), 'guard': 0}
if source.get('protocol_id') != 'uav-tgs-aaai27-hold8-v2' or source.get('scene') != scene:
    raise RuntimeError('not the frozen Hold-8 binding')
if source.get('counts') != {key: counts[key] for key in ('total','train','test')}:
    raise RuntimeError('Hold-8 binding counts mismatch')
records = source.get('files')
if not isinstance(records, list) or len(records) != counts['total']:
    raise RuntimeError('Hold-8 binding records mismatch')
expected = {
    split: [normalized_name(row.get('camera_name', '')) for row in records
            if isinstance(row, dict) and row.get('split') == split]
    for split in ('train', 'test')
}
runtime = {split: [normalized_name(row) for row in lines(path)]
           for split, path in (('train', train), ('test', test))}
if any(not name for names in expected.values() for name in names):
    raise RuntimeError('Hold-8 binding has an empty camera identity')
if expected != runtime:
    raise RuntimeError('Hold-8 runtime list identities/order mismatch')
if any(len(names) != len(set(names)) for names in runtime.values()):
    raise RuntimeError('Hold-8 runtime list has duplicate camera identities')
outputs = {}
for label, path in (('train',train),('test',test),('guard',guard)):
    outputs[f'{label}_list.txt'] = {'path': str(path), 'sha256': sha(path)}
compat_core = {
    'schema_name': 'uav_tgs_formal_scene_decode_binding', 'schema_version': 1,
    'status': 'passed', 'scene': scene, 'sfm_image_scope': 'shared_sfm_all_images',
    'counts': counts, 'outputs': outputs,
    'collection_hash': source['collection_hash'],
    'collection_split_hash': source['collection_split_hash'],
    'hold8_source_binding_sha256': sha(source_binding),
    'formal_list_sha256': source.get('list_sha256', {}),
    'compatibility_scope': 'empty guard compatibility for train-only materializer; no guard protocol introduced',
}
compat_core['binding_hash'] = hashlib.sha256(json.dumps(compat_core, sort_keys=True, separators=(',',':')).encode()).hexdigest()
Path(compat).write_text(json.dumps(compat_core, indent=2, sort_keys=True)+'\n', encoding='utf-8')
strict = {
    'schema': 'uav-tgs-hold8-openmvs-reference-protocol-v1',
    'scene_name': scene,
    'sfm_image_scope': 'shared_sfm_all_images',
    'reference_construction_scope': 'rgb_train_images_only_after_shared_sfm',
    'split': {'protocol_id': 'uav-tgs-aaai27-hold8-v2', **counts,
              'bound_split_sha256': source.get('bound_split_sha256')},
    'artifacts': {'train_union_source_root': str(Path(train_only).resolve()),
                  'strict_thermal_root': str(Path(thermal).resolve()),
                  'probe_camera_root': str(Path(thermal).resolve())},
    'lists': {'train_union': str(train), 'probe_camera_train': str(Path(thermal_train).resolve()),
              'probe_test': str(Path(thermal_test).resolve()),
              'reference_probe_exclusion': str(test)},
    'backend_recipe': {'backend': 'openmvs-2.4.0-refine-cuda-fail-closed',
                       'resolution_level': 1, 'max_resolution': 2000, 'min_resolution': 640,
                       'number_views': 8, 'number_views_fuse': 3, 'iterations': 4,
                       'refine_resolution_level': 1, 'refine_scales': 2,
                       'probe_resolution_arg': 4},
}
Path(protocol).write_text(json.dumps(strict, indent=2, sort_keys=True)+'\n', encoding='utf-8')
PY

export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
cd "$CODE"
{
  "$PY" tools/geometric_repeatability/materialize_train_only_colmap.py \
    --source-model "$WORK/sparse/0" --binding-manifest "$COMPAT_BINDING" \
    --sfm-image-scope shared_sfm_all_images --train-list "$RUNTIME/train_list.txt" \
    --test-list "$RUNTIME/test_list.txt" --guard-list "$EMPTY_GUARD" \
    --image-root "$WORK/images" --output-workspace "$TRAIN_ONLY" --image-mode hardlink

  "$PY" tools/geometric_repeatability/build_depth_reference.py \
    --strict_protocol_manifest "$PROTOCOL" --out_dir "$BASE" \
    --openmvs_interface_colmap_cmd "$OPENMVS/InterfaceCOLMAP" \
    --openmvs_densify_cmd "$OPENMVS/DensifyPointCloud" \
    --openmvs_reconstruct_mesh_cmd "$OPENMVS/ReconstructMesh" \
    --openmvs_refine_mesh_cmd "$OPENMVS/RefineMesh" --openmvs_cuda_device 0 \
    --openmvs_resolution_level 1 --openmvs_max_resolution 2000 \
    --openmvs_min_resolution 640 --openmvs_number_views 8 \
    --openmvs_number_views_fuse 3 --openmvs_iterations 4 \
    --openmvs_refine_resolution_level 1 --openmvs_refine_scales 2 --resolution_arg 4

  "$PY" tools/aaai27_hold8/bind_expected_depth_bundle.py --kind reference \
    --source-manifest "$BASE/reference_depth_manifest.json" \
    --collection-manifest "$COLLECTION" --scene-split-manifest "$BIND/bound_split.json" \
    --output-root "$BOUND"
} 2>&1 | tee "$LOG"

test "$(find "$BOUND/views" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq "$(wc -l < "$RUNTIME/test_list.txt")"
sha256sum "$BIND/bound_split.json" "$COLLECTION" "$PROTOCOL" "$COMPAT_BINDING" \
  "$BASE/reference_depth_manifest.json" "$BOUND/manifest.json" > "$OUT/receipt.sha256"
printf 'passed\n' > "$OUT/STATUS"
echo "$SCENE Hold-8 OpenMVS reference passed: $BOUND/manifest.json"
