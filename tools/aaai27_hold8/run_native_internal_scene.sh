#!/usr/bin/env bash
set -Eeuo pipefail

# Formal native-resolution internal runner. The legacy Hold-8 runner remains
# available by invoking run_phase1_internal_scene.sh directly, whose defaults
# still reproduce the historical quarter-resolution experiment root.
export UAV_TGS_EXPERIMENT_ID=aaai27_hold8_v2_native
export UAV_TGS_TRAIN_RESOLUTION=-1
export UAV_TGS_APPEARANCE_RESOLUTION=-1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_phase1_internal_scene.sh" "$@"
