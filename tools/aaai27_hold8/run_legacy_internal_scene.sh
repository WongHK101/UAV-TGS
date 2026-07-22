#!/usr/bin/env bash
set -Eeuo pipefail

# Explicit reproduction entrypoint for the historical quarter-resolution
# Hold-8 endpoints. Native-auto resolution is the default everywhere else.
export UAV_TGS_EXPERIMENT_ID=aaai27_hold8_v2
export UAV_TGS_TRAIN_RESOLUTION=4
export UAV_TGS_APPEARANCE_RESOLUTION=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/run_phase1_internal_scene.sh" "$@"
