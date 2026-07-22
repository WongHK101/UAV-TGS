#!/usr/bin/env bash
set -Eeuo pipefail

export UAV_TGS_EXPERIMENT_ID=aaai27_hold8_v2
export UAV_TGS_RGB_APPEARANCE_RESOLUTION=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/evaluate_internal_method.sh" "$@"
