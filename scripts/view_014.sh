#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
DATASET="${SPECIALIST_DATASET:-logs/real-capture-20261116}"
SESSION="20261116_192018_IAQEB2604VWWN7"
BLUEPRINT="$DATASET/blueprints/specialist.rbl"
EXTRA_LAYERS=()
MODE=dashboard
if [[ "${1:-}" == --help ]]; then
  printf 'Usage: bash scripts/view_014.sh [--official | --imu] [Rerun options]\nDefault: IMU, opening, tag boxes/IDs, 3D and both videos.\n--official: original upstream layout. --imu: compatibility alias for default.\n'
  exit 0
elif [[ "${1:-}" == --official ]]; then
  MODE=official
  shift
elif [[ "${1:-}" == --imu ]]; then
  shift
fi
if [[ "$MODE" == dashboard ]]; then
  DIAGNOSTICS="${SPECIALIST_014_DIAGNOSTICS:-logs/014-diagnostics}"
  STATE="logs/014-processing/window50/state_s8_r100/$SESSION.rrd"
  STATE_ARGS=()
  if [[ -f "$STATE" ]]; then
    STATE_ARGS+=(--state "$STATE")
  fi
  mamba run -n specialist-umi python scripts/make_capture_dashboard.py \
    --dataset "$DATASET" --session "$SESSION" --output-dir "$DIAGNOSTICS" \
    "${STATE_ARGS[@]}"
  BLUEPRINT="$DIAGNOSTICS/imu-dashboard.rbl"
  EXTRA_LAYERS+=("$DIAGNOSTICS/diagnostics.rrd")
fi
ASSETS=("$DATASET"/assets/*.rrd)
for path in "$DATASET/blueprints/specialist.rbl" \
  "$DATASET/base/$SESSION.rrd" "$DATASET/estimate/$SESSION.rrd" \
  "$DATASET/viz/$SESSION.rrd" "${ASSETS[@]}"; do
  if [[ ! -f "$path" ]]; then
    printf 'Missing recording or asset: %s\nRun bash scripts/process_014.sh first.\n' "$path" >&2
    exit 1
  fi
done
ASSET_ARGS=()
for asset in "${ASSETS[@]}"; do
  ASSET_ARGS+=(--asset "$asset")
done
NEW_ARGS=(--new)
for option in "$@"; do
  if [[ "$option" == --new || "$option" == --port || "$option" == --port=* ]]; then
    NEW_ARGS=()
  fi
done
exec mamba run -n specialist-umi rerun \
  "${NEW_ARGS[@]}" \
  "$BLUEPRINT" \
  "$DATASET/base/$SESSION.rrd" \
  "$DATASET/estimate/$SESSION.rrd" \
  "$DATASET/viz/$SESSION.rrd" "${EXTRA_LAYERS[@]}" "${ASSET_ARGS[@]}" "$@"
