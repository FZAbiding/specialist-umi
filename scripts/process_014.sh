#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
ENV_NAME="${SPECIALIST_ENV_NAME:-specialist-umi}"
SESSION="20261116_192018_IAQEB2604VWWN7"
DATASET="${SPECIALIST_DATASET:-logs/real-capture-20261116}"
PROCESSED="${SPECIALIST_014_OUTPUT:-logs/014-processing}"
MASTER="$DATASET/raw/IAQEB2604VWWN7/VID_20261116_192018_00_014.insv"
BASE="$DATASET/base/$SESSION.rrd"
ESTIMATE="$DATASET/estimate/$SESSION.rrd"
WINDOW_BASE="$PROCESSED/window50/base/$SESSION.rrd"
STATE="$PROCESSED/window50/state_s8_r100/$SESSION.rrd"

[[ -f "$MASTER" ]] || { echo "Missing local 014 master: $MASTER" >&2; exit 1; }
[[ -f "${MASTER/_00_/_10_}" ]] || { echo "Missing paired 014 lens" >&2; exit 1; }
mkdir -p "$PROCESSED"
if [[ ! -f "$BASE" ]]; then
  mamba run -n "$ENV_NAME" python scripts/convert.py --input-insvs "$MASTER" --output-dir "$DATASET"
fi
if [[ ! -f "$ESTIMATE" ]]; then
  mamba run -n "$ENV_NAME" python scripts/estimate.py --base "$BASE" \
    --anchor.tag-ids 0 1 --anchor.tag-size 0.1 --gripper.tag-size 0.01 --output-dir "$DATASET"
fi
# Explicit, clip-specific curation: keep the first 50 s, including the end taps
# near 47.68 s. The original recording and full-span reconstruction are kept.
if [[ ! -f "$WINDOW_BASE" ]]; then
  mamba run -n "$ENV_NAME" python scripts/convert.py --input-insvs "$MASTER" \
    --duration 50 --output-dir "$PROCESSED/window50"
fi
if [[ ! -f "$STATE" ]]; then
  mamba run -n "$ENV_NAME" python scripts/state.py --base "$WINDOW_BASE" \
    --estimate "$ESTIMATE" --output-dir "$PROCESSED/window50"
fi
mamba run -n "$ENV_NAME" python scripts/inspect_state.py "$STATE" --output "$PROCESSED/state-inspection.json"
mamba run -n "$ENV_NAME" python scripts/export_poses.py "$STATE" --hand right \
  --reference-estimate "$ESTIMATE" --reference-time "${SPECIALIST_014_REFERENCE_TIME:-1.0}" \
  --chunk-length "${SPECIALIST_ACTION_CHUNK_LENGTH:-1}" --output-dir "$PROCESSED/poses"
