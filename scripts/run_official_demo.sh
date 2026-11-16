#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
ENV_NAME="${SPECIALIST_ENV_NAME:-specialist-umi}"
OUT="${SPECIALIST_DEMO_DIR:-logs/official-demo}"
mamba run -n "$ENV_NAME" env MUJOCO_GL=egl python scripts/demo_official_sim.py --output-dir "$OUT"
mamba run -n "$ENV_NAME" sim-state "$OUT/episodes/pick-place-demo.npz" \
  --output-dir "$OUT/dataset" --hands both --image-size 360 --supersample 2
mamba run -n "$ENV_NAME" env MUJOCO_GL=egl sim-render "$OUT/episodes/pick-place-demo.npz" \
  --out "$OUT/pick-place-demo.mp4"
mamba run -n "$ENV_NAME" python scripts/inspect_state.py \
  "$OUT/dataset/state_s8_r100/pick-place-demo.rrd" --output "$OUT/state-inspection.json"
