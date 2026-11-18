#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
ENV_NAME="${SPECIALIST_ENV_NAME:-specialist-umi}"
SESSION="20261116_192018_IAQEB2604VWWN7"
DATASET="${SPECIALIST_DATASET:-logs/real-capture-20261116}"
PROCESSED="${SPECIALIST_014_OUTPUT:-logs/014-processing}"
OUTPUT="${SPECIALIST_PRETRAIN_OUTPUT:-logs/pretrain-20261117}"
PROTOCOL="${SPECIALIST_CAPTURE_PROTOCOL:-configs/capture_protocol.example.json}"
DESCRIPTION="${SPECIALIST_DESCRIPTION_ROOT:-$(dirname -- "$PROJECT_DIR")/xense-description}"
GENERATOR="${SPECIALIST_FORWARD_GENERATOR:-$(dirname -- "$PROJECT_DIR")/xense-description-sim/tools/build_flexiv_model.py}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
mkdir -p "$OUTPUT"
bash scripts/process_014.sh > "$OUTPUT/014-export.log" 2>&1
mamba run -n "$ENV_NAME" python scripts/export_calibration.py \
  "$DATASET/raw/IAQEB2604VWWN7/VID_20261116_192018_00_014.insv" \
  --base "$DATASET/base/$SESSION.rrd" --output "$OUTPUT/factory-calibration.json" \
  > "$OUTPUT/calibration-export.log" 2>&1
mamba run -n "$ENV_NAME" python scripts/prepare_capture.py \
  --state "$PROCESSED/window50/state_s8_r100/$SESSION.rrd" --poses-dir "$PROCESSED/poses" \
  --protocol "$PROTOCOL" --output-dir "$OUTPUT/episode014" > "$OUTPUT/prepare.log" 2>&1
mamba run -n "$ENV_NAME" python scripts/replay_capture_sim.py \
  --base "$DATASET/base/$SESSION.rrd" --estimate "$DATASET/estimate/$SESSION.rrd" \
  --state "$PROCESSED/window50/state_s8_r100/$SESSION.rrd" \
  --output-dir "$OUTPUT/specialist-sim" > "$OUTPUT/replay.log" 2>&1
mamba run -n "$ENV_NAME" python scripts/check_forward01.py \
  --poses "$PROCESSED/poses/right-poses.npz" --source-root "$DESCRIPTION" \
  --generator "$GENERATOR" --output-dir "$OUTPUT/forward01" > "$OUTPUT/forward01.log" 2>&1
mamba run -n "$ENV_NAME" python scripts/plot_pretrain_results.py --root "$OUTPUT" \
  > "$OUTPUT/plots.log" 2>&1
printf '训练前的软件检查与预览已生成：%s\n01结果包含失败和待标定项，请阅读forward01-result.json；这不是实机执行许可。\n' "$OUTPUT"
