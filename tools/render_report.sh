#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$PROJECT_DIR"
RENDERER="${SPECIALIST_VE_RENDERER:-/home/xense/.agents/skills/visual-explainer/kit/render.mjs}"
if [[ -d logs/setup/ve-browser ]]; then
  export PLAYWRIGHT_BROWSERS_PATH="$PROJECT_DIR/logs/setup/ve-browser"
fi
node "$RENDERER" reports/specialist-official-demo.src.html
