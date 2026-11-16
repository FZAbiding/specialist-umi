#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES_DIR="${SPECIALIST_SOURCES_DIR:-$(dirname -- "$PROJECT_DIR")}"
ENV_NAME="${SPECIALIST_ENV_NAME:-specialist-umi}"
cd -- "$PROJECT_DIR"
command -v mamba >/dev/null
command -v uv >/dev/null
if [[ -n "${SPECIALIST_PROXY:-}" ]]; then
  export HTTPS_PROXY="$SPECIALIST_PROXY" HTTP_PROXY="$SPECIALIST_PROXY"
  export NO_PROXY="${SPECIALIST_NO_PROXY:-${NO_PROXY:-files.pythonhosted.org,pypi.org,127.0.0.1,localhost}}"
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
if ! mamba run -n "$ENV_NAME" python -c 'import sys; assert sys.version_info[:2] == (3, 14)' >/dev/null 2>&1; then
  mamba env create -n "$ENV_NAME" -f environment.yml -y
fi
PREFIX="$(mamba run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
mkdir -p logs/setup
for name in specialist_description specialist_sim feetech_driver specialist_sdk insta360_driver; do
  PIN="$("$PREFIX/bin/python" -c 'import json,sys; print(json.load(open("configs/sources.lock.json"))[sys.argv[1]]["commit"])' "$name")"
  if [[ ! -e "$SOURCES_DIR/$name/.git" ]]; then
    git clone "https://github.com/specialist-ai/$name.git" "$SOURCES_DIR/$name"
    if ! git -C "$SOURCES_DIR/$name" cat-file -e "$PIN^{commit}" 2>/dev/null; then
      git -C "$SOURCES_DIR/$name" fetch origin "$PIN"
    fi
    git -C "$SOURCES_DIR/$name" checkout --detach "$PIN"
  elif [[ "$(git -C "$SOURCES_DIR/$name" rev-parse HEAD)" != "$PIN" ]]; then
    printf 'Source revision differs: %s (expected %s).\nUse a separate SPECIALIST_SOURCES_DIR for the pinned installation.\n' "$SOURCES_DIR/$name" "$PIN" >&2
    exit 1
  fi
done
if ! mamba run -n "$ENV_NAME" python -c 'import matplotlib' >/dev/null 2>&1; then
  mamba install -n "$ENV_NAME" -c conda-forge 'matplotlib>=3.10' -y
fi
uv export --frozen --no-hashes --no-emit-project --no-emit-package specialist-description \
  --output-file logs/setup/core.requirements.txt >/dev/null
WHEEL_OPTIONS=()
if [[ -d logs/setup/wheels ]]; then
  WHEEL_OPTIONS=(--find-links "$PROJECT_DIR/logs/setup/wheels")
fi
uv pip install --python "$PREFIX/bin/python" --no-sources \
  --index https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match \
  "${WHEEL_OPTIONS[@]}" \
  -r logs/setup/core.requirements.txt -e "$SOURCES_DIR/specialist_description" -e .
mamba run -n "$ENV_NAME" env \
  CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" \
  PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
  LDFLAGS="-L$PREFIX/lib -Wl,-rpath,$PREFIX/lib ${LDFLAGS:-}" \
  uv pip install --python "$PREFIX/bin/python" --no-sources \
  -e "$SOURCES_DIR/specialist_sim[dataset,handheld]" 'mujoco==3.17.0' 'warp-lang==1.17.0' \
  -e "$SOURCES_DIR/feetech_driver" -e "$SOURCES_DIR/specialist_sdk" \
  -e "$SOURCES_DIR/insta360_driver"
mamba run -n "$ENV_NAME" python -m pip check
printf '\n环境就绪：mamba activate %s\n' "$ENV_NAME"
