#!/usr/bin/env bash
# uvr-lite one-click installer (Linux / macOS)
# Usage: bash install.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================"
echo "  uvr-lite one-click installer"
echo "============================================"

# ---- 1. locate Python ----
PY="python3"
command -v "$PY" >/dev/null 2>&1 || { echo "[ERROR] Python 3 not found. Install Python 3.10+ first."; exit 1; }
echo "[1/5] Using $($PY --version)"
# pyproject 声明 requires-python >= 3.10：先比版本再建 venv，否则低版本会先建
# 环境、装完 torch 才在 pip/运行时失败，用户等半天只看到一句无从下手的报错。
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "[ERROR] $("$PY" --version 2>&1) is too old. uvr-lite requires Python 3.10 or newer."
    exit 1
fi

# ---- 2. create venv ----
if [ ! -d .venv ]; then
    echo "[2/5] Creating virtual environment .venv ..."
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- 3. install torch (auto-detect GPU) ----
echo "[3/5] Installing PyTorch (auto-detect GPU) ..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "      - NVIDIA GPU detected, installing CUDA build (cu128)"
    pip install --quiet "torch>=2.1" --index-url https://download.pytorch.org/whl/cu128 \
        || pip install --quiet "torch>=2.1"
else
    echo "      - No NVIDIA GPU detected, installing CPU build (slower separation)"
    pip install --quiet "torch>=2.1"
fi

# ---- 4. install package + deps ----
# 必须带 [ui] extras：GUI 依赖 PySide6-Essentials 只挂在 optional extra 上，
# 裸 `pip install -e .` 装出来的环境跑 `uvr-lite ui` 会 ImportError。
# 缺系统 Qt 库（如 libGL）时 PySide6 可能装不上，此时退回裸装保 CLI 可用。
echo "[4/5] Installing uvr-lite and dependencies (including GUI extras) ..."
pip install --quiet -e ".[ui]" \
    || { echo "      [WARN] GUI extras failed (missing system Qt libs?), continuing without GUI" >&2
         pip install --quiet -e .; }

# ---- 5. download model (SHA256 verified) ----
echo "[5/5] Downloading model weights (~320 MB, SHA256 verified) ..."
uvr-lite download

# ---- smoke test (GPU only; CPU too slow) ----
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "      Smoke test: generating 3s test tone and separating ..."
    python - <<'EOF'
import numpy as np, soundfile as sf
t = np.linspace(0, 3, 3 * 44100)
sf.write("_smoke.wav", np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1), 44100)
EOF
    uvr-lite separate _smoke.wav -o _smoke_out --format wav
    rm -rf _smoke.wav _smoke_out
    echo "      Smoke test passed!"
else
    echo "[INFO] CPU environment: smoke test skipped"
fi

echo
echo "============ INSTALLATION COMPLETE ============"
echo "Before each use, activate the virtual environment:"
echo "  source .venv/bin/activate"
echo
echo "Usage:"
echo "  uvr-lite separate song.mp3 -o output"
echo "  uvr-lite models        list models / status"
echo "  uvr-lite download      re-download models"
echo "==============================================="
