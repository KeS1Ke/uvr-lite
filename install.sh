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
echo "[4/5] Installing uvr-lite and dependencies ..."
pip install --quiet -e .

# ---- 5. download model (SHA256 verified) ----
echo "[5/5] Downloading model weights (~640 MB, SHA256 verified) ..."
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
echo "Usage:"
echo "  uvr-lite separate song.mp3 -o output"
echo "  uvr-lite models        list models / status"
echo "  uvr-lite download      re-download models"
echo "==============================================="
