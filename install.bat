@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   uvr-lite one-click installer (Windows)
echo ============================================

REM ---- 1. locate Python ----
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and check "Add to PATH".
    pause & exit /b 1
)
for /f "delims=" %%v in ('python -c "import sys; print(str(sys.version_info[0])+'.'+str(sys.version_info[1]))"') do set PYVER=%%v
echo [1/5] Using Python %PYVER%

REM ---- 2. create venv ----
if not exist .venv (
    echo [2/5] Creating virtual environment ".venv" ...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed & pause & exit /b 1 )
) else (
    echo [2/5] Virtual environment already exists
)
call .venv\Scripts\activate.bat

REM ---- 3. install torch (auto-detect GPU) ----
echo [3/5] Installing PyTorch (auto-detect GPU) ...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       - No NVIDIA GPU detected, installing CPU build (slower separation)
    pip install "torch>=2.1" --quiet
) else (
    echo       - NVIDIA GPU detected, installing CUDA build (cu128)
    pip install "torch>=2.1" --index-url https://download.pytorch.org/whl/cu128 --quiet
)
if errorlevel 1 (
    echo [WARN] torch install failed, retrying with CPU build ...
    pip install "torch>=2.1" --quiet
)
if errorlevel 1 ( echo [ERROR] torch installation failed & pause & exit /b 1 )

REM ---- 4. install package + deps ----
echo [4/5] Installing uvr-lite and dependencies ...
pip install -e . --quiet
if errorlevel 1 ( echo [ERROR] dependency installation failed & pause & exit /b 1 )

REM ---- 5. download model (SHA256 verified) ----
echo [5/5] Downloading model weights (~640 MB, SHA256 verified) ...
uvr-lite download
if errorlevel 1 ( echo [ERROR] model download failed & pause & exit /b 1 )

REM ---- smoke test (GPU only; CPU too slow) ----
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] CPU environment: smoke test skipped
) else (
    echo       Smoke test: generating 3s test tone and separating ...
    python -c "import numpy as np, soundfile as sf; t=np.linspace(0,3,3*44100); sf.write('_smoke.wav', np.stack([0.5*np.sin(2*np.pi*440*t)]*2, axis=1), 44100)"
    uvr-lite separate _smoke.wav -o _smoke_out --format wav
    if not exist "_smoke_out\_smoke-vocals.wav" ( echo [ERROR] smoke test failed & pause & exit /b 1 )
    del _smoke.wav 2>nul
    rmdir /s /q _smoke_out 2>nul
    echo       Smoke test passed!
)

echo.
echo ============ INSTALLATION COMPLETE ============
echo Usage:
echo   uvr-lite separate song.mp3 -o output
echo   uvr-lite separate a.mp3 b.flac -o out  ^(multiple files^)
echo   uvr-lite models        list models / status
echo   uvr-lite download      re-download models
echo ===============================================
pause
