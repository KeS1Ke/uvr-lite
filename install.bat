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
REM pyproject 声明 requires-python >= 3.10：先比版本再建 venv，否则低版本会先建
REM 环境、装完 torch 才在 pip/运行时失败，用户等半天只看到一句无从下手的报错。
REM 比较交给解释器自己做（bat 里 number 比较受区域设置影响，不用）。
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python %PYVER% is too old. uvr-lite requires Python 3.10 or newer.
    pause & exit /b 1
)

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
REM 必须带 [ui] extras：GUI 依赖 PySide6-Essentials 只挂在 optional extra 上，
REM 裸 `pip install -e .` 装出来的环境跑 `uvr-lite ui` 会 ImportError。
echo [4/5] Installing uvr-lite and dependencies (including GUI extras) ...
pip install -e ".[ui]" --quiet
if errorlevel 1 (
    echo [WARN] GUI extras installation failed, retrying without them ...
    pip install -e . --quiet
)
if errorlevel 1 ( echo [ERROR] dependency installation failed & pause & exit /b 1 )

REM ---- 5. download model (SHA256 verified) ----
echo [5/5] Downloading model weights (~320 MB, SHA256 verified) ...
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
echo Before each use, activate the virtual environment:
echo   PowerShell: .venv\Scripts\Activate.ps1
echo   cmd:        .venv\Scripts\activate.bat
echo.
echo Usage:
echo   uvr-lite separate song.mp3 -o output
echo   uvr-lite separate a.mp3 b.flac -o out  ^(multiple files^)
echo   uvr-lite models        list models / status
echo   uvr-lite download      re-download models
echo ===============================================
pause
