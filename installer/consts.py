# coding: utf-8
"""安装器固定版本与镜像源常量（票 6 执行链用）。

- 绿色 Python：python-build-standalone 固定版本 + 官方 SHA256（安装包免预装）
- torch：固定版本，CPU/CUDA 分流 + 清华 pytorch-wheels 镜像回退
- pip：默认清华 PyPI 加速，失败自动回退官方源
"""

# python-build-standalone（install_only：解压即用，约 50MB）
# 版本与 digest 来自官方 release assets 的 .sha256：
#   https://github.com/astral-sh/python-build-standalone/releases/tag/20260728
GREEN_PY_VERSION = "3.12.13"
GREEN_PY_TAG = "20260728"
GREEN_PY_FILENAME = (
    f"cpython-{GREEN_PY_VERSION}+{GREEN_PY_TAG}-"
    "x86_64-pc-windows-msvc-install_only.tar.gz"
)
GREEN_PY_SHA256 = "8a0e1ded37e11f4c72b9671bf134bb478b1b2d55efe53a3d6e589b166f1bf2e1"
GREEN_PY_URLS = [
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
    # ghproxy 镜像（GitHub 主源慢/失败时回退）
    "https://mirror.ghproxy.com/https://github.com/astral-sh/"
    f"python-build-standalone/releases/download/{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
]

# torch：cu128（RTX 30/40/50 系均可）与 CPU 两种构建，均带清华镜像回退
TORCH_VERSION = "2.7.1"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_MIRROR_CUDA = "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128"
TORCH_MIRROR_CPU = "https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu"

# 常规依赖 pip 源
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

MIN_PYTHON = (3, 10)
