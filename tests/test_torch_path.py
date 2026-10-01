"""安装场景下 torch 目录进 sys.path 的顺序。

CUDA 目录只有 wheel 解压出的 torch，sympy 等依赖在 torch_cpu/。
选用 CUDA 时两条路径都要在，且 torch_cuda 必须更靠前，否则 import torch
会命中 CPU 构建。
"""

import os
import sys
from pathlib import Path

import pytest

import uvr_lite


def _layout(tmp_path: Path) -> Path:
    base = tmp_path / "inst"
    (base / "torch_cuda" / "torch").mkdir(parents=True)
    (base / "torch_cpu" / "torch").mkdir(parents=True)
    (base / "torch_cpu" / "sympy").mkdir()
    return base


@pytest.fixture
def install_root(tmp_path, monkeypatch):
    base = _layout(tmp_path)
    monkeypatch.setattr(uvr_lite, "_base_dir", lambda: base)
    saved_path = list(sys.path)
    saved_managed = list(uvr_lite._managed_paths)
    saved_dir = uvr_lite._torch_dir_
    had_env = "UVR_TORCH" in os.environ
    old_env = os.environ.get("UVR_TORCH")
    yield base
    sys.path[:] = saved_path
    uvr_lite._managed_paths = saved_managed
    uvr_lite._torch_dir_ = saved_dir
    if had_env:
        os.environ["UVR_TORCH"] = old_env
    else:
        os.environ.pop("UVR_TORCH", None)


def test_cuda_keeps_cpu_deps_behind_cuda_torch(install_root):
    uvr_lite.set_torch_mode("cuda")
    cuda = str(install_root / "torch_cuda")
    cpu = str(install_root / "torch_cpu")
    assert sys.path[0] == cuda
    assert sys.path[1] == cpu


def test_cpu_mode_does_not_expose_cuda_torch(install_root):
    uvr_lite.set_torch_mode("cpu")
    assert sys.path[0] == str(install_root / "torch_cpu")
    assert str(install_root / "torch_cuda") not in sys.path


def test_switch_cuda_to_cpu_drops_cuda_prefix(install_root):
    uvr_lite.set_torch_mode("cuda")
    uvr_lite.set_torch_mode("cpu")
    assert sys.path[0] == str(install_root / "torch_cpu")
    assert str(install_root / "torch_cuda") not in sys.path
    assert sys.path.count(str(install_root / "torch_cpu")) == 1


def test_cuda_without_cpu_dir_still_inserts_cuda(tmp_path, monkeypatch):
    base = tmp_path / "inst"
    (base / "torch_cuda" / "torch").mkdir(parents=True)
    monkeypatch.setattr(uvr_lite, "_base_dir", lambda: base)
    saved_path = list(sys.path)
    saved_managed = list(uvr_lite._managed_paths)
    saved_dir = uvr_lite._torch_dir_
    had_env = "UVR_TORCH" in os.environ
    old_env = os.environ.get("UVR_TORCH")
    try:
        uvr_lite.set_torch_mode("cuda")
        assert sys.path[0] == str(base / "torch_cuda")
        assert str(base / "torch_cpu") not in sys.path
        assert not any("torch_cpu" in Path(entry).parts for entry in sys.path)
    finally:
        sys.path[:] = saved_path
        uvr_lite._managed_paths = saved_managed
        uvr_lite._torch_dir_ = saved_dir
        if had_env:
            os.environ["UVR_TORCH"] = old_env
        else:
            os.environ.pop("UVR_TORCH", None)
