# coding: utf-8
"""票 1（tdd）：CUDA 推理引擎下载安装（install_cuda_torch）。

用本地构造的假 wheel zip（与真实 wheel 同构：torch/ + dist-info/，含
.lib/include/bin）mock 掉 _download，验证：解压结构、裁剪、幂等、SHA 校验、
取消。真实 wheel 的 SHA256 常量（TORCH_CUDA_SHA256）在测试中被替换/保持。
"""

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

import uvr_lite.download as dl
from uvr_lite.download import cuda_torch_installed, install_cuda_torch


def make_wheel(path: Path) -> str:
    """构造与真实 wheel 同构的 zip，返回其 SHA256。"""
    files = {
        "torch/__init__.py": b"# fake torch\n",
        "torch/lib/torch.lib": b"L" * 100,             # 编译期导入库 → 应裁剪
        "torch/include/ATen/foo.h": b"h" * 10,          # 头文件目录 → 应裁剪
        "torch/bin/random.exe": b"E" * 10,              # 非 shm 可执行 → 应裁剪
        "torch/bin/torch_shm_manager.exe": b"S" * 10,   # 运行需要 → 保留
        "torch-2.7.1+cu128.dist-info/METADATA": b"Metadata\n",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_wheel(tmp_path, monkeypatch):
    """假 wheel 就位：SHA 常量替换为假值，_download 变为本地复制。"""
    wheel = tmp_path / "fake.whl"
    sha = make_wheel(wheel)
    cache = tmp_path / "cache"

    monkeypatch.setattr(dl, "TORCH_CUDA_SHA256", sha)
    monkeypatch.setattr(dl, "_wheel_cache_dir", lambda: cache)

    def fake_download(urls, dest, progress_callback=None, retries=2):
        dest.parent.mkdir(parents=True, exist_ok=True)  # 真实 _download 会建目录
        shutil.copy2(wheel, dest)

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(dl, "_REPORT_INTERVAL", 1)  # 小文件也触发进度回调
    return wheel


def test_install_extracts_structure_and_prunes(fake_wheel, tmp_path):
    dest = install_cuda_torch(tmp_path / "base")
    assert dest == tmp_path / "base" / "torch_cuda"
    # 解压结构（torch/ 与 dist-info/ 都在）
    assert (dest / "torch" / "__init__.py").exists()
    assert (dest / "torch-2.7.1+cu128.dist-info" / "METADATA").exists()
    # 裁剪：.lib / include / 非 shm 的 bin 全删，shm 保留
    assert not (dest / "torch" / "lib" / "torch.lib").exists()
    assert not (dest / "torch" / "include").exists()
    assert not (dest / "torch" / "bin" / "random.exe").exists()
    assert (dest / "torch" / "bin" / "torch_shm_manager.exe").exists()
    # wheel 中转文件已清理
    assert not (fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL).exists()


def test_install_idempotent_skips_download(fake_wheel, tmp_path, monkeypatch):
    install_cuda_torch(tmp_path / "base")
    calls = {"n": 0}

    def counting(urls, dest, progress_callback=None, retries=2):
        calls["n"] += 1
        shutil.copy2(fake_wheel, dest)

    monkeypatch.setattr(dl, "_download", counting)
    install_cuda_torch(tmp_path / "base")
    assert calls["n"] == 0, "已安装时不应再触发下载"
    assert cuda_torch_installed(tmp_path / "base")


def test_install_bad_sha_removes_wheel_and_raises(monkeypatch, tmp_path):
    """SHA 常量保持真实值（假 wheel 必然不匹配）→ 删缓存并报错。"""
    wheel = tmp_path / "fake.whl"
    make_wheel(wheel)
    cache = tmp_path / "cache"
    monkeypatch.setattr(dl, "_wheel_cache_dir", lambda: cache)

    def fake_download(urls, dest, progress_callback=None, retries=2):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, dest)

    monkeypatch.setattr(dl, "_download", fake_download)
    with pytest.raises(RuntimeError, match="SHA256 校验失败"):
        install_cuda_torch(tmp_path / "base")
    assert not (cache / dl.TORCH_CUDA_WHEEL).exists(), "损坏缓存应被删除"
    assert not cuda_torch_installed(tmp_path / "base")


def test_install_cancel_during_extract(fake_wheel, tmp_path):
    """解压阶段取消：抛 InterruptedError，wheel 缓存保留（下次直接解压）。"""
    calls = []

    def cb(done, total):
        calls.append((done, total))
        return done < total * 0.5  # 解压到一半取消

    with pytest.raises(InterruptedError):
        install_cuda_torch(tmp_path / "base", progress_callback=cb)
    assert (fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL).exists()
    # 半成品目录不完整（未全量解压）
    assert not (tmp_path / "base" / "torch_cuda" / "torch-2.7.1+cu128.dist-info" / "METADATA").exists()
    # 取消后重新安装：不再下载（wheel 已就绪），直接解压完成
    install_cuda_torch(tmp_path / "base")
    assert cuda_torch_installed(tmp_path / "base")


def test_cuda_torch_installed_detects_marker(fake_wheel, tmp_path):
    assert not cuda_torch_installed(tmp_path / "base")
    install_cuda_torch(tmp_path / "base")
    assert cuda_torch_installed(tmp_path / "base")
    (tmp_path / "base" / "torch_cuda" / "torch" / "__init__.py").unlink()
    assert not cuda_torch_installed(tmp_path / "base")
