# coding: utf-8
"""打包脚本的可单测部分：CPU torch wheel 的 SHA 校验与 URL 编码。

此前仅 CUDA wheel 有 SHA（install.iss 与 download.py 双向一致检查），CPU
wheel 下载后无校验直接 pip 安装——镜像被投毒时无防线。
"""

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # scripts/ 非包，从根目录按命名空间包导入

import scripts.build_installer as bi  # noqa: E402


@pytest.fixture
def fake_wheel_env(tmp_path, monkeypatch):
    """假 wheel 落盘 + 假 pip：聚焦 _install_torch 的校验与 URL 逻辑。"""
    from uvr_lite import download as dl

    content = b"fake torch cpu wheel " * 100
    seen = {}

    def fake_download(urls, dest, *a, **k):
        seen["urls"] = urls
        dest.write_bytes(content)

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(bi, "_pip", lambda *a, **k: None)
    return {"content": content, "sha": hashlib.sha256(content).hexdigest(),
            "dir": tmp_path, "seen": seen}


def test_install_torch_rejects_bad_sha(fake_wheel_env, monkeypatch):
    """SHA 不匹配 → 删除 wheel 并退出（投毒文件不得滞留进入 pip）。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="校验失败"):
        bi._install_torch(fake_wheel_env["dir"], "cpu", ["https://x/"])
    wheel = fake_wheel_env["dir"] / "torch-2.7.1+cpu-cp312-cp312-win_amd64.whl"
    assert not wheel.exists(), "校验失败后应删除 wheel"


def test_install_torch_sha_match_proceeds(fake_wheel_env, monkeypatch):
    """SHA 匹配 → 正常进入安装路径并返回 torch_cpu 目录。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", fake_wheel_env["sha"])
    dest = bi._install_torch(fake_wheel_env["dir"], "cpu", ["https://x/"])
    assert dest.name == "torch_cpu" and dest.exists()


def test_install_torch_urls_percent_encode_plus(fake_wheel_env, monkeypatch):
    """URL 中 "+" 必须 %2B 编码——官方源（S3/CloudFront）对字面 + 返回 403。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", fake_wheel_env["sha"])
    bi._install_torch(fake_wheel_env["dir"], "cpu",
                      ["https://mirror.example/cpu", "https://official.example/cpu"])
    urls = fake_wheel_env["seen"]["urls"]
    assert len(urls) == 2
    for u in urls:
        fname = u.rsplit("/", 1)[-1]
        assert fname == "torch-2.7.1%2Bcpu-cp312-cp312-win_amd64.whl"
