# coding: utf-8
"""Python 环境检测/下载/venv 相关测试。"""

import subprocess

from installer import python_env as pe


def test_parse_version():
    assert pe.parse_version("Python 3.12.9") == (3, 12, 9)
    assert pe.parse_version("Python 3.10.0") == (3, 10, 0)
    assert pe.parse_version(" 3.13.14 \n") == (3, 13, 14)
    assert pe.parse_version("garbage") is None
    assert pe.parse_version("") is None


def test_python_version(monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("x")
    monkeypatch.setattr(
        pe.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="3.12.9\n", stderr=""))
    assert pe.python_version(fake) == (3, 12, 9)


def test_find_system_python(monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("x")
    monkeypatch.setattr(pe.shutil, "which",
                        lambda name: str(fake) if name in ("python", "py") else None)
    monkeypatch.setattr(
        pe.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="3.13.14\n", stderr=""))
    assert pe.find_system_python() == fake


def test_find_system_python_too_old(monkeypatch, tmp_path):
    fake = tmp_path / "python.exe"
    fake.write_text("x")
    monkeypatch.setattr(pe.shutil, "which", lambda name: str(fake) if name == "python" else None)
    monkeypatch.setattr(
        pe.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="3.9.7\n", stderr=""))
    assert pe.find_system_python() is None


def test_find_system_python_none_found(monkeypatch):
    monkeypatch.setattr(pe.shutil, "which", lambda name: None)
    assert pe.find_system_python() is None


def test_download_green_python_sha256_mismatch(monkeypatch, tmp_path):
    """下载后 SHA256 与固定值不符 → 删除并报错。"""
    calls = {}

    def fake_download(urls, dest, progress_callback=None):
        calls["urls"] = urls
        dest.write_bytes(b"fake-bytes")

    monkeypatch.setattr(pe.download, "_download", fake_download)
    monkeypatch.setattr(pe.download, "sha256_of", lambda p: "0" * 64)
    import pytest

    with pytest.raises(RuntimeError, match="校验失败"):
        pe.download_green_python(tmp_path)
    assert not (tmp_path / pe.GREEN_PY_FILENAME).exists()


def test_download_green_python_ok(monkeypatch, tmp_path):
    """校验通过 → 解压出 python/python.exe，返回其路径。"""
    import io
    import tarfile

    # 构造一个安装包形态的 tar.gz：顶层 python/ 目录
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        data = b"fake-exe"
        info = tarfile.TarInfo("python/python.exe")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))

    def fake_download(urls, dest, progress_callback=None):
        dest.write_bytes(buf.getvalue())

    monkeypatch.setattr(pe.download, "_download", fake_download)
    monkeypatch.setattr(pe.download, "sha256_of", lambda p: pe.GREEN_PY_SHA256)
    exe = pe.download_green_python(tmp_path)
    assert exe == tmp_path / "python" / "python.exe"
    assert exe.read_bytes() == b"fake-exe"


def test_ensure_venv_creates(monkeypatch, tmp_path):
    python_exe = tmp_path / "python.exe"
    python_exe.write_text("x")
    venv = tmp_path / ".venv"
    ran = []

    def fake_run(cmd, **kw):
        ran.append(cmd)
        (venv / "Scripts").mkdir(parents=True)
        (venv / "Scripts" / "python.exe").write_text("x")

    monkeypatch.setattr(pe, "_run", fake_run)
    result = pe.ensure_venv(python_exe, venv)
    assert result == venv / "Scripts" / "python.exe"
    assert ran and ran[0][:3] == [str(python_exe), "-m", "venv"]


def test_ensure_venv_existing(tmp_path):
    venv = tmp_path / ".venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_text("x")
    assert pe.ensure_venv(tmp_path / "nope.exe", venv) == venv / "Scripts" / "python.exe"
