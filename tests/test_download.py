# coding: utf-8
"""票 4（tdd）：模型下载的进度回调、取消续传与多源回退（本地 HTTP 服务）。"""

import hashlib
import http.server
import os
import threading
from pathlib import Path

import pytest

import uvr_lite.download as dl

DATA = bytes(range(256)) * ((4 << 20) // 256)  # 4 MiB（>1MB 读缓冲，保证多次回调）


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: N802 —— 静默访问日志
        pass


@pytest.fixture
def http_server(tmp_path):
    (tmp_path / "model.bin").write_bytes(DATA)
    cwd = os.getcwd()
    os.chdir(tmp_path)  # SimpleHTTPRequestHandler 以 cwd 为根
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    os.chdir(cwd)


def test_download_progress_monotonic(http_server, tmp_path):
    dest = tmp_path / "m.ckpt"
    calls = []

    def cb(done, total):
        calls.append((done, total))
        return True

    dl._download([f"{http_server}/model.bin"], dest, progress_callback=cb)
    assert dest.read_bytes() == DATA
    dones = [d for d, _ in calls]
    assert dones == sorted(dones) and len(dones) > 1
    assert calls[-1] == (len(DATA), len(DATA))


def test_download_cancel_keeps_part(http_server, tmp_path):
    dest = tmp_path / "m.ckpt"

    def cb(done, total):
        return done < total * 0.6  # 超过 60% 后取消

    with pytest.raises(InterruptedError):
        dl._download([f"{http_server}/model.bin"], dest, progress_callback=cb)
    part = dest.with_suffix(".ckpt.part")
    assert part.exists(), "取消后应保留 .part 供续传"
    assert 0 < part.stat().st_size < len(DATA)


def test_download_resume_after_cancel(http_server, tmp_path):
    """取消后再次下载应从 .part 续传并完整落盘。"""
    dest = tmp_path / "m.ckpt"

    def cb(done, total):
        return done < total * 0.6

    with pytest.raises(InterruptedError):
        dl._download([f"{http_server}/model.bin"], dest, progress_callback=cb)
    dl._download([f"{http_server}/model.bin"], dest)  # 续传（无回调）
    assert dest.read_bytes() == DATA


def test_download_source_fallback(http_server, tmp_path):
    """主源 404 时自动切换备用源。"""
    dest = tmp_path / "m.ckpt"
    dl._download([f"{http_server}/missing.bin", f"{http_server}/model.bin"], dest)
    assert dest.read_bytes() == DATA


def test_download_all_sources_fail(http_server, tmp_path):
    with pytest.raises(RuntimeError, match="所有下载源均失败"):
        dl._download([f"{http_server}/a.bin", f"{http_server}/b.bin"],
                     tmp_path / "m.ckpt")


def _fake_registry(http_server):
    sha = hashlib.sha256(DATA).hexdigest()
    return {
        "test_model": {
            "model_type": "bs_roformer",
            "config": "x.yaml",
            "ckpt_url": f"{http_server}/model.bin",
            "mirror_urls": [],
            "sha256": sha,
            "description": "测试模型",
        }
    }


@pytest.fixture
def fake_download_env(http_server, tmp_path, monkeypatch):
    # get_model_info 引用 models 模块的注册表；download 的 models_dir 指向临时目录
    monkeypatch.setattr("uvr_lite.models.MODEL_REGISTRY", _fake_registry(http_server))
    monkeypatch.setattr(dl, "models_dir", lambda: tmp_path)
    return tmp_path


def test_ensure_model_ready_skips_download(fake_download_env, capsys):
    (fake_download_env / "test_model.ckpt").write_bytes(DATA)
    p = dl.ensure_model("test_model")
    assert "模型已就绪" in capsys.readouterr().out
    assert p == fake_download_env / "test_model.ckpt"


def test_ensure_model_bad_sha_redownloads(fake_download_env):
    (fake_download_env / "test_model.ckpt").write_bytes(b"corrupt")
    p = dl.ensure_model("test_model")
    assert p.read_bytes() == DATA, "校验失败应重新下载"


def test_ensure_model_downloads_with_progress(fake_download_env):
    calls = []

    def cb(done, total):
        calls.append((done, total))
        return True

    dl.ensure_model("test_model", progress_callback=cb)
    assert (fake_download_env / "test_model.ckpt").read_bytes() == DATA
    assert calls[-1] == (len(DATA), len(DATA))
