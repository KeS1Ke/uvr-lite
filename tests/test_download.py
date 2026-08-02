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


# ---------- 并行分段下载（Range server） ----------

import re as _re


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """支持 Range 的测试服务器（与阿里云等镜像行为一致：206 + Content-Range）。"""

    def log_message(self, *args):  # noqa: N802
        pass

    def _serve(self, head_only: bool = False) -> None:
        size = len(DATA)
        rng = self.headers.get("Range")
        status, body, extra = 200, DATA, {}
        if rng:
            m = _re.fullmatch(r"bytes=(\d+)-(\d*)", rng)
            if m:
                start, end = int(m.group(1)), int(m.group(2) or size - 1)
                body = DATA[start:end + 1]
                status = 206
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        extra["Accept-Ranges"] = "bytes"
        extra["Content-Length"] = str(len(body))
        self.send_response(status)
        for k, v in extra.items():
            self.send_header(k, v)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._serve()

    def do_HEAD(self):  # noqa: N802
        self._serve(head_only=True)


@pytest.fixture
def range_server(tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_parallel_download_matches_single(range_server, tmp_path):
    """并行下载结果与源数据完全一致（分段正确、合并正确）。"""
    dest = tmp_path / "big.ckpt"
    calls = []

    def cb(done, total):
        calls.append((done, total))
        return True

    dl._download([f"{range_server}/big.bin"], dest, progress_callback=cb)
    assert dest.read_bytes() == DATA
    assert calls[-1] == (len(DATA), len(DATA))
    # 无残留段文件
    assert not list(tmp_path.glob("*.part*"))


def test_parallel_download_cancel_keeps_segments(range_server, tmp_path):
    dest = tmp_path / "big.ckpt"
    cancelled = {"n": 0}

    def cb(done, total):
        cancelled["n"] += 1
        return done < total * 0.6

    with pytest.raises(InterruptedError):
        dl._download([f"{range_server}/big.bin"], dest, progress_callback=cb)
    parts = list(tmp_path.glob("*.part*"))
    assert parts, "取消后应保留段文件供续传"
    assert sum(p.stat().st_size for p in parts) < len(DATA)


def test_parallel_download_resume_after_cancel(range_server, tmp_path):
    """取消后重下：已下载段跳过（续传），最终完整。"""
    dest = tmp_path / "big.ckpt"

    def cb(done, total):
        return done < total * 0.6

    with pytest.raises(InterruptedError):
        dl._download([f"{range_server}/big.bin"], dest, progress_callback=cb)
    dl._download([f"{range_server}/big.bin"], dest)
    assert dest.read_bytes() == DATA


def test_parallel_fallback_to_single_when_no_range(http_server, tmp_path):
    """源不支持 Range（SimpleHTTPRequestHandler 返回 200）→ 单连接路径。"""
    dest = tmp_path / "m.ckpt"
    dl._download([f"{http_server}/model.bin"], dest)
    assert dest.read_bytes() == DATA


# ---------- UVR_MODEL_DIR 环境变量（安装场景：模型目录指向安装目录） ----------

def test_models_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("UVR_MODEL_DIR", str(tmp_path / "custom"))
    from uvr_lite import download
    monkeypatch.setattr(download, "repo_root", lambda: tmp_path / "repo")
    d = download.models_dir()
    assert d == tmp_path / "custom"
    assert d.exists()


def test_models_dir_default_no_env(monkeypatch, tmp_path):
    monkeypatch.delenv("UVR_MODEL_DIR", raising=False)
    from uvr_lite import download
    monkeypatch.setattr(download, "repo_root", lambda: tmp_path / "repo")
    assert download.models_dir() == tmp_path / "repo" / "models"
