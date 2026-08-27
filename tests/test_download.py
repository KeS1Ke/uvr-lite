"""票 4（tdd）：模型下载的进度回调、取消续传与多源回退（本地 HTTP 服务）。"""

import hashlib
import http.server
import os
import re
import threading

import pytest

import uvr_lite.download as dl

DATA = bytes(range(256)) * ((4 << 20) // 256)  # 4 MiB（>1MB 读缓冲，保证多次回调）


class _Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
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


def test_download_resume_progress_within_total(http_server, tmp_path):
    """续传场景进度回调不得双计 .part 已有字节（回归：曾报 2×existing+done 超 total）。"""
    dest = tmp_path / "m.ckpt"
    part = dest.with_suffix(".ckpt.part")
    part.write_bytes(DATA[: len(DATA) // 2])
    calls = []

    def cb(done, total):
        calls.append((done, total))
        return True

    dl._download([f"{http_server}/model.bin"], dest, progress_callback=cb)
    assert dest.read_bytes() == DATA
    assert calls, "应至少回调一次"
    assert max(d for d, _ in calls) <= len(DATA), "进度不得超过 total"
    assert calls[-1] == (len(DATA), len(DATA))


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


# ---------- 模块导入（tqdm 静默降级） ----------

def test_download_imports_without_tqdm(monkeypatch):
    """无 tqdm 环境（安装链绿色 Python 场景）应可导入 download 并降级为 None。

    回归：曾顶部无条件 `from tqdm.auto import tqdm`，下面的 try/except 降级
    是死代码——缺 tqdm 时模块导入直接 ImportError。
    """
    import importlib
    import sys

    import uvr_lite.download as dl

    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    dl = importlib.reload(dl)
    assert dl.tqdm is None, "缺 tqdm 时应静默降级为 None，而不是 ImportError"
    monkeypatch.undo()  # 恢复 sys.modules 后重载回真实状态，避免污染其他测试
    dl = importlib.reload(dl)
    assert dl.tqdm is not None


# ---------- 校验缓存（.verified 标记） ----------

def test_ensure_model_marks_verified(fake_download_env):
    """首次校验通过后生成标记。"""
    (fake_download_env / "test_model.ckpt").write_bytes(DATA)
    dl.ensure_model("test_model")
    assert dl._verified_marker(fake_download_env / "test_model.ckpt").exists()


def test_ensure_model_verified_marker_skips_hash(fake_download_env, monkeypatch):
    """标记命中（size+mtime 未变）→ 跳过全量 SHA256（640MB 读盘成本）。"""
    ckpt = fake_download_env / "test_model.ckpt"
    ckpt.write_bytes(DATA)
    dl.ensure_model("test_model")  # 首次：全量校验 + 写标记
    hashed = {"n": 0}
    real_sha = dl.sha256_of

    def counting_sha(path):
        hashed["n"] += 1
        return real_sha(path)

    monkeypatch.setattr(dl, "sha256_of", counting_sha)
    dl.ensure_model("test_model")
    assert hashed["n"] == 0, "标记命中后不应再读文件全量哈希"


def test_ensure_model_changed_file_revalidates(fake_download_env):
    """文件被替换（size/mtime 变化）→ 标记失效，重新校验并下载。"""
    ckpt = fake_download_env / "test_model.ckpt"
    ckpt.write_bytes(DATA)
    dl.ensure_model("test_model")
    ckpt.write_bytes(b"corrupt")
    dl.ensure_model("test_model")
    assert ckpt.read_bytes() == DATA, "内容变更应触发重新下载"


# ---------- 并行分段下载（Range server） ----------


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    """支持 Range 的测试服务器（与阿里云等镜像行为一致：206 + Content-Range）。"""

    def log_message(self, *args):
        pass

    def _serve(self, head_only: bool = False) -> None:
        size = len(DATA)
        rng = self.headers.get("Range")
        status, body, extra = 200, DATA, {}
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng)
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

    def do_GET(self):
        self._serve()

    def do_HEAD(self):
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

def test_repo_root_dev_layout():
    """dev 布局：{repo}/uvr_lite/download.py → 根 = {repo}（含 pyproject/msst）。

    回归：曾用固定 parents[2]，dev 下多跳一层解析到仓库上层，导致
    models_dir 错位触发误下载（安装布局正确、dev 错误的双布局问题）。
    """
    from uvr_lite import download

    root = download.repo_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "uvr_lite").is_dir()
    assert (root / "msst").is_dir()


def test_repo_root_install_layout(monkeypatch, tmp_path):
    """安装布局：{inst}/app/uvr_lite/__init__.py → 根 = {inst}。

    app/ 快照层结构与 dev 仓库根几乎相同（都含 uvr_lite/msst/pyproject），
    以父层含 models/ + torch_cpu/ 区分（install.iss 两个变体均安装这两目录）。
    """
    import uvr_lite
    from uvr_lite import download

    inst = tmp_path / "inst"
    (inst / "app" / "uvr_lite").mkdir(parents=True)
    (inst / "app" / "msst").mkdir(parents=True)
    (inst / "models").mkdir(parents=True)
    (inst / "torch_cpu").mkdir(parents=True)
    (inst / "app" / "pyproject.toml").write_text("")
    monkeypatch.setattr(uvr_lite, "__file__", str(inst / "app" / "uvr_lite" / "__init__.py"))

    assert download.repo_root() == inst


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
