# coding: utf-8
"""安装步骤函数测试（monkeypatch 外部副作用）。"""

import shutil
import subprocess
from pathlib import Path

from installer import steps
from installer.steps import StepContext


def make_ctx(tmp_path, upgrade=False, **kw):
    kw.setdefault("_cancel", lambda: False)
    kw.setdefault("_message", lambda s: None)
    kw.setdefault("_percent", lambda p: None)
    return StepContext(tmp_path, upgrade, **kw)


# ---------- step_prepare_python ----------

def test_prepare_reuses_existing_venv(tmp_path):
    venv = tmp_path / ".venv" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_text("x")
    ctx = make_ctx(tmp_path)
    steps.step_prepare_python(ctx)
    assert ctx.python_exe == ctx.venv_python


def test_prepare_reuses_system_python(tmp_path, monkeypatch):
    fake = tmp_path / "sys-python.exe"
    fake.write_text("x")
    monkeypatch.setattr(steps.python_env, "find_system_python", lambda: fake)
    monkeypatch.setattr(steps.python_env, "python_version", lambda p: (3, 13, 14))
    ctx = make_ctx(tmp_path)
    msgs = []
    ctx._message = msgs.append
    steps.step_prepare_python(ctx)
    assert ctx.python_exe == fake
    assert any("系统 Python 3.13.14" in m for m in msgs)


def test_prepare_downloads_green_python(tmp_path, monkeypatch):
    green = tmp_path / "python" / "python.exe"
    monkeypatch.setattr(steps.python_env, "find_system_python", lambda: None)
    monkeypatch.setattr(steps.python_env, "download_green_python",
                        lambda d, progress_callback=None, cancel=None: green)
    ctx = make_ctx(tmp_path)
    percents = []
    ctx._percent = percents.append
    steps.step_prepare_python(ctx)
    assert ctx.python_exe == green
    assert percents[-1] == 100


# ---------- step_create_venv ----------

def test_create_venv(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    ctx.python_exe = tmp_path / "python.exe"
    ran = []

    def fake_ensure_venv(py, venv, log_file=None, cancel=None):
        ran.append((py, venv))
        return venv / "Scripts" / "python.exe"

    monkeypatch.setattr(steps.python_env, "ensure_venv", fake_ensure_venv)
    steps.step_create_venv(ctx)
    assert ran == [(ctx.python_exe, ctx.venv_dir)]


def test_create_venv_skips_existing(tmp_path, monkeypatch):
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    (tmp_path / ".venv" / "Scripts" / "python.exe").write_text("x")
    ctx = make_ctx(tmp_path)
    ctx.python_exe = tmp_path / "python.exe"
    monkeypatch.setattr(steps.python_env, "ensure_venv", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应创建")))
    steps.step_create_venv(ctx)  # 不抛错即通过


# ---------- step_install_deps ----------

def test_install_deps_gpu_torch_then_pip(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    monkeypatch.setattr(steps, "_venv_has", lambda pip, pkg: False)
    monkeypatch.setattr(shutil, "which", lambda name: "C:/nvidia-smi.exe" if name == "nvidia-smi" else None)
    pip_cmds = []

    def fake_pip_install(ctx_, specs, indexes):
        pip_cmds.append((specs, indexes))

    monkeypatch.setattr(steps, "_pip_install_with_fallback", fake_pip_install)
    monkeypatch.setattr(steps.copy_app, "copy_app_source", lambda src, dest: 1)
    steps.step_install_deps(ctx)
    assert pip_cmds[0] == ([f"torch=={steps.TORCH_VERSION}"], [steps.TORCH_CUDA_INDEX, steps.TORCH_MIRROR_CUDA])
    assert pip_cmds[1][0] == [f"{ctx.app_dir}[ui]"]
    assert pip_cmds[1][1][0] == steps.PIP_INDEX


def test_install_deps_cpu_no_torch_skip(tmp_path, monkeypatch):
    """torch 已装 → 跳过 torch 安装，但 app[ui] 依赖照常安装。"""
    ctx = make_ctx(tmp_path)
    pip_specs = []
    monkeypatch.setattr(steps, "_venv_has", lambda pip, pkg: True)
    monkeypatch.setattr(steps, "_pip_install_with_fallback",
                        lambda ctx_, specs, indexes: pip_specs.append((specs, indexes)))
    monkeypatch.setattr(steps.copy_app, "copy_app_source", lambda src, dest: 1)
    steps.step_install_deps(ctx)
    assert len(pip_specs) == 1
    assert pip_specs[0][0][0] == f"{ctx.app_dir}[ui]"  # 只有 app 依赖，无 torch


def test_pip_fallback_all_sources_fail(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(steps.proc, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    import pytest

    with pytest.raises(RuntimeError, match="依赖安装失败"):
        steps._pip_install_with_fallback(ctx, ["pkg"], ["idx1", "idx2"])


def test_pip_fallback_second_source_succeeds(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    calls = []
    real = steps.proc.run

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "--index-url" in cmd and "good" not in cmd:
            raise RuntimeError("bad source")
        return None

    monkeypatch.setattr(steps.proc, "run", fake_run)
    steps._pip_install_with_fallback(ctx, ["pkg"], ["bad", "good"])
    assert len(calls) == 2
    assert calls[1][calls[1].index("--index-url") + 1] == "good"


def test_app_version_from_pyproject(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "pyproject.toml").write_text(
        '[project]\nname = "uvr-lite"\nversion = "0.3.1"\n', encoding="utf-8")
    assert steps._app_version(app) == "0.3.1"
    assert steps._app_version(tmp_path / "nope") == "0.0.0"


# ---------- step_download_model ----------

def test_download_model_sets_env_and_progress(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    calls = {}
    percents = []
    ctx._percent = percents.append
    monkeypatch.setattr(steps.os, "environ", {})  # 隔离真实环境

    def fake_ensure_model(name, progress_callback=None):
        calls["name"] = name
        assert progress_callback(500, 1000) is True
        progress_callback(1000, 1000)
        return tmp_path / "m.ckpt"

    monkeypatch.setattr(steps, "ensure_model", fake_ensure_model)
    steps.step_download_model(ctx)
    assert calls["name"] == "bs_roformer_ep317"
    assert steps.os.environ["UVR_MODEL_DIR"] == str(ctx.models_dir)
    assert percents[-1] == 100


def test_download_model_cancel_stops(tmp_path, monkeypatch):
    """取消时进度回调返回 False（ensure_model 据此中断下载）。"""
    ctx = make_ctx(tmp_path, _cancel=lambda: True)
    monkeypatch.setattr(steps.os, "environ", {})
    results = {}

    def fake_ensure_model(name, progress_callback=None):
        results["cb_return"] = progress_callback(10, 100)
        return tmp_path / "m.ckpt"

    monkeypatch.setattr(steps, "ensure_model", fake_ensure_model)
    steps.step_download_model(ctx)
    assert results["cb_return"] is False


# ---------- step_create_shortcuts / step_write_marker ----------

def test_create_shortcuts(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    monkeypatch.setattr(steps.shortcuts, "create_shortcuts", lambda d: None)
    steps.step_create_shortcuts(ctx)  # 不抛即通过


def test_write_marker(tmp_path, monkeypatch):
    ctx = make_ctx(tmp_path)
    ctx.python_exe = Path("C:/py/python.exe")
    monkeypatch.setattr(steps, "_app_version", lambda d: "9.9.9")
    steps.step_write_marker(ctx)
    data = steps.marker.read_marker(tmp_path)
    assert data["app"] == "uvr-lite"
    assert data["app_version"] == "9.9.9"
    assert data["torch"] == steps.TORCH_VERSION
    assert Path(data["python"]) == Path("C:/py/python.exe")
