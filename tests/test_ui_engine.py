"""推理引擎区：环境不支持 CUDA wheel 时不得给出可点的下载按钮。

从 tests/test_ui_drop.py 拆出（那边只保留路径拖放用例）。修前按钮永远可点，
用户点下去在 worker 线程里才炸（Linux/CPU 环境下尤甚）。断言的是「用户看到
的状态」，不是 download 层已单测过的判定逻辑：

- 不支持：label 只显示首句，tooltip 显示完整文案（含「怎么办」①②）
- 支持 / 已安装：状态与按钮不变
"""

import pytest
from PySide6.QtCore import QSettings

import uvr_lite.ui.main as M


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    import uvr_lite.log as L

    monkeypatch.setattr(L, "repo_root", lambda: tmp_path)
    w = M.MainWindow()
    w.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


def test_cuda_button_disabled_when_env_unsupported(qapp, win, monkeypatch):
    """状态栏一行放首句；完整原因（含 ①②）走 tooltip，GUI 用户能看到怎么办。"""
    reason = ("当前环境不匹配：当前是 ARM64 架构（非 x64）。"
              "CPU 引擎不受影响，可直接用（速度较慢）。若要 GPU 加速，二选一："
              "① 装官方安装包；② pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
    monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: False)
    monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: False)
    monkeypatch.setattr(M, "cuda_engine_block_reason", lambda *a, **k: reason)

    win._refresh_engine_status()

    assert not win.btn_cuda.isEnabled(), "环境不支持时按钮应禁用"
    assert win.btn_cuda.text() == "不可用"
    assert "不支持" in win.label_engine.text()
    assert win.label_engine.text() == (
        "当前环境不支持 CUDA 引擎 — 当前环境不匹配：当前是 ARM64 架构（非 x64）。")
    assert win.btn_cuda.toolTip() == reason, "tooltip 要显示完整文案（含怎么办）"


def test_cuda_button_enabled_when_env_supported(qapp, win, monkeypatch):
    """回归：受支持环境（原生 Windows 3.12）按钮照旧可点。"""
    monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: False)
    monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: True)

    win._refresh_engine_status()

    assert win.btn_cuda.isEnabled()
    assert win.btn_cuda.text() == "下载 CUDA 引擎"


def test_cuda_installed_state_wins_over_env_check(qapp, win, monkeypatch):
    """已安装过就不再谈支持与否（装好的引擎不因换环境而被判不可用）。"""
    monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: True)
    monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: False)

    win._refresh_engine_status()

    assert win.btn_cuda.text() == "已安装"
    assert "已安装" in win.label_engine.text()
