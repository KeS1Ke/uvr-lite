# coding: utf-8
"""StepRunner 编排测试：步骤顺序、percent 全局映射、取消、失败传播、日志。"""

import pytest
from PySide6.QtCore import QCoreApplication

from installer import runner
from installer.runner import StepRunner


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def run_runner(tmp_path, monkeypatch, fake_steps, upgrade=False):
    """构造 StepRunner 并同步跑完，返回收集到的信号序列。"""
    monkeypatch.setattr(runner, "_STEPS", fake_steps)
    r = StepRunner(tmp_path, upgrade)
    out = {"steps": [], "msgs": [], "percents": [], "finished": None}
    r.step.connect(lambda i, t, ti: out["steps"].append((i, t, ti)))
    r.message.connect(out["msgs"].append)
    r.percent.connect(out["percents"].append)
    r.finished.connect(lambda ok, err: out.update(finished=(ok, err)))
    r.run()
    return out


def test_run_sequence_and_percent_mapping(tmp_path, monkeypatch):
    calls = []

    def s1(ctx):
        calls.append("s1")
        ctx.percent(50)  # 步骤 0 的 50% → 全局 25%

    def s2(ctx):
        calls.append("s2")
        ctx.percent(100)

    out = run_runner(tmp_path, monkeypatch, [("A", s1), ("B", s2)])
    assert calls == ["s1", "s2"]
    assert out["steps"] == [(0, 2, "A"), (1, 2, "B")]
    assert out["finished"] == (True, "")
    assert 25 in out["percents"]
    assert out["percents"][-1] == 100


def test_run_cancel_before_steps(tmp_path, monkeypatch):
    def s1(ctx):
        raise AssertionError("不应执行")

    r = StepRunner(tmp_path, False)
    out = {}
    r.finished.connect(lambda ok, err: out.update(finished=(ok, err)))
    r.cancel()
    monkeypatch.setattr(runner, "_STEPS", [("A", s1)])
    r.run()
    assert out["finished"] == (False, "已取消安装")


def test_run_cancel_mid_step(tmp_path, monkeypatch):
    def s1(ctx):
        raise InterruptedError

    out = run_runner(tmp_path, monkeypatch, [("A", s1), ("B", lambda c: None)])
    assert out["finished"][0] is False
    assert "取消" in out["finished"][1]


def test_run_failure_aborts_with_step_name(tmp_path, monkeypatch):
    def s1(ctx):
        raise RuntimeError("boom")

    out = run_runner(tmp_path, monkeypatch, [("A", s1), ("B", lambda c: None)])
    assert out["finished"] == (False, "「A」失败：boom")


def test_run_writes_log(tmp_path, monkeypatch):
    def s1(ctx):
        ctx.message("detail line")

    run_runner(tmp_path, monkeypatch, [("A", s1)])
    log = (tmp_path / "logs" / "install.log").read_text(encoding="utf-8")
    assert "安装开始" in log
    assert "[1/1] A" in log
    assert "detail line" in log


def test_upgrade_flag_passed_to_context(tmp_path, monkeypatch):
    seen = {}

    def s1(ctx):
        seen["upgrade"] = ctx.upgrade

    run_runner(tmp_path, monkeypatch, [("A", s1)], upgrade=True)
    assert seen["upgrade"] is True


def test_state_persists_across_steps(tmp_path, monkeypatch):
    """步骤间共享同一 StepContext（python_exe 等状态跨步传递）。"""
    seen = {}

    def s1(ctx):
        ctx.python_exe = tmp_path / "python.exe"

    def s2(ctx):
        seen["python_exe"] = ctx.python_exe

    run_runner(tmp_path, monkeypatch, [("A", s1), ("B", s2)])
    assert seen["python_exe"] == tmp_path / "python.exe"
