"""SeparationWorker 进度跨文件测试：tracker 每文件重置，杜绝进度回跳。"""

import subprocess
import sys

import pytest
from PySide6.QtWidgets import QApplication

from uvr_lite.ui.worker import SeparationWorker, clear_separator_cache


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _clean_separator_cache():
    """模块级缓存跨用例隔离：默认参数键会留在缓存里影响后续断言。"""
    clear_separator_cache()
    yield
    clear_separator_cache()


def test_ui_import_does_not_load_torch():
    """UI 启动路径（ui.main → worker）不得加载 torch。

    torch 导入约 2-6s + 上 GB 内存；引擎应在首个分离任务时才加载。
    子进程隔离：测试进程本身可能已因其他用例导入 torch。
    """
    code = ("import sys; import uvr_lite.ui.main; "
            "sys.exit(0 if 'torch' not in sys.modules else 1)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"import uvr_lite.ui.main 不应加载 torch（实际已加载）: {r.stderr.decode()[-200:]}")


def _run_two_files(monkeypatch, files, params):
    """mock Separator 按文件序号回调：文件 1 完整序列，文件 2 重复同序列。"""
    seq = [("decode", 0, 1), ("decode", 1, 1),
           ("chunk", 0, 100), ("chunk", 50, 100), ("chunk", 100, 100),
           ("infer", 1, 1),
           ("write", 1, 2), ("write", 2, 2)]

    class FakeSeparator:
        def __init__(self, **kw):
            pass

        def separate(self, path, out_dir, progress_callback=None, **kw):
            for phase, done, total in seq:
                assert progress_callback(phase, done, total) is True
            return [path]

    # run() 内惰性 `from ..engine import Separator` → patch 定义处
    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    w = SeparationWorker(files, "out", params)
    progress = []
    w.progress.connect(lambda ph, d, t, idx, ftot, pct: progress.append((idx, ph, pct)))
    w.run()
    return progress


def test_file_two_chunk_starts_from_zero(tmp_path, monkeypatch, qapp):
    """文件 2 的 chunk 应从 ~5% 开始（tracker 重置），不能从 50% 起（_pass_done 残留）。"""
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f1.write_bytes(b"x")
    f2.write_bytes(b"x")
    progress = _run_two_files(monkeypatch, [f1, f2], {"bigshifts": 1})

    file2_chunks = [(ph, pct) for idx, ph, pct in progress if idx == 1 and ph == "chunk"]
    assert file2_chunks, "应收到文件 2 的 chunk 回调"
    assert file2_chunks[0][1] == 5, f"文件 2 chunk 起点应为 5%（实际 {file2_chunks[0][1]}%）"


def test_progress_monotonic_within_file(tmp_path, monkeypatch, qapp):
    """单文件内进度不应回跳（chunk 升到 50 后 infer 不应打回）。"""
    f1 = tmp_path / "a.wav"
    f1.write_bytes(b"x")
    progress = _run_two_files(monkeypatch, [f1], {"bigshifts": 1})
    pcts = [pct for _, _, pct in progress]
    assert pcts == sorted(pcts), f"文件内进度应单调: {pcts}"


def test_separator_reused_across_runs(tmp_path, monkeypatch, qapp):
    """同一进程里重复点「开始」：相同构造参数只 load 一次；num_overlap 变了再构造。"""
    clear_separator_cache()
    inits: list[dict] = []

    class FakeSeparator:
        def __init__(self, **kw):
            inits.append(kw)

        def separate(self, path, out_dir, progress_callback=None, **kw):
            return [path]

    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    params = {
        "model_name": "demo",
        "device": "cpu",
        "batch_size": 4,
        "num_overlap": 2,
        "bigshifts": 1,
    }
    try:
        SeparationWorker([f], "out", params).run()
        SeparationWorker([f], "out", dict(params)).run()
        assert len(inits) == 1
        SeparationWorker([f], "out", {**params, "num_overlap": 8}).run()
        assert len(inits) == 2
        assert inits[0]["num_overlap"] == 2
        assert inits[1]["num_overlap"] == 8
        assert all(kw.get("verbose") is False for kw in inits)
    finally:
        clear_separator_cache()


def _fake_engine_separator(monkeypatch, inits):
    class FakeSeparator:
        def __init__(self, **kw):
            inits.append(kw)

        def separate(self, path, out_dir, progress_callback=None, **kw):
            return [path]

    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)


def test_separator_cache_keeps_only_latest(tmp_path, monkeypatch, qapp):
    """只保留最近一次会话：切走再切回原参数要重建（避免两套权重同时驻留）。"""
    inits: list[dict] = []
    _fake_engine_separator(monkeypatch, inits)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    base = {"model_name": "demo", "device": "cpu", "num_overlap": 2, "bigshifts": 1}
    SeparationWorker([f], "out", dict(base)).run()
    SeparationWorker([f], "out", {**base, "num_overlap": 8}).run()
    SeparationWorker([f], "out", dict(base)).run()
    assert len(inits) == 3


def test_separator_construction_failure_not_cached(tmp_path, monkeypatch, qapp):
    """构造失败不缓存：逐文件报失败，下一次运行重试构造并成功。"""
    inits: list[dict] = []

    class FlakySeparator:
        def __init__(self, **kw):
            inits.append(kw)
            if len(inits) == 1:
                raise RuntimeError("load boom")

        def separate(self, path, out_dir, progress_callback=None, **kw):
            return [path]

    monkeypatch.setattr("uvr_lite.engine.Separator", FlakySeparator)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    params = {"model_name": "demo", "device": "cpu", "num_overlap": 2, "bigshifts": 1}

    failed: list[tuple[int, str]] = []
    finished1: list[tuple[int, int, bool]] = []
    w1 = SeparationWorker([f], "out", dict(params))
    w1.file_failed.connect(lambda idx, err: failed.append((idx, err)))
    w1.all_finished.connect(lambda ok, total, cancelled: finished1.append((ok, total, cancelled)))
    w1.run()

    assert len(failed) == 1 and "load boom" in failed[0][1]
    assert finished1 == [(0, 1, False)]

    done: list[int] = []
    finished2: list[tuple[int, int, bool]] = []
    w2 = SeparationWorker([f], "out", dict(params))
    w2.file_done.connect(lambda idx, written: done.append(idx))
    w2.all_finished.connect(lambda ok, total, cancelled: finished2.append((ok, total, cancelled)))
    w2.run()

    assert done == [0] and finished2 == [(1, 0, False)]
    assert len(inits) == 2


def test_cancel_mid_run_reports_cancelled(tmp_path, monkeypatch, qapp):
    """回调返回 False → 引擎抛 CancelledError → all_finished(..., cancelled=True)。"""
    from uvr_lite.engine import CancelledError

    class CancellableSeparator:
        def __init__(self, **kw):
            pass

        def separate(self, path, out_dir, progress_callback=None, **kw):
            if not progress_callback("chunk", 1, 1):
                raise CancelledError("用户取消")
            return [path]  # pragma: no cover - 取消后不应到达

    monkeypatch.setattr("uvr_lite.engine.Separator", CancellableSeparator)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    w = SeparationWorker([f], "out", {"bigshifts": 1})
    w.cancel()
    finished: list[tuple[int, int, bool]] = []
    w.all_finished.connect(lambda ok, total, cancelled: finished.append((ok, total, cancelled)))
    w.run()
    assert finished == [(0, 0, True)]
