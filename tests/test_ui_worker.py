# coding: utf-8
"""SeparationWorker 进度跨文件测试：tracker 每文件重置，杜绝进度回跳。"""

import pytest
from PySide6.QtCore import QCoreApplication

from uvr_lite.ui import worker
from uvr_lite.ui.worker import SeparationWorker


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def _run_two_files(monkeypatch, files, params):
    """mock separate_file 按文件序号回调：文件 1 完整序列，文件 2 重复同序列。"""
    seq = [("decode", 0, 1), ("decode", 1, 1),
           ("chunk", 0, 100), ("chunk", 50, 100), ("chunk", 100, 100),
           ("infer", 1, 1),
           ("write", 1, 2), ("write", 2, 2)]

    def fake_separate(path, out_dir, progress_callback=None, **kw):
        for phase, done, total in seq:
            assert progress_callback(phase, done, total) is True
        return [path]

    monkeypatch.setattr(worker, "separate_file", fake_separate)
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
