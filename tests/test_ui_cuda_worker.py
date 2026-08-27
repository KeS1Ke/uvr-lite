"""票 2（tdd）：CudaTorchWorker——进度上报、取消、失败与成功信号。"""

import pytest
from PySide6.QtCore import QCoreApplication

from uvr_lite.ui.worker import CudaTorchWorker


@pytest.fixture(scope="module")
def qapp():
    app = QCoreApplication.instance() or QCoreApplication([])
    yield app


def test_success_reports_progress_and_finished(monkeypatch, tmp_path, qapp):
    """mock install_cuda_torch 回调进度 → 信号完整；成功时 finished(True, '')。"""
    calls = []

    def fake_install(base, progress_callback=None):
        assert base == tmp_path
        for done, total in [(1, 10), (5, 10), (10, 10)]:
            assert progress_callback(done, total) is True
        return tmp_path / "torch_cuda"

    monkeypatch.setattr("uvr_lite.download.install_cuda_torch", fake_install)
    w = CudaTorchWorker(tmp_path)
    w.progress.connect(lambda d, t: calls.append((d, t)))
    results = []
    w.finished.connect(lambda ok, err: results.append((ok, err)))
    w.run()
    assert calls == [(1, 10), (5, 10), (10, 10)]
    assert results == [(True, "")]


def test_cancel_returns_false_and_finished_cancelled(monkeypatch, tmp_path, qapp):
    """进度信号槽里 cancel() → 回调返回 False → install_cuda_torch 抛
    InterruptedError → finished(False, 取消文案)。"""
    def fake_install(base, progress_callback=None):
        for done, total in [(1, 10), (5, 10)]:
            if not progress_callback(done, total):
                raise InterruptedError("解压已取消")
        return tmp_path / "torch_cuda"

    monkeypatch.setattr("uvr_lite.download.install_cuda_torch", fake_install)
    w = CudaTorchWorker(tmp_path)
    results = []
    w.finished.connect(lambda ok, err: results.append((ok, err)))

    def on_progress(done, total):
        w.cancel()  # 第一次进度回调即取消

    w.progress.connect(on_progress)
    w.run()
    assert results and results[0][0] is False
    assert "已取消" in results[0][1]


def test_failure_forwards_error(monkeypatch, tmp_path, qapp):
    def boom(base, progress_callback=None):
        raise RuntimeError("SHA256 校验失败")

    monkeypatch.setattr("uvr_lite.download.install_cuda_torch", boom)
    w = CudaTorchWorker(tmp_path)
    results = []
    w.finished.connect(lambda ok, err: results.append((ok, err)))
    w.run()
    assert results == [(False, "SHA256 校验失败")]
