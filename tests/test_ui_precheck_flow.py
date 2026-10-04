"""格式预检的**用户可见流程**（ADR-001 第 3 条）。

预检在分离 worker 的 run() 前段（加载模型之前），不在 GUI 槽里解码。
不合格文件在列表里标 ✗ 且不进分离；混合列表只把合格项交给 Separator；
超过 5 个不合格时状态栏只列前 5 个名字 + 省略号；全部不合格则弹窗并中止，
不构造 Separator。

QThread.start 改成在当前线程同步跑 worker（moveToThread 空操作），信号按
直接连接落到界面上，所以点完「开始分离」就能断言。不真起线程，也不加载模型。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QSettings, QThread

import uvr_lite.ui.main as M


class _InlineThread(QThread):
    """不启动操作系统线程：start() 同步把 started 发出去，worker.run() 就地执行。"""

    def start(self, *args, **kwargs) -> None:
        self.started.emit()

    def isRunning(self) -> bool:
        return False

    def quit(self) -> None:
        return None

    def wait(self, *args, **kwargs) -> bool:
        return True


@pytest.fixture
def win(qapp, tmp_path):
    w = M.MainWindow()
    # 别在测试里写用户注册表（QSettings 默认走 Windows 注册表），与 test_ui_log 同款
    w.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


def _wav(tmp_path: Path, name: str) -> Path:
    """写一个真能解码的音频（预检走 soundfile）。"""
    p = tmp_path / name
    sf.write(str(p), np.zeros(4410), 44100)
    return p


def _garbage(tmp_path: Path, name: str) -> Path:
    """写一个无法解码的假音频（soundfile 与 audioread 兜底都会失败）。"""
    p = tmp_path / name
    p.write_bytes(b"not audio at all" * 200)
    return p


def _prepare(tmp_path, monkeypatch):
    """装好「模型已下载」、假 Separator、同步线程，并拦下所有模态弹窗。

    返回 (提示文本, Separator 构造次数, separate 收到的路径)。
    """
    model = tmp_path / "fake.safetensors"
    model.write_bytes(b"x")
    monkeypatch.setattr(M, "model_file", lambda _name: model)
    monkeypatch.setattr(M, "QThread", _InlineThread)
    monkeypatch.setattr(M.SeparationWorker, "moveToThread", lambda *a, **k: None)
    created: list = []
    separated: list = []

    class _FakeSeparator:
        def __init__(self, **kw):
            created.append(kw)

        def separate(self, path, out_dir, progress_callback=None, **kw):
            separated.append(Path(path))
            return [path]

    monkeypatch.setattr("uvr_lite.engine.Separator", _FakeSeparator)
    hints: list = []
    for name in ("warning", "information"):
        monkeypatch.setattr(M.QMessageBox, name,
                            lambda *a: hints.append(a[-1]) or M.QMessageBox.Ok)
    monkeypatch.setattr(M.QMessageBox, "exec", lambda *a, **k: M.QMessageBox.Ok)
    return hints, created, separated


def _spy_status(win, monkeypatch) -> list:
    """记录状态栏每次 setText 的文本。

    混合列表下预检提示会被紧随其后的「准备中…」（main.py:598）覆盖，读最终文本
    只能看到后者，所以记录“设置过什么”，而不是“最后剩下什么”。
    """
    calls: list = []
    orig = win.label_status.setText
    monkeypatch.setattr(win.label_status, "setText",
                        lambda text: (calls.append(text), orig(text))[1])
    return calls


def _spy_log(monkeypatch) -> list:
    """拦下 _log()，记录预检写进日志的行（屏幕与日志用的是同一份截断名单）。"""
    records: list = []

    class _Recorder:
        def warning(self, msg, *args):
            records.append(msg % args if args else msg)

    monkeypatch.setattr(M, "_log", lambda: _Recorder())
    return records


def _texts(win) -> list:
    """列表当前每行的显示文本（顺序与 self._paths 一致）。"""
    return [win.list_files.item(i).text() for i in range(win.list_files.count())]


def test_all_good_queues_everything_without_warning(win, tmp_path, monkeypatch):
    """全合格：一个不漏地进队列，也不该出现「已跳过」。"""
    hints, created, separated = _prepare(tmp_path, monkeypatch)
    a, b = _wav(tmp_path, "a.wav"), _wav(tmp_path, "b.flac")
    win._add_paths([a, b])
    win._start_clicked()

    assert win._worker.files == [a.resolve(), b.resolve()]
    assert separated == [a.resolve(), b.resolve()]
    assert len(created) == 1
    assert hints == []
    assert "已跳过" not in win.label_status.text()


def test_mixed_list_marks_bad_and_queues_only_ok(win, tmp_path, monkeypatch):
    """混合：不合格行标 ✗（留在列表里，用户能看清是哪个），只有合格项进队列。"""
    _hints, _created, separated = _prepare(tmp_path, monkeypatch)
    status, logs = _spy_status(win, monkeypatch), _spy_log(monkeypatch)
    g1, b1 = _wav(tmp_path, "a.wav"), _garbage(tmp_path, "b.wav")
    g2, b2 = _wav(tmp_path, "c.wav"), _garbage(tmp_path, "d.wav")
    win._add_paths([g1, b1, g2, b2])
    win._start_clicked()

    assert separated == [g1.resolve(), g2.resolve()], "不合格项不得进队列"
    assert _texts(win) == [
        f"✓ {g1.name}",
        f"✗ {b1.name}（格式不支持）",
        f"✓ {g2.name}",
        f"✗ {b2.name}（格式不支持）",
    ]
    assert any("2 个文件无法识别为音频，已跳过：b.wav、d.wav" in t for t in status), status
    assert logs and "2 个文件预检失败" in logs[0] and "b.wav" in logs[0], logs


def test_more_than_five_bad_lists_only_first_five(win, tmp_path, monkeypatch):
    """超过 5 个不合格：状态栏与日志只列前 5 个名字 + 省略号，但每个坏文件仍在
    列表里单独标 ✗（持久通道，不依赖那条会被覆盖的状态栏提示）。"""
    _hints, _created, separated = _prepare(tmp_path, monkeypatch)
    status, logs = _spy_status(win, monkeypatch), _spy_log(monkeypatch)
    bads = [_garbage(tmp_path, f"bad{i}.wav") for i in range(1, 8)]
    good = _wav(tmp_path, "ok.wav")
    win._add_paths([*bads, good])
    win._start_clicked()

    assert separated == [good.resolve()]
    notice = next((t for t in status if t.startswith("7 个文件无法识别为音频")), None)
    assert notice is not None, status
    for p in bads[:5]:
        assert p.name in notice
    assert bads[5].name not in notice and bads[6].name not in notice
    assert notice.endswith("…"), "只列前 5 个时应以省略号收尾"
    assert logs and "7 个文件预检失败" in logs[0], logs
    # 被截断的名字不会丢：列表里每个坏文件都有自己的 ✗ 行
    assert [t for t in _texts(win) if t.startswith("✗ ")] == [
        f"✗ {p.name}（格式不支持）" for p in bads]


def test_all_bad_aborts_and_queues_nothing(win, tmp_path, monkeypatch):
    """全不合格：中止并弹窗说明，不加载模型（此时提示不会被完成框覆盖）。"""
    hints, created, separated = _prepare(tmp_path, monkeypatch)
    b1, b2 = _garbage(tmp_path, "x.wav"), _garbage(tmp_path, "y.wav")
    win._add_paths([b1, b2])
    win._start_clicked()

    assert created == [], "全不合格时不应加载模型"
    assert separated == []
    assert len(hints) == 1
    assert "都无法识别为音频格式" in hints[0]
    assert _texts(win) == [f"✗ {b1.name}（格式不支持）", f"✗ {b2.name}（格式不支持）"]
    assert "2 个文件无法识别为音频，已跳过" in win.label_status.text()
