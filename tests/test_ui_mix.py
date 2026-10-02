"""UI「合成」页：模式切换、自动配对列表、任务参数接线。

线程用「真 QThread 但 start() 不做事」的子类（与 test_ui_precheck_flow 同款），
只验证队列与参数，不真跑合成。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QMimeData, QPoint, QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QCloseEvent, QDropEvent

import uvr_lite.ui.main as M
from uvr_lite.ui.worker import CombineParams, CombineWorker


class _NoStartThread(QThread):
    def start(self) -> None:
        pass


@pytest.fixture
def win(qapp, tmp_path):
    w = M.MainWindow()
    w.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


def _audio(path: Path) -> Path:
    sf.write(str(path), np.full(800, 0.1, dtype=np.float32), 8000)
    return path


def _pair_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "stems"
    folder.mkdir()
    _audio(folder / "song-vocals.flac")
    _audio(folder / "song-instrumental.flac")
    return folder


def _drop(win, paths) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    event = QDropEvent(QPoint(10, 10), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    win.dropEvent(event)


def _switch_to_mix(win) -> None:
    win.combo_mode.setCurrentIndex(M.MODE_COMBINE)


def test_mode_switch_swaps_page_button_and_status(win):
    assert win.stack.currentWidget() is win.sep_page

    _switch_to_mix(win)

    assert win.stack.currentWidget() is win.mix_page
    assert win.btn_start.text() == "开始合成"
    assert "合成" in win.label_status.text()

    win.combo_mode.setCurrentIndex(M.MODE_SEPARATE)
    assert win.stack.currentWidget() is win.sep_page
    assert win.btn_start.text() == "开始分离"


def test_mix_drop_folder_pairs_and_lists(win, tmp_path):
    folder = _pair_folder(tmp_path)
    _audio(folder / "lonely-vocals.flac")
    _switch_to_mix(win)

    _drop(win, [folder])

    assert len(win._mix_pairs) == 1
    assert [p.name for p in win._mix_unmatched] == ["lonely-vocals.flac"]
    assert win.mix_list.count() == 2
    assert win.mix_list.item(0).text().startswith("✓")
    assert win.mix_list.item(1).text().startswith("✗")
    assert win._paths == [], "合成页拖放不应进分离队列"
    assert "已配对 1 组" in win.label_status.text()


def test_mix_drop_does_not_touch_separation_page_status(win, tmp_path):
    win._add_paths([_audio(tmp_path / "song.flac")])
    before = list(win._paths)
    _switch_to_mix(win)

    _drop(win, [_audio(tmp_path / "a-vocals.flac")])

    assert win._paths == before, "合成页拖放不得改动分离队列"
    assert win.mix_list.count() == 1  # 孤儿人声进未配对行


def test_mix_remove_selected_removes_pair(win, tmp_path):
    folder = _pair_folder(tmp_path)
    _switch_to_mix(win)
    _drop(win, [folder])

    win.mix_list.item(0).setSelected(True)
    win._mix_remove_selected()

    assert win._mix_pairs == []
    assert win._mix_files == []
    assert win.mix_stack.currentWidget() is win.mix_drop_hint


def test_mix_folder_dialog_adds_pairs(win, tmp_path, monkeypatch):
    folder = _pair_folder(tmp_path)
    monkeypatch.setattr(
        M.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))

    win._mix_add_folder_dialog()

    assert len(win._mix_pairs) == 1
    assert win.mix_list.count() == 1
    assert "已配对 1 组" in win.label_status.text()


def test_mix_start_builds_worker_with_params(win, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "QThread", _NoStartThread)
    folder = _pair_folder(tmp_path)
    _switch_to_mix(win)
    _drop(win, [folder])
    win.spin_vocal_gain.setValue(0.8)
    win.spin_inst_gain.setValue(1.2)
    win.mix_combo_format.setCurrentText("wav")
    win.mix_combo_pcm.setCurrentText("16")
    win.check_normalize.setChecked(True)
    out = tmp_path / "out"
    win.mix_edit_out.setText(str(out))

    win._start_clicked()

    assert win._mix_worker is not None
    assert win._mix_worker.pairs == [
        (folder.resolve() / "song-vocals.flac", folder.resolve() / "song-instrumental.flac")
    ]
    params = win._mix_worker.params
    assert (params.vocal_gain, params.inst_gain, params.fmt,
            params.pcm, params.normalize) == (0.8, 1.2, "wav", "PCM_16", True)
    assert win._mix_worker.out_dir == str(out.resolve())
    assert win.mix_list.item(0).text().startswith("⏳")


def test_mix_start_without_files_warns(win, monkeypatch):
    hints: list = []
    monkeypatch.setattr(M.QMessageBox, "information",
                        lambda *a: hints.append(a[-1]) or M.QMessageBox.Ok)
    _switch_to_mix(win)

    win._start_clicked()

    assert hints and "请先添加" in hints[0]
    assert win._mix_worker is None


def test_mix_start_with_unmatched_only_warns(win, tmp_path, monkeypatch):
    hints: list = []
    monkeypatch.setattr(M.QMessageBox, "information",
                        lambda *a: hints.append(a[-1]) or M.QMessageBox.Ok)
    _switch_to_mix(win)
    _drop(win, [_audio(tmp_path / "only-vocals.flac")])

    win._start_clicked()

    assert hints and "没有成对" in hints[0]
    assert win._mix_worker is None


def test_combine_worker_forwards_params_and_reports(tmp_path, monkeypatch, qapp):
    calls: list = []

    def fake_combine(vocals, instrumental, out_dir, **kw):
        calls.append((vocals, instrumental, out_dir, kw))
        return Path(out_dir) / "x-mix.flac"

    monkeypatch.setattr("uvr_lite.mix.combine", fake_combine)
    params = CombineParams(vocal_gain=0.5, inst_gain=0.75, fmt="wav",
                           pcm="PCM_16", normalize=True)
    worker = CombineWorker(
        [(tmp_path / "a-vocals.wav", tmp_path / "a-instrumental.wav")],
        str(tmp_path / "out"), params)
    done: list = []
    finished: list = []
    worker.file_done.connect(lambda idx, files: done.append((idx, files)))
    worker.all_finished.connect(
        lambda ok, failed, cancelled: finished.append((ok, failed, cancelled)))

    worker.run()

    assert finished == [(1, 0, False)]
    assert done and done[0][0] == 0
    assert calls[0][3] == {
        "vocal_gain": 0.5, "inst_gain": 0.75, "pcm": "PCM_16",
        "fmt": "wav", "normalize": True,
        "verbose": False, "progress_callback": worker._on_progress,
    }


class _LiveThread(QThread):
    """isRunning() 为真，直到 quit；wait 只记录，不阻塞。"""

    def __init__(self, order: list, parent=None):
        super().__init__(parent)
        self._order = order
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def quit(self) -> None:
        self._order.append("quit")
        self._running = False

    def wait(self, *args, **kwargs) -> bool:
        self._order.append("wait")
        return True


def _assert_quit_and_wait_before_dialog(order: list) -> None:
    assert "quit" in order and "wait" in order and "dialog" in order
    assert order.index("quit") < order.index("dialog")
    assert order.index("wait") < order.index("dialog")


def test_finish_slots_release_running_threads_before_dialog(win, monkeypatch):
    order: list = []

    def fake_exec(self):
        order.append("dialog")
        return 0

    monkeypatch.setattr(M.QMessageBox, "exec", fake_exec)

    win._mix_thread = _LiveThread(order, win)
    win._on_mix_finished(1, 0, False)
    _assert_quit_and_wait_before_dialog(order)

    order.clear()
    win._failed_names = []
    win._thread = _LiveThread(order, win)
    win._on_all_finished(1, 0, False)
    _assert_quit_and_wait_before_dialog(order)


def test_second_mix_start_quits_previous_thread(win, tmp_path, monkeypatch):
    created: list = []

    class _RecordingThread(QThread):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.quit_called = False
            self._running = True
            created.append(self)

        def start(self) -> None:
            pass

        def isRunning(self) -> bool:
            return self._running

        def quit(self) -> None:
            self.quit_called = True
            self._running = False

        def wait(self, *args, **kwargs) -> bool:
            return True

    monkeypatch.setattr(M, "QThread", _RecordingThread)
    folder = _pair_folder(tmp_path)
    _switch_to_mix(win)
    _drop(win, [folder])
    win.mix_edit_out.setText(str(tmp_path / "out"))

    win._start_clicked()
    win._start_clicked()

    assert len(created) == 2
    assert created[0].quit_called
    assert win._mix_thread is created[1]
    assert win._mix_thread is not created[0]


class _RecycledThread:
    """任务跑完、finished→deleteLater 之后只剩 Python 包装器的 QThread。

    此时 isRunning() 与 quit()/wait() 都会抛 RuntimeError（shiboken 已释放
    C++ 对象），照抄真实现：只有 isRunning() 会先被问到。
    """

    def isRunning(self) -> bool:
        raise RuntimeError(
            "Internal C++ object (PySide6.QtCore.QThread) already deleted.")

    def quit(self) -> None:
        raise AssertionError("已回收的线程不该再被 quit")

    def wait(self, _ms: int) -> bool:
        raise AssertionError("已回收的线程不该再被 wait")


class _LiveThreadRecorder:
    """仍在运行的线程：记录收尾动作，供断言「后面的线程确实被收尾了」。"""

    def __init__(self, order: list, name: str) -> None:
        self._order = order
        self._name = name
        self._running = True

    def isRunning(self) -> bool:
        return self._running

    def quit(self) -> None:
        self._order.append(f"{self._name}:quit")
        self._running = False

    def wait(self, ms: int) -> bool:
        self._order.append(f"{self._name}:wait:{ms}")
        return True


class _WorkerRecorder:
    def __init__(self, order: list, name: str) -> None:
        self._order = order
        self._name = name

    def cancel(self) -> None:
        self._order.append(f"{self._name}:cancel")


def test_close_event_survives_recycled_threads(qapp, win):
    """上一轮任务已回收 QThread 时，关窗仍要把剩下所有线程收尾完。

    裸调 isRunning() 抛的 RuntimeError 会逃出 closeEvent，后面几个线程就不再
    cancel+wait；带活销毁运行中的 QThread 会让进程 abort（0xC0000409），写到
    一半的输出也可能被截断。
    """
    order: list = []
    win._worker = _WorkerRecorder(order, "sep")
    win._mix_worker = _WorkerRecorder(order, "mix")
    win._dl_worker = _WorkerRecorder(order, "dl")
    win._cuda_worker = _WorkerRecorder(order, "cuda")
    win._thread = _RecycledThread()          # 分离任务跑完，线程已回收
    win._dl_thread = _RecycledThread()       # 下载任务同理
    win._mix_thread = _LiveThreadRecorder(order, "mix")    # 合成仍在跑
    win._cuda_thread = _LiveThreadRecorder(order, "cuda")  # CUDA 下载仍在跑

    event = QCloseEvent()
    win.closeEvent(event)  # 不得抛异常

    assert order == ["mix:cancel", "mix:quit", "mix:wait:5000",
                     "cuda:cancel", "cuda:quit", "cuda:wait:3000"], order
    assert event.isAccepted(), "窗口仍应正常关闭"
