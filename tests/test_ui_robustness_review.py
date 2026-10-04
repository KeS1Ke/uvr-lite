"""质量档、关窗超时、坏设置、合成 ignored 行、未装 CUDA 时拒绝 cuda 设备。

离屏 qapp 来自 conftest。不要 QApplication.quit()。
"""

from PySide6.QtCore import QMimeData, QPoint, QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QCloseEvent, QDropEvent

import uvr_lite.ui.main as M
from uvr_lite.quality import (
    QUALITY_CHOICES,
    QUALITY_PRESETS,
    QUALITY_STANDARD,
    quality_overlap,
)
from uvr_lite.ui.main import MainWindow, _settings_flag


class _NoStartThread(QThread):
    def start(self, *args, **kwargs) -> None:
        pass


class _TimeoutThread:
    """wait 超时：线程仍算在跑。"""

    def __init__(self) -> None:
        self._running = True
        self.calls: list = []

    def isRunning(self) -> bool:
        return self._running

    def quit(self):
        self.calls.append("quit")
        return None

    def wait(self, ms: int) -> bool:
        self.calls.append(("wait", ms))
        return False


class _Worker:
    def cancel(self):
        return None


def _quiesce(win) -> None:
    win._close_after_stop = False
    for attr in ("_thread", "_mix_thread", "_dl_thread", "_cuda_thread"):
        setattr(win, attr, None)
    for attr in ("_worker", "_mix_worker", "_dl_worker", "_cuda_worker"):
        setattr(win, attr, None)

    def _accept(event) -> None:
        event.accept()

    win.closeEvent = _accept
    win.close()


def test_settings_flag_false_string_is_false():
    """注册表里的 REG_SZ "false" 不能被 bool() 读成 True。"""
    assert bool("false") is True
    assert _settings_flag("false") is False
    assert _settings_flag("true") is True


def test_tta_false_string_and_corrupt_numbers(qapp, isolated_user_settings):
    """坏注册表值不能让 MainWindow() 抛出去；tta 的 "false" 读成未勾选。"""
    store = QSettings(str(isolated_user_settings), QSettings.IniFormat)
    store.setValue("tta", "false")
    store.setValue("bigshifts", "nope")
    store.setValue("batch_size", "100")
    store.setValue("mode", "xyz")
    store.setValue("mix_vocal_gain", "bad")
    store.setValue("mix_inst_gain", "9")
    store.setValue("num_overlap", "lots")
    store.sync()

    win = MainWindow()
    try:
        assert win.check_tta.isChecked() is False
        assert win.spin_bigshifts.value() == 1
        assert win.spin_batch.value() == 64
        assert win.combo_mode.currentIndex() == M.MODE_SEPARATE
        assert win.spin_vocal_gain.value() == 1.0
        assert win.spin_inst_gain.value() == 4.0
        assert win.combo_quality.currentData() == QUALITY_STANDARD
    finally:
        _quiesce(win)


def test_quality_combo_replaces_overlap_row(qapp, tmp_path, monkeypatch):
    """重叠窗口数换成质量档，不多一行；BigShifts / TTA 不跟档位清掉。"""
    monkeypatch.setattr(M, "repo_root", lambda: tmp_path)
    win = MainWindow()
    try:
        assert not hasattr(win, "spin_overlap")
        assert win.combo_quality.count() == len(QUALITY_CHOICES)
        labels = [win.combo_quality.itemText(i) for i in range(win.combo_quality.count())]
        data = [win.combo_quality.itemData(i) for i in range(win.combo_quality.count())]
        assert labels == [str(QUALITY_PRESETS[name]["menu"]) for name in QUALITY_CHOICES]
        assert data == list(QUALITY_CHOICES)
        assert win.combo_quality.currentData() == QUALITY_STANDARD
        tip = win.combo_quality.toolTip()
        assert str(QUALITY_PRESETS[QUALITY_STANDARD]["hint"]) in tip
        assert "ep317" in tip
        assert "karaoke" in tip
        assert "要更干净选「高」" in tip

        win.spin_bigshifts.setValue(3)
        win.check_tta.setChecked(True)
        win.combo_quality.setCurrentIndex(win.combo_quality.findData("fast"))
        assert win.spin_bigshifts.value() == 3
        assert win.check_tta.isChecked() is True
        assert str(QUALITY_PRESETS["fast"]["hint"]) in win.combo_quality.toolTip()

        win.settings.remove("quality")
        win.settings.setValue("num_overlap", 4)
        win._restore_settings()
        assert win.combo_quality.currentData() == "high"

        win.settings.setValue("num_overlap", 1)
        win.settings.remove("quality")
        win._restore_settings()
        assert win.combo_quality.currentData() == "fast"

        win.settings.setValue("quality", "high")
        win.settings.setValue("num_overlap", 1)
        win._restore_settings()
        assert win.combo_quality.currentData() == "high"

        model = tmp_path / "fake.safetensors"
        model.write_bytes(b"x")
        monkeypatch.setattr(M, "model_file", lambda _name: model)
        monkeypatch.setattr(M, "QThread", _NoStartThread)
        monkeypatch.setattr(
            M.QMessageBox, "information", lambda *a, **k: M.QMessageBox.Ok)
        win.combo_device.setCurrentText("cpu")
        win._add_paths([tmp_path / "song.wav"])
        win.combo_quality.setCurrentIndex(win.combo_quality.findData("fast"))
        win.check_tta.setChecked(True)
        win.spin_bigshifts.setValue(3)
        win._start_clicked()
        assert win._worker is not None
        assert win._worker.params.num_overlap == quality_overlap("fast") == 1
        assert win._worker.params.tta is True
        assert win._worker.params.bigshifts == 3

        win._worker = None
        win.combo_quality.setCurrentIndex(win.combo_quality.findData("high"))
        win._start_clicked()
        assert win._worker.params.num_overlap == 4
    finally:
        _quiesce(win)


def test_close_wait_timeout_ignores_and_retries(qapp):
    """wait 返回 False：ignore，状态栏提示，线程结束后再 close 一次。"""
    win = MainWindow()
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    try:
        thread = _TimeoutThread()
        win._worker = _Worker()
        win._thread = thread
        event = QCloseEvent()
        win.closeEvent(event)

        assert event.isAccepted() is False
        assert win.label_status.text() == "正在停止，请稍候…"
        assert win._close_after_stop is True
        assert thread.calls == ["quit", ("wait", 5000)]

        closes: list = []
        original = win.close

        def _close(*args, **kwargs):
            closes.append(1)
            return original(*args, **kwargs)

        win.close = _close
        thread._running = False
        win._on_task_thread_finished()
        assert closes == [1]
        assert win._close_after_stop is False
    finally:
        _quiesce(win)


def test_close_event_reentry_does_not_loop(qapp):
    """wait 里同步再进 closeEvent 时必须立刻 ignore，不能无限重入。"""
    win = MainWindow()
    entered = {"n": 0}
    original = win.closeEvent

    class _Reenter:
        def __init__(self) -> None:
            self._running = True

        def isRunning(self) -> bool:
            return self._running

        def quit(self):
            return None

        def wait(self, _ms: int) -> bool:
            nested = QCloseEvent()
            win.closeEvent(nested)
            assert nested.isAccepted() is False
            self._running = False
            return True

        def cancel(self):
            return None

    def _wrapped(event) -> None:
        entered["n"] += 1
        assert entered["n"] < 4, "closeEvent 同步重入没有停"
        original(event)

    win.closeEvent = _wrapped
    win._worker = _Reenter()
    win._thread = win._worker
    try:
        event = QCloseEvent()
        win.closeEvent(event)
        assert entered["n"] == 2
        assert event.isAccepted() is True
    finally:
        win.closeEvent = original
        _quiesce(win)


def test_ignored_stem_row_is_visible_and_removable(qapp, tmp_path, monkeypatch):
    """普通 song.mp3 拖进合成页要出现「非分离音轨」，能选中移除，且不能拿去合成。"""
    hints: list = []
    monkeypatch.setattr(
        M.QMessageBox, "information",
        lambda *a, **k: hints.append(a[-1]) or M.QMessageBox.Ok)
    win = MainWindow()
    try:
        win.combo_mode.setCurrentIndex(M.MODE_COMBINE)
        song = tmp_path / "song.mp3"
        song.write_bytes(b"not-a-real-decode")
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(song))])
        event = QDropEvent(
            QPoint(8, 8), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        win.dropEvent(event)

        assert win.mix_list.count() == 1
        assert "非分离音轨" in win.mix_list.item(0).text()
        assert song.name in win.mix_list.item(0).text()
        assert "已添加" in win.label_status.text()
        assert win._mix_pairs == []
        assert win._mix_ignored == [song.resolve()]

        win._start_clicked()
        assert win._mix_worker is None
        assert hints and "没有成对" in hints[0]

        win.mix_list.item(0).setSelected(True)
        win._mix_remove_selected()
        assert win.mix_list.count() == 0
        assert win._mix_files == []
        assert win.mix_stack.currentWidget() is win.mix_drop_hint
    finally:
        _quiesce(win)


def test_cuda_device_rejected_when_engine_missing(qapp, tmp_path, monkeypatch):
    """设备选 cuda 且引擎未安装：不起线程。auto/cpu 不拦截。"""
    monkeypatch.setattr(M, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: False)
    monkeypatch.setattr(M, "QThread", _NoStartThread)
    monkeypatch.setattr(
        M.QMessageBox, "information", lambda *a, **k: M.QMessageBox.Ok)
    model = tmp_path / "fake.safetensors"
    model.write_bytes(b"x")
    monkeypatch.setattr(M, "model_file", lambda _name: model)

    win = MainWindow()
    try:
        win._add_paths([tmp_path / "a.wav"])
        win.combo_device.setCurrentText("cuda")
        win._start_clicked()
        assert win._worker is None
        assert getattr(win, "_thread", None) is None
        assert "未安装 CUDA 引擎" in win.label_status.text()
        assert win._busy is False

        win.combo_device.setCurrentText("auto")
        win._start_clicked()
        assert win._worker is not None
        assert win._worker.params.device == "auto"

        win._worker = None
        win._busy = False
        win.combo_device.setCurrentText("cpu")
        win._start_clicked()
        assert win._worker is not None
        assert win._worker.params.device == "cpu"

        monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: True)
        win._worker = None
        win.combo_device.setCurrentText("cuda")
        win._start_clicked()
        assert win._worker is not None
        assert win._worker.params.device == "cuda"
    finally:
        _quiesce(win)


def test_set_busy_disables_download_and_respects_cuda(qapp, monkeypatch):
    win = MainWindow()
    try:
        monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: False)
        monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: True)
        win._set_busy(True)
        assert win.btn_download.isEnabled() is False
        assert win.btn_cuda.isEnabled() is False
        win._set_busy(False)
        assert win.btn_download.isEnabled() is True
        assert win.btn_cuda.isEnabled() is True

        monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: False)
        win._set_busy(True)
        win._set_busy(False)
        assert win.btn_download.isEnabled() is True
        assert win.btn_cuda.isEnabled() is False

        monkeypatch.setattr(M, "cuda_torch_installed", lambda *a, **k: True)
        monkeypatch.setattr(M, "cuda_engine_supported", lambda *a, **k: True)
        win._set_busy(False)
        assert win.btn_cuda.isEnabled() is False
    finally:
        _quiesce(win)
