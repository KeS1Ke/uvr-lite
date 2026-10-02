"""UI「合成」页：模式切换、自动配对列表、任务参数接线。

线程用「真 QThread 但 start() 不做事」的子类（与 test_ui_precheck_flow 同款），
只验证队列与参数，不真跑合成。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QMimeData, QPoint, QSettings, Qt, QThread, QUrl
from PySide6.QtGui import QDropEvent

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
