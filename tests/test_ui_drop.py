"""拖放添加文件/文件夹（审计缺陷 2：拖入目录被静默忽略）。

实现见 uvr_lite/ui/main.py 的 dropEvent：拖入的每一项按「目录 → 走
scan_audio_files 扫顶层音频」/「文件 → 走 is_audio 过滤」分流，与
「选择输入文件夹」按钮同一条扫描路径。修复前 dropEvent 只收 is_file() 的项，
目录既不扫描也无任何反馈（用户以为程序坏了）。

dropEvent 走真实的 QDropEvent（QMimeData + QUrl），而不是把分流逻辑抽出来
单独测——那条路一旦被重构，这个用例就测不到真实接线了。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QMimeData, QPoint, QSettings, Qt, QUrl
from PySide6.QtGui import QDropEvent

import uvr_lite.ui.main as M


@pytest.fixture
def win(qapp, tmp_path, monkeypatch):
    import uvr_lite.log as L

    monkeypatch.setattr(L, "repo_root", lambda: tmp_path)
    w = M.MainWindow()
    w.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


def _audio(path: Path, seconds: float = 0.1) -> Path:
    """写一个真音频（与 files.AUDIO_EXTS 的后缀一致即可，drop 路径不预检）。"""
    sf.write(str(path), np.zeros(int(44100 * seconds)), 44100)
    return path


def _drop_urls(win, urls) -> None:
    """构造真实 QDropEvent（直接给 QUrl，可混入非本地 URL）投给 dropEvent。"""
    mime = QMimeData()
    mime.setUrls(list(urls))
    event = QDropEvent(
        QPoint(10, 10),
        Qt.CopyAction,
        mime,
        Qt.LeftButton,
        Qt.NoModifier,
    )
    win.dropEvent(event)


def _drop(win, paths) -> None:
    """构造真实 QDropEvent 投给窗口的 dropEvent。"""
    _drop_urls(win, [QUrl.fromLocalFile(str(p)) for p in paths])


def test_drop_folder_adds_its_audio_files(qapp, win, tmp_path):
    """拖入文件夹：扫描其中顶层音频并入列（与「选择文件夹」按钮一致）。"""
    folder = tmp_path / "album"
    folder.mkdir()
    _audio(folder / "b.flac")
    _audio(folder / "a.mp3")
    (folder / "cover.txt").write_bytes(b"x")

    _drop(win, [folder])

    names = sorted(p.name for p in win._paths)
    assert names == ["a.mp3", "b.flac"], f"目录拖放应扫描出音频: {names}"


def test_drop_folder_is_not_recursive(qapp, win, tmp_path):
    """非递归语义与 scan_audio_files 一致（不扫子目录）。"""
    folder = tmp_path / "album"
    sub = folder / "disc2"
    sub.mkdir(parents=True)
    _audio(folder / "top.wav")
    _audio(sub / "nested.wav")

    _drop(win, [folder])

    assert [p.name for p in win._paths] == ["top.wav"]


def test_drop_mixes_files_and_folders(qapp, win, tmp_path):
    """一次拖入文件 + 文件夹 + 非音频文件：前两类都入列，非音频被过滤。"""
    folder = tmp_path / "album"
    folder.mkdir()
    _audio(folder / "in_folder.wav")
    loose = _audio(tmp_path / "loose.flac")
    junk = tmp_path / "readme.txt"
    junk.write_bytes(b"x")

    _drop(win, [loose, folder, junk])

    names = sorted(p.name for p in win._paths)
    assert names == ["in_folder.wav", "loose.flac"], f"应同时收文件与文件夹内容: {names}"


def test_drop_folder_without_audio_gives_feedback(qapp, win, tmp_path):
    """空文件夹/无音频文件夹：不给静默（审计的原始症状），状态栏说明。"""
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "notes.txt").write_bytes(b"x")

    _drop(win, [empty])

    assert win._paths == []
    assert "没有找到音频" in win.label_status.text()


def test_drop_folder_then_files_are_deduped(qapp, win, tmp_path):
    """同一文件经「文件夹 + 直接拖入」两条路进来只算一次。"""
    folder = tmp_path / "album"
    folder.mkdir()
    song = _audio(folder / "song.mp3")

    _drop(win, [folder])
    _drop(win, [song])

    assert [p.name for p in win._paths] == ["song.mp3"]


def test_drop_folder_matches_add_folder_dialog(qapp, win, tmp_path, monkeypatch):
    """同一条扫描路径：drop 与「选择文件夹」按钮对同一目录产出一致的路径集。"""
    from uvr_lite.ui.files import scan_audio_files

    folder = tmp_path / "album"
    folder.mkdir()
    _audio(folder / "a.wav")
    _audio(folder / "b.ogg")

    _drop(win, [folder])
    via_drop = [Path(p) for p in win._paths]

    win._paths = []
    win._rebuild_list()
    monkeypatch.setattr(
        M.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder))
    )
    win._add_folder_dialog()
    via_dialog = [Path(p) for p in win._paths]

    assert via_drop == via_dialog == [Path(p) for p in scan_audio_files(folder)]


# ---------- 非本地 URL（回归：Path("") 被当成 CWD） ----------

def test_drop_non_local_url_does_not_scan_or_add_cwd(qapp, win, tmp_path, monkeypatch):
    """拖入 https://… 时 toLocalFile() 是空串，Path("") 会被当成当前目录扫出音频。

    修前该 URL 会走目录分支扫 CWD 并静默入列；现在必须直接跳过、连扫描都不触发。
    """
    real_scan = M.scan_audio_files
    calls: list = []

    def spy(path):
        calls.append(path)
        return real_scan(path)

    monkeypatch.setattr(M, "scan_audio_files", spy)
    _audio(tmp_path / "cwd_trap.mp3")  # CWD 里有音频：修前会被扫进来
    monkeypatch.chdir(tmp_path)

    _drop_urls(win, [QUrl("https://example.com/song.mp3")])

    assert calls == [], "非本地 URL 不应触发任何扫描"
    assert win._paths == [], "不应把 CWD 里的音频当成拖入项"


def test_drop_non_local_url_mixed_keeps_local_file(qapp, win, tmp_path):
    """混拖非本地 URL + 本地文件：本地文件照常入列。"""
    song = _audio(tmp_path / "local.flac")

    _drop_urls(win, [QUrl("https://example.com/song.mp3"), QUrl.fromLocalFile(str(song))])

    assert [p.name for p in win._paths] == ["local.flac"]


# ---------- 不可读目录（OSError）：不冲事件循环、不静默 ----------

def test_drop_unreadable_folder_reports_and_keeps_list(qapp, win, tmp_path, monkeypatch):
    """scan 抛 OSError（PermissionError 等）不能冲出 dropEvent 被 Qt 吞掉：
    不抛异常、不改动已有列表、状态栏说明原因。"""
    folder = tmp_path / "locked"
    folder.mkdir()
    keep = _audio(tmp_path / "keep.wav")
    win._add_paths([keep])
    before = list(win._paths)

    def boom(_folder):
        raise PermissionError("permission denied")

    monkeypatch.setattr(M, "scan_audio_files", boom)

    _drop(win, [folder])  # 修前：异常冲出（被 Qt 打印后吞掉），用户无反馈

    assert win._paths == before, "读取失败不应改动已有列表"
    assert "无法读取" in win.label_status.text()


def test_add_folder_dialog_unreadable_reports(qapp, win, tmp_path, monkeypatch):
    """「选择文件夹」遇到不可读目录同样要提示，不把异常冒到事件循环。"""
    folder = tmp_path / "locked"
    folder.mkdir()
    monkeypatch.setattr(
        M.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(folder)))

    def boom(_folder):
        raise PermissionError("permission denied")

    monkeypatch.setattr(M, "scan_audio_files", boom)

    win._add_folder_dialog()

    assert win._paths == []
    assert "无法读取" in win.label_status.text()


# ---------- 状态栏文案：无「已添加 0 个」噪音、计数准确、重复/忽略有说明 ----------

def test_drop_empty_folder_has_no_zero_added_noise(qapp, win, tmp_path):
    """列表为空时不该出现「已添加 0 个文件」，状态栏只讲原因。"""
    empty = tmp_path / "empty"
    empty.mkdir()

    _drop(win, [empty])

    text = win.label_status.text()
    assert "已添加 0 个文件" not in text
    assert "没有找到音频" in text


def test_drop_non_audio_only_has_no_zero_added_noise(qapp, win, tmp_path):
    """只拖非音频文件：状态栏只有忽略提示，没有「已添加 0 个文件」。"""
    junk = tmp_path / "readme.txt"
    junk.write_bytes(b"x")

    _drop(win, [junk])

    text = win.label_status.text()
    assert "已添加 0 个文件" not in text
    assert "非音频" in text


def test_drop_many_empty_folders_count_matches(qapp, win, tmp_path):
    """多空目录文案的计数必须与实际数量一致（修前「等 5 个」易被读成另有 5 个）。"""
    dirs = []
    for i in range(5):
        d = tmp_path / f"empty{i}"
        d.mkdir()
        dirs.append(d)

    _drop(win, dirs)

    text = win.label_status.text()
    assert "共 5 个" in text, text


def test_drop_mixed_reports_ignored_non_audio(qapp, win, tmp_path):
    """混拖被 is_audio 过滤掉的本地文件要计数并提示。"""
    song = _audio(tmp_path / "song.wav")
    (tmp_path / "a.txt").write_bytes(b"x")
    (tmp_path / "b.pdf").write_bytes(b"x")

    _drop(win, [song, tmp_path / "a.txt", tmp_path / "b.pdf"])

    text = win.label_status.text()
    assert "已添加 1 个文件" in text
    assert "已忽略 2 个非音频文件" in text


def test_drop_same_file_twice_reports_no_new(qapp, win, tmp_path):
    """重复拖入同一文件：状态栏说明无新增、不出现「已添加 0 个」。"""
    song = _audio(tmp_path / "song.mp3")

    _drop(win, [song])
    assert "已添加 1 个文件" in win.label_status.text()

    _drop(win, [song])

    text = win.label_status.text()
    assert "无新增" in text
    assert "已添加 0 个文件" not in text
    assert [p.name for p in win._paths] == ["song.mp3"]
