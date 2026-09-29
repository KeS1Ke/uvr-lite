"""票 2：输入文件扫描（选择文件夹添加方式）的纯函数测试。

另含主窗口分离队列槽函数（uvr_lite/ui/main.py）的回归用例：worker 信号里的
file_idx 越界时，槽函数不得先抛 IndexError（票 T9）。
"""

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

from uvr_lite import __version__
from uvr_lite.ui.files import dedup_paths, is_audio, scan_audio_files
from uvr_lite.ui.main import MainWindow


def test_is_audio_extensions():
    assert is_audio(Path("a.mp3"))
    assert is_audio(Path("a.FLAC"))  # 大小写不敏感
    assert is_audio(Path("a.wav")) and is_audio(Path("a.ogg")) and is_audio(Path("a.m4a"))
    assert not is_audio(Path("a.txt"))
    assert not is_audio(Path("a.mp4"))


def test_scan_audio_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "a.flac").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.mp3").write_bytes(b"x")  # 非递归：不应被扫到

    found = scan_audio_files(tmp_path)
    names = [p.name for p in found]
    assert names == ["a.flac", "b.wav"], f"应过滤扩展名且按名称排序: {names}"
    assert all(p.is_absolute() for p in found)
    assert not any(p.parent == sub for p in found)


def test_scan_audio_files_missing_dir():
    assert scan_audio_files(Path("不存在/的/目录")) == []


def test_dedup_paths_preserves_order(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = dedup_paths([a, b, a, tmp_path / "a.wav"])
    assert out == [a.resolve(), b.resolve()]


# ---------- 格式预检（开始分离前快速校验，无需完整解码） ----------

def test_precheck_audio_valid_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    from uvr_lite.ui.files import precheck_audio

    p = tmp_path / "song.wav"
    sf.write(str(p), np.zeros(44100), 44100)
    assert precheck_audio(p) is True


def test_precheck_audio_valid_flac(tmp_path):
    """flac 靠 soundfile（libsndfile）探测——audioread 无外部后端会误判，回归测试。"""
    import numpy as np
    import soundfile as sf

    from uvr_lite.ui.files import precheck_audio

    p = tmp_path / "song.flac"
    sf.write(str(p), np.zeros(44100), 44100)
    assert precheck_audio(p) is True


def test_precheck_audio_unicode_filename(tmp_path):
    """日文文件名（空格/非 ASCII）不应影响探测。"""
    import numpy as np
    import soundfile as sf

    from uvr_lite.ui.files import precheck_audio

    p = tmp_path / "ネクライトーキー - 煙とブルー.flac"
    sf.write(str(p), np.zeros(44100), 44100)
    assert precheck_audio(p) is True


def test_precheck_audio_garbage_rejected(tmp_path):
    from uvr_lite.ui.files import precheck_audio

    p = tmp_path / "fake.doc"
    p.write_bytes(b"not audio content at all" * 100)
    assert precheck_audio(p) is False


def test_precheck_audio_missing_file(tmp_path):
    from uvr_lite.ui.files import precheck_audio

    assert precheck_audio(tmp_path / "nope.wav") is False


# ---------- 分离队列槽函数（uvr_lite/ui/main.py）：越界 file_idx ----------

@pytest.fixture()
def win(qapp, tmp_path, monkeypatch):
    """真主窗口（不显示）：QSettings 落 tmp（不写用户注册表），日志根也指向 tmp。

    qapp 由 tests/conftest.py 提供；本模块不得再建 QApplication（T1）。
    """
    import uvr_lite.log as L

    monkeypatch.setattr(L, "repo_root", lambda: tmp_path)
    w = MainWindow()
    w.settings = QSettings(str(tmp_path / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


def test_file_failed_out_of_range_reports_instead_of_raising(qapp, win, tmp_path):
    """file_idx 越界（worker 信号错位/队列不一致）应走失败分支，不能先抛 IndexError。

    槽函数里抛异常会直接冒到 Qt 事件循环：用户看到的是崩溃而不是"某个文件失败"。
    """
    win._ok_paths = [tmp_path / "a.wav"]
    win._failed_names = []

    win._on_file_failed(5, "引擎异常")  # 修前：IndexError

    assert len(win._failed_names) == 1, "失败文件仍要进汇总清单"
    assert "引擎异常" in win.label_status.text(), "状态栏仍要给中文失败提示"


def test_file_failed_in_range_marks_item_bad(qapp, win, tmp_path):
    """范围内索引照旧：列表项标 ✗、进失败清单、状态栏带文件名与原因。"""
    p = tmp_path / "a.wav"
    win._paths = [p]
    win._rebuild_list()
    win._ok_paths = [p]
    win._failed_names = []

    win._on_file_failed(0, "引擎异常")

    assert win.list_files.item(0).text().startswith("✗")
    assert win._failed_names == ["a.wav"]
    assert "a.wav" in win.label_status.text()


def test_file_done_in_range_marks_item_ok(qapp, win, tmp_path):
    """成功路径（_ok_count 删除后的回归保护）：列表项标 ✓、耗时进历史。"""
    p = tmp_path / "a.wav"
    win._paths = [p]
    win._rebuild_list()
    win._ok_paths = [p]
    win._file_times = []
    win._t_file = 0.0

    win._on_file_done(0, None)

    assert win.list_files.item(0).text().startswith("✓")
    assert len(win._file_times) == 1, "每个完成文件都要记一次耗时（ETA 用）"


# ---------- 阶段中文名单一来源（uvr_lite/ui/progress.py） ----------

_PHASES = ["decode", "infer", "chunk", "tta", "write"]


def test_phase_labels_live_in_progress_module():
    """中文阶段名只有一份、且就在 progress.py（main 从那里导入，不再自建第二张表）。"""
    from uvr_lite.ui.progress import PHASE_CN

    assert set(PHASE_CN) == set(_PHASES), "阶段键集合是 UI 与进度接线的契约"


@pytest.mark.parametrize("phase", _PHASES)
def test_every_labeled_phase_has_progress_weight(phase):
    """漂移守卫：PHASE_CN 的每个键都必须被 ProgressTracker 认识。

    未知阶段 on_progress 返回 0.0（进度条停在该阶段起点），所以 > 0 即"接线认识它"。
    """
    from uvr_lite.ui.progress import PHASE_CN, ProgressTracker

    assert phase in PHASE_CN
    assert ProgressTracker().on_progress(phase, 1, 1) > 0.0, f"{phase} 没有对应的进度权重"


@pytest.mark.parametrize("phase", _PHASES)
def test_status_line_uses_progress_phase_table(qapp, win, phase):
    """用户可见行为：状态栏显示的中文阶段名取自 progress.PHASE_CN。"""
    from uvr_lite.ui.progress import PHASE_CN

    win._file_times = []
    win._on_progress(phase, 1, 1, 0, 1, 50)

    assert PHASE_CN[phase] in win.label_status.text()


# ---------- 版本可见入口（uvr_lite/ui/main.py，ADR-001 默认项） ----------

def test_window_title_shows_name_and_version(qapp, win):
    """窗口标题带版本：用户不用点任何东西、截图标题就能确认装的是哪个版本。

    ADR-001「默认项」要求窗口标题用 uvr-lite（原本已是），但版本此前只有
    `uvr-lite --version` 能看到，界面里没有任何出口（T10）。
    """
    title = win.windowTitle()

    assert "uvr-lite" in title, "标题必须保留 ADR 约定的产品名"
    assert __version__ in title, "标题要能看到版本号"
