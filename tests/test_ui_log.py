"""「查看日志」入口：按钮点击打开日志目录、日志缺失时的降级不崩。

日志根目录全部自己 monkeypatch 到 tmp_path（另一个 conftest 也在重定向，这里
的用例不得依赖真实仓库根的 logs 位置）。
"""

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, Qt, QUrl
from PySide6.QtWidgets import QApplication, QMessageBox

import uvr_lite.log as L
import uvr_lite.ui.main as M
from uvr_lite.ui.main import _ONE_LINE_ERR, MainWindow, _short_error


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """日志根目录重定向到 tmp_path（已有 conftest 的做法，这里再保一次底）。"""
    monkeypatch.setattr(L, "repo_root", lambda: tmp_path)
    yield tmp_path


@pytest.fixture()
def win(qapp, isolated):
    w = MainWindow()
    # 别在测试里写用户注册表（QSettings 默认走 Windows 注册表）
    w.settings = QSettings(str(isolated / "settings.ini"), QSettings.IniFormat)
    yield w
    w.close()


class _FakeThread:
    """_on_*_finished 只用到 quit()/wait()，closeEvent 还会问 isRunning()。"""

    def quit(self) -> None:
        pass

    def wait(self, _ms: int) -> bool:
        return True

    def isRunning(self) -> bool:
        return False

    def start(self) -> None:
        pass


def _as_path(url) -> Path:
    """URL → Path（Qt 给的是正斜杠，Windows 上是反斜杠，交给 Path 归一化）。"""
    return Path(url.toLocalFile())


def _spy_open(monkeypatch):
    """拦下"交给系统打开"的动作，返回记录到的 URL 列表。"""
    opened = []
    monkeypatch.setattr(
        M.QDesktopServices, "openUrl", lambda url: opened.append(url) or True)
    return opened


def _spy_hint(monkeypatch):
    """拦下给用户的中文提示（避免测试里弹模态框）。"""
    hints = []
    monkeypatch.setattr(M.QMessageBox, "information",
                        lambda *a: hints.append(a[-1]) or QMessageBox.Ok)
    return hints


def test_button_exists_and_shows_log_path(qapp, win, isolated):
    """按钮在界面上，tooltip/无障碍描述里给出日志文件的完整路径。"""
    assert win.btn_open_log.text() == "查看日志"
    path = str(isolated / "logs" / "uvr-lite.log")
    assert path in win.btn_open_log.accessibleDescription()
    assert path in win.btn_open_log.toolTip(), "悬停也应看到日志在哪"
    assert str(isolated / "logs") not in win.label_status.text(), "状态行文案保持干净"


def test_click_opens_log_dir_not_file(qapp, win, isolated, monkeypatch):
    """点击后交给系统打开的是日志**目录**（.log 在用户机器上未必有关联程序）。"""
    opened = _spy_open(monkeypatch)
    win.btn_open_log.click()

    assert len(opened) == 1, f"应只打开一次，实际 {opened}"
    assert opened[0].isLocalFile()
    assert _as_path(opened[0]) == isolated / "logs"
    assert opened[0] != QUrl.fromLocalFile(str(isolated / "logs" / "uvr-lite.log"))


def test_click_creates_dir_when_log_missing(qapp, win, isolated, monkeypatch):
    """日志从没生成过：先建目录再打开，用户第一次点也有东西可看。"""
    assert not (isolated / "logs").exists()
    opened = _spy_open(monkeypatch)
    win.btn_open_log.click()

    assert (isolated / "logs").is_dir(), "应先 ensure_log_dir 再打开"
    assert _as_path(opened[0]) == isolated / "logs"


def test_click_with_existing_log_file(qapp, win, isolated, monkeypatch):
    """日志已存在时同样打开所在目录，而不是文件本身。"""
    (isolated / "logs").mkdir()
    (isolated / "logs" / "uvr-lite.log").write_text("出错了\n", encoding="utf-8")

    opened = _spy_open(monkeypatch)
    win.btn_open_log.click()
    assert _as_path(opened[0]) == isolated / "logs"


def test_open_failure_shows_hint(qapp, win, monkeypatch):
    """系统打不开（返回 False）→ 一句中文提示，不抛异常。"""
    hints = _spy_hint(monkeypatch)
    monkeypatch.setattr(M.QDesktopServices, "openUrl", lambda url: False)

    win.btn_open_log.click()
    assert hints and "日志" in hints[0]


def test_unavailable_dir_shows_hint_and_never_raises(qapp, win, tmp_path, monkeypatch):
    """logs 位置被同名文件占用 → 不去打开，给一句友好中文提示，绝不抛异常。"""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "logs").write_text("我是文件不是目录", encoding="utf-8")
    monkeypatch.setattr(L, "repo_root", lambda: blocked)

    opened = _spy_open(monkeypatch)
    hints = _spy_hint(monkeypatch)

    win.btn_open_log.click()  # 不得抛异常
    assert opened == [], "目录不可用时不该再调用系统打开"
    assert hints and "日志" in hints[0], "应给一句中文提示"


def test_repo_root_raising_does_not_crash_ui(qapp, win, monkeypatch):
    """异常布局下 repo_root 直接抛错：降级成提示，不能把窗口点崩。"""
    def boom():
        raise RuntimeError("无法定位 uvr-lite 根目录")

    monkeypatch.setattr(L, "repo_root", boom)
    hints = _spy_hint(monkeypatch)

    win.btn_open_log.click()
    assert hints and "日志" in hints[0]


def test_failure_labels_point_at_log_button(qapp, win):
    """模型下载/CUDA 引擎失败的一行标签：短提示 + tooltip 指向「查看日志」。"""
    win._dl_thread = win._cuda_thread = _FakeThread()

    win._on_dl_finished(False, "网络中断")
    assert "日志" in win.label_banner.text()
    assert "查看日志" in win.label_banner.toolTip()

    win._on_cuda_finished(False, "磁盘空间不足")
    assert "日志" in win.label_engine.text()
    assert "查看日志" in win.label_engine.toolTip()


def test_short_error_keeps_one_line():
    """超长错误摘要被截断（完整原因在日志里），空错误兜一句中文。"""
    assert _short_error("磁盘空间不足") == "磁盘空间不足", "短文案原样保留"
    assert _short_error("") == "未知原因"

    trimmed = _short_error("网络被断开" * 10)
    assert trimmed.endswith("…") and len(trimmed) == _ONE_LINE_ERR + 1


def test_long_failure_text_still_fits_and_keeps_hint(qapp, win):
    """最坏情况（超长中文错误）下「失败原因见日志」仍在同一行里没被挤掉。"""
    long_err = "网络连接被断开，远端返回 404，重试多次仍然失败"
    win._dl_thread = win._cuda_thread = _FakeThread()
    win._on_dl_finished(False, long_err)
    win._on_cuda_finished(False, long_err)

    win.setAttribute(Qt.WA_DontShowOnScreen, True)  # 布局要走一遍才拿得到真实宽度
    win.show()
    qapp.processEvents()
    for label in (win.label_banner, win.label_engine):
        assert label.width() > 0, "布局未生效，宽度无从比较"
        # 允许字体差异带来的几个像素误差（真撑破会是几十上百像素）
        assert label.sizeHint().width() <= int(label.width() * 1.1), "文案撑出了可用宽度"
        assert "日志" in label.text(), "短提示不能被错误详情挤出单行"
    # 同一行的按钮仍在标签右边（没被挤出去）
    assert win.btn_cuda.x() >= win.label_engine.x() + win.label_engine.width()
