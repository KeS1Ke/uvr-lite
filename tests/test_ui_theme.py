"""主题样式把主窗口压回 720 逻辑像素以内，且不改用户的 QSettings。

``MainWindow()`` 会读 ``QSettings("uvr-lite", "uvr-lite")``。建窗口前把模块里的
``QSettings`` 换成临时 ini；结束时清掉这份 ini，不走 ``closeEvent``（它会
``_save_settings``）。finally 仍把用户注册表里的 ``minimize_to_tray`` 设回进入
用例之前，并还原会话级 QApplication 的 style / font / palette。
"""

from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPalette
from PySide6.QtWidgets import QSystemTrayIcon

import uvr_lite.ui.main as M
import uvr_lite.ui.theme as theme
from uvr_lite.ui.main import MainWindow

_KEY = "minimize_to_tray"
_STYLE_BY_CLASS = {
    "QWindowsVistaStyle": "windowsvista",
    "QWindows11Style": "windows11",
    "QWindowsStyle": "windows",
    "QFusionStyle": "fusion",
}


def _user_store() -> QSettings:
    store = QSettings("uvr-lite", "uvr-lite")
    store.sync()
    return store


def _snapshot_minimize() -> tuple[bool, object]:
    store = _user_store()
    had = store.contains(_KEY)
    return had, store.value(_KEY) if had else None


def _restore_minimize(had: bool, previous: object) -> None:
    store = _user_store()
    if had:
        store.setValue(_KEY, previous)
    elif store.contains(_KEY):
        store.remove(_KEY)
    store.sync()


def _install_ini_settings(monkeypatch, ini: Path) -> None:
    class _IniSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(ini), QSettings.IniFormat)

    monkeypatch.setattr(M, "QSettings", _IniSettings)


def _style_name(qapp) -> str:
    name = qapp.style().objectName()
    if name:
        return name
    return _STYLE_BY_CLASS.get(qapp.style().metaObject().className(), "")


def _clear_ini(settings, tmp_path: Path) -> None:
    if settings is None or settings.format() != QSettings.IniFormat:
        return
    name = settings.fileName() or ""
    try:
        parent = Path(name).resolve().parent
    except OSError:
        return
    if parent != tmp_path.resolve():
        return
    settings.clear()
    settings.sync()


def _quiesce(qapp, win) -> None:
    """收起托盘图标并销毁窗口。不走 closeEvent，避免写回配置。"""
    if hasattr(win, "_save_settings"):
        win._save_settings = lambda: None

    def _accept(event) -> None:
        event.accept()

    win.closeEvent = _accept
    icons = [
        obj
        for obj in win.findChildren(QSystemTrayIcon)
        if isinstance(obj, QSystemTrayIcon)
    ]
    direct = getattr(win, "_tray", None)
    if isinstance(direct, QSystemTrayIcon) and direct not in icons:
        icons.append(direct)
    for icon in icons:
        icon.hide()
        icon.deleteLater()
    win.hide()
    win.deleteLater()
    qapp.processEvents()


def _title_ink_bottom(box, limit_y: int) -> int:
    """标题色在分组框顶部的最底一行。只扫到第一个内容控件之上，避开下拉箭头。"""
    image = QImage(box.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor(0, 0, 0, 0))
    box.render(image)
    target = QColor(theme.TOKENS["ink_2"])
    last = -1
    x1 = min(image.width(), 240)
    for y in range(max(0, min(limit_y, image.height()))):
        hits = 0
        for x in range(12, x1):
            color = image.pixelColor(x, y)
            if (
                abs(color.red() - target.red()) <= 16
                and abs(color.green() - target.green()) <= 16
                and abs(color.blue() - target.blue()) <= 16
                and color.alpha() > 180
            ):
                hits += 1
        if hits >= 6:
            last = y
    return last


def _assert_title_above(box, widget) -> None:
    top = widget.mapTo(box, widget.rect().topLeft()).y()
    ink = _title_ink_bottom(box, top)
    assert ink >= 0, f"{box.title()} 的标题没有画出来"
    assert ink < top, f"{box.title()} 压住了第一行 ink={ink} y={top}"


def test_stylesheet_drops_scorecard_and_stacked_metrics():
    """设计过程批注不进样式表；分组框不再叠水平/底内边距，输入框不再锁 min-height。"""
    doc = theme.__doc__ or ""
    assert "Hallmark" not in doc
    assert "oklch()" in doc and "var()" in doc
    css = theme._qss()
    assert "Hallmark" not in css
    assert "pre-emit critique" not in css
    assert "22px 12px 10px 12px" not in css
    assert "min-height: 20px" not in css
    assert "min-height: 22px" not in css
    assert "min-height: 14px" in css


def test_themed_window_fits_within_720(qapp, tmp_path, monkeypatch):
    """apply 之后最小高度不超过 720，680 的请求尺寸放得下，「开始分离」在窗口内。"""
    had, previous = _snapshot_minimize()
    ini = tmp_path / "settings.ini"
    window = None
    style_name = _style_name(qapp)
    old_font = QFont(qapp.font())
    old_palette = QPalette(qapp.palette())
    try:
        _install_ini_settings(monkeypatch, ini)
        window = MainWindow()
        window.setAttribute(Qt.WA_DontShowOnScreen, True)
        window.setAttribute(Qt.WA_ShowWithoutActivating, True)
        theme.apply(qapp, window)
        window.show()
        qapp.processEvents()

        min_h = window.minimumHeight()
        assert min_h <= 720, f"minimumHeight={min_h}"
        assert min_h <= 680, f"minimumHeight={min_h}"

        window.resize(760, 680)
        qapp.processEvents()
        assert window.height() <= 720, f"height={window.height()}"

        origin = window.btn_start.mapTo(window, window.btn_start.rect().topLeft())
        bottom = origin.y() + window.btn_start.height()
        assert origin.x() >= 0
        assert origin.y() >= 0
        assert origin.x() + window.btn_start.width() <= window.width()
        assert bottom <= window.height(), f"btn_start bottom={bottom} height={window.height()}"

        _assert_title_above(window.file_box, window.file_stack)
        _assert_title_above(window.combo_model.parentWidget(), window.combo_model)
        _assert_title_above(window.label_engine.parentWidget(), window.label_engine)
        _assert_title_above(window.edit_out.parentWidget(), window.edit_out)
    finally:
        try:
            if window is not None:
                window.setStyleSheet("")
                _quiesce(qapp, window)
                _clear_ini(getattr(window, "settings", None), tmp_path)
        finally:
            if style_name:
                qapp.setStyle(style_name)
            qapp.setFont(old_font)
            qapp.setPalette(old_palette)
            _restore_minimize(had, previous)
