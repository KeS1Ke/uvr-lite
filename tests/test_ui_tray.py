"""最小化到系统托盘：勾选记忆、隐藏与恢复。

不把窗口最小化到任务栏上抢焦点。隐藏/恢复直接调 ``_hide_to_tray`` /
``_restore_from_tray``。另有一条用例构造 ``QWindowStateChangeEvent`` 交给
``changeEvent``，再排空 ``QTimer``（见该用例 docstring）。

``MainWindow`` 构造时会碰到用户的 ``QSettings("uvr-lite", "uvr-lite")``。
这里在建窗口前把模块里的 ``QSettings`` 换成临时 ini，用例里再用
``setValue`` + ``_restore_settings`` 写 ``minimize_to_tray``。结束时
``clear`` 这份 ini，不调用 ``closeEvent``（它会 ``_save_settings``）。
finally 仍把用户注册表里的 ``minimize_to_tray`` 恢复成进入用例前的值。
"""

from pathlib import Path

import pytest
from PySide6.QtCore import QEventLoop, QSettings, Qt, QTimer
from PySide6.QtGui import QWindowStateChangeEvent
from PySide6.QtWidgets import QApplication, QCheckBox, QSystemTrayIcon

import uvr_lite.ui.main as M
from uvr_lite.ui.main import MainWindow

# qapp 夹具由 tests/conftest.py 提供（session 级单例，模块内不得再建）

_KEY = "minimize_to_tray"


def _user_store() -> QSettings:
    store = QSettings("uvr-lite", "uvr-lite")
    store.sync()
    return store


def _snapshot_minimize() -> tuple[bool, object]:
    store = _user_store()
    had = store.contains(_KEY)
    return had, store.value(_KEY) if had else None


def _restore_minimize(had: bool, previous: object) -> None:
    """把用户注册表中的 minimize_to_tray 设回进入用例之前。"""
    store = _user_store()
    if had:
        store.setValue(_KEY, previous)
    elif store.contains(_KEY):
        store.remove(_KEY)
    store.sync()


def _patch_tray(monkeypatch, available: bool) -> None:
    def _available() -> bool:
        return available

    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", _available)
    tray_cls = getattr(M, "QSystemTrayIcon", None)
    if tray_cls is not None and tray_cls is not QSystemTrayIcon:
        monkeypatch.setattr(tray_cls, "isSystemTrayAvailable", _available)


def _install_ini_settings(monkeypatch, ini: Path) -> None:
    """让 MainWindow() 里的 QSettings('uvr-lite', 'uvr-lite') 落到临时 ini。"""

    class _IniSettings(QSettings):
        def __init__(self, *_args, **_kwargs):
            super().__init__(str(ini), QSettings.IniFormat)

    monkeypatch.setattr(M, "QSettings", _IniSettings)


def _as_bool(value: object) -> bool:
    """QSettings 读回的布尔可能是 bool，也可能是 'true'/'false'。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise AssertionError(f"minimize_to_tray 的值无法当成布尔：{value!r}")


def _tray_icons(win) -> list:
    found = []
    seen: set[int] = set()

    def add(obj) -> None:
        if isinstance(obj, QSystemTrayIcon) and id(obj) not in seen:
            seen.add(id(obj))
            found.append(obj)

    for obj in win.findChildren(QSystemTrayIcon):
        add(obj)
    for obj in vars(win).values():
        add(obj)
    qapp = QApplication.instance()
    if qapp is not None:
        for obj in qapp.findChildren(QSystemTrayIcon):
            if obj.parent() is win:
                add(obj)
    return found


def _tray_visible(win) -> bool:
    return any(icon.isVisible() for icon in _tray_icons(win))


def _show_offscreen(qapp, win) -> None:
    """先 show 一次（托盘逻辑需要），但不把窗口画到屏幕上，也不激活。"""
    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win.setAttribute(Qt.WA_ShowWithoutActivating, True)
    win.show()
    qapp.processEvents()


def _pump(qapp) -> None:
    """排空 changeEvent 里可能挂上的 singleShot(0)，最多等 300ms。"""
    qapp.processEvents()
    loop = QEventLoop()
    QTimer.singleShot(0, loop.quit)
    QTimer.singleShot(300, loop.quit)
    loop.exec()
    qapp.processEvents()


def _pretend_minimized(monkeypatch, win) -> None:
    """查询层面当作已最小化，避免真的缩到任务栏抢焦点。"""
    monkeypatch.setattr(win, "isMinimized", lambda: True)
    monkeypatch.setattr(win, "windowState", lambda: Qt.WindowMinimized)


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
    """收起托盘图标并销毁窗口。不走 closeEvent，避免 _save_settings 写回配置。"""
    if hasattr(win, "_save_settings"):
        win._save_settings = lambda: None

    def _accept(event) -> None:
        event.accept()

    win.closeEvent = _accept
    icons = _tray_icons(win)
    direct = getattr(win, "_tray", None)
    if isinstance(direct, QSystemTrayIcon) and direct not in icons:
        icons.append(direct)
    for icon in icons:
        icon.hide()
        icon.deleteLater()
    win.hide()
    win.deleteLater()
    qapp.processEvents()


@pytest.fixture()
def tray_available(monkeypatch, request):
    """默认把系统托盘桩成可用；用例可用 indirect 参数改成 False。"""
    available = getattr(request, "param", True)
    _patch_tray(monkeypatch, bool(available))
    return bool(available)


@pytest.fixture()
def win(qapp, tmp_path, monkeypatch, tray_available):
    had, previous = _snapshot_minimize()
    ini = tmp_path / "settings.ini"
    window = None
    try:
        _install_ini_settings(monkeypatch, ini)
        window = MainWindow()
        yield window
    finally:
        try:
            if window is not None:
                _quiesce(qapp, window)
                _clear_ini(getattr(window, "settings", None), tmp_path)
        finally:
            _restore_minimize(had, previous)


def test_checkbox_offers_minimize_to_tray(qapp, win):
    """界面上有「最小化到系统托盘」勾选框。"""
    box = win.check_minimize_tray
    assert isinstance(box, QCheckBox)
    assert box.text() == "最小化到系统托盘"
    assert win.isAncestorOf(box)


def test_should_minimize_follows_checkbox(qapp, win):
    """托盘可用时：未勾选为 False，勾选后为 True，勾选框可操作。"""
    assert QSystemTrayIcon.isSystemTrayAvailable() is True
    assert win.check_minimize_tray.isEnabled() is True

    win.check_minimize_tray.setChecked(False)
    assert win._should_minimize_to_tray() is False

    win.check_minimize_tray.setChecked(True)
    assert win._should_minimize_to_tray() is True


@pytest.mark.parametrize("tray_available", [False], indirect=True)
def test_unavailable_tray_disables_checkbox_and_keeps_window(qapp, win, tray_available):
    """系统托盘不可用：勾选框禁用，_should_minimize_to_tray 为 False，窗口不被藏起。"""
    assert tray_available is False
    assert QSystemTrayIcon.isSystemTrayAvailable() is False

    _show_offscreen(qapp, win)
    assert win.isVisible()
    assert win.check_minimize_tray.isEnabled() is False

    win.check_minimize_tray.setChecked(True)
    assert win._should_minimize_to_tray() is False
    win._hide_to_tray()
    _pump(qapp)
    assert win.isVisible()
    assert not _tray_visible(win)


def test_setting_minimize_to_tray_roundtrip(qapp, win):
    """键 minimize_to_tray：写入后 _restore_settings 反映到勾选框，_save_settings 再写回。"""
    win.settings.remove(_KEY)
    win.settings.sync()
    win._restore_settings()
    assert win.check_minimize_tray.isChecked() is False
    assert win._should_minimize_to_tray() is False

    win.settings.setValue(_KEY, True)
    win.settings.sync()
    win._restore_settings()
    assert win.check_minimize_tray.isChecked() is True
    assert win._should_minimize_to_tray() is True

    win.check_minimize_tray.setChecked(False)
    win._save_settings()
    win.settings.sync()
    assert _as_bool(win.settings.value(_KEY)) is False

    win.check_minimize_tray.setChecked(True)
    win._save_settings()
    win.settings.sync()
    assert _as_bool(win.settings.value(_KEY)) is True


def test_hide_to_tray_then_restore_from_tray(qapp, win):
    """勾选后 _hide_to_tray 藏起窗口并显示托盘图标；_restore_from_tray 反过来。"""
    _show_offscreen(qapp, win)
    assert win.isVisible()

    win.check_minimize_tray.setChecked(True)
    assert win._should_minimize_to_tray() is True

    win._hide_to_tray()
    _pump(qapp)
    assert win.isHidden() or not win.isVisible()
    icons = _tray_icons(win)
    assert icons, f"应出现托盘图标，现有属性：{[n for n in vars(win) if 'tray' in n.lower()]}"
    assert any(icon.isVisible() for icon in icons)

    win.setAttribute(Qt.WA_DontShowOnScreen, True)
    win._restore_from_tray()
    _pump(qapp)
    assert win.isVisible()
    assert not win.isMinimized()
    assert not (win.windowState() & Qt.WindowMinimized)
    assert not _tray_visible(win)


def test_hide_to_tray_noop_when_unchecked(qapp, win):
    """未勾选时 _hide_to_tray 不得把窗口藏起来，也不亮托盘图标。"""
    _show_offscreen(qapp, win)
    assert win.isVisible()

    win.check_minimize_tray.setChecked(False)
    assert win._should_minimize_to_tray() is False

    win._hide_to_tray()
    _pump(qapp)
    assert win.isVisible()
    assert not win.isHidden()
    assert not _tray_visible(win)


def test_change_event_hides_when_checked(qapp, win, monkeypatch):
    """changeEvent(QWindowStateChangeEvent) 在勾选时把窗口收进托盘。

    不调用 showMinimized，也不改系统窗口状态（避免抢任务栏焦点）。
    实现若要等「已经最小化」才隐藏，这里把 isMinimized / windowState 桩成
    已最小化，再把 QWindowStateChangeEvent 直接交给 changeEvent。
    隐藏若经 QTimer.singleShot 推迟，则用 processEvents 和一段局部事件循环排空。
    未勾选时同一事件不得把窗口藏起来。
    """
    _show_offscreen(qapp, win)
    assert win.isVisible()
    _pretend_minimized(monkeypatch, win)

    win.check_minimize_tray.setChecked(False)
    win.changeEvent(QWindowStateChangeEvent(Qt.WindowNoState))
    _pump(qapp)
    assert win.isVisible(), "未勾选时 changeEvent 不应隐藏窗口"

    win.check_minimize_tray.setChecked(True)
    win.changeEvent(QWindowStateChangeEvent(Qt.WindowNoState))
    _pump(qapp)
    assert win.isHidden() or not win.isVisible()
    assert _tray_visible(win)


def test_quit_from_tray_action_closes_and_quits(qapp, win, monkeypatch):
    """托盘「退出」先 close()（藏图标、写入设置），再 QApplication.quit()。

    窗口已经隐藏，单靠 close() 不会走到 quitOnLastWindowClosed。
    quit 必须先桩掉，不能真的结束 session 级 QApplication。
    """
    _show_offscreen(qapp, win)
    win.check_minimize_tray.setChecked(True)
    win._hide_to_tray()
    _pump(qapp)
    assert win.isHidden() or not win.isVisible()
    assert _tray_visible(win)
    assert not win.settings.contains(_KEY)

    quit_calls: list[int] = []

    def _record_quit(*_args, **_kwargs) -> None:
        quit_calls.append(1)

    monkeypatch.setattr(QApplication, "quit", _record_quit)
    monkeypatch.setattr(M.QApplication, "quit", _record_quit)

    menu = win._tray_menu
    assert menu is not None
    quit_actions = [action for action in menu.actions() if action.text() == "退出"]
    assert len(quit_actions) == 1
    quit_actions[0].trigger()
    _pump(qapp)

    assert quit_calls == [1]
    assert not _tray_visible(win)
    win.settings.sync()
    assert _as_bool(win.settings.value(_KEY)) is True
