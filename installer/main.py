# coding: utf-8
"""uvr-lite 安装向导（Windows，PySide6）。

用法：
  python installer/main.py                 # 安装/覆盖升级
  python installer/main.py --uninstall     # 卸载（删除安装目录与快捷方式）

票 5：向导骨架——欢迎 → 安装目录 → 进行 → 完成 四页 + 卸载入口；
票 6 填充真实执行链。
"""

import ctypes
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

_ICON = Path(__file__).resolve().parent.parent / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico"
DEFAULT_DIR = Path.home() / "uvr-lite"

PAGE_WELCOME, PAGE_DIR, PAGE_RUN, PAGE_DONE = 0, 1, 2, 3


def _set_app_user_model_id(app_id: str) -> None:
    """Windows 任务栏图标跟随窗口图标（否则显示 Python 默认图标）。"""
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


class InstallerWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("uvr-lite 安装向导")
        self.setWindowIcon(QIcon(str(_ICON)))
        self.setFixedWidth(560)
        self.install_dir = Path(DEFAULT_DIR)
        self._runner = None
        self._thread = None
        self._failed = False

        self.stack = QStackedLayout(self)
        self._build_welcome()
        self._build_dir_page()
        self._build_run_page()
        self._build_done_page()

        self.btn_back = QPushButton("上一步", self)
        self.btn_next = QPushButton("下一步", self)
        self.btn_cancel = QPushButton("取消", self)
        nav = QHBoxLayout()
        # 先 attach 布局（获得宿主）再 addWidget，否则按钮会成为无 parent 的顶层窗口
        outer = QVBoxLayout(self)
        outer.addLayout(self.stack)
        outer.addLayout(nav)
        nav.addStretch(1)
        nav.addWidget(self.btn_back)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.btn_cancel)
        self.btn_back.clicked.connect(self._on_back)
        self.btn_next.clicked.connect(self._on_next)
        self.btn_cancel.clicked.connect(self._on_cancel)

        self._go(PAGE_WELCOME)

    # ---------- 页面 ----------

    def _build_welcome(self) -> None:
        page = QWidget(self)
        lay = QVBoxLayout(page)
        title = QLabel("欢迎使用 uvr-lite")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        body = QLabel(
            "将为你安装人声 / 伴奏分离工具：\n"
            "· 自动准备 Python 环境（无需预装）\n"
            "· 自动安装依赖与模型权重（约 4 GB，需联网）\n"
            "· 安装完成后桌面与开始菜单将出现 ♪ 快捷方式\n\n"
            "适合非专业用户：全程点击「下一步」即可。")
        body.setWordWrap(True)
        lay.addWidget(title)
        lay.addWidget(body)
        lay.addStretch(1)
        self.stack.addWidget(page)

    def _build_dir_page(self) -> None:
        page = QWidget(self)
        lay = QVBoxLayout(page)
        title = QLabel("选择安装位置")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        lay.addWidget(title)
        lay.addWidget(QLabel("所有文件（程序、Python、依赖、模型）都会安装到这个文件夹："))
        row = QHBoxLayout()
        self.edit_dir = QLineEdit(str(DEFAULT_DIR), page)
        self.btn_browse = QPushButton("浏览…", page)  # 显式 parent，防孤儿顶层窗口
        row.addWidget(self.edit_dir)
        row.addWidget(self.btn_browse)
        lay.addLayout(row)
        self.label_dir_hint = QLabel("")
        self.label_dir_hint.setWordWrap(True)
        self.label_dir_hint.setStyleSheet("color: #B8860B;")
        lay.addWidget(self.label_dir_hint)
        lay.addStretch(1)
        self.stack.addWidget(page)
        self.btn_browse.clicked.connect(self._browse_dir)
        self.edit_dir.textChanged.connect(self._on_dir_changed)

    def _build_run_page(self) -> None:
        page = QWidget(self)
        lay = QVBoxLayout(page)
        self.label_run_title = QLabel("正在安装…")
        self.label_run_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.label_run_status = QLabel("")
        self.label_run_status.setWordWrap(True)
        self.progress = QProgressBar(page)
        lay.addWidget(self.label_run_title)
        lay.addWidget(self.label_run_status)
        lay.addWidget(self.progress)

        # 页内取消确认（不弹独立窗口）
        self.confirm_row = QWidget(page)
        self.confirm_row.setVisible(False)
        self.confirm_row.setStyleSheet(
            "QWidget { background: #FFF8DC; border: 1px solid #E6C300; border-radius: 4px; }")
        cr = QHBoxLayout(self.confirm_row)
        cr.setContentsMargins(8, 6, 8, 6)
        cr.addWidget(QLabel("确定取消安装吗？已下载的部分会保留，下次可继续。"))
        self.btn_confirm_cancel = QPushButton("确认取消")
        self.btn_continue = QPushButton("继续安装")
        cr.addWidget(self.btn_confirm_cancel)
        cr.addWidget(self.btn_continue)
        lay.addWidget(self.confirm_row)
        self.btn_confirm_cancel.clicked.connect(self._confirm_cancel)
        self.btn_continue.clicked.connect(self._dismiss_confirm)

        lay.addStretch(1)
        self.stack.addWidget(page)

    def _build_done_page(self) -> None:
        page = QWidget(self)
        lay = QVBoxLayout(page)
        self.label_done = QLabel("")
        self.label_done.setWordWrap(True)
        self.label_done.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_launch = QPushButton("立即启动 uvr-lite")
        lay.addWidget(self.label_done)
        lay.addWidget(self.btn_launch)
        lay.addStretch(1)
        self.stack.addWidget(page)
        self.btn_launch.clicked.connect(self._launch_ui)

    # ---------- 导航 ----------

    def _go(self, page: int) -> None:
        self.stack.setCurrentIndex(page)
        self.btn_back.setEnabled(page in (PAGE_DIR,))
        self.btn_next.setText("开始安装" if page == PAGE_DIR else "下一步")
        self.btn_next.setVisible(page in (PAGE_WELCOME, PAGE_DIR))
        self.btn_cancel.setVisible(page not in (PAGE_DONE,))
        if page == PAGE_DIR:
            self._on_dir_changed()

    def _on_back(self) -> None:
        self._go(PAGE_WELCOME)

    def _on_next(self) -> None:
        if self.stack.currentIndex() == PAGE_WELCOME:
            self._go(PAGE_DIR)
        elif self.stack.currentIndex() == PAGE_DIR:
            self._start_install()

    def _on_cancel(self) -> None:
        if self.stack.currentIndex() == PAGE_RUN and self._runner is not None:
            # 页内确认，不弹独立窗口
            self.confirm_row.setVisible(True)
            self.btn_cancel.setEnabled(False)
            return
        self.reject()

    def _confirm_cancel(self) -> None:
        self.confirm_row.setVisible(False)
        if self._runner is not None:
            self._runner.cancel()
            self.label_run_status.setText("正在取消…")

    def _dismiss_confirm(self) -> None:
        self.confirm_row.setVisible(False)
        self.btn_cancel.setEnabled(True)

    # ---------- 目录页 ----------

    def _browse_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "选择安装位置", str(self.install_dir.parent))
        if folder:
            self.edit_dir.setText(folder)

    def _on_dir_changed(self) -> None:
        self.install_dir = Path(self.edit_dir.text().strip() or str(DEFAULT_DIR)).resolve()
        marker = self.install_dir / "install.json"
        if marker.exists():
            self.label_dir_hint.setText(
                "检测到该位置已安装 uvr-lite：将覆盖升级（保留已下载的模型，免重复下载）。")
        else:
            self.label_dir_hint.setText("")

    # ---------- 执行 ----------

    def _start_install(self) -> None:
        from .runner import StepRunner

        upgrade = (self.install_dir / "install.json").exists()
        self._runner = StepRunner(self.install_dir, upgrade)
        self._thread = QThread(self)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.step.connect(self._on_step)
        self._runner.message.connect(self.label_run_status.setText)
        self._runner.percent.connect(self.progress.setValue)
        self._runner.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._runner.deleteLater)

        self._go(PAGE_RUN)
        self.progress.setValue(0)
        self.btn_cancel.setEnabled(True)
        self.label_run_status.setText("准备中…")
        self._thread.start()

    def _on_step(self, idx, title) -> None:
        self.label_run_title.setText(f"正在安装…（{idx + 1}/5）{title}")

    def _on_finished(self, ok, error) -> None:
        self._thread.quit()
        self._thread.wait(5000)
        if ok:
            self.label_done.setText(
                f"安装完成！\n\n安装位置：{self.install_dir}\n"
                "桌面与开始菜单已创建 ♪ 快捷方式。")
            self._failed = False
        else:
            self.label_done.setText(f"安装未完成：\n{error}\n\n可重新运行本安装向导继续。")
            self._failed = True
        self._go(PAGE_DONE)
        self.btn_launch.setText("立即启动 uvr-lite" if ok else "关闭")
        self.btn_launch.setVisible(ok)

    def _launch_ui(self) -> None:
        import subprocess

        pythonw = self.install_dir / ".venv" / "Scripts" / "pythonw.exe"
        if pythonw.exists():
            subprocess.Popen([str(pythonw), "-m", "uvr_lite.ui"])
        self.accept()


def uninstall() -> int:
    """卸载：确认后删除安装目录与两处快捷方式。"""
    _set_app_user_model_id("uvr-lite.uninstaller")
    app = QApplication(sys.argv)
    app.setApplicationName("uvr-lite-installer")
    app.setWindowIcon(QIcon(str(_ICON)))
    target = QFileDialog.getExistingDirectory(
        None, "选择要卸载的 uvr-lite 安装目录", str(DEFAULT_DIR))
    if not target:
        return 0
    target = Path(target)
    marker = target / "install.json"
    if not marker.exists():
        QMessageBox.warning(None, "未找到安装", "该目录没有 uvr-lite 安装标记（install.json）。")
        return 1
    ret = QMessageBox.question(
        None, "确认卸载",
        f"将删除：\n· 安装目录 {target}\n· 桌面与开始菜单快捷方式\n\n"
        "此操作不可恢复，确定卸载吗？")
    if ret != QMessageBox.Yes:
        return 0
    import shutil

    from PySide6.QtCore import QStandardPaths

    ok = True
    try:
        for desktop in QStandardPaths.standardLocations(QStandardPaths.DesktopLocation):
            lnk = Path(desktop) / "uvr-lite.lnk"
            if lnk.exists():
                lnk.unlink()
        start = Path(QStandardPaths.writableLocation(QStandardPaths.ApplicationsLocation)) / "uvr-lite"
        if start.exists():
            shutil.rmtree(start, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        QMessageBox.warning(None, "卸载部分失败", f"删除时出错：{e}\n可手动删除剩余文件。")
        ok = False
    if ok:
        QMessageBox.information(None, "卸载完成", "uvr-lite 已卸载。")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--uninstall" in argv:
        return uninstall()
    _set_app_user_model_id("uvr-lite.installer")
    app = QApplication(sys.argv)
    app.setApplicationName("uvr-lite-installer")
    app.setWindowIcon(QIcon(str(_ICON)))
    win = InstallerWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
