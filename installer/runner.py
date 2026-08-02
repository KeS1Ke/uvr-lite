# coding: utf-8
"""安装器：安装步骤执行器（QThread 中运行）。

票 5：占位实现（演示页面流转）；票 6 填充真实执行链：
Python 检测/下载 → venv → torch → pip install -e . → 模型下载 → 快捷方式。
"""

from pathlib import Path
from typing import List

from PySide6.QtCore import QObject, Signal

_STEPS = [
    ("检查环境", "检测 Python 环境…"),
    ("准备安装目录", "准备安装目录…"),
    ("安装依赖", "安装依赖（约 4 GB，可能需要几分钟）…"),
    ("下载模型", "下载模型权重（约 640 MB）…"),
    ("创建快捷方式", "创建桌面与开始菜单快捷方式…"),
]


class StepRunner(QObject):
    """逐步骤执行安装；进度经信号上报。"""

    step = Signal(int, str)        # 步骤序号, 步骤标题
    message = Signal(str)          # 当前步骤的详细状态
    percent = Signal(int)          # 0-100
    finished = Signal(bool, str)   # 成功, 错误信息

    def __init__(self, install_dir: Path, upgrade: bool):
        super().__init__()
        self.install_dir = install_dir
        self.upgrade = upgrade
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            total = len(_STEPS)
            for idx, (title, detail) in enumerate(_STEPS):
                if self._cancel:
                    self.finished.emit(False, "已取消安装")
                    return
                self.step.emit(idx, title)
                self.message.emit(detail)
                # 票 6：在此执行真实步骤
                import time
                for pct in range(0, 101, 25):
                    if self._cancel:
                        self.finished.emit(False, "已取消安装")
                        return
                    time.sleep(0.2)
                    self.percent.emit(int((idx + pct / 100) / total * 100))
            self.percent.emit(100)
            self.finished.emit(True, "")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e))
