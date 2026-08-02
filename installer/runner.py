# coding: utf-8
"""安装器：安装步骤执行器（QThread 中运行）。

票 6：真实执行链——
  Python 检测/下载（build-standalone+镜像回退+SHA256+断点续传）
  → venv → torch（CPU/CUDA 分流+清华镜像）→ 复制代码快照 + pip install
  → 模型下载（进度）→ 快捷方式（桌面+开始菜单，♪）→ install.json 标记
覆盖升级：保留 venv 与模型，只更新代码/依赖/快捷方式；
中途取消：抛 InterruptedError 结束，下载的 .part 保留供续装。
"""

import os
import traceback
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from . import steps
from .steps import StepContext

# (标题, 步骤函数)
_STEPS: List = [
    ("准备 Python 环境", steps.step_prepare_python),
    ("创建虚拟环境", steps.step_create_venv),
    ("安装依赖", steps.step_install_deps),
    ("下载模型", steps.step_download_model),
    ("创建快捷方式", steps.step_create_shortcuts),
    ("完成安装", steps.step_write_marker),
]


class StepRunner(QObject):
    """逐步骤执行安装；进度经信号上报。"""

    step = Signal(int, int, str)        # 步骤序号, 总步骤数, 步骤标题
    message = Signal(str)               # 当前步骤的详细状态
    percent = Signal(int)               # 0-100
    finished = Signal(bool, str)        # 成功, 错误信息

    def __init__(self, install_dir: Path, upgrade: bool, src_dir: Optional[Path] = None):
        super().__init__()
        self.install_dir = install_dir
        self.upgrade = upgrade
        self.src_dir = src_dir
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        self.install_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.install_dir / "logs" / "install.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        total = len(_STEPS)

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n===== 安装开始（upgrade={self.upgrade}）=====\n")
            # ctx 跨步骤复用（步骤间传递 python_exe 等状态）
            ctx = StepContext(
                install_dir=self.install_dir,
                upgrade=self.upgrade,
                src_dir=self.src_dir,
                log_file=log_file,
                _cancel=lambda: self._cancel,
            )
            for idx, (title, fn) in enumerate(_STEPS):
                if self._cancel:
                    self.finished.emit(False, "已取消安装")
                    return
                self.step.emit(idx, total, title)
                log.write(f"[{idx + 1}/{total}] {title}\n")
                log.flush()

                def msg(text: str, _idx=idx) -> None:
                    self.message.emit(text)
                    log.write(f"    {text}\n")
                    log.flush()

                def pct(value: int, _idx=idx) -> None:
                    # 局部 0-100 → 全局
                    self.percent.emit(int((_idx + value / 100) / total * 100))

                ctx._message = msg
                ctx._percent = pct
                try:
                    fn(ctx)
                except InterruptedError:
                    log.write("    [取消]\n")
                    self.finished.emit(False, "安装已取消，已下载的部分会保留，可重新运行继续。")
                    return
                except Exception as e:  # noqa: BLE001
                    log.write(traceback.format_exc() + "\n")
                    self.finished.emit(False, f"「{title}」失败：{e}")
                    return
                self.percent.emit(int((idx + 1) / total * 100))
            self.percent.emit(100)
            self.finished.emit(True, "")
