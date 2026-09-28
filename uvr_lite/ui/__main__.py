"""无控制台启动入口：pythonw.exe -m uvr_lite.ui

日志在这里初始化：UI 由 pythonw 启动，没有控制台（未捕获异常的 traceback
打出来也没人看得见，等于丢弃），不先落盘就无从排错。
"""

import sys

from ..log import ensure_log_dir, log_exception
from .main import run


def _install_excepthook() -> None:
    """未捕获异常先写日志，再交回原 hook（保留默认崩溃行为，不吞异常）。"""
    original = sys.excepthook

    def hook(etype, value, tb):
        log_exception("界面发生未捕获异常", (etype, value, tb))
        if original is not None:
            original(etype, value, tb)

    sys.excepthook = hook


if __name__ == "__main__":
    ensure_log_dir()
    _install_excepthook()
    sys.exit(run())
