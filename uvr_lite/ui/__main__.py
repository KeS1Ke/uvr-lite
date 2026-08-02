# coding: utf-8
"""无控制台启动入口：pythonw.exe -m uvr_lite.ui"""

import sys

from .main import run

if __name__ == "__main__":
    sys.exit(run())
