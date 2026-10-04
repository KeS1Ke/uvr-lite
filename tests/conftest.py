"""pytest 全局隔离：单测不得把运行日志写进真实仓库根目录。

背景：``uvr_lite.log`` 把日志写到 ``repo_root()/logs/uvr-lite.log``，开发场景
下 ``repo_root()`` 就是仓库根。凡是走到失败路径的用例（下载失败、worker
异常、CLI 失败……）都会在真实仓库创建/追加 ``logs/`` 目录：既污染开发者的
工作区，又让用例之间互相串味——``uvr_lite.log._READY`` 按「logger 名 →
已挂目标文件」缓存，残留条目会让后续用例的 handler 仍指向上一个 tmp 路径。

本文件只做 path / fixture 处理，不放业务逻辑：

1. 把仓库根前置到 ``sys.path``（原因见下方注释）：保证 ``import uvr_lite``
   命中当前工作树，而不是 editable 安装可能指向的另一份 checkout。
2. autouse fixture：每个用例把 ``uvr_lite.log.repo_root`` 指向 pytest 的
   ``tmp_path``，并在用例前后关句柄 + 清 ``_READY`` 幂等缓存。
3. autouse fixture：把 ``MainWindow`` 构造时读的 ``QSettings("uvr-lite",
   "uvr-lite")`` 落到本例 tmp ini（见 ``isolated_user_settings``）。
4. session 级 ``qapp`` fixture：整个进程只建一个 QApplication。

与 tests/test_log.py 的关系：该文件自带同款的模块级 autouse fixture（也是
monkeypatch ``uvr_lite.log.repo_root``）。conftest 的 autouse 先设置、模块内
的后设置，所以**后者能覆盖前者**；monkeypatch 的 undo 是 LIFO 栈，收尾时两者
依次还原，最终回到真实的 repo_root。二者是叠加而非打架。
"""

import contextlib
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent  # tests/ 的上一层 = 仓库根

# ---------- sys.path 前置 ----------
# 本机存在 editable 安装（site-packages/__editable__.uvr_lite-*.pth →
# 本机 editable 安装的 .pth 指向仓库里的 uvr_lite）。editable finder 是 append 到
# sys.meta_path 末尾的，通常被排在前面的 PathFinder 抢先，因此「当前工作树
# 优先」其实是靠 cwd 恰好在 sys.path[0] 侥幸成立：换成 `pytest` 脚本入口
# （sys.path[0] 不再是 cwd）或从别处调用，import uvr_lite 就会解析到另一份
# 代码——改了源码，测试却跑旧实现。这里显式把仓库根插到最前消掉不确定性；
# 已存在时先移除再插入，避免重复条目。
_root = str(REPO_ROOT)
if _root in sys.path:
    sys.path.remove(_root)
sys.path.insert(0, _root)


def _forget_handlers(log) -> None:
    """卸掉所有已挂的日志 handler（并释放文件句柄）+ 清空 _READY 缓存。

    不清的后果有两种：一是 _READY[name] 还记着上一个 tmp 路径，get_logger
    判定「目标没变」直接复用旧 handler，日志写进已被删除或别人的目录；二是
    Windows 上文件句柄不 close 会锁住目录/文件。
    """
    for name in list(log._READY):
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
    # _READY 只是幂等标记字典；"是否提示过降级"（log._degraded_warned）属于
    # 一次性 stderr 提示，留着不动，免得每条用例重复刷屏。
    log._READY.clear()


@pytest.fixture(autouse=True)
def isolated_log_root(tmp_path, monkeypatch):
    """把日志根目录隔离到本例专属的 tmp_path（autouse，全 suite 生效）。

    想额外指定日志位置的用例（例如 test_log.py）只需在自己的 fixture 里再
    monkeypatch 一次 ``uvr_lite.log.repo_root``——后设置的生效，无需关掉本
    fixture。
    """
    import uvr_lite.log as log

    monkeypatch.setattr(log, "repo_root", lambda: tmp_path)
    _forget_handlers(log)  # 上一条用例残留的 handler 指向旧路径
    yield tmp_path
    _forget_handlers(log)  # 本例的 handler 指向本例 tmp，仍要关掉释放句柄


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    """让 ``MainWindow()`` 构造时读的 QSettings 落到本例专属的 tmp ini。

    背景：``MainWindow.__init__`` 建窗口时就 ``_restore_settings()``，读的是
    ``QSettings("uvr-lite", "uvr-lite")``（Windows 上是用户注册表）。各 UI 模块
    「先建窗口、再替换 ``w.settings``」的顺序挡不住这一次读取：开发机上存过
    ``mode=1``（合成页）、``tta=true``、``out_dir`` 等值，用例结果就取决于开发
    机的用户配置——窗口开在合成页时，拖放/预检/模式切换用例会整片失败。

    test_ui_theme / test_ui_tray 早已用同一个「建窗口前换掉模块里的 QSettings」
    手法自保，这里提到 conftest 做全 suite 兜底；模块内再 monkeypatch 一次仍然
    后设置生效（monkeypatch 的 undo 是 LIFO）。
    """
    from PySide6.QtCore import QSettings

    import uvr_lite.ui.main as main

    ini = tmp_path / "qsettings.ini"

    class _IniSettings(QSettings):
        def __init__(self, *args, **kwargs):
            super().__init__(str(ini), QSettings.IniFormat)

    monkeypatch.setattr(main, "QSettings", _IniSettings)
    yield ini


@pytest.fixture(scope="session")
def qapp():
    """全 suite 共用的 QApplication 单例（UI 测试统一从这里取）。

    Qt 每个进程只允许一个 QApplication/QCoreApplication，且**销毁后不能再建**
    第二个：原先 4 个 UI 测试模块各写一份 ``scope="module"`` 的 qapp，先建的
    模块跑完后解释器回收掉单例，下一个模块再建就直接触发 shiboken fastfail
    （PySide6 6.11.1 实测为进程 0xC0000409，整轮 pytest 无汇总中断，后面的
    用例一个都不跑）。所以这里建一次、session 结束前一直持有引用。

    返回 QApplication（它本身即 QCoreApplication），只需要信号/事件循环的用例
    同样可用；各模块不必再自己建。
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
