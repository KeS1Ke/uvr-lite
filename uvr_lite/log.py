"""本地运行日志：写 ``{安装目录}/logs/uvr-lite.log``，供用户报障时排错。

设计取舍（与 ADR-001「默认项」承诺一致）：

- 只写文件、只记 WARNING 及以上：目标用户是非专业人士，正常运行的 INFO
  流水账既看不懂也白占磁盘；出错时才有东西可查。
- 不挂 StreamHandler：UI 由 ``pythonw.exe`` 启动，没有控制台（stdout/stderr
  不可用），往那儿写要么丢内容，要么在 Windows 上直接抛 OSError。
- 不联网、不采集用户信息：这是本地排错日志，不是遥测，也不是结构化日志方案。
- 失败绝不影响主流程：只读安装目录、磁盘满、杀软锁句柄都可能让日志不可用，
  此时静默降级——宁可没有日志，也不能让"写日志"把分离任务或启动流程搞崩。
"""

import contextlib
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .download import repo_root

LOG_NAME = "uvr-lite"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "uvr-lite.log"
MAX_BYTES = 2 << 20       # 单文件上限 2 MB（超出即回滚）
BACKUP_COUNT = 3          # 保留 3 个回滚文件（总占用 ≤ 8 MB）
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# logger 名 → 已挂上的日志文件（幂等标记）。"" 表示当前不可用，已降级。
_READY: dict[str, str] = {}
_degraded_warned = False


def log_dir() -> Path:
    """日志目录：安装目录/logs（开发场景即仓库根/logs，已被 .gitignore 忽略）。

    根目录判定复用 download.repo_root：与 models/、torch.ini 同一层，安装
    场景指向安装目录，不会出现"日志写在一个地方、模型在另一个地方"。
    """
    return repo_root() / LOG_DIR_NAME


def log_path() -> Path:
    """日志文件路径（弹窗里给用户的就是这个）。"""
    return log_dir() / LOG_FILE_NAME


def ensure_log_dir() -> Path | None:
    """确保日志目录存在；不可用则返回 None（静默，不抛给调用方）。

    只读安装目录、权限不足、logs 被同名文件/其他程序占用都会走到 OSError，
    日志只是排错辅助，不能因此让启动或分离失败。
    """
    d = log_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def get_logger(name: str = LOG_NAME) -> logging.Logger:
    """取已接好文件 handler 的 logger；重复调用不重复挂 handler。

    幂等按「logger 名 + 目标文件」判定：测试/安装切换根目录时能自动重挂，
    同一目标下重复调用则直接返回（否则同一条日志会被写 N 遍）。
    """
    logger = logging.getLogger(name)
    target = _target_path()
    if not target:
        _degrade(logger)
        _READY[name] = ""
        return logger
    if _READY.get(name) == target and logger.handlers:
        return logger

    _reset_handlers(logger)
    logger.setLevel(logging.WARNING)  # 文件级过滤：DEBUG/INFO 不入文件
    logger.propagate = False          # 不冒泡到 root：避免被别人接的 handler 重复打一遍
    try:
        if ensure_log_dir() is None:
            raise OSError("日志目录不可用")
        handler = RotatingFileHandler(target, maxBytes=MAX_BYTES,
                                      backupCount=BACKUP_COUNT, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    except OSError as e:
        _degrade(logger, e)
        target = ""
    _READY[name] = target
    return logger


def log_exception(context: str, exc_info=None) -> None:
    """把带 traceback 的异常写进日志。

    context 用中文说明"在做什么时出错"（例如"分离失败: song.mp3"）；
    exc_info 省略时取当前活动的异常（``sys.exc_info()``）。
    """
    logger = get_logger()
    info = exc_info if exc_info is not None else sys.exc_info()
    if isinstance(info, BaseException):  # 传异常实例时补成 logging 认得的三元组
        info = (type(info), info, info.__traceback__)
    if info[0] is None:
        # 信号槽里只拿到错误字符串、没有堆栈可记录：至少留下时间点与上下文
        logger.error("%s（无异常堆栈）", context)
        return
    logger.exception(context, exc_info=info)


def log_hint() -> str:
    """给用户的中文提示后缀（含日志路径），日志不可用时为空串。"""
    try:
        return f"\n\n详细信息已写入日志：{log_path()}"
    except Exception:
        # repo_root() 在异常目录布局下会抛 RuntimeError：提示文案不该成为崩溃点
        return ""


# ---------- 内部实现 ----------

def _target_path() -> str:
    """当前日志文件路径；定位失败（异常布局/OSError）时返回 ""（=降级信号）。"""
    try:
        return str(log_path())
    except Exception:
        return ""


def _reset_handlers(logger: logging.Logger) -> None:
    for h in list(logger.handlers):
        logger.removeHandler(h)
        # 释放日志文件句柄（Windows 不关会锁住目录，卸载时删不掉）
        with contextlib.suppress(Exception):
            h.close()


def _degrade(logger: logging.Logger, reason: Exception | None = None) -> None:
    """降级到 NullHandler：所有 logging 调用变成空操作，主流程无感。"""
    _reset_handlers(logger)
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    logger.addHandler(logging.NullHandler())
    if reason is not None:
        _warn_once(f"日志不可用，本次运行不记录日志：{reason}")


def _warn_once(msg: str) -> None:
    """把降级原因提示一次（pythonw 无控制台时 sys.stderr 为 None，print 自动丢弃）。"""
    global _degraded_warned
    if _degraded_warned:
        return
    _degraded_warned = True
    print(f"[uvr-lite] {msg}", file=sys.stderr)
