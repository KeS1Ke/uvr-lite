"""uvr_lite/log.py：本地排错日志的位置、级别、幂等、回滚与降级健壮性。

全部用例都把 uvr_lite.log.repo_root 指向 tmp_path（单测不得碰真实配置/安装目录）。
"""

import contextlib
import logging
import sys
from logging.handlers import RotatingFileHandler

import pytest

import uvr_lite.log as L


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """日志根目录重定向到临时目录 + 用后关句柄（Windows 上打开的文件会锁目录）。"""
    monkeypatch.setattr(L, "repo_root", lambda: tmp_path)
    yield tmp_path
    for name in list(L._READY):
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            with contextlib.suppress(Exception):
                h.close()
    L._READY.clear()


def test_log_dir_and_path_under_root(isolated):
    assert L.log_dir() == isolated / "logs"
    assert L.log_path() == isolated / "logs" / "uvr-lite.log"


def test_ensure_log_dir_creates_dir(isolated):
    assert L.ensure_log_dir() == isolated / "logs"
    assert (isolated / "logs").is_dir()


def test_warning_written_debug_dropped(isolated):
    logger = L.get_logger("test.level")
    logger.debug("调试细节不入文件")
    logger.info("过程信息也不入文件")
    logger.warning("出错了")

    text = L.log_path().read_text(encoding="utf-8")
    assert "出错了" in text
    assert "调试细节" not in text and "过程信息" not in text


def test_get_logger_is_idempotent(isolated):
    """重复 get_logger 不得重复挂 handler（否则每条日志被写多遍）。"""
    a = L.get_logger("test.idempotent")
    b = L.get_logger("test.idempotent")
    assert a is b
    handlers = len(a.handlers)
    assert handlers == 1, f"应只有一个文件 handler，实际 {handlers}"
    L.get_logger("test.idempotent").warning("只有一行")

    lines = [ln for ln in L.log_path().read_text(encoding="utf-8").splitlines() if ln]
    assert len(lines) == 1, f"同一条日志只应落盘一次: {lines}"


def test_logger_writes_to_file_only(isolated):
    """只挂文件 handler：pythonw 无控制台，往 stdout 写要么丢内容要么抛 OSError。"""
    logger = L.get_logger("test.handler")
    assert logger.level == logging.WARNING
    assert logger.propagate is False
    h, = logger.handlers
    # 注意 FileHandler 本就是 StreamHandler 的子类：判"不是控制台流"而非判类型
    assert isinstance(h, RotatingFileHandler)
    assert h.stream not in (sys.stdout, sys.stderr)
    assert h.maxBytes == L.MAX_BYTES and h.backupCount == L.BACKUP_COUNT
    assert h.encoding == "utf-8"


def test_log_exception_records_traceback(isolated):
    try:
        raise ValueError("样本异常")
    except ValueError:
        L.log_exception("加载模型失败: bs_roformer_ep317")

    text = L.log_path().read_text(encoding="utf-8")
    assert "加载模型失败" in text
    assert "Traceback" in text and "ValueError: 样本异常" in text, "应带上 traceback"


def test_log_exception_without_active_exception(isolated):
    """没有活动异常时退化为一条 ERROR，绝不抛出去（例如信号槽里只有错误字符串）。"""
    L.log_exception("下载失败（只有文案）")
    text = L.log_path().read_text(encoding="utf-8")
    assert "下载失败（只有文案）" in text
    assert "Traceback" not in text


def test_rotation_rolls_over(isolated, monkeypatch):
    """单条超过上限即回滚（用缩小的 MAX_BYTES 验证，不真写 2MB 数据）。"""
    monkeypatch.setattr(L, "MAX_BYTES", 256)
    logger = L.get_logger("test.rotate")
    for i in range(20):
        logger.warning("第 %d 条日志: %s", i, "x" * 80)

    assert L.log_path().exists()
    assert L.log_path().with_suffix(".log.1").exists(), "超出上限应生成回滚文件"
    assert L.log_path().stat().st_size <= 256 + 200, "回滚后当前文件应回到小体积"


# ---------- 健壮性：日志不可用时不得影响主流程 ----------


def test_unwritable_root_silently_degrades(monkeypatch, tmp_path):
    """logs 位置被同名文件占用 → ensure_log_dir 返回 None、写日志不抛异常。"""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "logs").write_text("我是文件不是目录", encoding="utf-8")
    monkeypatch.setattr(L, "repo_root", lambda: blocked)

    assert L.ensure_log_dir() is None

    logger = L.get_logger("test.blocked")
    logger.warning("照样不能崩")           # 不得抛异常
    L.log_exception("排错记录")
    assert logger.handlers, "降级后至少要有 NullHandler（logging 调用变空操作）"
    assert isinstance(logger.handlers[0], logging.NullHandler)


def test_repo_root_raising_does_not_propagate(monkeypatch, tmp_path):
    """异常目录布局下 repo_root 本身抛 RuntimeError 也必须被日志兜住。"""
    def boom():
        raise RuntimeError("无法定位 uvr-lite 根目录")

    monkeypatch.setattr(L, "repo_root", boom)
    L.get_logger("test.boom").warning("不该炸")
    L.log_exception("不该炸")
    assert L.log_hint() == "", "定位失败时不应把半成品路径塞给用户"


def test_log_hint_contains_path(isolated):
    assert L.log_hint() == f"\n\n详细信息已写入日志：{isolated / 'logs' / 'uvr-lite.log'}"
