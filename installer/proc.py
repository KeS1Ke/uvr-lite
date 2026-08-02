# coding: utf-8
"""子进程辅助：输出实时写日志，失败抛带尾部的异常。"""

import subprocess
from pathlib import Path
from typing import Callable, List, Optional

_TAIL_LEN = 1200


def run(cmd: List[str], log_file: Optional[Path] = None,
        cancel: Optional[Callable[[], bool]] = None,
        env: Optional[dict] = None,
        timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    """执行子进程；stdout/stderr 实时写入日志文件（若给）。

    取消：每次输出块前检查 cancel()，为真则结束进程并抛 InterruptedError。
    失败：抛 RuntimeError（错误信息含输出尾部，便于定位）。
    """
    log_fh = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_file, "a", encoding="utf-8", errors="replace")
    tail: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if cancel is not None and cancel():
                proc.terminate()
                raise InterruptedError("安装已取消")
            tail.append(line.rstrip("\n"))
            if log_fh:
                log_fh.write(line)
                log_fh.flush()
            tail = tail[-200:]
        rc = proc.wait(timeout=timeout)
        if rc != 0:
            snippet = "\n".join(tail[-30:])
            raise RuntimeError(
                f"命令失败（退出码 {rc}）: {' '.join(cmd[:6])}...\n"
                f"最后输出:\n{snippet}")
        return proc
    finally:
        if log_fh:
            log_fh.close()
