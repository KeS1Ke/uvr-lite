# coding: utf-8
"""Python 环境：系统 Python 检测 / 绿色 Python 下载（固定版本+SHA256+断点续传）/ venv。"""

import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from uvr_lite import download  # 复用断点续传 + 多源回退
from .consts import GREEN_PY_FILENAME, GREEN_PY_SHA256, GREEN_PY_URLS, MIN_PYTHON
from .proc import run as _run

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def parse_version(text: str) -> Optional[Tuple[int, int, int]]:
    m = _VERSION_RE.search(text or "")
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def python_version(exe: Path) -> Optional[Tuple[int, int, int]]:
    out = subprocess.run(
        [str(exe), "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True, text=True, timeout=15)
    return parse_version(out.stdout)


def find_system_python() -> Optional[Path]:
    """检测系统 Python ≥ 3.10（python / py -3 / python3），没有则 None。"""
    candidates: List[List[str]] = [["python"], ["py", "-3"], ["python3"]]
    for cand in candidates:
        exe = shutil.which(cand[0])
        if not exe:
            continue
        cmd = [exe] + cand[1:] + ["-c", "import sys; print(sys.version.split()[0])"]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        ver = parse_version(out.stdout)
        if ver and ver >= MIN_PYTHON:
            return Path(exe)
    return None


def download_green_python(dest_dir: Path,
                          progress_callback: Optional[Callable[[int, int], bool]] = None,
                          cancel: Optional[Callable[[], bool]] = None) -> Path:
    """下载并解压绿色 Python 到 dest_dir/python/，返回 python.exe 路径。

    - 断点续传 + 多源回退（复用 uvr_lite.download._download）
    - 解压前校验官方 SHA256，不匹配则删除并报错
    - install_only 包顶层是 python/ 目录，解压时去掉该前缀
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / GREEN_PY_FILENAME
    if cancel is not None and cancel():
        raise InterruptedError("安装已取消")

    def cb(done: int, total: int) -> bool:
        if cancel is not None and cancel():
            return False
        if progress_callback is not None:
            return progress_callback(done, total)
        return True

    download._download(GREEN_PY_URLS, archive, cb)
    actual = download.sha256_of(archive)
    if actual != GREEN_PY_SHA256:
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"绿色 Python 校验失败: 期望 {GREEN_PY_SHA256[:16]}…，实际 {actual[:16]}…。"
            "下载源可能被篡改，已删除下载文件。")

    python_dir = dest_dir / "python"
    python_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("python/"):
                name = name[len("python/"):]
            elif name == "python":
                continue
            if not name:
                continue
            target = python_dir / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(target, "wb") as f:
                    while chunk := src.read(1 << 20):
                        f.write(chunk)
    archive.unlink(missing_ok=True)
    return python_dir / "python.exe"


def ensure_venv(python_exe: Path, venv_dir: Path,
                log_file: Optional[Path] = None,
                cancel: Optional[Callable[[], bool]] = None) -> Path:
    """创建（或复用已存在的）虚拟环境，返回 venv 的 python.exe。"""
    venv_py = venv_dir / "Scripts" / "python.exe"
    if venv_py.exists():
        return venv_py
    _run([str(python_exe), "-m", "venv", str(venv_dir)],
         log_file=log_file, cancel=cancel)
    return venv_py


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "python.exe"


def venv_pip(venv_dir: Path) -> Path:
    return venv_dir / "Scripts" / "pip.exe"
