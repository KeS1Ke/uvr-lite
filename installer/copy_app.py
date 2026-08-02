# coding: utf-8
"""代码快照复制：项目根 → 安装目录/app（排除开发/测试/缓存产物）。"""

import shutil
from pathlib import Path
from typing import Iterable, Set

# 需要复制进安装目录的顶层项（其余一律不复制）
_INCLUDE = ("uvr_lite", "msst", "pyproject.toml", "README.md")
# 复制过程中跳过的目录/后缀
_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "models", "node_modules",
                 ".pytest_cache", "tests", "installer", "scripts", "docs",
                 "graphify-out", ".idea", ".vscode", "build", "dist",
                 ".venv-Scripts", "__pycache__"}
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".egg-info")


def _ignore(dirpath: str, names: list) -> Set[str]:
    return {n for n in names
            if n in _EXCLUDE_DIRS
            or n.endswith(_EXCLUDE_SUFFIXES)
            or (Path(dirpath) / n).is_dir() and n.startswith(".")}


def copy_app_source(src: Path, dest: Path) -> int:
    """复制代码快照到 dest；返回复制的文件数。

    dest 已有内容时整体清空后重建（覆盖升级时替换旧代码）。
    """
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    count = 0
    for item in _INCLUDE:
        sp = src / item
        if not sp.exists():
            continue
        dp = dest / item
        if sp.is_dir():
            shutil.copytree(sp, dp, ignore=_ignore)
            count += sum(1 for _ in dp.rglob("*") if _.is_file())
        else:
            shutil.copy2(sp, dp)
            count += 1
    return count
