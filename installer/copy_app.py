# coding: utf-8
"""代码快照复制：项目根 → 安装目录/app（排除开发/测试/缓存产物）。"""

import shutil
from pathlib import Path
from typing import Iterable, Set

# 需要复制进安装目录的顶层项（其余一律不复制）
_INCLUDE = ("uvr_lite", "msst", "pyproject.toml", "README.md")
# 复制过程中跳过的目录/后缀。
# 注意 "models" 只排除仓库根的权重目录；msst/models/ 是模型定义代码，必须保留。
_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "models", "node_modules",
                 ".pytest_cache", "tests", "installer", "scripts", "docs",
                 "graphify-out", ".idea", ".vscode", "build", "dist"}
_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".egg-info")


def _make_ignore(root: Path):
    """copytree ignore 回调：顶层（root 下）排除 _EXCLUDE_DIRS 全部；
    子目录不排除 "models"（msst/models 是代码）。"""
    def _ignore(dirpath: str, names: list) -> Set[str]:
        top = Path(dirpath) == root
        exclude = _EXCLUDE_DIRS if top else _EXCLUDE_DIRS - {"models"}
        return {n for n in names
                if n in exclude
                or n.endswith(_EXCLUDE_SUFFIXES)
                or (Path(dirpath) / n).is_dir() and n.startswith(".")}
    return _ignore


def copy_app_source(src: Path, dest: Path) -> int:
    """复制代码快照到 dest；返回复制的文件数。

    dest 已有内容时整体清空后重建（覆盖升级时替换旧代码）。
    目录被占用（如 uvr-lite 窗口正开着）时抛带提示的 RuntimeError。
    """
    if dest.exists():
        try:
            shutil.rmtree(dest)
        except OSError as e:
            raise RuntimeError(
                f"程序目录被占用，无法更新：{e}\n"
                "请先关闭正在运行的 uvr-lite 窗口，再重新安装。") from e
    dest.mkdir(parents=True)
    count = 0
    for item in _INCLUDE:
        sp = src / item
        if not sp.exists():
            continue
        dp = dest / item
        if sp.is_dir():
            shutil.copytree(sp, dp, ignore=_make_ignore(src))
            count += sum(1 for _ in dp.rglob("*") if _.is_file())
        else:
            shutil.copy2(sp, dp)
            count += 1
    return count
