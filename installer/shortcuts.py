# coding: utf-8
"""快捷方式：桌面 + 开始菜单（♪ 图标），PowerShell COM 创建/删除 .lnk。"""

import shutil
import subprocess
from pathlib import Path
from typing import Dict

from PySide6.QtCore import QStandardPaths

_APP_ID = "uvr-lite"


def desktop_dir() -> Path:
    locs = QStandardPaths.standardLocations(QStandardPaths.DesktopLocation)
    return Path(locs[0]) if locs else Path.home() / "Desktop"


def start_menu_dir() -> Path:
    root = QStandardPaths.writableLocation(QStandardPaths.ApplicationsLocation)
    return Path(root) / "uvr-lite" if root else Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "uvr-lite"


def _ps_quote(s: str) -> str:
    """PowerShell 单引号字符串转义（' → ''）。"""
    return "'" + str(s).replace("'", "''") + "'"


def shortcut_spec(install_dir: Path) -> Dict[str, Path]:
    """快捷方式相关路径与参数（不创建）。"""
    app_dir = install_dir / "app"
    return {
        "desktop_lnk": desktop_dir() / "uvr-lite.lnk",
        "start_lnk": start_menu_dir() / "uvr-lite.lnk",
        "pythonw": install_dir / ".venv" / "Scripts" / "pythonw.exe",
        "icon": app_dir / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico",
        "workdir": app_dir,
        "args": f'-m uvr_lite.ui --model-dir "{install_dir / "models"}"',
    }


def create_shortcut(lnk: Path, target: Path, args: str,
                    icon: Path, workdir: Path) -> None:
    """创建单个 .lnk（WScript.Shell COM）。"""
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut({_ps_quote(str(lnk))}); "
        f"$s.TargetPath = {_ps_quote(str(target))}; "
        f"$s.Arguments = {_ps_quote(args)}; "
        f"$s.IconLocation = {_ps_quote(str(icon) + ',0')}; "
        f"$s.WorkingDirectory = {_ps_quote(str(workdir))}; "
        f"$s.Description = {_ps_quote(_APP_ID)}; "
        "$s.Save()"
    )
    lnk.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        check=True, capture_output=True, timeout=60)


def create_shortcuts(install_dir: Path) -> None:
    spec = shortcut_spec(install_dir)
    create_shortcut(spec["desktop_lnk"], spec["pythonw"], spec["args"],
                    spec["icon"], spec["workdir"])
    create_shortcut(spec["start_lnk"], spec["pythonw"], spec["args"],
                    spec["icon"], spec["workdir"])


def remove_shortcuts() -> None:
    """删除桌面与开始菜单快捷方式（卸载用）。"""
    for lnk in (desktop_dir() / "uvr-lite.lnk", start_menu_dir() / "uvr-lite.lnk"):
        lnk.unlink(missing_ok=True)
    start = start_menu_dir()
    if start.exists():
        shutil.rmtree(start, ignore_errors=True)
