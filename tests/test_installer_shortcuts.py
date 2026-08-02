# coding: utf-8
"""快捷方式测试：ps 转义、spec 构造、.lnk 创建/删除（Windows 实跑）。"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from installer import shortcuts


def test_ps_quote():
    assert shortcuts._ps_quote("plain") == "'plain'"
    assert shortcuts._ps_quote("D:\\a'b\\c") == "'D:\\a''b\\c'"


def test_shortcut_spec(tmp_path):
    spec = shortcuts.shortcut_spec(tmp_path)
    assert spec["pythonw"] == tmp_path / ".venv" / "Scripts" / "pythonw.exe"
    assert spec["icon"] == tmp_path / "app" / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico"
    assert '--model-dir' in spec["args"]
    assert str(tmp_path / "models") in spec["args"]
    assert spec["desktop_lnk"].name == "uvr-lite.lnk"
    assert spec["start_lnk"].parent.name == "uvr-lite"


def _read_lnk(lnk: Path):
    """用 PowerShell COM 读取 .lnk 属性。"""
    script = (
        "$ws = New-Object -ComObject WScript.Shell; "
        f"$s = $ws.CreateShortcut({shortcuts._ps_quote(str(lnk))}); "
        'Write-Output $s.TargetPath; Write-Output $s.Arguments; Write-Output $s.IconLocation'
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True, text=True, timeout=60, check=True)
    lines = out.stdout.strip().splitlines()
    return {"target": lines[0], "args": lines[1], "icon": lines[2]}


@pytest.mark.skipif(sys.platform != "win32", reason=".lnk 仅 Windows")
def test_create_shortcut_roundtrip(tmp_path):
    lnk = tmp_path / "test-uvr.lnk"
    icon = Path(__file__).resolve().parent.parent / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico"
    shortcuts.create_shortcut(lnk, Path("C:/Windows/System32/pythonw.exe"),
                              '-m uvr_lite.ui --model-dir "D:/x/models"',
                              icon, Path("D:/x/app"))
    data = _read_lnk(lnk)
    assert data["target"] == "C:\\Windows\\System32\\pythonw.exe"
    assert "uvr_lite.ui" in data["args"]
    assert "D:/x/models" in data["args"]
    assert data["icon"].lower().startswith(str(icon).lower())


@pytest.mark.skipif(sys.platform != "win32", reason=".lnk 仅 Windows")
def test_remove_shortcuts_cleanup(tmp_path, monkeypatch):
    lnk = tmp_path / "uvr-lite.lnk"
    monkeypatch.setattr(shortcuts, "desktop_dir", lambda: tmp_path)
    monkeypatch.setattr(shortcuts, "start_menu_dir", lambda: tmp_path / "start")
    icon = Path(__file__).resolve().parent.parent / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico"
    shortcuts.create_shortcut(lnk, Path("C:/x/pythonw.exe"), "", icon, tmp_path)
    (tmp_path / "start").mkdir()
    (tmp_path / "start" / "uvr-lite.lnk").write_text("x")

    shortcuts.remove_shortcuts()
    assert not lnk.exists()
    assert not (tmp_path / "start").exists()
