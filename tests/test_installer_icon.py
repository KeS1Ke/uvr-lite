# coding: utf-8
"""安装器图标路径（打包态 _MEIPASS 快照 vs 开发态项目根）测试。"""

import sys
from pathlib import Path

import pytest


def test_icon_dev_mode(tmp_path):
    from installer.main import _find_icon

    assert _find_icon().name == "uvr-lite.ico"
    assert _find_icon().exists()  # 开发态：项目根真实图标


def test_icon_packaged_mode(monkeypatch, tmp_path):
    """打包态：图标在 _MEIPASS/app-snapshot/uvr_lite/ui/resources/ 下。"""
    from installer.main import _find_icon

    fake = tmp_path / "meipass"
    ico_dir = fake / "app-snapshot" / "uvr_lite" / "ui" / "resources"
    ico_dir.mkdir(parents=True)
    (ico_dir / "uvr-lite.ico").write_bytes(b"ico")
    monkeypatch.setattr(sys, "_MEIPASS", str(fake), raising=False)
    assert _find_icon() == ico_dir / "uvr-lite.ico"


def test_icon_packaged_falls_back_to_dev(monkeypatch, tmp_path):
    """打包态快照里缺图标时回退开发态路径（不崩溃）。"""
    from installer.main import _find_icon

    fake = tmp_path / "meipass"
    (fake / "app-snapshot").mkdir(parents=True)  # 无 uvr_lite/ui/resources
    monkeypatch.setattr(sys, "_MEIPASS", str(fake), raising=False)
    assert _find_icon().name == "uvr-lite.ico"
