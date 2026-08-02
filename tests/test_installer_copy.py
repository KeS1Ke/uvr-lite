# coding: utf-8
"""代码快照复制测试。"""

from pathlib import Path

from installer.copy_app import _INCLUDE, copy_app_source


def test_copy_app_source_basic(tmp_path):
    src = tmp_path / "src"
    (src / "uvr_lite").mkdir(parents=True)
    (src / "uvr_lite" / "engine.py").write_text("x")
    (src / "msst").mkdir()
    (src / "msst" / "model.py").write_text("y")
    (src / "pyproject.toml").write_text("z")
    dest = tmp_path / "dest" / "app"

    count = copy_app_source(src, dest)
    assert (dest / "uvr_lite" / "engine.py").exists()
    assert (dest / "msst" / "model.py").exists()
    assert (dest / "pyproject.toml").exists()
    assert count == 3


def test_copy_app_source_excludes_dev_artifacts(tmp_path):
    src = tmp_path / "src"
    for rel in [
        "uvr_lite/__pycache__/x.pyc",
        "uvr_lite/ui/resources/a.ico",
        ".venv/Scripts/python.exe",
        ".git/config",
        "tests/test_x.py",
        "models/model.ckpt",
        "graphify-out/graph.json",
        "installer/main.py",
        "scripts/make_icon.py",
        "docs/CONTEXT.md",
        "uvr_lite/engine.py",
    ]:
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    dest = tmp_path / "dest" / "app"
    copy_app_source(src, dest)

    assert (dest / "uvr_lite" / "engine.py").exists()
    assert (dest / "uvr_lite" / "ui" / "resources" / "a.ico").exists()  # 包资源必须复制
    for excluded in [
        "uvr_lite/__pycache__", ".venv", ".git", "tests", "models",
        "graphify-out", "installer", "scripts", "docs",
    ]:
        assert not (dest / excluded).exists(), f"应排除 {excluded}"


def test_copy_app_source_clean_rebuild(tmp_path):
    src = tmp_path / "src"
    (src / "uvr_lite").mkdir(parents=True)
    (src / "uvr_lite" / "engine.py").write_text("v1")
    dest = tmp_path / "dest" / "app"
    copy_app_source(src, dest)
    (dest / "uvr_lite" / "stale.py").write_text("old")
    (src / "uvr_lite" / "engine.py").write_text("v2")

    copy_app_source(src, dest)
    assert not (dest / "uvr_lite" / "stale.py").exists()  # 旧文件被清掉
    assert (dest / "uvr_lite" / "engine.py").read_text() == "v2"


def test_copy_app_source_missing_item_skipped(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    dest = tmp_path / "dest" / "app"
    count = copy_app_source(src, dest)  # _INCLUDE 全缺失 → 不报错
    assert count == 0
    assert dest.exists()


def test_include_list_has_required_entries():
    assert "uvr_lite" in _INCLUDE
    assert "pyproject.toml" in _INCLUDE
