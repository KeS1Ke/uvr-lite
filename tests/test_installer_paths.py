# coding: utf-8
"""安装器路径解析：浏览选择位置后自动补 uvr-lite 子目录，防止文件散落。"""

from pathlib import Path

from installer.main import resolve_install_path


def test_appends_uvr_lite_subdir(tmp_path):
    # 选了 D:\ 这类父目录 → 自动补 \uvr-lite
    assert resolve_install_path(tmp_path) == tmp_path / "uvr-lite"


def test_keeps_dir_already_named_uvr_lite(tmp_path):
    # 所选目录本身就叫 uvr-lite → 不追加
    target = tmp_path / "uvr-lite"
    target.mkdir()
    assert resolve_install_path(target) == target


def test_keeps_existing_install_dir(tmp_path):
    # 所选目录已是安装目录（install.json 存在，覆盖升级）→ 不追加
    target = tmp_path / "uvr-lite"
    target.mkdir()
    (target / "install.json").write_text("{}", encoding="utf-8")
    assert resolve_install_path(target) == target


def test_resolves_relative_input(tmp_path):
    assert resolve_install_path(tmp_path / ".") == (tmp_path / "uvr-lite").resolve()
