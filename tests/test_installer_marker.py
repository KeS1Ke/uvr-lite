# coding: utf-8
"""install.json 标记读写测试。"""

from installer import marker


def test_marker_write_read_roundtrip(tmp_path):
    p = marker.write_marker(tmp_path, app_version="0.2.0", torch="2.7.1")
    assert p == marker.marker_path(tmp_path)
    data = marker.read_marker(tmp_path)
    assert data["app"] == "uvr-lite"
    assert data["app_version"] == "0.2.0"
    assert data["torch"] == "2.7.1"
    assert "installed_at" in data


def test_is_install_dir(tmp_path):
    assert not marker.is_install_dir(tmp_path)
    marker.write_marker(tmp_path)
    assert marker.is_install_dir(tmp_path)


def test_read_marker_missing(tmp_path):
    assert marker.read_marker(tmp_path) == {}


def test_read_marker_wrong_app(tmp_path):
    (tmp_path / "install.json").write_text('{"app": "other"}', encoding="utf-8")
    assert marker.read_marker(tmp_path) == {}


def test_read_marker_corrupt(tmp_path):
    (tmp_path / "install.json").write_text("{not-json", encoding="utf-8")
    assert marker.read_marker(tmp_path) == {}
