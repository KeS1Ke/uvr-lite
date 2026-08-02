# coding: utf-8
"""票 2：输入文件扫描（选择文件夹添加方式）的纯函数测试。"""

from pathlib import Path

from uvr_lite.ui.files import dedup_paths, is_audio, scan_audio_files


def test_is_audio_extensions():
    assert is_audio(Path("a.mp3"))
    assert is_audio(Path("a.FLAC"))  # 大小写不敏感
    assert is_audio(Path("a.wav")) and is_audio(Path("a.ogg")) and is_audio(Path("a.m4a"))
    assert not is_audio(Path("a.txt"))
    assert not is_audio(Path("a.mp4"))


def test_scan_audio_files_filters_and_sorts(tmp_path):
    (tmp_path / "b.wav").write_bytes(b"x")
    (tmp_path / "a.flac").write_bytes(b"x")
    (tmp_path / "note.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.mp3").write_bytes(b"x")  # 非递归：不应被扫到

    found = scan_audio_files(tmp_path)
    names = [p.name for p in found]
    assert names == ["a.flac", "b.wav"], f"应过滤扩展名且按名称排序: {names}"
    assert all(p.is_absolute() for p in found)
    assert not any(p.parent == sub for p in found)


def test_scan_audio_files_missing_dir():
    assert scan_audio_files(Path("不存在/的/目录")) == []


def test_dedup_paths_preserves_order(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = dedup_paths([a, b, a, tmp_path / "a.wav"])
    assert out == [a.resolve(), b.resolve()]
