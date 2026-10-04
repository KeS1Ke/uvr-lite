"""音轨配对：文件名识别 + 成对逻辑（CLI 批量与 UI 合成页共用）。"""

from pathlib import Path

from uvr_lite.stems import find_stem_pairs, pair_stems, split_stem


def test_split_stem_recognizes_engine_suffixes():
    assert split_stem(Path("song-vocals.flac")) == ("song", "vocals")
    assert split_stem(Path("song-instrumental.wav")) == ("song", "instrumental")
    assert split_stem(Path("song-inst.flac")) == ("song", "instrumental")
    assert split_stem(Path("song_vocals.ogg")) == ("song", "vocals")


def test_split_stem_case_insensitive_but_preserves_base():
    assert split_stem(Path("Song-Vocals.FLAC")) == ("Song", "vocals")


def test_split_stem_rejects_unrelated_names():
    assert split_stem(Path("song.flac")) is None
    assert split_stem(Path("-vocals.flac")) is None  # 只剩后缀，无基名


def test_pair_stems_groups_by_base_across_extensions(tmp_path):
    vocals = tmp_path / "song-vocals.flac"
    inst = tmp_path / "song-instrumental.wav"
    other_v = tmp_path / "other-vocals.flac"

    result = pair_stems([vocals, inst, other_v])

    assert [(p.vocals, p.instrumental) for p in result.pairs] == [(vocals, inst)]
    assert result.unmatched == [other_v]
    assert result.ignored == []


def test_pair_stems_ignored_and_unmatched(tmp_path):
    result = pair_stems([
        tmp_path / "plain.flac",
        tmp_path / "lonely-vocals.flac",
        tmp_path / "song-vocals.flac",
        tmp_path / "song-instrumental.flac",
    ])

    assert len(result.pairs) == 1
    assert [p.name for p in result.unmatched] == ["lonely-vocals.flac"]
    assert [p.name for p in result.ignored] == ["plain.flac"]


def test_pair_stems_duplicate_kind_keeps_extras_unmatched(tmp_path):
    """同一基名出现两份 vocals（flac/wav 副本）时不静默丢弃，多余项进 unmatched。"""
    result = pair_stems([
        tmp_path / "song-vocals.flac",
        tmp_path / "song-vocals.wav",
        tmp_path / "song-instrumental.flac",
    ])

    assert len(result.pairs) == 1
    assert result.pairs[0].vocals.name == "song-vocals.flac"
    assert [p.name for p in result.unmatched] == ["song-vocals.wav"]


def test_pair_stems_equal_counts_keeps_only_first_pair(tmp_path):
    """两边数量相等时也只配名称排序后的第一对，避免交叉写成同一个 mix 路径。"""
    result = pair_stems([
        tmp_path / "song-vocals.flac",
        tmp_path / "song-vocals.wav",
        tmp_path / "song-inst.wav",
        tmp_path / "song-instrumental.flac",
    ])

    assert len(result.pairs) == 1
    assert result.pairs[0].vocals.name == "song-vocals.flac"
    assert result.pairs[0].instrumental.name == "song-inst.wav"
    assert {p.name for p in result.unmatched} == {
        "song-instrumental.flac",
        "song-vocals.wav",
    }

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    result_dirs = pair_stems([
        dir_a / "song-vocals.flac",
        dir_a / "song-instrumental.flac",
        dir_b / "song-vocals.wav",
        dir_b / "song-instrumental.wav",
    ])
    assert len(result_dirs.pairs) == 2
    assert result_dirs.unmatched == []


def test_pair_stems_groups_relative_and_absolute_same_dir(tmp_path, monkeypatch):
    """相对路径与绝对路径、以及带 .. 的路径，resolve 后同一目录要分在一组。"""
    monkeypatch.chdir(tmp_path)
    album = tmp_path / "album"
    album.mkdir()
    rel = Path("album") / "song-vocals.flac"
    abs_inst = (album / "song-instrumental.wav").resolve()

    result = pair_stems([rel, abs_inst])

    assert len(result.pairs) == 1
    assert result.pairs[0].vocals == rel
    assert result.pairs[0].instrumental == abs_inst
    assert result.unmatched == []

    nested = album / "nested"
    via_parent = nested / ".." / "other-vocals.flac"
    direct = album / "other-instrumental.flac"
    assert str(via_parent.parent) != str(direct.parent)
    dotted = pair_stems([via_parent, direct])
    assert len(dotted.pairs) == 1
    assert dotted.unmatched == []

    # 调用方已经 resolve 过的路径，配对结果不变
    resolved = pair_stems([
        (album / "song-vocals.flac").resolve(),
        (album / "song-instrumental.wav").resolve(),
    ])
    assert len(resolved.pairs) == 1
    assert resolved.pairs[0].vocals == (album / "song-vocals.flac").resolve()
    assert resolved.unmatched == []


def test_find_stem_pairs_scans_folder(tmp_path):
    (tmp_path / "a-vocals.flac").write_bytes(b"x")
    (tmp_path / "a-instrumental.flac").write_bytes(b"x")
    (tmp_path / "b-vocals.flac").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c-vocals.flac").write_bytes(b"x")
    (sub / "c-instrumental.flac").write_bytes(b"x")

    result = find_stem_pairs(tmp_path)

    assert [p.vocals.name for p in result.pairs] == ["a-vocals.flac"]
    assert [p.name for p in result.unmatched] == ["b-vocals.flac"]
    assert result.ignored == [], "txt 不是音频，不进配对输入"
