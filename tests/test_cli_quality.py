"""CLI separate --quality：质量档变成重叠数，且不与 --num-overlap 静默覆盖。"""

import argparse
import sys
import types

from uvr_lite.cli import build_parser, main


def _patch_separator(monkeypatch):
    """挡住 engine（以及 torch），只记录 Separator 收到的参数。"""
    seen = {}

    class Separator:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def separate(self, *args, **kwargs):
            seen.setdefault("files", []).append(args[0] if args else None)
            return []

    eng = types.ModuleType("uvr_lite.engine")
    eng.Separator = Separator
    monkeypatch.setitem(sys.modules, "uvr_lite.engine", eng)
    return seen


def test_quality_help_lists_overlap_and_cpu_hint():
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    text = sub.choices["separate"]._option_string_actions["--quality"].help
    assert "重叠 1" in text
    assert "重叠 2" in text
    assert "重叠 4" in text
    assert "快适合 CPU、高更干净" in text


def test_module_doc_has_no_version_subcommand():
    import uvr_lite.cli as cli

    assert " / version" not in cli.__doc__
    assert "--version" in cli.__doc__


def test_quality_sets_overlap(monkeypatch, tmp_path):
    seen = _patch_separator(monkeypatch)

    assert main(["separate", "song.wav", "--quality", "fast", "-o", str(tmp_path)]) == 0
    assert seen["init"]["num_overlap"] == 1

    assert main(["separate", "song.wav", "--quality", "standard", "-o", str(tmp_path)]) == 0
    assert seen["init"]["num_overlap"] == 2

    assert main(["separate", "song.wav", "--quality", "high", "-o", str(tmp_path)]) == 0
    assert seen["init"]["num_overlap"] == 4
    assert seen["init"]["num_overlap"] is not None


def test_quality_alone_does_not_leave_overlap_none(monkeypatch, tmp_path):
    seen = _patch_separator(monkeypatch)

    assert main(["separate", "song.wav", "--quality", "fast", "-o", str(tmp_path)]) == 0

    assert seen["init"]["num_overlap"] == 1


def test_num_overlap_alone_still_passed(monkeypatch, tmp_path):
    seen = _patch_separator(monkeypatch)

    assert main(["separate", "song.wav", "--num-overlap", "3", "-o", str(tmp_path)]) == 0

    assert seen["init"]["num_overlap"] == 3


def test_neither_flag_leaves_overlap_none(monkeypatch, tmp_path):
    seen = _patch_separator(monkeypatch)

    assert main(["separate", "song.wav", "-o", str(tmp_path)]) == 0

    assert seen["init"]["num_overlap"] is None


def test_quality_and_num_overlap_exit_2(monkeypatch, capsys, tmp_path):
    seen = _patch_separator(monkeypatch)

    code = main([
        "separate", "song.wav",
        "--quality", "high", "--num-overlap", "2",
        "-o", str(tmp_path),
    ])

    assert code == 2
    assert "init" not in seen, "冲突时不能静默覆盖，也不该去加载模型"
    out = capsys.readouterr().out
    assert "--quality" in out and "--num-overlap" in out
    assert "不能同时" in out
