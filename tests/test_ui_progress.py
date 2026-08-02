# coding: utf-8
"""票 3（tdd）：推理接线的纯逻辑——文件内进度映射、ETA 估算、失败汇总。"""

import pytest

from uvr_lite.ui.progress import ProgressTracker, estimate_eta, summary_text


# ---------- ProgressTracker：阶段 → 文件内进度 0..1 ----------

def test_decode_phase():
    t = ProgressTracker()
    assert t.on_progress("decode", 0, 1) == 0.0
    assert t.on_progress("decode", 1, 1) == pytest.approx(0.05)


def test_bigshifts1_chunk_then_infer():
    t = ProgressTracker(bigshifts=1)
    assert t.on_progress("chunk", 1, 3) == pytest.approx(0.20)
    assert t.on_progress("chunk", 2, 3) == pytest.approx(0.35)
    assert t.on_progress("chunk", 3, 3) == pytest.approx(0.50)
    assert t.on_progress("infer", 1, 1) == pytest.approx(0.50)  # 单 pass 完成，与 chunk 终点衔接


def test_bigshifts4_multi_pass_monotonic():
    t = ProgressTracker(bigshifts=4)
    seq = [
        ("chunk", 1, 3), ("chunk", 3, 3),   # pass 1 内部
        ("infer", 1, 4),                     # pass 1 完成
        ("chunk", 1, 3), ("chunk", 3, 3),   # pass 2 内部（应从更高起点继续）
        ("infer", 4, 4),                     # 全部 pass 完成
    ]
    values = [t.on_progress(p, d, tot) for p, d, tot in seq]
    assert values == sorted(values), f"跨 pass 进度必须单调: {values}"
    assert values[3] < values[4], "pass 2 的 chunk 应从 pass 1 之后继续"
    assert values[-1] == pytest.approx(0.50)


def test_tta_and_write():
    t = ProgressTracker()
    assert t.on_progress("tta", 1, 2) == pytest.approx(0.70)
    assert t.on_progress("tta", 2, 2) == pytest.approx(0.90)
    assert t.on_progress("write", 1, 2) == pytest.approx(0.95)
    assert t.on_progress("write", 2, 2) == pytest.approx(1.00)


def test_unknown_phase_returns_zero():
    assert ProgressTracker().on_progress("unknown", 1, 1) == 0.0


# ---------- estimate_eta ----------

def test_eta_empty_history():
    assert estimate_eta([], 0, 1, 0.0) is None


def test_eta_remaining_files_and_current_fraction():
    # 已完成 2 首各 60s；当前第 3 首完成 50%；共 5 首
    # 剩余 = 2 首未开始 + 当前 50% = 2.5 首 → 150s
    assert estimate_eta([60.0, 60.0], done=2, total=5, file_pct=0.5) == pytest.approx(150.0)


def test_eta_last_file():
    assert estimate_eta([30.0], done=0, total=1, file_pct=0.25) == pytest.approx(22.5)


def test_eta_pct_clamped():
    assert estimate_eta([10.0], done=0, total=1, file_pct=2.0) == 0.0


# ---------- summary_text ----------

def test_summary_all_ok():
    assert summary_text(2, []) == "全部成功：2 个文件。"


def test_summary_with_failures():
    s = summary_text(2, ["坏文件.wav", "另一首.mp3"])
    assert "成功 2 个，失败 2 个" in s
    assert "坏文件.wav" in s and "另一首.mp3" in s
