"""音质三档纯函数：不启动 Qt，也不应因此导入 torch。"""

import subprocess
import sys

from uvr_lite.ui.presets import matching_preset, resolve_quality


def test_fast_ignores_manual_values():
    got = resolve_quality("fast", overlap=8, bigshifts=8, tta=True)
    assert got["num_overlap"] == 1
    assert got["bigshifts"] == 1
    assert got["tta"] is False


def test_standard_overlap_is_two_not_none():
    got = resolve_quality("standard", overlap=0, bigshifts=0, tta=True)
    assert got["num_overlap"] == 2
    assert got["num_overlap"] is not None
    assert got["bigshifts"] == 1
    assert got["tta"] is False


def test_high_quality():
    got = resolve_quality("high", overlap=1, bigshifts=1, tta=True)
    assert got["num_overlap"] == 2
    assert got["bigshifts"] == 2
    assert got["tta"] is False


def test_custom_overlap_zero_means_model_default():
    got = resolve_quality("custom", overlap=0, bigshifts=2, tta=True)
    assert got["num_overlap"] is None
    assert got["bigshifts"] == 2
    assert got["tta"] is True


def test_custom_overlap_three():
    got = resolve_quality("custom", overlap=3, bigshifts=4, tta=False)
    assert got["num_overlap"] == 3
    assert got["bigshifts"] == 4
    assert got["tta"] is False


def test_custom_bigshifts_at_least_one():
    assert resolve_quality("custom", overlap=2, bigshifts=0, tta=False)["bigshifts"] == 1
    assert resolve_quality("custom", overlap=2, bigshifts=-3, tta=False)["bigshifts"] == 1


def test_unknown_preset_falls_back_to_standard():
    got = resolve_quality("nope", overlap=0, bigshifts=0, tta=True)
    assert got["num_overlap"] == 2
    assert got["num_overlap"] is not None
    assert got["bigshifts"] == 1
    assert got["tta"] is False


def test_import_presets_does_not_load_torch_or_qt():
    code = (
        "import sys\n"
        "import uvr_lite.ui.presets\n"
        "bad = [m for m in ('torch', 'PySide6') if m in sys.modules]\n"
        "sys.exit(0 if not bad else 1)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]


def test_matching_preset_maps_three_tiers():
    assert matching_preset(1, 1, False) == "fast"
    assert matching_preset(2, 1, False) == "standard"
    assert matching_preset(2, 2, False) == "high"


def test_matching_preset_legacy_default_is_standard():
    """旧版默认 num_overlap=0（沿用模型配置）+ bigshifts=1 + 无 TTA → 标准档。"""
    assert matching_preset(0, 1, False) == "standard"
    assert matching_preset(-1, 1, False) == "standard"


def test_matching_preset_custom_cases():
    assert matching_preset(2, 1, True) == "custom"      # 开了 TTA
    assert matching_preset(3, 4, False) == "custom"     # 数值对不上任何档
    assert matching_preset(0, 2, False) == "custom"     # 0 只在旧默认（bigshifts=1）才归标准
    assert matching_preset(None, 1, False) == "custom"  # 非法输入

