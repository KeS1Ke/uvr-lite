"""票 2：demix 逐块窗口 + 尾批补齐，batch_size>1 不应改变输出。

运行（仓库根目录）: python -m pytest tests/test_demix_overlap.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict

# 与 tests/test_engine_progress.py 相同：msst 内模块按顶层包导入
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "msst") not in sys.path:
    sys.path.insert(0, str(ROOT / "msst"))

from utils.model_utils import demix  # noqa: E402


class RampModel(nn.Module):
    """位置相关变换：(B, C, T) → (B, 1, C, T)，同时记录每次前向的 batch 维。

    恒等模型不能用：窗口权重会被 result/counter 归一化相消，旧实现的批级
    窗口 bug 也会「通过」。乘以逐位置斜坡后，窗口差异才会体现在输出上。
    """

    def __init__(self):
        super().__init__()
        self.batch_dims: list[int] = []

    def forward(self, x):
        self.batch_dims.append(x.shape[0])
        ramp = torch.linspace(0.5, 1.5, x.shape[-1], dtype=x.dtype, device=x.device)
        return (x * ramp).unsqueeze(1)


def demix_config(batch_size: int, num_overlap: int = 2) -> ConfigDict:
    return ConfigDict({
        "training": {"use_amp": False, "instruments": ["vocals"]},
        "inference": {"chunk_size": 4096, "num_overlap": num_overlap,
                      "batch_size": batch_size},
        "audio": {"chunk_size": 4096},
    })


def test_demix_batch_padding_and_per_chunk_window_match():
    # 10000 samples → pad 后 7 块，batch=4 的末批只有 3 块，会走补零
    mix = np.random.RandomState(0).randn(2, 10000).astype(np.float32)
    model1 = RampModel()
    model4 = RampModel()
    out1 = demix(demix_config(1), model1, mix, "cpu", "bs_roformer", pbar=False)
    out4 = demix(demix_config(4), model4, mix, "cpu", "bs_roformer", pbar=False)

    for out in (out1, out4):
        vocals = out["vocals"]
        assert vocals.shape == (2, mix.shape[-1])
        assert vocals.ndim == 2
        assert np.isfinite(vocals).all()

    # 尾批补齐后，模型每次前向的 batch 维都必须等于配置 batch_size
    assert model4.batch_dims == [4, 4], model4.batch_dims
    assert model1.batch_dims == [1] * 7, model1.batch_dims
    np.testing.assert_allclose(out1["vocals"], out4["vocals"], atol=1e-4, rtol=0)


def test_demix_overlap_one_has_no_zero_samples_at_chunk_boundaries():
    """num_overlap=1 无重叠：窗口端点不能再留下零样本（旧实现每块边界 2 个）。"""
    mix = np.random.RandomState(1).randn(2, 10000).astype(np.float32) + 0.5
    model = RampModel()
    out = demix(demix_config(1, num_overlap=1), model, mix, "cpu", "bs_roformer", pbar=False)

    vocals = out["vocals"]
    assert vocals.shape == mix.shape
    assert np.abs(vocals).min() > 1e-6, "块边界不应出现零样本"
