"""demix 重叠窗：num_overlap=1 不得在块边界打出零样本。

不加载真实模型。假模型把输入原样送回，重叠加权后应还原输入。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ml_collections import ConfigDict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "msst") not in sys.path:
    sys.path.insert(0, str(ROOT / "msst"))

from utils.model_utils import _getWindowingArray, demix  # noqa: E402


class _Identity(nn.Module):
    """(batch, channels, time) -> (batch, 1, channels, time)。"""

    def forward(self, x):
        return x.unsqueeze(1)


def _config(chunk: int, overlap: int, batch: int = 1) -> ConfigDict:
    return ConfigDict({
        "training": {"use_amp": False, "instruments": ["vocals"]},
        "inference": {
            "chunk_size": chunk,
            "num_overlap": overlap,
            "batch_size": batch,
        },
        "audio": {"chunk_size": chunk},
    })


def _mix(n: int = 100) -> np.ndarray:
    """远离 0 的立体声，块边界被打成 0 时 allclose 会失败。"""
    mix = np.full((2, n), 0.4, dtype=np.float32)
    mix[0, ::7] = 0.25
    mix[1, 3::5] = 0.55
    return mix


def test_window_fade_zero_is_all_ones():
    window = _getWindowingArray(16, 0)
    assert torch.equal(window, torch.ones(16))
    assert float(window[0]) == 1.0 and float(window[-1]) == 1.0


def test_window_positive_fade_keeps_linear_ramp():
    """num_overlap>=2 仍用线性淡入淡出，端点为 0，中间为 1。"""
    window = _getWindowingArray(10, 3)
    expected = torch.tensor([0, 0.5, 1, 1, 1, 1, 1, 1, 0.5, 0], dtype=torch.float32)
    assert torch.allclose(window, expected)


def test_demix_overlap_one_matches_input_without_zero_edges():
    chunk = 32
    mix = _mix(100)
    out = demix(_config(chunk, 1), _Identity(), mix, "cpu", "bs_roformer", pbar=False)
    voc = out["vocals"]
    assert voc.shape == mix.shape
    assert np.isfinite(voc).all()
    assert np.allclose(voc, mix, atol=1e-5)
    # 每块交界的两个样本（旧 bug：窗端点为 0，nan_to_num 打成 0）
    for boundary in range(chunk, mix.shape[1], chunk):
        pair = voc[:, boundary - 1:boundary + 1]
        assert np.all(np.abs(pair) > 1e-3), f"overlap=1 zeroed the edge at {boundary}"
    assert np.all(np.abs(voc[:, 0]) > 1e-3)
    assert np.all(np.abs(voc[:, -1]) > 1e-3)


def test_demix_overlap_two_edges_are_finite():
    chunk = 40
    mix = _mix(100)
    out = demix(_config(chunk, 2), _Identity(), mix, "cpu", "bs_roformer", pbar=False)
    voc = out["vocals"]
    assert voc.shape == mix.shape
    assert np.isfinite(voc).all()
    assert not np.isnan(voc[:, :1]).any()
    assert not np.isnan(voc[:, -1:]).any()
    assert np.all(np.abs(voc[:, 0]) > 1e-3)
    assert np.all(np.abs(voc[:, -1]) > 1e-3)
    assert np.allclose(voc, mix, atol=1e-4)
