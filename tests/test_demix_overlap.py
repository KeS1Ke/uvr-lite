"""票 2：demix 逐块窗口 + 尾批补齐，batch_size>1 不应改变输出。

运行（仓库根目录）: python -m pytest tests/test_demix_overlap.py -v
"""

import sys
from pathlib import Path

import numpy as np
import torch.nn as nn
from ml_collections import ConfigDict

# 与 tests/test_engine_progress.py 相同：msst 内模块按顶层包导入
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "msst") not in sys.path:
    sys.path.insert(0, str(ROOT / "msst"))

from utils.model_utils import demix  # noqa: E402


class FakeModel(nn.Module):
    """逐样本恒等： (B, C, T) → (B, 1, C, T)。补零行不影响真实样本。"""

    def forward(self, x):
        return x.unsqueeze(1)


def demix_config(batch_size: int) -> ConfigDict:
    return ConfigDict({
        "training": {"use_amp": False, "instruments": ["vocals"]},
        "inference": {"chunk_size": 4096, "num_overlap": 2, "batch_size": batch_size},
        "audio": {"chunk_size": 4096},
    })


def test_demix_batch_padding_and_per_chunk_window_match():
    # 10000 samples → pad 后 7 块，batch=4 的末批只有 3 块，会走补零
    mix = np.random.RandomState(0).randn(2, 10000).astype(np.float32)
    out1 = demix(demix_config(1), FakeModel(), mix, "cpu", "bs_roformer", pbar=False)
    out4 = demix(demix_config(4), FakeModel(), mix, "cpu", "bs_roformer", pbar=False)

    for out in (out1, out4):
        vocals = out["vocals"]
        assert vocals.shape == (2, mix.shape[-1])
        assert vocals.ndim == 2
        assert np.isfinite(vocals).all()

    np.testing.assert_allclose(out1["vocals"], out4["vocals"], atol=1e-4, rtol=0)
