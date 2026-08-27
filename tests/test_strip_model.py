"""strip_model：fp16 瘦身 + safetensors 输出格式。"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # scripts/ 非包，从根目录按命名空间包导入

from scripts.strip_model import strip_ckpt  # noqa: E402


def test_strip_safetensors_output(tmp_path):
    """safetensors 输出：纯张量扁平存储，非张量键被跳过。"""
    src = tmp_path / "src.ckpt"
    torch.save({"state_dict": {
        "w": torch.randn(4, 4, dtype=torch.float32),
        "b": torch.zeros(4, dtype=torch.float32),
        "extra": "metadata",  # 非张量 → safetensors 无法存储
    }}, src)
    dst = tmp_path / "out.safetensors"

    strip_ckpt(src, dst, fmt="safetensors")

    from safetensors.torch import load_file

    sd = load_file(str(dst))
    assert set(sd) == {"w", "b"}
    assert sd["w"].dtype == torch.float16, "应转 fp16 存储"
    assert sd["w"].shape == (4, 4)


def test_strip_ckpt_output_keeps_wrapper(tmp_path):
    """ckpt 输出保持 {"state_dict": ...} 封装（兼容 msst 加载链）。"""
    src = tmp_path / "src.ckpt"
    torch.save({"state_dict": {"w": torch.randn(2, 2)}}, src)
    dst = tmp_path / "out.ckpt"

    strip_ckpt(src, dst, fmt="ckpt")

    ckpt = torch.load(dst, map_location="cpu", weights_only=True)
    assert set(ckpt) == {"state_dict"}
    assert ckpt["state_dict"]["w"].dtype == torch.float16
