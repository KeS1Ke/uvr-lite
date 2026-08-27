"""权重加载安全模式回归：engine 必须以 weights_only=True 加载 ckpt。

fp16-lite 分发格式为纯张量（{"state_dict": {name: Tensor}}），无 pickle
任意代码执行面；若权重文件混入非张量对象（numpy 数组/自定义类），
torch.load(weights_only=True) 抛 UnpicklingError，engine 应转为可读错误。
"""

from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import torch

import uvr_lite.engine as engine

_CKPT = Path("models") / "bs_roformer_ep317.ckpt"


def test_default_model_ckpt_is_weights_only_safe():
    """注册表默认模型权重必须是纯张量格式（weights_only=True 可加载）。"""
    if not _CKPT.exists():
        pytest.skip("本地无默认模型权重（CI / 无模型环境跳过）")
    obj = torch.load(_CKPT, map_location="cpu", weights_only=True)
    assert isinstance(obj, dict) and "state_dict" in obj


def _patch_engine(monkeypatch):
    """load_model 依赖最小化：假模型 + 假配置，聚焦 torch.load 行为。"""
    monkeypatch.setattr(engine, "get_model_info",
                        lambda name: {"model_type": "bs_roformer", "config": "x.yaml"})
    monkeypatch.setattr(engine, "config_path", lambda name: "fake.yaml")
    monkeypatch.setattr(engine, "get_model_from_config",
                        lambda mt, cfg: (torch.nn.Identity(), SimpleNamespace(training={})))
    monkeypatch.setattr(engine, "load_start_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: None)


def test_engine_load_model_rejects_pickle_payload(tmp_path, monkeypatch):
    """含非张量对象的 ckpt → RuntimeError（不静默回退到 weights_only=False）。"""
    _patch_engine(monkeypatch)
    bad = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"w": numpy.zeros(3)}}, bad)  # numpy 不在安全白名单

    with pytest.raises(RuntimeError, match="安全模式加载失败"):
        engine.load_model("m", bad, "cpu")


def test_engine_load_model_accepts_pure_tensors(tmp_path, monkeypatch):
    """纯张量 ckpt 正常通过 weights_only=True 加载。"""
    _patch_engine(monkeypatch)
    good = tmp_path / "good.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(3)}}, good)

    model, _ = engine.load_model("m", good, "cpu")
    assert model is not None
