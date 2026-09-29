"""权重加载安全模式回归：engine 必须以安全方式加载默认模型权重。

分发格式为纯张量（.safetensors 直接是 {name: Tensor}；历史 .ckpt 为
{"state_dict": {...}}），无 pickle 任意代码执行面；若权重文件混入非张量对象
（numpy 数组/自定义类），torch.load(weights_only=True) 抛 UnpicklingError，
engine 应转为可读错误。
"""

from types import SimpleNamespace

import numpy
import pytest
import torch

import uvr_lite.engine as engine
from uvr_lite.download import model_file
from uvr_lite.models import DEFAULT_MODEL


def test_default_model_weights_are_safe_to_load():
    """注册表默认模型的当前分发物必须是可安全加载的纯张量格式。"""
    # 路径取自注册表 filename（当前分发物），不是历史 .ckpt：旧写法盯着一份已不再
    # 分发的文件，结果在干净环境/CI 上永远 skip，这条唯一的安全断言实际失效。
    # 解析放在用例内而非模块级：model_file 经 models_dir() 会 mkdir 出 models/，
    # 模块级等于让「只收集测试」这一步也产生文件系统副作用（只读工作区会以
    # collection ERROR 收场，而不是这里的 skip）。
    ckpt = model_file(DEFAULT_MODEL)
    if not ckpt.exists():
        pytest.skip(f"本机无权重（CI / 未下载环境跳过）: {ckpt.name}")
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        obj = load_file(str(ckpt))
        assert obj and all(isinstance(v, torch.Tensor) for v in obj.values())
    else:
        obj = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert isinstance(obj, dict) and "state_dict" in obj


def _patch_engine(monkeypatch, seen: dict | None = None):
    """load_model 依赖最小化：假模型 + 假配置，聚焦 torch.load 行为。

    seen 非空时把 load_start_checkpoint 换成记录实参的 spy：只断言「没抛异常」
    会漏掉「权重读出来了却没交给加载器」——权重被静默丢弃时模型照样非 None。
    """
    monkeypatch.setattr(engine, "get_model_info",
                        lambda name: {"model_type": "bs_roformer", "config": "x.yaml"})
    monkeypatch.setattr(engine, "config_path", lambda name: "fake.yaml")
    monkeypatch.setattr(engine, "get_model_from_config",
                        lambda mt, cfg: (torch.nn.Identity(), SimpleNamespace(training={})))
    if seen is None:
        monkeypatch.setattr(engine, "load_start_checkpoint", lambda *a, **k: None)
    else:
        def spy(args, model, checkpoint, type_="inference", **kw):
            seen.update(args=args, model=model, checkpoint=checkpoint, type_=type_)

        monkeypatch.setattr(engine, "load_start_checkpoint", spy)
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: None)


def test_engine_load_model_rejects_pickle_payload(tmp_path, monkeypatch):
    """含非张量对象的 ckpt → RuntimeError（不静默回退到 weights_only=False）。"""
    _patch_engine(monkeypatch)
    bad = tmp_path / "bad.ckpt"
    torch.save({"state_dict": {"w": numpy.zeros(3)}}, bad)  # numpy 不在安全白名单

    with pytest.raises(RuntimeError, match="安全模式加载失败"):
        engine.load_model("m", bad, "cpu")


def test_engine_load_model_accepts_pure_tensors(tmp_path, monkeypatch):
    """纯张量 ckpt 正常通过 weights_only=True 加载，且权重确实交给加载器。"""
    seen = {}
    _patch_engine(monkeypatch, seen)
    good = tmp_path / "good.ckpt"
    torch.save({"state_dict": {"w": torch.zeros(3)}}, good)

    model, _ = engine.load_model("m", good, "cpu")

    assert seen["model"] is model, "权重必须装进 load_model 返回的那个模型"
    assert seen["args"].start_check_point == str(good), "加载器要看得到权重来源"
    assert seen["type_"] == "inference"
    assert set(seen["checkpoint"]["state_dict"]) == {"w"}, "ckpt 分支保留 state_dict 包装"
    assert seen["checkpoint"]["state_dict"]["w"].shape == (3,)


def test_engine_load_model_safetensors(tmp_path, monkeypatch):
    """.safetensors 权重走 safetensors.torch.load_file（无 pickle 载入面）。"""
    seen = {}
    _patch_engine(monkeypatch, seen)
    from safetensors.torch import save_file

    good = tmp_path / "good.safetensors"
    save_file({"w": torch.zeros(3)}, str(good))

    model, _ = engine.load_model("m", good, "cpu")

    assert seen["model"] is model
    # safetensors 分支交给加载器的是扁平 {name: Tensor}，没有 state_dict 包装
    assert set(seen["checkpoint"]) == {"w"}
    assert seen["checkpoint"]["w"].shape == (3,)
