"""torch.compile 门控与回退：CPU / 无 triton / UVR_COMPILE=0 不编译，失败退回 eager。"""

from types import SimpleNamespace

import torch

import uvr_lite.engine as engine


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        audio=SimpleNamespace(num_channels=2, chunk_size=4096),
        inference=SimpleNamespace(chunk_size=4096, batch_size=4),
    )


def test_warmup_small_shape_by_default():
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, x):
            self.shapes.append(tuple(x.shape))
            return x

    r1 = Recorder()
    assert engine._warmup(r1, _cfg(), "cpu") is True
    assert r1.shapes == [(1, 2, 2048)]  # max(4096 // 10, 2048)

    r2 = Recorder()
    assert engine._warmup(r2, _cfg(), "cpu", full_shape=True) is True
    assert r2.shapes == [(4, 2, 4096)]


def test_cpu_never_compiles(monkeypatch):
    compiled_calls: list[int] = []
    warmups: list[bool] = []
    monkeypatch.setattr(engine.torch, "compile", lambda *a, **k: compiled_calls.append(1))
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: warmups.append(k) or True)

    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cpu") is model
    assert compiled_calls == []
    assert len(warmups) == 1


def test_compile_skipped_without_triton(monkeypatch):
    compiled_calls: list[int] = []
    monkeypatch.setattr(engine, "_triton_available", lambda: False)
    monkeypatch.setattr(engine.torch, "compile", lambda *a, **k: compiled_calls.append(1))
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: True)

    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cuda") is model
    assert compiled_calls == []


def test_env_off_skips_compile(monkeypatch):
    compiled_calls: list[int] = []
    monkeypatch.setenv("UVR_COMPILE", "0")
    monkeypatch.setattr(engine, "_triton_available", lambda: True)
    monkeypatch.setattr(engine.torch, "compile", lambda *a, **k: compiled_calls.append(1))
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: True)

    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cuda") is model
    assert compiled_calls == []


def test_compile_failure_falls_back_to_eager(monkeypatch):
    monkeypatch.setattr(engine, "_triton_available", lambda: True)

    def boom(*a, **k):
        raise RuntimeError("no inductor")

    monkeypatch.setattr(engine.torch, "compile", boom)
    warmups: list[object] = []
    monkeypatch.setattr(engine, "_warmup", lambda m, *a, **k: warmups.append(m) or True)

    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cuda") is model
    assert warmups == [model]


def test_compiled_warmup_failure_falls_back_to_eager(monkeypatch):
    monkeypatch.setattr(engine, "_triton_available", lambda: True)
    compiled = object()
    monkeypatch.setattr(engine.torch, "compile", lambda *a, **k: compiled)
    seen: list[object] = []

    def fake_warmup(model, *a, **k):
        seen.append(model)
        return model is not compiled  # 编译模型预热失败，eager 成功

    monkeypatch.setattr(engine, "_warmup", fake_warmup)
    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cuda") is model
    assert seen == [compiled, model]


def test_compiled_model_returned_when_warmup_ok(monkeypatch):
    monkeypatch.setattr(engine, "_triton_available", lambda: True)
    compiled = object()
    monkeypatch.setattr(engine.torch, "compile", lambda *a, **k: compiled)
    monkeypatch.setattr(engine, "_warmup", lambda *a, **k: True)

    model = object()
    assert engine._compile_and_warmup(model, _cfg(), "cuda") is compiled
