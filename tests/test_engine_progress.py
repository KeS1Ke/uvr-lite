# coding: utf-8
"""票 1（tdd）：engine 进度回调 + CancelledError + 半成品清理。

运行（仓库根目录）: python -m pytest tests/test_engine_progress.py -v
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from ml_collections import ConfigDict

from uvr_lite import engine as engine_mod
from uvr_lite.engine import CancelledError, separate_file
from utils.model_utils import apply_tta, bigshifts_wrapper, demix


# ---------- 工具 ----------

def make_wav(path: Path, seconds: float = 0.3, sr: int = 44100) -> Path:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(path, np.stack([mono, mono], axis=1), sr)
    return path


def make_config():
    return SimpleNamespace(
        audio=SimpleNamespace(sample_rate=44100, num_channels=2),
        inference=SimpleNamespace(normalize=False, batch_size=1),
        training=SimpleNamespace(use_amp=False),
    )


def fake_ensure_model(model_name):
    return Path("fake.ckpt")


def fake_load_model(model_name, ckpt_path, device):
    return object(), make_config()


def fake_bigshifts(config, model, mix, device, model_type, pbar=False, bigshifts=1,
                   progress_cb=None, demix_progress_cb=None):
    """模拟 vendored bigshifts_wrapper：chunk 级 + pass 级回调（与真实实现同序）。"""
    total_chunks = 3
    for p in range(bigshifts):
        for c in range(total_chunks):
            if demix_progress_cb is not None:
                demix_progress_cb(c + 1, total_chunks)
        if progress_cb is not None:
            progress_cb(p + 1, bigshifts)
    return {"vocals": np.asarray(mix, dtype=float)}


def fake_apply_tta(config, model, mix, waveforms_orig, device, model_type,
                   bigshifts=1, pbar=False, progress_cb=None):
    n = 2
    for i in range(n):
        if progress_cb is not None:
            progress_cb(i + 1, n)
    return waveforms_orig


@pytest.fixture(autouse=True)
def patch_engine(monkeypatch):
    monkeypatch.setattr(engine_mod, "ensure_model", fake_ensure_model)
    monkeypatch.setattr(engine_mod, "load_model", fake_load_model)
    monkeypatch.setattr(engine_mod, "bigshifts_wrapper", fake_bigshifts)
    monkeypatch.setattr(engine_mod, "apply_tta", fake_apply_tta)
    monkeypatch.setattr(engine_mod, "prefer_target_instrument",
                        lambda config: ["vocals", "instrumental"])
    monkeypatch.setattr(engine_mod, "denormalize_audio", lambda est, p: est)


# ---------- engine 编排 ----------

def test_phase_order_and_progress(tmp_path):
    song = make_wav(tmp_path / "song.wav")
    out = tmp_path / "out"
    calls = []

    def cb(phase, done, total):
        calls.append((phase, done, total))
        return True

    written = separate_file(str(song), str(out), bigshifts=2, progress_callback=cb)

    assert len(written) == 2 and all(Path(p).exists() for p in written)
    assert calls[0] == ("decode", 0, 1)
    assert ("decode", 1, 1) in calls
    assert ("infer", 2, 2) in calls
    assert ("chunk", 3, 3) in calls
    assert calls[-1] == ("write", 2, 2)
    phases = [c[0] for c in calls]
    assert phases.index("decode") < phases.index("infer") < phases.index("write")


def test_cancel_during_decode(tmp_path):
    song = make_wav(tmp_path / "song.wav")

    def cb(phase, done, total):
        return not (phase == "decode" and done == 1)

    with pytest.raises(CancelledError):
        separate_file(str(song), str(tmp_path / "out"), progress_callback=cb)


def test_cancel_during_infer_no_output(tmp_path):
    song = make_wav(tmp_path / "song.wav")
    out = tmp_path / "out"

    def cb(phase, done, total):
        return not (phase == "infer" and done == 1)

    with pytest.raises(CancelledError):
        separate_file(str(song), str(out), bigshifts=2, progress_callback=cb)
    assert not out.exists() or not list(out.glob("*"))


def test_cancel_during_write_cleans_partial_outputs(tmp_path):
    song = make_wav(tmp_path / "song.wav")
    out = tmp_path / "out"

    def cb(phase, done, total):
        # 第一个声部写完后收到取消请求
        return not (phase == "write" and done == 1)

    with pytest.raises(CancelledError):
        separate_file(str(song), str(out), progress_callback=cb)
    assert not list(out.glob("*")), "取消后不应残留半成品文件"


def test_no_callback_backward_compatible(tmp_path):
    song = make_wav(tmp_path / "song.wav")
    out = tmp_path / "out"
    written = separate_file(str(song), str(out))  # 不传回调 = CLI 路径
    assert len(written) == 2 and all(Path(p).exists() for p in written)


def test_mono_input_upmixed_to_stereo(tmp_path, monkeypatch):
    """mono 输入 + stereo 模型（num_channels=2）：_load_audio 恒返回 2D
    (channels, samples)，引擎必须按 channels==1 复制双声道，否则模型
    stereo 断言崩溃（回归：旧代码用 len(shape)==1 判断，永不触发）。"""
    seen = {}

    def spy_bigshifts(config, model, mix, *a, **k):
        seen["channels"] = mix.shape[0]
        return {"vocals": np.asarray(mix, dtype=float)}

    monkeypatch.setattr(engine_mod, "bigshifts_wrapper", spy_bigshifts)
    song = tmp_path / "mono.wav"
    t = np.linspace(0, 0.3, int(0.3 * 44100), endpoint=False)
    sf.write(song, 0.5 * np.sin(2 * np.pi * 440 * t), 44100)  # 1D = mono

    written = separate_file(str(song), str(tmp_path / "out"))
    assert seen["channels"] == 2
    assert len(written) == 2 and all(Path(p).exists() for p in written)


def test_separator_reuses_model_across_files(tmp_path, monkeypatch):
    """会话复用：一个 Separator 处理多文件时，模型只加载一次。"""
    song1 = make_wav(tmp_path / "a.wav")
    song2 = make_wav(tmp_path / "b.wav")
    loads = []
    real_load = engine_mod.load_model  # fixture 已换成 fake，包装计数即可

    def counting_load_model(model_name, ckpt_path, device):
        loads.append(model_name)
        return real_load(model_name, ckpt_path, device)

    monkeypatch.setattr(engine_mod, "load_model", counting_load_model)
    from uvr_lite.engine import Separator

    sep = Separator(verbose=False)
    sep.separate(str(song1), str(tmp_path / "out"))
    sep.separate(str(song2), str(tmp_path / "out"))
    assert len(loads) == 1, "两个文件应共用一次模型加载（实际 %d 次）" % len(loads)


def test_tta_phase_reported(tmp_path):
    song = make_wav(tmp_path / "song.wav")
    phases = []

    def cb(phase, done, total):
        phases.append((phase, done, total))
        return True

    separate_file(str(song), str(tmp_path / "out"), tta=True, progress_callback=cb)
    assert ("tta", 1, 2) in phases and ("tta", 2, 2) in phases


# ---------- vendored 钩子 ----------

class FakeModel(nn.Module):
    """返回全零的假模型（形状 (batch, n_instr, chunk)）。"""

    def __init__(self, n_instr=1):
        super().__init__()
        self.n_instr = n_instr

    def forward(self, x):
        return torch.zeros((x.shape[0], self.n_instr, x.shape[2]))


def demix_config():
    # demix 内部用 `'chunk_size' in config.inference` 判断，需要 dict 语义的 ConfigDict
    return ConfigDict({
        "training": {"use_amp": False, "instruments": ["vocals"]},
        "inference": {"chunk_size": 4096, "num_overlap": 2, "batch_size": 1},
        "audio": {"chunk_size": 4096},
    })


def test_demix_reports_chunk_progress():
    mix = np.random.RandomState(0).randn(2, 12000)
    calls = []
    out = demix(demix_config(), FakeModel(), mix, "cpu", "bs_roformer",
                pbar=False, progress_cb=lambda d, t: calls.append((d, t)))
    assert len(calls) > 1
    dones = [d for d, t in calls]
    assert dones == sorted(dones), "进度应单调递增"
    for d, t in calls:
        assert 0 < d <= t
    assert calls[-1][0] == calls[-1][1], "最后一次应等于总数"
    assert out["vocals"].ndim == 2  # (channels, time)


def test_demix_callback_exception_propagates():
    mix = np.random.RandomState(0).randn(2, 12000)

    def boom(done, total):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        demix(demix_config(), FakeModel(), mix, "cpu", "bs_roformer",
              pbar=False, progress_cb=boom)


def test_bigshifts_wrapper_reports_per_pass(monkeypatch):
    calls = []

    def fake_demix(config, model, mix, device, model_type, pbar=False, progress_cb=None):
        return {"vocals": np.zeros((2, 100))}

    monkeypatch.setattr("utils.model_utils.demix", fake_demix)
    mix = np.random.RandomState(0).randn(2, 1000)
    out = bigshifts_wrapper(demix_config(), object(), mix, "cpu", "bs_roformer",
                            bigshifts=3, progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]
    assert out["vocals"].shape == (2, 100)


def test_apply_tta_reports_per_augmentation(monkeypatch):
    calls = []

    def fake_bigshifts(config, model, mix, device, model_type, pbar=False,
                       bigshifts=1, progress_cb=None, demix_progress_cb=None):
        return {"vocals": np.zeros((2, 100))}

    monkeypatch.setattr("utils.model_utils.bigshifts_wrapper", fake_bigshifts)
    mix = np.random.RandomState(0).randn(2, 100)
    orig = {"vocals": np.zeros((2, 100))}
    out = apply_tta(demix_config(), object(), mix, orig, "cpu", "bs_roformer",
                    progress_cb=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 2), (2, 2)]
    assert out["vocals"].shape == (2, 100)
