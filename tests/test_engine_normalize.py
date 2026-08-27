"""normalize 开启时 instrumental 的域一致性回归（B3）。

模型在归一化域输出 vocals；instrumental = 原始混合 - 反归一化后的 vocals，
两者必须同域。旧实现混域（原始域减归一化域）且写循环对 instrumental
二次 denorm——normalize=True 时输出错误。

本文件刻意不用真实模型（保持测试轻量），但 normalize/denormalize 走
msst 真实实现，文件写出为真实 soundfile。
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

# msst/ 内模块以顶层包导入，与 engine.py 的 sys.path 处理保持一致
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "msst") not in sys.path:
    sys.path.insert(0, str(ROOT / "msst"))

from utils.audio_utils import denormalize_audio, normalize_audio  # noqa: E402

import uvr_lite.engine as engine_mod  # noqa: E402
from uvr_lite.engine import separate_file  # noqa: E402


def _run(tmp_path, monkeypatch, normalize: bool, voc_scale: float = 0.5):
    """轻量编排：绕过模型加载，只走 separate() 的域处理与文件写出。"""
    sr = 8000
    rng = np.random.default_rng(7)
    mix = (rng.random((2, sr), dtype=np.float32) * 0.5 - 0.25).astype(np.float32)
    song = tmp_path / "song.wav"
    sf.write(song, mix.T, sr)

    config = SimpleNamespace(
        audio=SimpleNamespace(num_channels=2),
        inference=SimpleNamespace(normalize=normalize),
    )

    def _init(self, *a, **k):
        self.model_name = "bs_roformer_ep317"
        self.model = object()  # bigshifts_wrapper 形参（fake 不真实使用）
        self.device = "cpu"
        self.verbose = False
        self.sample_rate = sr
        self.config = config

    def _fake_bigshifts(config_, model, mix_arr, device, model_type, **kw):
        if normalize:
            norm_arr, _ = normalize_audio(mix_arr)  # 模型输入在归一化域
            return {"vocals": norm_arr * voc_scale, "instrumental": None}
        return {"vocals": mix_arr * voc_scale, "instrumental": None}

    monkeypatch.setattr(engine_mod.Separator, "__init__", _init)
    monkeypatch.setattr(engine_mod, "_load_audio", lambda path, sr_: mix.copy())
    monkeypatch.setattr(engine_mod, "bigshifts_wrapper", _fake_bigshifts)
    monkeypatch.setattr(engine_mod, "prefer_target_instrument",
                        lambda config_: ["vocals", "instrumental"])
    written = separate_file(str(song), str(tmp_path / "out"),
                            model_name="bs_roformer_ep317")
    return written, mix


def _read(wav_path):
    data, _ = sf.read(str(wav_path), dtype="float32", always_2d=True)
    return data.T


def test_instrumental_same_domain_when_normalize(tmp_path, monkeypatch):
    """normalize=True：instrumental 文件 == 原始混合 - 反归一化 vocals。"""
    written, mix = _run(tmp_path, monkeypatch, normalize=True)
    _, params = normalize_audio(mix)
    exp_vocals = denormalize_audio(normalize_audio(mix)[0] * 0.5, params)
    exp_instrumental = mix - exp_vocals

    vocals_path = next(p for p in written if "vocals" in p.name)
    instr_path = next(p for p in written if "instrumental" in p.name)
    assert np.allclose(_read(vocals_path), exp_vocals, atol=1e-5), "vocals 应反归一化写回"
    assert np.allclose(_read(instr_path), exp_instrumental, atol=1e-5), \
        "instrumental 必须与原始混合同域（回归 B3 混域/二次 denorm）"


def test_instrumental_without_normalize_unchanged(tmp_path, monkeypatch):
    """normalize=False：行为与旧版一致（instrumental = mix - vocals）。"""
    written, mix = _run(tmp_path, monkeypatch, normalize=False)
    instr_path = next(p for p in written if "instrumental" in p.name)
    assert np.allclose(_read(instr_path), mix * 0.5, atol=1e-5)
