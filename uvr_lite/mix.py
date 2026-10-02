"""合成引擎：把分离开的人声与伴奏重新合成为整曲（separate 的逆运算）。

不走模型、不 import torch：解码 → （按需重采样 / 补声道 / 补长度）→ 增益
求和 → 写出。同源分离结果默认精确求和（instrumental = mix − vocals），
不归一化即可近似无损还原原曲。

progress_callback(phase, done, total) -> bool 与分离引擎同契约：
phase 为 "decode" / "mix" / "write"；返回 False 时抛 CancelledError 并清理
已写出的半成品。
"""

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import load_audio, resample_audio
from .errors import CancelledError
from .stems import split_stem


def _fit_channels(data: np.ndarray, channels: int) -> np.ndarray:
    """单声道复制成目标声道数；已是目标声道数原样返回。"""
    if data.shape[0] == channels:
        return data
    if data.shape[0] == 1:
        return np.repeat(data, channels, axis=0)
    return data


def _fit_length(data: np.ndarray, frames: int) -> np.ndarray:
    """不足补零、超出截断（t=0 对齐）。"""
    cur = data.shape[1]
    if cur == frames:
        return data
    if cur < frames:
        pad = np.zeros((data.shape[0], frames - cur), dtype=data.dtype)
        return np.concatenate([data, pad], axis=1)
    return data[:, :frames]


def _output_stem(vocals: Path, out_name: str | None) -> str:
    """输出主名：优先 out_name，否则从 {stem}-vocals 反推 stem。"""
    if out_name:
        return out_name
    split = split_stem(vocals)
    if split is not None and split[1] == "vocals":
        return split[0]
    return vocals.stem


def combine(
    vocals: str | Path,
    instrumental: str | Path,
    out_dir: str | Path,
    vocal_gain: float = 1.0,
    inst_gain: float = 1.0,
    pcm: str = "PCM_24",
    fmt: str = "auto",  # auto | flac | wav
    normalize: bool = False,
    out_name: str | None = None,
    verbose: bool = True,
    progress_callback: Callable[[str, int, int], bool] | None = None,
) -> Path:
    """把人声与伴奏合成为一个文件，返回写出路径 `{stem}-mix.{ext}`。

    采样率取两者较高值；声道数取较大者（单声道自动复制）；长度按最长
    补零/截断（同源分离结果天然等长）。normalize=False 时保持精确求和。
    """
    vocals = Path(vocals)
    instrumental = Path(instrumental)
    if not vocals.exists():
        raise FileNotFoundError(f"人声轨不存在: {vocals}")
    if not instrumental.exists():
        raise FileNotFoundError(f"伴奏轨不存在: {instrumental}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cb(phase: str, done: int, total: int) -> None:
        if progress_callback is not None and not progress_callback(phase, done, total):
            raise CancelledError(f"用户在 {phase} 阶段取消了任务")

    if verbose:
        print(f"合成: {vocals.name} + {instrumental.name} | 增益: "
              f"人声 {vocal_gain:g} / 伴奏 {inst_gain:g}")

    _cb("decode", 0, 2)
    voc, sr_v = load_audio(vocals)
    _cb("decode", 1, 2)
    inst, sr_i = load_audio(instrumental)
    target_sr = max(sr_v, sr_i)
    if sr_v != target_sr:
        voc = resample_audio(voc, sr_v, target_sr)
    if sr_i != target_sr:
        inst = resample_audio(inst, sr_i, target_sr)
    _cb("decode", 2, 2)

    channels = max(voc.shape[0], inst.shape[0])
    frames = max(voc.shape[1], inst.shape[1])
    voc = _fit_length(_fit_channels(voc, channels), frames)
    inst = _fit_length(_fit_channels(inst, channels), frames)

    _cb("mix", 0, 1)
    mixed = voc * float(vocal_gain) + inst * float(inst_gain)
    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    if normalize and peak > 0:
        mixed = mixed * (0.891 / peak)  # 峰值 -1 dBFS
        peak = float(np.abs(mixed).max())
    elif peak > 1.0 and verbose:
        print(f"  警告: 峰值 {peak:.3f} 超过 0 dBFS，整数格式会削波"
              "（可勾选/传入峰值归一化）")
    _cb("mix", 1, 1)

    # 与分离引擎相同的 auto 规则：峰值 ≤ 1 落 FLAC，否则 WAV（PCM 会削波，
    # 这里只对齐行为；要真正避免削波请用 normalize）
    codec = "flac" if (fmt == "flac" or (fmt == "auto" and peak <= 1.0)) else "wav"
    out_path = out_dir / f"{_output_stem(vocals, out_name)}-mix.{codec}"

    written: list[Path] = []
    try:
        _cb("write", 0, 1)
        sf.write(out_path, mixed.T, target_sr, subtype=pcm)
        written.append(out_path)
        if verbose:
            print(f"  写出: {out_path}（峰值 {peak:.3f}）")
        _cb("write", 1, 1)
    except CancelledError:
        for p in written:
            with suppress(OSError):
                p.unlink(missing_ok=True)
        raise
    return out_path
