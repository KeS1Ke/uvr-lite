"""合成引擎：把分离开的人声与伴奏重新合成为整曲（separate 的逆运算）。

不走模型、不 import torch：解码 → （按需重采样 / 补声道 / 补长度）→ 增益
求和 → 写出。同源分离结果默认精确求和（instrumental = mix − vocals），
不归一化即可近似无损还原原曲。

progress_callback(phase, done, total) -> bool 与分离引擎同契约：
phase 为 "decode" / "mix" / "write"；返回 False 时抛 CancelledError，并删掉
本次已替换成功的成品。半成品只写在 `{最终名}.part`，异常时删掉，不占用成品名。
"""

import os
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


def _mix_output_path(out_dir: Path, stem: str, codec: str) -> Path:
    """空闲时 `{stem}-mix.{codec}`；已占用则 `{stem}-mix-2`、`-3`……直到空位。

    用 Path.exists()：Windows 上 Song-mix 与 song-mix 是同一个文件，也会让开。
    """
    candidate = out_dir / f"{stem}-mix.{codec}"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{stem}-mix-{n}.{codec}"
        n += 1
    return candidate


def _unlink_quiet(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)


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
    """把人声与伴奏合成为一个文件，返回写出路径。

    采样率取两者较高值；声道数取较大者（单声道自动复制）；两侧都不是
    单声道且声道数不同时抛出 ValueError。长度按最长补零/截断（同源分离
    结果天然等长）。normalize=False 时保持精确求和。文件名默认
    `{stem}-mix.{ext}`；该路径已存在时从 `{stem}-mix-2.{ext}` 起递增，
    避免截断已有文件。写出先落 `{最终名}.part`，成功后再替换。
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
    voc_ch = int(voc.shape[0])
    inst_ch = int(inst.shape[0])
    # 单声道可以复制；2ch+4ch 这类对不上时不要把 numpy 的 broadcast 错误抛出去
    if voc_ch != inst_ch and voc_ch != 1 and inst_ch != 1:
        raise ValueError(
            f"声道数对不上，无法合成：人声 {voc_ch} 声道，伴奏 {inst_ch} 声道"
            "（只有单声道可以复制到另一侧）"
        )
    voc = _fit_length(_fit_channels(voc, channels), frames)
    inst = _fit_length(_fit_channels(inst, channels), frames)

    _cb("mix", 0, 1)
    mixed = voc * float(vocal_gain) + inst * float(inst_gain)
    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    # peak<=0 或 <1e-4 视为静音，不放大；否则增益不超过 100
    silent = peak <= 0 or peak < 1e-4
    if normalize and not silent:
        gain = 0.891 / peak
        if gain > 100.0:
            gain = 100.0
            if verbose:
                print(
                    "  警告: 峰值过低，归一化增益已限制为 100"
                    f"（当前峰值 {peak:.6g}，不再放大到 -1 dBFS）"
                )
        mixed = mixed * gain
        peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    elif peak > 1.0 and verbose:
        print(f"  警告: 峰值 {peak:.3f} 超过 0 dBFS，整数格式会削波"
              "（可勾选/传入峰值归一化）")
    _cb("mix", 1, 1)

    # 与分离引擎相同的 auto 规则：峰值 ≤ 1 落 FLAC，否则 WAV（PCM 会削波，
    # 这里只对齐行为；要真正避免削波请用 normalize）
    codec = "flac" if (fmt == "flac" or (fmt == "auto" and peak <= 1.0)) else "wav"
    out_path = _mix_output_path(out_dir, _output_stem(vocals, out_name), codec)
    # 扩展名是 .part 时 soundfile 无法从文件名判断格式，必须显式传入 format
    part_path = out_path.with_name(out_path.name + ".part")

    written: list[Path] = []
    try:
        _cb("write", 0, 1)
        sf.write(part_path, mixed.T, target_sr, subtype=pcm, format=codec.upper())
        os.replace(part_path, out_path)
        written.append(out_path)
        if verbose:
            print(f"  写出: {out_path}（峰值 {peak:.3f}）")
        _cb("write", 1, 1)
    except CancelledError:
        _unlink_quiet(part_path)
        for p in written:
            _unlink_quiet(p)
        raise
    except (KeyboardInterrupt, Exception):
        _unlink_quiet(part_path)
        raise
    return out_path
