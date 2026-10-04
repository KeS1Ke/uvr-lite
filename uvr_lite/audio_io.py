"""音频解码 / 重采样 / 文件发现：soundfile + soxr（m4a 兜底 audioread）。

从 engine.py 抽出且不 import torch——合成（uvr_lite.mix）与 CLI 批量扫描
不付 torch 导入成本。engine._load_audio 保留为薄封装，兼容旧调用点与测试。
替代 librosa.load，连带省掉 scipy/numba/llvmlite 等约 320MB 依赖。
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

# 常见音频格式（soundfile 主链路可解码的子集；mp3 需 libsndfile ≥ 1.1）
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}


def is_audio(path: Path) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def scan_audio_files(folder: Path) -> list[Path]:
    """扫描文件夹下直接包含的音频文件（非递归），按名称排序。

    - 只取顶层文件（不进入子目录），行为可预期
    - 返回解析后的绝对路径，供列表去重与配对
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (p.resolve() for p in folder.iterdir() if p.is_file() and is_audio(p)),
        key=lambda p: p.name.lower(),
    )


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """解码为 (channels, samples) float32 与原生采样率，恒为 2D。

    主路径 soundfile（flac/wav/ogg/mp3 原生解码，libsndfile）；
    m4a 等 libsndfile 不支持的格式回退 audioread（需系统 ffmpeg）。
    兜底若抛出没有正文的 EOFError / NoBackendError，则带上 soundfile
    原文和「文件不是可解码的音频」；兜底异常已有正文时原样抛出。
    """
    try:
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except RuntimeError as sf_err:
        data, orig_sr = _audioread_or_explain(path, sf_err)
    return data.T, orig_sr  # (frames, channels) -> (channels, frames)


def _audioread_or_explain(path: Path, sf_err: Exception) -> tuple[np.ndarray, int]:
    """audioread 兜底。空消息的 EOF/无后端错误改写成带 soundfile 原文的说明。"""
    from audioread.exceptions import NoBackendError

    try:
        return _read_audioread(path)
    except (EOFError, NoBackendError) as err:
        if str(err).strip():
            raise
        sf_text = str(sf_err).strip() or type(sf_err).__name__
        raise RuntimeError(f"文件不是可解码的音频（{sf_text}）") from err


def _read_audioread(path: Path) -> tuple[np.ndarray, int]:
    """audioread 兜底解码（int16 PCM → float32，与 librosa 的 audioread 路径一致）。"""
    import audioread

    with audioread.audio_open(str(path)) as af:
        orig_sr = af.samplerate
        ch = af.channels
        blocks = [
            np.frombuffer(b, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
            for b in af
        ]
    return np.concatenate(blocks, axis=0), orig_sr


def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """(channels, samples) 重采样。soxr 的 2D 语义为 (samples, channels)。"""
    return soxr.resample(data.T, orig_sr, target_sr, quality="HQ").T


def load_audio(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """解码并（可选）重采样到 sr；返回 (data, 实际采样率)。

    data 恒为 2D (channels, samples)。engine 只需要数据，取 [0]。
    """
    data, orig_sr = read_audio(Path(path))
    if sr is not None and sr != orig_sr:
        data = resample_audio(data, orig_sr, sr)
        orig_sr = sr
    return data, orig_sr
