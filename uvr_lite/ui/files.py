# coding: utf-8
"""输入文件扫描：文件夹添加方式的音频文件发现。"""

from pathlib import Path
from typing import Iterable, List

# 常见音频格式（librosa/soundfile 可解码的子集）
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def scan_audio_files(folder: Path) -> List[Path]:
    """扫描文件夹下直接包含的音频文件（非递归），按名称排序。

    - 只取顶层文件（不进入子目录），行为可预期
    - 返回解析后的绝对路径，供列表去重
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        (p.resolve() for p in folder.iterdir() if p.is_file() and is_audio(p)),
        key=lambda p: p.name.lower(),
    )


def dedup_paths(paths: Iterable[Path]) -> List[Path]:
    """按解析后绝对路径去重，保持添加顺序。"""
    seen = set()
    result = []
    for p in paths:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(rp)
    return result
