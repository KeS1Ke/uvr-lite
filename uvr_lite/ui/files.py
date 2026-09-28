"""输入文件扫描：文件夹添加方式的音频文件发现。"""

from collections.abc import Iterable
from pathlib import Path

# 常见音频格式（engine._load_audio 主链路 soundfile 可解码的子集；mp3 需 libsndfile ≥ 1.1）
AUDIO_EXTS = {".mp3", ".flac", ".wav", ".ogg", ".m4a"}


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


def precheck_audio(path: Path) -> bool:
    """开始分离前的快速格式预检：读取音频头判断能否解码（不完整解码）。

    与 engine._load_audio 的实际解码链路一致（已不用 librosa，改 soundfile + soxr
    重采样；soundfile 优先、audioread 兜底）：
    - flac/wav/ogg → soundfile（libsndfile 原生支持，无需外部解码器）
    - mp3 → soundfile（需 libsndfile ≥ 1.1；Windows 自带，旧 Linux 会走到兜底）
    - m4a 等 → audioread 兜底（需系统 ffmpeg）
    文件内容真是音频时，即使后缀被改成 doc/txt 也能识别；真 Word 等非音频被拒绝。
    """
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames > 0 and info.samplerate > 0
    except Exception:
        pass
    import audioread

    try:
        with audioread.audio_open(str(path)) as f:
            _ = f.duration
        return True
    except Exception:
        return False


def scan_audio_files(folder: Path) -> list[Path]:
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


def dedup_paths(paths: Iterable[Path]) -> list[Path]:
    """按解析后绝对路径去重，保持添加顺序。"""
    seen = set()
    result = []
    for p in paths:
        rp = Path(p).resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(rp)
    return result
