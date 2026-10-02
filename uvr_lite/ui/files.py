"""输入文件扫描：文件夹添加方式的音频文件发现。

后缀表与扫描实现已上移到 ``uvr_lite.audio_io``（CLI 批量合成共用同一来源，
避免两份扩展名表漂移），这里保留旧导入路径（main/测试从本模块导入）。
"""

from collections.abc import Iterable
from pathlib import Path

from ..audio_io import AUDIO_EXTS as AUDIO_EXTS
from ..audio_io import is_audio as is_audio
from ..audio_io import scan_audio_files as scan_audio_files


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
