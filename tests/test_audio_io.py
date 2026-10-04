"""audio_io：从 engine 抽出的解码/重采样层（合成路径共用）。"""

import numpy as np
import pytest
import soundfile as sf

from uvr_lite.audio_io import load_audio, read_audio


def test_read_audio_returns_native_rate_and_channels(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros((1000, 2), dtype=np.float32), 8000)

    data, sr = read_audio(p)

    assert sr == 8000
    assert data.shape == (2, 1000), "恒为 (channels, frames)"


def test_read_audio_mono_is_2d(tmp_path):
    p = tmp_path / "mono.flac"
    sf.write(str(p), np.zeros(500, dtype=np.float32), 44100)

    data, _ = read_audio(p)

    assert data.ndim == 2 and data.shape == (1, 500)


def test_load_audio_resamples_to_requested_rate(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros(4000, dtype=np.float32), 8000)  # 0.5s

    data, sr = load_audio(p, 16000)

    assert sr == 16000
    assert abs(data.shape[1] - 8000) <= 2, f"0.5s@16k 应约 8000 帧: {data.shape}"


def test_empty_or_non_audio_keeps_soundfile_text(tmp_path):
    """空文件 / 非音频：兜底异常没有正文时，对外消息仍带 soundfile 原文。"""
    empty = tmp_path / "bad.wav"
    empty.write_bytes(b"")
    text = tmp_path / "notaudio.wav"
    text.write_bytes(b"this is not audio")

    for path in (empty, text):
        with pytest.raises(RuntimeError) as exc:
            read_audio(path)
        msg = str(exc.value).strip()
        assert msg, "异常消息不能是空的"
        assert "文件不是可解码的音频" in msg
        assert "Format not recognised" in msg


def test_audioread_error_with_message_is_not_replaced(monkeypatch, tmp_path):
    """兜底异常已有正文时原样抛出，不用空消息包装把它吞掉。"""
    from audioread.exceptions import NoBackendError

    path = tmp_path / "a.wav"
    path.write_bytes(b"not really audio")

    def boom(path):
        raise NoBackendError("backend said no")

    monkeypatch.setattr("uvr_lite.audio_io._read_audioread", boom)

    with pytest.raises(NoBackendError, match="backend said no"):
        read_audio(path)


def test_load_audio_without_resample_keeps_rate(tmp_path):
    p = tmp_path / "a.wav"
    sf.write(str(p), np.zeros(4000, dtype=np.float32), 8000)

    data, sr = load_audio(p)

    assert sr == 8000
    assert data.shape[1] == 4000
