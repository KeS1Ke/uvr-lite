"""分离写盘安全：不覆盖已有产物、失败不留半截成品、超 0 dBFS 用 float WAV。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import uvr_lite.engine as engine_mod
from uvr_lite.engine import CancelledError, separate_file


def _install(monkeypatch, mix: np.ndarray, vocals: np.ndarray, sr: int = 8000):
    """绕过模型加载，只走 separate() 的写盘路径。"""

    def _init(self, *args, **kwargs):
        self.model_name = "bs_roformer_ep317"
        self.model = object()
        self.device = "cpu"
        self.verbose = kwargs.get("verbose", True)
        self.sample_rate = sr
        self.config = type("C", (), {})()
        self.config.audio = type("A", (), {"num_channels": 2})()
        self.config.inference = type("I", (), {"normalize": False})()

    def _fake_bigshifts(config, model, mix_arr, device, model_type, **kwargs):
        return {"vocals": vocals.copy(), "instrumental": None}

    monkeypatch.setattr(engine_mod.Separator, "__init__", _init)
    monkeypatch.setattr(engine_mod, "_load_audio", lambda path, sr_: mix.copy())
    monkeypatch.setattr(engine_mod, "bigshifts_wrapper", _fake_bigshifts)
    monkeypatch.setattr(
        engine_mod, "prefer_target_instrument", lambda config: ["vocals", "instrumental"],
    )


def _song(tmp_path: Path, mix: np.ndarray, sr: int = 8000) -> Path:
    song = tmp_path / "song.wav"
    sf.write(song, mix.T, sr)
    return song


def test_marker_on_default_name_is_not_truncated(tmp_path, monkeypatch):
    """默认输出名上已有 marker 时，再次分离不得截断它，应写出 -2。"""
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    _install(monkeypatch, mix, mix * 0.5, sr)
    song = _song(tmp_path, mix, sr)
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "song-vocals.flac"
    payload = b"MARKER-NOT-AUDIO"
    marker.write_bytes(payload)

    written = separate_file(str(song), str(out), verbose=False)

    assert marker.read_bytes() == payload
    vocals = next(p for p in written if "vocals" in p.name)
    assert vocals.name == "song-vocals-2.flac"
    assert vocals.exists() and vocals.stat().st_size > len(payload)


def test_case_collision_also_yields(tmp_path, monkeypatch):
    """Windows 上 Song-vocals 与 song-vocals 是同一文件，也要让开。"""
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    _install(monkeypatch, mix, mix * 0.5, sr)
    song = _song(tmp_path, mix, sr)
    out = tmp_path / "out"
    out.mkdir()
    probe = out / "CaseProbe"
    probe.write_text("x", encoding="utf-8")
    if not (out / "caseprobe").exists():
        pytest.skip("大小写敏感的文件系统")
    probe.unlink()
    marker = out / "SONG-vocals.FLAC"
    marker.write_bytes(b"MARKER")

    written = separate_file(str(song), str(out), verbose=False)

    assert marker.read_bytes() == b"MARKER"
    assert any(p.name == "song-vocals-2.flac" for p in written)


def test_cancel_does_not_delete_previous_output(tmp_path, monkeypatch):
    """取消只删本次写出的 -2，不删上一轮占着默认名的文件。"""
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    _install(monkeypatch, mix, mix * 0.5, sr)
    song = _song(tmp_path, mix, sr)
    out = tmp_path / "out"
    out.mkdir()
    marker = out / "song-vocals.flac"
    marker.write_bytes(b"PREVIOUS")

    def cb(phase, done, total):
        return not (phase == "write" and done == 1)

    with pytest.raises(CancelledError):
        separate_file(str(song), str(out), verbose=False, progress_callback=cb)

    assert marker.read_bytes() == b"PREVIOUS"
    assert not (out / "song-vocals-2.flac").exists()
    assert not list(out.glob("*.part"))


def test_oserror_during_write_leaves_no_final_or_part(tmp_path, monkeypatch):
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    _install(monkeypatch, mix, mix * 0.5, sr)
    song = _song(tmp_path, mix, sr)
    out = tmp_path / "out"

    def exploding_write(path, data, samplerate, subtype=None, format=None, **kwargs):
        Path(path).write_bytes(b"PARTIAL")
        raise OSError("disk full")

    monkeypatch.setattr(engine_mod.sf, "write", exploding_write)

    with pytest.raises(OSError, match="disk full"):
        separate_file(str(song), str(out), verbose=False)

    names = [p.name for p in out.iterdir()] if out.exists() else []
    assert names == []
    assert not (out / "song-vocals.flac").exists()
    assert not (out / "song-instrumental.flac").exists()
    assert not list(out.glob("*.part"))


def test_peak_above_one_writes_float_wav(tmp_path, monkeypatch, capsys):
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    vocals = np.full((2, sr), 1.7, dtype=np.float32)
    _install(monkeypatch, mix, vocals, sr)
    song = _song(tmp_path, mix, sr)

    written = separate_file(str(song), str(tmp_path / "out"), verbose=True)
    voc = next(p for p in written if "vocals" in p.name)
    assert voc.suffix == ".wav"
    info = sf.info(str(voc))
    assert info.subtype == "FLOAT"
    data, _ = sf.read(str(voc), dtype="float32", always_2d=True)
    assert np.allclose(data, 1.7, atol=1e-5)
    assert float(np.max(np.abs(data))) > 1.0
    message = capsys.readouterr().out
    assert "整数 PCM 会削波" in message
    assert "32-bit float WAV" in message


def test_explicit_flac_peak_above_one_still_float_wav(tmp_path, monkeypatch, capsys):
    sr = 8000
    mix = np.full((2, sr), 0.2, dtype=np.float32)
    vocals = np.full((2, sr), 1.7, dtype=np.float32)
    _install(monkeypatch, mix, vocals, sr)
    song = _song(tmp_path, mix, sr)

    written = separate_file(str(song), str(tmp_path / "out"), fmt="flac", verbose=True)
    voc = next(p for p in written if "vocals" in p.name)
    assert voc.suffix == ".wav"
    assert sf.info(str(voc)).subtype == "FLOAT"
    data, _ = sf.read(str(voc), dtype="float32", always_2d=True)
    assert np.allclose(data, 1.7, atol=1e-5)
    message = capsys.readouterr().out
    assert "FLAC 无法保存超过 0 dBFS 的采样" in message
    assert "整数 PCM 会削波" in message
    assert "32-bit float WAV" in message
