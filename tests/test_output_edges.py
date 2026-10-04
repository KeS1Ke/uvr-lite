"""合成边界：.part 原子写、声道不匹配、归一化静音与增益上限。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvr_lite.errors import CancelledError
from uvr_lite.mix import combine


def _write(path: Path, data: np.ndarray, sr: int = 8000) -> Path:
    sf.write(str(path), data.T, sr, subtype="FLOAT")
    return path


def _read(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T


def _pair(tmp_path, vocals, inst):
    v = _write(tmp_path / "song-vocals.wav", vocals)
    i = _write(tmp_path / "song-instrumental.wav", inst)
    return v, i


def test_write_goes_to_part_then_replaces(monkeypatch, tmp_path):
    vocals = np.full((1, 80), 0.2, dtype=np.float32)
    inst = np.full((1, 80), 0.3, dtype=np.float32)
    v, i = _pair(tmp_path, vocals, inst)
    out_dir = tmp_path / "out"
    seen = []
    real_write = sf.write

    def wrapped(file, data, samplerate, **kwargs):
        seen.append(Path(file).name)
        return real_write(file, data, samplerate, **kwargs)

    monkeypatch.setattr("uvr_lite.mix.sf.write", wrapped)

    out = combine(str(v), str(i), str(out_dir), fmt="wav", verbose=False)

    assert seen == ["song-mix.wav.part"]
    assert out.name == "song-mix.wav"
    assert out.exists()
    assert not (out_dir / "song-mix.wav.part").exists()
    assert np.allclose(_read(out), vocals + inst, atol=1e-5)


def test_oserror_during_write_leaves_no_final_name(monkeypatch, tmp_path):
    v, i = _pair(
        tmp_path,
        np.full((1, 40), 0.2, dtype=np.float32),
        np.full((1, 40), 0.2, dtype=np.float32),
    )
    out_dir = tmp_path / "out"
    seen = []

    def boom(file, data, samplerate, **kwargs):
        seen.append(Path(file).name)
        Path(file).write_bytes(b"1234567")
        raise OSError("disk full")

    monkeypatch.setattr("uvr_lite.mix.sf.write", boom)

    with pytest.raises(OSError, match="disk full"):
        combine(str(v), str(i), str(out_dir), fmt="wav", verbose=False)

    assert seen == ["song-mix.wav.part"]
    assert list(out_dir.iterdir()) == []
    assert not (out_dir / "song-mix.wav").exists()


def test_cancel_mid_write_removes_part(monkeypatch, tmp_path):
    v, i = _pair(
        tmp_path,
        np.full((1, 40), 0.2, dtype=np.float32),
        np.full((1, 40), 0.2, dtype=np.float32),
    )
    out_dir = tmp_path / "out"

    def boom(file, data, samplerate, **kwargs):
        Path(file).write_bytes(b"1234567")
        raise CancelledError("写到一半取消")

    monkeypatch.setattr("uvr_lite.mix.sf.write", boom)

    with pytest.raises(CancelledError):
        combine(str(v), str(i), str(out_dir), fmt="wav", verbose=False)

    assert list(out_dir.iterdir()) == []


def test_non_cancel_after_replace_keeps_finished_file(tmp_path):
    v, i = _pair(
        tmp_path,
        np.full((1, 40), 0.2, dtype=np.float32),
        np.full((1, 40), 0.2, dtype=np.float32),
    )
    out_dir = tmp_path / "out"

    def cb(phase, done, total):
        if phase == "write" and done == 1:
            raise RuntimeError("回调失败")
        return True

    with pytest.raises(RuntimeError, match="回调失败"):
        combine(str(v), str(i), str(out_dir), fmt="wav", verbose=False, progress_callback=cb)

    finished = out_dir / "song-mix.wav"
    assert finished.exists()
    assert not (out_dir / "song-mix.wav.part").exists()
    assert finished.stat().st_size > 7


def test_channel_mismatch_raises_chinese_value_error(tmp_path):
    v, i = _pair(
        tmp_path,
        np.zeros((2, 32), dtype=np.float32),
        np.zeros((4, 32), dtype=np.float32),
    )

    with pytest.raises(ValueError) as exc:
        combine(str(v), str(i), str(tmp_path / "out"), fmt="wav", verbose=False)

    msg = str(exc.value)
    assert "2" in msg and "4" in msg
    assert "声道" in msg
    assert "broadcast" not in msg
    assert list((tmp_path / "out").glob("*.wav")) == []


def test_mono_can_still_expand_to_more_channels(tmp_path):
    v, i = _pair(
        tmp_path,
        np.full((1, 16), 0.1, dtype=np.float32),
        np.full((4, 16), 0.2, dtype=np.float32),
    )

    out = combine(str(v), str(i), str(tmp_path / "out"), fmt="wav", verbose=False)

    data = _read(out)
    assert data.shape == (4, 16)
    assert np.allclose(data, 0.3, atol=1e-5)


def test_normalize_silence_is_not_amplified(tmp_path):
    vocals = np.full((1, 64), 1e-5, dtype=np.float32)
    inst = np.full((1, 64), 1e-5, dtype=np.float32)
    v, i = _pair(tmp_path, vocals, inst)

    out = combine(str(v), str(i), str(tmp_path / "out"),
                  normalize=True, fmt="wav", verbose=False)

    assert float(np.abs(_read(out)).max()) == pytest.approx(2e-5, abs=1e-6)


def test_normalize_gain_capped_at_100(tmp_path, capsys):
    vocals = np.full((1, 64), 0.003, dtype=np.float32)
    inst = np.full((1, 64), 0.003, dtype=np.float32)
    v, i = _pair(tmp_path, vocals, inst)

    out = combine(str(v), str(i), str(tmp_path / "out"),
                  normalize=True, fmt="wav", verbose=True)

    assert float(np.abs(_read(out)).max()) == pytest.approx(0.6, abs=1e-3)
    assert "归一化增益已限制为 100" in capsys.readouterr().out


def test_normalize_ordinary_peak_still_hits_target(tmp_path, capsys):
    v, i = _pair(
        tmp_path,
        np.full((1, 32), 0.8, dtype=np.float32),
        np.full((1, 32), 0.8, dtype=np.float32),
    )

    out = combine(str(v), str(i), str(tmp_path / "out"),
                  normalize=True, fmt="wav", verbose=True)

    assert float(np.abs(_read(out)).max()) == pytest.approx(0.891, abs=1e-4)
    assert "归一化增益已限制为 100" not in capsys.readouterr().out
