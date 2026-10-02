"""合成引擎：人声＋伴奏 → 整曲（分离的逆运算）。"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from uvr_lite.errors import CancelledError
from uvr_lite.mix import combine


def _write(path: Path, data: np.ndarray, sr: int = 8000) -> Path:
    """写 WAV FLOAT，避免测试里量化误差干扰精确断言。"""
    sf.write(str(path), data.T, sr, subtype="FLOAT")
    return path


def _read(path: Path) -> np.ndarray:
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T


@pytest.fixture
def stems(tmp_path):
    """同源分离产物：inst = mix − vocals（float32）。返回 (人声路径, 伴奏路径, 数组...)。"""
    sr = 8000
    rng = np.random.default_rng(42)
    mix = (rng.standard_normal((2, sr)) * 0.2).astype(np.float32)
    vocals = (mix * 0.4).astype(np.float32)
    inst = (mix - vocals).astype(np.float32)
    v_path = _write(tmp_path / "song-vocals.wav", vocals, sr)
    i_path = _write(tmp_path / "song-instrumental.wav", inst, sr)
    return v_path, i_path, mix, vocals, inst


def test_combine_reconstructs_original_mix(stems, tmp_path):
    v, i, mix, _, _ = stems

    out = combine(str(v), str(i), str(tmp_path / "out"), fmt="wav")

    assert out.name == "song-mix.wav"
    assert np.allclose(_read(out), mix, atol=1e-4), "同源 stems 相加应还原原曲"


def test_combine_applies_gains(stems, tmp_path):
    v, i, _, vocals, inst = stems

    out = combine(str(v), str(i), str(tmp_path / "out"),
                  vocal_gain=0.5, inst_gain=1.0, fmt="wav")

    assert np.allclose(_read(out), vocals * 0.5 + inst, atol=1e-4)


def test_combine_pads_shorter_track(tmp_path):
    vocals = np.ones((1, 1000), dtype=np.float32) * 0.1
    inst = np.ones((1, 1500), dtype=np.float32) * 0.2
    v = _write(tmp_path / "a-vocals.wav", vocals)
    i = _write(tmp_path / "a-instrumental.wav", inst)

    out = combine(str(v), str(i), str(tmp_path / "out"), fmt="wav")

    data = _read(out)
    assert data.shape[1] == 1500, "长度取最长，短轨补零"
    assert np.allclose(data[0, :1000], 0.3, atol=1e-5)
    assert np.allclose(data[0, 1000:], 0.2, atol=1e-5)


def test_combine_mono_and_stereo_promotes_to_stereo(tmp_path):
    v = _write(tmp_path / "a-vocals.wav", np.full((1, 100), 0.1, dtype=np.float32))
    i = _write(tmp_path / "a-instrumental.wav", np.full((2, 100), 0.2, dtype=np.float32))

    out = combine(str(v), str(i), str(tmp_path / "out"), fmt="wav")

    data = _read(out)
    assert data.shape == (2, 100)
    assert np.allclose(data, 0.3, atol=1e-5)


def test_combine_normalize_caps_peak(tmp_path):
    v = _write(tmp_path / "a-vocals.wav", np.full((1, 100), 0.8, dtype=np.float32))
    i = _write(tmp_path / "a-instrumental.wav", np.full((1, 100), 0.8, dtype=np.float32))

    out = combine(str(v), str(i), str(tmp_path / "out"), normalize=True, fmt="wav")

    assert float(np.abs(_read(out)).max()) == pytest.approx(0.891, abs=1e-4)


def test_combine_auto_format_wav_when_peak_exceeds_unity(tmp_path):
    v = _write(tmp_path / "a-vocals.wav", np.full((1, 100), 0.8, dtype=np.float32))
    i = _write(tmp_path / "a-instrumental.wav", np.full((1, 100), 0.8, dtype=np.float32))

    out = combine(str(v), str(i), str(tmp_path / "out"), fmt="auto", verbose=False)

    assert out.suffix == ".wav", "峰值 > 1 时 auto 落 WAV（与分离引擎规则一致）"


def test_combine_progress_phases(stems, tmp_path):
    vocals, inst, _, _, _ = stems
    events = []

    combine(str(vocals), str(inst), str(tmp_path / "out"),
            progress_callback=lambda ph, d, t: events.append((ph, d, t)) or True)

    phases = [ph for ph, _, _ in events]
    assert phases[0] == "decode" and phases[-1] == "write"
    assert "mix" in phases


def test_combine_cancel_cleans_partial_output(stems, tmp_path):
    """写完才收到取消（write 1/1 返回 False）时，半成品必须删除。"""
    vocals, inst, _, _, _ = stems
    out_dir = tmp_path / "out"

    def cb(phase, done, total):
        return not (phase == "write" and done == 1)

    with pytest.raises(CancelledError):
        combine(str(vocals), str(inst), str(out_dir), progress_callback=cb)

    assert list(out_dir.glob("*.wav")) == [] and list(out_dir.glob("*.flac")) == []


def test_combine_cancel_before_write_writes_nothing(stems, tmp_path):
    vocals, inst, _, _, _ = stems
    out_dir = tmp_path / "out"

    with pytest.raises(CancelledError):
        combine(str(vocals), str(inst), str(out_dir),
                progress_callback=lambda ph, d, t: ph != "mix")

    assert list(out_dir.iterdir()) == []


def test_combine_missing_files_raise(tmp_path):
    with pytest.raises(FileNotFoundError, match="人声轨"):
        combine(str(tmp_path / "nope-vocals.wav"), str(tmp_path / "nope-inst.wav"),
                str(tmp_path / "out"))


def test_combine_same_stem_different_dirs_keeps_both(tmp_path):
    """不同目录、基名都是 song 的两对写到同一输出目录时，后一次不得盖掉前一次。"""
    left = np.full((1, 200), 0.2, dtype=np.float32)
    right = np.full((1, 200), 0.4, dtype=np.float32)
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    lv = _write(left_dir / "song-vocals.wav", left)
    li = _write(left_dir / "song-instrumental.wav", left)
    rv = _write(right_dir / "song-vocals.wav", right)
    ri = _write(right_dir / "song-instrumental.wav", right)
    out_dir = tmp_path / "out"

    first = combine(str(lv), str(li), str(out_dir), verbose=False)
    second = combine(str(rv), str(ri), str(out_dir), verbose=False)

    assert len(list(out_dir.iterdir())) == 2
    assert first.name == "song-mix.flac", "峰值 ≤ 1 时 auto 仍落 flac，且首次不改名"
    assert np.allclose(_read(first), left * 2, atol=1e-5), "第一次的和不得被第二次盖掉"
    assert second.name == "song-mix-2.flac"
    assert np.allclose(_read(second), right * 2, atol=1e-5)


def test_combine_existing_output_is_not_truncated(tmp_path):
    """预先放好的 song-mix.flac 不得被截断；峰值 ≤ 1 的新结果落到 song-mix-2.flac。"""
    sentinel = np.full((1, 160), 0.25, dtype=np.float32)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    preexisting = out_dir / "song-mix.flac"
    sf.write(str(preexisting), sentinel.T, 8000, subtype="PCM_24")
    before = _read(preexisting)

    vocals = np.full((1, 160), 0.2, dtype=np.float32)
    inst = np.full((1, 160), 0.3, dtype=np.float32)
    v = _write(tmp_path / "song-vocals.wav", vocals)
    i = _write(tmp_path / "song-instrumental.wav", inst)

    out = combine(str(v), str(i), str(out_dir), verbose=False)

    assert out.name == "song-mix-2.flac"
    assert np.array_equal(_read(preexisting), before), "已有 song-mix.flac 的样本不得被截断重写"
    assert np.allclose(_read(out), vocals + inst, atol=1e-5)
