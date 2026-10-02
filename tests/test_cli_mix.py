"""CLI `uvr-lite mix`：单对合成、--batch 自动配对、参数校验。"""

import numpy as np
import soundfile as sf

from uvr_lite.cli import main


def _stems(tmp_path, base="song"):
    sr = 8000
    v = tmp_path / f"{base}-vocals.wav"
    i = tmp_path / f"{base}-instrumental.wav"
    sf.write(str(v), np.full((100, 1), 0.2, dtype=np.float32), sr)
    sf.write(str(i), np.full((100, 1), 0.3, dtype=np.float32), sr)
    return v, i


def test_cli_mix_single_pair(tmp_path, capsys):
    v, i = _stems(tmp_path)
    out = tmp_path / "out"

    code = main(["mix", str(v), str(i), "-o", str(out)])

    assert code == 0
    assert (out / "song-mix.flac").exists()
    assert "song-mix.flac" in capsys.readouterr().out


def test_cli_mix_batch_pairs(tmp_path, capsys):
    _stems(tmp_path, "a")
    _stems(tmp_path, "b")
    out = tmp_path / "out"

    code = main(["mix", "--batch", str(tmp_path), "-o", str(out)])

    assert code == 0
    assert (out / "a-mix.flac").exists()
    assert (out / "b-mix.flac").exists()
    text = capsys.readouterr().out
    assert "写出" in text


def test_cli_mix_batch_reports_unmatched_and_ignored(tmp_path, capsys):
    _stems(tmp_path, "a")
    (tmp_path / "b-vocals.wav").write_bytes(b"x")
    (tmp_path / "plain.wav").write_bytes(b"x")

    code = main(["mix", "--batch", str(tmp_path), "-o", str(tmp_path / "out")])

    assert code == 0
    text = capsys.readouterr().out
    assert "未配对（缺伴奏）: b-vocals.wav" in text
    assert "跳过（非分离音轨命名）: plain.wav" in text


def test_cli_mix_batch_no_pairs_errors(tmp_path, capsys):
    (tmp_path / "plain.wav").write_bytes(b"x")

    code = main(["mix", "--batch", str(tmp_path), "-o", str(tmp_path / "out")])

    assert code == 1
    assert "没有可配对的音轨" in capsys.readouterr().out


def test_cli_mix_missing_argument_errors(tmp_path, capsys):
    v, _ = _stems(tmp_path)

    code = main(["mix", str(v), "-o", str(tmp_path / "out")])

    assert code == 1
    assert "请同时给出人声与伴奏文件" in capsys.readouterr().out


def test_cli_mix_gains_and_options(tmp_path):
    v, i = _stems(tmp_path)
    out = tmp_path / "out"

    code = main(["mix", str(v), str(i), "-o", str(out),
                 "--vocal-gain", "0.5", "--inst-gain", "0.5",
                 "--format", "wav", "--normalize"])

    assert code == 0
    assert (out / "song-mix.wav").exists()
