"""uvr-lite models：把可回收的退役权重列给用户看（T6 回收机制的唯一可见入口）。

退役文件（旧 .ckpt）不会再有代码路径碰它，回收动作只在「新权重校验通过」后发生
（download.py:_reclaim_retired）。用户想知道升级后能省多少磁盘，就得靠这条输出。
用例全程用 UVR_MODEL_DIR 指向 tmp_path：单测不得碰真实 models/。
"""

from uvr_lite.cli import main


def test_lists_existing_retired_weight(monkeypatch, tmp_path, capsys):
    """退役文件确实存在时：列出名字与体积，并说明回收时机。"""
    monkeypatch.setenv("UVR_MODEL_DIR", str(tmp_path))
    old = tmp_path / "bs_roformer_ep317.ckpt"
    old.write_bytes(b"x" * (2 << 20))  # 约 2 MB：体积要按退役文件本身算

    assert main(["models"]) == 0

    out = capsys.readouterr().out
    assert f"可回收: {old.name}（2 MB）" in out, out
    assert "新权重校验通过后自动删除" in out


def test_lists_retired_weight_of_second_model(monkeypatch, tmp_path, capsys):
    """第二个模型的退役名单同样生效（名单以注册表为准，不硬编码模型名）。"""
    monkeypatch.setenv("UVR_MODEL_DIR", str(tmp_path))
    old = tmp_path / "mel_band_karaoke.ckpt"
    old.write_bytes(b"x" * (1 << 20))

    assert main(["models"]) == 0

    out = capsys.readouterr().out
    assert f"可回收: {old.name}（1 MB）" in out, out


def test_absent_retired_weight_not_listed(monkeypatch, tmp_path, capsys):
    """没有退役文件时不得出现「可回收」行（不给用户无中生有的提示）。"""
    monkeypatch.setenv("UVR_MODEL_DIR", str(tmp_path))

    assert main(["models"]) == 0

    out = capsys.readouterr().out
    assert "可回收" not in out, out
    # 既有行格式不变：仍逐模型列出状态与简介
    assert "bs_roformer_ep317" in out and "未下载" in out
