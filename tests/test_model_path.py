"""权重路径以注册表 filename 为准（.safetensors），不再写死 .ckpt。"""

import uvr_lite.download as dl


def test_bs_roformer_model_file_uses_safetensors(tmp_path, monkeypatch):
    """真实注册表：bs_roformer_ep317 的本地文件名是 lite safetensors。"""
    monkeypatch.setattr(dl, "models_dir", lambda: tmp_path)

    path = dl.model_file("bs_roformer_ep317")
    assert path.name == "bs_roformer_ep317.lite.safetensors"
    assert not path.name.endswith(".ckpt")
    assert path == tmp_path / "bs_roformer_ep317.lite.safetensors"
    assert path.exists() is False

    path.write_bytes(b"weights")
    assert dl.model_file("bs_roformer_ep317").exists() is True


def test_karaoke_model_file_is_safetensors(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "models_dir", lambda: tmp_path)

    path = dl.model_file("mel_band_karaoke")
    assert path.name.endswith(".safetensors")
    assert "karaoke" in path.name
