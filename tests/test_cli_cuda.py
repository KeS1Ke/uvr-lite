"""票 3（tdd）：CLI `uvr-lite install-cuda` 命令接线。"""

from uvr_lite.cli import main


def test_install_cuda_wires_progress_callback(monkeypatch):
    """命令解析 → 调用 install_cuda_torch 且传入进度回调（模拟全程回调不炸）。"""
    calls = {}

    def fake_install(progress_callback=None):
        calls["cb"] = progress_callback
        # 模拟下载+解压全程（累计尺度）：首/中/末
        assert progress_callback is not None
        progress_callback(0, 100)
        progress_callback(50, 100)
        progress_callback(100, 100)
        return None

    monkeypatch.setattr("uvr_lite.download.install_cuda_torch", fake_install)
    assert main(["install-cuda"]) == 0
    assert calls["cb"] is not None
