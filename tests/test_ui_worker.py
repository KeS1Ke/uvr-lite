"""SeparationWorker 进度跨文件测试：tracker 每文件重置，杜绝进度回跳。"""

import subprocess
import sys

from uvr_lite.models import DEFAULT_MODEL
from uvr_lite.ui.worker import (
    CombineParams,
    CombineWorker,
    SeparationParams,
    SeparationWorker,
    friendly_error,
)


# qapp 夹具由 tests/conftest.py 提供（session 级单例，模块内不得再建）
def test_ui_import_does_not_load_torch():
    """UI 启动路径（ui.main → worker）不得加载 torch。

    torch 导入约 2-6s + 上 GB 内存；引擎应在首个分离任务时才加载。
    子进程隔离：测试进程本身可能已因其他用例导入 torch。
    """
    code = ("import sys; import uvr_lite.ui.main; "
            "sys.exit(0 if 'torch' not in sys.modules else 1)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True)
    assert r.returncode == 0, (
        f"import uvr_lite.ui.main 不应加载 torch（实际已加载）: {r.stderr.decode()[-200:]}")


def _run_two_files(monkeypatch, files, params: SeparationParams):
    """mock Separator 按文件序号回调：文件 1 完整序列，文件 2 重复同序列。"""
    seq = [("decode", 0, 1), ("decode", 1, 1),
           ("chunk", 0, 100), ("chunk", 50, 100), ("chunk", 100, 100),
           ("infer", 1, 1),
           ("write", 1, 2), ("write", 2, 2)]

    class FakeSeparator:
        def __init__(self, **kw):
            pass

        def separate(self, path, out_dir, progress_callback=None, **kw):
            for phase, done, total in seq:
                assert progress_callback(phase, done, total) is True
            return [path]

    # 桩文件不是真音频；预检在 run() 前段，单测只关心进度，不让它挡掉 Separator。
    monkeypatch.setattr("uvr_lite.ui.files.precheck_audio", lambda _p: True)
    # run() 内惰性 `from ..engine import Separator` → patch 定义处
    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    w = SeparationWorker(files, "out", params)
    progress = []
    w.progress.connect(lambda ph, d, t, idx, ftot, pct: progress.append((idx, ph, pct)))
    w.run()
    return progress


def test_file_two_chunk_starts_from_zero(tmp_path, monkeypatch, qapp):
    """文件 2 的 chunk 应从 ~5% 开始（tracker 重置），不能从 50% 起（_pass_done 残留）。"""
    f1 = tmp_path / "a.wav"
    f2 = tmp_path / "b.wav"
    f1.write_bytes(b"x")
    f2.write_bytes(b"x")
    progress = _run_two_files(monkeypatch, [f1, f2], SeparationParams(bigshifts=1))

    file2_chunks = [(ph, pct) for idx, ph, pct in progress if idx == 1 and ph == "chunk"]
    assert file2_chunks, "应收到文件 2 的 chunk 回调"
    assert file2_chunks[0][1] == 5, f"文件 2 chunk 起点应为 5%（实际 {file2_chunks[0][1]}%）"


def test_progress_monotonic_within_file(tmp_path, monkeypatch, qapp):
    """单文件内进度不应回跳（chunk 升到 50 后 infer 不应打回）。"""
    f1 = tmp_path / "a.wav"
    f1.write_bytes(b"x")
    progress = _run_two_files(monkeypatch, [f1], SeparationParams(bigshifts=1))
    pcts = [pct for _, _, pct in progress]
    assert pcts == sorted(pcts), f"文件内进度应单调: {pcts}"


def _capture_engine_calls(monkeypatch, tmp_path, params):
    """跑一次 worker，记录 Separator 构造与 separate 收到的关键字参数。"""
    captured = {}

    class FakeSeparator:
        def __init__(self, **kw):
            captured["init"] = kw

        def separate(self, path, out_dir, **kw):
            captured["separate"] = kw
            return [path]

    monkeypatch.setattr("uvr_lite.ui.files.precheck_audio", lambda _p: True)
    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    SeparationWorker([f], "out", params).run()
    return captured


def test_params_forwarded_to_engine_unchanged(tmp_path, monkeypatch, qapp):
    """params dict → dataclass 后，传给引擎的参数名与取值必须逐字不变。"""
    captured = _capture_engine_calls(monkeypatch, tmp_path, SeparationParams(
        model_name="mel_band_karaoke", device="cuda", fmt="flac", pcm="PCM_16",
        bigshifts=3, batch_size=2, num_overlap=4, tta=True,
    ))
    assert captured["init"] == {
        "model_name": "mel_band_karaoke",
        "device": "cuda",
        "batch_size": 2,
        "num_overlap": 4,
        "verbose": False,
    }
    sep_kw = dict(captured["separate"])
    assert sep_kw.pop("progress_callback") is not None
    assert sep_kw == {"pcm": "PCM_16", "fmt": "flac", "bigshifts": 3, "tta": True}


def test_params_defaults_match_previous_dict(tmp_path, monkeypatch, qapp):
    """默认值与原先 .get(key, default) 的默认值一致（含 batch/overlap 的 None）。"""
    captured = _capture_engine_calls(monkeypatch, tmp_path, SeparationParams())
    assert captured["init"] == {
        "model_name": DEFAULT_MODEL,
        "device": "auto",
        "batch_size": None,   # None = 回落模型配置（UI 的 0 值已转 None）
        "num_overlap": None,
        "verbose": False,
    }
    sep_kw = dict(captured["separate"])
    sep_kw.pop("progress_callback")
    assert sep_kw == {"pcm": "PCM_24", "fmt": "auto", "bigshifts": 1, "tta": False}


def test_all_rejected_does_not_load_model(tmp_path, monkeypatch, qapp):
    """预检全失败：不 import/构造 Separator，all_finished 只发一次。"""
    created = []

    class FakeSeparator:
        def __init__(self, **kw):
            created.append(kw)

    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    f = tmp_path / "nope.wav"
    f.write_bytes(b"not audio at all")
    worker = SeparationWorker([f], "out", SeparationParams())
    finished = []
    ready = []
    worker.precheck_ready.connect(lambda indices: ready.append(list(indices)))
    worker.all_finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert created == []
    assert ready == [[]]
    assert finished == [(0, 1, False)]


def test_cancel_during_precheck_skips_model(tmp_path, monkeypatch, qapp):
    """预检期间取消：当前文件之后不再继续，且不加载模型。"""
    created = []

    class FakeSeparator:
        def __init__(self, **kw):
            created.append(kw)

    monkeypatch.setattr("uvr_lite.engine.Separator", FakeSeparator)
    files = []
    for name in ("a.wav", "b.wav"):
        p = tmp_path / name
        p.write_bytes(b"x")
        files.append(p)
    worker = SeparationWorker(files, "out", SeparationParams())

    def _precheck(_path):
        worker.cancel()
        return True

    monkeypatch.setattr("uvr_lite.ui.files.precheck_audio", _precheck)
    finished = []
    worker.all_finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert created == []
    assert finished == [(0, 0, True)]


def test_engine_import_error_emits_all_finished_once(tmp_path, monkeypatch, qapp):
    """引擎导入失败不能让 run() 空退出：恰好一次 all_finished。"""
    monkeypatch.setattr("uvr_lite.ui.files.precheck_audio", lambda _p: True)
    monkeypatch.setitem(sys.modules, "uvr_lite.engine", None)
    f = tmp_path / "a.wav"
    f.write_bytes(b"x")
    worker = SeparationWorker([f], "out", SeparationParams())
    finished = []
    failed = []
    worker.file_failed.connect(lambda idx, err: failed.append((idx, err)))
    worker.all_finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert len(finished) == 1
    assert finished[0][0] == 0 and finished[0][2] is False
    assert failed and failed[0][0] == 0


def test_friendly_error_survives_missing_audioread(monkeypatch):
    """audioread 导入失败不得从 friendly_error 冒出去。"""
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "audioread" or name.startswith("audioread."):
            raise ImportError("no audioread")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert friendly_error(RuntimeError("boom")) == "RuntimeError: boom"


def test_combine_import_error_emits_once(tmp_path, monkeypatch, qapp):
    monkeypatch.setitem(sys.modules, "uvr_lite.mix", None)
    worker = CombineWorker(
        [(tmp_path / "a-vocals.wav", tmp_path / "a-instrumental.wav")],
        str(tmp_path), CombineParams())
    finished = []
    worker.all_finished.connect(lambda *args: finished.append(args))
    worker.run()
    assert len(finished) == 1
    assert finished[0][2] is False
