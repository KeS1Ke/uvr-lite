# coding: utf-8
"""分离任务 Worker：QThread 中逐文件调用引擎，进度/取消/失败经信号上报。

会话复用：run() 开始时创建一个 Separator（模型只加载一次），全部文件共用，
避免每文件重载 640MB 模型（批量场景的主要提速点）。
"""

from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import QObject, Signal

from ..engine import CancelledError, Separator
from ..models import DEFAULT_MODEL
from .progress import ProgressTracker


class SeparationWorker(QObject):
    # phase, done, total, file_idx, file_total, file_pct(0-100, 文件内)
    progress = Signal(str, int, int, int, int, int)
    file_done = Signal(int, list)      # file_idx, 写出文件列表
    file_failed = Signal(int, str)     # file_idx, 错误信息
    all_finished = Signal(int, int, bool)  # 成功数, 失败数, 是否取消

    def __init__(self, files: List[Path], out_dir: str, params: Dict):
        super().__init__()
        self.files = files
        self.out_dir = out_dir
        self.params = params
        self._cancel = False
        self._cur_idx = 0
        self._tracker = ProgressTracker(params.get("bigshifts", 1))

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        ok = failed = 0
        total = len(self.files)
        try:
            sep = Separator(
                model_name=self.params.get("model_name", DEFAULT_MODEL),
                device=self.params.get("device", "auto"),
                batch_size=self.params.get("batch_size"),
                num_overlap=self.params.get("num_overlap"),
                verbose=False,
            )
        except Exception as e:  # noqa: BLE001 —— 模型下载/加载失败 → 整个队列失败
            for idx in range(total):
                self.file_failed.emit(idx, friendly_error(e))
            self.all_finished.emit(0, total, False)
            return
        for idx, f in enumerate(self.files):
            self._cur_idx = idx
            # 每文件重置 tracker：_pass_done 残留会导致下一文件 chunk 从 50% 起算、
            # infer 回调再把进度打回（进度条回跳）
            self._tracker = ProgressTracker(self.params.get("bigshifts", 1))
            if self._cancel:
                break
            try:
                written = sep.separate(
                    str(f), str(self.out_dir),
                    pcm=self.params.get("pcm", "PCM_24"),
                    fmt=self.params.get("fmt", "auto"),
                    bigshifts=self.params.get("bigshifts", 1),
                    tta=self.params.get("tta", False),
                    progress_callback=self._on_progress,
                )
                ok += 1
                self.file_done.emit(idx, written)
            except CancelledError:
                self._cancel = True
                break
            except Exception as e:  # noqa: BLE001 —— 单文件失败跳过，继续队列
                failed += 1
                self.file_failed.emit(idx, friendly_error(e))
        self.all_finished.emit(ok, failed, self._cancel)

    def _on_progress(self, phase: str, done: int, total: int) -> bool:
        pct = int(round(self._tracker.on_progress(phase, done, total) * 100))
        self.progress.emit(phase, done, total, self._cur_idx, len(self.files), pct)
        return not self._cancel


class ModelDownloadWorker(QObject):
    """模型权重下载任务（复用 ensure_model：断点续传 + 多源回退 + SHA256）。"""

    progress = Signal(int, int)   # done_bytes, total_bytes
    finished = Signal(bool, str)  # 是否成功, 错误/取消信息

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ..download import ensure_model

            ensure_model(self.model_name, progress_callback=self._cb)
            self.finished.emit(True, "")
        except InterruptedError:
            self.finished.emit(False, "已取消（支持断点续传，可随时重新下载）")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e))

    def _cb(self, done: int, total: int) -> bool:
        self.progress.emit(done, total)
        return not self._cancel


class CudaTorchWorker(QObject):
    """CUDA 推理引擎下载安装任务（复用 install_cuda_torch：断点续传 + 镜像回退 + SHA256）。

    进度回调贯穿下载与解压阶段（累计尺度单调递增，进度条不回跳）；可取消
    （wheel 缓存保留，下次直接从解压开始）。
    """

    progress = Signal(int, int)   # done_bytes, total_bytes（下载+解压总量）
    finished = Signal(bool, str)  # 是否成功, 错误/取消信息

    def __init__(self, base: Path):
        super().__init__()
        self.base = Path(base)
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ..download import install_cuda_torch

            install_cuda_torch(self.base, progress_callback=self._cb)
            self.finished.emit(True, "")
        except InterruptedError:
            self.finished.emit(False, "已取消（断点续传，可随时重新下载）")
        except Exception as e:  # noqa: BLE001
            self.finished.emit(False, str(e))

    def _cb(self, done: int, total: int) -> bool:
        self.progress.emit(done, total)
        return not self._cancel


def friendly_error(e: Exception) -> str:
    """把异常转成非专业用户可读的中文信息。"""
    from audioread.exceptions import NoBackendError

    if isinstance(e, NoBackendError):
        return "音频解码失败：文件可能损坏或格式不受支持"
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}"
