"""分离任务 Worker：QThread 中逐文件调用引擎，进度/取消/失败经信号上报。

会话复用：同一进程内按 (model_name, device, batch_size, num_overlap) 缓存
Separator。run() 未命中或上次构造失败才重新构造（失败不入缓存，仍对每个
文件 file_failed）；命中则直接复用。一次任务内全部文件共用该实例，避免
重复加载 640MB 模型。换引擎或测试隔离时调用 clear_separator_cache()。

引擎惰性导入：worker 模块只依赖 Qt/轻量模块，torch（约 2-6s 导入 + 上 GB
内存）推迟到首个分离任务才加载——UI 启动、列表编辑零引擎开销。
"""

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..models import DEFAULT_MODEL
from .progress import ProgressTracker

# 进程内 Separator 会话：(model_name, device, batch_size, num_overlap) → 实例
_SEPARATOR_CACHE: dict[tuple, object] = {}


def clear_separator_cache() -> None:
    """丢掉已缓存的 Separator（测试隔离 / 换引擎后调用）。run() 不会自动清。"""
    _SEPARATOR_CACHE.clear()


def _cached_separator(Separator, params: dict):
    """按构造参数取 Separator：命中复用，未命中则构造并缓存。构造失败不缓存。

    只保留最近一次成功加载的会话。换模型或重叠窗口时先丢掉旧实例，
    避免两套 640MB 权重同时留在内存里。
    """
    model_name = params.get("model_name", DEFAULT_MODEL)
    device = params.get("device", "auto")
    # 0 与 None 在 engine 里同义（都表示沿用模型配置）；归一化避免重复加载
    batch_size = params.get("batch_size") or None
    num_overlap = params.get("num_overlap") or None
    key = (model_name, device, batch_size, num_overlap)
    cached = _SEPARATOR_CACHE.get(key)
    if cached is not None:
        return cached
    for stale in list(_SEPARATOR_CACHE):
        del _SEPARATOR_CACHE[stale]
    sep = Separator(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        num_overlap=num_overlap,
        verbose=False,
    )
    _SEPARATOR_CACHE[key] = sep
    return sep


class SeparationWorker(QObject):
    # phase, done, total, file_idx, file_total, file_pct(0-100, 文件内)
    progress = Signal(str, int, int, int, int, int)
    file_done = Signal(int, list)      # file_idx, 写出文件列表
    file_failed = Signal(int, str)     # file_idx, 错误信息
    all_finished = Signal(int, int, bool)  # 成功数, 失败数, 是否取消

    def __init__(self, files: list[Path], out_dir: str, params: dict):
        super().__init__()
        self.files = files
        self.out_dir = out_dir
        self.params = params
        self._cancel = False
        self._cur_idx = 0
        self._tracker = ProgressTracker(
            self.params.get("bigshifts", 1),
            tta=bool(self.params.get("tta", False)),
        )

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        # 惰性导入：首个分离任务才加载 torch（单测 patch 定义处 uvr_lite.engine.Separator）
        from ..engine import CancelledError, Separator

        ok = failed = 0
        total = len(self.files)
        try:
            sep = _cached_separator(Separator, self.params)
        except Exception as e:
            for idx in range(total):
                self.file_failed.emit(idx, friendly_error(e))
            self.all_finished.emit(0, total, self._cancel)
            return
        for idx, f in enumerate(self.files):
            self._cur_idx = idx
            # 每文件重置 tracker：_pass_done 残留会导致下一文件 chunk 从推理中途起算、
            # infer 回调再把进度打回（进度条回跳）
            self._tracker = ProgressTracker(
                self.params.get("bigshifts", 1),
                tta=bool(self.params.get("tta", False)),
            )
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
            except Exception as e:
                failed += 1
                self.file_failed.emit(idx, friendly_error(e))
        self.all_finished.emit(ok, failed, self._cancel)

    def _on_progress(self, phase: str, done: int, total: int) -> bool:
        pct = round(self._tracker.on_progress(phase, done, total) * 100)
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
        except Exception as e:
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
        except Exception as e:
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
