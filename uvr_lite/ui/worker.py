"""任务 Worker：QThread 中逐文件调用引擎，进度/取消/失败经信号上报。

- SeparationWorker：分离任务，run() 开始时创建一个 Separator（模型只加载
  一次），全部文件共用，避免每文件重载模型（批量场景的主要提速点）。
- CombineWorker：合成任务（人声＋伴奏回混），逐对调用 mix.combine，不加载
  模型；信号与 SeparationWorker 完全一致，主窗口接线可复用。

引擎惰性导入：worker 模块只依赖 Qt/轻量模块，torch（约 2-6s 导入 + 上 GB
内存）推迟到首个分离任务才加载——UI 启动、列表编辑、合成任务零引擎开销。
"""

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..log import log_exception
from ..models import DEFAULT_MODEL
from .progress import ProgressTracker


@dataclass
class SeparationParams:
    """UI 表单 → SeparationWorker 的参数（替代原先手拼的 dict）。

    字段名即引擎调用契约的一部分，默认值与 UI 表单的初始值一致；
    batch_size / num_overlap 为 None 时引擎回落模型自带配置（UI 的
    0 值已转成 None）。
    """

    model_name: str = DEFAULT_MODEL
    device: str = "auto"
    fmt: str = "auto"
    pcm: str = "PCM_24"
    bigshifts: int = 1
    batch_size: int | None = None
    num_overlap: int | None = None
    tta: bool = False


@dataclass
class CombineParams:
    """UI 合成表单 → CombineWorker 的参数。"""

    vocal_gain: float = 1.0
    inst_gain: float = 1.0
    fmt: str = "auto"
    pcm: str = "PCM_24"
    normalize: bool = False


class SeparationWorker(QObject):
    # phase, done, total, file_idx, file_total, file_pct(0-100, 文件内)
    progress = Signal(str, int, int, int, int, int)
    file_done = Signal(int, list)      # file_idx, 写出文件列表
    file_failed = Signal(int, str)     # file_idx, 错误信息
    all_finished = Signal(int, int, bool)  # 成功数, 失败数, 是否取消

    def __init__(self, files: list[Path], out_dir: str, params: SeparationParams):
        super().__init__()
        self.files = files
        self.out_dir = out_dir
        self.params = params
        self._cancel = False
        self._cur_idx = 0
        self._tracker = ProgressTracker(params.bigshifts)

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        # 惰性导入：首个分离任务才加载 torch（单测 patch 定义处 uvr_lite.engine.Separator）
        from ..engine import CancelledError, Separator

        ok = failed = 0
        total = len(self.files)
        try:
            sep = Separator(
                model_name=self.params.model_name,
                device=self.params.device,
                batch_size=self.params.batch_size,
                num_overlap=self.params.num_overlap,
                verbose=False,
            )
        except Exception as e:
            # 引擎加载失败（权重损坏/CUDA 不可用等）没有控制台可看 traceback，
            # 先落盘再按原样上报给用户（信号语义不变）
            log_exception(f"加载模型失败: {self.params.model_name}")
            for idx in range(total):
                self.file_failed.emit(idx, friendly_error(e))
            self.all_finished.emit(0, total, False)
            return
        for idx, f in enumerate(self.files):
            self._cur_idx = idx
            # 每文件重置 tracker：_pass_done 残留会导致下一文件 chunk 从 50% 起算、
            # infer 回调再把进度打回（进度条回跳）
            self._tracker = ProgressTracker(self.params.bigshifts)
            if self._cancel:
                break
            try:
                written = sep.separate(
                    str(f), str(self.out_dir),
                    pcm=self.params.pcm,
                    fmt=self.params.fmt,
                    bigshifts=self.params.bigshifts,
                    tta=self.params.tta,
                    progress_callback=self._on_progress,
                )
                ok += 1
                self.file_done.emit(idx, written)
            except CancelledError:
                self._cancel = True
                break
            except Exception as e:
                failed += 1
                log_exception(f"分离失败（第 {idx + 1}/{total} 个）: {f}")
                self.file_failed.emit(idx, friendly_error(e))
        self.all_finished.emit(ok, failed, self._cancel)

    def _on_progress(self, phase: str, done: int, total: int) -> bool:
        pct = round(self._tracker.on_progress(phase, done, total) * 100)
        self.progress.emit(phase, done, total, self._cur_idx, len(self.files), pct)
        return not self._cancel


class CombineWorker(QObject):
    """合成任务 Worker：逐对（人声, 伴奏）调用 mix.combine，信号与分离一致。"""

    # phase, done, total, file_idx, file_total, file_pct(0-100, 文件内)
    progress = Signal(str, int, int, int, int, int)
    file_done = Signal(int, list)      # file_idx, 写出文件列表
    file_failed = Signal(int, str)     # file_idx, 错误信息
    all_finished = Signal(int, int, bool)  # 成功数, 失败数, 是否取消

    def __init__(self, pairs: list[tuple[Path, Path]], out_dir: str, params: CombineParams):
        super().__init__()
        self.pairs = pairs
        self.out_dir = out_dir
        self.params = params
        self._cancel = False
        self._cur_idx = 0
        self._tracker = ProgressTracker()

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        # 惰性导入：mix 不 import torch，合成任务比分离任务更早可用
        from ..errors import CancelledError
        from ..mix import combine

        ok = failed = 0
        total = len(self.pairs)
        for idx, (vocals, instrumental) in enumerate(self.pairs):
            self._cur_idx = idx
            self._tracker = ProgressTracker()  # 每对重置，进度不回跳
            if self._cancel:
                break
            try:
                written = combine(
                    str(vocals), str(instrumental), str(self.out_dir),
                    vocal_gain=self.params.vocal_gain,
                    inst_gain=self.params.inst_gain,
                    pcm=self.params.pcm,
                    fmt=self.params.fmt,
                    normalize=self.params.normalize,
                    verbose=False,
                    progress_callback=self._on_progress,
                )
                ok += 1
                self.file_done.emit(idx, [str(written)])
            except CancelledError:
                self._cancel = True
                break
            except Exception as e:
                failed += 1
                log_exception(f"合成失败（第 {idx + 1}/{total} 对）: {vocals}")
                self.file_failed.emit(idx, friendly_error(e))
        self.all_finished.emit(ok, failed, self._cancel)

    def _on_progress(self, phase: str, done: int, total: int) -> bool:
        pct = round(self._tracker.on_progress(phase, done, total) * 100)
        self.progress.emit(phase, done, total, self._cur_idx, len(self.pairs), pct)
        return not self._cancel


class _DownloadWorker(QObject):
    """下载类任务的公共骨架：进度转发、取消标志、完成/失败信号。

    子类只需实现 _install()（调用具体的下载函数）与定制 _cancelled_text；
    取消语义统一为：回调返回 False → 下载函数抛 InterruptedError →
    finished(False, _cancelled_text)。
    """

    progress = Signal(int, int)   # done_bytes, total_bytes
    finished = Signal(bool, str)  # 是否成功, 错误/取消信息

    _cancelled_text = "已取消（断点续传，可随时重新下载）"

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            self._install()
        except InterruptedError:
            # 取消是用户主动行为，不是故障：不写日志（正常路径不该有噪声）
            self.finished.emit(False, self._cancelled_text)
        except Exception as e:
            # 下载失败的详细原因（哪个源、第几次重试）落盘，UI 只显示一行摘要
            log_exception(f"下载任务失败: {type(self).__name__}")
            self.finished.emit(False, str(e))
        else:
            self.finished.emit(True, "")

    def _install(self) -> None:
        raise NotImplementedError

    def _cb(self, done: int, total: int) -> bool:
        self.progress.emit(done, total)
        return not self._cancel


class ModelDownloadWorker(_DownloadWorker):
    """模型权重下载任务（复用 ensure_model：断点续传 + 多源回退 + SHA256）。"""

    _cancelled_text = "已取消（支持断点续传，可随时重新下载）"

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def _install(self) -> None:
        from ..download import ensure_model

        ensure_model(self.model_name, progress_callback=self._cb)


class CudaTorchWorker(_DownloadWorker):
    """CUDA 推理引擎下载安装任务（复用 install_cuda_torch：断点续传 + 镜像回退 + SHA256）。

    进度回调贯穿下载与解压阶段（累计尺度单调递增，进度条不回跳）；可取消
    （wheel 缓存保留，下次直接从解压开始）。
    """

    def __init__(self, base: Path):
        super().__init__()
        self.base = Path(base)

    def _install(self) -> None:
        from ..download import install_cuda_torch

        install_cuda_torch(self.base, progress_callback=self._cb)


def friendly_error(e: Exception) -> str:
    """把异常转成非专业用户可读的中文信息。"""
    from audioread.exceptions import NoBackendError

    if isinstance(e, NoBackendError):
        return "音频解码失败：文件可能损坏或格式不受支持"
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg}" if msg else f"{type(e).__name__}"
