"""分离引擎：封装 msst/ 推理子集（vendored，来自 ZFTurbo MSST），提供单文件人声/伴奏分离。

流程：soundfile+soxr 读音频（m4a 兜底 audioread）-> 归一化（可选）
-> BigShifts 圆形时移平均 -> BS-RoFormer 前向
-> instrumental = mix - vocals（数学无损）-> 写 FLAC/WAV。

会话复用：Separator 类把「模型加载」与「单文件分离」解耦——CLI 多文件与 GUI 批处理
共用一个 Separator，模型只加载一次（640MB ckpt + 图构建约 20s，逐文件重载是最大浪费）。
separate_file() 保留为薄封装（每次调用新建会话），兼容旧 API 与测试。
"""

import argparse
import pickle
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch

# msst/ 采用 `from models.xxx import ...` / `from utils.xxx import ...` 绝对导入，
# 因此必须把 msst 目录加入 sys.path（与上游 inference.py 的做法一致）。
_MSST_DIR = str(Path(__file__).resolve().parent.parent / "msst")
if _MSST_DIR not in sys.path:
    sys.path.insert(0, _MSST_DIR)

from utils.audio_utils import denormalize_audio, normalize_audio  # noqa: E402
from utils.model_utils import (  # noqa: E402
    apply_tta,
    bigshifts_wrapper,
    load_start_checkpoint,
    prefer_target_instrument,
)
from utils.settings import get_model_from_config  # noqa: E402

from .download import config_path, ensure_model  # noqa: E402
from .models import DEFAULT_MODEL, get_model_info  # noqa: E402


class CancelledError(Exception):
    """用户请求取消当前分离任务（进度回调返回 False 时抛出）。"""


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_model(model_name: str, ckpt_path: Path, device: str):
    """加载模型与配置（加载流程对齐 MSST 上游的 inference 惯例）。"""
    info = get_model_info(model_name)
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(info["model_type"], str(config_path(model_name)))
    # weights_only=True：只允许纯张量/基础类型，杜绝 pickle 任意代码执行面。
    # 注册表模型均为 fp16-lite 纯张量格式（{"state_dict": ...}），实测可加载；
    # 上游 exotic 格式失败时给出可行动的错误而非静默回退到不安全加载。
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"权重文件含非张量对象，安全模式加载失败: {ckpt_path}\n"
            "请删除该文件后重新下载；若持续失败请提 issue（可能上游格式变更）"
        ) from e
    args = argparse.Namespace(
        start_check_point=str(ckpt_path),
        model_type=info["model_type"],
        lora_checkpoint_loralib="",
    )
    load_start_checkpoint(args, model, checkpoint, type_="inference")
    model.to(device)
    model.eval()

    # CPU 上 AMP 无意义，且避免 autocast 兼容问题
    if device.startswith("cpu"):
        config.training["use_amp"] = False
    _warmup(model, config, device)
    return model, config


def _warmup(model, config, device: str) -> None:
    """预热：跑一次短前向，消除首块推理的 CUDA/cuDNN 内核选择抖动。失败静默。"""
    try:
        n_ch = int(getattr(config.audio, "num_channels", 2))
        chunk = int(getattr(config.inference, "chunk_size", 352800)) // 10
        dummy = torch.zeros(1, n_ch, max(chunk, 2048))
        with torch.inference_mode():
            model(dummy.to(device))
    except Exception:
        pass


def _load_audio(path: Path, sr: int) -> np.ndarray:
    """解码音频为 (channels, samples) float32 并重采样到 sr。

    主路径 soundfile（flac/wav/ogg/mp3 原生解码，libsndfile）；
    m4a 等 libsndfile 不支持的格式回退 audioread（需系统 ffmpeg）。
    替代 librosa.load，连带省掉 scipy/numba/llvmlite 等约 320MB 依赖。
    """
    try:
        data, orig_sr = sf.read(str(path), dtype="float32", always_2d=True)
    except RuntimeError:
        data, orig_sr = _read_audioread(path)
    if orig_sr != sr:
        # soxr 的 2D 语义为 (samples, channels)，与 soundfile/audioread 布局一致
        data = soxr.resample(data, orig_sr, sr, quality="HQ")
    return data.T  # (frames, channels) -> (channels, frames)


def _read_audioread(path: Path):
    """audioread 兜底解码（int16 PCM → float32，与 librosa 的 audioread 路径一致）。"""
    import audioread

    with audioread.audio_open(str(path)) as af:
        orig_sr = af.samplerate
        ch = af.channels
        blocks = [
            np.frombuffer(b, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
            for b in af
        ]
    return np.concatenate(blocks, axis=0), orig_sr


class Separator:
    """引擎会话：模型只加载一次，可连续分离多个文件。

    CLI 多文件与 GUI 批处理都通过它复用模型，避免每文件重复加载
    （640MB ckpt 读取 + 图构建约 20s/次）。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto",
                 batch_size: int | None = None, num_overlap: int | None = None,
                 verbose: bool = True):
        self.model_name = model_name
        self.device = pick_device(device)
        self.verbose = verbose

        ckpt = ensure_model(model_name)
        self.model, self.config = load_model(model_name, ckpt, self.device)

        # 低显存 GPU 可调小批大小防 OOM（默认取模型配置 yaml）
        if batch_size is not None and batch_size >= 1:
            self.config.inference["batch_size"] = batch_size
        # 质量/速度开关：num_overlap 越小越快（1 = 无重叠，约 2x 提速；默认取 yaml）
        if num_overlap is not None and num_overlap >= 1:
            self.config.inference["num_overlap"] = num_overlap

        self.sample_rate: int = getattr(self.config.audio, "sample_rate", 44100)

    def separate(
        self,
        input_path: str,
        out_dir: str,
        pcm: str = "PCM_24",
        fmt: str = "auto",  # auto | flac | wav
        bigshifts: int = 1,
        tta: bool = False,
        progress_callback: Callable[[str, int, int], bool] | None = None,
    ) -> list[Path]:
        """分离单个音频文件，输出 {stem}-vocals 与 {stem}-instrumental 两个文件。

        progress_callback(phase, done, total) -> bool：
          phase 为 "decode" / "infer" / "chunk" / "tta" / "write"；
          返回 False 表示请求取消，引擎抛 CancelledError 并清理已写出的半成品。
          不传（CLI 路径）时行为与旧版完全一致。
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"输入文件不存在: {input_path}")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _cb(phase: str, done: int, total: int) -> None:
            """上报进度；用户回调返回 False（取消请求）时抛 CancelledError。"""
            if progress_callback is not None and not progress_callback(phase, done, total):
                raise CancelledError(f"用户在 {phase} 阶段取消了任务")

        if self.verbose:
            print(f"分离: {input_path.name} | 模型: {self.model_name} | 设备: {self.device}"
                  f" | 采样率: {self.sample_rate}")

        _cb("decode", 0, 1)
        mix = _load_audio(input_path, self.sample_rate)  # (channels, samples)，恒为 2D
        if mix.shape[0] == 1 and getattr(self.config.audio, "num_channels", 1) == 2:
            mix = np.concatenate([mix, mix], axis=0)  # mono 输入按 stereo 模型复制双声道
        _cb("decode", 1, 1)

        mix_orig = mix.copy()

        norm_params = None
        if getattr(self.config.inference, "normalize", False):
            mix, norm_params = normalize_audio(mix)

        model_type = _model_type_of(self.model_name)
        waveforms = bigshifts_wrapper(
            self.config, self.model, mix, self.device,
            model_type=model_type, pbar=self.verbose, bigshifts=bigshifts,
            progress_cb=lambda done, total: _cb("infer", done, total),
            demix_progress_cb=lambda done, total: _cb("chunk", done, total),
        )
        if tta:
            waveforms = apply_tta(
                self.config, self.model, mix, waveforms, self.device,
                model_type, bigshifts=bigshifts, pbar=self.verbose,
                progress_cb=lambda done, total: _cb("tta", done, total),
            )

        # instrumental = 原混合 - 目标声部（数学无损）
        instruments = prefer_target_instrument(self.config)[:]
        target = "vocals" if "vocals" in [i.lower() for i in instruments] else instruments[0]
        target_key = next(i for i in instruments if i.lower() == target)
        waveforms["instrumental"] = mix_orig - waveforms[target_key]

        written: list[Path] = []
        try:
            for idx, (instr_key, stem_name) in enumerate(
                [(target_key, target), ("instrumental", "instrumental")], start=1
            ):
                est = waveforms[instr_key]
                if norm_params is not None:
                    est = denormalize_audio(est, norm_params)

                peak = float(np.abs(est).max())
                codec = "flac" if (fmt == "flac" or (fmt == "auto" and peak <= 1.0)) else "wav"

                out_path = out_dir / f"{input_path.stem}-{stem_name}.{codec}"
                sf.write(out_path, est.T, self.sample_rate, subtype=pcm)
                if self.verbose:
                    print(f"  写出: {out_path}（峰值 {peak:.3f}）")
                written.append(out_path)
                _cb("write", idx, 2)
        except CancelledError:
            # 取消：清理已写出的半成品，不留残缺文件
            for p in written:
                with suppress(OSError):
                    p.unlink(missing_ok=True)
            raise
        return written


def separate_file(
    input_path: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL,
    pcm: str = "PCM_24",
    device: str = "auto",
    fmt: str = "auto",  # auto | flac | wav
    bigshifts: int = 1,
    tta: bool = False,
    batch_size: int | None = None,
    num_overlap: int | None = None,
    verbose: bool = True,
    progress_callback: Callable[[str, int, int], bool] | None = None,
) -> list[Path]:
    """兼容薄封装：每次调用新建会话（模型加载一次）。

    批量场景请直接创建 Separator 复用，避免每文件重载 640MB 模型。
    """
    sep = Separator(model_name=model_name, device=device, batch_size=batch_size,
                    num_overlap=num_overlap, verbose=verbose)
    return sep.separate(input_path, out_dir, pcm=pcm, fmt=fmt,
                        bigshifts=bigshifts, tta=tta,
                        progress_callback=progress_callback)


def _model_type_of(model_name: str) -> str:
    return get_model_info(model_name)["model_type"]
