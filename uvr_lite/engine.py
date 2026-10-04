"""分离引擎：封装 msst/ 推理子集（vendored，来自 ZFTurbo MSST），提供单文件人声/伴奏分离。

流程：soundfile+soxr 读音频（m4a 兜底 audioread）-> 归一化（可选）
-> BigShifts 圆形时移平均 -> BS-RoFormer 前向
-> instrumental = mix - vocals（数学无损）-> 写 FLAC/WAV。

会话复用：Separator 类把「模型加载」与「单文件分离」解耦——CLI 多文件与 GUI 批处理
共用一个 Separator，模型只加载一次（fp16 权重约 320MB + 图构建约 20s，逐文件重载是最大浪费）。
separate_file() 保留为薄封装（每次调用新建会话），兼容旧 API 与测试。
"""

import argparse
import os
import pickle
import sys
import zipfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

import numpy as np
import soundfile as sf
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

from .audio_io import load_audio  # noqa: E402
from .download import config_path, ensure_model  # noqa: E402
from .errors import CancelledError  # noqa: E402  re-export（旧调用方兼容）
from .models import DEFAULT_MODEL, get_model_info  # noqa: E402


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_model(model_name: str, ckpt_path: Path, device: str,
               batch_size: int | None = None):
    """加载模型与配置（加载流程对齐 MSST 上游的 inference 惯例）。

    batch_size 为 None 表示调用方未指定。仅 CPU 会在此时把推理 batch 改为 1；
    显式传入的值各设备都保留。
    """
    info = get_model_info(model_name)
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(info["model_type"], str(config_path(model_name)))
    # .safetensors 无 pickle 载入面，加载更快；.ckpt 用 weights_only=True
    # 只允许纯张量/基础类型。注册表模型为 fp16 扁平格式（纯张量）；
    # 上游 exotic 格式失败时给出可行动的错误而非静默回退到不安全加载。
    if ckpt_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(str(ckpt_path), device="cpu")
    else:
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, zipfile.BadZipFile, EOFError, RuntimeError) as e:
            raise RuntimeError(
                f"权重文件含非张量对象或已损坏，安全模式加载失败: {ckpt_path}\n"
                "请删除该文件后重新下载；若持续失败请提 issue（可能上游格式变更）"
            ) from e
    args = argparse.Namespace(
        start_check_point=str(ckpt_path),
        model_type=info["model_type"],
        lora_checkpoint_loralib="",
    )
    load_start_checkpoint(args, model, checkpoint, type_="inference")

    # 线程池只能在 model.to 之前设；interop 每个进程只能成功设一次。
    if device.startswith("cpu"):
        torch.set_num_threads(os.cpu_count() or 1)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        torch.backends.mkldnn.enabled = True
        # CPU 上 AMP 无意义，且避免 autocast 兼容问题
        config.training["use_amp"] = False

    # 显式 batch 各设备都尊重。CPU 未指定时用 1：更大 batch 几乎不加速，只增内存。
    # inference 是 ConfigDict，不是 dict，不能用 isinstance(..., dict) 判断。
    if batch_size is not None and batch_size >= 1:
        config.inference["batch_size"] = batch_size
    elif device.startswith("cpu"):
        inference = getattr(config, "inference", None)
        if inference is not None:
            inference["batch_size"] = 1

    model.to(device)
    model.eval()
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
    """兼容旧调用点 / 测试 monkeypatch 的薄封装：解码并重采样为 (channels, samples)。

    实现在 uvr_lite.audio_io（与合成路径共用同一条解码链路）。
    """
    return load_audio(path, sr)[0]


def _separate_output_path(out_dir: Path, stem: str, kind: str, codec: str) -> Path:
    """空闲时 `{stem}-{kind}.{codec}`；已占用则 `{stem}-{kind}-2`、`-3`……直到空位。

    用 Path.exists()：Windows 上大小写不同的同名文件也视为占用。
    """
    candidate = out_dir / f"{stem}-{kind}.{codec}"
    n = 2
    while candidate.exists():
        candidate = out_dir / f"{stem}-{kind}-{n}.{codec}"
        n += 1
    return candidate


def _output_format(fmt: str, pcm: str, peak: float) -> tuple[str, str, str | None]:
    """返回 (codec, subtype, 警告或 None)。峰值 >1 时改为 32-bit float WAV。"""
    if peak > 1.0:
        if fmt == "flac":
            warn = (
                f"峰值 {peak:.3f} 超过 0 dBFS，FLAC 无法保存超过 0 dBFS 的采样，"
                "整数 PCM 会削波，本次改为 32-bit float WAV"
            )
        else:
            warn = (
                f"峰值 {peak:.3f} 超过 0 dBFS，整数 PCM 会削波，本次改为 32-bit float WAV"
            )
        return "wav", "FLOAT", warn
    codec = "flac" if (fmt == "flac" or (fmt == "auto" and peak <= 1.0)) else "wav"
    return codec, pcm, None


class Separator:
    """引擎会话：模型只加载一次，可连续分离多个文件。

    CLI 多文件与 GUI 批处理都通过它复用模型，避免每文件重复加载
    （fp16 权重读取 + 图构建约 20s/次）。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto",
                 batch_size: int | None = None, num_overlap: int | None = None,
                 verbose: bool = True):
        self.model_name = model_name
        self.device = pick_device(device)
        self.verbose = verbose

        ckpt = ensure_model(model_name)
        self.model, self.config = load_model(
            model_name, ckpt, self.device, batch_size=batch_size,
        )

        # 质量/速度开关：num_overlap 越小越快（1 = 无重叠，约 2x 提速；默认取 yaml）。
        # 窗已按重叠区收束，overlap=1 不再产生零样本。
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

        目标名已存在时改为 {stem}-{kind}-2、{stem}-{kind}-3……不覆盖旧文件。
        progress_callback(phase, done, total) -> bool：
          phase 为 "decode" / "infer" / "chunk" / "tta" / "write"；
          返回 False 表示请求取消，引擎抛 CancelledError 并只删除本次写出的成品。
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
                demix_progress_cb=lambda done, total: _cb("chunk", done, total),
            )

        # instrumental = 原混合 - 目标声部（数学无损）。两输入须同域：
        # vocals 只反归一化一次，写盘与相减共用这份原始域结果。
        instruments = prefer_target_instrument(self.config)[:]
        target = "vocals" if "vocals" in [i.lower() for i in instruments] else instruments[0]
        target_key = next(i for i in instruments if i.lower() == target)
        if norm_params is not None:
            waveforms[target_key] = denormalize_audio(waveforms[target_key], norm_params)
        waveforms["instrumental"] = mix_orig - waveforms[target_key]

        written: list[Path] = []
        part_path: Path | None = None
        try:
            for idx, (instr_key, stem_name) in enumerate(
                [(target_key, target), ("instrumental", "instrumental")], start=1
            ):
                est = waveforms[instr_key]
                peak = float(np.abs(est).max())
                codec, subtype, warn = _output_format(fmt, pcm, peak)
                if warn is not None and self.verbose:
                    print(f"  警告: {warn}")

                out_path = _separate_output_path(
                    out_dir, input_path.stem, stem_name, codec,
                )
                # 先写同目录 .part，成功后再换名为成品，避免半截文件顶着最终名。
                part_path = out_path.with_name(out_path.name + ".part")
                sf.write(
                    part_path, est.T, self.sample_rate,
                    subtype=subtype, format=codec.upper(),
                )
                os.replace(part_path, out_path)
                part_path = None
                written.append(out_path)
                if self.verbose:
                    print(f"  写出: {out_path}（峰值 {peak:.3f}）")
                _cb("write", idx, 2)
        except Exception as exc:
            if part_path is not None:
                with suppress(OSError):
                    part_path.unlink(missing_ok=True)
            # 取消只删本次已换名的成品；其它异常保留已完成的文件，只丢掉 .part。
            if isinstance(exc, CancelledError):
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

    批量场景请直接创建 Separator 复用，避免每文件重新加载模型。
    """
    sep = Separator(model_name=model_name, device=device, batch_size=batch_size,
                    num_overlap=num_overlap, verbose=verbose)
    return sep.separate(input_path, out_dir, pcm=pcm, fmt=fmt,
                        bigshifts=bigshifts, tta=tta,
                        progress_callback=progress_callback)


def _model_type_of(model_name: str) -> str:
    return get_model_info(model_name)["model_type"]
