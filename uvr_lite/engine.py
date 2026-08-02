# coding: utf-8
"""分离引擎：封装 msst/ 推理子集（vendored，来自 ZFTurbo MSST），提供单文件人声/伴奏分离。

流程：librosa 读音频 -> 归一化（可选）-> BigShifts 圆形时移平均 -> BS-RoFormer 前向
-> instrumental = mix - vocals（数学无损）-> 写 FLAC/WAV。
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import librosa
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

from .download import config_path, ensure_model  # noqa: E402
from .models import DEFAULT_MODEL, get_model_info  # noqa: E402


def pick_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def load_model(model_name: str, ckpt_path: Path, device: str):
    """加载模型与配置（结构同 MSST inference.py 的 proc_folder 前半段）。"""
    info = get_model_info(model_name)
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(info["model_type"], str(config_path(model_name)))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    return model, config


def separate_file(
    input_path: str,
    out_dir: str,
    model_name: str = DEFAULT_MODEL,
    pcm: str = "PCM_24",
    device: str = "auto",
    fmt: str = "auto",  # auto | flac | wav
    bigshifts: int = 1,
    tta: bool = False,
    batch_size: Optional[int] = None,
    verbose: bool = True,
) -> List[Path]:
    """分离单个音频文件，输出 {stem}-vocals 与 {stem}-instrumental 两个文件。"""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(device)
    ckpt = ensure_model(model_name)
    model, config = load_model(model_name, ckpt, device)

    # 低显存 GPU 可调小批大小防 OOM（默认取模型配置 yaml）
    if batch_size is not None and batch_size >= 1:
        config.inference["batch_size"] = batch_size

    sample_rate: int = getattr(config.audio, "sample_rate", 44100)
    if verbose:
        print(f"分离: {input_path.name} | 模型: {model_name} | 设备: {device} | 采样率: {sample_rate}")

    mix, sr = librosa.load(input_path, sr=sample_rate, mono=False)
    if len(mix.shape) == 1:
        mix = np.expand_dims(mix, axis=0)
        if getattr(config.audio, "num_channels", 1) == 2:
            mix = np.concatenate([mix, mix], axis=0)

    mix_orig = mix.copy()

    norm_params = None
    if getattr(config.inference, "normalize", False):
        mix, norm_params = normalize_audio(mix)

    model_type = _model_type_of(model_name)
    waveforms = bigshifts_wrapper(
        config, model, mix, device,
        model_type=model_type, pbar=verbose, bigshifts=bigshifts,
    )
    if tta:
        waveforms = apply_tta(
            config, model, mix, waveforms, device,
            model_type, bigshifts=bigshifts, pbar=verbose,
        )

    # instrumental = 原混合 - 目标声部（数学无损）
    instruments = prefer_target_instrument(config)[:]
    target = "vocals" if "vocals" in [i.lower() for i in instruments] else instruments[0]
    target_key = next(i for i in instruments if i.lower() == target)
    waveforms["instrumental"] = mix_orig - waveforms[target_key]

    written = []
    for instr_key, stem_name in [(target_key, target), ("instrumental", "instrumental")]:
        est = waveforms[instr_key]
        if norm_params is not None:
            est = denormalize_audio(est, norm_params)

        peak = float(np.abs(est).max())
        if fmt == "flac" or (fmt == "auto" and peak <= 1.0):
            codec = "flac"
        else:
            codec = "wav"

        out_path = out_dir / f"{input_path.stem}-{stem_name}.{codec}"
        sf.write(out_path, est.T, sample_rate, subtype=pcm)
        if verbose:
            print(f"  写出: {out_path}（峰值 {peak:.3f}）")
        written.append(out_path)
    return written


def _model_type_of(model_name: str) -> str:
    return get_model_info(model_name)["model_type"]
