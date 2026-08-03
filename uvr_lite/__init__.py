# coding: utf-8
"""uvr-lite：轻量级人声/伴奏分离工具。

推理引擎裁剪自 ZFTurbo Music-Source-Separation-Training（MIT），
模型（BS-RoFormer / Mel-Band RoFormer）与 Ultimate Vocal Remover 同源。

torch 二进制切换（全量安装包内含 CPU/CUDA 两套 torch，独立目录）：
  安装场景下 {app}/ 下存在 torch_cpu/ 与 torch_cuda/，本模块在导入时
  按用户选择（torch.ini 或环境变量 UVR_TORCH）把对应目录插入 sys.path，
  使后续 `import torch` 加载正确版本。开发场景（无这些目录）不干预，
  使用环境中已安装的 torch。
"""

import os
import sys
from pathlib import Path

__version__ = "0.1.0"


def _base_dir() -> Path:
    """安装场景 {app}（uvr_lite 位于 {app}/app/uvr_lite），开发场景仓库根。"""
    return Path(__file__).resolve().parents[2]


def _torch_dir() -> Path | None:
    """解析应使用的 torch 目录；无则返回 None（用环境已装 torch）。

    优先级：环境变量 UVR_TORCH（CLI 临时指定）> torch.ini（UI 持久选择）。
    mode 取值 cpu / cuda / auto（auto 优先 cuda，无则 cpu）。
    """
    base = _base_dir()
    mode = os.environ.get("UVR_TORCH", "")
    if not mode:
        ini = base / "torch.ini"
        if ini.exists():
            try:
                for line in ini.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("use="):
                        mode = line[4:].strip()
            except OSError:
                pass
    if mode == "cuda":
        cand = base / "torch_cuda"
    elif mode == "cpu":
        cand = base / "torch_cpu"
    else:  # auto / 未配置：有 cuda 目录则优先（无显卡时 torch 自动回退 CPU）
        cand = base / "torch_cuda" if (base / "torch_cuda").exists() else base / "torch_cpu"
    if cand.exists():
        return cand
    return None


_torch_dir_ = _torch_dir()
if _torch_dir_ is not None:
    _path = str(_torch_dir_)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def set_torch_mode(mode: str) -> None:
    """运行时切换 torch 二进制（CLI 用：解析 --device 后、import engine 前调用）。"""
    global _torch_dir_
    os.environ["UVR_TORCH"] = mode
    new = _torch_dir()
    if new is None:
        return
    new_path = str(new)
    if _torch_dir_ is not None:
        old_path = str(_torch_dir_)
        while old_path in sys.path:
            sys.path.remove(old_path)
    if new_path not in sys.path:
        sys.path.insert(0, new_path)
    _torch_dir_ = new
