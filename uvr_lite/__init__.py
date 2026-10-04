"""uvr-lite：轻量级人声/伴奏分离与合成工具。

推理引擎裁剪自 ZFTurbo Music-Source-Separation-Training（MIT），
模型（BS-RoFormer / Mel-Band RoFormer）与 Ultimate Vocal Remover 同源。

torch 二进制切换（安装目录内 CPU / CUDA 两套 torch，独立目录）：
  安装场景下 {app}/ 下存在 torch_cpu/，勾选或补装后还有 torch_cuda/。
  本模块在导入时按用户选择（torch.ini 或环境变量 UVR_TORCH）把对应目录
  插入 sys.path，使后续 `import torch` 加载正确版本。选用 CUDA 时再把
  torch_cpu 放在它后面：CPU 安装把 sympy、networkx、filelock、fsspec、
  jinja2、mpmath 等纯 Python 依赖装进了 torch_cpu/，CUDA 目录只是 wheel
  解压，不含这些包；缺了它们，`torch.utils.checkpoint` 会在 import sympy
  处失败。torch 包本身仍由更靠前的目录命中。开发场景（无这些目录）不干预，
  使用环境中已安装的 torch。
"""

import os
import sys
from pathlib import Path

__version__ = "0.1.9"


def _base_dir() -> Path:
    """定位根目录（models/、torch_cpu/、torch.ini 所在层）。

    - 开发场景：{repo}/uvr_lite/（uvr_lite 与 msst 平级于 {repo}）
    - 安装场景：{inst}/app/uvr_lite/（根为 {inst}；快照的 app/ 层结构
      与 dev 仓库根几乎相同，故以"父层含 models/ + torch_cpu/"区分）
    """
    f = Path(__file__).resolve()
    for parent in f.parents:
        if (parent / "uvr_lite").is_dir() and (parent / "msst").is_dir():
            if (parent.parent / "models").is_dir() and (parent.parent / "torch_cpu").is_dir():
                return parent.parent  # 安装 {inst}
            return parent  # dev {repo}
    raise RuntimeError(f"无法定位 uvr-lite 根目录（{f}）")


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


def _dependency_dirs(selected: Path) -> list[Path]:
    """CUDA 引擎还要带上 torch_cpu，那里有 wheel 没带上的纯 Python 依赖。"""
    if selected.name != "torch_cuda":
        return []
    cpu = selected.parent / "torch_cpu"
    if cpu.is_dir():
        return [cpu]
    return []


# 本模块插进 sys.path 的条目（切换引擎时整组撤下，避免 CUDA/CPU 叠在一起）。
_managed_paths: list[str] = []


def _apply_torch_paths(selected: Path | None) -> None:
    """把选中的 torch 目录放到 sys.path 最前，CUDA 时紧跟 torch_cpu。"""
    global _managed_paths
    for entry in _managed_paths:
        while entry in sys.path:
            sys.path.remove(entry)
    _managed_paths = []
    if selected is None:
        return
    ordered = [selected, *_dependency_dirs(selected)]
    for path in reversed(ordered):
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    _managed_paths = [str(path) for path in ordered]


_torch_dir_ = _torch_dir()
_apply_torch_paths(_torch_dir_)


def set_torch_mode(mode: str) -> None:
    """运行时切换 torch 二进制（CLI 用：解析 --device 后、import engine 前调用）。"""
    global _torch_dir_
    os.environ["UVR_TORCH"] = mode
    new = _torch_dir()
    if new is None:
        return
    _apply_torch_paths(new)
    _torch_dir_ = new
