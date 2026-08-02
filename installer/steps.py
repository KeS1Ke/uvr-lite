# coding: utf-8
"""安装执行步骤：每步一个函数，通过 StepContext 交互（可独立测试）。

步骤（installer/runner.py 编排）：
  1. 准备 Python 环境：复用系统 Python ≥3.10 或下载内置绿色 Python（SHA256 校验）
  2. 创建虚拟环境
  3. 安装依赖：torch（CPU/CUDA 分流 + 镜像回退）→ 复制代码快照 → pip install app[ui]
  4. 下载模型（断点续传 + 镜像回退 + SHA256；安装目录/models）
  5. 创建桌面 + 开始菜单快捷方式
  6. 写入 install.json 标记
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from uvr_lite import download as dl
from uvr_lite.download import ensure_model
from uvr_lite.models import DEFAULT_MODEL

from . import copy_app, marker, proc, python_env, shortcuts
from .consts import (
    PIP_INDEX,
    TORCH_CPU_INDEXES,
    TORCH_CUDA_INDEXES,
    TORCH_VERSION,
    torch_wheel_name,
    torch_wheel_urls,
)

# 项目根（开发态代码源；打包态由 src_dir 覆盖）
_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StepContext:
    """步骤上下文：路径、取消/消息/进度回调。percent 为局部 0-100。"""

    install_dir: Path
    upgrade: bool
    src_dir: Optional[Path] = None
    log_file: Optional[Path] = None
    python_exe: Optional[Path] = None
    _cancel: Callable[[], bool] = field(default=lambda: False)
    _message: Callable[[str], None] = field(default=lambda s: None)
    _percent: Callable[[int], None] = field(default=lambda p: None)

    @property
    def app_dir(self) -> Path:
        return self.install_dir / "app"

    @property
    def venv_dir(self) -> Path:
        return self.install_dir / ".venv"

    @property
    def venv_python(self) -> Path:
        return self.venv_dir / "Scripts" / "python.exe"

    @property
    def venv_pip(self) -> Path:
        return self.venv_dir / "Scripts" / "pip.exe"

    @property
    def models_dir(self) -> Path:
        return self.install_dir / "models"

    def cancel(self) -> bool:
        return self._cancel()

    def message(self, text: str) -> None:
        self._message(text)

    def percent(self, value: int) -> None:
        self._percent(max(0, min(100, value)))


# ---------- 辅助 ----------

def _venv_has(pip_exe: Path, pkg: str) -> bool:
    try:
        r = subprocess.run([str(pip_exe), "show", pkg],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except OSError:
        return False


def _pip_install_with_fallback(ctx: StepContext, specs: List[str],
                               indexes: List[str]) -> None:
    """依次尝试各 pip 源；全部失败抛最后一个错误。"""
    last_err: Optional[Exception] = None
    for idx in indexes:
        if ctx.cancel():
            raise InterruptedError("安装已取消")
        cmd = [str(ctx.venv_pip), "install", "--disable-pip-version-check", "--no-input"]
        if idx:
            cmd += ["--index-url", idx]
        cmd += specs
        try:
            proc.run(cmd, log_file=ctx.log_file, cancel=ctx.cancel)
            return
        except RuntimeError as e:
            last_err = e
    raise RuntimeError(f"依赖安装失败（已尝试 {len(indexes)} 个源）：{last_err}")


def _app_version(app_dir: Path) -> str:
    """从 app/pyproject.toml 读版本号（安装的是快照代码）。"""
    pyproject = app_dir / "pyproject.toml"
    if not pyproject.exists():
        return "0.0.0"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else "0.0.0"


def _venv_python_ver(ctx: StepContext) -> tuple:
    """查询 venv Python 主次版本（决定 torch wheel 文件名 cp3X 标签）。"""
    out = subprocess.run(
        [str(ctx.venv_python), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        capture_output=True, text=True, timeout=30)
    ver = python_env.parse_version(out.stdout)
    if ver is None:
        raise RuntimeError(f"无法识别 venv Python 版本: {out.stdout or out.stderr}")
    return ver


# ---------- 步骤 ----------

def step_prepare_python(ctx: StepContext) -> None:
    """优先复用已有 venv / 系统 Python；否则下载绿色 Python。"""
    if ctx.venv_python.exists():
        ctx.python_exe = ctx.venv_python
        ctx.message("虚拟环境已存在，复用其中的 Python")
        ctx.percent(100)
        return
    sys_py = python_env.find_system_python()
    if sys_py is not None:
        ctx.python_exe = sys_py
        ver = python_env.python_version(sys_py)
        ver_str = ".".join(map(str, ver)) if ver else "?"
        ctx.message(f"检测到系统 Python {ver_str}，直接复用（无需下载）")
        ctx.percent(100)
        return
    ctx.message("未检测到系统 Python，正在下载内置 Python（约 50 MB，校验 SHA256）…")

    def cb(done: int, total: int) -> bool:
        ctx.percent(int(done / total * 90) if total else 0)
        return not ctx.cancel()

    ctx.python_exe = python_env.download_green_python(
        ctx.install_dir, progress_callback=cb, cancel=ctx.cancel)
    ctx.message("内置 Python 解压完成")
    ctx.percent(100)


def step_create_venv(ctx: StepContext) -> None:
    if ctx.venv_python.exists():
        ctx.message("虚拟环境已存在，跳过创建")
        ctx.percent(100)
        return
    ctx.message("正在创建虚拟环境…")
    assert ctx.python_exe is not None, "python_exe 未确定"
    python_env.ensure_venv(ctx.python_exe, ctx.venv_dir,
                           log_file=ctx.log_file, cancel=ctx.cancel)
    ctx.message("虚拟环境创建完成")
    ctx.percent(100)


def step_install_deps(ctx: StepContext) -> None:
    # torch：按硬件分流；wheel 用自研下载器先下好（断点续传+超时+多源回退，
    # pip 大文件下载遇服务器断流会无限卡死），再 pip install 本地 wheel（依赖走清华）。
    # 已装则跳过（覆盖升级/续装）。
    if not _venv_has(ctx.venv_pip, "torch"):
        gpu = shutil.which("nvidia-smi") is not None
        py_ver = _venv_python_ver(ctx)
        wheel_name = torch_wheel_name(py_ver, gpu)
        ctx.message(
            "正在下载 PyTorch（" +
            ("检测到 NVIDIA 显卡，CUDA 版" if gpu else "CPU 版") +
            f"，约 {3300 if gpu else 220} MB，可断点续传）…")

        def cb(done: int, total: int) -> bool:
            ctx.percent(int(done / total * 70) if total else 0)
            return not ctx.cancel()

        cache_dir = ctx.install_dir / "cache"
        wheel_path = cache_dir / wheel_name
        dl._download(torch_wheel_urls(py_ver, gpu), wheel_path, cb)
        ctx.message("PyTorch 下载完成，正在安装…")
        _pip_install_with_fallback(ctx, [str(wheel_path)], [PIP_INDEX, ""])
    else:
        ctx.message("PyTorch 已安装，跳过")
    ctx.percent(40)

    # 复制代码快照 → pip install app[ui]（清华源，失败回退官方）
    if ctx.cancel():
        raise InterruptedError("安装已取消")
    src = ctx.src_dir or _REPO_ROOT
    ctx.message("正在复制程序文件…")
    copy_app.copy_app_source(src, ctx.app_dir)
    ctx.message("正在安装 uvr-lite 与依赖（界面库、音频库等）…")
    _pip_install_with_fallback(ctx, [f"{ctx.app_dir}[ui]"], [PIP_INDEX, ""])
    ctx.message("依赖安装完成")
    ctx.percent(100)


def step_download_model(ctx: StepContext) -> None:
    os.environ["UVR_MODEL_DIR"] = str(ctx.models_dir)
    ctx.message("正在下载模型权重（约 640 MB，SHA256 校验，可断点续传）…")

    def cb(done: int, total: int) -> bool:
        ctx.percent(int(done / total * 100) if total else 0)
        return not ctx.cancel()

    ensure_model(DEFAULT_MODEL, progress_callback=cb)
    ctx.message("模型就绪")
    ctx.percent(100)


def step_create_shortcuts(ctx: StepContext) -> None:
    ctx.message("正在创建桌面与开始菜单快捷方式（♪）…")
    shortcuts.create_shortcuts(ctx.install_dir)
    ctx.message("快捷方式创建完成")
    ctx.percent(100)


def step_write_marker(ctx: StepContext) -> None:
    ctx.message("正在完成安装…")
    marker.write_marker(
        ctx.install_dir,
        app_version=_app_version(ctx.app_dir),
        torch=TORCH_VERSION,
        python=str(ctx.python_exe) if ctx.python_exe else "",
        upgrade=ctx.upgrade,
    )
    ctx.message("安装完成")
    ctx.percent(100)
