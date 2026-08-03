# coding: utf-8
"""把 uvr-lite 打成半在线标准安装包（Inno Setup 7）。

用法: python scripts/build_installer.py [--out dist] [--iscc <ISCC.exe 路径>]
产物: dist/uvr-lite-setup_v0.1.0.exe（约 283MB）
  - 单包制：只内置 CPU torch（离线即装即用）；CUDA 引擎走半在线——
    安装时勾选附加任务联网下载（install.iss 内 DownloadTemporaryFile + SHA
    校验 + extractarchive 解压），或安装后应用内「推理引擎」区 / CLI
    `uvr-lite install-cuda` 补装（复用多段并发 + 断点续传 + 镜像回退下载器）
  - 含：代码快照 + 内置 Python+依赖 + CPU torch + fp16 瘦身模型权重
  - Inno 6+ 支持 >2GB 安装包（NSIS 有 ~2GB 硬限制，历史 full 变体无法打包）
发布约定: GitHub Releases 资产名固定 uvr-lite-setup.exe（README 的
  releases/latest/download 稳定链接依赖精确资产名），上传时重命名即可。

打包机首次打包需下载约 1GB（绿色 Python 50MB + CPU torch 0.7GB + 模型
  320MB + PySide6-Essentials 等依赖），国内镜像优先；下载/安装产物跨次
  构建复用（python/ torch_cpu/ models/ 存在即跳过，增量更新）。

前置: 本机安装 Inno Setup 7（默认探测 D:\\Inno setup / Program Files，可 --iscc 指定）
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from uvr_lite import __version__  # noqa: E402 —— 版本单一来源（与 pyproject.toml 同步维护）


def setup_name() -> str:
    return f"uvr-lite-setup_v{__version__}"

# ---------- 固定版本与源（原 installer/consts.py） ----------

# python-build-standalone（install_only，解压即用，约 50MB）
GREEN_PY_VERSION = "3.12.13"
GREEN_PY_TAG = "20260728"
GREEN_PY_FILENAME = (
    f"cpython-{GREEN_PY_VERSION}+{GREEN_PY_TAG}-"
    "x86_64-pc-windows-msvc-install_only.tar.gz"
)
GREEN_PY_SHA256 = "8a0e1ded37e11f4c72b9671bf134bb478b1b2d55efe53a3d6e589b166f1bf2e1"
GREEN_PY_URLS = [
    # 按实测速度排序：npmmirror(512KB/0.6s) → 南大(0.7s) → GitHub 官方(0.2MB/s) → ghproxy 兜底
    "https://registry.npmmirror.com/-/binary/python-build-standalone/"
    f"{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
    "https://mirror.nju.edu.cn/github-release/astral-sh/"
    f"python-build-standalone/{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
    "https://github.com/astral-sh/python-build-standalone/releases/download/"
    f"{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
    "https://mirror.ghproxy.com/https://github.com/astral-sh/"
    f"python-build-standalone/releases/download/{GREEN_PY_TAG}/{GREEN_PY_FILENAME}",
]

# torch：只内置 CPU 版（单包制）；CUDA 版由安装器/应用按需下载，源与 SHA
# 统一维护在 uvr_lite/download.py（TORCH_CUDA_URLS / TORCH_CUDA_SHA256），
# install.iss 内硬编码同款（_check_cuda_sha_consistency 防漂移）。
# 镜像按实测速度排序（2026-08-04 流式测速）：
#   SJTU 15-17MB/s > 官方 13-14MB/s > 阿里云 3-4MB/s（需浏览器 UA，403 已修）。
# 南大 mirror.nju.edu.cn 无 pytorch-wheels（404），未收录。
TORCH_VERSION = "2.7.1"
TORCH_CPU_INDEXES = [
    "https://mirror.sjtu.edu.cn/pytorch-wheels/cpu",
    "https://download.pytorch.org/whl/cpu",
    "https://mirrors.aliyun.com/pytorch-wheels/cpu",
]

# 常规依赖 pip 源（清华 PyPI）
PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

# 应用依赖（不含 torch：torch 单独 --target 安装；依赖装入绿色 Python 本体）
# 不用 librosa（连带 scipy/numba/llvmlite 等 ~320MB）：解码用 soundfile+soxr，
# mel 滤波器已 vendored（msst/models/bs_roformer/mel_filters.py）
DEPENDENCIES = [
    "numpy>=1.24", "soundfile>=0.12", "soxr>=0.3", "audioread>=3.0",
    "pyyaml>=6.0", "ml-collections>=0.1.1", "einops>=0.7", "beartype>=0.16",
    "packaging>=23", "tqdm>=4.60",
    "PySide6-Essentials>=6.6",
]


# ---------- 工具 ----------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download_to(urls: List[str], dest: Path,
                 progress_cb: Optional[Callable[[int, int], bool]] = None) -> None:
    """多源回退下载（单连接 + Range 续传），全部失败才报错。"""
    from uvr_lite import download as dl

    dl._download(urls, dest, progress_cb)


def _run(cmd: List[str], desc: str) -> None:
    print(f"  → {desc}")
    subprocess.run(cmd, check=True)


# ---------- bundle 准备 ----------

def _download_green_python(bundle_dir: Path) -> None:
    """下载并解压绿色 Python 到 bundle/python/（SHA256 校验）。"""
    dest = bundle_dir / "python"
    archive = bundle_dir / GREEN_PY_FILENAME
    print(f"[1/5] 下载内置 Python（约 50MB，SHA256 校验）…")
    _download_to(GREEN_PY_URLS, archive)
    actual = sha256_of(archive)
    if actual != GREEN_PY_SHA256:
        raise SystemExit(f"绿色 Python 校验失败: 期望 {GREEN_PY_SHA256[:16]}…，实际 {actual[:16]}…")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            name = member.name
            if name.startswith("python/"):
                name = name[len("python/"):]
            elif name == "python":
                continue
            if not name:
                continue
            target = dest / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    continue
                with open(target, "wb") as f:
                    while chunk := src.read(1 << 20):
                        f.write(chunk)
    archive.unlink(missing_ok=True)


def _pip(python_exe: Path, specs: List[str], index: str = PIP_INDEX,
         target: Optional[Path] = None, no_deps: bool = False) -> None:
    """pip 安装（默认清华源；失败自动回退官方源）。"""
    for idx in [index, ""]:
        cmd = [str(python_exe), "-m", "pip", "install", "--disable-pip-version-check",
               "--no-input"]
        if idx:
            cmd += ["--index-url", idx]
        if target is not None:
            cmd += ["--target", str(target)]
        if no_deps:
            cmd += ["--no-deps"]
        cmd += specs
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            last = e
    raise SystemExit(f"pip 安装失败: {specs}\n{last.stderr[-500:] if last else ''}")


def _prune_torch(dest: Path) -> None:
    """裁剪 torch 安装目录：删编译期文件（.lib / include / bin），运行时不需要。

    实测收益：torch_cuda 5.6G→4.7G（-900M）、torch_cpu 1.4G→461M（-940M），
    import + 真实分离（CPU/GPU）验证无损。bin/ 保留 torch_shm_manager.exe
    （torch 多进程共享内存需要；本应用单进程推理实际用不到，保守保留）。
    """
    t = dest / "torch"
    for f in (t / "lib").glob("*.lib"):
        f.unlink()
    shutil.rmtree(t / "include", ignore_errors=True)
    bin_dir = t / "bin"
    if bin_dir.is_dir():
        for f in bin_dir.iterdir():
            if f.name == "torch_shm_manager.exe":
                continue
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f, ignore_errors=True)
    print(f"    {dest.name} 裁剪完成（.lib/include/bin 已删）")


def _install_torch(bundle_dir: Path, tag: str, indexes: List[str]) -> Path:
    """把 torch 装到独立目录（--no-deps：依赖已在 python/ 内）。

    wheel 用自研多段下载器先下好（pip 大文件下载遇服务器断流会无限卡死，
    见 a7f9dff），再 pip 安装本地 wheel；目录已存在视为已就绪（跨次复用）。
    单包制只装 torch_cpu；CUDA 引擎由安装器/应用按需下载（install.iss 与
    uvr_lite.download.install_cuda_torch）。
    """
    dest = bundle_dir / f"torch_{tag}"
    if (dest / "torch" / "__init__.py").exists():
        _prune_torch(dest)  # 复用路径同样裁剪（幂等）
        print(f"[4/5] torch_{tag} 已就绪，复用")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # 绿色 Python 固定 3.12 → cp312 wheel（见 GREEN_PY_VERSION）；
    # 构建标签：CPU 为 +cpu（CUDA 为 +cu128，见 download.py）
    cp_tag = f"cp{GREEN_PY_VERSION.split('.')[0]}{GREEN_PY_VERSION.split('.')[1]}"
    wheel_tag = "cpu" if tag == "cpu" else "cu128"
    wheel_name = f"torch-{TORCH_VERSION}+{wheel_tag}-{cp_tag}-{cp_tag}-win_amd64.whl"
    wheel_path = bundle_dir / wheel_name
    print(f"[4/5] 下载 torch_{tag}（{wheel_name}，"
          f"约 {3300 if tag == 'cuda' else 700}MB，多段并行 + 镜像回退）…")
    from uvr_lite import download as dl

    dl._download([f"{idx}/{wheel_name}" for idx in indexes], wheel_path)
    print(f"[4/5] 安装 torch_{tag} 到 {dest}…")
    # 依赖 nvidia-* 用默认清华源拉取（index="" 会走 pypi.org 官方源卡死）
    _pip(bundle_dir / "python" / "python.exe", [str(wheel_path)],
         target=dest)
    wheel_path.unlink(missing_ok=True)
    _prune_torch(dest)
    return dest


# ---------- Qt 裁剪 ----------

def _prune_pyside6(site_packages: Path) -> None:
    """裁剪 PySide6-Essentials 中 GUI 用不到的部分（省 100-200MB）。

    只保留 QtCore/QtGui/QtWidgets（ui/ 的全部 import 面）：
    删 translations/（界面文案硬编码中文，无需 Qt 翻译）、qml/（无 QML 界面）、
    *.pyi 类型存根（仅 IDE 提示用）、以及其余 Qt 模块的 .pyd/.dll。
    裁剪安全性由随后的 _smoke_ui 冒烟测试兜底。
    """
    ps6 = site_packages / "PySide6"
    if not ps6.exists():
        return
    for sub in ("translations", "qml", "examples", "resources"):
        p = ps6 / sub
        if p.exists():
            print(f"    - 删除 {ps6.name}/{sub}/")
            shutil.rmtree(p)
    for f in list(ps6.glob("*.pyi")):
        f.unlink()
    keep = {"QtCore", "QtGui", "QtWidgets"}
    for f in list(ps6.glob("Qt*.pyd")):
        if f.stem not in keep:
            print(f"    - 删除 {ps6.name}/{f.name}")
            f.unlink()
    # 注意：Qt6*.dll 是共享依赖库（QtWidgets.pyd 可能依赖任意 Qt 模块的
    # DLL，实测删除后 DLL load failed），必须全部保留


def _smoke_ui(python_exe: Path) -> None:
    """UI 冒烟测试：裁剪后 Qt 模块仍可加载并完成一次事件循环。"""
    code = (
        "from PySide6.QtWidgets import QApplication;"
        "from PySide6.QtCore import QTimer;"
        "app = QApplication([]);"
        "QTimer.singleShot(0, app.quit);"
        "app.exec();"
        "print('UI SMOKE OK')"
    )
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    subprocess.run([str(python_exe), "-c", code], check=True, env=env)


def _check_cuda_sha_consistency() -> None:
    """install.iss 与 uvr_lite/download.py 的 CUDA wheel SHA 应一致（防漂移）。"""
    iss = (ROOT / "installer" / "install.iss").read_text(encoding="utf-8")
    dl_src = (ROOT / "uvr_lite" / "download.py").read_text(encoding="utf-8")
    m_iss = re.search(r"CUDA_SHA = '([0-9a-f]{64})'", iss)
    m_dl = re.search(r'TORCH_CUDA_SHA256 = "([0-9a-f]{64})"', dl_src)
    if not m_iss or not m_dl or m_iss.group(1) != m_dl.group(1):
        raise SystemExit("CUDA wheel SHA 不一致：install.iss 与 download.py 需同步修改")
    print(f"  CUDA wheel SHA 一致: {m_dl.group(1)[:16]}…")


def prepare_bundle(bundle_dir: Path) -> None:
    """组装打包源：app 快照 + python(绿色 Python+依赖) + torch_cpu + models。

    python/torch_cpu/models 下载安装产物跨次构建复用（大文件避免重下）。
    单包制：不内置 CUDA torch（省 3.3GB 包体，安装时按需下载）。
    """
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. 代码快照（每次按最新代码重建）
    for sub in ("app",):
        p = bundle_dir / sub
        if p.exists():
            shutil.rmtree(p)
    from installer.copy_app import copy_app_source

    count = copy_app_source(ROOT, bundle_dir / "app")
    print(f"[2/5] 代码快照: {bundle_dir / 'app'}（{count} 个文件）")

    # 2. 绿色 Python（存在则复用）
    python_exe = bundle_dir / "python" / "python.exe"
    if not python_exe.exists():
        _download_green_python(bundle_dir)
    print(f"[2/5] 内置 Python: {python_exe}")

    # 3. 依赖装入绿色 Python（存在关键包则复用）
    # 判定用 Qt6Widgets.dll（而非 PySide6 目录）：依赖被裁剪/损坏时强制重装
    need_deps = not (bundle_dir / "python" / "Lib" / "site-packages"
                     / "PySide6" / "Qt6Widgets.dll").exists()
    if need_deps:
        print("[3/5] 安装应用依赖（soundfile/PySide6-Essentials 等）…")
        _pip(python_exe, DEPENDENCIES)
        # rotary-embedding-torch 的依赖链会把 CPU torch（约 524MB）装进
        # site-packages，与 torch_cpu/ 目录重复 → 单独 --no-deps 安装，
        # torch 由 torch_cpu/torch_cuda 提供（应用启动时切换）
        _pip(python_exe, ["rotary-embedding-torch>=0.4"], no_deps=True)
        _prune_pyside6(bundle_dir / "python" / "Lib" / "site-packages")
        _smoke_ui(python_exe)
    else:
        print("[3/5] 应用依赖已就绪，复用")

    # 4. CPU torch（独立目录，应用内切换）
    _install_torch(bundle_dir, "cpu", TORCH_CPU_INDEXES)

    # 5. 模型权重（SHA256 校验；已就绪则复用）
    os.environ["UVR_MODEL_DIR"] = str(bundle_dir / "models")
    from uvr_lite.download import ensure_model
    from uvr_lite.models import DEFAULT_MODEL

    ensure_model(DEFAULT_MODEL)
    print(f"[4/5] 模型: {bundle_dir / 'models' / f'{DEFAULT_MODEL}.ckpt'}")


# ---------- 编译 ----------

def default_iscc() -> Path:
    """探测 Inno Setup 7 编译器路径（环境变量 ISCC 优先）。"""
    env_iscc = os.environ.get("ISCC", "").strip()
    candidates: list = []
    if env_iscc:
        candidates.append(Path(env_iscc))
    candidates += [
        Path("D:/Inno setup/Inno Setup 7/ISCC.exe"),
        Path("C:/Program Files (x86)/Inno Setup 7/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 7/ISCC.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit("未找到 ISCC.exe：请安装 Inno Setup 7，或用 --iscc 指定路径")


def build(iscc: Path, out_dir: Path) -> Path:
    # 输出路径与 bundle 路径均在 install.iss 内用相对路径配置
    # （相对脚本文件所在目录），避免 ISCC 对含空格路径参数的解析问题；
    # 只传无空格的版本号。半在线单包约 283MB（Inno 6+ 支持 >2GB 安装包）。
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_cuda_sha_consistency()
    cmd = [
        str(iscc),
        f"/DMyAppVersion={__version__}",
        str(ROOT / "installer" / "install.iss"),
    ]
    print(f"[5/5] ISCC 编译中（压缩需较长时间）…")
    subprocess.run(cmd, check=True)
    exe = out_dir / f"{setup_name()}.exe"
    if not exe.exists():
        raise SystemExit(f"打包失败：未找到产物 {exe}")
    print(f"完成: {exe}（{exe.stat().st_size / 1e6:.0f} MB）")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 uvr-lite 为半在线标准安装包（Inno Setup 7）")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    ap.add_argument("--iscc", default="", help="ISCC.exe 路径（默认自动探测）")
    ap.add_argument("--no-bundle", action="store_true",
                    help="跳过 bundle 准备（仅重新编译 .iss）")
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    bundle_dir = out_dir / "_bundle"
    iscc = Path(args.iscc).resolve() if args.iscc else default_iscc()
    print(f"ISCC: {iscc}")
    if not args.no_bundle:
        prepare_bundle(bundle_dir)
    build(iscc, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
