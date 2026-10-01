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
from collections.abc import Callable
from pathlib import Path

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

# torch：只内置 CPU 版（单包制）；CUDA 版由安装器/应用按需下载，源 / SHA /
# 字节数统一维护在 uvr_lite/download.py（TORCH_CUDA_URLS / _SHA256 / _SIZE），
# install.iss 内硬编码同款（build 前 _check_packaging_consistency 逐项核对防漂移）。
# 镜像排序：CPU wheel 索引沿用 2026-08-04 流式测速基线（SJTU 15-17MB/s >
# 官方 13-14MB/s > 阿里云 3-4MB/s，需浏览器 UA，403 已修）；CUDA wheel 于
# 2026-09-29 复测后反转为「官方 > SJTU > 阿里云」（见
# uvr_lite/download.py TORCH_CUDA_URLS 注释），install.iss 的 CUDA_URL 随其
# 首个镜像同步，_check_cuda_assets 会硬失败拦漂移。
# 南大 mirror.nju.edu.cn 无 pytorch-wheels（404），未收录。
TORCH_VERSION = "2.7.1"
# CPU wheel 内容哈希：构建期校验（此前仅 CUDA 有 SHA，CPU 无校验直接 pip
# 安装——镜像被投毒时无防线）。2026-08-27 SJTU 与官方下载字节级一致
# （215985616 字节，双源分别校验）；阿里云同大小（Content-Length 一致）。
TORCH_CPU_SHA256 = "0bc887068772233f532b51a3e8c8cfc682ae62bef74bf4e0c53526c8b9e4138f"
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
    "packaging>=23", "tqdm>=4.60", "safetensors>=0.4",
    "PySide6-Essentials>=6.6",
]

# 不随 DEPENDENCIES 批量装的两项：torch 由 _install_torch 装到独立目录（应用内
# 切换 CPU/CUDA）；rotary-embedding-torch 的依赖链会把 CPU torch（约 524MB）
# 拖进 site-packages，故在 prepare_bundle 里单独 --no-deps 安装
DEPS_INSTALLED_SEPARATELY = {"torch", "rotary-embedding-torch"}


# ---------- 工具 ----------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _download_to(urls: list[str], dest: Path,
                 progress_cb: Callable[[int, int], bool] | None = None) -> None:
    """多源回退下载（单连接 + Range 续传），全部失败才报错。"""
    from uvr_lite import download as dl

    dl._download(urls, dest, progress_cb)


# ---------- bundle 准备 ----------

def _download_green_python(bundle_dir: Path) -> None:
    """下载并解压绿色 Python 到 bundle/python/（SHA256 校验）。"""
    dest = bundle_dir / "python"
    archive = bundle_dir / GREEN_PY_FILENAME
    print("[1/5] 下载内置 Python（约 50MB，SHA256 校验）…")
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


def _pip(python_exe: Path, specs: list[str], index: str = PIP_INDEX,
         target: Path | None = None, no_deps: bool = False) -> None:
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

    清单与实现只在 uvr_lite.download._prune_torch_install 维护一份——打包链与
    应用内 install_cuda_torch 走同一段代码，避免两处漂移（实测收益见该函数）；
    install.iss 的 Pascal 版无法复用实现，由 _check_prune_list 文本核对防漂移。
    """
    from uvr_lite import download as dl

    dl._prune_torch_install(dest)
    print(f"    {dest.name} 裁剪完成（.lib/include/bin 已删）")


def _install_torch(bundle_dir: Path, indexes: list[str]) -> Path:
    """把 CPU torch 连同其纯 Python 依赖装到独立目录 torch_cpu/。

    不要加 --no-deps。依赖并不在绿色 Python 里：sympy、networkx、filelock、
    fsspec、jinja2、mpmath、markupsafe 会落到 torch_cpu/。CUDA 引擎只解压
    wheel，运行时靠 uvr_lite 在选用 torch_cuda 时把 torch_cpu 留在 sys.path
    后面才能 import 到这些包。改成 --no-deps 又不另装依赖，分离会在
    import sympy 处失败。

    wheel 用自研多段下载器先下好（pip 大文件下载遇服务器断流会无限卡死，
    见 a7f9dff），再 pip 安装本地 wheel；目录已存在视为已就绪（跨次复用）。
    半在线单包制只有 CPU 一条路径：CUDA 引擎由安装器/应用按需下载（install.iss
    与 uvr_lite.download.install_cuda_torch）。签名里不留 tag 参数——曾有个
    「tag=cuda」死分支，且 SHA 校验被 `if tag == "cpu":` 包住，等于传错 tag 就
    静默跳过校验；现在没有可传错的 tag，校验也无条件执行。
    """
    dest = bundle_dir / "torch_cpu"
    if (dest / "torch" / "__init__.py").exists():
        _prune_torch(dest)  # 复用路径同样裁剪（幂等）
        print("[4/5] torch_cpu 已就绪，复用")
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # 绿色 Python 固定 3.12 → cp312 wheel（见 GREEN_PY_VERSION）
    cp_tag = f"cp{GREEN_PY_VERSION.split('.')[0]}{GREEN_PY_VERSION.split('.')[1]}"
    wheel_name = f"torch-{TORCH_VERSION}+cpu-{cp_tag}-{cp_tag}-win_amd64.whl"
    wheel_path = bundle_dir / wheel_name
    print(f"[4/5] 下载 torch_cpu（{wheel_name}，约 700MB，多段并行 + 镜像回退）…")
    from uvr_lite import download as dl

    # URL 中 "+" 必须 %2B 编码：官方源（S3/CloudFront）对字面 + 返回 403
    # （此前官方回退源一直是坏的）；SJTU/阿里云对两种形式均可（已验证）。
    dl._download([f"{idx}/{wheel_name.replace('+', '%2B')}" for idx in indexes],
                 wheel_path)
    actual = sha256_of(wheel_path)
    if actual != TORCH_CPU_SHA256:
        wheel_path.unlink(missing_ok=True)
        raise SystemExit(
            f"CPU torch wheel 校验失败: 期望 {TORCH_CPU_SHA256[:16]}…，"
            f"实际 {actual[:16]}…。下载源可能已变更，请核实镜像内容。")
    print("    CPU wheel SHA256 校验通过")
    print(f"[4/5] 安装 torch_cpu 到 {dest}…")
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


# ---------- 打包链一致性校验 ----------
# install.iss 是 Pascal（Inno 脚本环境），无法 import Python 侧的常量与实现，
# 只能硬编码同款。漂移的代价：勾选 CUDA 引擎时下载/解压的可能是另一个文件
# （安装期 SHA 校验才失败，3.3GB 白下）、裁剪清单漏项白占约 900MB、装机依赖
# 漏包则应用启动即失败。故打包前文本核对，把漂移变成硬错误。

def _grab(iss: str, pattern: str, what: str) -> str:
    """从 install.iss 取一处硬编码值；取不到即报错（宁可停下也不跳过校验）。"""
    m = re.search(pattern, iss)
    if not m:
        raise SystemExit(f"install.iss 中找不到 {what}，打包链一致性校验无法完成")
    return m.group(1)


def _check_cuda_assets() -> None:
    """install.iss 与 download.py 的 CUDA 资产（文件名/SHA/字节数/镜像）必须一致。"""
    from uvr_lite import download as dl

    iss = (ROOT / "installer" / "install.iss").read_text(encoding="utf-8")
    # wheel 本质是 zip，install.iss 里按 .zip 处理（extractarchive 靠扩展名识别）
    wheel_zip = dl.TORCH_CUDA_WHEEL.removesuffix(".whl") + ".zip"
    names = {
        "CUDA_WHEEL 常量": _grab(iss, r"CUDA_WHEEL = '([^']+)'", "CUDA_WHEEL"),
        "[Files] 解压源": _grab(iss, r'Source: "\{tmp\}\\([^"]+)"', "[Files] Source"),
    }
    for what, name in names.items():
        if name != wheel_zip:
            raise SystemExit(f"{what} 与 download.py TORCH_CUDA_WHEEL 不一致: "
                             f"{name} != {wheel_zip}")
    size = int(_grab(iss, r"ExternalSize: (\d+)", "ExternalSize"))
    if size != dl.TORCH_CUDA_SIZE:
        raise SystemExit(f"ExternalSize 与 download.py TORCH_CUDA_SIZE 不一致: "
                         f"{size} != {dl.TORCH_CUDA_SIZE}")
    sha = _grab(iss, r"CUDA_SHA = '([0-9a-f]{64})'", "CUDA_SHA")
    if sha != dl.TORCH_CUDA_SHA256:
        raise SystemExit("CUDA wheel SHA 不一致：install.iss 与 download.py 需同步修改")
    url = _grab(iss, r"CUDA_URL = '([^']+)'", "CUDA_URL")
    if url != dl.TORCH_CUDA_URLS[0]:
        raise SystemExit("install.iss CUDA_URL 与 download.py 首个镜像不一致:\n"
                         f"  iss: {url}\n  dl : {dl.TORCH_CUDA_URLS[0]}")
    print(f"  CUDA 资产一致: {wheel_zip} / {size} 字节 / SHA {sha[:16]}…")


def _check_prune_list() -> None:
    """install.iss 的 Pascal 裁剪必须覆盖 download._prune_torch_install 的清单。

    Pascal 版在 Inno 脚本环境里跑，无法复用 Python 实现（这是它必须重复的
    原因）；清单唯一来源是 uvr_lite/download.py，这里核对标记词防两边漂移。
    """
    iss = (ROOT / "installer" / "install.iss").read_text(encoding="utf-8")
    required = {
        "'*.lib'": "torch/lib 编译期导入库",
        "T + 'include'": "torch/include 头文件目录",
        "T + 'bin'": "torch/bin 可执行文件",
        "'torch_shm_manager.exe'": "bin 内必须保留的 shm 管理器",
    }
    missing = [why for token, why in required.items() if token not in iss]
    if missing:
        raise SystemExit("install.iss 的 PruneTorchCuda 缺少裁剪项: " + "、".join(missing)
                         + "（唯一来源: uvr_lite/download.py _prune_torch_install）")
    print("  torch 裁剪清单一致: " + "、".join(required.values()))


def _pkg_name(spec: str) -> str:
    """依赖声明 → 归一化包名（PEP 503：小写 + 连字符，去掉版本/extra 约束）。"""
    return re.split(r"[<>=!~\[; (]", spec.strip(), maxsplit=1)[0].lower().replace("_", "-")


def _check_dependency_list() -> None:
    """pyproject.toml 的运行依赖必须都进 DEPENDENCIES（装机漏包 = 启动即失败）。"""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    if not m:
        raise SystemExit("pyproject.toml 未找到 [project].dependencies，依赖校验无法完成")
    want = {_pkg_name(s) for s in re.findall(r'"([^"]+)"', m.group(1))}
    have = {_pkg_name(s) for s in DEPENDENCIES}
    missing = want - have - DEPS_INSTALLED_SEPARATELY
    if missing:
        raise SystemExit(f"DEPENDENCIES 缺少 pyproject 运行依赖: {sorted(missing)}"
                         "（安装包里会缺包，应用启动即失败）")
    print(f"  依赖清单一致: pyproject 的 {len(want)} 项运行依赖均已覆盖安装包")


def _check_packaging_consistency() -> None:
    """打包前校验三处跨文件/跨语言的同款事实，漂移即硬失败。"""
    _check_cuda_assets()
    _check_prune_list()
    _check_dependency_list()


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
    _install_torch(bundle_dir, TORCH_CPU_INDEXES)

    # 5. 模型权重（SHA256 校验；已就绪则复用）。本地文件名由注册表的
    #    filename 字段决定（safetensors 或 ckpt），不写死扩展名
    os.environ["UVR_MODEL_DIR"] = str(bundle_dir / "models")
    from uvr_lite.download import ensure_model
    from uvr_lite.models import DEFAULT_MODEL, get_model_info

    model_file = bundle_dir / "models" / get_model_info(DEFAULT_MODEL).get(
        "filename", f"{DEFAULT_MODEL}.ckpt")
    ensure_model(DEFAULT_MODEL)
    print(f"[4/5] 模型: {model_file}")


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
    _check_packaging_consistency()
    cmd = [
        str(iscc),
        f"/DMyAppVersion={__version__}",
        str(ROOT / "installer" / "install.iss"),
    ]
    print("[5/5] ISCC 编译中（压缩需较长时间）…")
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
