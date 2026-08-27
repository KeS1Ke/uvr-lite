"""模型下载：从主源拉取权重到 models/ 目录，带 SHA256 完整性校验。

- 断点续传：`.part` 文件已存在时用 HTTP Range 头续传，避免中断后全量重下
- 多源回退：主源（GitHub Releases）失败时自动切换 HuggingFace 镜像

权重文件大（639MB+），不入 git；安装脚本与首次分离前自动调用本模块。
"""

import hashlib
import os
import shutil
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import _base_dir
from .models import MODEL_REGISTRY, get_model_info

# 浏览器 UA：阿里云 pytorch-wheels 等镜像对非浏览器 UA 返回 403（曾实测），
# GitHub/镜像站对默认 UA 也可能限速。
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

# tqdm 仅用于终端进度条装饰；安装链环境（绿色 Python）可能没有它，
# 下载逻辑本身走 progress_callback，无 tqdm 时静默降级。
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def repo_root() -> Path:
    """仓库/安装根目录（models/ 与 torch_cpu/ 等所在层）。

    与 __init__._base_dir 同一语义（dev {repo} 与安装 {inst} 层级不同，
    按目录特征判定，见 _base_dir 注释）。
    """
    return _base_dir()


def models_dir() -> Path:
    """模型目录：优先环境变量 UVR_MODEL_DIR（安装场景指向安装目录），
    否则为仓库根下的 models/。"""
    override = os.environ.get("UVR_MODEL_DIR")
    d = Path(override) if override else repo_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def config_path(name: str) -> Path:
    """模型配置 yaml（随包分发，位于 uvr_lite/configs/）"""
    info = get_model_info(name)
    return Path(__file__).resolve().parent / "configs" / info["config"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------- 校验缓存 ----------
# 权重文件 640MB，每轮运行全量 SHA256 约 1s+；用 {ckpt}.verified 标记记录
# (size, mtime_ns)，文件未变则跳过全量哈希。标记丢失/文件变更时重新校验。

def _verified_marker(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.name + ".verified")


def _check_verified(ckpt: Path) -> bool:
    """标记记录的 (size, mtime_ns) 与当前文件一致 → 视为已验证。"""
    marker = _verified_marker(ckpt)
    try:
        st = ckpt.stat()
        return marker.read_text(encoding="utf-8").strip() == f"{st.st_size}:{st.st_mtime_ns}"
    except (OSError, ValueError):
        return False


def _mark_verified(ckpt: Path) -> None:
    try:
        st = ckpt.stat()
        _verified_marker(ckpt).write_text(f"{st.st_size}:{st.st_mtime_ns}", encoding="utf-8")
    except OSError:
        pass


def _download_single(url: str, tmp: Path,
                     progress_callback: Callable[[int, int], bool] | None = None) -> None:
    """单连接下载（Range 断点续传）；失败抛异常由上层切换源。

    progress_callback(done_bytes, total_bytes) -> bool：返回 False 视为用户
    取消，抛 InterruptedError（保留 .part 供下次续传）。

    设置 socket 默认超时：服务器断流后 TCP 半开连接时，
    read 不会永远阻塞（否则安装器会无限卡死）。
    """
    socket.setdefaulttimeout(60)
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"User-Agent": UA}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        if e.code == 416 and existing:  # Range 超出文件末尾：文件已完整
            return
        raise

    resume = existing > 0 and resp.status == 206
    remaining = int(resp.headers.get("Content-Length", 0))
    total = existing + remaining if resume else remaining

    mode = "ab" if resume else "wb"
    # 进度起点：续传从 existing 起；服务器忽略 Range 重下时文件被截断，从 0 起
    start = existing if resume else 0
    bar = None
    if tqdm is not None:
        bar = tqdm(initial=start, total=total, unit="B", unit_scale=True,
                   desc=f"下载 {tmp.name[:-5]}", miniters=1)
    try:
        with open(tmp, mode) as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
                # f.tell() 即已落盘字节（续传含 existing），不存在双计
                if progress_callback is not None and not progress_callback(f.tell(), total):
                    raise InterruptedError("下载已取消")
    finally:
        if bar is not None:
            bar.close()


PARALLEL_SEGMENTS = 8        # 分段下载并发连接数
PARALLEL_MIN_SIZE = 64 << 20  # 64MB 以上才分段（小文件单连接足够）
_REPORT_INTERVAL = 4 << 20    # 进度回调限频：每 4MB 上报一次


def _probe_range(url: str) -> tuple:
    """探测源：返回 (total_bytes, supports_range)。

    用 Range: bytes=0-0 请求——支持 Range 的服务器返回 206 + Content-Range
    （可拿到总大小）；忽略 Range 的服务器返回 200 全量（立即关闭连接）。
    """
    socket.setdefaulttimeout(30)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": "bytes=0-0"})
    resp = urllib.request.urlopen(req)
    try:
        if resp.status == 206:
            cr = resp.headers.get("Content-Range", "")  # bytes 0-0/<total>
            total = int(cr.rsplit("/", 1)[1]) if "/" in cr else 0
            return total, True
        total = int(resp.headers.get("Content-Length", 0))
        return total, False
    finally:
        resp.close()


def _download_segment(url: str, part: Path, start: int, end: int,
                      retries: int,
                      cancel_check: Callable[[], bool] | None = None,
                      on_chunk: Callable[[int], None] | None = None) -> None:
    """下载 [start, end] 段到 part 文件；已下载部分（段文件大小）自动续传。

    失败重试 retries 次（Range 续传，成本低）；全部失败抛 RuntimeError。
    cancel_check() 返回 True 时抛 InterruptedError（段文件保留）；
    on_chunk(n) 每写入 n 字节调用一次（进度增量上报）。
    """
    socket.setdefaulttimeout(60)
    for attempt in range(1 + retries):
        done = part.stat().st_size if part.exists() else 0
        if start + done > end:
            return
        headers = {"User-Agent": UA, "Range": f"bytes={start + done}-{end}"}
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers=headers))
        except urllib.error.HTTPError as e:
            if e.code == 416 and done > 0:
                return
            raise
        try:
            with open(part, "ab" if done else "wb") as f:
                while chunk := resp.read(1 << 20):
                    if cancel_check is not None and cancel_check():
                        raise InterruptedError("下载已取消")
                    f.write(chunk)
                    if on_chunk is not None:
                        on_chunk(len(chunk))
            return
        except InterruptedError:
            raise
        except Exception as e:
            if attempt >= retries:
                raise RuntimeError(f"段 {start}-{end} 下载失败: {e}") from e
    raise RuntimeError(f"段 {start}-{end} 下载失败")


def _download_parallel(url: str, tmp: Path, total: int,
                       progress_callback: Callable[[int, int], bool] | None = None,
                       segments: int = PARALLEL_SEGMENTS,
                       retries: int = 2) -> None:
    """多连接分段下载到 tmp（段文件 tmp.s0..sN，完成后按序合并）。

    取消：progress_callback 返回 False → 各段线程抛 InterruptedError
    （段文件保留，下次续传）。
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    seg_size = (total + segments - 1) // segments
    parts = [tmp.with_name(f"{tmp.name}.s{i}") for i in range(segments)]
    lock = threading.Lock()
    state = {"done": 0, "last_report": 0, "cancel": False}

    def report(n: int) -> None:
        """增量累计已下载字节；限频上报进度，返回 False 置取消标志。"""
        with lock:
            state["done"] += n
            if (progress_callback is not None
                    and state["done"] - state["last_report"] >= _REPORT_INTERVAL):
                state["last_report"] = state["done"]
                if not progress_callback(state["done"], total):
                    state["cancel"] = True
            # 全部完成后补一次最终回调（限频可能截断最后不足 4MB）
            if progress_callback is not None and state["done"] >= total \
                    and state["last_report"] < total:
                state["last_report"] = total
                progress_callback(total, total)

    def work(i: int) -> None:
        start = i * seg_size
        end = min(start + seg_size - 1, total - 1)
        part = parts[i]
        with lock:  # 段文件已有内容（续传）先计入进度
            state["done"] += part.stat().st_size if part.exists() else 0
        _download_segment(url, part, start, end, retries,
                          cancel_check=lambda: state["cancel"],
                          on_chunk=report)

    with ThreadPoolExecutor(max_workers=segments) as ex:
        futures = [ex.submit(work, i) for i in range(segments)]
        for fut in as_completed(futures):
            fut.result()  # 取消/段失败异常在此传播

    # 按序合并段文件 → tmp
    with open(tmp, "wb") as out:
        for part in parts:
            if part.exists():
                with open(part, "rb") as p:
                    while chunk := p.read(1 << 20):
                        out.write(chunk)
                part.unlink()


def _download(urls: list[str], dest: Path,
              progress_callback: Callable[[int, int], bool] | None = None,
              retries: int = 2) -> None:
    """按顺序尝试各下载源（每源最多重试 retries 次）；全部失败才报错。

    大文件（≥64MB）且源支持 Range 时走多连接分段下载（提速）；
    否则单连接。用户取消（InterruptedError）不切换源、保留 .part 供续传。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in urls:
        for attempt in range(1 + retries):
            try:
                total, supports_range = _probe_range(url)
                if supports_range and total >= PARALLEL_MIN_SIZE:
                    _download_parallel(url, tmp, total, progress_callback)
                else:
                    _download_single(url, tmp, progress_callback)
                tmp.replace(dest)
                return
            except InterruptedError:
                raise
            except Exception as e:
                errors.append(f"{url}（第 {attempt + 1} 次）: {e}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError("所有下载源均失败:\n" + "\n".join(errors))


def ensure_model(name: str, force: bool = False,
                 progress_callback: Callable[[int, int], bool] | None = None) -> Path:
    """确保模型权重已下载且 SHA256 匹配；返回权重路径。

    校验缓存：{ckpt}.verified 标记（size+mtime）命中时跳过全量哈希。
    权重的本地文件名由注册表 filename 决定（.ckpt 或 .safetensors）。
    """
    info = get_model_info(name)
    ckpt = models_dir() / info.get("filename", f"{name}.ckpt")

    if ckpt.exists() and not force:
        if _check_verified(ckpt):
            print(f"模型已就绪: {ckpt.name}（{ckpt.stat().st_size / 1e6:.0f} MB）")
            return ckpt
        if sha256_of(ckpt) == info["sha256"]:
            _mark_verified(ckpt)
            print(f"模型已就绪: {ckpt.name}（{ckpt.stat().st_size / 1e6:.0f} MB）")
            return ckpt
        print(f"校验失败，重新下载: {ckpt.name}")
        ckpt.unlink()

    print(f"下载模型 {name}（{info['description']}）")
    urls = [info["ckpt_url"], *info.get("mirror_urls", [])]
    _download(urls, ckpt, progress_callback)
    actual = sha256_of(ckpt)
    if actual != info["sha256"]:
        ckpt.unlink()
        raise RuntimeError(
            f"SHA256 校验失败: 期望 {info['sha256']}，实际 {actual}。"
            f"下载源可能已变更，请检查 {info['ckpt_url']}"
        )
    _mark_verified(ckpt)
    print(f"完成: {ckpt}（{ckpt.stat().st_size / 1e6:.0f} MB）")
    return ckpt


def download_all() -> None:
    for name in MODEL_REGISTRY:
        ensure_model(name)


# ---------- CUDA 推理引擎（torch 二进制）下载 ----------
# 单包安装只含 CPU torch；CUDA 引擎由用户应用内/CLI 额外下载（半在线模式，
# 与 UVR 官方同策略）。wheel 自包含全部 CUDA 运行库（torch/lib 内 16 个
# cudnn/cublas/cufft DLL，无独立 nvidia-* 包，解压即用）。
# 镜像按实测速度排序（2026-08-04）：SJTU 15-17MB/s > 官方 13-14MB/s >
# 阿里云 3-4MB/s（需浏览器 UA，403 已修）；南大无 pytorch-wheels（404）。
# SHA256 于打包时下载一次算得（2bb8c05d…，3273024349 字节）。
TORCH_CUDA_WHEEL = "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
TORCH_CUDA_SHA256 = "2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30"
# URL 中 "+" 用 %2B 编码：官方源（S3/CloudFront）对字面 + 返回 403，此前
# 官方回退源一直是坏的；SJTU/阿里云对两种形式均可（与 install.iss 一致）。
TORCH_CUDA_WHEEL_ENC = TORCH_CUDA_WHEEL.replace("+", "%2B")
TORCH_CUDA_URLS = [
    f"https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cu128/{TORCH_CUDA_WHEEL_ENC}",
    f"https://download.pytorch.org/whl/cu128/{TORCH_CUDA_WHEEL_ENC}",
    f"https://mirrors.aliyun.com/pytorch-wheels/cu128/{TORCH_CUDA_WHEEL_ENC}",
]


def _wheel_cache_dir() -> Path:
    """wheel 中转目录（系统临时目录；.part 断点续传文件同处）。"""
    d = Path(tempfile.gettempdir()) / "uvr-lite"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cuda_torch_installed(base: Path | None = None) -> bool:
    """CUDA 引擎是否已安装（{base}/torch_cuda/torch/__init__.py 存在）。"""
    base = Path(base) if base else repo_root()
    return (base / "torch_cuda" / "torch" / "__init__.py").exists()


def _prune_torch_install(dest: Path) -> None:
    """裁剪 torch 目录：删编译期 .lib / include / bin（运行时不需要）。

    与打包脚本 build_installer._prune_torch 同款；实测 torch_cuda
    5.6G→4.7G（-900M），import + 真实分离（CPU/GPU）验证无损。
    bin/ 保留 torch_shm_manager.exe（torch 多进程共享内存需要）。
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


def install_cuda_torch(base: Path | None = None,
                       progress_callback: Callable[[int, int], bool] | None = None) -> Path:
    """下载并安装 CUDA 推理引擎到 {base}/torch_cuda（与应用同目录）。

    流程：已安装则直接返回 → _download（多段并发 + 断点续传 + 镜像回退）
    → SHA256 校验（不匹配删缓存并报错）→ zipfile 解压（压缩字节进度，
    与下载阶段同尺度，进度条不回跳）→ 裁剪 .lib/include/bin。

    progress_callback(done, total) 字节语义贯穿下载与解压阶段；返回 False
    视为取消（抛 InterruptedError；wheel 缓存保留，下次直接从解压开始）。
    """
    base = Path(base) if base else repo_root()
    dest = base / "torch_cuda"
    if cuda_torch_installed(base):
        print(f"CUDA 引擎已就绪: {dest}")
        return dest

    wheel = _wheel_cache_dir() / TORCH_CUDA_WHEEL
    print(f"下载 CUDA 推理引擎（{TORCH_CUDA_WHEEL}，约 3.3 GB，多段并行 + 镜像回退）…")
    download_total = 0

    def _track_download(done: int, total: int) -> bool:
        """记录下载阶段总字节（解压阶段进度接续用，避免进度条回跳）。"""
        nonlocal download_total
        download_total = total
        return True if progress_callback is None else progress_callback(done, total)

    _download(TORCH_CUDA_URLS, wheel, _track_download)
    actual = sha256_of(wheel)
    if actual != TORCH_CUDA_SHA256:
        wheel.unlink(missing_ok=True)
        wheel.with_suffix(wheel.suffix + ".part").unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 校验失败: 期望 {TORCH_CUDA_SHA256[:16]}…，实际 {actual[:16]}…。"
            f"下载源可能已变更，请稍后重试")

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # 手动迭代解压（extractall 无进度回调；3.3GB 解压约 1-2 分钟需可见进度）。
    # 进度按压缩字节累计并接续下载阶段（同一 total 尺度，进度条连续不回跳）。
    import zipfile

    offset = download_total
    with zipfile.ZipFile(wheel) as zf:
        total = offset + sum(i.compress_size for i in zf.infolist())
        done = last_report = offset
        for info in zf.infolist():
            zf.extract(info, dest)
            done += info.compress_size
            if progress_callback is not None and done - last_report >= _REPORT_INTERVAL:
                last_report = done
                if not progress_callback(done, total):
                    raise InterruptedError("解压已取消")
        if progress_callback is not None and last_report < total:
            progress_callback(total, total)
    wheel.unlink(missing_ok=True)

    _prune_torch_install(dest)
    print(f"完成: {dest}（CUDA 引擎已安装，重启应用后生效）")
    return dest


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        download_all()
    else:
        for a in args:
            ensure_model(a)
