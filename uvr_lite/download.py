"""模型下载：从注册表列出的下载源拉取权重到 models/ 目录，带 SHA256 完整性校验。

- 断点续传：`.part`（多分段时 `.part.s0..sN`）已存在时用 HTTP Range 头续传，
  避免中断后全量重下；取消与全部源失败都保留已下载分片
- 多源回退：按 ckpt_url → mirror_urls 顺序尝试，逐个失败才报错；注册表当前
  mirror_urls 为空（未发布同内容镜像，原因见 models.py），实际只有主源

权重为数百 MB 量级，不入 git；安装脚本与首次分离前自动调用本模块。
"""

import contextlib
import hashlib
import logging
import os
import platform
import re
import shutil
import socket
import struct
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

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


def _log() -> logging.Logger:
    """logger 惰性取用（log 模块的 log_dir() 反向依赖本模块的 repo_root()）。

    log 与本模块顶层互相 import 会形成循环（谁先被导入谁少了对方的符号），
    失败路径才走到这里，正常路径无额外开销。
    """
    from .log import get_logger

    return get_logger("uvr_lite.download")


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


def model_file(name: str) -> Path:
    """本地权重路径，以注册表 filename 为准（safetensors 或 .ckpt）。

    注册表的 filename 是本地文件名的唯一权威来源；调用方不得自行拼接
    `{name}.ckpt`，否则已存在的 safetensors 权重会被判成未下载而重复下载。
    """
    info = get_model_info(name)
    return models_dir() / info.get("filename", f"{name}.ckpt")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def retired_model_files(name: str) -> list[Path]:
    """退役（旧分发物）权重的本地路径，由注册表 retired_filenames 决定。

    与 model_file 同一权威来源（filename/retired_filenames 都在注册表里）；
    只给出路径，不判断存在性、不碰文件。CLI 的 models 列表可据此提示可回收项。
    """
    info = get_model_info(name)
    return [models_dir() / fn for fn in info.get("retired_filenames", [])]


# ---------- 校验缓存 ----------
# 权重文件数百 MB，每轮运行全量 SHA256 约 1s+。标记内容是
# `{期望 sha256}:{文件大小}`：sha 等于本次注册表期望且大小等于当前文件才跳过哈希。
# 旧的 `{size}:{mtime_ns}` 对不上这个格式，一律视为未命中，重新哈希后改写。

def _verified_marker(ckpt: Path) -> Path:
    return ckpt.with_name(ckpt.name + ".verified")


def _check_verified(ckpt: Path, expected_sha: str) -> bool:
    """标记里的 sha 等于本次期望、且大小等于当前文件 → 视为已验证。

    只认 `{sha256}:{size}`。旧的 `{size}:{mtime_ns}`（不绑 sha）一律未命中。
    """
    marker = _verified_marker(ckpt)
    try:
        recorded = marker.read_text(encoding="utf-8").strip()
        sha, sep, size_s = recorded.partition(":")
        if not sep or not sha or not size_s:
            return False
        return sha == expected_sha and int(size_s) == ckpt.stat().st_size
    except (OSError, ValueError):
        return False


def _mark_verified(ckpt: Path, expected_sha: str) -> None:
    try:
        size = ckpt.stat().st_size
        _verified_marker(ckpt).write_text(f"{expected_sha}:{size}", encoding="utf-8")
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
        # 服务器忽略 Range 时返回 200 全量：此时必须 "wb" 截断重写段文件，
        # 否则追加模式会把整段内容接在已下载字节之后 → 合并出的权重损坏。
        resume = done > 0 and resp.status == 206
        try:
            with open(part, "ab" if resume else "wb") as f:
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
    否则单连接。取消（InterruptedError）不切换源；取消与全部源失败都保留
    已下载分片（.part / .part.s*），下次从断点续传。
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
    # 全部源失败也保留已下载分片（单连接 .part / 分段 .part.s0..sN）：下次仍可
    # 按 Range 续传；残留分片不会被误当完整数据——ab/wb 由响应状态决定，
    # 落盘后还要过 SHA256 校验。
    # 下载失败的细节（每个源每次重试的异常）在 UI 上只会显示一行，落盘留存文件名
    # 与源后才有可能让用户报得出有效信息；异常消息本身保持原样不变。
    _log().error("所有下载源均失败: %s -> %s\n%s", dest.name, urls, "\n".join(errors))
    raise RuntimeError("所有下载源均失败:\n" + "\n".join(errors))


def _reclaim_retired(name: str, ckpt: Path) -> None:
    """回收退役的旧分发物（升级换权重格式后的一次性清理）。

    只允许在「新权重已就绪」之后调用：新权重还没拿到就删旧文件，用户手里
    一份可用的权重都不剩。删除失败（被占用/权限不足）只记日志，不影响使用。

    日志用 WARNING 而非 INFO：log.py 只让 WARNING 及以上落盘（见其模块注释），
    INFO 记录等于没记；删掉数百 MB 用户文件也该在报障日志里留痕。
    """
    for old in retired_model_files(name):
        if old == ckpt or not old.exists():
            # 注册表把当前 filename 误列进 retired 时，绝不能删刚校验通过的权重
            continue
        try:
            size = old.stat().st_size
            old.unlink()
        except OSError as e:
            _log().warning("退役权重回收失败（不影响使用）: %s: %s", old, e)
            continue
        # {old}.verified 是它的附属物，一并回收，否则留下指向空文件的孤儿；
        # 标记只是校验缓存，删不掉也无害，不打断回收流程
        with contextlib.suppress(OSError):
            _verified_marker(old).unlink(missing_ok=True)
        _log().warning("已回收退役权重: %s（%.0f MB），当前使用 %s",
                       old.name, size / 1e6, ckpt.name)


def ensure_model(name: str, force: bool = False,
                 progress_callback: Callable[[int, int], bool] | None = None) -> Path:
    """确保模型权重已下载且 SHA256 匹配；返回权重路径。

    校验缓存：{ckpt}.verified 记 `{期望 sha256}:{size}`，两者都对上才跳过全量哈希。
    权重的本地文件名由注册表 filename 决定（.ckpt 或 .safetensors）。
    新权重就绪（标记命中 / 哈希通过 / 下载后校验通过）才回收注册表里声明的
    退役权重（retired_filenames，见 models.py）；未就绪时不动任何文件。
    """
    info = get_model_info(name)
    ckpt = model_file(name)
    expected_sha = info["sha256"]

    if ckpt.exists() and not force:
        if _check_verified(ckpt, expected_sha):
            print(f"模型已就绪: {ckpt.name}（{ckpt.stat().st_size / 1e6:.0f} MB）")
            _reclaim_retired(name, ckpt)
            return ckpt
        if sha256_of(ckpt) == expected_sha:
            _mark_verified(ckpt, expected_sha)
            print(f"模型已就绪: {ckpt.name}（{ckpt.stat().st_size / 1e6:.0f} MB）")
            _reclaim_retired(name, ckpt)
            return ckpt
        print(f"校验失败，重新下载: {ckpt.name}")
        _log().warning("权重 SHA256 不匹配，删除缓存准备重新下载: %s", ckpt)
        ckpt.unlink()

    print(f"下载模型 {name}（{info['description']}）")
    urls = [info["ckpt_url"], *info.get("mirror_urls", [])]
    _download(urls, ckpt, progress_callback)
    actual = sha256_of(ckpt)
    if actual != expected_sha:
        ckpt.unlink()
        # 校验失败意味着下载源可能已变更（或被劫持）：留下实际哈希，便于用户
        # 报障时确认是源变了还是网络截断。异常消息保持原样（测试会断言）。
        _log().error("下载后 SHA256 校验失败: %s 期望 %s，实际 %s", ckpt, expected_sha, actual)
        raise RuntimeError(
            f"SHA256 校验失败: 期望 {expected_sha}，实际 {actual}。"
            f"下载源可能已变更，请检查 {info['ckpt_url']}"
        )
    _mark_verified(ckpt, expected_sha)
    print(f"完成: {ckpt}（{ckpt.stat().st_size / 1e6:.0f} MB）")
    _reclaim_retired(name, ckpt)
    return ckpt


def download_all() -> None:
    for name in MODEL_REGISTRY:
        ensure_model(name)


# ---------- CUDA 推理引擎（torch 二进制）下载 ----------
# 单包安装只含 CPU torch；CUDA 引擎由用户应用内/CLI 额外下载（半在线模式，
# 与 UVR 官方同策略）。wheel 自包含全部 CUDA 运行库（torch/lib 内 16 个
# cudnn/cublas/cufft DLL，无独立 nvidia-* 包，解压即用）。
# 镜像按实测速度排序（2026-09-29 复测：3.27GB wheel，8 并发聚合 / 单连接）：
#   官方 4.4 / 2.19MB/s > SJTU 2.0 / 2.02MB/s > 阿里云 0.17 / 0.02MB/s。
# 2026-08-04 旧基线为 SJTU 15-17 > 官方 13-14 > 阿里云 3-4，已反转。
# 回退只在「失败」时触发，慢但可用的源不会被跳过，所以最快的必须排首位。
# 阿里云实测已近不可用（0.17MB/s ≈ 320 分钟，且样点会超时），仅作末位兜底。
# 南大无 pytorch-wheels（404）。
# SHA256 与字节数于打包时下载一次算得；install.iss 的 CUDA_SHA / ExternalSize
# 硬编码同款，由 scripts/build_installer 打包前的一致性校验逐项核对防漂移。
#
# ABI 与平台：这份 wheel 只有 cp312 / win_amd64 一款，而 requires-python 声明
# >= 3.10。修前不校验环境，Windows 上的 3.10/3.11 会照下 cp312 wheel，import
# 时才炸（ABI 不匹配，3.3 GB 白花）；非 Windows 则下到一份根本装不上的
# win_amd64 wheel。现在装前先过 cuda_engine_requirements() 门槛，把「当前环境
# 拿不到这份 wheel」变成一条可行动的中文提示，而不是 3.3 GB 之后的 ImportError。
# 支持多 ABI 需要为每个 ABI 各记一份实测 SHA256/字节数，故留待按需扩充。
TORCH_CUDA_WHEEL = "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl"
TORCH_CUDA_SHA256 = "2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30"
# wheel 字节数（install.iss [Files] ExternalSize 的权威来源；本模块不读它）
TORCH_CUDA_SIZE = 3273024349

# wheel 文件名的解释器 ABI / 平台标签从文件名派生（单一来源）：手写重复常量
# 迟早与 TORCH_CUDA_WHEEL 漂移，CudaEngineRequirements 的 property 也走同一解析。
_WHEEL_TAGS_RE = re.compile(r"^.+-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def _parse_wheel_tags(wheel: str) -> tuple[str, str]:
    """从 wheel 文件名解析 (Python 标签, 平台标签)（PEP 427 文件名末三段）。"""
    m = _WHEEL_TAGS_RE.match(wheel)
    if m is None:
        raise ValueError(f"无法从 wheel 文件名解析 ABI/平台标签: {wheel}")
    return m.group("py"), m.group("plat")


TORCH_CUDA_PY_TAG, TORCH_CUDA_PLATFORM_TAG = _parse_wheel_tags(TORCH_CUDA_WHEEL)
# URL 中 "+" 用 %2B 编码：官方源（S3/CloudFront）对字面 + 返回 403，此前
# 官方回退源一直是坏的；SJTU/阿里云对两种形式均可（与 install.iss 一致）。
TORCH_CUDA_WHEEL_ENC = TORCH_CUDA_WHEEL.replace("+", "%2B")
TORCH_CUDA_URLS = [
    f"https://download.pytorch.org/whl/cu128/{TORCH_CUDA_WHEEL_ENC}",
    f"https://mirrors.sjtug.sjtu.edu.cn/pytorch-wheels/cu128/{TORCH_CUDA_WHEEL_ENC}",
    f"https://mirrors.aliyun.com/pytorch-wheels/cu128/{TORCH_CUDA_WHEEL_ENC}",
]


class CudaEngineUnsupportedError(RuntimeError):
    """当前环境与内置 CUDA wheel 的 ABI/平台不匹配（详见 cuda_engine_requirements）。"""


class CudaEngineRequirements(NamedTuple):
    """一份可用 CUDA wheel 的完整下载凭据。

    把 wheel 名 + SHA256 + 字节数 + 来源 URL 收在一处：日后要支持第二个 ABI，
    只需在这里多一张表，不必让 install_cuda_torch 里散落着 cp312 的假设。
    """

    wheel: str
    sha256: str
    size: int
    urls: list[str]

    @property
    def py_tag(self) -> str:
        return _parse_wheel_tags(self.wheel)[0]

    @property
    def platform_tag(self) -> str:
        return _parse_wheel_tags(self.wheel)[1]


# Windows 的 sys.platform 恒为 "win32"——32 位、x64 与 ARM64 都是；只看它会把
# win_amd64 wheel 放行给用不了的环境。以下是 platform.machine() 在 x64 上
# 各发行版的常见取值，统一转小写比较。
_X64_MACHINES = {"amd64", "x86_64", "x64", "em64t"}


def cuda_engine_requirements(platform_str: str | None = None,
                             version_info=None,
                             machine: str | None = None,
                             pointer_size: int | None = None) -> CudaEngineRequirements:
    """当前解释器能否使用内置 CUDA wheel；不能则抛 CudaEngineUnsupportedError。

    四个参数默认读 sys.platform / sys.version_info / platform.machine() /
    struct.calcsize("P")，仅为测试可注入而存在。

    放行条件 = Windows（win32）+ x64 架构 + 64 位解释器（指针 8 字节）+
    Python ABI 标签与 wheel 一致；缺一即以可行动的中文错误拦下。

    拦截而非「按 ABI 换 wheel」的原因：换 wheel 就得为每个 ABI 各记一份实测
    SHA256 与字节数（否则等于放弃完整性校验）。在补齐之前，让不匹配的环境
    立刻拿到可行动的错误，好过下载 3.3 GB 之后才在 import 时崩。

    报错文本要同时给出「缺什么」（ABI/平台/架构/位数）和「怎么办」（用 3.12
    的安装包，或 PyPI 装对应 CUDA 版 torch），这是非专业用户唯一能照做的信息。
    """
    plat = sys.platform if platform_str is None else platform_str
    version = sys.version_info if version_info is None else version_info
    arch = platform.machine() if machine is None else machine
    bits = struct.calcsize("P") if pointer_size is None else pointer_size

    running_py_tag = f"cp{version[0]}{version[1]}"
    if (plat == "win32" and arch.lower() in _X64_MACHINES and bits == 8
            and running_py_tag == TORCH_CUDA_PY_TAG):
        return CudaEngineRequirements(
            wheel=TORCH_CUDA_WHEEL,
            sha256=TORCH_CUDA_SHA256,
            size=TORCH_CUDA_SIZE,
            urls=list(TORCH_CUDA_URLS),
        )

    py_ver = f"{TORCH_CUDA_PY_TAG[2]}.{TORCH_CUDA_PY_TAG[3:]}"
    need = [f"64 位 x64 Windows + Python {py_ver}（{TORCH_CUDA_PY_TAG}）"]
    if plat != "win32":
        need.append(f"当前系统不是 Windows（sys.platform={plat}）")
    if arch.lower() not in _X64_MACHINES:
        need.append(f"当前是 {arch} 架构（非 x64）")
    if bits != 8:
        need.append(f"当前是 {bits * 8} 位解释器")
    if running_py_tag != TORCH_CUDA_PY_TAG:
        need.append(f"当前是 Python {version[0]}.{version[1]}（{running_py_tag}）")
    raise CudaEngineUnsupportedError(
        "CUDA 推理引擎只提供 "
        f"{TORCH_CUDA_PY_TAG}/{TORCH_CUDA_PLATFORM_TAG} 一款 wheel（{TORCH_CUDA_WHEEL}），"
        f"当前环境不匹配：{'；'.join(need)}。"
        "CPU 引擎不受影响，可直接用（速度较慢）。若要 GPU 加速，二选一："
        f"① 装官方安装包（内置 Python {py_ver} + CPU torch，CUDA 引擎在安装时勾选下载）；"
        f"② 在 Python {py_ver} 的 64 位 x64 Windows 环境里，"
        "改用 PyPI 装对应 CUDA 版 torch："
        "pip install torch --index-url https://download.pytorch.org/whl/cu128"
    )


def _wheel_cache_dir() -> Path:
    """wheel 中转目录（系统临时目录；.part 断点续传文件同处）。"""
    d = Path(tempfile.gettempdir()) / "uvr-lite"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cuda_complete_marker(dest: Path) -> Path:
    return dest / ".complete"


def _wheel_sha_sidecar(wheel: Path) -> Path:
    """已通过校验的 wheel 旁的侧车：内容是期望 sha256。"""
    return wheel.with_name(wheel.name + ".sha256")


def _cuda_wheel_cached(wheel: Path, expected_sha: str) -> bool:
    """wheel 在且侧车记录的 sha 等于本次期望 → 不必再下载。"""
    if not wheel.is_file():
        return False
    try:
        recorded = _wheel_sha_sidecar(wheel).read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return recorded == expected_sha


def cuda_torch_installed(base: Path | None = None) -> bool:
    """CUDA 引擎是否已安装。

    新安装：`torch_cuda/.complete` 与 `torch/__init__.py` 都在。
    旧的完整安装没有标记：`torch/__init__.py` 与 `torch/version.py` 都在则视为
    已安装，并补写 `.complete`（避免用户重下 3.3GB）。只有残缺目录
    （有 `__init__.py` 但没有 `version.py`，或正式目录缺关键文件）不算已安装。
    """
    base = Path(base) if base else repo_root()
    dest = base / "torch_cuda"
    init_py = dest / "torch" / "__init__.py"
    marker = _cuda_complete_marker(dest)
    if marker.is_file() and init_py.is_file():
        return True
    version_py = dest / "torch" / "version.py"
    if init_py.is_file() and version_py.is_file():
        try:
            marker.write_text("", encoding="utf-8")
        except OSError:
            _log().warning("无法补写 CUDA 完成标记（仍视为已安装）: %s", marker)
        return True
    return False


def cuda_engine_supported(platform_str: str | None = None, version_info=None,
                          machine: str | None = None, pointer_size: int | None = None) -> bool:
    """当前环境能否用内置 CUDA wheel（不抛异常版，给 UI 的可用性提示用）。"""
    try:
        cuda_engine_requirements(platform_str=platform_str, version_info=version_info,
                                 machine=machine, pointer_size=pointer_size)
    except CudaEngineUnsupportedError:
        return False
    return True


def cuda_engine_block_reason(platform_str: str | None = None, version_info=None,
                             machine: str | None = None, pointer_size: int | None = None) -> str:
    """环境不匹配时的**完整**原因说明（含「怎么办」①②）；环境可用时返回空串。

    UI 状态栏只截首句，完整文案进 tooltip。UI 不去跨模块 except 异常类：本模块
    可能被 importlib.reload（测试里就有一处），reload 会重建类对象，UI 手里的旧
    类引用再也抓不住新抛出的异常——表现为窗口构造直接失败。返回字符串没有这个
    身份问题。
    """
    try:
        cuda_engine_requirements(platform_str=platform_str, version_info=version_info,
                                 machine=machine, pointer_size=pointer_size)
    except CudaEngineUnsupportedError as e:
        return str(e)
    return ""


def _prune_torch_install(dest: Path) -> None:
    """裁剪 torch 目录：删编译期 .lib / include / bin（运行时不需要）。

    裁剪清单的**唯一来源**：打包脚本 build_installer._prune_torch 直接调用本
    函数（此前两份实现同款重复）；install.iss 的 Pascal 版在 Inno 脚本环境里
    跑，无法复用 Python 实现，只能硬编码同款，由 build_installer 打包前
    文本核对防漂移（_check_prune_list）。
    实测收益：torch_cuda 5.6G→4.7G（-900M）、torch_cpu 1.4G→461M（-940M），
    import + 真实分离（CPU/GPU）验证无损。
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
                       progress_callback: Callable[[int, int], bool] | None = None,
                       platform_str: str | None = None, version_info=None,
                       machine: str | None = None, pointer_size: int | None = None) -> Path:
    """下载并安装 CUDA 推理引擎到 {base}/torch_cuda（与应用同目录）。

    流程：已安装则直接返回 → 环境匹配检查 → wheel 已在且侧车 sha 匹配则跳过
    下载，否则 _download（多段并发 + 断点续传 + 镜像回退）→ SHA256 校验
    （不匹配删缓存并报错，消息含完整哈希与 URL；通过后把期望 sha 写到 wheel
    旁的侧车）→ 解压到 {base}/torch_cuda.partial（不先建正式目录）→ 裁剪
    .lib/include/bin → 换成正式目录，最后写 torch_cuda/.complete。

    已安装判断在环境门槛之前：装好的引擎不因解释器换代/换机而被判不支持，
    重复调用必须幂等返回，不再做任何环境检查。残缺目录不算已安装，成功时
    先清掉正式目录再换上临时目录里的完整树。

    环境检查在下载之前：wheel 只有 cp312/win_amd64 一款，不匹配时必须在花掉
    3.3 GB 之前就报出可行动的错误（见 cuda_engine_requirements）。

    platform_str / version_info / machine / pointer_size 只透传给环境检查，为测试
    可注入而存在，生产路径一律走默认值（即真实 sys.platform / sys.version_info /
    platform.machine() / struct.calcsize("P")）。

    解压或换目录任一步失败都删除临时目录；InterruptedError 继续向上抛。
    取消发生在解压阶段时 wheel 与侧车都留着，下次跳过下载直接解压。
    解压并换上正式目录成功之后才删 wheel 与侧车。

    progress_callback(done, total) 字节语义贯穿下载与解压阶段；返回 False
    视为取消（抛 InterruptedError）。
    """
    base = Path(base) if base else repo_root()
    dest = base / "torch_cuda"
    if cuda_torch_installed(base):
        print(f"CUDA 引擎已就绪: {dest}")
        return dest

    req = cuda_engine_requirements(platform_str=platform_str, version_info=version_info,
                                   machine=machine, pointer_size=pointer_size)
    wheel = _wheel_cache_dir() / req.wheel
    sidecar = _wheel_sha_sidecar(wheel)
    download_total = 0

    def _track_download(done: int, total: int) -> bool:
        """记录下载阶段总字节（解压阶段进度接续用，避免进度条回跳）。"""
        nonlocal download_total
        download_total = total
        return True if progress_callback is None else progress_callback(done, total)

    if _cuda_wheel_cached(wheel, req.sha256):
        print(f"CUDA wheel 已校验，跳过下载: {wheel.name}")
    else:
        print(f"下载 CUDA 推理引擎（{req.wheel}，约 3.3 GB，多段并行 + 镜像回退）…")
        _download(req.urls, wheel, _track_download)
        actual = sha256_of(wheel)
        if actual != req.sha256:
            wheel.unlink(missing_ok=True)
            wheel.with_suffix(wheel.suffix + ".part").unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            _log().error("CUDA 引擎 wheel SHA256 校验失败: 期望 %s，实际 %s", req.sha256, actual)
            sources = " ".join(req.urls)
            raise RuntimeError(
                f"SHA256 校验失败: 期望 {req.sha256}，实际 {actual}。"
                f"下载源可能已变更，请检查 {sources}")
        sidecar.write_text(req.sha256, encoding="utf-8")

    # 解压到临时目录，成功后再换成正式 torch_cuda。先建正式目录再往里解，
    # 中断时 torch/__init__.py 会先落盘，cuda_torch_installed() 会误判已就绪。
    partial = base / "torch_cuda.partial"
    import zipfile

    offset = download_total
    try:
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        # 手动迭代解压（extractall 无进度回调；3.3GB 解压约 1-2 分钟需可见进度）。
        # 进度按压缩字节累计并接续下载阶段（同一 total 尺度，进度条连续不回跳）。
        with zipfile.ZipFile(wheel) as zf:
            total = offset + sum(i.compress_size for i in zf.infolist())
            done = last_report = offset
            for info in zf.infolist():
                zf.extract(info, partial)
                done += info.compress_size
                if progress_callback is not None and done - last_report >= _REPORT_INTERVAL:
                    last_report = done
                    if not progress_callback(done, total):
                        raise InterruptedError("解压已取消")
            if progress_callback is not None and last_report < total:
                progress_callback(total, total)
        _prune_torch_install(partial)
        if dest.exists():
            shutil.rmtree(dest)
        partial.rename(dest)
        # 成功的最后一步：正式目录上的完成标记（内容为期望 sha256）
        _cuda_complete_marker(dest).write_text(req.sha256, encoding="utf-8")
    except BaseException as exc:
        if isinstance(exc, InterruptedError):
            # 解压阶段取消：临时目录清掉，wheel 与侧车留着，下次跳过下载直接解压。
            _log().warning("CUDA 引擎安装已取消，删除半成品目录: %s", partial)
        else:
            _log().warning("CUDA 引擎安装失败，删除半成品目录: %s", partial)
            if isinstance(exc, zipfile.BadZipFile):
                # 侧车已写上但包打不开：留着会让下次跳过下载后再次失败。
                wheel.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    wheel.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    print(f"完成: {dest}（CUDA 引擎已安装，重启应用后生效）")
    return dest


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        download_all()
    else:
        for a in args:
            ensure_model(a)
