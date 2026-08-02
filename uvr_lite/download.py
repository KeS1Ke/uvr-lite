# coding: utf-8
"""模型下载：从主源拉取权重到 models/ 目录，带 SHA256 完整性校验。

- 断点续传：`.part` 文件已存在时用 HTTP Range 头续传，避免中断后全量重下
- 多源回退：主源（GitHub Releases）失败时自动切换 HuggingFace 镜像

权重文件大（639MB+），不入 git；安装脚本与首次分离前自动调用本模块。
"""

import hashlib
import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

from tqdm.auto import tqdm

from .models import MODEL_REGISTRY, get_model_info

UA = "uvr-lite/0.1"


def repo_root() -> Path:
    """仓库根目录（uvr_lite 包所在目录的上一级）"""
    return Path(__file__).resolve().parent.parent


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


def _download_single(url: str, tmp: Path,
                     progress_callback: Optional[Callable[[int, int], bool]] = None) -> None:
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
    with open(tmp, mode) as f, tqdm(
        initial=existing, total=total, unit="B", unit_scale=True,
        desc=f"下载 {tmp.name[:-5]}", miniters=1,
    ) as bar:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
            if progress_callback is not None and not progress_callback(existing + bar.n, total):
                raise InterruptedError("下载已取消")


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
                      cancel_check: Optional[Callable[[], bool]] = None,
                      on_chunk: Optional[Callable[[int], None]] = None) -> None:
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
        except Exception as e:  # noqa: BLE001
            if attempt >= retries:
                raise RuntimeError(f"段 {start}-{end} 下载失败: {e}") from e
    raise RuntimeError(f"段 {start}-{end} 下载失败")


def _download_parallel(url: str, tmp: Path, total: int,
                       progress_callback: Optional[Callable[[int, int], bool]] = None,
                       segments: int = PARALLEL_SEGMENTS,
                       retries: int = 2) -> None:
    """多连接分段下载到 tmp（段文件 tmp.s0..sN，完成后按序合并）。

    取消：progress_callback 返回 False → 各段线程抛 InterruptedError
    （段文件保留，下次续传）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    seg_size = (total + segments - 1) // segments
    parts = [tmp.with_name(f"{tmp.name}.s{i}") for i in range(segments)]
    lock = threading.Lock()
    state = {"done": 0, "last_report": 0, "cancel": False}

    def report(n: int) -> None:
        """增量累计已下载字节；限频上报进度，返回 False 置取消标志。"""
        with lock:
            state["done"] += n
            if progress_callback is not None and state["done"] - state["last_report"] >= _REPORT_INTERVAL:
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


def _download(urls: List[str], dest: Path,
              progress_callback: Optional[Callable[[int, int], bool]] = None,
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
            except Exception as e:  # noqa: BLE001 —— 换源/重试（断点续传，重试成本低）
                errors.append(f"{url}（第 {attempt + 1} 次）: {e}")
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"所有下载源均失败:\n" + "\n".join(errors))


def ensure_model(name: str, force: bool = False,
                 progress_callback: Optional[Callable[[int, int], bool]] = None) -> Path:
    """确保模型权重已下载且 SHA256 匹配；返回权重路径。"""
    info = get_model_info(name)
    ckpt = models_dir() / f"{name}.ckpt"

    if ckpt.exists() and not force:
        if sha256_of(ckpt) == info["sha256"]:
            print(f"模型已就绪: {ckpt.name}（{ckpt.stat().st_size / 1e6:.0f} MB）")
            return ckpt
        print(f"校验失败，重新下载: {ckpt.name}")
        ckpt.unlink()

    print(f"下载模型 {name}（{info['description']}）")
    urls = [info["ckpt_url"]] + list(info.get("mirror_urls", []))
    _download(urls, ckpt, progress_callback)
    actual = sha256_of(ckpt)
    if actual != info["sha256"]:
        ckpt.unlink()
        raise RuntimeError(
            f"SHA256 校验失败: 期望 {info['sha256']}，实际 {actual}。"
            f"下载源可能已变更，请检查 {info['ckpt_url']}"
        )
    print(f"完成: {ckpt}（{ckpt.stat().st_size / 1e6:.0f} MB）")
    return ckpt


def download_all() -> None:
    for name in MODEL_REGISTRY:
        ensure_model(name)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "all":
        download_all()
    else:
        for a in args:
            ensure_model(a)
