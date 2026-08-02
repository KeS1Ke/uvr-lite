# coding: utf-8
"""模型下载：从主源拉取权重到 models/ 目录，带 SHA256 完整性校验。

- 断点续传：`.part` 文件已存在时用 HTTP Range 头续传，避免中断后全量重下
- 多源回退：主源（GitHub Releases）失败时自动切换 HuggingFace 镜像

权重文件大（639MB+），不入 git；安装脚本与首次分离前自动调用本模块。
"""

import hashlib
import os
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
    d = repo_root() / "models"
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


def _download_one(url: str, tmp: Path,
                  progress_callback: Optional[Callable[[int, int], bool]] = None) -> None:
    """从单个源下载（支持 Range 断点续传）；失败抛异常由上层切换源。

    progress_callback(done_bytes, total_bytes) -> bool：返回 False 视为用户
    取消，抛 InterruptedError（保留 .part 供下次续传）。
    """
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


def _download(urls: List[str], dest: Path,
              progress_callback: Optional[Callable[[int, int], bool]] = None) -> None:
    """按顺序尝试各下载源，全部失败才报错；完成后原子改名。

    用户取消（InterruptedError）不切换源、保留 .part 供续传。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    for url in urls:
        try:
            _download_one(url, tmp, progress_callback)
            tmp.replace(dest)
            return
        except InterruptedError:
            raise
        except Exception as e:  # noqa: BLE001 —— 切换下一源
            errors.append(f"{url}: {e}")
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
