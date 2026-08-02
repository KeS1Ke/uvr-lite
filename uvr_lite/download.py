# coding: utf-8
"""模型下载：从官方源拉取权重到 models/ 目录，带 SHA256 完整性校验。

权重文件大（639MB+），不入 git；安装脚本与首次分离前自动调用本模块。
"""

import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

from tqdm.auto import tqdm

from .models import MODEL_REGISTRY, get_model_info


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


def _download(url: str, dest: Path) -> None:
    """流式下载到临时文件，带进度条；完成后原子改名。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "uvr-lite/0.1"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"下载 {dest.name}", miniters=1
        ) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def ensure_model(name: str, force: bool = False) -> Path:
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
    _download(info["ckpt_url"], ckpt)
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
