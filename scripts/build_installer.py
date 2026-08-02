# coding: utf-8
"""把安装向导打包为单 exe（PyInstaller onefile，无 torch，约 40-60MB）。

用法: python scripts/build_installer.py [--out dist]
产物: dist/uvr-lite-setup.exe
  - 携带代码快照（uvr_lite/ + msst/ + pyproject.toml，app-snapshot 目录）
  - 向导运行时代码快照从 _MEIPASS/app-snapshot 读取（installer/main.py 已支持）
  - ♪ 图标

前置: pip install pyinstaller（仅打包机需要，安装器产物不含 PyInstaller）
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_ICON = ROOT / "uvr_lite" / "ui" / "resources" / "uvr-lite.ico"


def prepare_snapshot(build_dir: Path) -> Path:
    """用 copy_app 的排除规则生成代码快照（不复制 .git/.venv/权重等）。"""
    from installer.copy_app import copy_app_source

    snapshot = build_dir / "snapshot"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    count = copy_app_source(ROOT, snapshot)
    print(f"[1/2] 代码快照就绪: {snapshot}（{count} 个文件）")
    return snapshot


def build(snapshot: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--onefile", "--windowed",
        "--name", "uvr-lite-setup",
        "--icon", str(_ICON),
        "--distpath", str(out_dir),
        "--workpath", str(out_dir / "_build"),
        "--specpath", str(out_dir / "_build"),
        # Windows 的 --add-data 分隔符是 ';'（os.pathsep）
        "--add-data", f"{snapshot}{os.pathsep}app-snapshot",
        str(ROOT / "installer" / "main.py"),
    ]
    print("[2/2] PyInstaller 打包中（首次较慢，约 1-3 分钟）…")
    subprocess.run(cmd, check=True, cwd=ROOT)
    exe = out_dir / "uvr-lite-setup.exe"
    if not exe.exists():
        raise SystemExit(f"打包失败：未找到产物 {exe}")
    print(f"完成: {exe}（{exe.stat().st_size / 1e6:.0f} MB）")
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 uvr-lite 安装向导为单 exe")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    build_dir = out_dir / "_build"
    snapshot = prepare_snapshot(build_dir)
    exe = build(snapshot, out_dir)
    # 清理中间目录（快照与 PyInstaller 工作区）
    shutil.rmtree(build_dir, ignore_errors=True)
    print(f"清理完成，产物保留: {exe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
