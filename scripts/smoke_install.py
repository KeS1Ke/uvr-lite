# coding: utf-8
"""票 6 冒烟：命令行方式真实执行完整安装链（无 GUI）。

用法: python scripts/smoke_install.py [安装目录] [--upgrade]
信号（step/message/percent/finished）实时打印，供人工核对。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from installer.runner import StepRunner  # noqa: E402

install_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "uvr-lite"
upgrade = "--upgrade" in sys.argv
print(f"== 冒烟安装: {install_dir}（upgrade={upgrade}）==", flush=True)

r = StepRunner(install_dir, upgrade, src_dir=Path(__file__).resolve().parent.parent)
result = {"ok": False}
r.step.connect(lambda i, t, ti: print(f"\n>> 步骤 {i + 1}/{t}: {ti}", flush=True))
r.message.connect(lambda m: print(f"   {m}", flush=True))
r.percent.connect(lambda p: print(f"   [进度 {p}%]", flush=True))
r.finished.connect(lambda ok, err: (print(f"\n== 结果: {'成功' if ok else '失败'} | {err}", flush=True),
                                    result.update(ok=ok)))
r.run()
sys.exit(0 if result["ok"] else 1)
