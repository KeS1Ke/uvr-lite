# coding: utf-8
"""命令行入口：uvr-lite separate / download / models / version。"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .download import ensure_model, models_dir
from .engine import separate_file
from .models import MODEL_REGISTRY, DEFAULT_MODEL


def _cmd_separate(args: argparse.Namespace) -> int:
    pcm = f"PCM_{args.pcm}"
    for inp in args.input:
        separate_file(
            inp, args.out,
            model_name=args.model, pcm=pcm, device=args.device,
            fmt=args.format, bigshifts=args.bigshifts, tta=args.tta,
            batch_size=args.batch_size,
        )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    if args.model == "all":
        for name in MODEL_REGISTRY:
            ensure_model(name)
    else:
        ensure_model(args.model, force=args.force)
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    print(f"{'名称':<20} {'类型':<18} 状态")
    print("-" * 64)
    for name, info in MODEL_REGISTRY.items():
        ckpt = models_dir() / f"{name}.ckpt"
        status = f"{ckpt.stat().st_size / 1e6:.0f} MB" if ckpt.exists() else "未下载"
        print(f"{name:<20} {info['model_type']:<18} {status}")
        print(f"  {info['description']}")
    print(f"\n默认模型: {DEFAULT_MODEL}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvr-lite",
        description="轻量级人声/伴奏分离工具（BS-RoFormer，模型与 UVR 同源）",
    )
    parser.add_argument("--version", action="version", version=f"uvr-lite {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_sep = sub.add_parser("separate", help="分离人声/伴奏（核心命令）")
    p_sep.add_argument("input", nargs="+", help="输入音频文件（支持多文件）")
    p_sep.add_argument("--out", "-o", default="output", help="输出目录（默认 ./output）")
    # 注意：separate 只接受具体模型名（all 仅 download 命令用）
    p_sep.add_argument("--model", "-m", default=DEFAULT_MODEL,
                       choices=list(MODEL_REGISTRY), help="模型（默认 bs_roformer_ep317）")
    p_sep.add_argument("--format", default="auto", choices=["auto", "flac", "wav"],
                       help="输出格式：auto 按峰值自动选择（默认）；flac/wav 强制")
    p_sep.add_argument("--pcm", type=int, default=24, choices=[16, 24], help="FLAC 位深（默认 24）")
    p_sep.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="推理设备（默认 auto 自动检测）")
    p_sep.add_argument("--bigshifts", type=int, default=1,
                       help="圆形时移平均次数（>1 提升质量但线性增耗时，默认 1）")
    p_sep.add_argument("--batch-size", type=int, default=None,
                       help="推理批大小（默认取模型配置；低显存 GPU 可设 1 防 OOM）")
    p_sep.add_argument("--tta", action="store_true",
                       help="测试时增强（极性/声道反转平均，三倍耗时，默认关）")
    p_sep.set_defaults(func=_cmd_separate)

    p_dl = sub.add_parser("download", help="下载模型权重（带 SHA256 校验）")
    p_dl.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                      choices=list(MODEL_REGISTRY) + ["all"],
                      help="模型名或 all（默认 bs_roformer_ep317）")
    p_dl.add_argument("--force", action="store_true", help="强制重新下载")
    p_dl.set_defaults(func=_cmd_download)

    p_ls = sub.add_parser("models", help="列出可用模型与下载状态")
    p_ls.set_defaults(func=_cmd_models)

    p_ui = sub.add_parser("ui", help="启动桌面界面（需 pip install -e '.[ui]'）")
    p_ui.set_defaults(func=_cmd_ui)

    return parser


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .ui.main import run
    except ImportError:
        print("[ERROR] 桌面界面依赖未安装，请先运行: pip install -e '.[ui]'")
        return 1
    return run()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
