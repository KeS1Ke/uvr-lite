"""命令行入口：uvr-lite separate / mix / download / models / version。"""

import argparse
import sys
from pathlib import Path

from . import __version__
from .download import ensure_model, model_file, retired_model_files
from .log import log_exception, log_hint
from .models import DEFAULT_MODEL, MODEL_REGISTRY


def _cmd_separate(args: argparse.Namespace) -> int:
    pcm = f"PCM_{args.pcm}"
    # 全量安装包含 CPU/CUDA 两套 torch：--device 指定二进制，必须在
    # engine（import torch）之前切换（惰性 import）
    if args.device in ("cuda", "cpu"):
        from . import set_torch_mode

        set_torch_mode(args.device)
    from .engine import Separator

    # 会话复用：模型只加载一次，全部输入文件共用（避免每文件重载权重）
    sep = Separator(model_name=args.model, device=args.device,
                    batch_size=args.batch_size, num_overlap=args.num_overlap)
    for inp in args.input:
        sep.separate(
            inp, args.out, pcm=pcm,
            fmt=args.format, bigshifts=args.bigshifts, tta=args.tta,
        )
    return 0


def _cmd_mix(args: argparse.Namespace) -> int:
    """人声＋伴奏合成整曲：单对或 --batch 目录自动配对批量。"""
    from .mix import combine
    from .stems import find_stem_pairs, split_stem

    def _one(vocals: str, instrumental: str) -> None:
        # combine 的 verbose 输出已含「写出: 路径」，这里不再重复打印
        combine(
            vocals, instrumental, args.out,
            vocal_gain=args.vocal_gain, inst_gain=args.inst_gain,
            pcm=f"PCM_{args.pcm}", fmt=args.format, normalize=args.normalize,
        )

    if args.batch:
        result = find_stem_pairs(Path(args.batch))
        for p in result.unmatched:
            kind = split_stem(p)
            missing = "伴奏" if kind and kind[1] == "vocals" else "人声"
            print(f"未配对（缺{missing}）: {p.name}")
        for p in result.ignored:
            print(f"跳过（非分离音轨命名）: {p.name}")
        if not result.pairs:
            print("[ERROR] 没有可配对的音轨"
                  "（目录下需要 *-vocals 与 *-instrumental 成对存在）。")
            return 1
        for pair in result.pairs:
            _one(str(pair.vocals), str(pair.instrumental))
        return 0

    if not args.vocals or not args.instrumental:
        print("[ERROR] 请同时给出人声与伴奏文件，或使用 --batch DIR 批量配对。")
        return 1
    _one(args.vocals, args.instrumental)
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    if args.model == "all":
        for name in MODEL_REGISTRY:
            ensure_model(name)
    else:
        ensure_model(args.model, force=args.force)
    return 0


def _cmd_install_cuda(args: argparse.Namespace) -> int:
    """下载并安装 CUDA 推理引擎（终端进度条，断点续传 + 镜像回退）。"""
    from tqdm.auto import tqdm

    from .download import install_cuda_torch

    bar = {}

    def cb(done: int, total: int) -> bool:
        if "bar" not in bar:
            bar["bar"] = tqdm(total=total, unit="B", unit_scale=True,
                              desc="CUDA 引擎", miniters=1)
        bar["bar"].update(done - bar["bar"].n)
        return True

    try:
        install_cuda_torch(progress_callback=cb)
    finally:
        if "bar" in bar:
            bar["bar"].close()
    print("设备选择 cuda（或 auto）即可使用 GPU 加速。")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    print(f"{'名称':<20} {'类型':<18} 状态")
    print("-" * 64)
    for name, info in MODEL_REGISTRY.items():
        path = model_file(name)
        status = f"{path.stat().st_size / 1e6:.0f} MB" if path.exists() else "未下载"
        print(f"{name:<20} {info['model_type']:<18} {status}")
        print(f"  {info['description']}")
        # 退役权重（升级换格式后的旧文件）不会再被任何代码路径碰到，只有
        # ensure_model 在新权重就绪后回收它；这里把「还占着盘」的列出来，
        # 用户才知道重跑一次 download 能省多少空间（不存在则静默跳过）
        for old in retired_model_files(name):
            if old.exists():
                print(f"  可回收: {old.name}（{old.stat().st_size / 1e6:.0f} MB）"
                      "——新权重校验通过后自动删除")
    print(f"\n默认模型: {DEFAULT_MODEL}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uvr-lite",
        description="轻量级人声/伴奏分离与合成工具（BS-RoFormer，模型与 UVR 同源）",
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
    p_sep.add_argument("--num-overlap", type=int, default=None,
                       help="重叠窗口数（质量/速度开关：1 最快约 2x；默认取模型配置"
                            "（主力 2 / karaoke 4），显式传值才覆盖）")
    p_sep.add_argument("--tta", action="store_true",
                       help="测试时增强（极性/声道反转平均，三倍耗时，默认关）")
    p_sep.set_defaults(func=_cmd_separate)

    p_mix = sub.add_parser("mix", help="人声＋伴奏合成整曲（分离的逆运算）")
    p_mix.add_argument("vocals", nargs="?", help="人声轨（分离输出的 *-vocals）")
    p_mix.add_argument("instrumental", nargs="?",
                       help="伴奏轨（分离输出的 *-instrumental）")
    p_mix.add_argument("--batch", metavar="DIR",
                       help="扫描文件夹，按 *-vocals / *-instrumental 自动配对批量合成")
    p_mix.add_argument("--out", "-o", default="output", help="输出目录（默认 ./output）")
    p_mix.add_argument("--format", default="auto", choices=["auto", "flac", "wav"],
                       help="输出格式：auto 按峰值自动选择（默认）；flac/wav 强制")
    p_mix.add_argument("--pcm", type=int, default=24, choices=[16, 24],
                       help="FLAC 位深（默认 24）")
    p_mix.add_argument("--vocal-gain", type=float, default=1.0,
                       help="人声音量倍率（默认 1.0）")
    p_mix.add_argument("--inst-gain", type=float, default=1.0,
                       help="伴奏音量倍率（默认 1.0）")
    p_mix.add_argument("--normalize", action="store_true",
                       help="峰值归一到 -1 dBFS（防止叠加爆音；默认保持精确求和）")
    p_mix.set_defaults(func=_cmd_mix)

    p_dl = sub.add_parser("download", help="下载模型权重（带 SHA256 校验）")
    p_dl.add_argument("model", nargs="?", default=DEFAULT_MODEL,
                      choices=[*MODEL_REGISTRY, "all"],
                      help="模型名或 all（默认 bs_roformer_ep317）")
    p_dl.add_argument("--force", action="store_true", help="强制重新下载")
    p_dl.set_defaults(func=_cmd_download)

    p_ls = sub.add_parser("models", help="列出可用模型与下载状态")
    p_ls.set_defaults(func=_cmd_models)

    p_cuda = sub.add_parser("install-cuda",
                            help="下载并安装 CUDA 推理引擎（GPU 加速，约 3.3 GB，断点续传）")
    p_cuda.set_defaults(func=_cmd_install_cuda)

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
    try:
        return args.func(args)
    except Exception as e:
        # 命令层兜底：traceback 对非专业用户没有意义，落盘后给一句中文 +
        # 日志路径；成功路径的 stdout 与退出码完全不变（失败仍是退出码 1）
        log_exception(f"命令执行失败: {args.command}")
        # 日志路径走 log_hint() 而非 log_path()：异常目录布局下 repo_root 会抛
        # RuntimeError，兜底处理器不该在报错时再崩一次；日志不可用时它返回空串
        print(f"[ERROR] {args.command} 执行失败：{e}{log_hint()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
