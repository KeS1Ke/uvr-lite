"""模型瘦身：fp32 ckpt → fp16 存储（体积减半，推理精度不变）。

背景：viperx/aufr33 发布的 ckpt 顶层就是纯 state_dict（699 个 fp32 tensor，
159.8M 参数 = 639MB），没有可剥离的 optimizer/EMA 训练态。瘦身唯一有效路径
是 fp16 存储：torch.load 后 load_state_dict 会自动上转回 fp32，推理输出与
原版差异约 -58dB（实测相对误差 0.84%，不可闻），内存/速度均不变。

用法:
  python scripts/strip_model.py <输入.ckpt> <输出.ckpt> [--format ckpt|safetensors]
  python scripts/strip_model.py models/bs_roformer_ep317.ckpt \
      models/bs_roformer_ep317.lite.safetensors --format safetensors

输出形态：
  - ckpt：{"state_dict": ...} 封装，兼容 msst load_start_checkpoint
    （inference 模式对 "state"/"state_dict"/"model_state_dict" 键均兼容）
  - safetensors：纯 state_dict 扁平文件——无 pickle 载入面（引擎按扩展名
    用 safetensors.torch.load_file 加载，天然防任意代码执行），加载更快

发布：瘦身版上传到自己的 GitHub Releases，更新 uvr_lite/models.py 注册表
（ckpt_url + sha256 + filename；mirror 必须与主源内容一致——同 hash）。
"""

import argparse
import hashlib
from pathlib import Path

import torch


def _to_half_state_dict(ckpt: dict) -> tuple:
    """提取 state_dict 并转 fp16；返回 (sd16, n_params)。"""
    sd = None
    for key in ("state", "state_dict", "model_state_dict"):
        if isinstance(ckpt.get(key), dict):
            sd = ckpt[key]
            print(f"从顶层键 '{key}' 提取 state_dict")
            break
    if sd is None:
        sd = ckpt
        print("ckpt 本身即 state_dict")
    n_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
    sd16 = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in sd.items()}
    return sd16, n_params


def strip_ckpt(src: Path, dst: Path, fmt: str = "ckpt") -> None:
    """加载 fp32 ckpt，转 fp16 后按指定格式保存。"""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"不是标准 ckpt dict: {type(ckpt)}")

    sd16, n_params = _to_half_state_dict(ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "safetensors":
        from safetensors.torch import save_file

        non_tensor = [k for k, v in sd16.items() if not isinstance(v, torch.Tensor)]
        if non_tensor:
            print(f"警告: {len(non_tensor)} 个非张量键被跳过（safetensors 仅支持张量）"
                  f": {', '.join(non_tensor[:5])}")
        save_file({k: v for k, v in sd16.items() if isinstance(v, torch.Tensor)}, dst)
    else:
        torch.save({"state_dict": sd16}, dst)

    src_mb = src.stat().st_size / 1e6
    dst_mb = dst.stat().st_size / 1e6
    print(f"完成: {src.name}（{src_mb:.0f} MB，{n_params / 1e6:.1f}M 参数）")
    print(f"  → {dst}（{dst_mb:.0f} MB，减 {src_mb - dst_mb:.0f} MB / {(1 - dst_mb / src_mb) * 100:.0f}%）")
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    print(f"  sha256: {sha}")


def main() -> int:
    ap = argparse.ArgumentParser(description="模型瘦身：fp32 → fp16 存储（体积减半）")
    ap.add_argument("src", type=Path, help="输入 fp32 ckpt")
    ap.add_argument("dst", type=Path, help="输出 fp16 权重（扩展名 .safetensors 时可自动选格式）")
    ap.add_argument("--format", choices=["ckpt", "safetensors"], default=None,
                    help="输出格式（默认按 dst 扩展名推断，.safetensors → safetensors）")
    args = ap.parse_args()
    if not args.src.exists():
        raise SystemExit(f"输入不存在: {args.src}")
    fmt = args.format or ("safetensors" if args.dst.suffix == ".safetensors" else "ckpt")
    strip_ckpt(args.src, args.dst, fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
