"""模型瘦身：fp32 ckpt → fp16 存储（存储体积减半、无 pickle 载入面）。

背景：viperx/aufr33 发布的 ckpt 顶层就是纯 state_dict（699 个 tensor，
159.8M 参数；上游原版为 fp32，体积按上游记为 639MB——本仓库未独立核实，
见下），没有可剥离的 optimizer/EMA 训练态。瘦身唯一有效路径是 fp16 存储：
torch.load 后 load_state_dict 会自动上转回 fp32，内存/速度均不变。

精度口径（2026-09 订正；本仓库未做 fp32→fp16 端到端实测，故不再给出无口径的 dB）：
  - 权重级（可从定义推导）：fp16 规格化数单位舍入步长 2^-11 ≈ 0.049%
    （-66.2 dB）；单个权重的相对误差不超过它，整网误差按累加方式另计。
  - 输出级（音频 A/B 差异 dB）：需要 fp32 原版权重 + 同输入对比推理才能测。
    该原版不在本机——models/bs_roformer_ep317.ckpt 已是上一轮瘦身产物
    （699 个 tensor 全部 float16），且与分发的 .lite.safetensors 逐张量
    bitwise 相同；拿它再跑一次本脚本只会得到同内容副本（误差恒为 0，属
    假数据），因此本仓库不提供任何输出级 dB 数字。历史上此处曾写有一个未标
    口径的 dB 值与一个未标口径的相对误差百分比，两者互相矛盾，已随本次订正
    删除（历史数字与推导见 docs/adr/ADR-001-ui-wrapper.md 的 2026-09 T11 注记）。

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
