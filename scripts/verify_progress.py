# coding: utf-8
"""票 1 验收：真实模型分离验证进度回调实况。

用法: python scripts/verify_progress.py [--seconds N] [--cancel-at PHASE]
--cancel-at 可选: decode / infer / chunk / write（验证取消清理）；默认不取消。
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uvr_lite.engine import CancelledError, separate_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=15)
    ap.add_argument("--cancel-at", default=None,
                    help="在该阶段触发取消（decode/infer/chunk/write）")
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    sr = 44100
    t = np.linspace(0, args.seconds, args.seconds * sr, endpoint=False)
    melody = 0.4 * np.sin(2 * np.pi * 440 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    perc = 0.15 * np.random.RandomState(42).randn(len(t)) * (t % 1.0 < 0.02)
    bass = 0.2 * np.sin(2 * np.pi * 110 * t)
    mix = np.stack([melody + perc + bass, melody * 0.9 + perc + bass], axis=1)
    wav = Path("_verify_input.wav")
    sf.write(wav, mix, sr)

    out = Path("_verify_out")
    t0 = time.time()
    last = {"phase": None, "t": time.time()}

    def cb(phase, done, total):
        now = time.time()
        if phase != last["phase"]:
            print(f"[{now - t0:6.2f}s] 阶段 {phase:6s} 开始 (共 {total})")
            last["phase"] = phase
            last["t"] = now
        if phase in ("infer", "chunk", "tta") and (done % max(1, total // 5) == 0 or done == total):
            print(f"         {phase:6s} {done:4d}/{total}  ({now - last['t']:5.2f}s 增量)")
            last["t"] = now
        return not (args.cancel_at == phase and done == total)

    try:
        written = separate_file(str(wav), str(out), progress_callback=cb, verbose=False)
        print(f"\n✅ 完成: {len(written)} 个文件, 总耗时 {time.time() - t0:.1f}s")
        for p in written:
            print(f"   {p}  ({Path(p).stat().st_size / 1e6:.1f} MB)")
    except CancelledError:
        remain = list(out.glob("*")) if out.exists() else []
        print(f"\n⛔ 已取消（{args.cancel_at} 阶段），半成品清理: {'无残留 ✓' if not remain else f'残留: {remain}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
