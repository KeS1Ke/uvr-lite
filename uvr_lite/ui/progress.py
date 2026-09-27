"""推理接线的纯逻辑（无 Qt 依赖，可单测）。

- ProgressTracker: 阶段回调 → 文件内进度 0..1
- estimate_eta: 按历史平均耗时（或当前文件速度）估算剩余时间
- summary_text: 队列结束汇总文案
"""

import math


class ProgressTracker:
    """把引擎的阶段回调（decode/infer/chunk/tta/write）映射为文件内进度。

    默认不开 TTA：decode 5% → chunk/infer 90%（bigshifts 多 pass 均分，chunk
    为当前 pass 内部子进度）→ write 5%。推理结束 0.95，写出结束 1.0。

    tta=True 时保持旧权重：decode 5% / infer 45% / tta 40% / write 10%。
    """

    def __init__(self, bigshifts: int = 1, tta: bool = False):
        self.bigshifts = max(1, bigshifts)
        self.tta = bool(tta)
        self._pass_done = 0
        if self.tta:
            self._w_decode = 0.05
            self._w_infer = 0.45
            self._w_tta = 0.40
            self._w_write = 0.10
        else:
            self._w_decode = 0.05
            self._w_infer = 0.90
            self._w_tta = 0.0
            self._w_write = 0.05

    def on_progress(self, phase: str, done: int, total: int) -> float:
        frac = done / total if total else 0
        if phase == "decode":
            return self._w_decode * frac
        if phase == "chunk":
            return self._w_decode + (self._pass_done + frac) / self.bigshifts * self._w_infer
        if phase == "infer":
            self._pass_done = done
            return self._w_decode + (done / self.bigshifts) * self._w_infer
        if phase == "tta":
            return (self._w_decode + self._w_infer) + self._w_tta * frac
        if phase == "write":
            return (self._w_decode + self._w_infer + self._w_tta) + self._w_write * frac
        return 0.0


def estimate_eta(
    file_seconds: list[float],
    done: int,
    total: int,
    file_pct: float,
    elapsed_current: float | None = None,
) -> float | None:
    """估算剩余秒数。

    有历史耗时：平均耗时 × 剩余文件数（剩余 = (total-done-1) + (1-pct)，
    pct clamp 到 0..1，最小 0）。此时忽略 elapsed_current。

    无历史：没给 elapsed_current，或 file_pct < 0.10，返回 None（太早期
    的样本误差大；0.05 恰是 decode 终点，会把解码/加载耗时外推成整首）。
    否则按当前文件速度外推：rate = elapsed_current / pct，返回 rate × 剩余。
    """
    pct_raw = float(file_pct)
    pct = max(0.0, min(pct_raw, 1.0))
    remaining = max(0.0, (total - done - 1) + (1.0 - pct))
    if file_seconds:
        avg = sum(file_seconds) / len(file_seconds)
        return avg * remaining
    if (
        elapsed_current is None
        or pct_raw < 0.10
        or isinstance(elapsed_current, bool)
        or not isinstance(elapsed_current, (int, float))
        or not math.isfinite(elapsed_current)
        or elapsed_current < 0
    ):
        return None
    if pct <= 0:
        return None
    return (float(elapsed_current) / pct) * remaining


def summary_text(ok: int, failed: list[str]) -> str:
    """队列结束汇总文案：全成功或 成功 N/失败 M + 失败清单。"""
    if not failed:
        return f"全部成功：{ok} 个文件。"
    names = "\n".join(f"  · {n}" for n in failed)
    return f"成功 {ok} 个，失败 {len(failed)} 个：\n{names}"
