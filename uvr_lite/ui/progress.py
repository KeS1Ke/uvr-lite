"""推理接线的纯逻辑（无 Qt 依赖，可单测）。

- ProgressTracker: 阶段回调 → 文件内进度 0..1
- estimate_eta: 按历史平均耗时估算剩余时间
- summary_text: 队列结束汇总文案
"""



class ProgressTracker:
    """把引擎的阶段回调（decode/infer/chunk/tta/write）映射为文件内进度。

    权重分配：decode 5% → chunk/infer 45%（bigshifts 多 pass 均分，chunk
    为当前 pass 内部子进度）→ tta 40% → write 10%。
    """

    def __init__(self, bigshifts: int = 1):
        self.bigshifts = max(1, bigshifts)
        self._pass_done = 0

    def on_progress(self, phase: str, done: int, total: int) -> float:
        if phase == "decode":
            return 0.05 * (done / total if total else 0)
        if phase == "chunk":
            frac = done / total if total else 0
            return 0.05 + (self._pass_done + frac) / self.bigshifts * 0.45
        if phase == "infer":
            self._pass_done = done
            return 0.05 + (done / self.bigshifts) * 0.45
        if phase == "tta":
            return 0.50 + 0.40 * (done / total if total else 0)
        if phase == "write":
            return 0.90 + 0.10 * (done / total if total else 0)
        return 0.0


def estimate_eta(file_seconds: list[float], done: int, total: int, file_pct: float) -> float | None:
    """按已完成文件的平均耗时线性估算剩余秒数；无历史返回 None。"""
    if not file_seconds:
        return None
    avg = sum(file_seconds) / len(file_seconds)
    pct = max(0.0, min(float(file_pct), 1.0))
    remaining = (total - done - 1) + (1.0 - pct)
    return avg * remaining


def summary_text(ok: int, failed: list[str]) -> str:
    """队列结束汇总文案：全成功或 成功 N/失败 M + 失败清单。"""
    if not failed:
        return f"全部成功：{ok} 个文件。"
    names = "\n".join(f"  · {n}" for n in failed)
    return f"成功 {ok} 个，失败 {len(failed)} 个：\n{names}"
