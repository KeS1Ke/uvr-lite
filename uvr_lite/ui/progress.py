"""推理接线的纯逻辑（无 Qt 依赖，可单测）。

- PHASE_CN: 阶段中文名（状态栏与 ProgressTracker 共用的阶段名单）
- ProgressTracker: 阶段回调 → 文件内进度 0..1
- estimate_eta: 按历史平均耗时估算剩余时间
- summary_text: 队列结束汇总文案
"""

# 阶段中文名：单一来源——ui/main.py 从这里导入，不再自建第二张表（两份词表漂移时
# 状态栏会显示未翻译的阶段名）。键集合必须与 ProgressTracker.on_progress 认识的阶段
# 一致：漂移会让进度条停在该阶段起点，tests/test_ui_files.py 有按键遍历的行为守卫。
PHASE_CN = {
    "decode": "解码", "infer": "推理", "chunk": "推理",
    "tta": "增强", "mix": "合成", "write": "写出",
}


class ProgressTracker:
    """把引擎的阶段回调（decode/infer/chunk/tta/mix/write）映射为文件内进度。

    分离权重：decode 5% → chunk/infer 45%（bigshifts 多 pass 均分，chunk
    为当前 pass 内部子进度）→ tta 40% → write 10%。
    合成权重：decode 5%（两文件各半步）→ mix 55%→90%（占 35%）→ write 10%；
    分离路径不会收到 mix 回调，合成路径不会收到 infer/chunk/tta，共用一张表
    不互相干扰。
    """

    def __init__(self, bigshifts: int = 1):
        self.bigshifts = max(1, bigshifts)
        self._pass_done = 0

    def on_progress(self, phase: str, done: int, total: int) -> float:
        # 分支里的阶段名即 PHASE_CN 的键（不引用常量，保持本模块纯逻辑、不因词表变动而变行为）
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
        if phase == "mix":
            return 0.55 + 0.35 * (done / total if total else 0)
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
