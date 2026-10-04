"""分离质量档。

档位只改重叠窗口数，不换权重。BS-RoFormer ep317 已是本应用的人声主力，
公开 SDR 约 12.97 dB。karaoke 模型做的是主唱/和声，不是更干净的人声分离。

听起来不干净，多半是块与块的接缝，不是模型选错了。重叠越大，每个采样被
更多窗口平均，接缝和串音感越轻，耗时近似按重叠倍数增加。

- fast：重叠 1。无重叠窗修好之后约比标准快一倍，适合 CPU。
- standard：重叠 2。与模型配置默认一致。
- high：重叠 4。更干净，耗时约为标准的两倍。

bigshifts 和 TTA 不放进档位。它们另外按线性或约 3 倍计时，收益通常小于
把重叠从 2 提到 4。
"""

from __future__ import annotations

QUALITY_FAST = "fast"
QUALITY_STANDARD = "standard"
QUALITY_HIGH = "high"
QUALITY_CHOICES: tuple[str, ...] = (QUALITY_FAST, QUALITY_STANDARD, QUALITY_HIGH)

QUALITY_PRESETS: dict[str, dict[str, object]] = {
    QUALITY_FAST: {
        "num_overlap": 1,
        "label": "快",
        "menu": "快（约 2×，适合 CPU）",
        "hint": "重叠 1。约比标准快一倍，CPU 上优先用这档。块边界可能有轻微接缝。",
    },
    QUALITY_STANDARD: {
        "num_overlap": 2,
        "label": "标准",
        "menu": "标准（默认重叠）",
        "hint": "重叠 2，与模型配置一致。速度和接缝的折中。",
    },
    QUALITY_HIGH: {
        "num_overlap": 4,
        "label": "高",
        "menu": "高（重叠 4，更干净）",
        "hint": "重叠 4。接缝更干净、人声残留更少，耗时约为标准的两倍。",
    },
}


def quality_overlap(name: str) -> int:
    """质量档 → 重叠窗口数。未知名称抛 ValueError。"""
    key = (name or "").strip().lower()
    preset = QUALITY_PRESETS.get(key)
    if preset is None:
        known = ", ".join(QUALITY_CHOICES)
        raise ValueError(f"未知质量档: {name}（可用: {known}）")
    return int(preset["num_overlap"])


def quality_from_overlap(num_overlap: int | None) -> str:
    """旧设置里的重叠数映射回档位。0 / None / 2 → 标准；≤1 → 快；≥3 → 高。"""
    if num_overlap is None or num_overlap in (0, 2):
        return QUALITY_STANDARD
    if num_overlap <= 1:
        return QUALITY_FAST
    return QUALITY_HIGH
