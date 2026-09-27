"""音质档映射（无 Qt / torch 依赖，供界面与单测共用）。

标准档必须显式 num_overlap=2。若返回 None，karaoke 模型 yaml 里的
overlap=4 会在用户选了「标准」时偷偷生效，批量会变慢。
"""

PRESETS: dict[str, dict] = {
    "fast": {"num_overlap": 1, "bigshifts": 1, "tta": False, "label": "快速"},
    "standard": {"num_overlap": 2, "bigshifts": 1, "tta": False, "label": "标准"},
    "high": {"num_overlap": 2, "bigshifts": 2, "tta": False, "label": "高音质"},
}


def resolve_quality(preset: str, overlap: int, bigshifts: int, tta: bool) -> dict:
    """preset 为 fast/standard/high 时忽略后面三个，返回 num_overlap/bigshifts/tta。

    preset 为 custom 时用调用方传入的值（overlap<=0 则 num_overlap=None，
    表示沿用模型配置；bigshifts 至少 1）。未知 preset 按 standard。
    """
    spec = PRESETS.get(preset)
    if spec is None and preset != "custom":
        spec = PRESETS["standard"]
    if spec is not None:
        return {
            "num_overlap": spec["num_overlap"],
            "bigshifts": spec["bigshifts"],
            "tta": bool(spec["tta"]),
        }
    return {
        "num_overlap": _custom_overlap(overlap),
        "bigshifts": _custom_bigshifts(bigshifts),
        "tta": bool(tta),
    }


def matching_preset(overlap: int, bigshifts: int, tta: bool) -> str:
    """旧设置能否对上三档。tta 为真，或数值对不上，则视为 custom。"""
    if tta:
        return "custom"
    try:
        ov = int(overlap)
        bs = int(bigshifts)
    except (TypeError, ValueError):
        return "custom"
    for key, spec in PRESETS.items():
        if spec["num_overlap"] == ov and spec["bigshifts"] == bs and not spec["tta"]:
            return key
    return "custom"


def _custom_overlap(overlap: int):
    try:
        ov = int(overlap)
    except (TypeError, ValueError):
        return None
    return None if ov <= 0 else ov


def _custom_bigshifts(bigshifts: int) -> int:
    try:
        bs = int(bigshifts)
    except (TypeError, ValueError):
        return 1
    return bs if bs >= 1 else 1
