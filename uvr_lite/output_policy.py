"""分离与合成共用的写出格式策略，不依赖模型或 torch。"""


def output_format(fmt: str, pcm: str, peak: float) -> tuple[str, str, str | None]:
    """返回 (codec, subtype, warning)，超峰时用浮点 WAV 保留采样值。"""
    if peak > 1.0:
        if fmt == "flac":
            warning = (
                f"峰值 {peak:.3f} 超过 0 dBFS，FLAC 无法保存超过 0 dBFS 的采样，"
                "整数 PCM 会削波，本次改为 32-bit float WAV"
            )
        else:
            warning = (
                f"峰值 {peak:.3f} 超过 0 dBFS，整数 PCM 会削波，本次改为 32-bit float WAV"
            )
        return "wav", "FLOAT", warning
    return ("flac" if fmt in {"flac", "auto"} else "wav"), pcm, None
