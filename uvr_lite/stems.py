"""音轨配对：按文件名识别分离产物（人声 / 伴奏）并成对。

CLI 批量合成（uvr-lite mix --batch）与 UI「合成」页共用；纯 pathlib，
不依赖 torch/Qt。命名契约与 engine 输出对齐：`{stem}-vocals` /
`{stem}-instrumental`，另兼容 `-inst` 缩写（第三方工具常见）。
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .audio_io import scan_audio_files

# 按后缀长度从长到短匹配，避免 "-inst" 截了 "-instrumental" 的前缀；
# 小写比较，保留原文件名大小写用于展示。
VOCALS_SUFFIXES = ("-vocals", "_vocals", " vocals")
INSTRUMENTAL_SUFFIXES = (
    "-instrumental", "_instrumental", "-inst", "_inst", " instrumental",
)


@dataclass
class StemPair:
    """一组可合成的人声 + 伴奏。"""

    vocals: Path
    instrumental: Path


@dataclass
class PairResult:
    """配对结果：成对音轨、缺另一半的音轨、命名不属于分离产物的文件。"""

    pairs: list[StemPair] = field(default_factory=list)
    unmatched: list[Path] = field(default_factory=list)
    ignored: list[Path] = field(default_factory=list)


def split_stem(path: Path) -> tuple[str, str] | None:
    """拆出 (base, kind)；kind ∈ {"vocals", "instrumental"}，不匹配返回 None。

    只看文件名主干（不含扩展名），因此 `song-vocals.flac` 与
    `song-instrumental.wav` 扩展名不同也能配对。
    """
    name = Path(path).stem
    low = name.lower()
    for kind, suffixes in (
        ("instrumental", INSTRUMENTAL_SUFFIXES),
        ("vocals", VOCALS_SUFFIXES),
    ):
        for suf in suffixes:
            if low.endswith(suf) and len(low) > len(suf):
                return name[: len(name) - len(suf)], kind
    return None


def pair_stems(paths: Iterable[Path]) -> PairResult:
    """按 (父目录, 基名小写) 配对，按名称排序；同基名多候选时取名称靠前者。

    同一基名出现多个同类文件（如 flac/wav 副本）时，只配最先的一对，
    其余进 unmatched——宁可让用户手动处理，也不静默挑错文件。
    """
    groups: dict[tuple[str, str], dict[str, list[Path]]] = {}
    result = PairResult()
    for raw in paths:
        p = Path(raw)
        split = split_stem(p)
        if split is None:
            result.ignored.append(p)
            continue
        base, kind = split
        key = (str(p.parent), base.lower())
        groups.setdefault(key, {}).setdefault(kind, []).append(p)

    for kinds in groups.values():
        vocals = sorted(kinds.get("vocals", []), key=lambda p: p.name.lower())
        inst = sorted(kinds.get("instrumental", []), key=lambda p: p.name.lower())
        for i in range(min(len(vocals), len(inst))):
            result.pairs.append(StemPair(vocals=vocals[i], instrumental=inst[i]))
        result.unmatched.extend(vocals[len(inst):])
        result.unmatched.extend(inst[len(vocals):])

    def by_name(p: Path) -> tuple[str, str]:
        return (str(p.parent).lower(), p.name.lower())

    result.pairs.sort(key=lambda pr: by_name(pr.vocals))
    result.unmatched.sort(key=by_name)
    result.ignored.sort(key=by_name)
    return result


def find_stem_pairs(folder: Path) -> PairResult:
    """扫描文件夹顶层音频并配对（非递归，与 UI 选择文件夹同一条路）。"""
    return pair_stems(scan_audio_files(folder))
