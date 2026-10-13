"""音轨配对：按文件名识别分离产物（人声 / 伴奏）并成对。

CLI 批量合成（uvr-lite mix --batch）与 UI「合成」页共用；纯 pathlib，
不依赖 torch/Qt。命名契约与 engine 输出对齐：`{stem}-vocals` /
`{stem}-instrumental`，另兼容 `-inst` 缩写（第三方工具常见）。
"""

import re
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
    # 旧版把编号放在 kind 后，新版放在 kind 前；统一配对基名为 song-2。
    legacy = re.search(r"-([2-9]|[1-9][0-9]+)$", name)
    number = ""
    if legacy is not None:
        number = legacy.group(0)
        name = name[:legacy.start()]
    low = name.lower()
    for kind, suffixes in (
        ("instrumental", INSTRUMENTAL_SUFFIXES),
        ("vocals", VOCALS_SUFFIXES),
    ):
        for suf in suffixes:
            if low.endswith(suf) and len(low) > len(suf):
                return name[: len(name) - len(suf)] + number, kind
    return None


def separation_output_paths(out_dir: Path, stem: str,
                            outputs: list[tuple[str, str]]) -> list[Path]:
    """整对选择共同基名，任一格式/旧版编号的残留都占用该次分离编号。"""
    occupied = set()
    for path in out_dir.iterdir():
        split = split_stem(path)
        if split is not None:
            occupied.add(split[0].lower())
    index = 1
    while True:
        base = stem if index == 1 else f"{stem}-{index}"
        paths = [out_dir / f"{base}-{kind}.{codec}" for kind, codec in outputs]
        if base.lower() not in occupied and not any(path.exists() for path in paths):
            return paths
        index += 1


def pair_stems(paths: Iterable[Path]) -> PairResult:
    """按 (resolve 后的父目录, 基名小写) 配对，按名称排序；同基名多候选时取名称靠前者。

    同一基名出现多个同类文件（如 flac/wav 副本）时，只配最先的一对，
    其余进 unmatched——宁可让用户手动处理，也不静默挑错文件。
    相对路径与绝对路径指向同一目录时分在一组；已经 resolve 过的路径结果不变。
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
        key = (str(p.resolve().parent), base.lower())
        groups.setdefault(key, {}).setdefault(kind, []).append(p)

    for kinds in groups.values():
        vocals = sorted(kinds.get("vocals", []), key=lambda p: p.name.lower())
        inst = sorted(kinds.get("instrumental", []), key=lambda p: p.name.lower())
        if vocals and inst:
            result.pairs.append(StemPair(vocals=vocals[0], instrumental=inst[0]))
            result.unmatched.extend(vocals[1:])
            result.unmatched.extend(inst[1:])
        else:
            result.unmatched.extend(vocals)
            result.unmatched.extend(inst)

    def by_name(p: Path) -> tuple[str, str]:
        return (str(p.parent).lower(), p.name.lower())

    result.pairs.sort(key=lambda pr: by_name(pr.vocals))
    result.unmatched.sort(key=by_name)
    result.ignored.sort(key=by_name)
    return result


def find_stem_pairs(folder: Path) -> PairResult:
    """扫描文件夹顶层音频并配对（非递归，与 UI 选择文件夹同一条路）。"""
    return pair_stems(scan_audio_files(folder))
