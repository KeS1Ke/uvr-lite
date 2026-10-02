"""安装路径必须装上 UI extras（审计缺陷 3）。

GUI 依赖 PySide6-Essentials，只挂在 optional extra `ui` 上（pyproject.toml）。
install.sh / install.bat / 两份 README 的「手动安装」若只写 `pip install -e .`，
按文档装完 GUI 起不来——`uvr-lite ui` 直接 ImportError。修前四处全是裸 `-e .`。

这里把「文档与脚本的安装命令」当契约来测：凡是给用户的可执行安装命令，
必须带 extras 标记。用正则抓出 pip 调用并断言，避免逐字匹配整份脚本。
"""

import re
from pathlib import Path

import pytest

import uvr_lite

REPO = Path(uvr_lite.__file__).resolve().parent.parent

# 行内注释：只有 `#` 前是空白时才剥（URL 片段 `pkg#egg=x` 的 `#` 前无空白，不能误剥）。
_INLINE_COMMENT = re.compile(r"[ \t]+#.*$")

# 可编辑安装标志：`-e` 与 `--editable` 都要认。长选项写在前面，避免把 `--editable`
# 从中间劈成 `-e ditable`；`--extra-index-url` 里的 "-e" 后面不是空白，不会误匹配。
_EDITABLE_FLAG = r"(?:--editable|-e)"

# 匹配 `pip install ... [-e|--editable] <target>` 里的目标。行尾续行（shell 的 `\`）要
# 一并吃掉，否则 install.sh 的 `-e ".[ui]" \` 会被截成 ".`"，把带引号的 extras 判成不带。
_PIP_EDITABLE = re.compile(
    rf"pip\s+install\b[^\n]*?{_EDITABLE_FLAG}\s+(?P<target>[\"']?[^\s\"'\\\n]+)",
    re.IGNORECASE,
)

# target 方括号里的 extras 列表（`.[ui]` / `.[ui,dev]` / `.[ui;dev]`）
_EXTRAS = re.compile(r"\[([^\]]*)\]")


def _strip_inline_comment(line: str) -> str:
    """剥掉行内注释：仅当 `#` 前有空白时才算注释，保住 URL 片段里的 `#`。"""
    return _INLINE_COMMENT.sub("", line)


def _pip_install_targets(text: str) -> list[str]:
    return [m.group("target").strip("\"'") for m in _PIP_EDITABLE.finditer(text)]


def _extras_of(target: str) -> list[str]:
    """拆出 target 方括号内的 extras，按逗号/分号/空白分词（精确判定，非子串）。"""
    m = _EXTRAS.search(target)
    if m is None:
        return []
    return [part for part in re.split(r"[,;\s]+", m.group(1)) if part]


def _has_ui_extra(target: str) -> bool:
    """extras 分词里是否有精确的 `ui`（`.[gui]`/`.[uix]` 里的子串不算）。"""
    return "ui" in _extras_of(target)


def _without_comments(lines: list[str]) -> list[str]:
    """抹掉 shell 的 `#` 与 bat 的 `REM`/`::` 注释行，并剥掉行内注释。

    注释里也会出现 `pip install -e .`（说明为什么必须带 extras），不清掉会被
    当成真命令匹配到，断言就指向了注释而非真正的安装行。
    """
    out = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith(("#", "REM ", "rem ", "::")):
            out.append("")
            continue
        out.append(_strip_inline_comment(ln))
    return out


def _script_editable_installs(text: str) -> list[tuple[int, str, bool]]:
    """列出文本里每处可编辑安装：(行号, 目标, 是否主安装)。

    「主安装」= 脚本中第一处 `pip install -e`；其余都是失败兜底，必须带
    WARN 说明（否则用户看到的是一次静默降级）。修前主安装就是裸 `-e .`。
    """
    lines = _without_comments(text.split("\n"))
    out = []
    for i, line in enumerate(lines):
        m = _PIP_EDITABLE.search(line)
        if m is None:
            continue
        out.append((i + 1, m.group("target").strip("\"'"), not out))
    return out


def _readme_editable_installs(text: str) -> list[tuple[int, str]]:
    """README 里真正的 editable 安装行：(行号, 目标)。

    与脚本同源（`_PIP_EDITABLE`）：`--extra-index-url` 含 "-e" 子串但不是可编辑
    安装，旧的 `"-e" in ln` 子串判据会把这类行误当成安装命令。
    """
    return [(line, target) for line, target, _ in _script_editable_installs(text)]


def _has_warn_above(text_lines: list[str], idx: int) -> bool:
    """该行上方两行内是否已有 WARN 说明（有则确认是兜底而非静默降级）。"""
    return any("WARN" in ln for ln in text_lines[max(0, idx - 3):idx])


@pytest.mark.parametrize("script", ["install.bat", "install.sh"])
def test_one_click_installer_installs_ui_extra(script):
    """一键安装脚本的主安装命令必须带 [ui]，否则 GUI 缺 PySide6 依赖。"""
    text = (REPO / script).read_text(encoding="utf-8")
    installs = _script_editable_installs(text)
    assert installs, f"{script} 里应有一处 pip install -e"

    line, target, is_primary = installs[0]
    assert is_primary and _has_ui_extra(target), (
        f"{script}:{line} 主安装命令应为 `pip install -e \".[ui]\"`，实际 {target!r}")


@pytest.mark.parametrize("script", ["install.bat", "install.sh"])
def test_fallback_without_ui_extra_is_announced(script):
    """CLI-only 兜底（缺 Qt 系统库时）允许，但必须带 WARN，不能静默降级。"""
    lines = (REPO / script).read_text(encoding="utf-8").split("\n")
    clean = _without_comments(lines)
    for idx, target, is_primary in _script_editable_installs("\n".join(lines)):
        if is_primary or _has_ui_extra(target):
            continue
        assert _has_warn_above(clean, idx - 1), (
            f"{script}:{idx + 1} 有不带 ui extras 的安装兜底，上方须有 WARN 说明")


@pytest.mark.parametrize("readme", ["README.md", "README.zh-CN.md"])
def test_readme_manual_install_has_ui_extra(readme):
    """两份 README 的手动安装命令都要带 .[ui]。"""
    text = (REPO / readme).read_text(encoding="utf-8")
    installs = _readme_editable_installs(text)
    assert installs, f"{readme} 应给出 pip install -e 的手动安装命令"
    bad = [f"{readme}:{ln} {target!r}" for ln, target in installs if not _has_ui_extra(target)]
    assert not bad, f"{readme} 的手动安装命令漏了 .[ui]: {bad}"


def test_ui_extra_actually_declares_pyside():
    """前提守卫：`ui` extra 必须在 pyproject 里真的存在且带 PySide6。

    否则上面几条只是在断言「文档里写了个 extras 标记」，而该标记指向空 extra。
    """
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    ui_extra = data["project"].get("optional-dependencies", {}).get("ui", [])
    assert any("PySide6" in d for d in ui_extra), f"ui extra 未声明 PySide6: {ui_extra}"


def test_ui_error_hint_matches_documented_install_command():
    """CLI 报错里提示的安装命令要与文档一致（uvr_lite/cli.py 的 ui 子命令）。"""
    cli = (REPO / "uvr_lite" / "cli.py").read_text(encoding="utf-8")
    hinted = _pip_install_targets(cli)
    assert hinted, "cli.py 缺依赖时应提示 pip install 命令"
    assert all(_has_ui_extra(t) for t in hinted), f"提示命令应带 ui extras: {hinted}"


@pytest.mark.parametrize(
    ("command", "expect_ui"),
    [
        # 正常形态：短/长选项、引号、shell 续行都要认出 ui extra
        ('pip install -e ".[ui]"', True),
        ('pip install --editable ".[ui]"', True),
        ('pip install -e ".[ui]" \\', True),
        ("pip install --editable '.[ui,dev]'", True),
        # 行内注释不是命令的一部分：裸 `-e .` 不能靠注释里的 [ui] 蒙混过关
        ('pip install -e .  # [ui] = GUI extras (PySide6)', False),
        # 子串陷阱：gui/uix 里都含 "ui" 子串，但不是 ui extra
        ('pip install -e ".[gui]"', False),
        ('pip install -e ".[uix]"', False),
    ],
)
def test_editable_parser_requires_exact_ui_extra(command, expect_ui):
    """测试助手自测：先剥行内注释，再做 extras 分词精确判定。"""
    installs = _script_editable_installs(command)
    assert len(installs) == 1, f"应识别出一处 editable 安装: {command!r}"
    assert _has_ui_extra(installs[0][1]) is expect_ui


def test_editable_parser_ignores_non_editable_pip_lines():
    """测试助手自测：含 --extra-index-url 的 pip 行不是 editable 安装。"""
    text = (
        "pip install --extra-index-url https://example.com/simple requests\n"
        'pip install -e ".[ui]"'
    )
    installs = _readme_editable_installs(text)
    assert [target for _, target in installs] == [".[ui]"]


def test_readme_parser_rejects_ui_hidden_in_comment():
    """测试助手自测：README 行内注释里的 [ui] 不能给裸 `-e .` 背书。"""
    text = 'pip install -e .  # [ui] = GUI extras (PySide6); drop for CLI-only'
    installs = _readme_editable_installs(text)
    assert len(installs) == 1
    assert not _has_ui_extra(installs[0][1])


def test_strip_inline_comment_keeps_url_fragment():
    """测试助手自测：URL 片段里的 `#`（前面无空白）不当作注释剥掉。"""
    line = "pip install -e 'pkg @ https://example.com/pkg#egg=pkg'"
    assert _strip_inline_comment(line) == line
    stripped = _strip_inline_comment('pip install -e ".[ui]"  # [ui] = GUI extras')
    assert stripped.rstrip() == 'pip install -e ".[ui]"'


def test_comment_lines_are_ignored():
    """测试助手自测：shell `#` 与 bat `REM`/`::` 注释行里的 pip 命令不参与判定。"""
    text = '\n'.join([
        '# pip install -e ".[ui]"',
        'REM pip install -e ".[ui]"',
        ':: pip install -e ".[ui]"',
    ])
    assert _script_editable_installs(text) == []
