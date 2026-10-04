"""一键安装脚本的 Python 版本门槛（issue #6）。

pyproject.toml 声明 ``requires-python >= 3.10``，而 install.bat / install.sh 原先只
检查解释器是否存在：3.9 及更早会先建 .venv、装 torch，再在 pip 或运行时才失败——
用户等半天只看到一句无从下手的报错。门槛还必须是「比版本 → 建 venv」的顺序，
否则校验再准也拦不住已经建好的环境。

比较统一交给解释器自己（bat 里做数值比较会受区域设置影响，sh 不引入外部工具）。
"""

import re
from pathlib import Path

import pytest

import uvr_lite

REPO = Path(uvr_lite.__file__).resolve().parent.parent

# 门槛判据：解释器内的版本比较，3.10 是 pyproject 声明的下限
_GATE = re.compile(r"sys\.version_info\s*>=\s*\(\s*3\s*,\s*10\s*\)")
# 建 venv 的命令（bat 的 `python -m venv` 与 sh 的 `"$PY" -m venv` 都命中）
_VENV = re.compile(r"-m\s+venv")


@pytest.mark.parametrize("script", ["install.bat", "install.sh"])
def test_installer_gates_python_version_before_venv(script):
    """脚本必须在创建 .venv 之前比较 sys.version_info >= (3, 10)。"""
    text = (REPO / script).read_text(encoding="utf-8")
    gate = _GATE.search(text)
    assert gate, f"{script} 缺少 Python >= 3.10 的版本门槛"
    venv = _VENV.search(text)
    assert venv is None or gate.start() < venv.start(), (
        f"{script} 的版本门槛出现在建 venv 之后：低版本 Python 仍会先建环境才失败")


def test_pyproject_requires_python_matches_installer_gate():
    """前提守卫：门槛数字必须与 pyproject 的 requires-python 一致。

    改了下限却忘了改脚本时，这里先红，而不是等用户拿 3.11 去撞脚本里的 3.10。
    """
    import tomllib

    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.10", (
        "requires-python 变了：install.bat / install.sh 的版本门槛要同步修改")
