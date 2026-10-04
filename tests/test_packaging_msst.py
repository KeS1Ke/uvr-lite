"""wheel 必须带上 msst：site-packages 里与 uvr_lite 平级。

msst/、msst/utils、msst/models 没有 __init__.py，发现逻辑要开 namespace。
不在这里 pip install，也不覆盖 .artifacts/wheel 里的旧包。
"""

from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_finds_uvr_lite_and_msst_namespaces():
    with (ROOT / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    find = cfg["tool"]["setuptools"]["packages"]["find"]
    assert "uvr_lite*" in find["include"]
    assert "msst*" in find["include"]
    assert find["namespaces"] is True
    data = cfg["tool"]["setuptools"]["package-data"]
    assert data["uvr_lite"] == ["configs/*.yaml"]
    assert data["uvr_lite.ui"] == ["resources/*"]


def test_namespace_discovery_includes_msst_sources():
    from setuptools.config import expand

    with (ROOT / "pyproject.toml").open("rb") as f:
        find = tomllib.load(f)["tool"]["setuptools"]["packages"]["find"]
    found = set(expand.find_packages(root_dir=ROOT, **find))
    assert {"uvr_lite", "uvr_lite.ui", "msst", "msst.utils", "msst.models.bs_roformer"} <= found
    assert "tests" not in found
    assert "installer" not in found
    assert (ROOT / "msst" / "utils" / "model_utils.py").is_file()
    assert (ROOT / "msst" / "models" / "bs_roformer" / "bs_roformer.py").is_file()
    # 这两个目录故意没有 __init__.py，打包不能靠补文件改变导入风格
    assert not (ROOT / "msst" / "__init__.py").exists()
    assert not (ROOT / "msst" / "utils" / "__init__.py").exists()
