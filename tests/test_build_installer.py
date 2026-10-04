"""打包脚本的可单测部分：CPU torch wheel 的 SHA 校验、URL 编码与打包链一致性校验。

此前仅 CUDA wheel 有 SHA（install.iss 与 download.py 双向一致检查），CPU
wheel 下载后无校验直接 pip 安装——镜像被投毒时无防线。
一致性校验（CUDA 资产四元组 / Pascal 裁剪清单 / 装机依赖）此前只有一次性手跑
harness，用完即弃；这里固化成用例，改 install.iss 或 pyproject.toml 时会先红。
"""

import hashlib
import io
import re
import shutil
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # scripts/ 非包，从根目录按命名空间包导入

import scripts.build_installer as bi  # noqa: E402
from uvr_lite import download as dl  # noqa: E402


@pytest.fixture
def fake_wheel_env(tmp_path, monkeypatch):
    """假 wheel 落盘 + 假 pip：聚焦 _install_torch 的校验与 URL 逻辑。"""
    content = b"fake torch cpu wheel " * 100
    seen = {}

    def fake_download(urls, dest, *a, **k):
        seen["urls"] = urls
        dest.write_bytes(content)

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(bi, "_pip", lambda *a, **k: None)
    return {"content": content, "sha": hashlib.sha256(content).hexdigest(),
            "dir": tmp_path, "seen": seen}


def test_install_torch_rejects_bad_sha(fake_wheel_env, monkeypatch):
    """SHA 不匹配 → 删除 wheel 并退出（投毒文件不得滞留进入 pip）。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", "0" * 64)
    with pytest.raises(SystemExit, match="校验失败"):
        bi._install_torch(fake_wheel_env["dir"], ["https://x/"])
    wheel = fake_wheel_env["dir"] / "torch-2.7.1+cpu-cp312-cp312-win_amd64.whl"
    assert not wheel.exists(), "校验失败后应删除 wheel"


def test_install_torch_sha_match_proceeds(fake_wheel_env, monkeypatch):
    """SHA 匹配 → 正常进入安装路径并返回 torch_cpu 目录。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", fake_wheel_env["sha"])
    dest = bi._install_torch(fake_wheel_env["dir"], ["https://x/"])
    assert dest.name == "torch_cpu" and dest.exists()


def test_install_torch_urls_percent_encode_plus(fake_wheel_env, monkeypatch):
    """URL 中 "+" 必须 %2B 编码——官方源（S3/CloudFront）对字面 + 返回 403。"""
    monkeypatch.setattr(bi, "TORCH_CPU_SHA256", fake_wheel_env["sha"])
    bi._install_torch(fake_wheel_env["dir"],
                      ["https://mirror.example/cpu", "https://official.example/cpu"])
    urls = fake_wheel_env["seen"]["urls"]
    assert len(urls) == 2
    for u in urls:
        fname = u.rsplit("/", 1)[-1]
        assert fname == "torch-2.7.1%2Bcpu-cp312-cp312-win_amd64.whl"


# ---------- 打包链一致性校验（跨文件/跨语言的同款事实，漂移即打包前报错） ----------
# 校验读的是 ROOT 下的文本：把真实 install.iss / pyproject.toml 复制到 tmp_path
# 再改坏——既不动仓库文件，也保证校验的是真实文本而不是自造样例。

ISS_SRC = (ROOT / "installer" / "install.iss").read_text(encoding="utf-8")
PYPROJECT_SRC = (ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.fixture
def packaging_root(tmp_path, monkeypatch):
    """ROOT 指向 tmp_path 下的文本副本；返回该目录供各用例改坏。"""
    (tmp_path / "installer").mkdir()
    (tmp_path / "installer" / "install.iss").write_text(ISS_SRC, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(PYPROJECT_SRC, encoding="utf-8")
    monkeypatch.setattr(bi, "ROOT", tmp_path)
    return tmp_path


def _rewrite(path: Path, old: str, new: str, count: int = 1) -> None:
    """替换副本内容；待替换文本不存在即失败（说明源文件改版，用例需同步）。"""
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name} 中找不到 {old!r}，用例已失效需同步更新"
    path.write_text(text.replace(old, new, count), encoding="utf-8")


def _break_iss(root: Path, old: str, new: str, count: int = 1) -> None:
    _rewrite(root / "installer" / "install.iss", old, new, count)


SHA = "2bb8c05d48ba815b316879a18195d53a6472a03e297d971e916753f8e1053d30"


def test_packaging_consistency_passes_on_repo_text(packaging_root):
    """真实文本的副本必须通过三项校验：护栏自己不能是误报源。"""
    bi._check_packaging_consistency()


@pytest.mark.parametrize("old,new,needle", [
    (f"CUDA_SHA = '{SHA}'", "CUDA_SHA = '" + "0" * 64 + "'", "SHA 不一致"),
    ("ExternalSize: 3273024349", "ExternalSize: 1", "ExternalSize"),
    ("CUDA_WHEEL = 'torch-2.7.1+cu128", "CUDA_WHEEL = 'torch-9.9.9+cu999", "CUDA_WHEEL"),
    (r'Source: "{tmp}\torch-2.7.1+cu128-cp312-cp312-win_amd64.zip"',
     r'Source: "{tmp}\other.zip"', "[Files] 解压源"),
    (f"CUDA_URL = '{dl.TORCH_CUDA_URLS[0]}'",
     "CUDA_URL = 'https://example.com'", "CUDA_URL"),
], ids=["sha", "字节数", "wheel常量", "files解压源", "url"])
def test_cuda_asset_drift_is_fatal(packaging_root, old, new, needle):
    """CUDA 资产（SHA / 字节数 / 文件名 / 镜像）任一处漂移都必须报错。"""
    _break_iss(packaging_root, old, new)
    with pytest.raises(SystemExit, match=re.escape(needle)):
        bi._check_packaging_consistency()


def test_missing_sha_constant_is_fatal(packaging_root):
    """CUDA_SHA 整行被删 → 报错，而不是静默跳过 SHA 校验。"""
    iss = packaging_root / "installer" / "install.iss"
    text = iss.read_text(encoding="utf-8")
    kept = [ln for ln in text.splitlines() if "CUDA_SHA =" not in ln]
    assert len(kept) == len(text.splitlines()) - 1, "install.iss 中没有 CUDA_SHA 所在行"
    iss.write_text("\n".join(kept), encoding="utf-8")
    with pytest.raises(SystemExit, match="找不到 CUDA_SHA"):
        bi._check_packaging_consistency()


def test_missing_external_size_is_fatal(packaging_root):
    """ExternalSize 属性被删 → 报错（校验项缺失不得等于校验通过）。"""
    _break_iss(packaging_root, "ExternalSize: 3273024349; ", "")
    with pytest.raises(SystemExit, match="找不到 ExternalSize"):
        bi._check_packaging_consistency()


@pytest.mark.parametrize("token", ["'*.lib'", "T + 'include'", "T + 'bin'",
                                   "'torch_shm_manager.exe'"])
def test_prune_list_drift_is_fatal(packaging_root, token):
    """Pascal 裁剪清单少任何一类都报错（清单唯一来源在 download.py）。

    与 test_cuda_torch.py::test_install_extracts_structure_and_prunes 断言的
    同一份清单对应。注意标记词核对只能拦「少项/改名」，拦不住「Pascal 逻辑被
    改写但标记词还在」——那一档只能靠有 Inno 环境的机器实跑 PruneTorchCuda。
    """
    iss = packaging_root / "installer" / "install.iss"
    hits = iss.read_text(encoding="utf-8").count(token)
    assert hits, f"install.iss 中找不到 {token!r}，用例已失效需同步更新"
    # 改掉全部出现处：只要还剩一处，校验就该通过（第一版只改首处 → 用例自己红过）
    _break_iss(packaging_root, token, token[:-1] + "x" + token[-1], count=hits)
    with pytest.raises(SystemExit, match="缺少裁剪项"):
        bi._check_packaging_consistency()


def test_dependency_added_to_pyproject_only_is_fatal(packaging_root):
    """pyproject 新增运行依赖但没进 DEPENDENCIES → 装机漏包，必须打包前报错。"""
    _rewrite(packaging_root / "pyproject.toml", '"numpy>=1.24",',
             '"numpy>=1.24",\n    "brand-new-dep>=1.0",')
    with pytest.raises(SystemExit, match="brand-new-dep"):
        bi._check_packaging_consistency()


def test_dependency_dropped_from_installer_list_is_fatal(packaging_root, monkeypatch):
    """反向：DEPENDENCIES 手滑少一项（如 safetensors）同样报错。"""
    monkeypatch.setattr(bi, "DEPENDENCIES",
                        [d for d in bi.DEPENDENCIES if not d.startswith("safetensors")])
    with pytest.raises(SystemExit, match="safetensors"):
        bi._check_packaging_consistency()


def test_download_side_size_drift_is_fatal(packaging_root, monkeypatch):
    """download.py 侧字节数改坏、install.iss 不动 → 也要报错（双向核对）。"""
    monkeypatch.setattr(dl, "TORCH_CUDA_SIZE", 1)
    with pytest.raises(SystemExit, match="TORCH_CUDA_SIZE"):
        bi._check_packaging_consistency()


def test_pyproject_dependencies_block_missing_is_fatal(packaging_root):
    """pyproject 里解析不到 dependencies 块时不能静默通过。"""
    (packaging_root / "pyproject.toml").write_text('[project]\nname = "x"\n',
                                                   encoding="utf-8")
    with pytest.raises(SystemExit, match="dependencies"):
        bi._check_packaging_consistency()


def _write_tar(path: Path, members: list[tuple[str, bytes | None]]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            if data is None:
                info.type = tarfile.DIRTYPE
                tf.addfile(info)
            else:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


def test_green_python_bad_sha_deletes_archive(tmp_path, monkeypatch):
    """SHA256 不符：删掉坏 tar，且不得解压。哈希在解压之前。"""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    src = tmp_path / "ok.tar.gz"
    _write_tar(src, [("python/Lib/ok.py", b"print(1)\n")])
    monkeypatch.setattr(bi, "GREEN_PY_SHA256", "0" * 64)

    def fake_dl(urls, dest, progress_cb=None):
        shutil.copyfile(src, dest)

    monkeypatch.setattr(bi, "_download_to", fake_dl)
    with pytest.raises(SystemExit, match="校验失败"):
        bi._download_green_python(bundle)
    assert not (bundle / bi.GREEN_PY_FILENAME).exists()
    assert not (bundle / "python").exists()


def test_green_python_extracts_prefixed_members(tmp_path, monkeypatch):
    """python/ 前缀剥掉后落在 bundle/python 下，压缩包在成功后删除。"""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    src = tmp_path / "ok.tar.gz"
    _write_tar(src, [
        ("python", None),
        ("python/Lib/ok.py", b"ok"),
        ("python/python.exe", b"exe"),
    ])
    monkeypatch.setattr(bi, "GREEN_PY_SHA256", bi.sha256_of(src))

    def fake_dl(urls, dest, progress_cb=None):
        shutil.copyfile(src, dest)

    monkeypatch.setattr(bi, "_download_to", fake_dl)
    bi._download_green_python(bundle)
    assert (bundle / "python" / "Lib" / "ok.py").read_bytes() == b"ok"
    assert (bundle / "python" / "python.exe").read_bytes() == b"exe"
    assert not (bundle / "python" / "python").exists()
    assert not (bundle / bi.GREEN_PY_FILENAME).exists()


@pytest.mark.parametrize("name", [
    "/abs/evil.txt",
    "C:/Windows/evil.txt",
    "D:evil.txt",
    "python/../../outside.txt",
    "python/foo/../../outside.txt",
    "../outside.txt",
    "python/C:/evil.txt",
])
def test_green_python_rejects_unsafe_member(tmp_path, monkeypatch, name):
    """绝对路径、盘符，以及去掉 python/ 前缀后仍含 .. 的成员名都拒绝。"""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    src = tmp_path / "bad.tar.gz"
    _write_tar(src, [(name, b"evil"), ("python/Lib/ok.py", b"ok")])
    monkeypatch.setattr(bi, "GREEN_PY_SHA256", bi.sha256_of(src))

    def fake_dl(urls, dest, progress_cb=None):
        shutil.copyfile(src, dest)

    monkeypatch.setattr(bi, "_download_to", fake_dl)
    with pytest.raises(SystemExit, match="不安全路径"):
        bi._download_green_python(bundle)
    assert not (bundle / "python" / "Lib" / "ok.py").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "evil.txt").exists()
    assert not (bundle / "outside.txt").exists()


def test_drop_verified_markers_keeps_weights(tmp_path):
    """只删 *.verified，仓库/目录里的权重文件必须留下。"""
    models = tmp_path / "models"
    nested = models / "nested"
    nested.mkdir(parents=True)
    weight = models / "m.safetensors"
    weight.write_bytes(b"weights")
    (models / "m.safetensors.verified").write_text("abc:1", encoding="utf-8")
    (nested / "x.verified").write_text("nope", encoding="utf-8")
    bi._drop_verified_markers(models)
    assert weight.read_bytes() == b"weights"
    assert list(models.rglob("*.verified")) == []


def test_installer_excludes_verified_and_cuda_failure_continues():
    """安装脚本：模型排除 *.verified；CUDA 下载失败不中止安装。"""
    text = (ROOT / "installer" / "install.iss").read_text(encoding="utf-8")
    models = [ln for ln in text.splitlines() if r"{#BundleDir}\models" in ln]
    assert models and all("*.verified" in ln for ln in models)
    assert "RaiseException" not in text
    assert "CudaDownloadOK" in text
    assert "CUDA 未安装，可稍后在应用内补装" in text
    assert r"torch_cuda\.complete" in text
    assert r"torch\version.py" in text
    assert r"DirExists(ExpandConstant('{app}\torch_cuda\torch'))" not in text
    cuda_files = [ln for ln in text.splitlines() if "ExternalSize:" in ln]
    assert len(cuda_files) == 1
    assert "CudaExtractReady" in cuda_files[0]
