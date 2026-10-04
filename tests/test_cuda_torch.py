"""票 1（tdd）：CUDA 推理引擎下载安装（install_cuda_torch）。

用本地构造的假 wheel zip（与真实 wheel 同构：torch/ + dist-info/，含
.lib/include/bin）mock 掉 _download，验证：解压结构、裁剪、幂等、SHA 校验、
取消。真实 wheel 的 SHA256 常量（TORCH_CUDA_SHA256）在测试中被替换/保持。

**本机是 Linux + Python 3.13，而内置 wheel 是 cp312/win_amd64**（生产目标是
原生 Windows 3.12）。因此除 ABI/平台门槛的用例外，所有 install_cuda_torch
用例都显式注入一个「受支持的环境」，让它们测的是下载安装逻辑本身，而不是
被环境门槛先拦下。
"""

import hashlib
import shutil
import zipfile
from pathlib import Path

import pytest

import uvr_lite.download as dl
from uvr_lite.download import cuda_torch_installed, install_cuda_torch

# 受支持环境的架构/位数两要素（平台与版本按用例另注）
X64_ENV = {"machine": "AMD64", "pointer_size": 8}
# 假装运行在受支持的原生环境里（Windows + Python 3.12 + x64 + 64 位解释器）；
# machine/pointer_size 一并注入，安装路径用例在 ARM64/32 位开发机上也能跑。
SUPPORTED_ENV = {"platform_str": "win32", "version_info": (3, 12, 8, "final", 0), **X64_ENV}


def make_wheel(path: Path) -> str:
    """构造与真实 wheel 同构的 zip，返回其 SHA256。"""
    files = {
        "torch/__init__.py": b"# fake torch\n",
        "torch/lib/torch.lib": b"L" * 100,             # 编译期导入库 → 应裁剪
        "torch/include/ATen/foo.h": b"h" * 10,          # 头文件目录 → 应裁剪
        "torch/bin/random.exe": b"E" * 10,              # 非 shm 可执行 → 应裁剪
        "torch/bin/torch_shm_manager.exe": b"S" * 10,   # 运行需要 → 保留
        "torch-2.7.1+cu128.dist-info/METADATA": b"Metadata\n",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_wheel(tmp_path, monkeypatch):
    """假 wheel 就位：SHA 常量替换为假值，_download 变为本地复制。"""
    wheel = tmp_path / "fake.whl"
    sha = make_wheel(wheel)
    cache = tmp_path / "cache"

    monkeypatch.setattr(dl, "TORCH_CUDA_SHA256", sha)
    monkeypatch.setattr(dl, "_wheel_cache_dir", lambda: cache)

    def fake_download(urls, dest, progress_callback=None, retries=2):
        dest.parent.mkdir(parents=True, exist_ok=True)  # 真实 _download 会建目录
        shutil.copy2(wheel, dest)

    monkeypatch.setattr(dl, "_download", fake_download)
    monkeypatch.setattr(dl, "_REPORT_INTERVAL", 1)  # 小文件也触发进度回调
    return wheel


def test_install_extracts_structure_and_prunes(fake_wheel, tmp_path):
    dest = install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert dest == tmp_path / "base" / "torch_cuda"
    # 解压结构（torch/ 与 dist-info/ 都在）
    assert (dest / "torch" / "__init__.py").exists()
    assert (dest / "torch-2.7.1+cu128.dist-info" / "METADATA").exists()
    # 裁剪：.lib / include / 非 shm 的 bin 全删，shm 保留
    assert not (dest / "torch" / "lib" / "torch.lib").exists()
    assert not (dest / "torch" / "include").exists()
    assert not (dest / "torch" / "bin" / "random.exe").exists()
    assert (dest / "torch" / "bin" / "torch_shm_manager.exe").exists()
    # wheel 中转文件已清理
    assert not (fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL).exists()


def test_install_idempotent_skips_download(fake_wheel, tmp_path, monkeypatch):
    install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    calls = {"n": 0}

    def counting(urls, dest, progress_callback=None, retries=2):
        calls["n"] += 1
        shutil.copy2(fake_wheel, dest)

    monkeypatch.setattr(dl, "_download", counting)
    install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert calls["n"] == 0, "已安装时不应再触发下载"
    assert cuda_torch_installed(tmp_path / "base")


def test_install_bad_sha_removes_wheel_and_raises(monkeypatch, tmp_path):
    """SHA 常量保持真实值（假 wheel 必然不匹配）→ 删缓存并报错。

    报错要带完整 sha256 和下载 URL，不能只截断前 16 位。
    """
    wheel = tmp_path / "fake.whl"
    actual = make_wheel(wheel)
    cache = tmp_path / "cache"
    monkeypatch.setattr(dl, "_wheel_cache_dir", lambda: cache)

    def fake_download(urls, dest, progress_callback=None, retries=2):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wheel, dest)

    monkeypatch.setattr(dl, "_download", fake_download)
    with pytest.raises(RuntimeError, match="SHA256 校验失败") as exc:
        install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    msg = str(exc.value)
    assert dl.TORCH_CUDA_SHA256 in msg, f"期望哈希必须完整出现: {msg}"
    assert actual in msg, f"实际哈希必须完整出现: {msg}"
    assert dl.TORCH_CUDA_URLS[0] in msg, f"报错要带上 URL: {msg}"
    assert "…" not in msg, f"不应再截断哈希: {msg}"
    assert not (cache / dl.TORCH_CUDA_WHEEL).exists(), "损坏缓存应被删除"
    assert not cuda_torch_installed(tmp_path / "base")


def test_install_cancel_during_extract(fake_wheel, tmp_path, monkeypatch):
    """解压阶段取消：抛 InterruptedError，wheel 与侧车保留；重试不得再下载。"""
    calls = []
    downloads = {"n": 0}
    real_download = dl._download

    def counting(urls, dest, progress_callback=None, retries=2):
        downloads["n"] += 1
        return real_download(urls, dest, progress_callback, retries)

    monkeypatch.setattr(dl, "_download", counting)

    def cb(done, total):
        calls.append((done, total))
        return done < total * 0.5  # 解压到一半取消

    with pytest.raises(InterruptedError):
        install_cuda_torch(tmp_path / "base", progress_callback=cb, **SUPPORTED_ENV)
    wheel = fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL
    side = dl._wheel_sha_sidecar(wheel)
    assert wheel.exists()
    assert side.is_file()
    assert side.read_text(encoding="utf-8").strip() == dl.TORCH_CUDA_SHA256
    # 取消后被清理干净：残缺的 torch/__init__.py 不能让 installed 判真
    assert not cuda_torch_installed(tmp_path / "base")
    assert not (tmp_path / "base" / "torch_cuda").exists()
    assert not (tmp_path / "base" / "torch_cuda.partial").exists()
    assert downloads["n"] == 1
    # 取消后重新安装：侧车 sha 已匹配，不得再调用 _download
    install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert downloads["n"] == 1, "解压取消后重试不应再次下载"
    assert cuda_torch_installed(tmp_path / "base")
    assert calls, "解压阶段应上报过进度"


def test_install_cancel_clears_half_extracted_dest(fake_wheel, tmp_path, monkeypatch):
    """解压中途取消必须删掉半成品目录（回归：marker 先落盘被误判已安装）。

    torch/__init__.py 是安装的判定依据却几乎最先解压；只要它留下，
    cuda_torch_installed() 就返回 True，后续调用直接跳过安装。
    """
    import zipfile

    extracted = {"n": 0}

    class _AbortingZipFile(zipfile.ZipFile):
        def extract(self, member, path=None, pwd=None):
            extracted["n"] += 1
            if extracted["n"] > 1:      # 让 torch/__init__.py 先落盘
                raise InterruptedError("解压已取消")
            return super().extract(member, path, pwd)

    monkeypatch.setattr(zipfile, "ZipFile", _AbortingZipFile)
    with pytest.raises(InterruptedError):
        install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert extracted["n"] > 1, "应至少解压出一个成员（含 torch/__init__.py）"
    dest = tmp_path / "base" / "torch_cuda"
    assert not dest.exists(), "取消后半成品目录必须清理干净"
    assert not (tmp_path / "base" / "torch_cuda.partial").exists()
    assert not cuda_torch_installed(tmp_path / "base")
    # wheel 与侧车保留：下次跳过下载，直接解压
    wheel = fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL
    assert wheel.exists()
    assert dl._wheel_sha_sidecar(wheel).is_file()


def test_cuda_torch_installed_detects_marker(fake_wheel, tmp_path):
    assert not cuda_torch_installed(tmp_path / "base")
    install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert cuda_torch_installed(tmp_path / "base")
    (tmp_path / "base" / "torch_cuda" / "torch" / "__init__.py").unlink()
    assert not cuda_torch_installed(tmp_path / "base")


def test_cuda_urls_percent_encode_plus():
    """URL 中 "+" 必须 %2B 编码——官方源（S3/CloudFront）对字面 + 返回 403。"""
    for url in dl.TORCH_CUDA_URLS:
        fname = url.rsplit("/", 1)[-1]
        assert fname == "torch-2.7.1%2Bcu128-cp312-cp312-win_amd64.whl", url


# ---------- wheel 平台/ABI 与运行环境一致性（审计缺陷 1） ----------
# 内置 wheel 是 cp312 / win_amd64 单个文件，而 requires-python 声明 >= 3.10。
# 修前不校验环境：Windows 上的 3.10/3.11 会照下 cp312 wheel，import 即崩
# （ABI 不匹配）；非 Windows 会下到一个装不了也用不上的 win_amd64 wheel，
# 3.3 GB 白下。现在改为装前拦截 + 可行动的中文错误。
#
# sys.platform 在 32 位与 ARM64 Windows 上同样是 "win32"，只映射平台串会把
# win_amd64 wheel 放行给用不了的环境；这里连同 machine/pointer_size 一起注入，
# 用例不依赖宿主平台。
#
# 环境按参数注入（不 monkeypatch 全局 sys.platform / sys.version_info：
# 那两个是 pytest 自身也在读的模块属性，替掉会让 tmpdir 清理等内部逻辑炸掉）。

MISMATCH = [
    ("win32", (3, 11, 9, "final", 0)),   # Python 3.11：下到 cp312 会 ABI 不匹配
    ("win32", (3, 10, 14, "final", 0)),  # Python 3.10：同上
    ("win32", (3, 13, 1, "final", 0)),   # Python 3.13：cp312 覆盖不到
    ("linux", (3, 12, 0, "final", 0)),   # 非 Windows：win_amd64 wheel 装不了也用不上
    ("darwin", (3, 12, 0, "final", 0)),
    ("linux2", (3, 12, 0, "final", 0)),  # Python 2 的遗留平台串，仍应被拒
]


@pytest.mark.parametrize("sys_platform, version_info", MISMATCH)
def test_cuda_abi_compatibility_gate_rejects_mismatch(sys_platform, version_info):
    with pytest.raises(dl.CudaEngineUnsupportedError) as exc:
        dl.cuda_engine_requirements(platform_str=sys_platform, version_info=version_info,
                                    **X64_ENV)

    msg = str(exc.value)
    assert "cp312" in msg and "win_amd64" in msg, f"报错要说清需要的环境: {msg}"


def test_cuda_gate_rejects_arm64_windows():
    """sys.platform 同为 "win32"，但 ARM64 Windows 装不了 win_amd64 wheel。"""
    with pytest.raises(dl.CudaEngineUnsupportedError) as exc:
        dl.cuda_engine_requirements(platform_str="win32",
                                    version_info=(3, 12, 8, "final", 0),
                                    machine="ARM64", pointer_size=8)
    assert "ARM64" in str(exc.value), f"原因里要说明当前架构: {exc.value}"


def test_cuda_gate_rejects_32bit_interpreter():
    """32 位 Python 的 sys.platform 也是 "win32"，但指针 4 字节装不了 win_amd64。"""
    with pytest.raises(dl.CudaEngineUnsupportedError) as exc:
        dl.cuda_engine_requirements(platform_str="win32",
                                    version_info=(3, 12, 8, "final", 0),
                                    machine="AMD64", pointer_size=4)
    assert "32 位" in str(exc.value), f"原因里要说明解释器位数: {exc.value}"


@pytest.mark.parametrize("machine", ["AMD64", "amd64", "x86_64", "x64", "EM64T"])
def test_cuda_gate_accepts_x64_machine_aliases(machine):
    req = dl.cuda_engine_requirements(platform_str="win32",
                                      version_info=(3, 12, 8, "final", 0),
                                      machine=machine, pointer_size=8)
    assert req.wheel == dl.TORCH_CUDA_WHEEL


def test_cuda_abi_gate_accepts_native_windows_py312():
    req = dl.cuda_engine_requirements(platform_str="win32",
                                      version_info=(3, 12, 8, "final", 0), **X64_ENV)
    assert req.wheel == dl.TORCH_CUDA_WHEEL
    assert req.sha256 == dl.TORCH_CUDA_SHA256


def test_wheel_tags_derived_from_wheel_filename():
    """标签只从 wheel 文件名解析（单一来源），property 复用同一解析。"""
    assert dl._parse_wheel_tags(dl.TORCH_CUDA_WHEEL) == (
        dl.TORCH_CUDA_PY_TAG, dl.TORCH_CUDA_PLATFORM_TAG)
    req = dl.CudaEngineRequirements(wheel=dl.TORCH_CUDA_WHEEL, sha256="", size=0, urls=[])
    assert (req.py_tag, req.platform_tag) == (
        dl.TORCH_CUDA_PY_TAG, dl.TORCH_CUDA_PLATFORM_TAG)
    # 换一个 wheel 文件名，解析结果随之变化（防解析逻辑被写死）
    assert dl._parse_wheel_tags("foo-1.0-cp311-cp311-win32.whl") == ("cp311", "win32")


def test_cuda_install_rejects_unsupported_env_before_downloading(tmp_path, monkeypatch):
    """不支持的环境必须在下载之前就拦下（回归：先花 3.3 GB 再失败）。"""
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("环境不兼容时不应触发下载")

    monkeypatch.setattr(dl, "_download", boom)

    with pytest.raises(dl.CudaEngineUnsupportedError):
        dl.install_cuda_torch(tmp_path / "base", platform_str="win32",
                              version_info=(3, 11, 9, "final", 0))

    assert calls["n"] == 0
    assert not (tmp_path / "base" / "torch_cuda").exists()


def test_cuda_install_idempotent_wins_over_unsupported_env(tmp_path, monkeypatch):
    """回归：已安装则先返回，不再过环境门槛（解释器换代也不该报错）。

    旧的完整安装没有 .complete，但 torch/__init__.py 与 version.py 都在：
    视为已安装并补写标记，不重下。
    """
    torch_dir = tmp_path / "base" / "torch_cuda" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text("")
    (torch_dir / "version.py").write_text("__version__ = '2.7.1'\n")
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("已安装时不应触发下载")

    monkeypatch.setattr(dl, "_download", boom)
    dest = install_cuda_torch(tmp_path / "base", platform_str="linux",
                              version_info=(3, 12, 8, "final", 0))
    assert dest == tmp_path / "base" / "torch_cuda", "已安装时环境不支持也应照常返回"
    assert calls["n"] == 0
    assert (dest / ".complete").is_file(), "旧的完整安装应补写完成标记"


def test_incomplete_torch_cuda_is_not_installed(tmp_path):
    """只有 __init__.py、或只有 version.py，都不算已安装。"""
    torch_dir = tmp_path / "base" / "torch_cuda" / "torch"
    torch_dir.mkdir(parents=True)
    (torch_dir / "__init__.py").write_text("")
    assert not cuda_torch_installed(tmp_path / "base")
    (torch_dir / "__init__.py").unlink()
    (torch_dir / "version.py").write_text("")
    assert not cuda_torch_installed(tmp_path / "base")
    (torch_dir / "__init__.py").write_text("")
    (tmp_path / "base" / "torch_cuda" / ".complete").write_text("abc")
    assert cuda_torch_installed(tmp_path / "base")


def test_reinstall_replaces_incomplete_tree(fake_wheel, tmp_path):
    """残缺正式目录下次安装先清掉，再换成解压成功的树。"""
    base = tmp_path / "base"
    broken = base / "torch_cuda" / "torch"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("partial")
    (broken / "leftover.txt").write_text("x")
    assert not cuda_torch_installed(base)
    dest = install_cuda_torch(base, **SUPPORTED_ENV)
    assert cuda_torch_installed(base)
    assert (dest / ".complete").read_text(encoding="utf-8").strip() == dl.TORCH_CUDA_SHA256
    assert not (broken / "leftover.txt").exists()
    assert (dest / "torch" / "__init__.py").read_text(encoding="utf-8") == "# fake torch\n"
    assert not (base / "torch_cuda.partial").exists()


def test_extract_error_removes_partial(fake_wheel, tmp_path, monkeypatch):
    """解压失败（不只是取消）也要删掉临时目录，不能留下半成品。"""
    import zipfile

    class _Boom(zipfile.ZipFile):
        def extract(self, member, path=None, pwd=None):
            raise OSError("disk full")

    monkeypatch.setattr(zipfile, "ZipFile", _Boom)
    with pytest.raises(OSError, match="disk full"):
        install_cuda_torch(tmp_path / "base", **SUPPORTED_ENV)
    assert not (tmp_path / "base" / "torch_cuda.partial").exists()
    assert not (tmp_path / "base" / "torch_cuda").exists()
    assert not cuda_torch_installed(tmp_path / "base")
    # 不是坏 zip：wheel 与侧车留着，下次可直接解压
    wheel = fake_wheel.parent / "cache" / dl.TORCH_CUDA_WHEEL
    assert wheel.exists()
    assert dl._wheel_sha_sidecar(wheel).is_file()


def test_block_reason_full_when_unsupported_and_empty_when_supported():
    """UI 拿它做 tooltip：不支持时给完整文案（含怎么办），可用时给空串。"""
    reason = dl.cuda_engine_block_reason(platform_str="win32",
                                         version_info=(3, 11, 9, "final", 0), **X64_ENV)
    assert reason, "不支持的环境应有原因说明"
    assert "①" in reason and "pip install" in reason, f"完整文案要含可行动的出路: {reason}"

    assert dl.cuda_engine_block_reason(platform_str="win32",
                                       version_info=(3, 12, 8, "final", 0), **X64_ENV) == ""
