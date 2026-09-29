"""uvr_lite/cli.py：命令失败的兜底处理器（中文提示 + 日志路径）不得自身崩溃。

背景：``log_path()`` 依赖 ``repo_root()``，而后者在异常目录布局下会抛
RuntimeError（uvr_lite/__init__.py 的 ``_base_dir``）。错误分支若直接调
``log_path()``，"给用户一句友好中文 + 日志位置"这个动作本身会把异常抛出去，
用户看到的是原始 traceback——正好违背该分支的存在理由。这里锁住这条路径。
"""

import uvr_lite.log as L
from uvr_lite.cli import main


def _boom():
    """模拟 _base_dir 在异常目录布局下的行为。"""
    raise RuntimeError("无法定位 uvr-lite 根目录")


def _fail_ensure_model(*args, **kwargs):
    """让 download 命令在真实错误分支里失败（不联网、不落盘）。"""
    raise RuntimeError("模拟下载失败")


def test_error_path_survives_broken_root(monkeypatch, capsys):
    """repo_root 抛 RuntimeError 时：仍返回 1、仍给 [ERROR] 中文提示，不二次崩溃。"""
    monkeypatch.setattr("uvr_lite.cli.ensure_model", _fail_ensure_model)
    monkeypatch.setattr(L, "repo_root", _boom)

    assert main(["download"]) == 1

    out = capsys.readouterr().out
    assert "[ERROR]" in out, "用户可见的失败提示不能因为日志定位失败而消失"
    assert "download 执行失败" in out and "模拟下载失败" in out, f"应保留中文错误原因: {out!r}"
    # 日志确实不可用时宁可不提，也不给一个可能不存在的路径（log_hint 的既有契约）
    assert "详细信息已写入日志" not in out


def test_error_path_prints_log_path(monkeypatch, capsys, tmp_path):
    """目录布局正常时，失败提示仍带上日志文件路径（供用户报障）。"""
    monkeypatch.setattr("uvr_lite.cli.ensure_model", _fail_ensure_model)

    assert main(["download"]) == 1

    out = capsys.readouterr().out
    assert "[ERROR]" in out
    assert f"详细信息已写入日志：{tmp_path / 'logs' / 'uvr-lite.log'}" in out


def test_success_path_unchanged(monkeypatch, capsys, tmp_path):
    """成功路径不因本修复改变：退出码 0、不打印 [ERROR]、也不碰真实模型目录。"""
    # 模型目录重定向到 tmp（与 tests/test_cli_models.py 同款）：models 命令会经
    # retired_model_files→models_dir() 解析到仓库根 models/（mkdir + 读出真实退役
    # 权重），不隔离的话用例既污染工作区、结果又依赖本机是否留旧权重
    monkeypatch.setenv("UVR_MODEL_DIR", str(tmp_path))

    assert main(["models"]) == 0

    out = capsys.readouterr().out
    assert "[ERROR]" not in out
    assert "未下载" in out, "正常的列表输出照旧（tmp 里没有权重）"
    assert "可回收" not in out, "tmp 里没有退役权重，不该出现可回收行"
