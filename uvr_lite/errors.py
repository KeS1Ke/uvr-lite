"""共享异常：取消语义跨引擎（分离 / 合成）复用。

独立成模块是为了不把 torch 拖进合成路径——engine 与 mix 都从这里导入，
engine 侧继续 re-export（`from uvr_lite.engine import CancelledError` 兼容）。
"""


class CancelledError(Exception):
    """用户请求取消当前任务（进度回调返回 False 时抛出）。"""
