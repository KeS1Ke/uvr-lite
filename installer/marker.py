# coding: utf-8
"""安装标记 install.json：识别已安装目录（覆盖升级）、记录元信息。"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_APP_NAME = "uvr-lite"


def marker_path(install_dir: Path) -> Path:
    return install_dir / "install.json"


def is_install_dir(install_dir: Path) -> bool:
    return marker_path(install_dir).exists()


def write_marker(install_dir: Path, **meta: Any) -> Path:
    """写入 install.json（含安装时间），返回文件路径。"""
    data: Dict[str, Any] = {
        "app": _APP_NAME,
        "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    data.update(meta)
    path = marker_path(install_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_marker(install_dir: Path) -> Dict[str, Any]:
    """读取 install.json；不存在或 app 字段不符返回空 dict。"""
    path = marker_path(install_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) and data.get("app") == _APP_NAME else {}
