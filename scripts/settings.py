import json
import os
from pathlib import Path


SETTINGS_PATH = Path(os.environ.get("RAG_SETTINGS_PATH", "/sandbox/workspace/settings.json"))
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/sandbox/data"))


def _read_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_settings(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_data_root() -> Path:
    configured = _read_settings().get("data_root")
    if isinstance(configured, str) and configured.strip():
        return Path(configured).resolve()
    return DEFAULT_DATA_ROOT.resolve()


def set_data_root(path: str) -> Path:
    root = Path(path.strip()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = _read_settings()
    data["data_root"] = str(root)
    _write_settings(data)
    return root
