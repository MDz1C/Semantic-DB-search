"""Filesystem locations: bundled read-only resources and the per-user data folder.

Resources (the embedding model, icons, the sample workbook) ship with the app. When
frozen by PyInstaller they live under ``sys._MEIPASS``; in a source checkout they live
at the repository root.  User data (database snapshot, history, settings, caches)
always lives in a per-user writable folder, never next to the executable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_ID

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle)
    return Path(__file__).resolve().parents[2]


def model_dir() -> Path:
    override = os.environ.get("SEMDB_MODEL_DIR")
    if override:
        return Path(override)
    return resource_root() / "models" / MODEL_NAME


def assets_dir() -> Path:
    return resource_root() / "assets"


def sample_workbook() -> Path:
    return assets_dir() / "sample" / "sample_funds.xlsx"


def data_dir() -> Path:
    """Per-user writable folder (``%LOCALAPPDATA%\\SemanticDBSearch`` on Windows).

    ``SEMDB_DATA_DIR`` overrides it, which tests use to stay isolated.
    """
    override = os.environ.get("SEMDB_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / APP_ID
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / APP_ID
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        path = Path(base) / APP_ID
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
