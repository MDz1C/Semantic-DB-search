"""Persistent local state in SQLite: settings, the imported database, query history and the
embedding cache.  Everything lives under paths.data_dir(); nothing leaves the machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.types import HistoryEntry, Roles, Table


class SqliteEmbeddingCache:
    """Implements core.types.EmbeddingCache; vectors stored as float16 blobs keyed by sha1(model_id, text)."""

    def __init__(self, path: Path, max_entries: int = 400_000) -> None:
        raise NotImplementedError

    def get_many(self, model_id: str, texts: list[str]) -> dict[int, np.ndarray]:
        raise NotImplementedError

    def put_many(self, model_id: str, texts: list[str], vectors: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class Store:
    """Thread-safe facade over app.sqlite (+ embeddings.sqlite)."""

    def __init__(self, data_dir: Path) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # settings ---------------------------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        raise NotImplementedError

    def set_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError

    # dataset ----------------------------------------------------------------------------
    def save_dataset(self, table: Table, roles: Roles) -> None:
        raise NotImplementedError

    def load_dataset(self) -> tuple[Table, Roles] | None:
        raise NotImplementedError

    def save_roles(self, roles: Roles) -> None:
        raise NotImplementedError

    def has_dataset(self) -> bool:
        raise NotImplementedError

    def clear_dataset(self) -> None:
        raise NotImplementedError

    # history ----------------------------------------------------------------------------
    def add_entries(self, entries: list[HistoryEntry]) -> list[HistoryEntry]:
        raise NotImplementedError

    def load_entries(self) -> list[HistoryEntry]:
        raise NotImplementedError

    def delete_entries(self, ids: list[int]) -> None:
        raise NotImplementedError

    def clear_entries(self) -> None:
        raise NotImplementedError

    # embeddings -------------------------------------------------------------------------
    @property
    def embedding_cache(self) -> SqliteEmbeddingCache:
        raise NotImplementedError
