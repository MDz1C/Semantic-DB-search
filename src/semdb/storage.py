"""Persistent local state in SQLite: settings, the imported database, query history and the
embedding cache.  Everything lives under paths.data_dir(); nothing leaves the machine.

Both databases use WAL journaling and one connection each, shared across threads and
serialized by a lock.  A file that SQLite reports as corrupt (or that fails the integrity
check at startup) is renamed ``<name>.corrupt-<timestamp>`` and recreated; an app database
written by a newer, incompatible schema is moved aside the same way (``.incompatible-``),
never modified.  All (de)serialization is explicit JSON; nothing is pickled.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .core.text import normalize
from .core.types import FamilyEvidence, HistoryEntry, ResultRow, Role, Roles, Table

log = logging.getLogger(__name__)

APP_DB_NAME = "app.sqlite"
CACHE_DB_NAME = "embeddings.sqlite"
APP_SCHEMA_VERSION = 1
CACHE_SCHEMA_VERSION = 1

_CHUNK = 500  # keys per "IN (...)" list
_CORRUPTION_CODES = (11, 26)  # SQLITE_CORRUPT, SQLITE_NOTADB

_APP_SCHEMA = {
    "settings": "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    "dataset": (
        "CREATE TABLE IF NOT EXISTS dataset (id INTEGER PRIMARY KEY CHECK (id = 1), "
        "source_name TEXT NOT NULL, sheet_name TEXT NOT NULL, columns_json TEXT NOT NULL, "
        "roles_json TEXT NOT NULL, fingerprint TEXT NOT NULL, saved_at TEXT NOT NULL)"
    ),
    "dataset_rows": (
        "CREATE TABLE IF NOT EXISTS dataset_rows "
        "(idx INTEGER PRIMARY KEY, source_row INTEGER NOT NULL, cells_json TEXT NOT NULL)"
    ),
    "history": (
        "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL, "
        "query_key TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
    ),
}
_APP_COLUMNS = {
    "settings": {"key", "value"},
    "dataset": {"id", "source_name", "sheet_name", "columns_json", "roles_json", "fingerprint", "saved_at"},
    "dataset_rows": {"idx", "source_row", "cells_json"},
    "history": {"id", "query", "query_key", "created_at", "payload_json"},
}
_CACHE_SCHEMA = "CREATE TABLE IF NOT EXISTS vectors (key BLOB PRIMARY KEY, vec BLOB NOT NULL, used INTEGER NOT NULL)"


class SqliteEmbeddingCache:
    """Implements core.types.EmbeddingCache; vectors stored as float16 blobs keyed by sha1(model_id, text)."""

    def __init__(self, path: Path, max_entries: int = 400_000) -> None:
        self.path = Path(path)
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._tick = 0
        self._conn = _open_database(self.path, _prepare_cache)

    def get_many(self, model_id: str, texts: list[str]) -> dict[int, np.ndarray]:
        if not texts:
            return {}
        positions: dict[bytes, list[int]] = {}
        for position, text in enumerate(texts):
            positions.setdefault(_cache_key(model_id, text), []).append(position)
        found: list[tuple[bytes, bytes]] = []
        with self._lock:
            try:
                for chunk in _chunks(list(positions)):
                    marks = ",".join("?" * len(chunk))
                    found.extend(self._conn.execute(f"SELECT key, vec FROM vectors WHERE key IN ({marks})", chunk))
                if found:
                    tick = self._next_tick()
                    with _transaction(self._conn):
                        for chunk in _chunks([key for key, _ in found]):
                            marks = ",".join("?" * len(chunk))
                            self._conn.execute(f"UPDATE vectors SET used = ? WHERE key IN ({marks})", (tick, *chunk))
            except sqlite3.DatabaseError as exc:
                self._recover(exc)
                return {}
        return _decode_vectors(found, positions)

    def put_many(self, model_id: str, texts: list[str], vectors: np.ndarray) -> None:
        if not texts:
            return
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(f"expected {len(texts)} vectors, got array of shape {matrix.shape}")
        half = matrix.astype("<f2")
        with self._lock:
            tick = self._next_tick()
            rows = [(_cache_key(model_id, text), half[index].tobytes(), tick) for index, text in enumerate(texts)]
            try:
                with _transaction(self._conn):
                    self._conn.executemany(
                        "INSERT INTO vectors (key, vec, used) VALUES (?, ?, ?) "
                        "ON CONFLICT (key) DO UPDATE SET vec = excluded.vec, used = excluded.used",
                        rows,
                    )
                    excess = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] - self.max_entries
                    if excess > 0:  # least recently used first
                        self._conn.execute(
                            "DELETE FROM vectors WHERE rowid IN "
                            "(SELECT rowid FROM vectors ORDER BY used, rowid LIMIT ?)",
                            (excess,),
                        )
            except sqlite3.DatabaseError as exc:
                self._recover(exc)

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _next_tick(self) -> int:
        """Strictly increasing within the process, and wall-clock based across sessions."""
        self._tick = max(self._tick + 1, time.time_ns() // 1_000_000)
        return self._tick

    def _recover(self, exc: sqlite3.DatabaseError) -> None:
        """A cache failure is never fatal: log it, and start a fresh file if this one is corrupt
        (falling back to an in-memory cache for the session when even that fails)."""
        log.warning("Embedding cache error (%s); continuing without it", exc)
        if not _is_corruption(exc):
            return
        self._conn.close()
        try:
            moved = _move_aside(self.path, "corrupt")
            log.warning("Moved the corrupt embedding cache to %s", moved.name)
            self._conn = _open_database(self.path, _prepare_cache)
        except (OSError, sqlite3.Error) as error:
            log.warning("Embedding cache kept in memory for this session (%s)", error)
            self._conn = _connect(":memory:", _prepare_cache)


class Store:
    """Thread-safe facade over app.sqlite (+ embeddings.sqlite)."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = _open_database(self.data_dir / APP_DB_NAME, _prepare_app)
        try:
            self._cache = SqliteEmbeddingCache(self.data_dir / CACHE_DB_NAME)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        self._cache.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # settings ---------------------------------------------------------------------------
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0], object_hook=_decode_tagged)
        except ValueError:
            log.warning("Ignoring unreadable setting %r", key)
            return default

    def set_setting(self, key: str, value: Any) -> None:
        """Store any JSON-compatible value (tuples come back as lists); bytes are supported too."""
        payload = json.dumps(value, ensure_ascii=False, default=_encode_extra)
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, payload),
            )

    # dataset ----------------------------------------------------------------------------
    def save_dataset(self, table: Table, roles: Roles) -> None:
        if any(len(row) != len(table.columns) for row in table.rows) or len(table.source_rows) != len(table.rows):
            raise ValueError("table rows must match its columns and source_rows")
        encode = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode
        records = [(index, number, encode(row)) for index, (number, row) in enumerate(zip(table.source_rows, table.rows))]
        head = (
            table.source_name,
            table.sheet_name,
            encode(table.columns),
            encode(_roles_to_dict(roles, table.columns)),
            table.fingerprint(),
            _utc_now(),
        )
        with self._lock, _transaction(self._conn):
            self._conn.execute("DELETE FROM dataset_rows")
            self._conn.execute("DELETE FROM dataset")
            self._conn.execute(
                "INSERT INTO dataset (id, source_name, sheet_name, columns_json, roles_json, fingerprint, saved_at) "
                "VALUES (1, ?, ?, ?, ?, ?, ?)",
                head,
            )
            self._conn.executemany("INSERT INTO dataset_rows (idx, source_row, cells_json) VALUES (?, ?, ?)", records)

    def load_dataset(self) -> tuple[Table, Roles] | None:
        with self._lock:
            head = self._conn.execute(
                "SELECT source_name, sheet_name, columns_json, roles_json FROM dataset WHERE id = 1"
            ).fetchone()
            if head is None:
                return None
            records = self._conn.execute("SELECT source_row, cells_json FROM dataset_rows ORDER BY idx").fetchall()
        source_name, sheet_name, columns_json, roles_json = head
        try:
            columns = [str(name) for name in json.loads(columns_json)]
            width = len(columns)
            rows: list[list[str]] = []
            source_rows: list[int] = []
            loads = json.loads
            for number, cells_json in records:
                cells = loads(cells_json)
                if len(cells) != width:
                    cells = (cells + [""] * width)[:width]
                rows.append(cells)
                source_rows.append(int(number))
            roles = _roles_from_dict(loads(roles_json), columns)
        except (ValueError, TypeError) as exc:
            log.warning("The stored database could not be read (%s)", exc)
            return None
        table = Table(source_name=source_name, sheet_name=sheet_name, columns=columns, rows=rows, source_rows=source_rows)
        return table, roles

    def save_roles(self, roles: Roles) -> None:
        """Replace the stored roles of the current dataset (no-op when there is none)."""
        with self._lock, _transaction(self._conn):
            row = self._conn.execute("SELECT columns_json FROM dataset WHERE id = 1").fetchone()
            if row is not None:
                payload = json.dumps(_roles_to_dict(roles, json.loads(row[0])), ensure_ascii=False)
                self._conn.execute("UPDATE dataset SET roles_json = ? WHERE id = 1", (payload,))

    def has_dataset(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1 FROM dataset WHERE id = 1").fetchone() is not None

    def clear_dataset(self) -> None:
        with self._lock, _transaction(self._conn):
            self._conn.execute("DELETE FROM dataset_rows")
            self._conn.execute("DELETE FROM dataset")

    # history ----------------------------------------------------------------------------
    def add_entries(self, entries: list[HistoryEntry]) -> list[HistoryEntry]:
        """Store entries, replacing any stored entry whose query normalizes to the same key.

        Returns the stored entries (new objects with ids), oldest first.  Within one call a
        later entry with the same key supersedes an earlier one.
        """
        latest: dict[str, HistoryEntry] = {}
        for entry in entries:
            key = query_key(entry.query)
            latest.pop(key, None)
            latest[key] = entry
        encode = json.JSONEncoder(ensure_ascii=False, separators=(",", ":")).encode
        prepared = [(key, history_entry_to_dict(entry)) for key, entry in latest.items()]
        stored: list[HistoryEntry] = []
        with self._lock, _transaction(self._conn):
            for key, data in prepared:
                self._conn.execute("DELETE FROM history WHERE query_key = ?", (key,))
                cursor = self._conn.execute(
                    "INSERT INTO history (query, query_key, created_at, payload_json) VALUES (?, ?, ?, ?)",
                    (data["query"], key, data["created_at"], encode(data)),
                )
                stored.append(history_entry_from_dict({**data, "id": cursor.lastrowid}))
        return stored

    def load_entries(self) -> list[HistoryEntry]:
        """Every stored entry, oldest (first stored) to newest."""
        with self._lock:
            records = self._conn.execute("SELECT id, query, created_at, payload_json FROM history ORDER BY id").fetchall()
        entries: list[HistoryEntry] = []
        for entry_id, query, created_at, payload in records:
            try:
                data = json.loads(payload)
                entries.append(history_entry_from_dict({**data, "id": entry_id, "query": query, "created_at": created_at}))
            except (ValueError, TypeError, AttributeError) as exc:
                log.warning("Skipping unreadable history entry %s (%s)", entry_id, exc)
        return entries

    def delete_entries(self, ids: list[int]) -> None:
        with self._lock, _transaction(self._conn):
            for chunk in _chunks([int(entry_id) for entry_id in ids]):
                marks = ",".join("?" * len(chunk))
                self._conn.execute(f"DELETE FROM history WHERE id IN ({marks})", chunk)

    def clear_entries(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM history")

    # embeddings -------------------------------------------------------------------------
    @property
    def embedding_cache(self) -> SqliteEmbeddingCache:
        return self._cache


# ------------------------------------------------------------------------------------------
# JSON (de)serializers for the persisted types


def query_key(query: str) -> str:
    """History identity of a query: its normalized form ("iShares  CORE" == "ishares core")."""
    return normalize(query) or query.strip()


def evidence_to_dict(evidence: FamilyEvidence) -> dict[str, Any]:
    return {
        "column": evidence.column,
        "value": evidence.value,
        "strength": float(evidence.strength),
        "matched_text": evidence.matched_text,
    }


def evidence_from_dict(data: dict[str, Any]) -> FamilyEvidence:
    return FamilyEvidence(
        column=str(data.get("column", "")),
        value=str(data.get("value", "")),
        strength=float(data.get("strength", 0.0)),
        matched_text=str(data.get("matched_text", "")),
    )


def result_row_to_dict(result: ResultRow) -> dict[str, Any]:
    return {
        "rank": int(result.rank),
        "score": float(result.score),
        "source_row": int(result.source_row),
        "cells": [str(cell) for cell in result.cells],
        "detail": _plain(result.detail),
    }


def result_row_from_dict(data: dict[str, Any]) -> ResultRow:
    detail = data.get("detail")
    return ResultRow(
        rank=int(data.get("rank", 0)),
        score=float(data.get("score", 0.0)),
        source_row=int(data.get("source_row", 0)),
        cells=[str(cell) for cell in data.get("cells", [])],
        detail=detail if isinstance(detail, dict) else {},
    )


def history_entry_to_dict(entry: HistoryEntry) -> dict[str, Any]:
    return {
        "version": 1,
        "id": entry.id,
        "query": entry.query,
        "created_at": entry.created_at,
        "dataset_label": entry.dataset_label,
        "columns": list(entry.columns),
        "deciding": list(entry.deciding),
        "primary_column": entry.primary_column,
        "evidence": [evidence_to_dict(evidence) for evidence in entry.evidence],
        "results": [result_row_to_dict(result) for result in entry.results],
    }


def history_entry_from_dict(data: dict[str, Any]) -> HistoryEntry:
    entry_id = data.get("id")
    return HistoryEntry(
        id=None if entry_id is None else int(entry_id),
        query=str(data.get("query", "")),
        created_at=str(data.get("created_at", "")),
        dataset_label=str(data.get("dataset_label", "")),
        columns=[str(name) for name in data.get("columns", [])],
        deciding=[str(name) for name in data.get("deciding", [])],
        primary_column=str(data.get("primary_column", "")),
        evidence=[evidence_from_dict(item) for item in data.get("evidence", [])],
        results=[result_row_from_dict(item) for item in data.get("results", [])],
    )


def _plain(value: Any) -> Any:
    """JSON-native copy of ``value`` (numpy scalars/arrays, tuples, sets and enums converted)."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return _plain(value.value)
    return value


def _encode_extra(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    plain = _plain(value)
    if plain is value:
        raise TypeError(f"{type(value).__name__} values cannot be stored as settings")
    return plain


def _decode_tagged(data: dict[str, Any]) -> Any:
    if len(data) == 1 and isinstance(data.get("__bytes__"), str):
        return base64.b64decode(data["__bytes__"])
    return data


def _roles_to_dict(roles: Roles, columns: Sequence[str]) -> dict[str, str]:
    return {name: Role(roles.get(name, Role.IGNORE)).value for name in columns}


def _roles_from_dict(data: dict[str, Any], columns: list[str]) -> Roles:
    valid = {role.value for role in Role}
    return {name: Role(data[name]) if data.get(name) in valid else Role.IGNORE for name in columns}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------------------------------
# SQLite plumbing


def _open_database(path: Path, prepare: Callable[[sqlite3.Connection], None]) -> sqlite3.Connection:
    """Open (creating if needed); move a corrupt or incompatible file aside and start fresh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _connect(path, prepare)
    except sqlite3.DatabaseError as exc:
        if not _is_corruption(exc):
            raise
        reason, label = str(exc), "corrupt"
    except _IncompatibleSchema as exc:
        reason, label = str(exc), "incompatible"
    moved = _move_aside(path, label)
    log.warning("%s could not be used (%s); moved it to %s and started a new one", path.name, reason, moved.name)
    return _connect(path, prepare)


def _connect(target: Path | str, prepare: Callable[[sqlite3.Connection], None]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(target), timeout=10.0, isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        prepare(conn)
    except BaseException:
        conn.close()
        raise
    return conn


class _IncompatibleSchema(Exception):
    """The file was written by a newer version whose tables this version cannot use."""


def _prepare_app(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    problems = [row[0] for row in conn.execute("PRAGMA quick_check")]
    if problems != ["ok"]:
        raise sqlite3.DatabaseError(f"integrity check failed: {problems[0][:200]}")
    if version > APP_SCHEMA_VERSION:
        # Written by a newer version: use it as-is when its tables still fit, never alter it.
        for table, required in _APP_COLUMNS.items():
            info = conn.execute(f"PRAGMA table_info({table})").fetchall()
            names = {column[1] for column in info}
            # an unknown NOT NULL column without default would make our inserts fail
            blocking = [column[1] for column in info if column[1] not in required and column[3] and column[4] is None]
            if not required <= names or blocking:
                raise _IncompatibleSchema(f"schema version {version} has an incompatible '{table}' table")
        return
    with _transaction(conn):
        for statement in _APP_SCHEMA.values():
            conn.execute(statement)
        conn.execute("CREATE INDEX IF NOT EXISTS history_query_key ON history (query_key)")
        conn.execute(f"PRAGMA user_version = {APP_SCHEMA_VERSION}")


def _prepare_cache(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    with _transaction(conn):
        if version not in (0, CACHE_SCHEMA_VERSION):
            conn.execute("DROP TABLE IF EXISTS vectors")  # unknown layout: a cache is disposable
        conn.execute(_CACHE_SCHEMA)
        conn.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _is_corruption(exc: sqlite3.DatabaseError) -> bool:
    code = getattr(exc, "sqlite_errorcode", None)
    if code is None:  # raised by this module after a failed integrity check
        return type(exc) is sqlite3.DatabaseError
    return code & 0xFF in _CORRUPTION_CODES


def _move_aside(path: Path, label: str) -> Path:
    """Rename ``path`` (and its -wal/-shm/-journal companions) to ``<name>.<label>-<timestamp>``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.name}.{label}-{stamp}")
    counter = 1
    while target.exists():
        counter += 1
        target = path.with_name(f"{path.name}.{label}-{stamp}-{counter}")
    path.replace(target)
    for suffix in ("-wal", "-shm", "-journal"):
        companion = path.with_name(path.name + suffix)
        if companion.exists():
            companion.replace(target.with_name(target.name + suffix))
    return target


def _cache_key(model_id: str, text: str) -> bytes:
    return hashlib.sha1(f"{model_id}\x00{text}".encode("utf-8", "surrogatepass")).digest()


def _decode_vectors(found: list[tuple[bytes, bytes]], positions: dict[bytes, list[int]]) -> dict[int, np.ndarray]:
    """float16 blobs -> float32 rows, one vectorized conversion per vector size."""
    by_size: dict[int, list[tuple[bytes, bytes]]] = {}
    for key, blob in found:
        if blob and len(blob) % 2 == 0:
            by_size.setdefault(len(blob), []).append((key, blob))
    vectors: dict[int, np.ndarray] = {}
    for size, items in by_size.items():
        matrix = np.frombuffer(b"".join(blob for _, blob in items), dtype="<f2").reshape(len(items), size // 2)
        for (key, _), row in zip(items, matrix.astype(np.float32)):
            for position in positions[key]:
                vectors[position] = row
    return vectors


def _chunks(items: list[Any], size: int = _CHUNK) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
