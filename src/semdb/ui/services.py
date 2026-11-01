"""The UI's gateway to the search core, file import and export.

:class:`Backend` uses the real modules; tests and ``tools/ui_screenshots.py`` pass objects
with the same methods instead, so every window flow can run against fakes.  Core modules
are imported lazily so the window can be built before the core is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..core.types import (
    CancelFn,
    Cancelled,
    ColumnProfile,
    EmbeddingCache,
    Embedder,
    HistoryEntry,
    ProgressFn,
    QueryOutcome,
    Roles,
    Table,
)


class SearchIndexLike(Protocol):
    """What the UI uses of ``core.engine.SearchIndex``."""

    table: Table
    roles: Roles
    deciding_columns: list[str]
    search_columns: list[str]
    primary_column: str
    has_semantic: bool

    def search_many(
        self,
        queries: list[str],
        k: int = 50,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> list[QueryOutcome]: ...

    def to_history_entry(self, outcome: QueryOutcome, created_at: str) -> HistoryEntry: ...

    def learned_aliases(self) -> dict[str, dict[str, list[str]]]: ...


class StoreLike(Protocol):
    """What the UI uses of ``storage.Store``."""

    def get_setting(self, key: str, default: Any = None) -> Any: ...
    def set_setting(self, key: str, value: Any) -> None: ...
    def save_dataset(self, table: Table, roles: Roles) -> None: ...
    def load_dataset(self) -> tuple[Table, Roles] | None: ...
    def save_roles(self, roles: Roles) -> None: ...
    def add_entries(self, entries: list[HistoryEntry]) -> list[HistoryEntry]: ...
    def load_entries(self) -> list[HistoryEntry]: ...
    def delete_entries(self, ids: list[int]) -> None: ...
    def clear_entries(self) -> None: ...

    @property
    def embedding_cache(self) -> EmbeddingCache | None: ...


@dataclass(slots=True)
class ImportPreview:
    """Everything the import dialog shows for one file/sheet/header choice."""

    path: Path
    sheets: list[str]  # [""] for text files
    sheet: str
    header_row: int  # 0-based
    table: Table
    profiles: list[ColumnProfile]
    roles: Roles


class Backend:
    """Default implementation backed by ``semdb.core``."""

    def supported_extensions(self) -> tuple[str, ...]:
        from ..core.table import SUPPORTED_EXTENSIONS

        return SUPPORTED_EXTENSIONS

    def sample_workbook(self) -> Path:
        from .. import paths

        return paths.sample_workbook()

    def list_sheets(self, path: Path) -> list[str]:
        from ..core.table import list_sheets

        return list_sheets(path)

    def read_raw(self, path: Path, sheet: str | None, max_rows: int) -> list[list[str]]:
        from ..core.table import read_raw

        return read_raw(path, sheet, max_rows)

    def detect_header_row(self, raw_rows: list[list[str]]) -> int:
        from ..core.table import detect_header_row

        return detect_header_row(raw_rows)

    def read_table(
        self,
        path: Path,
        sheet: str | None,
        header_row: int | None,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> Table:
        from ..core.table import read_table

        return read_table(path, sheet, header_row, progress, cancel)

    def profile_columns(self, table: Table) -> list[ColumnProfile]:
        from ..core.profile import profile_columns

        return profile_columns(table)

    def suggest_roles(self, table: Table, profiles: list[ColumnProfile]) -> Roles:
        from ..core.profile import suggest_roles

        return suggest_roles(table, profiles)

    def load_embedder(self) -> Embedder | None:
        from ..core.embedder import load_default_embedder

        return load_default_embedder()

    def build_index(
        self,
        table: Table,
        roles: Roles,
        embedder: Embedder | None,
        cache: EmbeddingCache | None,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> SearchIndexLike:
        from ..core.engine import EngineConfig, SearchIndex

        return SearchIndex.build(table, roles, embedder, cache, EngineConfig(), progress, cancel)

    def export_history(self, path: Path, entries: list[HistoryEntry], k: int) -> None:
        from ..core.export import export_history

        export_history(path, entries, k)

    def inspect(
        self,
        path: Path,
        sheet: str | None = None,
        header_row: int | None = None,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> ImportPreview:
        """Read ``path`` for the import dialog: sheets, header row, full table, profiles, roles."""
        from ..core.table import TableError

        report = progress or (lambda _message, _fraction: None)
        report("Opening file…", 0.0)
        sheets = self.list_sheets(path)
        if not sheets:
            raise TableError("This workbook has no visible sheets.")
        chosen = sheet if sheet in sheets else sheets[0]
        sheet_arg = chosen or None
        if header_row is None:
            header_row = self.detect_header_row(self.read_raw(path, sheet_arg, 30))
        table = self.read_table(path, sheet_arg, header_row, progress, cancel)
        if cancel and cancel():
            raise Cancelled()
        report("Analyzing columns…", 1.0)
        profiles = self.profile_columns(table)
        roles = self.suggest_roles(table, profiles)
        return ImportPreview(Path(path), sheets, chosen, header_row, table, profiles, roles)
