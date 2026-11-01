"""The search engine: builds an immutable index over a Table and ranks rows for queries."""

from __future__ import annotations

from dataclasses import dataclass

from .types import (
    CancelFn,
    EmbeddingCache,
    Embedder,
    HistoryEntry,
    ProgressFn,
    QueryOutcome,
    Roles,
    Table,
)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    semantic_weight: float = 0.5  # share of embedding similarity in the text score (rest is lexical)
    strictness: float = 0.85  # how hard a deciding-column conflict pushes a row down (0..1)
    semantic_floor: float = 0.30  # cosine mapped to 0 when calibrating embedding similarity
    use_family: bool = True  # False disables deciding-column detection, cleaning and penalties (ablation)


class SearchIndex:
    """Immutable after ``build``; ``search``/``search_many`` are safe to call from any thread."""

    table: Table
    roles: Roles
    deciding_columns: list[str]
    search_columns: list[str]
    primary_column: str
    has_semantic: bool

    @classmethod
    def build(
        cls,
        table: Table,
        roles: Roles,
        embedder: Embedder | None = None,
        cache: EmbeddingCache | None = None,
        config: EngineConfig = EngineConfig(),
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> "SearchIndex":
        raise NotImplementedError

    def search(self, query: str, k: int = 50) -> QueryOutcome:
        raise NotImplementedError

    def search_many(
        self,
        queries: list[str],
        k: int = 50,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> list[QueryOutcome]:
        raise NotImplementedError

    def learned_aliases(self) -> dict[str, dict[str, list[str]]]:
        """{deciding column: {value: [aliases learned from the data]}} for display."""
        raise NotImplementedError

    def to_history_entry(self, outcome: QueryOutcome, created_at: str) -> HistoryEntry:
        """Snapshot an outcome (all columns of each matched row) for persistence."""
        raise NotImplementedError
