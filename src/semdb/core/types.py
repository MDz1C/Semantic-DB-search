"""Shared data types: the contract between the import, search, storage and UI layers.

Every module exchanges these types; none of them may change shape without updating
all layers (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import numpy as np

# (stage message, fraction done in [0, 1]); called from worker threads.
ProgressFn = Callable[[str, float], None]
# Returns True when the user asked to stop; long operations poll it and raise Cancelled.
CancelFn = Callable[[], bool]


class Cancelled(Exception):
    """Raised by long-running operations when their cancel callback returns True."""


class Role(str, Enum):
    """What a column means for matching."""

    DECIDING = "deciding"  # identity (brokerage, issuer, ISIN...): results must agree with the query
    SEARCH = "search"  # describes the item (fund name...): compared semantically and lexically
    IGNORE = "ignore"  # shown in results only

    @property
    def label(self) -> str:
        return {"deciding": "Deciding", "search": "Search", "ignore": "Display only"}[self.value]


Roles = dict[str, Role]


@dataclass(slots=True)
class Table:
    """A rectangular sheet. ``rows[i][j]`` is the display string of column j ('' when empty)."""

    source_name: str  # file name without directory, e.g. "funds.xlsx"
    sheet_name: str  # '' for CSV files
    columns: list[str]  # unique, non-empty header labels
    rows: list[list[str]]  # every row has exactly len(columns) cells
    source_rows: list[int]  # 1-based row number in the original sheet, one per row

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def label(self) -> str:
        return f"{self.source_name} · {self.sheet_name}" if self.sheet_name else self.source_name

    def column_index(self, name: str) -> int:
        return self.columns.index(name)

    def column_values(self, name: str) -> list[str]:
        j = self.columns.index(name)
        return [row[j] for row in self.rows]

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update("\x1f".join(self.columns).encode("utf-8"))
        for row in self.rows:
            digest.update(b"\x1e")
            digest.update("\x1f".join(row).encode("utf-8"))
        return digest.hexdigest()


@dataclass(slots=True)
class ColumnProfile:
    """Statistics about one column, used to suggest roles and to explain them in the UI."""

    name: str
    index: int
    non_empty: int
    unique: int
    avg_len: float
    numeric_ratio: float  # share of non-empty cells that parse as numbers
    date_ratio: float  # share of non-empty cells that look like dates
    is_identifier: bool  # code-like and (almost) unique per row: ISIN, ticker, id
    suggested_role: Role
    reason: str  # one short sentence shown next to the suggestion
    samples: list[str] = field(default_factory=list)  # up to 5 distinct example values


@dataclass(slots=True)
class FamilyEvidence:
    """A deciding-column value the query appears to name."""

    column: str
    value: str  # display form of the detected value, e.g. "BlackRock"
    strength: float  # confidence in [0, 1]
    matched_text: str  # query fragment that triggered it, e.g. "iShares"


@dataclass(slots=True)
class Match:
    """One ranked row for one query, with the evidence behind its score."""

    row: int  # index into Table.rows
    score: float  # final score in [0, 1] = text_score * family_factor
    text_score: float  # descriptive similarity in [0, 1]
    semantic: float  # calibrated embedding similarity in [0, 1] (0 when no model)
    lexical: float  # character/word TF-IDF cosine in [0, 1]
    family_factor: float  # multiplier in (0, 1]; 1 when deciding columns agree or are silent
    exact_id: str | None = None  # identifier column that matched the query exactly
    agreements: list[str] = field(default_factory=list)  # deciding columns agreeing with the query
    conflicts: list[str] = field(default_factory=list)  # deciding columns contradicting the query


@dataclass(slots=True)
class QueryOutcome:
    query: str
    cleaned_query: str  # query with detected identity fragments removed
    evidence: list[FamilyEvidence]
    matches: list[Match]  # best first


@dataclass(slots=True)
class ResultRow:
    """A frozen snapshot of one result, so history survives re-imports."""

    rank: int  # 1-based
    score: float
    source_row: int  # row number in the original sheet
    cells: list[str]  # aligned with HistoryEntry.columns
    detail: dict  # {"semantic", "lexical", "text_score", "family_factor", "exact_id", "agreements", "conflicts"}


@dataclass(slots=True)
class HistoryEntry:
    """One query and its results, as persisted and displayed."""

    id: int | None
    query: str
    created_at: str  # ISO 8601 UTC, e.g. "2026-11-01T17:32:05.123456+00:00"
    dataset_label: str  # Table.label at search time
    columns: list[str]  # Table.columns at search time
    deciding: list[str]  # deciding columns at search time
    primary_column: str  # column used as the result title (first search column)
    evidence: list[FamilyEvidence]
    results: list[ResultRow]  # up to MAX_STORED_RESULTS, best first


MAX_STORED_RESULTS = 50


class Embedder(Protocol):
    """Sentence-embedding model. Implementations must be thread-safe for ``encode``."""

    model_id: str  # stable id used as embedding-cache key prefix
    dim: int

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        """Return float32 array (len(texts), dim), rows L2-normalized; progress(done, total)."""
        ...


class EmbeddingCache(Protocol):
    """Persistent text -> vector cache, so unchanged rows are never re-embedded."""

    def get_many(self, model_id: str, texts: list[str]) -> dict[int, np.ndarray]:
        """Map position in ``texts`` -> cached float32 vector, for the texts that are cached."""
        ...

    def put_many(self, model_id: str, texts: list[str], vectors: np.ndarray) -> None: ...
