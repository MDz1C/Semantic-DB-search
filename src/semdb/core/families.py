"""Deciding columns: value vocabularies, automatically learned aliases, query detection and
row-local cleaning of identity fragments.  See docs/ARCHITECTURE.md, "Family model"."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import FamilyEvidence, Table


@dataclass(slots=True)
class DecidingColumn:
    name: str
    values: list[str]  # distinct display values, id = position
    codes: np.ndarray  # int32 per table row: value id, -1 when the cell is empty
    is_identifier: bool  # code-like and nearly unique per row (ISIN, ticker)
    indicators: dict[str, list[tuple[int, float]]]  # normalized token/phrase -> [(value id, weight)]
    learned_aliases: dict[int, list[str]] = field(default_factory=dict)  # value id -> mined alias tokens


@dataclass(slots=True)
class Detection:
    """What the query says about identity."""

    evidence: dict[str, dict[int, float]]  # column -> {value id: strength in (0, 1]}
    spans: list[tuple[int, int]]  # character spans of the query to drop before text matching
    found: list[FamilyEvidence]  # human-readable, strongest first


class FamilyModel:
    """Built once per (table, roles); immutable afterwards and safe to share across threads."""

    columns: list[DecidingColumn]

    @classmethod
    def build(cls, table: Table, deciding: list[str], search: list[str]) -> "FamilyModel":
        raise NotImplementedError

    def clean_cells(self, row: int, cells: list[str]) -> list[str]:
        """Remove high-confidence repeats of the row's own deciding values from its search cells."""
        raise NotImplementedError

    def detect(self, query: str) -> Detection:
        raise NotImplementedError

    def family_factor(self, detection: Detection, strictness: float) -> np.ndarray:
        """float32 (n_rows,) multipliers in (0, 1]: 1 = agrees or no evidence, low = conflicts."""
        raise NotImplementedError
