"""Exporting results to an Excel workbook (xlsxwriter; strings are never turned into formulas)."""

from __future__ import annotations

from pathlib import Path

from .types import HistoryEntry


def export_history(path: str | Path, entries: list[HistoryEntry], k: int) -> None:
    """Write a "Summary" sheet (query, then the top-k matches side by side) and a "Details"
    sheet (one row per query x result with every column).  Entries are written in the given order."""
    raise NotImplementedError
