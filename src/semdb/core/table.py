"""Reading spreadsheet files into :class:`Table` objects (offline, no macros, no formulas run).

Excel-family formats are parsed with python-calamine (Rust, memory-safe, never executes
macros); CSV/TSV/TXT with the standard library.  Every cell becomes a display string.
"""

from __future__ import annotations

from pathlib import Path

from .types import ProgressFn, CancelFn, Table

EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xlsb", ".xls", ".ods")
TEXT_EXTENSIONS = (".csv", ".tsv", ".txt")
SUPPORTED_EXTENSIONS = EXCEL_EXTENSIONS + TEXT_EXTENSIONS


class TableError(Exception):
    """User-facing import problem (unsupported type, password-protected, empty sheet...)."""


def list_sheets(path: str | Path) -> list[str]:
    """Visible sheet names in workbook order (``[""]`` for text files)."""
    raise NotImplementedError


def read_raw(path: str | Path, sheet: str | None = None, max_rows: int | None = None) -> list[list[str]]:
    """First ``max_rows`` raw rows of a sheet as display strings (no header handling), for previews."""
    raise NotImplementedError


def detect_header_row(raw_rows: list[list[str]]) -> int:
    """0-based index of the most plausible header row within the first rows of a sheet."""
    raise NotImplementedError


def read_table(
    path: str | Path,
    sheet: str | None = None,
    header_row: int | None = None,
    progress: ProgressFn | None = None,
    cancel: CancelFn | None = None,
) -> Table:
    """Read a whole sheet. ``header_row`` None means auto-detect. Raises TableError."""
    raise NotImplementedError
