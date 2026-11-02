"""Exporting results to an Excel workbook (xlsxwriter; strings are never turned into formulas).

Every text cell is written with ``write_string`` and the workbook is created with
``strings_to_formulas/urls/numbers`` off, so a cell such as ``=HYPERLINK(...)`` or
``+cmd|' /C calc'!A0`` stays inert text (no formula injection).  The workbook is assembled
in memory, written to a temporary file next to the target through a handle this module
controls, and moved into place, so a failed export (disk full, file open in Excel) never
leaves a half-written file behind.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
from pathlib import Path

import xlsxwriter
from xlsxwriter.exceptions import XlsxFileError
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet

from .types import HistoryEntry

WORKBOOK_OPTIONS = {"strings_to_formulas": False, "strings_to_urls": False, "strings_to_numbers": False}
SUMMARY_SHEET = "Summary"
DETAILS_SHEET = "Details"
DETAIL_HEADERS = ["Query", "Rank", "Match %", "Sheet row"]

_MAX_CELL_CHARS = 32_767  # Excel's limit for one cell
_MIN_WIDTH, _MAX_WIDTH, _SCORE_WIDTH = 8, 60, 9


class ExportError(OSError):
    """The workbook could not be written (folder not writable, file open in Excel...)."""


def export_history(path: str | Path, entries: list[HistoryEntry], k: int) -> None:
    """Write a "Summary" sheet (query, then the top-k matches side by side) and a "Details"
    sheet (one row per query x result with every column).  Entries are written in the given order."""
    target = Path(path)
    k = max(1, int(k))
    if entries:  # no trailing empty Match/Score pairs when every list is shorter than k
        k = max(1, min(k, max(len(entry.results) for entry in entries)))
    try:
        data = _workbook_bytes(entries, k)
    except (OSError, XlsxFileError, ValueError) as exc:
        raise ExportError(f"Could not create the workbook for '{target.name}' ({exc}).") from exc
    partial = target.with_name(f".{target.stem}.{os.getpid()}.partial.xlsx")
    try:
        with open(partial, "wb") as handle:
            handle.write(data)
        os.replace(partial, target)
    except OSError as exc:
        with contextlib.suppress(OSError):  # our handle is closed, so nothing holds the file
            partial.unlink(missing_ok=True)
        reason = f" ({exc.strerror})" if exc.strerror else ""
        raise ExportError(
            f"Could not save '{target.name}'{reason}. If it is open in Excel, close it and try "
            "again; otherwise choose a folder you can write to."
        ) from exc


def _workbook_bytes(entries: list[HistoryEntry], k: int) -> bytes:
    buffer = io.BytesIO()
    with xlsxwriter.Workbook(buffer, {**WORKBOOK_OPTIONS, "in_memory": True}) as workbook:
        workbook.set_properties({"title": "Search results", "comments": "Exported by Fundscope"})
        header = workbook.add_format(
            {"bold": True, "bg_color": "#EEF2FF", "font_color": "#111827", "bottom": 1, "bottom_color": "#CDD2DC"}
        )
        percent = workbook.add_format({"num_format": "0%"})
        _write_summary(workbook.add_worksheet(SUMMARY_SHEET), entries, k, header, percent)
        _write_details(workbook.add_worksheet(DETAILS_SHEET), entries, k, header, percent)
    return buffer.getvalue()


def _xml_safe(value: str) -> str:
    """Lone surrogates (from broken clipboard text) cannot be stored in XML: replace them."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return value.encode("utf-8", "replace").decode("utf-8")
    return value


def _write_summary(sheet: Worksheet, entries: list[HistoryEntry], k: int, header: Format, percent: Format) -> None:
    headers = ["Query"] + [label for i in range(1, k + 1) for label in (f"Match {i}", f"Score {i}")]
    widths = _write_header(sheet, headers, header)
    for row, entry in enumerate(entries, start=1):
        _text(sheet, row, 0, entry.query, widths)
        primary = entry.columns.index(entry.primary_column) if entry.primary_column in entry.columns else 0
        for i, result in enumerate(entry.results[:k]):
            title = result.cells[primary] if primary < len(result.cells) else ""
            _text(sheet, row, 1 + 2 * i, title or next((cell for cell in result.cells if cell), ""), widths)
            _score(sheet, row, 2 + 2 * i, result.score, percent)
    for col in range(2, len(headers), 2):
        widths[col] = _SCORE_WIDTH
    _finish(sheet, len(entries), widths)


def _write_details(sheet: Worksheet, entries: list[HistoryEntry], k: int, header: Format, percent: Format) -> None:
    columns: dict[str, int] = {}  # union of the entries' columns, first-seen order -> sheet column
    for entry in entries:
        for name in entry.columns:
            columns.setdefault(name, len(DETAIL_HEADERS) + len(columns))
    widths = _write_header(sheet, DETAIL_HEADERS + list(columns), header)
    widths[2] = _SCORE_WIDTH
    row = 0
    for entry in entries:
        targets = [columns[name] for name in entry.columns]
        for result in entry.results[:k]:
            row += 1
            _text(sheet, row, 0, entry.query, widths)
            sheet.write_number(row, 1, result.rank)
            _score(sheet, row, 2, result.score, percent)
            sheet.write_number(row, 3, result.source_row)
            for col, cell in zip(targets, result.cells):
                _text(sheet, row, col, cell, widths)
    _finish(sheet, row, widths)


def _write_header(sheet: Worksheet, headers: list[str], style: Format) -> list[int]:
    for col, label in enumerate(headers):
        sheet.write_string(0, col, _xml_safe(label[:_MAX_CELL_CHARS]), style)
    return [len(label) for label in headers]


def _text(sheet: Worksheet, row: int, col: int, value: str, widths: list[int]) -> None:
    if value:
        sheet.write_string(row, col, _xml_safe(value[:_MAX_CELL_CHARS]))
        widths[col] = max(widths[col], min(len(value), _MAX_WIDTH))


def _score(sheet: Worksheet, row: int, col: int, score: float, percent: Format) -> None:
    value = float(score)
    if math.isfinite(value):
        sheet.write_number(row, col, value, percent)


def _finish(sheet: Worksheet, last_row: int, widths: list[int]) -> None:
    """Frozen header row (and first column), autofilter, widths fitted to the content."""
    sheet.freeze_panes(1, 1)
    sheet.autofilter(0, 0, last_row, len(widths) - 1)
    for col, width in enumerate(widths):
        sheet.set_column(col, col, min(max(width + 2, _MIN_WIDTH), _MAX_WIDTH))
