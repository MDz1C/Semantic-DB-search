"""Excel export (ARCHITECTURE.md section 11, R13): sheet layout, numeric scores, formula-injection
safety, and writes that never leave a broken or half-written file behind."""

from __future__ import annotations

import os
import re
import sys
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from xml.etree import ElementTree

import pytest
from python_calamine import CalamineWorkbook

from semdb.core import export
from semdb.core.export import DETAIL_HEADERS, DETAILS_SHEET, SUMMARY_SHEET, ExportError, export_history
from semdb.core.types import FamilyEvidence, HistoryEntry, ResultRow

MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
RELATIONSHIP = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
BUILTIN_FORMATS = {0: "General", 9: "0%", 10: "0.00%"}
COLUMNS = ["ISIN", "Fund name", "Brokerage", "Currency"]


def _entry(
    query: str,
    results: list[tuple[list[str], float, int]],  # (cells, score, sheet row), best first
    *,
    columns: list[str] = COLUMNS,
    primary: str = "Fund name",
    evidence: list[FamilyEvidence] | None = None,
) -> HistoryEntry:
    rows = [
        ResultRow(rank, score, source_row, list(cells), {"semantic": score, "lexical": score, "text_score": score,
                                                         "family_factor": 1.0, "exact_id": None,
                                                         "agreements": [], "conflicts": []})
        for rank, (cells, score, source_row) in enumerate(results, start=1)
    ]  # fmt: skip
    return HistoryEntry(None, query, "2026-11-01T17:32:05+00:00", "funds.xlsx · Funds", list(columns),
                        ["ISIN", "Brokerage"], primary, evidence or [], rows)  # fmt: skip


@pytest.fixture
def entries() -> list[HistoryEntry]:
    ishares = _entry(
        "iShares Core S&P 500",
        [
            (["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF USD (Acc)", "BlackRock", "USD"], 0.95, 2),
            (["IE0031442068", "iShares Core S&P 500 UCITS ETF USD (Dist)", "BlackRock", "USD"], 0.93, 3),
            (["IE00B3XXRP09", "Vanguard S&P 500 UCITS ETF USD Distributing", "Vanguard", "USD"], 0.41, 6),
            (["IE00B6YX5C33", "SPDR S&P 500 UCITS ETF (Dist)", "State Street", "USD"], 0.38, 14),
        ],
        evidence=[FamilyEvidence("Brokerage", "BlackRock", 0.99, "iShares")],
    )
    vanguard = _entry(
        "Vanguard FTSE All-World",
        [
            (["IE00BK5BQT80", "Vanguard FTSE All-World UCITS ETF USD Accumulating", "Vanguard", "USD"], 0.97, 8),
            (["IE00B3RBWM25", "", "Vanguard", "USD"], 0.9, 9),  # no name: titled by its first filled cell
        ],
    )
    older = _entry(  # searched on a previous database with other columns
        "msci world", [(["iShares MSCI World", "0.20"], 0.7, 40)], columns=["Fund name", "TER"]
    )
    return [ishares, vanguard, older]


def _read(path: Path) -> dict[str, list[list]]:
    with CalamineWorkbook.from_path(str(path)) as workbook:
        return {name: workbook.get_sheet_by_name(name).to_python() for name in workbook.sheet_names}


def _export(folder: Path, entries: list[HistoryEntry], k: int) -> dict[str, list[list]]:
    target = folder / "results.xlsx"
    export_history(target, entries, k)
    return _read(target)


def _package(path: Path) -> tuple[dict[str, ElementTree.Element], ElementTree.Element]:
    """Parsed worksheets by sheet name, and the stylesheet, straight from the xlsx zip."""
    with zipfile.ZipFile(path) as package:
        workbook = ElementTree.fromstring(package.read("xl/workbook.xml"))
        relations = ElementTree.fromstring(package.read("xl/_rels/workbook.xml.rels"))
        targets = {relation.get("Id"): relation.get("Target").lstrip("/") for relation in relations}
        sheets = {}
        for sheet in workbook.iter(f"{MAIN}sheet"):
            part = targets[sheet.get(f"{RELATIONSHIP}id")]
            part = part if part.startswith("xl/") else "xl/" + part
            sheets[sheet.get("name")] = ElementTree.fromstring(package.read(part))
        return sheets, ElementTree.fromstring(package.read("xl/styles.xml"))


def _cell_style(path: Path, sheet: str, ref: str) -> tuple[str | None, str, bool]:
    """(XML type attribute, number format, bold) of one cell."""
    sheets, styles = _package(path)
    cell = next(cell for cell in sheets[sheet].iter(f"{MAIN}c") if cell.get("r") == ref)
    xf = list(styles.find(f"{MAIN}cellXfs"))[int(cell.get("s", "0"))]
    custom = {int(fmt.get("numFmtId")): fmt.get("formatCode") for fmt in styles.iter(f"{MAIN}numFmt")}
    format_id = int(xf.get("numFmtId", "0"))
    number_format = custom.get(format_id, BUILTIN_FORMATS.get(format_id, str(format_id)))
    font = list(styles.find(f"{MAIN}fonts"))[int(xf.get("fontId", "0"))]
    return cell.get("t"), number_format, font.find(f"{MAIN}b") is not None


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    path = tmp_path / "exports"
    path.mkdir()
    return path


# --- layout -----------------------------------------------------------------------------------


def test_summary_puts_the_top_matches_side_by_side(folder: Path, entries: list[HistoryEntry]) -> None:
    sheets = _export(folder, entries, k=3)
    assert list(sheets) == [SUMMARY_SHEET, DETAILS_SHEET]
    summary = sheets[SUMMARY_SHEET]
    assert summary[0] == ["Query", "Match 1", "Score 1", "Match 2", "Score 2", "Match 3", "Score 3"]
    assert summary[1:] == [
        ["iShares Core S&P 500",
         "iShares Core S&P 500 UCITS ETF USD (Acc)", pytest.approx(0.95),
         "iShares Core S&P 500 UCITS ETF USD (Dist)", pytest.approx(0.93),
         "Vanguard S&P 500 UCITS ETF USD Distributing", pytest.approx(0.41)],
        ["Vanguard FTSE All-World",
         "Vanguard FTSE All-World UCITS ETF USD Accumulating", pytest.approx(0.97),
         "IE00B3RBWM25", pytest.approx(0.9),
         "", ""],
        ["msci world", "iShares MSCI World", pytest.approx(0.7), "", "", "", ""],
    ]  # fmt: skip


@pytest.mark.parametrize(("k", "pairs"), [(1, 1), (2, 2), (4, 4), (50, 4), (0, 1)])
def test_k_sets_the_number_of_match_columns(folder: Path, entries: list[HistoryEntry], k: int, pairs: int) -> None:
    """k pairs of Match/Score columns, but never more than the longest list has (no empty pairs)."""
    sheets = _export(folder, entries, k)
    summary, details = sheets[SUMMARY_SHEET], sheets[DETAILS_SHEET]
    assert summary[0] == ["Query"] + [label for i in range(1, pairs + 1) for label in (f"Match {i}", f"Score {i}")]
    assert all(len(row) == 1 + 2 * pairs for row in summary)
    assert len(details) - 1 == sum(min(pairs, len(entry.results)) for entry in entries)


def test_details_lists_each_shown_result_with_every_column(folder: Path, entries: list[HistoryEntry]) -> None:
    details = _export(folder, entries, k=3)[DETAILS_SHEET]
    assert DETAIL_HEADERS == ["Query", "Rank", "Match %", "Sheet row"]
    assert details[0] == DETAIL_HEADERS + ["ISIN", "Fund name", "Brokerage", "Currency", "TER"]  # union of columns
    assert details[1:] == [
        ["iShares Core S&P 500", 1, pytest.approx(0.95), 2,
         "IE00B5BMR087", "iShares Core S&P 500 UCITS ETF USD (Acc)", "BlackRock", "USD", ""],
        ["iShares Core S&P 500", 2, pytest.approx(0.93), 3,
         "IE0031442068", "iShares Core S&P 500 UCITS ETF USD (Dist)", "BlackRock", "USD", ""],
        ["iShares Core S&P 500", 3, pytest.approx(0.41), 6,
         "IE00B3XXRP09", "Vanguard S&P 500 UCITS ETF USD Distributing", "Vanguard", "USD", ""],
        ["Vanguard FTSE All-World", 1, pytest.approx(0.97), 8,
         "IE00BK5BQT80", "Vanguard FTSE All-World UCITS ETF USD Accumulating", "Vanguard", "USD", ""],
        ["Vanguard FTSE All-World", 2, pytest.approx(0.9), 9, "IE00B3RBWM25", "", "Vanguard", "USD", ""],
        ["msci world", 1, pytest.approx(0.7), 40, "", "iShares MSCI World", "", "", "0.20"],  # cells follow names
    ]  # fmt: skip


def test_scores_are_numbers_formatted_as_percentages(folder: Path, entries: list[HistoryEntry]) -> None:
    target = folder / "results.xlsx"
    export_history(target, entries, k=3)
    sheets = _read(target)
    for sheet, row in ((SUMMARY_SHEET, sheets[SUMMARY_SHEET][1]), (DETAILS_SHEET, sheets[DETAILS_SHEET][1])):
        assert isinstance(row[2], float) and row[2] == pytest.approx(0.95), sheet  # 0.95, not "95%"
        cell_type, number_format, _bold = _cell_style(target, sheet, "C2")
        assert (cell_type, number_format) == (None, "0%"), sheet  # a number shown as a percentage
    rank_type, rank_format, _ = _cell_style(target, DETAILS_SHEET, "B2")
    assert (rank_type, rank_format) == (None, "General")


def test_header_is_bold_frozen_and_filterable(folder: Path, entries: list[HistoryEntry]) -> None:
    target = folder / "results.xlsx"
    export_history(target, entries, k=3)
    sheets, _styles = _package(target)
    for name, sheet in sheets.items():
        pane = sheet.find(f".//{MAIN}pane")
        assert pane is not None and pane.get("state") == "frozen" and pane.get("ySplit") == "1", name
        assert sheet.find(f"{MAIN}autoFilter") is not None, name
        assert _cell_style(target, name, "A1")[2], name


def test_empty_history_still_makes_a_valid_workbook(folder: Path) -> None:
    sheets = _export(folder, [], k=3)
    assert sheets[SUMMARY_SHEET] == [["Query", "Match 1", "Score 1", "Match 2", "Score 2", "Match 3", "Score 3"]]
    assert sheets[DETAILS_SHEET] == [DETAIL_HEADERS]


# --- formula injection --------------------------------------------------------------------------


def test_formula_like_text_stays_inert_text(folder: Path) -> None:
    payloads = [["@SUM(1)", "+cmd|' /C calc'!A0", "-2+3"],
                ['=HYPERLINK("http://evil.example","click")', "00123", "https://evil.example"]]  # fmt: skip
    entry = _entry("=1+1", [(payloads[0], 0.9, 2), (payloads[1], 0.8, 3)], columns=["=A1", "Fund name", "-Note"])
    target = folder / "results.xlsx"
    export_history(target, [entry], k=5)

    sheets = _read(target)  # every value comes back as the exact text, never evaluated
    assert sheets[SUMMARY_SHEET][1] == ["=1+1", "+cmd|' /C calc'!A0", pytest.approx(0.9), "00123", pytest.approx(0.8)]
    details = sheets[DETAILS_SHEET]
    assert details[0][4:] == ["=A1", "Fund name", "-Note"]
    assert [row[0] for row in details[1:]] == ["=1+1", "=1+1"]
    assert [row[4:] for row in details[1:]] == payloads

    with zipfile.ZipFile(target) as package:
        parts = [name for name in package.namelist() if name.startswith("xl/worksheets/")]
        assert len(parts) == 2
        for part in parts:
            xml = package.read(part).decode("utf-8")
            assert not re.search(r"<f[\s>/]", xml), f"formula element in {part}"
            assert "<hyperlink" not in xml, f"hyperlink in {part}"
    for sheet in _package(target)[0].values():
        assert {cell.get("t") for cell in sheet.iter(f"{MAIN}c")} <= {None, "s", "inlineStr"}  # text or number


# --- failures: never a broken or half-written file ----------------------------------------------


def _leftovers(folder: Path) -> list[str]:
    return [name for name in os.listdir(folder) if "partial" in name]


def test_export_replaces_an_existing_file_and_leaves_no_partial(folder: Path, entries: list[HistoryEntry]) -> None:
    target = folder / "results.xlsx"
    target.write_bytes(b"previous workbook")
    export_history(target, entries, k=3)
    assert os.listdir(folder) == ["results.xlsx"]
    assert _read(target)[SUMMARY_SHEET][1][0] == "iShares Core S&P 500"


@contextmanager
def _exclusive_handle(path: Path) -> Iterator[None]:
    """Hold ``path`` open with no sharing allowed (Windows), like a workbook open in Excel."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]  # fmt: skip
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    generic_read, no_sharing, open_existing, normal = 0x80000000, 0, 3, 0x80
    handle = kernel32.CreateFileW(str(path), generic_read, no_sharing, None, open_existing, normal, None)
    if handle in (None, ctypes.c_void_p(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform != "win32", reason="exclusive file locks are Windows behaviour")
def test_target_locked_by_another_program_raises_export_error(folder: Path, entries: list[HistoryEntry]) -> None:
    target = folder / "results.xlsx"
    target.write_bytes(b"previous workbook")
    with _exclusive_handle(target), pytest.raises(ExportError) as caught:
        export_history(target, entries, k=3)
    assert isinstance(caught.value, OSError)
    assert "results.xlsx" in str(caught.value)  # the message names the file to close
    assert target.read_bytes() == b"previous workbook"
    assert os.listdir(folder) == ["results.xlsx"]


@pytest.mark.parametrize("problem", ["target is a folder", "folder does not exist"])
def test_unwritable_target_raises_export_error(folder: Path, entries: list[HistoryEntry], problem: str) -> None:
    if problem == "target is a folder":
        target = folder / "results.xlsx"
        target.mkdir()
    else:
        target = folder / "missing" / "results.xlsx"
    with pytest.raises(ExportError, match=r"results\.xlsx"):
        export_history(target, entries, k=3)
    assert _leftovers(folder) == []


def test_failure_while_writing_keeps_the_previous_file(
    folder: Path, entries: list[HistoryEntry], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = folder / "results.xlsx"
    target.write_bytes(b"previous workbook")

    def disk_full(*_args: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(export, "_write_details", disk_full)
    with pytest.raises(ExportError):
        export_history(target, entries, k=3)
    assert target.read_bytes() == b"previous workbook"
    assert os.listdir(folder) == ["results.xlsx"]


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to delete a file that is still open")
def test_disk_full_while_saving_raises_export_error_and_leaves_no_partial(
    folder: Path, entries: list[HistoryEntry], monkeypatch: pytest.MonkeyPatch
) -> None:
    target = folder / "results.xlsx"
    target.write_bytes(b"previous workbook")

    def disk_full(*_args: object, **_kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(zipfile.ZipFile, "write", disk_full)  # xlsxwriter zips the parts while saving
    monkeypatch.setattr(zipfile.ZipFile, "writestr", disk_full)
    with pytest.raises(ExportError):
        export_history(target, entries, k=3)
    assert target.read_bytes() == b"previous workbook"
    assert os.listdir(folder) == ["results.xlsx"]
