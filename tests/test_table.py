"""Importing spreadsheets: sheets, header detection, cell conversion, text files, errors, speed."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import pytest
import xlsxwriter

from semdb.core.table import (
    READING_STAGE,
    TableError,
    cell_text,
    detect_header_row,
    is_date_text,
    is_number_text,
    list_sheets,
    read_raw,
    read_table,
)
from semdb.core.types import Cancelled


def _funds_workbook(path: Path) -> Path:
    """Title row, blank row, then a header with a duplicate, an empty label and an empty column."""
    workbook = xlsxwriter.Workbook(str(path))
    date = workbook.add_format({"num_format": "yyyy-mm-dd"})
    stamp = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm"})
    clock = workbook.add_format({"num_format": "hh:mm:ss"})
    funds = workbook.add_worksheet("Funds")
    workbook.add_worksheet("Secret").hide()
    workbook.add_worksheet("Archive").very_hidden()
    notes = workbook.add_worksheet("Notes")
    notes.write_row(0, 0, ["Note", "Author"])
    notes.write_row(1, 0, ["Check TER", "Ana"])

    funds.write(0, 0, "Fund universe — exported 2026-01-31")
    header = ["ISIN", "Fund name", "Brokerage", "", "TER", "Launch", "Listed", "Fund name", "", "Updated", "At"]
    funds.write_row(2, 0, header)
    rows = [
        ["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF", "BlackRock", "x", 0.07, dt.date(2010, 5, 19), True, "Core S&P"],
        ["FR0010315770", "Amundi MSCI World — Société\nGénérale", "Amundi", "", 0.3, dt.date(2006, 4, 18), False, "Monde"],
        None,  # an empty row in the middle is skipped
        ["IE00B3XXRP09", "Vanguard S&P 500 UCITS ETF", "Vanguard", "", 500.0, dt.date(2012, 5, 22), True, ""],
    ]
    for offset, values in enumerate(rows, start=3):
        if values is None:
            continue
        for col, value in enumerate(values):
            if isinstance(value, dt.date):
                funds.write_datetime(offset, col, dt.datetime.combine(value, dt.time()), date)
            elif isinstance(value, bool):
                funds.write_boolean(offset, col, value)
            elif isinstance(value, float):
                funds.write_number(offset, col, value)
            elif value:
                funds.write_string(offset, col, value)
    funds.write_datetime(3, 9, dt.datetime(2026, 1, 31, 13, 45), stamp)
    funds.write_datetime(3, 10, dt.time(9, 30, 15), clock)
    workbook.close()
    return path


@pytest.fixture
def funds_xlsx(tmp_path: Path) -> Path:
    return _funds_workbook(tmp_path / "funds.xlsx")


# ------------------------------------------------------------------------------- workbooks


def test_lists_only_visible_worksheets_in_order(funds_xlsx: Path) -> None:
    assert list_sheets(funds_xlsx) == ["Funds", "Notes"]


def test_reads_workbook_with_title_row_duplicates_and_empty_headers(funds_xlsx: Path) -> None:
    table = read_table(funds_xlsx)
    assert table.source_name == "funds.xlsx"
    assert table.sheet_name == "Funds"
    assert table.label == "funds.xlsx · Funds"
    # 'Column 9' (empty label, no data) is dropped; the empty label with data is named by position
    assert table.columns == [
        "ISIN", "Fund name", "Brokerage", "Column 4", "TER", "Launch", "Listed", "Fund name (2)", "Updated", "At",
    ]  # fmt: skip
    assert table.source_rows == [4, 5, 7]
    assert all(len(row) == len(table.columns) for row in table.rows)
    first, second, third = table.rows
    assert first == [
        "IE00B5BMR087", "iShares Core S&P 500 UCITS ETF", "BlackRock", "x", "0.07", "2010-05-19", "TRUE",
        "Core S&P", "2026-01-31 13:45", "09:30:15",
    ]  # fmt: skip
    assert second[1] == "Amundi MSCI World — Société Générale"  # accents kept, newline -> space
    assert second[4:7] == ["0.3", "2006-04-18", "FALSE"]
    assert third[4] == "500"  # integral float without ".0"


def test_header_row_can_be_chosen_and_validated(funds_xlsx: Path) -> None:
    table = read_table(funds_xlsx, header_row=3)  # use the first data row as header
    assert table.columns[0] == "IE00B5BMR087"
    assert table.source_rows == [5, 7]
    with pytest.raises(TableError, match="outside"):
        read_table(funds_xlsx, header_row=99)


def test_reads_other_sheets_including_hidden_on_request(funds_xlsx: Path) -> None:
    notes = read_table(funds_xlsx, sheet="Notes")
    assert notes.columns == ["Note", "Author"] and notes.rows == [["Check TER", "Ana"]]
    with pytest.raises(TableError, match="no sheet named"):
        read_table(funds_xlsx, sheet="Nope")
    with pytest.raises(TableError, match="no data"):
        read_table(funds_xlsx, sheet="Secret")


def test_read_raw_rows_are_aligned_with_sheet_rows(funds_xlsx: Path) -> None:
    raw = read_raw(funds_xlsx, max_rows=4)
    assert len(raw) == 4
    assert raw[0][0] == "Fund universe — exported 2026-01-31"
    assert raw[1] == [""] * len(raw[1])
    assert raw[2][:3] == ["ISIN", "Fund name", "Brokerage"]
    assert len({len(row) for row in raw}) == 1
    assert detect_header_row(read_raw(funds_xlsx, max_rows=60)) == 2
    assert read_raw(funds_xlsx, max_rows=0) == []


def test_data_starting_away_from_a1(tmp_path: Path) -> None:
    path = tmp_path / "offset.xlsx"
    workbook = xlsxwriter.Workbook(str(path))
    sheet = workbook.add_worksheet()
    sheet.write_row(4, 2, ["Name", "", "Kind"])
    sheet.write_row(5, 2, ["Alpha", "a", "x"])
    sheet.write_row(6, 2, ["Beta", "", "y"])
    workbook.close()
    raw = read_raw(path)
    assert len(raw) == 7 and raw[4] == ["Name", "", "Kind"]  # leading empty columns left out
    table = read_table(path)
    assert table.columns == ["Name", "Column 4", "Kind"]  # 'Column 4' = sheet column D
    assert table.source_rows == [6, 7]


def test_cell_conversion_rules() -> None:
    assert cell_text(None) == ""
    assert cell_text("  padded \r\n  text\t here ") == "padded text here"
    assert cell_text(True) == "TRUE" and cell_text(False) == "FALSE"
    assert cell_text(42) == "42"
    assert cell_text(500.0) == "500"
    assert cell_text(-3.0) == "-3"
    assert cell_text(0.07) == "0.07"
    assert cell_text(0.1 + 0.2) == "0.3"
    assert cell_text(1.5e-07) == "0.00000015"
    assert cell_text(4006381333931.0) == "4006381333931"
    assert cell_text(dt.datetime(2024, 1, 31)) == "2024-01-31"
    assert cell_text(dt.datetime(2024, 1, 31, 13, 45)) == "2024-01-31 13:45"
    assert cell_text(dt.datetime(2024, 1, 31, 13, 44, 59, 999_000)) == "2024-01-31 13:45"
    assert cell_text(dt.date(1999, 12, 31)) == "1999-12-31"
    assert cell_text(dt.time(9, 30)) == "09:30"
    assert cell_text(dt.time(9, 30, 15)) == "09:30:15"
    assert cell_text(dt.timedelta(hours=36)) == "36:00"


def test_number_and_date_recognition() -> None:
    for value in ["1,234.56", "1.234,56", "0,07", "-3", "12 %", "(1.5)", "$100", "1e5", "2024", "1 234"]:
        assert is_number_text(value), value
    for value in ["IE00B5BMR087", "S&P 500", "3yr", "-", "1.2.3", ""]:
        assert not is_number_text(value), value
    for value in ["2024-01-31", "2024-01-31 13:45", "31/01/2024", "31 Jan 2024", "Jan 31, 2024", "13:45", "févr. 2024"]:
        assert is_date_text(value), value
    for value in ["Decarbonization 2030", "March", "Mar 2025 Bond", "2024", "S&P 500"]:
        assert not is_date_text(value), value


def test_header_detection() -> None:
    header = ["ISIN", "Fund name", "Brokerage", "Currency", "TER", "Launch"]
    data = [["IE00B5BMR087", "iShares Core S&P 500", "BlackRock", "USD", "0.07", "2010-05-19"]] * 3
    assert detect_header_row([header, *data]) == 0
    assert detect_header_row([["Fund report"], [], header, *data]) == 2
    assert detect_header_row([["Source:", "Morningstar"], [], header, *data]) == 2  # too narrow to be the header
    assert detect_header_row([["Report"], ["Name", "2024", "2025"], ["Fund A", "1.5", "2.5"]]) == 1  # years
    assert detect_header_row([["a"], ["b"], ["c"]]) == 0
    assert detect_header_row([]) == 0


def test_progress_and_cancel(funds_xlsx: Path) -> None:
    calls: list[tuple[str, float]] = []
    read_table(funds_xlsx, progress=lambda stage, fraction: calls.append((stage, fraction)))
    assert calls and {stage for stage, _ in calls} == {READING_STAGE}
    fractions = [fraction for _, fraction in calls]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    with pytest.raises(Cancelled):
        read_table(funds_xlsx, cancel=lambda: True)


# ------------------------------------------------------------------------------ text files


def test_semicolon_csv_in_cp1252(tmp_path: Path) -> None:
    path = tmp_path / "fondos.csv"
    text = (
        "ISIN;Nombre del fondo;Gestora;TER\r\n"
        "ES0114105036;Bestinver Internacional;Bestinver Gestión;1,76\r\n"
        "ES0110407097;Cobas Internacional;Cobas Asset Management;1,50\r\n"
        "\r\n"
        "LU0996182563;Amundi Index MSCI World;Amundi Société;0,30\r\n"
    )
    path.write_bytes(text.encode("cp1252"))
    assert list_sheets(path) == [""]
    table = read_table(path)
    assert table.sheet_name == "" and table.label == "fondos.csv"
    assert table.columns == ["ISIN", "Nombre del fondo", "Gestora", "TER"]
    assert table.rows[0] == ["ES0114105036", "Bestinver Internacional", "Bestinver Gestión", "1,76"]
    assert table.rows[2][2] == "Amundi Société"
    assert table.source_rows == [2, 3, 5]


def test_utf8_bom_csv_with_quotes(tmp_path: Path) -> None:
    path = tmp_path / "funds.csv"
    text = 'Fund name,Brokerage,Comment\n"Vanguard, FTSE All-World",Vanguard,"two\nlines"\nAmundi Prime Global,Amundi,ok\n'
    path.write_bytes(text.encode("utf-8-sig"))
    table = read_table(path)
    assert table.columns == ["Fund name", "Brokerage", "Comment"]  # no BOM in the first header
    assert table.rows == [["Vanguard, FTSE All-World", "Vanguard", "two lines"], ["Amundi Prime Global", "Amundi", "ok"]]


def test_tsv_and_utf16_txt(tmp_path: Path) -> None:
    tsv = tmp_path / "funds.tsv"
    tsv.write_text("Name\tTicker\nCore, S&P 500\tCSPX\nAll-World\tVWRL\n", encoding="utf-8")
    assert read_table(tsv).rows == [["Core, S&P 500", "CSPX"], ["All-World", "VWRL"]]
    txt = tmp_path / "export.txt"  # Excel's "Unicode Text" export
    txt.write_bytes("Name\tTicker\r\nÉpargne Actions\tEPA\r\nZürich Fonds\tZRF\r\n".encode("utf-16"))
    table = read_table(txt)
    assert table.columns == ["Name", "Ticker"] and table.rows[1] == ["Zürich Fonds", "ZRF"]


def test_ragged_and_single_column_text(tmp_path: Path) -> None:
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b\n1,2,extra\n3\n", encoding="utf-8")
    table = read_table(ragged)
    assert table.columns == ["a", "b", "Column 3"]
    assert table.rows == [["1", "2", "extra"], ["3", "", ""]]
    single = tmp_path / "names.txt"
    single.write_text("Fund name\nAlpha Fund\nBeta Fund\n", encoding="utf-8")
    assert read_table(single).rows == [["Alpha Fund"], ["Beta Fund"]]


def test_wrong_extensions_are_detected_by_content(tmp_path: Path, funds_xlsx: Path) -> None:
    renamed = tmp_path / "actually_a_workbook.csv"
    renamed.write_bytes(funds_xlsx.read_bytes())
    assert read_table(renamed).columns[0] == "ISIN"
    disguised = tmp_path / "report.xls"  # many web systems export CSV text as ".xls"
    disguised.write_text("Name;Value\nAlpha;1\n", encoding="utf-8")
    assert read_table(disguised).rows == [["Alpha", "1"]]


# ---------------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("data.pdf", b"%PDF-1.7", "not a supported file type"),
        ("corrupt.xlsx", b"PK\x03\x04" + bytes(range(256)) * 20, "could not be read"),
        ("garbage.xls", bytes(range(256)) * 20, "could not be read"),
        ("empty.csv", b"", "is empty"),
        ("page.xls", b"<html><body><table><tr><td>1</td></tr></table></body></html>", "web page"),
        ("binary.csv", b"\x00\x01\x02\x03" * 100, "does not look like a text file"),
        ("blank.csv", b"\n\n , \n", "no data"),
        ("header_only.csv", b"Name,Value\n", "no data rows"),
    ],
)
def test_friendly_errors(tmp_path: Path, name: str, content: bytes, message: str) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(TableError, match=message):
        read_table(path)


def test_missing_file_and_password_protected(tmp_path: Path) -> None:
    with pytest.raises(TableError, match="was not found"):
        list_sheets(tmp_path / "missing.xlsx")
    encrypted = tmp_path / "secret.xlsx"  # encrypted OOXML = OLE container with an EncryptedPackage stream
    encrypted.write_bytes(bytes.fromhex("D0CF11E0A1B11AE1") + bytes(504) + "EncryptedPackage".encode("utf-16-le") + bytes(512))
    with pytest.raises(TableError, match="password-protected"):
        list_sheets(encrypted)


# ----------------------------------------------------------------------------- performance


def test_reads_45k_rows_quickly(tmp_path: Path) -> None:
    path = tmp_path / "big.xlsx"
    workbook = xlsxwriter.Workbook(str(path), {"constant_memory": True})
    sheet = workbook.add_worksheet("Funds")
    sheet.write_row(0, 0, [f"Column {i}" for i in range(15)])
    for row in range(1, 45_001):
        sheet.write_row(
            row,
            0,
            [f"IE{row:010d}", f"Fund {row} Core MSCI World UCITS ETF", "BlackRock", "USD", row * 0.01, row,
             "Ireland", "Equity", "MSCI World", "Physical", "Accumulating", True, 3, "Note", f"Extra {row % 7}"],
        )  # fmt: skip
    workbook.close()
    started = time.perf_counter()
    table = read_table(path)
    elapsed = time.perf_counter() - started
    assert len(table) == 45_000 and len(table.columns) == 15
    assert table.rows[-1][:2] == ["IE0000045000", "Fund 45000 Core MSCI World UCITS ETF"]
    assert elapsed < 10, f"reading 45k x 15 took {elapsed:.2f} s"
