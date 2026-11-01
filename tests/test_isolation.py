"""The import worker: files are parsed in a separate, memory-capped process."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import xlsxwriter

from semdb.core import isolation
from semdb.core.table import TableError, list_sheets, read_raw, read_table
from semdb.core.types import Cancelled


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(isolation.IN_PROCESS_ENV, raising=False)


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    path = tmp_path / "funds.xlsx"
    with xlsxwriter.Workbook(str(path)) as book:
        sheet = book.add_worksheet("Funds")
        sheet.write_row(0, 0, ["ISIN", "Fund name", "TER"])
        sheet.write_row(1, 0, ["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF", 0.07])
        sheet.write_row(2, 0, ["IE00BK5BQT80", "Vanguard FTSE All-World — Acumulación", 0.22])
        book.add_worksheet("Notes").write(0, 0, "x")
    return path


def test_worker_results_match_in_process_reading(worker: None, workbook: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fractions: list[float] = []
    sheets = list_sheets(workbook)
    raw = read_raw(workbook, "Funds", max_rows=2)
    table = read_table(workbook, "Funds", progress=lambda stage, fraction: fractions.append(fraction))
    monkeypatch.setenv(isolation.IN_PROCESS_ENV, "1")
    local = read_table(workbook, "Funds")
    assert sheets == ["Funds", "Notes"]
    assert raw == [["ISIN", "Fund name", "TER"], ["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF", "0.07"]]
    assert (table.columns, table.rows, table.source_rows) == (local.columns, local.rows, local.source_rows)
    assert table.rows[1][1] == "Vanguard FTSE All-World — Acumulación"
    assert fractions and fractions[-1] == 1.0


def test_reader_errors_come_back_as_friendly_messages(worker: None, tmp_path: Path) -> None:
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"PK\x03\x04" + b"\x00" * 200)
    with pytest.raises(TableError, match="could not be read"):
        read_table(broken)
    with pytest.raises(TableError, match="was not found"):  # answered without starting a worker
        list_sheets(tmp_path / "missing.xlsx")


def test_memory_cap_stops_a_runaway_allocation(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(isolation.TEST_OPS_ENV, "1")
    monkeypatch.setenv(isolation.MEMORY_ENV, "256")
    with pytest.raises(isolation.WorkerError) as caught:
        isolation.run("allocate", {"bytes": 2 << 30})
    assert caught.value.kind == "memory"
    assert isolation.run("allocate", {"bytes": 16 << 20}) == 16 << 20  # small allocations still work


def test_cancel_stops_the_worker_quickly(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(isolation.TEST_OPS_ENV, "1")
    started = time.monotonic()
    with pytest.raises(Cancelled):
        isolation.run("sleep", {"seconds": 30}, cancel=lambda: time.monotonic() - started > 0.5)
    assert time.monotonic() - started < 10


def test_a_crashing_worker_is_reported_not_fatal(worker: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(isolation.TEST_OPS_ENV, "1")
    with pytest.raises(isolation.WorkerError) as caught:
        isolation.run("exit", {"code": 3})
    assert caught.value.kind == "crashed"


def test_memory_limit_is_bounded() -> None:
    limit = isolation.memory_limit_bytes()
    assert 1 << 30 <= limit <= 3 << 30
