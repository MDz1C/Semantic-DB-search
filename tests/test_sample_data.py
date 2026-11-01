"""The synthetic fund universe (tools/make_sample_data.py) and the evaluation queries built on it."""

from __future__ import annotations

import datetime as dt
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest
from python_calamine import CalamineWorkbook

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import evaluate  # noqa: E402
import make_sample_data as msd  # noqa: E402
from semdb.core.text import normalize, tokens  # noqa: E402
from semdb.core.types import Match, QueryOutcome  # noqa: E402

ISIN = msd.COLUMNS.index("ISIN")
TICKER = msd.COLUMNS.index("Ticker")
NAME = msd.COLUMNS.index("Fund name")
BROKERAGE = msd.COLUMNS.index("Brokerage")


def luhn_valid_isin(isin: str) -> bool:
    """Independent ISO 6166 validation: Luhn over the whole code with letters as 10..35."""
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", isin):
        return False
    digits = [int(d) for ch in isin for d in str(int(ch, 36))]
    total = sum(d if i % 2 == 0 else (2 * d if d < 5 else 2 * d - 9) for i, d in enumerate(reversed(digits)))
    return total % 10 == 0


@pytest.fixture(scope="module")
def large_universe() -> tuple[list[msd.FundRecord], float]:
    started = time.perf_counter()
    records = msd.generate_records(45_000, seed=7)
    return records, time.perf_counter() - started


@pytest.fixture(scope="module")
def sample_records() -> list[msd.FundRecord]:
    return msd.generate_records(2500, seed=7)


def test_generation_is_deterministic() -> None:
    columns, rows = msd.generate(400, seed=11)
    assert (columns, rows) == msd.generate(400, seed=11)
    assert rows != msd.generate(400, seed=12)[1]
    assert columns == msd.COLUMNS
    assert len(rows) == 400


def test_rows_are_well_formed(sample_records: list[msd.FundRecord]) -> None:
    for record in sample_records:
        cells = dict(zip(msd.COLUMNS, record.cells))
        assert len(record.cells) == len(msd.COLUMNS)
        assert all(cells[name] for name in msd.COLUMNS), cells
        assert cells["Asset class"] in {"Equity", "Fixed Income", "Money Market", "Multi-Asset", "Commodity"}
        assert cells["Distribution"] in {"Acc", "Dist"}
        assert cells["Domicile"] in msd.DOMICILES.values()
        assert re.fullmatch(r"[A-Z]{3}", cells["Currency"])
        assert 0 < float(cells["TER (%)"]) < 4 and float(cells["AUM (M EUR)"]) > 0
        assert not cells["TER (%)"].endswith(".0") and not cells["AUM (M EUR)"].endswith(".0")
        assert dt.date(1985, 1, 1) <= dt.date.fromisoformat(cells["Launch date"]) <= dt.date(2026, 1, 1)
        expected = msd.compose_name(record.brand_text, record.sep, record.body, record.share_class)
        assert record.name == (expected.upper() if record.upper else expected)
    assert {r.cells[4] for r in sample_records} >= {
        "Renta Variable Internacional", "Renta Variable Euro", "Renta Fija Euro Corto Plazo",
        "Renta Fija Internacional", "Monetario", "Mixto Moderado", "Materias Primas",
    }


def test_known_isins_get_the_right_check_digit() -> None:
    for isin in ("US0378331005", "IE00B5BMR087", "IE00B4L5Y983", "LU1681048804", "DE0005933931"):
        assert msd.isin_check_digit(isin[:11]) == isin[11]
        assert luhn_valid_isin(isin)
    assert not luhn_valid_isin("IE00B5BMR088")


def test_scales_to_45k_unique_rows_quickly(large_universe: tuple[list[msd.FundRecord], float]) -> None:
    records, seconds = large_universe
    assert len(records) == 45_000
    assert seconds < 15, f"45,000 rows took {seconds:.1f}s"
    isins = [r.cells[ISIN] for r in records]
    tickers = [r.cells[TICKER] for r in records]
    assert len(set(isins)) == len(isins)
    assert len(set(tickers)) == len(tickers)
    assert len({normalize(r.name) for r in records}) == len(records)
    assert all(luhn_valid_isin(isin) for isin in isins)
    assert all(isin[:2] in msd.DOMICILES for isin in isins)
    assert all(re.fullmatch(r"[A-Z0-9]{3,5}", ticker) for ticker in tickers)


def test_tickers_never_double_as_words(sample_records: list[msd.FundRecord]) -> None:
    vocabulary = {token for record in sample_records for token in tokens(record.name)}
    assert not {r.cells[TICKER].lower() for r in sample_records} & vocabulary


def test_brands_differ_from_companies_and_vary(large_universe: tuple[list[msd.FundRecord], float]) -> None:
    records, _ = large_universe
    pairs = {(r.company, r.brand) for r in records}
    assert {("BlackRock", "iShares"), ("DWS", "Xtrackers"), ("Amundi", "Lyxor"),
            ("State Street Global Advisors", "SPDR"), ("Mutuactivos", "Mutuafondo")} <= pairs
    spellings = {r.brand_text for r in records}
    assert {"ISHS", "iSh", "ISHARES", "Xtr.", "X-trackers", "Vang.", "J.P. Morgan", "Van Eck"} <= spellings
    assert any(not r.brand_text for r in records), "some names omit the brand"
    assert any(r.upper for r in records), "some names are written in capitals"
    assert any("Selección" in r.name for r in records) and any("Seleccion" in r.name for r in records)

    ishares = [r for r in records if r.brand == "iShares"]
    main_share = sum(r.brand_text == "iShares" for r in ishares) / len(ishares)
    assert 0.75 < main_share < 0.95, "the main spelling dominates; variants are a minority"
    # The company never appears in an iShares name: only the brand links the row to BlackRock.
    assert not any("blackrock" in normalize(r.name) for r in ishares)


def test_same_exposure_exists_across_brokerages(sample_records: list[msd.FundRecord]) -> None:
    companies: dict[str, set[str]] = defaultdict(set)
    for record in sample_records:
        for exposure in ("S&P 500", "MSCI World", "MSCI Emerging Markets", "Euro Stoxx 50", "Physical Gold"):
            if re.search(rf"\b{re.escape(exposure)}\b(?! Small)", record.name, re.IGNORECASE):
                companies[exposure].add(record.company)
    assert all(len(companies[e]) >= 4 for e in ("S&P 500", "MSCI World", "MSCI Emerging Markets")), companies
    assert len(companies["Euro Stoxx 50"]) >= 3 and len(companies["Physical Gold"]) >= 2
    bodies: dict[str, set[str]] = defaultdict(set)
    for record in sample_records:
        bodies[normalize(evaluate.brand_free(record))].add(record.company)
    assert any(len(owners) >= 3 for owners in bodies.values()), "identical names once the brand is removed"


def test_xlsx_round_trip(tmp_path: Path) -> None:
    columns, rows = msd.generate(300, seed=3)
    path = tmp_path / "funds.xlsx"
    msd.write_xlsx(path, columns, rows)
    book = CalamineWorkbook.from_path(str(path))
    assert book.sheet_names == ["Funds"]
    values = book.get_sheet_by_name("Funds").to_python()
    assert values[0] == columns
    assert len(values) == len(rows) + 1
    for written, read in zip(rows, values[1:]):
        for name, text, cell in zip(columns, written, read):
            if name in msd.NUMBER_COLUMNS:
                assert isinstance(cell, float) and cell == float(text)
            elif name in msd.DATE_COLUMNS:
                assert isinstance(cell, dt.date) and cell == dt.date.fromisoformat(text)
            else:
                assert cell == text


def test_bundled_sample_matches_the_generator() -> None:
    path = msd.DEFAULT_OUT
    assert path.exists(), "run tools/make_sample_data.py"
    assert path.stat().st_size < 1_000_000
    values = CalamineWorkbook.from_path(str(path)).get_sheet_by_name("Funds").to_python()
    columns, rows = msd.generate(2500, seed=7)
    assert values[0] == columns
    assert [row[NAME] for row in values[1:]] == [row[NAME] for row in rows]
    assert [row[ISIN] for row in values[1:]] == [row[ISIN] for row in rows]


def test_evaluation_queries_cover_every_type() -> None:
    records = msd.generate_records(3000, seed=7)
    queries = evaluate.make_queries(records, 140, seed=7)
    assert len(queries) == 140
    assert {q.tag for q in queries} == set(evaluate.QUERY_TYPES)
    assert queries == evaluate.make_queries(records, 140, seed=7)
    for query in queries:
        record = records[query.target]
        assert query.target in query.accept and query.text.strip()
        if query.tag == "ISIN only":
            assert query.text == record.cells[ISIN] and not query.names_brand
        if query.tag == "brand removed":
            assert record.brand_text not in query.text
            assert all(normalize(evaluate.brand_free(records[row])) == normalize(query.text) for row in query.accept)
        if query.tag == "company instead of brand":
            assert evaluate.company_short(record.company) in query.text and record.brand not in query.text
        if query.tag in ("typo", "reordered tokens", "lowercase name"):
            assert query.text != record.name


class _FakeIndex:
    """Answers with fixed rows per query, to check the metrics without the engine."""

    def __init__(self, answers: dict[str, list[int]]) -> None:
        self.answers = answers

    def search(self, query: str, k: int = 50) -> QueryOutcome:
        matches = [Match(row, 1.0, 1.0, 0.0, 1.0, 1.0) for row in self.answers[query][:k]]
        return QueryOutcome(query=query, cleaned_query=query, evidence=[], matches=matches)


def test_metrics_on_a_fake_index() -> None:
    brokerages = ["BlackRock", "BlackRock", "Vanguard", "BlackRock", "Vanguard", "Amundi"]
    hit = evaluate.Query("iShares Core S&P 500", "exact name", 0, frozenset({0}), True)
    second = evaluate.Query("ISHS MSCI World", "brand abbreviated", 1, frozenset({1}), True)
    miss = evaluate.Query("IE00B5BMR087", "ISIN only", 3, frozenset({3}), False)
    index = _FakeIndex({hit.text: [0, 1, 2, 3, 4], second.text: [2, 1, 3], miss.text: [5]})
    scored = evaluate.run_queries(index, [hit, second, miss], brokerages)
    assert [s.rank for s in scored] == [1, 2, None]
    assert scored[0].brand_precision == pytest.approx(3 / 5)
    assert scored[1].brand_precision == pytest.approx(2 / 3)
    assert scored[2].brand_precision is None
    result = evaluate.metrics(scored)
    assert result["acc@1"] == pytest.approx(1 / 3)
    assert result["acc@5"] == pytest.approx(2 / 3)
    assert result["mrr@10"] == pytest.approx((1 + 1 / 2) / 3)
    assert result["brand_p@5"] == pytest.approx((3 / 5 + 2 / 3) / 2)
    assert result["brand_n"] == 2
