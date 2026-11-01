"""Family model: vocabularies, learned aliases, detection, cleaning and family factors
(ARCHITECTURE.md section 6)."""

from __future__ import annotations

import numpy as np
import pytest

from semdb.core.families import DecidingColumn, FamilyModel
from semdb.core.text import tokens
from semdb.core.types import Role, Roles, Table


def _table(columns: list[str], rows: list[list[str]]) -> Table:
    return Table("test.xlsx", "Sheet1", columns, [list(r) for r in rows], list(range(2, len(rows) + 2)))


def _model(table: Table, roles: Roles) -> FamilyModel:
    deciding = [c for c in table.columns if roles[c] is Role.DECIDING]
    search = [c for c in table.columns if roles[c] is Role.SEARCH]
    return FamilyModel.build(table, deciding, search)


def _column(model: FamilyModel, name: str) -> DecidingColumn:
    return next(column for column in model.columns if column.name == name)


def _weights(column: DecidingColumn, key: str) -> dict[str, float]:
    return {column.values[v]: w for v, w in column.indicators.get(key, [])}


def _strengths(model: FamilyModel, query: str) -> dict[str, dict[str, float]]:
    detection = model.detect(query)
    return {
        name: {_column(model, name).values[v]: s for v, s in values.items()}
        for name, values in detection.evidence.items()
    }


@pytest.fixture
def funds_model(funds_table: Table, funds_roles: Roles) -> FamilyModel:
    return _model(funds_table, funds_roles)


def test_vocabulary_codes_and_identifier_flags(funds_model: FamilyModel, funds_table: Table) -> None:
    isin, brokerage = _column(funds_model, "ISIN"), _column(funds_model, "Brokerage")
    assert isin.is_identifier and not brokerage.is_identifier
    assert brokerage.values == ["BlackRock", "Vanguard", "Amundi", "DWS", "State Street"]
    assert brokerage.codes.dtype == np.int32
    assert [brokerage.values[c] for c in brokerage.codes] == funds_table.column_values("Brokerage")
    with pytest.raises(ValueError):
        brokerage.codes[0] = 1  # immutable after build


def test_learns_ishares_alias_for_blackrock(funds_model: FamilyModel) -> None:
    brokerage = _column(funds_model, "Brokerage")
    blackrock = brokerage.values.index("BlackRock")
    assert brokerage.learned_aliases[blackrock][0] == "ishares"  # full coverage ranks first
    assert _weights(brokerage, "ishares") == {"BlackRock": pytest.approx(1.0)}
    # Display uses the spelling found in the data.
    assert funds_model.learned_alias_table()["Brokerage"]["BlackRock"][0] == "iShares"
    for value_id, aliases in brokerage.learned_aliases.items():
        assert brokerage.values[value_id] == "BlackRock" and "vanguard" not in aliases


def test_value_tokens_phrases_and_space_less_joins(funds_model: FamilyModel) -> None:
    brokerage = _column(funds_model, "Brokerage")
    assert _weights(brokerage, "vanguard") == {"Vanguard": pytest.approx(1.0)}
    for key in ("state", "street", "state street", "statestreet"):
        assert _weights(brokerage, key) == {"State Street": pytest.approx(1.0)}
    assert _strengths(funds_model, "StateStreet MSCI World") == {"Brokerage": {"State Street": pytest.approx(1.0)}}


def test_text_precision_smoothing_and_generic_tokens() -> None:
    rows = [
        ["Global X Lithium ETF", "Global X"],
        ["Global X Uranium ETF", "Global X"],
        ["Global X Robotics ETF", "Global X"],
        ["Vanguard Global Bond Fund", "Vanguard Group"],
        ["Vanguard Global Equity Fund", "Vanguard Group"],
        ["Amundi Global Aggregate", "Amundi Asset Management"],
        ["Amundi Prime Global", "Amundi Asset Management"],
        ["BNP Paribas Easy Europe", "BNP Paribas Asset Management"],
        ["DWS Invest Europe", "DWS Asset Management"],
        ["Candriam Bonds Euro", "Candriam Asset Management"],
    ]
    table = _table(["Name", "Company"], rows)
    model = _model(table, {"Name": Role.SEARCH, "Company": Role.DECIDING})
    company = _column(model, "Company")
    # "global": one value holds it; 3 of its rows mention it, 7 rows in all -> (3 + 2) / (7 + 2)
    assert _weights(company, "global") == {"Global X": pytest.approx(5 / 9)}
    # ...but the full phrase only occurs in Global X rows: max(0.9 * 1, 5/9)
    assert _weights(company, "global x") == {"Global X": pytest.approx(0.9)}
    assert _weights(company, "globalx") == {"Global X": pytest.approx(0.9)}
    # "asset"/"management" are shared by 4 values -> generic 3/4; never in the text -> precision 1
    assert list(_weights(company, "management").values()) == pytest.approx([0.75] * 4)
    assert "x" not in company.indicators  # short token of a multi-token value


def test_short_tokens_only_for_single_token_values() -> None:
    rows = [["JPMorgan US Value Fund", "J.P. Morgan"], ["AB American Growth", "AB"], ["Other Fund", "Other"]]
    model = _model(_table(["Name", "Company"], rows), {"Name": Role.SEARCH, "Company": Role.DECIDING})
    company = _column(model, "Company")
    assert "j" not in company.indicators and "p" not in company.indicators
    assert _weights(company, "ab") == {"AB": pytest.approx(1.0)}
    assert set(_weights(company, "jpmorgan")) == {"J.P. Morgan"}  # space-less join of "j p morgan"
    assert _strengths(model, "JPM US Value") == {"Company": {"J.P. Morgan": pytest.approx(0.6)}}  # abbreviation


def test_identifier_weights_for_ordinary_word_tickers() -> None:
    rows = [
        ["Vanguard FTSE All-World UCITS ETF", "VWRL", "IE00B3RBWM25"],
        ["Allstate Insurance Equity Fund", "ALL", "US0200021014"],
        ["Vanguard S&P 500 UCITS ETF", "VUSA", "IE00B3XXRP09"],
    ]
    table = _table(["Name", "Ticker", "ISIN"], rows)
    model = _model(table, {"Name": Role.SEARCH, "Ticker": Role.DECIDING, "ISIN": Role.DECIDING})
    ticker = _column(model, "Ticker")
    assert ticker.is_identifier
    assert _weights(ticker, "all") == {"ALL": pytest.approx(0.3)}  # "all" is a word of the text
    assert _weights(ticker, "vwrl") == {"VWRL": pytest.approx(1.0)}  # never in the text
    assert _weights(_column(model, "ISIN"), "ie00b3rbwm25") == {"IE00B3RBWM25": pytest.approx(1.0)}
    detection = model.detect("FTSE All-World")
    assert detection.evidence == {"Ticker": {1: pytest.approx(0.3)}}
    assert detection.spans == []  # too weak to be cut from the query


def test_detect_brand_alias_and_cut_span(funds_model: FamilyModel) -> None:
    query = "iShares Core S&P 500"
    detection = funds_model.detect(query)
    assert _strengths(funds_model, query) == {"Brokerage": {"BlackRock": pytest.approx(1.0)}}
    assert (0, 7) in detection.spans  # "iShares"
    assert detection.found[0].value == "BlackRock" and detection.found[0].column == "Brokerage"
    assert detection.found[0].matched_text.startswith("iShares")


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("ISHS Core MSCI World", "BlackRock"),  # abbreviation (+ data-learned "core" in this tiny table)
        ("iSh MSCI Europe", "BlackRock"),  # abbreviation only: 1.0 * 0.6
        ("Vangard FTSE All-World", "Vanguard"),  # typo: (93/100)^2
        ("Amundy Emerging Markets", None),  # ratio 83 < 85: not a typo of amundi
    ],
)
def test_detect_abbreviations_and_typos(funds_model: FamilyModel, query: str, expected: str | None) -> None:
    found = funds_model.detect(query).found
    assert (found[0].value if found else None) == expected
    if query.startswith("iSh "):
        assert found[0].strength == pytest.approx(0.6)
    if query.startswith("Vangard"):
        assert found[0].strength == pytest.approx((2 * 7 / 15) ** 2, abs=1e-3)
        assert found[0].matched_text == "Vangard"


def test_fuzzy_skips_established_corpus_words() -> None:
    rows = [
        ["Robeco Global Equity", "Robeco"],
        ["Robeco Euro Bonds", "Robeco"],
        ["Robeco Emerging Markets", "Robeco"],
        ["Robecos Value Fund", "Alpha"],
        ["Robecos Income Fund", "Beta"],
        ["Robecos Growth Fund", "Gamma"],
    ]
    model = _model(_table(["Name", "Company"], rows), {"Name": Role.SEARCH, "Company": Role.DECIDING})
    assert model.detect("Robecos Value").found == []  # df_text("robecos") = 3: a word, not a typo
    assert model.detect("Robecoo Value").found[0].value == "Robeco"  # unknown word: typo of robeco


def test_detect_nothing(funds_model: FamilyModel) -> None:
    for query in ["", "   ", "S&P 500", "MSCI World UCITS ETF"]:
        detection = funds_model.detect(query)
        assert detection.evidence == {} and detection.spans == [] and detection.found == []


def test_detect_spans_map_back_to_accented_text() -> None:
    rows = [
        ["Bankinter Índice España FI", "Bankinter Gestión de Activos"],
        ["Bankinter Renta Fija FI", "Bankinter Gestión de Activos"],
        ["Santander Acciones Españolas FI", "Santander Asset Management"],
    ]
    model = _model(_table(["Nombre", "Gestora"], rows), {"Nombre": Role.SEARCH, "Gestora": Role.DECIDING})
    query = "GESTIÓN Bankinter: Índice España"
    detection = model.detect(query)
    assert detection.found[0].value == "Bankinter Gestión de Activos"
    assert detection.found[0].matched_text == "GESTIÓN Bankinter"
    assert [query[s:e] for s, e in detection.spans] == ["GESTIÓN", "Bankinter"]
    assert model.clean_cells(0, ["Bankinter Índice España FI"]) == ["Índice España FI"]


def test_clean_cells_removes_brand_from_own_family_only(funds_model: FamilyModel, funds_table: Table) -> None:
    brokerage = _column(funds_model, "Brokerage")
    name = funds_table.column_index("Fund name")
    for row, cells in enumerate(funds_table.rows):
        cleaned = funds_model.clean_cells(row, [cells[name]])[0]
        family = brokerage.values[brokerage.codes[row]]
        if family == "BlackRock":
            assert "ishares" not in tokens(cleaned)
        elif family in ("Vanguard", "Amundi"):
            assert family.lower() not in tokens(cleaned)
        else:  # DWS / State Street: the brand (Xtrackers, SPDR) is not learned from 2 rows
            assert cleaned == cells[name]
    assert funds_model.clean_cells(0, ["iShares Core S&P 500 UCITS ETF USD (Acc)"])[0].startswith("S&P 500 UCITS ETF")
    vanguard_row = funds_table.column_values("Brokerage").index("Vanguard")
    assert funds_model.clean_cells(vanguard_row, ["iShares Core S&P 500"]) == ["iShares Core S&P 500"]


def test_clean_cells_keeps_cells_that_would_become_empty(funds_model: FamilyModel) -> None:
    assert funds_model.clean_cells(0, ["iShares", "", "BlackRock - iShares"]) == ["iShares", "", "BlackRock - iShares"]


def test_clean_cells_removes_own_identifier() -> None:
    rows = [["Fund IE00B5BMR087 Core", "IE00B5BMR087"], ["Other fund IE00B5BMR087", "IE0031442068"]]
    model = _model(_table(["Name", "ISIN"], rows), {"Name": Role.SEARCH, "ISIN": Role.DECIDING})
    assert model.clean_cells(0, [rows[0][0]]) == ["Fund Core"]
    assert model.clean_cells(1, [rows[1][0]]) == [rows[1][0]]


def test_family_factor_agree_conflict_empty() -> None:
    rows = [
        ["iShares Core S&P 500", "BlackRock"],
        ["iShares Core MSCI World", "BlackRock"],
        ["iShares MSCI EM", "BlackRock"],
        ["Vanguard S&P 500", "Vanguard"],
        ["Vanguard FTSE All-World", "Vanguard"],
        ["Vanguard Global Bond", "Vanguard"],
        ["Unknown S&P 500 tracker", ""],
    ]
    model = _model(_table(["Name", "Brokerage"], rows), {"Name": Role.SEARCH, "Brokerage": Role.DECIDING})
    brokerage = _column(model, "Brokerage")
    assert brokerage.codes[6] == -1
    detection = model.detect("Vanguard S&P 500")
    factor = model.family_factor(detection, 0.85)
    assert factor.dtype == np.float32 and factor.shape == (7,)
    np.testing.assert_allclose(factor, [0.15, 0.15, 0.15, 1, 1, 1, 1 - 0.85 * 0.5], atol=1e-6)
    assert model.agreement(detection, 3) == (["Brokerage"], [])
    assert model.agreement(detection, 0) == ([], ["Brokerage"])
    assert model.agreement(detection, 6) == ([], ["Brokerage"])  # empty cell: reduced strength
    np.testing.assert_allclose(model.family_factor(model.detect("S&P 500"), 0.85), np.ones(7))
    np.testing.assert_allclose(model.family_factor(detection, 0.0), np.ones(7))
    strict = model.family_factor(detection, 1.0)
    assert np.all(strict > 0) and strict[0] == pytest.approx(1e-3)


def test_family_factor_partial_evidence_and_columns_multiply(funds_model: FamilyModel) -> None:
    detection = funds_model.detect("iSh Core S&P 500 IE00B5BMR087")
    assert set(detection.evidence) == {"ISIN", "Brokerage"}
    factor = funds_model.family_factor(detection, 0.85)
    assert factor[0] == pytest.approx(1.0)  # its ISIN and its brokerage
    assert factor[1] == pytest.approx(0.15)  # right brokerage, other ISIN
    assert factor[4] == pytest.approx(0.15 * 0.15)  # Vanguard row: both columns conflict
    weak = funds_model.detect("iSh MSCI Europe")  # BlackRock at 0.6
    assert funds_model.family_factor(weak, 0.85)[4] == pytest.approx(1 - 0.85 * 0.6)


def test_alias_learning_needs_grouped_families_and_few_values() -> None:
    def build(n_values: int, rows_per_value: int) -> FamilyModel:
        rows = [[f"Brand{i} fund {j}", f"Company {i}"] for i in range(n_values) for j in range(rows_per_value)]
        return _model(_table(["Name", "Company"], rows), {"Name": Role.SEARCH, "Company": Role.DECIDING})

    learned = build(40, 3).columns[0].learned_aliases
    assert len(learned) == 40 and learned[0] == ["brand0"]
    assert build(40, 2).columns[0].learned_aliases == {}  # typical family smaller than 3 rows
    assert build(5001, 3).columns[0].learned_aliases == {}  # more than 5,000 distinct values


def test_rare_brand_spellings_are_learned_but_product_lines_are_not() -> None:
    rows = [[f"{'ASHS' if i < 3 else 'Alphashares'} {'Core ' if 5 <= i < 8 else ''}Fund {i}", "Alpha Group"] for i in range(20)]
    rows += [[f"Betafunds Fund {i}", "Beta Group"] for i in range(20)]
    model = _model(_table(["Name", "Company"], rows), {"Name": Role.SEARCH, "Company": Role.DECIDING})
    company = _column(model, "Company")
    alpha = company.values.index("Alpha Group")
    # "ashs" covers 3/20 rows (< 0.2) but abbreviates the learned "alphashares": kept.
    # "core" is just as precise and frequent but resembles no Alpha key: not an identity.
    assert company.learned_aliases[alpha] == ["alphashares", "ashs"]
    assert _weights(company, "ashs") == {"Alpha Group": pytest.approx(0.65)}
    assert "core" not in company.indicators
    assert model.detect("ASHS Core Fund").found[0].value == "Alpha Group"
    assert model.clean_cells(0, [rows[0][0]]) == ["Fund 0"]


def test_without_deciding_columns(funds_table: Table) -> None:
    model = FamilyModel.build(funds_table, [], ["Fund name"])
    assert model.columns == []
    assert model.detect("iShares Core S&P 500").found == []
    assert model.clean_cells(0, ["iShares Core"]) == ["iShares Core"]
    np.testing.assert_allclose(model.family_factor(model.detect("x"), 0.85), np.ones(len(funds_table)))
    assert model.learned_alias_table() == {}


def test_found_is_ordered_strongest_first(funds_model: FamilyModel) -> None:
    found = funds_model.detect("iSh vs Vanguard S&P 500").found
    assert [(e.value, round(e.strength, 2)) for e in found] == [("Vanguard", 1.0), ("BlackRock", 0.6)]
