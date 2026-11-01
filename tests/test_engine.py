"""Search engine: identity-aware ranking end to end (ARCHITECTURE.md section 9, R3-R6)."""

from __future__ import annotations

import json
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from semdb.core.engine import (
    STAGE_ALIASES,
    STAGE_ANALYZE,
    STAGE_EMBED,
    STAGE_FINAL,
    STAGE_INDEX,
    STAGE_SEARCH,
    EngineConfig,
    SearchIndex,
)
from semdb.core.text import tokens
from semdb.core.types import MAX_STORED_RESULTS, CancelFn, Cancelled, QueryOutcome, Role, Roles, Table


class FakeEmbedder:
    """Deterministic stand-in for the ONNX model: hashed character trigrams -> 64 dims."""

    model_id = "fake-trigram-64"
    dim = 64

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        with self._lock:
            self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            if cancel is not None and cancel():
                raise Cancelled()
            for token in tokens(text):
                padded = f" {token} "
                for j in range(len(padded) - 2):
                    h = zlib.crc32(padded[j : j + 3].encode("utf-8"))
                    out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
            else:
                out[i, 0] = 1.0
            if progress is not None:
                progress(i + 1, len(texts))
        return out

    @property
    def encoded(self) -> list[str]:
        return [text for call in self.calls for text in call]


class DictCache:
    """In-memory EmbeddingCache that stores float16 like the SQLite one."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], np.ndarray] = {}
        self.put_texts: list[str] = []

    def get_many(self, model_id: str, texts: list[str]) -> dict[int, np.ndarray]:
        return {
            i: self.store[(model_id, text)].astype(np.float32)
            for i, text in enumerate(texts)
            if (model_id, text) in self.store
        }

    def put_many(self, model_id: str, texts: list[str], vectors: np.ndarray) -> None:
        assert len(texts) == len(vectors)
        for text, vector in zip(texts, vectors):
            self.store[(model_id, text)] = np.asarray(vector, dtype=np.float16)
        self.put_texts.extend(texts)


def _table(columns: list[str], rows: list[list[str]]) -> Table:
    return Table("test.csv", "", columns, [list(r) for r in rows], list(range(2, len(rows) + 2)))


def _brokerage(table: Table, row: int) -> str:
    return table.rows[row][table.column_index("Brokerage")]


def _name(table: Table, row: int) -> str:
    return table.rows[row][table.column_index("Fund name")]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def index(funds_table: Table, funds_roles: Roles, embedder: FakeEmbedder) -> SearchIndex:
    return SearchIndex.build(funds_table, funds_roles, embedder=embedder)


def _check_outcome(outcome: QueryOutcome, k: int) -> None:
    assert len(outcome.matches) <= k
    scores = [m.score for m in outcome.matches]
    assert scores == sorted(scores, reverse=True)
    for m in outcome.matches:
        for value in (m.score, m.text_score, m.semantic, m.lexical):
            assert 0.0 <= value <= 1.0
        assert 0.0 < m.family_factor <= 1.0
        assert m.score == pytest.approx(m.text_score * m.family_factor, abs=1e-6)


# --- identity-aware ranking (R3, R4, R6) ------------------------------------------------


def test_ishares_query_finds_a_blackrock_sp500_fund(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("iShares Core S&P 500", k=10)
    _check_outcome(outcome, 10)
    top = outcome.matches[0]
    assert _brokerage(funds_table, top.row) == "BlackRock" and "S&P 500" in _name(funds_table, top.row)
    assert top.agreements == ["Brokerage"] and top.conflicts == []
    assert outcome.evidence[0].value == "BlackRock" and outcome.evidence[0].matched_text.startswith("iShares")
    assert "iShares" not in outcome.cleaned_query and "S&P 500" in outcome.cleaned_query
    vanguard = [m for m in outcome.matches if _brokerage(funds_table, m.row) == "Vanguard"]
    assert vanguard and all(m.conflicts == ["Brokerage"] and m.family_factor < 0.2 for m in vanguard)


def test_vanguard_query_ranks_every_blackrock_row_below_vanguard_sp500(
    index: SearchIndex, funds_table: Table
) -> None:
    outcome = index.search("Vanguard S&P 500", k=len(funds_table))
    rows = [m.row for m in outcome.matches]
    assert _brokerage(funds_table, rows[0]) == "Vanguard" and "S&P 500" in _name(funds_table, rows[0])
    vanguard_sp = [
        rank
        for rank, row in enumerate(rows)
        if _brokerage(funds_table, row) == "Vanguard" and "S&P 500" in _name(funds_table, row)
    ]
    blackrock = [rank for rank, row in enumerate(rows) if _brokerage(funds_table, row) == "BlackRock"]
    assert len(vanguard_sp) == 2 and blackrock
    assert max(vanguard_sp) < min(blackrock)


def test_brand_beats_a_competitor_whose_words_overlap_better(index: SearchIndex, funds_table: Table) -> None:
    """The brief's example: the Vanguard fund matches the rest of the words exactly, the iShares
    fund does not, yet a query naming iShares must put the iShares fund first."""
    outcome = index.search("iShares S&P 500 UCITS ETF USD Distributing", k=5)
    by_row = {m.row: m for m in outcome.matches}
    top = outcome.matches[0]
    assert _brokerage(funds_table, top.row) == "BlackRock"
    vanguard_dist = funds_table.column_values("Fund name").index("Vanguard S&P 500 UCITS ETF USD Distributing")
    assert by_row[vanguard_dist].text_score > top.text_score  # better text overlap...
    assert by_row[vanguard_dist].score < top.score  # ...but the wrong family


@pytest.mark.parametrize("query", ["ISHS Core MSCI World", "BlackRock Core MSCI World", "iSh Core MSCI World"])
def test_abbreviated_brand_or_company_name(index: SearchIndex, funds_table: Table, query: str) -> None:
    outcome = index.search(query, k=5)
    assert _name(funds_table, outcome.matches[0].row) == "iShares Core MSCI World UCITS ETF USD (Acc)"
    assert outcome.evidence[0].value == "BlackRock"
    assert outcome.cleaned_query == "MSCI World"


def test_isin_only_query_is_an_exact_hit(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("IE00B4L5Y983", k=5)
    top = outcome.matches[0]
    assert funds_table.rows[top.row][0] == "IE00B4L5Y983"
    assert top.exact_id == "ISIN" and top.score == pytest.approx(1.0)
    assert outcome.cleaned_query == ""
    assert all(m.exact_id is None and m.score < 0.5 for m in outcome.matches[1:])
    assert index.search("ie00 b4l5 y983", k=1).matches[0].row == top.row  # spaced, lower case


def test_unbranded_query_spans_several_brokerages(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("S&P 500", k=6)
    assert outcome.evidence == [] and outcome.cleaned_query == "S&P 500"
    assert all("S&P 500" in _name(funds_table, m.row) for m in outcome.matches)
    assert len({_brokerage(funds_table, m.row) for m in outcome.matches}) >= 3
    assert all(m.family_factor == 1.0 and m.agreements == m.conflicts == [] for m in outcome.matches)


def test_identity_only_query_ranks_the_family(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("iShares", k=8)
    assert outcome.cleaned_query == ""
    families = [_brokerage(funds_table, m.row) for m in outcome.matches]
    assert families[:6] == ["BlackRock"] * 6 and "BlackRock" not in families[6:]
    assert all(m.text_score == pytest.approx(0.5) for m in outcome.matches if m.exact_id is None)


def test_sub_brand_ties_follow_the_words_of_the_query(embedder: FakeEmbedder) -> None:
    """Lyxor and Amundi funds of one company clean to the same text and tie; the brand the
    user typed decides which comes first."""
    rows = [[f"{brand} {index} UCITS ETF", "Amundi"] for index in ("DAX", "CAC 40", "Euro Stoxx 50") for brand in ("Amundi", "Lyxor")]
    rows += [[f"Vanguard {index} UCITS ETF", "Vanguard"] for index in ("DAX", "CAC 40", "Euro Stoxx 50")]
    table = _table(["Fund name", "Brokerage"], rows)
    index = SearchIndex.build(table, {"Fund name": Role.SEARCH, "Brokerage": Role.DECIDING}, embedder=embedder)
    assert index.learned_aliases() == {"Brokerage": {"Amundi": ["Lyxor"]}}
    for query, expected in [
        ("Lyxor DAX", "Lyxor DAX UCITS ETF"),
        ("Amundi DAX", "Amundi DAX UCITS ETF"),
        ("Amndi DAX", "Amundi DAX UCITS ETF"),  # the typo is resolved to the key it matched
    ]:
        first, second = index.search(query, k=2).matches
        assert first.score == second.score  # identical cleaned text "DAX UCITS ETF"
        assert _name(table, first.row) == expected
    identity_only = index.search("Lyxor", k=3)
    assert [_name(table, m.row).split()[0] for m in identity_only.matches] == ["Lyxor"] * 3


def test_learned_aliases_and_cleaned_rows(index: SearchIndex) -> None:
    aliases = index.learned_aliases()
    assert aliases["Brokerage"]["BlackRock"][0] == "iShares"
    assert "Vanguard" not in aliases["Brokerage"]
    assert index.row_text(0).startswith("S&P 500 UCITS ETF")  # "iShares Core ... (Acc)" cleaned
    assert index.row_text(4) == "S&P 500 UCITS ETF USD Distributing"  # "Vanguard" cut


def test_use_family_false_disables_identity_logic(funds_table: Table, funds_roles: Roles) -> None:
    index = SearchIndex.build(funds_table, funds_roles, embedder=FakeEmbedder(), config=EngineConfig(use_family=False))
    assert index.learned_aliases() == {}
    assert index.row_text(0) == "iShares Core S&P 500 UCITS ETF USD (Acc)"
    outcome = index.search("iShares Core S&P 500", k=len(funds_table))
    assert outcome.evidence == [] and outcome.cleaned_query == "iShares Core S&P 500"
    assert all(m.family_factor == 1.0 and m.agreements == m.conflicts == [] for m in outcome.matches)
    assert index.search("IE00B4L5Y983", k=1).matches[0].exact_id == "ISIN"  # exact ids still work


# --- robustness ----------------------------------------------------------------------


def test_empty_and_whitespace_queries(index: SearchIndex) -> None:
    for query in ["", "   ", "\t", " - · ()"]:
        outcome = index.search(query)
        assert outcome.query == query and outcome.matches == [] and outcome.evidence == []


def test_table_without_search_columns(funds_table: Table) -> None:
    roles = {"ISIN": Role.DECIDING, "Fund name": Role.DECIDING, "Brokerage": Role.DECIDING, "Currency": Role.IGNORE, "TER": Role.IGNORE}
    index = SearchIndex.build(funds_table, roles, embedder=FakeEmbedder())
    assert index.search_columns == [] and index.primary_column == "ISIN"
    assert index.row_text(0) == "IE00B5BMR087 · iShares Core S&P 500 UCITS ETF USD (Acc) · BlackRock"
    outcome = index.search("Vanguard FTSE All-World Distributing", k=3)
    _check_outcome(outcome, 3)
    assert _name(funds_table, outcome.matches[0].row) == "Vanguard FTSE All-World UCITS ETF USD Distributing"


def test_only_ignored_columns() -> None:
    table = _table(["A", "B"], [["x", "y"], ["z", "w"]])
    index = SearchIndex.build(table, {"A": Role.IGNORE, "B": Role.IGNORE})
    assert index.primary_column == "A" and index.search("x").matches == []


def test_empty_table(embedder: FakeEmbedder) -> None:
    table = _table(["Fund name", "Brokerage"], [])
    index = SearchIndex.build(table, {"Fund name": Role.SEARCH, "Brokerage": Role.DECIDING}, embedder=embedder)
    assert index.search("anything").matches == []


def test_deciding_column_with_empty_cells(embedder: FakeEmbedder) -> None:
    rows = [
        ["iShares Core S&P 500 UCITS ETF", "BlackRock"],
        ["iShares Core MSCI World UCITS ETF", "BlackRock"],
        ["iShares MSCI EM UCITS ETF", "BlackRock"],
        ["Vanguard S&P 500 UCITS ETF", "Vanguard"],
        ["Vanguard FTSE All-World UCITS ETF", "Vanguard"],
        ["Vanguard Global Bond Index", "Vanguard"],
        ["Mystery S&P 500 UCITS ETF", ""],
    ]
    table = _table(["Fund name", "Brokerage"], rows)
    index = SearchIndex.build(table, {"Fund name": Role.SEARCH, "Brokerage": Role.DECIDING}, embedder=embedder)
    by_row = {m.row: m for m in index.search("Vanguard S&P 500", k=7).matches}
    assert by_row[3].family_factor == pytest.approx(1.0)
    assert by_row[6].family_factor == pytest.approx(1 - 0.85 * 0.5)
    assert by_row[0].family_factor == pytest.approx(0.15)
    assert by_row[6].conflicts == ["Brokerage"]
    assert index.row_text(6) == "Mystery S&P 500 UCITS ETF"


def test_accented_spanish_text(embedder: FakeEmbedder) -> None:
    rows = [
        ["Bankinter Índice España FI", "Bankinter Gestión de Activos"],
        ["Bankinter Renta Fija Corto Plazo FI", "Bankinter Gestión de Activos"],
        ["Bankinter Acciones Españolas FI", "Bankinter Gestión de Activos"],
        ["Santander Índice España FI", "Santander Asset Management"],
        ["Santander Renta Fija Corto Plazo FI", "Santander Asset Management"],
        ["Santander Acciones Españolas FI", "Santander Asset Management"],
    ]
    table = _table(["Nombre", "Gestora"], rows)
    index = SearchIndex.build(table, {"Nombre": Role.SEARCH, "Gestora": Role.DECIDING}, embedder=embedder)
    assert index.row_text(0) == "Índice España FI"
    outcome = index.search("GESTIÓN Bankinter: Índice España", k=3)
    assert outcome.cleaned_query == "Índice España"
    assert outcome.evidence[0].value == "Bankinter Gestión de Activos"
    assert outcome.evidence[0].matched_text == "GESTIÓN Bankinter"
    assert outcome.matches[0].row == 0
    assert index.search("santander acciones espanolas", k=1).matches[0].row == 5  # accent-insensitive


def test_ordinary_word_ticker_does_not_hijack(embedder: FakeEmbedder) -> None:
    rows = [
        ["Vanguard FTSE All-World UCITS ETF (Dist)", "VWRL", "Vanguard"],
        ["Vanguard FTSE All-World UCITS ETF (Acc)", "VWRA", "Vanguard"],
        ["Vanguard S&P 500 UCITS ETF", "VUSA", "Vanguard"],
        ["Allstate Insurance Equity Portfolio", "ALL", "Allstate"],
        ["Allstate Bond Income Trust", "ALB", "Allstate"],
        ["Allstate Growth Fund", "ALG", "Allstate"],
        # "all" is an ordinary word of this corpus (df > 2), not a misspelled brand
        ["SPDR MSCI All Country World UCITS ETF", "SPYY", "State Street"],
        ["iShares MSCI ACWI All Cap UCITS ETF", "ISAC", "BlackRock"],
    ]
    table = _table(["Fund name", "Ticker", "Brokerage"], rows)
    roles = {"Fund name": Role.SEARCH, "Ticker": Role.DECIDING, "Brokerage": Role.DECIDING}
    index = SearchIndex.build(table, roles, embedder=embedder)
    for query in ["FTSE All World", "Vanguard FTSE All-World", "all world fund"]:
        outcome = index.search(query, k=6)
        assert outcome.matches[0].row in (0, 1), query
        assert all(m.exact_id is None for m in outcome.matches)
        assert "all" in tokens(outcome.cleaned_query)  # too uncertain to be cut
    exact = index.search("ALL", k=1).matches[0]  # the whole query is the ticker: exact
    assert exact.row == 3 and exact.exact_id == "Ticker"
    assert index.search("VWRA", k=1).matches[0].exact_id == "Ticker"


def test_k_larger_than_table_and_non_positive_k(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("MSCI World", k=1000)
    assert 0 < len(outcome.matches) <= len(funds_table)
    assert len({m.row for m in outcome.matches}) == len(outcome.matches)
    assert index.search("MSCI World", k=0).matches == []
    assert len(index.search("MSCI World", k=1).matches) == 1


def test_deterministic_order_score_then_row() -> None:
    rows = [["Same Fund Name", "X"] if i % 2 == 0 else ["Something else entirely", "Y"] for i in range(10)]
    table = _table(["Fund name", "Brokerage"], rows)
    index = SearchIndex.build(table, {"Fund name": Role.SEARCH, "Brokerage": Role.IGNORE}, embedder=FakeEmbedder())
    first = [(m.row, m.score) for m in index.search("same fund name", k=4).matches]
    assert [row for row, _ in first] == [0, 2, 4, 6]  # equal scores: lower row first
    assert first == [(m.row, m.score) for m in index.search("same fund name", k=4).matches]


def test_lexical_only_without_embedder(funds_table: Table, funds_roles: Roles) -> None:
    index = SearchIndex.build(funds_table, funds_roles)
    assert not index.has_semantic
    outcome = index.search("iShares Core S&P 500", k=5)
    assert _brokerage(funds_table, outcome.matches[0].row) == "BlackRock"
    assert all(m.semantic == 0.0 and m.text_score == pytest.approx(m.lexical) for m in outcome.matches)


# --- batches, progress, cancel, cache ---------------------------------------------------------


def test_search_many_embeds_once_and_reports_progress(index: SearchIndex, embedder: FakeEmbedder) -> None:
    embedder.calls.clear()
    events: list[tuple[str, float]] = []
    queries = ["iShares Core S&P 500", "Vanguard FTSE All-World", "iShares Core S&P 500", "IE00B4L5Y983", ""]
    outcomes = index.search_many(queries, k=3, progress=lambda stage, f: events.append((stage, f)))
    assert [o.query for o in outcomes] == queries
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["s&p 500", "ftse all-world"]  # distinct cleaned texts, case-folded
    assert {stage for stage, _ in events} == {STAGE_SEARCH}
    fractions = [f for _, f in events]
    assert fractions == sorted(fractions) and fractions[0] == 0.0 and fractions[-1] == pytest.approx(1.0)
    assert [(m.row, m.score) for m in outcomes[0].matches] == [(m.row, m.score) for m in outcomes[2].matches]
    single = index.search("Vanguard FTSE All-World", k=3)
    assert [(m.row, m.score) for m in single.matches] == [(m.row, m.score) for m in outcomes[1].matches]


def test_search_many_cancel(index: SearchIndex) -> None:
    with pytest.raises(Cancelled):
        index.search_many(["S&P 500"] * 5, cancel=lambda: True)
    calls = iter(range(100))
    with pytest.raises(Cancelled):  # cancelled after a couple of queries
        index.search_many(["MSCI World"] * 40, cancel=lambda: next(calls) >= 3)


def test_build_progress_stages(funds_table: Table, funds_roles: Roles) -> None:
    events: list[tuple[str, float]] = []
    SearchIndex.build(funds_table, funds_roles, embedder=FakeEmbedder(), progress=lambda s, f: events.append((s, f)))
    stages = list(dict.fromkeys(stage for stage, _ in events))
    assert stages == [STAGE_ANALYZE, STAGE_ALIASES, STAGE_INDEX, STAGE_EMBED, STAGE_FINAL]
    assert all(0.0 <= f <= 1.0 for _, f in events)
    assert (STAGE_EMBED, 1.0) in events
    lexical_only: list[str] = []
    SearchIndex.build(funds_table, funds_roles, progress=lambda s, f: lexical_only.append(s))
    assert STAGE_EMBED not in lexical_only


def test_build_cancel(funds_table: Table, funds_roles: Roles) -> None:
    with pytest.raises(Cancelled):
        SearchIndex.build(funds_table, funds_roles, cancel=lambda: True)
    stage = {"name": ""}

    def cancel_while_embedding() -> bool:
        return stage["name"] == STAGE_EMBED

    with pytest.raises(Cancelled):
        SearchIndex.build(
            funds_table,
            funds_roles,
            embedder=FakeEmbedder(),
            progress=lambda s, f: stage.update(name=s),
            cancel=cancel_while_embedding,
        )


def test_embedding_cache_reuse_and_write_back(funds_table: Table, funds_roles: Roles) -> None:
    cache = DictCache()
    first_embedder = FakeEmbedder()
    first = SearchIndex.build(funds_table, funds_roles, embedder=first_embedder, cache=cache)
    # One search column here, so a row's text is that column's cleaned cell; embedded case-folded.
    row_texts = {first.row_text(r).lower() for r in range(len(funds_table))}
    assert set(first_embedder.encoded) == row_texts and len(first_embedder.encoded) == len(row_texts)
    assert set(cache.put_texts) == row_texts

    again = FakeEmbedder()
    second = SearchIndex.build(funds_table, funds_roles, embedder=again, cache=cache)
    assert again.calls == []  # everything came from the cache
    a, b = first.search("Vanguard FTSE All-World", k=5), second.search("Vanguard FTSE All-World", k=5)
    assert [m.row for m in a.matches] == [m.row for m in b.matches]
    assert [m.score for m in a.matches] == pytest.approx([m.score for m in b.matches], abs=2e-3)

    changed = Table(
        funds_table.source_name,
        funds_table.sheet_name,
        list(funds_table.columns),
        [list(row) for row in funds_table.rows],
        list(funds_table.source_rows),
    )
    changed.rows[6][1] = "Vanguard FTSE All-World High Dividend UCITS ETF"
    third_embedder = FakeEmbedder()
    SearchIndex.build(changed, funds_roles, embedder=third_embedder, cache=cache)
    assert third_embedder.encoded == ["ftse all-world high dividend ucits etf"]
    assert cache.put_texts[-1] == "ftse all-world high dividend ucits etf"


def test_to_history_entry_snapshot(index: SearchIndex, funds_table: Table) -> None:
    outcome = index.search("iShares Core S&P 500", k=MAX_STORED_RESULTS + 10)
    entry = index.to_history_entry(outcome, "2026-11-01T17:32:05.123456+00:00")
    assert entry.id is None and entry.query == "iShares Core S&P 500"
    assert entry.created_at == "2026-11-01T17:32:05.123456+00:00"
    assert entry.dataset_label == "funds.xlsx · Funds"
    assert entry.columns == funds_table.columns and entry.columns is not funds_table.columns
    assert entry.deciding == ["ISIN", "Brokerage"] and entry.primary_column == "Fund name"
    assert [e.value for e in entry.evidence] == ["BlackRock"]
    assert len(entry.results) == min(len(outcome.matches), MAX_STORED_RESULTS)
    for rank, (result, match) in enumerate(zip(entry.results, outcome.matches), start=1):
        assert result.rank == rank and result.score == match.score
        assert result.cells == funds_table.rows[match.row] and len(result.cells) == len(entry.columns)
        assert result.source_row == funds_table.source_rows[match.row]
        assert set(result.detail) == {"semantic", "lexical", "text_score", "family_factor", "exact_id", "agreements", "conflicts"}
        assert result.detail["family_factor"] == match.family_factor
        json.dumps(result.detail)  # persisted as JSON
    entry.results[0].cells[0] = "mutated"
    assert funds_table.rows[outcome.matches[0].row][0] != "mutated"  # a copy, not a view


def test_concurrent_searches_match_sequential(index: SearchIndex) -> None:
    queries = ["iShares Core S&P 500", "Vanguard S&P 500", "MSCI World", "IE00B4L5Y983", "SPDR", "Amundi EM"] * 5
    expected = [[(m.row, m.score) for m in index.search(q, k=5).matches] for q in queries]
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda q: [(m.row, m.score) for m in index.search(q, k=5).matches], queries))
    assert results == expected


# --- optional: the real multilingual model ------------------------------------------------


class _OnnxMiniLM:
    """Minimal private loader for the bundled model (mirrors ARCHITECTURE.md section 8)."""

    dim = 384

    def __init__(self, folder: Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self.model_id = f"test:{folder.name}"
        self._tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self._tokenizer.enable_truncation(128)
        self._tokenizer.enable_padding()
        self._session = onnxruntime.InferenceSession(str(folder / "model.onnx"), providers=["CPUExecutionProvider"])

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(texts), 64):
            batch = self._tokenizer.encode_batch(texts[start : start + 64])
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            hidden = self._session.run(
                None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)}
            )[0]
            pooled = (hidden * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
            out[start : start + len(batch)] = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
            if progress is not None:
                progress(start + len(batch), len(texts))
        return out


@lru_cache(maxsize=1)
def _real_model(folder: Path) -> _OnnxMiniLM:
    return _OnnxMiniLM(folder)


def test_real_model_sanity(model_dir: Path, funds_table: Table, funds_roles: Roles) -> None:
    index = SearchIndex.build(funds_table, funds_roles, embedder=_real_model(model_dir))
    expectations = {
        "iShares Core S&P 500": "iShares Core S&P 500 UCITS ETF USD (Acc)",
        "Vanguard FTSE All World acumulación": "Vanguard FTSE All-World UCITS ETF USD Accumulating",
        "fondo indexado S&P 500 de Vanguard que reparte dividendos": "Vanguard S&P 500 UCITS ETF USD Distributing",
        "BlackRock MSCI Europa sostenible": "iShares MSCI Europe SRI UCITS ETF EUR (Acc)",
    }
    for query, expected in expectations.items():
        outcome = index.search(query, k=5)
        _check_outcome(outcome, 5)
        assert _name(funds_table, outcome.matches[0].row) == expected, query
        assert outcome.matches[0].semantic > 0
