"""Lexical index: features, TF-IDF weighting and cosine scores (ARCHITECTURE.md section 7)."""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np
import pytest

from semdb.core.lexical import WORD_WEIGHT, LexicalIndex, token_features
from semdb.core.text import tokens
from semdb.core.types import Cancelled

DOCS = [
    "iShares Core S&P 500 UCITS ETF USD (Acc)",
    "Vanguard S&P 500 UCITS ETF USD Distributing",
    "SPDR S&P 400 US Mid Cap UCITS ETF",
    "Vanguard FTSE All-World UCITS ETF",
    "Xtrackers MSCI World UCITS ETF 1C",
    "Amundi Gestión Renta Fija Corto Plazo",
    "",
]


def _dense_reference(documents: list[str], query: str) -> np.ndarray:
    """Straightforward dense TF-IDF cosine, independent of the CSR implementation."""
    doc_counts = [Counter(f for t in tokens(d) for f in token_features(t)) for d in documents]
    n = len(documents)
    df = Counter(f for counts in doc_counts for f in counts)
    max_idf = math.log(1 + n) + 1

    def vector(counts: Counter[str]) -> dict[str, float]:
        out = {}
        for feature, tf in counts.items():
            idf = math.log((1 + n) / (1 + df[feature])) + 1 if feature in df else max_idf
            out[feature] = (1 + math.log(tf)) * idf * (WORD_WEIGHT if feature.startswith(("w:", "c:")) else 1.0)
        return out

    q = vector(Counter(f for t in tokens(query) for f in token_features(t)))
    q_norm = math.sqrt(sum(v * v for v in q.values()))
    scores = []
    for counts in doc_counts:
        d = vector(counts)
        d_norm = math.sqrt(sum(v * v for v in d.values()))
        dot = sum(q[f] * d[f] for f in q if f in d)
        scores.append(dot / (q_norm * d_norm) if q_norm and d_norm else 0.0)
    return np.array(scores)


def test_token_features() -> None:
    assert token_features("s&p") == [" s&", "s&p", "&p ", "w:s&p"]
    assert token_features("a") == [" a ", "w:a"]
    # split_alnum parts become extra word features, without duplicates
    assert token_features("s&p500")[-3:] == ["w:s&p500", "w:s&p", "w:500"]
    assert token_features("3yr")[-3:] == ["w:3yr", "w:3", "w:yr"]


def test_scores_match_dense_reference() -> None:
    rng = random.Random(7)
    words = ["ishares", "core", "s&p", "500", "msci", "world", "ucits", "etf", "usd", "acc", "dist", "gestión", "1c"]
    documents = [" ".join(rng.choice(words) for _ in range(rng.randint(0, 8))) for _ in range(60)]
    index = LexicalIndex(documents)
    for query in ["core s&p 500", "MSCI World ETF", "gestion zzz", "s&p500 acc", "unseen words only"]:
        np.testing.assert_allclose(index.scores(query), _dense_reference(documents, query), atol=1e-5)


def test_identical_document_scores_one_and_shape() -> None:
    index = LexicalIndex(DOCS)
    scores = index.scores(DOCS[0])
    assert scores.dtype == np.float32 and scores.shape == (len(DOCS),)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert np.all((scores >= 0) & (scores <= 1))
    assert len(index) == len(DOCS)


def test_exact_numbers_matter() -> None:
    index = LexicalIndex(DOCS)
    scores = index.scores("S&P 500")
    assert scores[0] > scores[2] and scores[1] > scores[2]  # 500 is not 400


def test_typos_accents_and_glued_tokens() -> None:
    index = LexicalIndex(DOCS)
    assert int(np.argmax(index.scores("Vangard FTSE Al World"))) == 3
    assert int(np.argmax(index.scores("GESTION renta fija"))) == 5
    assert int(np.argmax(index.scores("Core S&P500"))) == 0  # w:s&p and w:500 from split_alnum


def test_unseen_query_features_lower_the_score() -> None:
    index = LexicalIndex(DOCS)
    plain = index.scores("Xtrackers MSCI World")[4]
    noisy = index.scores("Xtrackers MSCI World qwzx")[4]
    assert 0 < noisy < plain


def test_empty_inputs() -> None:
    index = LexicalIndex(DOCS)
    assert not index.scores("").any()
    assert not index.scores("  ·  ").any()
    assert index.scores("etf")[6] == 0.0  # the empty document
    empty = LexicalIndex([])
    assert len(empty) == 0 and empty.scores("anything").shape == (0,)
    blank = LexicalIndex(["", ""])
    assert not blank.scores("abc").any()


def test_cancel() -> None:
    with pytest.raises(Cancelled):
        LexicalIndex(DOCS, cancel=lambda: True)


def test_index_arrays_are_read_only() -> None:
    index = LexicalIndex(DOCS)
    with pytest.raises(ValueError):
        index._weights[0] = 0.0


def test_abbreviation_concepts_link_spellings_with_no_letters_in_common() -> None:
    index = LexicalIndex([
        "MSCI USA UCITS ETF EUR Hedged (Dist)",
        "MSCI USA UCITS ETF EUR (Dist)",
        "MSCI USA UCITS ETF EUR Hedged (Acc)",
    ])
    f_beta, _ = index.match("MSCI USA UCITS ETF EUR Hdg Distributing")  # what ranking uses
    assert int(f_beta.argmax()) == 0  # Hdg = Hedged and Distributing = Dist
    _, recall = LexicalIndex(["Treasury Bond Accumulating", "Gold Mining Equity"]).match("bonos tesoro acumulación")
    assert recall[0] > 0.1 and recall[1] == 0  # Spanish spellings meet English ones via shared concepts
