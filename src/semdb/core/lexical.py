"""Character n-gram + word TF-IDF index (numpy only), for exact-token and typo-tolerant matching.

Every folded token contributes the character 3-grams of its space-padded form (``" s&p "`` ->
``" s&"``, ``"s&p"``, ``"&p "``; a one-letter token gives its padded self) plus the word
features ``w:<token>`` and ``w:<part>`` for each part of :func:`text.split_alnum`, the latter
weighted x1.5.  Weights are ``(1 + log tf) * idf`` with ``idf = log((1 + N) / (1 + df)) + 1``;
document vectors are L2-normalized and stored as postings per feature (CSR), so scoring a
query only touches the documents that share one of its features.

Besides the symmetric cosine, :meth:`LexicalIndex.match` measures how much of the *query*
a document covers: features count once, weighted by ``(idf * boost)²``; with ``I`` the
mass shared by query and document, ``recall = I / M_query`` and the recall-weighted
``F_beta = (1 + beta²) I / (beta² M_query + M_doc)``.  A query that is fully contained in a
longer name scores high instead of being diluted by the name's extra words.
"""

from __future__ import annotations

import math
from array import array
from collections import Counter

import numpy as np

from .text import split_alnum, tokens
from .types import CancelFn, Cancelled

WORD_WEIGHT = 1.5
_WORD_PREFIX = "w:"
_CONCEPT_PREFIX = "c:"
_CANCEL_EVERY = 2000

# Abbreviations that fund and security names use for the same idea (English and Spanish, folded).
# Both spellings share a "c:<concept>" feature, so "EUR Hdg (Acc)" meets "EUR Hedged Accumulating"
# although their letters barely overlap.  Only unambiguous, name-level shorthand belongs here.
CONCEPTS: dict[str, str] = {
    token: concept
    for concept, spellings in {
        "accumulating": "acc accum accumulating accumulation capitalisation capitalization acumulacion acumulativo",
        "distributing": "dist distributing distribution distribucion reparto",
        "hedged": "hdg hedged cubierto cubierta",
        "government": "govt gov government gobierno",
        "corporate": "corp corporate corporativo corporativa",
        "international": "intl international internacional",
        "inflation": "infl inflation inflacion",
        "treasury": "tsy treasury tesoro",
        "emerging": "em emerging emergentes emergente",
        "markets": "mkt mkts market markets mercados mercado",
        "dividend": "div dividend dividends dividendo dividendos",
        "bond": "bd bond bonds bono bonos",
        "equity": "eq equity equities acciones",
        "year": "yr yrs year years ano anos",
    }.items()
    for token in spellings.split()
}


def token_features(token: str) -> list[str]:
    """Features of one folded token: padded character 3-grams, then distinct word (and concept) features."""
    padded = f" {token} "
    grams = [padded[i : i + 3] for i in range(len(padded) - 2)]
    words = dict.fromkeys([_WORD_PREFIX + token, *(_WORD_PREFIX + part for part in split_alnum(token))])
    concept = CONCEPTS.get(token)
    if concept is not None:
        words[_CONCEPT_PREFIX + concept] = None
    return grams + list(words)


def _feature_weight(feature: str) -> float:
    return WORD_WEIGHT if feature.startswith((_WORD_PREFIX, _CONCEPT_PREFIX)) else 1.0


class LexicalIndex:
    """Cosine similarity between a query and every document, via an inverted index."""

    __slots__ = ("_n", "_vocab", "_idf", "_max_idf", "_indptr", "_doc_ids", "_weights", "_feature_mass", "_doc_mass")

    def __init__(self, documents: list[str], cancel: CancelFn | None = None) -> None:
        n = len(documents)
        vocab: dict[str, int] = {}
        boosts = array("d")
        token_ids: dict[str, array] = {}  # tokens repeat a lot (ucits, etf, usd...): featurize once
        features = array("i")
        lengths = np.zeros(n, dtype=np.int64)
        for doc_index, document in enumerate(documents):
            if cancel is not None and doc_index % _CANCEL_EVERY == 0 and cancel():
                raise Cancelled()
            before = len(features)
            for token in tokens(document):
                ids = token_ids.get(token)
                if ids is None:
                    ids = array("i")
                    for feature in token_features(token):
                        feature_id = vocab.get(feature)
                        if feature_id is None:
                            feature_id = vocab[feature] = len(vocab)
                            boosts.append(_feature_weight(feature))
                        ids.append(feature_id)
                    token_ids[token] = ids
                features.extend(ids)
            lengths[doc_index] = len(features) - before

        n_features = len(vocab)
        if len(features) == 0:
            feature_of = np.zeros(0, dtype=np.int64)
            doc_of = np.zeros(0, dtype=np.int32)
            tf = np.zeros(0, dtype=np.int64)
        else:
            feats = np.frombuffer(features, dtype=np.int32).astype(np.int64)
            docs = np.repeat(np.arange(n, dtype=np.int64), lengths)
            # One sort gives term frequencies *and* the CSR order (feature, then document).
            keys, tf = np.unique(feats * n + docs, return_counts=True)
            feature_of = keys // n
            doc_of = (keys - feature_of * n).astype(np.int32)
        df = np.bincount(feature_of, minlength=n_features)
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        boost = np.frombuffer(boosts, dtype=np.float64)
        feature_mass = (idf * boost) ** 2
        doc_mass = np.bincount(doc_of, weights=feature_mass[feature_of], minlength=n)
        weights = (1.0 + np.log(tf)) * idf[feature_of] * boost[feature_of]
        norms = np.sqrt(np.bincount(doc_of, weights=weights * weights, minlength=n))
        if weights.size:
            weights /= norms[doc_of]
        indptr = np.zeros(n_features + 1, dtype=np.int64)
        np.cumsum(df, out=indptr[1:])

        self._n = n
        self._vocab = vocab
        self._idf = idf
        self._max_idf = math.log(1.0 + n) + 1.0  # idf of a feature no document has
        self._indptr = indptr
        self._doc_ids = doc_of
        self._weights = weights.astype(np.float32)
        self._feature_mass = feature_mass
        self._doc_mass = doc_mass
        for frozen in (self._idf, self._indptr, self._doc_ids, self._weights, self._feature_mass, self._doc_mass):
            frozen.flags.writeable = False

    def __len__(self) -> int:
        return self._n

    def scores(self, query: str) -> np.ndarray:
        """float32 array (n_documents,) of cosine similarities in [0, 1]."""
        counts = Counter(feature for token in tokens(query) for feature in token_features(token))
        if not counts or self._n == 0:
            return np.zeros(self._n, dtype=np.float32)
        ids: list[int] = []
        query_weights: list[float] = []
        norm_sq = 0.0
        for feature, tf in counts.items():
            feature_id = self._vocab.get(feature)
            idf = self._max_idf if feature_id is None else float(self._idf[feature_id])
            weight = (1.0 + math.log(tf)) * idf * _feature_weight(feature)
            norm_sq += weight * weight
            if feature_id is not None:
                ids.append(feature_id)
                query_weights.append(weight)
        if not ids:
            return np.zeros(self._n, dtype=np.float32)
        scale = 1.0 / math.sqrt(norm_sq)
        starts = self._indptr[ids].tolist()
        ends = self._indptr[np.asarray(ids) + 1].tolist()
        docs = np.concatenate([self._doc_ids[s:e] for s, e in zip(starts, ends)])
        values = np.concatenate(
            [self._weights[s:e] * (w * scale) for s, e, w in zip(starts, ends, query_weights)]
        )
        out = np.bincount(docs, weights=values, minlength=self._n).astype(np.float32)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def match(self, query: str, beta: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """``(F_beta, recall)``, float32 arrays (n_documents,) in [0, 1]: how completely each
        document covers the query's features (see the module docstring)."""
        zeros = np.zeros(self._n, dtype=np.float32)
        features = {feature for token in tokens(query) for feature in token_features(token)}
        if not features or self._n == 0:
            return zeros, zeros.copy()
        ids: list[int] = []
        query_mass = 0.0
        for feature in features:
            feature_id = self._vocab.get(feature)
            if feature_id is None:
                query_mass += (self._max_idf * _feature_weight(feature)) ** 2
            else:
                ids.append(feature_id)
                query_mass += float(self._feature_mass[feature_id])
        if not ids:
            return zeros, zeros.copy()
        starts = self._indptr[ids]
        lengths = self._indptr[np.asarray(ids) + 1] - starts
        docs = np.concatenate([self._doc_ids[s : s + length] for s, length in zip(starts.tolist(), lengths.tolist())])
        shared = np.bincount(docs, weights=np.repeat(self._feature_mass[ids], lengths), minlength=self._n)
        b2 = beta * beta
        f_beta = (1.0 + b2) * shared / (b2 * query_mass + self._doc_mass)
        recall = shared / query_mass
        return np.clip(f_beta, 0.0, 1.0).astype(np.float32), np.clip(recall, 0.0, 1.0).astype(np.float32)
