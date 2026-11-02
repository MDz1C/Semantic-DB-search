"""Deciding columns: value vocabularies, automatically learned aliases, query detection and
row-local cleaning of identity fragments.  See docs/ARCHITECTURE.md, "Family model"."""

from __future__ import annotations

from array import array
from collections import Counter
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from rapidfuzz import fuzz, process

from .text import is_identifier_column, looks_like_code, ngrams, normalize, remove_spans, token_spans, tokens
from .types import CancelFn, Cancelled, FamilyEvidence, Table

SMOOTHING = 2.0  # alpha of the smoothed text precision of value tokens and phrases
GENERIC_VALUES = 3  # a token shared by more values than this is down-weighted (3 / nv)
PHRASE_PRECISION_WEIGHT = 0.9
AMBIGUOUS_ID_WEIGHT = 0.3  # short identifier that is also an ordinary word of the corpus
MIN_STRENGTH = 0.25  # detected values weaker than this are dropped
CUT_WEIGHT = 0.5  # indicators at least this strong are cut from queries and rows
MAX_ALIAS_VALUES = 5000
MIN_FAMILY_SIZE = 3
ALIAS_SUPPORT = 3
ALIAS_COVERAGE = 0.2
ALIAS_FULL_COVERAGE = 0.5
ALIAS_PRECISION = 0.85
MAX_FUZZY_DF = 2  # an established corpus word is not a misspelled brand
TYPO_RATIO = 85
ABBREVIATION_RATIO = 0.35
ABBREVIATION_WEIGHT = 0.6
MIN_FACTOR = 1e-3  # keeps factors in (0, 1] even at strictness 1
MIN_INDICATOR_WEIGHT = 0.01  # weaker indicators can never reach MIN_STRENGTH: not stored
_CANCEL_EVERY = 2000  # rows (or cheap items) between two polls of the cancel callback

Staging = dict[str, dict[int, float]]  # key -> {value id: weight} while building
Pattern = tuple[str, ...]  # a key split into tokens, matched against a cell's tokens


@dataclass(slots=True)
class DecidingColumn:
    name: str
    values: list[str]  # distinct display values, id = position
    codes: np.ndarray  # int32 per table row: value id, -1 when the cell is empty
    is_identifier: bool  # code-like and nearly unique per row (ISIN, ticker)
    indicators: dict[str, list[tuple[int, float]]]  # normalized token/phrase -> [(value id, weight)]
    learned_aliases: dict[int, list[str]] = field(default_factory=dict)  # value id -> mined alias tokens


@dataclass(slots=True)
class Detection:
    """What the query says about identity."""

    evidence: dict[str, dict[int, float]]  # column -> {value id: strength in (0, 1]}
    spans: list[tuple[int, int]]  # character spans of the query to drop before text matching
    found: list[FamilyEvidence]  # human-readable, strongest first
    keys: frozenset[str] = frozenset()  # indicator keys behind the spans ("amundi" for "Amndi")


@dataclass(slots=True)
class _ColumnAux:
    """Lookup structures derived from a DecidingColumn at build time."""

    value_tokens: list[Pattern]  # normalized tokens of every value
    removals: list[dict[str, list[Pattern]]] | None  # per value: first token -> patterns to cut
    typo_keys: list[str]  # unigram keys of length >= 4
    abbreviation_keys: dict[str, list[str]]  # first letter -> unigram keys
    alias_display: dict[str, str]  # learned alias token -> most common spelling in the data


def _check(cancel: CancelFn | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled()


class _TextStats:
    """Token statistics of the search text; only needed while building."""

    def __init__(self, table: Table, search: list[str], cancel: CancelFn | None = None) -> None:
        columns = [table.column_index(name) for name in search]
        vocab: dict[str, int] = {}
        per_cell: dict[str, tuple[list[str], str]] = {}
        pair_tokens = array("i")
        pair_counts = np.zeros(len(table.rows), dtype=np.int64)
        row_text: list[str] = []
        for r, row in enumerate(table.rows):
            if r % _CANCEL_EVERY == 0:
                _check(cancel)
            row_tokens: list[str] = []
            parts: list[str] = []
            for j in columns:
                cell = row[j]
                entry = per_cell.get(cell)
                if entry is None:
                    toks = tokens(cell)
                    entry = per_cell[cell] = (toks, " ".join(toks))
                if entry[0]:
                    row_tokens.extend(entry[0])
                    parts.append(entry[1])
            # "\x1f" separates cells so a phrase never matches across two of them.
            row_text.append(" " + " \x1f ".join(parts) + " ")
            distinct = dict.fromkeys(row_tokens)
            pair_tokens.extend([vocab.setdefault(t, len(vocab)) for t in distinct])
            pair_counts[r] = len(distinct)
        self.vocab = vocab
        self.words = list(vocab)
        self.row_text = row_text
        self.pair_tokens = np.array(pair_tokens, dtype=np.int64)
        self.pair_rows = np.repeat(np.arange(len(table.rows), dtype=np.int64), pair_counts)
        self.df = np.bincount(self.pair_tokens, minlength=len(vocab))
        self._postings = self.pair_rows[np.argsort(self.pair_tokens, kind="stable")]
        self._indptr = np.concatenate(([0], np.cumsum(self.df)))

    def text_df(self, token: str) -> int:
        token_id = self.vocab.get(token)
        return 0 if token_id is None else int(self.df[token_id])

    def df_of(self, token_ids: np.ndarray) -> np.ndarray:
        """Document frequencies for an id array where -1 means "not in the text"."""
        out = np.zeros(token_ids.shape, dtype=np.int64)
        known = token_ids >= 0
        out[known] = self.df[token_ids[known]]
        return out

    def rows_with(self, token_id: int) -> np.ndarray:
        return self._postings[self._indptr[token_id] : self._indptr[token_id + 1]]

    def phrase_rows(self, toks: Pattern) -> np.ndarray:
        """Rows whose text contains the phrase (contiguous tokens) or its space-less join."""
        found: list[int] = []
        ids = [self.vocab.get(t) for t in toks]
        if None not in ids:
            rarest = min(ids, key=lambda token_id: self.df[token_id])
            needle = " " + " ".join(toks) + " "
            found.extend(r for r in self.rows_with(rarest).tolist() if needle in self.row_text[r])
        joined = self.vocab.get("".join(toks))
        if joined is not None:
            found.extend(self.rows_with(joined).tolist())
        return np.unique(np.asarray(found, dtype=np.int64))


class _PairCounts:
    """n(v, t): rows with deciding value v whose search text contains token t."""

    def __init__(self, codes: np.ndarray, stats: _TextStats) -> None:
        self.width = max(len(stats.vocab), 1)
        member = codes[stats.pair_rows]
        keep = member >= 0
        keys = member[keep].astype(np.int64) * self.width + stats.pair_tokens[keep]
        self.keys, self.counts = np.unique(keys, return_counts=True)

    def count(self, values: np.ndarray, token_ids: np.ndarray) -> np.ndarray:
        if not self.keys.size or not values.size:
            return np.zeros(values.shape, dtype=np.int64)
        wanted = values * self.width + token_ids
        pos = np.minimum(np.searchsorted(self.keys, wanted), self.keys.size - 1)
        return np.where(self.keys[pos] == wanted, self.counts[pos], 0)


def _add(staging: Staging, key: str, value: int, weight: float) -> None:
    """Record ``key -> value``; a repeated (key, value) keeps the larger weight."""
    if weight < MIN_INDICATOR_WEIGHT:
        return
    entries = staging.setdefault(key, {})
    if weight > entries.get(value, 0.0):
        entries[value] = weight


def _vocabulary(raw: list[str], cancel: CancelFn | None = None) -> tuple[np.ndarray, list[str], list[Pattern]]:
    codes = np.full(len(raw), -1, dtype=np.int32)
    ids: dict[str, int] = {}
    values: list[str] = []
    value_tokens: list[Pattern] = []
    normalized: dict[str, str] = {}  # deciding values repeat a lot: normalize each once
    for r, cell in enumerate(raw):
        if r % _CANCEL_EVERY == 0:
            _check(cancel)
        key = normalized.get(cell)
        if key is None:
            key = normalized[cell] = normalize(cell)
        if not key:
            continue
        value_id = ids.get(key)
        if value_id is None:
            value_id = ids[key] = len(values)
            values.append(cell.strip())
            value_tokens.append(tuple(key.split(" ")))
        codes[r] = value_id
    return codes, values, value_tokens


def identifier_column(values: list[str]) -> bool:
    """``text.is_identifier_column`` with cheap exact pre-checks for repetitive or wordy columns."""
    filled = [cell for cell in values if cell.strip()]
    # Fewer distinct raw strings than 90 % can never pass (normalizing only merges values),
    if len(set(filled)) < 0.9 * len(filled):
        return False
    # nor can fewer than 90 % code-like cells.  A wordy column (names) shows that within its
    # first tenth, so only that many cells are counted: a code column then costs little extra.
    misses = 0
    for cell in filled[: len(filled) // 10 + 1000]:
        if not looks_like_code(cell):
            misses += 1
            if len(filled) - misses < 0.9 * len(filled):
                return False
    return is_identifier_column(values)


def _identifier_indicators(
    indicators: Staging, value_tokens: list[Pattern], stats: _TextStats, cancel: CancelFn | None = None
) -> None:
    for value_id, toks in enumerate(value_tokens):
        if value_id % _CANCEL_EVERY == 0:
            _check(cancel)
        phrase, compact = " ".join(toks), "".join(toks)
        if any(ch.isdigit() for ch in compact) or len(compact) >= 5:
            weight = 1.0
        else:
            # Tickers such as ALL that are also ordinary words must not hijack the ranking.
            df = min(stats.text_df(t) for t in toks)
            weight = 1.0 if df == 0 else AMBIGUOUS_ID_WEIGHT
        _add(indicators, phrase, value_id, weight)
        _add(indicators, compact, value_id, weight)


def _value_token_indicators(
    indicators: Staging, value_tokens: list[Pattern], pairs: _PairCounts, stats: _TextStats
) -> tuple[dict[str, list[int]], list[float]]:
    """Index each value token with ``generic * smoothed text precision``; returns the
    token -> values map and every value's best token weight."""
    token_values: dict[str, list[int]] = {}
    for value_id, toks in enumerate(value_tokens):
        for token in dict.fromkeys(toks):
            token_values.setdefault(token, []).append(value_id)
    best = [0.0] * len(value_tokens)
    if not token_values:
        return token_values, best
    names = list(token_values)
    holders = list(token_values.values())
    nv = np.fromiter((len(h) for h in holders), dtype=np.int64, count=len(holders))
    token_ids = np.fromiter((stats.vocab.get(t, -1) for t in names), dtype=np.int64, count=len(names))
    flat_token = np.repeat(np.arange(len(names)), nv)
    flat_value = np.fromiter(chain.from_iterable(holders), dtype=np.int64, count=int(nv.sum()))
    flat_ids = token_ids[flat_token]
    in_text = flat_ids >= 0
    # rows in S (value contains t) whose text also contains t = sum over v of n(v, t)
    hits = np.bincount(
        flat_token[in_text],
        weights=pairs.count(flat_value[in_text], flat_ids[in_text]),
        minlength=len(names),
    )
    precision = (hits + SMOOTHING) / (stats.df_of(token_ids) + SMOOTHING)
    generic = np.where(nv <= GENERIC_VALUES, 1.0, GENERIC_VALUES / nv)
    for name, holder, weight in zip(names, holders, (generic * precision).tolist()):
        short = len(name) <= 2
        for value_id in holder:
            if short and len(value_tokens[value_id]) > 1:
                continue
            _add(indicators, name, value_id, weight)
            best[value_id] = max(best[value_id], weight)
    return token_values, best


def _phrase_indicators(
    indicators: Staging,
    value_tokens: list[Pattern],
    token_values: dict[str, list[int]],
    best: list[float],
    codes: np.ndarray,
    stats: _TextStats,
    cancel: CancelFn | None = None,
) -> None:
    padded = [" " + " ".join(toks) + " " for toks in value_tokens]
    for value_id, toks in enumerate(value_tokens):
        if len(toks) < 2:
            continue
        _check(cancel)  # a phrase may scan many rows
        needle = padded[value_id]
        rarest = min(toks, key=lambda t: len(token_values[t]))
        holders = [u for u in token_values[rarest] if needle in padded[u]]
        occurrences = stats.phrase_rows(toks)
        hits = int(np.isin(codes[occurrences], holders).sum()) if occurrences.size else 0
        precision = (hits + SMOOTHING) / (occurrences.size + SMOOTHING)
        weight = max(PHRASE_PRECISION_WEIGHT * precision, best[value_id])
        _add(indicators, " ".join(toks), value_id, weight)
        _add(indicators, "".join(toks), value_id, weight)


def _learn_aliases(
    indicators: Staging,
    value_tokens: list[Pattern],
    codes: np.ndarray,
    pairs: _PairCounts,
    stats: _TextStats,
    cancel: CancelFn | None = None,
) -> dict[int, list[str]]:
    """Text tokens that (almost) only occur in one family's rows become indicators of it."""
    filled = codes[codes >= 0]
    sizes = np.bincount(filled, minlength=len(value_tokens))
    # Median over rows of the size of the row's family: is a typical row part of a group?
    typical = float(np.median(sizes[filled])) if filled.size else 0.0
    if len(value_tokens) > MAX_ALIAS_VALUES or typical < MIN_FAMILY_SIZE or not pairs.keys.size:
        return {}
    value_of = pairs.keys // pairs.width
    token_of = pairs.keys % pairs.width
    support = pairs.counts
    coverage = support / sizes[value_of]
    precision = support / stats.df[token_of]
    keep = (support >= ALIAS_SUPPORT) & (precision >= ALIAS_PRECISION)
    mined: dict[int, list[tuple[float, int, str]]] = {}
    rare: list[tuple[int, str, int, float, float]] = []  # precise but below the coverage threshold

    def learn(value_id: int, token: str, n: int, weight: float) -> None:
        _add(indicators, token, value_id, weight)
        mined.setdefault(value_id, []).append((-weight, -n, token))

    for i, (value_id, token_id, n, cov, prec) in enumerate(
        zip(
            value_of[keep].tolist(),
            token_of[keep].tolist(),
            support[keep].tolist(),
            coverage[keep].tolist(),
            precision[keep].tolist(),
        )
    ):
        if i % _CANCEL_EVERY == 0:
            _check(cancel)
        token = stats.words[token_id]
        if len(token) < 3 or token.isdigit() or token in value_tokens[value_id]:
            continue
        if cov >= ALIAS_COVERAGE:
            learn(value_id, token, n, prec * (0.5 + 0.5 * min(1.0, cov / ALIAS_FULL_COVERAGE)))
        else:
            rare.append((value_id, token, n, cov, prec))
    # A rare spelling of a strong key (ISHS / iShs for iShares, a recurring typo) is precise
    # and its spelling says what it means, unlike a product line such as "Core": learn it too.
    if rare:
        strong: dict[int, list[str]] = {}
        for key, entries in indicators.items():
            if " " not in key:
                for value_id, weight in entries.items():
                    if weight >= CUT_WEIGHT:
                        strong.setdefault(value_id, []).append(key)
        for i, (value_id, token, n, cov, prec) in enumerate(rare):
            if i % _CANCEL_EVERY == 0:
                _check(cancel)
            if any(_is_spelling_variant(token, key) for key in strong.get(value_id, ())):
                factor = max(ABBREVIATION_WEIGHT, 0.5 + 0.5 * min(1.0, cov / ALIAS_FULL_COVERAGE))
                learn(value_id, token, n, prec * factor)
    return {value_id: [token for *_, token in sorted(items)] for value_id, items in sorted(mined.items())}


def _alias_display(
    table: Table,
    search: list[str],
    codes: np.ndarray,
    learned: dict[int, list[str]],
    stats: _TextStats,
    cancel: CancelFn | None = None,
) -> dict[str, str]:
    """Most common original spelling of each learned alias among its family's rows."""
    columns = [table.column_index(name) for name in search]
    display: dict[str, str] = {}
    for value_id, aliases in learned.items():
        for alias in aliases:
            _check(cancel)
            rows = stats.rows_with(stats.vocab[alias])
            spellings: Counter[str] = Counter()
            for r in rows[codes[rows] == value_id][:50].tolist():
                for j in columns:
                    cell = table.rows[r][j]
                    spellings.update(cell[s:e] for token, s, e in token_spans(cell) if token == alias)
            if spellings:
                display[alias] = spellings.most_common(1)[0][0]
    return display


def _removals(indicators: Staging, value_tokens: list[Pattern]) -> list[dict[str, list[Pattern]]]:
    """Per value: the cell patterns cleaning cuts (keys with weight >= CUT_WEIGHT + full value)."""
    removals: list[dict[str, list[Pattern]]] = [{} for _ in value_tokens]
    for key, entries in indicators.items():
        pattern = tuple(key.split(" "))
        for value_id, weight in entries.items():
            if weight >= CUT_WEIGHT:
                removals[value_id].setdefault(pattern[0], []).append(pattern)
    for value_id, toks in enumerate(value_tokens):
        bucket = removals[value_id].setdefault(toks[0], [])
        if toks not in bucket:
            bucket.append(toks)
    return removals


def _build_column(
    table: Table, name: str, search: list[str], stats: _TextStats, cancel: CancelFn | None = None
) -> tuple[DecidingColumn, _ColumnAux]:
    raw = table.column_values(name)
    codes, values, value_tokens = _vocabulary(raw, cancel)
    _check(cancel)
    is_identifier = identifier_column(raw)
    _check(cancel)
    indicators: Staging = {}
    learned: dict[int, list[str]] = {}
    if is_identifier:
        _identifier_indicators(indicators, value_tokens, stats, cancel)
        aux = _ColumnAux(value_tokens, None, [], {}, {})
    else:
        pairs = _PairCounts(codes, stats)
        token_values, best = _value_token_indicators(indicators, value_tokens, pairs, stats)
        _phrase_indicators(indicators, value_tokens, token_values, best, codes, stats, cancel)
        learned = _learn_aliases(indicators, value_tokens, codes, pairs, stats, cancel)
        unigrams = [key for key in indicators if " " not in key]
        abbreviation_keys: dict[str, list[str]] = {}
        for key in unigrams:
            abbreviation_keys.setdefault(key[0], []).append(key)
        aux = _ColumnAux(
            value_tokens=value_tokens,
            removals=_removals(indicators, value_tokens),
            typo_keys=[key for key in unigrams if len(key) >= 4],
            abbreviation_keys=abbreviation_keys,
            alias_display=_alias_display(table, search, codes, learned, stats, cancel),
        )
    _check(cancel)
    codes.flags.writeable = False
    final = {key: sorted(entries.items()) for key, entries in indicators.items()}
    return DecidingColumn(name, values, codes, is_identifier, final, learned), aux


def _is_subsequence(short: str, long: str) -> bool:
    remaining = iter(long)
    return all(ch in remaining for ch in short)


def _is_abbreviation(token: str, key: str) -> bool:
    """``ishs`` -> ``ishares``: same first letter, a subsequence, not too short."""
    return (
        token[0] == key[0]
        and len(token) < len(key) <= len(token) / ABBREVIATION_RATIO
        and _is_subsequence(token, key)
    )


def _is_spelling_variant(token: str, key: str) -> bool:
    """A typo or an abbreviation of ``key`` (the rules of fuzzy query detection)."""
    return (len(key) >= 4 and fuzz.ratio(token, key) >= TYPO_RATIO) or _is_abbreviation(token, key)


def _fragment(query: str, spans: list[tuple[str, int, int]], ranges: list[tuple[int, int]]) -> str:
    """Original query text covered by token ranges; touching ranges read as one fragment."""
    merged: list[list[int]] = []
    for start, stop in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return ", ".join(query[spans[start][1] : spans[stop - 1][2]] for start, stop in merged)


class FamilyModel:
    """Built once per (table, roles); immutable afterwards and safe to share across threads."""

    columns: list[DecidingColumn]

    def __init__(
        self,
        columns: list[DecidingColumn],
        aux: list[_ColumnAux],
        df_text: dict[str, int],
        key_rows: dict[str, np.ndarray],
        n_rows: int,
    ) -> None:
        """Internal: use :meth:`build`."""
        self.columns = columns
        self._aux = aux
        self._df_text = df_text
        self._key_rows = key_rows  # token of an indicator key -> rows whose search text has it
        self._n_rows = n_rows

    @classmethod
    def build(
        cls, table: Table, deciding: list[str], search: list[str], *, cancel: CancelFn | None = None
    ) -> "FamilyModel":
        """``cancel`` is polled throughout (every few tens of milliseconds on 45k rows); when
        it returns True, ``Cancelled`` is raised."""
        if not deciding:
            return cls([], [], {}, {}, len(table.rows))
        stats = _TextStats(table, search, cancel)
        columns: list[DecidingColumn] = []
        aux: list[_ColumnAux] = []
        for name in deciding:
            _check(cancel)
            column, extra = _build_column(table, name, search, stats, cancel)
            columns.append(column)
            aux.append(extra)
        _check(cancel)
        key_tokens = {token for column in columns for key in column.indicators for token in key.split(" ")}
        key_rows = {t: stats.rows_with(stats.vocab[t]) for t in key_tokens if t in stats.vocab}
        df_text = dict(zip(stats.words, stats.df.tolist()))
        return cls(columns, aux, df_text, key_rows, len(table.rows))

    # cleaning ---------------------------------------------------------------------------
    def clean_cells(self, row: int, cells: list[str]) -> list[str]:
        """Remove high-confidence repeats of the row's own deciding values from its search cells."""
        plans = [
            self._patterns(index, int(column.codes[row]))
            for index, column in enumerate(self.columns)
            if column.codes[row] >= 0
        ]
        if not plans:
            return list(cells)
        return [_clean_cell(cell, plans) for cell in cells]

    def _patterns(self, column_index: int, value_id: int) -> dict[str, list[Pattern]]:
        aux = self._aux[column_index]
        if aux.removals is not None:
            return aux.removals[value_id]
        # Identifier column: the value itself, plus its space-less form when it is a confident key.
        toks = aux.value_tokens[value_id]
        patterns: dict[str, list[Pattern]] = {toks[0]: [toks]}
        compact = "".join(toks)
        entries = self.columns[column_index].indicators.get(compact, ())
        if len(toks) > 1 and any(v == value_id and w >= CUT_WEIGHT for v, w in entries):
            patterns.setdefault(compact, []).append((compact,))
        return patterns

    # detection --------------------------------------------------------------------------
    def detect(self, query: str) -> Detection:
        spans = token_spans(query)
        if not spans or not self.columns:
            return Detection({}, [], [])
        toks = [token for token, _, _ in spans]
        grams = ngrams(toks, 3) + [(toks[i] + toks[i + 1], i, i + 2) for i in range(len(toks) - 1)]
        evidence: dict[str, dict[int, float]] = {}
        cuts: set[tuple[int, int]] = set()
        cut_keys: set[str] = set()
        found: list[FamilyEvidence] = []
        for column, aux in zip(self.columns, self._aux):
            hits: list[tuple[int, float, int, int, str]] = []  # (value id, weight, first, stop, key)
            covered = [False] * len(toks)
            for key, start, stop in grams:
                entries = column.indicators.get(key)
                if entries:
                    hits.extend((value_id, weight, start, stop, key) for value_id, weight in entries)
                    covered[start:stop] = [True] * (stop - start)
            if not column.is_identifier:
                for i, token in enumerate(toks):
                    if covered[i] or len(token) < 3 or self._df_text.get(token, 0) > MAX_FUZZY_DF:
                        continue
                    fuzzy = self._fuzzy(column, aux, token)
                    hits.extend((v, w, i, i + 1, key) for v, (w, key) in fuzzy.items())
            if not hits:
                continue
            missing: dict[int, float] = {}
            for value_id, weight, *_ in hits:
                missing[value_id] = missing.get(value_id, 1.0) * (1.0 - weight)
            strengths = {v: 1.0 - m for v, m in missing.items() if 1.0 - m >= MIN_STRENGTH}
            if not strengths:
                continue
            evidence[column.name] = strengths
            for value_id in sorted(strengths, key=lambda v: (-strengths[v], v)):
                strong = [hit for hit in hits if hit[0] == value_id and hit[1] >= CUT_WEIGHT]
                cuts.update((spans[i][1], spans[j - 1][2]) for _, _, i, j, _ in strong)
                cut_keys.update(key for *_, key in strong)
                shown = [(i, j) for _, _, i, j, _ in strong or [hit for hit in hits if hit[0] == value_id]]
                value, strength = column.values[value_id], strengths[value_id]
                found.append(FamilyEvidence(column.name, value, strength, _fragment(query, spans, shown)))
        found.sort(key=lambda item: -item.strength)  # stable: ties keep column order
        return Detection(evidence, sorted(cuts), found, frozenset(cut_keys))

    def _fuzzy(self, column: DecidingColumn, aux: _ColumnAux, token: str) -> dict[int, tuple[float, str]]:
        """Best typo/abbreviation (weight, matched key) per value for an unrecognized query token."""
        best: dict[int, tuple[float, str]] = {}

        def offer(key: str, factor: float) -> None:
            for value_id, weight in column.indicators[key]:
                if weight * factor > best.get(value_id, (0.0, ""))[0]:
                    best[value_id] = (weight * factor, key)

        if aux.typo_keys:
            for key, ratio, _ in process.extract(
                token, aux.typo_keys, scorer=fuzz.ratio, score_cutoff=TYPO_RATIO, limit=None
            ):
                offer(key, (ratio / 100.0) ** 2)
        for key in aux.abbreviation_keys.get(token[0], ()):
            if _is_abbreviation(token, key):
                offer(key, ABBREVIATION_WEIGHT)
        return best

    # ranking ----------------------------------------------------------------------------
    def family_factor(self, detection: Detection, strictness: float) -> np.ndarray:
        """float32 (n_rows,) multipliers in (0, 1]: 1 = agrees or no evidence, low = conflicts."""
        factor = np.ones(self._n_rows, dtype=np.float32)
        strictness = min(max(float(strictness), 0.0), 1.0)
        for column in self.columns:
            evidence = detection.evidence.get(column.name)
            if not evidence:
                continue
            top = max(evidence.values())
            strength = np.zeros(len(column.values) + 1, dtype=np.float32)
            for value_id, value_strength in evidence.items():
                strength[value_id] = value_strength
            strength[-1] = 0.5 * top  # code -1 (empty cell) indexes this last slot
            factor *= 1.0 - strictness * (top - strength[column.codes])
        np.maximum(factor, MIN_FACTOR, out=factor)
        return factor

    def key_counts(self, keys: frozenset[str]) -> np.ndarray:
        """int32 per row: how many of ``keys`` (``Detection.keys``) its original search text
        contains; a phrase counts when all its tokens are present."""
        counts = np.zeros(self._n_rows, dtype=np.int32)
        for key in keys:
            postings = [self._key_rows.get(token) for token in key.split(" ")]
            if all(rows is not None for rows in postings):
                rows = postings[0]
                for other in postings[1:]:
                    rows = np.intersect1d(rows, other, assume_unique=True)
                counts[rows] += 1
        return counts

    def agreement(self, detection: Detection, row: int) -> tuple[list[str], list[str]]:
        """(agreeing, conflicting) deciding columns of ``row``: full vs reduced strength
        compared with the strongest evidence of each column the query named."""
        agrees: list[str] = []
        conflicts: list[str] = []
        for column in self.columns:
            evidence = detection.evidence.get(column.name)
            if not evidence:
                continue
            value_id = int(column.codes[row])
            if value_id >= 0 and evidence.get(value_id, 0.0) >= max(evidence.values()) - 1e-6:
                agrees.append(column.name)
            else:
                conflicts.append(column.name)
        return agrees, conflicts

    def learned_alias_table(self) -> dict[str, dict[str, list[str]]]:
        """{column: {value: [aliases as spelled in the data]}} for columns that learned any."""
        table: dict[str, dict[str, list[str]]] = {}
        for column, aux in zip(self.columns, self._aux):
            if column.learned_aliases:
                table[column.name] = {
                    column.values[value_id]: [aux.alias_display.get(alias, alias) for alias in aliases]
                    for value_id, aliases in column.learned_aliases.items()
                }
        return table


def _clean_cell(cell: str, plans: list[dict[str, list[Pattern]]]) -> str:
    spans = token_spans(cell)
    if not spans:
        return cell
    toks = [token for token, _, _ in spans]
    cuts: list[tuple[int, int]] = []
    for i, token in enumerate(toks):
        for plan in plans:
            for pattern in plan.get(token, ()):
                stop = i + len(pattern)
                if len(pattern) == 1 or (stop <= len(toks) and tuple(toks[i:stop]) == pattern):
                    cuts.append((spans[i][1], spans[stop - 1][2]))
    if not cuts:
        return cell
    cleaned = remove_spans(cell, cuts)
    return cleaned if tokens(cleaned) else cell
