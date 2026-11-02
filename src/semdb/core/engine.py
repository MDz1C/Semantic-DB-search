"""The search engine: builds an immutable index over a Table and ranks rows for queries.

Ranking (docs/ARCHITECTURE.md section 9): identity fragments the query names (via the family
model) are cut from it, the rest is compared with each search column of the row (cleaned of
the row's own identity) both lexically (how completely the column covers the query's
character/word features) and semantically (sentence embeddings); the best column counts, plus
a little for covering the query across the whole row.  The resulting text score is multiplied
by a family factor that pushes rows of a different identity down.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from .families import Detection, FamilyModel, identifier_column
from .lexical import LexicalIndex
from .text import normalize, remove_spans, tokens
from .types import (
    MAX_STORED_RESULTS,
    CancelFn,
    Cancelled,
    EmbeddingCache,
    Embedder,
    HistoryEntry,
    Match,
    ProgressFn,
    QueryOutcome,
    ResultRow,
    Role,
    Roles,
    Table,
)

log = logging.getLogger(__name__)

STAGE_ANALYZE = "Analyzing columns…"
STAGE_ALIASES = "Learning identity aliases…"
STAGE_INDEX = "Indexing words…"
STAGE_EMBED = "Understanding rows…"
STAGE_FINAL = "Finalizing…"
STAGE_SEARCH = "Searching…"

CELL_SEPARATOR = " · "
IDENTITY_ONLY_TEXT_SCORE = 0.5  # the query only named an identity: rank by identity
MATCH_BETA = 3.0  # lexical F-beta: covering the query matters 9x more than the column's extra words
FIELD_SHARE = 0.65  # lexical score = best column's F-beta, the rest = recall over the whole row
# Texts per encode call.  Each chunk is cached as soon as it is done, so a cancelled build keeps
# all but the chunk in progress (< 0.2 s of model time); the real model is as fast as with 1,024.
_EMBED_CHUNK = 256
_LOOKUP_CHUNK = 1024  # texts per embedding-cache lookup (~20-50 ms each), cancel polled between
_QUERY_EMBED_SHARE = 0.1  # share of the search progress bar spent embedding queries
_CANCEL_EVERY = 1000  # rows between two polls of the cancel callback (< ~50 ms on wide 45k tables)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    # Share of embedding similarity in the text score (rest is lexical).  Tuned on a held-out
    # synthetic universe: 0.15-0.25 is best (0.5 lost 3 points of Acc@1 to typos and near-duplicate
    # names); 0.2 leans slightly semantic for real-world paraphrases the benchmark does not model.
    semantic_weight: float = 0.2
    strictness: float = 0.85  # how hard a deciding-column conflict pushes a row down (0..1)
    semantic_floor: float = 0.30  # cosine mapped to 0 when calibrating embedding similarity
    use_family: bool = True  # False disables deciding-column detection, cleaning and penalties (ablation)


_NO_DETECTION = Detection({}, [], [])


@dataclass(frozen=True, slots=True)
class _Prepared:
    query: str
    detection: Detection
    cleaned: str
    has_tokens: bool  # the raw query has something to search for
    has_text: bool  # something descriptive is left after cutting identity fragments
    named_keys: frozenset[str]  # identity keys the query named ("lyxor"): they order exact ties


def _no_progress(stage: str, fraction: float) -> None:
    pass


def _check(cancel: CancelFn | None) -> None:
    if cancel is not None and cancel():
        raise Cancelled()


def _role(value: object) -> Role:
    try:
        return Role(value)
    except ValueError:
        return Role.IGNORE


class SearchIndex:
    """Immutable after ``build``; ``search``/``search_many`` are safe to call from any thread."""

    table: Table
    roles: Roles
    deciding_columns: list[str]
    search_columns: list[str]
    primary_column: str
    has_semantic: bool

    __slots__ = (
        "table",
        "roles",
        "deciding_columns",
        "search_columns",
        "primary_column",
        "has_semantic",
        "_config",
        "_family",
        "_fields",
        "_lexical",
        "_row_texts",
        "_embedder",
        "_vectors",
        "_text_ids",
        "_ids",
        "_ambiguous_ids",
    )

    @classmethod
    def build(
        cls,
        table: Table,
        roles: Roles,
        embedder: Embedder | None = None,
        cache: EmbeddingCache | None = None,
        config: EngineConfig = EngineConfig(),
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> "SearchIndex":
        report = progress or _no_progress
        started = time.perf_counter()

        report(STAGE_ANALYZE, 0.0)
        role_of = {name: _role(roles.get(name, Role.IGNORE)) for name in table.columns}
        deciding = [name for name in table.columns if role_of[name] is Role.DECIDING]
        search = [name for name in table.columns if role_of[name] is Role.SEARCH]
        # Without a search column every non-ignored column is text, compared uncleaned.
        text_columns = search or [name for name in table.columns if role_of[name] is not Role.IGNORE]
        primary = search[0] if search else (table.columns[0] if table.columns else "")
        report(STAGE_ANALYZE, 1.0)
        _check(cancel)

        report(STAGE_ALIASES, 0.0)
        family = FamilyModel.build(table, deciding, search, cancel=cancel) if config.use_family else None
        _check(cancel)
        fields = _field_texts(table, text_columns, family if search else None, report, cancel)
        if fields:
            row_texts = [CELL_SEPARATOR.join(cell for cell in cells if cell) for cells in zip(*fields)]
        else:
            row_texts = [""] * len(table.rows)
        _check(cancel)

        report(STAGE_INDEX, 0.0)
        field_indexes = []
        for done, texts in enumerate(fields):
            field_indexes.append(_field_index(texts, cancel))
            report(STAGE_INDEX, (done + 1) / (len(fields) + 1))
        lexical = LexicalIndex(row_texts, cancel)
        report(STAGE_INDEX, 1.0)

        vectors = text_ids = None
        if embedder is not None and fields:
            report(STAGE_EMBED, 0.0)
            flat = [text for texts in fields for text in texts]
            vectors, flat_ids = _embed_rows(flat, embedder, cache, report, cancel)
            text_ids = flat_ids.reshape(len(fields), len(table.rows))

        report(STAGE_FINAL, 0.0)
        known = {column.name: column.is_identifier for column in family.columns} if family else {}

        def is_identifier(name: str) -> bool:
            _check(cancel)
            return known[name] if name in known else identifier_column(table.column_values(name))

        id_columns = [
            name
            for name in table.columns
            if role_of[name] in (Role.DECIDING, Role.SEARCH) and is_identifier(name)
        ]
        ids, ambiguous = _id_lookup(table, id_columns, [c for c in text_columns if c not in id_columns], cancel)
        _check(cancel)
        report(STAGE_FINAL, 1.0)

        index = cls.__new__(cls)
        index.table = table
        index.roles = dict(roles)
        index.deciding_columns = deciding
        index.search_columns = search
        index.primary_column = primary
        index.has_semantic = vectors is not None
        index._config = config
        index._family = family
        index._fields = tuple(field_indexes)
        index._lexical = lexical
        index._row_texts = tuple(row_texts)
        index._embedder = embedder
        index._vectors = vectors
        index._text_ids = text_ids
        index._ids = ids
        index._ambiguous_ids = ambiguous
        log.info(
            "Index built in %.2fs: %d rows, deciding=%s, search=%s, semantic=%s, aliases=%s",
            time.perf_counter() - started,
            len(table.rows),
            deciding,
            search,
            index.has_semantic,
            index.learned_aliases(),
        )
        return index

    # querying ---------------------------------------------------------------------------
    def search(self, query: str, k: int = 50) -> QueryOutcome:
        return self.search_many([query], k)[0]

    def search_many(
        self,
        queries: list[str],
        k: int = 50,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> list[QueryOutcome]:
        report = progress or _no_progress
        report(STAGE_SEARCH, 0.0)
        prepared = [self._prepare(query) for query in queries]
        query_vectors = self._embed_queries(prepared, report, cancel)
        start = _QUERY_EMBED_SHARE if query_vectors else 0.0
        outcomes: list[QueryOutcome] = []
        for item in prepared:
            _check(cancel)
            vector = query_vectors.get(_embedding_text(item.cleaned)) if item.has_text else None
            outcomes.append(self._rank(item, vector, k))
            report(STAGE_SEARCH, start + (1.0 - start) * len(outcomes) / len(prepared))
        return outcomes

    def learned_aliases(self) -> dict[str, dict[str, list[str]]]:
        """{deciding column: {value: [aliases learned from the data]}} for display."""
        return self._family.learned_alias_table() if self._family is not None else {}

    def row_text(self, row: int) -> str:
        """The descriptive text of ``row`` as compared with queries (identity fragments cut)."""
        return self._row_texts[row]

    def to_history_entry(self, outcome: QueryOutcome, created_at: str) -> HistoryEntry:
        """Snapshot an outcome (all columns of each matched row) for persistence."""
        results = [
            ResultRow(
                rank=rank,
                score=match.score,
                source_row=self.table.source_rows[match.row],
                cells=list(self.table.rows[match.row]),
                detail={
                    "semantic": match.semantic,
                    "lexical": match.lexical,
                    "text_score": match.text_score,
                    "family_factor": match.family_factor,
                    "exact_id": match.exact_id,
                    "agreements": list(match.agreements),
                    "conflicts": list(match.conflicts),
                },
            )
            for rank, match in enumerate(outcome.matches[:MAX_STORED_RESULTS], start=1)
        ]
        return HistoryEntry(
            id=None,
            query=outcome.query,
            created_at=created_at,
            dataset_label=self.table.label,
            columns=list(self.table.columns),
            deciding=list(self.deciding_columns),
            primary_column=self.primary_column,
            evidence=list(outcome.evidence),
            results=results,
        )

    # internals --------------------------------------------------------------------------
    def _prepare(self, query: str) -> _Prepared:
        detection = self._family.detect(query) if self._family is not None else _NO_DETECTION
        cleaned = remove_spans(query, detection.spans)
        has_tokens, has_text = bool(tokens(query)), bool(tokens(cleaned))
        return _Prepared(query, detection, cleaned, has_tokens, has_text, detection.keys)

    def _embed_queries(
        self, prepared: list[_Prepared], report: ProgressFn, cancel: CancelFn | None
    ) -> dict[str, np.ndarray]:
        """One encode call for every distinct cleaned query that needs a vector (keyed by
        ``_embedding_text``)."""
        if self._embedder is None or not len(self.table.rows):
            return {}
        texts = list(dict.fromkeys(_embedding_text(item.cleaned) for item in prepared if item.has_text))
        if not texts:
            return {}

        def encoded(done: int, total: int) -> None:
            report(STAGE_SEARCH, _QUERY_EMBED_SHARE * done / max(total, 1))

        vectors = np.asarray(self._embedder.encode(texts, progress=encoded, cancel=cancel), dtype=np.float32)
        return dict(zip(texts, vectors))

    def _rank(self, item: _Prepared, query_vector: np.ndarray | None, k: int) -> QueryOutcome:
        n = len(self.table.rows)
        evidence = list(item.detection.found)
        if not item.has_tokens or n == 0 or k <= 0:
            return QueryOutcome(item.query, item.cleaned, evidence, [])
        config = self._config
        raw_lexical = None
        semantic = np.zeros(n, dtype=np.float32)
        if item.has_text:
            lexical = self._lexical_scores(item.cleaned)
            if query_vector is not None and self._vectors is not None:
                # One matrix-vector product per query (never batch-dependent); best column wins.
                cosine = (self._vectors @ query_vector)[self._text_ids].max(axis=0)
                floor = min(config.semantic_floor, 0.99)
                semantic = np.clip((cosine - floor) / (1.0 - floor), 0.0, 1.0).astype(np.float32)
                weight = config.semantic_weight
                text = weight * semantic + (1.0 - weight) * lexical
            else:
                text = lexical.copy()
        else:
            # Only an identity was named: rank by identity, ties by the raw query's words.
            lexical = raw_lexical = self._lexical.scores(item.query)
            text = np.full(n, IDENTITY_ONLY_TEXT_SCORE, dtype=np.float32)
        exact = self._exact_rows(item.query)
        if exact:
            text[list(exact)] = 1.0
        np.clip(text, 0.0, 1.0, out=text)
        if self._family is not None and item.detection.evidence:
            factor = self._family.family_factor(item.detection, config.strictness)
        else:
            factor = np.ones(n, dtype=np.float32)
        score = text * factor

        # Equal scores: first the rows whose original text has the identity the query named
        # ("Lyxor X" before "Amundi X", both cleaned to "X"), then the raw query's words.
        tiebreaks = []
        if item.named_keys and self._family is not None:
            tiebreaks.append(self._family.key_counts(item.named_keys))
        if raw_lexical is not None:
            tiebreaks.append(raw_lexical)
        matches = []
        for row in _top_rows(score, k, tiebreaks).tolist():
            agrees, conflicts = (
                self._family.agreement(item.detection, row) if self._family is not None else ([], [])
            )
            matches.append(
                Match(
                    row=row,
                    score=float(score[row]),
                    text_score=float(text[row]),
                    semantic=float(semantic[row]),
                    lexical=float(lexical[row]),
                    family_factor=float(factor[row]),
                    exact_id=exact.get(row),
                    agreements=agrees,
                    conflicts=conflicts,
                )
            )
        return QueryOutcome(item.query, item.cleaned, evidence, matches)

    def _lexical_scores(self, query: str) -> np.ndarray:
        """Best column's F-beta blended with the share of the query found anywhere in the row."""
        n = len(self.table.rows)
        if not self._fields:
            return np.zeros(n, dtype=np.float32)
        best = recall = None
        for index, ids in self._fields:
            f_beta, field_recall = index.match(query, MATCH_BETA)
            best = f_beta[ids] if best is None else np.maximum(best, f_beta[ids])
            if len(self._fields) == 1:
                recall = field_recall[ids]
        if recall is None:
            recall = self._lexical.match(query, MATCH_BETA)[1]
        return (FIELD_SHARE * best + (1.0 - FIELD_SHARE) * recall).astype(np.float32)

    def _exact_rows(self, query: str) -> dict[int, str]:
        """Rows whose identifier equals the whole query or one of its tokens -> column name."""
        if not self._ids:
            return {}
        toks = tokens(query)
        whole = " ".join(toks)
        keys = [(whole, True), (whole.replace(" ", ""), True), *((token, False) for token in toks)]
        hits: dict[int, str] = {}
        for key, is_whole in keys:
            if not is_whole and key in self._ambiguous_ids:
                continue
            for column, row in self._ids.get(key, ()):
                hits.setdefault(row, column)
        return hits


def _embedding_text(text: str) -> str:
    """What the embedding model sees: case-folded, because the model is case-sensitive and
    spreadsheets mix "ISHARES CORE S&P 500" with "iShares Core S&P 500" freely."""
    return text.lower()


def _field_texts(
    table: Table,
    columns: list[str],
    family: FamilyModel | None,
    report: ProgressFn,
    cancel: CancelFn | None,
) -> list[list[str]]:
    """Per text column, every row's cell cleaned of the row's own identity ('' when blank)."""
    indices = [table.column_index(name) for name in columns]
    total = max(len(table.rows), 1)
    fields: list[list[str]] = [[] for _ in indices]
    for r, row in enumerate(table.rows):
        if r % _CANCEL_EVERY == 0:
            _check(cancel)
            report(STAGE_ALIASES, r / total)
        cells = [row[j] for j in indices]
        if family is not None:
            cells = family.clean_cells(r, cells)
        for field, cell in zip(fields, cells):
            field.append(cell if cell.strip() else "")
    report(STAGE_ALIASES, 1.0)
    return fields


def _field_index(texts: list[str], cancel: CancelFn | None) -> tuple[LexicalIndex, np.ndarray]:
    """Lexical index over a column's distinct texts, plus each row's position among them."""
    position: dict[str, int] = {}
    ids = np.empty(len(texts), dtype=np.int32)
    for r, text in enumerate(texts):
        ids[r] = position.setdefault(text, len(position))
    ids.flags.writeable = False
    return LexicalIndex(list(position), cancel), ids


def _embed_rows(
    row_texts: list[str],
    embedder: Embedder,
    cache: EmbeddingCache | None,
    report: ProgressFn,
    cancel: CancelFn | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(unique vectors, per-row index into them); empty texts share a zero vector.

    The cache is read and the missing texts embedded in chunks, polling ``cancel`` between
    them, and each embedded chunk is cached at once: a cancelled build leaves all its finished
    chunks to the next one."""
    position: dict[str, int] = {}
    unique: list[str] = []
    text_ids = np.empty(len(row_texts), dtype=np.int32)
    for r, text in enumerate(row_texts):
        if r % _CANCEL_EVERY == 0:
            _check(cancel)
        if not text:
            text_ids[r] = -1
            continue
        text = _embedding_text(text)
        index = position.get(text)
        if index is None:
            index = position[text] = len(unique)
            unique.append(text)
        text_ids[r] = index
    vectors = np.zeros((len(unique) + 1, embedder.dim), dtype=np.float32)  # last row: empty text
    text_ids[text_ids < 0] = len(unique)

    total = max(len(unique), 1)
    have = np.zeros(len(unique), dtype=bool)
    done = 0
    if cache is not None:
        for first in range(0, len(unique), _LOOKUP_CHUNK):
            _check(cancel)
            for i, vector in cache.get_many(embedder.model_id, unique[first : first + _LOOKUP_CHUNK]).items():
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if vector.shape == (embedder.dim,) and np.all(np.isfinite(vector)):
                    vectors[first + i] = vector
                    have[first + i] = True
                    done += 1
            report(STAGE_EMBED, done / total)
    missing = sorted(np.flatnonzero(~have).tolist(), key=lambda i: len(unique[i]))
    report(STAGE_EMBED, done / total)
    for first in range(0, len(missing), _EMBED_CHUNK):
        _check(cancel)
        chunk = missing[first : first + _EMBED_CHUNK]
        texts = [unique[i] for i in chunk]

        def encoded(finished: int, _: int, base: int = done) -> None:
            report(STAGE_EMBED, (base + finished) / total)

        encoded_vectors = np.asarray(
            embedder.encode(texts, progress=encoded, cancel=cancel), dtype=np.float32
        )
        vectors[chunk] = encoded_vectors
        if cache is not None:
            cache.put_many(embedder.model_id, texts, encoded_vectors)
        done += len(chunk)
        report(STAGE_EMBED, done / total)
    _check(cancel)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    np.divide(vectors, norms, out=vectors, where=norms > 0)
    vectors.flags.writeable = False
    text_ids.flags.writeable = False
    return vectors, text_ids


def _id_lookup(
    table: Table, id_columns: list[str], other_text_columns: list[str], cancel: CancelFn | None = None
) -> tuple[dict[str, list[tuple[str, int]]], frozenset[str]]:
    """Normalized identifier (and its space-less form) -> [(column, row)], plus the keys that
    only count when they are the whole query: short letter-only codes that are also ordinary
    words of the descriptive text (a ticker "ALL" vs "FTSE All-World")."""
    lookup: dict[str, list[tuple[str, int]]] = {}
    for name in id_columns:
        j = table.column_index(name)
        for r, row in enumerate(table.rows):
            if r % _CANCEL_EVERY == 0:
                _check(cancel)
            key = normalize(row[j])
            if key:
                for variant in dict.fromkeys((key, key.replace(" ", ""))):
                    lookup.setdefault(variant, []).append((name, r))
    short = {key for key in lookup if len(key) < 5 and not any(ch.isdigit() for ch in key)}
    if not short:
        return lookup, frozenset()
    words: set[str] = set()
    seen: set[str] = set()
    indices = [table.column_index(name) for name in other_text_columns]
    for r, row in enumerate(table.rows):
        if r % _CANCEL_EVERY == 0:
            _check(cancel)
        for j in indices:
            cell = row[j]
            if cell not in seen:
                seen.add(cell)
                words.update(tokens(cell))
    return lookup, frozenset(short & words)


def _top_rows(score: np.ndarray, k: int, tiebreaks: list[np.ndarray]) -> np.ndarray:
    """Indices of the k best positive scores: score desc, then each tiebreak array desc (most
    important first), then row asc."""
    candidates = np.flatnonzero(score > 0)
    if candidates.size > k:
        values = score[candidates]
        kth = np.partition(values, candidates.size - k)[candidates.size - k]
        candidates = candidates[values >= kth]  # keeps every tie at the boundary
    keys = [candidates, *(-key[candidates] for key in reversed(tiebreaks)), -score[candidates]]
    return candidates[np.lexsort(keys)[:k]]
