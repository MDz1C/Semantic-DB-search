"""Responsiveness under load (ARCHITECTURE.md 12.3): the UI thread does not wait for the store
while workers write, cancelling an index build takes effect quickly in every stage, and a
cancelled build keeps the embeddings it finished."""

from __future__ import annotations

import bisect
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from semdb.core.engine import STAGE_ALIASES, STAGE_ANALYZE, STAGE_EMBED, STAGE_FINAL, SearchIndex
from semdb.core.families import FamilyModel
from semdb.core.types import CancelFn, Cancelled, FamilyEvidence, HistoryEntry, ResultRow, Role, Roles, Table
from semdb.storage import APP_DB_NAME, SqliteEmbeddingCache, Store

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from perf_engine import ROLES, synthetic_table  # noqa: E402

CANCEL_LIMIT = 0.25  # s; the target is ~150 ms, the margin absorbs slow or busy machines
ColdBuild = tuple[dict[str, float], SqliteEmbeddingCache]  # worst cancel wait per stage, filled cache
COLUMNS = ["ISIN", "Ticker", "Fund name", "Brokerage", "Category", "Asset class", "Region", "Currency",
           "Distribution", "Domicile", "TER (%)", "AUM (M EUR)", "Launch date"]  # fmt: skip


class FastEmbedder:
    """Random unit vectors, polling cancel before every text like the real model."""

    model_id = "fast-fake-32"
    dim = 32

    def __init__(self) -> None:
        self.seen: list[str] = []  # every text encoding reached, in order
        self.finished: list[str] = []  # texts of the encode calls that returned
        self._rng = np.random.default_rng(0)

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        for text in texts:
            if cancel is not None and cancel():
                raise Cancelled()
            self.seen.append(text)
        self.finished.extend(texts)
        if progress is not None:
            progress(len(texts), len(texts))
        vectors = self._rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class BuildClock:
    """Progress and cancel callbacks that timestamp stage starts and cancel polls; optionally
    cancels from ``delay`` seconds after ``stage`` started."""

    def __init__(self, stage: str | None = None, delay: float = 0.0) -> None:
        self.stages: list[tuple[float, str]] = []
        self.polls: list[float] = []
        self.requested: float | None = None  # when cancelling was requested
        self._target = stage
        self._delay = delay

    def progress(self, stage: str, fraction: float) -> None:
        if not self.stages or self.stages[-1][1] != stage:
            self.stages.append((time.perf_counter(), stage))

    def cancel(self) -> bool:
        now = time.perf_counter()
        self.polls.append(now)
        if self.requested is None:
            start = next((t for t, stage in self.stages if stage == self._target), None)
            if start is None or now < start + self._delay:
                return False
            self.requested = start + self._delay
        return True

    def worst_wait(self, end: float) -> dict[str, float]:
        """Per stage, the longest a cancel requested during it waits for the next poll (or for
        the build to return)."""
        worst: dict[str, float] = {}
        for i, (start, stage) in enumerate(self.stages):
            stop = self.stages[i + 1][0] if i + 1 < len(self.stages) else end
            for moment in [start, *(p for p in self.polls if start <= p < stop)]:
                after = bisect.bisect_right(self.polls, moment)
                answer = self.polls[after] if after < len(self.polls) else end
                worst[stage] = max(worst.get(stage, 0.0), min(answer, end) - moment)
        return worst


def _numbered(rows: int, seed: int) -> Table:
    """perf_engine's fund universe with a number in every name: like a real workbook, nearly
    every row has its own text to embed and look up."""
    table = synthetic_table(rows, seed)
    name = table.column_index("Fund name")
    for i, row in enumerate(table.rows):
        row[name] = f"{row[name]} {i}"
    return table


def _role_names(roles: Roles, role: Role) -> list[str]:
    return [name for name, value in roles.items() if value is role]


def _build(
    table: Table, cache: SqliteEmbeddingCache, clock: BuildClock, embedder: FastEmbedder | None = None
) -> float:
    """Build with the clock's callbacks; returns the time the build returned."""
    SearchIndex.build(table, ROLES, embedder or FastEmbedder(), cache, progress=clock.progress, cancel=clock.cancel)
    return time.perf_counter()


@pytest.fixture(scope="module")
def big_table() -> Table:
    return _numbered(45_000, seed=7)


@pytest.fixture(scope="module")
def cold_build(big_table: Table, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ColdBuild]:
    """One cold-cache build of the big table: its worst cancel waits, and the cache it filled."""
    cache = SqliteEmbeddingCache(tmp_path_factory.mktemp("cache") / "embeddings.sqlite")
    clock = BuildClock()
    end = _build(big_table, cache, clock)
    yield clock.worst_wait(end), cache
    cache.close()


# --- cancelling an index build -------------------------------------------------------------------


def _assert_quick(worst: dict[str, float], label: str) -> None:
    # Indexing words is left out: LexicalIndex (lexical.py) ends with one numpy pass of
    # 0.2-0.5 s on 45k rows that polls nothing.
    for stage in (STAGE_ANALYZE, STAGE_ALIASES, STAGE_EMBED, STAGE_FINAL):
        assert worst[stage] < CANCEL_LIMIT, f"{label}: a cancel during {stage!r} waits {worst[stage] * 1000:.0f} ms"


def test_cancel_is_polled_often_in_every_stage_of_a_cold_build(cold_build: ColdBuild) -> None:
    _assert_quick(cold_build[0], "cold cache")


def test_cancel_is_polled_often_in_every_stage_of_a_warm_build(big_table: Table, cold_build: ColdBuild) -> None:
    clock = BuildClock()
    embedder = FastEmbedder()
    end = _build(big_table, cold_build[1], clock, embedder)
    assert embedder.seen == []  # everything came from the cache: a lookup of ~45k texts
    _assert_quick(clock.worst_wait(end), "warm cache")


@pytest.mark.parametrize(
    ("stage", "delay"),
    [(STAGE_ALIASES, 0.1), (STAGE_EMBED, 0.05), (STAGE_FINAL, 0.0)],
    ids=["aliases", "embedding-lookup", "finalizing"],
)
def test_cancel_during_a_stage_stops_the_build_quickly(
    big_table: Table, cold_build: ColdBuild, stage: str, delay: float
) -> None:
    clock = BuildClock(stage, delay)
    with pytest.raises(Cancelled):
        _build(big_table, cold_build[1], clock)
    latency = time.perf_counter() - clock.requested
    assert latency < CANCEL_LIMIT, f"Cancelled raised {latency * 1000:.0f} ms after the request"


def test_family_model_build_polls_cancel(big_table: Table) -> None:
    deciding, search = _role_names(ROLES, Role.DECIDING), _role_names(ROLES, Role.SEARCH)
    polls: list[float] = []
    start = time.perf_counter()
    model = FamilyModel.build(big_table, deciding, search, cancel=lambda: polls.append(time.perf_counter()) or False)
    gaps = np.diff([start, *polls, time.perf_counter()])
    assert gaps.max() < CANCEL_LIMIT, f"FamilyModel.build went {gaps.max() * 1000:.0f} ms without polling cancel"
    assert [column.name for column in model.columns] == deciding
    with pytest.raises(Cancelled):
        FamilyModel.build(big_table, deciding, search, cancel=lambda: True)


def test_a_cancelled_build_keeps_the_embeddings_it_finished(tmp_path: Path) -> None:
    table = _numbered(3_000, seed=3)
    cache = SqliteEmbeddingCache(tmp_path / "embeddings.sqlite")
    first = FastEmbedder()
    with pytest.raises(Cancelled):  # cancelled a third of the way through "Understanding rows"
        SearchIndex.build(table, ROLES, first, cache, cancel=lambda: len(first.seen) >= 1_000)
    assert len(first.seen) - len(first.finished) < 256, "a cancel may lose at most one chunk of work"
    assert len(cache.get_many(first.model_id, first.finished)) == len(first.finished)

    again = FastEmbedder()  # "Build index" after the cancel
    index = SearchIndex.build(table, ROLES, again, cache)
    assert set(again.seen).isdisjoint(first.finished)  # nothing finished is embedded twice
    assert len(again.seen) == len(set(again.seen))
    assert index.has_semantic and index.search("MSCI World UCITS ETF", k=5).matches
    cache.close()


# --- the store while workers write ----------------------------------------------------------------


def _entry(query: str, results: int = 50) -> HistoryEntry:
    """A result list as a 50-result search of a 13-column sheet stores it (~21 KB of JSON)."""
    rows = [
        ResultRow(rank, round(1 - rank / 60, 4), rank + 1,
                  [f"IE00B{rank:07d}", f"T{rank}", f"{query} UCITS ETF USD (Acc) {rank}", "BlackRock", "Equity",
                   "Equity", "Global", "USD", "Accumulating", "Ireland", "0.20", "51234.5", "2009-09-25"],
                  {"semantic": 0.5, "lexical": 0.4, "text_score": 0.45, "family_factor": 1.0, "exact_id": None,
                   "agreements": ["Brokerage"], "conflicts": []})
        for rank in range(1, results + 1)
    ]  # fmt: skip
    return HistoryEntry(None, query, "2026-11-01T17:32:05.123456+00:00", "funds.xlsx · Funds", COLUMNS,
                        ["ISIN", "Ticker", "Brokerage"], "Fund name",
                        [FamilyEvidence("Brokerage", "BlackRock", 0.99, "iShares")], rows)  # fmt: skip


def _stored_setting(data_dir: Path, key: str) -> str | None:
    """The JSON of a setting as committed to the database file."""
    with closing(sqlite3.connect(data_dir / APP_DB_NAME)) as conn:
        row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


class _Holder:
    """Holds the store's connection on another thread, as a long worker transaction would."""

    def __init__(self, store: Store) -> None:
        self._store = store
        self._held = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self) -> None:
        with self._store._lock:
            self._held.set()
            self._release.wait(5)

    def __enter__(self) -> _Holder:
        self._thread.start()
        self._held.wait(5)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._release.set()
        self._thread.join()


def test_set_setting_does_not_wait_for_a_busy_connection(tmp_path: Path) -> None:
    data = tmp_path / "data"
    with Store(data) as store:
        with _Holder(store):
            start = time.perf_counter()
            store.set_setting("theme", "dark")
            elapsed = time.perf_counter() - start
            assert store.get_setting("theme") == "dark"  # visible at once
        store.has_dataset()  # the next use of the connection writes what was queued
        assert _stored_setting(data, "theme") == '"dark"'
        store.set_setting("theme", "light")  # connection free: written right away
        assert _stored_setting(data, "theme") == '"light"'
    assert elapsed < 0.05, f"set_setting waited {elapsed * 1000:.0f} ms for another thread's transaction"


def test_queued_settings_are_written_when_the_store_closes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    store = Store(data)
    with _Holder(store):
        store.set_setting("sort_mode", "alpha")
        store.set_setting("sort_mode", "recent")  # the later value wins
    store.close()
    with Store(data) as reopened:
        assert reopened.get_setting("sort_mode") == "recent"


def test_bulk_reads_do_not_wait_for_the_connection(tmp_path: Path, funds_table: Table, funds_roles: Roles) -> None:
    with Store(tmp_path / "data") as store:
        store.save_dataset(funds_table, funds_roles)
        store.add_entries([_entry("iShares Core S&P 500", results=3)])
        with _Holder(store):
            start = time.perf_counter()
            loaded, entries = store.load_dataset(), store.load_entries()
            elapsed = time.perf_counter() - start
    assert loaded == (funds_table, funds_roles)
    assert [entry.query for entry in entries] == ["iShares Core S&P 500"]
    assert elapsed < 0.1, f"loading waited {elapsed * 1000:.0f} ms for another thread's transaction"


def test_setting_writes_stay_fast_while_a_large_batch_of_lists_is_stored(tmp_path: Path) -> None:
    """A worker stores 400 result lists (~8 MB of JSON) while the UI thread saves settings.
    The bound is generous: on Windows each GIL hand-off between the threads can take a timer
    tick (15.6 ms), and one sqlite3 call needs a few."""
    batch = [_entry(f"query {n}") for n in range(400)]
    with Store(tmp_path / "data") as store:
        stored: list[HistoryEntry] = []
        errors: list[BaseException] = []
        done = threading.Event()

        def work() -> None:
            try:
                stored.extend(store.add_entries(batch))
            except BaseException as exc:  # reported by the main thread
                errors.append(exc)
            finally:
                done.set()

        worker = threading.Thread(target=work)
        waits: list[float] = []
        worker.start()
        while not done.is_set():
            start = time.perf_counter()
            store.set_setting("results_per_query", len(waits))
            waits.append(time.perf_counter() - start)
            time.sleep(0.002)
        worker.join()
        assert errors == []
        assert [entry.query for entry in stored] == [entry.query for entry in batch]
        assert len(store.load_entries()) == 400 and store.get_setting("results_per_query") == len(waits) - 1
    assert max(waits) < 0.1, f"a setting write waited {max(waits) * 1000:.0f} ms (over {len(waits)} writes)"


def test_the_write_ahead_log_is_checkpointed_outside_commits(tmp_path: Path) -> None:
    """Commits no longer checkpoint (that held the lock); bulk writes checkpoint afterwards, so
    the log still restarts instead of growing with every batch."""
    data = tmp_path / "data"
    with Store(data) as store:
        for batch in range(6):  # ~2 MB each
            store.add_entries([_entry(f"batch {batch} query {n}") for n in range(100)])
        assert (data / (APP_DB_NAME + "-wal")).stat().st_size < 8_000_000


def test_the_embedding_cache_still_prunes_to_its_limit(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / "embeddings.sqlite", max_entries=50)
    rng = np.random.default_rng(1)
    try:
        for n in range(20):
            texts = [f"text {n} {i}" for i in range(10)]
            cache.put_many("model", texts, rng.standard_normal((10, 8)).astype(np.float32))
            cache.put_many("model", texts[:5], rng.standard_normal((5, 8)).astype(np.float32))  # updates
            assert len(cache) <= 50
        assert len(cache) == 50
        assert sorted(cache.get_many("model", [f"text 19 {i}" for i in range(10)])) == list(range(10))
    finally:
        cache.close()
