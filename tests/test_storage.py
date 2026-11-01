"""Local persistence (ARCHITECTURE.md section 10, R11): settings, the imported database, query
history and the embedding cache; recovery from damaged files; thread safety; speed."""

from __future__ import annotations

import random
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from semdb.core.types import FamilyEvidence, HistoryEntry, ResultRow, Role, Roles, Table
from semdb.storage import APP_DB_NAME, CACHE_DB_NAME, SqliteEmbeddingCache, Store


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(data_dir: Path) -> Iterator[Store]:
    with Store(data_dir) as opened:
        yield opened


def _entry(query: str, *, results: int = 2, evidence: list[FamilyEvidence] | None = None) -> HistoryEntry:
    rows = [
        ResultRow(rank, round(1 - rank / 10, 2), rank + 1, [f"IE{rank:010d}", f"{query} fund {rank}", "BlackRock"],
                  {"semantic": 0.5, "lexical": 0.4, "text_score": 0.45, "family_factor": 1.0, "exact_id": None,
                   "agreements": ["Brokerage"], "conflicts": []})
        for rank in range(1, results + 1)
    ]  # fmt: skip
    return HistoryEntry(None, query, "2026-11-01T17:32:05.123456+00:00", "funds.xlsx · Funds",
                        ["ISIN", "Fund name", "Brokerage"], ["ISIN", "Brokerage"], "Fund name",
                        evidence or [], rows)  # fmt: skip


def _queries(store: Store) -> list[str]:
    return [entry.query for entry in store.load_entries()]


def _unit_vectors(count: int, dim: int = 384, seed: int = 0) -> np.ndarray:
    vectors = np.random.default_rng(seed).standard_normal((count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _execute(path: Path, *statements: str) -> None:
    with closing(sqlite3.connect(path)) as conn:
        for statement in statements:
            conn.execute(statement)
        conn.commit()


# --- settings -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [7, 0.25, "Fondos de inversión · ñ €", True, False, [530, 150],
     {"geometry": "AdnQywAD", "sizes": [1, 2.5, None]}, None],
    ids=["int", "float", "str", "true", "false", "list", "dict", "none"],
)  # fmt: skip
def test_settings_round_trip_and_persist(data_dir: Path, value: Any) -> None:
    with Store(data_dir) as store:
        store.set_setting("key", "first value")
        store.set_setting("key", value)  # overwrites
        assert store.get_setting("key", "default") == value
    with Store(data_dir) as store:  # the next start of the app (R11)
        loaded = store.get_setting("key", "default")
        assert loaded == value and type(loaded) is type(value)  # True stays a bool, not 1


def test_missing_setting_returns_the_default(store: Store) -> None:
    assert store.get_setting("missing") is None
    assert store.get_setting("missing", 5) == 5
    store.set_setting("present", None)
    assert store.get_setting("present", 5) is None  # a stored None is a value, not a missing key


# --- dataset ------------------------------------------------------------------------------------


def test_dataset_round_trip(store: Store, funds_table: Table, funds_roles: Roles) -> None:
    assert not store.has_dataset() and store.load_dataset() is None
    store.save_dataset(funds_table, funds_roles)
    assert store.has_dataset()
    table, roles = store.load_dataset()
    assert table == funds_table  # source name, sheet, columns, rows and sheet row numbers
    assert table.fingerprint() == funds_table.fingerprint()
    assert roles == funds_roles and all(type(role) is Role for role in roles.values())


def test_saving_a_second_dataset_replaces_the_first(store: Store, funds_table: Table, funds_roles: Roles) -> None:
    store.save_dataset(funds_table, funds_roles)
    other = Table("other.csv", "", ["Name", "Ticker"], [["Alpha", "AAA"], ["Beta", "BBB"]], [5, 9])
    other_roles = {"Name": Role.SEARCH, "Ticker": Role.DECIDING}
    store.save_dataset(other, other_roles)
    assert store.load_dataset() == (other, other_roles)  # no rows or roles left over from the first


def test_clear_dataset(store: Store, funds_table: Table, funds_roles: Roles) -> None:
    store.save_dataset(funds_table, funds_roles)
    store.clear_dataset()
    assert not store.has_dataset() and store.load_dataset() is None
    store.clear_dataset()  # clearing nothing is fine


def test_save_roles(store: Store, funds_table: Table, funds_roles: Roles) -> None:
    store.save_roles({"Fund name": Role.SEARCH})  # no dataset yet: nothing to update
    assert not store.has_dataset()
    store.save_dataset(funds_table, funds_roles)
    changed = dict(funds_roles, ISIN=Role.IGNORE, Currency=Role.DECIDING)
    store.save_roles({**changed, "Not a column": Role.DECIDING})
    table, roles = store.load_dataset()
    assert roles == changed and table == funds_table


def test_save_dataset_rejects_ragged_rows(store: Store) -> None:
    ragged = Table("bad.csv", "", ["A", "B"], [["1", "2"], ["3"]], [2, 3])
    with pytest.raises(ValueError):
        store.save_dataset(ragged, {"A": Role.SEARCH})
    assert not store.has_dataset()


# --- history ------------------------------------------------------------------------------------


def test_add_entries_assigns_ids_and_loads_oldest_first(store: Store) -> None:
    originals = [_entry("iShares Core S&P 500"), _entry("Vanguard FTSE All-World")]
    first = store.add_entries(originals)
    second = store.add_entries([_entry("Amundi MSCI EM")])
    ids = [entry.id for entry in first + second]
    assert all(isinstance(entry_id, int) for entry_id in ids) and ids == sorted(set(ids))
    assert [entry.id for entry in originals] == [None, None]  # the caller's objects are untouched
    loaded = store.load_entries()
    assert [entry.query for entry in loaded] == ["iShares Core S&P 500", "Vanguard FTSE All-World", "Amundi MSCI EM"]
    assert [entry.id for entry in loaded] == ids


def test_re_searching_a_query_replaces_its_list(store: Store) -> None:
    store.add_entries([_entry("iShares Core S&P 500", results=2), _entry("Vanguard FTSE All-World")])
    (refreshed,) = store.add_entries([_entry("  ISHARES core   S&P 500 ", results=3)])  # same words, other case/spaces
    loaded = store.load_entries()
    assert [entry.query for entry in loaded] == ["Vanguard FTSE All-World", "  ISHARES core   S&P 500 "]
    assert loaded[-1].id == refreshed.id and len(loaded[-1].results) == 3  # newest, with the new results
    # within one batch the later duplicate wins
    store.add_entries([_entry("Amundi MSCI EM", results=1), _entry("amundi msci em", results=4)])
    assert _queries(store)[-1] == "amundi msci em" and len(store.load_entries()[-1].results) == 4
    assert len(store.load_entries()) == 3


def test_delete_and_clear_entries(store: Store) -> None:
    stored = store.add_entries([_entry(f"query {n}") for n in range(4)])
    store.delete_entries([stored[1].id, stored[3].id, 999_999])  # unknown ids are ignored
    assert _queries(store) == ["query 0", "query 2"]
    store.delete_entries([])
    store.clear_entries()
    assert store.load_entries() == []


def test_evidence_and_results_survive_the_json_round_trip(data_dir: Path) -> None:
    evidence = [
        FamilyEvidence("Brokerage", "BlackRock", 0.99, "iShares"),
        FamilyEvidence("ISIN", "IE00B5BMR087", 1.0, "IE00B5BMR087"),
    ]
    entry = _entry("iShares Core S&P 500 «acc»", results=3, evidence=evidence)
    entry.results[0].detail = {  # what an engine may hand over: numpy scalars, tuples
        "semantic": np.float32(0.91), "lexical": np.float64(0.5), "text_score": 0.75, "family_factor": 1.0,
        "exact_id": "ISIN", "agreements": ("Brokerage",), "conflicts": [],
    }  # fmt: skip
    with Store(data_dir) as store:
        store.add_entries([entry])
    with Store(data_dir) as store:  # history persists after closing the app (R11)
        (loaded,) = store.load_entries()
    assert loaded.evidence == evidence
    for field in ("query", "created_at", "dataset_label", "columns", "deciding", "primary_column"):
        assert getattr(loaded, field) == getattr(entry, field), field
    assert [(r.rank, r.score, r.source_row, r.cells) for r in loaded.results] == [
        (r.rank, r.score, r.source_row, r.cells) for r in entry.results
    ]
    detail = loaded.results[0].detail
    assert detail["semantic"] == pytest.approx(0.91) and type(detail["semantic"]) is float
    assert detail["agreements"] == ["Brokerage"] and detail["exact_id"] == "ISIN"
    assert loaded.results[1].detail == entry.results[1].detail


# --- thread safety -------------------------------------------------------------------------------


def test_concurrent_writers_and_readers(store: Store) -> None:
    threads, rounds = 8, 25
    barrier = threading.Barrier(threads)
    errors: list[BaseException] = []
    cache = store.embedding_cache

    def work(t: int) -> None:
        try:
            barrier.wait()
            for i in range(rounds):
                store.add_entries([_entry(f"thread {t} query {i}", results=1)])
                store.set_setting(f"thread.{t}.{i}", [t, i])
                assert store.get_setting(f"thread.{t}.{i}") == [t, i]
                store.load_entries()
                cache.put_many("model", [f"text {t} {i}"], _unit_vectors(1, dim=16, seed=t * 100 + i))
                assert len(cache.get_many("model", [f"text {t} {i}", f"text {t} {max(0, i - 1)}"])) == 2
        except BaseException as exc:  # reported by the main thread
            errors.append(exc)

    workers = [threading.Thread(target=work, args=(t,)) for t in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert not any(worker.is_alive() for worker in workers)
    assert errors == []
    entries = store.load_entries()
    assert len(entries) == threads * rounds and len({entry.id for entry in entries}) == len(entries)
    for t in range(threads):  # each thread's lists are all there, in the order it added them
        mine = [entry.query for entry in entries if entry.query.startswith(f"thread {t} ")]
        assert mine == [f"thread {t} query {i}" for i in range(rounds)]
        assert all(store.get_setting(f"thread.{t}.{i}") == [t, i] for i in range(rounds))
    assert len(cache) == threads * rounds


# --- embedding cache ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[SqliteEmbeddingCache]:
    opened = SqliteEmbeddingCache(tmp_path / "cache" / CACHE_DB_NAME)
    yield opened
    opened.close()


def test_vectors_round_trip_through_float16(cache: SqliteEmbeddingCache) -> None:
    texts = ["iShares Core S&P 500", "Vanguard FTSE All-World", "", "Índice España 🇪🇸"]
    vectors = _unit_vectors(len(texts))
    cache.put_many("model-a", texts, vectors)
    found = cache.get_many("model-a", texts)
    assert sorted(found) == [0, 1, 2, 3]
    for position, vector in found.items():
        assert vector.dtype == np.float32 and vector.shape == (384,)
        assert np.abs(vector - vectors[position]).max() <= 1e-3


def test_model_ids_keep_their_own_vectors(cache: SqliteEmbeddingCache) -> None:
    first, second = _unit_vectors(1, seed=1), _unit_vectors(1, seed=2)
    cache.put_many("model-a", ["S&P 500"], first)
    assert cache.get_many("model-b", ["S&P 500"]) == {}
    cache.put_many("model-b", ["S&P 500"], second)
    assert np.abs(cache.get_many("model-a", ["S&P 500"])[0] - first[0]).max() <= 1e-3
    assert np.abs(cache.get_many("model-b", ["S&P 500"])[0] - second[0]).max() <= 1e-3
    assert len(cache) == 2


def test_get_many_maps_every_position_including_duplicates(cache: SqliteEmbeddingCache) -> None:
    vectors = _unit_vectors(2, seed=3)
    cache.put_many("model", ["alpha", "beta"], vectors)
    found = cache.get_many("model", ["beta", "missing", "alpha", "beta", "alpha"])
    assert sorted(found) == [0, 2, 3, 4]
    for position, expected in ((0, 1), (2, 0), (3, 1), (4, 0)):
        np.testing.assert_allclose(found[position], vectors[expected], atol=1e-3)
    assert cache.get_many("model", []) == {}


def test_put_many_rejects_mismatched_vectors(cache: SqliteEmbeddingCache) -> None:
    with pytest.raises(ValueError):
        cache.put_many("model", ["a", "b"], _unit_vectors(3, dim=8))
    assert len(cache) == 0


def test_lru_pruning_keeps_the_most_recently_used(tmp_path: Path) -> None:
    cache = SqliteEmbeddingCache(tmp_path / CACHE_DB_NAME, max_entries=3)
    try:
        vectors = _unit_vectors(4, dim=8, seed=4)
        for text, vector in zip("abc", vectors):
            cache.put_many("model", [text], vector[None, :])
        assert sorted(cache.get_many("model", ["a"])) == [0]  # reading "a" leaves "b" as least recently used
        cache.put_many("model", ["d"], vectors[3][None, :])
        assert len(cache) == 3
        assert sorted(cache.get_many("model", ["a", "b", "c", "d"])) == [0, 2, 3]  # "b" was dropped
    finally:
        cache.close()


# --- damaged or foreign files ----------------------------------------------------------------------


@pytest.mark.parametrize("damage", ["random bytes", "truncated database"])
def test_corrupt_app_database_is_moved_aside(data_dir: Path, damage: str) -> None:
    path = data_dir / APP_DB_NAME
    if damage == "random bytes":
        data_dir.mkdir(parents=True)
        path.write_bytes(random.Random(7).randbytes(8192))
    else:
        with Store(data_dir) as store:
            store.add_entries([_entry(f"query {n}", results=5) for n in range(300)])
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 3])
    damaged = path.read_bytes()

    with Store(data_dir) as store:
        moved = [p for p in data_dir.glob(f"{APP_DB_NAME}.corrupt-*") if not p.name.endswith(("-wal", "-shm"))]
        assert len(moved) == 1 and moved[0].read_bytes() == damaged  # kept for inspection, not deleted
        assert store.load_entries() == [] and store.load_dataset() is None and store.get_setting("theme") is None
        store.add_entries([_entry("fresh start")])
        store.set_setting("theme", "dark")
    with Store(data_dir) as store:
        assert _queries(store) == ["fresh start"] and store.get_setting("theme") == "dark"


def test_corrupt_embedding_cache_is_moved_aside(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    (data_dir / CACHE_DB_NAME).write_bytes(random.Random(8).randbytes(4096))
    with Store(data_dir) as store:
        assert list(data_dir.glob(f"{CACHE_DB_NAME}.corrupt-*"))
        cache = store.embedding_cache
        assert len(cache) == 0
        cache.put_many("model", ["text"], _unit_vectors(1, dim=8))
        assert sorted(cache.get_many("model", ["text"])) == [0]


def test_files_from_a_newer_version_never_crash(data_dir: Path) -> None:
    with Store(data_dir) as store:
        store.set_setting("theme", "dark")
        store.embedding_cache.put_many("model", ["text"], _unit_vectors(1, dim=8))
    _execute(data_dir / APP_DB_NAME, "ALTER TABLE history ADD COLUMN pinned INTEGER", "PRAGMA user_version = 7")
    _execute(data_dir / CACHE_DB_NAME, "PRAGMA user_version = 9")
    with Store(data_dir) as store:
        assert store.get_setting("theme") == "dark"  # compatible tables are used as they are
        store.add_entries([_entry("still works")])
        assert _queries(store) == ["still works"]
        assert len(store.embedding_cache) == 0  # an unknown cache layout is simply rebuilt
    with closing(sqlite3.connect(data_dir / APP_DB_NAME)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 7  # never rewritten as an older version


def test_close_releases_the_files(data_dir: Path, funds_table: Table, funds_roles: Roles) -> None:
    store = Store(data_dir)
    store.save_dataset(funds_table, funds_roles)
    store.add_entries([_entry("iShares Core S&P 500")])
    store.embedding_cache.put_many("model", ["text"], _unit_vectors(1, dim=8))
    store.close()
    shutil.rmtree(data_dir)  # Windows refuses while any database file is still open
    assert not data_dir.exists()


# --- speed -----------------------------------------------------------------------------------------


def test_45k_row_dataset_saves_and_loads_quickly(store: Store, funds_roles: Roles) -> None:
    rows = [
        [f"IE{n:010d}", f"iShares Core MSCI World UCITS ETF Series {n}", ["BlackRock", "Vanguard", "DWS"][n % 3],
         "USD", f"0.{n % 90 + 10}"]
        for n in range(45_000)
    ]  # fmt: skip
    columns = ["ISIN", "Fund name", "Brokerage", "Currency", "TER"]
    table = Table("universe.xlsx", "Funds", columns, rows, list(range(2, 45_002)))
    roles = dict(funds_roles)  # same columns as the shared fixture

    start = time.perf_counter()
    store.save_dataset(table, roles)
    saved = time.perf_counter() - start
    start = time.perf_counter()
    loaded = store.load_dataset()
    read = time.perf_counter() - start

    assert saved < 5.0, f"saving took {saved:.2f} s"
    assert read < 5.0, f"loading took {read:.2f} s"
    assert loaded == (table, roles)
