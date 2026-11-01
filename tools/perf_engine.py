"""Benchmark the search core on a synthetic ~45,000-row fund universe.

Usage:  .venv/Scripts/python.exe tools/perf_engine.py [--rows 45000] [--queries 300]

Measures FamilyModel.build, row cleaning and LexicalIndex construction (target <= 6 s
together), full SearchIndex.build with and without (random, 384-d) embeddings, and the
per-query latency of SearchIndex.search including the 45k x 384 matrix product (target
<= 40 ms).  Nothing here touches the network or the real model.  Exit status 0 when every
target is met.
"""

from __future__ import annotations

import argparse
import random
import statistics
import string
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semdb.core.engine import SearchIndex  # noqa: E402
from semdb.core.families import FamilyModel  # noqa: E402
from semdb.core.lexical import LexicalIndex  # noqa: E402
from semdb.core.types import CancelFn, Role, Table  # noqa: E402

T = TypeVar("T")

BRANDS = [  # (brand used in fund names, brokerage column value, abbreviations seen in some names)
    ("iShares", "BlackRock", ["iShs", "ISHS"]),
    ("Vanguard", "Vanguard", []),
    ("Xtrackers", "DWS", ["Xtr"]),
    ("Lyxor", "Amundi", []),
    ("Amundi", "Amundi", []),
    ("SPDR", "State Street Global Advisors", []),
    ("Invesco", "Invesco", []),
    ("HSBC", "HSBC Asset Management", []),
    ("UBS", "UBS Asset Management", []),
    ("JPMorgan", "J.P. Morgan Asset Management", ["JPM"]),
    ("Franklin", "Franklin Templeton", []),
    ("WisdomTree", "WisdomTree", []),
    ("VanEck", "VanEck", []),
    ("L&G", "Legal & General Investment Management", []),
    ("BNP Paribas Easy", "BNP Paribas Asset Management", []),
    ("Deka", "Deka Investment", []),
    ("Fidelity", "Fidelity International", []),
    ("Global X", "Global X", []),
    ("Ossiam", "Natixis Investment Managers", []),
    ("Candriam", "Candriam", []),
    ("Santander", "Santander Asset Management", []),
    ("BBVA", "BBVA Asset Management", []),
    ("Bankinter", "Bankinter Gestión de Activos", []),
    ("CaixaBank", "CaixaBank Asset Management", []),
    ("Allianz", "Allianz Global Investors", []),
]
INDICES = [  # (index, category)
    ("S&P 500", "Equity"), ("S&P 400 Mid Cap", "Equity"), ("S&P 600 Small Cap", "Equity"),
    ("MSCI World", "Equity"), ("MSCI ACWI", "Equity"), ("MSCI Emerging Markets", "Equity"),
    ("MSCI Europe", "Equity"), ("MSCI USA", "Equity"), ("MSCI Japan", "Equity"),
    ("MSCI China", "Equity"), ("MSCI India", "Equity"), ("MSCI World Small Cap", "Equity"),
    ("MSCI World Value Factor", "Equity"), ("MSCI World Quality Factor", "Equity"),
    ("MSCI World Momentum Factor", "Equity"), ("MSCI World Minimum Volatility", "Equity"),
    ("MSCI World ESG Leaders", "Equity"), ("MSCI Europe SRI", "Equity"), ("FTSE All-World", "Equity"),
    ("FTSE 100", "Equity"), ("FTSE 250", "Equity"), ("FTSE Developed Europe", "Equity"),
    ("FTSE Emerging Markets", "Equity"), ("Euro Stoxx 50", "Equity"), ("Stoxx Europe 600", "Equity"),
    ("DAX", "Equity"), ("CAC 40", "Equity"), ("IBEX 35", "Equity"), ("Nasdaq 100", "Equity"),
    ("Dow Jones Industrial Average", "Equity"), ("Russell 2000", "Equity"), ("Nikkei 225", "Equity"),
    ("Topix", "Equity"), ("Hang Seng", "Equity"), ("CSI 300", "Equity"),
    ("Global Aggregate Bond", "Fixed Income"), ("Euro Government Bond", "Fixed Income"),
    ("US Treasury 1-3yr", "Fixed Income"), ("US Treasury 7-10yr", "Fixed Income"),
    ("US Treasury 20+yr", "Fixed Income"), ("Euro Corporate Bond", "Fixed Income"),
    ("USD Corporate Bond", "Fixed Income"), ("Euro High Yield Corporate Bond", "Fixed Income"),
    ("Emerging Markets Bond", "Fixed Income"), ("Global Inflation Linked Bond", "Fixed Income"),
    ("Physical Gold", "Commodity"), ("Bloomberg Commodity", "Commodity"),
    ("Global Clean Energy", "Thematic"), ("Global Water", "Thematic"), ("Robotics & AI", "Thematic"),
    ("Cybersecurity", "Thematic"), ("Healthcare Innovation", "Thematic"), ("Global REIT", "Real Estate"),
    ("Dividend Aristocrats", "Equity"), ("Global Infrastructure", "Equity"), ("Semiconductor", "Thematic"),
    ("Biotech", "Thematic"), ("Battery Value-Chain", "Thematic"), ("Uranium Miners", "Thematic"),
    ("Copper Miners", "Thematic"),
]
SHARE_CLASSES = [
    "UCITS ETF (Acc)", "UCITS ETF (Dist)", "UCITS ETF 1C", "UCITS ETF 1D",
    "Index Fund A Acc", "Index Fund B Inc",
]
CURRENCIES = ["USD", "EUR", "GBP", "CHF", "EUR Hedged"]
ISIN_PREFIXES = ["IE00", "LU", "DE000", "FR00", "ES0"]
ROLES = {
    "ISIN": Role.DECIDING,
    "Fund name": Role.SEARCH,
    "Brokerage": Role.DECIDING,
    "Category": Role.SEARCH,
    "Currency": Role.IGNORE,
    "TER": Role.IGNORE,
}


def synthetic_table(n_rows: int, seed: int) -> Table:
    """brands x indices x share classes x currencies (45,000 combinations), shuffled."""
    rng = random.Random(seed)
    alphabet = string.ascii_uppercase + string.digits
    rows: list[list[str]] = []
    isins: set[str] = set()
    for brand, company, abbreviations in BRANDS:
        for index_name, category in INDICES:
            for share_class in SHARE_CLASSES:
                for currency in CURRENCIES:
                    shown = rng.choice(abbreviations) if abbreviations and rng.random() < 0.1 else brand
                    isin = ""
                    while not isin or isin in isins:
                        prefix = rng.choice(ISIN_PREFIXES)
                        body = "".join(rng.choices(alphabet, k=11 - len(prefix)))
                        isin = f"{prefix}{body}{rng.randrange(10)}"
                    isins.add(isin)
                    name = f"{shown} {index_name} {share_class} {currency}"
                    rows.append([isin, name, company, category, currency, f"{rng.uniform(0.03, 0.95):.2f}"])
    rng.shuffle(rows)
    for i in range(len(rows), n_rows):  # more rows than combinations: numbered copies
        base = rows[i % len(BRANDS)]
        rows.append([f"XX{i:010d}", f"{base[1]} {i}", *base[2:]])
    rows = rows[:n_rows]
    columns = list(ROLES)
    return Table("synthetic_funds.xlsx", "Funds", columns, rows, list(range(2, len(rows) + 2)))


class RandomEmbedder:
    """Stand-in for the ONNX model: random unit vectors of the real model's dimension."""

    model_id = "perf-random-384"
    dim = 384

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def encode(
        self,
        texts: list[str],
        progress: Callable[[int, int], None] | None = None,
        cancel: CancelFn | None = None,
    ) -> np.ndarray:
        vectors = self._rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if progress is not None:
            progress(len(texts), len(texts))
        return vectors


class StageClock:
    """ProgressFn that records how long each build stage took."""

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}
        self._stage: str | None = None
        self._since = time.perf_counter()

    def __call__(self, stage: str, fraction: float) -> None:
        if stage != self._stage:
            self.stop()
            self._stage = stage

    def stop(self) -> None:
        now = time.perf_counter()
        if self._stage is not None:
            self.durations[self._stage] = self.durations.get(self._stage, 0.0) + now - self._since
        self._stage, self._since = None, now


def sample_queries(table: Table, n: int, seed: int) -> list[str]:
    """Query mix: full names, names without the brand, brand + index, company instead of brand,
    a dropped letter, ISIN only, brand only, a lower-case prefix."""
    rng = random.Random(seed)
    queries: list[str] = []
    for i in range(n):
        isin, name, company, *_ = rng.choice(table.rows)
        words = name.split()
        kind = i % 8
        if kind == 0:
            queries.append(name)
        elif kind == 1:
            queries.append(" ".join(words[1:4]))
        elif kind == 2:
            queries.append(" ".join(words[:3]))
        elif kind == 3:
            queries.append(f"{company.split()[0]} {' '.join(words[1:3])}")
        elif kind == 4:
            j = rng.randrange(len(words))
            if len(words[j]) > 4:
                k = rng.randrange(1, len(words[j]) - 1)
                words[j] = words[j][:k] + words[j][k + 1 :]
            queries.append(" ".join(words[:5]))
        elif kind == 5:
            queries.append(isin)
        elif kind == 6:
            queries.append(words[0])
        else:
            queries.append(name.lower()[: max(8, len(name) // 2)])
    return queries


def timed(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def percentile(seconds: list[float], q: float) -> float:
    ms = sorted(s * 1000 for s in seconds)
    return ms[min(len(ms) - 1, int(q * len(ms)))]


def latency_line(label: str, seconds: list[float]) -> str:
    median = statistics.median(seconds) * 1000
    return (
        f"  {label:<34} median {median:6.2f} ms   p95 {percentile(seconds, 0.95):6.2f} ms"
        f"   max {percentile(seconds, 1.0):6.2f} ms"
    )


def report_build(title: str, seconds: float, clock: StageClock) -> None:
    print(f"\n{title}: {seconds:.2f} s")
    for stage, duration in clock.durations.items():
        print(f"  {stage:<30} {duration:6.2f} s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=45_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    table, generate_s = timed(lambda: synthetic_table(args.rows, args.seed))
    deciding = [c for c in table.columns if ROLES[c] is Role.DECIDING]
    search = [c for c in table.columns if ROLES[c] is Role.SEARCH]
    print(f"Synthetic table: {len(table):,} rows x {len(table.columns)} columns ({generate_s:.2f} s)")

    # The pre-embedding part of SearchIndex.build, component by component.
    family, family_s = timed(lambda: FamilyModel.build(table, deciding, search))
    search_idx = [table.column_index(c) for c in search]

    def cleaned_texts() -> list[str]:
        return [
            " · ".join(c for c in family.clean_cells(r, [row[j] for j in search_idx]) if c)
            for r, row in enumerate(table.rows)
        ]

    texts, clean_s = timed(cleaned_texts)
    lexical, lexical_s = timed(lambda: LexicalIndex(texts))
    build_total = family_s + clean_s + lexical_s
    print("\nBuild components (no embeddings)")
    print(f"  FamilyModel.build      {family_s:6.2f} s")
    print(f"  clean_cells (all rows) {clean_s:6.2f} s")
    print(f"  LexicalIndex           {lexical_s:6.2f} s   (target <= 5 s)")
    print(f"  total                  {build_total:6.2f} s   (target <= 6 s)")
    for column in family.columns:
        print(
            f"  {column.name}: {len(column.values):,} values, identifier={column.is_identifier}, "
            f"{len(column.indicators):,} indicator keys, "
            f"aliases learned for {len(column.learned_aliases)} values"
        )
    for column_name, values in family.learned_alias_table().items():
        pairs = ", ".join(f"{alias} -> {value}" for value, aliases in values.items() for alias in aliases)
        print(f"  learned ({column_name}): {pairs}")

    clock = StageClock()
    lexical_index, seconds = timed(lambda: SearchIndex.build(table, ROLES, progress=clock))
    clock.stop()
    report_build("SearchIndex.build, lexical only", seconds, clock)

    embedder = RandomEmbedder(args.seed)
    clock = StageClock()
    index, seconds = timed(lambda: SearchIndex.build(table, ROLES, embedder=embedder, progress=clock))
    clock.stop()
    report_build("SearchIndex.build, random 384-d embeddings", seconds, clock)

    queries = sample_queries(table, args.queries, args.seed)
    for query in queries[:5]:  # warm-up (BLAS threads, allocator)
        index.search(query)
    query_vector = embedder.encode(["probe"])[0]
    matrix = embedder.encode([f"row {i}" for i in range(len(table))])
    detect_s = [timed(lambda q=q: family.detect(q))[1] for q in queries]
    lexical_q = [timed(lambda q=q: lexical.scores(q))[1] for q in queries]
    matmul_s = [timed(lambda: matrix @ query_vector)[1] for _ in queries]
    lexical_only_s = [timed(lambda q=q: lexical_index.search(q))[1] for q in queries]
    hybrid_s = [timed(lambda q=q: index.search(q))[1] for q in queries]
    _, batch_s = timed(lambda: index.search_many(queries))
    print(f"\nPer-query latency over {len(queries)} queries (k = 50)")
    print(latency_line("FamilyModel.detect", detect_s))
    print(latency_line("LexicalIndex.scores (<= 15 ms)", lexical_q))
    print(latency_line(f"{len(table) // 1000}k x 384 matrix-vector product", matmul_s))
    print(latency_line("search, lexical only", lexical_only_s))
    print(latency_line("search, hybrid (<= 40 ms)", hybrid_s))
    print(f"  search_many({len(queries)}): {batch_s:.2f} s, {1000 * batch_s / len(queries):.2f} ms per query")

    ok = (
        build_total <= 6
        and lexical_s <= 5
        and percentile(lexical_q, 0.95) <= 15
        and percentile(hybrid_s, 0.95) <= 40
    )
    print(f"\n{'All targets met.' if ok else 'Some targets were missed.'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
