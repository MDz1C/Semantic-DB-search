"""Retrieval-quality benchmark on the synthetic fund universe (offline dev tool).

Indexes ``make_sample_data`` rows (ISIN, Ticker, Brokerage = deciding; Fund name,
Category = search), fires realistic variants of random target rows (exact name,
abbreviated brand, company instead of brand, typos, Spanish paraphrases...) and reports
Acc@1, Acc@5, MRR@10 and brand precision@5, overall and per query type, for the full
hybrid engine and its ablations.  Uses only the public engine/embedder/storage APIs.

    .venv/Scripts/python.exe tools/evaluate.py --rows 45000 --queries 600 --json build/eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from make_sample_data import ABBREVIATIONS, COLUMNS, FundRecord, compose_name, generate_records  # noqa: E402
from semdb.core.engine import EngineConfig, SearchIndex  # noqa: E402
from semdb.core.text import normalize, tokens  # noqa: E402
from semdb.core.types import Embedder, EmbeddingCache, QueryOutcome, Role, Table  # noqa: E402

CACHE_PATH = ROOT / "build" / "eval-cache" / "embeddings.sqlite"
K = 10
ROLES = {name: Role.IGNORE for name in COLUMNS} | {
    "ISIN": Role.DECIDING, "Ticker": Role.DECIDING, "Brokerage": Role.DECIDING,
    "Fund name": Role.SEARCH, "Category": Role.SEARCH,
}
BROKERAGE = COLUMNS.index("Brokerage")
# Targets are drawn per product kind so ETFs and Spanish funds are not drowned by the
# (far more numerous) SICAV share classes of a large universe.
TARGET_MIX = {"ETF": 0.5, "Fund": 0.25, "Index fund": 0.1, "FI": 0.15}

COMPANY_SUFFIXES = (" Asset Management", " Global Advisors", " Gestión de Activos", " Gestión", " Gestora",
                    " International", " Value Investors", " ETFs")
GENERIC_WORDS = {"ucits", "etf", "etc", "fund", "funds", "index", "fi", "fil", "clase"}
DROPPABLE = ("UCITS ETF", "Index Fund", "Fund", "FIL", "FI", "ETC")
SPANISH = (  # English phrase in a name -> how a Spanish speaker would write it (first match wins)
    ("Emerging Markets", "Mercados Emergentes"), ("USD Treasury Bond", "Bonos del Tesoro EEUU"),
    ("Euro Inflation Linked Govt Bond", "Bonos Gubernamentales Euro Ligados a la Inflación"),
    ("EUR Govt Bond", "Bonos Gubernamentales Euro"), ("Government Bond", "Bonos Gubernamentales"),
    ("High Yield Corporate Bond", "Bonos Corporativos de Alto Rendimiento"),
    ("High Yield Bond", "Bonos de Alto Rendimiento"), ("Corporate Bond", "Bonos Corporativos"),
    ("Global Aggregate Bond", "Bonos Agregados Globales"), ("Short Duration Bond", "Bonos de Corta Duración"),
    ("Physical Gold", "Oro Físico"), ("Physical Silver", "Plata Física"),
    ("Global Clean Energy", "Energía Limpia Global"), ("Smaller Companies", "Pequeñas Compañías"),
    ("Small Cap", "Pequeña Capitalización"), ("Information Technology", "Tecnología de la Información"),
    ("Health Care", "Salud"), ("Healthcare", "Salud"), ("Technology", "Tecnología"), ("Dividend", "Dividendos"),
    ("Water", "Agua"), ("Infrastructure", "Infraestructuras"), ("Real Estate", "Inmobiliario"),
    ("Semiconductor", "Semiconductores"), ("Artificial Intelligence", "Inteligencia Artificial"),
    ("Cybersecurity", "Ciberseguridad"), ("Uranium", "Uranio"), ("Defence", "Defensa"), ("Defense", "Defensa"),
    ("European", "Europea"), ("Europe", "Europa"), ("Japan", "Japón"), ("All-World", "Todo el Mundo"),
    ("World", "Mundial"),
)
ENGLISH = (  # Spanish phrase in an FI name -> English wording (all matches applied)
    ("Renta Fija Corto Plazo", "Short-Term Fixed Income"), ("Renta Fija", "Fixed Income"),
    ("Corto Plazo", "Short Term"), ("Largo Plazo", "Long Term"), ("Acciones Españolas", "Spanish Equities"),
    ("Grandes Compañías", "Large Caps"), ("Bolsa", "Equity"), ("Bonos", "Bonds"), ("Bono", "Bond"),
    ("Internacional", "International"), ("Índice", "Index"), ("Europeo", "European"), ("Europa", "Europe"),
    ("Emergentes", "Emerging Markets"), ("España", "Spain"), ("Dividendo", "Dividend"),
    ("Monetario", "Money Market"), ("Tecnología", "Technology"), ("Tecnológico", "Technology"),
    ("Sanidad", "Healthcare"), ("Selección", "Selection"), ("Sostenible", "Sustainable"), ("Japón", "Japan"),
    ("América", "America"), ("Mixto", "Mixed"), ("Dinero", "Cash"),
)
_REWORDS = {
    "Acc": "Accumulating", "Dist": "Distributing", "Accumulating": "Acc", "Distributing": "Dist",
    "acc": "accumulating", "dist": "distributing", "dis": "distributing", "Dis": "Distributing", "Inc": "Income",
    "Hedged": "Hdg", "Hdg": "Hedged", "hedged": "hdg", "(C)": "Acc", "(D)": "Dist", "Clase": "Class",
}
_REWORD_RE = re.compile(r"\([CD]\)|\b(?:" + "|".join(w for w in _REWORDS if w[0] != "(") + r")\b")
_ACCENT_CLASSES = {"á": "[áa]", "é": "[ée]", "í": "[íi]", "ó": "[óo]", "ú": "[úu]", "ñ": "[ñn]", "Í": "[ÍI]"}


def _phrase_re(phrase: str) -> re.Pattern[str]:
    """Whole-phrase pattern that also matches the phrase written without accents."""
    return re.compile(r"\b" + "".join(_ACCENT_CLASSES.get(ch, re.escape(ch)) for ch in phrase) + r"\b")


_SPANISH_RES = tuple((_phrase_re(en), es) for en, es in SPANISH)
_ENGLISH_RES = tuple((_phrase_re(es), en) for es, en in ENGLISH)


class Searchable(Protocol):
    def search(self, query: str, k: int = 50) -> QueryOutcome: ...


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    tag: str
    target: int  # row index of the record the query was made from
    accept: frozenset[int]  # rows that answer it equally well: the target plus indistinguishable rows
    names_brand: bool  # the query names the target's brand or company


@dataclass(slots=True)
class Universe:
    """Records plus lookups of rows that a brand-free query cannot tell apart."""

    records: list[FundRecord]
    by_body: dict[str, list[int]] = field(default_factory=dict)
    by_company_body: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by_body: dict[str, list[int]] = defaultdict(list)
        by_company_body: dict[tuple[str, str], list[int]] = defaultdict(list)
        for row, record in enumerate(self.records):
            key = normalize(brand_free(record))
            by_body[key].append(row)
            by_company_body[(record.company, key)].append(row)
        self.by_body, self.by_company_body = dict(by_body), dict(by_company_body)


def brand_free(record: FundRecord) -> str:
    return compose_name("", "", record.body, record.share_class)


def with_brand(record: FundRecord, brand: str | None = None, body: str | None = None) -> str:
    """The record's name in canonical case, with another brand and/or body."""
    return compose_name(record.brand if brand is None else brand, record.sep,
                        record.body if body is None else body, record.share_class)


def company_short(company: str) -> str:
    for suffix in COMPANY_SUFFIXES:
        if company.endswith(suffix):
            return company[: -len(suffix)]
    return company


# --- query builders: (record, universe, row, rng) -> Query | None when not applicable -----
Builder = Callable[[FundRecord, Universe, int, random.Random], "Query | None"]


def _exact(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query:
    return Query(rec.name, "exact name", row, frozenset({row}), bool(rec.brand_text))


def _lowercase(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query:
    return Query(rec.name.lower(), "lowercase name", row, frozenset({row}), bool(rec.brand_text))


def _abbreviated(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    for word, abbreviations in ABBREVIATIONS.items():
        if word in rec.brand:
            options = [a for a in abbreviations if a != rec.brand_text] or list(abbreviations)
            text = with_brand(rec, rec.brand.replace(word, rng.choice(options)))
            return Query(text, "brand abbreviated", row, frozenset({row}), True)
    return None


def _company(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    short = company_short(rec.company)
    if set(tokens(short)) & set(tokens(rec.brand)):
        return None  # the brand already is the company name
    accept = u.by_company_body[(rec.company, normalize(brand_free(rec)))]
    return Query(f"{short} {brand_free(rec)}", "company instead of brand", row, frozenset(accept), True)


def _brand_removed(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    if not rec.brand_text or len(tokens(rec.body)) < 2:
        return None
    text = brand_free(rec)
    return Query(text, "brand removed", row, frozenset(u.by_body[normalize(text)]), False)


def _brand_at_end(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query:
    return Query(f"{brand_free(rec)} {rec.brand}", "brand moved to end", row, frozenset({row}), True)


def _typo(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    words = rec.body.split()
    candidates = [i for i, w in enumerate(words) if w.isalpha() and len(w) >= 5 and w.lower() not in GENERIC_WORDS]
    if not candidates:
        return None
    i = rng.choice(candidates)
    word = words[i]
    swaps = [j for j in range(1, len(word) - 1) if word[j] != word[j + 1]]
    if swaps and rng.random() < 0.5:
        j = rng.choice(swaps)
        words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    else:
        j = rng.randrange(1, len(word))
        words[i] = word[:j] + word[j + 1:]
    return Query(with_brand(rec, body=" ".join(words)), "typo", row, frozenset({row}), True)


def _dropped(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    for phrase in DROPPABLE:
        body, count = re.subn(rf"\s*\b{re.escape(phrase)}\b", "", rec.body, count=1)
        if count and tokens(body):
            return Query(with_brand(rec, body=body.strip()), "dropped words", row, frozenset({row}), True)
    return None


def _reordered(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    original = with_brand(rec).split()
    if len(set(original)) < 3:
        return None
    words = list(original)
    while words == original:
        rng.shuffle(words)
    return Query(" ".join(words), "reordered tokens", row, frozenset({row}), True)


def _reworded(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    share_class = _REWORD_RE.sub(lambda m: _REWORDS[m.group()], rec.share_class)
    if share_class == rec.share_class:
        return None
    text = compose_name(rec.brand, rec.sep, rec.body, share_class)
    return Query(text, "share class reworded", row, frozenset({row}), True)


def _isin(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query:
    return Query(rec.cells[0], "ISIN only", row, frozenset({row}), False)


def _ticker(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query:
    short = rec.short_name if rec.short_name != rec.brand else ""
    text = " ".join(part for part in (rec.cells[1], rec.brand, short) if part)
    return Query(text, "ticker + short name", row, frozenset({row}), True)


def _spanish(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    if rec.kind == "FI":
        return None
    for pattern, spanish in _SPANISH_RES:
        body, count = pattern.subn(spanish, rec.body, count=1)
        if count:
            return Query(with_brand(rec, body=body), "Spanish paraphrase", row, frozenset({row}), True)
    return None


def _english(rec: FundRecord, u: Universe, row: int, rng: random.Random) -> Query | None:
    if rec.kind != "FI":
        return None
    body = rec.body
    for pattern, english in _ENGLISH_RES:
        body = pattern.sub(english, body)
    if body == rec.body:
        return None
    return Query(with_brand(rec, body=body), "English paraphrase", row, frozenset({row}), True)


BUILDERS: tuple[Builder, ...] = (
    _exact, _lowercase, _abbreviated, _company, _brand_removed, _brand_at_end, _typo, _dropped,
    _reordered, _reworded, _isin, _ticker, _spanish, _english,
)
QUERY_TYPES = (
    "exact name", "lowercase name", "brand abbreviated", "company instead of brand", "brand removed",
    "brand moved to end", "typo", "dropped words", "reordered tokens", "share class reworded",
    "ISIN only", "ticker + short name", "Spanish paraphrase", "English paraphrase",
)


def make_queries(records: list[FundRecord], count: int, seed: int = 7) -> list[Query]:
    """``count`` queries, query types in turn, each from a random applicable target row.

    A type that no row supports (English paraphrases need Spanish funds) is skipped, so a
    tiny universe can yield fewer queries.
    """
    rng = random.Random(seed + 1)
    universe = Universe(records)
    groups: dict[str, list[int]] = defaultdict(list)
    for row, record in enumerate(records):
        groups["ETF" if record.kind == "ETC" else record.kind].append(row)
    mix = [(group, weight) for group, weight in TARGET_MIX.items() if groups.get(group)]
    queries: list[Query] = []
    for i in range(count):
        builder = BUILDERS[i % len(BUILDERS)]
        for attempt in range(200):
            if attempt < 100:
                group = rng.choices([g for g, _ in mix], [w for _, w in mix])[0]
                row = rng.choice(groups[group])
            else:  # this type may not exist in the preferred kinds (tiny universes)
                row = rng.randrange(len(records))
            query = builder(records[row], universe, row, rng)
            if query is not None and query.text.strip():
                queries.append(query)
                break
    return queries


# --- scoring ----------------------------------------------------------------------------
@dataclass(slots=True)
class Scored:
    query: Query
    rows: list[int]  # returned rows, best first
    rank: int | None  # 1-based rank of the first acceptable row within the top K
    brand_precision: float | None  # share of the top 5 sharing the target's Brokerage
    latency_ms: float


def score(query: Query, rows: list[int], brokerages: list[str], latency_ms: float = 0.0) -> Scored:
    rank = next((i + 1 for i, row in enumerate(rows[:K]) if row in query.accept), None)
    precision = None
    if query.names_brand:
        top = rows[:5]
        target = brokerages[query.target]
        precision = sum(brokerages[row] == target for row in top) / len(top) if top else 0.0
    return Scored(query, rows, rank, precision, latency_ms)


def metrics(scored: list[Scored]) -> dict[str, float | int | None]:
    n = len(scored)
    if not n:
        return {"n": 0, "acc@1": None, "acc@5": None, "mrr@10": None, "brand_p@5": None, "brand_n": 0}
    branded = [s.brand_precision for s in scored if s.brand_precision is not None]
    return {
        "n": n,
        "acc@1": sum(s.rank == 1 for s in scored) / n,
        "acc@5": sum(s.rank is not None and s.rank <= 5 for s in scored) / n,
        "mrr@10": sum(1 / s.rank for s in scored if s.rank) / n,
        "brand_p@5": sum(branded) / len(branded) if branded else None,
        "brand_n": len(branded),
    }


def run_queries(index: Searchable, queries: list[Query], brokerages: list[str]) -> list[Scored]:
    out = []
    for query in queries:
        started = time.perf_counter()
        outcome = index.search(query.text, K)
        elapsed = (time.perf_counter() - started) * 1000
        out.append(score(query, [match.row for match in outcome.matches], brokerages, elapsed))
    return out


@dataclass(slots=True)
class ConfigResult:
    name: str
    build_seconds: float
    overall: dict
    by_type: dict[str, dict]
    latency_ms: dict[str, float]
    scored: list[Scored]


def summarize(name: str, build_seconds: float, scored: list[Scored]) -> ConfigResult:
    by_type = {tag: metrics([s for s in scored if s.query.tag == tag]) for tag in QUERY_TYPES}
    latencies = sorted(s.latency_ms for s in scored) or [0.0]
    latency = {
        "avg": sum(latencies) / len(latencies),
        "p95": latencies[min(len(latencies) - 1, math.ceil(0.95 * len(latencies)) - 1)],
        "max": latencies[-1],
    }
    return ConfigResult(name, build_seconds, metrics(scored), by_type, latency, scored)


def failures(result: ConfigResult, records: list[FundRecord], limit: int = 10) -> list[dict]:
    """Up to ``limit`` misses at rank 1, spread over query types."""
    misses = [s for s in result.scored if s.rank != 1]
    picked: list[Scored] = []
    seen: set[str] = set()
    for s in misses:  # one per type first, then the rest in order
        if s.query.tag not in seen:
            picked.append(s)
            seen.add(s.query.tag)
    picked += [s for s in misses if s not in picked]
    return [
        {
            "type": s.query.tag,
            "query": s.query.text,
            "expected": records[s.query.target].name,
            "got": records[s.rows[0]].name if s.rows else "(no result)",
            "expected_rank": s.rank,
        }
        for s in picked[:limit]
    ]


# --- running ----------------------------------------------------------------------------
def make_table(records: list[FundRecord]) -> Table:
    return Table(
        source_name="synthetic_funds.xlsx", sheet_name="Funds", columns=list(COLUMNS),
        rows=[record.cells for record in records], source_rows=list(range(2, len(records) + 2)),
    )


def configurations(embedder: Embedder | None) -> list[tuple[str, Embedder | None, EngineConfig]]:
    if embedder is None:
        return [("Full (lexical, no model)", None, EngineConfig()),
                ("No family model (lexical)", None, EngineConfig(use_family=False))]
    return [
        ("Full hybrid", embedder, EngineConfig()),
        ("No family model", embedder, EngineConfig(use_family=False)),
        ("Lexical only", None, EngineConfig()),
        ("Semantic heavy", embedder, EngineConfig(semantic_weight=0.9)),
    ]


def _progress_printer(label: str) -> Callable[[str, float], None]:
    last = {"stage": "", "step": -1}

    def report(stage: str, fraction: float) -> None:
        step = int(fraction * 10)
        if stage != last["stage"] or step != last["step"]:
            last.update(stage=stage, step=step)
            print(f"  [{label}] {stage} {fraction:4.0%}", file=sys.stderr, flush=True)

    return report


def evaluate(table: Table, queries: list[Query], configs: list[tuple[str, Embedder | None, EngineConfig]],
             cache: EmbeddingCache | None) -> list[ConfigResult]:
    brokerages = [row[BROKERAGE] for row in table.rows]
    results = []
    for name, embedder, config in configs:
        started = time.perf_counter()
        index = SearchIndex.build(table, ROLES, embedder=embedder, cache=cache if embedder else None,
                                  config=config, progress=_progress_printer(name))
        build_seconds = time.perf_counter() - started
        print(f"  [{name}] built in {build_seconds:.1f}s; running {len(queries)} queries", file=sys.stderr)
        results.append(summarize(name, build_seconds, run_queries(index, queries, brokerages)))
    return results


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _cell(text: str, width: int = 90) -> str:
    text = text.replace("|", "\\|")
    return text if len(text) <= width else text[: width - 1] + "…"


def render_report(results: list[ConfigResult], records: list[FundRecord], queries: list[Query],
                  rows: int, seed: int, model: str, generate_seconds: float) -> str:
    lines = [
        f"# Retrieval evaluation: {rows:,} rows, {len(queries)} queries, seed {seed}",
        "",
        f"Model: {model}. Universe generated in {generate_seconds:.1f}s. Metrics count a hit when the row is "
        "the target or indistinguishable from it for that query (same name without brand, for brand-free "
        "queries). Brand P@5 = share of the top 5 with the target's Brokerage, over queries naming a brand "
        "or company.",
        "",
        "| Configuration | Acc@1 | Acc@5 | MRR@10 | Brand P@5 | Build (s) | Avg query (ms) | p95 (ms) | Slowest (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        o = r.overall
        lines.append(
            f"| {r.name} | {_fmt(o['acc@1'])} | {_fmt(o['acc@5'])} | {_fmt(o['mrr@10'])} | {_fmt(o['brand_p@5'])} "
            f"| {r.build_seconds:.1f} | {r.latency_ms['avg']:.1f} | {r.latency_ms['p95']:.1f} | {r.latency_ms['max']:.1f} |"
        )
    for r in results:
        lines += ["", f"## {r.name}: per query type", "",
                  "| Query type | n | Acc@1 | Acc@5 | MRR@10 | Brand P@5 |", "|---|---:|---:|---:|---:|---:|"]
        for tag, m in r.by_type.items():
            if m["n"]:
                lines.append(f"| {tag} | {m['n']} | {_fmt(m['acc@1'])} | {_fmt(m['acc@5'])} | {_fmt(m['mrr@10'])} "
                             f"| {_fmt(m['brand_p@5'])} |")
    if results:
        lines += ["", f"## Failure examples ({results[0].name})", "",
                  "| Type | Query | Expected | Got (rank of expected) |", "|---|---|---|---|"]
        for f in failures(results[0], records):
            rank = f"#{f['expected_rank']}" if f["expected_rank"] else f"not in top {K}"
            lines.append(f"| {f['type']} | {_cell(f['query'], 60)} | {_cell(f['expected'], 70)} "
                         f"| {_cell(f['got'], 70)} ({rank}) |")
    return "\n".join(lines)


def report_json(results: list[ConfigResult], records: list[FundRecord], params: dict) -> dict:
    return {
        "params": params,
        "configs": [
            {"name": r.name, "build_seconds": r.build_seconds, "latency_ms": r.latency_ms,
             "overall": r.overall, "by_type": r.by_type}
            for r in results
        ],
        "failures": failures(results[0], records) if results else [],
        "queries": [asdict(s.query) | {"accept": sorted(s.query.accept), "rank": s.rank}
                    for s in (results[0].scored if results else [])],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark retrieval quality on the synthetic fund universe.")
    parser.add_argument("--rows", type=int, default=45000, help="universe size (default 45000)")
    parser.add_argument("--queries", type=int, default=600, help="number of queries (default 600)")
    parser.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    parser.add_argument("--json", type=Path, help="also write the full results as JSON")
    parser.add_argument("--no-model", action="store_true", help="skip the embedding model (lexical configurations)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # reports are often redirected to a file
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    started = time.perf_counter()
    records = generate_records(args.rows, args.seed)
    generate_seconds = time.perf_counter() - started
    table = make_table(records)
    queries = make_queries(records, args.queries, args.seed)

    embedder = cache = None
    if not args.no_model:
        from semdb.core.embedder import load_default_embedder
        from semdb.storage import SqliteEmbeddingCache

        embedder = load_default_embedder()
        if embedder is None:
            print("Embedding model unavailable: running the lexical configurations only.", file=sys.stderr)
        else:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            cache = SqliteEmbeddingCache(CACHE_PATH)
    try:
        results = evaluate(table, queries, configurations(embedder), cache)
    finally:
        if cache is not None:
            cache.close()

    model = embedder.model_id if embedder is not None else "none (lexical only)"
    print(render_report(results, records, queries, args.rows, args.seed, model, generate_seconds))
    if args.json:
        params = {"rows": args.rows, "queries": len(queries), "seed": args.seed, "model": model}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report_json(results, records, params), indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
