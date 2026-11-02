"""Render the key UI states to PNG files, in light and dark themes, for design review.

By default it uses an in-memory store and a fake search index over a realistic, generated
fund universe (45,210 rows: brokerages, fund names, ISINs).  ``--real`` uses the real store,
importer and search engine in a temporary data folder with the bundled sample workbook.

    .venv/Scripts/python.exe tools/ui_screenshots.py [--out DIR] [--real] [--scale 1.0]

Rendering uses the native Windows platform (real fonts) with windows kept off screen.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_OUT = Path(
    r"C:\Users\manuk\AppData\Local\Temp\claude\c--Users-manuk-Documents-Personal-projects-Semantic-DB-search"
    r"\0b9d854d-345b-4acb-b952-1d16fb4192b4\scratchpad\screens"
)
WINDOW_SIZE = (1320, 860)

# ---------------------------------------------------------------------------------------------------------------
# Realistic fake data
# ---------------------------------------------------------------------------------------------------------------
BRANDS = [  # brokerage, fund-name prefixes, ISIN countries, share-class suffixes
    ("BlackRock", ["iShares"], ["IE", "DE"],
     ["UCITS ETF USD (Acc)", "UCITS ETF USD (Dist)", "UCITS ETF EUR Hedged (Acc)", "UCITS ETF GBP Hedged (Dist)"]),
    ("Vanguard", ["Vanguard"], ["IE"],
     ["UCITS ETF USD Accumulating", "UCITS ETF USD Distributing", "UCITS ETF EUR Hedged Accumulating"]),
    ("Amundi", ["Amundi", "Lyxor"], ["LU", "FR"], ["UCITS ETF EUR (C)", "UCITS ETF USD (D)", "UCITS ETF Acc"]),
    ("DWS", ["Xtrackers"], ["IE", "LU"], ["UCITS ETF 1C", "UCITS ETF 1D", "UCITS ETF 2C EUR Hedged"]),
    ("State Street", ["SPDR"], ["IE"], ["UCITS ETF (Acc)", "UCITS ETF (Dist)"]),
    ("Invesco", ["Invesco"], ["IE"], ["UCITS ETF Acc", "UCITS ETF Dist"]),
    ("UBS", ["UBS ETF"], ["LU", "IE"], ["(USD) A-acc", "(EUR) A-dis", "(hedged to EUR) A-acc"]),
    ("HSBC", ["HSBC"], ["IE"], ["UCITS ETF USD", "UCITS ETF GBP"]),
    ("JPMorgan", ["JPM"], ["IE"], ["UCITS ETF USD (acc)", "UCITS ETF USD (dist)"]),
    ("Legal & General", ["L&G"], ["IE"], ["UCITS ETF", "UCITS ETF USD Acc"]),
    ("WisdomTree", ["WisdomTree"], ["IE", "JE"], ["UCITS ETF USD Acc", "UCITS ETF EUR Hedged Acc"]),
    ("VanEck", ["VanEck"], ["IE"], ["UCITS ETF A USD", "UCITS ETF"]),
    ("Fidelity", ["Fidelity"], ["IE"], ["UCITS ETF ACC-USD", "UCITS ETF INC-USD"]),
    ("Franklin Templeton", ["Franklin"], ["IE"], ["UCITS ETF", "UCITS ETF EUR Hedged"]),
    ("BNP Paribas", ["BNP Paribas Easy"], ["LU", "FR"], ["UCITS ETF EUR C", "UCITS ETF USD C"]),
]
INDICES = [
    ("Core S&P 500", "Equity"), ("S&P 500", "Equity"), ("MSCI World", "Equity"), ("Core MSCI World", "Equity"),
    ("MSCI EM IMI", "Equity"), ("FTSE All-World", "Equity"), ("MSCI Europe", "Equity"), ("Nasdaq 100", "Equity"),
    ("EURO STOXX 50", "Equity"), ("MSCI ACWI", "Equity"), ("Global Aggregate Bond", "Fixed income"),
    ("Euro Government Bond 1-3yr", "Fixed income"), ("USD Treasury Bond 7-10yr", "Fixed income"),
    ("MSCI World SRI", "Equity"), ("MSCI USA Quality Factor", "Equity"), ("FTSE 100", "Equity"), ("DAX", "Equity"),
    ("MSCI Japan", "Equity"), ("S&P 500 Equal Weight", "Equity"), ("Global Clean Energy", "Equity"),
    ("MSCI World Small Cap", "Equity"), ("MSCI Emerging Markets", "Equity"), ("STOXX Europe 600", "Equity"),
    ("MSCI China", "Equity"), ("S&P 400 US Mid Cap", "Equity"), ("Euro Corporate Bond", "Fixed income"),
    ("MSCI World Information Technology", "Equity"), ("Global Dividend", "Equity"), ("MSCI India", "Equity"),
    ("MSCI World Health Care", "Equity"), ("MSCI World Financials", "Equity"), ("MSCI USA", "Equity"),
    ("MSCI EMU", "Equity"), ("MSCI UK", "Equity"), ("MSCI Pacific ex-Japan", "Equity"),
    ("Global Infrastructure", "Equity"), ("Global Water", "Equity"), ("Cybersecurity", "Equity"),
    ("Automation & Robotics", "Equity"), ("Healthcare Innovation", "Equity"), ("Electric Vehicles", "Equity"),
    ("Euro Inflation Linked Govt Bond", "Fixed income"), ("USD Corporate Bond", "Fixed income"),
    ("J.P. Morgan EM Bond", "Fixed income"), ("Euro High Yield Corporate Bond", "Fixed income"),
    ("Japan Government Bond", "Fixed income"), ("Bloomberg Commodity", "Commodity"), ("Physical Gold", "Commodity"),
]
VARIANTS = ["", "", "", "ESG", "Screened", "Enhanced", "Paris-Aligned", "Climate", "Value", "Momentum", "Min Vol"]
TERS = ["0.03", "0.05", "0.07", "0.10", "0.12", "0.15", "0.18", "0.20", "0.22", "0.25", "0.30", "0.35", "0.45"]
KNOWN_FUNDS = [  # the brief's iShares-vs-Vanguard scenario, with real ISINs
    ["IE00B5BMR087", "iShares Core S&P 500 UCITS ETF USD (Acc)", "BlackRock", "Equity", "USD", "0.07"],
    ["IE0031442068", "iShares Core S&P 500 UCITS ETF USD (Dist)", "BlackRock", "Equity", "USD", "0.07"],
    ["IE00B4L5Y983", "iShares Core MSCI World UCITS ETF USD (Acc)", "BlackRock", "Equity", "USD", "0.20"],
    ["IE00BKM4GZ66", "iShares Core MSCI EM IMI UCITS ETF", "BlackRock", "Equity", "USD", "0.18"],
    ["IE00B3XXRP09", "Vanguard S&P 500 UCITS ETF USD Distributing", "Vanguard", "Equity", "USD", "0.07"],
    ["IE00BFMXXD54", "Vanguard S&P 500 UCITS ETF USD Accumulating", "Vanguard", "Equity", "USD", "0.07"],
    ["IE00BK5BQT80", "Vanguard FTSE All-World UCITS ETF USD Accumulating", "Vanguard", "Equity", "USD", "0.22"],
    ["IE00B3RBWM25", "Vanguard FTSE All-World UCITS ETF USD Distributing", "Vanguard", "Equity", "USD", "0.22"],
    ["LU1681048804", "Amundi S&P 500 UCITS ETF EUR (C)", "Amundi", "Equity", "EUR", "0.15"],
    ["LU1437016972", "Amundi MSCI Emerging Markets UCITS ETF", "Amundi", "Equity", "EUR", "0.20"],
    ["IE00BJ0KDQ92", "Xtrackers MSCI World UCITS ETF 1C", "DWS", "Equity", "USD", "0.19"],
    ["LU0274211480", "Xtrackers DAX UCITS ETF 1C", "DWS", "Equity", "EUR", "0.09"],
    ["IE00B6YX5C33", "SPDR S&P 500 UCITS ETF (Dist)", "State Street", "Equity", "USD", "0.03"],
    ["IE00BFY0GT14", "SPDR MSCI World UCITS ETF", "State Street", "Equity", "USD", "0.12"],
    ["IE00BYX2JD69", "iShares MSCI World SRI UCITS ETF EUR (Acc)", "BlackRock", "Equity", "EUR", "0.20"],
    ["IE00B52VJ196", "iShares MSCI Europe SRI UCITS ETF EUR (Acc)", "BlackRock", "Equity", "EUR", "0.20"],
    ["IE00BYZK4552", "iShares Automation & Robotics UCITS ETF USD (Acc)", "BlackRock", "Equity", "USD", "0.40"],
    ["IE00B4ND3602", "iShares Physical Gold ETC", "BlackRock", "Commodity", "USD", "0.12"],
]
COLUMNS = ["ISIN", "Fund name", "Brokerage", "Asset class", "Currency", "TER %"]
ALIASES = {
    "ishares": ("BlackRock", 0.99), "ishs": ("BlackRock", 0.86), "vanguard": ("Vanguard", 1.0),
    "amundi": ("Amundi", 1.0), "lyxor": ("Amundi", 0.93), "xtrackers": ("DWS", 0.97), "spdr": ("State Street", 0.98),
    "invesco": ("Invesco", 1.0), "hsbc": ("HSBC", 1.0), "jpm": ("JPMorgan", 0.84), "l&g": ("Legal & General", 0.95),
    "wisdomtree": ("WisdomTree", 1.0), "vaneck": ("VanEck", 1.0), "fidelity": ("Fidelity", 1.0),
}
GENERIC = {"ucits", "etf", "usd", "eur", "gbp", "chf", "acc", "dist", "1c", "1d", "2c", "hedged", "c", "d", "a",
           "accumulating", "distributing", "etc", "inc", "dis"}
QUERIES = [  # (query, minutes ago), oldest first
    ("Amundi MSCI Emerging Markets", 60 * 49),
    ("nasdaq 100 acumulación", 60 * 47),
    ("Physical gold", 60 * 27),
    ("S&P 400 midcap", 60 * 26),
    ("L&G clean energy", 60 * 25),
    ("IE00B4L5Y983", 60 * 5),
    ("SPDR S&P 500 dist", 60 * 3),
    ("Xtrackers DAX", 128),
    ("ishs msci world sri", 35),
    ("Vanguard FTSE All-World", 6),
    ("iShares Core S&P 500", 2),
]


def isin_check_digit(body: str) -> str:
    digits = "".join(str(int(ch, 36)) for ch in body)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 0:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return str((10 - total % 10) % 10)


def make_fund_table(rows: int = 45_210):
    from semdb.core.types import Table

    rng = random.Random(7)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    data = [list(r) for r in KNOWN_FUNDS]
    seen = {r[0] for r in data}
    flagships = {r[1] for r in data}  # stay unique
    copies: dict[str, int] = {}
    # Listings per generated name: the lists above make 16,272 names, so 2 copies capped the table
    # at 32,538 rows and the loop never ended.  Once nearly every name is used up, allow one more.
    limit, misses = 3, 0
    while len(data) < rows:
        brokerage, prefixes, countries, classes = rng.choice(BRANDS)
        index, asset = rng.choice(INDICES)
        if index.startswith("Core ") and brokerage not in ("BlackRock", "DWS"):
            index = index[5:]
        share_class = rng.choice(classes)
        name = " ".join(p for p in (rng.choice(prefixes), index, rng.choice(VARIANTS), share_class) if p)
        if name in flagships or copies.get(name, 0) >= limit:
            misses += 1
            if misses == 100_000:
                limit, misses = limit + 1, 0
            continue
        misses = 0
        body = rng.choice(countries) + "".join(rng.choice(alphabet) for _ in range(9))
        isin = body + isin_check_digit(body)
        if isin in seen:
            continue
        seen.add(isin)
        copies[name] = copies.get(name, 0) + 1
        currency = next((c for c in ("USD", "EUR", "GBP") if c in share_class), "EUR" if body[:2] in ("LU", "FR") else "USD")
        data.append([isin, name, brokerage, asset, currency, rng.choice(TERS)])
    return Table("funds_universe.xlsx", "Funds", list(COLUMNS), data, list(range(2, len(data) + 2)))


def fund_roles():
    from semdb.core.types import Role

    return {"ISIN": Role.DECIDING, "Fund name": Role.SEARCH, "Brokerage": Role.DECIDING,
            "Asset class": Role.SEARCH, "Currency": Role.IGNORE, "TER %": Role.IGNORE}


def fund_profiles(table):
    from semdb.core.types import ColumnProfile, Role

    reasons = {
        "ISIN": (Role.DECIDING, "Unique code per row — exact matches identify a row"),
        "Fund name": (Role.SEARCH, "Descriptive text"),
        "Brokerage": (Role.DECIDING, "Its values also appear inside 'Fund name' in 64 % of rows"),
        "Asset class": (Role.SEARCH, "Descriptive text"),
        "Currency": (Role.IGNORE, "Short codes with few distinct values"),
        "TER %": (Role.IGNORE, "Numbers — shown with results, not compared"),
    }
    profiles = []
    for j, name in enumerate(table.columns):
        values = [row[j] for row in table.rows[:4000]]
        samples: list[str] = []
        for value in values:
            if value and value not in samples:
                samples.append(value)
            if len(samples) == 5:
                break
        role, reason = reasons[name]
        profiles.append(ColumnProfile(name, j, len(table), len(set(values)), 12.0, 0.0, 0.0, name == "ISIN", role,
                                      reason, samples))
    return profiles


class MemoryStore:
    """In-memory stand-in for ``semdb.storage.Store``."""

    def __init__(self) -> None:
        self._settings: dict = {}
        self._dataset = None
        self._entries: list = []
        self._next_id = 1
        self._lock = threading.Lock()
        self.embedding_cache = None

    def get_setting(self, key, default=None):
        return self._settings.get(key, default)

    def set_setting(self, key, value) -> None:
        self._settings[key] = value

    def save_dataset(self, table, roles) -> None:
        self._dataset = (table, dict(roles))

    def load_dataset(self):
        return self._dataset

    def save_roles(self, roles) -> None:
        if self._dataset:
            self._dataset = (self._dataset[0], dict(roles))

    def has_dataset(self) -> bool:
        return self._dataset is not None

    def clear_dataset(self) -> None:
        self._dataset = None

    def add_entries(self, entries):
        from dataclasses import replace

        from semdb.core.text import normalize

        with self._lock:
            keys = {normalize(e.query) for e in entries}
            self._entries = [e for e in self._entries if normalize(e.query) not in keys]
            saved = []
            for entry in entries:
                saved.append(replace(entry, id=self._next_id))
                self._next_id += 1
            self._entries.extend(saved)
            return saved

    def load_entries(self):
        return list(self._entries)

    def delete_entries(self, ids) -> None:
        self._entries = [e for e in self._entries if e.id not in set(ids)]

    def clear_entries(self) -> None:
        self._entries = []

    def close(self) -> None:
        pass


class FakeIndex:
    """Plausible ranking over the fund table: brand detection, token overlap, family penalty, exact ISINs."""

    def __init__(self, table, roles) -> None:
        from semdb.core.text import tokens
        from semdb.core.types import Role

        self.table = table
        self.roles = dict(roles)
        self.deciding_columns = [c for c in table.columns if roles.get(c) == Role.DECIDING]
        self.search_columns = [c for c in table.columns if roles.get(c) == Role.SEARCH]
        self.primary_column = self.search_columns[0] if self.search_columns else table.columns[0]
        self.has_semantic = True
        name_col, brand_col, isin_col = (table.columns.index(c) for c in ("Fund name", "Brokerage", "ISIN"))
        alias_words = set(ALIASES)
        self._tokens = [set(tokens(row[name_col])) - alias_words for row in table.rows]
        self._brands = [row[brand_col] for row in table.rows]
        self._isins = {row[isin_col].lower(): i for i, row in enumerate(table.rows)}

    @staticmethod
    def _weight(token: str) -> float:
        return 0.2 if token in GENERIC else 1.0

    def search_many(self, queries, k=50, progress=None, cancel=None):
        from semdb.core.types import Cancelled

        outcomes = []
        for n, query in enumerate(queries, start=1):
            if cancel and cancel():
                raise Cancelled()
            outcomes.append(self.search(query, k))
            if progress:
                progress(f"Searching {n} of {len(queries)}…", n / len(queries))
        return outcomes

    def search(self, query, k=50):
        from semdb.core.text import remove_spans, token_spans, tokens
        from semdb.core.types import FamilyEvidence, Match, QueryOutcome

        evidence, spans = [], []
        for token, start, end in token_spans(query):
            if token in ALIASES:
                brand, strength = ALIASES[token]
                if all(e.value != brand for e in evidence):
                    evidence.append(FamilyEvidence("Brokerage", brand, strength, query[start:end]))
                spans.append((start, end))
        cleaned = remove_spans(query, spans)
        wanted = set(tokens(cleaned))
        exact = next((self._isins[t] for t in tokens(query) if t in self._isins), None)
        wanted_weight = sum(self._weight(t) for t in wanted) or 1.0
        scored = []
        for row, row_tokens in enumerate(self._tokens):
            common = wanted & row_tokens
            if not common and row != exact:
                continue
            recall = sum(self._weight(t) for t in common) / wanted_weight
            precision = sum(self._weight(t) for t in common) / (sum(self._weight(t) for t in row_tokens) or 1.0)
            text = 0.32 + 0.66 * (0.74 * recall + 0.26 * precision) if common else 0.0
            if common and row < len(KNOWN_FUNDS):
                text = min(0.99, text + 0.02)  # flagship share classes first among equally worded names
            exact_id = None
            if row == exact:
                text, exact_id = 1.0, "ISIN"
            factor, agreements, conflicts = 1.0, [], []
            if evidence:
                top = evidence[0]
                if self._brands[row] == top.value:
                    agreements.append("Brokerage")
                else:
                    factor = round(1 - 0.85 * top.strength, 3)
                    conflicts.append("Brokerage")
            scored.append((text * factor, row, text, factor, exact_id, agreements, conflicts))
        scored.sort(key=lambda item: (-item[0], item[1]))
        matches = []
        for score, row, text, factor, exact_id, agreements, conflicts in scored[:k]:
            jitter = (zlib.crc32(f"{query}|{row}".encode()) % 1000) / 1000 - 0.5
            semantic = max(0.0, min(1.0, text + 0.015 + 0.05 * jitter))
            lexical = max(0.0, min(1.0, text - 0.04 - 0.06 * jitter))
            matches.append(Match(row, score, text, semantic, lexical, factor, exact_id, agreements, conflicts))
        return QueryOutcome(query, cleaned, evidence, matches)

    def learned_aliases(self):
        return {"Brokerage": {"BlackRock": ["ishares"], "DWS": ["xtrackers"], "State Street": ["spdr"],
                              "Amundi": ["lyxor"], "Legal & General": ["l&g"]}}

    def to_history_entry(self, outcome, created_at):
        from semdb.core.types import HistoryEntry, ResultRow

        results = [
            ResultRow(
                rank=i + 1,
                score=m.score,
                source_row=self.table.source_rows[m.row],
                cells=list(self.table.rows[m.row]),
                detail={"semantic": m.semantic, "lexical": m.lexical, "text_score": m.text_score,
                        "family_factor": m.family_factor, "exact_id": m.exact_id, "agreements": m.agreements,
                        "conflicts": m.conflicts},
            )
            for i, m in enumerate(outcome.matches)
        ]
        return HistoryEntry(None, outcome.query, created_at, self.table.label, list(self.table.columns),
                            list(self.deciding_columns), self.primary_column, list(outcome.evidence), results)


def make_fake_backend(table, *, hold_build: bool = False):
    from semdb.ui.services import Backend, ImportPreview

    class FakeBackend(Backend):
        def __init__(self) -> None:
            self.exports: list = []
            self.build_started = threading.Event()

        def sample_workbook(self) -> Path:
            return ROOT / "assets" / "sample" / "sample_funds.xlsx"

        def inspect(self, path, sheet=None, header_row=None, progress=None, cancel=None):
            profiles = fund_profiles(table)
            roles = {p.name: p.suggested_role for p in profiles}
            return ImportPreview(Path(path), ["Funds", "Notes"], sheet or "Funds", 0, table, profiles, roles)

        def profile_columns(self, tbl):
            return fund_profiles(tbl)

        def load_embedder(self):
            return object()

        def build_index(self, tbl, roles, embedder, cache, progress=None, cancel=None):
            from semdb.core.types import Cancelled

            self.build_started.set()
            if hold_build:
                while not (cancel and cancel()):
                    if progress:
                        progress(f"Understanding rows… 18,240 of {len(tbl):,}", 0.403)
                    time.sleep(0.05)
                raise Cancelled()
            return FakeIndex(tbl, roles)

        def export_history(self, path, entries, k) -> None:
            self.exports.append((Path(path), list(entries), k))

    return FakeBackend()


def history_entries(index) -> list:
    now = datetime.now(timezone.utc)
    entries = []
    outcomes = index.search_many([q for q, _ in QUERIES])
    for (query, minutes), outcome in zip(QUERIES, outcomes):
        entries.append(index.to_history_entry(outcome, (now - timedelta(minutes=minutes)).isoformat()))
    return entries


# ---------------------------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------------------------
def wait_until(predicate, timeout: float = 15.0) -> bool:
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 30)
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def settle(seconds: float = 0.35) -> None:
    wait_until(lambda: False, seconds)


def save(widget, path: Path, keep_toasts: bool = False) -> None:
    settle()
    if not keep_toasts:  # transient notifications would cover the layout under review
        from semdb.ui.widgets import Toast

        for toast in widget.findChildren(Toast):
            toast.hide()
    widget.grab().save(str(path))
    print(f"  {path}")


def show_offscreen(widget, size: tuple[int, int] | None = None) -> None:
    from PySide6.QtCore import Qt

    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    if size:
        widget.resize(*size)
    widget.show()
    settle(0.2)


def new_window(store, backend, autostart: bool = True):
    from semdb.ui.main_window import MainWindow

    window = MainWindow(store, backend, autostart=autostart)
    window.resize(*WINDOW_SIZE)
    window._restore_settings()  # recompute splitter defaults for the final size
    show_offscreen(window, WINDOW_SIZE)
    return window


def close(widget) -> None:
    widget.close()
    widget.deleteLater()
    settle(0.1)


def render_fake(out: Path, mode: str) -> None:
    from semdb.ui import theme
    from semdb.ui.columns_editor import ColumnsDialog
    from semdb.ui.import_dialog import ImportDialog
    from semdb.ui.workers import TaskRunner

    theme.manager().set_mode(mode)
    table = make_fund_table()
    roles = fund_roles()
    index = FakeIndex(table, roles)
    entries = history_entries(index)

    print(f"[{mode}] empty")
    window = new_window(MemoryStore(), make_fake_backend(table))
    wait_until(lambda: window.pages.currentWidget() is window.empty_page)
    save(window, out / f"{mode}-01-empty.png")
    close(window)

    print(f"[{mode}] ready")
    store = MemoryStore()
    store.save_dataset(table, roles)
    window = new_window(store, make_fake_backend(table))
    wait_until(lambda: window.pages.currentWidget() is window.ready_page)
    save(window, out / f"{mode}-02-ready.png")
    close(window)

    print(f"[{mode}] results")
    store = MemoryStore()
    store.save_dataset(table, roles)
    store.add_entries(entries)
    window = new_window(store, make_fake_backend(table))
    wait_until(lambda: window.pages.currentWidget() is window.view and window._index is not None)
    window.view.clearSelection()
    window.details.show_empty()
    window.composer.set_text("Lyxor Core MSCI World\nvanguard s&p 500 acc\nIE00B5BMR087")
    save(window, out / f"{mode}-03-results.png")

    print(f"[{mode}] expanded + details")
    window.composer.clear()
    key = window.model.key_of(window.model.visible_entries()[0])
    window.view.reveal(key, expand=True)
    parent = window.model.query_index(key)
    window.view.setCurrentIndex(window.model.index(0, 0, parent))
    save(window, out / f"{mode}-04-expanded.png")

    print(f"[{mode}] query details")
    second = window.model.key_of(window.model.visible_entries()[2])
    window.view.reveal(second, expand=False)
    save(window, out / f"{mode}-07-query-details.png")
    close(window)

    print(f"[{mode}] import dialog")
    runner = TaskRunner()
    dialog = ImportDialog(ROOT / "funds_universe.xlsx", make_fake_backend(table), runner)
    show_offscreen(dialog, (960, 780))
    wait_until(lambda: dialog.import_button.isEnabled())
    save(dialog, out / f"{mode}-05-import.png")
    close(dialog)

    print(f"[{mode}] indexing")
    store = MemoryStore()
    store.save_dataset(table, roles)
    backend = make_fake_backend(table, hold_build=True)
    window = new_window(store, backend)
    wait_until(lambda: window.indexing_page.bar.value() not in (None, 0.0), 10)
    settle(0.3)
    save(window, out / f"{mode}-06-indexing.png")
    window.cancel_build()
    window.runner.wait(3000)
    close(window)

    print(f"[{mode}] columns dialog")
    dialog = ColumnsDialog(None, table, roles, make_fake_backend(table), runner, index.learned_aliases())
    show_offscreen(dialog, (860, 660))
    save(dialog, out / f"{mode}-08-columns.png")
    close(dialog)
    runner.wait(3000)


def render_real(out: Path, mode: str) -> None:
    from semdb.core.types import Role  # noqa: F401 - imported to fail early if the core is unavailable
    from semdb.storage import Store
    from semdb.ui import theme
    from semdb.ui.import_dialog import ImportDialog
    from semdb.ui.services import Backend
    from semdb.ui.workers import TaskRunner
    from semdb import paths

    theme.manager().set_mode(mode)
    backend = Backend()
    sample = backend.sample_workbook()
    if not sample.is_file():
        raise SystemExit(f"Sample workbook missing: {sample}")
    store = Store(paths.data_dir())
    try:
        print(f"[{mode}] empty")
        window = new_window(store, backend)
        wait_until(lambda: window.pages.currentWidget() is window.empty_page, 30)
        save(window, out / f"{mode}-real-01-empty.png")

        print(f"[{mode}] import dialog")
        runner = TaskRunner()
        dialog = ImportDialog(sample, backend, runner)
        show_offscreen(dialog, (960, 780))
        wait_until(lambda: dialog.import_button.isEnabled(), 60)
        save(dialog, out / f"{mode}-real-05-import.png")
        table, roles = dialog._preview.table, dialog.editor.roles()
        close(dialog)

        print(f"[{mode}] indexing")
        window.use_dataset(table, roles)
        wait_until(lambda: window.indexing_page.bar.value() not in (None, 0.0), 60)
        save(window, out / f"{mode}-real-06-indexing.png")

        print(f"[{mode}] ready")
        wait_until(lambda: window._index is not None, 600)
        save(window, out / f"{mode}-real-02-ready.png")

        print(f"[{mode}] results")
        window.run_search([q for q, _ in QUERIES])
        wait_until(lambda: not window.is_searching(), 120)
        window.view.collapse_all_lists()
        window.view.clearSelection()
        save(window, out / f"{mode}-real-03-results.png")

        print(f"[{mode}] expanded + details")
        key = window.model.key_of(window.model.visible_entries()[0])
        window.view.reveal(key, expand=True)
        window.view.setCurrentIndex(window.model.index(0, 0, window.model.query_index(key)))
        save(window, out / f"{mode}-real-04-expanded.png")
        close(window)
        runner.wait(3000)
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--real", action="store_true", help="use the real store, importer and engine")
    parser.add_argument("--scale", type=float, default=1.0, help="device pixel ratio of the images")
    parser.add_argument("--themes", default="light,dark")
    args = parser.parse_args(argv)

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    if args.scale != 1.0:
        os.environ["QT_SCALE_FACTOR"] = str(args.scale)
    os.environ.setdefault("QT_QPA_PLATFORM", "windows" if sys.platform == "win32" else "offscreen")
    os.environ["SEMDB_DATA_DIR"] = tempfile.mkdtemp(prefix="semdb-screens-")
    args.out.mkdir(parents=True, exist_ok=True)

    from PySide6.QtWidgets import QApplication

    from semdb import APP_NAME
    from semdb.ui import theme

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName(APP_NAME)
    theme.manager().install(app, "light")
    for mode in [m.strip() for m in args.themes.split(",") if m.strip()]:
        (render_real if args.real else render_fake)(args.out, mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
