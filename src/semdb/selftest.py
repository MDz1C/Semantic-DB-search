"""Headless end-to-end self-test: ``SemanticDBSearch.exe --selftest [--data-dir DIR] [--no-model]``.

Exercises the packaged app without opening a window: loads the bundled embedding model,
imports the bundled sample workbook, suggests column roles, saves and reloads the dataset,
builds the index twice (the second time from the on-disk embedding cache), runs known
queries, round-trips the results through the history store and exports them to Excel.  A
Python audit hook records any attempt to reach the network.  Qt is never imported.

The report goes to ``<data dir>/logs/selftest.log`` and to stdout when there is one (a frozen
windowed app has none).  ``--data-dir`` names a scratch folder for the test; without it a
throw-away folder is used and only its log is kept.  Exit status: 0 when every check passes,
1 when one fails, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO

from .core.types import QueryOutcome, Role, Roles, Table

QUERIES = ("iShares Core S&P 500", "Vanguard FTSE All-World")  # plus one ISIN taken from the data
RESULTS_PER_QUERY = 10
EXPORT_K = 5
# A Spanish query, its English paraphrase and an unrelated sentence: the model must rank the pair.
PROBES = ("fondo de renta variable europea", "European equity fund", "short-term US Treasury bonds")
NETWORK_PACKAGES = ("huggingface_hub", "hf_xet", "httpx", "httpx2", "httpcore", "httpcore2", "requests", "urllib3")
NETWORK_EVENTS = frozenset({
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg", "socket.getaddrinfo",
    "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr", "socket.getnameinfo",
    "urllib.Request", "http.client.connect", "ftplib.connect", "smtplib.connect", "poplib.connect",
    "imaplib.open", "webbrowser.open",
})


class _Abort(Exception):
    """A prerequisite failed, so the remaining steps cannot run."""


class _Report:
    """PASS/FAIL lines written to the log file (flushed per line) and echoed to stdout."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()
        self.passed = 0
        self.failed = 0

    def line(self, text: str = "") -> None:
        with self._lock:
            self._stream.write(text + "\n")
            self._stream.flush()
            _echo(text)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        self.line(f"{'PASS' if ok else 'FAIL'}  {name:<44} {detail}".rstrip())
        return ok

    def require(self, name: str, ok: bool, detail: str = "") -> None:
        if not self.check(name, ok, detail):
            raise _Abort()

    def detail(self, text: str) -> None:
        self.line(f"      {text}")


class _ReportHandler(logging.Handler):
    """Copies log records (e.g. why the model could not be loaded) into the report."""

    def __init__(self, report: _Report) -> None:
        super().__init__(logging.INFO)
        self.setFormatter(logging.Formatter("log %(levelname)s %(name)s: %(message)s"))
        self._report = report

    def emit(self, record: logging.LogRecord) -> None:
        try:
            for text in self.format(record).splitlines():
                self._report.detail(text)
        except Exception:
            self.handleError(record)


class _NetworkAudit:
    """Records socket and HTTP audit events while active.  Audit hooks cannot be removed, so a
    single hook is installed per process and goes quiet when the self-test ends."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.active = False
        self._installed = False

    def start(self) -> None:
        self.events.clear()
        self.active = True
        if not self._installed:
            sys.addaudithook(self._hook)
            self._installed = True

    def stop(self) -> None:
        self.active = False

    def _hook(self, event: str, args: tuple[Any, ...]) -> None:
        if self.active and event in NETWORK_EVENTS:
            shown = ", ".join(repr(arg) for arg in args if isinstance(arg, (str, int, tuple)))
            self.events.append(f"{event}({shown})"[:200])


_AUDIT = _NetworkAudit()


class _CountingEmbedder:
    """Wraps the real embedder and counts the texts it is asked to embed."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_id = inner.model_id
        self.dim = inner.dim
        self.texts = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def encode(self, texts: list[str], progress: Any = None, cancel: Any = None) -> Any:
        self.texts += len(texts)
        return self._inner.encode(texts, progress=progress, cancel=cancel)


@dataclass
class _Run:
    """State shared by the steps."""

    data_dir: Path
    use_model: bool
    embedder: _CountingEmbedder | None = None
    store: Any = None
    table: Table | None = None
    roles: Roles = field(default_factory=dict)
    isin_column: str = ""
    broker_column: str = ""
    index: Any = None
    queries: list[str] = field(default_factory=list)
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def reopen_store(self) -> None:
        """Close and reopen the database, as quitting and restarting the app would."""
        from .storage import Store

        if self.store is not None:
            self.store.close()
        self.store = Store(self.data_dir)

    def cell(self, row: int, column: str) -> str:
        return self.table.rows[row][self.table.column_index(column)]


def run_selftest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="SemanticDBSearch --selftest", add_help=False, exit_on_error=False)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--no-model", action="store_true")
    try:
        options, _ = parser.parse_known_args(argv)
    except (argparse.ArgumentError, SystemExit):
        _echo("usage: SemanticDBSearch --selftest [--data-dir DIR] [--no-model]")
        return 2

    temporary = options.data_dir is None
    if temporary:
        data_dir = Path(tempfile.mkdtemp(prefix="SemanticDBSearch-selftest-"))
    else:
        data_dir = options.data_dir.resolve()
    previous_data_dir = os.environ.get("SEMDB_DATA_DIR")
    os.environ["SEMDB_DATA_DIR"] = str(data_dir)
    # Defensive, as in the app: libraries that could phone home are told not to.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        from . import paths

        try:
            log_path = paths.logs_dir() / "selftest.log"
            stream = open(log_path, "w", encoding="utf-8", newline="\n")
        except OSError as exc:
            _echo(f"Self-test could not write its report in {data_dir}: {exc}")
            return 1
        with stream:
            report = _Report(stream)
            try:
                _run_with_diagnostics(report, _Run(data_dir=data_dir, use_model=not options.no_model), stream)
            except Exception:  # never let a self-test bug end in a modal error dialog of the windowed exe
                report.check("Self-test completes", False, "crashed:")
                _traceback_to(report)
        _echo(f"Report written to {log_path}")
        return 0 if report.failed == 0 else 1
    finally:
        if previous_data_dir is None:
            os.environ.pop("SEMDB_DATA_DIR", None)
        else:
            os.environ["SEMDB_DATA_DIR"] = previous_data_dir
        if temporary:
            _discard_all_but_logs(data_dir)


def _run_with_diagnostics(report: _Report, run: _Run, stream: TextIO) -> None:
    """Run the steps with log capture, a crash dump to the log and the network audit."""
    started = time.perf_counter()
    modules_before = set(sys.modules)
    handler = _ReportHandler(report)
    root, ours = logging.getLogger(), logging.getLogger("semdb")
    previous_level = ours.level
    root.addHandler(handler)
    ours.setLevel(logging.INFO)
    own_faulthandler = not faulthandler.is_enabled()
    if own_faulthandler:
        faulthandler.enable(file=stream)  # a native crash (e.g. in onnxruntime) still leaves a trace
    _AUDIT.start()
    try:
        _header(report, run)
        for title, step in STEPS:
            try:
                step(report, run)
            except _Abort:
                report.detail("Remaining steps skipped.")
                break
            except Exception as exc:
                report.check(f"{title} step", False, f"raised {type(exc).__name__}: {exc}")
                _traceback_to(report)
                break
    finally:
        _AUDIT.stop()
        if run.store is not None:
            try:
                run.store.close()
            except Exception as exc:
                report.check("Database closes", False, f"{type(exc).__name__}: {exc}")
        if own_faulthandler:
            faulthandler.disable()
        ours.setLevel(previous_level)
        root.removeHandler(handler)
    _check_isolation(report, set(sys.modules) - modules_before)
    total = report.passed + report.failed
    report.line()
    if report.failed:
        report.line(f"Result: FAILED - {report.failed} of {total} checks failed ({_seconds(started)})")
    else:
        report.line(f"Result: PASSED - all {total} checks passed ({_seconds(started)})")


def _header(report: _Report, run: _Run) -> None:
    from . import APP_NAME, __version__, paths

    report.line(f"{APP_NAME} {__version__} self-test, {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
    build = "frozen build" if paths.is_frozen() else "source checkout"
    report.line(f"Python {platform.python_version()} ({build}) on {platform.platform()}, {os.cpu_count()} CPUs")
    report.line(f"Resources: {paths.resource_root()}")
    report.line(f"Data:      {run.data_dir}")
    report.line()


def _step_model(report: _Report, run: _Run) -> None:
    """The bundled model loads and embeds a Spanish/English paraphrase closer than an unrelated text."""
    import numpy as np

    from . import paths
    from .core.embedder import load_default_embedder

    if not run.use_model:
        report.line(f"SKIP  {'Embedding model loads':<44} --no-model: lexical-only search")
        return
    started = time.perf_counter()
    embedder = load_default_embedder()
    if embedder is None:
        report.check("Embedding model loads", False, f"nothing usable in {paths.model_dir()}")
        return  # keep going lexical-only: the remaining checks still say a lot
    report.check("Embedding model loads", True, f"{embedder.model_id}, {embedder.dim}-d, {_seconds(started)}")
    started = time.perf_counter()
    vectors = embedder.encode(list(PROBES))
    well_formed = (
        vectors.shape == (len(PROBES), embedder.dim)
        and vectors.dtype == np.float32
        and bool(np.isfinite(vectors).all())
        and bool(np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3))
    )
    paraphrase, unrelated = float(vectors[0] @ vectors[1]), float(vectors[0] @ vectors[2])
    report.check(
        "Embeddings understand Spanish/English",
        well_formed and paraphrase > unrelated + 0.15,
        f"paraphrase {paraphrase:.2f} vs unrelated {unrelated:.2f}, {_seconds(started)}",
    )
    run.embedder = _CountingEmbedder(embedder)


def _step_import(report: _Report, run: _Run) -> None:
    """The bundled sample workbook imports through the normal reader."""
    from . import paths
    from .core.table import read_table

    sample = paths.sample_workbook()
    report.require("Sample workbook is bundled", sample.is_file(), str(sample))
    started = time.perf_counter()
    run.table = table = read_table(sample)
    report.require(
        "Sample workbook imports",
        len(table) > 0 and len(table.columns) >= 2,
        f"{len(table):,} rows x {len(table.columns)} columns from {table.label}, {_seconds(started)}",
    )


def _step_roles(report: _Report, run: _Run) -> None:
    """Profiling suggests roles; ISIN and Brokerage are then ticked as Deciding."""
    from .core.profile import profile_columns, suggest_roles

    table = run.table
    started = time.perf_counter()
    profiles = profile_columns(table)
    suggested = suggest_roles(table, profiles)
    report.check(
        "Column roles are suggested",
        len(profiles) == len(table.columns) and set(suggested) == set(table.columns)
        and Role.SEARCH in suggested.values(),
        f"{_seconds(started)}: " + ", ".join(f"{name} = {role.label}" for name, role in suggested.items()),
    )
    run.isin_column = _find_column(table, "isin")
    run.broker_column = _find_column(table, "brokerage", "broker")
    report.require(
        "Sample has ISIN and Brokerage columns",
        bool(run.isin_column and run.broker_column),
        f"columns: {', '.join(table.columns)}",
    )
    identity = (run.isin_column, run.broker_column)
    report.check(
        "ISIN and Brokerage suggested as Deciding",
        all(suggested.get(name) is Role.DECIDING for name in identity),
        ", ".join(f"{name} = {getattr(suggested.get(name), 'label', 'no role')}" for name in identity),
    )
    # The search checks below test ranking, not the heuristics: tick both as a user would.
    run.roles = {**suggested, **{name: Role.DECIDING for name in identity}}


def _step_persistence(report: _Report, run: _Run) -> None:
    """The imported dataset and its roles survive closing and reopening the database."""
    run.reopen_store()
    started = time.perf_counter()
    run.store.save_dataset(run.table, run.roles)
    run.reopen_store()
    loaded = run.store.load_dataset()
    same = loaded is not None and loaded[0].fingerprint() == run.table.fingerprint() and loaded[1] == run.roles
    report.require("Dataset survives a restart", same, f"saved and reloaded in {_seconds(started)}")
    run.table, run.roles = loaded


def _step_index(report: _Report, run: _Run) -> None:
    """Build the index, then rebuild it after a restart: every row embedding must come from the cache."""
    from .core.engine import SearchIndex

    stages: list[str] = []

    def progress(stage: str, _fraction: float) -> None:
        if stage not in stages:
            stages.append(stage)

    started = time.perf_counter()
    run.index = SearchIndex.build(run.table, run.roles, run.embedder, run.store.embedding_cache, progress=progress)
    embedded = f"embedded {run.embedder.texts:,} texts" if run.embedder else "lexical only"
    report.check(
        "Index builds",
        run.index.has_semantic == (run.embedder is not None),
        f"{_seconds(started)}, {embedded}; stages: {' > '.join(stages) or 'none reported'}",
    )
    if run.embedder is None:
        return
    run.reopen_store()
    run.embedder.texts = 0
    started = time.perf_counter()
    run.index = SearchIndex.build(run.table, run.roles, run.embedder, run.store.embedding_cache)
    report.check(
        "Rebuild after a restart uses the cache",
        run.embedder.texts == 0 and run.index.has_semantic,
        f"{_seconds(started)}, re-embedded {run.embedder.texts:,} texts",
    )


def _step_search(report: _Report, run: _Run) -> None:
    """Known queries return results and rank the right identity first."""
    from .core.text import fold, normalize

    codes = [row[run.table.column_index(run.isin_column)] for row in run.table.rows]
    codes = [code for code in codes if code.strip()]
    report.require("Sample has ISIN values", bool(codes))
    isin = codes[len(codes) // 2]
    run.queries = [*QUERIES, isin]
    started = time.perf_counter()
    run.outcomes = run.index.search_many(run.queries, k=RESULTS_PER_QUERY)
    per_query = (time.perf_counter() - started) * 1000 / len(run.queries)
    report.require(
        "Every query returns results",
        len(run.outcomes) == len(run.queries) and all(outcome.matches for outcome in run.outcomes),
        f"{per_query:.0f} ms per query over {len(run.table):,} rows",
    )
    for outcome in run.outcomes:
        _describe(report, run, outcome)

    ishares, vanguard, by_code = run.outcomes
    top = run.cell(by_code.matches[0].row, run.isin_column)
    report.check("ISIN query finds that fund first", normalize(top) == normalize(isin), f"{isin} -> {top}")
    broker = run.cell(ishares.matches[0].row, run.broker_column)
    report.check('"iShares" query puts BlackRock first', fold(broker) == "blackrock", f"top match: {broker}")
    if any("vanguard" in fold(value) for value in run.table.column_values(run.broker_column)):
        broker = run.cell(vanguard.matches[0].row, run.broker_column)
        report.check('"Vanguard" query puts Vanguard first', "vanguard" in fold(broker), f"top match: {broker}")


def _step_history(report: _Report, run: _Run) -> None:
    """Results persist across a restart and export to an Excel workbook."""
    from .core.export import export_history
    from .core.table import list_sheets, read_table

    now = datetime.now(timezone.utc).isoformat()
    saved = run.store.add_entries([run.index.to_history_entry(outcome, now) for outcome in run.outcomes])
    run.reopen_store()
    entries = [entry for entry in run.store.load_entries() if entry.query in run.queries]
    report.check(
        "Results survive a restart",
        all(entry.id is not None for entry in saved)
        and sorted(entry.query for entry in entries) == sorted(run.queries)
        and all(entry.results for entry in entries),
        f"{len(entries)} result lists reloaded",
    )
    path = run.data_dir / "selftest-export.xlsx"
    started = time.perf_counter()
    export_history(path, entries, EXPORT_K)
    sheets = list_sheets(path)
    exported: set[str] = set()
    if "Summary" in sheets:
        exported = set(read_table(path, sheet="Summary", header_row=0).column_values("Query"))
    report.check(
        "Results export to Excel",
        {"Summary", "Details"} <= set(sheets) and set(run.queries) <= exported,
        f"{path.name}: sheets {', '.join(sheets)}, {path.stat().st_size:,} bytes, {_seconds(started)}",
    )


STEPS: tuple[tuple[str, Callable[[_Report, _Run], None]], ...] = (
    ("Embedding model", _step_model),
    ("Sample import", _step_import),
    ("Column roles", _step_roles),
    ("Dataset storage", _step_persistence),
    ("Index build", _step_index),
    ("Search", _step_search),
    ("History and export", _step_history),
)


def _check_isolation(report: _Report, new_modules: set[str]) -> None:
    """The self-test must stay offline (R1) and headless.  ``new_modules`` are the modules it
    imported; a frozen build preloads PySide6.QtCore in PyInstaller's runtime hook, which is fine."""
    network = sorted(
        name for name in new_modules
        if name.split(".")[0] in NETWORK_PACKAGES or name.startswith("PySide6.QtNetwork")
    )
    report.check("No HTTP or network library was loaded", not network, ", ".join(network))
    report.check(
        "No network access was attempted",
        not _AUDIT.events,
        "; ".join(_AUDIT.events[:5]) or "Python audit hook saw no socket or HTTP activity",
    )
    qt = sorted(name for name in new_modules if name.startswith("PySide6"))
    report.check("No Qt module was imported (headless)", not qt, ", ".join(qt))


def _describe(report: _Report, run: _Run, outcome: QueryOutcome) -> None:
    identity = "; ".join(
        f"{found.column} -> {found.value} (from '{found.matched_text}', {found.strength:.0%})"
        for found in outcome.evidence
    )
    report.detail(f"\"{outcome.query}\"" + (f"   identity: {identity}" if identity else ""))
    for rank, match in enumerate(outcome.matches[:3], start=1):
        title = run.cell(match.row, run.index.primary_column)
        identity_cells = ", ".join(run.cell(match.row, name) for name in run.index.deciding_columns)
        report.detail(f"   {rank}. {match.score:4.0%}  {title}   [{identity_cells}]")


def _find_column(table: Table, *names: str) -> str:
    from .core.text import normalize

    for wanted in names:
        for column in table.columns:
            if normalize(column) == wanted:
                return column
    return ""


def _traceback_to(report: _Report) -> None:
    for text in traceback.format_exc().rstrip().splitlines():
        report.detail(text)


def _seconds(started: float) -> str:
    return f"{time.perf_counter() - started:.1f} s"


def _echo(text: str) -> None:
    """Print when there is a console; frozen windowed apps have ``sys.stdout = None``."""
    stream = sys.stdout
    if stream is None:
        return
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        stream.write(text.encode(encoding, "replace").decode(encoding) + "\n")
        stream.flush()
    except (OSError, ValueError, LookupError):
        pass


def _discard_all_but_logs(data_dir: Path) -> None:
    for child in data_dir.iterdir():
        if child.name == "logs":
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass  # a file still locked by another process; it is in the temp folder anyway
