"""Regression tests for main-window defects found in QA (ARCHITECTURE.md section 12): the details
panel's Re-run button following the index state, text typed into the composer during a search,
indexing progress that only moves forward, the details panel's result count and "Meaning" meter,
closing the window during a build, and the endless fund generator of tools/ui_screenshots.py.

The window runs against the real store and the real lexical engine (no language model); fakes
only hold a build or a search open, or replay the engine's progress.  Helpers follow
test_ui_smoke.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QPlainTextEdit, QPushButton, QWidget

from semdb.core.engine import SearchIndex
from semdb.core.types import CancelFn, Cancelled, HistoryEntry, ProgressFn, ResultRow, Role, Roles, Table
from semdb.storage import Store
from semdb.ui import theme
from semdb.ui.main_window import MainWindow
from semdb.ui.services import Backend
from semdb.ui.workers import Task

ROOT = Path(__file__).resolve().parents[1]
OpenWindow = Callable[..., MainWindow]


def wait_until(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    """Process Qt events (worker results arrive as queued signals) until ``predicate()`` holds."""
    deadline = time.monotonic() + timeout
    while True:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.002)


def labels(widget: QWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel)]


class LexicalBackend(Backend):
    """The real backend without the language model (``embedder`` pretends there is one).
    While ``release`` is clear, index builds wait (still honouring Cancel)."""

    def __init__(self, embedder: object | None = None) -> None:
        self.embedder = embedder
        self.release = threading.Event()
        self.release.set()

    def load_embedder(self) -> object | None:
        return self.embedder

    def build_index(self, table: Table, roles: Roles, embedder, cache, progress: ProgressFn | None = None,
                    cancel: CancelFn | None = None):
        while not self.release.wait(0.005):
            if cancel and cancel():
                raise Cancelled()
        return super().build_index(table, roles, None, None, progress, cancel)


class SlowIndex:
    """A ready index whose searches wait until ``release`` is set (a long search)."""

    def __init__(self, index: SearchIndex) -> None:
        self._index = index
        self.release = threading.Event()

    def __getattr__(self, name: str) -> object:
        return getattr(self._index, name)

    def search_many(self, queries: list[str], k: int = 50, progress: ProgressFn | None = None,
                    cancel: CancelFn | None = None):
        while not self.release.wait(0.005):
            if cancel and cancel():
                raise Cancelled()
        return self._index.search_many(queries, k, progress, cancel)


class TinyEmbedder:
    """Deterministic stand-in for the language model: the engine only needs unit vectors."""

    model_id, dim = "tiny-test", 8

    def encode(self, texts: list[str], progress=None, cancel=None) -> np.ndarray:
        rows = [np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dim) for text in texts]
        vectors = np.asarray(rows, dtype=np.float32).reshape(len(texts), self.dim)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        if progress:
            progress(len(texts), len(texts))
        return vectors


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication(["semdb-tests"])
    theme.manager().install(app, "light")
    yield app
    theme.manager().set_mode("light")


@pytest.fixture(autouse=True)
def no_dialogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """None of these flows asks the user anything: a modal dialog would be a defect."""
    opened: list[str] = []

    def refuse(dialog: QDialog, *_args: object) -> object:
        opened.append(type(dialog).__name__)
        return QDialog.DialogCode.Rejected

    def no_file(*_args: object, **_kwargs: object) -> tuple[str, str]:
        opened.append("file dialog")
        return "", ""  # as if the user pressed Cancel

    monkeypatch.setattr(QDialog, "exec", refuse)
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(no_file))
    yield
    assert opened == [], "a dialog was opened"


@pytest.fixture
def store(isolated_data_dir: Path) -> Iterator[Store]:
    with Store(isolated_data_dir) as opened:
        yield opened


@pytest.fixture
def open_window(qapp: QApplication, store: Store) -> Iterator[OpenWindow]:
    """Opens the main window like the app does; ``wait`` waits until its startup work is done."""
    windows: list[MainWindow] = []

    def open_(backend: Backend | None = None, *, wait: bool = True) -> MainWindow:
        window = MainWindow(store, backend or LexicalBackend(), autostart=False)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.resize(1320, 860)
        window.show()
        windows.append(window)
        window.start()
        if wait:
            assert wait_until(lambda: not window.runner.active()), "startup work did not finish"
        return window

    yield open_
    for window in windows:
        window.close()  # cancels held builds and searches, then waits for them
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def make_entry(table: Table, query: str, rows: list[int], semantic: list[float] | None = None,
               created_at: str = "2026-11-01T09:00:00+00:00") -> HistoryEntry:
    """A stored result list over ``rows``; ``semantic`` sets each result's meaning score."""
    results = []
    for rank, row in enumerate(rows, start=1):
        score = round(0.97 - 0.06 * rank, 2)
        meaning = score if semantic is None else semantic[rank - 1]
        detail = {"semantic": meaning, "lexical": score, "text_score": score, "family_factor": 1.0,
                  "exact_id": None, "agreements": [], "conflicts": []}  # fmt: skip
        results.append(ResultRow(rank, score, table.source_rows[row], list(table.rows[row]), detail))
    return HistoryEntry(None, query, created_at, table.label, list(table.columns), ["ISIN", "Brokerage"],
                        "Fund name", [], results)  # fmt: skip


def select_result(window: MainWindow, query: str, rank: int) -> None:
    model = window.model
    entry = next(entry for entry in model.visible_entries() if entry.query == query)
    parent = model.query_index(model.key_of(entry))
    window.view.setExpanded(parent, True)
    window.view.setCurrentIndex(model.index(rank - 1, 0, parent))
    assert window.details.current()[1] is entry.results[rank - 1]


def heading(window: MainWindow) -> str:
    return next(text for text in labels(window.details) if text.startswith("Result "))


def meter(window: MainWindow, name: str) -> str:
    """The number shown next to the details panel's ``name`` bar."""
    texts = labels(window.details)
    return texts[texts.index(name) + 1]


def rerun_button(window: MainWindow) -> QPushButton:
    (button,) = [b for b in window.details.findChildren(QPushButton) if b.text() == "Re-run"]
    return button


# --- defect 5: the Re-run button follows the index -------------------------------------------------


def test_rerun_button_follows_the_index_state(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles
) -> None:
    store.save_dataset(funds_table, funds_roles)
    store.add_entries([make_entry(funds_table, "Xtrackers DAX", [11, 10])])
    backend = LexicalBackend()
    backend.release.clear()  # the restart's index build is still running...
    window = open_window(backend, wait=False)
    assert wait_until(lambda: window.entries() and window._build_state == "building")
    window.view.reveal(window.model.key_of(window.entries()[0]))  # ...when the user selects a list
    assert window.details.current() == (window.entries()[0], None)
    assert not rerun_button(window).isEnabled()
    assert rerun_button(window).toolTip() == "Preparing the search index…"

    backend.release.set()
    assert wait_until(lambda: window._index is not None)
    assert rerun_button(window).isEnabled(), "Re-run stayed disabled although the index is ready"
    assert rerun_button(window).toolTip() == ""

    backend.release.clear()
    window.apply_roles({**funds_roles, "Currency": Role.SEARCH})  # Columns… > Apply rebuilds the index
    assert window.is_building()
    assert not rerun_button(window).isEnabled(), "Re-run looked usable during the rebuild"

    backend.release.set()
    assert wait_until(lambda: window._index is not None and not window.runner.active())
    rerun_button(window).click()
    assert window.is_searching()  # enabled means it really searches
    assert not rerun_button(window).isEnabled() and rerun_button(window).toolTip() == "A search is running"
    assert wait_until(lambda: not window.is_searching())
    assert rerun_button(window).isEnabled()


# --- defect 9: what the user types during a search survives it -------------------------------------

SUBMITTED = "iShares Core S&P 500\nXtrackers DAX"


def typing(text: str, position: int | None = None) -> Callable[[QPlainTextEdit], None]:
    """Types ``text`` at ``position`` (default: the end) key by key, like the user."""

    def edit(editor: QPlainTextEdit) -> None:
        cursor = editor.textCursor()
        if position is None:
            cursor.movePosition(QTextCursor.MoveOperation.End)
        else:
            cursor.setPosition(position)
        editor.setTextCursor(cursor)
        for number, line in enumerate(text.split("\n")):
            if number:
                QTest.keyClick(editor, Qt.Key.Key_Return)
            QTest.keyClicks(editor, line)

    return edit


@pytest.mark.parametrize(
    ("edit", "expected", "cursor"),
    [
        (lambda editor: None, "", 0),
        (typing("\nVanguard FTSE All-World"), "Vanguard FTSE All-World", 23),
        (typing("Vanguard FTSE All-World\n", 0), "Vanguard FTSE All-World", 23),
        (typing(" UCITS", len("iShares Core S&P 500")), "iShares Core S&P 500 UCITS\nXtrackers DAX", 26),
    ],
    ids=["untouched: cleared", "line added below", "line added above", "searched line edited: kept"],
)
def test_text_typed_during_a_search_is_kept(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles,
    edit: Callable[[QPlainTextEdit], None], expected: str, cursor: int,
) -> None:  # fmt: skip
    store.save_dataset(funds_table, funds_roles)
    window = open_window()
    slow = SlowIndex(window._index)
    window.set_index(slow)
    composer = window.composer
    composer.set_text(SUBMITTED)
    composer.search_button.click()
    assert window.is_searching()
    edit(composer.editor)
    slow.release.set()
    assert wait_until(lambda: not window.is_searching())
    assert sorted(entry.query for entry in window.entries()) == ["Xtrackers DAX", "iShares Core S&P 500"]
    assert composer.editor.toPlainText() == expected
    assert composer.editor.textCursor().position() == cursor  # the user keeps typing where they were


# --- defect 11: indexing progress only moves forward -----------------------------------------------


class ReplayBackend(LexicalBackend):
    """Reports ``script`` through the real worker (every report delivered), then builds silently."""

    def __init__(self, script: list[tuple[str, float]], embedder: object | None) -> None:
        super().__init__(embedder)
        self.script = script

    def build_index(self, table: Table, roles: Roles, embedder, cache, progress: ProgressFn | None = None,
                    cancel: CancelFn | None = None):
        for stage, fraction in self.script:
            progress(stage, fraction)
        return Backend.build_index(self, table, roles, None, None)


@pytest.mark.parametrize("semantic", [True, False], ids=["with the language model", "lexical only"])
def test_indexing_progress_only_moves_forward(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles,
    monkeypatch: pytest.MonkeyPatch, semantic: bool,
) -> None:  # fmt: skip
    embedder = TinyEmbedder() if semantic else None
    script: list[tuple[str, float]] = []  # the real engine's reports: each stage restarts at 0
    SearchIndex.build(funds_table, funds_roles, embedder, progress=lambda stage, f: script.append((stage, f)))
    monkeypatch.setattr(Task, "PROGRESS_INTERVAL", 0.0)
    store.save_dataset(funds_table, funds_roles)
    window = open_window(ReplayBackend(script, embedder), wait=False)
    shown: list[tuple[str, str, float | None, float | None]] = []
    on_progress = window._on_build_progress

    def record(generation: int, stage: str, fraction: float) -> None:
        on_progress(generation, stage, fraction)
        chip, card = window.dataset_chip, window.indexing_page
        shown.append((stage, chip.state()[1], chip._progress, card.bar.value()))

    window._on_build_progress = record
    assert wait_until(lambda: window._index is not None)
    assert [stage for stage, *_ in shown] == [stage for stage, _ in script]
    percents = [int(text.removeprefix("Indexing ").removesuffix("%")) for _, text, _, _ in shown]
    bars = [bar for *_, bar in shown]
    assert percents == sorted(percents), f"the chip went backwards: {percents}"
    assert bars == sorted(bars), f"the progress card went backwards: {bars}"
    assert all(chip_bar == bar for _, _, chip_bar, bar in shown), "the chip's bar and the card disagree"
    assert all(percent == int(bar * 100 + 1e-6) for percent, bar in zip(percents, bars)), "chip text vs card"
    assert percents[-1] == 100 and percents[0] == 0
    embedding = [bar for stage, *_, bar in shown if stage.startswith("Understanding rows")]
    assert (embedding[0] < 0.25 and embedding[-1] > 0.9) if semantic else not embedding  # embedding dominates


# --- defect 12: what the details panel says about a result -----------------------------------------


def test_details_count_only_the_results_a_list_has(open_window: OpenWindow, store: Store, funds_table: Table) -> None:
    store.add_entries([
        make_entry(funds_table, "Xtrackers DAX", [11], created_at="2026-10-31T09:00:00+00:00"),
        make_entry(funds_table, "iShares Core S&P 500", [0, 1, 4, 5, 8, 12, 2, 3]),
    ])  # fmt: skip
    window = open_window()
    assert window.model.k == 5
    select_result(window, "Xtrackers DAX", 1)
    assert heading(window) == "Result 1 of 1  ·  Sheet row 13", "a one-result list is not one of 5"
    select_result(window, "iShares Core S&P 500", 2)
    assert heading(window) == "Result 2 of 5  ·  Sheet row 3"
    window.k_stepper.setValue(1)
    select_result(window, "iShares Core S&P 500", 1)
    assert heading(window) == "Result 1 of 1  ·  Sheet row 2"


def test_meaning_is_not_shown_when_semantic_search_was_off(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles
) -> None:
    store.save_dataset(funds_table, funds_roles)
    window = open_window()  # no language model: a lexical index
    assert window._index is not None and not window._index.has_semantic
    window.run_search(["iShares Core S&P 500"])
    assert wait_until(lambda: not window.is_searching())
    select_result(window, "iShares Core S&P 500", 1)
    assert meter(window, "Meaning") == "—", "meaning was not compared, it is not 0 %"
    assert meter(window, "Exact words") not in ("—", "0%")

    searched_with_model = make_entry(funds_table, "Vanguard FTSE All-World", [6, 4], semantic=[0.91, 0.0])
    window.set_entries([*window.entries(), searched_with_model])
    select_result(window, "Vanguard FTSE All-World", 1)
    assert meter(window, "Meaning") == "91%"
    select_result(window, "Vanguard FTSE All-World", 2)
    assert meter(window, "Meaning") == "0%"  # compared, and nothing in common


# --- closing: background work is told to stop before anything can block ----------------------------


def test_closing_cancels_the_build_before_saving_the_window_state(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.save_dataset(funds_table, funds_roles)
    backend = LexicalBackend()
    backend.release.clear()
    window = open_window(backend, wait=False)
    assert wait_until(lambda: window._build_state == "building")
    build, save = window._build_task, store.set_setting
    cancelled_at_save: list[bool] = []

    def set_setting(key: str, value: object) -> None:  # waits for the store lock a big save_dataset holds
        cancelled_at_save.append(build.is_cancelled())
        save(key, value)

    monkeypatch.setattr(store, "set_setting", set_setting)
    window.close()
    assert cancelled_at_save and all(cancelled_at_save), "the build kept running while the window state was saved"
    assert wait_until(lambda: not window.runner.active(), 1.0)


# --- defect 13: the screenshot tool's data generator finishes --------------------------------------


def test_screenshot_tool_generates_its_whole_fund_universe() -> None:
    code = "\n".join([
        "import json, sys",
        f"sys.path.insert(0, {str(ROOT / 'tools')!r})",
        "import ui_screenshots as tool",
        "table = tool.make_fund_table()",
        "copies = {}",
        "for row in table.rows:",
        "    copies[row[1]] = copies.get(row[1], 0) + 1",
        "print(json.dumps({'rows': len(table), 'isins': len({row[0] for row in table.rows}),",
        "                  'flagships': [copies[row[1]] for row in tool.KNOWN_FUNDS],",
        "                  'most_copies': max(copies.values()), 'brands': len({row[2] for row in table.rows})}))",
    ])  # fmt: skip
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        pytest.fail("make_fund_table() was still generating rows after 60 s")
    assert done.returncode == 0, done.stderr
    stats = json.loads(done.stdout)
    assert stats["rows"] == stats["isins"] == 45_210
    assert stats["flagships"] == [1] * len(stats["flagships"])  # the brief's real funds stay unique
    assert stats["most_copies"] <= 3 and stats["brands"] == 15
