"""The results tree stays responsive with many long lists (ARCHITECTURE.md section 12.3).

A full layout of the tree used to ask Python for every cell of every visible row, and expanding
or collapsing every list reported each list separately to the window.  These tests time the
operations on 153 lists of 50 results (the bounds are generous; the timings before the fix were
1-2 s) and check, independently of machine load, how many calls and signals they make.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PySide6.QtCore import (
    QCoreApplication,
    QEvent,
    QEventLoop,
    QModelIndex,
    QObject,
    Qt,
    QtMsgType,
    qInstallMessageHandler,
)
from PySide6.QtGui import QContextMenuEvent
from PySide6.QtTest import QAbstractItemModelTester
from PySide6.QtWidgets import QApplication, QMenu

from semdb.core.types import CancelFn, FamilyEvidence, HistoryEntry, ProgressFn, ResultRow, Role, Table
from semdb.storage import Store
from semdb.ui import results_view, theme
from semdb.ui.main_window import MainWindow
from semdb.ui.results_model import SORT_ALPHA, SORT_RECENT, ResultsModel, order_columns
from semdb.ui.results_view import GROUP_GAP, QUERY_ROW_H, RESULT_ROW_H, ResultsView
from semdb.ui.services import Backend

COLUMNS = ["ISIN", "Ticker", "Fund name", "Brokerage", "Category", "Asset class", "Region", "Currency",
           "Distribution", "Domicile", "TER (%)", "AUM (M EUR)", "Launch date"]  # the sample's 13 columns
ROLES = {"ISIN": Role.DECIDING, "Brokerage": Role.DECIDING, "Fund name": Role.SEARCH}
INDEXES = ("MSCI World", "S&P 500", "MSCI EM")  # a filter for "msci" keeps two lists in three
MANY, K = 153, 50  # the QA's worst case: 7,803 rows when every list is open
BOUND_MS = 300  # before the fix: expand all 1.7 s, sort 1.6 s, one filter change 1.1-1.7 s


def make_entries(lists: int, results: int = K) -> list[HistoryEntry]:
    """``lists`` result lists, oldest first, each with ``results`` rows of 13 cells."""
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    entries = []
    for i in range(lists):
        query = f"{INDEXES[i % 3]} tracker {i:03d}"
        rows = []
        for rank in range(1, results + 1):
            brokerage = "BlackRock" if rank % 3 else "Vanguard"
            cells = [f"IE{i:04d}{rank:06d}", f"T{rank}", f"{query} UCITS ETF {rank}", brokerage, "Equity", "Equity",
                     "World", "USD", "Acc", "Ireland", "0.20", "1200.5", "2015-01-01"]
            detail = {"semantic": 0.5, "lexical": 0.8, "text_score": 0.7, "family_factor": 1.0, "exact_id": None,
                      "agreements": ["Brokerage"] if brokerage == "BlackRock" else [],
                      "conflicts": [] if brokerage == "BlackRock" else ["Brokerage"]}
            rows.append(ResultRow(rank, round(0.99 - rank * 0.01, 2), rank + 1, cells, detail))
        evidence = [FamilyEvidence("Brokerage", "BlackRock", 0.95, "iShares")] if i % 2 else []
        entries.append(HistoryEntry(None, query, (start + timedelta(minutes=i)).isoformat(), "funds.xlsx · Funds",
                                    list(COLUMNS), ["ISIN", "Brokerage"], "Fund name", evidence, rows))
    return entries


def settle() -> None:
    """Run what the operation left for the event loop (delayed layouts, repaints)."""
    for _ in range(3):
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)


def best_ms(action: Callable[[], object], prepare: Callable[[], object] | None = None, repeat: int = 3) -> float:
    """Shortest of ``repeat`` timings of ``action`` plus the work it leaves for the event loop."""
    times = []
    for _ in range(repeat):
        if prepare is not None:
            prepare()
            settle()
        start = time.perf_counter()
        action()
        settle()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def lists(model: ResultsModel) -> list[QModelIndex]:
    return [model.index(row, 0) for row in range(model.rowCount())]


def open_lists(view: ResultsView) -> list[str]:
    model = view.results_model()
    return [model.key_of(model.entry_for(index)) for index in lists(model) if view.isExpanded(index)]


def list_index(model: ResultsModel, query: str) -> QModelIndex:
    entry = next(entry for entry in model.all_entries() if entry.query == query)
    return model.query_index(model.key_of(entry))


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication(["semdb-tests"])
    theme.manager().install(app, "light")
    yield app


@pytest.fixture
def tree(qapp: QApplication) -> Iterator[ResultsView]:
    """A shown results tree over its own model, sized like the window's."""
    view = ResultsView(ResultsModel())
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.resize(1000, 640)
    view.show()
    yield view
    view.close()
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def load(view: ResultsView, count: int, k: int = K) -> list[HistoryEntry]:
    model = view.results_model()
    entries = make_entries(count)
    model.set_entries(entries)
    model.set_columns(order_columns(COLUMNS, ROLES, "Fund name"))
    view.apply_column_widths()
    model.set_k(k)
    settle()
    return entries


class Calls:
    """Counts the calls the tree makes into its delegate and model.  The counters are instance
    attributes: Qt's virtual calls see those, while PySide caches class methods per instance."""

    def __init__(self, view: ResultsView) -> None:
        self.counts: Counter[str] = Counter()
        model = view.results_model()
        for owner, name in ((view.itemDelegate(), "sizeHint"), (model, "index"), (model, "rowCount"),
                            (model, "parent")):
            setattr(owner, name, self._counting(name, getattr(owner, name)))

    def _counting(self, name: str, method: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            self.counts[name] += 1
            return method(*args, **kwargs)

        return wrapper


# --- timings ----------------------------------------------------------------------------------------


def test_expand_all_is_fast_with_many_long_lists(tree: ResultsView) -> None:
    load(tree, MANY)
    elapsed = best_ms(tree.expand_all_lists, prepare=tree.collapse_all_lists)
    assert len(open_lists(tree)) == MANY
    assert elapsed < BOUND_MS, f"expand all took {elapsed:.0f} ms"


def test_sort_and_filter_are_fast_with_every_list_open(tree: ResultsView) -> None:
    load(tree, MANY)
    tree.expand_all_lists()
    model = tree.results_model()

    def toggle_sort() -> None:
        model.set_sort_mode(SORT_ALPHA if model.sort_mode == SORT_RECENT else SORT_RECENT)

    sort = best_ms(toggle_sort)
    narrow = best_ms(lambda: model.set_filter("msci"), prepare=lambda: model.set_filter(""))
    widen = best_ms(lambda: model.set_filter(""), prepare=lambda: model.set_filter("msci"))
    assert model.rowCount() == MANY and len(open_lists(tree)) == MANY  # every list came back open
    assert max(sort, narrow, widen) < BOUND_MS, f"sort {sort:.0f} ms, filter {narrow:.0f} ms, clear {widen:.0f} ms"


# --- what a layout asks Python for ------------------------------------------------------------------


def test_a_full_layout_measures_each_row_once_not_each_cell(tree: ResultsView) -> None:
    load(tree, 40, k=10)
    tree.expand_all_lists()
    settle()
    calls = Calls(tree)
    tree.doItemsLayout()  # what a reset, a sort or a filter change ends in
    rows = 40 + 40 * 10
    assert tree.results_model().columnCount() == 14
    # Before: one sizeHint and one index per cell (14 per row), and ~12,000 rowCount calls
    # (hasIndex in index() and the delegate counting the siblings of every cell).
    assert rows <= calls.counts["sizeHint"] <= rows + 5, calls.counts
    assert rows <= calls.counts["index"] <= 2 * rows + 5, calls.counts
    assert calls.counts["rowCount"] <= 2 * 40 + 5, calls.counts  # hasChildren and rowCount per open list


def test_row_heights_columns_and_scrolling_survive_a_layout(tree: ResultsView) -> None:
    load(tree, 6, k=4)  # collapsed, the lists fit: no vertical scroll bar yet
    model, header = tree.results_model(), tree.header()
    header.resizeSection(3, 333)  # as if the user dragged a column wider
    widths = [header.sectionSize(c) for c in range(header.count())]
    assert not tree.verticalScrollBar().isVisible()
    tree.expand_all_lists()  # a full layout that brings the vertical scroll bar
    settle()
    assert tree.verticalScrollBar().isVisible()
    assert [header.sectionSize(c) for c in range(header.count())] == widths
    bar = tree.horizontalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(bar.maximum() // 2)
    scrolled = bar.value()
    model.set_sort_mode(SORT_ALPHA)  # a full layout with the dataset columns briefly hidden
    settle()
    assert [header.sectionSize(c) for c in range(header.count())] == widths
    assert all(not header.isSectionHidden(c) for c in range(header.count()))
    assert bar.value() == scrolled
    first = model.index(0, 0)
    assert tree.visualRect(first).height() == QUERY_ROW_H
    assert tree.visualRect(model.index(0, 0, first)).height() == RESULT_ROW_H
    assert tree.visualRect(model.index(3, 0, first)).height() == RESULT_ROW_H + GROUP_GAP  # the list's last row
    assert tree.visualRect(model.index(0, 5, first)).height() == RESULT_ROW_H  # every column of a row


def test_the_minute_tick_repaints_without_measuring_rows(tree: ResultsView) -> None:
    load(tree, 40, k=10)
    tree.expand_all_lists()
    settle()

    class Paints(QObject):
        count = 0

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.Paint:
                self.count += 1
            return False

    paints = Paints()
    tree.viewport().installEventFilter(paints)
    calls = Calls(tree)
    tree.results_model().refresh_times()  # "2 min ago" -> "3 min ago"
    settle()
    tree.viewport().removeEventFilter(paints)
    assert paints.count >= 1  # the new relative times are painted
    assert calls.counts["sizeHint"] == 0, calls.counts  # before: every row of every open list, in every column
    tree.doItemsLayout()
    assert calls.counts["sizeHint"] > 0  # (the counter does see the tree's calls)


# --- signals ----------------------------------------------------------------------------------------


def test_expanding_and_collapsing_every_list_is_reported_once(tree: ResultsView) -> None:
    load(tree, MANY, k=5)
    reports: Counter[str] = Counter()
    tree.expanded.connect(lambda _index: reports.update(["expanded"]))
    tree.collapsed.connect(lambda _index: reports.update(["collapsed"]))
    tree.expansion_changed.connect(lambda: reports.update(["changed"]))
    tree.expand_all_lists()
    assert reports == {"expanded": 1, "changed": 1}  # before: one per list and one per result row
    assert tree.any_expanded() and len(open_lists(tree)) == MANY
    reports.clear()
    tree.collapse_all_lists()
    assert reports == {"collapsed": 1, "changed": 1}  # before: one per list
    assert not tree.any_expanded() and open_lists(tree) == []
    reports.clear()
    tree.collapse_all_lists()  # nothing to close: the window still hears about it once
    assert reports == {"changed": 1}


def test_a_reset_reopens_the_lists_quietly(tree: ResultsView) -> None:
    entries = load(tree, 30, k=5)
    model = tree.results_model()
    tree.expand_all_lists()
    settle()
    bar = tree.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    scrolled = bar.value()
    reports: Counter[str] = Counter()
    tree.expanded.connect(lambda _index: reports.update(["expanded"]))
    tree.expansion_changed.connect(lambda: reports.update(["changed"]))
    model.set_entries(entries)  # a re-run or a deletion resets the model
    assert len(open_lists(tree)) == 30
    assert reports == {"expanded": 1, "changed": 1}  # before: 30 + 1
    assert bar.value() == scrolled  # laid out again before the position was restored


def test_sort_and_filter_keep_expansion_selection_and_current_row(tree: ResultsView) -> None:
    load(tree, 9, k=4)
    model = tree.results_model()
    resets = []
    model.modelAboutToBeReset.connect(lambda: resets.append(True))
    world, emerging = list_index(model, "MSCI World tracker 000"), list_index(model, "MSCI EM tracker 002")
    tree.setExpanded(world, True)
    tree.setExpanded(emerging, True)
    current = model.index(2, 0, world)  # rank 3 of the "MSCI World" list
    tree.setCurrentIndex(current)
    entry, result = model.entry_for(current), model.result_for(current)

    def check(expanded: set[str]) -> None:
        index = tree.currentIndex()
        assert (model.entry_for(index), model.result_for(index)) == (entry, result)
        assert [model.result_for(i) for i in tree.selectionModel().selectedRows(0)] == [result]
        assert set(open_lists(tree)) == expanded
        assert tree.any_expanded() == bool(expanded)

    both = {model.key_of(entry), model.key_of(model.entry_for(emerging))}
    model.set_sort_mode(SORT_ALPHA)
    check(both)
    model.set_filter("world")  # hides the other open list
    check({model.key_of(entry)})
    model.set_filter("")  # ...and brings it back open
    check(both)
    model.set_sort_mode(SORT_RECENT)
    check(both)
    assert resets == []  # layout changes: nothing was rebuilt from scratch


def test_a_filter_that_hides_every_open_list(tree: ResultsView) -> None:
    load(tree, 9, k=3)
    model = tree.results_model()
    tree.setExpanded(list_index(model, "S&P 500 tracker 001"), True)
    reports: Counter[str] = Counter()
    tree.expansion_changed.connect(lambda: reports.update(["changed"]))
    model.set_filter("msci")
    assert not tree.any_expanded() and reports["changed"] == 1  # the collapse button can say "Expand all"
    model.set_filter("")
    assert open_lists(tree) == [model.key_of(model.entry_for(list_index(model, "S&P 500 tracker 001")))]
    assert tree.any_expanded()


def test_the_model_stays_consistent_through_sorts_filters_and_k(qapp: QApplication) -> None:
    messages: list[str] = []

    def handler(mode: QtMsgType, _context: object, message: str) -> None:
        if mode != QtMsgType.QtDebugMsg:
            messages.append(message)

    previous = qInstallMessageHandler(handler)
    try:
        model = ResultsModel()
        tester = QAbstractItemModelTester(model, QAbstractItemModelTester.FailureReportingMode.Warning)
        model.set_entries(make_entries(12, 8))
        model.set_columns(order_columns(COLUMNS, ROLES, "Fund name"))
        for step in (lambda: model.set_k(3), lambda: model.set_sort_mode(SORT_ALPHA),
                     lambda: model.set_filter("msci"), lambda: model.set_k(8), lambda: model.set_filter("em"),
                     lambda: model.set_filter(""), lambda: model.set_sort_mode(SORT_RECENT)):
            step()  # the tester checks every signal, and the indexes it holds, along the way
        assert model.rowCount() == 12
        assert not model.index(12, 0).isValid() and not model.index(0, 14).isValid()
        assert not model.index(8, 0, model.index(0, 0)).isValid()  # K = 8 results per list
        assert not model.index(0, 0, model.index(0, 1)).isValid()  # results hang off column 0 only
        assert model.parent(model.index(4, 3, model.index(5, 0))) == model.index(5, 0)
        del tester
    finally:
        qInstallMessageHandler(previous)
    assert messages == []


# --- the window ---------------------------------------------------------------------------------------


class LexicalBackend(Backend):
    """The real backend without the language model (as in test_ui_smoke)."""

    def __init__(self, table: Table) -> None:
        self.table = table

    def load_embedder(self) -> None:
        return None

    def list_sheets(self, path: Path) -> list[str]:
        return [self.table.sheet_name]

    def read_raw(self, path: Path, sheet: str | None, max_rows: int) -> list[list[str]]:
        return [list(self.table.columns), *(list(row) for row in self.table.rows[: max_rows - 1])]

    def read_table(self, path: Path, sheet: str | None, header_row: int | None,
                   progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> Table:
        return self.table


def wait_until(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)
    return True


@pytest.fixture
def window(qapp: QApplication, isolated_data_dir: Path, funds_table: Table,
           monkeypatch: pytest.MonkeyPatch) -> Iterator[MainWindow]:
    """The main window showing 153 lists of 50 results; ``window.syncs`` counts collapse-button updates."""
    syncs = Counter()
    original = MainWindow._sync_collapse_button

    def counting(self: MainWindow) -> None:
        syncs["sync"] += 1
        original(self)

    monkeypatch.setattr(MainWindow, "_sync_collapse_button", counting)
    with Store(isolated_data_dir) as store:
        opened = MainWindow(store, LexicalBackend(funds_table), autostart=False)
        opened.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        opened.resize(1320, 860)
        opened.show()
        opened.start()
        assert wait_until(lambda: not opened.runner.active()), "startup work did not finish"
        opened.set_entries(make_entries(MANY))
        opened.k_stepper.setValue(K)
        settle()
        opened.syncs = syncs
        yield opened
        opened.close()
        opened.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_collapse_button_updates_once_per_collapse_or_expand_all(window: MainWindow) -> None:
    timings: dict[str, list[float]] = {"expand": [], "collapse": []}
    for _ in range(2):
        for action, label in (("expand", "Collapse all"), ("collapse", "Expand all")):
            window.syncs.clear()
            start = time.perf_counter()
            window.collapse_button.click()
            settle()
            timings[action].append((time.perf_counter() - start) * 1000)
            assert window.collapse_button.text() == label
            # Before: 7,804 updates (every list and every result row), each scanning every list.
            assert window.syncs["sync"] == 1, action
    expand, collapse = min(timings["expand"]), min(timings["collapse"])
    assert expand < BOUND_MS and collapse < 100, f"expand {expand:.0f} ms, collapse {collapse:.0f} ms"


# --- context menu -----------------------------------------------------------------------------------


def test_context_menus_do_not_accumulate(tree: ResultsView, monkeypatch: pytest.MonkeyPatch) -> None:
    load(tree, 3, k=2)
    model = tree.results_model()
    tree.expand_all_lists()
    settle()
    shown: list[list[str]] = []

    class DismissedMenu(QMenu):
        def exec(self, *_args: object) -> None:  # the user closes the menu without choosing
            shown.append([action.text() for action in self.actions() if not action.isSeparator()])

    monkeypatch.setattr(results_view, "menu", lambda parent=None, title="": DismissedMenu(title, parent))
    targets = [model.index(0, 0), model.index(1, 3, model.index(0, 0))]  # a query row, a result cell
    for i in range(20):
        position = tree.visualRect(targets[i % 2]).center()
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, position, tree.viewport().mapToGlobal(position))
        QApplication.sendEvent(tree.viewport(), event)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert len(shown) == 20 and "Re-run" in shown[0] and "Copy row" in shown[1]
    assert tree.findChildren(QMenu) == []  # before: 20 menus, their actions holding the lists
