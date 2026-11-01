"""Offscreen smoke tests of the desktop window (ARCHITECTURE.md section 12, R7-R13).

The window runs against the real store, the real profiler and the real lexical engine; the
language model is reported missing and the "file" behind the import dialog is an in-memory
table (the dependency-injection approach of tools/ui_screenshots.py).  Modal dialogs never
block: each expected one is answered through the ``dialogs`` fixture, any other one fails.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QModelIndex, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QLabel, QToolButton, QWidget
from python_calamine import CalamineWorkbook

from semdb.core.profile import profile_columns, suggest_roles
from semdb.core.types import CancelFn, FamilyEvidence, HistoryEntry, ProgressFn, ResultRow, Role, Roles, Table
from semdb.storage import Store
from semdb.ui import theme
from semdb.ui.import_dialog import ImportDialog
from semdb.ui.main_window import MainWindow
from semdb.ui.results_model import SORT_ALPHA, SORT_RECENT
from semdb.ui.services import Backend
from semdb.ui.workers import TaskRunner

OpenWindow = Callable[[], MainWindow]
ACCEPT, REJECT = QDialog.DialogCode.Accepted, QDialog.DialogCode.Rejected
RECENT = ["iShares Core S&P 500", "Xtrackers DAX", "Amundi MSCI Emerging Markets"]  # newest first
ALPHABETICAL = ["Amundi MSCI Emerging Markets", "iShares Core S&P 500", "Xtrackers DAX"]


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
    """The real backend without the language model; "reading a file" yields ``table``."""

    def __init__(self, table: Table) -> None:
        self.table = table

    def load_embedder(self) -> None:
        return None  # like an installation whose model folder is missing: lexical search only

    def list_sheets(self, path: Path) -> list[str]:
        return [self.table.sheet_name]

    def read_raw(self, path: Path, sheet: str | None, max_rows: int) -> list[list[str]]:
        return [list(self.table.columns), *(list(row) for row in self.table.rows[: max_rows - 1])]

    def read_table(
        self,
        path: Path,
        sheet: str | None,
        header_row: int | None,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> Table:
        return self.table


class DialogScript:
    """Stands in for the user at modal dialogs: each expected dialog consumes one answer (a
    DialogCode, or a callable that receives the dialog); unexpected dialogs are recorded."""

    def __init__(self) -> None:
        self.answers: list[object] = []
        self.shown: list[list[str]] = []  # label texts of every dialog shown
        self.unexpected: list[str] = []

    def expect(self, answer: object) -> None:
        self.answers.append(answer)

    def exec(self, dialog: QDialog) -> object:
        texts = labels(dialog)
        self.shown.append(texts)
        if not self.answers:
            self.unexpected.append(f"{type(dialog).__name__}: {texts}")
            return REJECT
        answer = self.answers.pop(0)
        return answer(dialog) if callable(answer) else answer

    def file_dialog(self, *args: object, **kwargs: object) -> tuple[str, str]:
        self.unexpected.append(f"file dialog {args[1:2]}")  # (parent, caption, ...)
        return "", ""  # as if the user pressed Cancel


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication(["semdb-tests"])
    theme.manager().install(app, "light")
    yield app
    theme.manager().set_mode("light")


@pytest.fixture(autouse=True)
def dialogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[DialogScript]:
    script = DialogScript()
    monkeypatch.setattr(QDialog, "exec", lambda dialog, *_args: script.exec(dialog))
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(script.file_dialog))
    yield script
    assert script.unexpected == [], "a dialog nobody expected was opened"
    assert script.answers == [], "an expected dialog was never opened"


@pytest.fixture
def store(isolated_data_dir: Path) -> Iterator[Store]:
    with Store(isolated_data_dir) as opened:
        yield opened


@pytest.fixture
def backend(funds_table: Table) -> LexicalBackend:
    return LexicalBackend(funds_table)


@pytest.fixture
def open_window(qapp: QApplication, store: Store, backend: LexicalBackend) -> Iterator[OpenWindow]:
    """Opens the main window like the app does and waits until its startup work is done."""
    windows: list[MainWindow] = []

    def open_() -> MainWindow:
        window = MainWindow(store, backend, autostart=False)
        window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        window.resize(1320, 860)
        window.show()
        windows.append(window)
        window.start()  # what the app schedules right after the window appears
        assert wait_until(lambda: not window.runner.active()), "startup work did not finish"
        return window

    yield open_
    for window in windows:
        window.close()  # saves the window state and waits for background work
        window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture
def history(funds_table: Table) -> list[HistoryEntry]:
    """Three hand-built result lists, oldest first; their A–Z order differs from their recency."""

    def entry(query: str, created_at: str, rows: list[int], evidence: tuple[FamilyEvidence, ...] = ()) -> HistoryEntry:
        named = {e.column: e.value for e in evidence}
        results = []
        for rank, row in enumerate(rows, start=1):
            cells = list(funds_table.rows[row])
            agree = [c for c, v in named.items() if cells[funds_table.column_index(c)] == v]
            conflict = [c for c in named if c not in agree]
            score = round(0.97 - 0.06 * rank, 2)
            detail = {"semantic": score, "lexical": score, "text_score": score,
                      "family_factor": 0.15 if conflict else 1.0, "exact_id": None,
                      "agreements": agree, "conflicts": conflict}  # fmt: skip
            results.append(ResultRow(rank, score, funds_table.source_rows[row], cells, detail))
        return HistoryEntry(None, query, created_at, funds_table.label, list(funds_table.columns),
                            ["ISIN", "Brokerage"], "Fund name", list(evidence), results)  # fmt: skip

    return [
        entry("Amundi MSCI Emerging Markets", "2026-10-30T09:00:00+00:00", [9, 3]),
        entry("Xtrackers DAX", "2026-10-31T09:00:00+00:00", [11, 10, 2, 13, 6, 7]),
        entry("iShares Core S&P 500", "2026-11-01T09:00:00+00:00", [0, 1, 4, 5, 8, 12, 2, 3],
              (FamilyEvidence("Brokerage", "BlackRock", 0.99, "iShares"),)),
    ]  # fmt: skip


def queries(window: MainWindow) -> list[str]:
    return [entry.query for entry in window.model.visible_entries()]


def child_counts(window: MainWindow) -> list[int]:
    model = window.model
    return [model.rowCount(model.index(row, 0)) for row in range(model.rowCount())]


def expanded(window: MainWindow) -> list[bool]:
    model = window.model
    return [window.view.isExpanded(model.index(row, 0)) for row in range(model.rowCount())]


def list_index(window: MainWindow, query: str) -> QModelIndex:
    model = window.model
    entry = next(entry for entry in model.visible_entries() if entry.query == query)
    return model.query_index(model.key_of(entry))


# --- states and the results tree -------------------------------------------------------------------


def test_empty_state_without_a_database(open_window: OpenWindow) -> None:
    window = open_window()
    assert window.pages.currentWidget() is window.empty_page
    assert window.dataset_bar.isHidden()
    for widget in (window.sort_control, window.collapse_button, window.filter_edit, window.export_button):
        assert widget.isHidden()  # nothing to sort, collapse, filter or export yet
    assert not window.reset_button.isEnabled()
    assert not window.composer.search_button.isEnabled()
    assert window.composer.search_button.toolTip() == "Import a database to start searching"
    assert window.model_text.text() == "Lexical only (model missing)"

    def left(widget: QWidget) -> int:
        return widget.mapTo(window, QPoint(0, 0)).x()

    # R7: results per query at the top left; Import, then Reset, at the top right
    assert left(window.k_stepper) < left(window.import_button) < left(window.reset_button)


def test_stored_lists_appear_collapsed(open_window: OpenWindow, store: Store, history: list[HistoryEntry]) -> None:
    store.add_entries(history)
    window = open_window()
    assert window.pages.currentWidget() is window.view
    assert queries(window) == RECENT
    assert expanded(window) == [False, False, False]  # R9: every list starts collapsed
    assert child_counts(window) == [5, 5, 2]  # 5 results per query by default; short lists show what they have
    assert window.results_count.text() == "3"
    assert window.collapse_button.text() == "Expand all"


def test_collapse_all_button_toggles_every_list(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry]
) -> None:
    store.add_entries(history)
    window = open_window()
    window.collapse_button.click()
    assert expanded(window) == [True, True, True]
    assert window.collapse_button.text() == "Collapse all"
    window.collapse_button.click()
    assert expanded(window) == [False, False, False]
    assert window.collapse_button.text() == "Expand all"
    window.view.setExpanded(list_index(window, "Xtrackers DAX"), True)  # one open list offers "Collapse all"
    assert window.collapse_button.text() == "Collapse all"


def test_results_per_query_sets_the_rows_of_every_list(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry]
) -> None:
    store.add_entries(history)
    window = open_window()
    window.collapse_button.click()
    window.k_stepper.setValue(3)
    assert child_counts(window) == [3, 3, 2]
    plus = next(button for button in window.k_stepper.findChildren(QToolButton) if button.toolTip() == "More")
    for _ in range(5):
        plus.click()
    assert window.k_stepper.value() == 8 and child_counts(window) == [8, 6, 2]
    window.k_stepper.setValue(1)
    assert child_counts(window) == [1, 1, 1]
    window.k_stepper.setValue(999)  # clamped to the 50 results stored per query
    assert window.k_stepper.value() == 50 and child_counts(window) == [8, 6, 2]
    assert expanded(window) == [True, True, True]
    assert store.get_setting("results_per_query") == 50  # remembered for the next start


def test_sort_modes_reorder_the_lists(open_window: OpenWindow, store: Store, history: list[HistoryEntry]) -> None:
    store.add_entries(history)
    window = open_window()
    window.view.setExpanded(list_index(window, "Xtrackers DAX"), True)
    QTest.keyClick(window.sort_control, Qt.Key.Key_Right)  # Recent -> A–Z
    assert window.sort_control.value() == SORT_ALPHA
    assert queries(window) == ALPHABETICAL
    assert expanded(window) == [False, False, True]  # the open list stays open where it moved
    assert store.get_setting("sort_mode") == SORT_ALPHA
    QTest.keyClick(window.sort_control, Qt.Key.Key_Left)
    assert window.sort_control.value() == SORT_RECENT
    assert queries(window) == RECENT and expanded(window) == [False, True, False]


def test_filter_narrows_the_lists(open_window: OpenWindow, store: Store, history: list[HistoryEntry]) -> None:
    store.add_entries(history)
    window = open_window()
    QTest.keyClicks(window.filter_edit, "dax")
    assert queries(window) == ["Xtrackers DAX"]
    assert window.results_count.text() == "1 of 3"
    window.filter_edit.setText("BLACKROCK")  # also matches the identity a query named
    assert queries(window) == ["iShares Core S&P 500"]
    window.filter_edit.setText("no such fund")
    assert window.model.rowCount() == 0
    assert window.pages.currentWidget() is window.no_matches_page
    window.filter_edit.clear()
    assert queries(window) == RECENT and window.pages.currentWidget() is window.view


def test_selecting_a_result_fills_the_details_panel(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry]
) -> None:
    store.add_entries(history)
    window = open_window()
    view, model = window.view, window.model
    parent = list_index(window, "iShares Core S&P 500")
    entry = model.entry_for(parent)
    view.setExpanded(parent, True)
    view.setCurrentIndex(model.index(1, 0, parent))  # its second result
    current_entry, current_result = window.details.current()
    assert current_entry is entry and current_result is entry.results[1]
    assert window.details.isVisible()
    texts = labels(window.details)
    assert any(text.startswith("Result 2 of 5") for text in texts)
    assert "iShares Core S&P 500 UCITS ETF USD (Dist)" in texts  # titled by the primary column
    assert "Brokerage agrees" in texts  # the identity check
    assert "IE0031442068" in texts  # every field of the row
    view.setCurrentIndex(parent)  # the query row: what was detected in it
    assert window.details.current() == (entry, None)
    assert "Brokerage → BlackRock" in labels(window.details)


def test_light_and_dark_themes_render(open_window: OpenWindow, store: Store, history: list[HistoryEntry]) -> None:
    store.add_entries(history)
    window = open_window()
    window.collapse_button.click()  # paint result rows, pills and score bars too
    window.view.setCurrentIndex(window.model.index(0, 0, window.model.index(0, 0)))
    for mode, dark in (("dark", True), ("light", False)):
        window.set_theme(mode)
        assert theme.manager().is_dark is dark and theme.tokens().dark is dark
        assert window.theme_actions[mode].isChecked()
        assert store.get_setting("theme") == mode
        image = window.grab().toImage()
        assert not image.isNull()
        header = image.pixelColor(10, 10)  # the header bar is painted in the theme's surface colour
        assert (header.lightness() < 128) is dark


# --- dialogs and flows ------------------------------------------------------------------------------


def test_import_dialog_suggests_roles_with_the_real_profiler(
    qapp: QApplication, funds_table: Table, backend: LexicalBackend
) -> None:
    runner = TaskRunner()
    dialog = ImportDialog(Path("funds.xlsx"), backend, runner)
    try:
        assert wait_until(dialog.import_button.isEnabled), "the preview never loaded"
        suggested = suggest_roles(funds_table, profile_columns(funds_table))
        assert dialog.editor.roles() == suggested
        assert suggested["ISIN"] is Role.DECIDING and suggested["Brokerage"] is Role.DECIDING
        assert suggested["Fund name"] is Role.SEARCH and suggested["TER"] is Role.IGNORE
        assert dialog.import_button.text() == "Import 16 rows"
        assert dialog.sheet_combo.isHidden()  # one sheet: nothing to choose
        assert dialog.header_stepper.value() == 1  # the detected header row, 1-based
        assert dialog.preview_view.model().rowCount() == len(funds_table)
        for name in funds_table.columns:  # nothing left to search on: Import is refused
            dialog.editor.set_role(name, Role.IGNORE)
        assert not dialog.import_button.isEnabled()
        dialog.editor.set_role("Fund name", Role.SEARCH)
        dialog.editor.set_role("Brokerage", Role.DECIDING)
        assert dialog.import_button.isEnabled()
        dialog.import_button.click()
        assert dialog.result() == ACCEPT
        assert dialog.table is funds_table
        assert dialog.roles == {"ISIN": Role.IGNORE, "Fund name": Role.SEARCH, "Brokerage": Role.DECIDING,
                                "Currency": Role.IGNORE, "TER": Role.IGNORE}  # fmt: skip
    finally:
        dialog.close()
        runner.wait(5000)
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_importing_a_file_builds_the_index(
    open_window: OpenWindow, store: Store, funds_table: Table, dialogs: DialogScript
) -> None:
    window = open_window()

    def import_when_loaded(dialog: QDialog) -> object:
        assert wait_until(dialog.import_button.isEnabled)
        dialog.import_button.click()
        return dialog.result()

    dialogs.expect(import_when_loaded)
    assert window.import_file(Path("funds.xlsx"))
    assert wait_until(lambda: not window.runner.active())
    assert window.pages.currentWidget() is window.ready_page
    assert not window.dataset_bar.isHidden() and window.dataset_name.full_text() == "funds.xlsx · Funds"
    assert window.composer.search_button.toolTip() == "Type or paste at least one query"  # ready, box empty
    stored = store.load_dataset()
    assert stored is not None and stored[0] == funds_table


def test_search_adds_an_expanded_list_and_persists_it(
    open_window: OpenWindow, store: Store, funds_table: Table, funds_roles: Roles
) -> None:
    store.save_dataset(funds_table, funds_roles)
    window = open_window()  # rebuilds the index of the stored database (lexical only)
    assert window.pages.currentWidget() is window.ready_page
    window.composer.set_text("iShares Core S&P 500")
    window.composer.search_button.click()
    assert wait_until(lambda: not window.is_searching() and not window.runner.active())
    (entry,) = window.entries()
    assert entry.query == "iShares Core S&P 500" and entry.id is not None
    assert [evidence.value for evidence in entry.evidence] == ["BlackRock"]
    top = entry.results[0].cells
    assert top[funds_table.column_index("Brokerage")] == "BlackRock" and "S&P 500" in top[1]
    assert expanded(window) == [True]  # a single query opens its new list
    assert window.composer.editor.toPlainText() == ""
    assert [stored.query for stored in store.load_entries()] == ["iShares Core S&P 500"]


def test_export_follows_the_visible_lists(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry], tmp_path: Path
) -> None:
    store.add_entries(history)
    window = open_window()
    window.k_stepper.setValue(2)
    window.sort_control.set_value(SORT_ALPHA, emit=True)
    window.export_results(tmp_path / "results")  # ".xlsx" is added
    assert wait_until(lambda: not window.runner.active())
    with CalamineWorkbook.from_path(str(tmp_path / "results.xlsx")) as workbook:
        summary = workbook.get_sheet_by_name("Summary").to_python()
    assert summary[0] == ["Query", "Match 1", "Score 1", "Match 2", "Score 2"]
    assert [row[0] for row in summary[1:]] == ALPHABETICAL
    assert window.status_text.text() == "Exported 3 queries to results.xlsx"


def test_reset_clears_every_list_after_confirmation(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry], dialogs: DialogScript
) -> None:
    store.add_entries(history)
    window = open_window()
    window.composer.set_text("Vanguard S&P 500")
    dialogs.expect(REJECT)  # "Cancel" keeps everything
    window.reset_button.click()
    assert len(window.entries()) == 3 and len(store.load_entries()) == 3
    dialogs.expect(ACCEPT)
    window.reset_button.click()
    assert "Clear all results?" in dialogs.shown[-1]
    assert "3 result lists will be removed from this computer. Your database stays loaded." in dialogs.shown[-1]
    assert window.entries() == [] and window.model.rowCount() == 0
    assert store.load_entries() == []  # also gone after a restart (R11)
    assert window.composer.editor.toPlainText() == ""
    assert window.pages.currentWidget() is window.empty_page  # no database in this test
    assert not window.reset_button.isEnabled()


def test_delete_key_removes_the_selected_list(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry], dialogs: DialogScript
) -> None:
    store.add_entries(history)
    window = open_window()
    window.view.setCurrentIndex(list_index(window, "Xtrackers DAX"))
    dialogs.expect(ACCEPT)
    QTest.keyClick(window.view, Qt.Key.Key_Delete)
    assert "Delete this result list?" in dialogs.shown[-1]
    assert queries(window) == ["iShares Core S&P 500", "Amundi MSCI Emerging Markets"]
    assert [entry.query for entry in store.load_entries()] == ["Amundi MSCI Emerging Markets", "iShares Core S&P 500"]


def test_deleting_the_selected_list_clears_the_details_panel(
    open_window: OpenWindow, store: Store, history: list[HistoryEntry], dialogs: DialogScript
) -> None:
    store.add_entries(history)
    window = open_window()
    window.view.setCurrentIndex(list_index(window, "Xtrackers DAX"))
    assert window.details.current()[0].query == "Xtrackers DAX"
    dialogs.expect(ACCEPT)
    QTest.keyClick(window.view, Qt.Key.Key_Delete)
    assert "Xtrackers DAX" not in queries(window)  # the list is gone...
    assert window.details.current() == (None, None)  # ...so the panel should show "Nothing selected"
