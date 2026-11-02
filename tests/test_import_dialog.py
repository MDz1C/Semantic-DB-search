"""Import dialog (ARCHITECTURE.md section 12.2): Import only ever takes a preview that matches the
sheet and header-row controls, and a sheet or header row that failed to load can be left again.

Small workbooks are written with xlsxwriter and read by the real backend (in-process, see
conftest.py), so every reload goes through the dialog's background task as in the app.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import xlsxwriter
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QWidget

from semdb.core.types import CancelFn, Cancelled, ProgressFn, Table
from semdb.ui import theme
from semdb.ui.import_dialog import ImportDialog
from semdb.ui.services import Backend
from semdb.ui.workers import TaskRunner

ACCEPT = QDialog.DialogCode.Accepted
FUNDS = [
    ["ISIN", "Brokerage", "Fund name"],
    ["IE00B5BMR087", "BlackRock", "iShares Core S&P 500 UCITS ETF USD (Acc)"],
    ["IE00B3XXRP09", "Vanguard", "Vanguard S&P 500 UCITS ETF USD Distributing"],
    ["LU1681048804", "Amundi", "Amundi S&P 500 UCITS ETF EUR (C)"],
    ["IE00B6YX5C33", "State Street", "SPDR S&P 500 UCITS ETF (Dist)"],
]
FUND_COLUMNS = FUNDS[0]
GROUPED = [["Identity", "Identity", "Product"], *FUNDS]  # group labels above the real header...
GROUP_COLUMNS = ["Identity", "Identity (2)", "Product"]  # ...are detected as the header (row 1)
NO_DATA = "There is no data in sheet 'Empty' of 'sheets.xlsx'."

OpenDialog = Callable[..., ImportDialog]


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


def settle(seconds: float) -> None:
    """Process events for ``seconds``, e.g. past the pause the dialog waits for after typing."""
    wait_until(lambda: False, seconds)


def visible_texts(widget: QWidget) -> list[str]:
    return [label.text() for label in widget.findChildren(QLabel) if label.isVisible()]


def editor_columns(dialog: ImportDialog) -> list[str]:
    return list(dialog.editor.roles())


def preview_columns(dialog: ImportDialog) -> list[str]:
    model = dialog.preview_view.model()
    return [model.headerData(column, Qt.Orientation.Horizontal) for column in range(model.columnCount())]


def type_header_row(dialog: ImportDialog, text: str) -> None:
    """Replace the header-row number as a user does: select it and type."""
    spin = dialog.header_stepper.spin_box()
    spin.setFocus()
    spin.selectAll()
    QTest.keyClicks(spin, text)


def write_workbook(path: Path, **sheets: list[list[str]]) -> Path:
    with xlsxwriter.Workbook(str(path)) as workbook:
        for name, rows in sheets.items():
            worksheet = workbook.add_worksheet(name)
            for number, row in enumerate(rows):
                worksheet.write_row(number, 0, row)
    return path


class GatedBackend(Backend):
    """The real backend, except that reading ``sheet`` waits for :meth:`release` (a big sheet)."""

    def __init__(self, sheet: str) -> None:
        super().__init__()
        self.sheet = sheet
        self._gate = threading.Event()

    def release(self) -> None:
        self._gate.set()

    def read_table(
        self,
        path: Path,
        sheet: str | None,
        header_row: int | None,
        progress: ProgressFn | None = None,
        cancel: CancelFn | None = None,
    ) -> Table:
        deadline = time.monotonic() + 10
        while sheet == self.sheet and not self._gate.wait(0.01):
            if (cancel is not None and cancel()) or time.monotonic() > deadline:
                raise Cancelled()
        return super().read_table(path, sheet, header_row, progress, cancel)


@pytest.fixture(scope="module")
def qapp() -> Iterator[QApplication]:
    app = QApplication.instance() or QApplication(["semdb-tests"])
    theme.manager().install(app, "light")
    yield app


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A modal dialog would block the run: refuse it and fail the test instead."""
    opened: list[str] = []

    def refuse(dialog: QDialog, *_args: object) -> QDialog.DialogCode:
        opened.append(type(dialog).__name__)
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", refuse)
    yield
    assert opened == [], f"unexpected modal dialogs: {opened}"


@pytest.fixture
def runner(qapp: QApplication) -> Iterator[TaskRunner]:
    runner = TaskRunner()
    yield runner
    runner.cancel_all()
    runner.wait(5000)


@pytest.fixture
def open_dialog(qapp: QApplication, runner: TaskRunner) -> Iterator[OpenDialog]:
    """Shows the import dialog for a file and waits for its first preview."""
    dialogs: list[ImportDialog] = []

    def open_(path: Path, backend: Backend | None = None) -> ImportDialog:
        dialog = ImportDialog(path, backend or Backend(), runner)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        dialog.show()  # Enter reaches the default button of a visible dialog only
        dialogs.append(dialog)
        assert wait_until(dialog.import_button.isEnabled), "the first preview never loaded"
        return dialog

    yield open_
    for dialog in dialogs:
        dialog.close()
    runner.wait(5000)
    for dialog in dialogs:
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


# --- the header row ----------------------------------------------------------------------------------


def test_enter_after_typing_a_header_row_previews_that_row_instead_of_importing(
    open_dialog: OpenDialog, tmp_path: Path
) -> None:
    dialog = open_dialog(write_workbook(tmp_path / "funds.xlsx", Funds=GROUPED))
    assert dialog.header_stepper.value() == 1 and editor_columns(dialog) == GROUP_COLUMNS
    type_header_row(dialog, "2")
    QTest.keyClick(dialog.header_stepper.spin_box(), Qt.Key.Key_Return)
    assert dialog.isVisible() and dialog.table is None, (
        f"Enter imported the columns of header row 1: {dialog.table.columns if dialog.table else None}"
    )
    assert wait_until(dialog.import_button.isEnabled), "the preview of header row 2 never loaded"
    assert preview_columns(dialog) == editor_columns(dialog) == FUND_COLUMNS  # the role editor follows
    QTest.keyClick(dialog.header_stepper.spin_box(), Qt.Key.Key_Return)  # nothing pending now: Enter imports
    assert dialog.result() == ACCEPT and dialog.table is not None and dialog.roles is not None
    assert dialog.table.columns == FUND_COLUMNS and dialog.table.source_rows[0] == 3
    assert list(dialog.roles) == FUND_COLUMNS


def test_import_waits_for_the_preview_of_a_changed_header_row(open_dialog: OpenDialog, tmp_path: Path) -> None:
    dialog = open_dialog(write_workbook(tmp_path / "funds.xlsx", Funds=GROUPED))
    type_header_row(dialog, "2")
    assert not dialog.import_button.isEnabled(), "Import still offers the preview of header row 1"
    dialog.import_button.click()  # within the typing pause
    assert dialog.isVisible() and dialog.table is None
    assert wait_until(dialog.import_button.isEnabled), "the preview of header row 2 never loaded"
    assert preview_columns(dialog) == editor_columns(dialog) == FUND_COLUMNS
    dialog.import_button.click()
    assert dialog.result() == ACCEPT and dialog.table is not None and dialog.table.columns == FUND_COLUMNS


def test_a_header_row_outside_the_sheet_can_be_corrected(open_dialog: OpenDialog, tmp_path: Path) -> None:
    dialog = open_dialog(write_workbook(tmp_path / "funds.xlsx", Funds=FUNDS))
    dialog.header_stepper.setValue(9)
    message = "Row 9 is outside sheet 'Funds' of 'funds.xlsx', which has 5 rows."
    assert wait_until(lambda: message in visible_texts(dialog))
    assert not dialog.import_button.isEnabled()
    dialog.header_stepper.setValue(1)
    assert wait_until(dialog.import_button.isEnabled), "the error page stayed after choosing a valid header row"
    assert dialog.preview_view.isVisible() and editor_columns(dialog) == FUND_COLUMNS


def test_a_pending_header_row_change_does_not_undo_a_sheet_switch(open_dialog: OpenDialog, tmp_path: Path) -> None:
    backend = GatedBackend("Grouped")
    dialog = open_dialog(write_workbook(tmp_path / "sheets.xlsx", Funds=FUNDS, Grouped=GROUPED), backend)
    type_header_row(dialog, "2")
    dialog.sheet_combo.setCurrentText("Grouped")  # before the typing pause is over
    settle(0.5)  # "Grouped" is still being read when the pause ends
    backend.release()
    assert wait_until(dialog.import_button.isEnabled), "the chosen sheet never loaded"
    assert dialog.sheet_combo.currentText() == "Grouped"
    assert dialog.header_stepper.value() == 1 and editor_columns(dialog) == GROUP_COLUMNS  # detected anew


def test_a_header_row_typed_while_a_sheet_loads_is_kept(open_dialog: OpenDialog, tmp_path: Path) -> None:
    backend = GatedBackend("Grouped")
    dialog = open_dialog(write_workbook(tmp_path / "sheets.xlsx", Funds=FUNDS, Grouped=GROUPED), backend)
    dialog.sheet_combo.setCurrentText("Grouped")
    type_header_row(dialog, "2")  # while "Grouped" is read with its detected header row
    backend.release()
    assert wait_until(dialog.import_button.isEnabled), "the chosen sheet never loaded"
    assert dialog.sheet_combo.currentText() == "Grouped" and dialog.header_stepper.value() == 2
    assert preview_columns(dialog) == editor_columns(dialog) == FUND_COLUMNS


# --- sheets that cannot be imported ------------------------------------------------------------------


def test_choosing_a_good_sheet_after_one_without_data_reloads_it(open_dialog: OpenDialog, tmp_path: Path) -> None:
    dialog = open_dialog(write_workbook(tmp_path / "sheets.xlsx", Funds=FUNDS, Empty=[], Grouped=GROUPED))
    dialog.sheet_combo.setCurrentText("Empty")
    assert wait_until(lambda: NO_DATA in visible_texts(dialog))
    assert not dialog.import_button.isEnabled()
    dialog.sheet_combo.setCurrentText("Funds")  # the sheet that was loaded before
    assert wait_until(dialog.import_button.isEnabled, 5), "the error page stayed after choosing a sheet with data"
    assert dialog.preview_view.isVisible() and editor_columns(dialog) == FUND_COLUMNS
    dialog.import_button.click()
    assert dialog.result() == ACCEPT and dialog.table is not None and dialog.table.sheet_name == "Funds"


def test_a_header_row_change_on_a_sheet_without_data_stays_on_that_sheet(
    open_dialog: OpenDialog, runner: TaskRunner, tmp_path: Path
) -> None:
    dialog = open_dialog(write_workbook(tmp_path / "sheets.xlsx", Funds=FUNDS, Empty=[], Grouped=GROUPED))
    dialog.sheet_combo.setCurrentText("Empty")
    assert wait_until(lambda: NO_DATA in visible_texts(dialog))
    dialog.header_stepper.setValue(2)
    settle(0.5)  # past the typing pause
    assert wait_until(lambda: not runner.active())
    assert dialog.sheet_combo.currentText() == "Empty", "the dialog went back to another sheet"
    assert NO_DATA in visible_texts(dialog) and not dialog.import_button.isEnabled()
    dialog.sheet_combo.setCurrentText("Grouped")
    assert wait_until(dialog.import_button.isEnabled), "the error page stayed after choosing a sheet with data"
    assert dialog.header_stepper.value() == 1 and editor_columns(dialog) == GROUP_COLUMNS
