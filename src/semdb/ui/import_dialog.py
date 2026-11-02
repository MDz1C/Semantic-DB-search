"""Import dialog (R2, R3): sheet and header row, a preview, and the role of every column."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QStackedWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..core.types import Roles, Table
from .columns_editor import EXPLAINER, INVALID_MESSAGE, RoleEditor, choices_from_profiles
from .results_model import plural
from .services import Backend, ImportPreview
from .widgets import Dialog, ElidedLabel, IconBadge, ProgressLine, Spinner, Stepper, button, label
from .workers import Task, TaskRunner

PREVIEW_ROWS = 30
MAX_HEADER_ROW = 30


class PreviewModel(QAbstractTableModel):
    """The first rows under the header, labelled with their sheet row numbers."""

    def __init__(self, table: Table, limit: int = PREVIEW_ROWS) -> None:
        super().__init__()
        self._columns = list(table.columns)
        self._rows = table.rows[:limit]
        self._numbers = table.source_rows[:limit]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            value = self._rows[index.row()][index.column()]
            return value if role == Qt.ItemDataRole.DisplayRole or len(value) > 30 else None
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._columns[section]
        return str(self._numbers[section]) if section < len(self._numbers) else None


class ImportDialog(Dialog):
    """Reads the file in the background; ``table``/``roles`` are set when the user accepts."""

    def __init__(self, path: str | Path, backend: Backend, runner: TaskRunner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(760, 600)
        self.resize(920, 720)
        self._path = Path(path)
        self._backend = backend
        self._runner = runner
        self._task: Task | None = None
        self._preview: ImportPreview | None = None
        self._open = True
        self._generation = 0
        self.table: Table | None = None
        self.roles: Roles | None = None

        head = QHBoxLayout()
        head.setSpacing(14)
        head.addWidget(IconBadge("file-spreadsheet", 44, radius=12, icon_ratio=0.5), 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label("Import database", "heading"))
        self._subtitle = ElidedLabel(self._path.name)
        self._subtitle.setProperty("role", "caption")
        titles.addWidget(self._subtitle)
        head.addLayout(titles, 1)
        self.body.addLayout(head)

        options = QHBoxLayout()
        options.setSpacing(8)
        self._sheet_label = label("Sheet", "muted")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setFixedHeight(34)
        self.sheet_combo.setMinimumWidth(220)
        self.sheet_combo.currentTextChanged.connect(self._on_sheet_changed)
        self._header_label = label("Header row", "muted")
        self.header_stepper = Stepper(1, MAX_HEADER_ROW, 1)
        self.header_stepper.setToolTip("Row that holds the column names (detected automatically)")
        self.header_stepper.valueChanged.connect(self._on_header_changed)
        self.header_stepper.spin_box().editingFinished.connect(self._apply_header)
        self._header_value = self.header_stepper.value()  # as last handled: Enter re-emits it unchanged
        options.addWidget(self._sheet_label)
        options.addWidget(self.sheet_combo)
        options.addSpacing(16)
        options.addWidget(self._header_label)
        options.addWidget(self.header_stepper)
        options.addStretch(1)
        self._options = QWidget()
        self._options.setLayout(options)
        options.setContentsMargins(0, 0, 0, 0)
        self._options.setEnabled(False)
        self.body.addWidget(self._options)

        self._pages = QStackedWidget()
        self._loading_page, self._loading_text, self._loading_bar = self._build_loading_page()
        self._content_page = self._build_content_page()
        self._error_page, self._error_text = self._build_error_page()
        for page in (self._loading_page, self._content_page, self._error_page):
            self._pages.addWidget(page)
        self.body.addWidget(self._pages, 1)

        self._invalid = label(INVALID_MESSAGE, "danger")
        self._invalid.hide()
        cancel = button("Cancel")
        cancel.clicked.connect(self.reject)
        self.import_button = button("Import", "primary", icon_name="import")
        self.import_button.setEnabled(False)
        self.import_button.setDefault(True)
        self.import_button.clicked.connect(self._accept)
        row = self.add_buttons(cancel, self.import_button)
        row.insertWidget(0, self._invalid)

        self._header_timer = QTimer(self)
        self._header_timer.setSingleShot(True)
        self._header_timer.setInterval(350)
        self._header_timer.timeout.connect(self._reload_header)
        self._load(None, None)

    # ------------------------------------------------------------------ pages
    def _build_loading_page(self) -> tuple[QWidget, QLabel, ProgressLine]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)
        spinner = Spinner(28)
        layout.addWidget(spinner, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(8)
        text = label(f"Reading {self._path.name}…", "strong")
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text)
        bar = ProgressLine()
        bar.setFixedWidth(280)
        bar.set_value(None)
        layout.addWidget(bar, 0, Qt.AlignmentFlag.AlignHCenter)
        hint = label("Large workbooks take a few seconds. Nothing leaves this computer.", "caption")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
        layout.addStretch(2)
        return page, text, bar

    def _build_content_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        preview_head = QHBoxLayout()
        preview_head.addWidget(label("Preview", "strong"))
        self._preview_caption = label("", "caption")
        preview_head.addWidget(self._preview_caption)
        preview_head.addStretch(1)
        layout.addLayout(preview_head)
        self.preview_view = QTableView()
        self.preview_view.setObjectName("Preview")
        self.preview_view.setFixedHeight(150)
        self.preview_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.preview_view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.preview_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.preview_view.setWordWrap(False)
        self.preview_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview_view.verticalHeader().setDefaultSectionSize(28)
        self.preview_view.verticalHeader().setMinimumWidth(40)
        self.preview_view.verticalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_view.horizontalHeader().setFixedHeight(30)
        self.preview_view.horizontalHeader().setDefaultAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.preview_view.horizontalHeader().setHighlightSections(False)
        self.preview_view.verticalHeader().setHighlightSections(False)
        layout.addWidget(self.preview_view)
        layout.addSpacing(12)
        layout.addWidget(label("How should each column be used?", "title"))
        layout.addWidget(label(EXPLAINER, "muted", wrap=True))
        layout.addSpacing(4)
        self.editor = RoleEditor()
        self.editor.changed.connect(self._validate)
        layout.addWidget(self.editor, 1)
        return page

    def _build_error_page(self) -> tuple[QWidget, QLabel]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addStretch(1)
        layout.addWidget(IconBadge("alert-circle", 56, color="danger", background="danger_soft"), 0,
                         Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(8)
        title = label("This file can't be imported", "title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        text = label("", "muted", wrap=True, selectable=True)
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text.setMaximumWidth(520)
        layout.addWidget(text, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(2)
        return page, text

    # ------------------------------------------------------------------ loading
    def _load(self, sheet: str | None, header_row: int | None) -> None:
        self._header_timer.stop()  # this request replaces a header-row change still waiting
        if self._task is not None:
            self._task.cancel()
        self._pages.setCurrentWidget(self._loading_page)
        self._loading_bar.set_value(None)
        self._generation += 1
        generation = self._generation
        path, backend = self._path, self._backend

        def current(callback):
            def call(*args) -> None:
                if self._open and generation == self._generation:
                    callback(*args)

            return call

        self._task = self._runner.start(
            lambda progress, cancel: backend.inspect(path, sheet, header_row, progress, cancel),
            name="import-preview",
            on_done=current(self._on_loaded),
            on_failed=current(self._on_failed),
            on_progress=current(self._on_progress),
        )
        self._validate()

    def _on_progress(self, message: str, fraction: float) -> None:
        self._loading_text.setText(message if message else f"Reading {self._path.name}…")
        self._loading_bar.set_value(fraction if fraction > 0 else None)

    def _on_failed(self, message: str) -> None:
        self._task = None
        self._error_text.setText(message)
        self._pages.setCurrentWidget(self._error_page)
        self._options.setEnabled(bool(self._preview))
        self._validate()

    def _on_loaded(self, preview: ImportPreview) -> None:
        self._task = None
        self._preview = preview
        table = preview.table
        self._subtitle.setText(
            f"{self._path.name}  ·  {plural(len(table), 'row')}  ·  {plural(len(table.columns), 'column')}"
        )
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        self.sheet_combo.addItems(preview.sheets)
        self.sheet_combo.setCurrentText(preview.sheet)
        self.sheet_combo.blockSignals(False)
        multiple = len(preview.sheets) > 1
        self._sheet_label.setVisible(multiple)
        self.sheet_combo.setVisible(multiple)
        if not self._header_timer.isActive():  # else keep the row typed while this was read
            self.header_stepper.blockSignals(True)
            self.header_stepper.setValue(preview.header_row + 1)
            self.header_stepper.blockSignals(False)
            self._header_value = self.header_stepper.value()
        self._options.setEnabled(True)

        model = PreviewModel(table)
        self.preview_view.setModel(model)
        self._size_preview_columns()
        shown = min(PREVIEW_ROWS, len(table))
        self._preview_caption.setText(f"first {plural(shown, 'row')} under the header")
        self.editor.set_choices(choices_from_profiles(preview.profiles, preview.roles))
        self.import_button.setText(f"Import {plural(len(table), 'row')}")
        self._pages.setCurrentWidget(self._content_page)
        self._validate()

    def _size_preview_columns(self) -> None:
        header = self.preview_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.preview_view.resizeColumnsToContents()
        for column in range(header.count()):
            header.resizeSection(column, max(80, min(260, header.sectionSize(column) + 16)))

    def _is_current(self) -> bool:
        """True when the preview on screen is what the sheet and header-row controls ask for."""
        preview = self._preview
        return (
            preview is not None
            and self._task is None
            and not self._header_timer.isActive()
            and self._pages.currentWidget() is self._content_page
            and self.sheet_combo.currentText() == preview.sheet
            and self.header_stepper.value() == preview.header_row + 1
        )

    def _on_sheet_changed(self, sheet: str) -> None:
        if self._preview is not None and not self._is_current():
            self._load(sheet, None)

    def _on_header_changed(self, value: int) -> None:
        if value == self._header_value:
            return
        self._header_value = value
        self._header_timer.start()
        self._validate()  # Import waits for the preview of the new row

    def _apply_header(self) -> None:
        """Enter (or leaving the field) applies a typed header row without waiting for the pause."""
        if self._header_timer.isActive():
            self._reload_header()

    def _reload_header(self) -> None:
        self._header_timer.stop()
        if self._preview is None or self._is_current():
            self._validate()  # back on the row already shown
        else:
            self._load(self.sheet_combo.currentText(), self.header_stepper.value() - 1)

    # ------------------------------------------------------------------ accept
    def _validate(self) -> None:
        valid = self.editor.is_valid()
        self._invalid.setVisible(self._pages.currentWidget() is self._content_page and not valid)
        self.import_button.setEnabled(valid and self._is_current())

    def _accept(self) -> None:
        preview = self._preview
        if preview is None or not self._is_current() or not self.editor.is_valid():
            return
        self.table = preview.table
        self.roles = self.editor.roles()
        self.accept()

    def done(self, result: int) -> None:
        self._open = False
        self._header_timer.stop()
        if self._task is not None:
            self._task.cancel()
        super().done(result)
