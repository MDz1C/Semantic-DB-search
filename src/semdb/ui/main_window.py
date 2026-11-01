"""The main window: header (R7), dataset bar, collapsible result lists (R9, R10) with a details
panel, the composer (R8) and the status bar.  It orchestrates import, indexing, search,
export and reset (R11); all long work runs on worker threads (ARCHITECTURE.md §12)."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QObject, QPoint, QRect, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QFont, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__, paths
from ..core.types import MAX_STORED_RESULTS, Cancelled, HistoryEntry, Role, Roles, Table
from . import icons, theme
from .columns_editor import ColumnsDialog
from .composer import Composer, parse_queries
from .details_panel import PANEL_WIDTH, DetailsPanel
from .import_dialog import ImportDialog
from .pages import EmptyImportPage, IndexingPage, NoMatchesPage, ReadyPage
from .results_model import (
    SORT_ALPHA,
    SORT_RECENT,
    ResultsModel,
    columns_from_history,
    entry_key,
    order_columns,
    plural,
    with_history_columns,
)
from .results_view import ResultsView
from .services import Backend, SearchIndexLike, StoreLike
from .widgets import (
    Card,
    CardCorners,
    Dialog,
    Divider,
    DropOverlay,
    ElidedLabel,
    IconBadge,
    IconWidget,
    LogoMark,
    SegmentedControl,
    StatusChip,
    Stepper,
    ToastManager,
    button,
    confirm,
    label,
    menu,
    show_error,
    tool_button,
)
from .workers import Task, TaskRunner

log = logging.getLogger(__name__)

DEFAULT_K = 5
_DRAG_EVENTS = (QEvent.Type.DragEnter, QEvent.Type.DragMove, QEvent.Type.DragLeave, QEvent.Type.Drop)
LICENCES = (
    "Built with Qt for Python (LGPLv3), ONNX Runtime (MIT), Hugging Face Tokenizers (Apache 2.0), NumPy (BSD), "
    "RapidFuzz (MIT), python-calamine (MIT) and XlsxWriter (BSD-2). Embedding model: "
    "paraphrase-multilingual-MiniLM-L12-v2 by sentence-transformers (Apache 2.0). Icons: Lucide (ISC)."
)


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.2f} s"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)} min {int(seconds % 60)} s"


def format_eta(seconds: float) -> str:
    if seconds < 10:
        return "A few seconds left"
    if seconds < 60:
        return f"About {max(10, round(seconds / 5) * 5)} s left"
    return f"About {round(seconds / 60)} min left"


class EtaTracker:
    """Estimates the time left in the current progress stage."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._stage = ""
        self._start = (0.0, 0.0)
        self._last = 0.0

    def update(self, message: str, fraction: float) -> str:
        stage = message.split("…")[0].strip()
        now = time.monotonic()
        if stage != self._stage or fraction + 0.01 < self._last:
            self._stage, self._start = stage, (now, fraction)
        self._last = fraction
        started, start_fraction = self._start
        elapsed, done = now - started, fraction - start_fraction
        if elapsed < 2 or done < 0.02 or fraction >= 0.99:
            return ""
        return format_eta(elapsed / done * (1.0 - fraction))


def example_queries(table: Table, roles: Roles, count: int = 3) -> list[str]:
    """A few short queries taken from the data, for the "Try an example" chips."""
    search = [c for c in table.columns if roles.get(c) == Role.SEARCH]
    column = search[0] if search else (table.columns[0] if table.columns else None)
    if column is None or not table.rows:
        return []
    j = table.columns.index(column)
    picks: list[str] = []
    for fraction in (0.11, 0.47, 0.83, 0.29, 0.65):
        start = int(len(table.rows) * fraction)
        for row in table.rows[start:start + 60]:
            words = row[j].split()
            if len(words) < 2:
                continue
            text = " ".join(words[:4]).rstrip(" -–,;:(")
            if len(text) > 34:
                text = text[:34].rsplit(" ", 1)[0]
            if text and text not in picks:
                picks.append(text)
                break
        if len(picks) == count:
            break
    return picks


def reveal_in_file_manager(path: Path) -> None:
    if sys.platform == "win32" and path.exists():
        subprocess.Popen(["explorer", "/select,", str(path)])  # noqa: S603,S607 - local file manager
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))


class MainWindow(QMainWindow):
    """``store`` persists everything; ``backend`` reaches the search core (fakes in tests)."""

    def __init__(
        self,
        store: StoreLike,
        backend: Backend | None = None,
        *,
        autostart: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.backend = backend or Backend()
        self.runner = TaskRunner(self)
        self._table: Table | None = None
        self._roles: Roles | None = None
        self._index: SearchIndexLike | None = None
        self._embedder = None
        self._embedder_state = "pending"  # pending | loading | ready | missing
        self._history_loaded = False
        self._dataset_loading = False
        self._started = False
        self._build_task: Task | None = None
        self._build_generation = 0
        self._build_state = "idle"  # idle | waiting | building | ready | cancelled | failed
        self._build_waiting = False
        self._build_notify = False
        self._build_started = 0.0
        self._build_progress = ("", 0.0)
        self._build_eta = ""
        self._build_error = ""
        self._eta = EtaTracker()
        self._search_task: Task | None = None
        self._search_started = 0.0
        self._entries: list[HistoryEntry] = []

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(icons.app_icon())
        self.setMinimumSize(880, 560)
        self.setAcceptDrops(True)
        self.model = ResultsModel(self)
        self._build_ui()
        self._build_actions()
        self._connect()
        self._restore_settings()
        self._drop_hide = QTimer(self)
        self._drop_hide.setSingleShot(True)
        self._drop_hide.setInterval(80)
        self._drop_hide.timeout.connect(self.drop_overlay.hide)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self._minute = QTimer(self)
        self._minute.setInterval(60_000)
        self._minute.timeout.connect(self._tick_minute)
        self._minute.start()
        theme.manager().changed.connect(self._on_theme_changed)
        self._refresh()
        if autostart:
            QTimer.singleShot(0, self.start)

    # ================================================================== layout
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("Central")
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_header())
        self.dataset_bar = self._build_dataset_bar()
        root.addWidget(self.dataset_bar)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(24, 16, 24, 16)
        body_layout.setSpacing(0)
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.vsplit.setHandleWidth(16)
        self.results_area = QWidget()
        area = QVBoxLayout(self.results_area)
        area.setContentsMargins(0, 0, 0, 0)
        area.setSpacing(12)
        area.addLayout(self._build_toolbar())
        self.hsplit = QSplitter(Qt.Orientation.Horizontal)
        self.hsplit.setChildrenCollapsible(False)
        self.hsplit.setHandleWidth(16)
        self.results_card = Card()
        card_layout = QVBoxLayout(self.results_card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        self.pages = QStackedWidget()
        self.view = ResultsView(self.model)
        self.empty_page = EmptyImportPage()
        self.ready_page = ReadyPage()
        self.indexing_page = IndexingPage()
        self.no_matches_page = NoMatchesPage()
        self.loading_page = QWidget()
        for page in (self.view, self.empty_page, self.ready_page, self.indexing_page, self.no_matches_page,
                     self.loading_page):
            self.pages.addWidget(page)
        card_layout.addWidget(self.pages)
        CardCorners(self.results_card)
        self.details = DetailsPanel()
        self.hsplit.addWidget(self.results_card)
        self.hsplit.addWidget(self.details)
        self.hsplit.setStretchFactor(0, 1)
        self.hsplit.setStretchFactor(1, 0)
        area.addWidget(self.hsplit, 1)
        self.composer = Composer()
        self.composer.setMinimumHeight(118)
        self.vsplit.addWidget(self.results_area)
        self.vsplit.addWidget(self.composer)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 0)
        body_layout.addWidget(self.vsplit)
        root.addWidget(body, 1)
        self.setCentralWidget(central)
        self.setStatusBar(self._build_status_bar())
        self.toasts = ToastManager(central, self._toast_area)
        self.drop_overlay = DropOverlay(central)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(64)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 20, 0)
        layout.setSpacing(0)
        layout.addWidget(LogoMark(28))
        layout.addSpacing(12)
        name = QLabel(APP_NAME)
        name.setFont(theme.font(11.5, QFont.Weight.DemiBold))
        layout.addWidget(name)
        layout.addSpacing(24)
        layout.addWidget(Divider(vertical=True, length=24))
        layout.addSpacing(24)
        layout.addWidget(label("Results per query", "muted"))
        layout.addSpacing(12)
        self.k_stepper = Stepper(1, MAX_STORED_RESULTS, DEFAULT_K)
        self.k_stepper.setToolTip("How many matches each query list shows (up to 50 are kept)")
        layout.addWidget(self.k_stepper)
        layout.addStretch(1)
        self.import_button = button("Import", "outline", icon_name="import", tooltip="Import a spreadsheet (Ctrl+O)")
        self.reset_button = button("Reset", "ghost", icon_name="rotate-ccw", tooltip="Clear all result lists")
        self.more_button = tool_button("ellipsis", "More options")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        layout.addWidget(self.import_button)
        layout.addSpacing(8)
        layout.addWidget(self.reset_button)
        layout.addSpacing(4)
        layout.addWidget(self.more_button)
        return header

    def _build_dataset_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("DatasetBar")
        bar.setFixedHeight(52)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(0)
        layout.addWidget(IconBadge("sheet", 30, radius=8, icon_ratio=0.5))
        layout.addSpacing(12)
        self.dataset_name = ElidedLabel("")
        self.dataset_name.setProperty("role", "strong")
        self.dataset_name.setMaximumWidth(360)
        layout.addWidget(self.dataset_name)
        layout.addSpacing(12)
        self.dataset_rows = label("", "caption")
        layout.addWidget(self.dataset_rows)
        layout.addSpacing(16)
        self.dataset_chip = StatusChip()
        layout.addWidget(self.dataset_chip)
        layout.addStretch(1)
        self.deciding_caption = label("Deciding", "caption")
        layout.addWidget(self.deciding_caption)
        layout.addSpacing(8)
        holder = QWidget()
        self._pills = QHBoxLayout(holder)
        self._pills.setContentsMargins(0, 0, 0, 0)
        self._pills.setSpacing(6)
        layout.addWidget(holder)
        layout.addSpacing(16)
        self.columns_button = button("Columns…", icon_name="sliders", height=32,
                                     tooltip="Choose deciding and search columns")
        layout.addWidget(self.columns_button)
        return bar

    def _build_toolbar(self) -> QHBoxLayout:
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        toolbar.addWidget(label("Results", "title"))
        self.results_count = QLabel("0")
        self.results_count.setProperty("pill", "count")
        self.results_count.setFixedHeight(20)
        self.results_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        toolbar.addWidget(self.results_count)
        toolbar.addSpacing(12)
        self.sort_control = SegmentedControl([(SORT_RECENT, "Recent"), (SORT_ALPHA, "A–Z")])
        self.sort_control.setToolTip("Order the query lists by recency or alphabetically")
        toolbar.addWidget(self.sort_control)
        self.collapse_button = button("Collapse all", "ghost", icon_name="chevrons-down-up",
                                      tooltip="Collapse or expand every list (Ctrl+Shift+C)")
        toolbar.addWidget(self.collapse_button)
        toolbar.addStretch(1)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter queries")
        self.filter_edit.setFixedHeight(34)
        self.filter_edit.setFixedWidth(240)
        self.filter_edit.setToolTip("Show only queries containing these words (Ctrl+F)")
        search_action = self.filter_edit.addAction(icons.icon("search"), QLineEdit.ActionPosition.LeadingPosition)
        icons.bind(search_action, "search", "text_faint")
        self._clear_filter = self.filter_edit.addAction(icons.icon("x"), QLineEdit.ActionPosition.TrailingPosition)
        icons.bind(self._clear_filter, "x", "text_muted")
        self._clear_filter.setToolTip("Clear filter")
        self._clear_filter.setVisible(False)
        toolbar.addWidget(self.filter_edit)
        self.export_button = button("Export", icon_name="download",
                                    tooltip="Save the visible lists as an Excel workbook (Ctrl+E)")
        toolbar.addWidget(self.export_button)
        self.details_button = tool_button("panel-right", "Show or hide the details panel", checkable=True)
        toolbar.addWidget(self.details_button)
        return toolbar

    def _build_status_bar(self) -> QStatusBar:
        status = QStatusBar()
        status.setSizeGripEnabled(False)
        status.setFixedHeight(30)
        left = QWidget()
        left_layout = QHBoxLayout(left)
        left_layout.setContentsMargins(16, 0, 0, 0)
        self.status_text = QLabel("")
        left_layout.addWidget(self.status_text)
        status.addWidget(left, 1)
        right = QWidget()
        layout = QHBoxLayout(right)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(6)
        layout.addWidget(IconWidget("lock", "text_muted", 13))
        offline = QLabel("Offline · data stays on this PC")
        offline.setToolTip("No internet connection is used. Files, searches and settings never leave this computer.")
        layout.addWidget(offline)
        layout.addSpacing(10)
        layout.addWidget(Divider(vertical=True, length=14))
        layout.addSpacing(10)
        self.model_icon = IconWidget("sparkles", "accent", 13)
        layout.addWidget(self.model_icon)
        self.model_text = QLabel("")
        layout.addWidget(self.model_text)
        status.addPermanentWidget(right)
        return status

    def _build_actions(self) -> None:
        def action(text: str, shortcut: str | QKeySequence.StandardKey, slot) -> QAction:
            act = QAction(text, self)
            act.setShortcut(QKeySequence(shortcut))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            act.triggered.connect(slot)
            self.addAction(act)
            return act

        self.import_action = action("Import…", "Ctrl+O", self.choose_file)
        self.export_action = action("Export…", "Ctrl+E", lambda: self.export_results())
        self.collapse_action = action("Collapse or expand all", "Ctrl+Shift+C", self.toggle_collapse_all)
        self.filter_action = action("Filter queries", "Ctrl+F", self._focus_filter)
        for keys in ("Ctrl+Return", "Ctrl+Enter"):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.activated.connect(self.composer.submit)

        more = menu(self)
        theme_menu = menu(more, "Theme")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.theme_actions: dict[str, QAction] = {}
        for mode, text, icon_name in (("system", "System", "monitor"), ("light", "Light", "sun"),
                                      ("dark", "Dark", "moon")):
            act = theme_menu.addAction(text)
            icons.bind(act, icon_name)
            act.setCheckable(True)
            act.triggered.connect(lambda _checked=False, m=mode: self.set_theme(m))
            group.addAction(act)
            self.theme_actions[mode] = act
        icons.bind(theme_menu.menuAction(), "sun")
        more.addMenu(theme_menu)
        folder = more.addAction("Open data folder")
        icons.bind(folder, "folder-open")
        folder.triggered.connect(self.open_data_folder)
        more.addSeparator()
        about = more.addAction(f"About {APP_NAME}")
        icons.bind(about, "info")
        about.triggered.connect(self.show_about)
        self.more_button.setMenu(more)

    def _connect(self) -> None:
        self.k_stepper.valueChanged.connect(self._on_k_changed)
        self.import_button.clicked.connect(self.choose_file)
        self.reset_button.clicked.connect(self.reset_results)
        self.columns_button.clicked.connect(self.edit_columns)
        self.sort_control.changed.connect(self._on_sort_changed)
        self.collapse_button.clicked.connect(self.toggle_collapse_all)
        self.filter_edit.textChanged.connect(self._on_filter_changed)
        self._clear_filter.triggered.connect(self.filter_edit.clear)
        self.export_button.clicked.connect(lambda: self.export_results())
        self.details_button.toggled.connect(self._on_details_toggled)
        self.view.focus_changed.connect(self._on_focus_changed)
        self.view.rerun_requested.connect(self.run_search)
        self.view.delete_requested.connect(self.delete_lists)
        self.view.expansion_changed.connect(self._sync_collapse_button)
        self.details.rerun_requested.connect(self.run_search)
        self.details.delete_requested.connect(self.delete_lists)
        self.composer.search_requested.connect(lambda queries: self.run_search(queries, from_composer=True))
        self.composer.cancel_requested.connect(self.cancel_search)
        self.empty_page.import_requested.connect(self.choose_file)
        self.empty_page.sample_requested.connect(self.import_sample)
        self.ready_page.example_clicked.connect(self._use_example)
        self.indexing_page.cancel_requested.connect(self.cancel_build)
        self.indexing_page.retry_requested.connect(lambda: self._request_build(notify=True))
        self.no_matches_page.clear_requested.connect(self.filter_edit.clear)

    # ================================================================== settings
    def _setting(self, key: str, default=None):
        try:
            value = self.store.get_setting(key, default)
        except Exception:
            log.exception("Could not read setting %s", key)
            return default
        return default if value is None else value

    def _save_setting(self, key: str, value) -> None:
        try:
            self.store.set_setting(key, value)
        except Exception:
            log.exception("Could not save setting %s", key)

    def _restore_settings(self) -> None:
        try:
            k = int(self._setting("results_per_query", DEFAULT_K))
        except (TypeError, ValueError):
            k = DEFAULT_K
        k = max(1, min(MAX_STORED_RESULTS, k))
        self.k_stepper.setValue(k)
        self.model.set_k(k)
        sort = self._setting("sort_mode", SORT_RECENT)
        sort = sort if sort in (SORT_RECENT, SORT_ALPHA) else SORT_RECENT
        self.sort_control.set_value(sort)
        self.model.set_sort_mode(sort)
        self.details_button.setChecked(bool(self._setting("details_visible", True)))
        mode = theme.manager().mode
        if mode in self.theme_actions:
            self.theme_actions[mode].setChecked(True)
        geometry = self._setting("window_geometry")
        restored = False
        if isinstance(geometry, str) and geometry:
            restored = self.restoreGeometry(QByteArray.fromBase64(geometry.encode("ascii")))
        if not restored:
            screen = self.screen() or QGuiApplication.primaryScreen()
            available = screen.availableGeometry() if screen else QRect(0, 0, 1440, 900)
            width = min(1320, int(available.width() * 0.94))
            height = min(860, int(available.height() * 0.94))
            self.resize(width, height)
            self.move(available.center() - QPoint(width // 2, height // 2))
        vertical = self._setting("splitter_main")
        horizontal = self._setting("splitter_results")
        height = max(self.height(), 560)
        width = max(self.width(), 880)
        self.vsplit.setSizes(vertical if self._valid_sizes(vertical) else [height - 330, 150])
        self.hsplit.setSizes(horizontal if self._valid_sizes(horizontal) else [width - 48 - 16 - PANEL_WIDTH,
                                                                             PANEL_WIDTH])

    @staticmethod
    def _valid_sizes(value) -> bool:
        return isinstance(value, list) and len(value) == 2 and all(isinstance(v, int) and v > 0 for v in value)

    def _save_window_state(self) -> None:
        self._save_setting("window_geometry", bytes(self.saveGeometry().toBase64()).decode("ascii"))
        self._save_setting("splitter_main", self.vsplit.sizes())
        if self.details.isVisible():
            self._save_setting("splitter_results", self.hsplit.sizes())

    # ================================================================== startup
    def start(self) -> None:
        """Load history, the stored database and the language model in the background."""
        if self._started:
            return
        self._started = True
        self._dataset_loading = True
        self._embedder_state = "loading"
        self._set_status("Opening…")
        self.runner.start(lambda _p, _c: self.store.load_entries(), name="load-history",
                          on_done=self._on_history_loaded, on_failed=self._on_history_failed)
        self.runner.start(lambda _p, _c: self.store.load_dataset(), name="load-dataset",
                          on_done=self._on_dataset_loaded, on_failed=self._on_dataset_failed)
        self.runner.start(lambda _p, _c: self.backend.load_embedder(), name="load-model",
                          on_done=self._on_embedder_loaded, on_failed=self._on_embedder_failed)
        self._refresh()

    def _on_history_loaded(self, entries: list[HistoryEntry]) -> None:
        self.set_entries(entries)
        if entries:
            self._set_status(f"{plural(len(entries), 'result list')} restored")

    def _on_history_failed(self, message: str) -> None:
        self.set_entries([])
        self.toasts.show(f"Your previous results could not be loaded. {message}", "error", timeout_ms=8000)

    def _on_dataset_loaded(self, stored: tuple[Table, Roles] | None) -> None:
        self._dataset_loading = False
        if stored:
            table, roles = stored
            self._set_dataset(table, roles)
            self._request_build(notify=False)
        else:
            self._refresh()
            if not self._entries:
                self._set_status("Import a database to get started")

    def _on_dataset_failed(self, message: str) -> None:
        self._dataset_loading = False
        self._refresh()
        self.toasts.show(f"The saved database could not be opened. Import it again. {message}", "error",
                         timeout_ms=8000)

    def _on_embedder_loaded(self, embedder) -> None:
        self._embedder = embedder
        self._embedder_state = "ready" if embedder is not None else "missing"
        if self._build_waiting:
            self._start_build()
        self._refresh()

    def _on_embedder_failed(self, message: str) -> None:
        log.warning("Language model unavailable: %s", message)
        self._on_embedder_loaded(None)

    # ================================================================== public state
    def set_entries(self, entries: list[HistoryEntry]) -> None:
        """Replace the displayed history (oldest → newest, as the store returns it)."""
        self._history_loaded = True
        self._entries = list(entries)
        self.model.set_entries(self._entries)
        self._refresh_columns()
        self._refresh()

    def entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def set_index(self, index: SearchIndexLike | None) -> None:
        """Use a ready index (tests and tools; the app builds its own)."""
        self._index = index
        self._build_state = "ready" if index is not None else "idle"
        self._refresh_columns()
        self._refresh()

    def use_dataset(self, table: Table, roles: Roles, *, save: bool = True) -> None:
        """Adopt an imported table: save it and build its index in the background."""
        self._index = None
        self._set_dataset(table, roles)
        if save:
            self.runner.start(lambda _p, _c: self.store.save_dataset(table, roles), name="save-dataset",
                              on_failed=lambda m: self.toasts.show(f"The database could not be saved. {m}", "error"))
        self._set_status(f"Imported {plural(len(table), 'row')} from {table.label}")
        self._request_build(notify=True)

    def _set_dataset(self, table: Table, roles: Roles) -> None:
        self._table, self._roles = table, dict(roles)
        self.dataset_name.setText(table.label)
        self.dataset_rows.setText(f"{plural(len(table), 'row')}  ·  {plural(len(table.columns), 'column')}")
        while self._pills.count():
            item = self._pills.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        deciding = [c for c in table.columns if self._roles.get(c) == Role.DECIDING]
        for name in deciding[:4]:
            pill = QLabel(name)
            pill.setProperty("pill", "accent")
            pill.setFixedHeight(22)
            self._pills.addWidget(pill)
        if len(deciding) > 4:
            more = QLabel(f"+{len(deciding) - 4}")
            more.setProperty("pill", "neutral")
            more.setFixedHeight(22)
            more.setToolTip(", ".join(deciding[4:]))
            self._pills.addWidget(more)
        if not deciding:
            self._pills.addWidget(label("none", "caption"))
        self.ready_page.set_dataset(len(table), example_queries(table, self._roles))
        self._refresh_columns()
        self._refresh()

    # ================================================================== indexing
    def _request_build(self, notify: bool) -> None:
        if self._table is None:
            return
        self._build_notify = self._build_notify or notify
        if self._build_task is not None:
            self._build_task.cancel()
            self._build_task = None
        if self._embedder_state in ("pending", "loading"):
            self._build_waiting = True
            self._build_state = "waiting"
            self._refresh()
            return
        self._start_build()

    def _start_build(self) -> None:
        if self._table is None or self._roles is None:
            return
        self._build_waiting = False
        if self._build_task is not None:
            self._build_task.cancel()
        self._build_generation += 1
        generation = self._build_generation
        table, roles, embedder = self._table, dict(self._roles), self._embedder
        cache = self._embedding_cache() if embedder is not None else None
        self._build_state = "building"
        self._build_progress = ("Analyzing columns…", 0.0)
        self._build_eta = ""
        self._build_started = time.monotonic()
        self._eta.reset()
        self._build_task = self.runner.start(
            lambda progress, cancel: self.backend.build_index(table, roles, embedder, cache, progress, cancel),
            name="build-index",
            on_done=lambda index: self._on_build_done(generation, index),
            on_failed=lambda message: self._on_build_failed(generation, message),
            on_progress=lambda stage, fraction: self._on_build_progress(generation, stage, fraction),
            on_cancelled=lambda: self._on_build_cancelled(generation),
        )
        self._set_status("Preparing the search index…")
        self._refresh()

    def _embedding_cache(self):
        try:
            return self.store.embedding_cache
        except Exception:
            log.exception("Embedding cache unavailable; continuing without it")
            return None

    def _on_build_progress(self, generation: int, stage: str, fraction: float) -> None:
        if generation != self._build_generation or self._build_state != "building":
            return
        self._build_progress = (stage, fraction)
        self._build_eta = self._eta.update(stage, fraction)
        self._update_build_widgets()

    def _on_build_done(self, generation: int, index: SearchIndexLike) -> None:
        if generation != self._build_generation:
            return
        self._build_task = None
        self._index = index
        self._build_state = "ready"
        elapsed = time.monotonic() - self._build_started
        rows = len(index.table) if getattr(index, "table", None) is not None else len(self._table or [])
        self._set_status(f"Ready · {plural(rows, 'row')} indexed in {format_seconds(elapsed)}")
        if self._build_notify:
            self.toasts.show("Index ready — you can search now.", "success")
        self._build_notify = False
        self._refresh_columns()
        self._refresh()

    def _on_build_failed(self, generation: int, message: str) -> None:
        if generation != self._build_generation:
            return
        self._build_task = None
        self._build_state = "failed"
        self._build_error = message
        self._build_notify = False
        self._set_status("The search index could not be built")
        self.toasts.show(message, "error", "Try again", lambda: self._request_build(notify=True), timeout_ms=8000)
        self._refresh()

    def _on_build_cancelled(self, generation: int) -> None:
        if generation != self._build_generation:
            return
        self._build_task = None
        self._build_state = "cancelled"
        self._build_notify = False
        self._set_status("Indexing cancelled")
        self._refresh()

    def cancel_build(self) -> None:
        if self._build_task is not None:
            self._build_task.cancel()

    def is_building(self) -> bool:
        return self._build_state in ("waiting", "building")

    # ================================================================== search
    def run_search(self, queries: list[str], from_composer: bool = False) -> None:
        """Search ``queries`` (deduplicated) and store one list per query."""
        queries = parse_queries("\n".join(queries))[0]
        index = self._index
        if not queries or index is None or self._search_task is not None:
            return
        store = self.store
        self._search_started = time.perf_counter()
        self.composer.set_busy(True, 0.0)
        self._set_status(f"Searching {plural(len(queries), 'query', 'queries')}…")

        def job(progress, cancel):
            outcomes = index.search_many(queries, k=MAX_STORED_RESULTS, progress=progress, cancel=cancel)
            if cancel():
                raise Cancelled()
            created = datetime.now(timezone.utc).isoformat()
            return store.add_entries([index.to_history_entry(outcome, created) for outcome in outcomes])

        self._search_task = self.runner.start(
            job,
            name="search",
            on_done=lambda saved: self._on_search_done(saved, len(queries), from_composer),
            on_failed=self._on_search_failed,
            on_progress=lambda _stage, fraction: self.composer.set_busy(True, fraction),
            on_cancelled=self._on_search_cancelled,
        )
        self._refresh()

    def _on_search_done(self, saved: list[HistoryEntry], count: int, from_composer: bool) -> None:
        self._search_task = None
        elapsed = time.perf_counter() - self._search_started
        keys = {entry_key(e) for e in saved}
        self._entries = [e for e in self._entries if entry_key(e) not in keys] + list(saved)
        if from_composer:
            self.composer.clear()
        self.composer.set_busy(False)
        if self.filter_edit.text():
            self.filter_edit.blockSignals(True)
            self.filter_edit.clear()
            self.filter_edit.blockSignals(False)
            self._clear_filter.setVisible(False)
            self.model.set_filter("")
        self.model.set_entries(self._entries)
        self._refresh_columns()
        self._refresh()
        visible = self.model.visible_entries()
        first = next((e for e in visible if entry_key(e) in keys), None)
        if first is not None:
            self.view.reveal(entry_key(first), expand=count == 1)
            self.view.setFocus()
        self._set_status(f"Searched {plural(count, 'query', 'queries')} in {format_seconds(elapsed)}")

    def _on_search_failed(self, message: str) -> None:
        self._search_task = None
        self.composer.set_busy(False)
        self._set_status("Search failed")
        self.toasts.show(message, "error", timeout_ms=8000)
        self._refresh()

    def _on_search_cancelled(self) -> None:
        self._search_task = None
        self.composer.set_busy(False)
        self._set_status("Search cancelled")
        self._refresh()

    def cancel_search(self) -> None:
        if self._search_task is not None:
            self._search_task.cancel()

    def is_searching(self) -> bool:
        return self._search_task is not None

    def _use_example(self, text: str) -> None:
        self.composer.set_text(text)
        self.composer.editor.setFocus()
        self.composer.editor.moveCursor(self.composer.editor.textCursor().MoveOperation.End)

    # ================================================================== lists
    def delete_lists(self, keys: list[str]) -> None:
        keyset = set(keys)
        doomed = [e for e in self._entries if entry_key(e) in keyset]
        if not doomed:
            return
        count = len(doomed)
        if count == 1:
            title, text = "Delete this result list?", f"“{doomed[0].query}” and its results will be removed from this computer."
        else:
            title, text = f"Delete {count} result lists?", f"{count} queries and their results will be removed from this computer."
        if not confirm(self, title, text, "Delete", icon_name="trash"):
            return
        ids = [e.id for e in doomed if e.id is not None]
        try:
            if ids:
                self.store.delete_entries(ids)
        except Exception:
            log.exception("Deleting entries failed")
            self.toasts.show("The lists could not be deleted.", "error")
            return
        self._entries = [e for e in self._entries if entry_key(e) not in keyset]
        self.view.forget_keys(list(keyset))
        self.model.set_entries(self._entries)
        self._refresh_columns()
        self._refresh()
        self._set_status(f"Deleted {plural(count, 'result list')}")

    def reset_results(self) -> None:
        count = len(self._entries)
        if not count:
            self.composer.clear()
            return
        if not confirm(
            self,
            "Clear all results?",
            f"{plural(count, 'result list')} will be removed from this computer. Your database stays loaded.",
            "Clear results",
            icon_name="rotate-ccw",
        ):
            return
        try:
            self.store.clear_entries()
        except Exception:
            log.exception("Clearing history failed")
            self.toasts.show("The results could not be cleared.", "error")
            return
        self._entries = []
        self.composer.clear()
        self.view.collapse_all_lists()
        self.model.set_entries([])
        self._refresh_columns()
        self._refresh()
        self._set_status(f"Cleared {plural(count, 'result list')}")

    def toggle_collapse_all(self) -> None:
        if self.view.any_expanded():
            self.view.collapse_all_lists()
        else:
            self.view.expand_all_lists()

    def _sync_collapse_button(self) -> None:
        expanded = self.view.any_expanded()
        self.collapse_button.setText("Collapse all" if expanded else "Expand all")
        icons.bind(self.collapse_button, "chevrons-down-up" if expanded else "chevrons-up-down")

    # ================================================================== export
    def export_results(self, path: str | Path | None = None) -> None:
        """Export the visible lists, in visible order, with the current results per query."""
        entries = self.model.visible_entries()
        if not entries:
            self.toasts.show("There is nothing to export yet.", "info")
            return
        if path is None:
            folder = self._setting("last_export_dir") or QStandardPaths.writableLocation(
                QStandardPaths.StandardLocation.DocumentsLocation)
            suggested = Path(folder) / f"{APP_NAME} results {date.today():%Y-%m-%d}.xlsx"
            chosen, _ = QFileDialog.getSaveFileName(self, "Export results", str(suggested), "Excel workbook (*.xlsx)")
            if not chosen:
                return
            path = chosen
        target = Path(path)
        if target.suffix.lower() != ".xlsx":
            target = target.with_name(target.name + ".xlsx")
        self._save_setting("last_export_dir", str(target.parent))
        k = self.model.k
        self.export_button.setEnabled(False)
        self._set_status(f"Exporting {plural(len(entries), 'list')}…")
        self.runner.start(
            lambda _p, _c: self.backend.export_history(target, entries, k),
            name="export",
            on_done=lambda _result: self._on_exported(target, len(entries)),
            on_failed=self._on_export_failed,
            on_finished=self._refresh,
        )

    def _on_exported(self, target: Path, count: int) -> None:
        self._set_status(f"Exported {plural(count, 'query', 'queries')} to {target.name}")
        self.toasts.show(f"Exported {plural(count, 'query', 'queries')}", "success", "Open folder",
                         lambda: reveal_in_file_manager(target))

    def _on_export_failed(self, message: str) -> None:
        self._set_status("Export failed")
        self.toasts.show(message, "error", timeout_ms=8000)

    # ================================================================== import
    def choose_file(self) -> None:
        folder = self._setting("last_import_dir") or QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation)
        patterns = " ".join(f"*{ext}" for ext in self.backend.supported_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Import a database", str(folder),
                                              f"Spreadsheets ({patterns});;All files (*)")
        if path:
            self._save_setting("last_import_dir", str(Path(path).parent))
            self.import_file(path)

    def import_sample(self) -> None:
        path = Path(self.backend.sample_workbook())
        if not path.is_file():
            self.toasts.show("The sample workbook is not included in this installation.", "error")
            return
        self.import_file(path)

    def import_file(self, path: str | Path) -> bool:
        """Open the import dialog for ``path``; returns True when a database was imported."""
        path = Path(path)
        if path.suffix.lower() not in self.backend.supported_extensions():
            show_error(self, "This file type is not supported",
                       f"{path.name} is not a spreadsheet this app can read. Choose an Excel (.xlsx, .xlsm, .xlsb, "
                       ".xls), OpenDocument (.ods) or CSV/TSV file.")
            return False
        dialog = ImportDialog(path, self.backend, self.runner, self)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        table, roles = dialog.table, dialog.roles
        dialog.deleteLater()
        if not accepted or table is None or roles is None:
            return False
        self.use_dataset(table, roles)
        return True

    def edit_columns(self) -> None:
        if self._table is None or self._roles is None:
            return
        aliases: dict[str, dict[str, list[str]]] = {}
        if self._index is not None:
            try:
                aliases = self._index.learned_aliases()
            except Exception:
                log.exception("Could not read learned aliases")
        dialog = ColumnsDialog(self, self._table, self._roles, self.backend, self.runner, aliases)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        roles = dialog.roles()
        dialog.deleteLater()
        if accepted:
            self.apply_roles(roles)

    def apply_roles(self, roles: Roles) -> None:
        if self._table is None or roles == self._roles:
            return
        self._index = None
        self.runner.start(lambda _p, _c: self.store.save_roles(roles), name="save-roles",
                          on_failed=lambda m: self.toasts.show(f"The column choice could not be saved. {m}", "error"))
        self._set_dataset(self._table, roles)
        self._request_build(notify=True)

    # ================================================================== misc actions
    def set_theme(self, mode: str) -> None:
        theme.manager().set_mode(mode)  # type: ignore[arg-type]
        self._save_setting("theme", theme.manager().mode)
        action = self.theme_actions.get(theme.manager().mode)
        if action is not None:
            action.setChecked(True)

    def open_data_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.data_dir())))

    def show_about(self) -> None:
        dialog = Dialog(self)
        dialog.setFixedWidth(500)
        head = QHBoxLayout()
        head.setSpacing(16)
        head.addWidget(LogoMark(48), 0, Qt.AlignmentFlag.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(label(APP_NAME, "heading"))
        titles.addWidget(label(f"Version {__version__}", "caption"))
        head.addLayout(titles, 1)
        dialog.body.addLayout(head)
        dialog.body.addWidget(label(
            "Runs entirely on this computer. It never connects to the internet: no web server, no open "
            "ports, no uploads. Your databases, searches and settings stay in your user folder:", "muted", wrap=True))
        dialog.body.addWidget(label(str(paths.data_dir()), "caption", wrap=True, selectable=True))
        dialog.body.addWidget(label(LICENCES, "faint", wrap=True))
        close = button("Close", "primary")
        close.clicked.connect(dialog.accept)
        dialog.add_buttons(close)
        dialog.exec()
        dialog.deleteLater()

    # ================================================================== reactions
    def _on_k_changed(self, value: int) -> None:
        self.model.set_k(value)
        self._save_setting("results_per_query", value)
        entry, result = self.details.current()
        if entry is not None:
            self._show_details(entry, result)

    def _on_sort_changed(self, mode: str) -> None:
        self.model.set_sort_mode(mode)
        self._save_setting("sort_mode", mode)
        self.view.scrollToTop()

    def _on_filter_changed(self, text: str) -> None:
        self._clear_filter.setVisible(bool(text))
        self.model.set_filter(text)
        self.no_matches_page.set_filter(text)
        self._refresh()

    def _focus_filter(self) -> None:
        if self.filter_edit.isVisible():
            self.filter_edit.setFocus()
            self.filter_edit.selectAll()

    def _on_details_toggled(self, visible: bool) -> None:
        self._save_setting("details_visible", visible)
        self._refresh()

    def _on_focus_changed(self, entry: HistoryEntry | None, result) -> None:
        self._show_details(entry, result)

    def _show_details(self, entry: HistoryEntry | None, result) -> None:
        key = self.model.key_of(entry) if entry is not None else ""
        self.details.show_entry(entry, result, self.model.columns(), self.model.k, key)

    def _on_theme_changed(self) -> None:
        self.details.refresh()
        self.view.viewport().update()

    def _tick_minute(self) -> None:
        self.model.refresh_times()
        entry, result = self.details.current()
        if entry is not None and result is None:
            self.details.refresh()

    def _toast_area(self) -> QRect:
        central = self.centralWidget()
        return QRect(self.results_area.mapTo(central, QPoint(0, 0)), self.results_area.size())

    def _set_status(self, text: str) -> None:
        self.status_text.setText(text)

    # ================================================================== refresh
    def _refresh_columns(self) -> None:
        if self._table is not None:
            primary = self._index.primary_column if self._index is not None else None
            base = order_columns(self._table.columns, self._roles, primary)
        else:
            base = columns_from_history(self._entries)
        self.model.set_columns(with_history_columns(base, self._entries))
        self.view.apply_column_widths()

    def _current_page(self) -> QWidget:
        if not self._history_loaded:
            return self.loading_page
        if self._entries:
            return self.view if self.model.visible_entries() else self.no_matches_page
        if self._table is None:
            return self.loading_page if self._dataset_loading else self.empty_page
        if self._index is not None:
            return self.ready_page
        return self.indexing_page

    def _refresh(self) -> None:
        has_entries = bool(self._entries)
        visible = self.model.visible_entries()
        # A model reset (delete, filter) clears the view's current index without a focus signal:
        # never leave the details panel showing a list that is gone or hidden.
        shown, _ = self.details.current()
        if shown is not None and self.model.key_of(shown) not in {self.model.key_of(e) for e in visible}:
            self._show_details(None, None)
        for widget in (self.results_count, self.sort_control, self.collapse_button, self.filter_edit,
                       self.export_button, self.details_button):
            widget.setVisible(has_entries)
        total = len(self._entries)
        self.results_count.setText(f"{len(visible):,} of {total:,}" if len(visible) != total else f"{total:,}")
        self.export_button.setEnabled(bool(visible) and not self._export_running())
        self.reset_button.setEnabled(has_entries)
        page = self._current_page()
        self.pages.setCurrentWidget(page)
        self.details.setVisible(page is self.view and self.details_button.isChecked())
        self.dataset_bar.setVisible(self._table is not None)
        self.columns_button.setEnabled(self._table is not None)
        self._sync_collapse_button()
        self._update_build_widgets()
        ready = self._index is not None
        if self._table is None:
            reason = "Import a database to start searching" if not self._dataset_loading else "Opening your database…"
        elif self._build_state in ("waiting", "building"):
            reason = "Preparing the search index…"
        elif self._build_state == "failed":
            reason = "The search index could not be built"
        elif self._build_state == "cancelled":
            reason = "Build the search index to start searching"
        else:
            reason = ""
        self.composer.set_ready(ready, reason)
        self.view.can_rerun = ready
        self.details.can_rerun = ready
        self._update_model_status()

    def _export_running(self) -> bool:
        return any(task.name == "export" for task in self.runner.active())

    def _update_build_widgets(self) -> None:
        stage, fraction = self._build_progress
        if self._dataset_loading and self._table is None:
            self.indexing_page.show_waiting("Opening your database…")
        if self._table is None:
            return
        state = self._build_state
        if state == "waiting":
            self.dataset_chip.set_state("working", "Preparing…")
            self.indexing_page.show_waiting("Loading the language model…")
        elif state == "building":
            self.dataset_chip.set_state("working", f"Indexing {round(fraction * 100)}%", fraction)
            self.indexing_page.show_progress(stage or "Preparing…", fraction, self._build_eta)
        elif state == "ready" and self._index is not None:
            self.dataset_chip.set_state("ready", "Ready")
        elif state == "failed":
            self.dataset_chip.set_state("error", "Index failed")
            self.indexing_page.show_stopped("The index could not be built", self._build_error, failed=True,
                                            button_text="Try again")
        elif state == "cancelled":
            self.dataset_chip.set_state("idle", "Not indexed")
            self.indexing_page.show_stopped(
                "Indexing was cancelled",
                "Build the index to start searching. Rows that were already processed are reused.",
                failed=False,
            )
        else:
            self.dataset_chip.set_state("idle", "Not indexed")

    def _update_model_status(self) -> None:
        if self._index is not None:
            semantic = bool(getattr(self._index, "has_semantic", False))
        else:
            semantic = self._embedder_state != "missing"
        if self._index is None and self._embedder_state in ("pending", "loading"):
            self.model_icon.set_icon("sparkles", "text_faint")
            self.model_text.setText("Loading language model…")
        elif semantic:
            self.model_icon.set_icon("sparkles", "accent")
            self.model_text.setText("Semantic search on")
        else:
            self.model_icon.set_icon("type", "warning")
            self.model_text.setText("Lexical only (model missing)")

    # ================================================================== events
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        kind = event.type()
        if kind not in _DRAG_EVENTS or not isinstance(watched, QWidget) or watched.window() is not self:
            return False
        if kind == QEvent.Type.DragLeave:
            self._drop_hide.start()
            return False
        path = self._dropped_path(event.mimeData())
        if path is None:
            return False
        if kind in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            self._drop_hide.stop()
            if not self.drop_overlay.isVisible():
                self.drop_overlay.show_overlay()
            return True
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        self.drop_overlay.hide()
        QTimer.singleShot(0, lambda: self.import_file(path))
        return True

    def _dropped_path(self, mime) -> Path | None:
        if mime is None or not mime.hasUrls():
            return None
        supported = self.backend.supported_extensions()
        for url in mime.urls():
            if url.isLocalFile():
                candidate = Path(url.toLocalFile())
                if candidate.suffix.lower() in supported:
                    return candidate
        return None

    def closeEvent(self, event) -> None:
        self._save_window_state()
        self.runner.cancel_all()
        self.runner.wait(3000)
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)
