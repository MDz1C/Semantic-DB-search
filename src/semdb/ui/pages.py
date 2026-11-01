"""States shown inside the results card when there is no list to show (ARCHITECTURE.md §12.2):
no database, ready to search, indexing, and no query matching the filter."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from .widgets import Card, DashedFrame, IconBadge, ProgressLine, Spinner, button, centered, label


def _text(text: str, role: str, width: int | None = None) -> QLabel:
    """Centred text; with ``width`` it wraps inside that width."""
    widget = label(text, role, wrap=width is not None)
    widget.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
    if width:
        widget.setFixedWidth(width)
    return widget


class EmptyImportPage(QWidget):
    """No database yet: one obvious action, plus the sample data and drag-and-drop."""

    import_requested = Signal()
    sample_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        zone = DashedFrame()
        outer.addWidget(zone)
        layout = QVBoxLayout(zone)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addLayout(centered(IconBadge("file-spreadsheet", 72, radius=20)))
        layout.addSpacing(20)
        layout.addLayout(centered(_text("Import your database", "hero")))
        layout.addSpacing(8)
        layout.addLayout(centered(_text(
            "Choose an Excel or CSV file. Everything stays on this computer — nothing is uploaded.", "muted", 380)))
        layout.addSpacing(24)
        self.import_button = button("Import file…", "primary", icon_name="import")
        self.import_button.setMinimumWidth(168)
        self.import_button.clicked.connect(self.import_requested)
        layout.addLayout(centered(self.import_button))
        layout.addSpacing(10)
        self.sample_button = button("Try with sample data", "link")
        self.sample_button.clicked.connect(self.sample_requested)
        layout.addLayout(centered(self.sample_button))
        layout.addSpacing(28)
        layout.addLayout(centered(_text("or drop a file anywhere in this window", "faint")))
        layout.addSpacing(4)
        layout.addLayout(centered(_text("Excel  ·  OpenDocument  ·  CSV  ·  TSV", "faint")))
        layout.addStretch(1)


class ReadyPage(QWidget):
    """Database loaded, nothing searched yet: how to start, with examples from the data."""

    example_clicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addLayout(centered(IconBadge("search", 64)))
        layout.addSpacing(20)
        self._title = _text("Ready to search", "hero")
        layout.addLayout(centered(self._title))
        layout.addSpacing(8)
        layout.addLayout(centered(_text(
            "Paste one query per line in the box below, then press Search. Each line gets its own ranked list.",
            "muted", 400)))
        layout.addSpacing(28)
        self._examples_label = _text("Try an example", "caption")
        layout.addLayout(centered(self._examples_label))
        layout.addSpacing(10)
        holder = QWidget()
        self._chips = QHBoxLayout(holder)
        self._chips.setContentsMargins(0, 0, 0, 0)
        self._chips.setSpacing(8)
        layout.addLayout(centered(holder))
        layout.addStretch(1)

    def set_dataset(self, rows: int, examples: list[str]) -> None:
        self._title.setText(f"Ready to search {rows:,} rows")
        while self._chips.count():
            item = self._chips.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for text in examples:
            chip = button(text, "chip", height=30)
            chip.setToolTip("Put this example in the search box")
            chip.clicked.connect(lambda _checked=False, value=text: self.example_clicked.emit(value))
            self._chips.addWidget(chip)
        self._examples_label.setVisible(bool(examples))


class IndexingPage(QWidget):
    """Progress card while the index is built; also explains a cancelled or failed build."""

    cancel_requested = Signal()
    retry_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.addStretch(1)
        card = Card(name="SoftCard")
        card.setFixedWidth(460)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 18)
        layout.setSpacing(0)
        head = QHBoxLayout()
        head.setSpacing(12)
        self._spinner = Spinner(22)
        self._badge = IconBadge("alert-circle", 28, color="danger", background="danger_soft")
        head.addWidget(self._spinner)
        head.addWidget(self._badge)
        self.title = label("Preparing your database", "title")
        head.addWidget(self.title, 1)
        layout.addLayout(head)
        layout.addSpacing(8)
        self.subtitle = label("", "caption", wrap=True)
        layout.addWidget(self.subtitle)
        layout.addSpacing(18)
        self.stage = label("", "strong")
        layout.addWidget(self.stage)
        layout.addSpacing(10)
        self.bar = ProgressLine()
        layout.addWidget(self.bar)
        layout.addSpacing(12)
        foot = QHBoxLayout()
        foot.setSpacing(8)
        self.eta = label("", "caption")
        foot.addWidget(self.eta, 1)
        self.cancel_button = button("Cancel", "ghost", height=30)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.retry_button = button("Build index", "primary", icon_name="refresh")
        self.retry_button.clicked.connect(self.retry_requested)
        foot.addWidget(self.cancel_button)
        foot.addWidget(self.retry_button)
        layout.addLayout(foot)
        outer.addLayout(centered(card))
        outer.addStretch(1)
        self.show_waiting("Opening your database…")

    def _mode(self, working: bool) -> None:
        self._spinner.setVisible(working)
        self._badge.setVisible(not working)
        self.bar.setVisible(working)
        self.cancel_button.setVisible(working)
        self.retry_button.setVisible(not working)
        self.stage.setVisible(working)

    def show_waiting(self, stage: str, cancellable: bool = False) -> None:
        self._mode(True)
        self.title.setText("Preparing your database")
        self.subtitle.setText("Searching becomes available as soon as this finishes.")
        self.stage.setText(stage)
        self.bar.set_value(None)
        self.eta.setText("")
        self.cancel_button.setVisible(cancellable)

    def show_progress(self, stage: str, fraction: float, eta: str) -> None:
        self._mode(True)
        self.title.setText("Preparing your database")
        self.subtitle.setText("The first time takes about a minute. After that it is cached on this computer.")
        self.stage.setText(stage)
        self.bar.set_value(fraction)
        self.eta.setText(eta)

    def show_stopped(self, title: str, text: str, *, failed: bool, button_text: str = "Build index") -> None:
        self._mode(False)
        tone = "danger" if failed else "accent"
        self._badge.set_icon("alert-circle" if failed else "info", tone, f"{tone}_soft")
        self.title.setText(title)
        self.subtitle.setText(text)
        self.retry_button.setText(button_text)


class NoMatchesPage(QWidget):
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.addStretch(1)
        layout.addLayout(centered(IconBadge("search", 48, color="text_muted", background="neutral_soft")))
        layout.addSpacing(14)
        self._title = _text("", "strong")
        layout.addLayout(centered(self._title))
        layout.addSpacing(6)
        clear = button("Clear filter", "link")
        clear.clicked.connect(self.clear_requested)
        layout.addLayout(centered(clear))
        layout.addStretch(2)

    def set_filter(self, text: str) -> None:
        self._title.setText(f"No queries match “{text.strip()}”")
