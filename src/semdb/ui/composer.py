"""The query composer at the bottom of the window (R8): one query per line, Ctrl+Enter searches."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QVBoxLayout, QWidget

from ..core.text import normalize
from . import icons
from .widgets import Card, IconWidget, ProgressButton, label, repolish, set_role

PLACEHOLDER = "Paste queries here — one per line. Each line becomes its own result list."


def parse_queries(text: str) -> tuple[list[str], int]:
    """Lines → queries: whitespace tidied, blanks skipped, duplicates (same normalized text)
    dropped keeping the first.  Returns ``(queries, duplicates_skipped)``."""
    queries: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for line in text.splitlines():
        query = " ".join(line.split())
        key = normalize(query)
        if not key:
            continue
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        queries.append(query)
    return queries, duplicates


class ComposerEdit(QPlainTextEdit):
    submit = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.submit.emit()
            return
        super().keyPressEvent(event)


class Composer(Card):
    search_requested = Signal(list)
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ready = False
        self._reason = ""
        self._busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 12, 12)
        layout.setSpacing(6)
        self.editor = ComposerEdit()
        self.editor.setObjectName("Composer")
        self.editor.setPlaceholderText(PLACEHOLDER)
        self.editor.setTabChangesFocus(True)
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.editor.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.editor.setAccessibleName("Queries, one per line")
        self.editor.document().setDocumentMargin(2)
        layout.addWidget(self.editor, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self._info_icon = IconWidget("info", "text_muted", 14)
        self._count = label("", "caption")
        footer.addWidget(self._info_icon)
        footer.addWidget(self._count)
        footer.addStretch(1)
        self._shortcut = QLabel("Ctrl+Enter")
        self._shortcut.setObjectName("Kbd")
        self._shortcut.setFixedHeight(22)
        footer.addWidget(self._shortcut)
        footer.addSpacing(4)
        self.search_button = ProgressButton("Search")
        self.search_button.setProperty("variant", "primary")
        self.search_button.setFixedHeight(34)
        self.search_button.setMinimumWidth(118)
        self.search_button.setIconSize(QSize(16, 16))
        self.search_button.setCursor(Qt.CursorShape.PointingHandCursor)
        icons.bind(self.search_button, "search", "on_accent", disabled="on_accent")
        footer.addWidget(self.search_button)
        layout.addLayout(footer)

        self._count_timer = QTimer(self)
        self._count_timer.setSingleShot(True)
        self._count_timer.setInterval(120)
        self._count_timer.timeout.connect(self._refresh)
        self.editor.textChanged.connect(self._count_timer.start)
        self.editor.submit.connect(self.submit)
        self.search_button.clicked.connect(self._on_button)
        self._refresh()

    # ------------------------------------------------------------------ API
    def queries(self) -> list[str]:
        return parse_queries(self.editor.toPlainText())[0]

    def set_text(self, text: str) -> None:
        self.editor.setPlainText(text)
        self._refresh()

    def clear(self) -> None:
        self.editor.clear()
        self._refresh()

    def set_ready(self, ready: bool, reason: str = "") -> None:
        """Whether Search is possible; ``reason`` explains why not (tooltip and footer)."""
        self._ready, self._reason = ready, reason
        self._refresh()

    def set_busy(self, busy: bool, progress: float | None = None) -> None:
        if busy != self._busy:
            self._busy = busy
            self.search_button.setProperty("variant", None if busy else "primary")
            repolish(self.search_button)
            self.search_button.setText("Cancel" if busy else "Search")
            if busy:
                icons.bind(self.search_button, "x", "text_muted")
            else:
                icons.bind(self.search_button, "search", "on_accent", disabled="on_accent")
        self.search_button.set_progress(progress if busy else None)
        self._refresh()

    def is_busy(self) -> bool:
        return self._busy

    def submit(self) -> None:
        if self._busy or not self._ready:
            return
        queries = self.queries()
        if queries:
            self.search_requested.emit(queries)
        else:
            self.editor.setFocus()

    # ------------------------------------------------------------------ internals
    def _on_button(self) -> None:
        if self._busy:
            self.cancel_requested.emit()
        else:
            self.submit()

    def _refresh(self) -> None:
        queries, duplicates = parse_queries(self.editor.toPlainText())
        count = len(queries)
        if self._busy:
            text, role, icon = f"Searching {count:,} {'query' if count == 1 else 'queries'}…", "caption", False
        elif not self._ready and self._reason:
            text, role, icon = self._reason, "caption", True
        elif count:
            text = f"{count:,} {'query' if count == 1 else 'queries'}"
            if duplicates:
                text += f"  ·  {duplicates:,} duplicate{'s' if duplicates != 1 else ''} skipped"
            role, icon = "muted", False
        else:
            text, role, icon = "Each line is searched separately", "caption", False
        self._count.setText(text)
        set_role(self._count, role)
        self._info_icon.setVisible(icon)
        enabled = self._busy or (self._ready and count > 0)
        self.search_button.setEnabled(enabled)
        if self._busy:
            tip = "Stop searching"
        elif not self._ready:
            tip = self._reason
        elif not count:
            tip = "Type or paste at least one query"
        else:
            tip = "Search every line (Ctrl+Enter)"
        self.search_button.setToolTip(tip)
        self._shortcut.setVisible(not self._busy and self._ready)
