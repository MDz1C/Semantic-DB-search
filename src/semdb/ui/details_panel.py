"""Right-hand panel explaining the selected result ("why this match") or query (R13)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ..core.text import remove_spans, tokens as text_tokens
from ..core.types import HistoryEntry, ResultRow
from . import theme
from .results_model import (
    ColumnInfo,
    format_timestamp,
    percent,
    plural,
    primary_value,
    relative_time,
    result_value,
    results_tsv,
)
from .widgets import Card, IconBadge, IconWidget, MeterBar, button, label

PANEL_WIDTH = 360
NO_RERUN = "Load a database to search again."
NO_MEANING = "Meaning was not compared in this search: no language model, or the query only named an identity."


def verdict(score: float) -> str:
    if score >= 0.85:
        return "Strong match"
    if score >= 0.60:
        return "Possible match"
    return "Weak match"


def compared_text(entry: HistoryEntry) -> str:
    """The query with confidently detected identity fragments removed (what the text was compared on)."""
    spans: list[tuple[int, int]] = []
    folded_query = entry.query.casefold()
    for evidence in entry.evidence:
        fragment = evidence.matched_text
        if evidence.strength < 0.5 or not fragment:
            continue
        start = entry.query.find(fragment)
        if start < 0:
            start = folded_query.find(fragment.casefold())
        if start >= 0:
            spans.append((start, start + len(fragment)))
    return remove_spans(entry.query, spans)


def compared_meaning(entry: HistoryEntry) -> bool:
    """Whether the search behind ``entry`` compared meaning: without the language model, or when
    the query only named an identity, every result's semantic score is 0."""
    return any((result.detail or {}).get("semantic") for result in entry.results)


class DetailsPanel(Card):
    rerun_requested = Signal(list)  # query strings
    delete_requested = Signal(list)  # entry keys

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(280)
        self._entry: HistoryEntry | None = None
        self._result: ResultRow | None = None
        self._columns: list[ColumnInfo] = []
        self._shown = 0
        self._key = ""
        self._can_rerun = False
        self._rerun_reason = ""
        self._rerun: QPushButton | None = None  # the shown query's Re-run button
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("Transparent")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        layout.addWidget(self._scroll)
        self.show_empty()

    # ------------------------------------------------------------------ public
    def current(self) -> tuple[HistoryEntry | None, ResultRow | None]:
        return self._entry, self._result

    def show_empty(self) -> None:
        self._entry, self._result = None, None
        body = self._new_body(center=True)
        body.addStretch(1)
        body.addWidget(IconBadge("mouse-pointer", 48, color="text_muted", background="neutral_soft"), 0,
                       Qt.AlignmentFlag.AlignHCenter)
        body.addSpacing(4)
        title = label("Nothing selected", "strong")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(title)
        hint = label("Select a query or a result to see what was detected and why it matched.", "caption", wrap=True)
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body.addWidget(hint)
        body.addStretch(2)

    def show_entry(self, entry: HistoryEntry | None, result: ResultRow | None, columns: list[ColumnInfo],
                   shown: int, key: str) -> None:
        self._columns, self._shown, self._key = columns, shown, key
        if entry is None:
            self.show_empty()
        elif result is None:
            self._show_query(entry)
        else:
            self._show_result(entry, result)

    def refresh(self) -> None:
        """Re-render (relative times, permissions) for the same selection."""
        if self._entry is not None:
            self.show_entry(self._entry, self._result, self._columns, self._shown, self._key)

    @property
    def can_rerun(self) -> bool:
        return self._can_rerun

    @can_rerun.setter
    def can_rerun(self, enabled: bool) -> None:
        self.set_rerun(enabled)

    def set_rerun(self, enabled: bool, reason: str = "") -> None:
        """Whether Re-run can search now (``reason`` says why not); applies to the shown query at once."""
        self._can_rerun, self._rerun_reason = bool(enabled), reason
        if self._rerun is not None:
            self._update_rerun(self._rerun)

    def _update_rerun(self, rerun: QPushButton) -> None:
        rerun.setEnabled(self._can_rerun)
        rerun.setToolTip("" if self._can_rerun else self._rerun_reason or NO_RERUN)

    def _shown_count(self, entry: HistoryEntry) -> int:
        """How many results the list of ``entry`` shows: K, or all of them when it has fewer."""
        return min(self._shown or len(entry.results), len(entry.results))

    # ------------------------------------------------------------------ building blocks
    def _new_body(self, center: bool = False) -> QVBoxLayout:
        self._rerun = None  # setWidget deletes the previous body and its buttons
        content = QWidget()
        content.setObjectName("Transparent")
        body = QVBoxLayout(content)
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(8 if center else 6)
        self._scroll.setWidget(content)
        return body

    @staticmethod
    def _section(body: QVBoxLayout, title: str) -> None:
        body.addSpacing(14)
        heading = QLabel(title)
        heading.setFont(theme.section_font())
        heading.setProperty("role", "muted")
        body.addWidget(heading)
        body.addSpacing(2)

    @staticmethod
    def _line(icon_name: str, color: str, text: str, caption: str = "") -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(10)
        layout.addWidget(IconWidget(icon_name, color, 16), 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(1)
        texts.addWidget(label(text, wrap=True, selectable=True))
        if caption:
            texts.addWidget(label(caption, "caption", wrap=True, selectable=True))
        layout.addLayout(texts, 1)
        return row

    @staticmethod
    def _meter(name: str, value: float | None, color: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(12)
        caption = label(name, "caption")
        caption.setFixedWidth(104)
        layout.addWidget(caption)
        layout.addWidget(MeterBar(value or 0.0, color), 1, Qt.AlignmentFlag.AlignVCenter)
        number = label("—" if value is None else percent(value), "caption")
        number.setFixedWidth(36)
        number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(number)
        return row

    # ------------------------------------------------------------------ result
    def _show_result(self, entry: HistoryEntry, result: ResultRow) -> None:
        self._entry, self._result = entry, result
        body = self._new_body()
        detail = result.detail or {}
        body.addWidget(label(f"Result {result.rank} of {self._shown_count(entry)}  ·  Sheet row "
                             f"{result.source_row}", "caption"))
        title = label(primary_value(entry, result), "heading", wrap=True, selectable=True)
        body.addWidget(title)
        body.addWidget(label(f"for “{entry.query}”", "caption", wrap=True))

        body.addSpacing(12)
        foreground = "success" if result.score >= 0.85 else "warning" if result.score >= 0.6 else "text_muted"
        score_row = QHBoxLayout()
        score_row.setSpacing(10)
        big = QLabel(percent(result.score))
        big.setFont(theme.font(22, QFont.Weight.DemiBold))
        big.setStyleSheet(f"color: {theme.css(theme.tokens().color(foreground))};")
        score_row.addWidget(big, 0, Qt.AlignmentFlag.AlignBottom)
        words = QVBoxLayout()
        words.setSpacing(0)
        words.addStretch(1)
        words.addWidget(label(verdict(result.score), "strong"))
        words.addWidget(label("overall match", "caption"))
        score_row.addLayout(words)
        score_row.addStretch(1)
        body.addLayout(score_row)

        self._section(body, "Why it matched")
        bar = "accent"
        body.addWidget(self._meter("Text similarity", detail.get("text_score"), bar))
        compared = compared_meaning(entry)
        meaning = self._meter("Meaning", detail.get("semantic") if compared else None, bar)
        if not compared:
            meaning.setToolTip(NO_MEANING)
        body.addWidget(meaning)
        body.addWidget(self._meter("Exact words", detail.get("lexical"), bar))
        factor = detail.get("family_factor")
        body.addWidget(self._meter("Identity", factor, "success" if (factor or 0) >= 0.99 else "danger"))

        lines = self._identity_lines(entry, result)
        if lines:
            body.addSpacing(6)
            for line in lines:
                body.addWidget(line)

        self._section(body, "Fields")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)
        for row, column in enumerate(entry.columns):
            name = label(column, "caption", wrap=True)
            name.setFixedWidth(104)
            grid.addWidget(name, row, 0, Qt.AlignmentFlag.AlignTop)
            value = result_value(entry, result, column)
            grid.addWidget(label(value or "—", None if value else "faint", wrap=True, selectable=True), row, 1,
                           Qt.AlignmentFlag.AlignTop)
        body.addLayout(grid)

        body.addSpacing(16)
        copy = button("Copy row", icon_name="copy")
        copy.clicked.connect(lambda: self._copy_row(entry, result))
        row = QHBoxLayout()
        row.addWidget(copy)
        row.addStretch(1)
        body.addLayout(row)
        body.addStretch(1)

    def _identity_lines(self, entry: HistoryEntry, result: ResultRow) -> list[QWidget]:
        detail = result.detail or {}
        named = {e.column: e for e in entry.evidence}
        lines: list[QWidget] = []
        exact = detail.get("exact_id")
        if exact:
            value = result_value(entry, result, exact)
            lines.append(self._line("check-circle", "accent", f"Exact {exact} match", value))
        for column in detail.get("agreements") or ():
            value = result_value(entry, result, column)
            lines.append(self._line("check-circle", "success", f"{column} agrees", value))
        for column in detail.get("conflicts") or ():
            value = result_value(entry, result, column) or "empty"
            expected = named.get(column)
            caption = f"This row has {value}; the query points to {expected.value}" if expected else value
            lines.append(self._line("x-circle", "danger", f"{column} differs", caption))
        if not lines:
            lines.append(self._line("info", "text_muted", "No identity named in the query",
                                    "Ranked on the descriptive text alone."))
        return lines

    def _copy_row(self, entry: HistoryEntry, result: ResultRow) -> None:
        columns = self._columns or [ColumnInfo(c, "search") for c in entry.columns]
        QGuiApplication.clipboard().setText(results_tsv([(entry, result)], columns))

    # ------------------------------------------------------------------ query
    def _show_query(self, entry: HistoryEntry) -> None:
        self._entry, self._result = entry, None
        body = self._new_body()
        body.addWidget(label(f"Query  ·  {relative_time(entry.created_at)}", "caption"))
        body.addWidget(label(entry.query, "heading", wrap=True, selectable=True))
        best = f"  ·  best {percent(entry.results[0].score)}" if entry.results else ""
        body.addWidget(label(f"{plural(self._shown_count(entry), 'result')} shown{best}", "caption"))

        self._section(body, "Detected identity")
        if entry.evidence:
            for evidence in entry.evidence:
                body.addWidget(self._line(
                    "fingerprint", "accent", f"{evidence.column} → {evidence.value}",
                    f"from “{evidence.matched_text}”  ·  {percent(evidence.strength)} confident",
                ))
        else:
            body.addWidget(self._line("info", "text_muted", "Nothing detected",
                                      "The whole query was compared as text."))

        self._section(body, "Compared text")
        cleaned = compared_text(entry)
        if text_tokens(cleaned):
            box = label(cleaned, wrap=True, selectable=True)
            box.setStyleSheet(
                f"background: {theme.css(theme.tokens().surface_alt)}; border: 1px solid "
                f"{theme.css(theme.tokens().border)}; border-radius: 8px; padding: 8px 10px;"
            )
            body.addWidget(box)
        else:
            body.addWidget(label("Only an identity was named, so rows were ranked by identity.", "caption",
                                 wrap=True))

        self._section(body, "Searched")
        body.addWidget(label(format_timestamp(entry.created_at), wrap=True))
        if entry.dataset_label:
            body.addWidget(label(entry.dataset_label, "caption", wrap=True))

        body.addSpacing(16)
        row = QHBoxLayout()
        row.setSpacing(8)
        rerun = button("Re-run", icon_name="refresh")
        rerun.clicked.connect(lambda: self.rerun_requested.emit([entry.query]))
        self._rerun = rerun
        self._update_rerun(rerun)
        delete = button("Delete", "danger-ghost", icon_name="trash")
        delete.clicked.connect(lambda: self.delete_requested.emit([self._key]))
        row.addWidget(rerun)
        row.addWidget(delete)
        row.addStretch(1)
        body.addLayout(row)
        body.addStretch(1)
