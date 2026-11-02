"""The results tree: collapsible query lists (R9) painted by a custom delegate.

Query rows span the full width and keep their content inside the visible part of the
viewport (so it stays put while wide result columns scroll horizontally); result rows show
rank + score, then the dataset columns with identity pills for deciding columns.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QModelIndex,
    QPersistentModelIndex,
    QRect,
    QRectF,
    QSize,
    Qt,
    Signal,
)
import math

from PySide6.QtGui import QFont, QFontMetrics, QFontMetricsF, QGuiApplication, QKeySequence, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QStyledItemDelegate,
    QToolTip,
    QTreeView,
    QWidget,
)

from ..core.types import HistoryEntry, ResultRow
from . import icons, theme
from .results_model import (
    COLUMN_ROLE,
    KEY_ROLE,
    ColumnInfo,
    ResultsModel,
    cell_state,
    percent,
    plural,
    primary_value,
    relative_time,
    result_value,
    results_tsv,
)
from .theme import tokens, with_alpha
from .widgets import menu

QUERY_ROW_H = 44
RESULT_ROW_H = 34
GROUP_GAP = 8  # breathing room below the last result of an expanded list
INDENT = 42  # x where query text and result ranks start
RANK_W = 18
CELL_PAD = 12
SCORE_W = 48
MATCH_COLUMN_W = INDENT + RANK_W + 10 + SCORE_W + 14
# Row size hints (copied by Qt; shared so a layout does not build thousands of them)
QUERY_SIZE = QSize(200, QUERY_ROW_H)
RESULT_SIZE = QSize(120, RESULT_ROW_H)
LAST_RESULT_SIZE = QSize(120, RESULT_ROW_H + GROUP_GAP)


def elide(metrics: QFontMetricsF, text: str, width: float) -> str:
    """``text`` cut with an ellipsis to ``width`` (sub-pixel overflow is tolerated)."""
    if metrics.horizontalAdvance(text) <= width + 0.5:
        return text
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(0.0, width))


class ResultsDelegate(QStyledItemDelegate):
    """Paints query rows and result cells; :class:`ResultsView` drives it from ``drawRow``."""

    def __init__(self, view: ResultsView) -> None:
        super().__init__(view)
        self._view = view
        self._model = view.results_model()
        self.f_query = theme.font(theme.BASE_PT, QFont.Weight.DemiBold)
        self.f_body = theme.font(theme.BASE_PT)
        self.f_caption = theme.font(theme.CAPTION_PT)
        self.f_pill = theme.font(theme.SMALL_PT, QFont.Weight.DemiBold)
        self.m_query = QFontMetricsF(self.f_query)
        self.m_body = QFontMetricsF(self.f_body)
        self.m_caption = QFontMetricsF(self.f_caption)
        self.m_pill = QFontMetricsF(self.f_pill)

    # ------------------------------------------------------------------ sizes
    def sizeHint(self, option, index: QModelIndex) -> QSize:  # noqa: N802
        """Depends on the row kind only (layouts ask once per visible row): no model round trips."""
        if not index.internalId():
            return QUERY_SIZE
        return LAST_RESULT_SIZE if self._model.is_last_result(index) else RESULT_SIZE

    def paint(self, painter: QPainter, option, index: QModelIndex) -> None:
        model = self._view.results_model()
        entry = model.entry_for(index)
        if entry is None:
            return
        if model.is_query(index):
            self.paint_query(painter, option.rect, entry, self._view.isExpanded(index), model)
        else:
            result = model.result_for(index)
            if result is not None:
                self.paint_result_cell(painter, option.rect, index.column(), entry, result, model)

    # ------------------------------------------------------------------ pills
    def pill_width(self, text: str, with_icon: bool = False) -> int:
        return math.ceil(self.m_pill.horizontalAdvance(text)) + 18 + (14 if with_icon else 0)

    def draw_pill(
        self,
        painter: QPainter,
        rect: QRectF,
        text: str,
        foreground,
        background,
        icon_name: str | None = None,
    ) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)
        x = rect.left() + 9
        if icon_name:
            icons.paint(painter, icon_name, QRectF(x - 1, rect.center().y() - 5.5, 11, 11), foreground, 2.4)
            x += 13
        painter.setPen(foreground)
        painter.setFont(self.f_pill)
        available = rect.right() - 9 - x
        if self.m_pill.horizontalAdvance(text) > available + 0.5:
            text = self.m_pill.elidedText(text, Qt.TextElideMode.ElideRight, available)
        painter.drawText(QRectF(x, rect.top(), available + 4, rect.height()),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), text)

    def draw_score(self, painter: QPainter, rect: QRectF, score: float) -> None:
        """Score pill whose tinted fill shows the score: green >= 85 %, amber >= 60 %, grey below."""
        t = tokens()
        score = max(0.0, min(1.0, score))
        foreground, _background = theme.score_colors(score, t)
        radius = rect.height() / 2
        shape = QPainterPath()
        shape.addRoundedRect(rect, radius, radius)
        painter.fillPath(shape, t.neutral_soft)
        painter.save()
        painter.setClipPath(shape)
        painter.fillRect(QRectF(rect.left(), rect.top(), rect.width() * score, rect.height()),
                         with_alpha(foreground, 0.26 if t.dark else 0.17))
        painter.restore()
        painter.setPen(foreground)
        painter.setFont(self.f_pill)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, percent(score))

    # ------------------------------------------------------------------ query rows
    def paint_query(self, painter: QPainter, rect: QRect, entry: HistoryEntry, expanded: bool,
                    model: ResultsModel) -> None:
        t = tokens()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = rect.top() + QUERY_ROW_H / 2
        icons.paint(painter, "chevron-down" if expanded else "chevron-right",
                    QRectF(rect.left() + 16, center - 8, 16, 16), t.text_muted)
        left = rect.left() + INDENT
        right = rect.right() - 16
        if entry.results:
            self.draw_score(painter, QRectF(right - SCORE_W - 2, center - 11, SCORE_W + 2, 22), entry.results[0].score)
            right -= SCORE_W + 16
        shown = min(model.k, len(entry.results))
        meta = f"{plural(shown, 'result')}  ·  {relative_time(entry.created_at, model.now())}"
        meta_width = self.m_caption.horizontalAdvance(meta)
        if right - left > meta_width + 160:
            painter.setPen(t.text_muted)
            painter.setFont(self.f_caption)
            painter.drawText(QRectF(right - meta_width - 4, rect.top(), meta_width + 4, QUERY_ROW_H),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), meta)
            right -= meta_width + 20
        available = max(0, right - left)
        pills = [e.value for e in entry.evidence if e.value]
        widths = [self.pill_width(v) for v in pills]
        query_width = self.m_query.horizontalAdvance(entry.query)
        pills_total = sum(widths) + 6 * max(0, len(widths) - 1)
        if pills and query_width + 12 + pills_total > available:
            text_budget = min(query_width, max(int(available * 0.6), available - pills_total - 12))
            room = available - text_budget - 12
            kept: list[int] = []
            used = 0
            for i, width in enumerate(widths):
                extra = width + (6 if kept else 0)
                more = len(widths) - i - 1
                reserve = (self.pill_width(f"+{more}") + 6) if more else 0
                if used + extra + reserve > room:
                    break
                kept.append(i)
                used += extra
            dropped = len(pills) - len(kept)
            pills = [pills[i] for i in kept] + ([f"+{dropped}"] if dropped and room >= self.pill_width("+9") else [])
            widths = [self.pill_width(v) for v in pills]
            pills_total = sum(widths) + 6 * max(0, len(widths) - 1)
            text_budget = max(0, available - (pills_total + 12 if pills else 0))
        else:
            text_budget = available
        painter.setPen(t.text)
        painter.setFont(self.f_query)
        query = elide(self.m_query, entry.query, text_budget)
        painter.drawText(QRectF(left, rect.top(), text_budget + 4, QUERY_ROW_H),
                         int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), query)
        x = left + self.m_query.horizontalAdvance(query) + 12
        for value, width in zip(pills, widths):
            self.draw_pill(painter, QRectF(x, center - 10, width, 20), value, t.accent, t.accent_soft)
            x += width + 6
        painter.restore()

    # ------------------------------------------------------------------ result cells
    def paint_result_cell(self, painter: QPainter, rect: QRect, column: int, entry: HistoryEntry,
                          result: ResultRow, model: ResultsModel) -> None:
        t = tokens()
        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = rect.top() + RESULT_ROW_H / 2
        if column == 0:
            painter.setPen(t.text_faint)
            painter.setFont(self.f_caption)
            painter.drawText(QRectF(rect.left() + INDENT, rect.top(), RANK_W, RESULT_ROW_H),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter), str(result.rank))
            self.draw_score(painter, QRectF(rect.left() + INDENT + RANK_W + 10, center - 10, SCORE_W, 20),
                            result.score)
            painter.restore()
            return
        columns = model.columns()
        if column - 1 >= len(columns):
            painter.restore()
            return
        info = columns[column - 1]
        text = model.cell_text(entry, result, column - 1)
        state = cell_state(result, info.name) if text else None
        inner = rect.adjusted(CELL_PAD, 0, -CELL_PAD, 0)
        if state is not None:
            foreground, background, icon_name = {
                "agree": (t.success, t.success_soft, "check"),
                "conflict": (t.danger, t.danger_soft, "x"),
                "exact": (t.accent, t.accent_soft, "check"),
            }[state]
            width = min(inner.width(), self.pill_width(text, with_icon=True))
            self.draw_pill(painter, QRectF(inner.left(), center - 11, width, 22), text, foreground, background,
                           icon_name)
        else:
            muted = info.kind in ("display", "extra")
            painter.setPen(t.text_muted if muted else t.text)
            painter.setFont(self.f_body)
            painter.drawText(QRectF(inner.left(), rect.top(), inner.width() + 4, RESULT_ROW_H),
                             int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                             elide(self.m_body, text, inner.width()))
        painter.restore()

    def text_fits(self, index: QModelIndex, width: int) -> bool:
        text = index.data(Qt.ItemDataRole.DisplayRole) or ""
        return self.m_body.horizontalAdvance(text) <= width - 2 * CELL_PAD + 0.5

    def helpEvent(self, event, view, option, index: QModelIndex) -> bool:  # noqa: N802
        """Tooltips for result cells only when the text is cut off or carries an explanation."""
        model = self._view.results_model()
        if event.type() == QEvent.Type.ToolTip and index.isValid() and not model.is_query(index):
            result = model.result_for(index)
            info = index.data(COLUMN_ROLE)
            if index.column() > 0 and result is not None and isinstance(info, ColumnInfo):
                width = self._view.header().sectionSize(index.column())
                if cell_state(result, info.name) is None and self.text_fits(index, width):
                    QToolTip.hideText()
                    return True
        return super().helpEvent(event, view, option, index)


class ResultsHeader(QHeaderView):
    """Flat header aligned with the painted cells; deciding columns get a fingerprint mark."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setObjectName("ResultsHeader")
        self.setFixedHeight(36)
        self.setSectionsClickable(False)
        self.setHighlightSections(False)
        self.setSectionsMovable(False)
        self.setMinimumSectionSize(64)
        self.setStretchLastSection(False)
        self._font = theme.font(theme.CAPTION_PT, QFont.Weight.DemiBold)
        self._metrics = QFontMetricsF(self._font)

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), t.surface)
        painter.fillRect(QRect(0, self.height() - 1, self.viewport().width(), 1), t.border)
        painter.end()
        super().paintEvent(event)

    def paintSection(self, painter: QPainter, rect: QRect, logical: int) -> None:  # noqa: N802
        if not rect.isValid():
            return
        t = tokens()
        model = self.model()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(rect, t.surface)
        painter.fillRect(QRect(rect.left(), rect.bottom(), rect.width(), 1), t.border)
        painter.setFont(self._font)
        painter.setPen(t.text_muted)
        align = int(Qt.AlignmentFlag.AlignVCenter)
        if logical == 0:
            painter.drawText(QRect(rect.left() + INDENT, rect.top(), RANK_W, rect.height()),
                             align | int(Qt.AlignmentFlag.AlignRight), "#")
            painter.drawText(QRect(rect.left() + INDENT + RANK_W + 10, rect.top(), SCORE_W, rect.height()),
                             align | int(Qt.AlignmentFlag.AlignHCenter), "Match")
        else:
            info = model.headerData(logical, Qt.Orientation.Horizontal, COLUMN_ROLE)
            x = rect.left() + CELL_PAD
            if isinstance(info, ColumnInfo) and info.kind == "deciding":
                icons.paint(painter, "fingerprint", QRectF(x, rect.center().y() - 6.5, 13, 13), t.accent, 2.0)
                x += 19
            text = str(model.headerData(logical, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole) or "")
            available = rect.right() - CELL_PAD - x
            painter.drawText(QRect(x, rect.top(), available + 4, rect.height()),
                             align | int(Qt.AlignmentFlag.AlignLeft), elide(self._metrics, text, available))
        if self.visualIndex(logical) < self.count() - 1:
            painter.fillRect(QRect(rect.right(), rect.top() + 11, 1, rect.height() - 22), t.border)
        painter.restore()


class ResultsView(QTreeView):
    """Tree of query lists.  Emits requests the window acts on; copies to the clipboard itself."""

    rerun_requested = Signal(list)  # query strings
    delete_requested = Signal(list)  # entry keys
    focus_changed = Signal(object, object)  # (HistoryEntry | None, ResultRow | None)
    expansion_changed = Signal()

    def __init__(self, model: ResultsModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Results")
        self._model = model
        self._narrow = False  # a full layout measuring the match column only is running
        self._delegate = ResultsDelegate(self)
        self._header = ResultsHeader(self)
        self.setHeader(self._header)
        self.setItemDelegate(self._delegate)
        self.setModel(model)
        self.setUniformRowHeights(False)
        self.setRootIsDecorated(False)
        self.setIndentation(0)
        self.setItemsExpandable(True)
        self.setExpandsOnDoubleClick(False)
        self.setAnimated(True)
        self.setAllColumnsShowFocus(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setMouseTracking(True)
        self.setFrameShape(QTreeView.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.verticalScrollBar().setSingleStep(24)
        self.can_rerun = False
        self._hover = QPersistentModelIndex()
        self._expanded_keys: set[str] = set()
        self._saved_selection: list[tuple[str, int | None]] = []
        self._saved_current: tuple[str, int | None] | None = None
        self._saved_scroll = 0
        self._auto_fit = True
        self._fitting = False
        self._primary_fit = 220  # content width of the title column, from apply_column_widths
        self._fit_had_samples = False  # the last fit could measure real values
        self._column_signature: tuple[str, ...] = ()
        model.modelAboutToBeReset.connect(self._capture_state)
        model.modelReset.connect(self._restore_state)
        model.layoutChanged.connect(self._on_layout_changed)
        model.times_changed.connect(self.viewport().update)
        self.expanded.connect(self._on_expanded)
        self.collapsed.connect(self._on_collapsed)
        self.selectionModel().currentChanged.connect(self._on_current_changed)
        self._header.sectionResized.connect(self._on_section_resized)
        theme.manager().changed.connect(self.viewport().update)
        self.apply_column_widths()

    def results_model(self) -> ResultsModel:
        return self._model

    # ------------------------------------------------------------------ painting
    def drawRow(self, painter: QPainter, option, index: QModelIndex) -> None:  # noqa: N802
        t = tokens()
        model = self._model
        width = self.viewport().width()
        top, height = option.rect.top(), option.rect.height()
        is_query = model.is_query(index)
        selected = self.selectionModel().isSelected(index)
        hovered = self._hover.isValid() and self._hover.row() == index.row() and self._hover.parent() == index.parent()
        if is_query:
            base, hover = t.surface, t.hover
            content = QRect(0, top, width, height)
        else:
            base, hover = t.surface_alt, t.hover_alt
            content = QRect(0, top, width, RESULT_ROW_H)
        painter.fillRect(QRect(0, top, width, height), base)
        if selected:
            painter.fillRect(content, t.selection)
        elif hovered:
            painter.fillRect(content, hover)
        if is_query and index.row() > 0:
            painter.fillRect(QRect(0, top, width, 1), t.border)
        entry = model.entry_for(index)
        if entry is None:
            return
        if is_query:
            self._delegate.paint_query(painter, QRect(0, top, width, QUERY_ROW_H), entry, self.isExpanded(index), model)
            return
        result = model.result_for(index)
        if result is None:
            return
        header = self.header()
        for logical in range(header.count()):
            if header.isSectionHidden(logical):
                continue
            x, size = header.sectionViewportPosition(logical), header.sectionSize(logical)
            if x + size <= 0 or x >= width:
                continue
            self._delegate.paint_result_cell(painter, QRect(x, top, size, RESULT_ROW_H), logical, entry, result, model)

    # ------------------------------------------------------------------ layout
    def doItemsLayout(self) -> None:  # noqa: N802
        """Lay every visible row out again (resets, layout changes, collapse all...).

        The tree measures a row by asking the delegate about each of its cells, Python calls per
        row and column; heights depend on the row kind alone, so the dataset columns are hidden
        while it runs (hidden sections are skipped) and come back at the same widths."""
        header = self.header()
        others = [c for c in range(1, header.count()) if not header.isSectionHidden(c)]
        if self._narrow or not others:
            super().doItemsLayout()
            return
        scroll = self.horizontalScrollBar().value()
        blocked = header.blockSignals(True)  # the widths end as they were: nothing to report
        self._narrow = True
        try:
            for column in others:
                header.setSectionHidden(column, True)
            super().doItemsLayout()
        finally:
            for column in others:
                header.setSectionHidden(column, False)
            self._narrow = False
            header.blockSignals(blocked)
        self.updateGeometries()  # scroll ranges for the full width again
        self.horizontalScrollBar().setValue(scroll)

    # ------------------------------------------------------------------ columns
    def apply_column_widths(self, force: bool = False) -> None:
        """Size columns from their content when the column set changes (or ``force``)."""
        columns = self._model.columns()
        signature = tuple(c.name for c in columns)
        # Columns first sized before any result existed were fitted to their headers only:
        # refit them once real values can be measured.
        unmeasured = not self._fit_had_samples and bool(self._model.visible_entries())
        if signature == self._column_signature and not force and not unmeasured:
            self._fit_primary()
            return
        self._column_signature = signature
        self._auto_fit = True
        header = self.header()
        self._fitting = True
        try:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            header.resizeSection(0, MATCH_COLUMN_W)
            samples = self._sample_texts(columns)
            self._fit_had_samples = any(samples)
            for position, info in enumerate(columns, start=1):
                header.setSectionResizeMode(position, QHeaderView.ResizeMode.Interactive)
                values = samples[position - 1]
                pill = 30 if info.kind == "deciding" else 0
                width = QFontMetrics(theme.font(theme.CAPTION_PT, QFont.Weight.DemiBold)).horizontalAdvance(info.name)
                width += 2 * CELL_PAD + (19 if info.kind == "deciding" else 0)
                if values:
                    lengths = sorted(math.ceil(self._delegate.m_body.horizontalAdvance(v)) for v in values)
                    typical = lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))]
                    width = max(width, typical + 2 * CELL_PAD + pill)
                cap = 420 if info.kind in ("primary", "search") else 240
                header.resizeSection(position, max(72, min(cap, width)))
                if info.kind == "primary":
                    self._primary_fit = max(220, min(cap, width))
        finally:
            self._fitting = False
        self._fit_primary()

    def _sample_texts(self, columns: list[ColumnInfo]) -> list[list[str]]:
        samples: list[list[str]] = [[] for _ in columns]
        for entry in self._model.visible_entries()[:40]:
            for result in entry.results[:10]:
                for position, info in enumerate(columns):
                    value = result_value(entry, result, info.name)
                    if value:
                        samples[position].append(value)
        return samples

    def _fit_primary(self) -> None:
        """Let the primary column absorb spare width until the user resizes a column; it never
        shrinks below its fitted content width (the other columns scroll instead)."""
        columns = self._model.columns()
        if not self._auto_fit or not columns or self._narrow:  # (hidden columns measure 0 meanwhile)
            return
        header = self.header()
        others = sum(header.sectionSize(i) for i in range(header.count()) if i != 1)
        target = max(self._primary_fit, self.viewport().width() - others)
        if header.sectionSize(1) != target:
            self._fitting = True
            header.resizeSection(1, target)
            self._fitting = False

    def _on_section_resized(self, logical: int, old: int, new: int) -> None:
        if not self._fitting and logical > 0:
            self._auto_fit = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_primary()

    # ------------------------------------------------------------------ state across resets
    def _capture_state(self) -> None:
        model = self._model
        self._saved_selection = []
        for index in self.selectionModel().selectedRows(0):
            entry = model.entry_for(index)
            if entry is not None:
                result = model.result_for(index)
                self._saved_selection.append((model.key_of(entry), result.rank if result else None))
        current = self.currentIndex()
        entry = model.entry_for(current)
        if entry is not None:
            result = model.result_for(current)
            self._saved_current = (model.key_of(entry), result.rank if result else None)
        else:
            self._saved_current = None
        self._saved_scroll = self.verticalScrollBar().value()
        self._hover = QPersistentModelIndex()

    def _restore_state(self) -> None:
        model = self._model
        reopened = self._reopen()
        selection = QItemSelection()
        for key, rank in self._saved_selection:
            index = self._locate(key, rank)
            if index.isValid():
                selection.select(index, index.siblingAtColumn(model.columnCount() - 1))
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        current = self._locate(*self._saved_current) if self._saved_current else QModelIndex()
        if current.isValid():
            self.selectionModel().setCurrentIndex(current, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.selectionModel().select(selection, flags)
        self.executeDelayedItemsLayout()  # unless scrolling to the restored current row already did
        self.verticalScrollBar().setValue(self._saved_scroll)
        self._announce(reopened, True)

    def _locate(self, key: str, rank: int | None) -> QModelIndex:
        parent = self._model.query_index(key)
        if rank is None or not parent.isValid():
            return parent
        return self._model.index(rank - 1, 0, parent)  # invalid when the list now shows fewer results

    def _on_layout_changed(self) -> None:
        """Sorting and filtering keep the expansion of every list that stays visible; the lists a
        filter brings back reopen when they were open before it hid them."""
        self._announce(self._reopen(), True)

    def _reopen(self) -> QModelIndex | None:
        """Expand the shown lists remembered as open that the tree has closed; the first of them."""
        model = self._model
        closed = [index for index in map(model.query_index, self._expanded_keys & model.visible_keys())
                  if not self.isExpanded(index)]
        if not closed:
            return None
        self._open_quietly(closed)
        return min(closed, key=QModelIndex.row)

    def _open_quietly(self, indexes: list[QModelIndex]) -> None:
        """Expand ``indexes`` with one layout for all and no per-list signals (see _announce)."""
        self.scheduleDelayedItemsLayout()  # the tree then only records each expansion
        with self._quiet():
            for index in indexes:
                self.setExpanded(index, True)

    @contextmanager
    def _quiet(self) -> Iterator[None]:
        blocked = self.blockSignals(True)
        try:
            yield
        finally:
            self.blockSignals(blocked)

    def _announce(self, index: QModelIndex | None, expanded: bool) -> None:
        """Report a batch once: as the view's own signal for one changed list (so listeners of
        expanded/collapsed and of expansion_changed each run once), or as expansion_changed."""
        if index is None:
            self.expansion_changed.emit()
        else:
            (self.expanded if expanded else self.collapsed).emit(index)

    def _on_expanded(self, index: QModelIndex) -> None:
        key = self._model.data(index, KEY_ROLE)
        if key:
            self._expanded_keys.add(key)
        self.expansion_changed.emit()

    def _on_collapsed(self, index: QModelIndex) -> None:
        self._expanded_keys.discard(self._model.data(index, KEY_ROLE))
        self.expansion_changed.emit()

    def forget_keys(self, keys: list[str]) -> None:
        self._expanded_keys.difference_update(keys)

    # ------------------------------------------------------------------ expansion helpers
    def any_expanded(self) -> bool:
        """No per-row work: the tree has open exactly the shown lists whose keys are remembered."""
        return not self._expanded_keys.isdisjoint(self._model.visible_keys())

    def expand_all_lists(self) -> None:
        closed = [index for index in self._lists() if not self.isExpanded(index)]
        if closed:
            self._open_quietly(closed)
            self.executeDelayedItemsLayout()
        self._expanded_keys.update(self._model.visible_keys())
        self._announce(closed[0] if closed else None, True)

    def collapse_all_lists(self) -> None:
        opened = next((index for index in self._lists() if self.isExpanded(index)), None)
        with self._quiet():  # collapseAll would report every list it closes
            self.collapseAll()
        self._expanded_keys.clear()
        self._announce(opened, False)

    def _lists(self) -> list[QModelIndex]:
        model = self._model
        return [model.index(row, 0) for row in range(model.rowCount())]

    def toggle(self, index: QModelIndex) -> None:
        index = index.siblingAtColumn(0)
        if self._model.is_query(index):
            self.setExpanded(index, not self.isExpanded(index))

    def reveal(self, key: str, expand: bool = False) -> None:
        """Select and scroll to the list of ``key``."""
        index = self._model.query_index(key)
        if not index.isValid():
            return
        if expand:
            self.setExpanded(index, True)
        flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
        self.selectionModel().setCurrentIndex(index, flags)
        self.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtTop)
        self.horizontalScrollBar().setValue(0)

    # ------------------------------------------------------------------ selection helpers
    def selected_items(self) -> tuple[list[HistoryEntry], list[tuple[HistoryEntry, ResultRow]]]:
        """Selected query entries and selected (entry, result) pairs, in display order."""
        model = self._model
        indexes = self.selectionModel().selectedRows(0)

        def order(index: QModelIndex) -> tuple[int, int]:
            if model.is_query(index):
                return index.row(), -1
            return index.internalId() - 1, index.row()

        queries: list[HistoryEntry] = []
        results: list[tuple[HistoryEntry, ResultRow]] = []
        for index in sorted(indexes, key=order):
            entry = model.entry_for(index)
            if entry is None:
                continue
            result = model.result_for(index)
            if result is None:
                queries.append(entry)
            else:
                results.append((entry, result))
        return queries, results

    def copy_selection(self) -> bool:
        queries, results = self.selected_items()
        if results:
            QGuiApplication.clipboard().setText(results_tsv(results, self._model.columns()))
            return True
        if queries:
            QGuiApplication.clipboard().setText("\r\n".join(e.query for e in queries) + "\r\n")
            return True
        return False

    def _on_current_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        self.focus_changed.emit(self._model.entry_for(current), self._model.result_for(current))

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, event) -> None:
        index = self.indexAt(event.position().toPoint())
        plain = event.modifiers() == Qt.KeyboardModifier.NoModifier
        super().mousePressEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and plain
            and index.isValid()
            and self._model.is_query(index)
        ):
            self.toggle(index)

    def mouseMoveEvent(self, event) -> None:
        index = self.indexAt(event.position().toPoint())
        index = index.siblingAtColumn(0) if index.isValid() else QModelIndex()
        if QModelIndex(self._hover) != index:
            self._update_row(QModelIndex(self._hover))
            self._hover = QPersistentModelIndex(index)
            self._update_row(index)
        super().mouseMoveEvent(event)

    def _update_row(self, index: QModelIndex) -> None:
        if index.isValid():
            rect = self.visualRect(index)
            self.viewport().update(QRect(0, rect.top(), self.viewport().width(), rect.height()))

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selection()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            queries, _ = self.selected_items()
            if queries:
                self.delete_requested.emit([self._model.key_of(e) for e in queries])
            return
        current = self.currentIndex()
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space) and self._model.is_query(current):
            self.toggle(current)
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:
        index = self.indexAt(event.pos())
        if not index.isValid():
            return
        if not self.selectionModel().isSelected(index):
            flags = QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows
            self.selectionModel().setCurrentIndex(index, flags)
        model = self._model
        entry = model.entry_for(index)
        if entry is None:
            return
        popup = menu(self)
        result = model.result_for(index)
        if result is None:
            queries, _ = self.selected_items()
            queries = queries or [entry]
            many = len(queries) > 1
            expanded = self.isExpanded(index.siblingAtColumn(0))
            if not many:
                toggle = popup.addAction("Collapse" if expanded else "Expand")
                toggle.triggered.connect(lambda: self.toggle(index))
            rerun = popup.addAction(icons.icon("refresh"), f"Re-run {len(queries)} queries" if many else "Re-run")
            rerun.setEnabled(self.can_rerun)
            rerun.triggered.connect(lambda: self.rerun_requested.emit([e.query for e in queries]))
            popup.addSeparator()
            copy_query = popup.addAction(icons.icon("copy"), "Copy queries" if many else "Copy query")
            copy_query.triggered.connect(
                lambda: QGuiApplication.clipboard().setText("\r\n".join(e.query for e in queries) + "\r\n"))
            copy_results = popup.addAction("Copy results")
            copy_results.triggered.connect(lambda: self._copy_results(queries))
            popup.addSeparator()
            delete = popup.addAction(icons.icon("trash", "danger"), f"Delete {len(queries)} lists" if many else "Delete")
            delete.triggered.connect(lambda: self.delete_requested.emit([model.key_of(e) for e in queries]))
        else:
            copy_row = popup.addAction(icons.icon("copy"), "Copy row")
            copy_row.triggered.connect(self.copy_selection)
            if index.column() > 0:
                value = model.cell_text(entry, result, index.column() - 1)
                copy_cell = popup.addAction("Copy cell")
                copy_cell.setEnabled(bool(value))
                copy_cell.triggered.connect(lambda: QGuiApplication.clipboard().setText(value))
            title = primary_value(entry, result)
            short = QFontMetrics(popup.font()).elidedText(title, Qt.TextElideMode.ElideRight, 260)
            copy_primary = popup.addAction(f"Copy “{short}”")
            copy_primary.triggered.connect(lambda: QGuiApplication.clipboard().setText(title))
        popup.exec(event.globalPos())
        popup.deleteLater()  # with its actions and the entries their slots hold

    def _copy_results(self, entries: list[HistoryEntry]) -> None:
        k = self._model.k
        items = [(entry, result) for entry in entries for result in entry.results[:k]]
        QGuiApplication.clipboard().setText(results_tsv(items, self._model.columns()))

    def viewportEvent(self, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Leave:
            self._update_row(QModelIndex(self._hover))
            self._hover = QPersistentModelIndex()
        return super().viewportEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(640, 400)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        if dx:
            self.viewport().update()
