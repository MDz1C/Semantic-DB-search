"""Item model behind the results tree: one top-level row per query, its first K results below.

Every stored entry keeps up to 50 results; the view shows the first K (R9).  Queries are
sorted by recency or alphabetically (R10) and can be narrowed by a text filter.  Column 0
is the match (rank + score); the remaining columns are dataset columns ordered
primary → other search → deciding → display only (ARCHITECTURE.md §12.2).
"""

from __future__ import annotations

from collections.abc import Iterable, KeysView, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QPersistentModelIndex, Qt, Signal

from ..core.text import fold, normalize
from ..core.types import HistoryEntry, ResultRow, Role

KIND_ROLE = Qt.ItemDataRole.UserRole + 1
ENTRY_ROLE = Qt.ItemDataRole.UserRole + 2
RESULT_ROLE = Qt.ItemDataRole.UserRole + 3
KEY_ROLE = Qt.ItemDataRole.UserRole + 4
COLUMN_ROLE = Qt.ItemDataRole.UserRole + 5
CELL_STATE_ROLE = Qt.ItemDataRole.UserRole + 6

QUERY = "query"
RESULT = "result"
SORT_RECENT = "recent"
SORT_ALPHA = "alpha"

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_PER_ROW_SIGNAL_LIMIT = 2000  # above this, K changes reset the model instead of per-row signals
_QUERY_FLAGS = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
# Results never have children: the tree neither expands them (expandAll) nor asks for their rows.
_RESULT_FLAGS = _QUERY_FLAGS | Qt.ItemFlag.ItemNeverHasChildren
# Sorting and filtering move whole rows; the columns (and so the header) stay as they are.
_ROWS_MOVED = QAbstractItemModel.LayoutChangeHint.VerticalSortHint
_WITH_HINT = "QList<QPersistentModelIndex>,QAbstractItemModel::LayoutChangeHint"  # signal overload

AnyIndex = QModelIndex | QPersistentModelIndex


@dataclass(frozen=True, slots=True)
class ColumnInfo:
    name: str
    kind: str  # "primary" | "search" | "deciding" | "display" | "extra"


@dataclass(slots=True)
class _Meta:
    key: str
    timestamp: float
    haystack: str
    position: int
    cell_map: list[int]


def entry_key(entry: HistoryEntry) -> str:
    """Identity of a query list; matches the store's ``query_key``."""
    return normalize(entry.query) or entry.query.strip().casefold()


def parse_time(created_at: str) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def relative_time(created_at: str, now: datetime | None = None) -> str:
    """"just now", "2 min ago", "3 h ago", "yesterday", "12 Mar", "12 Mar 2025"."""
    stamp = parse_time(created_at)
    if stamp is None:
        return ""
    now = now or datetime.now(timezone.utc)
    seconds = (now - stamp).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    local, local_now = stamp.astimezone(), now.astimezone()
    if seconds < 6 * 3600 or local.date() == local_now.date():
        return f"{int(seconds // 3600)} h ago"
    if local.date() == local_now.date() - timedelta(days=1):
        return "yesterday"
    if local.year == local_now.year:
        return f"{local.day} {_MONTHS[local.month - 1]}"
    return f"{local.day} {_MONTHS[local.month - 1]} {local.year}"


def format_timestamp(created_at: str) -> str:
    """"1 Nov 2026 at 17:32" in local time."""
    stamp = parse_time(created_at)
    if stamp is None:
        return created_at
    local = stamp.astimezone()
    return f"{local.day} {_MONTHS[local.month - 1]} {local.year} at {local:%H:%M}"


def percent(score: float) -> str:
    return f"{round(max(0.0, min(1.0, score)) * 100)}%"


def plural(count: int, noun: str, plural_form: str | None = None) -> str:
    return f"{count:,} {noun if count == 1 else (plural_form or noun + 's')}"


def order_columns(
    columns: list[str],
    roles: Mapping[str, Role] | None,
    primary: str | None = None,
) -> list[ColumnInfo]:
    """Dataset columns in display order: primary → deciding → other search → display only.

    The deciding columns sit right after the title: they are the identity the match hinges on
    (Brokerage, ISIN) and must stay visible without scrolling."""
    if not columns:
        return []
    roles = roles or {}
    search = [c for c in columns if roles.get(c) == Role.SEARCH]
    if primary not in columns:
        primary = search[0] if search else columns[0]
    other_search = [c for c in search if c != primary]
    deciding = [c for c in columns if roles.get(c) == Role.DECIDING and c != primary]
    used = {primary, *other_search, *deciding}
    rest = [c for c in columns if c not in used]
    return (
        [ColumnInfo(primary, "primary")]
        + [ColumnInfo(c, "deciding") for c in deciding]
        + [ColumnInfo(c, "search") for c in other_search]
        + [ColumnInfo(c, "display") for c in rest]
    )


def columns_from_history(entries: Iterable[HistoryEntry]) -> list[ColumnInfo]:
    """Display columns when no database is loaded: taken from the most recent entry."""
    entries = list(entries)
    if not entries:
        return []
    newest = max(entries, key=lambda e: (parse_time(e.created_at) or datetime.min.replace(tzinfo=timezone.utc)))
    primary = newest.primary_column if newest.primary_column in newest.columns else None
    first = primary or (newest.columns[0] if newest.columns else None)
    if first is None:
        return []
    deciding = [c for c in newest.columns if c in newest.deciding and c != first]
    rest = [c for c in newest.columns if c != first and c not in deciding]
    return (
        [ColumnInfo(first, "primary")]
        + [ColumnInfo(c, "deciding") for c in deciding]
        + [ColumnInfo(c, "search") for c in rest]
    )


def with_history_columns(base: list[ColumnInfo], entries: Iterable[HistoryEntry], limit: int = 30) -> list[ColumnInfo]:
    """Append columns that only older entries have (e.g. from a previous database)."""
    known = {c.name for c in base}
    extra: list[ColumnInfo] = []
    ordered = sorted(entries, key=lambda e: e.created_at, reverse=True)
    for entry in ordered:
        for name in entry.columns:
            if name not in known and len(extra) < limit:
                known.add(name)
                extra.append(ColumnInfo(name, "extra"))
    return base + extra


def cell_state(result: ResultRow, column: str) -> str | None:
    """"exact", "agree" or "conflict" when the ranking evidence concerns ``column``."""
    detail = result.detail or {}
    if detail.get("exact_id") == column:
        return "exact"
    if column in (detail.get("agreements") or ()):
        return "agree"
    if column in (detail.get("conflicts") or ()):
        return "conflict"
    return None


def result_value(entry: HistoryEntry, result: ResultRow, column: str) -> str:
    try:
        return result.cells[entry.columns.index(column)]
    except (ValueError, IndexError):
        return ""


def primary_value(entry: HistoryEntry, result: ResultRow) -> str:
    value = result_value(entry, result, entry.primary_column)
    if value:
        return value
    return next((cell for cell in result.cells if cell), f"Row {result.source_row}")


def _tsv_cell(value: str) -> str:
    return " ".join(str(value).replace("\t", " ").split())


def results_tsv(items: list[tuple[HistoryEntry, ResultRow]], columns: list[ColumnInfo]) -> str:
    """Selected results as tab-separated text with a header row (pastes into Excel)."""
    header = ["Query", "Rank", "Match"] + [c.name for c in columns]
    lines = ["\t".join(header)]
    for entry, result in items:
        cells = [entry.query, str(result.rank), percent(result.score)]
        cells += [result_value(entry, result, c.name) for c in columns]
        lines.append("\t".join(_tsv_cell(c) for c in cells))
    return "\r\n".join(lines) + "\r\n"


class ResultsModel(QAbstractItemModel):
    """Two-level model.  Top-level internal id is 0; a result's internal id is its parent row + 1."""

    times_changed = Signal()  # now() moved on: the relative times painted are stale

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._entries: list[HistoryEntry] = []
        self._visible: list[HistoryEntry] = []
        self._counts: list[int] = []
        self._rows: dict[str, int] = {}  # key -> row of each visible list
        self._meta: dict[int, _Meta] = {}
        self._columns: list[ColumnInfo] = []
        self._k = 5
        self._sort = SORT_RECENT
        self._filter_terms: list[str] = []
        self._filter_text = ""
        self._now = datetime.now(timezone.utc)

    # ----------------------------------------------------------------- public API
    @property
    def k(self) -> int:
        return self._k

    @property
    def sort_mode(self) -> str:
        return self._sort

    @property
    def filter_text(self) -> str:
        return self._filter_text

    def now(self) -> datetime:
        return self._now

    def columns(self) -> list[ColumnInfo]:
        return list(self._columns)

    def all_entries(self) -> list[HistoryEntry]:
        return list(self._entries)

    def visible_entries(self) -> list[HistoryEntry]:
        """Entries in the order shown (sorted and filtered)."""
        return list(self._visible)

    def set_entries(self, entries: Iterable[HistoryEntry]) -> None:
        self.beginResetModel()
        self._entries = list(entries)
        self._meta = {id(e): self._make_meta(e, i) for i, e in enumerate(self._entries)}
        self._now = datetime.now(timezone.utc)
        self._rebuild_visible()
        self.endResetModel()

    def set_columns(self, columns: list[ColumnInfo]) -> None:
        if columns == self._columns:
            return
        self.beginResetModel()
        self._columns = list(columns)
        for entry in self._entries:
            self._meta[id(entry)].cell_map = self._cell_map(entry)
        self.endResetModel()

    def set_sort_mode(self, mode: str) -> None:
        mode = mode if mode in (SORT_RECENT, SORT_ALPHA) else SORT_RECENT
        if mode == self._sort:
            return
        self._sort = mode
        self._move_rows()

    def set_filter(self, text: str) -> None:
        terms = fold(text).split()
        self._filter_text = text
        if terms == self._filter_terms:
            return
        self._filter_terms = terms
        self._move_rows()

    def set_k(self, k: int) -> None:
        k = max(1, int(k))
        if k == self._k:
            return
        if len(self._visible) > _PER_ROW_SIGNAL_LIMIT:
            self.beginResetModel()
            self._k = k
            self._counts = [min(k, len(e.results)) for e in self._visible]
            self.endResetModel()
            return
        self._k = k
        for row, entry in enumerate(self._visible):
            before, after = self._counts[row], min(k, len(entry.results))
            if after == before:
                continue
            parent = self.index(row, 0)
            if after > before:
                self.beginInsertRows(parent, before, after - 1)
                self._counts[row] = after
                self.endInsertRows()
            else:
                self.beginRemoveRows(parent, after, before - 1)
                self._counts[row] = after
                self.endRemoveRows()

    def refresh_times(self) -> None:
        """Re-evaluate relative times ("2 min ago") of every query row.

        Only painting shows them, no item data changes: views repaint on ``times_changed``
        (dataChanged would make the tree measure every row of every list again)."""
        self._now = datetime.now(timezone.utc)
        self.times_changed.emit()

    def is_query(self, index: AnyIndex) -> bool:
        return index.isValid() and index.internalId() == 0

    def entry_for(self, index: AnyIndex) -> HistoryEntry | None:
        """The entry of a query row, or the owning entry of a result row."""
        if not index.isValid():
            return None
        row = index.row() if index.internalId() == 0 else index.internalId() - 1
        return self._visible[row] if 0 <= row < len(self._visible) else None

    def result_for(self, index: AnyIndex) -> ResultRow | None:
        if not index.isValid() or index.internalId() == 0:
            return None
        entry = self.entry_for(index)
        if entry is None or index.row() >= len(entry.results):
            return None
        return entry.results[index.row()]

    def is_last_result(self, index: AnyIndex) -> bool:
        """Whether ``index`` is the last result its list shows (constant time: layouts ask per row)."""
        list_row = index.internalId() - 1
        return 0 <= list_row < len(self._counts) and index.row() == self._counts[list_row] - 1

    def query_index(self, key: str) -> QModelIndex:
        row = self._rows.get(key)
        return QModelIndex() if row is None else self.createIndex(row, 0, 0)

    def visible_keys(self) -> KeysView[str]:
        """Keys of the lists shown (a live set-like view)."""
        return self._rows.keys()

    def key_of(self, entry: HistoryEntry) -> str:
        meta = self._meta.get(id(entry))
        return meta.key if meta else entry_key(entry)

    def cell_text(self, entry: HistoryEntry, result: ResultRow, position: int) -> str:
        meta = self._meta.get(id(entry))
        if meta is None or position >= len(meta.cell_map):
            return ""
        source = meta.cell_map[position]
        return result.cells[source] if 0 <= source < len(result.cells) else ""

    # ----------------------------------------------------------------- Qt model
    # The tree calls index/parent/rowCount/flags for every row it lays out: plain bookkeeping,
    # no calls back into Qt (hasIndex would call rowCount and columnCount again).
    def index(self, row: int, column: int, parent: AnyIndex = QModelIndex()) -> QModelIndex:
        if not 0 <= column <= len(self._columns):
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, 0) if 0 <= row < len(self._visible) else QModelIndex()
        counts, list_row = self._counts, parent.row()
        if parent.internalId() or parent.column() or list_row >= len(counts) or not 0 <= row < counts[list_row]:
            return QModelIndex()
        return self.createIndex(row, column, list_row + 1)

    def parent(self, index: AnyIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        list_id = index.internalId()  # 0 for query rows and invalid indexes
        return self.createIndex(list_id - 1, 0, 0) if list_id else QModelIndex()

    def rowCount(self, parent: AnyIndex = QModelIndex()) -> int:  # noqa: N802
        if not parent.isValid():
            return len(self._visible)
        list_row = parent.row()
        if parent.internalId() or parent.column() or list_row >= len(self._counts):
            return 0
        return self._counts[list_row]

    def columnCount(self, parent: AnyIndex = QModelIndex()) -> int:  # noqa: N802
        return 1 + len(self._columns)

    def flags(self, index: AnyIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return _RESULT_FLAGS if index.internalId() else _QUERY_FLAGS

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return "Match" if section == 0 else self._columns[section - 1].name
        if role == COLUMN_ROLE and section > 0:
            return self._columns[section - 1]
        if role == Qt.ItemDataRole.ToolTipRole and section > 0:
            info = self._columns[section - 1]
            if info.kind == "deciding":
                return f"{info.name} is a deciding column: results should agree with the identity a query names."
            if info.kind == "primary":
                return f"{info.name} describes each row and is compared with your queries."
        return None

    def data(self, index: AnyIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        entry = self.entry_for(index)
        if entry is None:
            return None
        if index.internalId() == 0:
            return self._query_data(entry, index.column(), role)
        if index.row() >= len(entry.results):
            return None
        return self._result_data(entry, entry.results[index.row()], index.column(), role)

    # ----------------------------------------------------------------- internals
    def _query_data(self, entry: HistoryEntry, column: int, role: int):
        if role == KIND_ROLE:
            return QUERY
        if role == ENTRY_ROLE:
            return entry
        if role == KEY_ROLE:
            return self._meta[id(entry)].key
        if role == Qt.ItemDataRole.DisplayRole:
            return entry.query if column == 0 else None
        if role == Qt.ItemDataRole.ToolTipRole:
            lines = [entry.query, f"{plural(len(entry.results), 'result')} · searched {format_timestamp(entry.created_at)}"]
            lines += [f"{e.column} → {e.value} (from “{e.matched_text}”, {percent(e.strength)})" for e in entry.evidence]
            return "\n".join(lines)
        return None

    def _result_data(self, entry: HistoryEntry, result: ResultRow, column: int, role: int):
        if role == KIND_ROLE:
            return RESULT
        if role == ENTRY_ROLE:
            return entry
        if role == RESULT_ROLE:
            return result
        info = self._columns[column - 1] if 0 < column <= len(self._columns) else None
        if role == COLUMN_ROLE:
            return info
        if role == CELL_STATE_ROLE:
            return cell_state(result, info.name) if info else None
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            if column == 0:
                return f"#{result.rank} · {percent(result.score)} match"
            text = self.cell_text(entry, result, column - 1)
            if role == Qt.ItemDataRole.ToolTipRole and info is not None:
                state = cell_state(result, info.name)
                note = {"agree": "agrees with the query", "conflict": "differs from the query's identity",
                        "exact": "matched the query exactly"}.get(state or "")
                return f"{text}\n{info.name} {note}" if note and text else (text or None)
            return text
        return None

    def _make_meta(self, entry: HistoryEntry, position: int) -> _Meta:
        stamp = parse_time(entry.created_at)
        values = " ".join(e.value for e in entry.evidence)
        return _Meta(
            key=entry_key(entry),
            timestamp=stamp.timestamp() if stamp else 0.0,
            haystack=fold(f"{entry.query} {values}"),
            position=position,
            cell_map=self._cell_map(entry),
        )

    def _cell_map(self, entry: HistoryEntry) -> list[int]:
        lookup = {name: i for i, name in enumerate(entry.columns)}
        return [lookup.get(info.name, -1) for info in self._columns]

    def _rebuild_visible(self) -> None:
        entries = self._entries
        if self._filter_terms:
            entries = [e for e in entries if all(t in self._meta[id(e)].haystack for t in self._filter_terms)]
        if self._sort == SORT_ALPHA:
            entries = sorted(entries, key=lambda e: (fold(e.query).strip(), e.query, self._meta[id(e)].position))
        else:
            entries = sorted(entries, key=lambda e: (-self._meta[id(e)].timestamp, self._meta[id(e)].position))
        self._visible = entries
        self._counts = [min(self._k, len(e.results)) for e in entries]
        self._rows = {}
        for row, entry in enumerate(entries):
            self._rows.setdefault(self._meta[id(entry)].key, row)

    def _move_rows(self) -> None:
        """Re-sort and re-filter as a layout change rather than a reset: views keep the expansion,
        selection and current row of every list that stays visible (their persistent indexes
        follow it), those of lists the filter hides become invalid."""
        self.layoutAboutToBeChanged[_WITH_HINT].emit([], _ROWS_MOVED)
        before = self._visible
        self._rebuild_visible()
        now = {id(entry): row for row, entry in enumerate(self._visible)}
        moved = [now.get(id(entry), -1) for entry in before]  # old row -> new row, -1 when hidden
        sources = self.persistentIndexList()
        targets = []
        for index in sources:
            list_id = index.internalId()
            old = list_id - 1 if list_id else index.row()
            row = moved[old] if 0 <= old < len(moved) else -1
            if row < 0:
                targets.append(QModelIndex())
            elif list_id:
                targets.append(self.createIndex(index.row(), index.column(), row + 1))
            else:
                targets.append(self.createIndex(row, index.column(), 0))
        self.changePersistentIndexList(sources, targets)
        self.layoutChanged[_WITH_HINT].emit([], _ROWS_MOVED)
