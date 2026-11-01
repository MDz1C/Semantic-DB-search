"""Choosing what each column means (R3): a role editor shared by the import dialog and the
Columns dialog, which also shows identity aliases learned from the data (R4)."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from ..core.types import ColumnProfile, Role, Roles, Table
from .services import Backend
from .theme import tokens
from .widgets import Card, Dialog, ElidedLabel, IconWidget, SegmentedControl, button, label
from .workers import TaskRunner

EXPLAINER = (
    "Deciding = identity that must match (brokerage, issuer, ISIN). "
    "Search = text that describes the item (fund name)."
)
INVALID_MESSAGE = "Choose at least one Search or Deciding column."

ROLE_OPTIONS = [(Role.DECIDING.value, "Deciding"), (Role.SEARCH.value, "Search"), (Role.IGNORE.value, "Display only")]
ROLE_ACCENTS = {
    Role.DECIDING.value: ("accent_soft", "accent"),
    Role.SEARCH.value: ("segment", "text"),
    Role.IGNORE.value: ("segment", "text_muted"),
}


@dataclass(slots=True)
class ColumnChoice:
    name: str
    samples: list[str]
    role: Role
    suggested: Role | None = None
    reason: str = ""


def choices_from_profiles(profiles: list[ColumnProfile], roles: Roles) -> list[ColumnChoice]:
    return [
        ColumnChoice(p.name, list(p.samples), roles.get(p.name, p.suggested_role), p.suggested_role, p.reason)
        for p in profiles
    ]


def quick_samples(table: Table, limit: int = 5, scan: int = 2000) -> dict[str, list[str]]:
    """Up to ``limit`` distinct non-empty values per column from the first rows (cheap, UI thread)."""
    samples: dict[str, list[str]] = {name: [] for name in table.columns}
    for row in table.rows[:scan]:
        for name, value in zip(table.columns, row):
            bucket = samples[name]
            if value and len(bucket) < limit and value not in bucket:
                bucket.append(value)
    return samples


def describe_aliases(aliases: dict[str, dict[str, list[str]]], limit: int = 14) -> str:
    parts = [f"{word} → {value}" for values in aliases.values() for value, words in values.items() for word in words]
    text = "   ·   ".join(parts[:limit])
    if len(parts) > limit:
        text += f"   ·   +{len(parts) - limit} more"
    return text


class _RoleRow(QWidget):
    changed = Signal()

    def __init__(self, choice: ColumnChoice, divider: bool) -> None:
        super().__init__()
        self.choice = choice
        self._divider = divider
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(16)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(label(choice.name, "strong"))
        self._samples = ElidedLabel(" · ".join(choice.samples) if choice.samples else "Empty column")
        self._samples.setProperty("role", "caption")
        texts.addWidget(self._samples)
        layout.addLayout(texts, 1)
        self._reason = label("", "faint", wrap=True)
        self._reason.setFixedWidth(220)
        layout.addWidget(self._reason)
        self.segment = SegmentedControl(ROLE_OPTIONS, choice.role.value, accents=ROLE_ACCENTS, height=32,
                                        segment_padding=12)
        self.segment.setAccessibleName(f"Role of {choice.name}")
        self.segment.changed.connect(lambda _value: self.changed.emit())
        layout.addWidget(self.segment)
        self.set_reason(choice.suggested, choice.reason)

    def set_reason(self, suggested: Role | None, reason: str) -> None:
        self.choice.suggested, self.choice.reason = suggested, reason
        self._reason.setText(reason)
        tip = f"Suggested: {suggested.label}" if suggested else ""
        self.setToolTip(f"{tip} — {reason}" if tip and reason else tip)

    def set_samples(self, samples: list[str]) -> None:
        self._samples.setText(" · ".join(samples) if samples else "Empty column")

    def role(self) -> Role:
        return Role(self.segment.value())

    def paintEvent(self, event) -> None:
        if self._divider:
            painter = QPainter(self)
            painter.fillRect(QRect(16, self.height() - 1, self.width() - 32, 1), tokens().border)


class RoleEditor(QWidget):
    """One row per column with a Deciding | Search | Display only control."""

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[_RoleRow] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        card = Card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(1, 4, 1, 4)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("Transparent")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        card_layout.addWidget(self._scroll)
        outer.addWidget(card)

    def set_choices(self, choices: list[ColumnChoice]) -> None:
        content = QWidget()
        content.setObjectName("Transparent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._rows = []
        for i, choice in enumerate(choices):
            row = _RoleRow(choice, divider=i < len(choices) - 1)
            row.changed.connect(self.changed)
            layout.addWidget(row)
            self._rows.append(row)
        layout.addStretch(1)
        self._scroll.setWidget(content)
        self.changed.emit()

    def set_reasons(self, profiles: list[ColumnProfile]) -> None:
        by_name = {p.name: p for p in profiles}
        for row in self._rows:
            profile = by_name.get(row.choice.name)
            if profile is not None:
                row.set_reason(profile.suggested_role, profile.reason)
                if profile.samples:
                    row.set_samples(list(profile.samples))

    def roles(self) -> Roles:
        return {row.choice.name: row.role() for row in self._rows}

    def set_role(self, name: str, role: Role) -> None:
        for row in self._rows:
            if row.choice.name == name:
                row.segment.set_value(role.value, emit=True)

    def is_valid(self) -> bool:
        return any(role in (Role.SEARCH, Role.DECIDING) for role in self.roles().values())


class ColumnsDialog(Dialog):
    """Change column roles of the loaded database; Apply rebuilds the index (the cache keeps it quick)."""

    def __init__(
        self,
        parent: QWidget | None,
        table: Table,
        roles: Roles,
        backend: Backend,
        runner: TaskRunner,
        aliases: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumSize(720, 520)
        self.resize(860, 660)
        head = QVBoxLayout()
        head.setSpacing(4)
        head.addWidget(label("How should each column be used?", "heading"))
        head.addWidget(label(f"{table.label}  ·  {len(table):,} rows", "caption"))
        self.body.addLayout(head)
        self.body.addWidget(label(EXPLAINER, "muted", wrap=True))
        self.editor = RoleEditor()
        samples = quick_samples(table)
        self.editor.set_choices([
            ColumnChoice(name, samples[name], roles.get(name, Role.IGNORE)) for name in table.columns
        ])
        self.body.addWidget(self.editor, 1)
        text = describe_aliases(aliases or {})
        if text:
            box = Card(name="SoftCard")
            box_layout = QHBoxLayout(box)
            box_layout.setContentsMargins(14, 12, 14, 12)
            box_layout.setSpacing(12)
            box_layout.addWidget(IconWidget("sparkles", "accent", 18), 0, Qt.AlignmentFlag.AlignTop)
            texts = QVBoxLayout()
            texts.setSpacing(2)
            texts.addWidget(label("Learned automatically from your data", "strong"))
            texts.addWidget(label(text, "caption", wrap=True, selectable=True))
            box_layout.addLayout(texts, 1)
            self.body.addWidget(box)
        self._invalid = label(INVALID_MESSAGE, "danger")
        self._invalid.hide()
        cancel = button("Cancel")
        cancel.clicked.connect(self.reject)
        self.apply_button = button("Apply", "primary")
        self.apply_button.setToolTip("Rebuild the search index with these roles")
        self.apply_button.clicked.connect(self.accept)
        self.apply_button.setDefault(True)
        row = self.add_buttons(cancel, self.apply_button)
        row.insertWidget(0, self._invalid)
        self.editor.changed.connect(self._validate)
        self._validate()
        self._open = True
        self._profile_task = runner.start(
            lambda _progress, _cancel: backend.profile_columns(table),
            name="profile-columns",
            on_done=self._on_profiles,
        )

    def _on_profiles(self, profiles: list[ColumnProfile]) -> None:
        if self._open:
            self.editor.set_reasons(profiles)

    def roles(self) -> Roles:
        return self.editor.roles()

    def _validate(self) -> None:
        valid = self.editor.is_valid()
        self._invalid.setVisible(not valid)
        self.apply_button.setEnabled(valid)

    def done(self, result: int) -> None:
        self._open = False
        self._profile_task.cancel()
        super().done(result)
