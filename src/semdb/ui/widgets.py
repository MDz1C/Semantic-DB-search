"""Small reusable widgets: cards, buttons, stepper, segmented control, chips, toasts, overlays,
confirmation dialogs.  Colours always come from :mod:`theme` tokens at paint time."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from PySide6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics, QIcon, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME
from . import icons, theme
from .theme import CONTROL_HEIGHT, tokens, with_alpha


def repolish(widget: QWidget) -> None:
    """Re-evaluate stylesheet rules after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def label(text: str = "", role: str | None = None, *, wrap: bool = False, selectable: bool = False) -> QLabel:
    widget = QLabel(text)
    if role:
        widget.setProperty("role", role)
    widget.setWordWrap(wrap)
    if selectable:
        widget.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        widget.setCursor(Qt.CursorShape.IBeamCursor)
    return widget


def set_role(widget: QLabel, role: str) -> None:
    if widget.property("role") != role:
        widget.setProperty("role", role)
        repolish(widget)


class Button(QPushButton):
    """Push button whose icon sits a fixed 8 px from its label (Qt's own gap is too tight)."""

    ICON_GAP = 8

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        if self.icon().isNull() or not self.text():
            return hint
        return QSize(hint.width() + self.ICON_GAP, hint.height())

    def paintEvent(self, event) -> None:
        if self.icon().isNull() or not self.text():
            super().paintEvent(event)
            return
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        # Not option.icon: the style option's copy of a Python-engine icon renders null pixmaps.
        icon, size = self.icon(), self.iconSize()
        option.icon = QIcon()
        painter.drawControl(QStyle.ControlElement.CE_PushButtonBevel, option)
        text_width = option.fontMetrics.horizontalAdvance(option.text)
        shift = (size.width() + self.ICON_GAP) / 2
        label_option = QStyleOptionButton(option)
        label_option.rect = option.rect.translated(round(shift), 0)
        painter.drawControl(QStyle.ControlElement.CE_PushButtonLabel, label_option)
        if not self.isEnabled():
            mode = QIcon.Mode.Disabled
        elif self.underMouse():
            mode = QIcon.Mode.Active
        else:
            mode = QIcon.Mode.Normal
        x = self.width() / 2 + shift - text_width / 2 - self.ICON_GAP - size.width()
        y = (self.height() - size.height()) / 2
        pixmap = icon.pixmap(size, self.devicePixelRatioF(), mode)
        painter.drawPixmap(QPointF(round(x), round(y)), pixmap)


def button(
    text: str,
    variant: str = "secondary",
    icon_name: str | None = None,
    icon_color: str | None = None,
    *,
    height: int = CONTROL_HEIGHT,
    tooltip: str = "",
) -> QPushButton:
    """A push button in one of the stylesheet variants: primary, secondary, outline, ghost, danger,
    danger-ghost, link, chip."""
    widget = Button(text)
    if variant != "secondary":
        widget.setProperty("variant", variant)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    if variant != "link":
        widget.setFixedHeight(height)
    if icon_name:
        default = {"primary": "on_accent", "outline": "accent", "danger": "on_accent", "danger-ghost": "danger"}
        color = icon_color or default.get(variant, "text_muted")
        disabled = "on_accent" if variant == "primary" else "text_faint"
        icons.bind(widget, icon_name, color, disabled=disabled, active="text" if color == "text_muted" else color)
        widget.setIconSize(QSize(16, 16))
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


def tool_button(
    icon_name: str,
    tooltip: str,
    *,
    size: int = CONTROL_HEIGHT,
    icon_size: int = 18,
    checkable: bool = False,
    color: str = "text_muted",
) -> QToolButton:
    widget = QToolButton()
    icons.bind(widget, icon_name, color)
    widget.setIconSize(QSize(icon_size, icon_size))
    widget.setFixedSize(size, size)
    widget.setToolTip(tooltip)
    widget.setCheckable(checkable)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.setAutoRaise(True)
    return widget


def menu(parent: QWidget | None = None, title: str = "") -> QMenu:
    """A popup menu with real rounded corners (translucent, frameless window)."""
    widget = QMenu(title, parent)
    widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    widget.setWindowFlags(
        widget.windowFlags() | Qt.WindowType.FramelessWindowHint | Qt.WindowType.NoDropShadowWindowHint
    )
    return widget


class Card(QFrame):
    """Rounded surface with a hairline border (``QFrame#Card`` in the stylesheet)."""

    def __init__(self, parent: QWidget | None = None, name: str = "Card") -> None:
        super().__init__(parent)
        self.setObjectName(name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)


class Divider(QFrame):
    def __init__(self, vertical: bool = False, length: int | None = None) -> None:
        super().__init__()
        self.setObjectName("Divider")
        if vertical:
            self.setFixedWidth(1)
            if length:
                self.setFixedHeight(length)
        else:
            self.setFixedHeight(1)
            if length:
                self.setFixedWidth(length)


class IconWidget(QWidget):
    """A themed icon painted as vector art."""

    def __init__(self, name: str, color: str = "text_muted", size: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._color = color
        self.setFixedSize(size, size)

    def set_icon(self, name: str, color: str | None = None) -> None:
        self._name = name
        if color:
            self._color = color
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        icons.paint(painter, self._name, QRectF(self.rect()), self._color)


class IconBadge(QWidget):
    """An icon centred on a soft rounded tile (empty states, dialogs)."""

    def __init__(
        self,
        name: str,
        size: int = 64,
        *,
        color: str = "accent",
        background: str = "accent_soft",
        radius: float | None = None,
        icon_ratio: float = 0.44,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._name, self._color, self._background = name, color, background
        self._radius = size / 2 if radius is None else radius
        self._ratio = icon_ratio
        self.setFixedSize(size, size)

    def set_icon(self, name: str, color: str | None = None, background: str | None = None) -> None:
        self._name = name
        self._color = color or self._color
        self._background = background or self._background
        self.update()

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(t.color(self._background))
        painter.drawRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        side = self.width() * self._ratio
        offset = (self.width() - side) / 2
        icons.paint(painter, self._name, QRectF(offset, offset, side, side), self._color)


class LogoMark(QWidget):
    def __init__(self, size: int = 28, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        icons.paint_logo(painter, QRectF(self.rect()))


class ElidedLabel(QLabel):
    """Single-line label that elides with an ellipsis instead of growing."""

    def __init__(self, text: str = "", mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight) -> None:
        super().__init__()
        self._full = text
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = text
        self.setToolTip(text if text else "")
        super().setText(text)
        self.update()

    def full_text(self) -> str:
        return self._full

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, super().minimumSizeHint().height())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = self.contentsRect()
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.setFont(self.font())
        text = self.fontMetrics().elidedText(self._full, self._mode, rect.width())
        painter.drawText(rect, int(self.alignment() | Qt.AlignmentFlag.AlignVCenter), text)


class Stepper(QFrame):
    """``[ − 5 + ]`` integer control with a typed value in the middle."""

    valueChanged = Signal(int)

    def __init__(self, minimum: int, maximum: int, value: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Stepper")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(CONTROL_HEIGHT)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(0)
        self._minus = tool_button("minus", "Fewer", size=26, icon_size=14)
        self._minus.setObjectName("StepButton")
        self._plus = tool_button("plus", "More", size=26, icon_size=14)
        self._plus.setObjectName("StepButton")
        self._spin = QSpinBox()
        self._spin.setRange(minimum, maximum)
        self._spin.setValue(value)
        self._spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._spin.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._spin.setFrame(False)
        self._spin.setFixedWidth(38)
        self._spin.setAccessibleName("Results per query")
        layout.addWidget(self._minus)
        layout.addWidget(self._spin)
        layout.addWidget(self._plus)
        self._minus.clicked.connect(lambda: self._spin.stepBy(-1))
        self._plus.clicked.connect(lambda: self._spin.stepBy(1))
        self._spin.valueChanged.connect(self._on_changed)
        self._sync_buttons()

    def value(self) -> int:
        return self._spin.value()

    def setValue(self, value: int) -> None:  # noqa: N802
        self._spin.setValue(value)

    def spin_box(self) -> QSpinBox:
        return self._spin

    def _on_changed(self, value: int) -> None:
        self._sync_buttons()
        self.valueChanged.emit(value)

    def _sync_buttons(self) -> None:
        self._minus.setEnabled(self._spin.value() > self._spin.minimum())
        self._plus.setEnabled(self._spin.value() < self._spin.maximum())


class SegmentedControl(QWidget):
    """A row of mutually exclusive segments, e.g. ``Recent | A–Z``."""

    changed = Signal(str)

    def __init__(
        self,
        options: Sequence[tuple[str, str]],
        value: str | None = None,
        *,
        height: int = CONTROL_HEIGHT,
        accents: Mapping[str, tuple[str, str]] | None = None,
        segment_padding: int = 14,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._options = list(options)
        self._value = value if value is not None else self._options[0][0]
        self._accents = dict(accents or {})
        self._padding = segment_padding
        self._hover = -1
        self._font = theme.font(theme.BASE_PT - 0.5, QFont.Weight.DemiBold)
        self.setFixedHeight(height)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def value(self) -> str:
        return self._value

    def set_value(self, value: str, emit: bool = False) -> None:
        if value == self._value or value not in {key for key, _ in self._options}:
            return
        self._value = value
        self.update()
        if emit:
            self.changed.emit(value)

    def keys(self) -> list[str]:
        return [key for key, _ in self._options]

    def _natural_widths(self) -> list[int]:
        metrics = QFontMetrics(self._font)
        return [metrics.horizontalAdvance(text) + 2 * self._padding for _, text in self._options]

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(sum(self._natural_widths()) + 6, self.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return self.sizeHint()

    def _rects(self) -> list[QRectF]:
        widths = self._natural_widths()
        scale = (self.width() - 6) / max(1, sum(widths))
        rects: list[QRectF] = []
        x = 3.0
        for width in widths:
            w = width * scale
            rects.append(QRectF(x, 3, w, self.height() - 6))
            x += w
        return rects

    def _index_at(self, pos: QPointF) -> int:
        for i, rect in enumerate(self._rects()):
            if rect.contains(pos):
                return i
        return -1

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        outer = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        painter.setPen(QPen(t.border, 1))
        painter.setBrush(t.track)
        painter.drawRoundedRect(outer, 8, 8)
        painter.setFont(self._font)
        for i, ((key, text), rect) in enumerate(zip(self._options, self._rects())):
            if key == self._value:
                background, foreground = self._accents.get(key, ("segment", "text"))
                border = "accent_border" if background == "accent_soft" else "border_strong"
                if not t.dark and background == "segment":
                    border = "border"
                painter.setPen(QPen(t.color(border), 1))
                painter.setBrush(t.color(background))
                painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)
                painter.setPen(t.color(foreground))
            else:
                painter.setPen(t.text if i == self._hover else t.text_muted)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        if self.hasFocus() and self.property("keyboardFocus"):
            painter.setPen(QPen(t.accent, 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(outer, 8, 8)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            index = self._index_at(event.position())
            if index >= 0:
                self.set_value(self._options[index][0], emit=True)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        index = self._index_at(event.position())
        if index != self._hover:
            self._hover = index
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = -1
        self.update()

    def focusInEvent(self, event) -> None:
        keyboard = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        self.setProperty("keyboardFocus", keyboard)
        super().focusInEvent(event)

    def focusOutEvent(self, event) -> None:
        self.setProperty("keyboardFocus", False)
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        keys = self.keys()
        index = keys.index(self._value)
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Up):
            self.set_value(keys[max(0, index - 1)], emit=True)
        elif event.key() in (Qt.Key.Key_Right, Qt.Key.Key_Down):
            self.set_value(keys[min(len(keys) - 1, index + 1)], emit=True)
        else:
            super().keyPressEvent(event)


class StatusChip(QWidget):
    """Coloured dot + short state text, optionally with a tiny progress bar."""

    _KINDS = {
        "ready": ("success", "success_soft"),
        "working": ("accent", "accent_soft"),
        "warning": ("warning", "warning_soft"),
        "error": ("danger", "danger_soft"),
        "idle": ("text_muted", "neutral_soft"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._kind = "idle"
        self._text = ""
        self._progress: float | None = None
        self._font = theme.font(theme.CAPTION_PT, QFont.Weight.DemiBold)
        self.setFixedHeight(24)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_state(self, kind: str, text: str, progress: float | None = None) -> None:
        self._kind, self._text, self._progress = kind, text, progress
        self.setToolTip(text)
        self.updateGeometry()
        self.update()

    def state(self) -> tuple[str, str]:
        return self._kind, self._text

    def sizeHint(self) -> QSize:  # noqa: N802
        width = 10 + 8 + 7 + QFontMetrics(self._font).horizontalAdvance(self._text) + 10
        if self._progress is not None:
            width += 48
        return QSize(width, 24)

    def paintEvent(self, event) -> None:
        t = tokens()
        fg_name, bg_name = self._KINDS.get(self._kind, self._KINDS["idle"])
        fg, bg = t.color(fg_name), t.color(bg_name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
        painter.setBrush(fg)
        painter.drawEllipse(QRectF(10, 8, 8, 8))
        painter.setPen(fg)
        painter.setFont(self._font)
        text_rect = QRectF(25, 0, self.width() - 35, self.height())
        painter.drawText(text_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter), self._text)
        if self._progress is not None:
            bar = QRectF(self.width() - 50, 10, 40, 4)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(with_alpha(fg, 0.22))
            painter.drawRoundedRect(bar, 2, 2)
            painter.setBrush(fg)
            painter.drawRoundedRect(QRectF(bar.left(), bar.top(), max(4.0, bar.width() * self._progress), 4), 2, 2)


class ProgressButton(QPushButton):
    """Push button that can show a progress fill behind its label (Search → Cancel)."""

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._progress: float | None = None

    def set_progress(self, fraction: float | None) -> None:
        self._progress = None if fraction is None else max(0.0, min(1.0, fraction))
        self.update()

    def progress(self) -> float | None:
        return self._progress

    def paintEvent(self, event) -> None:
        if self._progress is None:
            super().paintEvent(event)
            return
        t = tokens()
        painter = QStylePainter(self)
        option = QStyleOptionButton()
        self.initStyleOption(option)
        text = option.text
        option.text = ""
        option.icon = QIcon()
        painter.drawControl(QStyle.ControlElement.CE_PushButton, option)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inner = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        clip = QPainterPath()
        clip.addRoundedRect(inner, 7, 7)
        painter.setClipPath(clip)
        painter.fillRect(QRectF(inner.left(), inner.top(), inner.width() * self._progress, inner.height()),
                         with_alpha(t.accent, 0.16))
        painter.setClipping(False)
        painter.setPen(t.text)
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class MeterBar(QWidget):
    """Thin horizontal bar showing a value in [0, 1]."""

    def __init__(self, value: float, color: str = "accent", height: int = 6, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value = max(0.0, min(1.0, value))
        self._color = color
        self.setFixedHeight(height)
        self.setMinimumWidth(40)

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = self.height() / 2
        painter.setBrush(t.track if not t.dark else t.border)
        painter.drawRoundedRect(QRectF(self.rect()), radius, radius)
        if self._value > 0:
            width = max(self.height(), self.width() * self._value)
            painter.setBrush(t.color(self._color))
            painter.drawRoundedRect(QRectF(0, 0, width, self.height()), radius, radius)


class ProgressLine(QWidget):
    """Rounded progress bar; ``None`` shows an indeterminate sliding segment."""

    def __init__(self, height: int = 6, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: float | None = 0.0
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setFixedHeight(height)

    def set_value(self, value: float | None) -> None:
        self._value = None if value is None else max(0.0, min(1.0, value))
        if self._value is None and self.isVisible():
            self._timer.start()
        elif self._value is not None:
            self._timer.stop()
        self.update()

    def value(self) -> float | None:
        return self._value

    def _tick(self) -> None:
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def showEvent(self, event) -> None:
        if self._value is None:
            self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        radius = self.height() / 2
        bounds = QRectF(self.rect())
        painter.setBrush(t.track if not t.dark else t.border)
        painter.drawRoundedRect(bounds, radius, radius)
        painter.setBrush(t.accent)
        if self._value is None:
            clip = QPainterPath()
            clip.addRoundedRect(bounds, radius, radius)
            painter.setClipPath(clip)
            width = bounds.width() * 0.3
            x = -width + (bounds.width() + width) * self._phase
            painter.drawRoundedRect(QRectF(x, 0, width, self.height()), radius, radius)
        elif self._value > 0:
            painter.drawRoundedRect(QRectF(0, 0, max(self.height(), bounds.width() * self._value), self.height()),
                                    radius, radius)


class CardCorners(QWidget):
    """Overlay that rounds an opaque child (a tree view) inside a card: paints the page colour
    outside the rounded rectangle plus the card's hairline border."""

    def __init__(self, card: QWidget, radius: float = 12, outside: str = "bg") -> None:
        super().__init__(card)
        self._radius = radius
        self._outside = outside
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        card.installEventFilter(self)
        self.setGeometry(card.rect())
        self.raise_()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Type.Resize, QEvent.Type.Show):
            self.setGeometry(self.parentWidget().rect())
            self.raise_()
        elif event.type() == QEvent.Type.ChildAdded:
            self.raise_()
        return False

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = QRectF(self.rect())
        rounded = QPainterPath()
        rounded.addRoundedRect(bounds.adjusted(0.5, 0.5, -0.5, -0.5), self._radius, self._radius)
        outside = QPainterPath()
        outside.addRect(bounds)
        painter.fillPath(outside.subtracted(rounded), t.color(self._outside))
        painter.setPen(QPen(t.border, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(rounded)


class Spinner(QWidget):
    """Indeterminate activity indicator."""

    def __init__(self, size: int = 20, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(size, size)

    def _tick(self) -> None:
        self._angle = (self._angle + 8) % 360
        self.update()

    def showEvent(self, event) -> None:
        self._timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = max(2.0, self.width() / 9)
        rect = QRectF(self.rect()).adjusted(width, width, -width, -width)
        painter.setPen(QPen(t.border, width))
        painter.drawEllipse(rect)
        pen = QPen(t.accent, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, -self._angle * 16, 100 * 16)


class DashedFrame(QWidget):
    """A dashed rounded outline (drop-zone affordance) around its layout."""

    def __init__(self, parent: QWidget | None = None, radius: float = 16) -> None:
        super().__init__(parent)
        self._radius = radius

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(t.border_strong, 1.4)
        pen.setDashPattern([5, 4])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), self._radius, self._radius)


class Toast(QFrame):
    """Transient notification; ``ToastManager`` positions and stacks them."""

    closed = Signal()

    _KIND_ICONS = {
        "info": ("info", "accent"),
        "success": ("check-circle", "success"),
        "error": ("alert-circle", "danger"),
        "warning": ("alert-triangle", "warning"),
    }

    def __init__(
        self,
        parent: QWidget,
        message: str,
        kind: str = "info",
        action_text: str | None = None,
        action: Callable[[], None] | None = None,
        timeout_ms: int = 4000,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Toast")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 8, 10)
        layout.setSpacing(10)
        name, color = self._KIND_ICONS.get(kind, self._KIND_ICONS["info"])
        layout.addWidget(IconWidget(name, color, 18), 0, Qt.AlignmentFlag.AlignVCenter)
        self._label = QLabel(message)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(320)
        layout.addWidget(self._label, 1)
        if action_text and action:
            link = button(action_text, "link")
            link.clicked.connect(action)
            link.clicked.connect(self.dismiss)
            layout.addWidget(link)
        close = tool_button("x", "Dismiss", size=24, icon_size=14)
        close.clicked.connect(self.dismiss)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignVCenter)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(tokens().shadow)
        self.setGraphicsEffect(shadow)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(timeout_ms)
        self._timer.timeout.connect(self.dismiss)
        if timeout_ms > 0:
            self._timer.start()

    def message(self) -> str:
        return self._label.text()

    def enterEvent(self, event) -> None:
        self._timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._timer.interval() > 0:
            self._timer.start(2000)
        super().leaveEvent(event)

    def dismiss(self) -> None:
        self._timer.stop()
        self.hide()
        self.closed.emit()
        self.deleteLater()


class ToastManager(QObject):
    """Stacks toasts at the bottom-right corner of ``anchor()`` (a rect in ``host`` coordinates)."""

    def __init__(self, host: QWidget, anchor: Callable[[], QRect], limit: int = 3) -> None:
        super().__init__(host)
        self._host = host
        self._anchor = anchor
        self._limit = limit
        self._toasts: list[Toast] = []
        host.installEventFilter(self)

    def show(
        self,
        message: str,
        kind: str = "info",
        action_text: str | None = None,
        action: Callable[[], None] | None = None,
        timeout_ms: int = 4000,
    ) -> Toast:
        toast = Toast(self._host, message, kind, action_text, action, timeout_ms)
        toast.closed.connect(lambda t=toast: self._forget(t))
        self._toasts.append(toast)
        while len(self._toasts) > self._limit:
            self._toasts[0].dismiss()
        toast.show()
        toast.raise_()
        self.relayout()
        return toast

    def toasts(self) -> list[Toast]:
        return list(self._toasts)

    def _forget(self, toast: Toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
        self.relayout()

    def relayout(self) -> None:
        area = self._anchor()
        y = area.bottom() - 16
        for toast in reversed(self._toasts):
            toast.adjustSize()
            toast.move(area.right() - 16 - toast.width(), y - toast.height())
            toast.raise_()
            y -= toast.height() + 8

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._host and event.type() == QEvent.Type.Resize:
            self.relayout()
        return False


class DropOverlay(QWidget):
    """Full-window hint shown while a file is dragged over the app."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.hide()
        parent.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Resize:
            self.setGeometry(self.parentWidget().rect())
        return False

    def show_overlay(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()

    def paintEvent(self, event) -> None:
        t = tokens()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), t.scrim)
        box = QRectF(0, 0, 440, 240)
        box.moveCenter(QRectF(self.rect()).center())
        pen = QPen(t.accent, 2)
        pen.setDashPattern([6, 4])
        painter.setPen(pen)
        painter.setBrush(t.surface)
        painter.drawRoundedRect(box, 16, 16)
        badge = QRectF(0, 0, 64, 64)
        badge.moveCenter(QPointF(box.center().x(), box.top() + 72))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(t.accent_soft)
        painter.drawEllipse(badge)
        icons.paint(painter, "import", badge.adjusted(18, 18, -18, -18), t.accent)
        painter.setPen(t.text)
        painter.setFont(theme.font(theme.HERO_PT, QFont.Weight.DemiBold))
        painter.drawText(QRectF(box.left(), box.top() + 118, box.width(), 32), Qt.AlignmentFlag.AlignCenter,
                         "Drop to import")
        painter.setPen(t.text_muted)
        painter.setFont(theme.font())
        painter.drawText(QRectF(box.left(), box.top() + 154, box.width(), 24), Qt.AlignmentFlag.AlignCenter,
                         "Excel, OpenDocument or CSV file")


class Dialog(QDialog):
    """Base dialog: app-titled, surface background, 24 px content margins."""

    def __init__(self, parent: QWidget | None = None, title: str = APP_NAME) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(24, 24, 24, 20)
        self.body.setSpacing(16)

    def add_buttons(self, *buttons: QPushButton) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        for widget in buttons:
            row.addWidget(widget)
        self.body.addLayout(row)
        return row


class MessageDialog(Dialog):
    """Icon + title + text, with one or two buttons."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        text: str,
        *,
        confirm_text: str = "OK",
        cancel_text: str | None = None,
        icon_name: str = "info",
        tone: str = "accent",
        destructive: bool = False,
        details: str = "",
    ) -> None:
        super().__init__(parent)
        self.setFixedWidth(460)
        top = QHBoxLayout()
        top.setSpacing(16)
        top.addWidget(IconBadge(icon_name, 40, color=tone, background=f"{tone}_soft"), 0, Qt.AlignmentFlag.AlignTop)
        texts = QVBoxLayout()
        texts.setSpacing(6)
        heading = label(title, "title", wrap=True)
        body = label(text, "muted", wrap=True, selectable=bool(details))
        texts.addWidget(heading)
        texts.addWidget(body)
        if details:
            detail = label(details, "faint", wrap=True, selectable=True)
            texts.addWidget(detail)
        top.addLayout(texts, 1)
        self.body.addLayout(top)
        buttons: list[QPushButton] = []
        if cancel_text:
            cancel = button(cancel_text)
            cancel.clicked.connect(self.reject)
            buttons.append(cancel)
        confirm = button(confirm_text, "danger" if destructive else "primary")
        confirm.clicked.connect(self.accept)
        confirm.setDefault(True)
        buttons.append(confirm)
        self.add_buttons(*buttons)
        self.confirm_button = confirm


def confirm(
    parent: QWidget | None,
    title: str,
    text: str,
    confirm_text: str,
    *,
    destructive: bool = True,
    icon_name: str = "alert-triangle",
) -> bool:
    dialog = MessageDialog(
        parent,
        title,
        text,
        confirm_text=confirm_text,
        cancel_text="Cancel",
        icon_name=icon_name,
        tone="danger" if destructive else "accent",
        destructive=destructive,
    )
    return dialog.exec() == QDialog.DialogCode.Accepted


def show_error(parent: QWidget | None, title: str, text: str, details: str = "") -> None:
    MessageDialog(parent, title, text, icon_name="alert-circle", tone="danger", details=details).exec()


def centered(widget: QWidget) -> QHBoxLayout:
    """``widget`` centred horizontally without an alignment flag, so wrapped labels keep their
    height-for-width (layouts ignore it for aligned items)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    row.addWidget(widget)
    row.addStretch(1)
    return row


def hbox(*items: QWidget | int | None, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QHBoxLayout:
    """Horizontal layout from widgets; an int adds that much spacing, ``None`` adds a stretch."""
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)
    for item in items:
        if item is None:
            layout.addStretch(1)
        elif isinstance(item, int):
            layout.addSpacing(item)
        else:
            layout.addWidget(item)
    return layout
