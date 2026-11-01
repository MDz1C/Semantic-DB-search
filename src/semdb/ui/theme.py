"""Visual system: design tokens (light/dark), fonts, the Qt palette and one generated stylesheet.

The app follows the operating-system colour scheme live unless the user picks Light or
Dark (ARCHITECTURE.md §12.1).  Custom-painted widgets read colours from :func:`tokens` at
paint time and repaint on :attr:`ThemeManager.changed`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QStyleFactory

ThemeMode = Literal["system", "light", "dark"]
THEME_MODES: tuple[ThemeMode, ...] = ("system", "light", "dark")

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Arial"]
BASE_PT = 10.0
CAPTION_PT = 9.0
SMALL_PT = 8.5
TITLE_PT = 12.0
HEADING_PT = 13.0
HERO_PT = 15.0

SPACE = 8  # spacing grid unit
CONTROL_HEIGHT = 34
RADIUS_CARD = 12
RADIUS_CONTROL = 8

# name: (light, dark).  The first block is the specified palette, the second is derived from it.
_COLORS: dict[str, tuple[str, str]] = {
    "bg": ("#F5F6FA", "#0E1014"),
    "surface": ("#FFFFFF", "#161A21"),
    "surface_alt": ("#F8F9FC", "#1B2029"),
    "border": ("#E3E6ED", "#262C36"),
    "border_strong": ("#CDD2DC", "#343B47"),
    "text": ("#111827", "#E7EAF0"),
    "text_muted": ("#6B7280", "#9AA3B2"),
    "text_faint": ("#9CA3AF", "#6B7383"),
    "accent": ("#4F46E5", "#818CF8"),
    "accent_hover": ("#4338CA", "#A5B4FC"),
    "accent_soft": ("#EEF2FF", "#1E2040"),
    "on_accent": ("#FFFFFF", "#0E1014"),
    "success": ("#059669", "#34D399"),
    "success_soft": ("#ECFDF5", "#0F2A22"),
    "warning": ("#D97706", "#FBBF24"),
    "warning_soft": ("#FFFBEB", "#2B2412"),
    "danger": ("#DC2626", "#F87171"),
    "danger_soft": ("#FEF2F2", "#2C1618"),
    "selection": ("#E0E7FF", "#272A4D"),
    # derived
    "hover": ("#F4F5F9", "#1B2029"),
    "hover_alt": ("#EEF0F5", "#222833"),
    "pressed": ("#E9EBF1", "#262C36"),
    "track": ("#EEF0F4", "#0F1217"),
    "segment": ("#FFFFFF", "#262C36"),
    "neutral_soft": ("#F1F3F7", "#222833"),
    "accent_disabled": ("#B4B0F4", "#3B416C"),
    "accent_border": ("#C7D2FE", "#3A3F78"),
    "tooltip_bg": ("#1F2430", "#2A303B"),
    "tooltip_text": ("#F5F6FA", "#E7EAF0"),
    "scrim": ("#E6F5F6FA", "#E60E1014"),  # #AARRGGBB
    "shadow": ("#14101828", "#66000000"),
}


@dataclass(frozen=True, slots=True)
class Tokens:
    """Resolved colours for one scheme."""

    dark: bool
    bg: QColor
    surface: QColor
    surface_alt: QColor
    border: QColor
    border_strong: QColor
    text: QColor
    text_muted: QColor
    text_faint: QColor
    accent: QColor
    accent_hover: QColor
    accent_soft: QColor
    on_accent: QColor
    success: QColor
    success_soft: QColor
    warning: QColor
    warning_soft: QColor
    danger: QColor
    danger_soft: QColor
    selection: QColor
    hover: QColor
    hover_alt: QColor
    pressed: QColor
    track: QColor
    segment: QColor
    neutral_soft: QColor
    accent_disabled: QColor
    accent_border: QColor
    tooltip_bg: QColor
    tooltip_text: QColor
    scrim: QColor
    shadow: QColor

    def color(self, name: str) -> QColor:
        """Colour by token name (``"accent"``...), for code that receives roles as strings."""
        return QColor(getattr(self, name))


def make_tokens(dark: bool) -> Tokens:
    values = {name: QColor(pair[1] if dark else pair[0]) for name, pair in _COLORS.items()}
    return Tokens(dark=dark, **values)


def css(color: QColor) -> str:
    """QColor as a stylesheet colour literal (rgba when translucent)."""
    if color.alpha() == 255:
        return color.name()
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {color.alpha()})"


def mix(a: QColor, b: QColor, amount: float) -> QColor:
    """Blend ``amount`` of ``b`` into ``a``."""
    inv = 1.0 - amount
    return QColor(
        round(a.red() * inv + b.red() * amount),
        round(a.green() * inv + b.green() * amount),
        round(a.blue() * inv + b.blue() * amount),
        round(a.alpha() * inv + b.alpha() * amount),
    )


def with_alpha(color: QColor, alpha: float) -> QColor:
    out = QColor(color)
    out.setAlphaF(max(0.0, min(1.0, alpha)))
    return out


def font(size: float = BASE_PT, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(FONT_FAMILIES)
    f.setPointSizeF(size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def caption_font(weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    return font(CAPTION_PT, weight)


def section_font() -> QFont:
    """Small uppercase label used for panel sections ("FIELDS")."""
    f = font(SMALL_PT, QFont.Weight.DemiBold)
    f.setCapitalization(QFont.Capitalization.AllUppercase)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)
    return f


def build_palette(t: Tokens) -> QPalette:
    p = QPalette()
    roles = {
        QPalette.ColorRole.Window: t.bg,
        QPalette.ColorRole.WindowText: t.text,
        QPalette.ColorRole.Base: t.surface,
        QPalette.ColorRole.AlternateBase: t.surface_alt,
        QPalette.ColorRole.ToolTipBase: t.tooltip_bg,
        QPalette.ColorRole.ToolTipText: t.tooltip_text,
        QPalette.ColorRole.PlaceholderText: t.text_faint,
        QPalette.ColorRole.Text: t.text,
        QPalette.ColorRole.Button: t.surface,
        QPalette.ColorRole.ButtonText: t.text,
        QPalette.ColorRole.BrightText: t.danger,
        QPalette.ColorRole.Light: t.surface,
        QPalette.ColorRole.Midlight: t.surface_alt,
        QPalette.ColorRole.Mid: t.border,
        QPalette.ColorRole.Dark: t.border_strong,
        QPalette.ColorRole.Shadow: t.shadow,
        QPalette.ColorRole.Highlight: t.selection,
        QPalette.ColorRole.HighlightedText: t.text,
        QPalette.ColorRole.Link: t.accent,
        QPalette.ColorRole.LinkVisited: t.accent,
        QPalette.ColorRole.Accent: t.accent,
    }
    for group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for role, color in roles.items():
            p.setColor(group, role, color)
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        p.setColor(QPalette.ColorGroup.Disabled, role, t.text_faint)
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, t.surface_alt)
    return p


def build_qss(t: Tokens, chevron_icon: str = "", check_icon: str = "") -> str:
    """The application stylesheet.  ``chevron_icon``/``check_icon`` are file paths of themed SVGs."""
    c = {f.name: css(getattr(t, f.name)) for f in fields(Tokens) if f.name != "dark"}
    arrow = f'image: url("{chevron_icon}");' if chevron_icon else ""
    check = f'image: url("{check_icon}");' if check_icon else ""
    return f"""
* {{ outline: 0; }}
QWidget {{ color: {c['text']}; }}
QMainWindow, QWidget#Central {{ background: {c['bg']}; }}
QDialog {{ background: {c['surface']}; }}
QToolTip {{
    background: {c['tooltip_bg']}; color: {c['tooltip_text']};
    border: 1px solid {c['tooltip_bg']}; padding: 5px 8px; font-size: 9pt;
}}
QFrame#Header, QFrame#DatasetBar {{ background: {c['surface']}; border: none; border-bottom: 1px solid {c['border']}; }}
QFrame#Card {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; }}
QFrame#Toast {{ background: {c['surface']}; border: 1px solid {c['border_strong']}; border-radius: 10px; }}
QFrame#SoftCard {{ background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 10px; }}
QFrame#Divider {{ background: {c['border']}; border: none; }}
QWidget#Transparent, QScrollArea#Transparent, QScrollArea#Transparent > QWidget > QWidget {{ background: transparent; border: none; }}

QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {c['text_muted']}; }}
QLabel[role="caption"] {{ color: {c['text_muted']}; font-size: 9pt; }}
QLabel[role="faint"] {{ color: {c['text_faint']}; font-size: 9pt; }}
QLabel[role="title"] {{ font-size: 12pt; font-weight: 600; }}
QLabel[role="heading"] {{ font-size: 13pt; font-weight: 600; }}
QLabel[role="hero"] {{ font-size: 15pt; font-weight: 600; }}
QLabel[role="strong"] {{ font-weight: 600; }}
QLabel[role="danger"] {{ color: {c['danger']}; font-size: 9pt; }}
QLabel[pill="accent"] {{ background: {c['accent_soft']}; color: {c['accent']}; border-radius: 11px; padding: 0 9px; font-size: 9pt; font-weight: 600; }}
QLabel[pill="neutral"] {{ background: {c['neutral_soft']}; color: {c['text_muted']}; border-radius: 11px; padding: 0 9px; font-size: 9pt; font-weight: 600; }}
QLabel[pill="count"] {{ background: {c['surface']}; color: {c['text_muted']}; border: 1px solid {c['border']}; border-radius: 10px; padding: 0 7px; font-size: 9pt; font-weight: 600; }}
QLabel#Kbd {{ background: {c['surface']}; color: {c['text_muted']}; border: 1px solid {c['border_strong']}; border-radius: 5px; padding: 0 5px; font-size: 8.5pt; }}

QPushButton {{
    background: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border_strong']}; border-radius: 8px;
    padding: 0 14px; font-weight: 600;
}}
QPushButton:hover {{ background: {c['hover']}; }}
QPushButton:pressed {{ background: {c['pressed']}; }}
QPushButton:disabled {{ color: {c['text_faint']}; border-color: {c['border']}; background: {c['surface']}; }}
QPushButton:focus {{ border-color: {c['accent']}; }}
QPushButton[variant="primary"] {{ background: {c['accent']}; color: {c['on_accent']}; border: 1px solid {c['accent']}; }}
QPushButton[variant="primary"]:hover {{ background: {c['accent_hover']}; border-color: {c['accent_hover']}; }}
QPushButton[variant="primary"]:pressed {{ background: {c['accent_hover']}; }}
QPushButton[variant="primary"]:disabled {{ background: {c['accent_disabled']}; border-color: {c['accent_disabled']}; color: {c['on_accent']}; }}
QPushButton[variant="primary"]:focus {{ border-color: {c['accent_hover']}; }}
QPushButton[variant="outline"] {{ background: {c['surface']}; color: {c['accent']}; border: 1px solid {c['accent_border']}; }}
QPushButton[variant="outline"]:hover {{ background: {c['accent_soft']}; border-color: {c['accent']}; }}
QPushButton[variant="outline"]:pressed {{ background: {c['accent_soft']}; }}
QPushButton[variant="ghost"] {{ background: transparent; color: {c['text_muted']}; border: 1px solid transparent; }}
QPushButton[variant="ghost"]:hover {{ background: {c['hover']}; color: {c['text']}; }}
QPushButton[variant="ghost"]:pressed {{ background: {c['pressed']}; }}
QPushButton[variant="ghost"]:disabled {{ background: transparent; border-color: transparent; color: {c['text_faint']}; }}
QPushButton[variant="ghost"]:focus {{ border-color: {c['accent_border']}; }}
QPushButton[variant="danger"] {{ background: {c['danger']}; color: {c['on_accent']}; border: 1px solid {c['danger']}; }}
QPushButton[variant="danger"]:hover {{ background: {c['danger']}; border-color: {c['text']}; }}
QPushButton[variant="danger-ghost"] {{ background: transparent; color: {c['danger']}; border: 1px solid transparent; }}
QPushButton[variant="danger-ghost"]:hover {{ background: {c['danger_soft']}; }}
QPushButton[variant="link"] {{ background: transparent; color: {c['accent']}; border: none; padding: 0 2px; }}
QPushButton[variant="link"]:hover {{ color: {c['accent_hover']}; text-decoration: underline; }}
QPushButton[variant="chip"] {{
    background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border_strong']};
    border-radius: 15px; padding: 0 12px; font-weight: 400;
}}
QPushButton[variant="chip"]:hover {{ background: {c['accent_soft']}; color: {c['accent']}; border-color: {c['accent_border']}; }}

QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 8px; padding: 0; }}
QToolButton:hover {{ background: {c['hover']}; }}
QToolButton:pressed {{ background: {c['pressed']}; }}
QToolButton:checked {{ background: {c['accent_soft']}; border-color: {c['accent_border']}; }}
QToolButton:focus {{ border-color: {c['accent_border']}; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QToolButton#StepButton {{ border-radius: 6px; }}

QLineEdit {{
    background: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border_strong']}; border-radius: 8px; padding: 0 8px;
    selection-background-color: {c['selection']}; selection-color: {c['text']};
}}
QLineEdit:hover {{ border-color: {c['text_faint']}; }}
QLineEdit:focus {{ border-color: {c['accent']}; }}
QLineEdit:disabled {{ color: {c['text_faint']}; background: {c['surface_alt']}; }}
QPlainTextEdit, QTextEdit {{
    background: {c['surface']}; color: {c['text']}; border: none;
    selection-background-color: {c['selection']}; selection-color: {c['text']};
}}
QPlainTextEdit#Composer {{ background: transparent; font-size: 10.5pt; }}

QFrame#Stepper {{ background: {c['surface']}; border: 1px solid {c['border_strong']}; border-radius: 8px; }}
QFrame#Stepper:hover {{ border-color: {c['text_faint']}; }}
QSpinBox {{
    background: transparent; border: none; color: {c['text']}; font-weight: 600; padding: 0;
    selection-background-color: {c['selection']}; selection-color: {c['text']};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    subcontrol-origin: border; width: 0px; height: 0px; border: none; background: transparent;
}}
QSpinBox::up-arrow, QSpinBox::down-arrow {{ image: none; width: 0; height: 0; }}

QComboBox {{
    background: {c['surface']}; color: {c['text']};
    border: 1px solid {c['border_strong']}; border-radius: 8px; padding: 0 30px 0 10px;
}}
QComboBox:hover {{ border-color: {c['text_faint']}; }}
QComboBox:focus {{ border-color: {c['accent']}; }}
QComboBox::drop-down {{ border: none; width: 28px; subcontrol-origin: padding; subcontrol-position: center right; }}
QComboBox::down-arrow {{ {arrow} width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{
    background: {c['surface']}; color: {c['text']}; border: 1px solid {c['border_strong']};
    padding: 4px; selection-background-color: {c['selection']}; selection-color: {c['text']};
}}

QMenu {{ background: {c['surface']}; border: 1px solid {c['border_strong']}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 24px 7px 10px; border-radius: 6px; color: {c['text']}; background: transparent; }}
QMenu::item:selected {{ background: {c['hover_alt']}; }}
QMenu::item:disabled {{ color: {c['text_faint']}; }}
QMenu::icon {{ padding-left: 10px; }}
QMenu::indicator {{ width: 16px; height: 16px; padding-left: 8px; }}
QMenu::indicator:checked {{ {check} }}
QMenu::separator {{ height: 1px; background: {c['border']}; margin: 5px 6px; }}
QMenu::right-arrow {{ width: 0; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px 2px 2px 0; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0 2px 2px 2px; }}
QScrollBar::handle:vertical {{ background: {c['border_strong']}; border-radius: 4px; min-height: 36px; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: {c['border_strong']}; border-radius: 4px; min-width: 36px; margin: 2px 0; }}
QScrollBar::handle:hover {{ background: {c['text_faint']}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
QAbstractScrollArea::corner {{ background: transparent; border: none; }}

QTreeView#Results {{ background: {c['surface']}; border: none; border-radius: 12px; }}
QTableView#Preview {{
    background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 8px;
    gridline-color: {c['border']}; selection-background-color: {c['selection']}; selection-color: {c['text']};
}}
QTableView#Preview QHeaderView::section {{
    background: {c['surface_alt']}; color: {c['text_muted']}; border: none;
    border-bottom: 1px solid {c['border']}; border-right: 1px solid {c['border']};
    padding: 0 8px; font-size: 9pt; font-weight: 600;
}}
QTableView#Preview QTableCornerButton::section {{ background: {c['surface_alt']}; border: none; border-bottom: 1px solid {c['border']}; }}

QSplitter::handle {{ background: transparent; }}
QStatusBar {{ background: {c['bg']}; border-top: 1px solid {c['border']}; color: {c['text_muted']}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ color: {c['text_muted']}; font-size: 9pt; }}
QProgressBar {{ background: {c['track']}; border: none; border-radius: 3px; color: transparent; }}
QProgressBar::chunk {{ background: {c['accent']}; border-radius: 3px; }}
"""


class ThemeManager(QObject):
    """Owns the current scheme; re-applies palette + stylesheet when it changes."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._mode: ThemeMode = "system"
        self._tokens = make_tokens(False)
        self._app: QApplication | None = None
        self._updating_hint = False

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def tokens(self) -> Tokens:
        return self._tokens

    @property
    def is_dark(self) -> bool:
        return self._tokens.dark

    def install(self, app: QApplication, mode: ThemeMode = "system") -> None:
        """Set the Fusion style, base font and the theme on ``app``."""
        self._app = app
        app.setStyle(QStyleFactory.create("Fusion"))
        app.setFont(font())
        app.styleHints().colorSchemeChanged.connect(self._on_scheme_changed)
        self.set_mode(mode, force=True)

    def set_mode(self, mode: ThemeMode, force: bool = False) -> None:
        if mode not in THEME_MODES:
            mode = "system"
        if mode == self._mode and not force:
            return
        self._mode = mode
        if self._app is not None:
            hints = self._app.styleHints()
            requested = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}.get(
                mode, Qt.ColorScheme.Unknown
            )
            self._updating_hint = True
            try:
                hints.setColorScheme(requested)  # also themes the native title bar
            finally:
                self._updating_hint = False
        self._apply()

    def _system_is_dark(self) -> bool:
        if self._app is None:
            return False
        return self._app.styleHints().colorScheme() == Qt.ColorScheme.Dark

    def _on_scheme_changed(self, _scheme: Qt.ColorScheme) -> None:
        if not self._updating_hint and self._mode == "system":
            self._apply()

    def _apply(self) -> None:
        dark = self._mode == "dark" or (self._mode == "system" and self._system_is_dark())
        self._tokens = make_tokens(dark)
        if self._app is not None:
            from . import icons  # local import: icons depends on this module

            self._app.setPalette(build_palette(self._tokens))
            self._app.setStyleSheet(
                build_qss(
                    self._tokens,
                    chevron_icon=icons.stylesheet_icon("chevron-down", self._tokens.text_muted),
                    check_icon=icons.stylesheet_icon("check", self._tokens.accent),
                )
            )
        self.changed.emit()


_manager: ThemeManager | None = None


def manager() -> ThemeManager:
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager


def tokens() -> Tokens:
    return manager().tokens


def score_colors(score: float, t: Tokens | None = None) -> tuple[QColor, QColor]:
    """(foreground, soft background) for a match score: green >= 85 %, amber >= 60 %, grey below."""
    t = t or tokens()
    if score >= 0.85:
        return t.success, t.success_soft
    if score >= 0.60:
        return t.warning, t.warning_soft
    return t.text_muted, t.neutral_soft
