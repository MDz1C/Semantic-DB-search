"""Icons: inline Lucide SVGs recoloured per theme and rendered with QtSvg at any pixel ratio.

Icon artwork from Lucide (https://lucide.dev), ISC License:
Copyright (c) for portions of Lucide are held by Cole Bemis 2013-2022 as part of Feather (MIT).
All other copyright (c) for Lucide are held by Lucide Contributors 2022.
Permission to use, copy, modify, and/or distribute this software for any purpose with or
without fee is hereby granted, provided that the above copyright notice and this permission
notice appear in all copies.
"""

from __future__ import annotations

import tempfile
import weakref
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QByteArray, QObject, QRectF, QSize, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QIconEngine, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QAbstractButton

from .. import paths
from . import theme

STROKE = 1.8

_PATHS: dict[str, str] = {
    "import": '<path d="M12 3v12"/><path d="m8 11 4 4 4-4"/>'
    '<path d="M8 5H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-4"/>',
    "rotate-ccw": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
    "ellipsis": '<circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/><circle cx="5" cy="12" r="1"/>',
    "chevron-right": '<path d="m9 18 6-6-6-6"/>',
    "chevron-down": '<path d="m6 9 6 6 6-6"/>',
    "chevrons-down-up": '<path d="m7 20 5-5 5 5"/><path d="m7 4 5 5 5-5"/>',
    "chevrons-up-down": '<path d="m7 15 5 5 5-5"/><path d="m7 9 5-5 5 5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "x": '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "copy": '<rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>'
    '<path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>',
    "trash": '<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
    '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/>'
    '<line x1="14" x2="14" y1="11" y2="17"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/>'
    '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/>'
    '<line x1="12" x2="12" y1="15" y2="3"/>',
    "lock": '<rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 '
    '9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 '
    '.964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/><path d="M20 3v4"/>'
    '<path d="M22 5h-4"/><path d="M4 17v2"/><path d="M5 18H3"/>',
    "panel-right": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M15 3v18"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/>'
    '<path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/>'
    '<path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/>'
    '<line x1="12" x2="12" y1="17" y2="21"/>',
    "folder-open": '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 '
    '0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
    '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "alert-circle": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/>'
    '<line x1="12" x2="12.01" y1="16" y2="16"/>',
    "check-circle": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
    "x-circle": '<circle cx="12" cy="12" r="10"/><path d="m15 9-6 6"/><path d="m9 9 6 6"/>',
    "file-spreadsheet": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
    '<path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M8 13h2"/><path d="M14 13h2"/><path d="M8 17h2"/>'
    '<path d="M14 17h2"/>',
    "sheet": '<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><line x1="3" x2="21" y1="9" y2="9"/>'
    '<line x1="3" x2="21" y1="15" y2="15"/><line x1="9" x2="9" y1="9" y2="21"/>'
    '<line x1="15" x2="15" y1="9" y2="21"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/>'
    '<path d="M3 12A9 3 0 0 0 21 12"/>',
    "sliders": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/>'
    '<line x1="21" x2="12" y1="12" y2="12"/><line x1="8" x2="3" y1="12" y2="12"/>'
    '<line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" y1="20" y2="20"/>'
    '<line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
    '<line x1="16" x2="16" y1="18" y2="22"/>',
    "minus": '<path d="M5 12h14"/>',
    "plus": '<path d="M5 12h14"/><path d="M12 5v14"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "fingerprint": '<path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4"/><path d="M14 13.12c0 2.38 0 6.38-1 8.88"/>'
    '<path d="M17.29 21.02c.12-.6.43-2.3.5-3.02"/><path d="M2 12a10 10 0 0 1 18-6"/><path d="M2 16h.01"/>'
    '<path d="M21.8 16c.2-2 .131-5.354 0-6"/><path d="M5 19.5C5.5 18 6 15 6 12a6 6 0 0 1 .34-2"/>'
    '<path d="M8.65 22c.21-.66.45-1.32.57-2"/><path d="M9 6.8a6 6 0 0 1 9 5.2v2"/>',
    "corner-down-left": '<polyline points="9 10 4 15 9 20"/><path d="M20 4v7a4 4 0 0 1-4 4H4"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>',
    "text-search": '<path d="M21 6H3"/><path d="M10 12H3"/><path d="M10 18H3"/><circle cx="17" cy="15" r="3"/>'
    '<path d="m21 19-1.9-1.9"/>',
    "list-filter": '<path d="M3 6h18"/><path d="M7 12h10"/><path d="M10 18h4"/>',
    "mouse-pointer": '<path d="M12.586 12.586 19 19"/><path d="M3.688 3.037a.497.497 0 0 0-.651.651l6.5 15.999a.501'
    '.501 0 0 0 .947-.062l1.569-6.083a2 2 0 0 1 1.448-1.479l6.124-1.579a.5.5 0 0 0 .063-.947z"/>',
    "type": '<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/>'
    '<line x1="12" x2="12" y1="4" y2="20"/>',
    "shield-check": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 '
    '4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "hard-drive": '<line x1="22" x2="2" y1="12" y2="12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 '
    '2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><line x1="6" x2="6.01" y1="16" y2="16"/>'
    '<line x1="10" x2="10.01" y1="16" y2="16"/>',
}

# Drawn when assets/logo.svg is absent: indigo tile, table rows and a lens.
_FALLBACK_LOGO = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#6366F1"/><stop offset="1" stop-color="#4338CA"/></linearGradient></defs>
<rect width="64" height="64" rx="16" fill="url(#g)"/>
<g fill="none" stroke="#FFFFFF" stroke-width="4.5" stroke-linecap="round">
<path d="M15 21h20" stroke-opacity="0.65"/><path d="M15 31h12" stroke-opacity="0.65"/>
<path d="M15 41h8" stroke-opacity="0.65"/><circle cx="39" cy="37" r="9"/><path d="M45.5 43.5l5.5 5.5"/>
</g></svg>"""

ColorSpec = QColor | str  # a QColor or a theme token name such as "text_muted"


def names() -> list[str]:
    return sorted(_PATHS)


def _resolve(color: ColorSpec) -> QColor:
    return QColor(color) if isinstance(color, QColor) else theme.tokens().color(color)


def svg_bytes(name: str, color: QColor, stroke: float = STROKE) -> bytes:
    body = _PATHS[name].replace("currentColor", color.name())
    opacity = f' stroke-opacity="{color.alphaF():.3f}"' if color.alpha() < 255 else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color.name()}"{opacity} stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    ).encode()


@lru_cache(maxsize=1024)
def _renderer(name: str, rgba: int, stroke: float) -> QSvgRenderer:
    return QSvgRenderer(QByteArray(svg_bytes(name, QColor.fromRgba(rgba), stroke)))


def paint(painter: QPainter, name: str, rect: QRectF, color: ColorSpec, stroke: float = STROKE) -> None:
    """Draw icon ``name`` into ``rect`` (vector, crisp at any scale)."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _renderer(name, _resolve(color).rgba(), stroke).render(painter, rect)
    painter.restore()


def pixmap(name: str, size: int, color: ColorSpec, dpr: float = 1.0, stroke: float = STROKE) -> QPixmap:
    pm = QPixmap(max(1, round(size * dpr)), max(1, round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    paint(painter, name, QRectF(0, 0, size, size), color, stroke)
    painter.end()
    return pm


class _SvgIconEngine(QIconEngine):
    """Renders on demand at the exact size and device pixel ratio requested."""

    def __init__(self, name: str, normal: QColor, disabled: QColor, active: QColor, stroke: float) -> None:
        super().__init__()
        self._args = (name, QColor(normal), QColor(disabled), QColor(active), stroke)

    def _color(self, mode: QIcon.Mode) -> QColor:
        _, normal, disabled, active, _ = self._args
        if mode == QIcon.Mode.Disabled:
            return disabled
        if mode in (QIcon.Mode.Active, QIcon.Mode.Selected):
            return active
        return normal

    def paint(self, painter: QPainter, rect, mode: QIcon.Mode, state: QIcon.State) -> None:
        paint(painter, self._args[0], QRectF(rect), self._color(mode), self._args[4])

    def pixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State) -> QPixmap:
        return self.scaledPixmap(size, mode, state, 1.0)

    def scaledPixmap(self, size: QSize, mode: QIcon.Mode, state: QIcon.State, scale: float) -> QPixmap:
        side = min(size.width(), size.height())
        return pixmap(self._args[0], side, self._color(mode), scale, self._args[4])

    def clone(self) -> QIconEngine:
        return _SvgIconEngine(*self._args)


def icon(
    name: str,
    color: ColorSpec = "text_muted",
    *,
    disabled: ColorSpec = "text_faint",
    active: ColorSpec | None = None,
    stroke: float = STROKE,
) -> QIcon:
    normal = _resolve(color)
    return QIcon(_SvgIconEngine(name, normal, _resolve(disabled), _resolve(active or color), stroke))


_bindings: weakref.WeakKeyDictionary[QObject, tuple] = weakref.WeakKeyDictionary()
_hooked = False


def bind(target: QAbstractButton | QAction, name: str, color: str = "text_muted", **kwargs) -> None:
    """Set a themed icon on a button or action and refresh it whenever the theme changes."""
    global _hooked
    _bindings[target] = (name, color, kwargs)
    target.setIcon(icon(name, color, **kwargs))
    if not _hooked:
        theme.manager().changed.connect(refresh_bound)
        _hooked = True


def refresh_bound() -> None:
    for target, (name, color, kwargs) in list(_bindings.items()):
        try:
            target.setIcon(icon(name, color, **kwargs))
        except RuntimeError:  # the C++ object is gone
            _bindings.pop(target, None)


def stylesheet_icon(name: str, color: QColor) -> str:
    """Path of a themed SVG file for stylesheet ``image: url(...)`` rules (combo arrows, menu checks)."""
    folder = Path(tempfile.gettempdir()) / "Fundscope-ui"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}-{color.name()[1:]}.svg"
    if not path.exists():
        path.write_bytes(svg_bytes(name, color))
    return path.as_posix()


def _logo_renderer() -> QSvgRenderer:
    custom = paths.assets_dir() / "logo.svg"
    if custom.is_file():
        renderer = QSvgRenderer(str(custom))
        if renderer.isValid():
            return renderer
    return QSvgRenderer(QByteArray(_FALLBACK_LOGO))


def paint_logo(painter: QPainter, rect: QRectF) -> None:
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    _logo_renderer().render(painter, rect)
    painter.restore()


def logo_pixmap(size: int, dpr: float = 1.0) -> QPixmap:
    pm = QPixmap(max(1, round(size * dpr)), max(1, round(size * dpr)))
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    paint_logo(painter, QRectF(0, 0, size, size))
    painter.end()
    return pm


def app_icon() -> QIcon:
    """Window/taskbar icon: assets/icon.png when present, else the drawn logo."""
    png = paths.assets_dir() / "icon.png"
    if png.is_file():
        loaded = QIcon(str(png))
        if not loaded.isNull():
            return loaded
    result = QIcon()
    for size in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        result.addPixmap(logo_pixmap(size))
    return result
