"""Generate the Fundscope mark: assets/logo.svg, assets/icon.png (256 px) and assets/icon.ico.

The mark is a "breakout lens": a magnifying glass (search) whose lens holds a rising stock
line; the line's last stroke breaks out through a gap in the rim as an arrow, like a price
breaking out through resistance.  One SVG is the single source of truth; the PNG and every
ICO frame are rendered from it with Qt's SVG engine (supersampled), so the window icon, the
installer and the in-app logo always match.  Dev-time tool: ``python tools/make_icon.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SUPERSAMPLE = 4

TILE_FROM, TILE_TO = "#1E3A8A", "#0F172A"  # deep navy: the UI accent's blue family, finance-grade
LINE = "#34D399"  # emerald: the rising market
WHITE = "#FFFFFF"

LENS_R, RING_W, HANDLE_W, LINE_W = 60.0, 21.0, 30.0, 20.0
HANDLE_LEN = 54.0
GAP_MARGIN = 4.5  # clear space between the breaking line and the cut rim ends
CHART = [(-30.0, 22.0), (-11.0, 3.0), (4.0, 14.0), (34.0, -24.0)]  # lens-relative, y down
ARROW_TIP = (78.0, -70.0)
ARROW_LEN, ARROW_HALF = 30.0, 19.0


def _f(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _geometry() -> dict:
    """Every shape in lens-centred coordinates, plus the gap where the line exits the rim."""
    tip = ARROW_TIP
    last = CHART[-1]
    dx, dy = tip[0] - last[0], tip[1] - last[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    base = (tip[0] - ARROW_LEN * ux, tip[1] - ARROW_LEN * uy)
    corners = [(base[0] + s * ARROW_HALF * -uy, base[1] + s * ARROW_HALF * ux) for s in (1, -1)]
    # Where the breakout segment crosses the rim's centre line (|last + t*(dx,dy)| = R).
    a = dx * dx + dy * dy
    b = 2 * (last[0] * dx + last[1] * dy)
    c = last[0] ** 2 + last[1] ** 2 - LENS_R**2
    t = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    exit_angle = math.atan2(last[1] + t * dy, last[0] + t * dx)
    half_gap = math.asin(min(1.0, (LINE_W / 2 + RING_W / 2 + GAP_MARGIN) / LENS_R))
    handle_from = LENS_R + RING_W / 2 - 2
    handle = [(handle_from * math.cos(math.pi / 4), handle_from * math.sin(math.pi / 4))]
    handle.append((handle[0][0] + HANDLE_LEN * math.cos(math.pi / 4), handle[0][1] + HANDLE_LEN * math.sin(math.pi / 4)))
    return {"base": base, "corners": corners, "exit": exit_angle, "half_gap": half_gap, "handle": handle}


def svg() -> str:
    g = _geometry()
    # Centre the mark's visual bounding box on the 256 tile.
    xs = [-LENS_R - RING_W / 2, g["handle"][1][0] + HANDLE_W / 2, ARROW_TIP[0]]
    ys = [ARROW_TIP[1], g["handle"][1][1] + HANDLE_W / 2, LENS_R + RING_W / 2]
    cx = 128 - (min(xs) + max(xs)) / 2
    cy = 128 - (min(ys) + max(ys)) / 2

    def pt(p: tuple[float, float]) -> str:
        return f"{_f(p[0] + cx)} {_f(p[1] + cy)}"

    start, end = g["exit"] + g["half_gap"], g["exit"] - g["half_gap"] + 2 * math.pi
    rim_from = (LENS_R * math.cos(start), LENS_R * math.sin(start))
    rim_to = (LENS_R * math.cos(end), LENS_R * math.sin(end))
    chart = " L ".join(pt(p) for p in [*CHART, g["base"]])
    arrow = " L ".join(pt(p) for p in [ARROW_TIP, *g["corners"]])
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256">
  <title>Fundscope</title>
  <defs>
    <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{TILE_FROM}"/>
      <stop offset="1" stop-color="{TILE_TO}"/>
    </linearGradient>
  </defs>
  <rect x="8" y="8" width="240" height="240" rx="56" fill="url(#tile)"/>
  <circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(LENS_R - RING_W / 2)}" fill="{WHITE}" fill-opacity="0.14"/>
  <path d="M {pt(rim_from)} A {_f(LENS_R)} {_f(LENS_R)} 0 1 1 {pt(rim_to)}" fill="none" stroke="{WHITE}" stroke-width="{_f(RING_W)}" stroke-linecap="round"/>
  <path d="M {pt(g['handle'][0])} L {pt(g['handle'][1])}" stroke="{WHITE}" stroke-width="{_f(HANDLE_W)}" stroke-linecap="round"/>
  <path d="M {chart}" fill="none" stroke="{LINE}" stroke-width="{_f(LINE_W)}" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M {arrow} Z" fill="{LINE}" stroke="{LINE}" stroke-width="6" stroke-linejoin="round"/>
</svg>
"""


def render(svg_text: str, size: int):
    """RGBA Pillow image of the SVG at ``size`` px (rendered larger, then downsampled)."""
    from PIL import Image
    from PySide6.QtCore import QByteArray, QRectF, Qt
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer

    if QGuiApplication.instance() is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        render.app = QGuiApplication(sys.argv[:1])  # kept alive for later renders
    big = size * SUPERSAMPLE
    image = QImage(big, big, QImage.Format.Format_RGBA8888)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg_text.encode("utf-8"))).render(painter, QRectF(0, 0, big, big))
    painter.end()
    data = bytes(image.constBits())
    frame = Image.frombuffer("RGBA", (big, big), data, "raw", "RGBA", image.bytesPerLine(), 1)
    return frame.resize((size, size), Image.Resampling.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ASSETS)
    parser.add_argument("--preview", type=Path, help="also write a strip of every icon size here")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    text = svg()
    (args.out / "logo.svg").write_text(text, encoding="utf-8")
    frames = {size: render(text, size) for size in ICO_SIZES}
    frames[256].save(args.out / "icon.png")
    frames[256].save(args.out / "icon.ico", format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                     append_images=[frames[s] for s in ICO_SIZES if s != 256])
    if args.preview:
        from PIL import Image

        strip = Image.new("RGBA", (sum(ICO_SIZES) + 12 * (len(ICO_SIZES) + 1), 280), (245, 246, 250, 255))
        x = 12
        for size in ICO_SIZES:
            strip.alpha_composite(frames[size], (x, 268 - size))
            x += size + 12
        strip.save(args.preview)
    print(f"Wrote logo.svg, icon.png and icon.ico ({len(ICO_SIZES)} sizes) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
