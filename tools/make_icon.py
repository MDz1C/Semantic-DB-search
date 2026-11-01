"""Generate the app icon: assets/icon.png (256 px), assets/icon.ico (16-256 px) and assets/logo.svg.

The mark is a white table with a magnifying glass on a rounded indigo-to-violet tile.  Two
variants share one layout: a detailed one for 40 px and up, and a bolder, simpler one drawn
on a 16-unit pixel grid for 16-32 px icons and the ~22 px logo in the app header, so small
sizes stay crisp.  Rasters are supersampled and box-filtered; the 256 px PNG is rendered at
1024 px and downsampled.  Dev-time only (Pillow + numpy):

    python tools/make_icon.py [--out assets] [--preview DIR]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
GRADIENT = ("#4F46E5", "#7C3AED")  # tile, top-left -> bottom-right
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
SMALL_MAX = 32  # sizes up to this use the bold variant

Box = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class Mark:
    """Icon geometry in design units on a ``grid`` x ``grid`` canvas."""

    grid: int
    tile: Box
    tile_radius: float
    table: Box
    table_radius: float
    stroke: float  # table border and grid-line thickness
    header: float  # height of the solid header band, from the table top
    rows: tuple[float, ...]  # top edge of each horizontal grid line
    cols: tuple[float, ...]  # left edge of each vertical grid line
    lens: Point  # centre of the magnifier
    lens_radius: float  # outer radius of the ring
    ring: float  # ring thickness
    handle: Point  # centre of the handle's round end
    handle_width: float
    gap: float  # knock-out between the magnifier and the table
    glass: float  # opacity of the white tint inside the lens

    def snapped(self, size: int) -> Mark:
        """The mark in pixels of a ``size`` px icon, edges of straight shapes on whole pixels."""
        u = size / self.grid

        def px(v: float) -> float:
            return float(math.floor(v * u + 0.5))

        def scaled(point: Point) -> Point:
            return point[0] * u, point[1] * u

        return replace(
            self,
            grid=size,
            tile=(px(self.tile[0]), px(self.tile[1]), px(self.tile[2]), px(self.tile[3])),
            tile_radius=self.tile_radius * u,
            table=(px(self.table[0]), px(self.table[1]), px(self.table[2]), px(self.table[3])),
            table_radius=self.table_radius * u,
            stroke=max(1.0, px(self.stroke)),
            header=max(1.0, px(self.header)),
            rows=tuple(px(y) for y in self.rows),
            cols=tuple(px(x) for x in self.cols),
            lens=scaled(self.lens),
            lens_radius=self.lens_radius * u,
            ring=max(1.25, self.ring * u),
            handle=scaled(self.handle),
            handle_width=max(1.5, self.handle_width * u),
            gap=max(1.0, px(self.gap)),
        )

    def handle_start(self) -> Point:
        """Where the handle leaves the ring (inside it, so the two shapes merge)."""
        cx, cy = self.lens
        hx, hy = self.handle
        dist = math.hypot(hx - cx, hy - cy)
        inset = self.lens_radius - self.ring / 2
        return cx + (hx - cx) / dist * inset, cy + (hy - cy) / dist * inset

    def cells(self) -> list[tuple[Box, tuple[bool, bool, bool, bool]]]:
        """Holes of the table body as (box, rounded corners tl/tr/br/bl)."""
        x0, y0, x1, y1 = self.table
        s = self.stroke
        tops = [y0 + self.header, *(y + s for y in self.rows)]
        bottoms = [*self.rows, y1 - s]
        lefts = [x0 + s, *(x + s for x in self.cols)]
        rights = [*self.cols, x1 - s]
        out = []
        for i, (top, bottom) in enumerate(zip(tops, bottoms)):
            last_row = i == len(tops) - 1
            for j, (left, right) in enumerate(zip(lefts, rights)):
                if right <= left or bottom <= top:  # squeezed out by pixel snapping
                    continue
                corners = (False, False, last_row and j == len(lefts) - 1, last_row and j == 0)
                out.append(((left, top, right, bottom), corners))
        return out


DETAILED = Mark(
    grid=64,
    tile=(2, 2, 62, 62),
    tile_radius=14,
    table=(10, 11, 44, 43),
    table_radius=5,
    stroke=3,
    header=9,
    rows=(24.65, 32.35),
    cols=(20,),
    lens=(40, 39.5),
    lens_radius=12,
    ring=4.25,
    handle=(51, 50.5),
    handle_width=6,
    gap=3,
    glass=0.16,
)

BOLD = Mark(
    grid=16,
    tile=(0, 0, 16, 16),
    tile_radius=3.5,
    table=(2, 2, 12, 12),
    table_radius=1.5,
    stroke=1,
    header=2,
    rows=(7,),
    cols=(5,),
    lens=(10.75, 10.75),
    lens_radius=3.5,
    ring=1.5,
    handle=(13.6, 13.6),
    handle_width=2,
    gap=1,
    glass=0.18,
)


def _hex(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float64)


def _capsule(draw: ImageDraw.ImageDraw, p0: Point, p1: Point, width: float, k: float) -> None:
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    nx, ny = -(y1 - y0) / length * width / 2, (x1 - x0) / length * width / 2
    corners = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
    draw.polygon([(x * k, y * k) for x, y in corners], fill=255)
    for x, y in (p0, p1):
        _disc(draw, (x, y), width / 2, k, 255)


def _disc(draw: ImageDraw.ImageDraw, center: Point, radius: float, k: float, fill: int) -> None:
    cx, cy = center
    draw.ellipse(_scale((cx - radius, cy - radius, cx + radius, cy + radius), k), fill=fill)


def _scale(box: Box, k: float) -> list[float]:
    # Pillow boxes are inclusive of the far edge: shrink it by one canvas pixel.
    return [box[0] * k, box[1] * k, box[2] * k - 1, box[3] * k - 1]


def render(mark: Mark, size: int) -> Image.Image:
    """RGBA icon of ``size`` px, drawn at >= 1024 px and box-filtered down."""
    ss = max(2, math.ceil(1024 / size))
    n = size * ss
    k = n / mark.grid

    def layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new("L", (n, n), 0)
        return image, ImageDraw.Draw(image)

    tile, draw = layer()
    draw.rounded_rectangle(_scale(mark.tile, k), radius=mark.tile_radius * k, fill=255)

    table, draw = layer()
    draw.rounded_rectangle(_scale(mark.table, k), radius=mark.table_radius * k, fill=255)
    inner_radius = max(mark.table_radius - mark.stroke, 0) * k
    for box, corners in mark.cells():
        draw.rounded_rectangle(_scale(box, k), radius=inner_radius, fill=0, corners=corners)

    magnifier, draw = layer()
    _disc(draw, mark.lens, mark.lens_radius, k, 255)
    _disc(draw, mark.lens, mark.lens_radius - mark.ring, k, 0)
    _capsule(draw, mark.handle_start(), mark.handle, mark.handle_width, k)

    knockout, draw = layer()
    _disc(draw, mark.lens, mark.lens_radius + mark.gap, k, 255)
    _capsule(draw, mark.handle_start(), mark.handle, mark.handle_width + 2 * mark.gap, k)

    glass, draw = layer()
    _disc(draw, mark.lens, mark.lens_radius - mark.ring / 2, k, 255)

    def alpha(image: Image.Image) -> np.ndarray:
        return np.asarray(image, dtype=np.float64) / 255.0

    white = np.maximum.reduce(
        [alpha(table) * (1.0 - alpha(knockout)), alpha(magnifier), alpha(glass) * mark.glass]
    )[..., None]

    x0, y0, x1, y1 = (v * k for v in mark.tile)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64) + 0.5
    t = np.clip(((xx - x0) + (yy - y0)) / ((x1 - x0) + (y1 - y0)), 0.0, 1.0)[..., None]
    start, end = (_hex(c) for c in GRADIENT)
    rgb = (start * (1.0 - t) + end * t) * (1.0 - white) + 255.0 * white

    rgba = np.dstack([rgb, alpha(tile) * 255.0]).round().astype(np.uint8)
    return Image.fromarray(rgba, "RGBA").reduce(ss)


def icon_image(size: int) -> Image.Image:
    """Large sizes are downsampled from a 1024 px master; small ones are pixel-snapped."""
    if size >= 128:
        return render(DETAILED, 1024).resize((size, size), Image.Resampling.LANCZOS)
    return render((BOLD if size <= SMALL_MAX else DETAILED).snapped(size), size)


def _num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _rounded_rect_path(
    box: Box, radius: float, corners: tuple[bool, bool, bool, bool] = (True, True, True, True)
) -> str:
    """Clockwise rounded rectangle as path data; corners are cubic Béziers (SVG Tiny has no arcs)."""
    x0, y0, x1, y1 = box
    tl, tr, br, bl = (radius if rounded else 0.0 for rounded in corners)
    k = 1 - 0.5523  # control-point inset of a quarter-circle Bézier, as a share of the radius
    p = _num
    d = [f"M{p(x0 + tl)} {p(y0)}H{p(x1 - tr)}"]
    if tr:
        d.append(f"C{p(x1 - tr * k)} {p(y0)} {p(x1)} {p(y0 + tr * k)} {p(x1)} {p(y0 + tr)}")
    d.append(f"V{p(y1 - br)}")
    if br:
        d.append(f"C{p(x1)} {p(y1 - br * k)} {p(x1 - br * k)} {p(y1)} {p(x1 - br)} {p(y1)}")
    d.append(f"H{p(x0 + bl)}")
    if bl:
        d.append(f"C{p(x0 + bl * k)} {p(y1)} {p(x0)} {p(y1 - bl * k)} {p(x0)} {p(y1 - bl)}")
    d.append(f"V{p(y0 + tl)}")
    if tl:
        d.append(f"C{p(x0)} {p(y0 + tl * k)} {p(x0 + tl * k)} {p(y0)} {p(x0 + tl)} {p(y0)}")
    return "".join(d) + "Z"


def svg(mark: Mark) -> str:
    """The mark as plain SVG shapes (SVG Tiny 1.2 subset, renders with QtSvg)."""
    p = _num
    g = mark.grid
    x0, y0, x1, y1 = mark.tile
    cx, cy = mark.lens
    (sx, sy), (hx, hy) = mark.handle_start(), mark.handle
    inner_radius = max(mark.table_radius - mark.stroke, 0)
    table = _rounded_rect_path(mark.table, mark.table_radius) + "".join(
        _rounded_rect_path(box, inner_radius, corners) for box, corners in mark.cells()
    )
    handle = f'x1="{p(sx)}" y1="{p(sy)}" x2="{p(hx)}" y2="{p(hy)}" stroke-linecap="round"'
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{g}" height="{g}" viewBox="0 0 {g} {g}">',
        "  <defs>",
        f'    <linearGradient id="tile" gradientUnits="userSpaceOnUse" x1="{p(x0)}" y1="{p(y0)}"'
        f' x2="{p(x1)}" y2="{p(y1)}">',
        f'      <stop offset="0" stop-color="{GRADIENT[0]}"/>',
        f'      <stop offset="1" stop-color="{GRADIENT[1]}"/>',
        "    </linearGradient>",
        "  </defs>",
        f'  <rect x="{p(x0)}" y="{p(y0)}" width="{p(x1 - x0)}" height="{p(y1 - y0)}"'
        f' rx="{p(mark.tile_radius)}" fill="url(#tile)"/>',
        f'  <path fill="#FFFFFF" fill-rule="evenodd" d="{table}"/>',
        f'  <circle cx="{p(cx)}" cy="{p(cy)}" r="{p(mark.lens_radius + mark.gap)}" fill="url(#tile)"/>',
        f'  <line {handle} stroke="url(#tile)" stroke-width="{p(mark.handle_width + 2 * mark.gap)}"/>',
        f'  <circle cx="{p(cx)}" cy="{p(cy)}" r="{p(mark.lens_radius - mark.ring / 2)}" fill="#FFFFFF"'
        f' fill-opacity="{p(mark.glass)}" stroke="#FFFFFF" stroke-width="{p(mark.ring)}"/>',
        f'  <line {handle} stroke="#FFFFFF" stroke-width="{p(mark.handle_width)}"/>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def preview(directory: Path) -> None:
    """Contact sheet: each ICO size at 1x on light and dark, plus 8x pixel zooms of small sizes."""
    directory.mkdir(parents=True, exist_ok=True)
    images = {size: icon_image(size) for size in ICO_SIZES}
    width = sum(ICO_SIZES) + 16 * (len(ICO_SIZES) + 1)
    sheet = Image.new("RGBA", (width, 2 * 256 + 48), "#FFFFFF")
    sheet.paste(Image.new("RGBA", (width, 256 + 24), "#202020"), (0, 256 + 24))
    x = 16
    for size, image in images.items():
        sheet.alpha_composite(image, (x, 12 + 256 - size))
        sheet.alpha_composite(image, (x, 256 + 36 + 256 - size))
        x += size + 16
    sheet.save(directory / "sheet.png")
    zooms = [images[s].resize((s * 8, s * 8), Image.Resampling.NEAREST) for s in (16, 20, 24, 32, 40, 48)]
    zoom = Image.new("RGBA", (sum(z.width for z in zooms) + 16 * 7, 48 * 8 + 32), "#FFFFFF")
    x = 16
    for image in zooms:
        zoom.alpha_composite(image, (x, 16))
        x += image.width + 16
    zoom.save(directory / "zoom.png")
    for size, image in images.items():
        image.save(directory / f"icon-{size}.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "assets", help="output folder (default: assets)")
    parser.add_argument("--preview", type=Path, help="also write inspection sheets to this folder")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    frames = [icon_image(size) for size in ICO_SIZES]
    frames[-1].save(args.out / "icon.png", optimize=True)
    frames[-1].save(args.out / "icon.ico", sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1])
    (args.out / "logo.svg").write_text(svg(BOLD), encoding="utf-8", newline="\n")
    if args.preview:
        preview(args.preview)
    for name in ("icon.png", "icon.ico", "logo.svg"):
        print(f"wrote {args.out / name} ({(args.out / name).stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
