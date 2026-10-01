"""Generates the Sigma logo (SVG, PNG and ICO) from one geometry definition.

    python scripts/make_logo.py

The mark is a sample space Ω (the ring), an event A inside it (the disc, with a Σ — the σ-algebra) and Pollux, the
four-pointed orange star, as one outcome ω in the complement. A compact variant without the Σ and with heavier strokes
is used at favicon sizes, where the full mark loses detail. Pillow (already a dependency) draws the PNG/ICO files and
matplotlib's font outlines draw the «Sigma» lettering, so nothing new is needed.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from matplotlib import font_manager
from matplotlib.textpath import TextPath
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "sigma" / "presentation" / "web" / "static" / "img"
DOCS = ROOT / "docs"

INDIGO, INDIGO_DARK, INDIGO_PALE = "#4f46e5", "#3730a3", "#eef2ff"
DISC_FROM, DISC_TO = "#6366f1", "#3730a3"
ORANGE, ORANGE_GLOW = "#f59e0b", "#fbbf24"
INK = "#1e1b4b"

# Geometry in a 256 × 256 box. «full» keeps the Σ; «compact» drops it and thickens everything for tiny sizes.
FULL = {
    "ring": (128, 128, 116, 9),  # cx, cy, r, stroke
    "disc": (98, 150, 66),
    "sigma": [(124, 112), (80, 112), (106, 150), (80, 188), (124, 188)],
    "sigma_width": 11,
    "star": (186, 74, 30),
    "glow": 44,
}
COMPACT = {
    "ring": (128, 128, 118, 16),
    "disc": (96, 152, 56),
    "sigma": None,
    "sigma_width": 0,
    "star": (180, 80, 32),
    "glow": 46,
}


# ------------------------------------------------------------------ shapes
def star_points(cx: float, cy: float, r: float, k: float = 0.18, steps: int = 28) -> list[tuple[float, float]]:
    """The four-pointed concave star as a polygon (quadratic curves through the four tips)."""
    k *= r
    tips = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    controls = [(cx + k, cy - k), (cx + k, cy + k), (cx - k, cy + k), (cx - k, cy - k)]
    points: list[tuple[float, float]] = []
    for i in range(4):
        p0, c, p1 = tips[i], controls[i], tips[(i + 1) % 4]
        for s in range(steps):
            t = s / steps
            points.append(
                (
                    (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * c[0] + t**2 * p1[0],
                    (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * c[1] + t**2 * p1[1],
                )
            )
    return points


def star_path(cx: float, cy: float, r: float, k: float = 0.18) -> str:
    k *= r
    f = lambda v: f"{v:.1f}"  # noqa: E731
    return (
        f"M{f(cx)} {f(cy - r)}Q{f(cx + k)} {f(cy - k)} {f(cx + r)} {f(cy)}"
        f"Q{f(cx + k)} {f(cy + k)} {f(cx)} {f(cy + r)}"
        f"Q{f(cx - k)} {f(cy + k)} {f(cx - r)} {f(cy)}"
        f"Q{f(cx - k)} {f(cy - k)} {f(cx)} {f(cy - r)}Z"
    )


# ------------------------------------------------------------------ SVG
def mark_svg_body(g: dict, uid: str = "") -> str:
    cx, cy, r, stroke = g["ring"]
    dx, dy, dr = g["disc"]
    sx, sy, sr = g["star"]
    parts = [
        "<defs>"
        f'<linearGradient id="a{uid}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{DISC_FROM}"/>'
        f'<stop offset="1" stop-color="{DISC_TO}"/></linearGradient>'
        f'<radialGradient id="h{uid}"><stop offset="0" stop-color="{ORANGE_GLOW}" stop-opacity=".6"/>'
        f'<stop offset="1" stop-color="{ORANGE_GLOW}" stop-opacity="0"/></radialGradient></defs>',
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{INDIGO_PALE}" stroke="{INDIGO}" stroke-width="{stroke}"/>',
        f'<circle cx="{dx}" cy="{dy}" r="{dr}" fill="url(#a{uid})"/>',
    ]
    if g["sigma"]:
        points = " ".join(f"{x},{y}" for x, y in g["sigma"])
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="#fff" stroke-width="{g["sigma_width"]}" '
            'stroke-linejoin="round" stroke-linecap="butt"/>'
        )
    parts.append(f'<clipPath id="c{uid}"><circle cx="{cx}" cy="{cy}" r="{r - stroke / 2}"/></clipPath>')
    parts.append(f'<circle cx="{sx}" cy="{sy}" r="{g["glow"]}" fill="url(#h{uid})" clip-path="url(#c{uid})"/>')
    parts.append(f'<path d="{star_path(sx, sy, sr)}" fill="{ORANGE}"/>')
    return "".join(parts)


def mark_svg(g: dict, title: str = "Sigma") -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" width="256" height="256" role="img" '
        f'aria-label="{title}"><title>{title}</title>{mark_svg_body(g)}</svg>\n'
    )


def lettering_path(text: str, height: float) -> tuple[str, float]:
    """SVG path data for ``text`` in DejaVu Sans Bold, scaled so capitals are ``height`` tall; returns (path, width)."""
    path = TextPath((0, 0), text, size=100, prop=font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    scale = height / 72.9  # cap height of DejaVu Sans Bold at size 100
    commands: list[str] = []
    for vertices, code in path.iter_segments():
        pts = [(x * scale, -y * scale) for x, y in zip(vertices[0::2], vertices[1::2], strict=False)]
        if code == 1:
            commands.append("M" + " ".join(f"{x:.1f} {y:.1f}" for x, y in pts))
        elif code == 2:
            commands.append("L" + " ".join(f"{x:.1f} {y:.1f}" for x, y in pts))
        elif code == 3:
            commands.append("Q" + " ".join(f"{x:.1f} {y:.1f}" for x, y in pts))
        elif code == 4:
            commands.append("C" + " ".join(f"{x:.1f} {y:.1f}" for x, y in pts))
        elif code == 79:
            commands.append("Z")
    bounds = path.get_extents()
    return "".join(commands), bounds.x1 * scale


def wordmark_svg() -> str:
    text, width = lettering_path("Sigma", 118)
    total_width = 256 + 36 + width
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_width:.0f} 256" role="img" aria-label="Sigma">'
        f"<title>Sigma</title>{mark_svg_body(FULL, 'w')}"
        f'<g transform="translate(292 188)"><path d="{text}" fill="{INK}"/></g></svg>\n'
    )


# ------------------------------------------------------------------ PNG (Pillow, 4× supersampling)
def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def render_mark(g: dict, size: int, background: str | None = None, padding: float = 0.0) -> Image.Image:
    scale = 4
    big = size * scale
    canvas = Image.new("RGBA", (big, big), _rgb(background) + (255,) if background else (0, 0, 0, 0))
    inner = big * (1 - 2 * padding)
    k = inner / 256
    offset = big * padding

    def at(x: float, y: float) -> tuple[float, float]:
        return offset + x * k, offset + y * k

    draw = ImageDraw.Draw(canvas)
    cx, cy, r, stroke = g["ring"]
    (x0, y0), (x1, y1) = at(cx - r, cy - r), at(cx + r, cy + r)
    draw.ellipse((x0, y0, x1, y1), fill=_rgb(INDIGO_PALE), outline=_rgb(INDIGO), width=round(stroke * k))

    # the disc A with a diagonal gradient
    dx, dy, dr = g["disc"]
    (a0, b0), (a1, b1) = at(dx - dr, dy - dr), at(dx + dr, dy + dr)
    box = (int(a0), int(b0), int(a1), int(b1))
    w, h = box[2] - box[0], box[3] - box[1]
    t = (np.add.outer(np.arange(h), np.arange(w)) / max(w + h - 2, 1)).clip(0, 1)
    c0, c1 = np.array(_rgb(DISC_FROM)), np.array(_rgb(DISC_TO))
    gradient = Image.fromarray((c0 + (c1 - c0) * t[..., None]).astype("uint8"), "RGB").convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, w - 1, h - 1), fill=255)
    canvas.paste(gradient, box[:2], mask)

    if g["sigma"]:
        pts = [at(x, y) for x, y in g["sigma"]]
        width = g["sigma_width"] * k
        for p, q in zip(pts, pts[1:], strict=False):
            v = np.array(q) - np.array(p)
            n = np.array([-v[1], v[0]]) / np.linalg.norm(v) * width / 2
            draw.polygon([tuple(p + n), tuple(q + n), tuple(q - n), tuple(p - n)], fill=(255, 255, 255, 255))
        for x, y in pts[1:-1]:  # round joins
            draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=(255, 255, 255, 255))

    # Pollux with its glow
    sx, sy, sr = g["star"]
    glow = Image.new("RGBA", (big, big), _rgb(ORANGE_GLOW) + (0,))
    gx, gy = at(sx, sy)
    gr = g["glow"] * k
    ImageDraw.Draw(glow).ellipse((gx - gr, gy - gr, gx + gr, gy + gr), fill=_rgb(ORANGE_GLOW) + (150,))
    glow = glow.filter(ImageFilter.GaussianBlur(gr * 0.45))
    inside = Image.new("L", (big, big), 0)  # the glow never leaves the ring
    ring_r = (r - stroke / 2) * k
    rx, ry = at(cx, cy)
    ImageDraw.Draw(inside).ellipse((rx - ring_r, ry - ring_r, rx + ring_r, ry + ring_r), fill=255)
    glow.putalpha(Image.fromarray(np.minimum(np.array(glow.getchannel("A")), np.array(inside)).astype("uint8")))
    canvas.alpha_composite(glow)
    ImageDraw.Draw(canvas).polygon([at(x, y) for x, y in star_points(sx, sy, sr)], fill=_rgb(ORANGE) + (255,))
    return canvas.resize((size, size), Image.LANCZOS)


def render_wordmark(height: int = 256) -> Image.Image:
    mark = render_mark(FULL, height)
    font = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))
    from PIL import ImageFont

    face = ImageFont.truetype(font, int(height * 0.58))
    left, top, right, bottom = face.getbbox("Sigma")
    width = height + int(height * 0.14) + right
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(mark, (0, 0))
    ImageDraw.Draw(canvas).text(
        (height + int(height * 0.14), int(height * 0.73) - bottom), "Sigma", font=face, fill=_rgb(INK) + (255,)
    )
    return canvas


def png_bytes(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()


def main() -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    (IMG / "logo-mark.svg").write_text(mark_svg(FULL), encoding="utf-8")
    (IMG / "logo-mark-small.svg").write_text(mark_svg(COMPACT), encoding="utf-8")
    (IMG / "favicon.svg").write_text(mark_svg(COMPACT), encoding="utf-8")
    (IMG / "logo-wordmark.svg").write_text(wordmark_svg(), encoding="utf-8")
    for size in (512, 256):
        (IMG / f"logo-mark-{size}.png").write_bytes(png_bytes(render_mark(FULL, size)))
    (IMG / "apple-touch-icon.png").write_bytes(png_bytes(render_mark(COMPACT, 180, background="#ffffff", padding=0.1)))
    (IMG / "logo-wordmark.png").write_bytes(png_bytes(render_wordmark(256)))
    render_mark(COMPACT, 256).save(IMG / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    (DOCS / "logo.png").write_bytes(png_bytes(render_wordmark(180)))
    print("Logotipo generado en", IMG)


if __name__ == "__main__":
    main()
