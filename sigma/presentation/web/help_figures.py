"""Small SVG figures that illustrate the theoretical reference (help page).

Each function returns a standalone, inline SVG string (no external assets). Curves are computed
with numpy/scipy for accuracy and emitted as polylines. The style follows the app's muted palette.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

BRAND = "#2f6bb0"
GRAY = "#64748b"
RED = "#dc2626"
GREEN = "#16a34a"
AMBER = "#d97706"
GRID = "#d8dee6"

_FONT = 'font-family="Segoe UI, system-ui, sans-serif"'


def _wrap(w: int, h: int, body: str) -> str:
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" style="max-width:100%;height:auto" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def _line(xs, ys, x0, y0, w, h, xmin, xmax, ymin, ymax) -> str:
    sx = x0 + (np.asarray(xs) - xmin) / (xmax - xmin) * w
    sy = y0 + h - (np.asarray(ys) - ymin) / (ymax - ymin) * h
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(sx, sy))


def _axis(x0, y0, w, h, color=GRAY) -> str:
    return (
        f'<line x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" stroke="{color}" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" stroke="{color}" stroke-width="1"/>'
    )


def _label(x, y, text, color=GRAY, anchor="middle", size=10) -> str:
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" text-anchor="{anchor}" {_FONT}>{text}</text>'


# --------------------------------------------------------------------------- distributions
def normal_pdf() -> str:
    w, h = 440, 210
    x = np.linspace(-4, 4, 200)
    y = stats.norm.pdf(x)
    x0, y0, pw, ph = 45, 18, 380, 150
    pts = _line(x, y, x0, y0, pw, ph, -4, 4, 0, 0.42)
    # shade between -1 and +1 (one sigma)
    xs1 = np.linspace(-1, 1, 80)
    fill = f"{x0},{y0 + ph} " + _line(xs1, stats.norm.pdf(xs1), x0, y0, pw, ph, -4, 4, 0, 0.42)
    body = (
        _axis(x0, y0, pw, ph)
        + f'<polygon points="{fill}" fill="{BRAND}" opacity="0.18"/>'
        + f'<polyline points="{pts}" fill="none" stroke="{BRAND}" stroke-width="2"/>'
        + _label(x0 + pw / 2, y0 - 4, "μ", GRAY)
        + f'<line x1="{x0 + pw/2}" y1="{y0}" x2="{x0 + pw/2}" y2="{y0 + ph}" stroke="{GRAY}" stroke-width="0.8" stroke-dasharray="3,3"/>'
        + _label(x0 + pw / 2, y0 + ph + 12, "μ ± σ (≈ 68 %)", GRAY)
    )
    return _wrap(w, h, body)


def t_vs_normal() -> str:
    w, h = 440, 210
    x = np.linspace(-4, 4, 240)
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    for df, color in ((3, RED), (10, GREEN)):
        y = stats.t.pdf(x, df)
        body += f'<polyline points="{_line(x, y, x0, y0, pw, ph, -4, 4, 0, 0.42)}" fill="none" stroke="{color}" stroke-width="1.8"/>'
    yn = stats.norm.pdf(x)
    body += f'<polyline points="{_line(x, yn, x0, y0, pw, ph, -4, 4, 0, 0.42)}" fill="none" stroke="{BRAND}" stroke-width="2"/>'
    body += _label(x0 + pw - 60, y0 + 14, "Normal", BRAND, "start")
    body += _label(x0 + pw - 60, y0 + 28, "t (gl=3, colas pesadas)", RED, "start")
    body += _label(x0 + pw - 60, y0 + 42, "t (gl=10)", GREEN, "start")
    return _wrap(w, h, body)


def chi2_pdf() -> str:
    w, h = 440, 210
    x = np.linspace(0.001, 12, 240)
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    for df, color in ((1, RED), (3, GREEN), (5, BRAND)):
        y = stats.chi2.pdf(x, df)
        body += f'<polyline points="{_line(x, y, x0, y0, pw, ph, 0, 12, 0, 0.5)}" fill="none" stroke="{color}" stroke-width="1.8"/>'
    body += _label(x0 + pw - 55, y0 + 16, "χ² (gl=1)", RED, "start")
    body += _label(x0 + pw - 55, y0 + 30, "χ² (gl=3)", GREEN, "start")
    body += _label(x0 + pw - 55, y0 + 44, "χ² (gl=5)", BRAND, "start")
    return _wrap(w, h, body)


def f_pdf() -> str:
    w, h = 440, 210
    x = np.linspace(0.001, 5, 240)
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    for dfn, dfd, color in ((2, 10, RED), (5, 20, BRAND), (10, 50, GREEN)):
        y = stats.f.pdf(x, dfn, dfd)
        body += f'<polyline points="{_line(x, y, x0, y0, pw, ph, 0, 5, 0, 1.1)}" fill="none" stroke="{color}" stroke-width="1.8"/>'
    body += _label(x0 + pw - 55, y0 + 16, "F (2,10)", RED, "start")
    body += _label(x0 + pw - 55, y0 + 30, "F (5,20)", BRAND, "start")
    body += _label(x0 + pw - 55, y0 + 44, "F (10,50)", GREEN, "start")
    return _wrap(w, h, body)


# --------------------------------------------------------------------------- discrete distributions
def _pmf(ks, ps):
    """Bar chart of a probability mass function over integer support."""
    w, h = 440, 200
    x0, y0, pw, ph = 45, 18, 380, 140
    ks = [int(k) for k in ks]
    ps = [float(p) for p in ps]
    kmin, kmax = min(ks), max(ks)
    span = kmax - kmin + 1
    ymax = max(ps) * 1.15 or 1.0
    body = _axis(x0, y0, pw, ph)
    bar_w = pw / span * 0.7
    for k, p in zip(ks, ps):
        cx = x0 + (k - kmin + 0.5) / span * pw
        bh = p / ymax * ph
        body += f'<rect x="{cx - bar_w / 2:.1f}" y="{y0 + ph - bh:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{BRAND}" opacity="0.85"/>'
        if span <= 16:
            body += _label(cx, y0 + ph + 12, str(k), GRAY, size=8)
    body += _label(x0 + pw / 2, y0 - 6, "k", GRAY, size=10)
    return _wrap(w, h, body)


def binomial_pmf() -> str:
    ks = np.arange(0, 11)
    return _pmf(ks, stats.binom.pmf(ks, 10, 0.5))


def poisson_pmf() -> str:
    ks = np.arange(0, 13)
    return _pmf(ks, stats.poisson.pmf(ks, 3.0))


def geometric_pmf() -> str:
    ks = np.arange(1, 13)
    return _pmf(ks, stats.geom.pmf(ks, 0.3))


def hypergeometric_pmf() -> str:
    ks = np.arange(0, 11)
    return _pmf(ks, stats.hypergeom.pmf(ks, 50, 10, 10))


def negative_binomial_pmf() -> str:
    ks = np.arange(0, 16)
    return _pmf(ks, stats.nbinom.pmf(ks, 3, 0.5))


# --------------------------------------------------------------------------- models
def regression_scatter() -> str:
    w, h = 440, 210
    rng = np.random.default_rng(7)
    n = 28
    x = np.linspace(0, 10, n)
    y = 1.2 * x + 3.0 + rng.normal(0, 2.5, n)
    b, a = np.polyfit(x, y, 1)
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    sx = x0 + (x - 0) / 10 * pw
    sy = y0 + ph - (y - 0) / (max(y) + 3) * ph
    for xi, yi in zip(sx, sy):
        body += f'<circle cx="{xi:.1f}" cy="{yi:.1f}" r="2.6" fill="{BRAND}" opacity="0.75"/>'
    fit = a + b * x
    body += f'<polyline points="{_line(x, fit, x0, y0, pw, ph, 0, 10, 0, max(y)+3)}" fill="none" stroke="{RED}" stroke-width="2"/>'
    # one residual
    xi = x[18]
    yi = y[18]
    rx = x0 + xi / 10 * pw
    ry = y0 + ph - yi / (max(y) + 3) * ph
    ryf = y0 + ph - (a + b * xi) / (max(y) + 3) * ph
    body += f'<line x1="{rx:.1f}" y1="{ry:.1f}" x2="{rx:.1f}" y2="{ryf:.1f}" stroke="{GREEN}" stroke-width="1.5" stroke-dasharray="3,2"/>'
    body += _label(rx + 4, (ry + ryf) / 2, "εᵢ", GREEN, "start")
    body += _label(x0 + pw / 2, y0 + ph + 12, "ŷ = β̂₀ + β̂₁x  (recta de mínimos cuadrados)", GRAY)
    return _wrap(w, h, body)


def boxplot() -> str:
    w, h = 440, 200
    cx, y0, ph = 220, 18, 150
    # data coordinates for a box: min=5, q1=40, med=65, q3=80, max=95
    def yy(v):
        return y0 + ph - (v - 0) / 100 * ph
    body = f'<line x1="{cx}" y1="{yy(5)}" x2="{cx}" y2="{yy(95)}" stroke="{GRAY}" stroke-width="1.2"/>'
    body += f'<line x1="{cx - 22}" y1="{yy(5)}" x2="{cx + 22}" y2="{yy(5)}" stroke="{GRAY}" stroke-width="1.2"/>'
    body += f'<line x1="{cx - 22}" y1="{yy(95)}" x2="{cx + 22}" y2="{yy(95)}" stroke="{GRAY}" stroke-width="1.2"/>'
    body += f'<rect x="{cx - 32}" y="{yy(80)}" width="64" height="{yy(40) - yy(80)}" fill="{BRAND}" opacity="0.15" stroke="{BRAND}" stroke-width="1.5"/>'
    body += f'<line x1="{cx - 32}" y1="{yy(65)}" x2="{cx + 32}" y2="{yy(65)}" stroke="{BRAND}" stroke-width="2"/>'
    body += f'<circle cx="{cx}" cy="{yy(2)}" r="3" fill="{RED}"/>'
    body += _label(cx, y0 + ph + 22, "atípico (outlier)", RED)
    body += _label(cx + 46, yy(40), "Q1", GRAY, "start")
    body += _label(cx + 46, yy(65), "mediana", GRAY, "start")
    body += _label(cx + 46, yy(80), "Q3", GRAY, "start")
    return _wrap(w, h, body)


def roc_curve() -> str:
    w, h = 440, 210
    x0, y0, pw, ph = 45, 18, 380, 170
    body = _axis(x0, y0, pw, ph)
    body += f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0}" stroke="{GRAY}" stroke-width="1" stroke-dasharray="4,4"/>'
    # a typical ROC (concave)
    fpr = np.linspace(0, 1, 200)
    tpr = 1 - np.exp(-3.5 * fpr)
    body += f'<polyline points="{_line(fpr, tpr, x0, y0, pw, ph, 0, 1, 0, 1)}" fill="none" stroke="{BRAND}" stroke-width="2"/>'
    body += _label(x0 + pw - 30, y0 + ph - 30, "AUC", BRAND, "start")
    body += _label(x0 + 5, y0 + ph - 8, "azar (AUC=0.5)", GRAY, "start")
    return _wrap(w, h, body)


def kaplan_meier() -> str:
    w, h = 440, 210
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    # a step survival curve: drops at t=2,5,9,14,18
    steps = [(0, 1.0), (2, 1.0), (2, 0.88), (5, 0.88), (5, 0.74), (9, 0.74), (9, 0.61), (14, 0.61), (14, 0.50), (18, 0.50), (18, 0.38), (20, 0.38)]
    pts = []
    for t, s in steps:
        sx = x0 + t / 20 * pw
        sy = y0 + ph - s * ph
        pts.append(f"{sx:.1f},{sy:.1f}")
    body += f'<polyline points="{" ".join(pts)}" fill="none" stroke="{BRAND}" stroke-width="2"/>'
    for t, s in [(2, 0.88), (9, 0.61), (18, 0.38)]:
        sx = x0 + t / 20 * pw
        sy = y0 + ph - s * ph
        body += f'<line x1="{sx:.1f}" y1="{y0+ph}" x2="{sx:.1f}" y2="{sy:.1f}" stroke="{GRAY}" stroke-width="0.8" stroke-dasharray="3,3"/>'
    body += _label(x0 + pw / 2, y0 + ph + 12, "tiempo → (las marcas │ son censuras)", GRAY)
    return _wrap(w, h, body)


def power_curve() -> str:
    w, h = 440, 210
    d = np.linspace(0, 1.2, 200)
    # power of a two-sample t test grows with effect size (n=50, alpha=0.05)
    power = stats.nct.sf(stats.t.ppf(0.975, 98), 98, d * np.sqrt(50 / 2))
    x0, y0, pw, ph = 45, 18, 380, 150
    body = _axis(x0, y0, pw, ph)
    body += f'<polyline points="{_line(d, power, x0, y0, pw, ph, 0, 1.2, 0, 1)}" fill="none" stroke="{BRAND}" stroke-width="2"/>'
    body += f'<line x1="{x0}" y1="{y0 + ph - 0.8 * ph}" x2="{x0 + pw}" y2="{y0 + ph - 0.8 * ph}" stroke="{GRAY}" stroke-width="0.8" stroke-dasharray="3,3"/>'
    body += _label(x0 + 8, y0 + ph - 0.8 * ph - 4, "potencia = 0.80", GRAY, "start")
    body += _label(x0 + pw / 2, y0 + ph + 12, "tamaño del efecto (d) →", GRAY)
    return _wrap(w, h, body)


def control_chart() -> str:
    w, h = 440, 210
    rng = np.random.default_rng(3)
    n = 24
    x = rng.normal(10, 1, n)
    x[15] = 14  # out of control
    x0, y0, pw, ph = 45, 18, 380, 150
    lo, hi = 6, 15
    body = _axis(x0, y0, pw, ph)
    # center line and limits
    for v, c, lbl in ((10, BRAND, "LC (μ)"), (13, RED, "LCS (μ+3σ)"), (7, RED, "LCI (μ−3σ)")):
        sy = y0 + ph - (v - lo) / (hi - lo) * ph
        body += f'<line x1="{x0}" y1="{sy:.1f}" x2="{x0+pw}" y2="{sy:.1f}" stroke="{c}" stroke-width="1.2" stroke-dasharray="4,3"/>'
        body += _label(x0 + 4, sy - 3, lbl, c, "start", 9)
    xs = x0 + np.arange(n) / (n - 1) * pw
    ys = y0 + ph - (x - lo) / (hi - lo) * ph
    pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in zip(xs, ys))
    body += f'<polyline points="{pts}" fill="none" stroke="{GRAY}" stroke-width="1.5"/>'
    body += f'<circle cx="{xs[15]:.1f}" cy="{ys[15]:.1f}" r="3.5" fill="{RED}"/>'
    body += _label(xs[15], ys[15] - 10, "fuera de control", RED, "middle", 9)
    return _wrap(w, h, body)


# --------------------------------------------------------------------------- index of figures by key
FIGURES = {
    "normal": normal_pdf,
    "t": t_vs_normal,
    "chi2": chi2_pdf,
    "f": f_pdf,
    "binomial": binomial_pmf,
    "poisson": poisson_pmf,
    "geometric": geometric_pmf,
    "hypergeometric": hypergeometric_pmf,
    "negative_binomial": negative_binomial_pmf,
    "regresion": regression_scatter,
    "boxplot": boxplot,
    "roc": roc_curve,
    "kaplan_meier": kaplan_meier,
    "potencia": power_curve,
    "control": control_chart,
}
