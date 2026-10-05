"""Control estadístico de procesos (SPC): cartas de control de Shewhart, CUSUM/EWMA y capacidad del proceso."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import hist_chart, line_chart
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import InvalidInputError

MODULE = "calidad"
GROUP_CHARTS = "Cartas de control"
GROUP_CAPABILITY = "Capacidad del proceso"

MEASURE = RoleSpec("measure", "Mediciones", "numeric", help="La variable que se mide en cada observación o subgrupo.")
SUBGROUP = RoleSpec("subgroup", "Subgrupo (opcional)", "categorical", optional=True, help="Si se elige, cada valor agrupa las mediciones de un subgrupo.")
PART = RoleSpec("part", "Pieza (o unidad)", "categorical", help="La pieza o unidad medida (fuente de variación entre piezas).")
OPERATOR = RoleSpec("operator", "Operador (o instrumento)", "categorical", help="Quién (o con qué) se midió: fuente de variación entre operadores.")
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
TOLERANCE = FieldSpec("tolerance", "Tolerancia de especificación (opcional)", "number", optional=True, minimum=0.0001, help="USL − LSL; con ella se calcula el %GRR sobre la tolerancia.")

_CHARTS = (
    ("imr", "Individual y rango móvil (I-MR)"),
    ("xbar_r", "Media y rango (X̄-R)"),
    ("xbar_s", "Media y desviación estándar (X̄-S)"),
    ("cusum", "CUSUM tabular"),
    ("ewma", "EWMA"),
)

# n -> (A2, D3, D4, A3, B3, B4, d2) — constantes de Shewhart
_SHEWHART: dict[int, tuple[float, float, float, float, float, float, float]] = {
    2: (1.880, 0.000, 3.267, 2.659, 0.000, 3.267, 1.128),
    3: (1.023, 0.000, 2.574, 1.954, 0.000, 2.568, 1.693),
    4: (0.729, 0.000, 2.282, 1.628, 0.000, 2.266, 2.059),
    5: (0.577, 0.000, 2.114, 1.427, 0.000, 2.089, 2.326),
    6: (0.483, 0.000, 2.004, 1.287, 0.030, 1.970, 2.534),
    7: (0.419, 0.076, 1.924, 1.182, 0.118, 1.882, 2.704),
    8: (0.373, 0.136, 1.864, 1.099, 0.185, 1.815, 2.847),
    9: (0.337, 0.184, 1.816, 1.032, 0.239, 1.761, 2.970),
    10: (0.308, 0.223, 1.777, 0.975, 0.284, 1.716, 3.078),
    11: (0.285, 0.256, 1.744, 0.927, 0.321, 1.679, 3.173),
    12: (0.266, 0.284, 1.716, 0.886, 0.354, 1.646, 3.258),
    13: (0.249, 0.308, 1.692, 0.850, 0.382, 1.618, 3.336),
    14: (0.235, 0.329, 1.671, 0.817, 0.406, 1.594, 3.407),
    15: (0.223, 0.348, 1.652, 0.789, 0.428, 1.572, 3.472),
    16: (0.212, 0.364, 1.636, 0.763, 0.448, 1.552, 3.532),
    17: (0.203, 0.379, 1.621, 0.739, 0.466, 1.534, 3.588),
    18: (0.194, 0.392, 1.608, 0.718, 0.482, 1.518, 3.640),
    19: (0.187, 0.404, 1.596, 0.698, 0.497, 1.503, 3.689),
    20: (0.180, 0.414, 1.586, 0.680, 0.510, 1.490, 3.735),
    21: (0.173, 0.425, 1.575, 0.663, 0.523, 1.477, 3.778),
    22: (0.167, 0.434, 1.566, 0.647, 0.534, 1.466, 3.819),
    23: (0.162, 0.443, 1.557, 0.633, 0.545, 1.455, 3.858),
    24: (0.157, 0.452, 1.548, 0.619, 0.555, 1.445, 3.895),
    25: (0.153, 0.459, 1.541, 0.606, 0.565, 1.435, 3.931),
}


def _subgroups(frame: pd.DataFrame, roles: dict[str, Any], size: int) -> tuple[pd.DataFrame, int]:
    """Build subgroups: by the ``subgroup`` column when present, otherwise consecutive chunks of ``size``."""
    measure = pd.to_numeric(frame[roles["measure"]], errors="coerce")
    if roles.get("subgroup"):
        data = pd.DataFrame({"x": measure, "sub": frame[roles["subgroup"]].astype(str)}).dropna()
    else:
        data = pd.DataFrame({"x": measure}).dropna()
        data["sub"] = np.arange(len(data)) // max(size, 1)
    if len(data) < 4:
        raise InvalidInputError("Se necesitan al menos 4 mediciones.")
    return data, data["sub"].nunique()


def _shewhart(n: int) -> tuple[float, float, float, float, float, float, float]:
    return _SHEWHART.get(int(n), _SHEWHART[25])


def _control_chart(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    chart = p["chart"]
    needs_subgroups = chart in ("xbar_r", "xbar_s")
    data, n_sub = _subgroups(frame, roles, int(p["subgroup_size"]))
    x = data["x"].to_numpy(dtype=float)
    out = Output(dict(_CHARTS)[chart], f"n = {len(x)} mediciones")

    if needs_subgroups:
        if roles.get("subgroup") is None and int(p["subgroup_size"]) <= 1:
            raise InvalidInputError("Para X̄-R / X̄-S elija una columna de subgrupo o un tamaño de subgrupo mayor que 1.")
        grouped = data.groupby("sub")["x"].agg(["mean", "std", "size", lambda s: s.max() - s.min()])
        grouped.columns = ["mean", "std", "size", "range"]
        n_const = int(round(grouped["size"].mean()))
        A2, D3, D4, A3, B3, B4, _d2 = _shewhart(n_const)
        grand = float(grouped["mean"].mean())
        if chart == "xbar_r":
            center2 = float(grouped["range"].mean())
            ucl1, lcl1 = grand + A2 * center2, grand - A2 * center2
            ucl2, lcl2 = D4 * center2, D3 * center2
            label2 = "Rango"
            stat2 = grouped["range"].to_numpy()
        else:
            center2 = float(grouped["std"].mean())
            ucl1, lcl1 = grand + A3 * center2, grand - A3 * center2
            ucl2, lcl2 = B4 * center2, B3 * center2
            label2 = "Desv. est."
            stat2 = grouped["std"].to_numpy()
        stat1 = grouped["mean"].to_numpy()
        idx = np.arange(1, len(grouped) + 1)
        out.tiles(
            [
                ("Subgrupos", len(grouped)),
                ("Tamaño de subgrupo", n_const),
                ("Media global", grand),
                (f"Promedio de {label2.lower()}", center2),
            ]
        )
        out.chart(
            line_chart(
                "Carta X̄", "Subgrupo", "Media", [{"name": "Media", "x": idx, "y": stat1}],
                hlines=[{"value": grand}, {"value": ucl1, "dash": "dash"}, {"value": lcl1, "dash": "dash"}],
            )
        )
        out.chart(
            line_chart(
                f"Carta {label2}", "Subgrupo", label2, [{"name": label2, "x": idx, "y": stat2}],
                hlines=[{"value": center2}, {"value": ucl2, "dash": "dash"}, {"value": lcl2, "dash": "dash"}],
            )
        )
        out.alert(
            "info",
            f"Límites X̄: {lcl1:.4g} a {ucl1:.4g} · Límites {label2}: {lcl2:.4g} a {ucl2:.4g} (constantes para n = {n_const}).",
        )
        out.set(grand_mean=grand, ucl=ucl1, lcl=lcl1, n_subgroups=len(grouped))
        return out.build()

    # individuals charts
    if chart == "imr":
        mr = np.abs(np.diff(x))
        mr_bar = float(mr.mean())
        center = float(x.mean())
        ucl, lcl = center + 2.66 * mr_bar, center - 2.66 * mr_bar
        idx = np.arange(1, len(x) + 1)
        out.tiles(
            [
                ("Media", center),
                ("Rango móvil medio", mr_bar),
                ("Límites X", f"{lcl:.4g} a {ucl:.4g}"),
            ]
        )
        out.chart(
            line_chart(
                "Carta de individuos (X)", "Observación", "Valor", [{"name": "X", "x": idx, "y": x}],
                hlines=[{"value": center}, {"value": ucl, "dash": "dash"}, {"value": lcl, "dash": "dash"}],
            )
        )
        out.chart(
            line_chart(
                "Carta de rangos móviles (MR)", "Observación", "Rango móvil", [{"name": "MR", "x": idx[1:], "y": mr}],
                hlines=[{"value": mr_bar}, {"value": 3.267 * mr_bar, "dash": "dash"}, {"value": 0.0}],
            )
        )
        out.alert("info", f"Límites X = media ± 2.66·MR̄ = {center:.4g} ± {2.66 * mr_bar:.4g}.")
        out.set(mean=center, mr_bar=mr_bar, ucl=ucl, lcl=lcl)
        return out.build()

    if chart == "ewma":
        lam = float(p["lambda_"])
        if not 0 < lam <= 1:
            raise InvalidInputError("λ debe estar en (0, 1].")
        L = float(p["L"])
        ewma = np.zeros(len(x))
        ewma[0] = x[0]
        for i in range(1, len(x)):
            ewma[i] = lam * x[i] + (1 - lam) * ewma[i - 1]
        center = float(x.mean())
        sigma = float(x.std(ddof=1))
        idx = np.arange(1, len(x) + 1)
        factor = np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * np.arange(1, len(x) + 1))))
        out.tiles([("Media", center), ("λ", lam), ("L", L)])
        out.chart(
            line_chart(
                "Carta EWMA", "Observación", "EWMA", [{"name": "EWMA", "x": idx, "y": ewma}],
                bands=[
                    {"name": "Límites", "x": idx, "lower": center - L * sigma * factor, "upper": center + L * sigma * factor}
                ],
                hlines=[{"value": center}],
            )
        )
        out.alert("info", f"EWMA con λ = {lam:g} y límites a ±{L:g}·σ·√(λ/(2−λ)·(1−(1−λ)^(2t))).")
        out.set(mean=center, ewma_last=float(ewma[-1]))
        return out.build()

    # cusum
    k = float(p["k"])
    h = float(p["h"])
    sigma = float(x.std(ddof=1))
    if sigma <= 0:
        raise InvalidInputError("Las mediciones son constantes: no hay variación que vigilar.")
    std = (x - x.mean()) / sigma
    c_pos = np.zeros(len(x))
    c_neg = np.zeros(len(x))
    for i in range(len(x)):
        if i:
            c_pos[i] = max(0.0, c_pos[i - 1] + std[i] - k)
            c_neg[i] = min(0.0, c_neg[i - 1] + std[i] + k)
        else:
            c_pos[i] = max(0.0, std[i] - k)
            c_neg[i] = min(0.0, std[i] + k)
    idx = np.arange(1, len(x) + 1)
    out.tiles([("k (tolerancia)", k), ("h (intervalo de decisión)", h)])
    out.chart(
        line_chart(
            "CUSUM tabular", "Observación", "CUSUM estandarizado",
            [{"name": "C⁺", "x": idx, "y": c_pos}, {"name": "C⁻", "x": idx, "y": c_neg}],
            hlines=[{"value": h, "dash": "dash"}, {"value": -h, "dash": "dash"}, {"value": 0.0}],
        )
    )
    alarmed = bool((c_pos > h).any() or (c_neg < -h).any())
    out.alert(
        "warning" if alarmed else "success",
        "La CUSUM cruza el intervalo de decisión: se detecta un cambio en la media." if alarmed else "La CUSUM no cruza el intervalo de decisión: proceso estable.",
    )
    out.set(k=k, h=h, alarmed=alarmed)
    return out.build()


def _control_chart_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nimport pandas as pd\n\n"
        + load_line(dataset)
        + f"x = df[{py_name(roles['measure'])}].to_numpy(float)\n"
        + "# límites I-MR\nmr = np.abs(np.diff(x))\n"
        + "print('media', x.mean(), 'UCL', x.mean() + 2.66 * mr.mean(), 'LCL', x.mean() - 2.66 * mr.mean())\n"
    )


# ============================================================================ capability
def _capability(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    data, n_sub = _subgroups(frame, roles, int(p.get("subgroup_size", 1)))
    x = data["x"].to_numpy(dtype=float)
    usl = p.get("usl")
    lsl = p.get("lsl")
    if usl is None and lsl is None:
        raise InvalidInputError("Indique al menos un límite de especificación (USL o LSL).")
    if usl is not None and lsl is not None and usl <= lsl:
        raise InvalidInputError("El límite superior (USL) debe ser mayor que el inferior (LSL).")
    mean = float(x.mean())
    overall = float(x.std(ddof=1))
    # within-subgroup sigma: R̄/d2 (subgroups reales) o MR̄/1.128 (individuos)
    individuals = not roles.get("subgroup") and int(p.get("subgroup_size", 1)) <= 1
    if individuals:
        within = float(np.abs(np.diff(x)).mean()) / 1.128
    else:
        grouped = data.groupby("sub")["x"]
        r_bar = float(grouped.apply(lambda s: s.max() - s.min()).mean())
        n_const = int(round(grouped.size().mean()))
        within = r_bar / _shewhart(n_const)[6]
    if within <= 0 or overall <= 0:
        raise InvalidInputError("Las mediciones son constantes: no se puede estimar la dispersión.")

    def _indices(sigma: float) -> tuple[float | None, float | None, tuple[float | None, float | None]]:
        cp = ((usl - lsl) / (6 * sigma)) if (usl is not None and lsl is not None) else None
        cpu = ((usl - mean) / (3 * sigma)) if usl is not None else None
        cpl = ((mean - lsl) / (3 * sigma)) if lsl is not None else None
        if cpu is None:
            cpk = cpl
        elif cpl is None:
            cpk = cpu
        else:
            cpk = min(cpu, cpl)
        return cp, cpk, (cpu, cpl)

    cp, cpk, _cpu_cpl = _indices(within)
    pp, ppk, _ppu_ppl = _indices(overall)
    # expected PPM outside spec (normal)
    ppm = 0.0
    if usl is not None:
        ppm += float(stats.norm.sf((usl - mean) / overall) * 1e6)
    if lsl is not None:
        ppm += float(stats.norm.cdf((lsl - mean) / overall) * 1e6)

    out = Output("Capacidad del proceso", f"n = {len(x)} · dentro de subgrupos σ = {within:.4g} · global σ = {overall:.4g}")
    out.tiles(
        [
            ("Cp", cp),
            ("Cpk", cpk),
            ("Pp", pp),
            ("Ppk", ppk),
            ("PPM esperado", ppm),
        ]
    )
    ok = (cpk if cpk is not None else 0) >= 1.33
    out.alert(
        "success" if ok else "warning",
        "Proceso capaz (Cpk ≥ 1.33)." if ok else "Proceso no capaz (Cpk < 1.33): hay producción fuera de especificación.",
    )
    vlines = [{"value": mean, "label": "media"}]
    if usl is not None:
        vlines.append({"value": usl, "label": "USL", "dash": "dash"})
    if lsl is not None:
        vlines.append({"value": lsl, "label": "LSL", "dash": "dash"})
    out.chart(hist_chart("Distribución de las mediciones", "Valor", x, norm="count", vlines=vlines))
    out.set(cp=cp, cpk=cpk, pp=pp, ppk=ppk, ppm=ppm, mean=mean, within_sigma=within, overall_sigma=overall)
    return out.build()


def _capability_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\n\n"
        + load_line(dataset)
        + f"x = df[{py_name(roles['measure'])}].to_numpy(float)\n"
        + "s = x.std(ddof=1)\n"
        + f"cpk = min(({p.get('usl', 'None')!r} - x.mean()) / (3 * s), (x.mean() - {p.get('lsl', 'None')!r}) / (3 * s)) if {p.get('usl') is not None and p.get('lsl') is not None} else None\n"
        + "print('Cpk', cpk)\n"
    )


# ============================================================================ gage R&R
GROUP_MSA = "Sistema de medición"


def _gage_rr(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    data = pd.DataFrame(
        {
            "x": pd.to_numeric(frame[roles["measure"]], errors="coerce"),
            "part": frame[roles["part"]].astype(str),
            "op": frame[roles["operator"]].astype(str),
        }
    ).dropna()
    if len(data) < 6:
        raise InvalidInputError("Se necesitan al menos 6 mediciones completas.")
    n_parts = data["part"].nunique()
    n_ops = data["op"].nunique()
    if n_parts < 2 or n_ops < 2:
        raise InvalidInputError("Se necesitan al menos 2 piezas y 2 operadores.")
    cells = data.groupby(["part", "op"])["x"]
    sizes = cells.size()
    r = int(round(sizes.mean()))
    balanced = bool((sizes == r).all())

    grand = float(data["x"].mean())
    ss_part = float(sum(len(g) * (g["x"].mean() - grand) ** 2 for _, g in data.groupby("part")))
    ss_op = float(sum(len(g) * (g["x"].mean() - grand) ** 2 for _, g in data.groupby("op")))
    ss_cell = float(sum(len(g) * (g.mean() - grand) ** 2 for _, g in cells))
    ss_interaction = ss_cell - ss_part - ss_op
    ss_error = float(sum(((g - g.mean()) ** 2).sum() for _, g in cells))

    df_part = n_parts - 1
    df_op = n_ops - 1
    df_int = df_part * df_op
    df_error = len(data) - n_parts * n_ops
    if df_error < 1:
        raise InvalidInputError("Se necesitan réplicas: más de una medición por pieza y operador.")

    ms_part = ss_part / df_part
    ms_op = ss_op / df_op
    ms_int = ss_interaction / df_int if df_int > 0 else 0.0
    ms_error = ss_error / df_error
    f_int = ms_int / ms_error if ms_error > 0 else float("inf")
    p_int = float(stats.f.sf(f_int, df_int, df_error)) if ms_error > 0 else 1.0

    if p_int < 0.25:  # interacción significativa (criterio AIAG)
        var_interaction = max((ms_int - ms_error) / r, 0.0)
        var_repeatability = ms_error
        var_operator = max((ms_op - ms_int) / (n_parts * r), 0.0)
    else:
        var_interaction = 0.0
        var_repeatability = ms_error
        var_operator = max((ms_op - ms_error) / (n_parts * r), 0.0)
    var_part = max((ms_part - ms_int) / (n_ops * r), 0.0)

    ev = 5.15 * math.sqrt(var_repeatability)
    av = 5.15 * math.sqrt(var_operator)
    grr = math.sqrt(ev**2 + av**2)
    pv = 5.15 * math.sqrt(var_part)
    tv = math.sqrt(grr**2 + pv**2)
    percent_grr = 100 * grr / tv if tv > 0 else 0.0
    ndc = int(1.41 * pv / grr) if grr > 0 else 99

    out = Output("Estudio R&R (sistema de medición)", f"{n_parts} piezas × {n_ops} operadores × {r} réplica(s)")
    out.tiles(
        [
            ("%GRR (de la variación total)", percent_grr),
            ("ndc (categorías distintas)", ndc),
            ("Repetibilidad (EV)", ev),
            ("Reproducibilidad (AV)", av),
        ]
    )
    tolerance = p.get("tolerance")
    if tolerance:
        out.tiles([("%GRR (de la tolerancia)", 100 * grr / tolerance)])
    verdict = "aceptable" if percent_grr < 10 else "marginal" if percent_grr <= 30 else "inaceptable"
    out.alert(
        "success" if percent_grr < 10 else "warning" if percent_grr <= 30 else "danger",
        f"Sistema de medición {verdict} (%GRR = {percent_grr:.1f} %).",
    )
    out.table(
        "Componentes de varianza",
        ["Fuente", "Varianza (σ²)", "Desviación (5.15σ)", "% de la variación total"],
        [
            ["Repetibilidad (equipo)", var_repeatability, ev, 100 * ev / tv if tv > 0 else None],
            ["Reproducibilidad (operadores)", var_operator, av, 100 * av / tv if tv > 0 else None],
            ["GRR total", var_repeatability + var_operator, grr, percent_grr],
            ["Piezas", var_part, pv, 100 * pv / tv if tv > 0 else None],
            ["Variación total", var_repeatability + var_operator + var_part, tv, 100.0],
        ],
        note="5.15σ cubre el 99 % de la variación. ndc < 5 indica que el sistema no distingue bien las piezas.",
    )
    if not balanced:
        out.alert("info", "Diseño no balanceado (distinto número de réplicas por celda): se usa el número medio de réplicas.")
    return out.set(percent_grr=percent_grr, ndc=ndc, ev=ev, av=av, grr=grr, pv=pv, tv=tv).build()


# ============================================================================ catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "control_chart",
        "Cartas de control (I-MR, X̄-R, X̄-S, CUSUM, EWMA)",
        GROUP_CHARTS,
        "Vigila la estabilidad de un proceso con límites de control de Shewhart (I-MR, X̄-R, X̄-S) o con CUSUM y EWMA.",
        _control_chart,
        roles=(MEASURE, SUBGROUP),
        params=(
            FieldSpec("chart", "Carta", "select", "imr", _CHARTS),
            FieldSpec("subgroup_size", "Tamaño de subgrupo (sin columna de subgrupo)", "integer", 1, minimum=1, maximum=25),
            FieldSpec("lambda_", "λ de EWMA", "number", 0.2, minimum=0.01, maximum=1.0, visible_if=("chart", ("ewma",))),
            FieldSpec("L", "L de EWMA", "number", 3.0, minimum=1.0, maximum=6.0, visible_if=("chart", ("ewma",))),
            FieldSpec("k", "k del CUSUM (tolerancia)", "number", 0.5, minimum=0.01, maximum=3.0, visible_if=("chart", ("cusum",))),
            FieldSpec("h", "h del CUSUM (intervalo de decisión)", "number", 5.0, minimum=1.0, maximum=10.0, visible_if=("chart", ("cusum",))),
        ),
        code=_control_chart_code,
        example={"roles": {"measure": "x1"}, "params": {"chart": "imr"}},
    ),
    _spec(
        "capability",
        "Capacidad del proceso (Cp, Cpk, Pp, Ppk)",
        GROUP_CAPABILITY,
        "Mide qué tan capaz es el proceso respecto a los límites de especificación y estima el PPM fuera de especificación.",
        _capability,
        roles=(MEASURE, SUBGROUP),
        params=(
            FieldSpec("usl", "Límite superior (USL)", "number", optional=True),
            FieldSpec("lsl", "Límite inferior (LSL)", "number", optional=True),
            FieldSpec("subgroup_size", "Tamaño de subgrupo (sin columna de subgrupo)", "integer", 1, minimum=1, maximum=25),
        ),
        code=_capability_code,
        example={"roles": {"measure": "x1"}, "params": {"usl": 16, "lsl": 4}},
    ),
    _spec(
        "gage_rr",
        "Estudio R&R (repetibilidad y reproducibilidad)",
        GROUP_MSA,
        "Evalúa el sistema de medición separando repetibilidad (equipo), reproducibilidad (operadores) y la variación entre piezas (método ANOVA).",
        _gage_rr,
        roles=(MEASURE, PART, OPERATOR),
        params=(TOLERANCE,),
        example={"roles": {"measure": "yc", "part": "cl", "operator": "g"}, "params": {}},
    ),
)

registry.register(*SPECS)
