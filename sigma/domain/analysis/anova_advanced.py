"""Advanced ANOVA (Minitab *ANOVA*/*GLM*, SPSS *General Linear Model*): N factors, ANCOVA, repeated measures, MANOVA, nested, post hoc."""

from __future__ import annotations

import re
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import interval_chart, line_chart, scatter_chart
from sigma.domain.analysis.modeling import Design, build_design, dropped_note
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.plots import groups_of
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "anova_avanzado"
ALPHA = FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5)
MAX_FACTORS = 4


def _label(design: Design, term: str) -> str:
    text = re.sub(r"C\((v\d+), Sum\)", r"\1", term)  # a factor is shown by the column's name
    return "Error" if term == "Residual" else design.pretty(text)


def _tokens(design: Design) -> dict[str, str]:
    return {name: token for token, name in design.names.items()}


# ============================================================================ N-way ANOVA and ANCOVA
def _glm_anova(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, factors, covariates = roles["response"], roles["factors"], roles.get("covariates", [])
    if len(factors) > MAX_FACTORS:
        raise InvalidInputError(f"Se admiten hasta {MAX_FACTORS} factores.")
    design = build_design(frame, response, [*factors, *covariates], force_factors=tuple(factors))
    tok = _tokens(design)
    ftok, ctok = [tok[f] for f in factors], [tok[c] for c in covariates]
    base = [f"C({t}, Sum)" for t in ftok]
    terms = list(base)
    if p["interactions"] != "none" and len(base) > 1:
        depth = 2 if p["interactions"] == "two" else len(base)
        for size in range(2, depth + 1):
            terms += [":".join(c) for c in combinations(base, size)]
    terms += ctok
    typ = {"I": 1, "II": 2, "III": 3}[p["ss_type"]]
    formula = f"{design.response} ~ " + " + ".join(terms)
    fit = smf.ols(formula, design.data).fit()
    n = len(design.data)
    if fit.df_resid < 1:
        raise InvalidInputError(
            "El modelo tiene tantos parámetros como observaciones: reduzca los factores o las interacciones."
        )
    table = anova_lm(fit, typ=typ)
    error = table.loc["Residual"]
    mse = float(error["sum_sq"] / error["df"])
    out = Output(
        f"ANOVA {'de ' + str(len(factors)) + ' factores' if len(factors) > 1 else 'de un factor'}{' con covariables (ANCOVA)' if covariates else ''}: {response}",
        f"suma de cuadrados tipo {p['ss_type']} · n = {n}",
    )
    dropped_note(out, design)
    rows = []
    for term, row in table.iterrows():
        if term in ("Intercept",):
            continue
        if term == "Residual":
            rows.append(["Error", int(row["df"]), float(row["sum_sq"]), mse, None, None, None, None])
            continue
        f, pv, df, ss = float(row["F"]), float(row["PR(>F)"]), int(row["df"]), float(row["sum_sq"])
        omega = df * (f - 1) / (df * (f - 1) + n)
        rows.append(
            [_label(design, str(term)), df, ss, ss / df, f, P(pv), ss / (ss + float(error["sum_sq"])), max(omega, 0.0)]
        )
    total = float(((design.data[design.response] - design.data[design.response].mean()) ** 2).sum())
    rows.append(["Total", n - 1, total, None, None, None, None, None])
    out.table(
        "Análisis de varianza",
        ["Fuente", "gl", "Suma de cuadrados", "Cuadrado medio", "F", "Valor p", "η² parcial", "ω² parcial"],
        rows,
        note="η² parcial = SC del efecto / (SC del efecto + SC del error). ω² parcial es la versión menos sesgada."
        + (" Con tipo III los factores usan codificación suma (desvíos de la media global)." if typ == 3 else ""),
    )
    significant = [r[0] for r in rows if isinstance(r[5], P) and float(r[5]) < p["alpha"]]
    out.tiles(
        [
            ("Observaciones", n),
            ("R²", float(fit.rsquared)),
            ("R² ajustado", float(fit.rsquared_adj)),
            ("Efectos significativos", len(significant)),
        ]
    )
    out.alert(
        "info" if not significant else "success",
        ("Efectos significativos al " + f"{100 * (1 - p['alpha']):g} %: " + ", ".join(significant) + ".")
        if significant
        else "Ningún efecto es significativo con ese nivel.",
    )
    cells = design.data.groupby(ftok, observed=True)[design.response]
    groups = [g.to_numpy() for _, g in cells if len(g) > 1]
    checks = []
    if len(groups) > 1:
        lev = stats.levene(*groups, center="median")
        checks.append(
            ["Homogeneidad de varianzas entre celdas (Brown-Forsythe)", float(lev.statistic), P(float(lev.pvalue))]
        )
    resid = np.asarray(fit.resid)
    sample = resid if n <= 5000 else np.random.default_rng(1).choice(resid, 5000, replace=False)
    shapiro = stats.shapiro(sample)
    checks.append(["Normalidad de los residuos (Shapiro-Wilk)", float(shapiro.statistic), P(float(shapiro.pvalue))])
    for c, token in zip(covariates, ctok, strict=True):
        reduced = fit
        full = smf.ols(
            f"{design.response} ~ " + " + ".join([*terms, *[f"{b}:{token}" for b in base]]), design.data
        ).fit()
        table_slopes = anova_lm(reduced, full)
        checks.append(
            [
                f"Homogeneidad de pendientes: {c} × factores",
                float(table_slopes.iloc[1]["F"]),
                P(float(table_slopes.iloc[1]["Pr(>F)"])),
            ]
        )
    out.table(
        "Comprobación de supuestos",
        ["Prueba", "Estadístico", "Valor p"],
        checks,
        note="Valores p pequeños indican incumplimiento. En ANCOVA las pendientes de la covariable deben ser iguales entre grupos.",
    )
    cells_table = design.data.groupby(ftok, observed=True)[design.response].agg(["count", "mean", "std"]).reset_index()
    out.table(
        "Medias por celda",
        [*factors, "n", "Media", "Desv. est."],
        [
            [
                *(str(row[t]) for t in ftok),
                int(row["count"]),
                float(row["mean"]),
                float(row["std"]) if row["std"] == row["std"] else None,
            ]
            for _, row in cells_table.iterrows()
        ],
    )
    # estimated marginal means: every factor combination at the covariates' mean, averaged with equal weights
    levels = {t: sorted(design.data[t].astype(str).unique()) for t in ftok}
    grid = pd.DataFrame(list(product(*levels.values())), columns=ftok)
    for t in ctok:
        grid[t] = float(design.data[t].mean())
    train = patsy.dmatrix(formula.split("~", 1)[1], design.data, return_type="dataframe")
    if list(train.columns) != list(fit.params.index):
        raise ComputationError("No se pudieron reconstruir las columnas del diseño para las medias marginales.")
    matrix = np.asarray(patsy.build_design_matrices([train.design_info], grid, return_type="dataframe")[0])
    crit = stats.t.ppf(1 - p["alpha"] / 2, fit.df_resid)
    cov = np.asarray(fit.cov_params())
    for t, name in zip(ftok, factors, strict=True):
        marginal = []
        for level in levels[t]:
            weights = (grid[t] == level).to_numpy(float)
            weights /= weights.sum()
            row = weights @ matrix
            est, se = float(row @ np.asarray(fit.params)), float(np.sqrt(row @ cov @ row))
            marginal.append((level, est, se, est - crit * se, est + crit * se))
        out.table(
            f"Medias marginales estimadas de «{name}»" + (" (covariables en su media)" if covariates else ""),
            ["Nivel", "Media", "Error est.", "IC inf.", "IC sup."],
            [list(m) for m in marginal],
        )
        out.chart(
            interval_chart(
                f"Efecto principal de {name}",
                name,
                response,
                [{"name": m[0], "mean": m[1], "lower": m[3], "upper": m[4]} for m in marginal],
            )
        )
    if len(ftok) >= 2:
        a, b = ftok[0], ftok[1]
        predicted = (
            grid.assign(_y=matrix @ np.asarray(fit.params)).groupby([a, b], observed=True)["_y"].mean().reset_index()
        )
        out.chart(
            line_chart(
                f"Interacción {factors[0]} × {factors[1]}",
                factors[0],
                response,
                [
                    {
                        "name": f"{factors[1]} = {level}",
                        "x": part[a].astype(str).to_numpy(),
                        "y": part["_y"].to_numpy(),
                        "markers": True,
                    }
                    for level, part in predicted.groupby(b, observed=True)
                ],
            )
        )
    return out.set(
        n=n, r2=float(fit.rsquared), effects={r[0]: float(r[5]) for r in rows if isinstance(r[5], P)}, mse=mse
    ).build()


def _glm_anova_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    terms = (
        " * ".join(f"C(Q('{f}'), Sum)" for f in roles["factors"])
        if p["interactions"] != "none"
        else " + ".join(f"C(Q('{f}'), Sum)" for f in roles["factors"])
    )
    covs = "".join(f" + Q('{c}')" for c in roles.get("covariates", []))
    typ = {"I": 1, "II": 2, "III": 3}[p["ss_type"]]
    return (
        "import statsmodels.formula.api as smf\nfrom statsmodels.stats.anova import anova_lm\n\n"
        + load_line(dataset)
        + f"fit = smf.ols(\"Q('{roles['response']}') ~ {terms}{covs}\", df).fit()\nprint(anova_lm(fit, typ={typ}))\n"
    )


# ============================================================================ repeated measures
def helmert(k: int) -> np.ndarray:
    """(k − 1) × k orthonormal contrasts orthogonal to the constant vector."""
    rows = []
    for i in range(1, k):
        row = np.zeros(k)
        row[:i] = 1.0
        row[i] = -float(i)
        rows.append(row / np.linalg.norm(row))
    return np.array(rows)


def sphericity(data: np.ndarray) -> dict[str, float]:
    """Mauchly's test and the Greenhouse-Geisser / Huynh-Feldt epsilons for an ``n × k`` array of repeated measures."""
    n, k = data.shape
    contrasts = data @ helmert(k).T
    s = np.cov(contrasts, rowvar=False).reshape(k - 1, k - 1)
    m = k - 1
    trace, trace2 = float(np.trace(s)), float(np.trace(s @ s))
    eps_gg = trace**2 / (m * trace2)
    eps_hf = min(1.0, (n * m * eps_gg - 2) / (m * (n - 1 - m * eps_gg)))
    if k == 2:
        return {"W": 1.0, "chi2": 0.0, "df": 0, "p": 1.0, "gg": 1.0, "hf": 1.0}
    determinant = float(np.linalg.det(s))
    w = max(determinant, 1e-300) / (trace / m) ** m
    chi2 = -(n - 1 - (2 * m**2 + m + 2) / (6 * m)) * np.log(w)
    df = k * (k - 1) // 2 - 1
    return {
        "W": float(w),
        "chi2": float(chi2),
        "df": df,
        "p": float(stats.chi2.sf(chi2, df)),
        "gg": float(eps_gg),
        "hf": float(max(eps_hf, 1 / m)),
    }


def _repeated(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    names, alpha = roles["vars"], p["alpha"]
    data = frame[names].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = data.shape
    if n < 3:
        raise InvalidInputError("Se necesitan al menos 3 sujetos con todas las mediciones.")
    x = data.to_numpy(float)
    grand = x.mean()
    ss_subject = k * float(((x.mean(axis=1) - grand) ** 2).sum())
    ss_cond = n * float(((x.mean(axis=0) - grand) ** 2).sum())
    ss_total = float(((x - grand) ** 2).sum())
    ss_error = ss_total - ss_subject - ss_cond
    df_c, df_e = k - 1, (n - 1) * (k - 1)
    f = (ss_cond / df_c) / (ss_error / df_e)
    sph = sphericity(x)
    eta_p, eta_g = ss_cond / (ss_cond + ss_error), ss_cond / (ss_cond + ss_subject + ss_error)
    out = Output("ANOVA de medidas repetidas", f"{k} mediciones en {n} sujetos")
    if len(data) < len(frame):
        out.alert("info", f"Se omitieron {len(frame) - len(data)} sujeto(s) con mediciones faltantes.")
    out.tiles([("Sujetos", n), ("Mediciones", k), ("F", float(f)), ("η² parcial", eta_p)])
    rows = [
        ["Sin corrección", df_c, df_e, f, P(float(stats.f.sf(f, df_c, df_e)))],
        [
            "Greenhouse-Geisser",
            df_c * sph["gg"],
            df_e * sph["gg"],
            f,
            P(float(stats.f.sf(f, df_c * sph["gg"], df_e * sph["gg"]))),
        ],
        [
            "Huynh-Feldt",
            df_c * sph["hf"],
            df_e * sph["hf"],
            f,
            P(float(stats.f.sf(f, df_c * sph["hf"], df_e * sph["hf"]))),
        ],
    ]
    out.table(
        "Efecto de la medición",
        ["Corrección", "gl (efecto)", "gl (error)", "F", "Valor p"],
        rows,
        note="Si no se cumple la esfericidad, use Greenhouse-Geisser (o Huynh-Feldt si ε > 0.75).",
    )
    out.table(
        "Tabla de análisis de varianza",
        ["Fuente", "gl", "Suma de cuadrados", "Cuadrado medio", "F", "Valor p", "η² parcial", "η² generalizado"],
        [
            ["Mediciones", df_c, ss_cond, ss_cond / df_c, f, P(float(stats.f.sf(f, df_c, df_e))), eta_p, eta_g],
            ["Sujetos", n - 1, ss_subject, ss_subject / (n - 1), None, None, None, None],
            ["Error", df_e, ss_error, ss_error / df_e, None, None, None, None],
        ],
    )
    if k >= 3:
        violated = sph["p"] < alpha
        out.table(
            "Esfericidad",
            ["Prueba", "W de Mauchly", "χ²", "gl", "Valor p", "ε Greenhouse-Geisser", "ε Huynh-Feldt"],
            [["Mauchly", sph["W"], sph["chi2"], sph["df"], P(sph["p"]), sph["gg"], sph["hf"]]],
        )
        out.alert(
            "warning" if violated else "success",
            "Se viola la esfericidad (Mauchly): use la corrección de Greenhouse-Geisser."
            if violated
            else "No hay evidencia de que se viole la esfericidad.",
        )
    centred = x - x.mean(axis=1, keepdims=True) + grand  # Cousineau-Morey intervals remove between-subject differences
    crit = stats.t.ppf(1 - alpha / 2, n - 1)
    spread = centred.std(axis=0, ddof=1) / np.sqrt(n) * np.sqrt(k / (k - 1))
    means = x.mean(axis=0)
    out.chart(
        interval_chart(
            f"Medias por medición (IC {100 * (1 - alpha):g} % intrasujeto)",
            "Medición",
            "Media",
            [
                {"name": c, "mean": float(m), "lower": float(m - crit * s), "upper": float(m + crit * s)}
                for c, m, s in zip(names, means, spread, strict=True)
            ],
        )
    )
    pairs = list(combinations(range(k), 2))
    raw = [float(stats.ttest_rel(x[:, i], x[:, j]).pvalue) for i, j in pairs]
    method = p["adjust"]
    adjusted = multipletests(raw, method=method)[1] if len(raw) > 1 else raw
    out.table(
        f"Comparaciones por pares (t pareada, ajuste {dict(_ADJUST)[method]})",
        ["Medición A", "Medición B", "Diferencia (A − B)", "t", "Valor p", "Valor p ajustado"],
        [
            [
                names[i],
                names[j],
                float((x[:, i] - x[:, j]).mean()),
                float(stats.ttest_rel(x[:, i], x[:, j]).statistic),
                P(r),
                P(float(a)),
            ]
            for (i, j), r, a in zip(pairs, raw, adjusted, strict=True)
        ],
    )
    return out.set(
        f=float(f),
        p=float(stats.f.sf(f, df_c, df_e)),
        p_gg=rows[1][4] and float(rows[1][4]),
        mauchly_p=sph["p"],
        epsilon_gg=sph["gg"],
        epsilon_hf=sph["hf"],
        eta_p=eta_p,
    ).build()


_ADJUST = (("holm", "Holm"), ("bonferroni", "Bonferroni"), ("sidak", "Šidák"), ("fdr_bh", "Benjamini-Hochberg (FDR)"))


def _repeated_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.stats.anova import AnovaRM\n\n"
        + load_line(dataset)
        + f"long = df[{roles['vars']!r}].dropna().rename_axis('sujeto').reset_index().melt(id_vars='sujeto', var_name='medicion', value_name='y')\nprint(AnovaRM(long, 'y', 'sujeto', within=['medicion']).fit())\n"
    )


# ============================================================================ MANOVA
def box_m(groups: list[np.ndarray]) -> tuple[float, float, int]:
    """Box's M test of equal covariance matrices: (chi-square, p-value, df)."""
    g, p = len(groups), groups[0].shape[1]
    sizes = np.array([len(x) for x in groups])
    covs = [np.cov(x, rowvar=False) for x in groups]
    pooled = sum((n - 1) * c for n, c in zip(sizes, covs, strict=True)) / (sizes.sum() - g)
    m = (sizes.sum() - g) * np.log(np.linalg.det(pooled)) - sum(
        (n - 1) * np.log(np.linalg.det(c)) for n, c in zip(sizes, covs, strict=True)
    )
    c = (np.sum(1 / (sizes - 1)) - 1 / (sizes.sum() - g)) * (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    df = p * (p + 1) * (g - 1) // 2
    chi2 = float(m * (1 - c))
    return chi2, float(stats.chi2.sf(chi2, df)), df


def _manova(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    responses, factors, alpha = roles["responses"], roles["factors"], p["alpha"]
    design = build_design(frame, None, [*responses, *factors], force_factors=tuple(factors))
    tok = _tokens(design)
    rtok, ftok = [tok[r] for r in responses], [tok[f] for f in factors]
    data = design.data
    n, q = len(data), len(responses)
    if n < q + len(ftok) + 5:
        raise InvalidInputError("Hay muy pocas filas completas para el número de variables.")
    rhs = " * ".join(f"C({t}, Sum)" for t in ftok)
    try:
        result = MANOVA.from_formula(f"{' + '.join(rtok)} ~ {rhs}", data).mv_test()
    except (np.linalg.LinAlgError, ValueError) as error:
        raise ComputationError(
            f"No se pudo ajustar el MANOVA ({error}). Revise que las respuestas no sean combinaciones exactas unas de otras."
        ) from None
    out = Output("MANOVA", f"{q} variables de respuesta · factores: {', '.join(factors)} · n = {n}")
    dropped_note(out, design)
    out.tiles([("Observaciones", n), ("Respuestas", q), ("Factores", len(factors))])
    rows = []
    wilks = {}
    for term, block in result.results.items():
        if term == "Intercept":
            continue
        label = _label(design, term)
        stat = block["stat"]
        for test, values in stat.iterrows():
            rows.append(
                [
                    label,
                    str(test),
                    float(values["Value"]),
                    float(values["F Value"]),
                    float(values["Num DF"]),
                    float(values["Den DF"]),
                    P(float(values["Pr > F"])),
                ]
            )
            if test.startswith("Wilks"):
                wilks[label] = float(values["Pr > F"])
    out.table(
        "Pruebas multivariadas",
        ["Efecto", "Estadístico", "Valor", "F", "gl (num)", "gl (den)", "Valor p"],
        rows,
        note="Pillai es la más robusta ante incumplimientos; Wilks es la más usada; Roy es la más potente pero la más sensible.",
    )
    univariate = []
    for response, name in zip(rtok, responses, strict=True):
        fit = smf.ols(f"{response} ~ {rhs}", data).fit()
        table = anova_lm(fit, typ=3)
        for term, row in table.iterrows():
            if term in ("Intercept", "Residual"):
                continue
            univariate.append(
                [
                    name,
                    _label(design, str(term)),
                    int(row["df"]),
                    float(row["F"]),
                    P(float(row["PR(>F)"])),
                    "sí" if float(row["PR(>F)"]) < alpha / q else "no",
                ]
            )
    out.table(
        "Pruebas univariadas de seguimiento",
        ["Respuesta", "Efecto", "gl", "F", "Valor p", f"Significativo con α/{q}"],
        univariate,
        note=f"Se usa la corrección de Bonferroni: α/{q} = {alpha / q:.4f}.",
    )
    cells = data.groupby(ftok, observed=True)
    arrays = [part[rtok].to_numpy(float) for _, part in cells if len(part) > q + 1]
    if len(arrays) >= 2 and len(arrays) == cells.ngroups:
        chi2, pv, df = box_m(arrays)
        out.table(
            "Igualdad de matrices de covarianza (M de Box)",
            ["χ²", "gl", "Valor p"],
            [[chi2, df, P(pv)]],
            note="M de Box es muy sensible a la no normalidad; con grupos de igual tamaño el MANOVA tolera diferencias moderadas.",
        )
    means = cells[rtok].mean().reset_index()
    out.table(
        "Medias por celda",
        [*factors, *responses],
        [[*(str(r[t]) for t in ftok), *(float(r[t]) for t in rtok)] for _, r in means.iterrows()],
    )
    return out.set(n=n, wilks_p=wilks).build()


def _manova_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    rhs = " * ".join(f"C(Q('{f}'))" for f in roles["factors"])
    return (
        "from statsmodels.multivariate.manova import MANOVA\n\n"
        + load_line(dataset)
        + f'print(MANOVA.from_formula("{" + ".join(roles["responses"])} ~ {rhs}", df).mv_test())\n'
    )


# ============================================================================ nested ANOVA
def _nested(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, a_name, b_name = roles["response"], roles["factor_a"], roles["factor_b"]
    if a_name == b_name:
        raise InvalidInputError("El factor anidado debe ser distinto del factor principal.")
    design = build_design(frame, response, [a_name, b_name], force_factors=(a_name, b_name))
    tok = _tokens(design)
    a, b = tok[a_name], tok[b_name]
    data = design.data.assign(_cell=design.data[a] + "/" + design.data[b])
    counts = data.groupby(a, observed=True)[b].nunique()
    sizes = data.groupby("_cell", observed=True).size()
    if len(counts) < 2 or counts.min() < 2:
        raise InvalidInputError("Cada nivel del factor principal necesita al menos 2 niveles del factor anidado.")
    if sizes.min() < 2:
        raise InvalidInputError("Cada combinación necesita al menos 2 observaciones para estimar el error.")
    fit = smf.ols(f"{design.response} ~ C({a}) + C({a}):C({b})", data).fit()
    table = anova_lm(fit, typ=1)
    row_a, row_b, row_e = table.iloc[0], table.iloc[1], table.loc["Residual"]
    ms_a, ms_b, mse = (
        float(row_a["sum_sq"] / row_a["df"]),
        float(row_b["sum_sq"] / row_b["df"]),
        float(row_e["sum_sq"] / row_e["df"]),
    )
    random_b = bool(p["b_random"])
    f_a = ms_a / (ms_b if random_b else mse)
    df_a_den = float(row_b["df"]) if random_b else float(row_e["df"])
    f_b = ms_b / mse
    p_a, p_b = float(stats.f.sf(f_a, row_a["df"], df_a_den)), float(stats.f.sf(f_b, row_b["df"], row_e["df"]))
    n = len(data)
    out = Output(
        f"ANOVA anidado: {response}",
        f"«{b_name}» anidado en «{a_name}» ({'aleatorio' if random_b else 'fijo'}) · n = {n}",
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            (f"Niveles de {a_name}", len(counts)),
            (f"Combinaciones {a_name}/{b_name}", len(sizes)),
            ("Error estándar residual", float(np.sqrt(mse))),
        ]
    )
    out.table(
        "Análisis de varianza",
        ["Fuente", "gl", "Suma de cuadrados", "Cuadrado medio", "Denominador de F", "F", "Valor p"],
        [
            [
                a_name,
                int(row_a["df"]),
                float(row_a["sum_sq"]),
                ms_a,
                f"{b_name}({a_name})" if random_b else "Error",
                f_a,
                P(p_a),
            ],
            [f"{b_name}({a_name})", int(row_b["df"]), float(row_b["sum_sq"]), ms_b, "Error", f_b, P(p_b)],
            ["Error", int(row_e["df"]), float(row_e["sum_sq"]), mse, None, None, None],
        ],
        note="Con el factor anidado aleatorio, el factor principal se prueba contra el cuadrado medio del factor anidado.",
    )
    balanced = sizes.nunique() == 1 and counts.nunique() == 1
    if random_b:
        m, nb = float(sizes.mean()), float(counts.mean())
        var_b = max((ms_b - mse) / m, 0.0)
        var_a = max((ms_a - ms_b) / (nb * m), 0.0)
        total = var_a + var_b + mse
        out.table(
            "Componentes de la varianza",
            ["Fuente", "Varianza estimada", "% del total"],
            [
                [a_name, var_a, 100 * var_a / total],
                [f"{b_name}({a_name})", var_b, 100 * var_b / total],
                ["Error", mse, 100 * mse / total],
            ],
            note="Estimación por cuadrados medios esperados"
            + ("" if balanced else " (aproximada: el diseño no está balanceado)")
            + ".",
        )
    if not balanced:
        out.alert(
            "info",
            "El diseño no está balanceado: la suma de cuadrados secuencial depende del orden y los componentes son aproximados.",
        )
    means = data.groupby(a, observed=True)[design.response].agg(["count", "mean", "std"]).reset_index()
    out.table(
        f"Medias por «{a_name}»",
        [a_name, "n", "Media", "Desv. est."],
        [[str(r[a]), int(r["count"]), float(r["mean"]), float(r["std"])] for _, r in means.iterrows()],
    )
    cells = data.groupby("_cell", observed=True)[design.response].mean().reset_index()
    out.chart(
        scatter_chart(
            f"Medias de cada «{b_name}» dentro de «{a_name}»",
            f"{a_name}/{b_name}",
            response,
            [
                {
                    "name": "Media de la combinación",
                    "x": cells["_cell"].to_numpy(),
                    "y": cells[design.response].to_numpy(),
                }
            ],
            height=360,
        )
    )
    return out.set(n=n, p_a=p_a, p_b=p_b, mse=mse).build()


def _nested_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    a, b = roles["factor_a"], roles["factor_b"]
    return (
        "import statsmodels.formula.api as smf\nfrom statsmodels.stats.anova import anova_lm\n\n"
        + load_line(dataset)
        + f"fit = smf.ols(\"Q('{roles['response']}') ~ C(Q('{a}')) + C(Q('{a}')):C(Q('{b}'))\", df).fit()\nprint(anova_lm(fit, typ=1))  # F de A contra B(A) si B es aleatorio\n"
    )


# ============================================================================ post hoc comparisons
_POSTHOC = (
    ("tukey", "Tukey HSD (todas las parejas, varianzas iguales)"),
    ("games_howell", "Games-Howell (varianzas desiguales)"),
    ("dunnett", "Dunnett (cada grupo contra un control)"),
    ("bonferroni", "Bonferroni"),
    ("holm", "Holm"),
    ("sidak", "Šidák"),
    ("scheffe", "Scheffé"),
    ("lsd", "Diferencia mínima significativa de Fisher (sin ajuste)"),
)


def _posthoc(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, factor, method, alpha = roles["response"], roles["factor"], p["method"], p["alpha"]
    parts = [(label, values.to_numpy(float)) for label, values in groups_of(frame, response, factor)]
    parts = [(label, v) for label, v in parts if len(v) >= 2]
    k = len(parts)
    if k < 2:
        raise InvalidInputError("Se necesitan al menos 2 grupos con 2 datos cada uno.")
    labels = [label for label, _ in parts]
    arrays = [v for _, v in parts]
    n = sum(len(v) for v in arrays)
    means = [float(v.mean()) for v in arrays]
    sizes = [len(v) for v in arrays]
    df_error = n - k
    mse = float(sum(((v - v.mean()) ** 2).sum() for v in arrays) / df_error)
    rows: list[list[Any]] = []
    if method == "dunnett":
        control = (p.get("control") or "").strip() or labels[0]
        if control not in labels:
            raise InvalidInputError(f"El grupo de control «{control}» no existe en «{factor}».")
        ci = control and labels.index(control)
        others = [i for i in range(k) if i != ci]
        result = stats.dunnett(*[arrays[i] for i in others], control=arrays[ci], alternative="two-sided")
        interval = result.confidence_interval(confidence_level=1 - alpha)
        for position, i in enumerate(others):
            diff = means[i] - means[ci]
            rows.append(
                [
                    labels[i],
                    labels[ci],
                    diff,
                    float(np.sqrt(mse * (1 / sizes[i] + 1 / sizes[ci]))),
                    float(result.statistic[position]),
                    float(result.pvalue[position]),
                    float(interval.low[position]),
                    float(interval.high[position]),
                ]
            )
    else:
        pairs = list(combinations(range(k), 2))
        raw, partial = [], []
        for i, j in pairs:
            diff = means[i] - means[j]
            if method == "games_howell":
                v1, v2 = arrays[i].var(ddof=1) / sizes[i], arrays[j].var(ddof=1) / sizes[j]
                se, df = (
                    float(np.sqrt(v1 + v2)),
                    float((v1 + v2) ** 2 / (v1**2 / (sizes[i] - 1) + v2**2 / (sizes[j] - 1))),
                )
                q = abs(diff) / se * np.sqrt(2)
                pv = float(stats.studentized_range.sf(q, k, df))
                half = float(stats.studentized_range.ppf(1 - alpha, k, df) / np.sqrt(2) * se)
                partial.append((diff, se, abs(diff) / se, pv, diff - half, diff + half))
            elif method == "tukey":
                se = float(np.sqrt(mse / 2 * (1 / sizes[i] + 1 / sizes[j])))
                q = abs(diff) / se
                pv = float(stats.studentized_range.sf(q, k, df_error))
                half = float(stats.studentized_range.ppf(1 - alpha, k, df_error) * se)
                partial.append((diff, se, q, pv, diff - half, diff + half))
            else:
                se = float(np.sqrt(mse * (1 / sizes[i] + 1 / sizes[j])))
                t = diff / se
                raw_p = float(2 * stats.t.sf(abs(t), df_error))
                if method == "scheffe":
                    f = t**2 / (k - 1)
                    pv = float(stats.f.sf(f, k - 1, df_error))
                    half = float(np.sqrt((k - 1) * stats.f.ppf(1 - alpha, k - 1, df_error)) * se)
                    partial.append((diff, se, t, pv, diff - half, diff + half))
                else:
                    raw.append(raw_p)
                    m = len(pairs)
                    level = {
                        "bonferroni": alpha / m,
                        "sidak": 1 - (1 - alpha) ** (1 / m),
                        "holm": alpha / m,
                        "lsd": alpha,
                    }[method]
                    half = float(stats.t.ppf(1 - level / 2, df_error) * se)
                    partial.append((diff, se, t, raw_p, diff - half, diff + half))
        if method in ("bonferroni", "holm", "sidak"):
            adjusted = multipletests(raw, method=method)[1]
            partial = [(d, se, t, float(a), lo, hi) for (d, se, t, _, lo, hi), a in zip(partial, adjusted, strict=True)]
        for (i, j), (diff, se, stat_value, pv, low, high) in zip(pairs, partial, strict=True):
            rows.append([labels[i], labels[j], diff, se, stat_value, pv, low, high])
    out = Output(
        f"Comparaciones múltiples: {response} según {factor}",
        f"{dict(_POSTHOC)[method].split(' (')[0]} · {k} grupos · n = {n}",
    )
    significant = sum(1 for r in rows if r[5] < alpha)
    out.tiles(
        [
            ("Grupos", k),
            ("Comparaciones", len(rows)),
            (f"Diferencias significativas (α = {alpha:g})", significant),
            ("Error cuadrático medio", mse),
        ]
    )
    if method == "lsd":
        out.alert(
            "warning",
            "Sin ajuste por comparaciones múltiples la probabilidad de algún falso positivo crece rápido con el número de grupos. Úsela solo tras un ANOVA significativo.",
        )
    if method == "holm":
        out.alert(
            "info",
            "Holm ajusta los valores p; no tiene un intervalo de confianza simultáneo simple, por eso los límites mostrados usan Bonferroni.",
        )
    stat_name = "q" if method in ("tukey", "games_howell") else "t" if method != "dunnett" else "t de Dunnett"
    ci_label = f"IC {100 * (1 - alpha):g} % simultáneo"
    out.table(
        "Diferencias entre pares",
        [
            "Grupo A",
            "Grupo B",
            "Diferencia (A − B)",
            "Error est.",
            stat_name,
            "Valor p ajustado",
            f"{ci_label} inf.",
            "sup.",
            "¿Difieren?",
        ],
        [[a, b, d, se, st, P(pv), lo, hi, "sí" if pv < alpha else "no"] for a, b, d, se, st, pv, lo, hi in rows],
    )
    out.chart(
        interval_chart(
            "Diferencias y sus intervalos de confianza",
            "Comparación",
            "Diferencia de medias",
            [{"name": f"{a} − {b}", "mean": d, "lower": lo, "upper": hi} for a, b, d, _, _, _, lo, hi in rows[:40]],
            hlines=[{"value": 0.0}],
        )
    )
    out.table(
        "Medias por grupo",
        ["Grupo", "n", "Media", "Desv. est."],
        [[label, sizes[i], means[i], float(arrays[i].std(ddof=1))] for i, label in enumerate(labels)],
    )
    return out.set(
        significant=significant,
        comparisons=len(rows),
        pairs={f"{a}|{b}": float(pv) for a, b, _, _, _, pv, _, _ in rows},
    ).build()


def _posthoc_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    response, factor = py_name(roles["response"]), py_name(roles["factor"])
    if p["method"] == "tukey":
        return (
            "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n\n"
            + load_line(dataset)
            + f"d = df[[{response}, {factor}]].dropna()\nprint(pairwise_tukeyhsd(d[{response}], d[{factor}], alpha={p['alpha']}))\n"
        )
    if p["method"] == "dunnett":
        return (
            "from scipy import stats\n\n"
            + load_line(dataset)
            + f"groups = {{k: g[{response}].dropna() for k, g in df.groupby({factor})}}\ncontrol = groups.pop({(p.get('control') or '...')!r})\nprint(stats.dunnett(*groups.values(), control=control))\n"
        )
    return (
        "import itertools\nfrom scipy import stats\nfrom statsmodels.stats.multitest import multipletests\n\n"
        + load_line(dataset)
        + f"groups = {{k: g[{response}].dropna() for k, g in df.groupby({factor})}}\npairs = list(itertools.combinations(groups, 2))\nraw = [stats.ttest_ind(groups[a], groups[b]).pvalue for a, b in pairs]\nprint(multipletests(raw, method='{p['method'] if p['method'] in ('bonferroni', 'holm') else 'bonferroni'}')[1])\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


RESPONSE = RoleSpec("response", "Respuesta (Y)", "numeric")

SPECS = (
    _spec(
        "glm_anova", "ANOVA de varios factores y ANCOVA", "Modelos lineales",
        "Hasta 4 factores con interacciones y covariables, suma de cuadrados tipo I, II o III, medias marginales y gráficos de efectos.",
        _glm_anova,
        roles=(RESPONSE, RoleSpec("factors", "Factores (categorías)", "any", multiple=True, max_count=MAX_FACTORS), RoleSpec("covariates", "Covariables numéricas (ANCOVA, opcional)", "numeric", multiple=True, optional=True)),
        params=(FieldSpec("interactions", "Interacciones", "select", "full", (("none", "Solo efectos principales"), ("two", "Hasta de dos factores"), ("full", "Todas"))), FieldSpec("ss_type", "Suma de cuadrados", "select", "III", (("I", "Tipo I (secuencial)"), ("II", "Tipo II"), ("III", "Tipo III (la habitual con interacciones)"))), ALPHA),
        code=_glm_anova_code, example={"roles": {"response": "y", "factors": ["g", "h"], "covariates": ["x1"]}, "params": {"interactions": "two"}},
    ),
    _spec(
        "rm_anova", "ANOVA de medidas repetidas", "Modelos lineales",
        "Un mismo sujeto medido varias veces: esfericidad de Mauchly, correcciones de Greenhouse-Geisser y Huynh-Feldt, y comparaciones por pares.",
        _repeated,
        roles=(RoleSpec("vars", "Mediciones (una columna por condición)", "numeric", multiple=True, min_count=2),),
        params=(FieldSpec("adjust", "Ajuste de las comparaciones por pares", "select", "holm", _ADJUST), ALPHA),
        code=_repeated_code, example={"roles": {"vars": ["r1", "r2", "r3"]}, "params": {}},
    ),
    _spec(
        "manova", "MANOVA", "Modelos lineales",
        "Compara grupos en varias respuestas a la vez: Wilks, Pillai, Hotelling-Lawley y Roy, con pruebas univariadas de seguimiento y M de Box.",
        _manova,
        roles=(RoleSpec("responses", "Respuestas numéricas (2 o más)", "numeric", multiple=True, min_count=2), RoleSpec("factors", "Factores", "any", multiple=True, max_count=3)),
        params=(ALPHA,), code=_manova_code, example={"roles": {"responses": ["x1", "x2", "x3"], "factors": ["g"]}, "params": {}},
    ),
    _spec(
        "nested", "ANOVA anidado", "Modelos lineales",
        "Un factor cuyos niveles solo tienen sentido dentro de otro (lotes dentro de proveedores, aulas dentro de escuelas), con componentes de la varianza.",
        _nested,
        roles=(RESPONSE, RoleSpec("factor_a", "Factor principal", "any"), RoleSpec("factor_b", "Factor anidado en el principal", "any")),
        params=(FieldSpec("b_random", "El factor anidado es aleatorio", "boolean", True, help="Aleatorio: los niveles son una muestra de muchos posibles (lo habitual)."), ALPHA),
        code=_nested_code, example={"roles": {"response": "ya", "factor_a": "g", "factor_b": "aula"}, "params": {}},
    ),
    _spec(
        "posthoc", "Comparaciones múltiples (post hoc)", "Comparaciones",
        "Qué grupos difieren de cuáles: Tukey, Games-Howell, Dunnett, Bonferroni, Holm, Šidák, Scheffé o Fisher, con intervalos de confianza.",
        _posthoc,
        roles=(RESPONSE, RoleSpec("factor", "Factor (grupos)", "categorical")),
        params=(FieldSpec("method", "Método", "select", "tukey", _POSTHOC), FieldSpec("control", "Grupo de control (vacío = el primero)", "text", optional=True, visible_if=("method", ("dunnett",))), ALPHA),
        code=_posthoc_code, example={"roles": {"response": "y", "factor": "g"}, "params": {"method": "games_howell"}},
    ),
)  # fmt: skip

registry.register(*SPECS)
