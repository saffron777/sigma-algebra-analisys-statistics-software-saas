"""Regression models beyond OLS (Minitab *Regression*, SPSS *Regression/Generalized Linear Models*, EViews equations)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import auc, roc_curve
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import bar_chart, line_chart, scatter_chart
from sigma.domain.analysis.expressions import evaluate
from sigma.domain.analysis.modeling import (
    ALPHA,
    PREDICTIONS,
    PREDICTORS,
    RESPONSE,
    build_design,
    ci_columns,
    coefficient_table,
    dropped_note,
    encode_binary,
    fit_failure,
    formula_code,
    need_enough,
    prediction_frame,
)
from sigma.domain.analysis.output import Output, P, strong
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "modelos"
FIT_ERRORS = (PerfectSeparationError, np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError)
INTERACTIONS = FieldSpec(
    "interactions",
    "Interacciones",
    "select",
    "none",
    (("none", "Ninguna"), ("two", "Entre pares de predictores"), ("full", "Todas")),
)
BINARY_RESPONSE = RoleSpec("response", "Respuesta binaria (2 valores)", "any")


def _fit(model, **kwargs):
    try:
        return model.fit(**kwargs)
    except FIT_ERRORS as error:
        raise fit_failure(error) from None


def _extreme(out: Output, result, limit: float = 12.0) -> None:
    params = np.asarray(result.params, dtype=float)
    if np.any(np.abs(params) > limit):
        out.alert(
            "warning",
            "Algún coeficiente es enorme: puede haber separación perfecta (un predictor separa casi por completo las clases). Revise o simplifique el modelo.",
        )


# ============================================================================ binary logistic / probit / cloglog
_LINKS = (("logit", "Logit (regresión logística)"), ("probit", "Probit"), ("cloglog", "Complemento log-log"))


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, groups: int = 10) -> tuple[float, float, int]:
    """Hosmer-Lemeshow chi-square over deciles of predicted risk: (statistic, p-value, groups used)."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, min(groups, len(y)))
    statistic = 0.0
    for part in parts:
        n, observed, expected = len(part), y[part].sum(), p[part].sum()
        mean = expected / n
        if 0 < mean < 1:
            statistic += (observed - expected) ** 2 / (n * mean * (1 - mean))
    df = max(len(parts) - 2, 1)
    return float(statistic), float(stats.chi2.sf(statistic, df)), len(parts)


def _binary(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha = roles["response"], roles["predictors"], p["alpha"]
    y, event, other = encode_binary(frame[response], response, (p.get("event") or "").strip() or None)
    work = frame.copy()
    work[response] = y
    design = build_design(work, response, predictors, interactions=p["interactions"])
    events = int(design.data[design.response].sum())
    n = len(design.data)
    if events < 1 or events == n:
        raise InvalidInputError("La respuesta tiene un solo valor entre las filas completas: no hay nada que modelar.")
    link = {"logit": links.Logit(), "probit": links.Probit(), "cloglog": links.CLogLog()}[p["link"]]
    result = _fit(smf.glm(design.formula, design.data, family=families.Binomial(link=link)))
    k = len(result.params) - 1
    need_enough(design, k + 1)
    prob = np.asarray(result.fittedvalues)
    yv = design.data[design.response].to_numpy()
    llf, llnull = float(result.llf), float(result.llnull)
    lr = 2 * (llf - llnull)
    cox = 1 - np.exp(2 * (llnull - llf) / n)
    nagelkerke = cox / (1 - np.exp(2 * llnull / n))
    out = Output(
        f"Regresión {dict(_LINKS)[p['link']].split(' (')[0].lower()}: {response}",
        f"evento = {event} · referencia = {other} · n = {n}",
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("Eventos", events),
            ("χ² de razón de verosimilitud", lr),
            ("McFadden R²", 1 - llf / llnull),
        ]
    )
    if events / max(k, 1) < 10:
        out.alert(
            "warning",
            f"Solo hay {events} eventos para {k} parámetro(s) ({events / max(k, 1):.1f} por parámetro): se recomiendan al menos 10 para que los coeficientes sean estables.",
        )
    _extreme(out, result)
    coefficient_table(
        out,
        design,
        result,
        alpha,
        exponentiate="Odds ratio" if p["link"] == "logit" else None,
        note="Odds ratio = exp(coeficiente): cuánto se multiplican las ventajas del evento por cada unidad (o frente a la categoría de referencia)."
        if p["link"] == "logit"
        else None,
    )
    p_lr = float(stats.chi2.sf(lr, k))
    hl, hl_p, groups = hosmer_lemeshow(yv, prob)
    out.table(
        "Ajuste del modelo",
        ["Medida", "Valor", "gl", "Valor p"],
        [
            ["Razón de verosimilitud (frente al modelo sin predictores)", lr, k, P(p_lr)],
            ["Hosmer-Lemeshow", hl, groups - 2, P(hl_p)],
            ["R² de McFadden", 1 - llf / llnull, None, None],
            ["R² de Cox-Snell", float(cox), None, None],
            ["R² de Nagelkerke", float(nagelkerke), None, None],
            ["Log-verosimilitud", llf, None, None],
            ["AIC", -2 * llf + 2 * (k + 1), None, None],
            ["BIC", -2 * llf + (k + 1) * np.log(n), None, None],
        ],
        note="Hosmer-Lemeshow: un valor p pequeño indica mal ajuste (calibración).",
    )
    cutoff = p["cutoff"]
    pred = (prob >= cutoff).astype(int)
    tp, tn = int(((pred == 1) & (yv == 1)).sum()), int(((pred == 0) & (yv == 0)).sum())
    fp, fn = int(((pred == 1) & (yv == 0)).sum()), int(((pred == 0) & (yv == 1)).sum())
    out.table(
        f"Clasificación con corte {cutoff:g}",
        ["", f"Predicho = {event}", f"Predicho = {other}", "Total"],
        [[f"Real = {event}", tp, fn, tp + fn], [f"Real = {other}", fp, tn, fp + tn]],
    )
    accuracy = (tp + tn) / n
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    fpr, tpr, _ = roc_curve(yv, prob)
    area = float(auc(fpr, tpr))
    out.table(
        "Medidas de clasificación",
        ["Medida", "Valor"],
        [
            ["Exactitud", accuracy],
            ["Sensibilidad", sens],
            ["Especificidad", spec],
            ["Valor predictivo positivo", tp / max(tp + fp, 1)],
            ["Valor predictivo negativo", tn / max(tn + fn, 1)],
            ["Área bajo la curva ROC (AUC)", strong(area)],
        ],
    )
    out.chart(
        line_chart(
            f"Curva ROC (AUC = {area:.3f})",
            "1 − especificidad",
            "Sensibilidad",
            [
                {"name": "Modelo", "x": fpr, "y": tpr},
                {"name": "Azar", "x": np.array([0.0, 1.0]), "y": np.array([0.0, 1.0]), "dash": "dash"},
            ],
            height=380,
        )
    )
    deviance_resid = np.asarray(result.resid_deviance)
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(
            frame, design, probabilidad=prob, clase_predicha=pred.astype(float), residuo_deviance=deviance_resid
        ),
    )
    return out.set(
        n=n,
        events=events,
        auc=area,
        accuracy=accuracy,
        mcfadden=1 - llf / llnull,
        lr_p=p_lr,
        hl_p=hl_p,
        coefficients={design.pretty(str(t)): float(v) for t, v in result.params.items()},
    ).build()


def _binary_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    family = {"logit": "Logit", "probit": "Probit", "cloglog": "CLogLog"}[p["link"]]
    return (
        "import statsmodels.api as sm\n"
        + load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"df['y01'] = (df[{py_name(roles['response'])}].astype(str) == sorted(df[{py_name(roles['response'])}].dropna().astype(str).unique())[-1]).astype(int)\n"
        + "formula = formula.replace("
        + repr('Q("' + roles["response"] + '")')
        + ", 'y01')\n"
        + f"model = smf.glm(formula, df, family=sm.families.Binomial(link=sm.families.links.{family}())).fit()\nprint(model.summary())\n"
    )


# ============================================================================ ordinal
def _levels_of(series: pd.Series) -> list[Any]:
    values = list(pd.unique(series.dropna()))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


def _ordinal(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha = roles["response"], roles["predictors"], p["alpha"]
    design = build_design(frame, response, predictors, response_kind="raw", interactions=p["interactions"])
    y = design.data[design.response]
    levels = _levels_of(y)
    if len(levels) < 3:
        raise InvalidInputError(
            f"«{response}» debe tener al menos 3 categorías ordenadas (tiene {len(levels)}); con 2 use la regresión binaria."
        )
    if len(levels) > 15:
        raise InvalidInputError(
            f"«{response}» tiene {len(levels)} categorías; para una respuesta ordinal se admiten hasta 15."
        )
    rhs = design.formula.split("~", 1)[1]
    x = patsy.dmatrix(rhs, design.data, return_type="dataframe").drop(columns="Intercept", errors="ignore")
    if x.shape[1] == 0:
        raise InvalidInputError("Elija al menos un predictor.")
    endog = pd.Series(pd.Categorical(y, categories=levels, ordered=True), index=y.index)
    n = len(endog)
    need_enough(design, x.shape[1] + len(levels) - 1, per_parameter=3)
    model = OrderedModel(endog, x, distr=p["distr"])
    result = _fit(model, method="bfgs", disp=False, maxiter=1000)
    k = x.shape[1]
    counts = endog.value_counts().reindex(levels).to_numpy()
    llnull = float((counts * np.log(counts / n)).sum())
    llf = float(result.llf)
    lr = 2 * (llf - llnull)
    probs = np.asarray(result.predict(x))
    predicted = probs.argmax(axis=1)
    actual = np.array([levels.index(v) for v in y])
    accuracy = float((predicted == actual).mean())
    out = Output(
        f"Regresión ordinal ({'logit' if p['distr'] == 'logit' else 'probit'}): {response}",
        f"{len(levels)} categorías ordenadas · n = {n}",
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("Categorías", len(levels)),
            ("χ² de razón de verosimilitud", lr),
            ("Exactitud", accuracy),
        ]
    )
    params = result.params.iloc[:k]
    se = result.bse.iloc[:k]
    ci = result.conf_int(alpha=alpha).iloc[:k]
    z = params / se
    pv = result.pvalues.iloc[:k]
    head = ["Término", "Coeficiente", "Error est.", "z", "Valor p", *ci_columns(alpha)]
    exp_head = ["Odds ratio", *[f"{c} (OR)" for c in ci_columns(alpha)]] if p["distr"] == "logit" else []
    rows = []
    for term in params.index:
        row = [
            design.pretty(str(term)),
            float(params[term]),
            float(se[term]),
            float(z[term]),
            P(float(pv[term])),
            float(ci.loc[term].iloc[0]),
            float(ci.loc[term].iloc[1]),
        ]
        if exp_head:
            row += [
                float(np.exp(params[term])),
                float(np.exp(ci.loc[term].iloc[0])),
                float(np.exp(ci.loc[term].iloc[1])),
            ]
        rows.append(row)
    out.table(
        "Coeficientes (modelo de probabilidades proporcionales)",
        head + exp_head,
        rows,
        note="Un coeficiente positivo desplaza la respuesta hacia las categorías más altas.",
    )
    thresholds = model.transform_threshold_params(result.params)[1:-1]
    out.table(
        "Umbrales entre categorías",
        ["Entre", "Umbral"],
        [[f"{levels[i]} | {levels[i + 1]}", float(t)] for i, t in enumerate(thresholds)],
    )
    out.table(
        "Ajuste del modelo",
        ["Medida", "Valor", "gl", "Valor p"],
        [
            ["Razón de verosimilitud (frente a solo umbrales)", lr, k, P(float(stats.chi2.sf(lr, k)))],
            ["R² de McFadden", 1 - llf / llnull, None, None],
            ["Log-verosimilitud", llf, None, None],
            ["AIC", -2 * llf + 2 * (k + len(levels) - 1), None, None],
            ["BIC", -2 * llf + (k + len(levels) - 1) * np.log(n), None, None],
        ],
    )
    matrix = pd.crosstab(
        pd.Categorical(actual, categories=range(len(levels))), pd.Categorical(predicted, categories=range(len(levels)))
    ).to_numpy()
    out.table(
        "Categoría real (filas) frente a predicha (columnas)",
        ["", *[str(v) for v in levels]],
        [[str(levels[i]), *(int(c) for c in matrix[i])] for i in range(len(levels))],
    )
    prediction = prediction_frame(frame, design, categoria_predicha_codigo=predicted.astype(float))
    for i, level in enumerate(levels):
        prediction[f"prob_{level}"] = pd.Series(probs[:, i], index=design.used).reindex(frame.index)
    out.dataset("predicciones", PREDICTIONS.label, prediction)
    return out.set(
        n=n,
        accuracy=accuracy,
        lr_p=float(stats.chi2.sf(lr, k)),
        coefficients={design.pretty(str(t)): float(v) for t, v in params.items()},
    ).build()


def _ordinal_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import pandas as pd\nimport patsy\nfrom statsmodels.miscmodels.ordinal_model import OrderedModel\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['response'])}, *{roles['predictors']!r}]].dropna()\n"
        + f"y = pd.Series(pd.Categorical(d[{py_name(roles['response'])}], categories=sorted(d[{py_name(roles['response'])}].unique()), ordered=True))\n"
        + f"X = patsy.dmatrix(' + '.join({roles['predictors']!r}), d, return_type='dataframe').drop(columns='Intercept')\n"
        + f"print(OrderedModel(y, X, distr={p['distr']!r}).fit(method='bfgs').summary())\n"
    )


# ============================================================================ multinomial
def _multinomial(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha = roles["response"], roles["predictors"], p["alpha"]
    design = build_design(frame, response, predictors, response_kind="raw", interactions=p["interactions"])
    y = design.data[design.response]
    levels = [str(v) for v in _levels_of(y)]
    if len(levels) < 3 or len(levels) > 12:
        raise InvalidInputError(f"«{response}» necesita entre 3 y 12 categorías (tiene {len(levels)}).")
    reference = (p.get("reference") or "").strip() or levels[0]
    if reference not in levels:
        raise InvalidInputError(f"La categoría de referencia «{reference}» no existe en «{response}».")
    order = [reference, *[v for v in levels if v != reference]]
    codes = y.astype(str).map({v: i for i, v in enumerate(order)})
    data = design.data.assign(**{design.response: codes})
    rhs = design.formula.split("~", 1)[1]
    n = len(data)
    result = _fit(smf.mnlogit(f"{design.response} ~ {rhs}", data), method="newton", disp=False, maxiter=200)
    k = result.params.shape[0] - 1
    need_enough(design, (k + 1) * (len(order) - 1), per_parameter=2)
    probs = np.asarray(result.predict(data))
    predicted = probs.argmax(axis=1)
    accuracy = float((predicted == codes.to_numpy()).mean())
    z = stats.norm.ppf(1 - alpha / 2)
    out = Output(
        f"Regresión logística multinomial: {response}", f"referencia = {reference} · {len(order)} categorías · n = {n}"
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("Categorías", len(order)),
            ("χ² de razón de verosimilitud", float(result.llr)),
            ("McFadden R²", float(result.prsquared)),
        ]
    )
    rows = []
    for j, category in enumerate(order[1:]):
        for term in result.params.index:
            coef, se, pv = (
                float(result.params.iloc[:, j][term]),
                float(result.bse.iloc[:, j][term]),
                float(result.pvalues.iloc[:, j][term]),
            )
            rows.append(
                [
                    category,
                    "Constante" if term == "Intercept" else design.pretty(str(term)),
                    coef,
                    se,
                    coef / se,
                    P(pv),
                    coef - z * se,
                    coef + z * se,
                    float(np.exp(coef)),
                    float(np.exp(coef - z * se)),
                    float(np.exp(coef + z * se)),
                ]
            )
    out.table(
        f"Coeficientes frente a «{reference}»",
        [
            "Categoría",
            "Término",
            "Coeficiente",
            "Error est.",
            "z",
            "Valor p",
            *ci_columns(alpha),
            "Razón de riesgos relativos",
            "RRR inf.",
            "RRR sup.",
        ],
        rows,
        note="Razón de riesgos relativos = exp(coeficiente): cambio en las probabilidades relativas de la categoría frente a la de referencia.",
    )
    out.table(
        "Ajuste del modelo",
        ["Medida", "Valor", "gl", "Valor p"],
        [
            ["Razón de verosimilitud", float(result.llr), (k) * (len(order) - 1), P(float(result.llr_pvalue))],
            ["R² de McFadden", float(result.prsquared), None, None],
            ["Log-verosimilitud", float(result.llf), None, None],
            ["AIC", float(result.aic), None, None],
            ["BIC", float(result.bic), None, None],
        ],
    )
    matrix = pd.crosstab(
        pd.Categorical(codes.to_numpy(), categories=range(len(order))),
        pd.Categorical(predicted, categories=range(len(order))),
    ).to_numpy()
    out.table(
        f"Real (filas) frente a predicho (columnas) · exactitud {accuracy:.1%}",
        ["", *order],
        [[order[i], *(int(c) for c in matrix[i])] for i in range(len(order))],
    )
    prediction = prediction_frame(frame, design, categoria_predicha_codigo=predicted.astype(float))
    for i, level in enumerate(order):
        prediction[f"prob_{level}"] = pd.Series(probs[:, i], index=design.used).reindex(frame.index)
    out.dataset("predicciones", PREDICTIONS.label, prediction)
    return out.set(n=n, accuracy=accuracy, llr_p=float(result.llr_pvalue)).build()


def _multinomial_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + "model = smf.mnlogit(formula, df.dropna(subset=["
        + py_name(roles["response"])
        + ", *predictors])).fit()\nprint(model.summary())\n"
    )


# ============================================================================ counts
_COUNT = (
    ("poisson", "Poisson"),
    ("quasi", "Cuasi-Poisson (corrige la sobredispersión)"),
    ("negbin", "Binomial negativa"),
)


def cameron_trivedi(y: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """Dispersion test of Cameron and Trivedi: (alpha, p-value) from ((y − μ)² − y)/μ = α·μ."""
    target = ((y - mu) ** 2 - y) / mu
    fit = sm.OLS(target, mu).fit()
    return float(fit.params[0]), float(fit.pvalues[0])


def _count(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha, kind = roles["response"], roles["predictors"], p["alpha"], p["family"]
    extra = {"exposure": roles["exposure"]} if roles.get("exposure") else None
    design = build_design(frame, response, predictors, interactions=p["interactions"], extra=extra)
    y = design.data[design.response].to_numpy()
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise InvalidInputError(f"«{response}» debe contener conteos: enteros no negativos.")
    offset = np.log(design.data[design.extra["exposure"]]) if extra else None
    if extra and (design.data[design.extra["exposure"]] <= 0).any():
        raise InvalidInputError("La exposición debe ser positiva.")
    n = len(y)
    if kind == "negbin":
        result = _fit(
            smf.negativebinomial(design.formula, design.data, offset=offset), method="bfgs", disp=False, maxiter=500
        )
        mu = (
            np.asarray(result.predict(design.data, offset=offset))
            if offset is not None
            else np.asarray(result.predict(design.data))
        )
        dispersion = float(result.params["alpha"])
        free = result.params.drop("alpha")
        k = len(free)
    else:
        result = _fit(
            smf.glm(design.formula, design.data, family=families.Poisson(), offset=offset),
            **({"scale": "X2"} if kind == "quasi" else {}),
        )
        mu = np.asarray(result.fittedvalues)
        k = len(result.params)
    need_enough(design, k + 1)
    pearson = float(((y - mu) ** 2 / np.where(mu > 0, mu, 1)).sum())
    df_resid = n - k
    ratio = pearson / df_resid
    ct_alpha, ct_p = cameron_trivedi(y, np.where(mu > 0, mu, 1e-9))
    out = Output(
        f"Regresión de conteos ({dict(_COUNT)[kind].split(' (')[0]}): {response}",
        f"n = {n}" + (f" · exposición = {roles['exposure']}" if extra else ""),
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("Media de la respuesta", float(y.mean())),
            ("Pearson χ² / gl", ratio),
            ("Parámetros", k),
        ]
    )
    if kind == "poisson" and ratio > 1.5 and ct_p < alpha:
        out.alert(
            "warning",
            f"Hay sobredispersión (χ² de Pearson / gl = {ratio:.2f}; prueba de Cameron-Trivedi p = {ct_p:.4f}): los errores estándar de Poisson son demasiado pequeños. Use cuasi-Poisson o binomial negativa.",
        )
    elif kind == "poisson":
        out.alert("success", f"Sin evidencia fuerte de sobredispersión (χ² de Pearson / gl = {ratio:.2f}).")
    elif kind == "negbin" and (dispersion < 1e-3 or not np.isfinite(np.asarray(result.bse)).all()):
        out.alert(
            "info",
            "La dispersión estimada es casi cero (los datos no tienen más variabilidad que la de Poisson): los errores "
            "estándar pueden no calcularse y el modelo de Poisson es suficiente.",
        )
    coefficient_table(
        out,
        design,
        result if kind != "negbin" else _Subset(result, free.index),
        alpha,
        exponentiate="Razón de tasas",
        stat_label="z",
        note="Razón de tasas = exp(coeficiente): cuánto se multiplica el conteo esperado por cada unidad del predictor.",
    )
    fit_rows = [["Dispersión (χ² de Pearson / gl)", ratio, None], ["Cameron-Trivedi: α", ct_alpha, P(ct_p)]]
    if kind == "negbin":
        fit_rows.insert(0, ["α de la binomial negativa", dispersion, None])
    if kind != "quasi":
        fit_rows += [
            ["Log-verosimilitud", float(result.llf), None],
            ["AIC", float(result.aic), None],
            ["BIC", -2 * float(result.llf) + (k + (1 if kind == "negbin" else 0)) * np.log(n), None],
        ]
    if kind != "negbin":
        fit_rows.insert(
            0, ["Devianza residual", float(result.deviance), P(float(stats.chi2.sf(result.deviance, df_resid)))]
        )
    out.table("Ajuste del modelo", ["Medida", "Valor", "Valor p"], fit_rows)
    top = int(min(max(y.max(), 3), 15))
    ks = np.arange(top + 1)
    observed = np.array([(y == v).mean() for v in ks])
    if kind == "negbin":
        size = 1 / dispersion
        expected = np.array([stats.nbinom.pmf(v, size, size / (size + mu)).mean() for v in ks])
    else:
        expected = np.array([stats.poisson.pmf(v, mu).mean() for v in ks])
    out.chart(
        bar_chart(
            "Frecuencias observadas y esperadas por el modelo",
            "Conteo",
            "Proporción de filas",
            [str(v) for v in ks],
            [{"name": "Observadas", "y": observed}, {"name": "Esperadas", "y": expected}],
        )
    )
    residual = (y - mu) / np.sqrt(np.where(mu > 0, mu, 1))
    out.dataset(
        "predicciones", PREDICTIONS.label, prediction_frame(frame, design, ajustado=mu, residuo_pearson=residual)
    )
    return out.set(
        n=n,
        dispersion=ratio,
        cameron_trivedi_p=ct_p,
        coefficients={
            design.pretty(str(t)): float(v) for t, v in (free if kind == "negbin" else result.params).items()
        },
    ).build()


class _Subset:
    """A results object restricted to some parameters (the dispersion parameter of a negative binomial is shown apart)."""

    def __init__(self, result, keep) -> None:
        self.params, self.bse, self.pvalues = result.params[keep], result.bse[keep], result.pvalues[keep]
        self.tvalues = result.tvalues[keep]
        self._ci = result.conf_int
        self._keep = keep

    def conf_int(self, alpha: float):
        return self._ci(alpha=alpha).loc[self._keep]


def _count_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    family = "sm.families.Poisson()"
    scale = ".fit(scale='X2')" if p["family"] == "quasi" else ".fit()"
    if p["family"] == "negbin":
        return (
            load_line(dataset)
            + formula_code(roles["response"], roles["predictors"])
            + "model = smf.negativebinomial(formula, df).fit()\nprint(model.summary())\n"
        )
    return (
        "import statsmodels.api as sm\n"
        + load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"model = smf.glm(formula, df, family={family}){scale}\nprint(model.summary())\n"
    )


# ============================================================================ generalised linear model
_FAMILIES = (
    ("gaussian", "Normal (gaussiana)"),
    ("binomial", "Binomial (0/1 o proporciones)"),
    ("poisson", "Poisson"),
    ("gamma", "Gamma (positiva y asimétrica)"),
    ("inverse_gaussian", "Gaussiana inversa"),
    ("negative_binomial", "Binomial negativa (α fijo)"),
)
_GLM_LINKS = {
    "gaussian": ("identity", "log", "inverse_power"),
    "binomial": ("logit", "probit", "cloglog", "log"),
    "poisson": ("log", "identity", "sqrt"),
    "gamma": ("inverse_power", "log", "identity"),
    "inverse_gaussian": ("inverse_squared", "log", "identity", "inverse_power"),
    "negative_binomial": ("log", "identity", "sqrt"),
}
_LINK_CLASSES = {
    "identity": links.Identity,
    "log": links.Log,
    "inverse_power": links.InversePower,
    "logit": links.Logit,
    "probit": links.Probit,
    "cloglog": links.CLogLog,
    "sqrt": links.Sqrt,
    "inverse_squared": links.InverseSquared,
}
_ALL_LINKS = tuple(
    (name, name.replace("_", " "))
    for name in ("default", "identity", "log", "logit", "probit", "cloglog", "inverse_power", "inverse_squared", "sqrt")
)


def _glm(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha, family_name = roles["response"], roles["predictors"], p["alpha"], p["family"]
    design = build_design(frame, response, predictors, interactions=p["interactions"])
    y = design.data[design.response].to_numpy()
    if family_name == "binomial" and ((y < 0).any() or (y > 1).any()):
        raise InvalidInputError("Con la familia binomial la respuesta debe estar entre 0 y 1.")
    if family_name in ("gamma", "inverse_gaussian") and (y <= 0).any():
        raise InvalidInputError("Con esa familia la respuesta debe ser estrictamente positiva.")
    if family_name in ("poisson", "negative_binomial") and (y < 0).any():
        raise InvalidInputError("Con esa familia la respuesta no puede ser negativa.")
    link_name = p["link"] if p["link"] != "default" else _GLM_LINKS[family_name][0]
    if link_name not in _GLM_LINKS[family_name]:
        raise InvalidInputError(
            f"La liga «{link_name}» no es válida para esa familia; use " + ", ".join(_GLM_LINKS[family_name]) + "."
        )
    link = _LINK_CLASSES[link_name]()
    family = {
        "gaussian": families.Gaussian,
        "binomial": families.Binomial,
        "poisson": families.Poisson,
        "gamma": families.Gamma,
        "inverse_gaussian": families.InverseGaussian,
        "negative_binomial": lambda link: families.NegativeBinomial(link=link, alpha=p["nb_alpha"]),
    }[family_name](link=link)
    result = _fit(
        smf.glm(design.formula, design.data, family=family), **({"scale": "X2"} if p["scale"] == "pearson" else {})
    )
    k, n = len(result.params), len(y)
    need_enough(design, k + 1)
    out = Output(
        f"Modelo lineal generalizado: {response}",
        f"familia {dict(_FAMILIES)[family_name].split(' (')[0].lower()} · liga {link_name.replace('_', ' ')} · n = {n}",
    )
    dropped_note(out, design)
    null_dev, dev = float(result.null_deviance), float(result.deviance)
    out.tiles(
        [
            ("Observaciones", n),
            ("Devianza residual", dev),
            ("Devianza nula", null_dev),
            ("Pseudo-R² (1 − dev/dev nula)", 1 - dev / null_dev if null_dev else 0.0),
        ]
    )
    exp_label = {"log": "exp(coef)", "logit": "Odds ratio"}.get(link_name)
    coefficient_table(out, design, result, alpha, stat_label="z", exponentiate=exp_label)
    scale = float(result.scale)
    fit = [
        [
            "Devianza residual",
            dev,
            n - k,
            P(float(stats.chi2.sf(dev, n - k))) if family_name in ("binomial", "poisson") else None,
        ],
        ["Devianza nula", null_dev, n - 1, None],
        ["χ² de Pearson", float(result.pearson_chi2), n - k, None],
        ["Parámetro de escala", scale, None, None],
        ["Log-verosimilitud", float(result.llf), None, None],
        ["AIC", float(result.aic), None, None],
    ]
    out.table(
        "Ajuste del modelo",
        ["Medida", "Valor", "gl", "Valor p"],
        fit,
        note="El valor p de la devianza solo es una prueba de bondad de ajuste con conteos o binarios agrupados.",
    )
    fitted = np.asarray(result.fittedvalues)
    pearson = np.asarray(result.resid_pearson)
    out.chart(
        scatter_chart(
            "Residuos de Pearson frente a valores ajustados",
            "Valor ajustado",
            "Residuo de Pearson",
            [{"name": "Residuos", "x": fitted, "y": pearson}],
            hlines=[{"value": 0}],
        )
    )
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(
            frame, design, ajustado=fitted, residuo_pearson=pearson, residuo_deviance=np.asarray(result.resid_deviance)
        ),
    )
    return out.set(
        n=n,
        deviance=dev,
        null_deviance=null_dev,
        aic=float(result.aic),
        coefficients={design.pretty(str(t)): float(v) for t, v in result.params.items()},
    ).build()


def _glm_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    family = {
        "gaussian": "Gaussian",
        "binomial": "Binomial",
        "poisson": "Poisson",
        "gamma": "Gamma",
        "inverse_gaussian": "InverseGaussian",
        "negative_binomial": f"NegativeBinomial(alpha={p['nb_alpha']})",
    }[p["family"]]
    family = family if "(" in family else family + "()"
    return (
        "import statsmodels.api as sm\n"
        + load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"model = smf.glm(formula, df, family=sm.families.{family}).fit()\nprint(model.summary())\n"
    )


# ============================================================================ quantile regression
_QUANTILES = tuple((f"{q:g}", f"{q:g}") for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))


def _quantile(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha = roles["response"], roles["predictors"], p["alpha"]
    design = build_design(frame, response, predictors, interactions=p["interactions"])
    quantiles = sorted({float(q) for q in p["quantiles"]})
    if not quantiles:
        raise InvalidInputError("Elija al menos un cuantil.")
    model = smf.quantreg(design.formula, design.data)
    fits = {q: _fit(model, q=q, max_iter=2000) for q in quantiles}
    ols = _fit(smf.ols(design.formula, design.data))
    n = len(design.data)
    need_enough(design, len(ols.params) + 1)
    out = Output(f"Regresión cuantílica: {response}", f"cuantiles {', '.join(f'{q:g}' for q in quantiles)} · n = {n}")
    dropped_note(out, design)
    out.tiles([("Observaciones", n), ("Cuantiles", len(quantiles)), ("R² de MCO", float(ols.rsquared))])
    terms = list(ols.params.index)
    head = ["Término", "MCO", *[f"q = {q:g}" for q in quantiles]]
    rows = []
    for term in terms:
        rows.append(
            [
                "Constante" if term == "Intercept" else design.pretty(str(term)),
                float(ols.params[term]),
                *(float(fits[q].params[term]) for q in quantiles),
            ]
        )
    out.table("Coeficientes por cuantil", head, rows)
    out.table(
        "Valores p por cuantil",
        ["Término", *[f"q = {q:g}" for q in quantiles]],
        [
            [
                "Constante" if t == "Intercept" else design.pretty(str(t)),
                *(P(float(fits[q].pvalues[t])) for q in quantiles),
            ]
            for t in terms
        ],
    )
    out.table(
        "Pseudo-R² por cuantil", ["Cuantil", "Pseudo-R²"], [[f"{q:g}", float(fits[q].prsquared)] for q in quantiles]
    )
    focus = design.numeric[0] if design.numeric else None
    if focus:
        grid = np.round(np.linspace(0.05, 0.95, 19), 2)
        path = {q: fits[q] if q in fits else _fit(model, q=float(q), max_iter=2000) for q in grid}
        coef = np.array([float(path[q].params[focus]) for q in grid])
        low = np.array([float(path[q].conf_int(alpha).loc[focus].iloc[0]) for q in grid])
        high = np.array([float(path[q].conf_int(alpha).loc[focus].iloc[1]) for q in grid])
        out.chart(
            line_chart(
                f"Efecto de {design.names[focus]} según el cuantil",
                "Cuantil",
                "Coeficiente",
                [{"name": "Regresión cuantílica", "x": grid, "y": coef, "markers": True}],
                bands=[{"name": f"IC {100 * (1 - alpha):g} %", "x": grid, "lower": low, "upper": high}],
                hlines=[{"value": float(ols.params[focus]), "label": "MCO", "dash": "dash"}],
            )
        )
    return out.set(
        n=n,
        coefficients={
            f"{q:g}": {design.pretty(str(t)): float(v) for t, v in fits[q].params.items()} for q in quantiles
        },
    ).build()


def _quantile_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"for q in {[float(q) for q in p['quantiles']]!r}:\n    print(q, smf.quantreg(formula, df).fit(q=q).params.to_dict())\n"
    )


# ============================================================================ robust regression
_ROBUST = (
    ("hc3", "MCO con errores robustos HC3 (heterocedasticidad)"),
    ("hc1", "MCO con errores robustos HC1"),
    ("hc0", "MCO con errores robustos HC0 (White)"),
    ("hac", "MCO con errores HAC de Newey-West (autocorrelación y heterocedasticidad)"),
    ("huber", "Regresión robusta M de Huber"),
    ("tukey", "Regresión robusta M de Tukey (bicuadrada)"),
    ("hampel", "Regresión robusta M de Hampel"),
)


def _robust(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha, method = roles["response"], roles["predictors"], p["alpha"], p["method"]
    design = build_design(frame, response, predictors, interactions=p["interactions"])
    ols = _fit(smf.ols(design.formula, design.data))
    n = len(design.data)
    need_enough(design, len(ols.params) + 1)
    out = Output(f"Regresión robusta: {response}", f"{dict(_ROBUST)[method]} · n = {n}")
    dropped_note(out, design)
    if method in ("hc0", "hc1", "hc3", "hac"):
        kwargs = (
            {"cov_type": method.upper()}
            if method != "hac"
            else {"cov_type": "HAC", "cov_kwds": {"maxlags": int(p["lags"])}}
        )
        robust = _fit(smf.ols(design.formula, design.data), **kwargs)
        bp = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)
        dw = float(sm.stats.stattools.durbin_watson(ols.resid))
        out.tiles(
            [
                ("Observaciones", n),
                ("R²", float(ols.rsquared)),
                ("Breusch-Pagan p", P(float(bp[1]))),
                ("Durbin-Watson", dw),
            ]
        )
        rows = []
        for term in ols.params.index:
            ratio = float(robust.bse[term] / ols.bse[term])
            rows.append(
                [
                    "Constante" if term == "Intercept" else design.pretty(str(term)),
                    float(ols.params[term]),
                    float(ols.bse[term]),
                    float(robust.bse[term]),
                    ratio,
                    float(robust.tvalues[term]),
                    P(float(robust.pvalues[term])),
                    float(robust.conf_int(alpha).loc[term].iloc[0]),
                    float(robust.conf_int(alpha).loc[term].iloc[1]),
                ]
            )
        out.table(
            "Coeficientes con errores robustos",
            [
                "Término",
                "Coeficiente",
                "Error est. clásico",
                "Error est. robusto",
                "Robusto / clásico",
                "t",
                "Valor p",
                *ci_columns(alpha),
            ],
            rows,
            note="Los coeficientes son los de MCO; solo cambian los errores estándar, valores p e intervalos.",
        )
        out.alert(
            "info",
            "Breusch-Pagan sugiere heterocedasticidad: los errores robustos son preferibles."
            if bp[1] < alpha
            else "No hay evidencia clara de heterocedasticidad; los errores robustos y los clásicos deberían ser parecidos.",
        )
        resid_source = ols
        weights = None
    else:
        norm = {
            "huber": sm.robust.norms.HuberT(),
            "tukey": sm.robust.norms.TukeyBiweight(),
            "hampel": sm.robust.norms.Hampel(),
        }[method]
        robust = _fit(smf.rlm(design.formula, design.data, M=norm))
        weights = np.asarray(robust.weights)
        down = int((weights < 0.5).sum())
        out.tiles(
            [
                ("Observaciones", n),
                ("Con peso < 0.5", down),
                ("Escala robusta", float(robust.scale)),
                ("R² de MCO", float(ols.rsquared)),
            ]
        )
        rows = []
        for term in robust.params.index:
            rows.append(
                [
                    "Constante" if term == "Intercept" else design.pretty(str(term)),
                    float(ols.params[term]),
                    float(ols.bse[term]),
                    float(robust.params[term]),
                    float(robust.bse[term]),
                    float(robust.tvalues[term]),
                    P(float(robust.pvalues[term])),
                    float(robust.conf_int(alpha).loc[term].iloc[0]),
                    float(robust.conf_int(alpha).loc[term].iloc[1]),
                ]
            )
        out.table(
            "MCO frente a regresión robusta",
            ["Término", "Coef. MCO", "EE MCO", "Coef. robusto", "EE robusto", "z", "Valor p", *ci_columns(alpha)],
            rows,
        )
        if down:
            out.alert(
                "info",
                f"{down} observación(es) recibieron un peso menor que 0.5: son atípicas respecto al ajuste y influyen menos en el modelo.",
            )
        resid_source = robust
    fitted, residual = np.asarray(resid_source.fittedvalues), np.asarray(resid_source.resid)
    out.chart(
        scatter_chart(
            "Residuos frente a valores ajustados",
            "Valor ajustado",
            "Residuo",
            [{"name": "Residuos", "x": fitted, "y": residual}],
            hlines=[{"value": 0}],
        )
    )
    if weights is not None:
        out.chart(
            line_chart(
                "Pesos asignados por la estimación robusta",
                "Observación",
                "Peso",
                [{"name": "Peso", "x": np.arange(1, n + 1), "y": weights, "markers": True}],
            )
        )
    columns = {"ajustado": fitted, "residuo": residual}
    if weights is not None:
        columns["peso_robusto"] = weights
    out.dataset("predicciones", PREDICTIONS.label, prediction_frame(frame, design, **columns))
    return out.set(
        n=n,
        coefficients={design.pretty(str(t)): float(v) for t, v in robust.params.items()},
        standard_errors={design.pretty(str(t)): float(v) for t, v in robust.bse.items()},
    ).build()


def _robust_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    base = load_line(dataset) + formula_code(roles["response"], roles["predictors"])
    if p["method"] in ("hc0", "hc1", "hc3"):
        return base + f"print(smf.ols(formula, df).fit(cov_type={p['method'].upper()!r}).summary())\n"
    if p["method"] == "hac":
        return (
            base
            + f"print(smf.ols(formula, df).fit(cov_type='HAC', cov_kwds={{'maxlags': {int(p['lags'])}}}).summary())\n"
        )
    norm = {"huber": "HuberT", "tukey": "TukeyBiweight", "hampel": "Hampel"}[p["method"]]
    return (
        "import statsmodels.api as sm\n"
        + base
        + f"print(smf.rlm(formula, df, M=sm.robust.norms.{norm}()).fit().summary())\n"
    )


# ============================================================================ weighted least squares
def _wls(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha = roles["response"], roles["predictors"], p["alpha"]
    extra = {"weights": roles["weights"]} if roles.get("weights") else None
    design = build_design(frame, response, predictors, interactions=p["interactions"], extra=extra)
    ols = _fit(smf.ols(design.formula, design.data))
    n = len(design.data)
    need_enough(design, len(ols.params) + 1)
    if extra:
        weights = design.data[design.extra["weights"]].to_numpy()
        if (weights <= 0).any():
            raise InvalidInputError("Los pesos deben ser estrictamente positivos.")
        origin = f"pesos de la columna «{roles['weights']}»"
    else:
        scale_model = _fit(sm.OLS(np.abs(ols.resid), sm.add_constant(ols.fittedvalues)))
        spread = np.asarray(scale_model.fittedvalues)
        floor = max(float(spread.max()) * 0.01, 1e-9)
        weights = 1.0 / np.maximum(spread, floor) ** 2
        origin = "pesos estimados: 1 / (desviación predicha por |residuo| ~ ajustado)²"
    wls = _fit(smf.wls(design.formula, design.data, weights=weights))
    bp_before = sm.stats.diagnostic.het_breuschpagan(ols.resid, ols.model.exog)[1]
    bp_after = sm.stats.diagnostic.het_breuschpagan(wls.resid * np.sqrt(weights), wls.model.exog)[1]
    out = Output(f"Mínimos cuadrados ponderados: {response}", f"{origin} · n = {n}")
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("R² ponderado", float(wls.rsquared)),
            ("Breusch-Pagan antes", P(float(bp_before))),
            ("Breusch-Pagan después", P(float(bp_after))),
        ]
    )
    rows = []
    for term in wls.params.index:
        rows.append(
            [
                "Constante" if term == "Intercept" else design.pretty(str(term)),
                float(ols.params[term]),
                float(ols.bse[term]),
                float(wls.params[term]),
                float(wls.bse[term]),
                float(wls.tvalues[term]),
                P(float(wls.pvalues[term])),
                float(wls.conf_int(alpha).loc[term].iloc[0]),
                float(wls.conf_int(alpha).loc[term].iloc[1]),
            ]
        )
    out.table(
        "MCO frente a mínimos cuadrados ponderados",
        ["Término", "Coef. MCO", "EE MCO", "Coef. MCP", "EE MCP", "t", "Valor p", *ci_columns(alpha)],
        rows,
    )
    out.chart(
        scatter_chart(
            "Residuos estandarizados por el peso",
            "Valor ajustado",
            "Residuo × √peso",
            [
                {
                    "name": "Residuos ponderados",
                    "x": np.asarray(wls.fittedvalues),
                    "y": np.asarray(wls.resid) * np.sqrt(weights),
                }
            ],
            hlines=[{"value": 0}],
        )
    )
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(
            frame, design, ajustado=np.asarray(wls.fittedvalues), residuo=np.asarray(wls.resid), peso=weights
        ),
    )
    return out.set(
        n=n,
        r2=float(wls.rsquared),
        bp_before=float(bp_before),
        bp_after=float(bp_after),
        coefficients={design.pretty(str(t)): float(v) for t, v in wls.params.items()},
    ).build()


def _wls_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    weights = f"df[{py_name(roles['weights'])}]" if roles.get("weights") else "1 / fitted_scale ** 2  # pesos estimados"
    return (
        load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"weights = {weights}\nprint(smf.wls(formula, df, weights=weights).fit().summary())\n"
    )


# ============================================================================ polynomial regression and interactions
def _polynomial(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, alpha, degree = roles["response"], roles["predictors"], p["alpha"], int(p["degree"])
    design = build_design(frame, response, predictors, degree=degree, interactions=p["interactions"])
    if not design.numeric and degree > 1:
        raise InvalidInputError("Los polinomios necesitan al menos un predictor numérico.")
    result = _fit(smf.ols(design.formula, design.data))
    n = len(design.data)
    need_enough(design, len(result.params) + 1)
    out = Output(f"Regresión polinómica / con interacciones: {response}", f"grado {degree} · n = {n}")
    dropped_note(out, design)
    out.tiles(
        [
            ("Observaciones", n),
            ("R²", float(result.rsquared)),
            ("R² ajustado", float(result.rsquared_adj)),
            ("Error estándar residual", float(np.sqrt(result.scale))),
        ]
    )
    coefficient_table(out, design, result, alpha, stat_label="t")
    sequence = []
    previous = None
    for d in range(1, degree + 1):
        fit = _fit(
            smf.ols(
                build_design(frame, response, predictors, degree=d, interactions=p["interactions"]).formula, design.data
            )
        )
        if previous is not None:
            f, pv, _ = fit.compare_f_test(previous)
            sequence.append([f"Grado {d} frente a {d - 1}", float(f), P(float(pv)), float(fit.rsquared_adj)])
        else:
            sequence.append(["Grado 1", float(fit.fvalue), P(float(fit.f_pvalue)), float(fit.rsquared_adj)])
        previous = fit
    out.table(
        "Selección del grado (prueba F secuencial)",
        ["Comparación", "F", "Valor p", "R² ajustado"],
        sequence,
        note="Un valor p pequeño indica que subir el grado mejora el ajuste.",
    )
    out.table(
        "Análisis de varianza del modelo",
        ["Fuente", "gl", "Suma de cuadrados", "Cuadrado medio", "F", "Valor p"],
        [
            [
                "Regresión",
                int(result.df_model),
                float(result.ess),
                float(result.ess / result.df_model),
                float(result.fvalue),
                P(float(result.f_pvalue)),
            ],
            ["Error", int(result.df_resid), float(result.ssr), float(result.scale), None, None],
            ["Total", int(result.df_model + result.df_resid), float(result.centered_tss), None, None, None],
        ],
    )
    if design.numeric:
        focus = design.numeric[0]
        grid = pd.DataFrame({t: float(design.data[t].mean()) for t in design.numeric}, index=range(120))
        for f in design.factors:
            grid[f] = design.data[f].mode().iloc[0]
        grid[focus] = np.linspace(design.data[focus].min(), design.data[focus].max(), 120)
        prediction = result.get_prediction(grid).summary_frame(alpha=alpha)
        out.chart(
            scatter_chart(
                f"{response} según {design.names[focus]} (otros predictores en su media)",
                design.names[focus],
                response,
                [{"name": "Datos", "x": design.data[focus].to_numpy(), "y": design.data[design.response].to_numpy()}],
                lines=[{"name": "Ajuste", "x": grid[focus].to_numpy(), "y": prediction["mean"].to_numpy()}],
                bands=[
                    {
                        "name": f"IC {100 * (1 - alpha):g} %",
                        "x": grid[focus].to_numpy(),
                        "lower": prediction["mean_ci_lower"].to_numpy(),
                        "upper": prediction["mean_ci_upper"].to_numpy(),
                    }
                ],
            )
        )
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(frame, design, ajustado=np.asarray(result.fittedvalues), residuo=np.asarray(result.resid)),
    )
    return out.set(
        n=n,
        r2=float(result.rsquared),
        r2_adj=float(result.rsquared_adj),
        coefficients={design.pretty(str(t)): float(v) for t, v in result.params.items()},
    ).build()


def _polynomial_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        load_line(dataset)
        + formula_code(
            roles["response"], roles["predictors"], extra='  # añada I(Q("x") ** 2)… para los grados superiores'
        )
        + "print(smf.ols(formula, df).fit().summary())\n"
    )


# ============================================================================ nonlinear least squares
_NONLINEAR = {
    "exponential": ("a * exp(b * x)", ("a", "b"), "Crecimiento exponencial"),
    "decay": ("a * exp(-b * x) + c", ("a", "b", "c"), "Decaimiento exponencial con asíntota"),
    "power": ("a * x ** b", ("a", "b"), "Potencia"),
    "logistic": ("a / (1 + exp(-b * (x - c)))", ("a", "b", "c"), "Crecimiento logístico"),
    "michaelis": ("a * x / (b + x)", ("a", "b"), "Michaelis-Menten (saturación)"),
    "gompertz": ("a * exp(-b * exp(-c * x))", ("a", "b", "c"), "Gompertz"),
    "quadratic_peak": ("a + b * x + c * x ** 2", ("a", "b", "c"), "Parábola"),
    "custom": ("", (), "Fórmula propia"),
}


def _starting_values(model: str, x: np.ndarray, y: np.ndarray) -> list[float]:
    span_y = float(np.ptp(y)) or 1.0
    span_x = float(np.ptp(x)) or 1.0
    top = float(y.max())
    return {
        "exponential": [max(float(y.min()), 1e-3), 1.0 / span_x],
        "decay": [span_y, 3.0 / span_x, float(y.min())],
        "power": [max(top / max(float(np.abs(x).max()) ** 1.0, 1e-9), 1e-3), 1.0],
        "logistic": [top * 1.1, 4.0 / span_x, float(np.median(x))],
        "michaelis": [top * 1.2, float(np.median(x))],
        "gompertz": [top * 1.1, 2.0, 3.0 / span_x],
        "quadratic_peak": [float(y.mean()), 0.0, 0.0],
    }[model]


def _nonlinear(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    from scipy.optimize import curve_fit

    xn, yn, alpha, kind = roles["x"], roles["y"], p["alpha"], p["model"]
    data = pd.DataFrame(
        {"x": pd.to_numeric(frame[xn], errors="coerce"), "y": pd.to_numeric(frame[yn], errors="coerce")}
    ).dropna()
    if len(data) < 6:
        raise InvalidInputError("Se necesitan al menos 6 pares de datos.")
    x, y = data["x"].to_numpy(float), data["y"].to_numpy(float)
    if kind == "custom":
        expression = (p["expression"] or "").strip()
        names = tuple(n.strip() for n in str(p["parameters"] or "").replace(";", ",").split(",") if n.strip())
        if not expression or not names:
            raise InvalidInputError("Escriba la fórmula (con x) y los nombres de los parámetros separados por comas.")
        if len(names) > 6 or "x" in names or len(set(names)) != len(names):
            raise InvalidInputError("Use hasta 6 parámetros con nombres distintos y distintos de «x».")
        evaluate(
            expression, pd.DataFrame({"x": [1.0, 2.0], **{n: [1.0, 1.0] for n in names}})
        )  # reject bad formulas early
    else:
        expression, names, _ = _NONLINEAR[kind]
    holder = pd.DataFrame({"x": x, **{n: np.ones(len(x)) for n in names}})

    def function(values: np.ndarray, *theta: float) -> np.ndarray:
        local = (
            holder
            if len(values) == len(holder)
            else pd.DataFrame({"x": values, **{n: np.ones(len(values)) for n in names}})
        )
        for name, t in zip(names, theta, strict=True):
            local[name] = t
        return evaluate(expression, local).to_numpy(float)

    if p.get("start"):
        try:
            start = [float(v) for v in str(p["start"]).replace(";", ",").split(",") if v.strip()]
        except ValueError:
            raise InvalidInputError("Los valores iniciales deben ser números separados por comas.") from None
        if len(start) != len(names):
            raise InvalidInputError(f"Indique {len(names)} valores iniciales (uno por parámetro: {', '.join(names)}).")
    elif kind == "custom":
        start = [1.0] * len(names)
    else:
        start = _starting_values(kind, x, y)
    try:
        theta, covariance = curve_fit(function, x, y, p0=start, maxfev=20000)
    except (RuntimeError, ValueError) as error:
        raise ComputationError(
            f"El ajuste no convergió ({error}). Pruebe otros valores iniciales o un modelo más sencillo."
        ) from None
    if not np.all(np.isfinite(covariance)):
        raise ComputationError(
            "El ajuste no produjo errores estándar finitos: el modelo está sobreparametrizado o mal condicionado."
        )
    n, k = len(x), len(names)
    fitted = function(x, *theta)
    resid = y - fitted
    sse, sst = float((resid**2).sum()), float(((y - y.mean()) ** 2).sum())
    df = n - k
    se = np.sqrt(np.diag(covariance))
    crit = stats.t.ppf(1 - alpha / 2, df)
    out = Output(
        f"Regresión no lineal: {dict((k_, v[2]) for k_, v in _NONLINEAR.items())[kind]}", f"y = {expression} · n = {n}"
    )
    out.tiles(
        [
            ("Observaciones", n),
            ("R²", 1 - sse / sst),
            ("Error estándar residual", float(np.sqrt(sse / df))),
            ("AIC", n * np.log(sse / n) + 2 * (k + 1)),
        ]
    )
    out.table(
        "Parámetros estimados",
        ["Parámetro", "Estimación", "Error est.", "t", "Valor p", *ci_columns(alpha)],
        [
            [
                name,
                float(t_),
                float(s_),
                float(t_ / s_),
                P(float(2 * stats.t.sf(abs(t_ / s_), df))),
                float(t_ - crit * s_),
                float(t_ + crit * s_),
            ]
            for name, t_, s_ in zip(names, theta, se, strict=True)
        ],
        note="Errores estándar por aproximación lineal (matriz de covarianza del ajuste).",
    )
    grid = np.linspace(x.min(), x.max(), 150)
    curve = function(grid, *theta)
    eps = 1e-6
    jac = np.column_stack([(function(grid, *(theta + eps * np.eye(k)[i])) - curve) / eps for i in range(k)])
    half = crit * np.sqrt(np.einsum("ij,jk,ik->i", jac, covariance, jac))
    out.chart(
        scatter_chart(
            f"{yn} según {xn}",
            xn,
            yn,
            [{"name": "Datos", "x": x, "y": y}],
            lines=[{"name": "Ajuste", "x": grid, "y": curve}],
            bands=[
                {
                    "name": f"IC {100 * (1 - alpha):g} % de la curva",
                    "x": grid,
                    "lower": curve - half,
                    "upper": curve + half,
                }
            ],
        )
    )
    out.chart(
        scatter_chart(
            "Residuos frente a valores ajustados",
            "Valor ajustado",
            "Residuo",
            [{"name": "Residuos", "x": fitted, "y": resid}],
            hlines=[{"value": 0}],
        )
    )
    return out.set(
        n=n, r2=1 - sse / sst, parameters={name: float(t_) for name, t_ in zip(names, theta, strict=True)}
    ).build()


def _nonlinear_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    expression, names, _ = (
        _NONLINEAR[p["model"]]
        if p["model"] != "custom"
        else (p["expression"], [n.strip() for n in str(p["parameters"] or "").split(",")], "")
    )
    python = str(expression).replace("exp(", "np.exp(").replace("^", "**")
    return (
        "import numpy as np\nfrom scipy.optimize import curve_fit\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['x'])}, {py_name(roles['y'])}]].dropna()\n"
        + f"def f(x, {', '.join(names)}):\n    return {python}\n\n"
        + f"theta, cov = curve_fit(f, d[{py_name(roles['x'])}], d[{py_name(roles['y'])}], p0={[1.0] * len(names)!r}, maxfev=20000)\nprint(theta, np.sqrt(np.diag(cov)))\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


_OUT = (PREDICTIONS,)
BINARY = "Respuestas categóricas"
COUNTS = "Conteos y GLM"
LINEAR = "Lineales robustos y flexibles"

SPECS = (
    _spec("binary", "Regresión logística / probit (respuesta binaria)", BINARY, "Odds ratios, ajuste (Hosmer-Lemeshow, R²), tabla de clasificación y curva ROC.", _binary, roles=(BINARY_RESPONSE, PREDICTORS), params=(FieldSpec("link", "Liga", "select", "logit", _LINKS), FieldSpec("event", "Valor que cuenta como evento (vacío = el mayor)", "text", optional=True), FieldSpec("cutoff", "Corte de clasificación", "number", 0.5, minimum=0.01, maximum=0.99), INTERACTIONS, ALPHA), outputs=_OUT, code=_binary_code, example={"roles": {"response": "b", "predictors": ["x1", "x2", "g"]}, "params": {}}),
    _spec("ordinal", "Regresión ordinal", BINARY, "Respuesta con categorías ordenadas (satisfacción 1-5, grados): probabilidades proporcionales con logit o probit.", _ordinal, heavy=True, roles=(RoleSpec("response", "Respuesta ordinal (3 o más categorías)", "any"), PREDICTORS), params=(FieldSpec("distr", "Distribución", "select", "logit", (("logit", "Logit"), ("probit", "Probit"))), INTERACTIONS, ALPHA), outputs=_OUT, code=_ordinal_code, example={"roles": {"response": "r1", "predictors": ["x1", "g"]}, "params": {}}),
    _spec("multinomial", "Regresión logística multinomial", BINARY, "Respuesta nominal con 3 o más categorías sin orden, frente a una categoría de referencia.", _multinomial, heavy=True, roles=(RoleSpec("response", "Respuesta nominal (3 a 12 categorías)", "any"), PREDICTORS), params=(FieldSpec("reference", "Categoría de referencia (vacío = la primera)", "text", optional=True), INTERACTIONS, ALPHA), outputs=_OUT, code=_multinomial_code, example={"roles": {"response": "g", "predictors": ["x1", "x2"]}, "params": {}}),
    _spec("count", "Regresión de conteos (Poisson, cuasi-Poisson, binomial negativa)", COUNTS, "Razones de tasas con exposición opcional y diagnóstico de sobredispersión.", _count, roles=(RoleSpec("response", "Conteo (enteros ≥ 0)", "numeric"), PREDICTORS, RoleSpec("exposure", "Exposición (opcional)", "numeric", optional=True, help="Tiempo, población o tamaño en riesgo: entra como log(exposición).")), params=(FieldSpec("family", "Modelo", "select", "poisson", _COUNT), INTERACTIONS, ALPHA), outputs=_OUT, code=_count_code, example={"roles": {"response": "count", "predictors": ["x1", "x2"]}, "params": {"family": "negbin"}}),
    _spec("glm", "Modelo lineal generalizado (GLM)", COUNTS, "Elija la familia y la liga: normal, binomial, Poisson, gamma, gaussiana inversa o binomial negativa.", _glm, roles=(RoleSpec("response", "Respuesta", "numeric"), PREDICTORS), params=(FieldSpec("family", "Familia", "select", "gamma", _FAMILIES), FieldSpec("link", "Liga", "select", "default", _ALL_LINKS, help="«default» usa la liga canónica de la familia."), FieldSpec("nb_alpha", "α de la binomial negativa", "number", 1.0, minimum=1e-6, visible_if=("family", ("negative_binomial",))), FieldSpec("scale", "Escala", "select", "model", (("model", "Según la familia"), ("pearson", "Estimada con χ² de Pearson (cuasi-verosimilitud)"))), INTERACTIONS, ALPHA), outputs=_OUT, code=_glm_code, example={"roles": {"response": "count", "predictors": ["x1", "g"]}, "params": {"family": "poisson"}}),
    _spec("quantile", "Regresión cuantílica", LINEAR, "Cómo cambia el efecto de los predictores en distintos puntos de la distribución de Y (mediana, colas).", _quantile, roles=(RESPONSE, PREDICTORS), params=(FieldSpec("quantiles", "Cuantiles", "multiselect", ["0.1", "0.5", "0.9"], _QUANTILES), INTERACTIONS, ALPHA), code=_quantile_code, example={"roles": {"response": "y", "predictors": ["x1", "x2"]}, "params": {}}),
    _spec("robust", "Regresión robusta y errores robustos", LINEAR, "Errores HC/HAC para heterocedasticidad y autocorrelación, o estimación M (Huber, Tukey, Hampel) contra atípicos.", _robust, roles=(RESPONSE, PREDICTORS), params=(FieldSpec("method", "Método", "select", "hc3", _ROBUST), FieldSpec("lags", "Rezagos de Newey-West", "integer", 4, minimum=0, maximum=100, visible_if=("method", ("hac",))), INTERACTIONS, ALPHA), outputs=_OUT, code=_robust_code, example={"roles": {"response": "y", "predictors": ["x1", "x2"]}, "params": {"method": "huber"}}),
    _spec("wls", "Mínimos cuadrados ponderados", LINEAR, "Para varianza no constante: pesos propios o estimados, comparado con MCO y con Breusch-Pagan antes y después.", _wls, roles=(RESPONSE, PREDICTORS, RoleSpec("weights", "Pesos (opcional; vacío = estimarlos)", "numeric", optional=True)), params=(INTERACTIONS, ALPHA), outputs=_OUT, code=_wls_code, example={"roles": {"response": "y", "predictors": ["x1", "x2"]}, "params": {}}),
    _spec("polynomial", "Regresión polinómica e interacciones", LINEAR, "Grados superiores e interacciones entre predictores, con prueba F secuencial para elegir el grado y curva con banda de confianza.", _polynomial, roles=(RESPONSE, PREDICTORS), params=(FieldSpec("degree", "Grado máximo", "integer", 2, minimum=1, maximum=6), INTERACTIONS, ALPHA), outputs=_OUT, code=_polynomial_code, example={"roles": {"response": "y", "predictors": ["x1", "g"]}, "params": {"degree": 2, "interactions": "two"}}),
    _spec("nonlinear", "Regresión no lineal", LINEAR, "Ajusta curvas de crecimiento, saturación o la fórmula que usted escriba, con errores estándar y banda de confianza.", _nonlinear, roles=(RoleSpec("x", "Variable X", "numeric"), RoleSpec("y", "Respuesta Y", "numeric")), params=(FieldSpec("model", "Modelo", "select", "michaelis", tuple((k, v[2]) for k, v in _NONLINEAR.items())), FieldSpec("expression", "Fórmula con x y los parámetros", "text", visible_if=("model", ("custom",)), help="Ejemplo: a * x ** b + c"), FieldSpec("parameters", "Nombres de los parámetros", "text", visible_if=("model", ("custom",)), help="Separados por comas: a, b, c"), FieldSpec("start", "Valores iniciales (opcional)", "text", optional=True, help="Uno por parámetro, separados por comas; vacío = automáticos."), ALPHA), code=_nonlinear_code, example={"roles": {"x": "x3", "y": "x1"}, "params": {"model": "quadratic_peak"}}),
)  # fmt: skip

registry.register(*SPECS)
