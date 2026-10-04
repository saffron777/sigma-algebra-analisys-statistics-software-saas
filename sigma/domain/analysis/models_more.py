"""Model selection, regularisation, instrumental variables, mixed models and specification tests."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, enet_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats import diagnostic
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart, scatter_chart
from sigma.domain.analysis.modeling import (
    ALPHA,
    PREDICTIONS,
    PREDICTORS,
    RESPONSE,
    Design,
    build_design,
    ci_columns,
    coefficient_table,
    dropped_note,
    formula_code,
    need_enough,
    prediction_frame,
)
from sigma.domain.analysis.models import FIT_ERRORS, _fit
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import ComputationError, InvalidInputError

MODULE = "modelos"
SELECTION = "Selección y regularización"
ADVANCED = "Endogeneidad, efectos mixtos y diagnóstico"
MAX_SUBSET_PREDICTORS = 15


def _term(design: Design, token: str) -> str:
    return f"C({token})" if token in design.factors else token


def _ols(design: Design, terms: list[str]):
    rhs = " + ".join(terms) if terms else "1"
    return smf.ols(f"{design.response} ~ {rhs}", design.data).fit()


# ============================================================================ stepwise
def _stepwise(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, direction, criterion = roles["response"], roles["predictors"], p["direction"], p["criterion"]
    design = build_design(frame, response, predictors)
    tokens = design.numeric + design.factors
    names = {t: design.names[t] for t in tokens}
    n = len(design.data)
    enter, leave = p["alpha_enter"], p["alpha_remove"]
    current = list(tokens) if direction == "backward" else []

    def score(fit) -> float:
        return float(fit.aic if criterion == "aic" else fit.bic)

    def partial_p(bigger, smaller) -> float:
        return float(anova_lm(smaller, bigger).iloc[1]["Pr(>F)"])

    fit = _ols(design, [_term(design, t) for t in current])
    steps = [["Inicio", "—", "—", float(fit.rsquared_adj), score(fit) if criterion != "pvalue" else None]]
    for step in range(1, p["max_steps"] + 1):
        moved = False
        if direction in ("forward", "both"):
            best = None
            for token in (t for t in tokens if t not in current):
                trial = _ols(design, [_term(design, t) for t in [*current, token]])
                value = partial_p(trial, fit) if criterion == "pvalue" else score(trial)
                ok = value < enter if criterion == "pvalue" else value < score(fit) - 1e-9
                if ok and (best is None or value < best[0]):
                    best = (value, token, trial)
            if best:
                current.append(best[1])
                fit, moved = best[2], True
                steps.append(
                    [
                        f"Paso {step}",
                        "Entra",
                        names[best[1]],
                        float(fit.rsquared_adj),
                        best[0] if criterion == "pvalue" else score(fit),
                    ]
                )
        if not moved and direction in ("backward", "both") and current:
            worst = None
            for token in current:
                trial = _ols(design, [_term(design, t) for t in current if t != token])
                value = partial_p(fit, trial) if criterion == "pvalue" else score(trial)
                ok = value > leave if criterion == "pvalue" else value < score(fit) - 1e-9
                if ok and (worst is None or (value > worst[0] if criterion == "pvalue" else value < worst[0])):
                    worst = (value, token, trial)
            if worst:
                current.remove(worst[1])
                fit, moved = worst[2], True
                steps.append(
                    [
                        f"Paso {step}",
                        "Sale",
                        names[worst[1]],
                        float(fit.rsquared_adj),
                        worst[0] if criterion == "pvalue" else score(fit),
                    ]
                )
        if not moved:
            break
    full = _ols(design, [_term(design, t) for t in tokens])
    need_enough(design, len(full.params) + 1)
    label = {"pvalue": "valor p del F parcial", "aic": "AIC", "bic": "BIC"}[criterion]
    out = Output(
        f"Regresión paso a paso: {response}",
        f"{ {'forward': 'hacia adelante', 'backward': 'hacia atrás', 'both': 'en ambas direcciones'}[direction] } · criterio {label} · n = {n}",
    )
    dropped_note(out, design)
    out.tiles(
        [
            ("Predictores candidatos", len(tokens)),
            ("Predictores en el modelo", len(current)),
            ("R² ajustado final", float(fit.rsquared_adj)),
            ("R² ajustado completo", float(full.rsquared_adj)),
        ]
    )
    out.table("Historia de la selección", ["Paso", "Acción", "Término", "R² ajustado", label.capitalize()], steps)
    if not current:
        out.alert(
            "warning", "Ningún predictor cumplió el criterio de entrada: el modelo final solo tiene la constante."
        )
    coefficient_table(out, design, fit, p["alpha"], stat_label="t", title="Modelo final")
    out.table(
        "Comparación del modelo final con el completo",
        ["Modelo", "Predictores", "R² ajustado", "AIC", "BIC", "Error estándar residual"],
        [
            ["Final", len(current), float(fit.rsquared_adj), float(fit.aic), float(fit.bic), float(np.sqrt(fit.scale))],
            [
                "Completo",
                len(tokens),
                float(full.rsquared_adj),
                float(full.aic),
                float(full.bic),
                float(np.sqrt(full.scale)),
            ],
        ],
        note="Los valores p y los intervalos del modelo elegido por pasos son demasiado optimistas (no descuentan la selección): valide con datos nuevos.",
    )
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(frame, design, ajustado=np.asarray(fit.fittedvalues), residuo=np.asarray(fit.resid)),
    )
    return out.set(selected=[names[t] for t in current], r2_adj=float(fit.rsquared_adj), n=n).build()


def _stepwise_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "# Selección paso a paso: statsmodels no la trae hecha; este es el esquema hacia adelante por AIC\n"
        + load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + "print(smf.ols(formula, df).fit().summary())  # reajuste con los términos elegidos\n"
    )


# ============================================================================ best subsets
def _subset_stats(xtx: np.ndarray, xty: np.ndarray, yty: float, idx: tuple[int, ...]) -> float:
    if not idx:
        return yty
    sub = list(idx)
    beta = np.linalg.solve(xtx[np.ix_(sub, sub)], xty[sub])
    return float(yty - beta @ xty[sub])


def _best_subsets(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors = roles["response"], roles["predictors"]
    if len(predictors) > MAX_SUBSET_PREDICTORS:
        raise InvalidInputError(
            f"Los mejores subconjuntos admiten hasta {MAX_SUBSET_PREDICTORS} predictores (hay {len(predictors)})."
        )
    data = frame[[response, *predictors]].apply(pd.to_numeric, errors="coerce").dropna()
    n, k = len(data), len(predictors)
    if n <= k + 2:
        raise InvalidInputError(f"Se necesitan más de {k + 2} filas completas (hay {n}).")
    y = data[response].to_numpy(float)
    x = data[predictors].to_numpy(float)
    yc, xc = y - y.mean(), x - x.mean(axis=0)
    if (xc.std(axis=0) == 0).any():
        raise InvalidInputError("Algún predictor es constante.")
    xtx, xty, sst = xc.T @ xc, xc.T @ yc, float(yc @ yc)
    mse_full = _subset_stats(xtx, xty, sst, tuple(range(k))) / (n - k - 1)
    keep = int(p["per_size"])
    rows: list[dict[str, Any]] = []
    for size in range(1, k + 1):
        results = []
        for subset in combinations(range(k), size):
            try:
                sse = _subset_stats(xtx, xty, sst, subset)
            except np.linalg.LinAlgError:
                continue
            results.append((sse, subset))
        results.sort(key=lambda r: r[0])
        for sse, subset in results[:keep]:
            q = size + 1
            rows.append(
                {
                    "size": size,
                    "subset": subset,
                    "r2": 1 - sse / sst,
                    "r2adj": 1 - (sse / (n - q)) / (sst / (n - 1)),
                    "cp": sse / mse_full - (n - 2 * q),
                    "aic": n * np.log(sse / n) + 2 * q,
                    "bic": n * np.log(sse / n) + q * np.log(n),
                    "s": float(np.sqrt(sse / (n - q))),
                }
            )
    best_bic = min(rows, key=lambda r: r["bic"])
    best_adj = max(rows, key=lambda r: r["r2adj"])
    out = Output(
        f"Mejores subconjuntos: {response}",
        f"{k} predictores · {2**k - 1:,} modelos evaluados · n = {n}".replace(",", " "),
    )
    out.tiles(
        [
            ("Predictores", k),
            ("Mejor por BIC", ", ".join(predictors[i] for i in best_bic["subset"])),
            ("Mejor R² ajustado", ", ".join(predictors[i] for i in best_adj["subset"])),
        ]
    )
    head = ["Predictores", "R²", "R² ajustado", "Cp de Mallows", "AIC", "BIC", "S", *predictors]
    table = [
        [
            r["size"],
            r["r2"],
            r["r2adj"],
            r["cp"],
            r["aic"],
            r["bic"],
            r["s"],
            *("●" if i in r["subset"] else "" for i in range(k)),
        ]
        for r in rows
    ]
    out.table(
        "Mejores modelos de cada tamaño",
        head,
        table,
        note="Elija modelos con Cp cercano al número de parámetros (predictores + 1), BIC/AIC bajos y R² ajustado alto.",
    )
    top = [next(r for r in rows if r["size"] == size) for size in range(1, k + 1)]  # the best model of each size
    out.chart(
        line_chart(
            "R² ajustado del mejor modelo de cada tamaño",
            "Número de predictores",
            "R² ajustado",
            [{"name": "R² ajustado", "x": [r["size"] for r in top], "y": [r["r2adj"] for r in top], "markers": True}],
        )
    )
    out.chart(
        line_chart(
            "BIC del mejor modelo de cada tamaño",
            "Número de predictores",
            "BIC",
            [{"name": "BIC", "x": [r["size"] for r in top], "y": [r["bic"] for r in top], "markers": True}],
        )
    )
    return out.set(
        best_bic=[predictors[i] for i in best_bic["subset"]],
        best_adj_r2=[predictors[i] for i in best_adj["subset"]],
        n=n,
    ).build()


def _best_subsets_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from itertools import combinations\nimport numpy as np\nimport statsmodels.api as sm\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['response'])}, *{roles['predictors']!r}]].dropna()\nbest = min(((sm.OLS(d[{py_name(roles['response'])}], sm.add_constant(d[list(c)])).fit().bic, c) for r in range(1, {len(roles['predictors'])} + 1) for c in combinations({roles['predictors']!r}, r)))\nprint(best)\n"
    )


# ============================================================================ ridge, lasso, elastic net
_PENALTIES = (
    ("lasso", "Lasso (L1): elimina predictores"),
    ("ridge", "Ridge (L2): reduce sin eliminar"),
    ("elastic", "Red elástica (L1 + L2)"),
)


def _cv_path_model(cls, grid: int, **kwargs):
    """scikit-learn renamed ``n_alphas`` to ``alphas`` (an int) in recent versions."""
    try:
        return cls(n_alphas=grid, **kwargs)
    except TypeError:
        return cls(alphas=grid, **kwargs)


def _ridge_cv(z: np.ndarray, y: np.ndarray, cv: KFold) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge over a grid of penalties: (alphas, mean CV mse, sd of the fold mse)."""
    alphas = np.logspace(-3, 3, 60)
    errors = np.zeros((len(alphas), cv.get_n_splits()))
    for f, (train, test) in enumerate(cv.split(z)):
        centre = y[train].mean()
        gram, target = z[train].T @ z[train], z[train].T @ (y[train] - centre)
        for i, a in enumerate(alphas):
            w = np.linalg.solve(gram + a * np.eye(z.shape[1]), target)
            errors[i, f] = float(((y[test] - centre - z[test] @ w) ** 2).mean())
    return alphas, errors.mean(axis=1), errors.std(axis=1)


def _regularized(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, kind = roles["response"], roles["predictors"], p["penalty"]
    design = build_design(frame, response, predictors)
    rhs = design.formula.split("~", 1)[1]
    matrix = patsy.dmatrix(rhs, design.data, return_type="dataframe").drop(columns="Intercept", errors="ignore")
    n, k = matrix.shape
    if n < 20:
        raise InvalidInputError("La regresión regularizada necesita al menos 20 filas completas.")
    folds = int(p["folds"])
    if folds > n // 2:
        raise InvalidInputError("Hay demasiados pliegues para tan pocas filas.")
    y = design.data[design.response].to_numpy(float)
    scaler = StandardScaler().fit(matrix.to_numpy(float))
    z = scaler.transform(matrix.to_numpy(float))
    cv = KFold(folds, shuffle=True, random_state=int(p["seed"]))
    seed = int(p["seed"])
    if kind == "ridge":
        cv_alphas, cv_mse, cv_sd = _ridge_cv(z, y, cv)
        pick = int(cv_mse.argmin())
        if p["rule"] == "one_se":  # the strongest penalty whose error is within one standard error of the best
            within = np.flatnonzero(cv_mse <= cv_mse[pick] + cv_sd[pick] / np.sqrt(folds))
            pick = int(within.max())
        chosen = float(cv_alphas[pick])
        weights = np.linalg.solve(z.T @ z + chosen * np.eye(k), z.T @ (y - y.mean()))
        intercept_z, l1 = float(y.mean()), 0.0
        path_alphas = np.logspace(-3, 3, 50)
        paths = np.array([np.linalg.solve(z.T @ z + a * np.eye(k), z.T @ (y - y.mean())) for a in path_alphas]).T
        best_mse = float(cv_mse.min())
        curve = (cv_alphas, np.sqrt(cv_mse), np.sqrt(cv_mse + cv_sd) - np.sqrt(cv_mse))
    else:
        if kind == "lasso":
            model = _cv_path_model(LassoCV, 80, cv=cv, random_state=seed, max_iter=20000).fit(z, y)
            cv_alphas, mse_path, l1 = model.alphas_, model.mse_path_, 1.0
        else:
            model = _cv_path_model(
                ElasticNetCV, 60, cv=cv, l1_ratio=[0.2, 0.5, 0.8, 0.95], random_state=seed, max_iter=20000
            ).fit(z, y)
            i = int(np.argmin(np.abs(np.asarray(model.l1_ratio) - model.l1_ratio_)))
            cv_alphas, mse_path, l1 = model.alphas_[i], model.mse_path_[i], float(model.l1_ratio_)
        chosen, weights, intercept_z = float(model.alpha_), model.coef_, float(model.intercept_)
        if p["rule"] == "one_se":
            mean_cv = mse_path.mean(axis=1)
            best = int(mean_cv.argmin())
            within = np.flatnonzero(mean_cv <= mean_cv[best] + mse_path.std(axis=1)[best] / np.sqrt(folds))
            chosen = float(cv_alphas[within].max())
            refit = (
                Lasso(alpha=chosen, max_iter=50000)
                if kind == "lasso"
                else ElasticNet(alpha=chosen, l1_ratio=l1, max_iter=50000)
            ).fit(z, y)
            weights, intercept_z = refit.coef_, float(refit.intercept_)
        top = float(np.abs(z.T @ (y - y.mean())).max() / (n * max(l1, 1e-3)))
        path_alphas, paths, _ = enet_path(z, y - y.mean(), l1_ratio=l1, alphas=np.geomspace(top, top * 1e-3, 60))
        mean_mse = mse_path.mean(axis=1)
        best_mse = float(mean_mse.min())
        curve = (cv_alphas, np.sqrt(mean_mse), np.sqrt(mean_mse + mse_path.std(axis=1)) - np.sqrt(mean_mse))
    scale = scaler.scale_
    coef = weights / scale
    intercept = float(intercept_z - (weights * scaler.mean_ / scale).sum())
    fitted = z @ weights + intercept_z
    r2 = 1 - float(((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    ols = sm.OLS(y, sm.add_constant(matrix.to_numpy(float))).fit()
    out = Output(
        f"Regresión regularizada ({dict(_PENALTIES)[kind].split(' (')[0]}): {response}",
        f"validación cruzada de {folds} pliegues · n = {n} · {k} términos",
    )
    dropped_note(out, design)
    zeros = int((np.abs(coef) < 1e-10).sum())
    out.tiles(
        [
            ("λ elegido", chosen),
            ("R² (entrenamiento)", r2),
            ("RMSE de validación cruzada", float(np.sqrt(best_mse))),
            ("Coeficientes en cero", zeros),
        ]
    )
    rows = [["Constante", intercept, float(ols.params[0])]] + [
        [design.pretty(str(name)), float(c), float(o)]
        for name, c, o in zip(matrix.columns, coef, ols.params[1:], strict=True)
    ]
    note = (
        "Las columnas se estandarizaron para penalizar por igual; los coeficientes se devuelven a su escala original."
    )
    out.table(
        "Coeficientes (en la escala original)",
        ["Término", "Regularizado", "MCO (sin penalizar)"],
        rows,
        note=note + (f" Mezcla L1 elegida: {l1:g}." if kind == "elastic" else ""),
    )
    shown = [(design.pretty(str(name)), paths[i]) for i, name in enumerate(matrix.columns)][:12]
    out.chart(
        line_chart(
            "Trayectoria de los coeficientes estandarizados",
            "log₁₀(λ)",
            "Coeficiente",
            [{"name": name, "x": np.log10(path_alphas), "y": path} for name, path in shown],
            vlines=[{"value": float(np.log10(chosen)), "label": "λ elegido"}],
        )
    )
    order = np.argsort(curve[0])
    out.chart(
        line_chart(
            "Error de validación cruzada",
            "log₁₀(λ)",
            "RMSE",
            [{"name": "RMSE medio", "x": np.log10(curve[0][order]), "y": curve[1][order]}],
            bands=[
                {
                    "name": "+ 1 desv. est.",
                    "x": np.log10(curve[0][order]),
                    "lower": curve[1][order],
                    "upper": (curve[1] + curve[2])[order],
                }
            ],
            vlines=[{"value": float(np.log10(chosen)), "label": "λ elegido"}],
        )
    )
    out.dataset("predicciones", PREDICTIONS.label, prediction_frame(frame, design, ajustado=fitted, residuo=y - fitted))
    return out.set(
        lambda_=chosen,
        l1_ratio=l1,
        r2=r2,
        nonzero=int(k - zeros),
        n=n,
        coefficients={design.pretty(str(name)): float(c) for name, c in zip(matrix.columns, coef, strict=True)},
    ).build()


def _regularized_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    model = {
        "lasso": "LassoCV(cv=5)",
        "ridge": "RidgeCV(alphas=np.logspace(-3, 3, 60), cv=5)",
        "elastic": "ElasticNetCV(cv=5, l1_ratio=[0.2, 0.5, 0.8, 0.95])",
    }[p["penalty"]]
    return (
        "import numpy as np\nimport pandas as pd\nfrom sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV\nfrom sklearn.preprocessing import StandardScaler\n\n"
        + load_line(dataset)
        + f"d = df[[{py_name(roles['response'])}, *{roles['predictors']!r}]].dropna()\nX = pd.get_dummies(d[{roles['predictors']!r}], drop_first=True).astype(float)\n"
        + f"Z = StandardScaler().fit_transform(X)\nmodel = {model}.fit(Z, d[{py_name(roles['response'])}])\nprint(dict(zip(X.columns, model.coef_)))\n"
    )


# ============================================================================ instrumental variables
def _iv(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    from linearmodels.iv import IV2SLS

    response, endogenous, instruments = roles["response"], roles["endogenous"], roles["instruments"]
    exogenous = roles.get("exogenous", [])
    if len(instruments) < len(endogenous):
        raise InvalidInputError(
            f"Se necesitan al menos tantos instrumentos ({len(instruments)}) como variables endógenas ({len(endogenous)})."
        )
    if set(endogenous) & set(instruments) or set(endogenous) & set(exogenous) or set(instruments) & set(exogenous):
        raise InvalidInputError("Una columna no puede tener dos funciones (endógena, exógena o instrumento).")
    design = build_design(frame, response, [*endogenous, *exogenous, *instruments])
    token = {name: tok for tok, name in design.names.items()}
    data = design.data
    y = data[token[response]]
    exog_names = [token[c] for c in exogenous]
    exog = patsy.dmatrix(
        " + ".join(_term(design, t) for t in exog_names) if exog_names else "1", data, return_type="dataframe"
    )
    endog = data[[token[c] for c in endogenous]]
    instr = data[[token[c] for c in instruments]]
    n = len(data)
    cov = {"unadjusted": "unadjusted", "robust": "robust"}[p["cov"]]
    try:
        result = IV2SLS(y, exog, endog, instr).fit(cov_type=cov)
    except FIT_ERRORS + (ValueError, KeyError) as error:
        raise ComputationError(f"No se pudo ajustar el modelo de variables instrumentales: {error}") from None
    ols = sm.OLS(y, pd.concat([exog, endog], axis=1)).fit()
    out = Output(
        f"Variables instrumentales (2MCO): {response}",
        f"endógenas: {', '.join(endogenous)} · instrumentos: {', '.join(instruments)} · n = {n}",
    )
    dropped_note(out, design)
    first = result.first_stage.diagnostics
    weakest = float(first["f.stat"].min())
    out.tiles(
        [
            ("Observaciones", n),
            ("R²", float(result.rsquared)),
            ("F de la primera etapa (mínimo)", weakest),
            ("Sobreidentificación", "sí" if len(instruments) > len(endogenous) else "no"),
        ]
    )
    if weakest < 10:
        out.alert(
            "warning",
            f"Instrumentos débiles: el F de la primera etapa ({weakest:.1f}) es menor que 10. Las estimaciones por variables instrumentales pueden estar muy sesgadas e imprecisas.",
        )
    conf = result.conf_int(level=1 - p["alpha"])
    rows = []
    for term in result.params.index:
        name = "Constante" if term == "Intercept" else design.pretty(str(term))
        rows.append(
            [
                name + (" (endógena)" if term in endog.columns else ""),
                float(result.params[term]),
                float(result.std_errors[term]),
                float(result.tstats[term]),
                P(float(result.pvalues[term])),
                float(conf.loc[term].iloc[0]),
                float(conf.loc[term].iloc[1]),
                float(ols.params[term]) if term in ols.params.index else None,
            ]
        )
    out.table(
        "Estimaciones de 2MCO",
        ["Término", "Coeficiente", "Error est.", "t", "Valor p", *ci_columns(p["alpha"]), "Coef. MCO"],
        rows,
        note="La última columna es la estimación por MCO ignorando la endogeneidad: si difiere mucho, la corrección importa.",
    )
    out.table(
        "Primera etapa por variable endógena",
        ["Variable", "R² parcial", "F", "Valor p"],
        [
            [design.pretty(str(name)), float(row["partial.rsquared"]), float(row["f.stat"]), P(float(row["f.pval"]))]
            for name, row in first.iterrows()
        ],
    )
    tests = []
    wu = result.wu_hausman()
    tests.append(["Wu-Hausman (¿son endógenas?)", float(wu.stat), P(float(wu.pval))])
    if len(instruments) > len(endogenous):
        sargan = result.sargan
        tests.append(
            ["Sargan (sobreidentificación: ¿instrumentos válidos?)", float(sargan.stat), P(float(sargan.pval))]
        )
    out.table(
        "Pruebas de especificación",
        ["Prueba", "Estadístico", "Valor p"],
        tests,
        note="Wu-Hausman con valor p pequeño: la variable es endógena y MCO sería inconsistente. Sargan con valor p pequeño: algún instrumento no es válido.",
    )
    return out.set(
        n=n,
        coefficients={design.pretty(str(t)): float(v) for t, v in result.params.items()},
        first_stage_f=weakest,
        wu_hausman_p=float(wu.pval),
    ).build()


def _iv_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    controls = " + ".join(roles.get("exogenous", [])) or "1"
    endog, instr = " + ".join(roles["endogenous"]), " + ".join(roles["instruments"])
    formula = (
        f"{roles['response']} ~ 1 + {controls} + [{endog} ~ {instr}]"
        if controls != "1"
        else f"{roles['response']} ~ 1 + [{endog} ~ {instr}]"
    )
    return (
        "from linearmodels.iv import IV2SLS\n\n"
        + load_line(dataset)
        + f"print(IV2SLS.from_formula({formula!r}, df.dropna()).fit(cov_type={p['cov']!r}))\n"
    )


# ============================================================================ mixed effects
def _mixed(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, group, slope, alpha = (
        roles["response"],
        roles["predictors"],
        roles["group"],
        roles.get("slope"),
        p["alpha"],
    )
    # the grouping column is kept apart from the fixed effects
    combined = frame[list(dict.fromkeys([response, *predictors, group, *([slope] if slope else [])]))].dropna()
    base = build_design(combined, response, predictors)
    groups = combined.loc[base.used, group].astype(str)
    counts = groups.value_counts()
    if len(counts) < 3:
        raise InvalidInputError(f"«{group}» necesita al menos 3 grupos (tiene {len(counts)}).")
    if counts.max() < 2:
        raise InvalidInputError("Cada grupo debe tener varias observaciones.")
    n = len(base.data)
    data = base.data.assign(_g=groups.to_numpy())
    random = None
    if slope:
        data["_s"] = pd.to_numeric(combined.loc[base.used, slope], errors="coerce").to_numpy(float)
        random = "~ _s"
    try:
        model = smf.mixedlm(base.formula, data, groups=data["_g"], re_formula=random)
        result = model.fit(reml=p["method"] == "reml", method="lbfgs")
    except FIT_ERRORS as error:
        raise ComputationError(f"No se pudo ajustar el modelo mixto: {error}") from None
    if not np.isfinite(np.asarray(result.bse_fe)).all() or not np.isfinite(np.asarray(result.cov_re)).all():
        raise ComputationError(
            "La estimación del modelo mixto no es estable: hay pocos grupos o casi no hay variación entre ellos. "
            "Pruebe sin pendiente aleatoria o con una regresión ordinaria."
        )
    ols = smf.ols(base.formula, base.data).fit()
    var_group = float(result.cov_re.iloc[0, 0])
    resid_var = float(result.scale)
    icc = var_group / (var_group + resid_var) if var_group + resid_var > 0 else 0.0
    out = Output(
        f"Modelo lineal mixto: {response}",
        f"intercepto aleatorio por «{group}»"
        + (f" y pendiente aleatoria de {slope}" if slope else "")
        + f" · {len(counts)} grupos · n = {n}",
    )
    dropped_note(out, base)
    out.tiles(
        [
            ("Observaciones", n),
            ("Grupos", len(counts)),
            ("CCI (correlación intraclase)", icc),
            ("Varianza residual", resid_var),
        ]
    )
    fixed = result.fe_params
    se, pv = result.bse_fe, result.pvalues[fixed.index]
    z = stats.norm.ppf(1 - alpha / 2)
    rows = [
        [
            "Constante" if t == "Intercept" else base.pretty(str(t)),
            float(fixed[t]),
            float(se[t]),
            float(fixed[t] / se[t]),
            P(float(pv[t])),
            float(fixed[t] - z * se[t]),
            float(fixed[t] + z * se[t]),
            float(ols.params[t]),
            float(ols.bse[t]),
        ]
        for t in fixed.index
    ]
    out.table(
        "Efectos fijos",
        ["Término", "Coeficiente", "Error est.", "z", "Valor p", *ci_columns(alpha), "Coef. MCO", "EE MCO"],
        rows,
        note="Los errores estándar de MCO ignoran el agrupamiento y suelen ser demasiado pequeños.",
    )
    components = [
        [f"Varianza entre grupos (intercepto de «{group}»)", var_group],
        ["Varianza residual", resid_var],
        ["Correlación intraclase (CCI)", icc],
    ]
    if slope:
        components.insert(1, [f"Varianza de la pendiente de {slope}", float(result.cov_re.iloc[1, 1])])
    out.table(
        "Componentes de la varianza",
        ["Componente", "Valor"],
        components,
        note="CCI = proporción de la variabilidad que se debe a diferencias entre grupos.",
    )
    ll_ols = float(sm.OLS(base.data[base.response], patsy.dmatrix(base.formula.split("~", 1)[1], base.data)).fit().llf)
    if p["method"] == "ml" or not slope:
        ml = smf.mixedlm(base.formula, data, groups=data["_g"]).fit(reml=False, method="lbfgs")
        lr = max(2 * (float(ml.llf) - ll_ols), 0.0)
        out.table(
            "¿Hace falta el efecto aleatorio?",
            ["Prueba", "χ²", "Valor p"],
            [["Razón de verosimilitud (intercepto aleatorio frente a MCO)", lr, P(float(0.5 * stats.chi2.sf(lr, 1)))]],
            note="Valor p con la mezcla 50:50 de χ² (la varianza está en la frontera del espacio de parámetros).",
        )
    effects = result.random_effects
    blup = pd.DataFrame(
        {
            "grupo": list(effects.keys()),
            "efecto": [float(v.iloc[0]) for v in effects.values()],
            "n": [int(counts[g]) for g in effects],
        }
    ).sort_values("efecto", ascending=False)
    out.table(
        "Efectos aleatorios (BLUP) más extremos",
        ["Grupo", "Efecto sobre el intercepto", "n"],
        [[r.grupo, r.efecto, r.n] for r in pd.concat([blup.head(5), blup.tail(5)]).drop_duplicates().itertuples()],
    )
    fitted = np.asarray(result.fittedvalues)
    out.chart(
        scatter_chart(
            "Residuos frente a valores ajustados",
            "Valor ajustado",
            "Residuo",
            [{"name": "Residuos", "x": fitted, "y": np.asarray(result.resid)}],
            hlines=[{"value": 0}],
        )
    )
    out.dataset(
        "predicciones",
        PREDICTIONS.label,
        prediction_frame(frame, base, ajustado=fitted, residuo=np.asarray(result.resid)),
    )
    return out.set(
        n=n,
        groups=len(counts),
        icc=icc,
        var_group=var_group,
        resid_var=resid_var,
        coefficients={base.pretty(str(t)): float(v) for t, v in fixed.items()},
    ).build()


def _mixed_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    random = f", re_formula='~ Q(\"{roles['slope']}\")'" if roles.get("slope") else ""
    return (
        load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + f"d = df.dropna(subset=[{py_name(roles['response'])}, {py_name(roles['group'])}, *predictors])\nmodel = smf.mixedlm(formula, d, groups=d[{py_name(roles['group'])}]{random}).fit(reml={p['method'] == 'reml'})\nprint(model.summary())\n"
    )


# ============================================================================ specification tests
_TESTS = (
    ("breusch_pagan", "Breusch-Pagan (heterocedasticidad)"),
    ("white", "White (heterocedasticidad general)"),
    ("reset", "RESET de Ramsey (forma funcional)"),
    ("harvey_collier", "Harvey-Collier (linealidad)"),
    ("durbin_watson", "Durbin-Watson (autocorrelación de primer orden)"),
    ("breusch_godfrey", "Breusch-Godfrey (autocorrelación de orden superior)"),
    ("jarque_bera", "Jarque-Bera (normalidad de los residuos)"),
    ("shapiro", "Shapiro-Wilk (normalidad de los residuos)"),
    ("cusum", "CUSUM (estabilidad de los parámetros)"),
    ("chow", "Chow (cambio estructural en un punto)"),
)


def chow_test(y: np.ndarray, x: np.ndarray, split: int) -> tuple[float, float, int, int]:
    """Chow F test for a structural break after row ``split``: (F, p, df1, df2)."""
    k = x.shape[1]
    n = len(y)
    if split < k + 1 or n - split < k + 1:
        raise InvalidInputError(f"Cada tramo necesita más de {k} filas para estimar el modelo.")
    rss = [float(sm.OLS(y[s], x[s]).fit().ssr) for s in (slice(None), slice(0, split), slice(split, None))]
    f = ((rss[0] - rss[1] - rss[2]) / k) / ((rss[1] + rss[2]) / (n - 2 * k))
    return float(f), float(stats.f.sf(f, k, n - 2 * k)), k, n - 2 * k


def _diagnostics(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    response, predictors, chosen, alpha = roles["response"], roles["predictors"], p["tests"], p["alpha"]
    if not chosen:
        raise InvalidInputError("Elija al menos una prueba.")
    work = frame.sort_values(roles["order"], kind="stable") if roles.get("order") else frame
    design = build_design(work, response, predictors)
    result = _fit(smf.ols(design.formula, design.data))
    n = len(design.data)
    need_enough(design, len(result.params) + 2)
    exog = np.asarray(result.model.exog)
    resid = np.asarray(result.resid)
    k = exog.shape[1]
    rows: list[list[Any]] = []

    def add(label: str, statistic: float, pvalue: float | None, ok_when_high_p: bool = True, detail: str = "") -> None:
        if pvalue is None:
            rows.append([label, statistic, None, detail or "—"])
            return
        problem = pvalue < alpha
        verdict = (
            ("posible problema" if problem else "sin evidencia de problema")
            if ok_when_high_p
            else ("rechaza" if problem else "no rechaza")
        )
        rows.append([label, statistic, P(pvalue), f"{verdict}{(' · ' + detail) if detail else ''}"])

    for key in chosen:
        label = dict(_TESTS)[key]
        if key == "breusch_pagan":
            stat, pv, *_ = diagnostic.het_breuschpagan(resid, exog)
            add(label, float(stat), float(pv))
        elif key == "white":
            if k > 20:
                add(label, float("nan"), None, detail="omitida: demasiados regresores")
                continue
            stat, pv, *_ = diagnostic.het_white(resid, exog)
            add(label, float(stat), float(pv))
        elif key == "reset":
            test = diagnostic.linear_reset(result, power=int(p["reset_power"]), use_f=True)
            add(f"{label}, potencias 2 a {int(p['reset_power'])}", float(test.statistic), float(test.pvalue))
        elif key == "harvey_collier":
            stat, pv = diagnostic.linear_harvey_collier(result)
            add(label, float(stat), float(pv))
        elif key == "durbin_watson":
            dw = float(durbin_watson(resid))
            add(label, dw, None, detail="cerca de 2: sin autocorrelación; hacia 0: positiva; hacia 4: negativa")
        elif key == "breusch_godfrey":
            lags = int(p["lags"])
            stat, pv, fstat, fpv = diagnostic.acorr_breusch_godfrey(result, nlags=lags, result_object=False)
            add(f"{label}, {lags} rezagos", float(stat), float(pv))
        elif key == "jarque_bera":
            stat, pv, *_ = jarque_bera(resid)
            add(label, float(stat), float(pv))
        elif key == "shapiro":
            sample = resid if n <= 5000 else np.random.default_rng(1).choice(resid, 5000, replace=False)
            test = stats.shapiro(sample)
            add(label, float(test.statistic), float(test.pvalue))
        elif key == "cusum":
            stat, pv, _ = diagnostic.breaks_cusumolsresid(resid, ddof=k)
            add(label, float(stat), float(pv))
        elif key == "chow":
            split = int(round(float(p["break_at"]) * n)) if 0 < float(p["break_at"]) < 1 else int(p["break_at"])
            f, pv, df1, df2 = chow_test(np.asarray(design.data[design.response]), exog, split)
            add(f"{label} tras la fila {split} (F con {df1}, {df2} gl)", f, pv)
    out = Output(
        f"Pruebas de especificación del modelo lineal: {response}",
        f"{len(chosen)} pruebas · n = {n}" + (f" · filas ordenadas por {roles['order']}" if roles.get("order") else ""),
    )
    dropped_note(out, design)
    problems = sum(1 for r in rows if isinstance(r[2], P) and float(r[2]) < alpha)
    out.tiles(
        [
            ("Observaciones", n),
            ("Pruebas", len(rows)),
            ("Con posible problema", problems),
            ("R²", float(result.rsquared)),
        ]
    )
    out.table(
        "Resultados",
        ["Prueba", "Estadístico", "Valor p", "Lectura"],
        rows,
        note="Un valor p pequeño en estas pruebas señala un incumplimiento del supuesto (no «rechazar» es lo deseable).",
    )
    fitted = np.asarray(result.fittedvalues)
    standardized = resid / resid.std(ddof=k)
    out.chart(
        scatter_chart(
            "Residuos estandarizados frente a ajustados",
            "Valor ajustado",
            "Residuo estandarizado",
            [{"name": "Residuos", "x": fitted, "y": standardized}],
            hlines=[{"value": 0}, {"value": 2, "dash": "dot"}, {"value": -2, "dash": "dot"}],
        )
    )
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    out.chart(
        scatter_chart(
            "Gráfico de normalidad de los residuos",
            "Cuantil normal",
            "Residuo estandarizado",
            [{"name": "Residuos", "x": theoretical, "y": np.sort(standardized)}],
            lines=[{"name": "Normal", "x": np.array([-3.0, 3.0]), "y": np.array([-3.0, 3.0]), "dash": "dash"}],
        )
    )
    out.chart(
        line_chart(
            "Residuos en el orden de las filas",
            "Observación",
            "Residuo",
            [{"name": "Residuo", "x": np.arange(1, n + 1), "y": resid}],
            hlines=[{"value": 0}],
        )
    )
    if "cusum" in chosen:
        cumulative = np.cumsum(resid) / (resid.std(ddof=k) * np.sqrt(n))
        bound = 0.948 * (1 + 2 * np.arange(1, n + 1) / n)
        out.chart(
            line_chart(
                "CUSUM de los residuos",
                "Observación",
                "CUSUM",
                [
                    {"name": "CUSUM", "x": np.arange(1, n + 1), "y": cumulative},
                    {"name": "Límite 5 %", "x": np.arange(1, n + 1), "y": bound, "dash": "dash"},
                    {"name": "Límite inferior", "x": np.arange(1, n + 1), "y": -bound, "dash": "dash"},
                ],
            )
        )
    return out.set(
        n=n, problems=problems, tests={r[0]: (float(r[2]) if isinstance(r[2], P) else None) for r in rows}
    ).build()


def _diagnostics_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "from statsmodels.stats import diagnostic\nfrom statsmodels.stats.stattools import durbin_watson\n\n"
        + load_line(dataset)
        + formula_code(roles["response"], roles["predictors"])
        + "fit = smf.ols(formula, df).fit()\nprint(diagnostic.het_breuschpagan(fit.resid, fit.model.exog))\nprint(diagnostic.het_white(fit.resid, fit.model.exog))\nprint(diagnostic.linear_reset(fit, power=3, use_f=True))\nprint(diagnostic.acorr_breusch_godfrey(fit, nlags=4))\nprint(durbin_watson(fit.resid))\n"
    )


# ============================================================================ the catalogue
def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec("stepwise", "Regresión paso a paso", SELECTION, "Selecciona predictores hacia adelante, hacia atrás o en ambas direcciones por valor p, AIC o BIC.", _stepwise, roles=(RESPONSE, PREDICTORS), params=(FieldSpec("direction", "Dirección", "select", "both", (("forward", "Hacia adelante"), ("backward", "Hacia atrás"), ("both", "En ambas direcciones"))), FieldSpec("criterion", "Criterio", "select", "aic", (("aic", "AIC"), ("bic", "BIC"), ("pvalue", "Valor p del F parcial"))), FieldSpec("alpha_enter", "α para entrar", "number", 0.15, minimum=0.001, maximum=0.5, visible_if=("criterion", ("pvalue",))), FieldSpec("alpha_remove", "α para salir", "number", 0.15, minimum=0.001, maximum=0.5, visible_if=("criterion", ("pvalue",))), FieldSpec("max_steps", "Máximo de pasos", "integer", 30, minimum=1, maximum=100), ALPHA), outputs=(PREDICTIONS,), code=_stepwise_code, example={"roles": {"response": "y", "predictors": ["x1", "x2", "x3", "g"]}, "params": {"criterion": "bic"}}),
    _spec("best_subsets", "Mejores subconjuntos", SELECTION, "Evalúa todas las combinaciones de predictores y muestra las mejores de cada tamaño con R², Cp, AIC y BIC.", _best_subsets, heavy=True, roles=(RESPONSE, RoleSpec("predictors", "Predictores numéricos (2 a 15)", "numeric", multiple=True, min_count=2, max_count=MAX_SUBSET_PREDICTORS)), params=(FieldSpec("per_size", "Modelos a mostrar por tamaño", "integer", 2, minimum=1, maximum=5),), code=_best_subsets_code, example={"roles": {"response": "y", "predictors": ["x1", "x2", "x3"]}, "params": {}}),
    _spec("regularized", "Ridge, lasso y red elástica", SELECTION, "Regresión penalizada con validación cruzada: coeficientes, trayectoria y error según el parámetro λ.", _regularized, heavy=True, roles=(RESPONSE, PREDICTORS), params=(FieldSpec("penalty", "Penalización", "select", "lasso", _PENALTIES), FieldSpec("rule", "Elección de λ", "select", "min", (("min", "El de menor error de validación"), ("one_se", "Regla de 1 error estándar (modelo más simple)"))), FieldSpec("folds", "Pliegues de validación cruzada", "integer", 5, minimum=3, maximum=20), FieldSpec("seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1)), outputs=(PREDICTIONS,), code=_regularized_code, example={"roles": {"response": "y", "predictors": ["x1", "x2", "x3", "g"]}, "params": {"penalty": "lasso"}}),
    _spec("iv", "Variables instrumentales (2MCO)", ADVANCED, "Corrige la endogeneidad con instrumentos: primera etapa, Wu-Hausman y Sargan.", _iv, roles=(RESPONSE, RoleSpec("endogenous", "Variables endógenas", "numeric", multiple=True), RoleSpec("exogenous", "Controles exógenos (opcional)", "any", multiple=True, optional=True), RoleSpec("instruments", "Instrumentos", "numeric", multiple=True)), params=(FieldSpec("cov", "Errores estándar", "select", "robust", (("robust", "Robustos"), ("unadjusted", "Clásicos"))), ALPHA), code=_iv_code, example={"roles": {"response": "y", "endogenous": ["x1"], "exogenous": ["x2"], "instruments": ["x3", "r1"]}, "params": {}}),
    _spec("mixed", "Modelo lineal mixto", ADVANCED, "Datos agrupados o medidas repetidas: efectos fijos y varianza entre grupos (intercepto y pendiente aleatorios).", _mixed, heavy=True, roles=(RESPONSE, PREDICTORS, RoleSpec("group", "Grupo (efecto aleatorio)", "categorical"), RoleSpec("slope", "Pendiente aleatoria de (opcional)", "numeric", optional=True)), params=(FieldSpec("method", "Estimación", "select", "reml", (("reml", "REML"), ("ml", "Máxima verosimilitud"))), ALPHA), outputs=(PREDICTIONS,), code=_mixed_code, example={"roles": {"response": "yc", "predictors": ["x1"], "group": "cl"}, "params": {}}),
    _spec("diagnostics", "Pruebas de especificación de un modelo lineal", ADVANCED, "White, Breusch-Pagan, RESET, Durbin-Watson, Breusch-Godfrey, normalidad, CUSUM y Chow, con gráficos de residuos.", _diagnostics, roles=(RESPONSE, PREDICTORS, RoleSpec("order", "Ordenar las filas por (opcional)", "any", optional=True, help="Para las pruebas de autocorrelación y cambio estructural: fecha, tiempo u otro orden.")), params=(FieldSpec("tests", "Pruebas", "multiselect", ["breusch_pagan", "white", "reset", "durbin_watson", "jarque_bera"], _TESTS), FieldSpec("reset_power", "Potencia máxima del RESET", "integer", 3, minimum=2, maximum=5), FieldSpec("lags", "Rezagos de Breusch-Godfrey", "integer", 4, minimum=1, maximum=40), FieldSpec("break_at", "Punto de cambio de Chow (fracción 0-1 o número de fila)", "number", 0.5, minimum=0.05), ALPHA), code=_diagnostics_code, example={"roles": {"response": "y", "predictors": ["x1", "x2"], "order": "t"}, "params": {"tests": ["breusch_pagan", "white", "reset", "breusch_godfrey", "cusum", "chow", "shapiro"]}}),
)  # fmt: skip

registry.register(*SPECS)
