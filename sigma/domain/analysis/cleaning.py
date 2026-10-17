"""Data quality: missing values and outliers (SPSS *Missing Value Analysis*/*Explore*, Minitab *Data*, EViews *smpl*)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import box_chart
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import (
    RESULT_KEY,
    cell,
    check_new_columns,
    column_name,
    load_line,
    preview,
    transformed,
)
from sigma.domain.datasets import classify_column
from sigma.domain.errors import InvalidInputError

GROUP = "Calidad de los datos"
_SAVE = "\n# sigma.save(df, 'nombre_del_dataset_nuevo')\n"
_OUTPUT = (OutputSpec(RESULT_KEY, "Datos limpios"),)
_COMMON = {"module": "transformar", "group": GROUP, "outputs": _OUTPUT, "can_replace": True}


# ============================================================================ missing values
_METHODS = (
    ("drop_rows", "Eliminar las filas con algún faltante en las columnas elegidas"),
    ("drop_columns", "Eliminar las columnas con demasiados faltantes"),
    ("mean", "Reemplazar por la media (numéricas)"),
    ("median", "Reemplazar por la mediana (numéricas)"),
    ("mode", "Reemplazar por la moda (cualquier tipo)"),
    ("constant", "Reemplazar por un valor fijo"),
    ("ffill", "Arrastrar el último valor conocido (hacia adelante)"),
    ("bfill", "Traer el siguiente valor conocido (hacia atrás)"),
    ("interpolate", "Interpolación lineal (numéricas)"),
)
_NUMERIC_ONLY = {"mean", "median", "interpolate"}


def _fill_value(series: pd.Series, p: dict[str, Any]) -> Any:
    method, kind = p["method"], classify_column(series)
    if method == "mean":
        return float(pd.to_numeric(series, errors="coerce").mean())
    if method == "median":
        return float(pd.to_numeric(series, errors="coerce").median())
    if method == "mode":
        modes = series.mode(dropna=True)
        return modes.iloc[0] if len(modes) else None
    text = str(p.get("value") or "").strip()
    if not text:
        raise InvalidInputError("Indique el valor con el que reemplazar.")
    if kind == "numeric":
        try:
            return float(text.replace(",", "."))
        except ValueError:
            raise InvalidInputError(f"«{series.name}» es numérica: el valor «{text}» no es un número.") from None
    if kind == "datetime":
        try:
            return pd.Timestamp(text)
        except (ValueError, TypeError):
            raise InvalidInputError(
                f"«{series.name}» es de fecha: «{text}» no es una fecha válida (AAAA-MM-DD)."
            ) from None
    return text


def _missing(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    method = p["method"]
    chosen = roles.get("cols") or list(frame.columns)
    explicit = bool(roles.get("cols"))
    with_gaps = [c for c in chosen if frame[c].isna().any()]
    after = frame.copy()
    table: list[list[Any]] = []
    skipped: list[str] = []
    created: list[str] = []
    if method == "drop_rows":
        after = frame.dropna(subset=chosen).reset_index(drop=True)
        if after.empty:
            raise InvalidInputError("Todas las filas tienen algún faltante en esas columnas: no quedaría ninguna.")
        for col in with_gaps:
            table.append([col, int(frame[col].isna().sum()), 0, "se eliminan las filas"])
    elif method == "drop_columns":
        share = frame[chosen].isna().mean() * 100
        dropped = [c for c in chosen if share[c] > p["threshold"]]
        if len(dropped) == frame.shape[1]:
            raise InvalidInputError("Se quitarían todas las columnas: suba el umbral.")
        after = frame.drop(columns=dropped)
        table = [
            [c, int(frame[c].isna().sum()), float(share[c]), "se elimina la columna" if c in dropped else "se conserva"]
            for c in chosen
            if share[c] > 0 or c in dropped
        ]
        table = [[c, n, f"{s:.1f} %", what] for c, n, s, what in table]
    else:
        for col in chosen:
            series = frame[col]
            if method in _NUMERIC_ONLY and classify_column(series) not in ("numeric", "boolean"):
                skipped.append(col)
                continue
            missing = int(series.isna().sum())
            if not missing:
                continue
            if method in ("ffill", "bfill"):
                filled, used = (series.ffill() if method == "ffill" else series.bfill()), method
            elif method == "interpolate":
                filled, used = (
                    pd.to_numeric(series, errors="coerce").interpolate(method="linear", limit_direction="both"),
                    "lineal",
                )
            else:
                value = _fill_value(series, p)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    skipped.append(col)
                    continue
                filled, used = series.fillna(value), cell(value)
            if p["flag"]:
                flag = column_name(after, f"{col}_faltante")
                after[flag] = series.isna().astype("int64")
                created.append(flag)
            after[col] = filled
            table.append(
                [
                    col,
                    missing,
                    int(filled.isna().sum()),
                    used if isinstance(used, str) else float(used) if isinstance(used, int | float) else str(used),
                ]
            )
        if skipped and explicit and len(skipped) == len(chosen):
            raise InvalidInputError(
                "Ese método solo se aplica a columnas numéricas (o a columnas con algún dato presente)."
            )
    check_new_columns(len(created))
    label = dict(_METHODS)[method]
    out = transformed("Valores faltantes", label, frame, after, created=created)
    total_before = int(frame[chosen].isna().sum().sum())
    total_after = int(after[[c for c in chosen if c in after.columns]].isna().sum().sum())
    out.tiles(
        [
            ("Faltantes antes", total_before),
            ("Faltantes después", total_after),
            ("Columnas con faltantes", len(with_gaps)),
        ]
    )
    if table:
        header = {"drop_rows": "Qué se hace", "drop_columns": "Qué se hace"}.get(method, "Valor usado")
        columns = [
            "Columna",
            "Faltantes antes",
            "Faltantes después" if method != "drop_columns" else "% faltante",
            header,
        ]
        out.table("Detalle por columna", columns, table)
    else:
        out.alert("info", "Las columnas elegidas no tenían valores faltantes que reemplazar.")
    if skipped:
        out.alert(
            "info", f"Se omitieron por no ser del tipo adecuado (o no tener ningún dato): {', '.join(skipped[:8])}."
        )
    if method == "drop_rows":
        out.alert("info", f"Se eliminaron {len(frame) - len(after)} de {len(frame)} filas.")
    if created:
        preview(out, after, created[:4])
    return out.set(missing_before=total_before, missing_after=total_after).build()


def _missing_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    cols = roles.get("cols") or None
    target = repr(cols) if cols else "df.columns"
    m = p["method"]
    body = {
        "drop_rows": f"df = df.dropna(subset={target}).reset_index(drop=True)\n",
        "drop_columns": f"limit = {p['threshold']}\ndf = df.drop(columns=[c for c in {target} if df[c].isna().mean() * 100 > limit])\n",
        "mean": f"for c in {target}:\n    df[c] = df[c].fillna(df[c].mean())\n",
        "median": f"for c in {target}:\n    df[c] = df[c].fillna(df[c].median())\n",
        "mode": f"for c in {target}:\n    df[c] = df[c].fillna(df[c].mode().iloc[0])\n",
        "constant": f"df[{target}] = df[{target}].fillna({p.get('value')!r})\n",
        "ffill": f"df[{target}] = df[{target}].ffill()\n",
        "bfill": f"df[{target}] = df[{target}].bfill()\n",
        "interpolate": f"for c in {target}:\n    df[c] = df[c].interpolate(limit_direction='both')\n",
    }[m]
    return load_line(dataset) + body + _SAVE


# ============================================================================ multivariate imputation
def _impute_model(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer

    cols, extra = roles["cols"], roles.get("predictors", [])
    use = [*cols, *(c for c in extra if c not in cols)]
    data = frame[use].apply(pd.to_numeric, errors="coerce")
    gaps = data[cols].isna()
    if not gaps.to_numpy().any():
        raise InvalidInputError("Las columnas elegidas no tienen valores faltantes.")
    if (data.notna().sum() == 0).any():
        raise InvalidInputError("Una de las columnas no tiene ningún dato: no hay de dónde aprender.")
    mean, sd = data.mean(), data.std(ddof=1).replace(0, 1.0).fillna(1.0)
    scaled = (data - mean) / sd  # distances and regressions behave better on comparable scales
    if p["method"] == "knn":
        imputer = KNNImputer(n_neighbors=int(p["k"]), weights="distance")
    else:
        imputer = IterativeImputer(
            max_iter=int(p["max_iter"]), random_state=int(p["seed"]), sample_posterior=False, skip_complete=True
        )
    filled = pd.DataFrame(imputer.fit_transform(scaled), columns=use, index=data.index) * sd + mean
    after = frame.copy()
    rows = []
    for col in cols:
        before_series = data[col]
        after[col] = before_series.where(before_series.notna(), filled[col])
        rows.append(
            [
                col,
                int(gaps[col].sum()),
                float(before_series.mean()),
                float(after[col].mean()),
                float(before_series.std(ddof=1)),
                float(after[col].std(ddof=1)),
            ]
        )
    label = (
        f"KNN (k = {p['k']})" if p["method"] == "knn" else f"MICE / imputación encadenada ({p['max_iter']} iteraciones)"
    )
    out = transformed("Imputación multivariante", label, frame, after)
    out.tiles([("Valores imputados", int(gaps.to_numpy().sum())), ("Columnas", len(cols)), ("Predictores", len(use))])
    out.table(
        "Efecto sobre cada columna",
        ["Columna", "Imputados", "Media antes", "Media después", "Desv. antes", "Desv. después"],
        rows,
        note="Una buena imputación deja la media y la dispersión casi iguales.",
    )
    return out.set(imputed=int(gaps.to_numpy().sum())).build()


def _impute_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    use = [*roles["cols"], *(c for c in roles.get("predictors", []) if c not in roles["cols"])]
    if p["method"] == "knn":
        make = f"KNNImputer(n_neighbors={p['k']}, weights='distance')"
        header = "from sklearn.impute import KNNImputer\n"
    else:
        make = f"IterativeImputer(max_iter={p['max_iter']}, random_state={p['seed']})"
        header = "from sklearn.experimental import enable_iterative_imputer  # noqa\nfrom sklearn.impute import IterativeImputer\n"
    return (
        "import pandas as pd\n"
        + header
        + "\n"
        + load_line(dataset)
        + f"cols = {use!r}\nz = (df[cols] - df[cols].mean()) / df[cols].std()\n"
        + f"df[cols] = pd.DataFrame({make}.fit_transform(z), columns=cols, index=df.index) * df[cols].std() + df[cols].mean()\n"
        + _SAVE
    )


# ============================================================================ outliers
_DETECTORS = (
    ("iqr", "Rango intercuartílico (k × IQR fuera de los cuartiles, como el diagrama de caja)"),
    ("zscore", "Puntuación z (|z| mayor que k)"),
    ("mad", "Z modificada con MAD (robusta; |z*| mayor que k)"),
    ("grubbs", "Prueba de Grubbs repetida (supone normalidad)"),
)
_ACTIONS = (
    ("report", "Solo informar (no cambiar los datos)"),
    ("flag", "Añadir una columna indicadora 0/1 por variable"),
    ("missing", "Reemplazar los atípicos por dato faltante"),
    ("winsorize", "Limitar al valor límite más cercano (winsorizar)"),
    ("remove", "Eliminar las filas con algún atípico"),
)


def grubbs_critical(n: int, alpha: float) -> float:
    """Critical value of the two-sided Grubbs statistic for ``n`` observations."""
    t = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    return float((n - 1) / np.sqrt(n) * np.sqrt(t**2 / (n - 2 + t**2)))


def _grubbs(values: pd.Series, alpha: float) -> tuple[pd.Series, float, float]:
    """Repeated two-sided Grubbs test; returns the outlier mask and the range of what is left."""
    mask = pd.Series(False, index=values.index)
    live = values.dropna()
    while len(live) > 6:
        z = (live - live.mean()).abs() / live.std(ddof=1)
        worst = z.idxmax()
        if z[worst] <= grubbs_critical(len(live), alpha):
            break
        mask[worst] = True
        live = live.drop(worst)
    return mask, float(live.min()), float(live.max())


def _detect(values: pd.Series, method: str, k: float, alpha: float) -> tuple[pd.Series, float, float]:
    """(outlier mask, lower limit, upper limit) of one column."""
    data = values.dropna()
    if len(data) < 5:
        raise InvalidInputError(f"«{values.name}» tiene muy pocos datos para buscar atípicos (mínimo 5).")
    if method == "grubbs":
        return _grubbs(values, alpha)
    if method == "iqr":
        q1, q3 = data.quantile([0.25, 0.75])
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    elif method == "zscore":
        low, high = data.mean() - k * data.std(ddof=1), data.mean() + k * data.std(ddof=1)
    else:
        median = data.median()
        mad = (data - median).abs().median()
        if mad == 0:
            return pd.Series(False, index=values.index), float(data.min()), float(data.max())
        low, high = median - k * mad / 0.6745, median + k * mad / 0.6745
    mask = (values < low) | (values > high)
    return mask.fillna(False), float(low), float(high)


def _outliers(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    cols, method, action = roles["cols"], p["method"], p["action"]
    k = p["threshold"] if method != "grubbs" else 0.0
    data = frame[cols].apply(pd.to_numeric, errors="coerce")
    after = frame.copy()
    summary, found, any_outlier = [], [], pd.Series(False, index=frame.index)
    created: list[str] = []
    for col in cols:
        mask, low, high = _detect(data[col], method, k, p["alpha"])
        any_outlier |= mask
        n = int(mask.sum())
        summary.append(
            [col, int(data[col].notna().sum()), n, 100 * n / max(int(data[col].notna().sum()), 1), low, high]
        )
        for row in data.index[mask][:15]:
            found.append([col, int(row) + 1, float(data.at[row, col])])
        if action == "flag":
            name = column_name(after, f"{col}_atipico")
            after[name] = mask.astype("int64")
            created.append(name)
        elif action == "missing":
            after[col] = data[col].mask(mask)
        elif action == "winsorize":
            after[col] = data[col].clip(low, high)
    if action == "remove":
        after = after[~any_outlier].reset_index(drop=True)
        if after.empty:
            raise InvalidInputError("Todas las filas tienen algún atípico: no quedaría ninguna.")
    check_new_columns(len(created))
    out = transformed(
        "Valores atípicos",
        dict(_DETECTORS)[method].split(" (")[0] + (f" · k = {k:g}" if method != "grubbs" else f" · α = {p['alpha']:g}"),
        frame,
        after,
        created=created,
    )
    total = sum(r[2] for r in summary)
    out.tiles(
        [("Atípicos detectados", total), ("Filas afectadas", int(any_outlier.sum())), ("Columnas revisadas", len(cols))]
    )
    out.alert(
        "warning" if total else "success",
        (
            f"Se detectaron {total} valor(es) atípico(s) en {int(any_outlier.sum())} fila(s). "
            if total
            else "No se detectaron valores atípicos. "
        )
        + {
            "report": "Los datos no se modificaron: elija una acción y un nombre para guardar el resultado.",
            "flag": "Se añadieron columnas indicadoras.",
            "missing": "Quedaron como dato faltante.",
            "winsorize": "Se limitaron a los valores límite.",
            "remove": "Se eliminaron esas filas.",
        }[action],
    )
    out.table(
        "Resumen por columna", ["Columna", "Datos", "Atípicos", "%", "Límite inferior", "Límite superior"], summary
    )
    if found:
        out.table(
            "Primeros valores atípicos",
            ["Columna", "Fila", "Valor"],
            found[:30],
            note="Los números de fila empiezan en 1.",
        )
    for col in cols[:6]:
        out.chart(
            box_chart(
                f"Diagrama de caja: {col}",
                col,
                [{"name": col, "values": data[col].to_numpy()}],
                points=False,
                height=320,
            )
        )
    return out.set(outliers=total, rows_affected=int(any_outlier.sum())).build()


def _outliers_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    k = p["threshold"]
    head = "import pandas as pd\n\n" + load_line(dataset)
    detect = {
        "iqr": f"q1, q3 = x.quantile([0.25, 0.75])\n    low, high = q1 - {k} * (q3 - q1), q3 + {k} * (q3 - q1)\n",
        "zscore": f"low, high = x.mean() - {k} * x.std(), x.mean() + {k} * x.std()\n",
        "mad": f"mad = (x - x.median()).abs().median()\n    low, high = x.median() - {k} * mad / 0.6745, x.median() + {k} * mad / 0.6745\n",
        "grubbs": "# la prueba de Grubbs repetida está implementada en Sigma Algebra; aquí se muestra con límites de 3 desviaciones\n    low, high = x.mean() - 3 * x.std(), x.mean() + 3 * x.std()\n",
    }[p["method"]]
    action = {
        "report": "print(col, int(out.sum()), 'atípicos')\n",
        "flag": "df[col + '_atipico'] = out.astype(int)\n",
        "missing": "df[col] = x.mask(out)\n",
        "winsorize": "df[col] = x.clip(low, high)\n",
        "remove": "flagged |= out\n",
    }[p["action"]]
    tail = "df = df[~flagged].reset_index(drop=True)\n" if p["action"] == "remove" else ""
    init = "flagged = pd.Series(False, index=df.index)\n" if p["action"] == "remove" else ""
    body = f"for col in {roles['cols']!r}:\n    x = pd.to_numeric(df[col], errors='coerce')\n    {detect}    out = (x < low) | (x > high)\n    {action.replace(chr(10), chr(10) + '    ').rstrip()}\n"
    return head + init + body + tail + _SAVE


# ============================================================================ the catalogue
SPECS = (
    AnalysisSpec(
        "missing",
        "Valores faltantes: eliminar o reemplazar",
        compute=_missing,
        description="Elimina filas o columnas con faltantes, o los reemplaza por la media, la mediana, la moda, un valor fijo, el valor vecino o una interpolación.",
        roles=(RoleSpec("cols", "Columnas (si no elige ninguna, todas)", "any", multiple=True, optional=True),),
        params=(
            FieldSpec("method", "Método", "select", "mean", _METHODS),
            FieldSpec("value", "Valor de reemplazo", "text", visible_if=("method", ("constant",))),
            FieldSpec(
                "threshold",
                "Eliminar si faltan más del (%)",
                "number",
                50,
                minimum=0,
                maximum=99.9,
                visible_if=("method", ("drop_columns",)),
            ),
            FieldSpec(
                "flag",
                "Añadir una columna indicadora 0/1 de lo que faltaba",
                "boolean",
                False,
                help="Útil para modelos: conserva la información de que el dato se imputó.",
            ),
        ),
        code=_missing_code,
        example={"roles": {"cols": ["m"]}, "params": {"method": "median", "flag": True}},
        **_COMMON,
    ),
    AnalysisSpec(
        "impute_model",
        "Imputación multivariante (KNN / MICE)",
        compute=_impute_model,
        description="Estima los faltantes numéricos a partir de las demás columnas: vecinos más cercanos o regresiones encadenadas (MICE).",
        roles=(
            RoleSpec("cols", "Columnas con faltantes a imputar", "numeric", multiple=True),
            RoleSpec(
                "predictors",
                "Otras columnas que ayudan a predecir (opcional)",
                "numeric",
                multiple=True,
                optional=True,
                min_count=1,
            ),
        ),
        params=(
            FieldSpec(
                "method",
                "Método",
                "select",
                "mice",
                (("mice", "MICE (regresiones encadenadas)"), ("knn", "K vecinos más cercanos")),
            ),
            FieldSpec("k", "Vecinos (k)", "integer", 5, minimum=1, maximum=100, visible_if=("method", ("knn",))),
            FieldSpec(
                "max_iter", "Iteraciones", "integer", 10, minimum=1, maximum=100, visible_if=("method", ("mice",))
            ),
            FieldSpec(
                "seed", "Semilla", "integer", 123, minimum=0, maximum=2**31 - 1, visible_if=("method", ("mice",))
            ),
        ),
        heavy=True,
        code=_impute_code,
        example={"roles": {"cols": ["m"], "predictors": ["x1", "x3", "y"]}, "params": {"method": "knn"}},
        **_COMMON,
    ),
    AnalysisSpec(
        "outliers",
        "Valores atípicos",
        compute=_outliers,
        description="Detecta atípicos (IQR, z, MAD, Grubbs) y los deja marcados, reemplazados, limitados o eliminados.",
        roles=(RoleSpec("cols", "Columnas numéricas", "numeric", multiple=True),),
        params=(
            FieldSpec("method", "Método de detección", "select", "iqr", _DETECTORS),
            FieldSpec(
                "threshold",
                "k (umbral)",
                "number",
                1.5,
                minimum=0.1,
                maximum=20,
                help="IQR: 1.5 (3 = extremos) · z: 3 · MAD: 3.5",
                visible_if=("method", ("iqr", "zscore", "mad")),
            ),
            FieldSpec(
                "alpha", "Nivel α", "number", 0.05, minimum=0.001, maximum=0.5, visible_if=("method", ("grubbs",))
            ),
            FieldSpec("action", "Qué hacer con ellos", "select", "report", _ACTIONS),
        ),
        code=_outliers_code,
        example={"roles": {"cols": ["o"]}, "params": {"method": "iqr", "action": "flag"}},
        **_COMMON,
    ),
)

registry.register(*SPECS)
