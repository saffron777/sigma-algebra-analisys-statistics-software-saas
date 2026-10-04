"""Time series (Minitab *Time Series*, EViews series views, SPSS *Forecasting*)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import line_chart, stem_chart
from sigma.domain.analysis.output import Output, P, bad
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec
from sigma.domain.errors import InvalidInputError

MIN_OBSERVATIONS = 10


def ordered_series(frame: pd.DataFrame, roles: dict[str, Any]) -> tuple[pd.Series, str]:
    """The series sorted by its time column (when given), without missing values, and its name."""
    name = roles["series"]
    data = frame[[name]].copy()
    time_name = roles.get("time")
    if time_name:
        data[time_name] = frame[time_name]
        data = data.sort_values(time_name, kind="stable")
        if data[time_name].duplicated().any():
            raise InvalidInputError(f"La columna de tiempo «{time_name}» tiene fechas repetidas.")
        data = data.set_index(time_name)
    values = pd.to_numeric(data[name], errors="coerce")
    dropped = int(values.isna().sum())
    values = values.dropna()
    if len(values) < MIN_OBSERVATIONS:
        raise InvalidInputError(f"La serie necesita al menos {MIN_OBSERVATIONS} observaciones válidas.")
    values.attrs["dropped"] = dropped
    return values.astype(float), name


def _acf_pacf(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    series, name = ordered_series(frame, roles)
    n = len(series)
    lags = min(int(p["lags"]), n // 2 - 1)
    if lags < 1:
        raise InvalidInputError("La serie es demasiado corta para esa cantidad de rezagos.")
    alpha = p["alpha"]
    values = series.to_numpy()
    r, r_ci = acf(values, nlags=lags, alpha=alpha, fft=True, result_object=False)
    pr, pr_ci = pacf(values, nlags=lags, alpha=alpha, method=p["pacf_method"], result_object=False)
    lb_lags = list(range(1, lags + 1))
    lb = acorr_ljungbox(values, lags=lb_lags, return_df=True)
    out = Output(f"Autocorrelación de {name}", f"n = {n} · {lags} rezagos · bandas al {100 * (1 - alpha):g} %")
    if series.attrs.get("dropped"):
        out.alert("info", f"Se omitieron {series.attrs['dropped']} valor(es) faltante(s).")
    significant = [k for k in range(1, lags + 1) if abs(r[k]) > abs(r_ci[k, 1] - r[k])]
    out.tiles(
        [
            ("Observaciones", n),
            ("Media", float(values.mean())),
            ("Desv. estándar", float(values.std(ddof=1))),
            ("Rezagos significativos (ACF)", len(significant)),
        ]
    )
    q_final = float(lb["lb_stat"].iloc[-1])
    p_final = float(lb["lb_pvalue"].iloc[-1])
    out.alert(
        "warning" if p_final < alpha else "success",
        f"Ljung-Box con {lags} rezagos: Q = {q_final:.3f}, p = {p_final:.4f} → "
        + ("hay autocorrelación en la serie." if p_final < alpha else "no hay evidencia de autocorrelación."),
    )
    x = np.arange(1, lags + 1)
    out.chart(
        stem_chart("Función de autocorrelación (ACF)", "Rezago", "Autocorrelación", x, r[1:], band=r_ci[1:, 1] - r[1:])
    )
    out.chart(
        stem_chart(
            "Autocorrelación parcial (PACF)", "Rezago", "Autocorrelación parcial", x, pr[1:], band=pr_ci[1:, 1] - pr[1:]
        )
    )
    out.table(
        "Autocorrelaciones",
        ["Rezago", "ACF", "PACF", "Q de Ljung-Box", "Valor p"],
        [
            [
                k,
                bad(float(r[k])) if k in significant else float(r[k]),
                float(pr[k]),
                float(lb["lb_stat"].iloc[k - 1]),
                P(float(lb["lb_pvalue"].iloc[k - 1])),
            ]
            for k in range(1, lags + 1)
        ],
        note="En rojo: autocorrelaciones fuera de las bandas de confianza.",
    )
    out.chart(
        line_chart(
            "Serie",
            "Observación" if not roles.get("time") else "Tiempo",
            name,
            [
                {
                    "name": name,
                    "x": [str(v) for v in series.index] if roles.get("time") else np.arange(1, n + 1),
                    "y": values,
                }
            ],
        )
    )
    out.set(n=n, lags=lags, ljung_box_q=q_final, ljung_box_p=p_final, significant_lags=significant, acf=r[1:6].tolist())
    return out.build()


def _acf_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    time = roles.get("time")
    prepare = f'df = sigma.load("{dataset}")\n' + (f'df = df.sort_values("{time}")\n' if time else "")
    return (
        "import numpy as np\n"
        "from statsmodels.tsa.stattools import acf, pacf\n"
        "from statsmodels.stats.diagnostic import acorr_ljungbox\n\n"
        f'{prepare}x = df["{roles["series"]}"].dropna().to_numpy(dtype=float)\n'
        f"r = acf(x, nlags={p['lags']}, fft=True)\n"
        f'pr = pacf(x, nlags={p["lags"]}, method="{p["pacf_method"]}")\n'
        f"print(acorr_ljungbox(x, lags={p['lags']}))\n"
    )


SPECS = (
    AnalysisSpec(
        "acf_pacf",
        "Autocorrelación (ACF, PACF y Ljung-Box)",
        "Exploración",
        "series",
        "Funciones de autocorrelación simple y parcial con bandas de confianza y prueba de Ljung-Box.",
        compute=_acf_pacf,
        roles=(
            RoleSpec("series", "Serie", "numeric"),
            RoleSpec(
                "time",
                "Columna de tiempo (opcional)",
                "datetime",
                optional=True,
                help="Si se omite se usa el orden de las filas.",
            ),
        ),
        params=(
            FieldSpec("lags", "Rezagos", "integer", 24, minimum=1, maximum=200),
            FieldSpec("alpha", "Nivel de significancia α", "number", 0.05, minimum=0.001, maximum=0.5),
            FieldSpec(
                "pacf_method",
                "Método de la PACF",
                "select",
                "ywm",
                (("ywm", "Yule-Walker (ajustado)"), ("ols", "Regresiones MCO"), ("ldb", "Levinson-Durbin")),
            ),
        ),
        code=_acf_code,
        example={"roles": {"series": "s", "time": "t"}, "params": {"lags": 12}},
    ),
)

registry.register(*SPECS)
