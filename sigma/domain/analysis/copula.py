"""Cópulas: dependencia entre variables con marginales arbitrarias (Gaussiana, Clayton, Gumbel, Frank)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.distributions.copula.api import ClaytonCopula, FrankCopula, GaussianCopula, GumbelCopula

from sigma.domain.analysis import registry
from sigma.domain.analysis.charts import scatter_chart
from sigma.domain.analysis.output import Output
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, OutputSpec, RoleSpec
from sigma.domain.analysis.support import load_line, py_name
from sigma.domain.errors import InvalidInputError

MODULE = "econometria"
GROUP_COPULA = "Dependencia y cópulas"

X = RoleSpec("x", "Primera variable", "numeric")
Y = RoleSpec("y", "Segunda variable", "numeric")

_FAMILIES = (
    ("gaussian", "Gaussiana (elíptica, sin dependencia de colas)"),
    ("clayton", "Clayton (dependencia en la cola inferior)"),
    ("gumbel", "Gumbel (dependencia en la cola superior)"),
    ("frank", "Frank (simétrica, sin dependencia de colas)"),
)


def _tau(family: str, cop, param: float) -> float:
    if family == "gaussian":
        return float(2 / np.pi * np.arcsin(np.clip(param, -1, 1)))
    return float(cop.tau(param))


def _tail_dependence(family: str, param: float) -> tuple[float, float]:
    """(lower, upper) tail dependence."""
    if family == "clayton":
        return (float(2 ** (-1 / param)) if param > 0 else 0.0), 0.0
    if family == "gumbel":
        return 0.0, (float(2 - 2 ** (1 / param)) if param >= 1 else 0.0)
    return 0.0, 0.0  # gaussian / frank


def _copula(frame: pd.DataFrame, roles: dict[str, Any], p: dict[str, Any]):
    data = pd.DataFrame({"x": pd.to_numeric(frame[roles["x"]], errors="coerce"), "y": pd.to_numeric(frame[roles["y"]], errors="coerce")}).dropna()
    if len(data) < 20:
        raise InvalidInputError("Se necesitan al menos 20 observaciones pareadas.")
    x, y = data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float)
    if x.std() == 0 or y.std() == 0:
        raise InvalidInputError("Una de las variables es constante.")
    family = p["family"]
    xy = np.column_stack([x, y])
    cop = {"gaussian": GaussianCopula, "clayton": ClaytonCopula, "gumbel": GumbelCopula, "frank": FrankCopula}[family](k_dim=2)
    param = float(cop.fit_corr_param(xy))
    tau = _tau(family, cop, param)
    lower, upper = _tail_dependence(family, param)
    pearson = float(stats.pearsonr(x, y)[0])
    out = Output(f"Cópula {dict(_FAMILIES)[family].split(' (')[0]}", f"n = {len(data)} pares")
    out.tiles(
        [
            ("Correlación de Pearson", pearson),
            ("τ de Kendall", tau),
            ("Parámetro", param),
            ("Dependencia cola inferior", lower),
            ("Dependencia cola superior", upper),
        ]
    )
    out.alert(
        "info",
        "La cópula separa la estructura de dependencia de las distribuciones marginales: τ y la dependencia de colas describen la asociación sin suponer normalidad.",
    )
    series = [{"name": "Datos", "x": x, "y": y}]
    n_sim = int(p["simulate"])
    if n_sim > 0:
        if family == "gaussian":
            sim_cop = GaussianCopula(corr=param, k_dim=2)
        else:
            sim_cop = {"clayton": ClaytonCopula, "gumbel": GumbelCopula, "frank": FrankCopula}[family](theta=param)
        sim = sim_cop.rvs(n_sim)
        series.append(
            {
                "name": f"Simulado ({n_sim})",
                "x": np.quantile(x, sim[:, 0]),
                "y": np.quantile(y, sim[:, 1]),
            }
        )
    out.chart(scatter_chart("Datos y muestra simulada", roles["x"], roles["y"], series))
    if n_sim > 0:
        out.dataset("copula", "Muestra simulada de la cópula", pd.DataFrame({"u": sim[:, 0], "v": sim[:, 1]}))
    return out.set(family=family, parameter=param, tau=tau, pearson=pearson, tail_lower=lower, tail_upper=upper).build()


def _copula_code(roles: dict[str, Any], p: dict[str, Any], dataset: str) -> str:
    return (
        "import numpy as np\nfrom statsmodels.distributions.copula.api import GaussianCopula\n\n"
        + load_line(dataset)
        + f"x = df[{py_name(roles['x'])}].to_numpy(float)\ny = df[{py_name(roles['y'])}].to_numpy(float)\n"
        + "cop = GaussianCopula(k_dim=2)\ncorr = cop.fit_corr_param(np.column_stack([x, y]))\n"
        + "print('correlación', corr, 'tau', 2 / np.pi * np.arcsin(corr))\n"
    )


def _spec(id_: str, label: str, group: str, description: str, compute, **kw) -> AnalysisSpec:
    return AnalysisSpec(id_, label, group, MODULE, description, compute, **kw)


SPECS = (
    _spec(
        "copula",
        "Cópula (dependencia entre variables)",
        GROUP_COPULA,
        "Ajusta una cópula (Gaussiana, Clayton, Gumbel o Frank) a dos variables y reporta la τ de Kendall y la dependencia de colas.",
        _copula,
        roles=(X, Y),
        params=(
            FieldSpec("family", "Familia de cópula", "select", "gaussian", _FAMILIES),
            FieldSpec("simulate", "Puntos a simular (0 = no)", "integer", 0, minimum=0, maximum=5000),
        ),
        outputs=(OutputSpec("copula", "Muestra simulada"),),
        code=_copula_code,
        example={"roles": {"x": "x1", "y": "x2"}, "params": {"family": "gaussian", "simulate": 200}},
    ),
)

registry.register(*SPECS)
