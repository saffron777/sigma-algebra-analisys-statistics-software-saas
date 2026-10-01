"""Charts for the descriptive-statistics module."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from sigma.domain.distributions import get_distribution
from sigma.domain.results import CorrelationResult, DistributionFit, FrequencyTable
from sigma.presentation.web.charts.common import PALETTE, as_list, figure_json, new_figure

PLOT_LIMIT = 3000


def _sample(values: np.ndarray) -> np.ndarray:
    if len(values) <= PLOT_LIMIT:
        return values
    return np.random.default_rng(0).choice(values, PLOT_LIMIT, replace=False)


def histogram_figure(name: str, groups: dict[str, np.ndarray], fits: list[DistributionFit]) -> dict[str, Any]:
    """Histogram of all the data (with fitted densities) or, without fits, overlaid per group."""
    overall = list(groups) == ["Todos"] or bool(fits)
    title = f"Histograma de {name}" + (" con ajustes" if fits else "")
    fig = new_figure(title, name, "Densidad" if overall and fits else "Frecuencia")
    if overall:
        data = _sample(groups["Todos"])
        fig.add_trace(
            go.Histogram(
                x=as_list(data),
                histnorm="probability density" if fits else "",
                name="Datos",
                marker={"color": PALETTE[0], "line": {"color": "white", "width": 0.6}},
                opacity=0.8,
            )
        )
        if fits:
            grid = np.linspace(float(data.min()), float(data.max()), 240)
            for i, fit in enumerate(fits[:4]):
                dist = get_distribution(fit.distribution).build(fit.params)
                fig.add_trace(
                    go.Scatter(
                        x=as_list(grid),
                        y=as_list(dist.pdf(grid)),
                        mode="lines",
                        name=f"{fit.label} (p KS = {fit.ks_p_value:.3f})",
                        line={"color": PALETTE[(i + 2) % len(PALETTE)], "width": 2.4},
                    )
                )
    else:
        for i, (group, values) in enumerate(g for g in groups.items() if g[0] != "Todos"):
            fig.add_trace(
                go.Histogram(
                    x=as_list(_sample(values)), name=group, opacity=0.6, marker={"color": PALETTE[i % len(PALETTE)]}
                )
            )
        fig.update_layout(barmode="overlay")
    return figure_json(fig)


def box_figure(name: str, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    fig = new_figure(f"Diagrama de caja de {name}", None, name)
    shown = {k: v for k, v in groups.items() if k != "Todos"} or groups
    for i, (group, values) in enumerate(shown.items()):
        fig.add_trace(
            go.Box(
                y=as_list(_sample(values)),
                name=group,
                boxmean="sd",
                boxpoints="outliers",
                marker={"color": PALETTE[i % len(PALETTE)]},
                line={"width": 1.6},
            )
        )
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def frequency_bar_figure(table: FrequencyTable) -> dict[str, Any]:
    fig = new_figure(f"Frecuencias de {table.name}", table.name, "Porcentaje (%)")
    fig.add_trace(
        go.Bar(
            x=[r.value for r in table.rows],
            y=[round(r.percent, 3) for r in table.rows],
            customdata=[r.count for r in table.rows],
            marker={"color": PALETTE[0]},
            hovertemplate="%{x}<br>%{y:.1f} %  (n=%{customdata})<extra></extra>",
        )
    )
    fig.update_xaxes(type="category")
    return figure_json(fig)


def qq_figure(name: str, values: np.ndarray) -> dict[str, Any]:
    (osm, osr), (slope, intercept, _) = stats.probplot(_sample(values), dist="norm")
    fig = new_figure(f"Gráfico Q-Q normal de {name}", "Cuantiles teóricos (normal)", "Cuantiles de la muestra")
    fig.add_trace(
        go.Scatter(
            x=as_list(osm),
            y=as_list(osr),
            mode="markers",
            name="Datos",
            marker={"color": PALETTE[0], "size": 6, "opacity": 0.75},
        )
    )
    line = np.array([float(np.min(osm)), float(np.max(osm))])
    fig.add_trace(
        go.Scatter(
            x=as_list(line),
            y=as_list(intercept + slope * line),
            mode="lines",
            name="Referencia",
            line={"color": "#ef4444", "width": 2},
        )
    )
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def correlation_figure(corr: CorrelationResult) -> dict[str, Any]:
    z = [[np.nan if v is None else v for v in row] for row in corr.matrix]
    text = [["" if v is None else f"{v:.2f}" for v in row] for row in corr.matrix]
    names = [str(c) for c in corr.columns]
    fig = new_figure(f"Matriz de correlación ({corr.method})", None, None, height=max(340, 60 * len(names) + 140))
    fig.add_trace(
        go.Heatmap(
            z=z,
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=text,
            texttemplate="%{text}",
            xgap=2,
            ygap=2,
            hovertemplate="%{y} × %{x}: r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    return figure_json(fig)
