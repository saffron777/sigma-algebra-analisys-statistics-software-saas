"""Charts for the regression module."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from sigma.domain.results import RegressionResult
from sigma.presentation.web.charts.common import PALETTE, as_list, figure_json, new_figure, thin

BAND_FILL = "rgba(79,70,229,0.20)"
PRED_FILL = "rgba(79,70,229,0.08)"


def fit_figure(result: RegressionResult) -> dict[str, Any]:
    """Scatter + fitted line with confidence/prediction bands (simple regression)."""
    data = result.plot_data
    x, y, line = data["x"], data["observed"], data["line"]
    pct = f"{100 * result.confidence:g} %"
    fig = new_figure(f"Recta de regresión: {result.response} vs {data['x_name']}", data["x_name"], result.response)
    idx = thin(x)
    fig.add_trace(
        go.Scatter(
            x=as_list(line["grid"]),
            y=as_list(line["pi_high"]),
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(line["grid"]),
            y=as_list(line["pi_low"]),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=PRED_FILL,
            name=f"Banda de predicción {pct}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(line["grid"]),
            y=as_list(line["ci_high"]),
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(line["grid"]),
            y=as_list(line["ci_low"]),
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BAND_FILL,
            name=f"IC de la media {pct}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(x[idx]),
            y=as_list(y[idx]),
            mode="markers",
            name="Observaciones",
            marker={"color": PALETTE[1], "size": 7, "opacity": 0.75},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=as_list(line["grid"]),
            y=as_list(line["fit"]),
            mode="lines",
            name="Recta ajustada",
            line={"color": PALETTE[0], "width": 2.6},
        )
    )
    return figure_json(fig)


def observed_vs_fitted_figure(result: RegressionResult) -> dict[str, Any]:
    observed, fitted = result.plot_data["observed"], result.plot_data["fitted"]
    idx = thin(observed)
    fig = new_figure("Valores observados vs ajustados", "Ajustado", "Observado")
    fig.add_trace(
        go.Scatter(
            x=as_list(fitted[idx]),
            y=as_list(observed[idx]),
            mode="markers",
            name="Observaciones",
            marker={"color": PALETTE[1], "size": 7, "opacity": 0.75},
        )
    )
    low, high = float(min(observed.min(), fitted.min())), float(max(observed.max(), fitted.max()))
    fig.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Ajuste perfecto (y = x)",
            line={"color": "#ef4444", "dash": "dash"},
        )
    )
    return figure_json(fig)


def residuals_vs_fitted_figure(result: RegressionResult) -> dict[str, Any]:
    fitted, resid = result.plot_data["fitted"], result.plot_data["residuals"]
    idx = thin(fitted)
    fig = new_figure("Residuos vs valores ajustados", "Valor ajustado", "Residuo")
    fig.add_trace(
        go.Scatter(
            x=as_list(fitted[idx]),
            y=as_list(resid[idx]),
            mode="markers",
            name="Residuos",
            marker={"color": PALETTE[0], "size": 7, "opacity": 0.75},
        )
    )
    fig.add_hline(y=0, line={"color": "#ef4444", "dash": "dash"})
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def residual_qq_figure(result: RegressionResult) -> dict[str, Any]:
    standardized = result.plot_data["standardized"]
    (osm, osr), (slope, intercept, _) = stats.probplot(standardized, dist="norm")
    fig = new_figure("Q-Q de los residuos estandarizados", "Cuantiles teóricos", "Cuantiles de los residuos")
    fig.add_trace(
        go.Scatter(
            x=as_list(osm), y=as_list(osr), mode="markers", marker={"color": PALETTE[0], "size": 6, "opacity": 0.75}
        )
    )
    ends = np.array([float(np.min(osm)), float(np.max(osm))])
    fig.add_trace(
        go.Scatter(x=as_list(ends), y=as_list(intercept + slope * ends), mode="lines", line={"color": "#ef4444"})
    )
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def scale_location_figure(result: RegressionResult) -> dict[str, Any]:
    fitted, standardized = result.plot_data["fitted"], result.plot_data["standardized"]
    idx = thin(fitted)
    fig = new_figure("Escala-ubicación (homocedasticidad)", "Valor ajustado", "√|residuo estandarizado|")
    fig.add_trace(
        go.Scatter(
            x=as_list(fitted[idx]),
            y=as_list(np.sqrt(np.abs(standardized[idx]))),
            mode="markers",
            marker={"color": PALETTE[2], "size": 7, "opacity": 0.75},
        )
    )
    fig.update_layout(showlegend=False)
    return figure_json(fig)


def cooks_figure(result: RegressionResult) -> dict[str, Any]:
    cooks = result.plot_data["cooks"]
    n = len(cooks)
    idx = thin(cooks)
    fig = new_figure("Distancia de Cook (observaciones influyentes)", "Observación", "Distancia de Cook")
    fig.add_trace(go.Bar(x=(idx + 1).tolist(), y=as_list(cooks[idx]), marker={"color": PALETTE[3]}))
    fig.add_hline(
        y=4 / n, line={"color": "#ef4444", "dash": "dash"}, annotation_text="4/n", annotation_position="top right"
    )
    fig.update_layout(showlegend=False)
    return figure_json(fig)
