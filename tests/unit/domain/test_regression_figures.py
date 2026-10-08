"""The report figures of a big dataset stay small: one point per row would be tens of megabytes of JSON.

A regression on 1.4 M rows sent a 55.8 MB answer, 57.8 MB of it a single Q-Q figure with one marker per row, which
froze the browser tab. Every chart that draws one marker per observation must down-sample first.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sigma.domain.statistics.regression import fit_regression
from sigma.presentation.web.charts import regression_plots as plots
from sigma.presentation.web.charts.common import MAX_POINTS

ROWS = 200_000
MAX_FIGURE_BYTES = 2 * 1024 * 1024  # a figure a browser can draw without effort


@pytest.fixture(scope="module")
def big_result():
    rng = np.random.default_rng(11)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "y": rng.normal(0, 1, ROWS),
            "x1": rng.normal(0, 1, ROWS),
            "x2": rng.gamma(2, 1, ROWS),
        }
    )
    return fit_regression(frame, "y", ["x1", "x2"])


@pytest.mark.parametrize(
    "builder",
    [
        plots.observed_vs_fitted_figure,
        plots.residuals_vs_fitted_figure,
        plots.residual_qq_figure,
        plots.scale_location_figure,
        plots.cooks_figure,
    ],
)
def test_every_figure_of_a_big_dataset_stays_small(big_result, builder):
    figure = builder(big_result)
    size = len(json.dumps(figure))
    assert size < MAX_FIGURE_BYTES, f"{builder.__name__} pesa {size / 1e6:.1f} MB"
    for trace in figure["data"]:
        for key in ("x", "y"):
            values = trace.get(key)
            if isinstance(values, list):
                assert len(values) <= max(MAX_POINTS, 3000), f"{builder.__name__}: {len(values)} puntos en {key}"


def test_the_qq_chart_keeps_the_shape_of_the_distribution(big_result):
    """Sampling must not flatten the chart: the extremes and the straight reference line are still there."""
    figure = plots.residual_qq_figure(big_result)
    points = figure["data"][0]
    xs, ys = np.asarray(points["x"]), np.asarray(points["y"])
    assert len(xs) > 100 and len(xs) < ROWS / 10
    assert float(np.min(xs)) < -3 and float(np.max(xs)) > 3  # the tails survived
    assert ys[0] < ys[-1] and float(np.max(ys)) > 2  # still increasing, as a Q-Q must be
    line = figure["data"][1]
    assert len(line["x"]) == 2  # the reference line still joins the real ends
