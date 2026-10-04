"""The graph catalogue: the numbers behind each chart against scipy / statsmodels / pandas, and the figures."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from sigma.domain.errors import InvalidInputError
from sigma.presentation.web.charts.render import render_chart
from tests.analysis_data import run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def graph(analysis_id, frame, roles=None, params=None):
    return run("graficos", analysis_id, frame, roles, params)


def figures(output):
    return [render_chart(block["spec"]) for block in output.blocks if block["type"] == "chart"]


def table(output, title):
    block = next(b for b in output.blocks if b["type"] == "table" and b["title"] == title)
    return block["columns"], block["rows"]


def value(cell):
    return getattr(cell, "value", cell)


# ------------------------------------------------------------------ histogram
def test_the_histogram_summarises_and_draws_a_density(frame):
    out = graph("histogram", frame, {"var": "x1"}, {"curve": "normal", "bins": 12})
    assert out.result["mean"] == pytest.approx(frame["x1"].mean()) and out.result["skew"] == pytest.approx(
        stats.skew(frame["x1"])
    )
    (figure,) = figures(out)
    bars = [t for t in figure["data"] if t["type"] == "bar"]
    width = bars[0]["width"]
    assert sum(bars[0]["y"]) * width == pytest.approx(1.0)  # a density integrates to one
    curve = next(t for t in figure["data"] if t["type"] == "scatter")
    assert curve["name"] == "Normal ajustada" and max(curve["y"]) == pytest.approx(
        stats.norm.pdf(frame["x1"].mean(), frame["x1"].mean(), frame["x1"].std()), rel=0.02
    )
    _, rows = table(out, "Estadísticos")
    assert value(rows[0][-1]) > 0 and rows[0][1] == len(frame)


def test_groups_share_their_bins_and_can_be_split_in_panels(frame):
    overlay = figures(graph("histogram", frame, {"var": "y", "group": "g"}, {"curve": "none", "norm": "count"}))[0]
    bars = [t for t in overlay["data"] if t["type"] == "bar"]
    assert [t["name"] for t in bars] == ["A", "B", "C"]
    assert bars[0]["x"] == bars[1]["x"] and sum(sum(t["y"]) for t in bars) == len(frame)
    panels = figures(graph("histogram", frame, {"var": "y", "group": "g"}, {"layout": "panels", "curve": "kde"}))[0]
    assert len({trace["xaxis"] for trace in panels["data"] if trace["type"] == "bar"}) == 3
    percent = figures(graph("histogram", frame, {"var": "y"}, {"norm": "percent", "curve": "none"}))[0]
    assert sum(percent["data"][0]["y"]) == pytest.approx(100)


def test_histogram_needs_enough_data(frame):
    with pytest.raises(InvalidInputError, match="al menos"):
        graph("histogram", frame.head(2), {"var": "x1"})
    with pytest.raises(InvalidInputError, match="categorías"):
        graph("histogram", frame.assign(id=range(len(frame))).astype({"id": str}), {"var": "x1", "group": "id"})


# ------------------------------------------------------------------ box plots
def test_box_statistics_match_pandas(frame):
    out = graph("boxplot", frame, {"vars": ["y"], "group": "g"}, {"style": "box"})
    columns, rows = table(out, "Estadísticos")
    for row in rows:
        values = frame.loc[frame["g"] == row[0], "y"]
        assert (
            row[1] == len(values)
            and row[2] == pytest.approx(values.mean())
            and row[6] == pytest.approx(values.median())
        )
        q1, q3 = values.quantile([0.25, 0.75])
        outside = ((values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))).sum()
        assert row[-1] == outside
    assert len(figures(out)[0]["data"]) == 3


def test_box_styles_and_several_variables(frame):
    many = graph("boxplot", frame, {"vars": ["x1", "x2", "x3"]}, {"style": "strip"})
    traces = figures(many)[0]["data"]
    assert [t["name"] for t in traces] == ["x1", "x2", "x3"] and traces[0]["boxpoints"] == "all"
    assert figures(graph("boxplot", frame, {"vars": ["y"]}, {"style": "violin"}))[0]["data"][0]["type"] == "violin"
    with pytest.raises(InvalidInputError, match="una sola variable"):
        graph("boxplot", frame, {"vars": ["x1", "x2"], "group": "g"})


# ------------------------------------------------------------------ probability plot
def test_the_probability_plot_matches_scipy(frame):
    out = graph("probability", frame, {"var": "y"}, {"distribution": "norm"})
    mean, sd = stats.norm.fit(frame["y"])
    assert out.result["parameters"] == pytest.approx([mean, sd])
    assert out.result["correlation"] > 0.99
    anderson = stats.anderson(frame["y"].to_numpy(), dist="norm", method="interpolate")
    _, rows = table(out, "Bondad de ajuste")
    assert rows[-1][1] == pytest.approx(anderson.statistic)
    ks = stats.kstest(frame["y"], stats.norm(mean, sd).cdf)
    assert out.result["ks_p"] == pytest.approx(ks.pvalue)
    points = figures(out)[0]["data"][0]
    assert points["x"] == sorted(points["x"]) and points["y"] == sorted(points["y"])


def test_the_probability_plot_tells_a_poor_fit_from_a_good_one():
    rng = np.random.default_rng(4)
    skewed = pd.DataFrame({"v": rng.exponential(2.0, 300)})
    normal = graph("probability", skewed, {"var": "v"}, {"distribution": "norm"})
    exponential = graph("probability", skewed, {"var": "v"}, {"distribution": "expon"})
    assert normal.result["correlation"] < exponential.result["correlation"] and exponential.result["ks_p"] > 0.05
    assert normal.result["ks_p"] < 0.001
    with pytest.raises(InvalidInputError, match="positivos"):
        graph("probability", skewed.assign(v=skewed["v"] - 3), {"var": "v"}, {"distribution": "lognorm"})


# ------------------------------------------------------------------ scatter
def test_scatter_correlations_and_fit_match_references(frame):
    out = graph("scatter", frame, {"x": "x1", "y": "y"}, {"fit": "linear"})
    r = stats.pearsonr(frame["x1"], frame["y"])
    assert out.result["pearson"] == pytest.approx(r.statistic)
    columns, rows = table(out, "Asociación")
    assert rows[0][4] == pytest.approx(stats.spearmanr(frame["x1"], frame["y"]).statistic)
    assert rows[0][6] == pytest.approx(r.statistic**2)
    figure = figures(out)[0]
    fit = next(t for t in figure["data"] if t.get("name") == "Ajuste")
    slope, intercept = np.polyfit(frame["x1"], frame["y"], 1)
    assert np.polyval([slope, intercept], fit["x"][10]) == pytest.approx(fit["y"][10])
    assert any(t.get("fill") == "tonexty" for t in figure["data"])  # the confidence band


def test_polynomial_and_lowess_fits(frame):
    quad = graph("scatter", frame, {"x": "x1", "y": "y"}, {"fit": "quadratic"})
    _, rows = table(quad, "Asociación")
    coefficients = np.polyfit(frame["x1"], frame["y"], 2)
    predicted = np.polyval(coefficients, frame["x1"])
    assert rows[0][6] == pytest.approx(
        1 - ((frame["y"] - predicted) ** 2).sum() / ((frame["y"] - frame["y"].mean()) ** 2).sum()
    )
    smooth = graph("scatter", frame, {"x": "x1", "y": "y"}, {"fit": "lowess", "smooth": 0.5})
    line = next(t for t in figures(smooth)[0]["data"] if t.get("name") == "Ajuste")
    reference = lowess(frame["y"], frame["x1"], frac=0.5)
    assert line["y"] == pytest.approx(reference[:, 1].tolist())


def test_bubbles_groups_and_panels(frame):
    bubbles = graph("scatter", frame, {"x": "x1", "y": "y", "size": "x3"}, {"fit": "none"})
    sizes = figures(bubbles)[0]["data"][0]["marker"]["size"]
    assert min(sizes) == pytest.approx(6) and max(sizes) == pytest.approx(36)
    grouped = graph("scatter", frame, {"x": "x1", "y": "y", "group": "g"}, {"fit": "linear"})
    assert len(table(grouped, "Asociación")[1]) == 3
    panels = figures(
        graph("scatter", frame, {"x": "x1", "y": "y", "group": "g"}, {"fit": "linear", "layout": "panels"})
    )[0]
    assert len({trace["xaxis"] for trace in panels["data"]}) == 3
    with pytest.raises(InvalidInputError, match="al menos"):
        graph("scatter", frame.head(3), {"x": "x1", "y": "y"})


def test_the_scatter_matrix(frame):
    out = graph("splom", frame, {"vars": ["y", "x1", "x2"], "group": "g"})
    matrix = frame[["y", "x1", "x2"]].corr()
    columns, rows = table(out, "Correlaciones de Pearson")
    assert value(rows[0][2]) == pytest.approx(matrix.at["y", "x1"])
    traces = figures(out)[0]["data"]
    assert [t["type"] for t in traces] == ["splom"] * 3 and len(traces[0]["dimensions"]) == 3
    with pytest.raises(InvalidInputError):
        graph("splom", frame, {"vars": ["y"]})


# ------------------------------------------------------------------ lines
def test_line_plot_scales(frame):
    base = graph("line_plot", frame, {"ys": ["s"], "x": "t"}, {"scale": "index100"})
    line = figures(base)[0]["data"][0]
    assert line["y"][0] == pytest.approx(100) and line["x"][0] == "2000-01-01"
    z = figures(graph("line_plot", frame, {"ys": ["s", "y"], "x": "t"}, {"scale": "zscore"}))[0]["data"]
    assert [np.nanmean(t["y"]) for t in z] == pytest.approx([0, 0], abs=1e-9)
    smooth = figures(graph("line_plot", frame, {"ys": ["s"], "x": "t"}, {"smooth": 6}))[0]["data"]
    assert smooth[1]["name"].endswith("(media móvil 6)") and smooth[1]["y"][5] == pytest.approx(
        frame["s"].iloc[:6].mean()
    )


def test_line_plot_sorts_by_x_and_splits_by_group(frame):
    shuffled = frame.sample(frac=1, random_state=1)
    ordered = figures(graph("line_plot", shuffled, {"ys": ["s"], "x": "t"}, {"sort": True}))[0]["data"][0]["x"]
    assert ordered == sorted(ordered)
    by_group = figures(graph("line_plot", frame, {"ys": ["s"], "x": "t", "group": "g"}))[0]["data"]
    assert [t["name"] for t in by_group] == ["A", "B", "C"]
    with pytest.raises(InvalidInputError, match="una sola variable"):
        graph("line_plot", frame, {"ys": ["s", "y"], "group": "g"})


# ------------------------------------------------------------------ bars, Pareto, pie
def test_bar_counts_percent_and_means(frame):
    counts = graph("bar", frame, {"category": "g"}, {"stat": "count"})
    _, rows = table(counts, "Valores")
    assert {r[0]: r[1] for r in rows} == frame["g"].value_counts().to_dict()
    percent = graph("bar", frame, {"category": "g"}, {"stat": "percent"})
    assert sum(r[1] for r in table(percent, "Valores")[1]) == pytest.approx(100)
    means = graph("bar", frame, {"category": "g", "value": "y"}, {"stat": "mean", "error": "ci"})
    figure = figures(means)[0]["data"][0]
    group = frame.loc[frame["g"] == "A", "y"]
    half = stats.t.ppf(0.975, len(group) - 1) * group.std() / np.sqrt(len(group))
    assert figure["y"][0] == pytest.approx(group.mean()) and figure["error_y"]["array"][0] == pytest.approx(half)


def test_bars_can_be_sorted_grouped_and_need_their_value(frame):
    desc = figures(graph("bar", frame, {"category": "g", "value": "y"}, {"stat": "sum", "sort": "desc"}))[0]["data"][0][
        "y"
    ]
    assert desc == sorted(desc, reverse=True)
    stacked = figures(graph("bar", frame, {"category": "g", "group": "h"}, {"stacked": True}))[0]
    assert len(stacked["data"]) == 2 and stacked["layout"]["barmode"] == "stack"
    with pytest.raises(InvalidInputError, match="columna numérica"):
        graph("bar", frame, {"category": "g"}, {"stat": "mean"})


def test_pareto_cumulative_percentages(frame):
    out = graph("pareto", frame, {"category": "g", "value": "count"})
    _, rows = table(out, "Categorías")
    cumulative = [r[3] for r in rows]
    assert cumulative[-1] == pytest.approx(100) and cumulative == sorted(cumulative)
    totals = frame.groupby("g")["count"].sum().sort_values(ascending=False)
    assert [r[0] for r in rows] == list(totals.index) and rows[0][1] == totals.iloc[0]
    figure = figures(out)[0]
    assert figure["data"][1]["yaxis"] == "y2" and figure["layout"]["yaxis2"]["range"] == [0, 105]
    many = frame.assign(k=(frame.index % 20).astype(str))
    grouped = graph("pareto", many, {"category": "k"}, {"top": 5})
    assert table(grouped, "Categorías")[1][-1][0] == "Otros"
    with pytest.raises(InvalidInputError, match="no negativos"):
        graph("pareto", frame.assign(n=-1.0), {"category": "g", "value": "n"})


def test_pie_shares(frame):
    out = graph("pie", frame, {"category": "g"}, {"donut": True})
    figure = figures(out)[0]["data"][0]
    assert figure["type"] == "pie" and figure["hole"] == 0.45 and sum(figure["values"]) == len(frame)


# ------------------------------------------------------------------ interval plot
def test_interval_plot_matches_scipy(frame):
    out = graph("interval", frame, {"var": "y", "group": "g"}, {"confidence": 95})
    _, rows = table(out, "Intervalos de confianza")
    for row in rows:
        values = frame.loc[frame["g"] == row[0], "y"]
        low, high = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
        assert (row[5], row[6]) == pytest.approx((low, high))
    figure = figures(out)[0]
    assert sum(1 for t in figure["data"] if t["type"] == "scatter") == 3
    wide = graph("interval", frame, {"var": "y", "group": "g"}, {"confidence": 99})
    assert table(wide, "Intervalos de confianza")[1][0][6] > rows[0][6]
    assert any(b["type"] == "alert" for b in out.blocks)  # B is well above C: non-overlapping intervals are pointed out


# ------------------------------------------------------------------ heat map and surfaces
def test_correlation_heat_map(frame):
    names = ["y", "x1", "x2", "x3"]
    out = graph("heatmap", frame, {"vars": names}, {"method": "spearman", "cluster": True})
    heat = figures(out)[0]["data"][0]
    matrix = frame[names].corr(method="spearman")
    order = heat["x"]
    assert sorted(order) == sorted(names) and heat["y"] == order
    assert heat["z"][1][0] == pytest.approx(matrix.loc[order[1], order[0]])
    _, rows = table(out, "Pares con mayor correlación")
    strongest = max(
        ((a, b) for i, a in enumerate(names) for b in names[i + 1 :]), key=lambda ab: abs(matrix.at[ab[0], ab[1]])
    )
    assert {rows[0][0], rows[0][1]} == set(strongest)
    assert value(rows[0][3]) == pytest.approx(matrix.at[strongest[0], strongest[1]])
    with pytest.raises(InvalidInputError):
        graph("heatmap", frame, {"vars": ["y"]})


def test_contours_surfaces_and_density(frame):
    contour = figures(graph("contour", frame, {"x": "x1", "y": "x2", "z": "y"}, {"grid": 20}))[0]["data"][0]
    assert contour["type"] == "contour" and len(contour["z"]) == 20 and len(contour["z"][0]) == 20
    surface = figures(
        graph(
            "contour",
            frame,
            {"x": "x1", "y": "x2", "z": "y"},
            {"kind": "surface", "interpolation": "cubic", "grid": 15},
        )
    )[0]["data"][0]
    assert surface["type"] == "surface"
    density = graph("contour", frame, {"x": "x1", "y": "x2"}, {"grid": 25})
    peak = figures(density)[0]["data"][0]["z"]
    assert max(max(v for v in row if v is not None) for row in peak) > 0
    assert "densidad" in density.subtitle


def test_scatter_3d_has_one_trace_per_group(frame):
    traces = figures(graph("scatter3d", frame, {"x": "x1", "y": "x2", "z": "y", "group": "g"}))[0]["data"]
    assert [t["type"] for t in traces] == ["scatter3d"] * 3 and sum(len(t["x"]) for t in traces) == len(frame)


def test_missing_values_are_reported_not_hidden(frame):
    out = graph("scatter", frame, {"x": "m", "y": "y"}, {"fit": "none"})
    assert any(b["type"] == "alert" and "omitieron" in b["message"] for b in out.blocks)
