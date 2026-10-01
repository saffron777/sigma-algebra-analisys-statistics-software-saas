import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sigma.domain.errors import InvalidInputError
from sigma.domain.statistics.anova import run_anova


@pytest.fixture
def three_groups():
    rng = np.random.default_rng(7)
    a, b, c = rng.normal(10, 2, 20), rng.normal(12, 2, 22), rng.normal(10.5, 2, 18)
    frame = pd.DataFrame({"y": np.r_[a, b, c], "g": ["A"] * 20 + ["B"] * 22 + ["C"] * 18})
    return frame, (a, b, c)


def test_one_way_matches_scipy(three_groups):
    frame, samples = three_groups
    result = run_anova(frame, "y", ["g"])
    expected = stats.f_oneway(*samples)
    between = result.table[0]
    assert between.f_value == pytest.approx(expected.statistic)
    assert between.p_value == pytest.approx(expected.pvalue)
    assert (between.df, result.table[1].df, result.table[2].df) == (2, 57, 59)
    assert between.sum_sq + result.table[1].sum_sq == pytest.approx(result.table[2].sum_sq)
    assert between.eta_sq == pytest.approx(between.sum_sq / result.table[2].sum_sq)
    assert 0 <= between.omega_sq <= between.eta_sq


def test_one_way_extras_match_references(three_groups):
    frame, samples = three_groups
    result = run_anova(frame, "y", ["g"])
    assert result.kruskal["p_value"] == pytest.approx(stats.kruskal(*samples).pvalue)
    tukey = pairwise_tukeyhsd(frame["y"], frame["g"])
    ours = {(r.group1, r.group2): r for r in result.post_hoc}
    assert len(ours) == 3
    assert ours[("A", "B")].p_adj == pytest.approx(tukey.pvalues[0])
    assert ours[("A", "B")].mean_diff == pytest.approx(tukey.meandiffs[0])
    assert result.welch["p_value"] < 0.05
    names = {a.test for a in result.assumptions}
    assert {"Shapiro-Wilk", "Levene (Brown-Forsythe)", "Bartlett"} <= names
    assert [g.group for g in result.groups] == ["A", "B", "C"]


@pytest.fixture
def balanced_design():
    """3x2 balanced design, 10 replicates per cell, strong main effects and no interaction."""
    rng = np.random.default_rng(11)
    rows = []
    for a, effect_a in zip(["a1", "a2", "a3"], [0, 4, 8], strict=True):
        for b, effect_b in zip(["b1", "b2"], [0, 3], strict=True):
            for value in rng.normal(50 + effect_a + effect_b, 2, 10):
                rows.append({"y": value, "A": a, "B": b})
    return pd.DataFrame(rows)


def test_two_way_matches_hand_computed_balanced_sums_of_squares(balanced_design):
    frame = balanced_design
    result = run_anova(frame, "y", ["A", "B"], ss_type=2)
    grand = frame["y"].mean()
    ss_a = 20 * ((frame.groupby("A")["y"].mean() - grand) ** 2).sum()
    ss_b = 30 * ((frame.groupby("B")["y"].mean() - grand) ** 2).sum()
    rows = {r.source: r for r in result.table}
    assert rows["A"].sum_sq == pytest.approx(ss_a)
    assert rows["B"].sum_sq == pytest.approx(ss_b)
    assert rows["A"].p_value < 1e-6 and rows["B"].p_value < 1e-6
    assert rows["A × B"].p_value > 0.01
    assert rows["Total"].df == 59 and rows["Error (residual)"].df == 54
    assert [r.source for r in result.table] == ["A", "B", "A × B", "Error (residual)", "Total"]
    assert len(result.groups) == 6
    assert result.post_hoc[0].group1.startswith("A: ")


def test_two_way_runs_on_unbalanced_data(sales_frame):
    result = run_anova(sales_frame, "ventas", ["region", "canal"], ss_type=3)
    assert result.table[-1].df == len(sales_frame) - 1


def test_two_way_additive_has_no_interaction_row(sales_frame):
    result = run_anova(sales_frame, "ventas", ["region", "canal"], interaction=False)
    assert [row.source for row in result.table][:3] == ["region", "canal", "Error (residual)"]


@pytest.mark.parametrize("ss_type", [1, 2, 3])
def test_all_ss_types_run(sales_frame, ss_type):
    result = run_anova(sales_frame, "ventas", ["region", "canal"], ss_type=ss_type)
    assert result.ss_type == ss_type and all(math.isfinite(t.statistic) for t in result.f_tests)


def test_validation(three_groups, sales_frame):
    frame, _ = three_groups
    with pytest.raises(InvalidInputError):
        run_anova(frame, "y", [])
    with pytest.raises(InvalidInputError):
        run_anova(frame, "y", ["y"])
    with pytest.raises(InvalidInputError):
        run_anova(frame.assign(g="A"), "y", ["g"])
    with pytest.raises(InvalidInputError):
        run_anova(sales_frame, "region", ["canal"])
    with pytest.raises(InvalidInputError):
        run_anova(frame.iloc[[0, 20, 21, 45, 46]], "y", ["g"])  # groups with one observation
