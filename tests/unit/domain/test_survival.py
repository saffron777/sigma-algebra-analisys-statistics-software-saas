"""Supervivencia: Kaplan-Meier, log-rank, Cox y ajuste paramétrico, contrastados contra cálculos de referencia."""

import numpy as np

from sigma.domain.analysis.survival import _km, _logrank, _median
from tests.analysis_data import run, standard_frame


def test_kaplan_meier_matches_a_manual_calculation():
    time = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    event = np.array([1, 1, 0, 1, 1, 0], dtype=int)
    times, surv, at_risk = _km(time, event)
    # t=1: 5/6 · t=2: (5/6)(4/5)=4/6 · t=4: (4/6)(2/3)=4/9 · t=5: (4/9)(1/2)=2/9
    assert np.allclose(times, [1, 2, 4, 5])
    assert np.allclose(surv, [5 / 6, 4 / 6, 4 / 9, 2 / 9])
    assert at_risk == [6, 5, 3, 2]
    assert _median(times, surv) == 4  # first S <= 0.5 is at t=4


def test_logrank_detects_a_group_difference():
    rng = np.random.default_rng(0)
    n = 300
    group = rng.choice([0, 1], n)
    time = rng.exponential(scale=np.where(group == 0, 8.0, 20.0), size=n)  # group 1 survives longer
    event = (rng.uniform(size=n) < 0.8).astype(int)
    _stat, _df, pvalue, observed, expected = _logrank(time, event, group, "logrank")
    assert pvalue < 1e-4
    assert observed[0] > expected[0]  # group 0 has more events than expected under the null


def test_logrank_finds_no_difference_within_one_population():
    rng = np.random.default_rng(1)
    n = 400
    group = rng.choice([0, 1], n)
    time = rng.exponential(scale=15.0, size=n)
    event = (rng.uniform(size=n) < 0.8).astype(int)
    _stat, _df, pvalue, _observed, _expected = _logrank(time, event, group, "logrank")
    assert pvalue > 0.05


def test_cox_recovers_the_known_hazard_ratio():
    frame = standard_frame()
    out = run("supervivencia", "cox", frame, {"time": "tm", "event": "ev", "predictors": ["x1"]}, {})
    hr = out.result["hazard_ratios"][0]
    assert 0.95 < hr < 1.25  # true HR = exp(0.08) ≈ 1.083


def test_parametric_weibull_gives_a_reasonable_median():
    frame = standard_frame()
    out = run("supervivencia", "parametric", frame, {"time": "tm", "event": "ev"}, {"distribution": "weibull"})
    assert 3 < out.result["median"] < 15
    assert "fitted" in out.datasets and len(out.datasets["fitted"].frame) > 0
