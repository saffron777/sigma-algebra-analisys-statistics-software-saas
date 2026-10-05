"""Cópulas: la dependencia estimada coincide con la esperada para datos correlacionados."""

import numpy as np
import pandas as pd
import pytest

from tests.analysis_data import run


def _correlated(n=600, rho=0.7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y})


def test_gaussian_copula_recovers_tau():
    frame = _correlated(rho=0.7)
    out = run("econometria", "copula", frame, {"x": "x", "y": "y"}, {"family": "gaussian", "simulate": 0})
    assert out.result["tau"] == pytest.approx(2 / np.pi * np.arcsin(0.7), abs=0.05)
    assert out.result["tail_lower"] == 0 and out.result["tail_upper"] == 0


def test_clayton_has_lower_tail_dependence_and_gumbel_upper():
    frame = _correlated(rho=0.7)
    clayton = run("econometria", "copula", frame, {"x": "x", "y": "y"}, {"family": "clayton", "simulate": 0})
    assert clayton.result["tail_lower"] > 0 and clayton.result["tail_upper"] == 0
    gumbel = run("econometria", "copula", frame, {"x": "x", "y": "y"}, {"family": "gumbel", "simulate": 0})
    assert gumbel.result["tail_lower"] == 0 and gumbel.result["tail_upper"] > 0
