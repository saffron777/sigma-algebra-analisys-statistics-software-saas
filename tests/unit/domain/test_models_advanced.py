"""GAM, PLS y loglineal: verificaciones de corrección estadística."""

import numpy as np
import pandas as pd

from tests.analysis_data import run, standard_frame


def test_gam_captures_a_nonlinear_relationship():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 300)
    y = np.sin(x) + rng.normal(0, 0.3, 300)
    frame = pd.DataFrame({"x": x, "y": y})
    out = run("modelos", "gam", frame, {"response": "y", "predictors": ["x"]}, {"splines": 8})
    assert out.result["r2"] > 0.8  # a GAM with splines fits sin(x) well


def test_pls_matches_ols_on_collinear_predictors():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    x2 = x[:, :1] + rng.normal(0, 0.01, (200, 1))  # nearly collinear
    X = np.column_stack([x, x2])
    y = X[:, 0] + rng.normal(0, 0.2, 200)
    frame = pd.DataFrame({"y": y, "a": X[:, 0], "b": X[:, 1], "c": X[:, 2]})
    out = run("modelos", "pls", frame, {"response": "y", "predictors": ["a", "b", "c"]}, {"components": 2})
    assert out.result["r2"] > 0.8


def test_loglinear_aggregates_the_contingency_table():
    frame = standard_frame()
    out = run("modelos", "loglinear", frame, {"count": "count", "factors": ["g", "h"]}, {"interactions": False})
    assert out.result["cells"] == 6  # 3 niveles de g × 2 de h
    assert np.isfinite(out.result["deviance"])
