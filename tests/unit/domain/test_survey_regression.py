"""Regresión de encuestas: los modelos ponderados se ajustan y reportan coeficientes."""

import numpy as np

from tests.analysis_data import run, standard_frame


def test_survey_ols_fits_with_weights():
    frame = standard_frame()
    out = run("encuestas", "survey_ols", frame, {"response": "y", "predictors": ["x1", "x2"], "weight": "x3"}, {})
    assert 0.5 < out.result["r2"] < 1.0


def test_survey_logistic_reports_odds_ratios():
    frame = standard_frame()
    out = run("encuestas", "survey_logistic", frame, {"response": "h", "predictors": ["x1", "x2"], "weight": "x3"}, {})
    assert np.isfinite(out.result["aic"])
    assert "predicciones" in out.datasets
