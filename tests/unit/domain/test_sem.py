"""Ecuaciones estructurales: el análisis de rutas recupera los coeficientes de ruta esperados."""

import numpy as np
import pandas as pd
import pytest

from tests.analysis_data import run


def _coef_table(out):
    return next(b for b in out.blocks if b["type"] == "table")


def test_path_analysis_standardized_coefficient_equals_the_correlation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    y = 2 * x + rng.normal(0, 0.5, 300)
    frame = pd.DataFrame({"y": y, "x": x})
    out = run("sem", "path_analysis", frame, {"variables": ["y", "x"]}, {"model": "y ~ x"})
    table = _coef_table(out)
    std_coef = table["rows"][1][2]  # fila de «x», columna «Estandarizado (ruta)»
    assert std_coef == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=0.01)
    assert std_coef > 0.9


def test_path_analysis_handles_two_predictors():
    rng = np.random.default_rng(1)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 1.5 * x1 - 2 * x2 + rng.normal(0, 0.5, 200)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    out = run("sem", "path_analysis", frame, {"variables": ["y", "x1", "x2"]}, {"model": "y ~ x1 + x2"})
    table = _coef_table(out)
    assert table["rows"][1][2] > 0  # x1 -> efecto positivo
    assert table["rows"][2][2] < 0  # x2 -> efecto negativo
