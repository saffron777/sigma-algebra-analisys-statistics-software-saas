"""Diseño de experimentos: el diseño factorial se genera y el modelo recupera los efectos conocidos."""

import itertools

import numpy as np
import pandas as pd
import pytest

from tests.analysis_data import run


def test_full_factorial_generates_the_coded_design():
    out = run("doe", "factorial", pd.DataFrame(), {}, {"factors": 3, "center_points": 0})
    design = out.datasets["diseño"].frame
    assert len(design) == 8 and list(design.columns) == ["A", "B", "C"]
    assert set(design["A"]) == {-1.0, 1.0}


def test_factorial_recovers_the_known_effects():
    design = np.array(list(itertools.product([-1, 1], repeat=2)), dtype=float)
    y = 10 + 3 * design[:, 0] - 2 * design[:, 1]  # y = 10 + 3A − 2B
    frame = pd.DataFrame({"y": y})
    out = run("doe", "factorial", frame, {"response": "y"}, {"factors": 2})
    # params: [Intercept, A, B, A:B]
    coef = out.result["coefficients"]
    assert coef[0] == pytest.approx(10)
    assert coef[1] == pytest.approx(3)
    assert coef[2] == pytest.approx(-2)
    assert coef[3] == pytest.approx(0)


def test_response_surface_has_axial_and_center_runs():
    out = run("doe", "response_surface", pd.DataFrame(), {}, {"factors": 2, "center_points": 3})
    design = out.datasets["diseño"].frame
    assert len(design) == 4 + 4 + 3  # factorial + axial + centres
