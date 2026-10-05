"""Control de calidad (SPC): límites de control y capacidad, contrastados contra las fórmulas de Shewhart."""

import numpy as np
import pandas as pd
import pytest

from sigma.domain.analysis.spc import _shewhart
from tests.analysis_data import run


def test_imr_limits_match_the_shewhart_formula():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = run("calidad", "control_chart", frame, {"measure": "x"}, {"chart": "imr"})
    mean, mr_bar = out.result["mean"], out.result["mr_bar"]
    assert mean == pytest.approx(3.0)
    assert mr_bar == pytest.approx(1.0)  # all moving ranges are 1
    assert out.result["ucl"] == pytest.approx(mean + 2.66 * mr_bar)
    assert out.result["lcl"] == pytest.approx(mean - 2.66 * mr_bar)


def test_xbar_r_limits_use_a2_d3_d4():
    # three subgroups of size 4 with known means and ranges
    rng = np.random.default_rng(0)
    g = np.repeat(["A", "B", "C"], 4)
    x = np.array([2, 3, 4, 5, 10, 11, 12, 13, 6, 7, 8, 9], dtype=float)
    frame = pd.DataFrame({"x": x, "g": g})
    out = run("calidad", "control_chart", frame, {"measure": "x", "subgroup": "g"}, {"chart": "xbar_r"})
    # each subgroup range is 3, so R̄ = 3 and X̄̄ = mean of [3.5, 11.5, 7.5] = 7.5
    a2, _d3, d4, _a3, _b3, _b4, _d2 = _shewhart(4)
    assert out.result["grand_mean"] == pytest.approx(7.5)
    assert out.result["ucl"] == pytest.approx(7.5 + a2 * 3.0)
    assert out.result["lcl"] == pytest.approx(7.5 - a2 * 3.0)


def test_capability_matches_manual_calculation():
    frame = pd.DataFrame({"x": [8.0, 9.0, 10.0, 11.0, 12.0]})
    out = run("calidad", "capability", frame, {"measure": "x"}, {"usl": 13, "lsl": 7})
    mr_bar = np.abs(np.diff(frame["x"].to_numpy())).mean()  # all diffs are 1
    within = mr_bar / 1.128
    assert out.result["cp"] == pytest.approx((13 - 7) / (6 * within))
    assert out.result["cpk"] == pytest.approx(min((13 - 10) / (3 * within), (10 - 7) / (3 * within)))


def test_cusum_detects_a_mean_shift():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)])
    frame = pd.DataFrame({"x": x})
    out = run("calidad", "control_chart", frame, {"measure": "x"}, {"chart": "cusum", "h": 5, "k": 0.5})
    assert out.result["alarmed"] is True


def test_gage_rr_decomposes_a_known_measurement_system():
    rng = np.random.default_rng(0)
    parts = np.repeat(["P1", "P2", "P3", "P4"], 6)  # 4 piezas × 3 operadores × 2 réplicas
    ops = np.tile(np.repeat(["A", "B", "C"], 2), 4)
    effect = {"P1": 0.0, "P2": 5.0, "P3": 10.0, "P4": 15.0}
    x = np.array([effect[p] for p in parts]) + rng.normal(0, 0.2, len(parts))
    frame = pd.DataFrame({"x": x, "part": parts, "op": ops})
    out = run("calidad", "gage_rr", frame, {"measure": "x", "part": "part", "operator": "op"}, {})
    assert out.result["percent_grr"] < 30  # la variación de piezas domina
    assert out.result["ndc"] >= 5
