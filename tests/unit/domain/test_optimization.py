"""Optimización: la solución de la mochila continua y entera es la esperada."""

import pandas as pd

from tests.analysis_data import run


def test_knapsack_continuous_chooses_by_value_per_unit():
    frame = pd.DataFrame({"value": [3.0, 2.0], "cost": [1.0, 1.0], "bound": [1.0, 1.0]})
    out = run("optimizacion", "lp", frame, {"value": "value", "cost": "cost", "bound": "bound"}, {"budget": 1, "sense": "maximize"})
    assert out.result["objective"] == 3.0  # elegir el elemento 1 (valor 3) consume todo el presupuesto


def test_knapsack_integer_is_0_1():
    frame = pd.DataFrame({"value": [3.0, 2.0], "cost": [1.0, 1.0], "bound": [1.0, 1.0]})
    out = run(
        "optimizacion",
        "lp",
        frame,
        {"value": "value", "cost": "cost", "bound": "bound"},
        {"budget": 1, "sense": "maximize", "integer": True},
    )
    assert out.result["objective"] == 3.0
    x = out.datasets["solucion"].frame["x"].to_numpy()
    assert set(x) <= {0.0, 1.0}
