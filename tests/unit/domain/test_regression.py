import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from sigma.domain.errors import ComputationError, InvalidInputError
from sigma.domain.statistics.regression import fit_regression


def test_simple_regression_matches_numpy_and_statsmodels(sales_frame):
    result = fit_regression(sales_frame, "ventas", ["precio"])
    slope, intercept = np.polyfit(sales_frame["precio"], sales_frame["ventas"], 1)
    coefficients = {c.term: c for c in result.coefficients}
    assert coefficients["Intercepto"].estimate == pytest.approx(intercept)
    assert coefficients["precio"].estimate == pytest.approx(slope)
    reference = sm.OLS(sales_frame["ventas"], sm.add_constant(sales_frame["precio"])).fit()
    assert result.r_squared == pytest.approx(reference.rsquared)
    assert result.f_p_value == pytest.approx(reference.f_pvalue)
    assert result.is_simple and "line" in result.plot_data
    assert result.coefficients[1].vif == 1.0


def test_multiple_regression_recovers_known_coefficients(sales_frame):
    result = fit_regression(sales_frame, "ventas", ["precio", "publicidad", "competencia"])
    estimates = {c.term: c.estimate for c in result.coefficients}
    assert estimates["precio"] == pytest.approx(2.0, abs=0.15)
    assert estimates["publicidad"] == pytest.approx(3.0, abs=0.2)
    assert estimates["competencia"] == pytest.approx(-1.5, abs=0.3)
    assert result.r_squared > 0.9
    assert not result.is_simple
    assert result.df_model == 3 and result.df_resid == 196
    assert [row.source for row in result.anova] == ["Regresión", "Residual", "Total"]
    assert result.anova[0].sum_sq + result.anova[1].sum_sq == pytest.approx(result.anova[2].sum_sq)
    assert all(c.vif is not None and c.vif < 5 for c in result.coefficients if c.term != "Intercepto")


def test_confidence_intervals_follow_level(sales_frame):
    narrow = fit_regression(sales_frame, "ventas", ["precio"], confidence=0.80).coefficients[1]
    wide = fit_regression(sales_frame, "ventas", ["precio"], confidence=0.99).coefficients[1]
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_categorical_predictor_creates_dummies(sales_frame):
    result = fit_regression(sales_frame, "ventas", ["precio", "region"])
    terms = [c.term for c in result.coefficients]
    assert "region[Norte]" in terms and "region[Sur]" in terms and "region[Centro]" not in terms
    assert any("referencia" in note and "Centro" in note for note in result.notes)
    assert not result.is_simple


def test_prediction_intervals_are_wider_than_mean_intervals(sales_frame):
    new = [{"precio": 50, "region": "Sur"}]
    result = fit_regression(sales_frame, "ventas", ["precio", "region"], predict=new)
    row = result.predictions[0]
    assert row.mean_ci_low < row.prediction < row.mean_ci_high
    assert (row.pred_ci_high - row.pred_ci_low) > (row.mean_ci_high - row.mean_ci_low)
    with pytest.raises(InvalidInputError):
        fit_regression(sales_frame, "ventas", ["precio", "region"], predict=[{"precio": 50, "region": "Oeste"}])


def test_missing_values_are_dropped_and_reported(sales_frame):
    damaged = sales_frame.copy()
    damaged.loc[:9, "precio"] = np.nan
    result = fit_regression(damaged, "ventas", ["precio", "publicidad"])
    assert result.n_dropped == 10 and result.n_obs == 190
    assert any("10 fila" in note for note in result.notes)


def test_no_intercept_model(sales_frame):
    result = fit_regression(sales_frame, "ventas", ["precio"], intercept=False)
    assert all(c.term != "Intercepto" for c in result.coefficients)
    assert not result.has_intercept


def test_diagnostics_and_influence_present(sales_frame):
    result = fit_regression(sales_frame, "ventas", ["precio", "publicidad"])
    names = {d.name for d in result.diagnostics}
    assert {"Durbin-Watson", "Jarque-Bera (normalidad)", "Breusch-Pagan (homocedasticidad)"} <= names
    assert len(result.influential) == 10
    assert result.influential[0].cooks_distance >= result.influential[-1].cooks_distance
    assert result.equation.startswith("ventas = ")


def test_invalid_inputs(sales_frame):
    with pytest.raises(InvalidInputError):
        fit_regression(sales_frame, "ventas", [])
    with pytest.raises(InvalidInputError):
        fit_regression(sales_frame, "ventas", ["ventas"])
    with pytest.raises(InvalidInputError):
        fit_regression(sales_frame, "region", ["precio"])  # non-numeric response
    with pytest.raises(InvalidInputError):
        fit_regression(sales_frame.head(2), "ventas", ["precio", "publicidad"])


def test_perfect_collinearity_is_reported(sales_frame):
    frame = sales_frame.assign(duplicado=sales_frame["precio"] * 2)
    with pytest.raises(ComputationError):
        fit_regression(frame, "ventas", ["precio", "duplicado"])


def test_constant_predictor_is_reported(sales_frame):
    with pytest.raises(ComputationError):
        fit_regression(sales_frame.assign(k=1.0), "ventas", ["precio", "k"])


def test_result_with_pandas_nullable_columns():
    frame = pd.DataFrame({"y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.8], "x": pd.array([1, 2, 3, 4, 5, 6], dtype="Int64")})
    result = fit_regression(frame, "y", ["x"])
    assert result.coefficients[1].estimate == pytest.approx(0.99, abs=0.05)
