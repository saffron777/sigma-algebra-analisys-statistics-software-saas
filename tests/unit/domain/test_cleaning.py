"""Missing values and outliers against pandas, scikit-learn and published critical values."""

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer

from sigma.domain.analysis.cleaning import grubbs_critical
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def clean(analysis_id, frame, roles=None, params=None):
    return run("transformar", analysis_id, frame, roles, params)


# ------------------------------------------------------------------ missing values
def test_simple_replacements_match_pandas(frame):
    column = frame["m"]
    for method, fill in (("mean", column.mean()), ("median", column.median())):
        out = clean("missing", frame, {"cols": ["m"]}, {"method": method})
        result = offered(out)
        assert result["m"].to_numpy() == pytest.approx(column.fillna(fill).to_numpy())
        assert out.result["missing_before"] == column.isna().sum() and out.result["missing_after"] == 0
    constant = offered(clean("missing", frame, {"cols": ["m"]}, {"method": "constant", "value": "0,5"}))
    assert constant["m"].isna().sum() == 0 and (constant.loc[frame["m"].isna(), "m"] == 0.5).all()
    mode = offered(
        clean("missing", frame.assign(c=frame["g"].where(frame["x1"] > 9)), {"cols": ["c"]}, {"method": "mode"})
    )
    assert mode["c"].isna().sum() == 0


def test_neighbour_methods_and_interpolation(frame):
    column = frame["m"]
    assert offered(clean("missing", frame, {"cols": ["m"]}, {"method": "ffill"}))["m"].tolist()[1:] == pytest.approx(
        column.ffill().tolist()[1:], nan_ok=True
    )
    assert offered(clean("missing", frame, {"cols": ["m"]}, {"method": "bfill"}))[
        "m"
    ].dropna().tolist() == pytest.approx(column.bfill().dropna().tolist())
    line = offered(clean("missing", frame, {"cols": ["m"]}, {"method": "interpolate"}))["m"]
    assert line.to_numpy() == pytest.approx(column.interpolate(limit_direction="both").to_numpy())


def test_missing_values_can_leave_an_indicator(frame):
    result = offered(clean("missing", frame, {"cols": ["m"]}, {"method": "mean", "flag": True}))
    assert result["m_faltante"].tolist() == frame["m"].isna().astype(int).tolist()


def test_rows_and_columns_with_missing_values_can_be_dropped(frame):
    rows = offered(clean("missing", frame, {"cols": ["m"]}, {"method": "drop_rows"}))
    assert len(rows) == frame["m"].notna().sum() and not rows["m"].isna().any()
    sparse = frame.assign(mostly=np.where(np.arange(len(frame)) % 4 == 0, 1.0, np.nan))
    columns = offered(clean("missing", sparse, params={"method": "drop_columns", "threshold": 50}))
    assert "mostly" not in columns.columns and "m" in columns.columns
    all_missing = pd.DataFrame({"a": [np.nan, np.nan], "b": [1.0, np.nan]})
    with pytest.raises(InvalidInputError, match="ninguna"):
        clean("missing", all_missing, {"cols": ["a"]}, {"method": "drop_rows"})


def test_missing_value_methods_check_their_inputs(frame):
    with pytest.raises(InvalidInputError, match="Valor de reemplazo"):
        clean("missing", frame, {"cols": ["m"]}, {"method": "constant"})
    with pytest.raises(InvalidInputError, match="no es un número"):
        clean("missing", frame, {"cols": ["m"]}, {"method": "constant", "value": "abc"})
    with pytest.raises(InvalidInputError, match="numéricas"):
        clean("missing", frame.assign(c=frame["g"].where(frame["x1"] > 9)), {"cols": ["c"]}, {"method": "mean"})
    nothing = clean("missing", frame, {"cols": ["x1"]}, {"method": "mean"})
    assert any(b["type"] == "alert" for b in nothing.blocks)


# ------------------------------------------------------------------ multivariate imputation
def test_knn_imputation_matches_scikit_learn(frame):
    out = clean("impute_model", frame, {"cols": ["m"], "predictors": ["x1", "x3"]}, {"method": "knn", "k": 5})
    result = offered(out)
    assert result["m"].isna().sum() == 0 and out.result["imputed"] == frame["m"].isna().sum()
    kept = frame["m"].notna()
    assert result.loc[kept, "m"].to_numpy() == pytest.approx(
        frame.loc[kept, "m"].to_numpy()
    )  # observed data is never touched
    data = frame[["m", "x1", "x3"]]
    mean, sd = data.mean(), data.std()
    expected = (
        KNNImputer(n_neighbors=5, weights="distance").fit_transform((data - mean) / sd) * sd.to_numpy()
        + mean.to_numpy()
    )
    assert result["m"].to_numpy() == pytest.approx(expected[:, 0])


def test_mice_recovers_a_relationship_better_than_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=400)
    truth = 2 * x + rng.normal(scale=0.3, size=400)
    holes = rng.choice(400, 60, replace=False)
    data = pd.DataFrame({"x": x, "v": truth})
    data.loc[holes, "v"] = np.nan
    result = offered(clean("impute_model", data, {"cols": ["v"], "predictors": ["x"]}, {"method": "mice"}))
    mice_error = np.abs(result.loc[holes, "v"].to_numpy() - truth[holes]).mean()
    mean_error = np.abs(data["v"].mean() - truth[holes]).mean()
    assert mice_error < 0.4 * mean_error
    with pytest.raises(InvalidInputError, match="no tienen valores faltantes"):
        clean("impute_model", data.dropna(), {"cols": ["v"]}, {"method": "knn"})


# ------------------------------------------------------------------ outliers
def test_the_iqr_rule_finds_the_extreme_values(frame):
    out = clean("outliers", frame, {"cols": ["o"]}, {"method": "iqr", "threshold": 1.5, "action": "flag"})
    result = offered(out)
    q1, q3 = frame["o"].quantile([0.25, 0.75])
    expected = (frame["o"] < q1 - 1.5 * (q3 - q1)) | (frame["o"] > q3 + 1.5 * (q3 - q1))
    assert result["o_atipico"].tolist() == expected.astype(int).tolist()
    assert out.result["outliers"] == expected.sum() >= 4
    assert result["o"].equals(frame["o"])  # flagging does not change the data


def test_the_other_detectors(frame):
    z = clean("outliers", frame, {"cols": ["o"]}, {"method": "zscore", "threshold": 3, "action": "flag"})
    zscores = (frame["o"] - frame["o"].mean()).abs() / frame["o"].std()
    assert offered(z)["o_atipico"].sum() == (zscores > 3).sum()
    mad = clean("outliers", frame, {"cols": ["o"]}, {"method": "mad", "threshold": 3.5, "action": "flag"})
    median = frame["o"].median()
    robust = 0.6745 * (frame["o"] - median).abs() / (frame["o"] - median).abs().median()
    assert offered(mad)["o_atipico"].sum() == (robust > 3.5).sum()


def test_grubbs_critical_values_match_the_published_table():
    # two-sided alpha = 0.05 (NIST handbook table)
    assert grubbs_critical(10, 0.05) == pytest.approx(2.290, abs=0.001)
    assert grubbs_critical(20, 0.05) == pytest.approx(2.709, abs=0.001)
    assert grubbs_critical(30, 0.05) == pytest.approx(2.908, abs=0.001)


def test_grubbs_removes_outliers_one_at_a_time():
    rng = np.random.default_rng(3)
    values = np.append(rng.normal(50, 2, 30), [80.0, 20.0])
    data = pd.DataFrame({"v": values})
    out = clean("outliers", data, {"cols": ["v"]}, {"method": "grubbs", "alpha": 0.05, "action": "flag"})
    flagged = offered(out)["v_atipico"].to_numpy().astype(bool)
    assert set(np.flatnonzero(flagged)) == {30, 31}
    quiet = clean("outliers", pd.DataFrame({"v": rng.normal(0, 1, 30)}), {"cols": ["v"]}, {"method": "grubbs"})
    assert quiet.result["outliers"] == 0


def test_outlier_actions(frame):
    flagged_rows = clean("outliers", frame, {"cols": ["o"]}, {"action": "flag"}).result["rows_affected"]
    removed = offered(clean("outliers", frame, {"cols": ["o"]}, {"action": "remove"}))
    assert len(removed) == len(frame) - flagged_rows
    missing = offered(clean("outliers", frame, {"cols": ["o"]}, {"action": "missing"}))
    assert missing["o"].isna().sum() == flagged_rows
    wins = offered(clean("outliers", frame, {"cols": ["o"]}, {"action": "winsorize"}))
    q1, q3 = frame["o"].quantile([0.25, 0.75])
    assert wins["o"].max() == pytest.approx(q3 + 1.5 * (q3 - q1)) and wins["o"].min() == pytest.approx(
        q1 - 1.5 * (q3 - q1)
    )
    report = clean("outliers", frame, {"cols": ["o"]}, {"action": "report"})
    pd.testing.assert_frame_equal(offered(report), frame)  # still on offer, unchanged
    assert [b["type"] for b in report.blocks].count("chart") == 1


def test_outliers_need_enough_data():
    with pytest.raises(InvalidInputError, match="muy pocos"):
        clean("outliers", pd.DataFrame({"v": [1.0, 2.0, 3.0]}), {"cols": ["v"]})
