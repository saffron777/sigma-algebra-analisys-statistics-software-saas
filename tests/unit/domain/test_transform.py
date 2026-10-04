"""Creating and modifying variables: every result against pandas / numpy / scipy."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sigma.domain.analysis.recoding import parse_rules, recode_series
from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def transform(analysis_id, frame, roles=None, params=None):
    return run("transformar", analysis_id, frame, roles, params)


# ------------------------------------------------------------------ compute
def test_compute_adds_the_formula_as_a_column(frame):
    out = transform("compute", frame, params={"name": "ratio", "expression": "x1 / x2 + log(x3 + 1)"})
    result = offered(out)
    assert list(result.columns) == [*frame.columns, "ratio"]
    expected = frame["x1"] / frame["x2"] + np.log(frame["x3"] + 1)
    assert result["ratio"].to_numpy() == pytest.approx(expected.to_numpy())
    assert out.result == {**out.result, "column": "ratio", "n_missing": 0}
    pd.testing.assert_frame_equal(result[frame.columns], frame)  # the rest is untouched


def test_compute_reports_missing_results_and_protects_existing_columns(frame):
    out = transform("compute", frame, params={"name": "z", "expression": "where(x1 > 10, x1, nan)"})
    assert out.result["n_missing"] == int((frame["x1"] <= 10).sum())
    assert any(b["type"] == "alert" for b in out.blocks)
    with pytest.raises(InvalidInputError, match="Ya existe"):
        transform("compute", frame, params={"name": "X1", "expression": "x1 * 2"})  # case-insensitive clash
    replaced = offered(transform("compute", frame, params={"name": "x1", "expression": "x1 * 2", "overwrite": True}))
    assert replaced["x1"].to_numpy() == pytest.approx(2 * frame["x1"].to_numpy()) and replaced.shape == frame.shape
    for bad in ("", "nope + 1", "x1.mean()"):
        with pytest.raises(InvalidInputError):
            transform("compute", frame, params={"name": "n", "expression": bad})
    with pytest.raises(InvalidInputError, match="nombre"):
        transform("compute", frame, params={"name": "  ", "expression": "x1"})


def test_compute_text_dates_and_booleans(frame):
    texts = offered(transform("compute", frame, params={"name": "etiqueta", "expression": 'concat(g, "-", h)'}))
    assert texts["etiqueta"].iloc[0] == f"{frame['g'].iloc[0]}-{frame['h'].iloc[0]}"
    flags = offered(transform("compute", frame, params={"name": "alto", "expression": "x1 > mean(x1)"}))
    assert flags["alto"].dtype == bool and flags["alto"].sum() == (frame["x1"] > frame["x1"].mean()).sum()
    years = offered(transform("compute", frame, params={"name": "anio", "expression": "year(t)"}))
    assert years["anio"].iloc[0] == 2000


# ------------------------------------------------------------------ recode
def test_recode_rules_are_parsed_and_applied():
    series = pd.Series([1, 2, 3, 4, 5, 6, np.nan, 8])
    result, counts = recode_series(series, "1, 2 -> 10\n3..5 -> 20\n>=8 -> 30\nmissing -> 0\nelse -> 99")
    assert result.tolist() == [10, 10, 20, 20, 20, 99, 0, 30]
    assert [n for _, n in counts] == [2, 3, 1, 1, 1]
    kept, _ = recode_series(series, "1 -> 100")
    assert kept.tolist()[:3] == [100.0, 2.0, 3.0] and math.isnan(kept.iloc[6])  # unmatched values are copied
    text, _ = recode_series(pd.Series(["a", "b", None, "c"]), '"a", b -> grupo uno\nmissing -> sin dato\nelse -> otros')
    assert text.tolist() == ["grupo uno", "grupo uno", "sin dato", "otros"]
    mixed, _ = recode_series(series, "<=2 -> bajo")
    assert mixed.tolist()[:3] == ["bajo", "bajo", "3"] and pd.isna(mixed.iloc[6])


def test_recode_rule_errors_are_explained():
    for text, fragment in (
        ("", "al menos una regla"),
        ("1 2", "valor -> resultado"),
        ("1 ->", "valor -> resultado"),
        ("a..b -> 1", "rango"),
        ("else -> 1\nelse -> 2", "else"),
    ):
        with pytest.raises(InvalidInputError, match=fragment):
            parse_rules(text)
    with pytest.raises(InvalidInputError, match="numéricas"):
        recode_series(pd.Series(["a", "b"]), "1..5 -> x")


def test_recode_page_analysis(frame):
    out = transform("recode", frame, {"col": "g"}, {"name": "grupo2", "rules": "A -> primero\nB, C -> resto"})
    result = offered(out)
    expected = frame["g"].map({"A": "primero", "B": "resto", "C": "resto"})
    assert result["grupo2"].tolist() == expected.tolist()
    assert sum(n for _, n in out.result["rules"]) == len(frame)


# ------------------------------------------------------------------ bin
def test_equal_width_bins_match_pandas_cut(frame):
    result = offered(
        transform("bin", frame, {"col": "x1"}, {"name": "caja", "method": "width", "bins": 5, "labels": "number"})
    )
    expected = pd.cut(frame["x1"], 5, labels=False, include_lowest=True) + 1
    assert result["caja"].astype(int).tolist() == expected.tolist()


def test_quantile_and_custom_bins(frame):
    quartiles = offered(transform("bin", frame, {"col": "x1"}, {"name": "q", "method": "quantile", "bins": 4}))
    counts = quartiles["q"].value_counts()
    assert counts.max() - counts.min() <= 2 and quartiles["q"].nunique() == 4
    custom = offered(
        transform("bin", frame, {"col": "x1"}, {"name": "c", "method": "custom", "cuts": "8 12", "labels": "interval"})
    )
    assert set(custom["c"].dropna()) == {"≤ 8", "(8, 12]", "> 12"}
    assert (custom["c"] == "(8, 12]").sum() == ((frame["x1"] > 8) & (frame["x1"] <= 12)).sum()
    with pytest.raises(InvalidInputError, match="corte"):
        transform("bin", frame, {"col": "x1"}, {"name": "c", "method": "custom", "cuts": "uno dos"})
    constant = frame.assign(k=5.0)
    with pytest.raises(InvalidInputError, match="constante"):
        transform("bin", constant, {"col": "k"}, {"name": "c", "method": "width"})


# ------------------------------------------------------------------ dummies
def test_dummies_match_get_dummies(frame):
    out = transform("dummies", frame, {"cols": ["g", "h"]}, {"drop_first": False})
    result = offered(out)
    expected = pd.get_dummies(frame[["g", "h"]], prefix_sep="_", dtype=int)
    assert list(result.columns) == [*frame.columns, *expected.columns]
    assert result[list(expected.columns)].to_numpy() == pytest.approx(expected.to_numpy())
    reduced = offered(transform("dummies", frame, {"cols": ["g"]}, {"drop_first": True, "keep_original": False}))
    assert "g" not in reduced.columns and "g_A" not in reduced.columns and {"g_B", "g_C"} <= set(reduced.columns)


def test_dummies_refuse_too_many_levels_and_name_clashes(frame):
    wide = frame.assign(id=[f"c{i}" for i in range(len(frame))])
    with pytest.raises(InvalidInputError, match="demasiadas"):
        transform("dummies", wide, {"cols": ["id"]})
    clash = frame.assign(g_B=1)
    with pytest.raises(InvalidInputError, match="Ya existe"):
        transform("dummies", clash, {"cols": ["g"]}, {"drop_first": False})


# ------------------------------------------------------------------ standardise and rescale
def test_zscore_and_minmax_match_scipy(frame):
    result = offered(transform("standardize", frame, {"cols": ["x1", "x2"]}, {"method": "zscore"}))
    assert result["x1_z"].to_numpy() == pytest.approx(stats.zscore(frame["x1"], ddof=1))
    assert result["x1_z"].mean() == pytest.approx(0, abs=1e-12) and result["x1_z"].std() == pytest.approx(1)
    scaled = offered(transform("standardize", frame, {"cols": ["x1"]}, {"method": "minmax"}))
    assert scaled["x1_mm"].min() == 0 and scaled["x1_mm"].max() == 1
    robust = offered(transform("standardize", frame, {"cols": ["x1"]}, {"method": "robust"}))
    assert robust["x1_rob"].median() == pytest.approx(0)
    replaced = offered(transform("standardize", frame, {"cols": ["x1"]}, {"method": "center", "mode": "replace"}))
    assert replaced.shape == frame.shape and replaced["x1"].mean() == pytest.approx(0, abs=1e-12)


def test_standardise_reports_the_parameters_used_and_refuses_constants(frame):
    out = transform("standardize", frame, {"cols": ["x1"]}, {"method": "zscore"})
    parameters = out.result["parameters"]["x1"]
    assert parameters["shift"] == pytest.approx(frame["x1"].mean()) and parameters["divisor"] == pytest.approx(
        frame["x1"].std()
    )
    with pytest.raises(InvalidInputError, match="constante"):
        transform("standardize", frame.assign(k=3.0), {"cols": ["k"]})
    with pytest.raises(InvalidInputError, match="Ya existe"):
        transform("standardize", frame.assign(x1_z=0), {"cols": ["x1"]}, {"method": "zscore"})


# ------------------------------------------------------------------ mathematical transformations
def test_simple_functions_match_numpy(frame):
    for name, suffix, function in (
        ("log", "_ln", np.log),
        ("sqrt", "_raiz", np.sqrt),
        ("square", "_cuad", np.square),
        ("log1p", "_log1p", np.log1p),
    ):
        result = offered(transform("math", frame, {"cols": ["x3"]}, {"function": name}))
        assert result[f"x3{suffix}"].to_numpy() == pytest.approx(function(frame["x3"].to_numpy()))


def test_boxcox_and_yeo_johnson_estimate_lambda_like_scipy(frame):
    positive = frame.assign(skewed=np.exp(frame["x2"] / 3))
    out = transform("math", positive, {"cols": ["skewed"]}, {"function": "boxcox"})
    values, lam = stats.boxcox(positive["skewed"].to_numpy())
    assert out.result["lambdas"]["skewed"] == pytest.approx(lam)
    assert offered(out)["skewed_bc"].to_numpy() == pytest.approx(values)
    yj = transform("math", frame, {"cols": ["x1"]}, {"function": "yeojohnson"})
    assert yj.result["lambdas"]["x1"] == pytest.approx(stats.yeojohnson(frame["x1"].to_numpy())[1])


def test_functions_outside_their_domain_leave_missing_values_or_refuse(frame):
    centred = frame.assign(c=frame["x1"] - frame["x1"].mean())
    result = offered(transform("math", centred, {"cols": ["c"]}, {"function": "log"}))
    assert result["c_ln"].isna().sum() == (centred["c"] <= 0).sum()
    with pytest.raises(InvalidInputError, match="positivos"):
        transform("math", centred, {"cols": ["c"]}, {"function": "boxcox"})
    with pytest.raises(InvalidInputError, match="ningún valor"):
        transform("math", frame.assign(n=-1.0), {"cols": ["n"]}, {"function": "sqrt"})


# ------------------------------------------------------------------ lags, differences, rolling
def test_time_operations_match_pandas(frame):
    s = frame["s"]
    for operation, k, expected in (
        ("lag", 2, s.shift(2)),
        ("lead", 1, s.shift(-1)),
        ("diff", 12, s.diff(12)),
        ("pct_change", 1, s.pct_change()),
        ("log_diff", 1, np.log(s).diff()),
        ("cumsum", 1, s.cumsum()),
        ("rolling_mean", 6, s.rolling(6).mean()),
        ("rolling_std", 6, s.rolling(6).std()),
    ):
        result = offered(
            transform("time_ops", frame, {"cols": ["s"], "time": "t"}, {"operation": operation, "periods": k})
        )
        new = [c for c in result.columns if c not in frame.columns][0]
        assert result[new].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True), operation
    ewma = offered(transform("time_ops", frame, {"cols": ["s"]}, {"operation": "ewma", "alpha": 0.4}))
    assert ewma["s_ewma"].to_numpy() == pytest.approx(s.ewm(alpha=0.4, adjust=False).mean().to_numpy())


def test_time_operations_respect_the_date_order_and_the_groups(frame):
    shuffled = frame.sample(frac=1, random_state=3).reset_index(drop=True)
    result = offered(transform("time_ops", shuffled, {"cols": ["s"], "time": "t"}, {"operation": "lag", "periods": 1}))
    by_time = result.sort_values("t")["s_lag1"].to_numpy()
    assert by_time[1:] == pytest.approx(frame["s"].to_numpy()[:-1])  # lagged along the dates, rows keep their place
    assert result["t"].tolist() == shuffled["t"].tolist()
    grouped = offered(
        transform("time_ops", frame, {"cols": ["s"], "time": "t", "by": "g"}, {"operation": "diff", "periods": 1})
    )
    expected = frame.sort_values("t").groupby("g")["s"].diff().reindex(frame.index)
    assert grouped["s_d1"].to_numpy() == pytest.approx(expected.to_numpy(), nan_ok=True)


# ------------------------------------------------------------------ dates and text
def test_date_parts_match_pandas(frame):
    result = offered(
        transform(
            "date_parts",
            frame,
            {"col": "t"},
            {"parts": ["year", "quarter", "month", "month_name", "weekday", "is_weekend", "year_month", "week"]},
        )
    )
    dt = frame["t"].dt
    assert result["t_year"].tolist() == dt.year.tolist() and result["t_quarter"].tolist() == dt.quarter.tolist()
    assert result["t_weekday"].tolist() == (dt.weekday + 1).tolist()
    assert result["t_is_weekend"].tolist() == (dt.weekday >= 5).astype(int).tolist()
    assert result["t_year_month"].iloc[0] == "2000-01" and result["t_month_name"].iloc[0] == "enero"
    assert result["t_week"].tolist() == dt.isocalendar().week.tolist()
    with pytest.raises(InvalidInputError, match="al menos una"):
        transform("date_parts", frame, {"col": "t"}, {"parts": []})


def test_text_cleaning():
    dirty = pd.DataFrame({"c": ["  Árbol  Grande ", "ÁRBOL grande", None, "otro"]})
    out = transform(
        "text_ops",
        dirty,
        {"cols": ["c"]},
        {"operations": ["trim", "collapse", "lower", "no_accents"], "mode": "replace"},
    )
    assert offered(out)["c"].tolist()[:2] == ["arbol grande", "arbol grande"]
    assert pd.isna(offered(out)["c"].iloc[2])
    with pytest.raises(InvalidInputError, match="operación"):
        transform("text_ops", dirty, {"cols": ["c"]}, {"operations": []})
