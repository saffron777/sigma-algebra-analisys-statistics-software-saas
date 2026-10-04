"""Rows and structure: filter, sort, duplicates, samples, partitions, column selection, aggregation and reshaping."""

import pandas as pd
import pytest

from sigma.domain.errors import InvalidInputError
from tests.analysis_data import offered, run, standard_frame


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def manage(analysis_id, frame, roles=None, params=None):
    return run("transformar", analysis_id, frame, roles, params)


# ------------------------------------------------------------------ filter and sort
def test_filter_keeps_or_drops_the_rows_that_meet_the_condition(frame):
    condition = 'x1 > 10 and g == "A"'
    kept = offered(manage("filter", frame, params={"condition": condition}))
    expected = frame[(frame["x1"] > 10) & (frame["g"] == "A")].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, expected)
    dropped = offered(manage("filter", frame, params={"condition": condition, "action": "drop"}))
    assert len(kept) + len(dropped) == len(frame)
    assert list(dropped.index) == list(range(len(dropped)))


def test_filter_needs_a_true_or_false_condition_and_some_rows(frame):
    with pytest.raises(InvalidInputError, match="verdadero o falso"):
        manage("filter", frame, params={"condition": "x1 + 1"})
    with pytest.raises(InvalidInputError, match="Ninguna fila"):
        manage("filter", frame, params={"condition": "x1 > 1000"})
    with pytest.raises(InvalidInputError):
        manage("filter", frame, params={"condition": "x1.mean() > 1"})
    missing = offered(manage("filter", frame, params={"condition": "notna(m)"}))
    assert len(missing) == frame["m"].notna().sum() and not missing["m"].isna().any()


def test_sort_is_stable_and_places_missing_values(frame):
    result = offered(manage("sort", frame, {"cols": ["g", "x1"]}, {"descending": True}))
    expected = frame.sort_values(["g", "x1"], ascending=False, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)
    first = offered(manage("sort", frame, {"cols": ["m"]}, {"missing": "first"}))
    assert first["m"].iloc[: frame["m"].isna().sum()].isna().all()
    last = offered(manage("sort", frame, {"cols": ["m"]}))
    assert last["m"].iloc[-1:].isna().all() and last["m"].dropna().is_monotonic_increasing


# ------------------------------------------------------------------ duplicates
def test_duplicates_are_found_removed_and_flagged():
    data = pd.DataFrame({"a": [1, 1, 2, 3, 3, 3], "b": ["x", "x", "y", "z", "z", "w"]})
    removed = manage("duplicates", data, params={"keep": "first"})
    assert offered(removed).to_dict("list") == data.drop_duplicates().reset_index(drop=True).to_dict("list")
    assert removed.result["duplicates"] == 2
    last = offered(manage("duplicates", data, {"cols": ["a"]}, {"keep": "last"}))
    assert last["b"].tolist() == ["x", "y", "w"]
    none = offered(manage("duplicates", data, {"cols": ["a", "b"]}, {"keep": "none"}))
    assert none.to_dict("list") == {"a": [2, 3], "b": ["y", "w"]}
    flagged = offered(manage("duplicates", data, {"cols": ["a"]}, {"keep": "first", "action": "flag"}))
    assert flagged["duplicado"].tolist() == [0, 1, 0, 0, 1, 1]
    clean = manage("duplicates", pd.DataFrame({"a": [1, 2, 3]}))
    assert clean.result["duplicates"] == 0 and any(b["type"] == "alert" for b in clean.blocks)
    with pytest.raises(InvalidInputError, match="Ya existe"):
        manage("duplicates", data.assign(duplicado=0), params={"action": "flag"})


# ------------------------------------------------------------------ samples
def test_a_sample_is_reproducible_and_keeps_the_original_order(frame):
    one = offered(manage("sample", frame, params={"size": 50, "seed": 5}))
    two = offered(manage("sample", frame, params={"size": 50, "seed": 5}))
    other = offered(manage("sample", frame, params={"size": 50, "seed": 6}))
    pd.testing.assert_frame_equal(one, two)
    assert not one["y"].equals(other["y"]) and len(one) == 50
    assert one["t"].is_monotonic_increasing and one["y"].isin(frame["y"]).all()
    half = offered(manage("sample", frame, params={"mode": "fraction", "size": 0.25}))
    assert len(half) == round(0.25 * len(frame))
    boot = offered(manage("sample", frame, params={"size": 400, "replace": True}))
    assert len(boot) == 400 and boot["t"].duplicated().any()


def test_a_stratified_sample_keeps_the_proportions(frame):
    result = manage("sample", frame, {"strata": "g"}, {"size": 60})
    sample = offered(result)
    assert len(sample) == 60
    proportions = frame["g"].value_counts(normalize=True)
    for level, share in proportions.items():
        assert (sample["g"] == level).sum() == pytest.approx(60 * share, abs=1.01)


def test_sample_sizes_are_checked(frame):
    for params in ({"size": 1000}, {"mode": "fraction", "size": 1.5}, {"size": 0}):
        with pytest.raises(InvalidInputError):
            manage("sample", frame, params=params)


# ------------------------------------------------------------------ partition
def test_a_random_partition_is_disjoint_complete_and_reproducible(frame):
    out = manage("partition", frame, params={"test_fraction": 0.25, "validation_fraction": 0.1, "seed": 9})
    parts = {key: offer.frame for key, offer in out.datasets.items()}
    assert set(parts) == {"train", "test", "validation"}
    assert sum(len(p) for p in parts.values()) == len(frame)
    assert len(parts["test"]) == round(0.25 * len(frame)) and len(parts["validation"]) == round(0.1 * len(frame))
    stamps = [set(p["t"]) for p in parts.values()]
    assert not (stamps[0] & stamps[1]) and not (stamps[0] & stamps[2]) and not (stamps[1] & stamps[2])
    again = manage("partition", frame, params={"test_fraction": 0.25, "validation_fraction": 0.1, "seed": 9})
    pd.testing.assert_frame_equal(parts["test"], again.datasets["test"].frame)
    assert "validation" not in manage("partition", frame).datasets


def test_a_stratified_partition_keeps_each_category_in_every_set(frame):
    out = manage("partition", frame, {"strata": "g"}, {"test_fraction": 0.3})
    train, test = out.datasets["train"].frame, out.datasets["test"].frame
    for level, share in frame["g"].value_counts(normalize=True).items():
        assert (test["g"] == level).mean() == pytest.approx(share, abs=0.05)
        assert (train["g"] == level).mean() == pytest.approx(share, abs=0.05)


def test_a_sequential_partition_puts_the_latest_rows_in_the_test_set(frame):
    shuffled = frame.sample(frac=1, random_state=2).reset_index(drop=True)
    out = manage("partition", shuffled, {"time": "t"}, {"method": "sequential", "test_fraction": 0.2})
    train, test = out.datasets["train"].frame, out.datasets["test"].frame
    assert train["t"].max() < test["t"].min() and len(test) == round(0.2 * len(frame))
    with pytest.raises(InvalidInputError, match="sumar menos"):
        manage("partition", frame, params={"test_fraction": 0.6, "validation_fraction": 0.4})
    with pytest.raises(InvalidInputError, match="10 filas"):
        manage("partition", frame.head(5))


# ------------------------------------------------------------------ select and rename columns
def test_columns_are_kept_in_the_order_given_and_renamed(frame):
    result = offered(
        manage("select_columns", frame, {"cols": ["x1", "y", "g"]}, {"rename": "y -> respuesta\nx1 -> predictor"})
    )
    assert list(result.columns) == ["predictor", "respuesta", "g"]
    assert result["respuesta"].tolist() == frame["y"].tolist()
    dropped = offered(manage("select_columns", frame, {"cols": ["x1", "y"]}, {"mode": "drop"}))
    assert "x1" not in dropped.columns and dropped.shape[1] == frame.shape[1] - 2


def test_column_selection_errors(frame):
    with pytest.raises(InvalidInputError, match="no está"):
        manage("select_columns", frame, {"cols": ["x1"]}, {"rename": "y -> otra"})
    with pytest.raises(InvalidInputError, match="mismo nombre"):
        manage("select_columns", frame, {"cols": ["x1", "y"]}, {"rename": "y -> X1"})
    with pytest.raises(InvalidInputError, match="nombre actual"):
        manage("select_columns", frame, {"cols": ["x1"]}, {"rename": "x1 =>"})
    with pytest.raises(InvalidInputError, match="todas"):
        manage("select_columns", frame, {"cols": list(frame.columns)}, {"mode": "drop"})


# ------------------------------------------------------------------ aggregate
def test_aggregation_matches_groupby(frame):
    out = manage(
        "aggregate",
        frame,
        {"by": ["g", "h"], "values": ["y", "x1"]},
        {"functions": ["count", "mean", "std", "q1", "max"]},
    )
    result = offered(out)
    expected = frame.groupby(["g", "h"])
    assert list(result[["g", "h"]].itertuples(index=False, name=None)) == list(expected.size().index)
    assert result["n"].tolist() == expected.size().tolist()
    assert result["y_mean"].to_numpy() == pytest.approx(expected["y"].mean().to_numpy())
    assert result["x1_std"].to_numpy() == pytest.approx(expected["x1"].std().to_numpy())
    assert result["y_q1"].to_numpy() == pytest.approx(expected["y"].quantile(0.25).to_numpy())
    assert result["x1_count"].tolist() == expected["x1"].count().tolist()
    assert out.result["groups"] == len(result) == 6


def test_aggregation_checks_its_columns(frame):
    with pytest.raises(InvalidInputError, match="a la vez"):
        manage("aggregate", frame, {"by": ["x1"], "values": ["x1"]})
    with pytest.raises(InvalidInputError, match="al menos una función"):
        manage("aggregate", frame, {"by": ["g"], "values": ["y"]}, {"functions": []})


# ------------------------------------------------------------------ reshape
def test_melt_matches_pandas(frame):
    result = offered(
        manage("melt", frame, {"ids": ["g"], "values": ["x1", "x2"]}, {"var_name": "medida", "value_name": "dato"})
    )
    expected = pd.melt(frame, id_vars=["g"], value_vars=["x1", "x2"], var_name="medida", value_name="dato")
    pd.testing.assert_frame_equal(result, expected)
    with pytest.raises(InvalidInputError, match="mismo tipo"):
        manage("melt", frame, {"values": ["x1", "g"]})
    with pytest.raises(InvalidInputError, match="a la vez"):
        manage("melt", frame, {"ids": ["x1"], "values": ["x1", "x2"]})
    with pytest.raises(InvalidInputError, match="distintos"):
        manage("melt", frame, {"values": ["x1", "x2"]}, {"var_name": "v", "value_name": "V"})


def test_pivot_matches_pivot_table(frame):
    result = offered(manage("pivot", frame, {"index": ["h"], "columns": "g", "values": "y"}, {"aggfunc": "median"}))
    expected = frame.pivot_table(index="h", columns="g", values="y", aggfunc="median")
    assert result["h"].tolist() == list(expected.index)
    for level in ("A", "B", "C"):
        assert result[level].to_numpy() == pytest.approx(expected[level].to_numpy())
    with pytest.raises(InvalidInputError, match="distintas"):
        manage("pivot", frame, {"index": ["g"], "columns": "g", "values": "y"})
    wide = frame.assign(id=[f"c{i}" for i in range(len(frame))])
    with pytest.raises(InvalidInputError, match="demasiadas"):
        manage("pivot", wide.assign(id=wide["id"].astype("category")), {"index": ["h"], "columns": "id", "values": "y"})


def test_every_transformation_that_can_replace_declares_it():
    from sigma.domain.analysis import registry

    for spec in registry.catalog("transformar"):
        assert spec.outputs, spec.id
        if spec.id in ("partition", "aggregate", "melt", "pivot"):
            assert not spec.can_replace, spec.id


# ------------------------------------------------------------------ combining two datasets
@pytest.fixture
def reference():
    return pd.DataFrame({"g": ["A", "B", "D"], "peso": [1.5, 2.5, 9.0], "y": [0.0, 0.0, 0.0]})


def test_merge_matches_pandas_for_every_kind_of_join(frame, reference):
    for how in ("inner", "left", "right", "outer"):
        out = manage(
            "merge",
            frame,
            {"keys": ["g"], "other_keys": ["g"]},
            {"other": "@lookup", "how": how, "relation": "m:1"},
        )
        expected = frame.merge(reference.rename(columns={"y": "y_otro"}), how=how, on="g")
        result = offered(out)
        assert len(result) == len(expected), how
        assert sorted(result.columns) == sorted(expected.columns), how
        assert result["peso"].fillna(-1).sort_values().tolist() == expected["peso"].fillna(-1).sort_values().tolist()
    left = manage("merge", frame, {"keys": ["g"], "other_keys": ["g"]}, {"other": "@lookup", "how": "left"})
    assert left.result["right_only"] == 0 and left.result["matched"] == (frame["g"].isin(["A", "B"])).sum()
    assert offered(left)["y_otro"].isna().sum() == (frame["g"] == "C").sum()
    assert offered(left)["y"].tolist() == frame["y"].tolist()  # the main data keeps its own column


def test_merge_origin_column_and_differently_named_keys(frame, reference):
    from sigma.domain.analysis import registry
    from sigma.domain.analysis.spec import prepare_inputs

    renamed = reference.rename(columns={"g": "grupo"})
    spec = registry.get("transformar", "merge")
    params = {"other": "x", "how": "outer", "keep_origin": True}
    roles, clean = prepare_inputs(spec, frame, {"keys": ["g"], "other_keys": ["grupo"]}, params, {"other": renamed})
    result = offered(spec.compute(frame, roles, {**clean, "other": renamed}))
    assert "grupo" not in result.columns and result["g"].isna().sum() == 0  # the unmatched "D" row got its key
    assert set(result["origen"]) == {"en ambos", "solo en este dataset", "solo en el otro"}


def test_merge_protects_against_accidental_row_multiplication(frame):
    repeated = frame[["g"]].head(10)
    from sigma.domain.analysis import registry
    from sigma.domain.analysis.spec import prepare_inputs

    spec = registry.get("transformar", "merge")
    roles, params = prepare_inputs(
        spec, frame, {"keys": ["g"], "other_keys": ["g"]}, {"other": "x"}, {"other": repeated}
    )
    with pytest.raises(InvalidInputError, match="relación"):
        spec.compute(frame, roles, {**params, "other": repeated})
    loose = {**params, "relation": "m:m", "other": repeated}
    assert len(offered(spec.compute(frame, roles, loose))) > len(frame)


def test_merge_checks_its_inputs(frame):
    with pytest.raises(InvalidInputError, match="misma cantidad"):
        manage("merge", frame, {"keys": ["g", "h"], "other_keys": ["g"]}, {"other": "@lookup"})
    with pytest.raises(InvalidInputError, match="tipos distintos"):
        manage("merge", frame, {"keys": ["x1"], "other_keys": ["g"]}, {"other": "@lookup"})
    with pytest.raises(InvalidInputError, match="segundo dataset"):
        manage("merge", frame, {"keys": ["g"], "other_keys": ["nada"]}, {"other": "@lookup"})
    with pytest.raises(InvalidInputError, match="Elija"):
        manage("merge", frame, {"keys": ["g"], "other_keys": ["g"]})
    unmatched = frame.assign(g="Z")
    with pytest.raises(InvalidInputError, match="Ninguna fila"):
        manage("merge", unmatched, {"keys": ["g"], "other_keys": ["g"]}, {"other": "@lookup", "how": "inner"})


def test_append_stacks_rows_and_marks_their_origin(frame):
    out = manage("append", frame, params={"other": "@more", "label_main": "base", "label_other": "extra"})
    result = offered(out)
    assert len(result) == len(frame) + 30 and result["origen"].tolist().count("extra") == 30
    assert result["origen"].iloc[0] == "base" and list(result.index) == list(range(len(result)))
    narrow = frame[["y", "x1", "g"]]
    common = offered(manage("append", narrow, params={"other": "@more", "mode": "common", "source": False}))
    assert list(common.columns) == ["y", "x1", "g"] and len(common) == len(narrow) + 30
    union = offered(manage("append", narrow.assign(solo_aqui=1), params={"other": "@more", "source": False}))
    assert union["solo_aqui"].isna().sum() == 30 and union["x2"].isna().sum() == len(narrow)


def test_append_checks_compatibility(frame):
    with pytest.raises(InvalidInputError, match="ninguna columna"):
        manage("append", pd.DataFrame({"zzz": [1, 2]}), params={"other": "@more"})
    with pytest.raises(InvalidInputError, match="tipos distintos"):
        manage("append", frame.assign(y=frame["y"].astype(str)), params={"other": "@more"})
    with pytest.raises(InvalidInputError, match="Ya existe"):
        manage("append", frame.assign(origen=1), params={"other": "@more"})
