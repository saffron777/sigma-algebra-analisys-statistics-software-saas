"""The analysis catalogue: validation of what the user sends, the registry and the two first analyses."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.power import TTestIndPower, TTestPower
from statsmodels.tsa.stattools import acf

from sigma.domain.analysis import registry
from sigma.domain.analysis.modules import BY_KEY, CATALOG_MODULES
from sigma.domain.analysis.output import Output, P
from sigma.domain.analysis.spec import AnalysisSpec, FieldSpec, RoleSpec, prepare_inputs
from sigma.domain.errors import ComputationError, InvalidInputError
from tests.analysis_data import standard_frame


def spec_with(roles=(), params=()) -> AnalysisSpec:
    return AnalysisSpec("x", "X", "G", "series", "d", lambda *_: Output("x").build(), roles=roles, params=params)


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


# ------------------------------------------------------------------ roles
def test_roles_must_exist_and_have_the_right_kind(frame):
    spec = spec_with(
        roles=(
            RoleSpec("num", "Numérica", "numeric"),
            RoleSpec("cat", "Grupo", "categorical"),
            RoleSpec("when", "Tiempo", "datetime"),
        )
    )
    good = prepare_inputs(spec, frame, {"num": "y", "cat": "g", "when": "t"}, {})[0]
    assert good == {"num": "y", "cat": "g", "when": "t"}
    for bad in (
        {"num": "g", "cat": "g", "when": "t"},
        {"num": "y", "cat": "y", "when": "t"},
        {"num": "y", "cat": "g", "when": "y"},
    ):
        with pytest.raises(InvalidInputError):
            prepare_inputs(spec, frame, bad, {})
    with pytest.raises(InvalidInputError, match="no existe"):
        prepare_inputs(spec, frame, {"num": "nada", "cat": "g", "when": "t"}, {})
    with pytest.raises(InvalidInputError, match="Elija"):
        prepare_inputs(spec, frame, {"num": "y"}, {})


def test_multiple_and_optional_roles(frame):
    spec = spec_with(
        roles=(
            RoleSpec("vars", "Variables", "numeric", multiple=True, min_count=2, max_count=3),
            RoleSpec("g", "Grupo", "categorical", optional=True),
        )
    )
    roles, _ = prepare_inputs(spec, frame, {"vars": ["x1", "x2"]}, {})
    assert roles == {"vars": ["x1", "x2"]}
    for bad in (["x1"], ["x1", "x2", "x3", "y"], ["x1", "x1"], []):
        with pytest.raises(InvalidInputError):
            prepare_inputs(spec, frame, {"vars": bad}, {})


def test_a_dataset_is_needed_to_check_columns():
    with pytest.raises(InvalidInputError):
        prepare_inputs(spec_with(roles=(RoleSpec("a", "A"),)), None, {"a": "y"}, {})


# ------------------------------------------------------------------ parameters
def test_parameters_are_coerced_defaulted_and_range_checked():
    spec = spec_with(
        params=(
            FieldSpec("alpha", "α", "number", 0.05, minimum=0.001, maximum=0.5),
            FieldSpec("lags", "Rezagos", "integer", 10, minimum=1),
            FieldSpec("kind", "Tipo", "select", "a", (("a", "A"), ("b", "B"))),
            FieldSpec("flag", "Opción", "boolean", False),
            FieldSpec("many", "Varios", "multiselect", ["a"], (("a", "A"), ("b", "B"))),
            FieldSpec("name", "Nombre", "text", optional=True),
        )
    )
    assert prepare_inputs(spec, None, {}, {})[1] == {
        "alpha": 0.05,
        "lags": 10,
        "kind": "a",
        "flag": False,
        "many": ["a"],
        "name": None,
    }
    clean = prepare_inputs(
        spec,
        None,
        {},
        {"alpha": "0.1", "lags": "12", "kind": "b", "flag": "true", "many": ["a", "b"], "name": " hola "},
    )[1]
    assert clean == {"alpha": 0.1, "lags": 12, "kind": "b", "flag": True, "many": ["a", "b"], "name": "hola"}
    for bad in (
        {"alpha": 0},
        {"alpha": "x"},
        {"alpha": float("nan")},
        {"lags": 1.5},
        {"lags": 0},
        {"kind": "z"},
        {"many": ["z"]},
    ):
        with pytest.raises(InvalidInputError):
            prepare_inputs(spec, None, {}, bad)


def test_a_required_parameter_without_default_is_refused():
    with pytest.raises(InvalidInputError, match="Indique"):
        prepare_inputs(spec_with(params=(FieldSpec("v", "Valor", "number"),)), None, {}, {})


def test_hidden_parameters_are_ignored():
    spec = spec_with(
        params=(
            FieldSpec("mode", "Modo", "select", "a", (("a", "A"), ("b", "B"))),
            FieldSpec("extra", "Extra", "number", 5, minimum=1, visible_if=("mode", ("b",))),
        )
    )
    assert prepare_inputs(spec, None, {}, {"mode": "a", "extra": -3})[1] == {
        "mode": "a",
        "extra": 5,
    }  # not shown: not checked
    with pytest.raises(InvalidInputError):
        prepare_inputs(spec, None, {}, {"mode": "b", "extra": -3})


# ------------------------------------------------------------------ registry and output
def test_every_module_has_a_unique_key_and_the_registry_knows_them():
    assert len({m.key for m in CATALOG_MODULES}) == len(CATALOG_MODULES) == len(BY_KEY)
    for module in CATALOG_MODULES:
        assert module.path.startswith("/") and module.criteria and module.description
        registry.catalog(module.key)
    with pytest.raises(InvalidInputError):
        registry.catalog("inventado")
    with pytest.raises(InvalidInputError):
        registry.get("series", "inventado")


def test_registering_twice_or_in_an_unknown_module_fails():
    spec = registry.get("potencia", "t_power")
    with pytest.raises(ValueError):
        registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(AnalysisSpec("y", "Y", "G", "no_existe", "d", spec.compute))


def test_the_output_builder_collects_blocks_datasets_and_numbers():
    out = (
        Output("Título", "sub")
        .heading("H")
        .tiles([("a", 1.5)])
        .table("T", ["x"], [[P(0.01)]])
        .text("p")
        .alert("info", "m")
        .dataset("k", "Etiqueta", pd.DataFrame({"a": [1]}))
        .set(n=3)
        .build()
    )
    assert [b["type"] for b in out.blocks] == ["heading", "tiles", "table", "text", "alert"]
    assert (
        out.datasets["k"].label == "Etiqueta" and out.result == {"n": 3} and isinstance(out.blocks[2]["rows"][0][0], P)
    )


# ------------------------------------------------------------------ the first analyses
def test_t_power_matches_the_textbook_and_statsmodels():
    spec = registry.get("potencia", "t_power")
    base = {"design": "two_sample", "solve_for": "n", "effect_size": 0.5, "power": 0.8, "alpha": 0.05}

    def run(**changes):
        roles, params = prepare_inputs(spec, None, {}, {**base, **changes})
        return spec.compute(pd.DataFrame(), roles, params).result

    result = run()
    assert result["n"] == 64 and 0.8 <= result["power"] < 0.81  # d = 0.5, power 0.8 → 64 per group
    assert run(design="one_sample")["n"] == int(
        np.ceil(TTestPower().solve_power(effect_size=0.5, power=0.8, alpha=0.05))
    )
    assert run(design="paired", alternative="larger")["n"] < run(design="paired")["n"]
    power = run(solve_for="power", n=30)["power"]
    assert power == pytest.approx(TTestIndPower().solve_power(effect_size=0.5, nobs1=30, alpha=0.05), abs=1e-9)
    effect = run(solve_for="effect", n=64)["effect_size"]
    assert effect == pytest.approx(0.5, abs=0.01)
    assert (
        run(design="two_sample", ratio=2.0)["n"] < run()["n"] * 1.0 + 1
    )  # unequal allocation needs a smaller first group


def test_t_power_refuses_nonsense():
    spec = registry.get("potencia", "t_power")
    roles, params = prepare_inputs(
        spec, None, {}, {"design": "one_sample", "solve_for": "n", "effect_size": 0, "power": 0.8}
    )
    with pytest.raises(InvalidInputError):
        spec.compute(pd.DataFrame(), roles, params)
    with pytest.raises(InvalidInputError):
        prepare_inputs(spec, None, {}, {"design": "x"})


def test_acf_pacf_match_statsmodels(frame):
    spec = registry.get("series", "acf_pacf")
    roles, params = prepare_inputs(spec, frame, {"series": "s", "time": "t"}, {"lags": 12})
    out = spec.compute(frame, roles, params)
    expected = acf(frame["s"].to_numpy(), nlags=12, fft=True, result_object=False)
    assert out.result["acf"] == pytest.approx(expected[1:6].tolist())
    assert out.result["ljung_box_p"] < 0.001  # a trending, seasonal series is autocorrelated
    assert [b["type"] for b in out.blocks].count("chart") == 3
    short = frame.head(5)
    with pytest.raises(InvalidInputError):
        spec.compute(short, *prepare_inputs(spec, short, {"series": "s"}, {}))
    shuffled = frame.sample(frac=1, random_state=1)
    ordered = spec.compute(shuffled, *prepare_inputs(spec, shuffled, {"series": "s", "time": "t"}, {"lags": 12}))
    assert ordered.result["acf"] == pytest.approx(out.result["acf"])  # the time column puts the rows in order


def test_computation_errors_are_domain_errors():
    assert issubclass(ComputationError, Exception)
