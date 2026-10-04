"""The expression evaluator: right answers against pandas/numpy, and hostile expressions that must be refused."""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sigma.domain.analysis import expressions
from sigma.domain.analysis.expressions import FUNCTIONS, REFERENCE, evaluate
from sigma.domain.errors import InvalidInputError


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "b": [2, 0, 3, 4, 5, 6],
            "g": ["x", "y", "x", None, "y", "z"],
            "f": pd.to_datetime(["2020-01-01", "2020-02-15", None, "2021-03-01", "2021-12-31", "2022-06-30"]),
            "Nombre raro": [1, 2, 3, 4, 5, 6],
            "ok": [True, False, True, True, False, True],
        }
    )


def values(expression, frame):
    return evaluate(expression, frame).tolist()


def close(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        if isinstance(want, float) and math.isnan(want):
            assert isinstance(got, float) and math.isnan(got)
        else:
            assert got == pytest.approx(want)


# ------------------------------------------------------------------ the answers
def test_arithmetic_matches_pandas(frame):
    a, b = frame["a"], frame["b"]
    close(values("a + b * 2 - 1", frame), (a + b * 2 - 1).tolist())
    close(values("a / b", frame), (a / b).replace(np.inf, np.nan).tolist())
    close(values("(a - mean(a)) / std(a)", frame), ((a - a.mean()) / a.std()).tolist())
    close(values("a ** 2 + -b", frame), (a**2 - b).tolist())
    close(values("b % 4", frame), (b.astype(float) % 4).tolist())
    close(values("b // 4", frame), (b.astype(float) // 4).tolist())


def test_division_by_zero_and_overflow_give_missing_values(frame):
    assert all(math.isnan(v) for v in values("1 / 0", frame))
    assert math.isnan(values("b / 0", frame)[0])
    assert all(math.isnan(v) for v in values("9 ** 9 ** 9 ** 9", frame))
    assert math.isnan(values("log(b)", frame)[1])  # log(0) = -inf is stored as missing
    result = evaluate("1 / b", frame)
    assert np.isfinite(result.dropna()).all()


def test_comparisons_with_missing_values_are_false(frame):
    assert values("a > 2", frame) == [False, False, False, True, True, True]
    assert values("a != 2", frame) == [True, False, False, True, True, True]
    assert values('g == "x"', frame) == [True, False, True, False, False, False]
    assert values('g != "x"', frame) == [False, True, False, False, True, True]
    assert values("0 < a < 5", frame) == [True, True, False, True, False, False]
    assert values('g in ["x", "z"]', frame) == [True, False, True, False, False, True]
    assert values('f >= "2021-01-01"', frame) == [False, False, False, True, True, True]


def test_conditions_combine(frame):
    assert values("a > 1 and b < 5", frame) == [False, True, False, True, False, False]
    assert values("(a > 1) | (b == 0)", frame) == [False, True, False, True, True, True]
    assert values("not (a > 1)", frame) == [True, False, True, False, False, False]
    assert values("ok & (b > 2)", frame) == [False, False, True, True, False, True]


def test_where_nests_and_keeps_types(frame):
    assert values('where(b > 4, "alto", where(b > 2, "medio", "bajo"))', frame) == [
        "bajo",
        "bajo",
        "medio",
        "medio",
        "alto",
        "alto",
    ]
    assert evaluate("where(b > 2, 1, 0)", frame).dtype.kind == "i"
    close(values("where(a > 2, a, 0)", frame), [0.0, 0.0, 0.0, 4.0, 5.0, 6.0])


def test_statistical_helpers_match_references(frame):
    a = frame["a"]
    close(values("zscore(a)", frame), ((a - a.mean()) / a.std(ddof=1)).tolist())
    close(values("minmax(a)", frame), ((a - a.min()) / (a.max() - a.min())).tolist())
    close(values("rank(a)", frame), a.rank().tolist())
    close(values("cumsum(b)", frame), frame["b"].cumsum().astype(float).tolist())
    close(values("median(a)", frame)[:1], [a.median()])
    close(values("quantile(a, 0.25)", frame)[:1], [a.quantile(0.25)])
    close(values("rowmean(a, b)", frame), frame[["a", "b"]].mean(axis=1).tolist())
    close(values("rowsum(a, b)", frame), frame[["a", "b"]].sum(axis=1, min_count=1).tolist())
    close(values("rownvalid(a, b)", frame), [2, 2, 1, 2, 2, 2])
    close(values("pnorm(0)", frame)[:1], [0.5])
    close(values("qnorm(0.975)", frame)[:1], [1.959963984540054])


def test_time_series_helpers(frame):
    a = frame["a"]
    close(values("lag(a)", frame), a.shift(1).tolist())
    close(values("lead(a, 2)", frame), a.shift(-2).tolist())
    close(values("diff(a)", frame), a.diff().tolist())
    close(
        values("pct_change(b)", frame),
        frame["b"].astype(float).pct_change(fill_method=None).replace(np.inf, np.nan).tolist(),
    )
    close(values("rolling_mean(b, 3)", frame), frame["b"].rolling(3).mean().tolist())
    close(values("log(a) - lag(log(a), 1)", frame), (np.log(a) - np.log(a).shift(1)).tolist())


def test_text_helpers(frame):
    assert values("upper(g)", frame)[:2] == ["X", "Y"]
    assert values('g + "_1"', frame)[:2] == ["x_1", "y_1"]
    assert values('concat(g, "-", b)', frame)[:3] == ["x-2", "y-0", "x-3"]
    assert values('contains(g, "x")', frame) == [True, False, True, False, False, False]
    assert values('replace(g, "x", "w")', frame)[:3] == ["w", "y", "w"]
    assert values("substr(concat(g, g), 2, 1)", frame)[:2] == ["x", "y"]
    close(values("length(g)", frame), [1.0, 1.0, 1.0, math.nan, 1.0, 1.0])
    close(values('to_number(concat("1,5", ""))', frame)[:1], [1.5])


def test_date_helpers(frame):
    close(values("year(f)", frame), [2020, 2020, math.nan, 2021, 2021, 2022])
    close(values("month(f)", frame)[:2], [1, 2])
    close(values("quarter(f)", frame)[:2], [1, 1])
    close(values("days_between(f, add_days(f, 10))", frame)[:2], [10, 10])
    close(values("days_between(f, f)", frame)[:2], [0, 0])


def test_columns_with_spaces_and_constants(frame):
    close(values('col("Nombre raro") * pi', frame)[:2], [math.pi, 2 * math.pi])
    assert values("5", frame) == [5] * 6
    assert values("True", frame) == [True] * 6


def test_the_result_is_aligned_with_the_rows(frame):
    shifted = frame.set_index(pd.Index([10, 20, 30, 40, 50, 60]))
    result = evaluate("a * 2", shifted)
    assert list(result.index) == [10, 20, 30, 40, 50, 60] and result.name is None
    assert len(evaluate("1", frame.iloc[0:0])) == 0


def test_every_function_is_documented():
    documented = " ".join(signature for signature, _ in REFERENCE)
    import re

    words = set(re.findall(r"[a-z_0-9]+", documented))
    assert [name for name in FUNCTIONS if name not in words] == []


def test_wrong_arguments_have_clear_messages(frame):
    for expression, fragment in (
        ("log(a, b)", "argumento"),
        ("where(a, 1, 2)", "condiciones"),
        ('where(a > 1, 1, "x")', "mismo tipo"),
        ("a + g", "texto"),
        ("g > 3", "comparar"),
        ("year(a)", "fecha"),
        ("nope + 1", "nope"),
        ("lag(a, a)", "entero"),
        ("round(a, 99)", "decimales"),
        ("min(a, b)", "minimum"),
        ("isin(a, 3)", "lista"),
        ("", "Escriba"),
        ('f >= "no es fecha"', "fecha"),
    ):
        with pytest.raises(InvalidInputError, match=fragment):
            evaluate(expression, frame)
    with pytest.raises(InvalidInputError, match="Quiso decir"):
        evaluate("sqr(a)", frame)


# ------------------------------------------------------------------ hostile expressions
HOSTILE = [
    "a.__class__",
    "a.mean()",
    "a.values",
    "().__class__.__bases__[0].__subclasses__()",
    "__import__('os').system('echo hi')",
    "__builtins__",
    "open('x')",
    "exec('1')",
    "eval('1')",
    "compile('1', 'x', 'eval')",
    "getattr(a, 'x')",
    "globals()",
    "[x for x in a]",
    "{x for x in a}",
    "(x for x in a)",
    "{1: 2}",
    "{1, 2}",
    "lambda: 1",
    "(lambda: 1)()",
    "a[0]",
    "a[1:3]",
    "col('a')[0]",
    "1 if a else 2",
    "f'{a}'",
    "x := 3",
    "await a",
    "yield a",
    "a if True else b",
    "*a",
    "print(*a)",
    "abs(a=1)",
    "abs(**{})",
    "a @ b",
    "a << 3",
    "a >> 3",
    "~a",
    "1 is 1",
    "a is None",
    "a is not None",
    "b'abc'",
    "1j",
    "...",
    "'a' * 100000000",
    "g * 5",
    "'x' ** 2",
    "col(g)",
    "col('a', 'b')",
    "col('no existe')",
    "col(1)",
    "FUNCTIONS",
    "evaluate('1', 1)",
    "import os",
    "a; b",
    "a\nb",
    "a = 3",
    "\x00",
    "a +",
    "(((",
    ")",
    "'" + "x" * 300 + "'",
    "abs(" * 200 + "1" + ")" * 200,
    "1+" * 400 + "1",
    "(" * 400 + "1" + ")" * 400,
    "x" * 2000,
    "[" * 100 + "1" + "]" * 100,
    "isin(a, [" + ",".join(["1"] * 500) + "])",
    "rolling_mean(a, 1000000000)",
    "lag(a, 1000000000)",
    "round(a, 1000000)",
    "concat(" + ",".join(["g"] * 20) + ")",
]


@pytest.mark.parametrize("expression", HOSTILE)
def test_hostile_expressions_are_refused_without_effects(expression, frame):
    before = frame.copy()
    with pytest.raises(InvalidInputError):
        evaluate(expression, frame)
    pd.testing.assert_frame_equal(frame, before)


def test_nothing_is_executed_by_a_refused_expression(tmp_path, frame):
    marker = tmp_path / "marker"
    with pytest.raises(InvalidInputError):
        evaluate(f"__import__('pathlib').Path(r'{marker}').write_text('x')", frame)
    assert not marker.exists()


def test_giant_arithmetic_does_not_hang_or_exhaust_memory(frame):
    started = time.monotonic()
    for expression in ("9 ** 9 ** 9 ** 9 ** 9", "a ** b ** a ** b", "999999999999 ** 999999999999", "10 ** 400"):
        result = evaluate(expression, frame)
        assert result.dtype.kind == "f"
    assert time.monotonic() - started < 3


def test_a_big_dataset_is_handled_vectorised():
    big = pd.DataFrame({"x": np.arange(300_000, dtype=float), "g": np.where(np.arange(300_000) % 2, "a", "b")})
    started = time.monotonic()
    out = evaluate('where(g == "a", log(x + 1) - lag(log(x + 1), 1), zscore(x))', big)
    assert len(out) == 300_000 and time.monotonic() - started < 5


def test_module_has_no_eval_or_exec():
    source = Path(expressions.__file__).read_text(encoding="utf-8")
    for forbidden in ("eval(", "exec(", "compile(", "__import__"):
        assert forbidden not in source.replace("pd.eval", "")
