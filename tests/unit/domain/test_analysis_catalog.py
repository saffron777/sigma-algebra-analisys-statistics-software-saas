"""Every analysis of the catalogue runs on the standard test dataset and produces a valid, exportable report."""

import json

import pandas as pd
import pytest

from sigma.domain.analysis import registry
from sigma.domain.analysis.output import Cell
from sigma.domain.analysis.spec import prepare_inputs
from sigma.presentation.web.reports.analysis import build_report
from sigma.presentation.web.serializers import to_jsonable
from tests.analysis_data import standard_frame

SPECS = registry.all_specs()


@pytest.fixture(scope="module")
def frame():
    return standard_frame()


def test_the_catalogue_is_not_empty_and_ids_are_unique():
    assert SPECS
    keys = [(s.module, s.id) for s in SPECS]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.module}/{s.id}")
def test_every_analysis_has_a_runnable_example_and_a_valid_declaration(spec):
    assert spec.example is not None, "declare `example=` so the catalogue test can run it"
    assert spec.label and spec.group and spec.description
    names = [r.name for r in spec.roles] + [f.name for f in spec.params]
    assert len(names) == len(set(names)), "role and parameter names must be unique"
    for field in spec.params:
        assert field.kind in ("number", "integer", "select", "text", "boolean", "multiselect", "code")
        if field.kind in ("select", "multiselect"):
            assert field.options, f"{field.name}: a choice needs options"
        if field.visible_if:
            assert field.visible_if[0] in names[: names.index(field.name)], "the controlling parameter must come first"
    for output in spec.outputs:
        assert output.key and output.label


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.module}/{s.id}")
def test_every_example_produces_a_valid_report(spec, frame):
    data = frame if spec.needs_dataset else None
    roles, params = prepare_inputs(spec, data, spec.example["roles"], spec.example["params"])
    out = spec.compute(frame if spec.needs_dataset else pd.DataFrame(), roles, params)
    assert out.title and out.blocks
    for block in out.blocks:
        assert block["type"] in ("heading", "tiles", "table", "text", "alert", "chart")
        if block["type"] == "table":
            assert all(len(row) == len(block["columns"]) for row in block["rows"]), block["title"]
    for key in {o.key for o in spec.outputs if o.required}:
        assert key in out.datasets, f"the required output {key!r} is missing"
    for offer in out.datasets.values():
        assert isinstance(offer.frame, pd.DataFrame) and len(offer.frame.columns)

    builder = build_report(out)
    report = builder.report()
    text = json.dumps(
        to_jsonable({"report": report, "figures": builder.figures, "result": out.result}), allow_nan=False
    )
    assert "NaN" not in text and "Infinity" not in text
    for block in report["blocks"]:
        if block["type"] == "figure":
            assert 0 <= block["index"] < len(builder.figures)
    for figure in builder.figures:
        assert figure["data"]
    for table in (b for b in out.blocks if b["type"] == "table"):
        for row in table["rows"]:
            for cell in row:
                assert not isinstance(cell, Cell) or isinstance(cell.tone, str)

    if spec.code:
        source = spec.code(roles, params, "mis_datos")
        assert "import" in source or "=" in source
        compile(source.replace('sigma.load("mis_datos")', "None"), "<codigo>", "exec")
