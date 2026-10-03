"""Opening, re-opening and refusing databases on every engine."""

import dataclasses

import pandas as pd
import pytest
from sqlalchemy import inspect, text

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.backend import create_backend
from sigma.infrastructure.persistence.sql import schema as sql_schema
from sigma.infrastructure.persistence.sql.engine import build_engine
from tests.conformance.conftest import wipe


def test_reopening_keeps_the_data(backend, uid, real_server):
    backend.datasets(uid).add("persistente", pd.DataFrame({"a": [1, 2, 3]}), "csv", {})
    backend.settings_store().set("clave", "valor")
    again = create_backend(backend.settings)  # a restart of the application
    try:
        assert again.datasets(uid).get_by_name("persistente").data["a"].tolist() == [1, 2, 3]
        assert again.settings_store().get("clave") == "valor" and again.users().count() == 1
    finally:
        again.close()


def test_the_schema_version_is_recorded(backend):
    if not backend.settings.is_relational:
        pytest.skip("only relational engines keep a schema table")
    with backend.database.begin() as conn:
        assert conn.execute(text("SELECT version FROM sigma_schema")).scalar() == sql_schema.SCHEMA_VERSION


def test_a_database_that_is_not_sigmas_is_refused(backend):
    if not backend.settings.is_relational:
        pytest.skip("a document database has no foreign tables to collide with")
    settings = backend.settings
    backend.close()
    wipe(settings)
    engine = build_engine(settings)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE tabla_ajena (id INTEGER)"))
    engine.dispose()
    with pytest.raises(InvalidInputError):
        create_backend(settings)
    wipe(settings)


def test_a_dedicated_schema_keeps_sigma_apart(backend):
    if backend.settings.engine not in ("postgresql", "mssql"):
        pytest.skip("dedicated schemas are offered for PostgreSQL and SQL Server")
    settings = dataclasses.replace(backend.settings, schema="sigma_app")
    wipe(settings)
    app = create_backend(settings)
    try:
        owner = app.users().add("ana", "Ana", "x", "user", False, False).id
        info = app.datasets(owner).add("datos", pd.DataFrame({"x": [1, 2]}), "csv", {})
        assert app.datasets(owner).get(info.id).data["x"].tolist() == [1, 2]
        names = inspect(app.database.engine).get_table_names(schema="sigma_app")
        assert "users" in names and any(n.startswith("ds_") for n in names)
        assert backend.users().count() == 0  # the default schema (the fixture's) never saw the account
    finally:
        app.close()
        wipe(settings)
