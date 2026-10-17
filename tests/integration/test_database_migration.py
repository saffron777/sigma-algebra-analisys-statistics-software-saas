"""Moving a running Sigma Algebra to another database from the administration page: copy, switch, keep working."""

import os
import time

import pandas as pd
import pytest

from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.presentation.web.migration import MAINTENANCE_KEY
from tests.conformance.conftest import ENV, wipe
from tests.integration.conftest import database_under_test

pytestmark = pytest.mark.skipif(
    database_under_test() is not None, reason="la aplicación de pruebas ya corre sobre otro motor"
)


def available_targets():
    targets = [pytest.param(engine, id=engine) for engine, variable in ENV.items() if os.environ.get(variable)]
    return targets or [pytest.param(None, id="sin-servidores", marks=pytest.mark.skip(reason="sin SIGMA_TEST_*_URL"))]


def request_for(settings: DatabaseSettings) -> dict:
    return {
        "engine": settings.engine,
        "host": settings.host,
        "port": settings.port,
        "database": settings.database,
        "schema": settings.schema,
        "username": settings.username,
        "password": settings.password,
        "options": dict(settings.options),
        "migrate": True,
    }


def wait_for(api, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = api.get("/api/admin/database/migration").parsed
        if job["state"] in ("done", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError("la migración no terminó a tiempo")


@pytest.mark.parametrize("engine", available_targets())
def test_migrating_the_running_application_keeps_everyone_signed_in_and_their_data(
    engine, app, api, sales_frame, tmp_path
):
    from tests.integration.conftest import import_frame

    target = DatabaseSettings.from_url(os.environ[ENV[engine]])
    wipe(target)
    try:
        dataset = import_frame(api, sales_frame, "ventas")
        notebook = api.post("/api/notebooks", {"name": "mi análisis"}).parsed["notebook"]
        api.patch("/api/auth/profile", {"display_name": "Admin Ñ"})
        before_users = api.get("/api/users").parsed["users"]

        started = api.post("/api/admin/database/apply", request_for(target))
        assert started.status_code == 200 and started.parsed["migrating"] is True
        job = wait_for(api)
        assert job["state"] == "done", job
        assert job["report"]["copied"]["datasets"] == 1 and job["setup_required"] is False

        sigma = app.extensions["sigma"]
        assert sigma.config.database.engine == target.engine and sigma.backend.describe()["counts"]["datasets"] == 1
        assert api.get("/api/users").parsed["users"] == before_users  # same ids, same epochs: the session still works
        assert [d["id"] for d in api.get("/api/datasets").parsed["datasets"]] == [dataset["id"]]
        preview = api.get(f"/api/datasets/{dataset['id']}/preview?limit=3").parsed
        assert len(preview["rows"]) == 3
        assert [n["id"] for n in api.get("/api/notebooks").parsed["notebooks"]] == [notebook["id"]]
        status = api.get("/api/admin/database").parsed
        assert status["current"]["engine"] == target.engine and status["source"] == "file"
        # new work lands in the new database
        again = import_frame(api, pd.DataFrame({"x": [1, 2]}), "nuevo")
        assert again["id"] != dataset["id"] and api.get("/api/datasets").parsed["datasets"]
    finally:
        app.extensions["sigma"].close()
        wipe(target)


def test_the_maintenance_gate_closes_everything_but_the_progress(app, api):
    app.config[MAINTENANCE_KEY] = True
    try:
        assert api.get("/api/datasets").status_code == 503
        assert api.get("/api/datasets").parsed["error"]["code"] == "maintenance"
        assert api.raw.get("/datos").status_code == 503
        assert api.get("/health").status_code == 200
        assert api.get("/api/admin/database/migration").status_code == 200
    finally:
        app.config[MAINTENANCE_KEY] = False
    assert api.get("/api/datasets").status_code == 200


def test_a_migration_to_a_database_with_data_is_refused_and_nothing_changes(app, api, tmp_path):
    target = DatabaseSettings.sqlite(tmp_path / "ocupada.db")
    from sigma.infrastructure.persistence.backend import create_backend

    other = create_backend(target)
    other.users().add("alguien", "Alguien", "x", "user", False, False)
    other.close()
    # SQLite requests always mean the server's own file, so exercise the refusal through the use case instead
    from sigma.application.use_cases.migration import MigrateData
    from sigma.domain.errors import InvalidInputError

    source = app.extensions["sigma"].backend.transfer(tmp_path / "nb")
    busy = create_backend(target)
    try:
        with pytest.raises(InvalidInputError):
            MigrateData().execute(source, busy.transfer(tmp_path / "nb2"))
    finally:
        busy.close()
    assert api.get("/api/datasets").status_code == 200
