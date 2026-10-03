"""Choosing the database from the browser: the administration page, the wizard's step and the hot switch."""

import json

import pytest

from sigma.app_factory import create_app
from sigma.config import Config
from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from tests.integration.conftest import PASSWORD, ApiClient, database_under_test

pytestmark = pytest.mark.skipif(
    database_under_test() is not None, reason="estas pruebas parten de la configuración SQLite de las pruebas"
)

UNREACHABLE = {
    "engine": "postgresql",
    "host": "127.0.0.1",
    "port": 1,
    "database": "x",
    "username": "u",
    "password": "p",
}


# ------------------------------------------------------------------ the administration page
def test_status_describes_the_database_without_secrets(api):
    status = api.get("/api/admin/database").parsed
    assert status["current"]["engine"] == "sqlite" and status["editable"] is True and status["source"] == "default"
    assert status["live"]["counts"]["users"] == 1 and status["live"]["server_version"]
    assert {e["engine"] for e in status["engines"]} == {"sqlite", "mssql", "mysql", "postgresql", "oracle", "mongodb"}
    assert '"password"' not in json.dumps(status)


def test_only_administrators_reach_it(api, make_client, anonymous):
    user = make_client("luis")
    for method, url in (("get", "/api/admin/database"), ("post", "/api/admin/database/test")):
        assert getattr(user, method)(url).status_code == 403
    assert user.raw.get("/basedatos").status_code == 403
    assert api.raw.get("/basedatos").status_code == 200
    fresh = ApiClient(anonymous.raw.application.test_client())
    assert fresh.get("/api/admin/database").status_code == 401


def test_the_menu_offers_the_page_only_to_administrators(api, make_client):
    assert 'href="/basedatos"' in api.raw.get("/").get_data(as_text=True)
    assert 'href="/basedatos"' not in make_client("luis").raw.get("/").get_data(as_text=True)


def test_testing_the_current_sqlite_file_reports_its_contents(api):
    report = api.post("/api/admin/database/test", {"engine": "sqlite"}).parsed
    assert report["ok"] and report["state"] == "sigma" and report["counts"]["users"] == 1
    assert [s["name"] for s in report["steps"]][:2] == ["Controlador", "Conexión"]
    assert api.post("/api/admin/database/apply", {"engine": "sqlite"}).status_code == 422  # already in use


def test_an_unreachable_server_is_explained_in_spanish_without_the_password(api):
    response = api.post("/api/admin/database/test", {**UNREACHABLE, "password": "SECRETO-123"})
    report = response.parsed
    assert response.status_code == 200 and report["ok"] is False
    assert "conectar" in report["message"] or "respondió" in report["message"]
    assert "SECRETO-123" not in json.dumps(report)
    applied = api.post("/api/admin/database/apply", UNREACHABLE)
    assert applied.status_code == 422 and applied.parsed["error"]["message"] == report["message"]


@pytest.mark.parametrize(
    "bad",
    [
        {"engine": "firebird", "host": "h", "database": "d"},
        {"engine": "mysql", "host": "bad host!", "database": "d"},
        {"engine": "mysql", "host": "h", "database": "d; DROP"},
        {"engine": "mysql", "host": "", "database": "d"},
        {"engine": "postgresql", "host": "h", "database": "d", "options": {"sslmode": "nope"}},
    ],
)
def test_malformed_settings_are_rejected(api, bad):
    assert api.post("/api/admin/database/test", bad).status_code == 422


def test_unknown_options_are_ignored_not_forwarded(api, monkeypatch):
    seen = {}

    def fake_test(self, settings):
        seen["options"] = dict(settings.options)
        raise RuntimeError("stop")

    monkeypatch.setattr("sigma.infrastructure.persistence.connection_tester.ConnectionTester.test", fake_test)
    body = {
        "engine": "mysql",
        "host": "h",
        "database": "d",
        "options": {"init_command": "DROP DATABASE x", "ssl_disabled": "true"},
    }
    assert api.post("/api/admin/database/test", body).status_code == 500  # the stub blows up after parsing
    assert seen["options"] == {"ssl_disabled": "true"}


# ------------------------------------------------------------------ the hot switch
@pytest.fixture
def two_sqlite_app(tmp_path):
    """Running on ``other.db`` while the default file is ``sigma.db`` (which «sqlite» in the page means)."""
    config = Config.from_env(
        testing=True,
        instance_dir=tmp_path,
        db_path=tmp_path / "sigma.db",
        database=DatabaseSettings.sqlite(tmp_path / "other.db"),
        notebooks_dir=tmp_path / "notebooks",
        sample_dir=tmp_path / "samples",
    )
    app = create_app(config)
    app.testing = True
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    client = ApiClient(app.test_client())
    assert client.login("admin").status_code == 200
    return app, client, tmp_path


def test_switching_rebuilds_the_application_on_the_new_database(two_sqlite_app):
    app, client, tmp_path = two_sqlite_app
    old = app.extensions["sigma"]
    assert client.get("/api/admin/database").parsed["current"]["path"].endswith("other.db")
    result = client.post("/api/admin/database/apply", {"engine": "sqlite"})
    assert result.status_code == 200 and result.parsed["switched"] and result.parsed["setup_required"]
    assert app.extensions["sigma"] is not old and app.extensions["sigma"].config.database.path.endswith("sigma.db")
    saved = json.loads((tmp_path / "database.json").read_text(encoding="utf-8"))
    assert saved == {"engine": "sqlite", "path": str(tmp_path / "sigma.db")}
    # the new database is empty: nobody is signed in any more and the wizard is offered again
    assert client.get("/api/datasets").status_code == 401
    assert client.get("/api/auth/status").parsed["setup_required"] is True
    created = client.post("/api/auth/setup", {"username": "nuevo", "password": PASSWORD})
    assert created.status_code == 201
    # …and the old database was left untouched
    old_db = DatabaseSettings.sqlite(tmp_path / "other.db")
    assert any(u.username == "admin" for u in app.extensions["sigma"].backend.users().list()) is False
    from sigma.infrastructure.persistence.backend import create_backend

    back = create_backend(old_db)
    try:
        assert [u.username for u in back.users().list()] == ["admin"]
    finally:
        back.close()


def test_a_failed_switch_leaves_the_application_as_it_was(two_sqlite_app, monkeypatch):
    app, client, _ = two_sqlite_app
    before = app.extensions["sigma"]

    def broken(config):
        raise OSError("disco lleno")

    monkeypatch.setattr("sigma.presentation.web.switching.build_container", broken, raising=False)
    from sigma.presentation.web.switching import ContainerSwitcher

    switcher = ContainerSwitcher(app, close_delay=0, build=broken)
    with pytest.raises(Exception) as error:
        switcher(DatabaseSettings.sqlite("x.db"))
    assert "disco lleno" in str(error.value) or "No se pudo conectar" in str(error.value)
    assert app.extensions["sigma"] is before and client.get("/api/datasets").status_code == 200


def test_environment_variable_settings_cannot_be_changed_from_the_page(tmp_path):
    from sigma.domain.errors import PermissionDeniedError
    from sigma.infrastructure.persistence.administration import DatabaseAdmin
    from sigma.infrastructure.persistence.backend import create_backend

    current = DatabaseSettings.sqlite(tmp_path / "a.db")
    backend = create_backend(current)
    try:
        admin = DatabaseAdmin(
            tmp_path, tmp_path / "a.db", current, backend, environ={"SIGMA_DATABASE_URL": "sqlite:///x"}
        )
        admin.switcher = lambda settings: False
        assert admin.status()["editable"] is False and admin.status()["source"] == "env"
        with pytest.raises(PermissionDeniedError):
            admin.apply({"engine": "sqlite"})
    finally:
        backend.close()


# ------------------------------------------------------------------ the first-run wizard
def test_the_wizard_step_is_public_until_the_first_account_exists(app, anonymous):
    status = anonymous.get("/api/setup/database")
    assert status.status_code == 200 and status.parsed["current"]["engine"] == "sqlite"
    assert anonymous.post("/api/setup/database/test", {"engine": "sqlite"}).parsed["ok"]
    assert anonymous.raw.get("/setup").status_code == 200
    anonymous.post("/api/auth/setup", {"username": "ana", "password": PASSWORD})
    assert anonymous.get("/api/setup/database").status_code == 403
    assert anonymous.post("/api/setup/database/apply", {"engine": "sqlite"}).status_code == 403


def test_the_setup_page_offers_the_database_step(anonymous):
    html = anonymous.raw.get("/setup").get_data(as_text=True)
    assert "databaseChooser('/api/setup/database', 'setup')" in html and "Base de datos" in html
