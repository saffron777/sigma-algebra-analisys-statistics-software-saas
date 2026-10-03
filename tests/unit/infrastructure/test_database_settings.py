import json
import os
import stat
from pathlib import Path

import pytest

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence import database_settings as ds
from sigma.infrastructure.persistence.database_settings import DatabaseSettings


# ------------------------------------------------------------------ URL <-> settings
@pytest.mark.parametrize(
    "url, engine, port, database",
    [
        ("mssql+pyodbc://sa:p%40ss@srv:1433/sigma?driver=ODBC+Driver+17+for+SQL+Server", "mssql", 1433, "sigma"),
        ("mysql+pymysql://root:x@localhost/sigma", "mysql", None, "sigma"),
        ("mariadb+pymysql://root:x@localhost:3307/sigma", "mysql", 3307, "sigma"),
        ("postgresql+psycopg://u:p@db:5432/sigma", "postgresql", 5432, "sigma"),
        ("postgres://u:p@db/sigma", "postgresql", None, "sigma"),
        ("oracle+oracledb://u:p@host:1521/?service_name=XEPDB1", "oracle", 1521, "XEPDB1"),
    ],
)
def test_urls_round_trip_through_settings(url, engine, port, database):
    settings = DatabaseSettings.from_url(url)
    assert (settings.engine, settings.port, settings.database) == (engine, port, database)
    again = DatabaseSettings.from_url(settings.to_url())
    assert again.same_target(settings) and again.password == settings.password


def test_passwords_with_special_characters_survive_the_url():
    settings = DatabaseSettings(engine="postgresql", host="db", database="s", username="u", password="p@ss:/w?rd#1")
    assert DatabaseSettings.from_url(settings.to_url()).password == "p@ss:/w?rd#1"


def test_the_schema_travels_in_the_url_but_not_in_the_engine_url():
    settings = DatabaseSettings.from_url("postgresql+psycopg://u:p@db/s?sigma_schema=sigma")
    assert settings.schema == "sigma" and "sigma_schema" not in settings.options
    assert "sigma_schema" not in str(settings.sqlalchemy_url()) and "sigma_schema=sigma" in settings.to_url()


def test_sqlite_url_and_unknown_engine(tmp_path):
    path = tmp_path / "a.db"
    settings = DatabaseSettings.from_url(DatabaseSettings.sqlite(path).to_url())
    assert settings.is_sqlite and Path(settings.path).resolve() == path.resolve()
    with pytest.raises(InvalidInputError):
        DatabaseSettings.from_url("firebird://x/y")


def test_describe_never_exposes_the_password():
    settings = DatabaseSettings(engine="mysql", host="h", database="d", username="u", password="SECRETO")
    description = settings.describe()
    assert description["has_password"] is True and "SECRETO" not in json.dumps(description)
    assert "SECRETO" not in repr(settings) and "SECRETO" not in settings.to_url(hide_password=True)


def test_validation_messages_are_in_spanish():
    for bad in (
        DatabaseSettings(engine="sqlite"),
        DatabaseSettings(engine="mssql", database="x"),
        DatabaseSettings(engine="mysql", host="h"),
        DatabaseSettings(engine="postgresql", host="h", database="d", schema="mal esquema"),
        DatabaseSettings(engine="oracle", host="h", database="d", port=70000),
    ):
        with pytest.raises(InvalidInputError):
            bad.validate()
    DatabaseSettings(engine="mysql", host="h", database="d", port=3306).validate()


def test_a_blank_password_keeps_the_stored_one():
    stored = DatabaseSettings(engine="mysql", host="h", database="d", username="u", password="viejo")
    form = DatabaseSettings(engine="mysql", host="h", database="otra", username="u")
    assert form.with_password_from(stored).password == "viejo"
    assert (
        DatabaseSettings(engine="mysql", host="h2", database="d", username="u").with_password_from(stored).password
        == ""
    )
    assert (
        DatabaseSettings(engine="mysql", host="h", database="d", username="u", password="nuevo")
        .with_password_from(stored)
        .password
        == "nuevo"
    )
    other_engine = DatabaseSettings(engine="postgresql", host="h", database="d", username="u")
    assert other_engine.with_password_from(stored).password == ""  # never leak a password across engines


def test_mysql_gets_utf8mb4_and_mssql_an_odbc_driver():
    assert DatabaseSettings(engine="mysql", host="h", database="d").sqlalchemy_url().query["charset"] == "utf8mb4"
    assert "driver" in DatabaseSettings(engine="mssql", host="h", database="d").sqlalchemy_url().query


# ------------------------------------------------------------------ precedence and persistence
def test_precedence_env_then_file_then_default(tmp_path):
    default = tmp_path / "sigma.db"
    assert ds.load_settings(tmp_path, default, {}).is_sqlite and ds.settings_source(tmp_path, {}) == "default"

    chosen = DatabaseSettings(engine="mysql", host="h", database="d", username="u", password="p")
    ds.save_settings(tmp_path, chosen)
    assert ds.settings_source(tmp_path, {}) == "file"
    assert ds.load_settings(tmp_path, default, {}).same_target(chosen)

    env = {"SIGMA_DATABASE_URL": "postgresql+psycopg://a:b@pg/x"}
    assert ds.settings_source(tmp_path, env) == "env"
    assert ds.load_settings(tmp_path, default, env).engine == "postgresql"


def test_the_settings_file_is_private_and_atomic(tmp_path):
    ds.save_settings(tmp_path, DatabaseSettings(engine="mysql", host="h", database="d", password="p"))
    path = ds.settings_path(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["engine"] == "mysql"
    assert not path.with_suffix(".tmp").exists()
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_corrupt_settings_file_falls_back_to_sqlite(tmp_path):
    ds.settings_path(tmp_path).write_text("{no es json", encoding="utf-8")
    assert ds.load_settings(tmp_path, tmp_path / "sigma.db", {}).is_sqlite


def test_engine_list_for_the_ui_names_every_engine():
    engines = {e["engine"]: e for e in ds.engines_for_ui()}
    assert set(engines) == {"sqlite", "mssql", "mysql", "postgresql", "oracle", "mongodb"}
    assert engines["sqlite"]["driver_available"] is True and engines["mongodb"]["relational"] is False


# ------------------------------------------------------------------ settings typed into the browser
SQLITE = Path("instance") / "sigma.db"


def request(**fields):
    return {"engine": "postgresql", "host": "db.interna", "port": 5432, "database": "sigma", **fields}


def test_sqlite_from_a_request_always_means_the_servers_own_file():
    settings = DatabaseSettings.from_request({"engine": "sqlite", "path": "C:/Windows/evil.db"}, SQLITE)
    assert settings.is_sqlite and settings.path == str(SQLITE)


def test_a_blank_password_is_kept_only_for_the_same_target():
    stored = DatabaseSettings(
        engine="postgresql", host="db.interna", port=5432, database="s", username="u", password="vieja"
    )
    same = DatabaseSettings.from_request(request(username="u"), SQLITE, stored)
    assert same.password == "vieja"
    for change in ({"host": "otro.servidor"}, {"port": 5433}, {"username": "otro"}):
        moved = DatabaseSettings.from_request({**request(username="u"), **change}, SQLITE, stored)
        assert moved.password == ""  # a new host is never handed the old server's password
    assert DatabaseSettings.from_request(request(username="u", password="nueva"), SQLITE, stored).password == "nueva"


@pytest.mark.parametrize(
    "bad",
    [
        request(host=""),
        request(host="a b"),
        request(host="x;y"),
        request(database=""),
        request(database="a'b"),
        request(username="u\x00"),
        request(password="p\nq"),
        request(port="abc"),
        request(port=70000),
        request(schema="mal esquema"),
        request(options={"sslmode": "invalido"}),
        {"engine": "nope", "host": "h", "database": "d"},
    ],
)
def test_bad_requests_are_rejected(bad):
    with pytest.raises(InvalidInputError):
        DatabaseSettings.from_request(bad, SQLITE)


def test_only_known_options_survive_a_request():
    settings = DatabaseSettings.from_request(
        request(options={"sslmode": "require", "sslrootcert": "/etc/passwd", "options": "-c x=1"}), SQLITE
    )
    assert dict(settings.options) == {"sslmode": "require"}
    mssql = DatabaseSettings.from_request(
        {
            "engine": "mssql",
            "host": r"srv\SQLEXPRESS",
            "database": "sigma",
            "options": {"driver": "ODBC Driver 18 for SQL Server", "trusted_connection": "yes"},
        },
        SQLITE,
    )
    assert mssql.host == r"srv\SQLEXPRESS" and mssql.options["driver"] == "ODBC Driver 18 for SQL Server"
    with pytest.raises(InvalidInputError):
        DatabaseSettings.from_request(
            {"engine": "mssql", "host": "h", "database": "d", "options": {"driver": "x;DROP"}}, SQLITE
        )
