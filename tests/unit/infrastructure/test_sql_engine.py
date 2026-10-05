"""``ensure_database``: crea la base de datos destino cuando aún no existe."""

from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql import engine as engine_module


class _FakeResult:
    def __init__(self, scalar_value=None) -> None:
        self.scalar_value = scalar_value

    def scalar(self):
        return self.scalar_value


class _FakeConnection:
    def __init__(self, scalar_value=None) -> None:
        self.statements: list[str] = []
        self.scalar_value = scalar_value

    def execute(self, statement, *args, **_kwargs):
        self.statements.append(str(statement))
        return _FakeResult(self.scalar_value)


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self.connection = connection
        self.disposed = False

    def connect(self):
        return self

    def execution_options(self, **_kwargs):
        return self

    def __enter__(self):
        return self.connection

    def __exit__(self, *_args):
        return False

    def dispose(self) -> None:
        self.disposed = True


def _build_fake(monkeypatch, connection):
    engine = _FakeEngine(connection)
    built: dict = {}

    def fake_build_engine(target, connect_timeout=15):
        built["target"] = target
        built["timeout"] = connect_timeout
        return engine

    monkeypatch.setattr(engine_module, "build_engine", fake_build_engine)
    return engine, built


def test_mssql_creates_the_database_when_it_is_missing(monkeypatch):
    connection = _FakeConnection()
    engine, built = _build_fake(monkeypatch, connection)
    settings = DatabaseSettings(engine="mssql", host="srv", database="sigma", username="sa", password="p")

    engine_module.ensure_database(settings, connect_timeout=8)

    assert built["target"].database == "master" and built["timeout"] == 8
    assert any("CREATE DATABASE [sigma]" in statement for statement in connection.statements)
    assert engine.disposed is True


def test_mysql_creates_the_database_when_it_is_missing(monkeypatch):
    connection = _FakeConnection()
    engine, built = _build_fake(monkeypatch, connection)
    settings = DatabaseSettings(engine="mysql", host="h", database="sigma", username="u", password="p")

    engine_module.ensure_database(settings, connect_timeout=8)

    assert built["target"].database is None  # conecta sin esquema por defecto
    assert any("CREATE DATABASE IF NOT EXISTS `sigma`" in statement for statement in connection.statements)
    assert engine.disposed is True


def test_postgresql_creates_the_database_when_it_is_missing(monkeypatch):
    connection = _FakeConnection(scalar_value=None)  # pg_database no la encuentra
    engine, built = _build_fake(monkeypatch, connection)
    settings = DatabaseSettings(engine="postgresql", host="h", database="sigma", username="u", password="p")

    engine_module.ensure_database(settings, connect_timeout=8)

    assert built["target"].database == "postgres"
    assert any('CREATE DATABASE "sigma"' in statement for statement in connection.statements)
    assert engine.disposed is True


def test_postgresql_skips_when_the_database_exists(monkeypatch):
    connection = _FakeConnection(scalar_value=1)  # pg_database ya la encuentra
    _engine, built = _build_fake(monkeypatch, connection)
    settings = DatabaseSettings(engine="postgresql", host="h", database="sigma", username="u", password="p")

    engine_module.ensure_database(settings, connect_timeout=8)

    assert built["target"].database == "postgres"
    assert not any("CREATE DATABASE" in statement for statement in connection.statements)


def test_sqlite_mongodb_and_oracle_are_untouched(monkeypatch):
    def fail_build(*_args, **_kwargs):
        raise AssertionError("build_engine no debe llamarse para este motor")

    monkeypatch.setattr(engine_module, "build_engine", fail_build)
    engine_module.ensure_database(DatabaseSettings.sqlite("x.db"))
    engine_module.ensure_database(DatabaseSettings(engine="mongodb", host="h", database="d"))
    engine_module.ensure_database(DatabaseSettings(engine="oracle", host="h", database="d"))
