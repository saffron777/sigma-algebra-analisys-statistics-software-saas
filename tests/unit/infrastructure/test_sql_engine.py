"""``ensure_database``: crea la base de datos destino cuando aún no existe."""

from sqlalchemy import inspect, text

from sigma.infrastructure.persistence.database_settings import DatabaseSettings
from sigma.infrastructure.persistence.sql import engine as engine_module
from sigma.infrastructure.persistence.sql.schema import SCHEMA_VERSION


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


# ------------------------------------------------------------------ upgrading an existing table
def _dialect(name: str):
    from sqlalchemy.dialects import mssql, mysql, oracle, postgresql, sqlite

    return {
        "sqlite": sqlite.dialect(),
        "mssql": mssql.dialect(),
        "mysql": mysql.dialect(),
        "postgresql": postgresql.dialect(),
        "oracle": oracle.dialect(),
    }[name]


class _Inspector:
    """``inspect(engine).get_columns(table)`` with a fixed answer (the column names the fake table has)."""

    def __init__(self, dialect, columns) -> None:
        self.dialect = dialect
        self._columns = list(columns)

    def get_columns(self, table, schema=None):  # noqa: ARG002 - same signature as SQLAlchemy's inspector
        return [{"name": name} for name in self._columns]


class _AddColumnEngine:
    """Enough of an ``Engine`` for ``_add_column``: a dialect, an inspection and a recording ``begin()``."""

    def __init__(self, dialect, columns) -> None:
        self.dialect = _dialect(dialect)
        self.statements: list[str] = []
        self._columns = columns

    def begin(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        return False

    def execute(self, statement, *_args, **_kwargs):
        self.statements.append(str(statement))


def _run_add_column(monkeypatch, dialect: str, columns) -> list[str]:
    engine = _AddColumnEngine(dialect, columns)
    monkeypatch.setattr(engine_module, "inspect", lambda _engine: _Inspector(engine.dialect, engine._columns))
    engine_module._add_column(engine, engine_module.schema.ai_usage, engine_module.schema.ai_usage.c.provider)
    return engine.statements


def test_sql_server_says_add_without_column(monkeypatch):
    """SQL Server and Oracle reject «ADD COLUMN»: the migration must not depend on the generic syntax."""
    statements = _run_add_column(monkeypatch, "mssql", ["id", "user_id", "model"])
    assert len(statements) == 1
    statement = statements[0]
    assert statement.startswith("ALTER TABLE ai_usage ADD provider ")
    assert "ADD COLUMN" not in statement and "NVARCHAR(40)" in statement
    assert "NOT NULL" not in statement  # SQL Server refuses a NOT NULL column added without a default


def test_oracle_also_says_add(monkeypatch):
    statements = _run_add_column(monkeypatch, "oracle", ["id", "user_id", "model"])
    assert len(statements) == 1 and "ADD" in statements[0] and "ADD COLUMN" not in statements[0]


def test_the_other_engines_say_add_column(monkeypatch):
    for dialect in ("sqlite", "mysql", "postgresql"):
        statements = _run_add_column(monkeypatch, dialect, ["id", "user_id", "model"])
        assert len(statements) == 1, dialect
        assert statements[0].startswith("ALTER TABLE ai_usage ADD COLUMN provider"), dialect


def test_a_column_that_already_exists_is_left_alone(monkeypatch):
    """The version stamp can be interrupted by a crash: the step must be idempotent, never raise."""
    for dialect in ("sqlite", "mssql", "mysql", "postgresql", "oracle"):
        assert _run_add_column(monkeypatch, dialect, ["id", "user_id", "model", "provider"]) == [], dialect


def test_the_upgrade_adds_the_column_to_a_real_table(tmp_path):
    """End to end on a real engine (SQLite): the pending upgrades reach the current schema version."""
    from sigma.infrastructure.persistence.sql.ai_repositories import SqlAiUsageRepository
    from sigma.infrastructure.persistence.sql.engine import Database
    from tests.support import add_user

    path = tmp_path / "upgrade.db"
    first = Database(path)
    with first.begin() as conn:
        conn.execute(text("DROP TABLE ai_usage"))  # the shape a version-7 database has
        conn.execute(
            text(
                "CREATE TABLE ai_usage (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, module TEXT NOT NULL,"
                " model TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,"
                " created_at TEXT NOT NULL)"
            )
        )
        conn.execute(text("UPDATE sigma_schema SET version = 7"))
    first.dispose()

    second = Database(path)  # opening it runs the pending upgrade
    try:
        assert "provider" in {item["name"] for item in inspect(second.engine).get_columns("ai_usage")}
        with second.begin() as conn:
            assert conn.execute(text("SELECT version FROM sigma_schema")).scalar() == SCHEMA_VERSION
    finally:
        second.dispose()

    third = Database(path)  # and opening it again is harmless (the step is idempotent)
    try:
        assert "provider" in {item["name"] for item in inspect(third.engine).get_columns("ai_usage")}
        user = add_user(third, "ana")
        SqlAiUsageRepository(third).add(user, "regression", "gpt-5", 10, 5, "openai")
        with third.begin() as conn:
            assert conn.execute(text("SELECT provider FROM ai_usage")).scalar() == "openai"
    finally:
        third.dispose()
