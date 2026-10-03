"""The connection test reports what it finds, and explains failures in Spanish, on every engine."""

import dataclasses

import pytest

from sigma.domain.database import STATE_EMPTY, STATE_FOREIGN, STATE_SIGMA
from sigma.infrastructure.persistence.connection_tester import ConnectionTester
from tests.conformance.conftest import wipe

pytestmark = pytest.mark.usefixtures("real_server")


@pytest.fixture
def tester():
    return ConnectionTester(timeout_seconds=6)


def test_an_existing_sigma_database_is_recognised_with_its_counts(backend, uid, tester):
    report = tester.test(backend.settings)
    assert report.ok and report.state == STATE_SIGMA and report.server_version
    assert report.counts and report.counts["users"] == 1
    assert [step.name for step in report.steps] == ["Controlador", "Conexión", "Contenido", "Permisos"] or (
        report.steps[0].name == "Controlador" and all(step.ok for step in report.steps)
    )


def test_an_empty_database_is_ready(backend, tester):
    settings = backend.settings
    backend.close()
    wipe(settings)
    report = tester.test(settings)
    assert report.ok and report.state == STATE_EMPTY and "vacía" in report.message


def test_foreign_tables_are_refused(backend, tester):
    if backend.settings.engine == "mongodb":
        pytest.skip("covered by the MongoDB backend tests")
    from sqlalchemy import text

    settings = backend.settings
    backend.close()
    wipe(settings)
    from sigma.infrastructure.persistence.sql.engine import build_engine

    engine = build_engine(settings)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE contabilidad (id INTEGER)"))
    engine.dispose()
    report = tester.test(settings)
    assert not report.ok and report.state == STATE_FOREIGN and "no son de Sigma" in report.message
    wipe(settings)


def test_a_wrong_password_is_explained(backend, tester):
    settings = backend.settings
    if settings.is_sqlite or not settings.password:
        pytest.skip("this target has no password (SQLite / Windows authentication)")
    report = tester.test(dataclasses.replace(settings, password="clave-incorrecta-xyz"))
    assert not report.ok and "contraseña" in report.message and "clave-incorrecta-xyz" not in report.message


def test_a_missing_database_is_explained(backend, tester):
    settings = backend.settings
    if settings.is_sqlite or settings.engine == "mongodb":
        pytest.skip("not applicable")
    report = tester.test(dataclasses.replace(settings, database="no_existe_sigma_xyz"))
    assert not report.ok
    assert any(word in report.message.lower() for word in ("no existe", "servicio", "acceso", "rechaz"))


def test_an_unreachable_server_is_explained(backend, tester):
    settings = backend.settings
    if settings.is_sqlite:
        pytest.skip("not applicable")
    report = tester.test(dataclasses.replace(settings, host="127.0.0.1", port=1))
    assert not report.ok and report.message and "Traceback" not in report.message
