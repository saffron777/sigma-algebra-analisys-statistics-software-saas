"""Las explicaciones en español de los fallos de conexión (SQL Server)."""

from sigma.infrastructure.database_errors import explain_driver_error


def _mssql(message: str) -> str:
    return explain_driver_error(RuntimeError(message), "mssql")


def test_a_missing_database_is_not_reported_as_a_bad_password():
    message = "Login failed for user 'sa'. Reason: Failed to open the explicitly specified database 'sigma'."
    explanation = _mssql(message)
    assert "no existe" in explanation
    assert "contraseña" not in explanation


def test_cannot_open_database_is_explained():
    explanation = _mssql("Cannot open database 'sigma' requested by the login. The login failed.")
    assert "no existe" in explanation


def test_a_bad_password_is_still_explained():
    explanation = _mssql("Login failed for user 'sa'.")
    assert "contraseña" in explanation


def test_missing_create_permission_is_explained():
    explanation = _mssql("CREATE DATABASE permission denied in database 'master'.")
    assert "permiso" in explanation


def test_a_tls_failure_is_explained():
    explanation = _mssql("SSL Provider: The certificate chain was issued by an authority that is not trusted.")
    assert "certificado" in explanation


def test_the_password_never_appears_in_the_explanation():
    explanation = explain_driver_error(RuntimeError("error inesperado SECRETO"), "mssql", "SECRETO")
    assert "SECRETO" not in explanation and "***" in explanation
