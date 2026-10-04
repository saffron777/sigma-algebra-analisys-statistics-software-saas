"""Plain-Spanish explanations of database driver failures (connection tests and external-source imports)."""

from __future__ import annotations

import re

# (regular expression over the driver's message, explanation) — first match wins
_ERRORS: dict[str, list[tuple[str, str]]] = {
    "mssql": [
        (r"18456|Login failed", "El servidor rechazó el usuario o la contraseña."),
        (r"4060|Cannot open database", "La base de datos no existe o el usuario no tiene acceso a ella."),
        (
            r"certificate verify failed|certificate chain|SSL Provider",
            "El certificado TLS del servidor no es de confianza. "
            "Active «Confiar en el certificado del servidor» o instale su certificado.",
        ),
        (
            r"Data source name not found|driver.*not found|IM002",
            "No está instalado el controlador ODBC indicado para SQL Server.",
        ),
        (r"timeout|timed out|HYT00", "El servidor no respondió a tiempo."),
        (
            r"network-related|08001|Named Pipes|could not open a connection|TCP Provider",
            "No se pudo conectar con el servidor. Revise el nombre, el puerto y que el servicio esté activo.",
        ),
    ],
    "mysql": [
        (r"\(1045,|Access denied", "El servidor rechazó el usuario o la contraseña."),
        (r"\(1049,|Unknown database", "La base de datos no existe."),
        (r"\(1044,", "El usuario no tiene permiso sobre esa base de datos."),
        (r"\(1146,|\(1142,|\(1143,", "La tabla o vista no existe, o el usuario no tiene permiso para leerla."),
        (
            r"\(2003,|\(2002,|Can't connect|Connection refused",
            "No se pudo conectar con el servidor. Revise el servidor, el puerto y que el servicio esté activo.",
        ),
        (
            r"\(2005,|Name or service not known|getaddrinfo",
            "No se encontró el servidor. Revise el nombre o la dirección.",
        ),
        (r"timed out|\(2013,|\(2006,", "El servidor no respondió a tiempo o cortó la conexión."),
        (r"SSL|TLS", "Falló la conexión segura (TLS) con el servidor."),
    ],
    "postgresql": [
        (
            r"password authentication failed|no password supplied|authentication failed",
            "El servidor rechazó el usuario o la contraseña.",
        ),
        (r'database ".*" does not exist', "La base de datos no existe."),
        (r"relation \".*\" does not exist", "La tabla o vista no existe en ese esquema."),
        (r"permission denied", "El usuario no tiene permiso sobre esa base de datos."),
        (
            r"Connection refused|could not connect|is the server running",
            "No se pudo conectar con el servidor. Revise el servidor, el puerto y que el servicio esté activo.",
        ),
        (
            r"could not translate host name|Name or service not known|getaddrinfo",
            "No se encontró el servidor. Revise el nombre o la dirección.",
        ),
        (r"timeout|timed out", "El servidor no respondió a tiempo."),
        (
            r"SSL|TLS|no pg_hba.conf entry",
            "El servidor rechazó la conexión (revise el modo SSL y las reglas de acceso del servidor).",
        ),
    ],
    "oracle": [
        (r"ORA-01017|invalid credential|invalid username/password", "El servidor rechazó el usuario o la contraseña."),
        (
            r"ORA-12514|ORA-12505|ORA-12154|DPY-6001|service.*not registered",
            "El servicio de Oracle indicado no existe en ese servidor.",
        ),
        (
            r"ORA-12541|DPY-6005|Connection refused|no listener",
            "No se pudo conectar con el servidor. Revise el servidor, el puerto y que el listener esté activo.",
        ),
        (
            r"DPY-4|getaddrinfo|Name or service not known",
            "No se encontró el servidor. Revise el nombre o la dirección.",
        ),
        (r"timeout|timed out|DPY-6000", "El servidor no respondió a tiempo."),
        (r"ORA-00942", "La tabla o vista no existe, o el usuario no tiene permiso para leerla."),
        (r"ORA-01466", "La tabla se modificó hace un instante: vuelva a intentarlo en unos segundos."),
        (
            r"ORA-01045|ORA-01031|insufficient privileges",
            "El usuario no tiene privilegios suficientes (necesita crear tablas).",
        ),
    ],
}

_ERRORS["mongodb"] = [
    (r"Authentication failed|auth failed|AuthenticationFailed", "El servidor rechazó el usuario o la contraseña."),
    (
        r"not authorized|Unauthorized|requires authentication",
        "El usuario no tiene permiso (o el servidor exige autenticarse).",
    ),
    (
        r"timed out|ServerSelectionTimeout|No servers found|Connection refused|actively refused",
        "No se pudo conectar con el servidor. Revise el servidor, el puerto y que el servicio esté activo.",
    ),
    (
        r"getaddrinfo|Name or service not known|nodename nor servname",
        "No se encontró el servidor. Revise el nombre o la dirección.",
    ),
    (r"SSL|TLS|certificate", "Falló la conexión segura (TLS) con el servidor."),
]


def explain_driver_error(
    error: BaseException, engine: str, password: str = "", prefix: str = "No se pudo conectar"
) -> str:
    """A Spanish sentence for a driver failure; the password never appears in it."""
    raw = str(getattr(error, "orig", None) or error)
    if password:
        raw = raw.replace(password, "***")
    for pattern, explanation in _ERRORS.get(engine, []):
        if re.search(pattern, raw, re.IGNORECASE):
            return explanation
    first = raw.strip().splitlines()[0][:240] if raw.strip() else error.__class__.__name__
    return f"{prefix}: {first}"
