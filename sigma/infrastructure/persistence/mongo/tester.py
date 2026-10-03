"""The connection test for MongoDB (separate module: pymongo is an optional dependency)."""

from __future__ import annotations

import re
import uuid

from sigma.domain.database import STATE_EMPTY, STATE_FOREIGN, STATE_SIGMA, CheckStep, ConnectionReport
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available

_ERRORS = [
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


def explain(error: BaseException, password: str) -> str:
    raw = str(error)
    if password:
        raw = raw.replace(password, "***")
    for pattern, explanation in _ERRORS:
        if re.search(pattern, raw, re.IGNORECASE):
            return explanation
    first = raw.strip().splitlines()[0][:240] if raw.strip() else error.__class__.__name__
    return f"No se pudo conectar: {first}"


def test_mongo(settings: DatabaseSettings, timeout_seconds: int) -> ConnectionReport:
    label = ENGINES["mongodb"]["label"]
    if not driver_available("mongodb"):
        message = f"Falta el controlador de {label}. Instálelo con: pip install {ENGINES['mongodb']['pip']}"
        return ConnectionReport(False, "mongodb", message, steps=(CheckStep("Controlador", False, message),))
    from pymongo import MongoClient

    steps = [CheckStep("Controlador", True, "pymongo")]
    counts = None
    client = MongoClient(settings.to_url(), serverSelectionTimeoutMS=timeout_seconds * 1000)
    try:
        version = str(client.server_info().get("version", ""))
        steps.append(CheckStep("Conexión", True, f"MongoDB {version}"))
        database = client[settings.database]
        collections = sorted(database.list_collection_names())
        if not collections:
            state, detail = STATE_EMPTY, "Base vacía: Sigma creará sus colecciones."
        elif "sigma_schema" in collections:
            state, detail = STATE_SIGMA, "Ya contiene datos de Sigma: se usarán tal como están."
        else:
            state, detail = STATE_FOREIGN, f"Contiene {len(collections)} colecciones que no son de Sigma."
        steps.append(CheckStep("Contenido", state != STATE_FOREIGN, detail))
        if state == STATE_SIGMA:
            counts = {
                "users": database["users"].count_documents({}),
                "datasets": database["datasets"].count_documents({}),
                "results": database["saved_results"].count_documents({}),
            }
        probe = database[f"sigma_probe_{uuid.uuid4().hex[:10]}"]
        probe.insert_one({"x": 1})
        probe.drop()
        steps.append(CheckStep("Permisos", True, "Puede crear, escribir y borrar colecciones."))
    except Exception as error:  # noqa: BLE001 - pymongo has many exception types
        explanation = explain(error, settings.password)
        steps.append(CheckStep("Conexión", False, explanation))
        return ConnectionReport(False, "mongodb", explanation, steps=tuple(steps))
    finally:
        client.close()
    ok = state != STATE_FOREIGN
    headline = (
        "Conexión correcta. La base ya contiene datos de Sigma."
        if state == STATE_SIGMA
        else "Conexión correcta. La base está vacía y lista para usarse."
        if ok
        else "La base de datos ya tiene colecciones que no son de Sigma. Use una base vacía."
    )
    return ConnectionReport(ok, "mongodb", headline, version, state, counts, tuple(steps))
