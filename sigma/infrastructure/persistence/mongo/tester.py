"""The connection test for MongoDB (separate module: pymongo is an optional dependency)."""

from __future__ import annotations

import uuid

from sigma.domain.database import STATE_EMPTY, STATE_FOREIGN, STATE_SIGMA, CheckStep, ConnectionReport
from sigma.infrastructure.database_errors import explain_driver_error
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available


def explain(error: BaseException, password: str) -> str:
    return explain_driver_error(error, "mongodb", password)


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
