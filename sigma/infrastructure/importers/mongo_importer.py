"""Read-only import from MongoDB: the documents of a collection (optionally filtered) flattened into columns.

Only ``find`` is ever used, so nothing can be written. The filter is extended JSON (``{"edad": {"$gt": 30}}``,
``{"fecha": {"$gte": {"$date": "2024-01-01T00:00:00Z"}}}``) and the operators that run JavaScript or arbitrary
expressions on the server (``$where``, ``$function``, ``$accumulator``, ``$expr``) are refused.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError, SigmaError
from sigma.infrastructure.database_errors import explain_driver_error
from sigma.infrastructure.importers.common import as_bool, as_int, check_host_allowed
from sigma.infrastructure.persistence.database_settings import ENGINES, DatabaseSettings, driver_available

DEFAULT_MAX_ROWS = 1_000_000
QUERY_TIMEOUT_MS = 120_000
CONNECT_TIMEOUT_MS = 10_000
CATALOG_LIMIT = 500
SAMPLE_DOCUMENTS = 50
_COLLECTION = re.compile(r"^[\w.\-$ ]{1,120}$", re.UNICODE)
_FIELD = re.compile(r"^[\w.\-$ ]{1,120}$", re.UNICODE)
_FORBIDDEN_OPERATORS = {"$where", "$function", "$accumulator", "$expr"}

ClientFactory = Callable[[DatabaseSettings], Any]


def mongo_client(settings: DatabaseSettings) -> Any:
    meta = ENGINES["mongodb"]
    if not driver_available("mongodb"):
        raise InvalidInputError(f"Falta el controlador de {meta['label']}. Instálelo con: pip install {meta['pip']}")
    from pymongo import MongoClient

    return MongoClient(
        settings.to_url(),
        serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS,
        connectTimeoutMS=CONNECT_TIMEOUT_MS,
        socketTimeoutMS=QUERY_TIMEOUT_MS,
        appname="sigma-import",
    )


def parse_filter(text: str) -> dict[str, Any]:
    """The filter document, from extended JSON; operators that execute code on the server are rejected at any depth."""
    if not text.strip():
        return {}
    from bson import json_util

    try:
        value = json_util.loads(text)
    except ValueError as error:
        raise InvalidInputError(f"El filtro no es un JSON válido: {str(error)[:120]}") from None
    if not isinstance(value, dict):
        raise InvalidInputError('El filtro debe ser un objeto JSON, por ejemplo {"edad": {"$gt": 30}}.')
    _reject_operators(value)
    return value


def _reject_operators(value: Any) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_OPERATORS:
                raise InvalidInputError(
                    f"El operador {key} no está permitido en el filtro (ejecuta código en el servidor)."
                )
            _reject_operators(inner)
    elif isinstance(value, list):
        for item in value:
            _reject_operators(item)


def plain(value: Any) -> Any:
    """BSON values as things pandas understands (ObjectId/UUID → text, Decimal128 → float, binary → hex)."""
    from bson import Binary, Decimal128, ObjectId
    from bson.int64 import Int64

    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Decimal128):
        return float(value.to_decimal())
    if isinstance(value, Binary | bytes | bytearray):
        return bytes(value).hex()
    if isinstance(value, Int64):
        return int(value)
    if value is None or isinstance(value, str | int | float | bool) or hasattr(value, "isoformat"):
        return value
    return str(value)


class MongoImporter:
    source_type = "mongodb"

    def __init__(self, client_factory: ClientFactory = mongo_client, allowed_hosts: tuple[str, ...] = ()) -> None:
        self._client_factory = client_factory
        self._allowed_hosts = allowed_hosts

    # ------------------------------------------------------------------ reading
    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        options = request.options
        max_rows = as_int(options.get("max_rows"), "Máximo de documentos", DEFAULT_MAX_ROWS, minimum=1)
        collection = self._collection_name(options)
        query = parse_filter(str(options.get("filter") or ""))
        projection = self._projection(options.get("fields"))
        settings = self._settings(options)
        client = self._client_factory(settings)
        try:
            cursor = client[settings.database][collection].find(query, projection).limit(max_rows + 1)
            documents = list(cursor.max_time_ms(QUERY_TIMEOUT_MS))
        except SigmaError:
            raise
        except Exception as error:  # noqa: BLE001 - pymongo has many exception types
            raise ImportFailedError(self._describe(error, settings)) from error
        finally:
            client.close()
        truncated = len(documents) > max_rows
        if not documents:
            raise ImportFailedError("La colección no tiene documentos con ese filtro.")
        frame = pd.json_normalize([plain(d) for d in documents[:max_rows]], sep=".")
        detected = {
            "engine": ENGINES["mongodb"]["label"],
            "host": settings.host,
            "database": settings.database,
            "collection": collection,
            "rows_truncated": truncated,
            **({"query": str(options.get("filter")).strip()[:500]} if query else {}),
        }
        return frame, detected

    # ------------------------------------------------------------------ explorer
    def inspect(self, request: ImportRequest) -> dict[str, Any]:
        """Collections of the database and, for ``options['collection']``, the fields found in a sample of documents."""
        options = request.options
        settings = self._settings(options)
        client = self._client_factory(settings)
        try:
            database = client[settings.database]
            names = sorted(n for n in database.list_collection_names() if not n.startswith("system."))
            result: dict[str, Any] = {
                "tables": [{"name": n, "kind": "colección"} for n in names[:CATALOG_LIMIT]],
                "truncated": len(names) > CATALOG_LIMIT,
            }
            requested = str(options.get("collection") or options.get("table") or "").strip()
            if requested:
                self._collection_name({"collection": requested})
                sample = [
                    plain(d) for d in database[requested].find().limit(SAMPLE_DOCUMENTS).max_time_ms(QUERY_TIMEOUT_MS)
                ]
                frame = pd.json_normalize(sample, sep=".") if sample else pd.DataFrame()
                result["columns"] = [{"name": str(c), "type": self._type_of(frame[c])} for c in frame.columns]
            return result
        except SigmaError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ImportFailedError(self._describe(error, settings)) from error
        finally:
            client.close()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _type_of(series: pd.Series) -> str:
        first = series.dropna()
        return type(first.iloc[0]).__name__ if len(first) else "null"

    @staticmethod
    def _collection_name(options: dict[str, Any]) -> str:
        name = str(options.get("collection") or "").strip()
        if not name:
            raise InvalidInputError("Indique la colección de MongoDB.")
        if not _COLLECTION.match(name) or name.startswith("system."):
            raise InvalidInputError("El nombre de la colección contiene caracteres no permitidos.")
        return name

    @staticmethod
    def _projection(fields: Any) -> dict[str, int] | None:
        names = [f.strip() for f in str(fields or "").split(",") if f.strip()]
        for name in names:
            if not _FIELD.match(name):
                raise InvalidInputError(f"El nombre de campo «{name}» contiene caracteres no permitidos.")
        return {name: 1 for name in names} or None

    def _settings(self, options: dict[str, Any]) -> DatabaseSettings:
        extra: dict[str, str] = {}
        if as_bool(options.get("tls"), False):
            extra["tls"] = "true"
        if options.get("auth_source"):
            extra["authSource"] = str(options["auth_source"])
        settings = DatabaseSettings.from_request(
            {
                "engine": "mongodb",
                "host": options.get("host"),
                "port": options.get("port"),
                "database": options.get("database"),
                "username": options.get("username"),
                "password": options.get("password"),
                "options": extra,
            },
            sqlite_path="",
        )
        check_host_allowed(settings.host, self._allowed_hosts)
        return settings

    @staticmethod
    def _describe(error: Exception, settings: DatabaseSettings) -> str:
        label = ENGINES["mongodb"]["label"]
        return explain_driver_error(
            error, "mongodb", settings.password, prefix=f"No se pudo consultar {label} ({settings.host})"
        )
