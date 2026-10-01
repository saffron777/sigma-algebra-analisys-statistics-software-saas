"""Importer for external JSON APIs (REST endpoints returning JSON)."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.common import as_int

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
ALLOWED_METHODS = ("GET", "POST")


def parse_pairs(raw: Any, separator: str, label: str) -> dict[str, str]:
    """Accept a dict or text with one ``key<separator>value`` pair per line."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    pairs: dict[str, str] = {}
    for line in str(raw).replace("&", "\n").splitlines():
        if not line.strip():
            continue
        key, found, value = line.partition(separator)
        if not found or not key.strip():
            raise InvalidInputError(f"{label}: use el formato «clave{separator}valor», una por línea.")
        pairs[key.strip()] = value.strip()
    return pairs


def _walk(document: Any, path: str) -> Any:
    current = document
    for key in (p for p in path.split(".") if p):
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            raise InvalidInputError(f"La ruta de registros «{path}» no existe en la respuesta (falló en «{key}»).")
    return current


def _lists_of_objects(node: Any, depth: int) -> list[list]:
    if isinstance(node, list):
        return [node] if node and isinstance(node[0], dict) else []
    if isinstance(node, dict) and depth > 0:
        return [found for child in node.values() for found in _lists_of_objects(child, depth - 1)]
    return []


def _find_records(document: Any) -> Any:
    """Heuristic: the document itself if it is a list, else the largest list of objects (up to 3 levels deep)."""
    if isinstance(document, list):
        return document
    if isinstance(document, dict):
        candidates = _lists_of_objects(document, depth=3)
        return max(candidates, key=len) if candidates else [document]
    raise ImportFailedError("La respuesta JSON no contiene registros utilizables.")


class JsonApiImporter:
    source_type = "json_api"

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        options = request.options
        url = str(options.get("url") or "").strip()
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise InvalidInputError("Indique una URL válida que empiece por http:// o https://.")
        method = str(options.get("method") or "GET").upper()
        if method not in ALLOWED_METHODS:
            raise InvalidInputError("El método debe ser GET o POST.")
        timeout = as_int(options.get("timeout"), "Tiempo de espera", DEFAULT_TIMEOUT_SECONDS, minimum=1)
        max_bytes = as_int(options.get("max_bytes"), "Tamaño máximo", DEFAULT_MAX_BYTES, minimum=1)
        body = self._parse_body(options.get("body"))

        document = self._download(
            method,
            url,
            parse_pairs(options.get("headers"), ":", "Cabeceras"),
            parse_pairs(options.get("params"), "=", "Parámetros"),
            body,
            timeout,
            max_bytes,
        )
        path = str(options.get("records_path") or "").strip()
        records = _walk(document, path) if path else _find_records(document)
        frame = self._to_frame(records)
        safe_url = urlunsplit(
            (parts.scheme, parts.hostname + (f":{parts.port}" if parts.port else ""), parts.path, "", "")
        )
        return frame, {"url": safe_url, "method": method, "records_path": path or None}

    @staticmethod
    def _parse_body(raw: Any) -> Any:
        if raw in (None, ""):
            return None
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidInputError(f"El cuerpo de la petición no es un JSON válido: {error.msg}.") from error

    @staticmethod
    def _download(method, url, headers, params, body, timeout, max_bytes) -> Any:
        try:
            with requests.request(
                method, url, headers=headers, params=params, json=body, timeout=(5, timeout), stream=True
            ) as response:
                response.raise_for_status()
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_content(chunk_size=65536):
                    received += len(chunk)
                    if received > max_bytes:
                        raise ImportFailedError(
                            f"La respuesta supera el tamaño máximo permitido ({max_bytes // (1024 * 1024)} MB)."
                        )
                    chunks.append(chunk)
        except requests.HTTPError as error:
            raise ImportFailedError(f"La API respondió con error HTTP {error.response.status_code}.") from error
        except requests.RequestException as error:
            raise ImportFailedError(f"No se pudo conectar con la API: {type(error).__name__}.") from error
        try:
            return json.loads(b"".join(chunks).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ImportFailedError("La respuesta de la API no es un JSON válido.") from error

    @staticmethod
    def _to_frame(records: Any) -> pd.DataFrame:
        if isinstance(records, dict):
            records = [records]
        if not isinstance(records, list):
            raise ImportFailedError("La ruta indicada no apunta a una lista de registros.")
        if not records:
            raise ImportFailedError("La API devolvió una lista vacía.")
        if all(isinstance(r, dict) for r in records):
            return pd.json_normalize(records, sep=".")
        return pd.DataFrame({"valor": records})
