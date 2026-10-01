"""Maps domain/validation/HTTP errors to JSON (API) or HTML (pages) responses."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

from sigma.domain.errors import (
    ComputationError,
    DatasetNotFoundError,
    ImportFailedError,
    InvalidInputError,
    KernelError,
    NotebookNotFoundError,
    ResultNotFoundError,
    SigmaError,
)

log = logging.getLogger(__name__)

_STATUS = {
    InvalidInputError: 422,
    ComputationError: 422,
    ImportFailedError: 422,
    DatasetNotFoundError: 404,
    ResultNotFoundError: 404,
    NotebookNotFoundError: 404,
    KernelError: 503,
}
_HTTP_MESSAGES = {
    400: "Solicitud inválida.",
    403: "Acceso denegado.",
    404: "Recurso no encontrado.",
    405: "Método no permitido.",
    413: "El archivo supera el tamaño máximo permitido.",
}


def _wants_json() -> bool:
    return request.path.startswith("/api/")


def _error_body(code: str, message: str, details=None) -> dict:
    return {"error": {"code": code, "message": message, "details": details}}


_VALIDATION_MESSAGES = {
    "missing": "es obligatorio",
    "float_parsing": "debe ser un número",
    "int_parsing": "debe ser un número entero",
    "int_from_float": "debe ser un número entero",
    "string_type": "debe ser texto",
    "bool_parsing": "debe ser verdadero o falso",
    "list_type": "debe ser una lista",
    "dict_type": "debe ser un objeto",
    "too_short": "debe tener al menos {min_length} elemento(s)",
    "too_long": "admite como máximo {max_length} elemento(s)",
    "string_too_short": "no puede estar vacío",
    "greater_than": "debe ser mayor que {gt}",
    "greater_than_equal": "debe ser mayor o igual que {ge}",
    "less_than": "debe ser menor que {lt}",
    "less_than_equal": "debe ser menor o igual que {le}",
    "literal_error": "valor no permitido (opciones: {expected})",
    "finite_number": "debe ser un número finito",
}


def _friendly_validation(error: ValidationError) -> list[dict[str, str]]:
    details = []
    for item in error.errors(include_url=False, include_context=True, include_input=False):
        template = _VALIDATION_MESSAGES.get(item["type"])
        try:
            message = template.format(**item.get("ctx", {})) if template else item["msg"]
        except (KeyError, IndexError):
            message = item["msg"]
        details.append({"field": ".".join(str(p) for p in item["loc"]), "message": message})
    return details


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(SigmaError)
    def handle_sigma_error(error: SigmaError):
        status = next((code for cls, code in _STATUS.items() if isinstance(error, cls)), 400)
        return jsonify(_error_body(error.code, error.message, error.details)), status

    @app.errorhandler(ValidationError)
    def handle_validation_error(error: ValidationError):
        details = _friendly_validation(error)
        summary = "; ".join(f"{d['field']}: {d['message']}" for d in details[:3])
        return jsonify(_error_body("validation_error", f"Datos inválidos — {summary}", details)), 422

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        message = (
            error.description
            if error.code in (400, 403) and error.description
            else _HTTP_MESSAGES.get(error.code or 500, error.name)
        )
        if _wants_json():
            return jsonify(_error_body(f"http_{error.code}", message)), error.code or 500
        return render_template("pages/error.html", status=error.code, message=message), error.code or 500

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.path)
        message = "Ocurrió un error inesperado en el servidor."
        if _wants_json():
            return jsonify(_error_body("internal_error", message)), 500
        return render_template("pages/error.html", status=500, message=message), 500
