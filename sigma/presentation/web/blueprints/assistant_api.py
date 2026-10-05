"""Asistente guiado: qué análisis usar, por pregunta y tipos de variable."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from sigma.domain.assistant import GOALS, KINDS, recommend
from sigma.domain.errors import InvalidInputError

bp = Blueprint("assistant_api", __name__, url_prefix="/api/assistant")


@bp.get("/catalog")
def catalog():
    return jsonify(
        {
            "goals": [{"id": key, "label": label} for key, label in GOALS],
            "kinds": [{"id": key, "label": label} for key, label in KINDS],
        }
    )


@bp.get("/recommend")
def recommend_analysis():
    goal = request.args.get("goal", "")
    primary = request.args.get("primary", "")
    secondary = request.args.get("secondary") or None
    valid_goals = {key for key, _ in GOALS}
    valid_kinds = {key for key, _ in KINDS}
    if goal not in valid_goals or primary not in valid_kinds:
        raise InvalidInputError("Elija una pregunta y un tipo de variable válidos.")
    if secondary is not None and secondary not in valid_kinds:
        raise InvalidInputError("El tipo de la segunda variable no es válido.")
    rec = recommend(goal, primary, secondary)
    return jsonify({"module": rec.module, "label": rec.label, "href": rec.href, "reason": rec.reason})
