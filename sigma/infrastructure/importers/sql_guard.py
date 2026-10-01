"""Read-only guard for user-supplied SQL: only a single SELECT/WITH statement is accepted."""

from __future__ import annotations

import re

from sigma.domain.errors import InvalidInputError

_COMMENTS = re.compile(r"/\*.*?\*/|--[^\r\n]*", re.DOTALL)
_LITERALS_AND_IDENTIFIERS = re.compile(r"'(?:[^']|'')*'|\[[^\]]*\]|\"(?:[^\"]|\"\")*\"")
_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|MERGE|GRANT|REVOKE|DENY|INTO|"
    r"OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|WAITFOR|SHUTDOWN|DBCC|BACKUP|RESTORE|KILL|RECONFIGURE|"
    r"XP_\w+|SP_\w+)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[\w .\-$#@]{1,128}$", re.UNICODE)


def validate_readonly_query(sql: str) -> str:
    """Return the query without a trailing ``;`` or raise if it could modify anything."""
    if not sql or not sql.strip():
        raise InvalidInputError("Escriba una consulta SQL.")
    scrubbed = _LITERALS_AND_IDENTIFIERS.sub("''", _COMMENTS.sub(" ", sql)).strip().rstrip(";").strip()
    if ";" in scrubbed:
        raise InvalidInputError("Solo se permite una única sentencia SQL.")
    if not _START.match(scrubbed):
        raise InvalidInputError("Solo se permiten consultas que empiecen por SELECT o WITH.")
    forbidden = _FORBIDDEN.search(scrubbed)
    if forbidden:
        raise InvalidInputError(
            f"La consulta contiene «{forbidden.group(0).upper()}», que no está permitido: "
            "la importación es de solo lectura."
        )
    return sql.strip().rstrip(";").strip()


def _bracket(identifier: str, label: str) -> str:
    if not _IDENTIFIER.match(identifier or ""):
        raise InvalidInputError(f"El nombre de {label} contiene caracteres no permitidos.")
    return "[" + identifier.replace("]", "]]") + "]"


def build_table_query(table: str, schema: str | None, max_rows: int) -> str:
    """``SELECT TOP (n) * FROM [schema].[table]`` with validated, bracket-quoted identifiers."""
    target = _bracket(table, "la tabla")
    if schema:
        target = f"{_bracket(schema, 'el esquema')}.{target}"
    return f"SELECT TOP ({int(max_rows) + 1}) * FROM {target}"
