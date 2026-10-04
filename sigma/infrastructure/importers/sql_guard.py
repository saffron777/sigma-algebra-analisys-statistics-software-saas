"""Read-only guard for user-supplied SQL: only a single SELECT/WITH statement is accepted, per SQL dialect.

This is the first of two layers. The second is the session itself: every external connection is opened read-only
(see ``external_sql_importer``), so even something the guard overlooked cannot write. A read-only account is still the
best protection.
"""

from __future__ import annotations

import re

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.importers.sql_scrub import scrub

_START = re.compile(r"^\s*\(*\s*(SELECT|WITH)\b", re.IGNORECASE)
_IDENTIFIER = re.compile(r"^[\w .\-$#@]{1,128}$", re.UNICODE)

_COMMON = r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|GRANT|REVOKE|INTO"
# what must not appear in a SELECT, by engine (matched on the text left after ``scrub``)
_FORBIDDEN_WORDS = {
    "mssql": _COMMON
    + r"|DENY|EXEC|EXECUTE|OPENROWSET|OPENQUERY|OPENDATASOURCE|BULK|WAITFOR|SHUTDOWN|DBCC|BACKUP|RESTORE|KILL"
    + r"|RECONFIGURE|XP_\w+|SP_\w+",
    "mysql": _COMMON
    + r"|CALL|LOAD|OUTFILE|DUMPFILE|LOAD_FILE|SLEEP|BENCHMARK|GET_LOCK|RELEASE_LOCK|IS_FREE_LOCK|LOCK|UNLOCK|HANDLER"
    + r"|PREPARE|DEALLOCATE",
    "postgresql": _COMMON
    + r"|CALL|DO|COPY|VACUUM|LISTEN|NOTIFY|LOCK|PG_SLEEP\w*|PG_READ_\w+|PG_LS_\w+|PG_STAT_FILE|PG_TERMINATE_BACKEND"
    + r"|PG_CANCEL_BACKEND|PG_RELOAD_CONF|PG_ROTATE_LOGFILE|PG_ADVISORY_\w+|PG_TRY_ADVISORY_\w+|PG_LOGICAL_\w+"
    + r"|PG_CREATE_\w+|PG_DROP_\w+|PG_STAT_RESET\w*|LO_\w+|DBLINK\w*|SET_CONFIG|NEXTVAL|SETVAL|QUERY_TO_XML\w*"
    + r"|CURSOR_TO_XML\w*|TABLE_TO_XML\w*|SCHEMA_TO_XML\w*|DATABASE_TO_XML\w*",
    "oracle": _COMMON + r"|CALL|LOCK|DBMS_\w+|UTL_\w+|HTTPURITYPE|SYS_XMLGEN|XMLGEN",
}
_FORBIDDEN_PHRASES = {
    "mssql": [],
    "mysql": [r"FOR\s+UPDATE", r"FOR\s+SHARE", r"LOCK\s+IN\s+SHARE\s+MODE"],
    "postgresql": [r"FOR\s+(NO\s+KEY\s+)?UPDATE", r"FOR\s+(KEY\s+)?SHARE"],
    "oracle": [r"FOR\s+UPDATE"],
}
_FORBIDDEN = {
    dialect: re.compile(
        r"\b(" + "|".join([words, *_FORBIDDEN_PHRASES[dialect]]) + r")\b",
        re.IGNORECASE,
    )
    for dialect, words in _FORBIDDEN_WORDS.items()
}


def validate_readonly_query(sql: str, dialect: str = "mssql") -> str:
    """Return the query without a trailing ``;`` or raise if it could modify anything."""
    if not sql or not sql.strip():
        raise InvalidInputError("Escriba una consulta SQL.")
    scrubbed = scrub(sql, dialect).strip().rstrip(";").strip()
    if ";" in scrubbed:
        raise InvalidInputError("Solo se permite una única sentencia SQL.")
    if not _START.match(scrubbed):
        raise InvalidInputError("Solo se permiten consultas que empiecen por SELECT o WITH.")
    forbidden = _FORBIDDEN[dialect].search(scrubbed)
    if forbidden:
        word = " ".join(forbidden.group(0).upper().split())
        raise InvalidInputError(
            f"La consulta contiene «{word}», que no está permitido: la importación es de solo lectura."
        )
    return sql.strip().rstrip(";").strip()


def quote_identifier(identifier: str, label: str, dialect: str) -> str:
    """A validated identifier in the engine's quoting (the allowed characters never include a quote)."""
    if not _IDENTIFIER.match(identifier or ""):
        raise InvalidInputError(f"El nombre de {label} contiene caracteres no permitidos.")
    if dialect == "mssql":
        return "[" + identifier + "]"
    if dialect == "mysql":
        return "`" + identifier + "`"
    return '"' + identifier + '"'


def build_table_query(table: str, schema: str | None, max_rows: int, dialect: str = "mssql") -> str:
    """``SELECT * FROM schema.table`` limited to ``max_rows + 1`` rows (the extra one reveals a truncation)."""
    target = quote_identifier(table, "la tabla", dialect)
    if schema:
        target = f"{quote_identifier(schema, 'el esquema', dialect)}.{target}"
    limit = int(max_rows) + 1
    if dialect == "mssql":
        return f"SELECT TOP ({limit}) * FROM {target}"
    if dialect == "oracle":
        return f"SELECT * FROM (SELECT * FROM {target}) WHERE ROWNUM <= {limit}"  # valid before 12c too
    return f"SELECT * FROM {target} LIMIT {limit}"
