"""Blanks out comments, string literals and quoted identifiers of a SQL text, following each engine's own lexical rules.

The read-only guard (``sql_guard``) looks for forbidden words in what is left. A regular expression is not enough for
that: every engine delimits text differently (``$$…$$`` and nested comments in PostgreSQL, ``q'[…]'`` in Oracle,
backslash escapes and ``/*! … */`` in MySQL…) and a guard that disagrees with the server about where a literal ends
can be fooled into ignoring code the server will run. So this scanner reads the text the way the server does.
Whatever it cannot read unambiguously (a literal or comment that never closes) is rejected.
"""

from __future__ import annotations

import re

from sigma.domain.errors import InvalidInputError

DIALECTS = ("mssql", "mysql", "postgresql", "oracle")
_IDENTIFIER_CHAR = re.compile(r"[\w$#@\u0080-￿]")
_DOLLAR_TAG = re.compile(r"\$(?:[^\W\d]\w*)?\$")
_LINE_END = "\r\n"
_BLANKS = " \t\r\n\f\v"
_Q_CLOSERS = {"[": "]", "{": "}", "(": ")", "<": ">"}
_COMMENT, _LITERAL, _IDENT = " ", "''", '""'


def scrub(sql: str, dialect: str) -> str:
    """``sql`` with every comment replaced by a space, every literal by ``''`` and every quoted identifier by ``""``."""
    if dialect not in DIALECTS:
        raise ValueError(f"unknown SQL dialect {dialect!r}")
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        following = sql[i + 1] if i + 1 < n else ""
        if c == "-" and following == "-" and _line_comment_starts(sql, i, dialect) or c == "#" and dialect == "mysql":
            i = _skip_line(sql, i)
            out.append(_COMMENT)
        elif c == "/" and following == "*":
            i = _skip_block_comment(sql, i, dialect)
            out.append(_COMMENT)
        elif c == "'" or (c == '"' and dialect == "mysql"):
            if c == "'" and _is_oracle_q_literal(out, dialect):
                i = _skip_oracle_q(sql, i)
            else:
                i = _skip_literal(sql, i, c, _backslash_escapes(out, dialect))
            out.append(_LITERAL)
        elif c == '"':
            i = _skip_quoted(sql, i, '"')
            out.append(_IDENT)
        elif c == "`" and dialect == "mysql":
            i = _skip_quoted(sql, i, "`")
            out.append(_IDENT)
        elif c == "[" and dialect == "mssql":
            i = _skip_quoted(sql, i, "]")
            out.append(_IDENT)
        elif c == "$" and dialect == "postgresql" and _starts_dollar_quote(sql, i, out):
            i = _skip_dollar_quote(sql, i)
            out.append(_LITERAL)
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ------------------------------------------------------------------ comments
def _line_comment_starts(sql: str, i: int, dialect: str) -> bool:
    if dialect != "mysql":
        return True
    after = sql[i + 2 : i + 3]  # MySQL needs whitespace (or the end) after the two dashes
    return after == "" or after in _BLANKS or ord(after) < 32


def _skip_line(sql: str, i: int) -> int:
    n = len(sql)
    while i < n and sql[i] not in _LINE_END:
        i += 1
    return i


def _skip_block_comment(sql: str, i: int, dialect: str) -> int:
    if dialect == "mysql" and sql[i + 2 : i + 3] == "!":
        raise InvalidInputError("Los comentarios ejecutables de MySQL (/*! … */) no están permitidos.")
    depth, n = 1, len(sql)
    i += 2
    while i < n:
        if sql.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        elif dialect in ("postgresql", "mssql") and sql.startswith("/*", i):
            depth += 1  # these engines nest block comments
            i += 2
        else:
            i += 1
    raise InvalidInputError("Hay un comentario /* sin cerrar en la consulta.")


# ------------------------------------------------------------------ literals
def _backslash_escapes(out: list[str], dialect: str) -> bool:
    """Whether the literal about to start treats ``\\`` as an escape (always in MySQL, ``E'…'`` in PostgreSQL)."""
    if dialect == "mysql":
        return True
    if dialect == "postgresql":
        previous = out[-1] if out else ""
        before = out[-2] if len(out) > 1 else ""
        return previous in ("e", "E") and not _IDENTIFIER_CHAR.match(before or " ")
    return False


def _skip_literal(sql: str, i: int, quote: str, backslash: bool) -> int:
    n = len(sql)
    i += 1
    while i < n:
        c = sql[i]
        if backslash and c == "\\":
            i += 2
        elif c == quote:
            if sql[i + 1 : i + 2] == quote:  # a doubled quote is an escaped quote
                i += 2
            else:
                return i + 1
        else:
            i += 1
    raise InvalidInputError("Hay un texto entre comillas sin cerrar en la consulta.")


def _is_oracle_q_literal(out: list[str], dialect: str) -> bool:
    """``q'[…]'`` / ``nq'[…]'``: the quote right after a lone ``q`` (optionally ``n``) starts alternative quoting."""
    if dialect != "oracle" or not out or out[-1] not in ("q", "Q"):
        return False
    before = out[-2] if len(out) > 1 else ""
    if before in ("n", "N"):
        before = out[-3] if len(out) > 2 else ""
    return not _IDENTIFIER_CHAR.match(before or " ")


def _skip_oracle_q(sql: str, i: int) -> int:
    delimiter = sql[i + 1 : i + 2]
    if not delimiter or delimiter in _BLANKS:
        raise InvalidInputError("Hay un texto q'…' mal formado en la consulta.")
    closer = _Q_CLOSERS.get(delimiter, delimiter)
    end = sql.find(closer + "'", i + 2)
    if end < 0:
        raise InvalidInputError("Hay un texto q'…' sin cerrar en la consulta.")
    return end + 2


def _starts_dollar_quote(sql: str, i: int, out: list[str]) -> bool:
    if out and _IDENTIFIER_CHAR.match(out[-1][-1]):  # ``a$b$`` is an identifier, ``$1`` a parameter
        return False
    return _DOLLAR_TAG.match(sql, i) is not None


def _skip_dollar_quote(sql: str, i: int) -> int:
    tag = _DOLLAR_TAG.match(sql, i).group(0)  # type: ignore[union-attr]
    end = sql.find(tag, i + len(tag))
    if end < 0:
        raise InvalidInputError("Hay un texto $$…$$ sin cerrar en la consulta.")
    return end + len(tag)


# ------------------------------------------------------------------ quoted identifiers
def _skip_quoted(sql: str, i: int, closer: str) -> int:
    n = len(sql)
    i += 1
    while i < n:
        if sql[i] == closer:
            if sql[i + 1 : i + 2] == closer:  # a doubled closer is an escaped one
                i += 2
                continue
            return i + 1
        i += 1
    raise InvalidInputError("Hay un nombre entre comillas o corchetes sin cerrar en la consulta.")
