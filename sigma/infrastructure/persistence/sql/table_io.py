"""Reading and writing a DataFrame as a table of its own (datasets and staged import previews), on any SQL engine.

The rows go to a table with a generated name (``ds_<24 hex>`` / ``stg_<24 hex>`` — short enough for Oracle), columns
``c0 … cN`` (so odd, long or duplicate-looking column names never meet an engine's identifier rules) and a ``_row``
key that gives a stable order for paging. The original column names live in the metadata row (``col_map`` / dtypes).
Tables created by older Sigma versions (SQLite, original column names, implicit ``rowid`` order) are read as they are.
"""

from __future__ import annotations

import math
import re
import uuid
from collections.abc import Iterator
from typing import Any

import pandas as pd
from pandas.api import types as pdt
from sqlalchemy import BigInteger, Column, Double, Integer, MetaData, Table, literal_column, select
from sqlalchemy.engine import Connection

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.persistence.frame_codec import decode_frame, encode_frame
from sigma.infrastructure.persistence.sql.schema import LongText

CHUNK_ROWS = 5000
# conservative column limits (the real ones depend on row size and server version)
MAX_COLUMNS = {"sqlite": 1900, "mssql": 280, "mysql": 900, "mariadb": 900, "postgresql": 1000, "oracle": 900}
_TABLE_NAME = re.compile(r"^(ds|stg)_[0-9a-f]{24,32}$")


def new_table_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _check_name(name: str) -> str:
    if not _TABLE_NAME.match(name or ""):
        raise InvalidInputError("Nombre de tabla de datos inválido.")
    return name


def column_kind(series: pd.Series) -> str:
    if pdt.is_bool_dtype(series):
        return "bool"
    if pdt.is_integer_dtype(series):
        return "int"
    if pdt.is_float_dtype(series):
        return "float"
    return "text"


_SQL_TYPES = {"bool": Integer, "int": BigInteger, "float": Double, "text": LongText}


def column_values(series: pd.Series, kind: str) -> list[Any]:
    """Plain Python values of a column slice, ready for any driver (NaN/NA → NULL, bool → 0/1, text always ``str``)."""
    values = series.astype(object).where(series.notna(), None).tolist()
    if kind == "int":
        return [None if v is None else int(v) for v in values]
    if kind == "bool":
        return [None if v is None else int(bool(v)) for v in values]
    if kind == "float":
        return [None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v) for v in values]
    return [None if v is None else (v if isinstance(v, str) else str(v)) for v in values]


def write_frame(conn: Connection, table_name: str, frame: pd.DataFrame) -> list[str]:
    """Create ``table_name`` and load ``frame`` (inside the caller's transaction); returns the physical columns."""
    _check_name(table_name)
    encoded, _dtypes = encode_frame(frame)
    names = list(encoded.columns)
    limit = MAX_COLUMNS.get(conn.dialect.name, 800)
    if len(names) > limit:
        raise InvalidInputError(
            f"El dataset tiene {len(names)} columnas; este motor de base de datos admite hasta {limit}."
        )
    physical = [f"c{i}" for i in range(len(names))]
    kinds = [column_kind(encoded[name]) for name in names]
    table = Table(
        table_name,
        MetaData(),
        Column("_row", BigInteger, primary_key=True, autoincrement=False),
        *[Column(p, _SQL_TYPES[k]) for p, k in zip(physical, kinds, strict=True)],
    )
    table.create(conn)
    for rows in _batches(encoded, kinds, physical):
        conn.execute(table.insert(), rows)
    return physical


def _batches(encoded: pd.DataFrame, kinds: list[str], physical: list[str]) -> Iterator[list[dict[str, Any]]]:
    total = len(encoded)
    for start in range(0, total, CHUNK_ROWS):
        chunk = encoded.iloc[start : start + CHUNK_ROWS]
        columns = [column_values(chunk.iloc[:, i], kinds[i]) for i in range(len(physical))]
        yield [
            {"_row": start + offset, **{p: columns[i][offset] for i, p in enumerate(physical)}}
            for offset in range(len(chunk))
        ]


def read_frame(
    conn: Connection,
    table_name: str,
    names: list[str],
    col_map: list[str] | None,
    dtypes: dict[str, str],
    limit: int | None = None,
    offset: int | None = None,
) -> pd.DataFrame:
    """Rows ``offset … offset+limit`` (all of them when ``limit`` is ``None``) with the original names and dtypes."""
    _check_name(table_name)
    physical = col_map if col_map is not None else names  # no map = a table from an older version (same names)
    columns = [Column(p) for p in physical]
    if col_map is not None:
        table = Table(table_name, MetaData(), Column("_row", BigInteger), *columns)
        order = table.c["_row"]
    else:
        table = Table(table_name, MetaData(), *columns)
        order = literal_column("rowid")  # only SQLite files can contain such tables
    statement = select(*[table.c[p] for p in physical]).order_by(order)
    if limit is not None:
        statement = statement.limit(int(limit)).offset(int(offset or 0))
    frame = pd.read_sql_query(statement, conn)
    frame.columns = names
    return decode_frame(frame, dtypes)


def drop_table(conn: Connection, table_name: str | None) -> None:
    if table_name:
        Table(_check_name(table_name), MetaData()).drop(conn, checkfirst=True)
