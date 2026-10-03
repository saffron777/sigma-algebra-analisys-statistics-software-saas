"""A DataFrame as documents of a few thousand rows each (datasets and staged previews), keeping the pandas dtypes."""

from __future__ import annotations

import uuid
from typing import Any

import pandas as pd

from sigma.infrastructure.persistence.frame_codec import decode_frame, encode_frame
from sigma.infrastructure.persistence.mongo.core import MongoCore
from sigma.infrastructure.persistence.sql.table_io import column_kind, column_values

CHUNK_TARGET_BYTES = 4 * 1024 * 1024  # well below MongoDB's 16 MB document limit
MAX_CHUNK_ROWS = 5000
COLLECTION = "frame_chunks"


def new_owner(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _weight(value: Any) -> int:
    return len(value) + 8 if isinstance(value, str) else 10


def write_frame(core: MongoCore, owner: str, frame: pd.DataFrame) -> None:
    encoded, _dtypes = encode_frame(frame)
    names = list(encoded.columns)
    kinds = [column_kind(encoded[name]) for name in names]
    chunks = core.col(COLLECTION)
    seq = 0

    def store(rows: list[list[Any]], start: int) -> None:
        nonlocal seq
        chunks.insert_one({"owner": owner, "seq": seq, "start": start, "end": start + len(rows), "rows": rows})
        seq += 1

    total = len(encoded)
    for low in range(0, total, MAX_CHUNK_ROWS):
        high = min(total, low + MAX_CHUNK_ROWS)
        columns = [column_values(encoded.iloc[low:high, i], kinds[i]) for i in range(len(names))]
        rows = [list(row) for row in zip(*columns, strict=True)] if names else [[] for _ in range(high - low)]
        group: list[list[Any]] = []
        weight, group_start = 0, low
        for row in rows:
            group.append(row)
            weight += sum(_weight(v) for v in row)
            if weight >= CHUNK_TARGET_BYTES:  # text-heavy rows: close the document early
                store(group, group_start)
                group_start += len(group)
                group, weight = [], 0
        if group:
            store(group, group_start)
    if total == 0:
        store([], 0)


def read_frame(
    core: MongoCore,
    owner: str,
    names: list[str],
    dtypes: dict[str, str],
    limit: int | None = None,
    offset: int | None = None,
) -> pd.DataFrame:
    offset = int(offset or 0)
    query: dict[str, Any] = {"owner": owner}
    if limit is not None:
        query["end"] = {"$gt": offset}
        query["start"] = {"$lt": offset + int(limit)}
    rows: list[list[Any]] = []
    first_start = None
    for chunk in core.col(COLLECTION).find(query).sort("seq", 1):
        if first_start is None:
            first_start = chunk["start"]
        rows.extend(chunk["rows"])
    if limit is not None and rows:
        skip = offset - first_start
        rows = rows[max(skip, 0) : max(skip, 0) + int(limit)]
    frame = pd.DataFrame(rows, columns=names)
    return decode_frame(frame, dtypes)


def drop_frame(core: MongoCore, owner: str | None) -> None:
    if owner:
        core.col(COLLECTION).delete_many({"owner": owner})
