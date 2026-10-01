"""In-memory staging area for imports awaiting the user's confirmation."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

from sigma.application.ports import StagedImport
from sigma.domain.errors import InvalidInputError


class InMemoryImportStaging:
    def __init__(self, capacity: int = 8, ttl_seconds: float = 1800.0) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._items: OrderedDict[str, tuple[float, StagedImport]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, staged: StagedImport) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._evict()
            self._items[token] = (time.monotonic(), staged)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)
        return token

    def take(self, token: str) -> StagedImport:
        with self._lock:
            self._evict()
            entry = self._items.pop(token, None)
        if entry is None:
            raise InvalidInputError("La vista previa expiró o ya fue guardada. Vuelva a cargar la fuente de datos.")
        return entry[1]

    def _evict(self) -> None:
        now = time.monotonic()
        for token in [t for t, (stamp, _) in self._items.items() if now - stamp > self._ttl]:
            del self._items[token]
