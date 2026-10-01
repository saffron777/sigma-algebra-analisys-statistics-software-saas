"""In-memory brute-force protection for the login form."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from sigma.domain.errors import RateLimitedError


class InMemoryLoginThrottle:
    """Blocks a (username, IP) pair after too many failures, and an IP after many failures overall."""

    def __init__(
        self,
        max_failures: int = 5,
        window_seconds: float = 300.0,
        lockout_seconds: float = 300.0,
        max_ip_failures: int = 25,
        clock=time.monotonic,
    ) -> None:
        self._max = max_failures
        self._window = window_seconds
        self._lockout = lockout_seconds
        self._max_ip = max_ip_failures
        self._clock = clock
        self._failures: dict[str, deque[float]] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _keys(username: str, ip: str) -> tuple[str, str]:
        return f"u:{username.strip().lower()}|{ip}", f"ip:{ip}"

    def check(self, username: str, ip: str) -> None:
        now = self._clock()
        with self._lock:
            for key in self._keys(username, ip):
                until = self._locked_until.get(key, 0.0)
                if until > now:
                    raise RateLimitedError(
                        "Demasiados intentos fallidos. Espere unos minutos antes de volver a intentarlo.",
                        {"retry_after": int(until - now) + 1},
                    )

    def record_failure(self, username: str, ip: str) -> None:
        now = self._clock()
        user_key, ip_key = self._keys(username, ip)
        with self._lock:
            for key, limit in ((user_key, self._max), (ip_key, self._max_ip)):
                attempts = self._failures[key]
                attempts.append(now)
                while attempts and now - attempts[0] > self._window:
                    attempts.popleft()
                if len(attempts) >= limit:
                    self._locked_until[key] = now + self._lockout
                    attempts.clear()

    def reset(self, username: str, ip: str) -> None:
        user_key, _ = self._keys(username, ip)
        with self._lock:
            self._failures.pop(user_key, None)
            self._locked_until.pop(user_key, None)
