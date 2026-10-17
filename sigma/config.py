"""Application configuration, read from environment variables (and an optional ``.env`` file)."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from sigma.infrastructure.persistence.database_settings import DatabaseSettings, load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Minimal ``KEY=VALUE`` loader; real environment variables always win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _load_or_create_secret(instance_dir: Path) -> str:
    """Persist a random secret key so sessions survive restarts without being committed anywhere."""
    path = instance_dir / "secret.key"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    instance_dir.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32)
    path.write_text(key, encoding="utf-8")
    return key


# Values copied from the example file: they are not a real secret, so they are ignored.
PLACEHOLDER_SECRETS = frozenset({"cambia-esta-clave", "cambiame", "changeme", "change-me", "secret"})


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 5000
    secret_key: str = ""
    source_url: str = "https://github.com/saffron777/Sigma"
    instance_dir: Path = PROJECT_ROOT / "instance"
    db_path: Path = PROJECT_ROOT / "instance" / "sigma.db"  # the SQLite file used when no other database is chosen
    database: DatabaseSettings | None = None  # which database to persist to (None = SQLite at db_path)
    notebooks_dir: Path = PROJECT_ROOT / "instance" / "notebooks"
    sample_dir: Path = PROJECT_ROOT / "sample_data"
    max_upload_bytes: int = 100 * 1024 * 1024
    analysis_timeout_seconds: int = 120  # heavy analyses (models, forests, bootstrap…) run in a process with this limit
    analysis_max_parallel: int = 2  # heavy analyses at once
    external_db_hosts: tuple[str, ...] = ()  # servers the data page may read from (empty = any)
    cell_timeout_seconds: int = 60
    kernel_idle_minutes: int = 30
    max_kernels_per_user: int = 3
    max_kernels_total: int = 8
    session_hours: float = 12.0
    draft_retention_days: int = 14
    ai_requests_per_hour: int = 20
    secure_cookies: bool = False
    allowed_hosts: tuple[str, ...] = ("127.0.0.1", "localhost", "[::1]", "::1")
    testing: bool = False
    debug: bool = False

    @classmethod
    def from_env(cls, **overrides) -> Config:
        load_dotenv(PROJECT_ROOT / ".env")
        instance_dir = Path(os.environ.get("SIGMA_INSTANCE_DIR", PROJECT_ROOT / "instance"))
        values = {
            "host": os.environ.get("SIGMA_HOST", "127.0.0.1"),
            "port": int(os.environ.get("SIGMA_PORT", "5000")),
            "source_url": os.environ.get("SIGMA_SOURCE_URL", "https://github.com/saffron777/Sigma").strip(),
            "instance_dir": instance_dir,
            "db_path": Path(os.environ.get("SIGMA_DB_PATH", instance_dir / "sigma.db")),
            "notebooks_dir": Path(os.environ.get("SIGMA_NOTEBOOKS_DIR", instance_dir / "notebooks")),
            "max_upload_bytes": int(os.environ.get("SIGMA_MAX_UPLOAD_MB", "100")) * 1024 * 1024,
            "analysis_timeout_seconds": int(os.environ.get("SIGMA_ANALYSIS_TIMEOUT_SECONDS", "120")),
            "analysis_max_parallel": int(os.environ.get("SIGMA_ANALYSIS_MAX_PARALLEL", "2")),
            "external_db_hosts": tuple(
                h.strip() for h in os.environ.get("SIGMA_EXTERNAL_DB_HOSTS", "").split(",") if h.strip()
            ),
            "cell_timeout_seconds": int(os.environ.get("SIGMA_CELL_TIMEOUT_SECONDS", "60")),
            "kernel_idle_minutes": int(os.environ.get("SIGMA_KERNEL_IDLE_MINUTES", "30")),
            "max_kernels_per_user": int(os.environ.get("SIGMA_MAX_KERNELS_PER_USER", "3")),
            "max_kernels_total": int(os.environ.get("SIGMA_MAX_KERNELS_TOTAL", "8")),
            "session_hours": float(os.environ.get("SIGMA_SESSION_HOURS", "12")),
            "draft_retention_days": int(os.environ.get("SIGMA_DRAFT_RETENTION_DAYS", "14")),
            "ai_requests_per_hour": int(os.environ.get("SIGMA_AI_REQUESTS_PER_HOUR", "20")),
            "secure_cookies": os.environ.get("SIGMA_SECURE_COOKIES", "").lower() in {"1", "true", "yes"},
            "debug": os.environ.get("SIGMA_DEBUG", "").lower() in {"1", "true", "yes"},
        }
        extra_hosts = tuple(h.strip() for h in os.environ.get("SIGMA_ALLOWED_HOSTS", "").split(",") if h.strip())
        if extra_hosts:
            values["allowed_hosts"] = (*cls.allowed_hosts, *extra_hosts)
        values.update(overrides)
        if values.get("database") is None:
            values["database"] = (
                DatabaseSettings.sqlite(values["db_path"])
                if values.get("testing")  # tests never read the developer's environment or instance/database.json
                else load_settings(values["instance_dir"], values["db_path"])
            )
        secret = values.pop("secret_key", None) or os.environ.get("SIGMA_SECRET_KEY") or ""
        if secret.strip().lower() in PLACEHOLDER_SECRETS:
            secret = ""  # copied example value: not a real secret
        if not secret:
            secret = secrets.token_hex(32) if values.get("testing") else _load_or_create_secret(values["instance_dir"])
        return cls(secret_key=secret, **values)
