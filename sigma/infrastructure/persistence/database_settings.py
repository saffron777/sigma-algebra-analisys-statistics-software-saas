"""Which database Sigma Algebra persists to, and how to reach it.

The choice cannot live in the database itself, so it is read, in this order, from:

1. the ``SIGMA_DATABASE_URL`` environment variable (deployments),
2. ``instance/database.json`` (written by the setup wizard / the administration page; mode 0600),
3. SQLite in ``instance/sigma.db`` (the default).

A setting is canonically a URL (``mssql+pyodbc://user:pass@host/db?driver=...``); the JSON file keeps the same data as
fields so the administration page can edit them. The password never leaves this module except inside the URL handed
to the engine / the kernels.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.engine import URL, make_url

from sigma.domain.errors import InvalidInputError

SETTINGS_FILE = "database.json"

# engine key -> what the UI and the factory need to know about it
ENGINES: dict[str, dict[str, Any]] = {
    "sqlite": {
        "label": "SQLite (archivo local)",
        "drivername": "sqlite",
        "default_port": None,
        "package": None,
        "pip": None,
        "relational": True,
    },
    "mssql": {
        "label": "Microsoft SQL Server",
        "drivername": "mssql+pyodbc",
        "default_port": 1433,
        "package": "pyodbc",
        "pip": "pyodbc (y el controlador «ODBC Driver 17/18 for SQL Server»)",
        "relational": True,
    },
    "mysql": {
        "label": "MySQL / MariaDB",
        "drivername": "mysql+pymysql",
        "default_port": 3306,
        "package": "pymysql",
        "pip": "pymysql",
        "relational": True,
    },
    "postgresql": {
        "label": "PostgreSQL",
        "drivername": "postgresql+psycopg",
        "default_port": 5432,
        "package": "psycopg",
        "pip": "psycopg[binary]",
        "relational": True,
    },
    "oracle": {
        "label": "Oracle Database",
        "drivername": "oracle+oracledb",
        "default_port": 1521,
        "package": "oracledb",
        "pip": "oracledb",
        "relational": True,
    },
    "mongodb": {
        "label": "MongoDB",
        "drivername": "mongodb",
        "default_port": 27017,
        "package": "pymongo",
        "pip": "pymongo",
        "relational": False,
    },
}


# URL parameters a request may set per engine (anything else is ignored): the driver is never handed free-form options
ALLOWED_OPTIONS: dict[str, dict[str, Any]] = {
    "mssql": {
        "driver": None,
        "TrustServerCertificate": {"yes", "no"},
        "Encrypt": {"yes", "no", "strict"},
        "trusted_connection": {"yes", "no"},
    },
    "mysql": {"ssl_disabled": {"true", "false"}},
    "postgresql": {"sslmode": {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}},
    "oracle": {},
    "mongodb": {"tls": {"true", "false"}, "authSource": None, "replicaSet": None},
    "sqlite": {},
}
_HOST = re.compile(r"^[A-Za-z0-9._\-\\[\]:]{1,255}$")
_NAME = re.compile(r"^[\w .\-$#]{1,128}$")
_FREE = re.compile(r"^[^\x00-\x1f\x7f]{0,256}$")


def engine_of(drivername: str) -> str:
    base = drivername.split("+", 1)[0].lower()
    if base == "postgres":
        base = "postgresql"
    if base == "mariadb":
        base = "mysql"
    if base not in ENGINES:
        raise InvalidInputError(f"Motor de base de datos no admitido: «{drivername}».")
    return base


def _clean_options(engine: str, raw: Mapping[str, Any]) -> dict[str, str]:
    allowed = ALLOWED_OPTIONS.get(engine, {})
    cleaned: dict[str, str] = {}
    for key, value in raw.items():
        if key not in allowed or value in (None, ""):
            continue
        value = str(value).strip()
        choices = allowed[key]
        if choices is not None and value not in choices:
            raise InvalidInputError(f"Valor no válido para «{key}».")
        if choices is None and not _NAME.match(value.replace("(", "").replace(")", "")):
            raise InvalidInputError(f"Valor no válido para «{key}».")
        cleaned[key] = value
    return cleaned


def driver_available(engine: str) -> bool:
    package = ENGINES[engine]["package"]
    return package is None or importlib.util.find_spec(package) is not None


def default_odbc_driver() -> str:
    """The best SQL Server ODBC driver installed (newest first); falls back to the 18 name."""
    try:
        import pyodbc

        installed = set(pyodbc.drivers())
    except Exception:  # noqa: BLE001 - no pyodbc / no ODBC manager: the connection test explains it
        installed = set()
    for name in (
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server",
    ):
        if name in installed:
            return name
    return "ODBC Driver 18 for SQL Server"


@dataclass(frozen=True)
class DatabaseSettings:
    engine: str = "sqlite"
    path: str = ""  # SQLite file
    host: str = ""
    port: int | None = None
    database: str = ""  # database / schema / service name
    username: str = ""
    password: str = field(default="", repr=False)
    options: Mapping[str, str] = field(default_factory=dict)  # extra URL parameters (driver, sslmode, …)
    schema: str = ""  # a dedicated schema inside a shared database (SQL Server / PostgreSQL / Oracle)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def sqlite(cls, path: str | Path) -> DatabaseSettings:
        return cls(engine="sqlite", path=str(path))

    @classmethod
    def from_url(cls, text: str) -> DatabaseSettings:
        url = make_url(text)
        engine = engine_of(url.drivername)
        if engine == "sqlite":
            return cls.sqlite(url.database or "sigma.db")
        options = {k: (v if isinstance(v, str) else ",".join(v)) for k, v in url.query.items()}
        schema = options.pop("sigma_schema", "")
        if engine == "oracle" and "service_name" in options:
            database = options.pop("service_name")
        else:
            database = url.database or ""
        return cls(
            engine=engine,
            host=url.host or "",
            port=url.port,
            database=database,
            username=url.username or "",
            password=url.password or "",
            options=options,
            schema=schema,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatabaseSettings:
        engine = engine_of(str(data.get("engine", "sqlite")))
        if engine == "sqlite":
            return cls.sqlite(str(data.get("path", "")))
        port = data.get("port")
        return cls(
            engine=engine,
            host=str(data.get("host", "")).strip(),
            port=int(port) if port not in (None, "") else None,
            database=str(data.get("database", "")).strip(),
            username=str(data.get("username", "")).strip(),
            password=str(data.get("password", "")),
            options={str(k): str(v) for k, v in (data.get("options") or {}).items() if str(v) != ""},
            schema=str(data.get("schema", "")).strip(),
        )

    # ------------------------------------------------------------------ views
    @property
    def is_sqlite(self) -> bool:
        return self.engine == "sqlite"

    @property
    def is_relational(self) -> bool:
        return bool(ENGINES[self.engine]["relational"])

    def validate(self) -> None:
        """Required fields per engine, with messages the setup form can show as they are."""
        if self.is_sqlite:
            if not self.path:
                raise InvalidInputError("Indique la ruta del archivo SQLite.")
            return
        if not self.host:
            raise InvalidInputError("Indique el servidor (host).")
        if not self.database and self.engine != "mongodb":
            raise InvalidInputError("Indique el nombre de la base de datos.")
        if self.engine == "mongodb" and not self.database:
            raise InvalidInputError("Indique el nombre de la base de datos de MongoDB.")
        if self.schema and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", self.schema):
            raise InvalidInputError("El esquema solo puede tener letras, números y guion bajo.")
        if self.port is not None and not 0 < self.port < 65536:
            raise InvalidInputError("El puerto debe estar entre 1 y 65535.")

    def sqlalchemy_url(self, with_schema: bool = False) -> URL:
        meta = ENGINES[self.engine]
        if self.is_sqlite:
            return URL.create("sqlite", database=str(Path(self.path).resolve()))
        query = dict(self.options)
        database: str | None = self.database or None
        if self.engine == "mssql":
            query.setdefault("driver", default_odbc_driver())
        elif self.engine == "mysql":
            query.setdefault("charset", "utf8mb4")
        elif self.engine == "oracle":
            query["service_name"] = self.database
            database = None
        if with_schema and self.schema:
            query["sigma_schema"] = self.schema  # carried to kernels; the engine itself never sees it
        return URL.create(
            meta["drivername"],
            username=self.username or None,
            password=self.password or None,
            host=self.host or None,
            port=self.port,
            database=database,
            query=query,
        )

    def to_url(self, hide_password: bool = False) -> str:
        """The canonical URL (schema included): what the kernels and ``SIGMA_DATABASE_URL`` carry."""
        return self.sqlalchemy_url(with_schema=True).render_as_string(hide_password=hide_password)

    def describe(self) -> dict[str, Any]:
        """Everything the administration page may show: never the password."""
        meta = ENGINES[self.engine]
        return {
            "engine": self.engine,
            "label": meta["label"],
            "path": self.path if self.is_sqlite else None,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "schema": self.schema,
            "username": self.username,
            "has_password": bool(self.password),
            "options": dict(self.options),
            "driver_available": driver_available(self.engine),
        }

    def to_dict(self) -> dict[str, Any]:
        if self.is_sqlite:
            return {"engine": "sqlite", "path": self.path}
        return {
            "engine": self.engine,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "schema": self.schema,
            "username": self.username,
            "password": self.password,
            "options": dict(self.options),
        }

    def with_password_from(self, previous: DatabaseSettings | None) -> DatabaseSettings:
        """A form that leaves the password blank keeps the stored one (the page never receives it).

        Only when the target is unchanged: otherwise a new host could be handed the old server's password.
        """
        if (
            self.password
            or previous is None
            or (previous.engine, previous.host, previous.port, previous.username)
            != (self.engine, self.host, self.port, self.username)
            or not self.username
        ):
            return self
        return DatabaseSettings(**{**vars(self), "password": previous.password})

    @classmethod
    def from_request(
        cls, data: Mapping[str, Any], sqlite_path: Path | str, previous: DatabaseSettings | None = None
    ) -> DatabaseSettings:
        """Settings typed into the setup wizard / administration page: strictly checked, never trusted.

        SQLite always means the server's own file (the page cannot point Sigma Algebra at an arbitrary path).
        """
        engine = engine_of(str(data.get("engine") or ""))
        if engine == "sqlite":
            return cls.sqlite(sqlite_path)
        text = {key: str(data.get(key) or "").strip() for key in ("host", "database", "schema", "username")}
        password = str(data.get("password") or "")
        if not _HOST.match(text["host"]):
            raise InvalidInputError("El servidor (host) tiene caracteres no válidos o está vacío.")
        if not _NAME.match(text["database"]):
            raise InvalidInputError("El nombre de la base de datos tiene caracteres no válidos o está vacío.")
        if text["username"] and not _FREE.match(text["username"]):
            raise InvalidInputError("El usuario tiene caracteres no válidos.")
        if not _FREE.match(password):
            raise InvalidInputError("La contraseña tiene caracteres no válidos.")
        raw_port = data.get("port")
        try:
            port = int(raw_port) if raw_port not in (None, "") else None
        except (TypeError, ValueError):
            raise InvalidInputError("El puerto debe ser un número.") from None
        options = _clean_options(engine, data.get("options") or {})
        settings = cls(
            engine=engine,
            host=text["host"],
            port=port,
            database=text["database"],
            username=text["username"],
            password=password,
            options=options,
            schema=text["schema"],
        ).with_password_from(previous)
        settings.validate()
        return settings

    def same_target(self, other: DatabaseSettings) -> bool:
        return self.to_url() == other.to_url()


def engines_for_ui() -> list[dict[str, Any]]:
    return [
        {
            "engine": key,
            "label": meta["label"],
            "default_port": meta["default_port"],
            "driver_available": driver_available(key),
            "install_hint": meta["pip"],
            "relational": meta["relational"],
        }
        for key, meta in ENGINES.items()
    ]


# ------------------------------------------------------------------ persistence of the choice
def settings_path(instance_dir: Path) -> Path:
    return Path(instance_dir) / SETTINGS_FILE


def load_settings(
    instance_dir: Path, default_sqlite_path: Path, environ: Mapping[str, str] | None = None
) -> DatabaseSettings:
    environ = os.environ if environ is None else environ
    url = (environ.get("SIGMA_DATABASE_URL") or "").strip()
    if url:
        return DatabaseSettings.from_url(url)
    path = settings_path(instance_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DatabaseSettings.sqlite(default_sqlite_path)
    if data.get("engine") == "sqlite" and not data.get("path"):
        data["path"] = str(default_sqlite_path)
    return DatabaseSettings.from_dict(data)


def save_settings(instance_dir: Path, settings: DatabaseSettings) -> None:
    path = settings_path(instance_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    with contextlib.suppress(OSError):
        temporary.chmod(0o600)  # best effort: it holds the database password
    os.replace(temporary, path)


def settings_source(instance_dir: Path, environ: Mapping[str, str] | None = None) -> str:
    """Where the active choice comes from: ``env``, ``file`` or ``default`` (what the page may change)."""
    environ = os.environ if environ is None else environ
    if (environ.get("SIGMA_DATABASE_URL") or "").strip():
        return "env"
    return "file" if settings_path(instance_dir).is_file() else "default"
