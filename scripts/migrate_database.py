"""Copies everything Sigma stores from one database to another (ids included); the source is only read.

    python scripts/migrate_database.py --to "postgresql+psycopg://user:pass@host/sigma"
    python scripts/migrate_database.py --from "sqlite:///instance/sigma.db" --to "mysql+pymysql://u:p@host/sigma" --use

The destination must be empty (Sigma creates its tables there). Without ``--from`` the database currently configured
for Sigma is the source. With ``--use`` the destination is also saved as the database Sigma starts with
(``instance/database.json``); stop Sigma before migrating so nobody writes while the data is copied.

URLs: sqlite:///path.db · mssql+pyodbc://user:pass@host/db?driver=ODBC+Driver+18+for+SQL+Server ·
mysql+pymysql://user:pass@host/db · postgresql+psycopg://user:pass@host/db ·
oracle+oracledb://user:pass@host:1521/?service_name=SERVICE · mongodb://host:27017/db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.application.use_cases.migration import MigrateData  # noqa: E402
from sigma.config import Config  # noqa: E402
from sigma.domain.errors import SigmaError  # noqa: E402
from sigma.infrastructure.persistence.backend import create_backend  # noqa: E402
from sigma.infrastructure.persistence.database_settings import DatabaseSettings, save_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", help="URL de la base de origen (por defecto, la configurada)")
    parser.add_argument("--to", dest="target", required=True, help="URL de la base de destino (debe estar vacía)")
    parser.add_argument("--notebooks", type=Path, help="carpeta de notebooks (por defecto, la configurada)")
    parser.add_argument("--use", action="store_true", help="dejar el destino como base de datos de Sigma al terminar")
    args = parser.parse_args()

    config = Config.from_env()
    notebooks = args.notebooks or config.notebooks_dir
    source_settings = DatabaseSettings.from_url(args.source) if args.source else config.database
    target_settings = DatabaseSettings.from_url(args.target)
    if source_settings.same_target(target_settings):
        print("El origen y el destino son la misma base de datos.", file=sys.stderr)
        return 2

    source = target = None
    try:
        print(f"Origen:  {source_settings.describe()['label']}  ({_where(source_settings)})")
        print(f"Destino: {target_settings.describe()['label']}  ({_where(target_settings)})")
        source = create_backend(source_settings)
        target = create_backend(target_settings)
        report = MigrateData().execute(source.transfer(notebooks), target.transfer(notebooks), progress=_print_progress)
    except SigmaError as error:
        print(f"\nError: {error.message}", file=sys.stderr)
        return 1
    finally:
        for backend in (source, target):
            if backend is not None:
                backend.close()
    print("\nCopiado:")
    for key, count in report.copied.items():
        print(f"  {key:<18} {count}")
    for key, count in report.skipped.items():
        print(f"  (omitidos en {key}: {count})")
    if args.use:
        save_settings(config.instance_dir, target_settings)
        print(f"\nSigma usará ahora el destino (guardado en {config.instance_dir / 'database.json'}).")
    else:
        print("\nPara usar el destino: SIGMA_DATABASE_URL=<url> o repita con --use.")
    return 0


def _where(settings: DatabaseSettings) -> str:
    return settings.path if settings.is_sqlite else f"{settings.host}:{settings.port or ''}/{settings.database}"


def _print_progress(key: str, label: str, done: int, total: int) -> None:
    if done == total:
        print(f"  {label}: {done}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
