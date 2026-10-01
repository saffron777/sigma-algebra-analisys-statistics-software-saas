"""Loads the CSV files of ``sample_data/`` into one user's datasets (skips the ones already loaded).

    python scripts/make_sample_data.py              # (re)generate the CSV files
    python scripts/load_sample_data.py --user ana   # defaults to the first administrator

The same thing is available in the web UI: Datos → «Cargar datos de ejemplo».
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.bootstrap import build_container  # noqa: E402
from sigma.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", help="usuario que recibe los datos (por defecto, el primer administrador)")
    args = parser.parse_args()

    container = build_container(Config.from_env())
    users = container.list_users.execute()
    if not users:
        print(
            "Todavía no hay cuentas: abra Sigma en el navegador y cree el administrador (o use scripts/create_user.py)."
        )
        return 1
    if args.user:
        target = next((u for u in users if u.username.lower() == args.user.lower()), None)
        if target is None:
            print(f"No existe el usuario «{args.user}». Usuarios: {', '.join(u.username for u in users)}")
            return 1
    else:
        target = next((u for u in users if u.is_admin), users[0])

    outcome = container.for_user(target).load_sample_data.execute()
    for name in outcome["created"]:
        print(f"✓ {name}")
    for name in outcome["skipped"]:
        print(f"· {name}: ya existe, se omite")
    print(f"Datos de ejemplo cargados para «{target.username}».")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
