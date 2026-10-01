"""Creates an account (or resets a password) without using the browser.

    python scripts/create_user.py ana --admin                 # asks for the password
    python scripts/create_user.py luis --console              # regular user allowed to use the Python console
    python scripts/create_user.py ana --reset-password        # new password for an existing account

The very first account is always created as administrator.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.bootstrap import build_container  # noqa: E402
from sigma.config import Config  # noqa: E402
from sigma.domain.errors import SigmaError  # noqa: E402


def ask_password() -> str:
    while True:
        first = getpass.getpass("Contraseña: ")
        if first == getpass.getpass("Repita la contraseña: "):
            return first
        print("Las contraseñas no coinciden.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("username")
    parser.add_argument("--name", default="", help="nombre para mostrar")
    parser.add_argument("--admin", action="store_true", help="crear como administrador")
    parser.add_argument("--console", action="store_true", help="permitir la consola de Python")
    parser.add_argument("--ai", action="store_true", help="permitir la interpretación con IA")
    parser.add_argument(
        "--reset-password", action="store_true", help="restablecer la contraseña de una cuenta existente"
    )
    args = parser.parse_args()

    container = build_container(Config.from_env())
    try:
        if args.reset_password:
            target = next(
                (u for u in container.list_users.execute() if u.username.lower() == args.username.lower()), None
            )
            if target is None:
                print(f"No existe el usuario «{args.username}».")
                return 1
            container.reset_password.execute(target.id, ask_password())
            print(f"Contraseña de «{target.username}» restablecida (sus sesiones abiertas se cerraron).")
        elif container.get_setup_state.setup_required():
            user, adopted = container.create_first_admin.execute(args.username, args.name, ask_password())
            print(f"Administrador «{user.username}» creado. Datos existentes adoptados: {adopted}")
        else:
            role = "admin" if args.admin else "user"
            user = container.create_user.execute(args.username, args.name, ask_password(), role, args.console, args.ai)
            print(f"Usuario «{user.username}» creado ({role}).")
    except SigmaError as error:
        print(f"Error: {error.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
