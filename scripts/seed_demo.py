"""Seeds the public demo instance: a regular demo account plus the bundled sample datasets.

    SIGMA_DEMO_MODE=1 python scripts/seed_demo.py

The demo account has no console and no AI access; visitors enter through ``/demo`` and cannot use
imports, administration or destructive endpoints (see ``sigma/presentation/web/auth.py``). The
demo volume is disposable: ``demo/reset.sh`` restores it from a snapshot.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.bootstrap import build_container  # noqa: E402
from sigma.config import Config  # noqa: E402
from sigma.domain.errors import SigmaError  # noqa: E402


def main() -> int:
    config = Config.from_env()
    container = build_container(config)
    try:
        users = container.list_users.execute()
        if not users:
            # The demo route logs in the regular demo account; the administrator only exists to
            # satisfy setup and is never exposed.
            container.create_first_admin.execute(
                "demo-admin", "Administrador de la demo", secrets.token_urlsafe(32)
            )
            users = container.list_users.execute()
        demo = next((u for u in users if u.username.lower() == config.demo_user.lower()), None)
        if demo is None:
            demo = container.create_user.execute(
                config.demo_user,
                "Demo",
                secrets.token_urlsafe(32),
                "user",
                False,
                False,
            )
        outcome = container.for_user(demo).load_sample_data.execute()
        print(f"Usuario demo: {demo.username} (sin consola ni IA)")
        print(f"Datasets nuevos: {len(outcome['created'])}; existentes: {len(outcome['skipped'])}")
    except SigmaError as error:
        print(f"Error: {error.message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
