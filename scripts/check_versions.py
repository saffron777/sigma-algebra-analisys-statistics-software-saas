"""Check that the version is the same in sigma/__init__.py, pyproject.toml and package.json.

    python scripts/check_versions.py
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma import __version__  # noqa: E402  (the project root must be importable when run as a script)


def versions() -> dict[str, str]:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    return {
        "sigma/__init__.py": __version__,
        "pyproject.toml": str(pyproject["project"]["version"]),
        "package.json": str(package["version"]),
    }


def main() -> int:
    found = versions()
    labels = ", ".join(f"{name}={value}" for name, value in found.items())
    if len(set(found.values())) != 1:
        print(f"Las versiones no coinciden: {labels}")
        return 1
    print(f"Versión coherente: {next(iter(found.values()))} ({labels})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
