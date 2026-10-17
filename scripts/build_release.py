"""Build the Sigma release artifacts.

    npm ci && npm run build
    python scripts/collect_licenses.py collect
    python scripts/build_release.py

Produces in ``dist/``:

- ``sigma-<version>-source.tar.gz``: corresponding source (no generated assets).
- ``sigma-<version>.tar.gz`` and ``.zip``: ready to run (source + built browser assets).
- ``SHA256SUMS``.

The script refuses to package data that must never be published (``.env``, ``instance/``,
``.git``, ``.gitconfig.backup``, databases, etc.).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
STAGING = DIST / ".staging"
EXCLUDED_DIRS = {
    ".git",
    ".github",  # included separately in the tree copy? no: releases do not need workflow files
    ".venv",
    ".uv-cache",
    "node_modules",
    "instance",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".pytest-run",
    "pytest_pre",
    "pytest_tmp",
    ".pytest-tmp",
    ".pytest-tmp2",
    ".acl-recovery",
    ".chrome-check-profile",
    ".ai-tmp",
}
EXCLUDED_FILES = {".env", ".env.local", ".gitconfig.backup", ".DS_Store", ".tailwind-check.css"}
EXCLUDED_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".log", ".pyc")
FORBIDDEN = ("/instance/", "/.git/", "/.venv/", "/node_modules/", ".db", ".sqlite")


def version() -> str:
    text = (ROOT / "sigma/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("No se encontró __version__ en sigma/__init__.py")
    return match.group(1)


def _ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        relative = path.as_posix()
        if (
            name in EXCLUDED_DIRS
            or name in EXCLUDED_FILES
            or name.startswith((".gitconfig", ".uv-cache", ".acl-", ".pytest-tmp", ".chrome-"))
            or relative.endswith("sigma/presentation/web/static/vendor")
            or relative.endswith("sigma/presentation/web/static/css/app.css")
            or (path.is_file() and name.endswith(EXCLUDED_SUFFIXES))
        ):
            ignored.add(name)
    return ignored


def _forbidden(name: str) -> bool:
    normalized = f"/{name.replace(chr(92), '/').strip('/')}"
    if normalized.endswith("/.env.example"):
        return False  # the example file documents the configuration and contains no secret
    if normalized.endswith("/.env") or "/.env." in normalized or "/.gitconfig" in normalized:
        return True
    return any(marker in normalized for marker in FORBIDDEN)


def _copy_tree(target: Path) -> None:
    shutil.copytree(ROOT, target, ignore=_ignore, symlinks=False, dirs_exist_ok=True)
    # generated Python license texts and inventory belong in the release
    for extra in (ROOT / "LICENSES/python",):
        if extra.is_dir():
            shutil.copytree(extra, target / extra.relative_to(ROOT), dirs_exist_ok=True)


def _write_release_note(target: Path) -> None:
    note = f"""Sigma {version()}

Copyright (C) 2026 Frank Miranda
Licencia: AGPL-3.0-or-later. Texto completo en LICENSE; avisos en THIRD-PARTY-NOTICES.md.

Código fuente: https://github.com/saffron777/Sigma
Instalación: consulte README.md (Docker, Windows o Linux/macOS).
Este paquete se distribuye SIN GARANTÍA, según los términos de la AGPL.
"""
    (target / "RELEASE.txt").write_text(note, encoding="utf-8")


def _tarball(source: Path, destination: Path, arcname: str) -> None:
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"{arcname}/{path.relative_to(source).as_posix()}")


def _zip(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source).as_posix())


def _checksums(paths: list[Path]) -> None:
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    (DIST / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    current = version()
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.MULTILINE)
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    for name, found in (("pyproject.toml", pyproject.group(1) if pyproject else None), ("package.json", package.get("version"))):
        if found != current:
            print(f"Versión incoherente: sigma={current}, {name}={found}")
            return 1
    for asset in (
        ROOT / "sigma/presentation/web/static/vendor/plotly.min.js",
        ROOT / "sigma/presentation/web/static/css/app.css",
    ):
        if not asset.is_file():
            print(f"Falta {asset.relative_to(ROOT)}; ejecute `npm ci && npm run build` antes de empaquetar.")
            return 1
    for required in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md", "TRADEMARKS.md", "requirements.lock"):
        if not (ROOT / required).is_file():
            print(f"Falta {required}.")
            return 1

    from collect_licenses import collect

    if collect() != 0:
        return 1

    if DIST.exists():
        shutil.rmtree(DIST)
    STAGING.mkdir(parents=True)
    source = STAGING / f"sigma-{current}"
    full = STAGING / f"sigma-{current}-full"
    _copy_tree(source)
    _copy_tree(full)
    for generated in (
        "sigma/presentation/web/static/vendor",
        "sigma/presentation/web/static/css/app.css",
    ):
        origin = ROOT / generated
        destination = full / generated
        if origin.is_dir():
            shutil.copytree(origin, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
    _write_release_note(source)
    _write_release_note(full)

    for tree in (source, full):
        for path in tree.rglob("*"):
            if path.is_file() and _forbidden(path.relative_to(tree).as_posix()):
                print(f"Contenido prohibido en el artefacto: {path.relative_to(tree)}")
                return 1

    source_tar = DIST / f"sigma-{current}-source.tar.gz"
    full_tar = DIST / f"sigma-{current}.tar.gz"
    full_zip = DIST / f"sigma-{current}.zip"
    _tarball(source, source_tar, f"sigma-{current}")
    _tarball(full, full_tar, f"sigma-{current}")
    _zip(full, full_zip)
    _checksums([source_tar, full_tar, full_zip])
    shutil.rmtree(STAGING)
    print(f"Artefactos generados en {DIST}:")
    for path in (source_tar, full_tar, full_zip, DIST / "SHA256SUMS"):
        print(f" - {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
