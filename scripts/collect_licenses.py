"""Collect and check the third-party licenses shipped with Sigma.

    python scripts/collect_licenses.py collect   # copy texts and write LICENSES/INVENTORY.md
    python scripts/collect_licenses.py check     # fail if a dependency has an unknown license

The allowlist is deliberately conservative: everything must be compatible with
AGPL-3.0-or-later. This is a technical check, not legal advice.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LICENSES = ROOT / "LICENSES"
FRONTEND = {
    "alpinejs": ("LICENSES/frontend/alpinejs.txt", "MIT"),
    "dompurify": ("node_modules/dompurify/LICENSE", "Apache-2.0"),
    "jspreadsheet-ce": ("node_modules/jspreadsheet-ce/LICENSE", "MIT"),
    "jsuites": ("node_modules/jsuites/LICENSE", "MIT"),
    "katex": ("node_modules/katex/LICENSE", "MIT"),
    "marked": ("node_modules/marked/LICENSE.md", "MIT"),
    "plotly.js-dist-min": ("node_modules/plotly.js-dist-min/LICENSE", "MIT"),
    "tailwindcss": ("node_modules/tailwindcss/LICENSE", "MIT"),
}
# Packages whose installed metadata does not carry a machine-readable license expression.
KNOWN_LICENSES = {
    "jspreadsheet-ce": "MIT",
    "jsuites": "MIT",
    "semopy": "MIT",
    "arch": "NCSA",
    "linearmodels": "NCSA",
}
ALLOWED = (
    "0BSD",
    "AGPL-3.0",
    "Apache",
    "BSD",
    "CC0-1.0",
    "CC-BY-4.0",
    "HPND",
    "ISC",
    "LGPL-3.0",
    "MIT",
    "MPL-2.0",
    "NCSA",
    "PSF",
    "PYTHON SOFTWARE FOUNDATION",
    "Python-2.0",
    "UPL-1.0",
    "ZPL",
    "Zlib",
)
DENIED = ("GPL-2", "SSPL", "BUSL", "ELASTIC", "COMMONS CLAUSE", "NONCOMMERCIAL", "PROPRIETARY")


def _requirement_names() -> list[str]:
    names: list[str] = []
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def _license_of(name: str) -> str:
    if name in KNOWN_LICENSES:
        return KNOWN_LICENSES[name]
    try:
        meta = distribution(name).metadata
    except PackageNotFoundError:
        return "MISSING"
    classifiers = [item for item in (meta.get_all("Classifier") or []) if item.startswith("License ::")]
    parts = [meta.get("License-Expression") or "", *classifiers, meta.get("License") or ""]
    return " | ".join(" ".join(str(part).split()) for part in parts if part)[:800]


def _allowed(expression: str) -> bool:
    if not expression or expression in {"UNKNOWN", "MISSING"}:
        return False
    upper = expression.upper()
    if any(denied in upper for denied in DENIED):
        return False
    if "PUBLIC DOMAIN" in upper or "DUAL LICENSE" in upper:
        return True
    return any(allowed.upper() in upper for allowed in ALLOWED)


def _short_license(name: str) -> str:
    """A short, table-friendly license name for the inventory (the full text is in LICENSES/)."""
    if name in KNOWN_LICENSES:
        return KNOWN_LICENSES[name]
    try:
        meta = distribution(name).metadata
    except PackageNotFoundError:
        return "MISSING"
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return expression
    classifiers = [item for item in (meta.get_all("Classifier") or []) if item.startswith("License ::")]
    if classifiers:
        return classifiers[0].rsplit("::", 1)[-1].strip()
    text = " ".join((meta.get("License") or "").split())
    return text[:60] or "UNKNOWN"


def _package_version(package: str) -> str:
    path = ROOT / "node_modules" / package / "package.json"
    if not path.is_file():
        return "no instalado"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("version") or "?")
    except json.JSONDecodeError:
        return "?"


def _copy(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    if source.resolve() == target.resolve():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def _copy_python_licenses(names: list[str]) -> None:
    """Copy the license files of installed Python distributions into LICENSES/python/."""
    for name in names:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            continue
        license_files = [
            item
            for item in (dist.files or [])
            if item.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")) and dist.locate_file(item).is_file()
        ]
        target = LICENSES / "python" / f"{name}-{dist.version}"
        for index, item in enumerate(license_files, start=1):
            source = dist.locate_file(item)
            suffix = source.suffix or ".txt"
            _copy(Path(str(source)), target / f"{index:02d}-{source.stem[:40]}{suffix}")


def collect() -> int:
    failures: list[str] = []
    for package, (license_file, _license) in FRONTEND.items():
        source = ROOT / license_file
        if package == "alpinejs" and not source.is_file():
            # Alpine's npm package ships only package.json; the text is committed in LICENSES/frontend.
            if not (ROOT / "LICENSES/frontend/alpinejs.txt").is_file():
                failures.append("alpinejs: falta LICENSES/frontend/alpinejs.txt")
            continue
        if not _copy(source, ROOT / f"LICENSES/frontend/{package}.txt"):
            failures.append(f"{package}: no se encontró {license_file}")
    if not (ROOT / "LICENSES/fonts/DejaVu.txt").is_file():
        failures.append("falta LICENSES/fonts/DejaVu.txt")

    names = _requirement_names()
    _copy_python_licenses(names)
    rows: list[tuple[str, str, str, str]] = []
    for name in names:
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = "no instalado"
        rows.append((name, installed, _short_license(name), "Python (instalado por pip)"))
    for package, (_file, license_name) in FRONTEND.items():
        rows.append((package, _package_version(package), license_name, "Frontend (incluido en la app)"))

    inventory = ["# Inventario de licencias", "", "Generado por `scripts/collect_licenses.py`.", ""]
    inventory.append("| Componente | Versión | Licencia | Distribución |")
    inventory.append("|---|---:|---|---|")
    inventory.extend(
        f"| {str(name).replace('|', chr(92) + '|')} | {installed} | "
        f"{str(license_name).replace('|', chr(92) + '|')} | {where} |"
        for name, installed, license_name, where in rows
    )
    (LICENSES / "INVENTORY.md").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    print(f"Escrito LICENSES/INVENTORY.md ({len(rows)} componentes).")
    for item in failures:
        print(f"AVISO: {item}")
    return 1 if failures else 0


def check() -> int:
    problems: list[str] = []
    for required in ("LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.md"):
        if not (ROOT / required).is_file():
            problems.append(f"falta {required}")
    for name in _requirement_names():
        expression = _license_of(name)
        if not _allowed(expression):
            problems.append(f"{name}: licencia no reconocida o no permitida: {expression!r}")
    for package, (_license_file, license_name) in FRONTEND.items():
        if not _allowed(license_name):
            problems.append(f"{package}: licencia no permitida: {license_name!r}")
        if not (ROOT / f"LICENSES/frontend/{package}.txt").is_file():
            problems.append(f"{package}: falta LICENSES/frontend/{package}.txt")
    if problems:
        print("Revisión de licencias FALLIDA:")
        for problem in problems:
            print(f" - {problem}")
        return 1
    print("Revisión de licencias OK (AGPL-3.0-or-later compatible).")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"collect", "check"}:
        print(__doc__)
        return 2
    return collect() if argv[1] == "collect" else check()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
