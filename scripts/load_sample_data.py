"""Loads the CSV files of ``sample_data/`` into Sigma's SQLite database (skips the ones already loaded).

python scripts/make_sample_data.py   # (re)generate the CSV files
python scripts/load_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sigma.application.ports import ImportRequest  # noqa: E402
from sigma.bootstrap import build_container  # noqa: E402
from sigma.config import Config  # noqa: E402
from sigma.domain.errors import InvalidInputError  # noqa: E402


def main() -> None:
    container = build_container(Config.from_env())
    existing = {d.name.lower() for d in container.list_datasets.execute()}
    for path in sorted((ROOT / "sample_data").glob("*.csv")):
        if path.stem.lower() in existing:
            print(f"· {path.stem}: ya existe, se omite")
            continue
        preview = container.preview_import.execute(ImportRequest("csv", {}, path.name, path.read_bytes()))
        try:
            info = container.commit_import.execute(preview.token, path.stem)
        except InvalidInputError as error:
            print(f"· {path.stem}: {error.message}")
            continue
        print(f"✓ {info.name}: {info.n_rows} filas x {info.n_cols} columnas")


if __name__ == "__main__":
    main()
