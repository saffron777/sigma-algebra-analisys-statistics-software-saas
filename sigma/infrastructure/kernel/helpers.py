"""The ``sigma`` helper object available inside console kernels (not used by the web app itself).

sigma.datasets()                    # table of the datasets stored in Sigma
df = sigma.load("ventas")           # a dataset as a pandas DataFrame
sigma.save(df, "ventas_limpias")    # store a DataFrame as a new Sigma dataset
"""

from __future__ import annotations

import os

import pandas as pd

from sigma.domain.datasets import normalize_columns
from sigma.domain.errors import DatasetNotFoundError, InvalidInputError
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository


class SigmaHelper:
    def __init__(self) -> None:
        self._repository: SqliteDatasetRepository | None = None

    @property
    def _repo(self) -> SqliteDatasetRepository:
        if self._repository is None:
            path, user = os.environ.get("SIGMA_DB_PATH"), os.environ.get("SIGMA_USER_ID")
            if not path or not user:
                raise RuntimeError("Este kernel no está conectado a una cuenta de Sigma.")
            self._repository = SqliteDatasetRepository(Database(path), int(user))
        return self._repository

    def datasets(self) -> pd.DataFrame:
        """Datasets stored in Sigma (name, rows, columns, source, creation date)."""
        return pd.DataFrame(
            [
                {
                    "nombre": d.name,
                    "filas": d.n_rows,
                    "columnas": d.n_cols,
                    "origen": d.source_type,
                    "creado": d.created_at,
                }
                for d in self._repo.list()
            ],
            columns=["nombre", "filas", "columnas", "origen", "creado"],
        )

    def load(self, name: str) -> pd.DataFrame:
        """Return the dataset called ``name`` as a DataFrame."""
        try:
            return self._repo.get_by_name(name).data
        except DatasetNotFoundError as error:
            available = ", ".join(d.name for d in self._repo.list()) or "(ninguno)"
            raise KeyError(f"{error.message} Datasets disponibles: {available}") from None

    def save(self, frame: pd.DataFrame, name: str, replace: bool = False) -> str:
        """Store ``frame`` as dataset ``name``; ``replace=True`` overwrites an existing one."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("sigma.save() espera un DataFrame de pandas.")
        tidy = normalize_columns(frame)
        try:
            existing = self._repo.get_by_name(name)
        except DatasetNotFoundError:
            self._repo.add(name, tidy, "console", {"created_from": "console"})
            return f"Dataset «{name}» creado ({len(tidy)} filas × {tidy.shape[1]} columnas)."
        if not replace:
            raise InvalidInputError(
                f"Ya existe el dataset «{name}». Use sigma.save(df, '{name}', replace=True) para reemplazarlo."
            )
        self._repo.replace_data(existing.info.id, tidy)
        return f"Dataset «{name}» actualizado ({len(tidy)} filas × {tidy.shape[1]} columnas)."

    def __repr__(self) -> str:
        return "sigma — datasets(), load('nombre'), save(df, 'nombre', replace=False)"


sigma = SigmaHelper()
