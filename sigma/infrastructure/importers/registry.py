"""Maps a ``source_type`` to its importer."""

from __future__ import annotations

from sigma.application.ports import DatasetImporter
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.importers.external_sql_importer import ExternalSqlImporter
from sigma.infrastructure.importers.file_importers import CsvImporter, ExcelImporter, TextImporter
from sigma.infrastructure.importers.json_api_importer import JsonApiImporter
from sigma.infrastructure.importers.mongo_importer import MongoImporter
from sigma.infrastructure.importers.sqlserver_importer import SqlServerImporter
from sigma.infrastructure.persistence.database_settings import ENGINES, driver_available

# the databases the «Base de datos» tab can read from: source type -> engine key of ``ENGINES``
EXTERNAL_DATABASES = {
    "sqlserver": "mssql",
    "mysql": "mysql",
    "postgresql": "postgresql",
    "oracle": "oracle",
    "mongodb": "mongodb",
}


class DefaultImporterRegistry:
    def __init__(self, importers: list[DatasetImporter] | None = None, allowed_hosts: tuple[str, ...] = ()) -> None:
        items = importers or [
            CsvImporter(),
            ExcelImporter(),
            TextImporter(),
            SqlServerImporter(allowed_hosts=allowed_hosts),
            ExternalSqlImporter("mysql", allowed_hosts=allowed_hosts),
            ExternalSqlImporter("postgresql", allowed_hosts=allowed_hosts),
            ExternalSqlImporter("oracle", allowed_hosts=allowed_hosts),
            MongoImporter(allowed_hosts=allowed_hosts),
            JsonApiImporter(),
        ]
        self._importers = {importer.source_type: importer for importer in items}

    def get(self, source_type: str) -> DatasetImporter:
        try:
            return self._importers[source_type]
        except KeyError:
            raise InvalidInputError(f"Fuente de datos no soportada: «{source_type}».") from None

    def source_types(self) -> list[str]:
        return list(self._importers)

    def external_databases(self) -> list[dict[str, object]]:
        """The database engines a user can read from, and whether this server has the driver for each."""
        return [
            {
                "source_type": source_type,
                "label": ENGINES[engine]["label"],
                "default_port": ENGINES[engine]["default_port"],
                "driver_available": driver_available(engine),
                "install_hint": ENGINES[engine]["pip"],
            }
            for source_type, engine in EXTERNAL_DATABASES.items()
            if source_type in self._importers
        ]
