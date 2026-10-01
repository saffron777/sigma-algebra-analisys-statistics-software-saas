"""Maps a ``source_type`` to its importer."""

from __future__ import annotations

from sigma.application.ports import DatasetImporter
from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.importers.file_importers import CsvImporter, ExcelImporter, TextImporter
from sigma.infrastructure.importers.json_api_importer import JsonApiImporter
from sigma.infrastructure.importers.sqlserver_importer import SqlServerImporter


class DefaultImporterRegistry:
    def __init__(self, importers: list[DatasetImporter] | None = None) -> None:
        items = importers or [CsvImporter(), ExcelImporter(), TextImporter(), SqlServerImporter(), JsonApiImporter()]
        self._importers = {importer.source_type: importer for importer in items}

    def get(self, source_type: str) -> DatasetImporter:
        try:
            return self._importers[source_type]
        except KeyError:
            raise InvalidInputError(f"Fuente de datos no soportada: «{source_type}».") from None

    def source_types(self) -> list[str]:
        return list(self._importers)
