"""Importers for CSV, plain-text and Excel files."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError
from sigma.infrastructure.importers.common import as_int, decode_bytes, read_delimited, require_file


class CsvImporter:
    source_type = "csv"

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        return read_delimited(require_file(request), request.options)


class TextImporter:
    """Delimited text (space/tab/other) or fixed-width columns."""

    source_type = "text"

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        content = require_file(request)
        if request.options.get("layout") == "fixed":
            return self._read_fixed(content, request.options)
        return read_delimited(content, request.options, candidates=",;\t| ")

    @staticmethod
    def _read_fixed(content: bytes, options: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        text, encoding = decode_bytes(content, options.get("encoding"))
        raw_widths = str(options.get("widths") or "").replace(";", ",").strip()
        try:
            widths = [int(w) for w in raw_widths.split(",") if w.strip()] if raw_widths else None
            frame = pd.read_fwf(
                io.StringIO(text),
                widths=widths,
                header=0 if options.get("header", True) else None,
                skiprows=as_int(options.get("skip_rows"), "Filas a omitir", 0),
            )
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
            raise ImportFailedError(f"No se pudo interpretar el texto de ancho fijo: {error}") from error
        return frame, {"layout": "fixed", "encoding": encoding, "widths": widths or "auto"}


class ExcelImporter:
    source_type = "excel"

    def inspect(self, request: ImportRequest) -> dict[str, Any]:
        workbook = self._open(require_file(request))
        return {"sheets": [str(name) for name in workbook.sheet_names]}

    def read(self, request: ImportRequest) -> tuple[pd.DataFrame, dict[str, Any]]:
        workbook = self._open(require_file(request))
        sheets = [str(name) for name in workbook.sheet_names]
        sheet = self._resolve_sheet(request.options.get("sheet"), sheets)
        header_row = as_int(request.options.get("header_row"), "Fila de encabezado", 1)
        try:
            frame = workbook.parse(sheet_name=sheet, header=header_row - 1 if header_row else None)
        except (ValueError, KeyError) as error:
            raise ImportFailedError(f"No se pudo leer la hoja «{sheet}»: {error}") from error
        if not header_row:
            frame.columns = [f"columna_{i + 1}" for i in range(frame.shape[1])]
        return frame, {"sheet": sheet, "sheets": sheets, "header_row": header_row}

    @staticmethod
    def _open(content: bytes) -> pd.ExcelFile:
        try:
            return pd.ExcelFile(io.BytesIO(content))
        except Exception as error:  # noqa: BLE001 - xlrd/openpyxl raise many unrelated exception types
            raise ImportFailedError(
                "No se pudo abrir el libro de Excel (¿es un .xlsx/.xls válido y sin contraseña?)."
            ) from error

    @staticmethod
    def _resolve_sheet(option: Any, sheets: list[str]) -> str:
        if option in (None, ""):
            return sheets[0]
        text = str(option)
        if text in sheets:
            return text
        if text.isdigit() and 1 <= int(text) <= len(sheets):
            return sheets[int(text) - 1]
        raise InvalidInputError(f"La hoja «{text}» no existe. Hojas disponibles: {', '.join(sheets)}.")
