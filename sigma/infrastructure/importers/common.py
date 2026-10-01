"""Helpers shared by the file-based importers."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

import pandas as pd

from sigma.application.ports import ImportRequest
from sigma.domain.errors import ImportFailedError, InvalidInputError

_DELIMITER_NAMES = {"comma": ",", "semicolon": ";", "tab": "\t", "pipe": "|", "space": " "}
_ENCODINGS_TO_TRY = ("utf-8-sig", "cp1252", "latin-1")


def require_file(request: ImportRequest) -> bytes:
    if not request.file_content:
        raise InvalidInputError("Seleccione un archivo para importar.")
    return request.file_content


def decode_bytes(content: bytes, encoding: str | None) -> tuple[str, str]:
    """Decode text, auto-detecting between UTF-8 (with/without BOM) and Windows-1252/Latin-1."""
    if encoding and encoding != "auto":
        try:
            return content.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError) as error:
            raise ImportFailedError(f"No se pudo leer el archivo con la codificación «{encoding}».") from error
    for candidate in _ENCODINGS_TO_TRY:
        try:
            return content.decode(candidate), candidate
        except UnicodeDecodeError:
            continue
    raise ImportFailedError(
        "No se pudo determinar la codificación del archivo."
    )  # unreachable: latin-1 decodes anything


def as_int(value: Any, label: str, default: int, minimum: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise InvalidInputError(f"«{label}» debe ser un número entero.") from None
    if number < minimum:
        raise InvalidInputError(f"«{label}» debe ser ≥ {minimum}.")
    return number


def as_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "si", "sí", "on"}


def resolve_delimiter(option: str | None, sample: str, candidates: str = ",;\t|") -> str:
    """Return the delimiter to use: explicit option, or sniffed from a text sample."""
    if option and option != "auto":
        return _DELIMITER_NAMES.get(option, option.replace("\\t", "\t"))
    try:
        return csv.Sniffer().sniff(sample[:65536], delimiters=candidates).delimiter
    except csv.Error:
        return ","


def resolve_decimal(option: str | None, delimiter: str, sample: str) -> str:
    if option and option != "auto":
        return option
    if delimiter != "," and re.search(r"\d,\d", sample) and not re.search(r"\d\.\d", sample):
        return ","
    return "."


def read_delimited(
    content: bytes, options: dict[str, Any], candidates: str = ",;\t|"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    text, encoding = decode_bytes(content, options.get("encoding"))
    if not text.strip():
        raise ImportFailedError("El archivo está vacío.")
    sample = text[:65536]
    raw_delimiter = options.get("delimiter")
    whitespace = raw_delimiter == "whitespace"
    delimiter = r"\s+" if whitespace else resolve_delimiter(raw_delimiter, sample, candidates)
    decimal = resolve_decimal(options.get("decimal"), delimiter, sample)
    has_header = as_bool(options.get("header"), True)
    skip = as_int(options.get("skip_rows"), "Filas a omitir", 0)
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            decimal=decimal,
            header=0 if has_header else None,
            skiprows=skip,
            engine="python" if len(delimiter) > 1 else "c",
            skipinitialspace=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as error:
        raise ImportFailedError(f"No se pudo interpretar el archivo: {error}") from error
    if not has_header:
        frame.columns = [f"columna_{i + 1}" for i in range(frame.shape[1])]
    shown = "whitespace" if whitespace else {v: k for k, v in _DELIMITER_NAMES.items()}.get(delimiter, delimiter)
    return frame, {"delimiter": shown, "encoding": encoding, "decimal": decimal, "header": has_header}
