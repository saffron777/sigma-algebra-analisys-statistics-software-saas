"""Image validation for uploads (logos) and the charts the browser sends for the reports."""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from sigma.domain.errors import InvalidInputError
from sigma.domain.report_templates import MAX_LOGO_BYTES

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_CHART_BYTES = 8 * 1024 * 1024
_MAX_PIXELS = 24_000_000  # a decompression-bomb guard, well above any real chart
_LOGO_BOX = (600, 240)

Image.MAX_IMAGE_PIXELS = _MAX_PIXELS


class PillowImageProcessor:
    def normalize_logo(self, data: bytes) -> bytes:
        """PNG or JPEG only (never SVG: it can carry scripts); re-encoded as a clean, downsized PNG."""
        if not data:
            raise InvalidInputError("El archivo del logotipo está vacío.")
        if len(data) > MAX_LOGO_BYTES:
            raise InvalidInputError("El logotipo supera los 2 MB.")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in ("PNG", "JPEG"):
                    raise InvalidInputError("El logotipo debe ser una imagen PNG o JPG.")
                image.load()
                image = ImageOps.exif_transpose(image)
                image.thumbnail(_LOGO_BOX, Image.LANCZOS)
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA" if "A" in image.getbands() or image.mode == "P" else "RGB")
                out = io.BytesIO()
                image.save(out, format="PNG", optimize=True)  # re-encoding drops metadata and any trailing payload
                return out.getvalue()
        except InvalidInputError:
            raise
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise InvalidInputError("El archivo no es una imagen PNG o JPG válida.") from None

    def validate_chart(self, data: bytes) -> bytes:
        """A chart image from the browser: must be a readable PNG within the size limit."""
        if not data.startswith(PNG_MAGIC):
            raise InvalidInputError("Una de las imágenes del informe no es un PNG válido.")
        if len(data) > MAX_CHART_BYTES:
            raise InvalidInputError("Una de las imágenes del informe supera los 8 MB.")
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise InvalidInputError("Una de las imágenes del informe está dañada.") from None
        return data
