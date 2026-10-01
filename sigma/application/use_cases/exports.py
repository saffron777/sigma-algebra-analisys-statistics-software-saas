"""Exporting reports: a result (or several) from the browser, a dataset sheet, or a notebook."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sigma.application.ports import (
    DatasetRepository,
    DocumentExporter,
    ImageProcessor,
    NotebookStore,
    TemplateRepository,
)
from sigma.domain.dataset_report import dataset_data_table, dataset_section
from sigma.domain.documents import (
    ORIENTATIONS,
    PAGE_SIZES,
    Branding,
    LayoutOptions,
    ReportDocument,
    ReportSection,
    clean_blocks,
    safe_filename,
)
from sigma.domain.errors import InvalidInputError
from sigma.domain.notebook_report import NotebookReportOptions, notebook_section

__all__ = ["NotebookReportOptions"]
from sigma.domain.report_templates import ReportTemplate

MAX_SECTIONS = 200
MAX_IMAGES_TOTAL_BYTES = 80 * 1024 * 1024


@dataclass
class ExportedFile:
    filename: str
    content_type: str
    data: bytes


@dataclass
class SectionInput:
    title: str
    subtitle: str | None = None
    meta: list[str] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    images: dict[int, bytes] = field(default_factory=dict)  # figure index -> PNG drawn by the browser


@dataclass
class ExportOptions:
    """What the export dialog lets the user decide; ``None`` keeps the template's value."""

    template_id: str | None = None
    title: str | None = None
    subtitle: str | None = None
    author: str = ""
    cover: bool | None = None
    toc: bool | None = None
    page_size: str | None = None
    orientation: str | None = None


class ExporterRegistry:
    def __init__(self, exporters: list[DocumentExporter]) -> None:
        self._by_format = {e.format: e for e in exporters}

    def get(self, fmt: str) -> DocumentExporter:
        try:
            return self._by_format[fmt]
        except KeyError:
            raise InvalidInputError(f"Formato de exportación no admitido: «{fmt}» (use pdf, docx o xlsx).") from None

    def formats(self) -> list[str]:
        return list(self._by_format)


class DocumentAssembler:
    """Applies a template and the dialog's choices to a list of sections, producing the document to render."""

    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def build(
        self,
        title: str,
        sections: list[ReportSection],
        options: ExportOptions,
        data_tables=None,
    ) -> ReportDocument:
        template = self._template(options.template_id)
        branding = Branding()
        layout = LayoutOptions()
        if template:
            logo = self._templates.get_logo(template.id) if template.has_logo else None
            branding = Branding(
                organization=template.organization,
                primary_color=template.primary_color,
                secondary_color=template.secondary_color,
                footer_text=template.footer_text,
                cover_text=template.cover_text,
                logo_png=logo,
            )
            layout = LayoutOptions(
                page_size=template.page_size,
                orientation=template.orientation,
                cover=template.show_cover,
                toc=template.show_toc,
            )
        if options.cover is not None:
            layout.cover = options.cover
        if options.toc is not None:
            layout.toc = options.toc
        if options.page_size is not None:
            if options.page_size not in PAGE_SIZES:
                raise InvalidInputError(f"Tamaño de página inválido: {', '.join(PAGE_SIZES)}.")
            layout.page_size = options.page_size
        if options.orientation is not None:
            if options.orientation not in ORIENTATIONS:
                raise InvalidInputError("La orientación debe ser «portrait» o «landscape».")
            layout.orientation = options.orientation
        return ReportDocument(
            title=(options.title or title).strip()[:200] or "Informe",
            subtitle=(options.subtitle or "").strip()[:300] or None,
            author=(options.author or "").strip()[:120],
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            sections=sections,
            branding=branding,
            layout=layout,
            data_tables=data_tables or [],
        )

    def _template(self, template_id: str | None) -> ReportTemplate | None:
        if template_id:
            return self._templates.get(template_id)  # 404 when it is not the user's
        return self._templates.get_default()


def _file(exporter: DocumentExporter, document: ReportDocument) -> ExportedFile:
    return ExportedFile(
        safe_filename(document.title, exporter.extension, document.created_at[:10]),
        exporter.content_type,
        exporter.export(document),
    )


class ExportReport:
    """The report the browser built (one result, or several for the combined report) plus its chart images."""

    def __init__(self, exporters: ExporterRegistry, templates: TemplateRepository, images: ImageProcessor) -> None:
        self._exporters = exporters
        self._assembler = DocumentAssembler(templates)
        self._images = images

    def execute(self, fmt: str, sections: list[SectionInput], options: ExportOptions) -> ExportedFile:
        exporter = self._exporters.get(fmt)
        if not sections:
            raise InvalidInputError("No hay nada que exportar.")
        if len(sections) > MAX_SECTIONS:
            raise InvalidInputError(f"Un informe admite como máximo {MAX_SECTIONS} secciones.")
        total = sum(len(png) for s in sections for png in s.images.values())
        if total > MAX_IMAGES_TOTAL_BYTES:
            raise InvalidInputError("Las imágenes del informe superan los 80 MB.")
        built = [self._section(s) for s in sections]
        title = options.title or (built[0].title if len(built) == 1 else "Informe de resultados")
        return _file(exporter, self._assembler.build(title, built, options))

    def _section(self, section: SectionInput) -> ReportSection:
        title = (section.title or "").strip()[:300]
        if not title:
            raise InvalidInputError("Cada sección necesita un título.")
        size = (max(section.images) + 1) if section.images else 0
        if size > 500:
            raise InvalidInputError("Demasiados gráficos en una sección.")
        images: list[bytes | None] = [None] * size
        for index, data in section.images.items():
            images[index] = self._images.validate_chart(data)
        return ReportSection(
            title=title,
            subtitle=(section.subtitle or "").strip()[:400] or None,
            meta=[str(m)[:300] for m in section.meta[:12]],
            blocks=clean_blocks(section.blocks),
            images=images,
        )


class ExportDatasetReport:
    """The dataset's «ficha» (summary, columns, statistics, first rows); Excel also carries every row."""

    def __init__(self, exporters: ExporterRegistry, templates: TemplateRepository, datasets: DatasetRepository) -> None:
        self._exporters = exporters
        self._assembler = DocumentAssembler(templates)
        self._datasets = datasets

    def execute(self, dataset_id: str, fmt: str, options: ExportOptions) -> ExportedFile:
        exporter = self._exporters.get(fmt)
        dataset = self._datasets.get(dataset_id)
        section = dataset_section(dataset.info, dataset.data)
        data_tables = [dataset_data_table(dataset.info, dataset.data)] if fmt == "xlsx" else []
        document = self._assembler.build(f"Dataset {dataset.info.name}", [section], options, data_tables)
        return _file(exporter, document)


class ExportNotebookReport:
    """A notebook as a report (text cells, code and outputs)."""

    def __init__(self, exporters: ExporterRegistry, templates: TemplateRepository, notebooks: NotebookStore) -> None:
        self._exporters = exporters
        self._assembler = DocumentAssembler(templates)
        self._notebooks = notebooks

    def execute(
        self, notebook_id: str, fmt: str, options: ExportOptions, content: NotebookReportOptions | None = None
    ) -> ExportedFile:
        exporter = self._exporters.get(fmt)
        notebook = self._notebooks.get(notebook_id)
        section = notebook_section(notebook, content or NotebookReportOptions())
        document = self._assembler.build(notebook.name, [section], options)
        return _file(exporter, document)
