"""Report templates: create, edit, choose the default, upload the logo."""

from __future__ import annotations

from sigma.application.ports import ImageProcessor, TemplateRepository
from sigma.domain.errors import TemplateNotFoundError
from sigma.domain.report_templates import ReportTemplate, validate_fields


class ListTemplates:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self) -> list[ReportTemplate]:
        return self._templates.list()


class GetTemplate:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str) -> ReportTemplate:
        return self._templates.get(template_id)


class CreateTemplate:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, make_default: bool = False, **fields) -> ReportTemplate:
        return self._templates.add(validate_fields(**fields), make_default)


class UpdateTemplate:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str, **fields) -> ReportTemplate:
        return self._templates.update(template_id, validate_fields(**fields))


class DeleteTemplate:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str) -> None:
        self._templates.delete(template_id)


class SetDefaultTemplate:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str, default: bool = True) -> ReportTemplate:
        return self._templates.set_default(template_id, default)


class UploadTemplateLogo:
    def __init__(self, templates: TemplateRepository, images: ImageProcessor) -> None:
        self._templates = templates
        self._images = images

    def execute(self, template_id: str, data: bytes) -> ReportTemplate:
        self._templates.get(template_id)  # 404 before looking at the file
        return self._templates.set_logo(template_id, self._images.normalize_logo(data))


class RemoveTemplateLogo:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str) -> ReportTemplate:
        return self._templates.set_logo(template_id, None)


class GetTemplateLogo:
    def __init__(self, templates: TemplateRepository) -> None:
        self._templates = templates

    def execute(self, template_id: str) -> bytes:
        logo = self._templates.get_logo(template_id)
        if logo is None:
            raise TemplateNotFoundError("La plantilla no tiene logotipo.")
        return logo
