"""Report templates (colours, texts, page set-up and logo) on any SQL engine, scoped to one user."""

from __future__ import annotations

import re
import uuid

from sqlalchemy import case, delete, func, insert, select, update
from sqlalchemy.engine import Connection, Row
from sqlalchemy.exc import IntegrityError

from sigma.domain.errors import InvalidInputError, TemplateNotFoundError
from sigma.domain.report_templates import ReportTemplate, TemplateFields
from sigma.infrastructure.persistence.sql import schema
from sigma.infrastructure.persistence.sql.engine import Database
from sigma.infrastructure.persistence.sql.helpers import bump_sync, now_iso

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_t = schema.report_templates
_PUBLIC = [
    _t.c.id,
    _t.c.name,
    _t.c.organization,
    _t.c.primary_color,
    _t.c.secondary_color,
    _t.c.footer_text,
    _t.c.cover_text,
    case((_t.c.logo.is_not(None), 1), else_=0).label("has_logo"),
    _t.c.page_size,
    _t.c.orientation,
    _t.c.show_cover,
    _t.c.show_toc,
    _t.c.is_default,
    _t.c.created_at,
    _t.c.updated_at,
]


class SqlTemplateRepository:
    def __init__(self, database: Database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ writes
    def add(self, fields: TemplateFields, make_default: bool = False) -> ReportTemplate:
        template_id = uuid.uuid4().hex
        now = now_iso()
        try:
            with self._db.begin() as conn:
                first = (
                    conn.execute(select(func.count()).select_from(_t).where(_t.c.user_id == self._user)).scalar_one()
                    == 0
                )
                conn.execute(
                    insert(_t).values(
                        id=template_id,
                        user_id=self._user,
                        **self._values(fields),
                        is_default=0,
                        created_at=now,
                        updated_at=now,
                    )
                )
                if make_default or first:  # the first template is the default until another one is chosen
                    self._mark_default(conn, template_id)
                bump_sync(conn, self._user, "templates")
        except IntegrityError:
            raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
        return self.get(template_id)

    def update(self, template_id: str, fields: TemplateFields) -> ReportTemplate:
        self.get(template_id)
        try:
            with self._db.begin() as conn:
                conn.execute(
                    update(_t)
                    .where(_t.c.id == template_id, _t.c.user_id == self._user)
                    .values(**self._values(fields), updated_at=now_iso())
                )
                bump_sync(conn, self._user, "templates")
        except IntegrityError:
            raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
        return self.get(template_id)

    def delete(self, template_id: str) -> None:
        template = self.get(template_id)
        with self._db.begin() as conn:
            conn.execute(delete(_t).where(_t.c.id == template_id, _t.c.user_id == self._user))
            if template.is_default:  # somebody must stay the default
                heir = conn.execute(
                    select(_t.c.id).where(_t.c.user_id == self._user).order_by(_t.c.created_at, _t.c.id).limit(1)
                ).first()
                if heir:
                    self._mark_default(conn, heir.id)
            bump_sync(conn, self._user, "templates")

    def set_default(self, template_id: str, default: bool) -> ReportTemplate:
        self.get(template_id)
        with self._db.begin() as conn:
            if default:
                self._mark_default(conn, template_id)
            else:
                conn.execute(
                    update(_t)
                    .where(_t.c.id == template_id, _t.c.user_id == self._user)
                    .values(is_default=0, updated_at=now_iso())
                )
            bump_sync(conn, self._user, "templates")
        return self.get(template_id)

    def set_logo(self, template_id: str, logo: bytes | None) -> ReportTemplate:
        self.get(template_id)
        with self._db.begin() as conn:
            conn.execute(
                update(_t)
                .where(_t.c.id == template_id, _t.c.user_id == self._user)
                .values(logo=logo, updated_at=now_iso())
            )
            bump_sync(conn, self._user, "templates")
        return self.get(template_id)

    # ------------------------------------------------------------------ reads
    def get(self, template_id: str) -> ReportTemplate:
        if not _ID_PATTERN.match(template_id or ""):
            raise TemplateNotFoundError("La plantilla no existe.")
        with self._db.begin() as conn:
            row = conn.execute(select(*_PUBLIC).where(_t.c.id == template_id, _t.c.user_id == self._user)).first()
        if row is None:
            raise TemplateNotFoundError("La plantilla no existe.")
        return self._to_template(row)

    def get_default(self) -> ReportTemplate | None:
        with self._db.begin() as conn:
            row = conn.execute(select(*_PUBLIC).where(_t.c.user_id == self._user, _t.c.is_default == 1)).first()
        return self._to_template(row) if row else None

    def list(self) -> list[ReportTemplate]:
        with self._db.begin() as conn:
            rows = conn.execute(
                select(*_PUBLIC).where(_t.c.user_id == self._user).order_by(_t.c.is_default.desc(), _t.c.name)
            ).all()
        return [self._to_template(row) for row in rows]

    def get_logo(self, template_id: str) -> bytes | None:
        self.get(template_id)
        with self._db.begin() as conn:
            row = conn.execute(select(_t.c.logo).where(_t.c.id == template_id, _t.c.user_id == self._user)).first()
        return bytes(row.logo) if row and row.logo is not None else None

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _values(fields: TemplateFields) -> dict[str, object]:
        return {
            "name": fields.name,
            "name_key": fields.name.lower(),
            "organization": fields.organization,
            "primary_color": fields.primary_color,
            "secondary_color": fields.secondary_color,
            "footer_text": fields.footer_text,
            "cover_text": fields.cover_text,
            "page_size": fields.page_size,
            "orientation": fields.orientation,
            "show_cover": int(fields.show_cover),
            "show_toc": int(fields.show_toc),
        }

    def _mark_default(self, conn: Connection, template_id: str) -> None:
        conn.execute(update(_t).where(_t.c.user_id == self._user, _t.c.is_default == 1).values(is_default=0))
        conn.execute(
            update(_t)
            .where(_t.c.id == template_id, _t.c.user_id == self._user)
            .values(is_default=1, updated_at=now_iso())
        )

    @staticmethod
    def _to_template(row: Row) -> ReportTemplate:
        return ReportTemplate(
            id=row.id,
            name=row.name,
            organization=row.organization or "",  # Oracle stores '' as NULL
            primary_color=row.primary_color,
            secondary_color=row.secondary_color,
            footer_text=row.footer_text or "",
            cover_text=row.cover_text or "",
            has_logo=bool(row.has_logo),
            page_size=row.page_size,
            orientation=row.orientation,
            show_cover=bool(row.show_cover),
            show_toc=bool(row.show_toc),
            is_default=bool(row.is_default),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
