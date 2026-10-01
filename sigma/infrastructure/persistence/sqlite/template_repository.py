"""SQLite store for the user's report templates (colours, texts, page set-up and logo)."""

from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import UTC, datetime

from sigma.domain.errors import InvalidInputError, TemplateNotFoundError
from sigma.domain.report_templates import ReportTemplate, TemplateFields

_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_COLUMNS = (
    "id, name, organization, primary_color, secondary_color, footer_text, cover_text, "
    "logo IS NOT NULL AS has_logo, page_size, orientation, show_cover, show_toc, is_default, created_at, updated_at"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SqliteTemplateRepository:
    def __init__(self, database, user_id: int) -> None:
        self._db = database
        self._user = int(user_id)

    # ------------------------------------------------------------------ writes
    def add(self, fields: TemplateFields, make_default: bool = False) -> ReportTemplate:
        template_id = uuid.uuid4().hex
        now = _now()
        with self._db.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            first = (
                conn.execute("SELECT COUNT(*) FROM report_templates WHERE user_id = ?", (self._user,)).fetchone()[0]
                == 0
            )
            try:
                conn.execute(
                    "INSERT INTO report_templates (id, user_id, name, organization, primary_color, secondary_color,"
                    " footer_text, cover_text, page_size, orientation, show_cover, show_toc, is_default, created_at,"
                    " updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (template_id, self._user, *self._values(fields), now, now),
                )
            except sqlite3.IntegrityError:
                raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
            if make_default or first:  # the first template is the default until another one is chosen
                self._mark_default(conn, template_id)
        return self.get(template_id)

    def update(self, template_id: str, fields: TemplateFields) -> ReportTemplate:
        self.get(template_id)
        with self._db.connection() as conn:
            try:
                conn.execute(
                    "UPDATE report_templates SET name = ?, organization = ?, primary_color = ?, secondary_color = ?,"
                    " footer_text = ?, cover_text = ?, page_size = ?, orientation = ?, show_cover = ?, show_toc = ?,"
                    " updated_at = ? WHERE id = ? AND user_id = ?",
                    (*self._values(fields), _now(), template_id, self._user),
                )
            except sqlite3.IntegrityError:
                raise InvalidInputError(f"Ya existe una plantilla llamada «{fields.name}».") from None
        return self.get(template_id)

    def delete(self, template_id: str) -> None:
        template = self.get(template_id)
        with self._db.connection() as conn:
            conn.execute("DELETE FROM report_templates WHERE id = ? AND user_id = ?", (template_id, self._user))
            if template.is_default:  # somebody must stay the default
                heir = conn.execute(
                    "SELECT id FROM report_templates WHERE user_id = ? ORDER BY created_at LIMIT 1", (self._user,)
                ).fetchone()
                if heir:
                    self._mark_default(conn, heir["id"])

    def set_default(self, template_id: str, default: bool) -> ReportTemplate:
        self.get(template_id)
        with self._db.connection() as conn:
            if default:
                self._mark_default(conn, template_id)
            else:
                conn.execute(
                    "UPDATE report_templates SET is_default = 0, updated_at = ? WHERE id = ? AND user_id = ?",
                    (_now(), template_id, self._user),
                )
        return self.get(template_id)

    def set_logo(self, template_id: str, logo: bytes | None) -> ReportTemplate:
        self.get(template_id)
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE report_templates SET logo = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (logo, _now(), template_id, self._user),
            )
        return self.get(template_id)

    # ------------------------------------------------------------------ reads
    def get(self, template_id: str) -> ReportTemplate:
        if not _ID_PATTERN.match(template_id or ""):
            raise TemplateNotFoundError("La plantilla no existe.")
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM report_templates WHERE id = ? AND user_id = ?", (template_id, self._user)
            ).fetchone()
        if row is None:
            raise TemplateNotFoundError("La plantilla no existe.")
        return self._to_template(row)

    def get_default(self) -> ReportTemplate | None:
        with self._db.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM report_templates WHERE user_id = ? AND is_default = 1", (self._user,)
            ).fetchone()
        return self._to_template(row) if row else None

    def list(self) -> list[ReportTemplate]:
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM report_templates WHERE user_id = ? ORDER BY is_default DESC, name",
                (self._user,),
            ).fetchall()
        return [self._to_template(row) for row in rows]

    def get_logo(self, template_id: str) -> bytes | None:
        self.get(template_id)
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT logo FROM report_templates WHERE id = ? AND user_id = ?", (template_id, self._user)
            ).fetchone()
        return bytes(row["logo"]) if row and row["logo"] is not None else None

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _values(fields: TemplateFields) -> tuple:
        return (
            fields.name,
            fields.organization,
            fields.primary_color,
            fields.secondary_color,
            fields.footer_text,
            fields.cover_text,
            fields.page_size,
            fields.orientation,
            int(fields.show_cover),
            int(fields.show_toc),
        )

    def _mark_default(self, conn: sqlite3.Connection, template_id: str) -> None:
        conn.execute("UPDATE report_templates SET is_default = 0 WHERE user_id = ? AND is_default = 1", (self._user,))
        conn.execute(
            "UPDATE report_templates SET is_default = 1, updated_at = ? WHERE id = ? AND user_id = ?",
            (_now(), template_id, self._user),
        )

    @staticmethod
    def _to_template(row: sqlite3.Row) -> ReportTemplate:
        return ReportTemplate(
            id=row["id"],
            name=row["name"],
            organization=row["organization"],
            primary_color=row["primary_color"],
            secondary_color=row["secondary_color"],
            footer_text=row["footer_text"],
            cover_text=row["cover_text"],
            has_logo=bool(row["has_logo"]),
            page_size=row["page_size"],
            orientation=row["orientation"],
            show_cover=bool(row["show_cover"]),
            show_toc=bool(row["show_toc"]),
            is_default=bool(row["is_default"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
