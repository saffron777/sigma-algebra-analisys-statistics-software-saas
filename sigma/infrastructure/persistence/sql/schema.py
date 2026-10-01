"""The relational schema as code (SQLAlchemy Core): SQLite, SQL Server, MySQL/MariaDB, PostgreSQL and Oracle.

Portability rules (see docs/ARCHITECTURE.md): no triggers (revisions are bumped by the repositories), no
``COLLATE NOCASE`` (case-insensitive uniqueness uses a lower-cased ``*_key`` column), dates are ISO-8601 text,
booleans are 0/1 integers, long text/JSON uses ``LongText`` and binaries use ``Blob``.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Double,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Unicode,
    UnicodeText,
    UniqueConstraint,
)
from sqlalchemy.dialects import mysql

SCHEMA_VERSION = 7  # v1–v6 are the SQLite-only history kept in legacy_sqlite.py; v7 is the portable baseline

LongText = UnicodeText().with_variant(mysql.LONGTEXT(), "mysql", "mariadb")
Blob = LargeBinary().with_variant(mysql.LONGBLOB(), "mysql", "mariadb")
Str = Unicode

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Str(64), nullable=False),
    Column("username_key", Str(64), nullable=False),
    Column("display_name", Str(120), nullable=False),
    Column("password_hash", Str(255), nullable=False),
    Column("role", Str(16), nullable=False),
    Column("can_use_console", Integer, nullable=False, default=0),
    Column("can_use_ai", Integer, nullable=False, default=0),
    Column("is_active", Integer, nullable=False, default=1),
    Column("auth_epoch", Integer, nullable=False, default=1),
    Column("created_at", Str(40), nullable=False),
    Column("last_login_at", Str(40)),
    UniqueConstraint("username_key", name="uq_users_username_key"),
    CheckConstraint("role IN ('admin', 'user')", name="ck_users_role"),
)

datasets = Table(
    "datasets",
    metadata,
    Column("id", Str(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE")),  # NULL only for pre-accounts data
    Column("name", Str(120), nullable=False),
    Column("name_key", Str(120), nullable=False),
    Column("source_type", Str(24), nullable=False),
    Column("source_meta", LongText, nullable=False),
    Column("n_rows", BigInteger, nullable=False),
    Column("n_cols", Integer, nullable=False),
    Column("dtypes", LongText, nullable=False),
    Column("columns_info", LongText, nullable=False),
    Column("created_at", Str(40), nullable=False),
    Column("table_name", Str(40), nullable=False),  # the physical table that holds the rows
    Column("col_map", LongText),  # JSON list of physical column names; NULL = same as the dataset's column names
    UniqueConstraint("user_id", "name_key", name="uq_datasets_user_name"),
    Index("ix_datasets_user", "user_id"),
)

saved_results = Table(
    "saved_results",
    metadata,
    Column("id", Str(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE")),
    Column("module", Str(24), nullable=False),
    Column("analysis_type", Str(80), nullable=False),
    Column("title", Str(200), nullable=False),
    Column("notes", LongText, nullable=False),
    Column("dataset_id", Str(32)),  # a plain reference: the repositories clear it when the dataset is deleted
    Column("dataset_name", Str(120)),
    Column("params", LongText, nullable=False),
    Column("result", LongText, nullable=False),
    Column("report", LongText, nullable=False),
    Column("figures", LongText, nullable=False),
    Column("created_at", Str(40), nullable=False),
    Column("updated_at", Str(40), nullable=False),
    Column("interpretation", LongText),
    Index("ix_saved_results_user", "user_id"),
    Index("ix_saved_results_module", "module"),
    Index("ix_saved_results_created", "created_at"),
)

sync_state = Table(
    "sync_state",
    metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("scope", Str(24), primary_key=True),
    Column("revision", Integer, nullable=False, default=0),
)

workspace = Table(
    "workspace",
    metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("module", Str(24), primary_key=True),
    Column("state", LongText),
    Column("payload", LongText),
    Column("revision", Integer, nullable=False, default=1),
    Column("updated_at", Str(40), nullable=False),
)

import_drafts = Table(
    "import_drafts",
    metadata,
    Column("id", Str(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("source_type", Str(24), nullable=False),
    Column("file_name", Str(255)),
    Column("file_size", BigInteger, nullable=False, default=0),
    Column("options", LongText, nullable=False),
    Column("staged_table", Str(40)),
    Column("n_rows", BigInteger),
    Column("n_cols", Integer),
    Column("dtypes", LongText),
    Column("columns_info", LongText),
    Column("col_map", LongText),
    Column("source_meta", LongText, nullable=False),
    Column("detected", LongText, nullable=False),
    Column("suggested_name", Str(120)),
    Column("created_at", Str(40), nullable=False),
    Column("updated_at", Str(40), nullable=False),
    UniqueConstraint("user_id", "source_type", name="uq_import_drafts_user_source"),
)

# an uploaded file is stored in pieces so that no single statement exceeds a server's packet limit (MySQL, SQL Server…)
import_draft_files = Table(
    "import_draft_files",
    metadata,
    Column("draft_id", Str(32), ForeignKey("import_drafts.id", ondelete="CASCADE"), primary_key=True),
    Column("seq", Integer, primary_key=True),
    Column("data", Blob, nullable=False),
)

report_templates = Table(
    "report_templates",
    metadata,
    Column("id", Str(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("name", Str(80), nullable=False),
    Column("name_key", Str(80), nullable=False),
    Column("organization", Str(120)),
    Column("primary_color", Str(7), nullable=False),
    Column("secondary_color", Str(7), nullable=False),
    Column("footer_text", Str(200)),
    Column("cover_text", Str(300)),
    Column("logo", Blob),
    Column("page_size", Str(8), nullable=False),
    Column("orientation", Str(12), nullable=False),
    Column("show_cover", Integer, nullable=False, default=1),
    Column("show_toc", Integer, nullable=False, default=1),
    Column("is_default", Integer, nullable=False, default=0),
    Column("created_at", Str(40), nullable=False),
    Column("updated_at", Str(40), nullable=False),
    UniqueConstraint("user_id", "name_key", name="uq_templates_user_name"),
)

app_settings = Table(
    "app_settings",
    metadata,
    Column("key", Str(64), primary_key=True),
    Column("value", Str(1000), nullable=False),
)

ai_usage = Table(
    "ai_usage",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("module", Str(24), nullable=False),
    Column("model", Str(80), nullable=False),
    Column("input_tokens", Integer, nullable=False, default=0),
    Column("output_tokens", Integer, nullable=False, default=0),
    Column("created_at", Str(40), nullable=False),
    Index("ix_ai_usage_user_time", "user_id", "created_at"),
)

# notebooks live here for every engine except SQLite (which keeps plain .ipynb files, openable in Jupyter)
notebooks = Table(
    "notebooks",
    metadata,
    Column("id", Str(32), primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("name", Str(120), nullable=False),
    Column("n_cells", Integer, nullable=False, default=0),
    Column("created_at", Str(40), nullable=False),
    Column("updated_at", Str(40), nullable=False),
    Column("revision", BigInteger, nullable=False, default=1),
    Column("content", LongText, nullable=False),
    Index("ix_notebooks_user", "user_id"),
)

schema_version = Table(
    "sigma_schema",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=False),
    Column("version", Integer, nullable=False),
)

# the data tables of datasets and the staged previews are created on the fly (see table_io.py)
FLOAT = Double
