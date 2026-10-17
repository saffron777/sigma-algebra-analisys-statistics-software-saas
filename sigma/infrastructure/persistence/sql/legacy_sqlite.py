"""The SQLite-only schema history (v1–v6) and the step that lifts such a file to the portable baseline (v7).

New databases — on any engine, SQLite included — are created from ``schema.py`` and never run these scripts. They
exist so that a ``sigma.db`` written by an earlier Sigma Algebra keeps working: ``upgrade_legacy_sqlite`` replays what is
missing (tracked with ``PRAGMA user_version``) and then applies :data:`V7_SCRIPT`. Never edit a released script.
"""

from __future__ import annotations

import re
import sqlite3


def _bump_trigger(table: str, scope: str) -> str:
    """Triggers that bump a user's revision for ``scope`` on every insert/update/delete of ``table``."""
    bump = (
        "INSERT INTO sync_state (user_id, scope, revision) VALUES ({row}.user_id, '{scope}', 1) "
        "ON CONFLICT (user_id, scope) DO UPDATE SET revision = revision + 1;"
    )
    parts = []
    for event, row in (("INSERT", "NEW"), ("UPDATE", "NEW"), ("DELETE", "OLD")):
        # the EXISTS guard matters when an account is deleted: its rows cascade away after the user row is gone
        condition = f"{row}.user_id IS NOT NULL AND EXISTS (SELECT 1 FROM users WHERE id = {row}.user_id)"
        parts.append(
            f"CREATE TRIGGER trg_{table}_{event.lower()} AFTER {event} ON {table} "
            f"WHEN {condition} BEGIN {bump.format(row=row, scope=scope)} END;"
        )
    return "\n".join(parts)


V1_BASE = """
CREATE TABLE datasets (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    source_type  TEXT NOT NULL,
    source_meta  TEXT NOT NULL DEFAULT '{}',
    n_rows       INTEGER NOT NULL,
    n_cols       INTEGER NOT NULL,
    dtypes       TEXT NOT NULL,
    columns_info TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE saved_results (
    id            TEXT PRIMARY KEY,
    module        TEXT NOT NULL,
    analysis_type TEXT NOT NULL,
    title         TEXT NOT NULL,
    notes         TEXT NOT NULL DEFAULT '',
    dataset_id    TEXT REFERENCES datasets(id) ON DELETE SET NULL,
    dataset_name  TEXT,
    params        TEXT NOT NULL,
    result        TEXT NOT NULL,
    report        TEXT NOT NULL,
    figures       TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);
CREATE INDEX idx_saved_results_module  ON saved_results(module);
CREATE INDEX idx_saved_results_created ON saved_results(created_at DESC);
"""

V2_ACCOUNTS = f"""
CREATE TABLE users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    display_name    TEXT NOT NULL,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('admin', 'user')),
    can_use_console INTEGER NOT NULL DEFAULT 0,
    can_use_ai      INTEGER NOT NULL DEFAULT 0,
    is_active       INTEGER NOT NULL DEFAULT 1,
    auth_epoch      INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL,
    last_login_at   TEXT
);

CREATE TABLE datasets_v2 (
    id           TEXT PRIMARY KEY,
    user_id      INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name         TEXT NOT NULL COLLATE NOCASE,
    source_type  TEXT NOT NULL,
    source_meta  TEXT NOT NULL DEFAULT '{{}}',
    n_rows       INTEGER NOT NULL,
    n_cols       INTEGER NOT NULL,
    dtypes       TEXT NOT NULL,
    columns_info TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    UNIQUE (user_id, name)
);
INSERT INTO datasets_v2 (id, user_id, name, source_type, source_meta, n_rows, n_cols, dtypes, columns_info, created_at)
    SELECT id, NULL, name, source_type, source_meta, n_rows, n_cols, dtypes, columns_info, created_at FROM datasets;
DROP TABLE datasets;
ALTER TABLE datasets_v2 RENAME TO datasets;
CREATE INDEX idx_datasets_user ON datasets(user_id);

ALTER TABLE saved_results ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX idx_saved_results_user ON saved_results(user_id);

CREATE TABLE sync_state (
    user_id  INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    scope    TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, scope)
);
{_bump_trigger("datasets", "datasets")}
{_bump_trigger("saved_results", "results")}
"""

V3_WORKSPACE = """
CREATE TABLE workspace (
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module     TEXT NOT NULL,
    state      TEXT,
    payload    TEXT,
    revision   INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, module)
);
"""

V4_IMPORT_DRAFTS = f"""
CREATE TABLE import_drafts (
    id             TEXT PRIMARY KEY,
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_type    TEXT NOT NULL,
    file_name      TEXT,
    file_blob      BLOB,
    file_size      INTEGER NOT NULL DEFAULT 0,
    options        TEXT NOT NULL DEFAULT '{{}}',
    staged_table   TEXT,
    n_rows         INTEGER,
    n_cols         INTEGER,
    dtypes         TEXT,
    columns_info   TEXT,
    source_meta    TEXT NOT NULL DEFAULT '{{}}',
    detected       TEXT NOT NULL DEFAULT '{{}}',
    suggested_name TEXT,
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    UNIQUE (user_id, source_type)
);
{_bump_trigger("import_drafts", "drafts")}
"""

V5_REPORT_TEMPLATES = f"""
CREATE TABLE report_templates (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL COLLATE NOCASE,
    organization    TEXT NOT NULL DEFAULT '',
    primary_color   TEXT NOT NULL,
    secondary_color TEXT NOT NULL,
    footer_text     TEXT NOT NULL DEFAULT '',
    cover_text      TEXT NOT NULL DEFAULT '',
    logo            BLOB,
    page_size       TEXT NOT NULL DEFAULT 'A4',
    orientation     TEXT NOT NULL DEFAULT 'portrait',
    show_cover      INTEGER NOT NULL DEFAULT 1,
    show_toc        INTEGER NOT NULL DEFAULT 1,
    is_default      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    UNIQUE (user_id, name)
);
{_bump_trigger("report_templates", "templates")}
"""

V6_AI = """
ALTER TABLE saved_results ADD COLUMN interpretation TEXT;

CREATE TABLE app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE ai_usage (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    module        TEXT NOT NULL,
    model         TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE INDEX idx_ai_usage_user_time ON ai_usage(user_id, created_at);
"""

MIGRATIONS: tuple[str, ...] = (V1_BASE, V2_ACCOUNTS, V3_WORKSPACE, V4_IMPORT_DRAFTS, V5_REPORT_TEMPLATES, V6_AI)

# ---------------------------------------------------------------- v7: the portable baseline
_TRIGGER_NAMES = [
    f"trg_{table}_{event}"
    for table in ("datasets", "saved_results", "import_drafts", "report_templates")
    for event in ("insert", "update", "delete")
]

_DROP_TRIGGERS = "\n".join(f"DROP TRIGGER IF EXISTS {name};" for name in _TRIGGER_NAMES)

V7_SCRIPT = f"""
ALTER TABLE users ADD COLUMN username_key TEXT;
UPDATE users SET username_key = lower(username);
CREATE UNIQUE INDEX IF NOT EXISTS uq_users_username_key ON users(username_key);

ALTER TABLE datasets ADD COLUMN name_key TEXT;
ALTER TABLE datasets ADD COLUMN table_name TEXT;
ALTER TABLE datasets ADD COLUMN col_map TEXT;
UPDATE datasets SET name_key = lower(name), table_name = 'ds_' || id;
CREATE UNIQUE INDEX IF NOT EXISTS uq_datasets_user_name ON datasets(user_id, name_key);

ALTER TABLE report_templates ADD COLUMN name_key TEXT;
UPDATE report_templates SET name_key = lower(name);
CREATE UNIQUE INDEX IF NOT EXISTS uq_templates_user_name ON report_templates(user_id, name_key);

ALTER TABLE import_drafts ADD COLUMN col_map TEXT;
CREATE TABLE IF NOT EXISTS import_draft_files (
    draft_id TEXT NOT NULL REFERENCES import_drafts(id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    data     BLOB NOT NULL,
    PRIMARY KEY (draft_id, seq)
);
INSERT INTO import_draft_files (draft_id, seq, data)
    SELECT id, 0, file_blob FROM import_drafts WHERE file_blob IS NOT NULL;
UPDATE import_drafts SET file_blob = NULL;

{_DROP_TRIGGERS}
"""

_STAGED_TABLE = re.compile(r"^stg_[0-9a-f]{32}$")


def upgrade_legacy_sqlite(conn: sqlite3.Connection) -> None:
    """Lift a SQLite file written by an earlier Sigma Algebra (``user_version`` 0–6) to the portable v7 baseline."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.execute("PRAGMA foreign_keys = OFF")  # rebuilding a table requires it
    for target, script in enumerate(MIGRATIONS, start=1):
        if target > version:
            _run(conn, script, target)
    _run(conn, V7_SCRIPT, 7)
    # the parsed previews of pending imports were stored under the old naming; they are cheap to rebuild from the
    # file, which is kept, so drop them instead of teaching the new code a second layout
    rows = conn.execute("SELECT id, staged_table FROM import_drafts WHERE staged_table IS NOT NULL").fetchall()
    for _draft_id, table in rows:
        if _STAGED_TABLE.match(table or ""):
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(
        "UPDATE import_drafts SET staged_table = NULL, n_rows = NULL, n_cols = NULL, dtypes = NULL,"
        " columns_info = NULL, detected = '{}', suggested_name = NULL WHERE staged_table IS NOT NULL"
    )
    conn.commit()
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        raise sqlite3.IntegrityError(f"La actualización dejó {len(violations)} referencias huérfanas.")


def _run(conn: sqlite3.Connection, script: str, target: int) -> None:
    try:
        conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {target};\nCOMMIT;")
    except sqlite3.Error:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
