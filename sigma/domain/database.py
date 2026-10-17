"""What the administration pages learn about a database before Sigma Algebra is pointed at it."""

from __future__ import annotations

from dataclasses import dataclass, field

# what an inspected database contains, as far as Sigma Algebra is concerned
STATE_EMPTY = "empty"  # nothing in it: Sigma Algebra creates its own tables
STATE_SIGMA = "sigma"  # tables from a previous Sigma Algebra installation: its data is used as it is
STATE_FOREIGN = "foreign"  # somebody else's tables: Sigma Algebra will not touch them


@dataclass(frozen=True)
class CheckStep:
    name: str
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class ConnectionReport:
    ok: bool
    engine: str
    message: str  # the headline, in Spanish, ready to show
    server_version: str = ""
    state: str = ""  # STATE_* once the database could be inspected
    counts: dict[str, int] | None = None  # users / datasets / results of an existing Sigma Algebra database
    steps: tuple[CheckStep, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "engine": self.engine,
            "message": self.message,
            "server_version": self.server_version,
            "state": self.state,
            "counts": self.counts,
            "steps": [{"name": s.name, "ok": s.ok, "detail": s.detail} for s in self.steps],
        }


# what a migration copies, in dependency order (each entity may refer to the ones before it)
MIGRATION_ENTITIES: tuple[tuple[str, str], ...] = (
    ("users", "Cuentas"),
    ("app_settings", "Ajustes del servidor"),
    ("report_templates", "Plantillas de informe"),
    ("datasets", "Conjuntos de datos"),
    ("saved_results", "Resultados guardados"),
    ("workspace", "Espacio de trabajo"),
    ("sync_state", "Revisiones de sincronización"),
    ("import_drafts", "Importaciones pendientes"),
    ("ai_usage", "Uso de la IA"),
    ("notebooks", "Notebooks"),
)
