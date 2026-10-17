"""HTML pages. Each page is a shell; its behaviour lives in ``static/js/modules/<page>.js``."""

from __future__ import annotations

from collections.abc import Callable

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, send_file, url_for

from sigma import __version__
from sigma.config import PROJECT_ROOT
from sigma.domain.analysis.modules import CATALOG_MODULES
from sigma.presentation.web.auth import login_user, require_admin, require_console

bp = Blueprint("pages", __name__)

# Public legal texts (also linked from the login page: the AGPL notice must be visible without a session).
LEGAL_FILES = {
    "licencia": "LICENSE",
    "terceros": "THIRD-PARTY-NOTICES.md",
    "marcas": "TRADEMARKS.md",
}

PAGES = {
    "home": ("/", "Inicio", "pages/home.html"),
    "asistente": ("/asistente", "Asistente: ¿qué análisis usar?", "pages/asistente.html"),
    "ayuda": ("/ayuda", "Ayuda", "pages/ayuda.html"),
    "datos": ("/datos", "Datos", "pages/datos.html"),
    "descriptiva": ("/descriptiva", "Estadística descriptiva", "pages/descriptiva.html"),
    "distribuciones": ("/distribuciones", "Distribuciones", "pages/distribuciones.html"),
    "hipotesis": ("/hipotesis", "Pruebas de hipótesis", "pages/hipotesis.html"),
    "regresion": ("/regresion", "Regresión lineal", "pages/regresion.html"),
    "anova": ("/anova", "ANOVA", "pages/anova.html"),
    "mercados": ("/mercados", "Análisis de mercados", "pages/mercados.html"),
    "proyectos": ("/proyectos", "Diseño de encuestas", "pages/encuestas.html"),
    "consola": ("/consola", "Consola (Python · R)", "pages/consola.html"),
    "resultados": ("/resultados", "Resultados guardados", "pages/resultados.html"),
    "plantillas": ("/plantillas", "Plantillas de informe", "pages/plantillas.html"),
    "cuenta": ("/cuenta", "Mi cuenta", "pages/cuenta.html"),
    "usuarios": ("/usuarios", "Usuarios", "pages/usuarios.html"),
    "basedatos": ("/basedatos", "Base de datos", "pages/basedatos.html"),
    **{m.key: (m.path, m.label, "pages/catalogo.html") for m in CATALOG_MODULES},
}

# pages that need more than a valid session (the guard raises a PermissionDeniedError → 403 page)
GUARDS: dict[str, Callable[[], None]] = {
    "consola": require_console,
    "usuarios": require_admin,
    "basedatos": require_admin,
}


def _register(key: str, path: str, title: str, template: str) -> None:
    guard = GUARDS.get(key)

    def view():
        if guard:
            guard()
        return render_template(template, active=key, page_title=title, module=key)

    view.__name__ = f"page_{key}"
    bp.add_url_rule(path, view_func=view)


for _key, (_path, _title, _template) in PAGES.items():
    _register(_key, _path, _title, _template)


@bp.get("/health")
def health():
    return jsonify({"status": "ok", "app": "Sigma Algebra", "version": __version__})


@bp.get("/demo")
def demo_login():
    """Log the public demo account in, so visitors can explore without a password."""
    if not current_app.config.get("SIGMA_DEMO_MODE"):
        abort(404)
    container = current_app.extensions["sigma"]
    username = str(current_app.config.get("SIGMA_DEMO_USER", "demo"))
    user = next((u for u in container.list_users.execute() if u.username.lower() == username.lower()), None)
    if user is None:
        abort(503, description="La demo no está inicializada. Ejecute demo/seed.sh.")
    login_user(user)
    return redirect(url_for("pages.page_home"))


def _legal_text(name: str):
    """Serve one of the fixed legal files; no user input is ever used as a path."""
    path = PROJECT_ROOT / LEGAL_FILES[name]
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="text/plain", max_age=0)


@bp.get("/licencia")
def license_text():
    return _legal_text("licencia")


@bp.get("/terceros")
def third_party_notices():
    return _legal_text("terceros")


@bp.get("/marcas")
def trademarks():
    return _legal_text("marcas")
