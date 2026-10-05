"""HTML pages. Each page is a shell; its behaviour lives in ``static/js/modules/<page>.js``."""

from __future__ import annotations

from collections.abc import Callable

from flask import Blueprint, jsonify, render_template

from sigma import __version__
from sigma.domain.analysis.modules import CATALOG_MODULES
from sigma.presentation.web.auth import require_admin, require_console

bp = Blueprint("pages", __name__)

PAGES = {
    "home": ("/", "Inicio", "pages/home.html"),
    "asistente": ("/asistente", "Asistente: ¿qué análisis usar?", "pages/asistente.html"),
    "datos": ("/datos", "Datos", "pages/datos.html"),
    "descriptiva": ("/descriptiva", "Estadística descriptiva", "pages/descriptiva.html"),
    "distribuciones": ("/distribuciones", "Distribuciones", "pages/distribuciones.html"),
    "hipotesis": ("/hipotesis", "Pruebas de hipótesis", "pages/hipotesis.html"),
    "regresion": ("/regresion", "Regresión lineal", "pages/regresion.html"),
    "anova": ("/anova", "ANOVA", "pages/anova.html"),
    "mercados": ("/mercados", "Análisis de mercados", "pages/mercados.html"),
    "consola": ("/consola", "Consola Python", "pages/consola.html"),
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
    return jsonify({"status": "ok", "app": "Sigma", "version": __version__})
