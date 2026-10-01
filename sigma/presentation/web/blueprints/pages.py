"""HTML pages. Each page is a shell; its behaviour lives in ``static/js/modules/<page>.js``."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template

from sigma import __version__

bp = Blueprint("pages", __name__)

PAGES = {
    "home": ("/", "Inicio", "pages/home.html"),
    "datos": ("/datos", "Datos", "pages/datos.html"),
    "descriptiva": ("/descriptiva", "Estadística descriptiva", "pages/descriptiva.html"),
    "distribuciones": ("/distribuciones", "Distribuciones", "pages/distribuciones.html"),
    "hipotesis": ("/hipotesis", "Pruebas de hipótesis", "pages/hipotesis.html"),
    "regresion": ("/regresion", "Regresión lineal", "pages/regresion.html"),
    "anova": ("/anova", "ANOVA", "pages/anova.html"),
    "mercados": ("/mercados", "Análisis de mercados", "pages/mercados.html"),
    "consola": ("/consola", "Consola Python", "pages/consola.html"),
    "resultados": ("/resultados", "Resultados guardados", "pages/resultados.html"),
}


def _register(key: str, path: str, title: str, template: str) -> None:
    def view():
        return render_template(template, active=key, page_title=title)

    view.__name__ = f"page_{key}"
    bp.add_url_rule(path, view_func=view)


for _key, (_path, _title, _template) in PAGES.items():
    _register(_key, _path, _title, _template)


@bp.get("/health")
def health():
    return jsonify({"status": "ok", "app": "Sigma", "version": __version__})
