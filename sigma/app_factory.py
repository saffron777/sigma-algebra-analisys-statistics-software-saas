"""Flask application factory."""

from __future__ import annotations

from flask import Flask

from sigma import __copyright__, __license__, __version__
from sigma.bootstrap import build_container
from sigma.config import Config
from sigma.domain.analysis.modules import CATALOG_MODULES
from sigma.presentation.web import help_content
from sigma.presentation.web.auth import init_auth
from sigma.presentation.web.blueprints import register_blueprints
from sigma.presentation.web.blueprints.pages import PAGES
from sigma.presentation.web.errors import register_error_handlers
from sigma.presentation.web.migration import MigrationRunner, init_maintenance
from sigma.presentation.web.security import init_security
from sigma.presentation.web.switching import ContainerSwitcher

NAV_GROUPS = (
    ("Datos", ("home", "asistente", "datos", "transformar", "graficos", "resultados", "plantillas")),
    (
        "Estadística básica",
        ("descriptiva", "distribuciones", "hipotesis", "pruebas", "potencia", "remuestreo", "mercados", "encuestas", "calidad"),
    ),
    ("Modelos", ("regresion", "modelos", "anova", "anova_avanzado", "supervivencia", "sem", "doe", "optimizacion")),
    ("Series y econometría", ("series", "econometria")),
    ("Multivariado y ciencia de datos", ("multivariado", "aprendizaje")),
    ("Programación", ("consola",)),
)

# Menu-bar structure (traditional statistics-software look): top-level menu -> ordered items.
# Item markers: ("__sep__",), ("__label__", "texto"), ("__logout__",), ("__about__",); otherwise ("page_key",).
MENU = (
    ("Archivo", (("home",), ("datos",), ("resultados",), ("plantillas",), ("__sep__",), ("cuenta",), ("__logout__",))),
    ("Datos", (("datos",), ("transformar",))),
    (
        "Analizar",
        (
            ("__label__", "Descriptiva"),
            ("descriptiva",), ("distribuciones",),
            ("__label__", "Pruebas"),
            ("hipotesis",), ("pruebas",), ("potencia",), ("remuestreo",),
            ("__label__", "Mercados y encuestas"),
            ("mercados",), ("proyectos",), ("encuestas",),
            ("__label__", "Regresión y modelos"),
            ("regresion",), ("modelos",), ("anova",), ("anova_avanzado",), ("supervivencia",), ("sem",),
            ("__label__", "Diseño y optimización"),
            ("doe",), ("optimizacion",),
            ("__label__", "Series y econometría"),
            ("series",), ("econometria",),
            ("__label__", "Multivariado y ML"),
            ("multivariado",), ("aprendizaje",),
            ("__label__", "Calidad"),
            ("calidad",),
        ),
    ),
    ("Gráficos", (("graficos",),)),
    ("Utilidades", (("asistente",), ("consola",), ("__sep__",), ("usuarios",), ("basedatos",))),
    ("Ayuda", (("__help__", "/ayuda#guia", "Guía de uso"), ("__help__", "/ayuda#referencia", "Referencia teórica"), ("__sep__",), ("__help__", "/ayuda#acerca", "Acerca de"))),
)


def create_app(config: Config | None = None) -> Flask:
    config = config or Config.from_env()
    app = Flask(__name__, template_folder="presentation/web/templates", static_folder="presentation/web/static")
    app.json.sort_keys = False  # type: ignore[attr-defined]
    app.json.ensure_ascii = False  # type: ignore[attr-defined]
    app.json.compact = True  # type: ignore[attr-defined]
    app.config["TEMPLATES_AUTO_RELOAD"] = config.debug

    container = build_container(config)
    app.extensions["sigma"] = container
    switcher = ContainerSwitcher(app, close_delay=0 if config.testing else 2.0)
    switcher.migrator = MigrationRunner(app, switcher)
    container.database_admin.switcher = switcher
    container.database_admin.migrator = switcher.migrator
    init_security(app, config)  # Host/Origin/CSRF run first…
    init_maintenance(app)  # …then the gate that closes the application while a migration copies the data
    init_auth(app)  # …then the session check
    register_blueprints(app)
    register_error_handlers(app)

    nav = [
        {"title": title, "items": [{"key": key, "href": PAGES[key][0], "label": PAGES[key][1]} for key in keys]}
        for title, keys in NAV_GROUPS
    ]
    # Menu bar: resolve page keys to href/label; keep the special markers (__sep__/__label__/__logout__/__about__/__help__) as-is.
    def _menu_item(item: tuple) -> dict:
        key = item[0]
        if key == "__label__":
            return {"kind": "__label__", "label": item[1], "href": None, "key": None}
        if key == "__help__":
            return {"kind": "__help__", "label": item[2], "href": item[1], "key": None}
        if key in ("__sep__", "__logout__", "__about__"):
            return {"kind": key, "label": None, "href": None, "key": None}
        return {"kind": "page", "label": PAGES[key][1], "href": PAGES[key][0], "key": key}

    menu = [
        {"title": title, "items": [_menu_item(item) for item in items]}
        for title, items in MENU
    ]
    app.jinja_env.globals.update(
        nav_groups=nav,
        menu=menu,
        sigma_version=__version__,
        license_name=__license__,
        copyright_holder=__copyright__,
        source_url=config.source_url,
        catalog_modules=[(m.key, m.label, m.description) for m in CATALOG_MODULES],
        help_guide=help_content.GUIDE,
        help_reference=help_content.REFERENCE,
        help_about=help_content.ABOUT,
    )
    return app
