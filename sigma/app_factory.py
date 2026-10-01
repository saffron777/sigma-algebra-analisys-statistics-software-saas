"""Flask application factory."""

from __future__ import annotations

from flask import Flask

from sigma import __version__
from sigma.bootstrap import build_container
from sigma.config import Config
from sigma.presentation.web.auth import init_auth
from sigma.presentation.web.blueprints import register_blueprints
from sigma.presentation.web.blueprints.pages import PAGES
from sigma.presentation.web.errors import register_error_handlers
from sigma.presentation.web.security import init_security

NAV_GROUPS = (
    ("General", ("home", "datos", "resultados", "plantillas")),
    ("Análisis", ("descriptiva", "distribuciones", "hipotesis", "regresion", "anova", "mercados")),
    ("Programación", ("consola",)),
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
    init_security(app, config)  # Host/Origin/CSRF run first…
    init_auth(app, container)  # …then the session check
    register_blueprints(app)
    register_error_handlers(app)

    nav = [
        {"title": title, "items": [{"key": key, "href": PAGES[key][0], "label": PAGES[key][1]} for key in keys]}
        for title, keys in NAV_GROUPS
    ]
    app.jinja_env.globals.update(nav_groups=nav, sigma_version=__version__)
    return app
