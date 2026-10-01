"""Production-style entry point for a WSGI server, e.g. ``waitress-serve --listen=127.0.0.1:5000 wsgi:app``."""

from sigma.app_factory import create_app

app = create_app()
