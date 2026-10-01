"""Development entry point: ``python run.py`` (serves on http://127.0.0.1:5000 by default)."""

from __future__ import annotations

from sigma.app_factory import create_app
from sigma.config import Config


def main() -> None:
    config = Config.from_env()
    app = create_app(config)
    print(f"Sigma disponible en http://{config.host}:{config.port}  (Ctrl+C para detener)")
    # SIGMA_DEBUG=1 enables the reloader and template auto-reload (development only)
    app.run(host=config.host, port=config.port, threaded=True, debug=config.debug, use_reloader=config.debug)


if __name__ == "__main__":
    main()
