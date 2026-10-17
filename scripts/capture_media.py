"""Captures the README/landing screenshots and the animated GIF from a real local instance.

    python -m pip install -r requirements-media.txt
    python scripts/capture_media.py

It starts a throwaway app, seeds the demo user + sample datasets, drives the installed Chrome with
Playwright, and writes docs/screenshots/*.png and docs/demo.gif. No real user data is used.
"""

from __future__ import annotations

import secrets
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werkzeug.serving import make_server  # noqa: E402

from sigma.app_factory import create_app  # noqa: E402
from sigma.config import Config  # noqa: E402

SHOTS = ROOT / "docs" / "screenshots"
GIF = ROOT / "docs" / "demo.gif"
PASSWORD = "Clave-Segura-2024!"


def _seed() -> tuple:
    instance = ROOT / "instance-media"
    config = Config.from_env(
        testing=True,
        instance_dir=instance,
        db_path=instance / "sigma.db",
        notebooks_dir=instance / "notebooks",
        sample_dir=ROOT / "sample_data",
        demo_mode=True,
        demo_user="demo",
    )
    app = create_app(config)
    app.testing = True
    container = app.extensions["sigma"]
    if not container.list_users.execute():
        container.create_first_admin.execute("admin", "Administrador", PASSWORD)
    if not any(u.username == "demo" for u in container.list_users.execute()):
        container.create_user.execute("demo", "Demo", secrets.token_urlsafe(24), "user", False, False)
    demo = next(u for u in container.list_users.execute() if u.username == "demo")
    container.for_user(demo).load_sample_data.execute()
    return app


def _capture(base: str) -> list[Path]:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(f"{base}/demo", wait_until="networkidle")
        page.wait_for_timeout(700)

        home = SHOTS / "03-inicio.png"
        page.goto(f"{base}/", wait_until="networkidle")
        page.wait_for_timeout(500)
        page.screenshot(path=str(home))
        frames.append(home)

        page.goto(f"{base}/datos", wait_until="networkidle")
        page.wait_for_timeout(1000)
        data_list = SHOTS / "01-datos.png"
        page.screenshot(path=str(data_list))
        frames.append(data_list)

        view = page.get_by_role("button", name="Ver").first
        if view.count():
            view.click()
            page.wait_for_timeout(1200)
            viewer = SHOTS / "01b-datos-grilla.png"
            page.screenshot(path=str(viewer))
            frames.append(viewer)

        page.get_by_role("button", name="Crear dataset manual").click()
        page.locator("#manual-name").fill("clientes")
        page.get_by_role("button", name="Añadir variable").click()
        page.locator("div[role='dialog'] input[placeholder='nombre de la variable']").first.fill("sexo")
        page.locator("div[role='dialog'] select").first.select_option("categorical")
        page.get_by_role("button", name="Definir categorías").click()
        codes = page.locator("div[role='dialog'] input[placeholder='1']")
        while codes.count() < 2:
            page.get_by_role("button", name="Añadir categoría").click()
        labels = page.locator("div[role='dialog'] input[placeholder='Masculino']")
        for index, (code, label) in enumerate((("1", "Femenino"), ("2", "Masculino"))):
            codes.nth(index).fill(code)
            labels.nth(index).fill(label)
        page.wait_for_timeout(500)
        editorial = SHOTS / "02-variables-categoricas.png"
        page.screenshot(path=str(editorial))
        frames.append(editorial)
        browser.close()
    return frames


def _gif(frames: list[Path]) -> None:
    from PIL import Image, ImageOps

    images = []
    for frame in frames:
        with Image.open(frame) as image:
            images.append(ImageOps.fit(image.convert("RGB"), (960, 600), method=Image.Resampling.LANCZOS))
    if not images:
        return
    images[0].save(
        GIF,
        save_all=True,
        append_images=images[1:],
        duration=1200,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> int:
    app = _seed()
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        frames = _capture(base)
        _gif(frames)
        print(f"Capturas: {SHOTS}")
        print(f"GIF: {GIF}")
    finally:
        server.shutdown()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
