"""Throwaway: prove what the running app actually serves for the AI admin page.

Run from the project root as ``python scripts/verify_ai_ui.py`` with ``PYTHONPATH`` pointing at the project root
(or through the virtual environment's interpreter). It uses ``instance/verify-ai.db`` and ``instance/verify-ai.json``
so a real installation is left alone, and it never sends a request to a provider.
"""

from __future__ import annotations

import re
from pathlib import Path

from sigma.app_factory import create_app
from sigma.config import Config
from sigma.infrastructure.persistence.database_settings import DatabaseSettings

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "instance"
config_path = INSTANCE / "verify-ai.json"
database_path = INSTANCE / "verify-ai.db"
for leftover in (config_path, database_path):
    leftover.unlink(missing_ok=True)

config = Config.from_env(
    testing=True,
    instance_dir=INSTANCE,
    db_path=database_path,
    notebooks_dir=INSTANCE / "notebooks",
    sample_dir=INSTANCE / "samples",
    database=DatabaseSettings.sqlite(database_path),
)
app = create_app(config)
app.testing = True
container = app.extensions["sigma"]
container.ai_settings._path = config_path
container.create_first_admin.execute("admin", "Administrador", "Clave-Segura-2024!")
client = app.test_client()
with client.session_transaction() as session:
    session["csrf"] = "tok"
client.post("/api/auth/login", json={"username": "admin", "password": "Clave-Segura-2024!"}, headers={"X-CSRF-Token": "tok"})

page = client.get("/usuarios")
html = page.get_data(as_text=True)
script = client.get("/static/js/modules/usuarios.js")
js = script.get_data(as_text=True)

print("GET /usuarios          ->", page.status_code, page.headers.get("Cache-Control"))
print("GET usuarios.js        ->", script.status_code, "content-type:", script.headers.get("Content-Type"))
print("  Cache-Control        ->", script.headers.get("Cache-Control"))
print("  bytes                ->", len(js))
print("  declares aiForm?     ->", "aiForm: emptyAiForm()" in js)
print("  suggestions() method ->", "suggestions() {" in js)
print("  old 'form.suggestions' in JS? ->", "form.suggestions" in js)
print("  old 'form.suggestions' in HTML? ->", "form.suggestions" in html)
print("  'aiForm.suggestions' in HTML? ->", "aiForm.suggestions" in html)
print("  drop-down option      ->", "Otro modelo (escribir a mano)…" in html)
print("  bindings in the card  ->", len(re.findall(r"aiForm\.", html)))

# the AI button of an analysis page is rendered (disabled) with the reason, instead of being hidden
descriptiva = client.get("/descriptiva").get_data(as_text=True)
print("GET /descriptiva       -> 'Interpretar con IA':", "Interpretar con IA" in descriptiva)
print("  disabled when off     ->", 'disabled="$store.ai.busy || !aiAvailable()"' in descriptiva)
print("  shows the reason      ->", "$store.ai.status.reason" in descriptiva)

container.close()
for leftover in (config_path, database_path):
    leftover.unlink(missing_ok=True)
print("ok")
