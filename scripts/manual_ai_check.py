"""Throwaway manual check of the AI administration endpoints (reuses the workspace ``instance`` folder).

Run it from the project root with ``.\.venv\Scripts\python.exe scripts\manual_ai_check.py`` (with ``PYTHONPATH``
pointing at the project root). It uses ``instance/manual-check.db`` and ``instance/manual-check-ai.json`` so a real
installation is left alone, and it needs no network: it never sends a real request.
"""
from __future__ import annotations

import json
from pathlib import Path

from sigma.app_factory import create_app
from sigma.config import Config
from sigma.infrastructure.persistence.database_settings import DatabaseSettings

ROOT = Path(__file__).resolve().parent.parent
INSTANCE = ROOT / "instance"
config_path = INSTANCE / "manual-check-ai.json"
database_path = INSTANCE / "manual-check.db"

for leftover in (config_path, database_path):
    leftover.unlink(missing_ok=True)  # a previous run must not make the check fail

config = Config.from_env(
    testing=True,
    instance_dir=INSTANCE,
    db_path=database_path,
    notebooks_dir=INSTANCE / "notebooks",
    sample_dir=INSTANCE / "samples",
    database=DatabaseSettings.sqlite(database_path),
    ai_requests_per_hour=20,
)
app = create_app(config)
app.testing = True
container = app.extensions["sigma"]
container.ai_settings._path = config_path  # the check must not touch a real installation
container.create_first_admin.execute("admin", "Administrador", "Clave-Segura-2024!")
client = app.test_client()
with client.session_transaction() as session:
    session["csrf"] = "manual-token"
login = client.post(
    "/api/auth/login",
    json={"username": "admin", "password": "Clave-Segura-2024!"},
    headers={"X-CSRF-Token": "manual-token"},
)
print("login", login.status_code)
with client.session_transaction() as session:
    headers = {"X-CSRF-Token": session["csrf"]}


def call(method, url, payload=None):
    response = (
        getattr(client, method)(url, json=payload, headers=headers)
        if payload is not None
        else getattr(client, method)(url, headers=headers)
    )
    body = response.get_data(as_text=True)
    print(f"{method.upper()} {url} -> {response.status_code}")
    parsed = json.loads(body) if body and response.mimetype == "application/json" else body
    if response.status_code >= 400:
        print("   ", parsed)
    return response.status_code, parsed


_, catalog = call("get", "/api/admin/ai")
print("providers:", len(catalog["providers"]), "models:", catalog["models"], "configured:", catalog["configured"])

_, added = call("post", "/api/admin/ai/servers", {"provider": "ollama", "model": "llama3.1", "label": "Ollama local"})
ollama = next(m for m in added["models"] if m["provider"] == "ollama")
print("ollama:", ollama["id"], ollama["base_url"], "available:", ollama["available"], "key:", ollama["key_required"])

_, added = call(
    "post", "/api/admin/ai/servers", {"provider": "openai", "model": "gpt-5", "key": "sk-openai-MANUAL-1234"}
)
openai_entry = next(m for m in added["models"] if m["provider"] == "openai")
print("openai:", openai_entry["key_hint"], "available:", openai_entry["available"])
print("key leaked?", "MANUAL-1234" in json.dumps(added))

_, patched = call("patch", f"/api/admin/ai/servers/{openai_entry['id']}", {"make_default": True, "label": "ChatGPT"})
print("default is ChatGPT:", patched["default_server_id"] == openai_entry["id"])

_, checked = call("get", "/api/ai/status")
print("status:", checked["available"], checked["model"], checked["provider"], len(checked["models"]))

_, gemini = call("post", "/api/admin/ai/servers", {"provider": "gemini", "model": "gemini-2.5-pro", "key": "AIza-x"})
print("gemini url:", next(m for m in gemini["models"] if m["provider"] == "gemini")["base_url"])

_, deleted = call("delete", f"/api/admin/ai/servers/{ollama['id']}")
print("after delete:", [m["provider"] for m in deleted["models"]])

call("post", "/api/admin/ai/servers", {"provider": "azure_openai", "model": "d"})
call("post", "/api/admin/ai/servers", {"provider": "ollama", "model": "llama3.1", "base_url": "ftp://x"})
call("post", f"/api/admin/ai/servers/{openai_entry['id']}/test")

# --- la receta para DeepSeek, paso a paso -------------------------------------------------------
print("\n== Añadir DeepSeek y dárselo a un usuario ==")
_, deepseek = call(
    "post",
    "/api/admin/ai/servers",
    {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "label": "DeepSeek",
        "key": "sk-DEEPSEEK-DE-EJEMPLO-1234",  # la clave real, la que da platform.deepseek.com
    },
)
entry = next(m for m in deepseek["models"] if m["provider"] == "deepseek")
print("deepseek:", entry["id"], entry["base_url"], "listo:", entry["available"], "clave visible:", entry["key_hint"])
_, patched = call("patch", f"/api/admin/ai/servers/{entry['id']}", {"make_default": True})
print("predeterminado del servidor:", patched["default_server_id"] == entry["id"])

# el permiso «IA» es por cuenta: se marca la casilla de la columna IA
_, created = call(
    "post",
    "/api/users",
    {"username": "ana", "display_name": "Ana", "password": "Clave-Segura-2024!", "role": "user", "can_use_ai": True},
)
ana = created["user"]
print("usuario:", ana["username"], "permiso IA:", ana["can_use_ai"])
call("patch", f"/api/users/{ana['id']}", {"can_use_ai": False})
print("permiso retirado:", call("get", "/api/users")[1]["users"][-1]["can_use_ai"] is False)
call("patch", f"/api/users/{ana['id']}", {"can_use_ai": True})

print("\nai.json:", json.dumps(json.loads(config_path.read_text(encoding="utf-8")), indent=1))
container.close()
for leftover in (config_path, database_path):
    leftover.unlink(missing_ok=True)
