"""The public demo: auto-login, banner, and the guard that keeps it read-mostly."""

from __future__ import annotations

import io

import pytest

from sigma.app_factory import create_app
from sigma.config import Config

PASSWORD = "Clave-Segura-2024!"


@pytest.fixture
def demo_app(tmp_path):
    config = Config.from_env(
        testing=True,
        instance_dir=tmp_path,
        db_path=tmp_path / "sigma.db",
        notebooks_dir=tmp_path / "notebooks",
        sample_dir=tmp_path / "samples",
        demo_mode=True,
        demo_user="demo",
    )
    application = create_app(config)
    application.testing = True
    container = application.extensions["sigma"]
    container.create_first_admin.execute("admin", "Administrador", PASSWORD)
    container.create_user.execute("demo", "Demo", PASSWORD, "user", False, False)
    return application


def _demo_client(app):
    client = app.test_client()
    response = client.get("/demo")
    assert response.status_code == 302
    with client.session_transaction() as session:
        return client, session["csrf"]


def test_demo_route_is_disabled_by_default(app):
    assert app.test_client().get("/demo").status_code == 404


def test_demo_auto_login_and_banner(demo_app):
    client, _csrf = _demo_client(demo_app)
    page = client.get("/")
    html = page.get_data(as_text=True)
    assert page.status_code == 200
    assert "Demo pública" in html
    assert "Sigma Algebra" in html


def test_demo_user_cannot_reach_admin(demo_app):
    client, _csrf = _demo_client(demo_app)
    assert client.get("/usuarios").status_code == 403
    assert client.get("/consola").status_code == 403


def test_demo_blocks_imports_and_deletions(demo_app):
    client, csrf = _demo_client(demo_app)
    headers = {"X-CSRF-Token": csrf}
    preview = client.post(
        "/api/datasets/import/preview",
        headers=headers,
        data={"source_type": "csv", "options": "{}", "file": (io.BytesIO(b"x,y\n1,2\n"), "x.csv")},
        content_type="multipart/form-data",
    )
    assert preview.status_code == 403
    assert client.delete("/api/datasets/1", headers=headers).status_code == 403
    assert client.patch("/api/datasets/1", headers=headers, json={"name": "otro"}).status_code == 403


def test_demo_still_allows_reading_and_analysis_forms(demo_app):
    client, _csrf = _demo_client(demo_app)
    assert client.get("/datos").status_code == 200
    assert client.get("/descriptiva").status_code == 200
    assert client.get("/api/datasets").status_code == 200
