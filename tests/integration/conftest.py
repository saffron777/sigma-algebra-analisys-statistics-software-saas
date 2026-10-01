"""Fixtures for API tests: an isolated app (own SQLite file) and CSRF-aware, signed-in clients."""

from __future__ import annotations

import io
import json

import pytest

from sigma.app_factory import create_app
from sigma.config import Config

PASSWORD = "Clave-Segura-2024!"


class ApiClient:
    """Wraps the Flask test client: sends the CSRF token of the current session and parses JSON strictly."""

    def __init__(self, flask_client) -> None:
        self.raw = flask_client
        self.csrf = ""
        self._prepare_session()

    def _prepare_session(self) -> None:
        with self.raw.session_transaction() as session:
            session["csrf"] = self.csrf = self.csrf or "test-csrf-token"

    def login(self, username: str, password: str = PASSWORD):
        response = self.post("/api/auth/login", {"username": username, "password": password})
        if response.status_code == 200:  # logging in rotates the CSRF token
            with self.raw.session_transaction() as session:
                self.csrf = session["csrf"]
        return response

    def logout(self):
        response = self.post("/api/auth/logout")
        self.csrf = ""
        self._prepare_session()
        return response

    def _call(self, method: str, url: str, **kwargs):
        headers = {"X-CSRF-Token": self.csrf, **kwargs.pop("headers", {})}
        response = getattr(self.raw, method)(url, headers=headers, **kwargs)
        if response.mimetype == "application/json":
            # strict parsing: NaN / Infinity tokens would break JSON.parse in the browser
            response.parsed = json.loads(
                response.get_data(as_text=True), parse_constant=lambda token: pytest.fail(f"non-JSON constant {token}")
            )
        return response

    def get(self, url, **kw):
        return self._call("get", url, **kw)

    def post(self, url, payload=None, **kw):
        if payload is not None:
            kw["json"] = payload
        return self._call("post", url, **kw)

    def put(self, url, payload=None, **kw):
        return self._call("put", url, json=payload, **kw)

    def patch(self, url, payload=None, **kw):
        return self._call("patch", url, json=payload, **kw)

    def delete(self, url, **kw):
        return self._call("delete", url, **kw)


@pytest.fixture
def app(tmp_path):
    config = Config.from_env(
        testing=True,
        instance_dir=tmp_path,
        db_path=tmp_path / "sigma.db",
        notebooks_dir=tmp_path / "notebooks",
        sample_dir=tmp_path / "samples",
    )
    application = create_app(config)
    application.testing = True
    return application


@pytest.fixture
def anonymous(app) -> ApiClient:
    """A client without a session (no account has been created yet)."""
    return ApiClient(app.test_client())


@pytest.fixture
def api(app, anonymous) -> ApiClient:
    """A client signed in as the administrator ``admin`` (console and AI allowed)."""
    app.extensions["sigma"].create_first_admin.execute("admin", "Administrador", PASSWORD)
    assert anonymous.login("admin").status_code == 200
    return anonymous


@pytest.fixture
def make_client(app, api):
    """Factory of signed-in clients for extra accounts: ``make_client("luis", console=True)``."""
    services = app.extensions["sigma"]

    def _make(username: str, *, role: str = "user", console: bool = False, ai: bool = False) -> ApiClient:
        services.create_user.execute(username, username.title(), PASSWORD, role, console, ai)
        client = ApiClient(app.test_client())
        assert client.login(username).status_code == 200
        return client

    return _make


@pytest.fixture
def sales_dataset(api, sales_frame) -> dict:
    """The synthetic sales table imported through the real upload endpoint."""
    return import_frame(api, sales_frame, "ventas")


def import_frame(client: ApiClient, frame, name: str) -> dict:
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    preview = client.post(
        "/api/datasets/import/preview",
        data={"source_type": "csv", "options": "{}", "file": (io.BytesIO(csv_bytes), f"{name}.csv")},
        content_type="multipart/form-data",
    )
    assert preview.status_code == 200, preview.get_data(as_text=True)
    commit = client.post("/api/datasets/import/commit", {"token": preview.parsed["token"], "name": name})
    assert commit.status_code == 201, commit.get_data(as_text=True)
    return commit.parsed["dataset"]
