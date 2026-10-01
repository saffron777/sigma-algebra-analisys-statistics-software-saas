"""Fixtures for API tests: an isolated app (own SQLite file) and a CSRF-aware client."""

from __future__ import annotations

import io
import json

import pytest

from sigma.app_factory import create_app
from sigma.config import Config

CSRF = "test-csrf-token"


class ApiClient:
    """Thin wrapper around the Flask test client that always sends the CSRF header and parses JSON strictly."""

    def __init__(self, flask_client) -> None:
        self.raw = flask_client
        with flask_client.session_transaction() as session:
            session["csrf"] = CSRF

    def _call(self, method: str, url: str, **kwargs):
        headers = {"X-CSRF-Token": CSRF, **kwargs.pop("headers", {})}
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
        testing=True, instance_dir=tmp_path, db_path=tmp_path / "sigma.db", notebooks_dir=tmp_path / "notebooks"
    )
    application = create_app(config)
    application.testing = True
    return application


@pytest.fixture
def api(app) -> ApiClient:
    return ApiClient(app.test_client())


@pytest.fixture
def sales_dataset(api, sales_frame) -> dict:
    """The synthetic sales table imported through the real upload endpoint."""
    csv_bytes = sales_frame.to_csv(index=False).encode("utf-8")
    preview = api.post(
        "/api/datasets/import/preview",
        data={"source_type": "csv", "options": "{}", "file": (io.BytesIO(csv_bytes), "ventas.csv")},
        content_type="multipart/form-data",
    )
    assert preview.status_code == 200, preview.get_data(as_text=True)
    commit = api.post("/api/datasets/import/commit", {"token": preview.parsed["token"], "name": "ventas"})
    assert commit.status_code == 201
    return commit.parsed["dataset"]
