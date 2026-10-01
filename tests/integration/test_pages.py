import re

import pytest

from sigma.presentation.web.blueprints.pages import PAGES

PAGE_PATHS = [path for path, _title, _template in PAGES.values()]


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_every_page_renders_with_layout_and_csrf_token(app, path):
    client = app.test_client()
    response = client.get(path)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert '<meta name="csrf-token" content="' in html
    assert "Sigma" in html and 'x-data="shell"' in html
    assert "{{" not in html and "{%" not in html  # no template syntax leaked
    token = re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)
    with client.session_transaction() as session:
        assert session["csrf"] == token


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_pages_load_only_local_assets(app, path):
    html = app.test_client().get(path).get_data(as_text=True)
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert external == [], f"external resources would break offline use / the CSP: {external}"


def test_navigation_marks_the_active_page(app):
    html = app.test_client().get("/regresion").get_data(as_text=True)
    assert html.count('aria-current="page"') == 1
    assert re.search(r'href="/regresion"\s+aria-current="page"', html)


def test_unknown_page_renders_the_error_template(app):
    response = app.test_client().get("/no-existe")
    assert response.status_code == 404 and "Recurso no encontrado" in response.get_data(as_text=True)


def test_static_assets_are_served(app):
    client = app.test_client()
    for asset in (
        "js/app.js",
        "js/report.js",
        "js/output.js",
        "css/app.css",
        "vendor/plotly.min.js",
        "vendor/alpine.min.js",
        "vendor/marked.min.js",
        "vendor/purify.min.js",
        "img/favicon.svg",
    ):
        assert client.get(f"/static/{asset}").status_code == 200, asset
