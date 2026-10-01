import re

import pytest

from sigma.presentation.web.blueprints.pages import PAGES

PAGE_PATHS = [path for path, _title, _template in PAGES.values()]


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_every_page_renders_with_layout_and_csrf_token(api, path):
    response = api.raw.get(path)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert '<meta name="csrf-token" content="' in html
    assert "Sigma" in html and 'x-data="shell"' in html
    assert "{{" not in html and "{%" not in html  # no template syntax leaked
    token = re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)
    with api.raw.session_transaction() as session:
        assert session["csrf"] == token


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_pages_load_only_local_assets(api, path):
    html = api.raw.get(path).get_data(as_text=True)
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', html)
    assert external == [], f"external resources would break offline use / the CSP: {external}"


def test_navigation_marks_the_active_page(api):
    html = api.raw.get("/regresion").get_data(as_text=True)
    assert html.count('aria-current="page"') == 1
    assert re.search(r'href="/regresion"\s+aria-current="page"', html)


def test_unknown_page_renders_the_error_template(api):
    response = api.raw.get("/no-existe")
    assert response.status_code == 404 and "Recurso no encontrado" in response.get_data(as_text=True)


def test_static_assets_are_served_without_signing_in(app):
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


def test_login_and_setup_pages_render(app, anonymous):
    setup = anonymous.raw.get("/setup")
    assert setup.status_code == 200 and "cuenta de administrador" in setup.get_data(as_text=True)
    app.extensions["sigma"].create_first_admin.execute("admin", "", "Clave-Segura-2024!")
    login = anonymous.raw.get("/login?next=/regresion")
    assert login.status_code == 200 and 'data-next="/regresion"' in login.get_data(as_text=True)
    assert anonymous.raw.get("/setup").status_code == 302  # setup is only available before the first account
