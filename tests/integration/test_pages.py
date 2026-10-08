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


def test_static_assets_must_be_revalidated_so_no_stale_script_runs(app):
    """A cached module once made the page fail with errors from code that no longer existed."""
    client = app.test_client()
    assert client.get("/static/js/modules/usuarios.js").headers["Cache-Control"] == "no-cache, must-revalidate"
    assert client.get("/regresion").headers["Cache-Control"] == "no-store"


def test_login_and_setup_pages_render(app, anonymous):
    setup = anonymous.raw.get("/setup")
    assert setup.status_code == 200 and "cuenta de administrador" in setup.get_data(as_text=True)
    app.extensions["sigma"].create_first_admin.execute("admin", "", "Clave-Segura-2024!")
    login = anonymous.raw.get("/login?next=/regresion")
    assert login.status_code == 200 and 'data-next="/regresion"' in login.get_data(as_text=True)
    assert anonymous.raw.get("/setup").status_code == 302  # setup is only available before the first account


WORKSPACE_PAGES = [
    "datos",
    "descriptiva",
    "distribuciones",
    "hipotesis",
    "regresion",
    "anova",
    "mercados",
    "resultados",
]


@pytest.mark.parametrize("key", WORKSPACE_PAGES)
def test_workspace_modules_offer_clear_and_the_sync_conflict_bar(api, key):
    path = PAGES[key][0]
    html = api.raw.get(path).get_data(as_text=True)
    assert "clearWorkspace()" in html and "Limpiar" in html  # «Limpiar» empties the module's workspace
    assert 'x-show="wsConflict"' in html and "Cargar la versión nueva" in html  # another window changed it


def test_home_can_clear_every_module_and_the_console_page_shows_the_notebook_conflict_bar(api):
    assert "Limpiar todos los módulos" in api.raw.get("/").get_data(as_text=True)
    console = api.raw.get("/consola").get_data(as_text=True)
    assert "Sobrescribir con la mía" in console and "Otra ventana modificó este notebook" in console


EXPORT_PAGES = ["descriptiva", "distribuciones", "hipotesis", "regresion", "anova", "mercados"]


@pytest.mark.parametrize("key", EXPORT_PAGES)
def test_analysis_pages_offer_export_and_print(api, key):
    html = api.raw.get(PAGES[key][0]).get_data(as_text=True)
    assert "openExport()" in html and "Exportar" in html and "printResult()" in html and "Imprimir" in html


def test_export_dialog_and_templates_page_are_available_everywhere(api):
    home = api.raw.get("/").get_data(as_text=True)
    assert 'id="export-title"' in home and "Gestionar plantillas" in home  # the dialog is part of every layout
    assert 'href="/plantillas"' in home  # …and the templates page is in the menu
    templates = api.raw.get("/plantillas").get_data(as_text=True)
    assert "Mis plantillas" in templates and "Logotipo" in templates and "No se admite SVG" in templates


def test_results_data_and_console_pages_have_their_export_controls(api):
    assert "Exportar informe" in api.raw.get("/resultados").get_data(as_text=True)
    assert "Exportar ficha" in api.raw.get("/datos").get_data(as_text=True)
    console = api.raw.get("/consola").get_data(as_text=True)
    assert "Exportar informe" in console and "Imprimir el notebook" in console and "Salto de página antes" in console


def test_ai_controls_are_in_the_pages_that_use_them(api):
    regression = api.raw.get("/regresion").get_data(as_text=True)
    assert (
        "Interpretar con IA" in regression
        and "Interpretación (IA)" in regression
        and "¿Qué quieres saber?" in regression
    )
    # the button is never hidden without an explanation: it is disabled and the hint says why
    assert "aiAvailable()" in regression and "$store.ai.status.reason" in regression
    assert "Interpretar con IA" in api.raw.get("/descriptiva").get_data(as_text=True)
    assert "marked.min.js" in api.raw.get("/").get_data(as_text=True)  # Markdown is rendered with marked + DOMPurify
    assert "Interpretar con IA" in api.raw.get("/resultados").get_data(as_text=True)
    assert "Interpretar con IA" in api.raw.get("/datos").get_data(as_text=True)
    assert "Interpretar con IA" in api.raw.get("/consola").get_data(as_text=True)
    users = api.raw.get("/usuarios").get_data(as_text=True)
    assert "Interpretación con IA" in users and "Modelos configurados" in users and "Añadir un modelo" in users
    assert "Probar el predeterminado" in users and "base de datos" in users
    # the «Modelo» field of the add form is a list of the chosen provider's models, with a way out for the rest
    assert 'x-model="aiForm.model"' in users and "Otro modelo (escribir a mano)…" in users
    assert 'x-for="name in suggestions()"' in users and "pickModel()" in users and "providerChanged()" in users
    assert "aiForm.suggestions" not in users  # the list always comes from the helper: it is never undefined


def test_the_logo_is_used_in_the_layouts_and_served(api, app):
    home = api.raw.get("/").get_data(as_text=True)
    assert "img/logo-mark.svg" in home and 'rel="apple-touch-icon"' in home and "img/favicon.ico" in home
    login = ApiClientless(app).raw.get("/login").get_data(as_text=True)
    assert "img/logo-mark.svg" in login and "img/favicon.svg" in login
    for asset in (
        "logo-mark.svg",
        "logo-mark-small.svg",
        "logo-wordmark.svg",
        "logo-mark-256.png",
        "favicon.ico",
        "apple-touch-icon.png",
    ):
        assert api.raw.get(f"/static/img/{asset}").status_code == 200, asset


class ApiClientless:
    """A client with no session (the login page is public)."""

    def __init__(self, app):
        self.raw = app.test_client()
