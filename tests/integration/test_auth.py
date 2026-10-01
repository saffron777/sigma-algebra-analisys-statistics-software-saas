import pytest

from sigma.presentation.web.auth import safe_next
from tests.integration.conftest import PASSWORD, ApiClient

PROTECTED_API = ["/api/datasets", "/api/results", "/api/distributions", "/api/hypothesis/catalog", "/api/auth/me"]


# ------------------------------------------------------------------ first-run setup
def test_before_any_account_everything_points_to_setup(anonymous):
    status = anonymous.get("/api/auth/status")
    assert status.parsed == {"setup_required": True, "user": None}
    api_call = anonymous.get("/api/datasets")
    assert api_call.status_code == 401 and api_call.parsed["error"]["code"] == "setup_required"
    page = anonymous.raw.get("/regresion")
    assert page.status_code == 302 and page.headers["Location"] == "/setup"
    assert anonymous.get("/health").status_code == 200  # monitoring keeps working


def test_setup_creates_the_first_administrator_and_signs_in(anonymous):
    csrf_before = anonymous.csrf
    created = anonymous.post("/api/auth/setup", {"username": "ana", "display_name": "Ana Pérez", "password": PASSWORD})
    assert created.status_code == 201
    user = created.parsed["user"]
    assert user["role"] == "admin" and user["can_use_console"] and user["can_use_ai"] and "password" not in str(user)
    with anonymous.raw.session_transaction() as session:
        anonymous.csrf = session["csrf"]
        assert session["uid"] == user["id"]
    assert anonymous.csrf != csrf_before  # the CSRF token is rotated on sign-in
    assert user["last_login_at"]  # setup counts as the first sign-in
    assert anonymous.get("/api/auth/me").parsed["user"]["username"] == "ana"
    again = anonymous.post("/api/auth/setup", {"username": "otro", "password": PASSWORD})
    assert again.status_code == 403  # setup closes after the first account


def test_setup_validates_username_and_password(anonymous):
    assert anonymous.post("/api/auth/setup", {"username": "a b", "password": PASSWORD}).status_code == 422
    weak = anonymous.post("/api/auth/setup", {"username": "ana", "password": "corta"})
    assert weak.status_code == 422 and "10 caracteres" in weak.parsed["error"]["message"]
    assert anonymous.post("/api/auth/setup", {"username": "ana"}).status_code == 422


# ------------------------------------------------------------------ guard
@pytest.mark.parametrize("path", PROTECTED_API)
def test_protected_api_returns_401_without_a_session(app, anonymous, path):
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    response = anonymous.get(path)
    assert response.status_code == 401 and response.parsed["error"]["code"] == "unauthenticated"


def test_pages_redirect_to_login_with_a_return_path(app, anonymous):
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    response = anonymous.raw.get("/regresion?load=abc")
    assert response.status_code == 302
    assert response.headers["Location"] == "/login?next=/regresion%3Fload%3Dabc"
    assert anonymous.raw.get("/login").status_code == 200


def test_safe_next_blocks_open_redirects():
    assert safe_next("/regresion?x=1") == "/regresion?x=1"
    for evil in (None, "", "https://evil.example", "//evil.example", "/\\evil", "javascript:alert(1)", "regresion"):
        assert safe_next(evil) == "/"


# ------------------------------------------------------------------ login / logout
def test_login_failures_are_generic(app, anonymous):
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    wrong = anonymous.post("/api/auth/login", {"username": "admin", "password": "incorrecta-123"})
    unknown = anonymous.post("/api/auth/login", {"username": "fantasma", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.parsed["error"]["message"] == unknown.parsed["error"]["message"] == "Usuario o contraseña incorrectos."


def test_login_is_throttled_after_repeated_failures(app, anonymous):
    app.extensions["sigma"].create_first_admin.execute("admin", "", PASSWORD)
    for _ in range(5):
        assert anonymous.post("/api/auth/login", {"username": "admin", "password": "mala-clave-123"}).status_code == 401
    blocked = anonymous.post("/api/auth/login", {"username": "admin", "password": PASSWORD})  # even the right one
    assert blocked.status_code == 429 and int(blocked.headers["Retry-After"]) > 0


def test_logout_ends_the_session(api):
    assert api.get("/api/datasets").status_code == 200
    assert api.post("/api/auth/logout").status_code == 200
    assert api.get("/api/datasets").status_code == 401


def test_login_is_case_insensitive_on_username(app, api):
    client = ApiClient(app.test_client())
    assert client.login("ADMIN").status_code == 200


def test_disabled_accounts_cannot_sign_in(app, api, make_client):
    other = make_client("luis")
    luis_id = other.get("/api/auth/me").parsed["user"]["id"]
    assert api.patch(f"/api/users/{luis_id}", {"is_active": False}).status_code == 200
    assert other.get("/api/datasets").status_code == 401  # the open session dies with the account
    assert ApiClient(app.test_client()).login("luis").status_code == 401


# ------------------------------------------------------------------ sessions are tied to the account epoch
def test_changing_the_password_closes_the_other_sessions_only(app, api):
    second = ApiClient(app.test_client())
    assert second.login("admin").status_code == 200
    changed = api.post(
        "/api/auth/change-password", {"current_password": PASSWORD, "new_password": "Otra-clave-larga-9"}
    )
    assert changed.status_code == 200
    assert api.get("/api/datasets").status_code == 200  # the session that changed it survives
    assert second.get("/api/datasets").status_code == 401  # every other session is closed
    assert ApiClient(app.test_client()).login("admin", PASSWORD).status_code == 401
    assert ApiClient(app.test_client()).login("admin", "Otra-clave-larga-9").status_code == 200


def test_change_password_checks_the_current_one_and_the_policy(api):
    wrong = api.post(
        "/api/auth/change-password", {"current_password": "nope-nope-123", "new_password": "Nueva-clave-larga-9"}
    )
    assert wrong.status_code == 422 and "actual" in wrong.parsed["error"]["message"]
    same = api.post("/api/auth/change-password", {"current_password": PASSWORD, "new_password": PASSWORD})
    assert same.status_code == 422 and "distinta" in same.parsed["error"]["message"]
    weak = api.post("/api/auth/change-password", {"current_password": PASSWORD, "new_password": "123"})
    assert weak.status_code == 422


def test_admin_password_reset_signs_the_user_out(app, api, make_client):
    other = make_client("luis")
    luis = other.get("/api/auth/me").parsed["user"]["id"]
    assert api.post(f"/api/users/{luis}/reset-password", {"new_password": "Reiniciada-clave-77"}).status_code == 200
    assert other.get("/api/datasets").status_code == 401
    assert ApiClient(app.test_client()).login("luis", "Reiniciada-clave-77").status_code == 200


def test_profile_update(api):
    assert (
        api.patch("/api/auth/profile", {"display_name": "  Nuevo   Nombre "}).parsed["user"]["display_name"]
        == "Nuevo Nombre"
    )
    assert api.patch("/api/auth/profile", {}).status_code == 422


# ------------------------------------------------------------------ administration
def test_only_administrators_can_manage_accounts(api, make_client):
    regular = make_client("luis")
    for call in (
        lambda: regular.get("/api/users"),
        lambda: regular.post("/api/users", {"username": "x1x", "password": PASSWORD}),
        lambda: regular.patch("/api/users/1", {"is_active": False}),
        lambda: regular.delete("/api/users/1"),
    ):
        assert call().status_code == 403
    forbidden = regular.raw.get("/usuarios")
    assert forbidden.status_code == 403 and "<title>Error 403 · Sigma</title>" in forbidden.get_data(as_text=True)
    assert api.raw.get("/usuarios").status_code == 200
    assert api.get("/api/users").status_code == 200


def test_create_update_and_list_accounts(api):
    created = api.post(
        "/api/users", {"username": "marta", "display_name": "Marta", "password": PASSWORD, "can_use_console": True}
    )
    assert created.status_code == 201 and created.parsed["user"]["can_use_console"] is True
    assert api.post("/api/users", {"username": "MARTA", "password": PASSWORD}).status_code == 422  # duplicate
    marta = created.parsed["user"]["id"]
    updated = api.patch(f"/api/users/{marta}", {"role": "admin", "can_use_ai": True}).parsed["user"]
    assert updated["role"] == "admin" and updated["can_use_ai"] is True
    assert {u["username"] for u in api.get("/api/users").parsed["users"]} == {"admin", "marta"}
    assert "password" not in api.get("/api/users").get_data(as_text=True).lower().replace("password_hash", "")


def test_the_last_active_administrator_is_protected(api, make_client):
    me = api.get("/api/auth/me").parsed["user"]["id"]
    assert api.patch(f"/api/users/{me}", {"role": "user"}).status_code == 422
    assert api.patch(f"/api/users/{me}", {"is_active": False}).status_code == 422
    assert api.delete(f"/api/users/{me}").status_code == 422  # nobody deletes themselves
    second = make_client("jefe", role="admin")
    assert api.patch(f"/api/users/{me}", {"is_active": False}).status_code == 200  # now another admin exists
    jefe = second.get("/api/auth/me").parsed["user"]["id"]
    assert second.patch(f"/api/users/{jefe}", {"role": "user"}).status_code == 422  # …and now the last one is protected


def test_deleting_an_account_removes_its_data_and_kernels(app, api, make_client, sales_frame):
    from tests.integration.conftest import import_frame

    victim = make_client("luis", console=True)
    luis = victim.get("/api/auth/me").parsed["user"]["id"]
    dataset = import_frame(victim, sales_frame, "ventas")
    notebook = victim.post("/api/notebooks", {"name": "mio"}).parsed["notebook"]
    folder = app.extensions["sigma"].config.notebooks_dir / str(luis)
    assert folder.is_dir() and (folder / f"{notebook['id']}.ipynb").is_file()
    assert api.delete(f"/api/users/{luis}").status_code == 200
    assert not folder.exists()
    with app.extensions["sigma"].database.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM datasets WHERE id = ?", (dataset["id"],)).fetchone()[0] == 0
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'ds_%'")]
    assert f"ds_{dataset['id']}" not in tables
    assert victim.get("/api/datasets").status_code == 401


# ------------------------------------------------------------------ permissions
def test_console_needs_an_explicit_permission(api, make_client):
    without = make_client("luis")
    assert without.get("/api/notebooks").status_code == 403
    assert without.post("/api/notebooks", {"name": "x"}).status_code == 403
    assert without.raw.get("/consola").status_code == 403
    html = without.raw.get("/").get_data(as_text=True)
    assert 'href="/consola"' not in html  # not even offered in the menu
    with_permission = make_client("marta", console=True)
    assert with_permission.get("/api/notebooks").status_code == 200
    assert with_permission.raw.get("/consola").status_code == 200
    assert 'href="/consola"' in with_permission.raw.get("/").get_data(as_text=True)


def test_menu_shows_the_account_block_and_admin_link(api, make_client):
    html = api.raw.get("/").get_data(as_text=True)
    assert "Administrador" in html and 'href="/usuarios"' in html and 'href="/cuenta"' in html
    regular_html = make_client("luis").raw.get("/").get_data(as_text=True)
    assert 'href="/usuarios"' not in regular_html and "Cerrar sesión" in regular_html


def test_responses_are_never_cached(api):
    assert "no-store" in api.raw.get("/regresion").headers["Cache-Control"]
    assert "no-store" in api.get("/api/datasets").headers["Cache-Control"]
