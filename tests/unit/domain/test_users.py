import pytest

from sigma.domain.errors import InvalidInputError
from sigma.domain.users import validate_display_name, validate_password, validate_role, validate_username


@pytest.mark.parametrize("name", ["ana", "Luis.Perez", "user_01", "a@b.co", "x" * 40, "9abc"])
def test_valid_usernames(name):
    assert validate_username(f"  {name}  ") == name


@pytest.mark.parametrize("name", ["", "ab", "x" * 41, "con espacio", "_empieza", "ñandú", "a/b", "../etc"])
def test_invalid_usernames(name):
    with pytest.raises(InvalidInputError):
        validate_username(name)


def test_password_policy():
    assert validate_password("Una-clave-larga-1", "ana") == "Una-clave-larga-1"
    for bad in ("corta", "1234567890", "Password123", "aaaaaaaaaaaa", "xxxxxxxxxxxxxxxx"):
        with pytest.raises(InvalidInputError):
            validate_password(bad, "ana")
    with pytest.raises(InvalidInputError, match="nombre de usuario"):
        validate_password("la-clave-de-ANA-2024", "ana")
    with pytest.raises(InvalidInputError):
        validate_password("p" * 201, "ana")


def test_role_and_display_name():
    assert validate_role("admin") == "admin"
    with pytest.raises(InvalidInputError):
        validate_role("root")
    assert validate_display_name("  Ana   María  ", "ana") == "Ana María"
    assert validate_display_name("   ", "ana") == "ana"
