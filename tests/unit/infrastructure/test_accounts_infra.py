import sqlite3

import pandas as pd
import pytest

from sigma.domain.errors import DatasetNotFoundError, InvalidInputError, RateLimitedError, UserNotFoundError
from sigma.infrastructure.persistence.sqlite.database import Database
from sigma.infrastructure.persistence.sqlite.dataset_repository import SqliteDatasetRepository
from sigma.infrastructure.persistence.sqlite.migrations import MIGRATIONS, V1_BASE
from sigma.infrastructure.persistence.sqlite.user_repository import SqliteUserRepository
from sigma.infrastructure.persistence.user_data import SqliteLegacyDataAdopter
from sigma.infrastructure.security.login_throttle import InMemoryLoginThrottle
from sigma.infrastructure.security.password_hasher import WerkzeugPasswordHasher


# ------------------------------------------------------------------------------ migration v1 → v2
@pytest.fixture
def legacy_db(tmp_path):
    """A database as the single-user version left it: schema v1, one dataset, one saved result."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(V1_BASE + "PRAGMA user_version = 1;")
    conn.execute('CREATE TABLE "ds_' + "a" * 32 + '" (x INTEGER)')
    conn.execute('INSERT INTO "ds_' + "a" * 32 + '" VALUES (1), (2)')
    conn.execute(
        "INSERT INTO datasets VALUES (?, 'ventas', 'csv', '{}', 2, 1, '{\"x\": \"int64\"}',"
        ' \'[{"name":"x","kind":"numeric","pandas_dtype":"int64","n_missing":0,"n_unique":2}]\', \'ayer\')',
        ("a" * 32,),
    )
    conn.execute(
        "INSERT INTO saved_results VALUES"
        " ('r1', 'anova', 't', 'Mi análisis', '', ?, 'ventas', '{}', '{}', '{}', '[]', 'ayer', 'ayer')",
        ("a" * 32,),
    )
    conn.commit()
    conn.close()
    return path


def test_migration_keeps_legacy_rows_unowned_until_adopted(legacy_db, tmp_path):
    database = Database(legacy_db)
    with database.connection() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
        assert conn.execute("SELECT user_id FROM datasets").fetchone()[0] is None
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    owner = SqliteUserRepository(database).add("ana", "Ana", "x", "admin", True, True).id
    assert SqliteDatasetRepository(database, owner).list() == []  # invisible to everyone until adopted
    adopted = SqliteLegacyDataAdopter(database, tmp_path / "nb").adopt(owner)
    assert adopted["datasets"] == 1 and adopted["results"] == 1
    repo = SqliteDatasetRepository(database, owner)
    assert [d.name for d in repo.list()] == ["ventas"] and repo.get(repo.list()[0].id).data["x"].tolist() == [1, 2]


def test_adoption_moves_legacy_notebooks_into_the_users_folder(legacy_db, tmp_path):
    notebooks = tmp_path / "nb"
    notebooks.mkdir()
    (notebooks / ("b" * 32 + ".ipynb")).write_text("{}", encoding="utf-8")
    database = Database(legacy_db)
    owner = SqliteUserRepository(database).add("ana", "Ana", "x", "admin", True, True).id
    assert SqliteLegacyDataAdopter(database, notebooks).adopt(owner)["notebooks"] == 1
    assert (notebooks / str(owner) / ("b" * 32 + ".ipynb")).is_file() and not list(notebooks.glob("*.ipynb"))


def test_dataset_names_are_unique_per_user_not_globally(tmp_path):
    database = Database(tmp_path / "s.db")
    users = SqliteUserRepository(database)
    a, b = (users.add(name, name, "x", "user", False, False).id for name in ("ana", "luis"))
    frame = pd.DataFrame({"x": [1]})
    SqliteDatasetRepository(database, a).add("ventas", frame, "csv", {})
    SqliteDatasetRepository(database, b).add("VENTAS", frame, "csv", {})  # same name, other owner: allowed
    with pytest.raises(InvalidInputError):
        SqliteDatasetRepository(database, a).add("ventas", frame, "csv", {})


def test_a_foreign_dataset_id_behaves_like_a_missing_one(tmp_path):
    database = Database(tmp_path / "s.db")
    users = SqliteUserRepository(database)
    a, b = (users.add(name, name, "x", "user", False, False).id for name in ("ana", "luis"))
    info = SqliteDatasetRepository(database, a).add("privado", pd.DataFrame({"x": [1]}), "csv", {})
    intruder = SqliteDatasetRepository(database, b)
    for call in (
        lambda: intruder.get_info(info.id),
        lambda: intruder.get(info.id),
        lambda: intruder.preview(info.id, 5, 0),
        lambda: intruder.rename(info.id, "robado"),
        lambda: intruder.delete(info.id),
        lambda: intruder.get_by_name("privado"),
        lambda: intruder.replace_data(info.id, pd.DataFrame({"x": [9]})),
    ):
        with pytest.raises(DatasetNotFoundError):
            call()
    assert intruder.list() == []
    assert SqliteDatasetRepository(database, a).get(info.id).data["x"].tolist() == [1]


def test_sync_revisions_are_bumped_by_triggers(tmp_path):
    database = Database(tmp_path / "s.db")
    owner = SqliteUserRepository(database).add("ana", "Ana", "x", "user", False, False).id
    repo = SqliteDatasetRepository(database, owner)

    def revision():
        with database.connection() as conn:
            row = conn.execute(
                "SELECT revision FROM sync_state WHERE user_id = ? AND scope = 'datasets'", (owner,)
            ).fetchone()
        return row[0] if row else 0

    assert revision() == 0
    info = repo.add("uno", pd.DataFrame({"x": [1]}), "csv", {})
    after_add = revision()
    repo.rename(info.id, "dos")
    after_rename = revision()
    repo.delete(info.id)
    assert 0 < after_add < after_rename < revision()


def test_deleting_a_user_cascades_to_their_rows(tmp_path):
    database = Database(tmp_path / "s.db")
    users = SqliteUserRepository(database)
    owner = users.add("ana", "Ana", "x", "user", False, False).id
    SqliteDatasetRepository(database, owner).add("uno", pd.DataFrame({"x": [1]}), "csv", {})
    users.delete(owner)
    with database.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM datasets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM sync_state").fetchone()[0] == 0


# ------------------------------------------------------------------------------ users repository
def test_user_repository_crud_and_epoch(tmp_path):
    users = SqliteUserRepository(Database(tmp_path / "u.db"))
    created = users.add("Ana", "Ana", "hash1", "admin", True, True)
    assert users.find_with_hash("ana")[1] == "hash1"  # case-insensitive login
    with pytest.raises(InvalidInputError):
        users.add("ANA", "Otra", "h", "user", False, False)
    assert users.get(created.id).auth_epoch == 1
    assert users.set_password(created.id, "hash2").auth_epoch == 2
    assert users.update(created.id, is_active=False).auth_epoch == 3  # disabling invalidates sessions
    assert users.update(created.id, display_name="Nueva").auth_epoch == 3  # a cosmetic edit does not
    assert users.count() == 1 and users.count_active_admins() == 0 and users.count_active_admins(excluding=99) == 0
    users.delete(created.id)
    with pytest.raises(UserNotFoundError):
        users.get(created.id)
    assert users.find_with_hash("ana") is None


# ------------------------------------------------------------------------------ hashing and throttling
def test_password_hasher_roundtrip_and_salting():
    hasher = WerkzeugPasswordHasher()
    first, second = hasher.hash("clave-1234567"), hasher.hash("clave-1234567")
    assert first != second and first.startswith("scrypt:")
    assert hasher.verify(first, "clave-1234567") and not hasher.verify(first, "otra")
    assert not hasher.verify("not-a-hash", "x") and not hasher.verify("", "x")
    assert not hasher.verify(hasher.dummy_hash, "clave-1234567")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_throttle_locks_after_repeated_failures_and_recovers():
    clock = FakeClock()
    throttle = InMemoryLoginThrottle(max_failures=3, lockout_seconds=60, clock=clock)
    for _ in range(3):
        throttle.check("ana", "1.1.1.1")
        throttle.record_failure("ana", "1.1.1.1")
    with pytest.raises(RateLimitedError) as locked:
        throttle.check("ANA", "1.1.1.1")
    assert 1 <= locked.value.details["retry_after"] <= 61
    throttle.check("ana", "2.2.2.2")  # another IP is unaffected
    throttle.check("luis", "1.1.1.1")  # another user is unaffected
    clock.now += 61
    throttle.check("ana", "1.1.1.1")


def test_success_resets_the_counter_and_old_failures_expire():
    clock = FakeClock()
    throttle = InMemoryLoginThrottle(max_failures=3, window_seconds=100, clock=clock)
    throttle.record_failure("ana", "ip")
    throttle.record_failure("ana", "ip")
    throttle.reset("ana", "ip")
    throttle.record_failure("ana", "ip")
    throttle.check("ana", "ip")  # 1 failure since the reset
    clock.now += 101
    throttle.record_failure("ana", "ip")
    throttle.record_failure("ana", "ip")
    throttle.check("ana", "ip")  # the earlier failures fell out of the window


def test_an_ip_hammering_many_usernames_gets_blocked():
    throttle = InMemoryLoginThrottle(max_failures=5, max_ip_failures=6, clock=FakeClock())
    for i in range(6):
        throttle.record_failure(f"user{i}", "9.9.9.9")
    with pytest.raises(RateLimitedError):
        throttle.check("someone-new", "9.9.9.9")
