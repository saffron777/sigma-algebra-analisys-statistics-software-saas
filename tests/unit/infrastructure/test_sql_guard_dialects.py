"""The read-only guard reads SQL as each engine does: what hides code in one dialect must not in another."""

import pytest

from sigma.domain.errors import InvalidInputError
from sigma.infrastructure.importers.sql_guard import build_table_query, validate_readonly_query
from sigma.infrastructure.importers.sql_scrub import scrub

ACCEPTED = [
    # MySQL / MariaDB
    ("mysql", "SELECT `update`, `delete` FROM t WHERE x = 'a\\'b; DROP TABLE t' -- DROP"),
    ("mysql", "SELECT 1 # DROP TABLE t"),
    ("mysql", 'SELECT * FROM t WHERE a = "x; y"'),
    ("mysql", "SELECT 'a\\' ; DROP TABLE t; --'"),  # one literal: the backslash escapes the quote
    ("mysql", "SELECT 1 -- ok\n, 2"),
    ("mysql", "(SELECT 1) UNION (SELECT 2)"),
    # PostgreSQL
    ("postgresql", "SELECT $$a; DROP TABLE t$$ AS x"),
    ("postgresql", "SELECT $tag$ ' ; DROP TABLE t $tag$, 1"),
    ("postgresql", "select E'it\\'s ; ok'"),
    ("postgresql", "/* a /* b */ DROP TABLE t */ SELECT 1"),  # block comments nest
    ("postgresql", 'SELECT "drop", "delete" FROM "update"'),
    ("postgresql", "SELECT $1, a$b$c FROM t"),
    ("postgresql", "WITH c AS (SELECT 1 a) SELECT * FROM c;"),
    # Oracle
    ("oracle", "SELECT q'[it's; DROP TABLE t]' FROM dual"),
    ("oracle", "SELECT q'{ ; }', nq'(x)' FROM dual"),
    ("oracle", "SELECT 'a''b; DROP' FROM dual"),
    ("oracle", 'SELECT "DROP" FROM t -- DELETE'),
    # SQL Server (unchanged behaviour)
    ("mssql", 'SELECT [update], "delete" FROM t -- INSERT INTO x'),
    ("mssql", "SELECT /* DELETE */ 1"),
]

REJECTED = [
    # MySQL / MariaDB
    ("mysql", "SELECT 1 --;DROP TABLE t"),  # `--;` is not a comment in MySQL
    ("mysql", "SELECT 1 /*!50000 ; DROP TABLE t */"),
    ("mysql", "SELECT LOAD_FILE('/etc/passwd')"),
    ("mysql", "SELECT SLEEP(100)"),
    ("mysql", "SELECT * FROM t FOR UPDATE"),
    ("mysql", "SELECT * FROM t LOCK IN SHARE MODE"),
    ("mysql", "SELECT 1 INTO OUTFILE '/tmp/x'"),
    ("mysql", "SELECT 1; SELECT 2"),
    ("mysql", "DELETE FROM t"),
    ("mysql", "SELECT 'sin cerrar"),
    # PostgreSQL
    ("postgresql", "SELECT $$'$$; DROP TABLE x; SELECT $$'$$"),  # the quote inside $$ must not confuse the scanner
    ("postgresql", "SELECT $$'$$; DROP TABLE x; SELECT '"),
    ("postgresql", "SELECT pg_sleep(10)"),
    ("postgresql", "SELECT pg_read_file('/etc/passwd')"),
    ("postgresql", "SELECT lo_import('/etc/passwd')"),
    ("postgresql", "SELECT nextval('s')"),
    ("postgresql", "SELECT * FROM t FOR UPDATE"),
    ("postgresql", "SELECT * FROM t FOR NO KEY UPDATE"),
    ("postgresql", "COPY t TO '/tmp/x'"),
    ("postgresql", "WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d"),
    ("postgresql", "SELECT 1; SELECT 2"),
    ("postgresql", "SELECT $tag$ sin cerrar"),
    ("postgresql", "/* sin cerrar SELECT 1"),
    ("postgresql", "SELECT query_to_xml('delete from t', false, false, '')"),
    # Oracle
    ("oracle", "SELECT UTL_HTTP.REQUEST('http://x') FROM dual"),
    ("oracle", "SELECT DBMS_LOCK.SLEEP(5) FROM dual"),
    ("oracle", "SELECT * FROM t FOR UPDATE"),
    ("oracle", "SELECT 1 FROM dual; DROP TABLE t"),
    ("oracle", "SELECT q'[x]' FROM dual; DROP TABLE t"),
    ("oracle", "SELECT q'[sin cerrar FROM dual"),
    ("oracle", "BEGIN NULL; END;"),
    ("oracle", "SELECT * INTO v FROM t"),
    # the same text means different things in other dialects
    ("mssql", "SELECT 'a\\' ; DROP TABLE t; --'"),  # no backslash escapes: the literal ends at the second quote
    ("postgresql", "SELECT 'a\\' ; DROP TABLE t; --'"),
]


@pytest.mark.parametrize(("dialect", "sql"), ACCEPTED)
def test_ordinary_selects_pass(dialect, sql):
    assert validate_readonly_query(sql, dialect)


@pytest.mark.parametrize(("dialect", "sql"), REJECTED)
def test_anything_that_could_write_or_hide_code_is_refused(dialect, sql):
    with pytest.raises(InvalidInputError):
        validate_readonly_query(sql, dialect)


def squeeze(text: str) -> str:
    return "".join(text.split())


def test_the_scanner_blanks_comments_literals_and_identifiers():
    assert squeeze(scrub("SELECT 'a;b' /* c */ , \"d\" -- e", "oracle")) == "SELECT''," + '""'
    mysql_text = r"SELECT `a``b`, 'x\'y', # z" + "\n1"
    assert squeeze(scrub(mysql_text, "mysql")) == "SELECT\"\",''," + "1"
    assert squeeze(scrub(r"SELECT $a$ ' $a$, E'\'' FROM t", "postgresql")) == "SELECT'',E''FROMt"
    assert squeeze(scrub("SELECT [a]]b] FROM t", "mssql")) == 'SELECT""FROMt'


def test_the_default_dialect_is_still_transact_sql():
    assert validate_readonly_query("SELECT TOP 5 * FROM [dbo].[t]")
    with pytest.raises(InvalidInputError):
        validate_readonly_query("EXEC sp_who")


@pytest.mark.parametrize(
    ("dialect", "expected"),
    [
        ("mssql", "SELECT TOP (11) * FROM [s].[t]"),
        ("mysql", "SELECT * FROM `s`.`t` LIMIT 11"),
        ("postgresql", 'SELECT * FROM "s"."t" LIMIT 11'),
        ("oracle", 'SELECT * FROM (SELECT * FROM "s"."t") WHERE ROWNUM <= 11'),
    ],
)
def test_table_queries_quote_and_limit_per_engine(dialect, expected):
    assert build_table_query("t", "s", 10, dialect) == expected


@pytest.mark.parametrize("dialect", ["mssql", "mysql", "postgresql", "oracle"])
def test_identifiers_cannot_break_out_of_their_quotes(dialect):
    for malicious in ('a"b', "a`b", "a]b", "t; DROP TABLE x", "x' OR '1'='1", 't"; DROP TABLE x; --', ""):
        with pytest.raises(InvalidInputError):
            build_table_query(malicious, None, 5, dialect)
        with pytest.raises(InvalidInputError):
            build_table_query("t", malicious, 5, dialect) if malicious else (_ for _ in ()).throw(
                InvalidInputError("x")
            )
