"""The SQL boundary, attacked on purpose.

This is the most important test file in the project. Everything else can be
wrong and the system degrades; if this is wrong, the agent can drop the
warehouse it is supposed to be diagnosing.

The suite is written as two parametrised tables:

* **safe** — realistic investigative queries that must be *allowed*. A guard
  that blocks real work gets disabled by the first person it inconveniences, so
  usability is part of the security requirement, not a nicety.
* **attacks** — statements that must be rejected, including the multi-statement
  bypass the reference implementation is vulnerable to, a data-modifying CTE,
  and `SELECT ... INTO`.
"""
from __future__ import annotations

import pytest

from app.tools.sql_guard import (
    ALLOWED_SCHEMAS,
    UnsafeQuery,
    assert_read_only,
    is_read_only,
    missing_guard_nodes,
    referenced_tables,
)

SAFE_QUERIES = [
    "SELECT count(*) FROM warehouse.claims WHERE claim_date = '2026-09-20'",
    "WITH x AS (SELECT * FROM warehouse.claims) SELECT count(*) FROM x",
    (
        "SELECT c.claim_id FROM warehouse.claims c "
        "JOIN warehouse.member_eligibility m ON m.member_id = c.member_id"
    ),
    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'warehouse'",
    "SELECT claim_date, count(*) FROM warehouse.claims GROUP BY 1 ORDER BY 1 DESC LIMIT 10",
    "SELECT a FROM warehouse.claims UNION SELECT b FROM warehouse.policy_dim",
    "SELECT 1",
    "SELECT * FROM warehouse.claims",  # unqualified schema is fine: search_path handles it
]

ATTACKS = [
    pytest.param("DROP TABLE warehouse.claims", id="drop"),
    pytest.param("select 1; drop table warehouse.claims", id="multi-statement-drop"),
    pytest.param("DELETE FROM warehouse.claims", id="delete"),
    pytest.param("INSERT INTO warehouse.claims VALUES ('x')", id="insert"),
    pytest.param("UPDATE warehouse.claims SET claim_amount = 0", id="update"),
    pytest.param("TRUNCATE warehouse.claims", id="truncate"),
    pytest.param("CREATE TABLE warehouse.evil (x int)", id="create"),
    pytest.param("ALTER TABLE warehouse.claims ADD COLUMN evil int", id="alter"),
    pytest.param(
        "WITH deleted AS (DELETE FROM warehouse.claims RETURNING *) SELECT * FROM deleted",
        id="data-modifying-cte",
    ),
    pytest.param("SELECT * INTO warehouse.stolen FROM warehouse.claims", id="select-into"),
    pytest.param("COPY warehouse.claims TO '/tmp/exfil.csv'", id="copy-to-file"),
    pytest.param("SELECT * FROM platform.incidents", id="other-schema"),
    pytest.param("SELECT * FROM pg_catalog.pg_shadow", id="pg-catalog"),
    pytest.param("ATTACH DATABASE '/tmp/evil.db' AS evil", id="attach"),
    pytest.param("GRANT ALL ON warehouse.claims TO PUBLIC", id="grant"),
    pytest.param("VACUUM", id="vacuum"),
    pytest.param("SET search_path = public", id="set"),
    pytest.param("select 1; select 2", id="two-selects"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
]


@pytest.mark.parametrize("sql", SAFE_QUERIES)
def test_investigative_queries_are_allowed(sql: str) -> None:
    assert_read_only(sql)  # must not raise
    assert is_read_only(sql)


@pytest.mark.parametrize("sql", ATTACKS)
def test_attacks_are_rejected(sql: str) -> None:
    with pytest.raises(UnsafeQuery):
        assert_read_only(sql)
    assert not is_read_only(sql)


def test_multi_statement_rejection_explains_itself() -> None:
    """The reason is fed back to the agent, so it has to be actionable."""
    with pytest.raises(UnsafeQuery) as excinfo:
        assert_read_only("select 1; drop table warehouse.claims")
    message = str(excinfo.value)
    assert "one statement" in message
    assert "smuggle" in message or "Multi-statement" in message


def test_data_modifying_cte_names_the_operation() -> None:
    with pytest.raises(UnsafeQuery) as excinfo:
        assert_read_only("WITH x AS (DELETE FROM warehouse.claims RETURNING *) SELECT * FROM x")
    assert "DELETE" in str(excinfo.value)


def test_schema_allow_list_is_enforced_not_sanitised() -> None:
    assert "warehouse" in ALLOWED_SCHEMAS
    assert "platform" not in ALLOWED_SCHEMAS
    with pytest.raises(UnsafeQuery):
        assert_read_only("SELECT * FROM platform.reports")


def test_referenced_tables_are_reported() -> None:
    tables = referenced_tables(
        "SELECT * FROM warehouse.claims c JOIN warehouse.member_eligibility m ON true"
    )
    assert tables == ["claims", "member_eligibility"]


def test_referenced_tables_is_empty_for_a_rejected_query() -> None:
    assert referenced_tables("DROP TABLE warehouse.claims") == []


def test_guard_is_not_silently_degraded_by_a_dependency_bump() -> None:
    """A renamed sqlglot node would silently remove a protection.

    This test documents which names the installed version does not provide, so
    an upgrade that drops one fails here rather than in production. The two
    currently missing (``AlterTable``, ``Call``) are covered by ``Alter`` and
    ``Command`` respectively.
    """
    missing = set(missing_guard_nodes())
    assert missing <= {"AlterTable", "Call"}, (
        f"sqlglot no longer provides {missing}; the forbidden-node guard has a hole"
    )
    # Whatever the version, these must exist.
    for required in ("Insert", "Update", "Delete", "Drop", "Copy", "Command"):
        assert required not in missing, f"{required} is required for the guard to work"
