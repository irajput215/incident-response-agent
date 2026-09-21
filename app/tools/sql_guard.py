"""A read-only SQL boundary that is a boundary, not a prefix check.

The naive version of this guard — ``if not sql.strip().lower().startswith("select")``
— is bypassable in one line: ``select 1; drop table victim`` starts with
"select" and the ``DROP`` still executes. This project does not ship that.

Three layers, because each catches something the others do not:

**1. Parse the statement (sqlglot).** Exactly one statement, and its root must be
a read-only shape. If sqlglot cannot parse it into a known node it becomes
``exp.Command``, which is rejected — so unknown syntax fails closed rather than
sneaking through.

**2. Reject dangerous nodes *anywhere* in the tree.** This is the layer a
root-only check misses. PostgreSQL supports **data-modifying CTEs**:

    WITH deleted AS (DELETE FROM warehouse.claims RETURNING *) SELECT * FROM deleted

The root node is a ``SELECT``. The statement deletes your data. Walking the
tree for ``Insert``/``Update``/``Delete``/``Copy``/``Command`` is what stops it.

**3. Constrain which schemas may be read.** Every referenced table must be
schema-qualified into the allow-list, or unqualified (which resolves through the
session's ``search_path``, set to ``warehouse``). ``platform.*`` and
``pg_catalog.*`` are therefore not reachable from a query.

Layer 4 is not in this file but matters just as much: the query runs on a
connection opened with ``default_transaction_read_only=on``, so the *engine*
refuses writes even if all three checks above are somehow defeated. See
:mod:`app.tools.database`.

**What this does not defend against** — stated plainly, because a guard whose
limits are undocumented will be trusted beyond them:

* A read-only role can still read any row it can see. There is no row-level or
  column-level policy here; if the warehouse holds PII, this tool can return it.
* ``statement_timeout`` bounds runtime but not the cost of a query that has
  already started scanning.
* Resource exhaustion via a pathological query is mitigated, not prevented.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

# Where the agent is allowed to look. `information_schema` is included so the
# schema tools can introspect; `warehouse` is the simulated estate.
ALLOWED_SCHEMAS: frozenset[str] = frozenset({"warehouse", "information_schema"})

# The only statement roots we accept.
_READ_ONLY_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select,
    exp.Union,
    exp.Except,
    exp.Intersect,
    exp.Subquery,
)

# Node types that must never appear anywhere in the tree. `exp.Command` is the
# catch-all sqlglot uses for statements it does not model (COPY, SET, ATTACH,
# VACUUM, ...), so including it makes unrecognised syntax fail closed.
#
# Built by name rather than directly, because sqlglot renames and removes node
# classes between major versions (30.x has no `AlterTable` or `Call`). A guard
# that raises AttributeError on import is worse than one that is slightly less
# specific, so missing names are reported rather than fatal.
_FORBIDDEN_NAMES: tuple[str, ...] = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "AlterTable",
    "Command",
    "Copy",
    "Merge",
    "TruncateTable",
    "Grant",
    "Set",
    "Use",
    "Attach",
    "Detach",
    "Call",
    "Transaction",
    "Commit",
    "Rollback",
    "Into",  # `SELECT ... INTO new_table` is a write wearing a SELECT costume
)


def _forbidden_nodes() -> tuple[type[exp.Expression], ...]:
    found = []
    for name in _FORBIDDEN_NAMES:
        node = getattr(exp, name, None)
        if isinstance(node, type) and issubclass(node, exp.Expression):
            found.append(node)
    return tuple(found)


_FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = _forbidden_nodes()


def missing_guard_nodes() -> list[str]:
    """Names this sqlglot version does not provide.

    Exposed so a test can assert the guard is not silently degraded after a
    dependency bump: if sqlglot renames a node, the protection for it vanishes
    quietly, and a silent hole in a security boundary is the worst kind.
    """
    return [name for name in _FORBIDDEN_NAMES if not hasattr(exp, name)]


class UnsafeQuery(ValueError):
    """The statement is not a plain read. Carries a reason for the agent."""


def assert_read_only(
    sql: str,
    *,
    allowed_schemas: frozenset[str] = ALLOWED_SCHEMAS,
    dialect: str = "postgres",
) -> exp.Expression:
    """Validate ``sql`` as a single, read-only, allow-listed query.

    Returns the parsed expression so callers do not pay to parse twice. Raises
    :class:`UnsafeQuery` with a human-readable reason, which matters: the reason
    is fed back to the agent, so a *useful* rejection lets it correct itself
    while a terse one just makes it retry the same thing.
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeQuery("empty query")

    try:
        statements = sqlglot.parse(text, read=dialect)
    except Exception as exc:  # sqlglot raises a variety of parse errors
        raise UnsafeQuery(f"could not parse SQL: {exc}") from exc

    statements = [s for s in statements if s is not None]
    if not statements:
        raise UnsafeQuery("empty query")
    if len(statements) > 1:
        # This is the exact bypass the naive guard allows.
        raise UnsafeQuery(
            f"exactly one statement is allowed, got {len(statements)}. "
            "Multi-statement SQL is rejected because it can smuggle a write "
            "past a read-only-looking prefix."
        )

    statement = statements[0]

    if not isinstance(statement, _READ_ONLY_ROOTS):
        raise UnsafeQuery(
            f"only read-only queries are allowed; got {type(statement).__name__.upper()}"
        )

    # Walk *every* node, not just the root: a data-modifying CTE has a SELECT root.
    for node in statement.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeQuery(
                f"statement contains a disallowed {type(node).__name__.upper()} operation"
            )

    for table in statement.find_all(exp.Table):
        schema = table.db
        if schema and schema.lower() not in allowed_schemas:
            raise UnsafeQuery(
                f"table {table.sql()} is outside the readable schemas "
                f"({', '.join(sorted(allowed_schemas))})"
            )

    return statement


def referenced_tables(sql: str, *, dialect: str = "postgres") -> list[str]:
    """Names of the tables a query touches, for logging and evaluation."""
    try:
        statement = assert_read_only(sql, dialect=dialect)
    except UnsafeQuery:
        return []
    return sorted({t.name for t in statement.find_all(exp.Table) if t.name})


def is_read_only(sql: str, *, dialect: str = "postgres") -> bool:
    """Convenience boolean wrapper, for tests and the API's error messages."""
    try:
        assert_read_only(sql, dialect=dialect)
    except UnsafeQuery:
        return False
    return True
