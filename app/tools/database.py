"""Data investigation tools: query the warehouse, check its shape and contents.

Every query here runs on a connection opened with
``default_transaction_read_only=on``. That is the fourth layer of the SQL
boundary and the only one the *engine* enforces — the parse guard in
:mod:`app.tools.sql_guard` can be reasoned about, this one cannot be argued
with.

Two more bounds live here, both of which matter in production and neither of
which is visible in a demo:

* ``statement_timeout`` — a runaway aggregate is killed by the server rather
  than pinning a connection until someone notices.
* a hard row cap on returned results — an agent that asks for a million rows
  gets an error, not an out-of-memory.

Query errors are re-raised as ``ValueError`` carrying the database's own
message. That is intentional: this text is what a self-repair step feeds back to
the model, and "column claim_dt does not exist" is the single most useful thing
you can tell it.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.config import Settings
from app.estate import TableInfo
from app.schemas import EvidenceSource
from app.tools.registry import Tool, ToolContext, ToolOutput, object_schema
from app.tools.sql_guard import UnsafeQuery, assert_read_only

DEFAULT_QUERY_LIMIT = 200
MAX_QUERY_LIMIT = 1000
STATEMENT_TIMEOUT_MS = 5000


def quote_ident(ident: str) -> str:
    """Quote a SQL *identifier*, escaping embedded double quotes.

    Identifiers are double-quoted and escape by doubling; values are
    single-quoted. Conflating the two is the classic injection bug, and it is
    why no value in this module is ever formatted into a string.
    """
    return '"' + str(ident).replace('"', '""') + '"'


@contextmanager
def read_only_connection(
    settings: Settings, *, statement_timeout_ms: int = STATEMENT_TIMEOUT_MS
) -> Iterator[psycopg.Connection[dict[str, Any]]]:
    """A connection the server will not let you write through.

    ``options`` is passed to libpq as startup options, so the restriction applies
    to the whole session rather than to one statement — there is no window in
    which an unwrapped query could slip through.
    """
    options = (
        "-c default_transaction_read_only=on "
        f"-c statement_timeout={int(statement_timeout_ms)} "
        "-c search_path=warehouse,information_schema"
    )
    try:
        conn = psycopg.connect(
            settings.database_url,
            row_factory=dict_row,
            options=options,
            connect_timeout=5,
        )
    except psycopg.OperationalError as exc:
        raise ValueError(f"cannot reach the warehouse: {exc}") from exc
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _jsonable(value: Any) -> Any:
    """Make a database value safe to put in JSON and in a prompt."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    return value


def _rows_jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _jsonable(v) for k, v in row.items()} for row in rows]


def _table_info(ctx: ToolContext, table: str) -> TableInfo:
    """Resolve a table name against the estate.

    An allow-list rather than string sanitisation: the agent may only ask about
    tables the estate declares, so there is no name it can invent that resolves
    to something else.
    """
    info = ctx.estate.table(table)
    if info is None:
        known = ", ".join(sorted(ctx.estate.tables)) or "none"
        raise ValueError(f"unknown table {table!r}; known tables: {known}")
    return info


def _one(cur: psycopg.Cursor[dict[str, Any]]) -> dict[str, Any]:
    """Fetch exactly one row.

    An aggregate query always returns a row, so a ``None`` here means the query
    was wrong, not that the data is missing — surfacing that as an error beats
    letting ``None["n"]`` raise an opaque TypeError four frames later.
    """
    row = cur.fetchone()
    if row is None:
        raise ValueError("internal query returned no rows")
    return row


def _columns(
    conn: psycopg.Connection[dict[str, Any]], info: TableInfo
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable
              FROM information_schema.columns
             WHERE table_schema = %s AND table_name = %s
             ORDER BY ordinal_position
            """,
            [info.schema_name, info.name],
        )
        return cur.fetchall()


# --------------------------------------------------------------------------- #
# run_sql
# --------------------------------------------------------------------------- #
def _run_sql(ctx: ToolContext, query: str, limit: int = DEFAULT_QUERY_LIMIT) -> ToolOutput:
    limit = max(1, min(int(limit), MAX_QUERY_LIMIT))

    try:
        assert_read_only(query)
    except UnsafeQuery as exc:
        # Rejected before the database sees it, and the reason is returned so
        # the agent can correct the query rather than retry blindly.
        ctx.metrics.incr("tool.run_sql.rejected")
        return ToolOutput(
            summary=f"Query rejected by the read-only guard: {exc}",
            data={"query": query, "rejected": True, "reason": str(exc), "rows": []},
        )

    # Wrapping rather than rewriting: injecting a LIMIT into arbitrary SQL
    # requires understanding its shape, and a subquery wrapper cannot be fooled
    # by ORDER BY, GROUP BY, CTEs or unions.
    wrapped = f"SELECT * FROM (\n{query.rstrip().rstrip(';')}\n) AS guarded_query LIMIT {limit}"

    with read_only_connection(ctx.settings) as conn, conn.cursor() as cur:
        try:
            cur.execute(wrapped)
            rows = cur.fetchall()
        except psycopg.Error as exc:
            # Collapsed to one line: this text is fed back to the model for
            # self-repair and logged as a structured field, and a raw newline
            # breaks both.
            raise ValueError(" ".join(f"{type(exc).__name__}: {exc}".split())) from exc

    payload = _rows_jsonable(rows)
    return ToolOutput(
        summary=f"Query returned {len(payload)} row(s)" + (" (capped)" if len(payload) >= limit else ""),
        data={"query": query, "row_count": len(payload), "rows": payload, "limit": limit},
    )


# --------------------------------------------------------------------------- #
# check_table_schema
# --------------------------------------------------------------------------- #
def _check_table_schema(ctx: ToolContext, table: str) -> ToolOutput:
    info = _table_info(ctx, table)
    with read_only_connection(ctx.settings) as conn:
        columns = _columns(conn, info)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS n FROM information_schema.tables "
                "WHERE table_schema = %s AND table_name = %s",
                [info.schema_name, info.name],
            )
            exists = bool(_one(cur)["n"])

    missing = [c for c in info.required_columns if c not in {col["column_name"] for col in columns}]
    return ToolOutput(
        summary=(
            f"{info.qualified} has {len(columns)} column(s)"
            + (f"; MISSING required column(s): {', '.join(missing)}" if missing else "")
            if exists
            else f"{info.qualified} does not exist"
        ),
        data={
            "table": info.qualified,
            "exists": exists,
            "columns": columns,
            "required_columns": list(info.required_columns),
            "missing_required_columns": missing,
            "expected_column_count": len(columns),
        },
    )


# --------------------------------------------------------------------------- #
# check_row_count
# --------------------------------------------------------------------------- #
def _check_row_count(
    ctx: ToolContext, table: str, partition: str | None = None
) -> ToolOutput:
    info = _table_info(ctx, table)
    part_col = quote_ident(info.partition_column)

    with read_only_connection(ctx.settings) as conn, conn.cursor() as cur:
        if not partition:
            cur.execute(
                f"SELECT {part_col}::text AS p, count(*) AS n "
                f"FROM {quote_ident(info.schema_name)}.{quote_ident(info.name)} "
                f"GROUP BY 1 ORDER BY 1 DESC LIMIT 10"
            )
            partitions = [
                {"partition": row["p"], "row_count": int(row["n"])} for row in cur.fetchall()
            ]
            return ToolOutput(
                summary=(
                    f"{info.qualified}: {len(partitions)} recent partition(s); "
                    f"latest {partitions[0]['partition'] if partitions else 'none'}"
                ),
                data={
                    "table": info.qualified,
                    "partition_column": info.partition_column,
                    "expected_rows_per_partition": info.expected_rows_per_partition,
                    "partitions": partitions,
                },
            )

        cur.execute(
            f"SELECT count(*) AS n FROM {quote_ident(info.schema_name)}.{quote_ident(info.name)} "
            f"WHERE {part_col}::text = %s",
            [partition],
        )
        count = int(_one(cur)["n"])

        # Compare against the immediately preceding partition: that is the
        # baseline an engineer actually uses ("yesterday had 200, today has 0").
        cur.execute(
            f"SELECT {part_col}::text AS p, count(*) AS n "
            f"FROM {quote_ident(info.schema_name)}.{quote_ident(info.name)} "
            f"WHERE {part_col}::text < %s "
            f"GROUP BY 1 ORDER BY 1 DESC LIMIT 1",
            [partition],
        )
        previous = cur.fetchone()

    comparable = int(previous["n"]) if previous else None
    previous_partition = previous["p"] if previous else None
    expected = info.expected_rows_per_partition
    is_empty = count == 0
    deviation = None if comparable in (None, 0) else round((count - comparable) / comparable, 3)

    return ToolOutput(
        summary=(
            f"{info.qualified} partition {partition}: {count} row(s) "
            f"(expected ~{expected}"
            + (f", previous partition {previous_partition} had {comparable}" if comparable is not None else "")
            + (") — PARTITION IS EMPTY" if is_empty else ")")
        ),
        data={
            "table": info.qualified,
            "partition": partition,
            "row_count": count,
            "expected_rows_per_partition": expected,
            "previous_partition": previous_partition,
            "previous_row_count": comparable,
            "deviation_vs_previous": deviation,
            "is_empty": is_empty,
        },
    )


# --------------------------------------------------------------------------- #
# check_null_rate
# --------------------------------------------------------------------------- #
def _check_null_rate(
    ctx: ToolContext, table: str, column: str, partition: str | None = None
) -> ToolOutput:
    info = _table_info(ctx, table)

    with read_only_connection(ctx.settings) as conn:
        available = {col["column_name"] for col in _columns(conn, info)}
        if column not in available:
            raise ValueError(
                f"column {column!r} does not exist on {info.qualified}; "
                f"available columns: {', '.join(sorted(available))}"
            )
        where = ""
        params: list[Any] = []
        if partition:
            where = f"WHERE {quote_ident(info.partition_column)}::text = %s"
            params = [partition]
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT count(*) AS total, "
                f"count(*) FILTER (WHERE {quote_ident(column)} IS NULL) AS nulls "
                f"FROM {quote_ident(info.schema_name)}.{quote_ident(info.name)} {where}",
                params,
            )
            row = _one(cur)

    total = int(row["total"])
    nulls = int(row["nulls"])
    rate = round(nulls / total, 4) if total else 0.0
    threshold = info.null_rate_threshold
    breaches = rate > threshold

    return ToolOutput(
        summary=(
            f"{info.qualified}.{column}"
            + (f" partition {partition}" if partition else "")
            + f": null rate {rate:.4f} ({nulls}/{total}); threshold {threshold}"
            + (" — BREACHES THRESHOLD" if breaches else " — within threshold")
        ),
        data={
            "table": info.qualified,
            "column": column,
            "partition": partition,
            "total_rows": total,
            "null_count": nulls,
            "null_rate": rate,
            "threshold": threshold,
            "breaches_threshold": breaches,
        },
    )


# --------------------------------------------------------------------------- #
# check_latest_partition
# --------------------------------------------------------------------------- #
def _check_latest_partition(ctx: ToolContext, table: str) -> ToolOutput:
    info = _table_info(ctx, table)
    part_col = quote_ident(info.partition_column)
    qualified = f"{quote_ident(info.schema_name)}.{quote_ident(info.name)}"

    with read_only_connection(ctx.settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT {part_col}::text AS p, count(*) AS n FROM {qualified} "
            f"GROUP BY 1 ORDER BY 1 DESC LIMIT 30"
        )
        partitions = [(row["p"], int(row["n"])) for row in cur.fetchall()]
        cur.execute(f"SELECT max({part_col})::text AS latest FROM {qualified}")
        latest = _one(cur)["latest"]

    # Gap detection. A missing day in an otherwise daily series separates
    # "yesterday's partition is late" from "the schedule changed", and it is
    # invisible if you only look at the latest value.
    #
    # The step is assumed to be one day rather than inferred from two
    # observations: inferring it from the first two partitions produces silent
    # nonsense on irregular data, and the estate's partitions are daily.
    parsed: list[date] = []
    for name, _ in partitions:
        try:
            parsed.append(date.fromisoformat(str(name)))
        except (TypeError, ValueError):
            continue

    present = set(parsed)
    gaps: list[str] = []
    if len(present) > 1:
        span_start, span_end = min(present), max(present)
        day = span_start
        while day < span_end:
            day += timedelta(days=1)
            if day not in present:
                gaps.append(day.isoformat())
                if len(gaps) >= 30:  # bound the loop on pathological input
                    break

    return ToolOutput(
        summary=(
            f"{info.qualified}: latest partition {latest or 'none'}"
            + (f"; {len(gaps)} gap(s) in the partition series: {', '.join(gaps[:5])}" if gaps else "")
        ),
        data={
            "table": info.qualified,
            "partition_column": info.partition_column,
            "latest_partition": latest,
            "recent_partitions": [{"partition": p, "row_count": n} for p, n in partitions],
            "gaps": gaps,
            "expected_rows_per_partition": info.expected_rows_per_partition,
        },
    )


def build_tools(ctx: ToolContext) -> list[Tool]:
    """The data-investigation tools, with ``ctx`` already bound."""
    return [
        Tool(
            name="run_sql",
            description=(
                "Run a read-only SQL query against the warehouse and return rows. "
                "Only single SELECT/WITH queries are permitted; writes and DDL are "
                "rejected. Use it to measure the data directly."
            ),
            parameters=object_schema(
                {
                    "query": {
                        "type": "string",
                        "description": "A single read-only SQL statement",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_QUERY_LIMIT,
                        "description": f"Maximum rows to return (default {DEFAULT_QUERY_LIMIT})",
                    },
                },
                required=["query"],
            ),
            fn=lambda **kwargs: _run_sql(ctx, **kwargs),
            source=EvidenceSource.SQL,
        ),
        Tool(
            name="check_table_schema",
            description=(
                "List a table's columns and types, and report whether any column the "
                "pipeline requires is missing. Use it to test a schema-change hypothesis."
            ),
            parameters=object_schema(
                {"table": {"type": "string", "description": "Table name"}},
                required=["table"],
            ),
            fn=lambda **kwargs: _check_table_schema(ctx, **kwargs),
            source=EvidenceSource.SCHEMA,
        ),
        Tool(
            name="check_row_count",
            description=(
                "Count rows in a table, optionally for one partition, and compare with "
                "the previous partition and the expected volume. Omit partition for a "
                "per-partition breakdown."
            ),
            parameters=object_schema(
                {
                    "table": {"type": "string", "description": "Table name"},
                    "partition": {
                        "type": "string",
                        "description": "Partition value, e.g. 2026-09-20",
                    },
                },
                required=["table"],
            ),
            fn=lambda **kwargs: _check_row_count(ctx, **kwargs),
            source=EvidenceSource.SQL,
        ),
        Tool(
            name="check_null_rate",
            description=(
                "Measure the null rate of one column, optionally for a partition, and "
                "report whether it breaches the table's configured threshold."
            ),
            parameters=object_schema(
                {
                    "table": {"type": "string", "description": "Table name"},
                    "column": {"type": "string", "description": "Column to measure"},
                    "partition": {"type": "string", "description": "Optional partition value"},
                },
                required=["table", "column"],
            ),
            fn=lambda **kwargs: _check_null_rate(ctx, **kwargs),
            source=EvidenceSource.SQL,
        ),
        Tool(
            name="check_latest_partition",
            description=(
                "Find the newest partition present in a table and detect gaps in the "
                "partition series. Use it to establish whether the expected data "
                "arrived at all."
            ),
            parameters=object_schema(
                {"table": {"type": "string", "description": "Table name"}},
                required=["table"],
            ),
            fn=lambda **kwargs: _check_latest_partition(ctx, **kwargs),
            source=EvidenceSource.SQL,
        ),
    ]
