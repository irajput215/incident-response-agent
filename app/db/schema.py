"""The platform's own schema, applied as versioned forward-only migrations.

The agent needs to remember things across restarts: what incident it was asked
about, what it investigated, what it concluded, who approved the fix, and what
it cost. That is this schema.

**Why migrations rather than ``CREATE TABLE IF NOT EXISTS`` at import time.**
``IF NOT EXISTS`` cannot express a change: adding a column to an existing table
is invisible to it, so the second deployment silently runs against the old
shape. A version table plus an ordered list makes "what shape is this database?"
a queryable fact, and makes an upgrade a thing that either fully applies or
loudly does not.

Migrations are **forward-only** — no ``down`` — because a rollback that drops a
column drops the incident history with it, and an incident history you can lose
is not an audit trail.
"""
from __future__ import annotations

from typing import Any

import psycopg

from app.observability import get_logger, log

_log = get_logger("app.db.schema")

SCHEMA = "platform"

# --- migration 1: the core incident-response record -------------------------
_V1 = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

-- An alert: what failed, when, and the error text we were given.
CREATE TABLE IF NOT EXISTS {SCHEMA}.incidents (
    incident_id   TEXT PRIMARY KEY,
    pipeline      TEXT        NOT NULL,
    run_id        TEXT        NOT NULL,
    status        TEXT        NOT NULL,
    severity      TEXT,
    category      TEXT,
    error         TEXT,
    logs          TEXT,
    alert_ts      TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- The same pipeline failing the same run twice is one incident, not two.
    UNIQUE (pipeline, run_id)
);
CREATE INDEX IF NOT EXISTS incidents_status_idx  ON {SCHEMA}.incidents (status);
CREATE INDEX IF NOT EXISTS incidents_created_idx ON {SCHEMA}.incidents (created_at DESC);

-- One agent attempt at one incident. Separate from the incident because an
-- incident can be re-investigated after new evidence arrives, and because the
-- two-phase write lives here: a row appears with status='RUNNING' *before*
-- any work happens, so a crash leaves a visible orphan rather than silence.
CREATE TABLE IF NOT EXISTS {SCHEMA}.investigations (
    investigation_id TEXT PRIMARY KEY,
    incident_id      TEXT NOT NULL REFERENCES {SCHEMA}.incidents (incident_id) ON DELETE CASCADE,
    status           TEXT NOT NULL,
    planner          TEXT,
    rounds           INT  NOT NULL DEFAULT 0,
    llm_calls        INT  NOT NULL DEFAULT 0,
    tool_calls       INT  NOT NULL DEFAULT 0,
    tokens_in        BIGINT NOT NULL DEFAULT 0,
    tokens_out       BIGINT NOT NULL DEFAULT 0,
    error            TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at      TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS investigations_incident_idx ON {SCHEMA}.investigations (incident_id);
CREATE INDEX IF NOT EXISTS investigations_status_idx   ON {SCHEMA}.investigations (status);

-- Node-level trace: the timeline a human reads after the fact.
CREATE TABLE IF NOT EXISTS {SCHEMA}.agent_steps (
    step_id          BIGSERIAL PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES {SCHEMA}.investigations (investigation_id) ON DELETE CASCADE,
    node             TEXT NOT NULL,
    seq              INT  NOT NULL,
    ok               BOOLEAN NOT NULL,
    summary          TEXT NOT NULL DEFAULT '',
    detail           JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    duration_ms      DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_steps_investigation_idx
    ON {SCHEMA}.agent_steps (investigation_id, seq);

-- Every observation the agent made, with what it supports or refutes.
CREATE TABLE IF NOT EXISTS {SCHEMA}.evidence (
    evidence_id      BIGSERIAL PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES {SCHEMA}.investigations (investigation_id) ON DELETE CASCADE,
    source           TEXT NOT NULL,
    tool             TEXT NOT NULL,
    summary          TEXT NOT NULL,
    detail           JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    supports         TEXT,
    refutes          TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Re-running an investigation re-observes the same facts. Without this the
    -- evidence table doubles on every retry and the "how much evidence did we
    -- gather?" number becomes a lie.
    UNIQUE (investigation_id, tool, summary)
);
CREATE INDEX IF NOT EXISTS evidence_investigation_idx ON {SCHEMA}.evidence (investigation_id);

-- Tool invocations: the tool-selection evaluation reads this table.
CREATE TABLE IF NOT EXISTS {SCHEMA}.tool_calls (
    call_id          BIGSERIAL PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES {SCHEMA}.investigations (investigation_id) ON DELETE CASCADE,
    tool             TEXT NOT NULL,
    arguments        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    ok               BOOLEAN NOT NULL,
    error            TEXT,
    duration_ms      DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tool_calls_investigation_idx ON {SCHEMA}.tool_calls (investigation_id);

-- Human decisions. Append-only on purpose: "who approved this, and when" is
-- exactly the question you cannot answer if approvals are mutable rows.
CREATE TABLE IF NOT EXISTS {SCHEMA}.approvals (
    approval_id      BIGSERIAL PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES {SCHEMA}.investigations (investigation_id) ON DELETE CASCADE,
    approved         BOOLEAN NOT NULL,
    approver         TEXT NOT NULL,
    note             TEXT NOT NULL DEFAULT '',
    decided_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS approvals_investigation_idx ON {SCHEMA}.approvals (investigation_id);

-- The final artefact, stored whole. The JSONB column is the report; the
-- scalar columns are the parts you filter and sort on.
CREATE TABLE IF NOT EXISTS {SCHEMA}.reports (
    investigation_id TEXT PRIMARY KEY
        REFERENCES {SCHEMA}.investigations (investigation_id) ON DELETE CASCADE,
    incident_id      TEXT NOT NULL,
    status           TEXT NOT NULL,
    summary          TEXT NOT NULL DEFAULT '',
    report           JSONB NOT NULL,
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS reports_incident_idx ON {SCHEMA}.reports (incident_id);
"""

# Ordered, append-only. Never edit an applied migration — add a new one.
MIGRATIONS: tuple[tuple[int, str], ...] = ((1, _V1),)

LATEST_VERSION = MIGRATIONS[-1][0]

# Bootstrap: the schema has to exist before the version table can live in it.
# Both statements are unparameterised, so psycopg runs them as one simple query.
_MIGRATION_TABLE = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};
CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_migrations (
    version    INT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def applied_versions(conn: psycopg.Connection[dict[str, Any]]) -> set[int]:
    with conn.cursor() as cur:
        cur.execute(_MIGRATION_TABLE)
        cur.execute(f"SELECT version FROM {SCHEMA}.schema_migrations")
        return {row["version"] for row in cur.fetchall()}


def migrate(conn: psycopg.Connection[dict[str, Any]]) -> list[int]:
    """Apply any unapplied migrations in order. Returns the versions applied.

    Idempotent: running it twice applies nothing the second time. Each migration
    runs in the caller's transaction, so a failure part-way leaves the database
    at a known version rather than half-upgraded.
    """
    already = applied_versions(conn)
    newly_applied: list[int] = []

    for version, sql in MIGRATIONS:
        if version in already:
            continue
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                f"INSERT INTO {SCHEMA}.schema_migrations (version) VALUES (%s) "
                "ON CONFLICT (version) DO NOTHING",
                [version],
            )
        newly_applied.append(version)
        log(_log, 20, "migration_applied", version=version)

    if not newly_applied:
        log(_log, 10, "migrations_up_to_date", version=LATEST_VERSION)
    return newly_applied


def current_version(conn: psycopg.Connection[dict[str, Any]]) -> int:
    versions = applied_versions(conn)
    return max(versions) if versions else 0


def ensure_schema(conn: psycopg.Connection[dict[str, Any]]) -> int:
    """Migrate to the latest version and return it."""
    migrate(conn)
    return current_version(conn)


def table_names(conn: psycopg.Connection[dict[str, Any]]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            [SCHEMA],
        )
        return [row["table_name"] for row in cur.fetchall()]


def describe(conn: psycopg.Connection[dict[str, Any]]) -> dict[str, Any]:
    """A small summary for ``/health`` — is the store actually there?"""
    return {
        "version": current_version(conn),
        "tables": table_names(conn),
    }
