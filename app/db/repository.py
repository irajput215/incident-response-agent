"""Repositories: every SQL statement about platform state lives here.

Keeping SQL out of the graph nodes means the nodes read as the workflow they
are — "triage, then investigate, then conclude" — instead of a mix of control
flow and string concatenation. It also means the persistence contract can be
tested without running an LLM.

Row mappers live here too, so the shape of a database row is converted to a
domain object in exactly one place.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from app.observability import get_logger, log
from app.schemas import (
    ApprovalDecision,
    Evidence,
    Incident,
    IncidentCreate,
    IncidentReport,
    IncidentStatus,
    Triage,
)

_log = get_logger("app.db.repository")

_SCHEMA = "platform"


def new_id(prefix: str) -> str:
    """A short, sortable-enough, collision-safe identifier."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class IncidentRepository:
    """Reads and writes the incident-response record."""

    def __init__(self, conn: psycopg.Connection[dict[str, Any]]) -> None:
        self._conn = conn

    # --- incidents ---------------------------------------------------------
    def create_incident(self, alert: IncidentCreate) -> Incident:
        """Record an alert. Idempotent on ``(pipeline, run_id)``.

        Alerts get re-delivered — a webhook retries, an operator re-POSTs. The
        unique constraint plus ``ON CONFLICT`` means the second delivery
        enriches the existing incident instead of creating a duplicate, which is
        what makes "how many incidents did we have?" a trustworthy number.
        """
        incident_id = new_id("INC")
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.incidents
                    (incident_id, pipeline, run_id, status, error, logs, alert_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pipeline, run_id) DO UPDATE SET
                    error      = COALESCE(EXCLUDED.error, {_SCHEMA}.incidents.error),
                    logs       = COALESCE(EXCLUDED.logs,  {_SCHEMA}.incidents.logs),
                    updated_at = now()
                RETURNING *
                """,
                [
                    incident_id,
                    alert.pipeline,
                    alert.run_id,
                    IncidentStatus.OPEN.value,
                    alert.error,
                    alert.logs,
                    alert.timestamp,
                ],
            )
            row = cur.fetchone()
        self._conn.commit()
        if row is None:  # pragma: no cover - RETURNING always yields a row
            raise RuntimeError("INSERT ... RETURNING produced no row")
        return _to_incident(row)

    def get_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.incidents WHERE incident_id = %s", [incident_id]
            )
            return cur.fetchone()

    def find_incident(self, pipeline: str, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.incidents WHERE pipeline = %s AND run_id = %s",
                [pipeline, run_id],
            )
            return cur.fetchone()

    def list_incidents(
        self, *, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {_SCHEMA}.incidents"
        params: list[Any] = []
        if status:
            sql += " WHERE status = %s"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def set_incident_status(
        self,
        incident_id: str,
        status: IncidentStatus,
        *,
        severity: str | None = None,
        category: str | None = None,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {_SCHEMA}.incidents
                   SET status     = %s,
                       severity   = COALESCE(%s, severity),
                       category   = COALESCE(%s, category),
                       updated_at = now()
                 WHERE incident_id = %s
                """,
                [status.value, severity, category, incident_id],
            )
        self._conn.commit()

    def apply_triage(self, incident_id: str, triage: Triage) -> None:
        self.set_incident_status(
            incident_id,
            IncidentStatus.INVESTIGATING,
            severity=triage.severity.value,
            category=triage.category.value,
        )

    def count_incidents(self) -> dict[str, int]:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT status, count(*) AS n FROM {_SCHEMA}.incidents GROUP BY status")
            return {row["status"]: int(row["n"]) for row in cur.fetchall()}

    # --- investigations ----------------------------------------------------
    def start_investigation(self, incident_id: str, planner: str) -> str:
        """Phase 1 of the two-phase write: record the *intent* before the work.

        A row appears with ``status='RUNNING'`` before any node executes, so a
        process killed mid-investigation leaves a visible orphan instead of no
        trace at all. That is the difference between "the agent crashed" and
        "the agent never ran", which at 2am is the whole question.
        """
        investigation_id = new_id("INV")
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.investigations
                    (investigation_id, incident_id, status, planner)
                VALUES (%s, %s, %s, %s)
                """,
                [investigation_id, incident_id, IncidentStatus.INVESTIGATING.value, planner],
            )
        self._conn.commit()
        log(_log, 20, "investigation_started", investigation_id=investigation_id, planner=planner)
        return investigation_id

    def update_investigation(
        self,
        investigation_id: str,
        status: IncidentStatus,
        *,
        rounds: int = 0,
        llm_calls: int = 0,
        tool_calls: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str | None = None,
    ) -> None:
        """Update status and counters **without closing** the investigation.

        This is what a paused-for-approval investigation gets: it has real cost
        so far, and a real status, but it is not finished. Leaving
        ``finished_at`` null is what distinguishes "waiting for a human" from
        "the process died", and the orphan sweeper keys off exactly that.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {_SCHEMA}.investigations
                   SET status     = %s,
                       rounds     = %s,
                       llm_calls  = %s,
                       tool_calls = %s,
                       tokens_in  = %s,
                       tokens_out = %s,
                       error      = %s
                 WHERE investigation_id = %s
                """,
                [
                    status.value,
                    rounds,
                    llm_calls,
                    tool_calls,
                    tokens_in,
                    tokens_out,
                    error,
                    investigation_id,
                ],
            )
        self._conn.commit()

    def finish_investigation(
        self,
        investigation_id: str,
        status: IncidentStatus,
        *,
        rounds: int = 0,
        llm_calls: int = 0,
        tool_calls: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str | None = None,
    ) -> None:
        """Phase 2: record the outcome and the cost of getting there."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {_SCHEMA}.investigations
                   SET status      = %s,
                       rounds      = %s,
                       llm_calls   = %s,
                       tool_calls  = %s,
                       tokens_in   = %s,
                       tokens_out  = %s,
                       error       = %s,
                       finished_at = now()
                 WHERE investigation_id = %s
                """,
                [
                    status.value,
                    rounds,
                    llm_calls,
                    tool_calls,
                    tokens_in,
                    tokens_out,
                    error,
                    investigation_id,
                ],
            )
        self._conn.commit()
        log(
            _log,
            20,
            "investigation_finished",
            investigation_id=investigation_id,
            status=status.value,
            rounds=rounds,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
        )

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.investigations WHERE investigation_id = %s",
                [investigation_id],
            )
            return cur.fetchone()

    def latest_investigation(self, incident_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {_SCHEMA}.investigations
                 WHERE incident_id = %s
                 ORDER BY started_at DESC LIMIT 1
                """,
                [incident_id],
            )
            return cur.fetchone()

    def sweep_orphaned_investigations(self, max_age_minutes: int = 30) -> list[str]:
        """Mark investigations that will never finish.

        Without this, every crash leaves a row in ``RUNNING`` forever and the
        dashboard slowly fills with phantoms. Sweeping is the honest response:
        we cannot know they died, so we age them out.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE {_SCHEMA}.investigations
                   SET status      = %s,
                       error       = COALESCE(error, 'orphaned: process exited without finishing'),
                       finished_at = now()
                 WHERE status = %s
                   AND started_at < now() - %s::interval
                RETURNING investigation_id
                """,
                [
                    IncidentStatus.FAILED.value,
                    IncidentStatus.INVESTIGATING.value,
                    f"{max_age_minutes} minutes",
                ],
            )
            swept = [row["investigation_id"] for row in cur.fetchall()]
        self._conn.commit()
        if swept:
            log(_log, 30, "orphans_swept", count=len(swept))
        return swept

    # --- trace -------------------------------------------------------------
    def record_step(
        self,
        investigation_id: str,
        *,
        seq: int,
        node: str,
        ok: bool,
        summary: str = "",
        detail: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.agent_steps
                    (investigation_id, node, seq, ok, summary, detail, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [investigation_id, node, seq, ok, summary, Jsonb(detail or {}), duration_ms],
            )
        self._conn.commit()

    def get_steps(self, investigation_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.agent_steps WHERE investigation_id = %s ORDER BY seq",
                [investigation_id],
            )
            return cur.fetchall()

    def record_evidence(self, investigation_id: str, evidence: Evidence) -> None:
        """Store one observation. Idempotent per (tool, summary)."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.evidence
                    (investigation_id, source, tool, summary, detail, supports, refutes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (investigation_id, tool, summary) DO NOTHING
                """,
                [
                    investigation_id,
                    evidence.source.value,
                    evidence.tool,
                    evidence.summary,
                    Jsonb(evidence.detail),
                    evidence.supports,
                    evidence.refutes,
                ],
            )
        self._conn.commit()

    def get_evidence(self, investigation_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.evidence WHERE investigation_id = %s ORDER BY evidence_id",
                [investigation_id],
            )
            return cur.fetchall()

    def record_tool_call(
        self,
        investigation_id: str,
        *,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.tool_calls
                    (investigation_id, tool, arguments, ok, error, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [investigation_id, tool, Jsonb(arguments), ok, error, duration_ms],
            )
        self._conn.commit()

    def get_tool_calls(self, investigation_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.tool_calls WHERE investigation_id = %s ORDER BY call_id",
                [investigation_id],
            )
            return cur.fetchall()

    # --- approvals and reports ---------------------------------------------
    def record_approval(self, investigation_id: str, decision: ApprovalDecision) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.approvals
                    (investigation_id, approved, approver, note, decided_at)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING approval_id
                """,
                [
                    investigation_id,
                    decision.approved,
                    decision.approver,
                    decision.note,
                    decision.decided_at,
                ],
            )
            row = cur.fetchone()
        self._conn.commit()
        if row is None:  # pragma: no cover - RETURNING always yields a row
            raise RuntimeError("INSERT ... RETURNING produced no row")
        log(
            _log,
            20,
            "approval_recorded",
            investigation_id=investigation_id,
            approved=decision.approved,
            approver=decision.approver,
        )
        return int(row["approval_id"])

    def latest_approval(self, investigation_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT * FROM {_SCHEMA}.approvals
                 WHERE investigation_id = %s
                 ORDER BY decided_at DESC LIMIT 1
                """,
                [investigation_id],
            )
            return cur.fetchone()

    def save_report(self, investigation_id: str, report: IncidentReport) -> None:
        """Upsert the report. One report per investigation, rewritten on resume."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {_SCHEMA}.reports
                    (investigation_id, incident_id, status, summary, report, generated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (investigation_id) DO UPDATE SET
                    status       = EXCLUDED.status,
                    summary      = EXCLUDED.summary,
                    report       = EXCLUDED.report,
                    generated_at = EXCLUDED.generated_at
                """,
                [
                    investigation_id,
                    report.incident_id,
                    report.status.value,
                    report.summary,
                    Jsonb(report.model_dump(mode="json")),
                    report.generated_at,
                ],
            )
        self._conn.commit()

    def get_report(self, investigation_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT * FROM {_SCHEMA}.reports WHERE investigation_id = %s",
                [investigation_id],
            )
            return cur.fetchone()

    def report_for_incident(self, incident_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.*, i.investigation_id
                  FROM {_SCHEMA}.reports r
                  JOIN {_SCHEMA}.investigations i ON i.investigation_id = r.investigation_id
                 WHERE r.incident_id = %s
                 ORDER BY r.generated_at DESC LIMIT 1
                """,
                [incident_id],
            )
            return cur.fetchone()

    # --- operator helpers ---------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Counts for ``/health``: is the platform actually accumulating state?"""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    (SELECT count(*) FROM {_SCHEMA}.incidents)      AS incidents,
                    (SELECT count(*) FROM {_SCHEMA}.investigations) AS investigations,
                    (SELECT count(*) FROM {_SCHEMA}.evidence)       AS evidence,
                    (SELECT count(*) FROM {_SCHEMA}.tool_calls)     AS tool_calls,
                    (SELECT count(*) FROM {_SCHEMA}.reports)        AS reports,
                    (SELECT count(*) FROM {_SCHEMA}.approvals)      AS approvals
                """
            )
            row = cur.fetchone() or {}
        return {key: int(value) for key, value in row.items()}

    def reset(self) -> None:
        """Delete all platform state. **Tests only** — never call this in a request."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"TRUNCATE {_SCHEMA}.reports, {_SCHEMA}.approvals, {_SCHEMA}.tool_calls, "
                f"{_SCHEMA}.evidence, {_SCHEMA}.agent_steps, {_SCHEMA}.investigations, "
                f"{_SCHEMA}.incidents RESTART IDENTITY CASCADE"
            )
        self._conn.commit()


def _to_incident(row: dict[str, Any]) -> Incident:
    """Map an ``incidents`` row onto the domain model."""
    return Incident(
        incident_id=row["incident_id"],
        pipeline=row["pipeline"],
        run_id=row["run_id"],
        status=IncidentStatus(row["status"]),
        timestamp=row["alert_ts"],
        error=row.get("error"),
        logs=row.get("logs"),
        created_at=row.get("created_at") or datetime.now(tz=UTC),
    )


def sweep_after() -> timedelta:
    """Default orphan age before a ``RUNNING`` investigation is considered dead."""
    return timedelta(minutes=30)
