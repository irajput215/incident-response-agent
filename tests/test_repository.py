"""Persistence: migrations, the two-phase investigation write, and idempotency.

The interesting assertions are about *invariants the schema enforces* rather
than conventions the code remembers. Idempotency in particular is tested by
doing the thing twice and counting rows, because that is the only way to know
the constraint is doing the work.
"""
from __future__ import annotations

from datetime import UTC, datetime

from app.db.repository import new_id
from app.db.schema import LATEST_VERSION, current_version, describe, migrate
from app.schemas import (
    ApprovalDecision,
    Evidence,
    EvidenceSource,
    FailureCategory,
    IncidentCreate,
    IncidentStatus,
    RootCause,
    Severity,
    Triage,
)

FIXED_TS = datetime(2026, 9, 21, 2, 14, tzinfo=UTC)


def alert(run_id: str = "run_98321", **overrides: object) -> IncidentCreate:
    payload = {
        "pipeline": "customer_claims_daily",
        "run_id": run_id,
        "timestamp": FIXED_TS,
        "error": "SparkException: File not found",
        "table": "claims",
    }
    payload.update(overrides)
    return IncidentCreate.model_validate(payload)


# --------------------------------------------------------------------------- #
# migrations
# --------------------------------------------------------------------------- #
def test_migrations_reach_the_latest_version(conn) -> None:
    assert current_version(conn) == LATEST_VERSION


def test_migrations_are_idempotent(conn) -> None:
    """Running them twice must apply nothing the second time."""
    assert migrate(conn) == []
    assert current_version(conn) == LATEST_VERSION


def test_the_schema_is_queryable_as_a_fact(conn) -> None:
    described = describe(conn)
    assert described["version"] == LATEST_VERSION
    for table in ("incidents", "investigations", "agent_steps", "evidence", "tool_calls", "approvals", "reports"):
        assert table in described["tables"]


# --------------------------------------------------------------------------- #
# incidents
# --------------------------------------------------------------------------- #
def test_an_alert_is_recorded(repository) -> None:
    incident = repository.create_incident(alert())
    assert incident.incident_id.startswith("INC-")
    assert incident.status is IncidentStatus.OPEN
    assert repository.get_incident(incident.incident_id) is not None


def test_a_redelivered_alert_enriches_rather_than_duplicates(repository) -> None:
    """Webhooks retry. "How many incidents did we have?" must stay trustworthy."""
    first = repository.create_incident(alert(error=None))
    second = repository.create_incident(alert(error="now with an exception"))

    assert first.incident_id == second.incident_id
    assert len(repository.list_incidents()) == 1
    assert repository.get_incident(first.incident_id)["error"] == "now with an exception"


def test_different_runs_are_different_incidents(repository) -> None:
    repository.create_incident(alert(run_id="run_1"))
    repository.create_incident(alert(run_id="run_2"))
    assert len(repository.list_incidents()) == 2


def test_triage_is_persisted_onto_the_incident(repository) -> None:
    incident = repository.create_incident(alert())
    repository.apply_triage(
        incident.incident_id,
        Triage(
            category=FailureCategory.DATA_SOURCE_FAILURE,
            severity=Severity.HIGH,
            initial_hypothesis="missing partition",
        ),
    )
    row = repository.get_incident(incident.incident_id)
    assert row["status"] == IncidentStatus.INVESTIGATING.value
    assert row["severity"] == "HIGH"
    assert row["category"] == "DATA_SOURCE_FAILURE"


def test_incidents_can_be_filtered_by_status(repository) -> None:
    incident = repository.create_incident(alert())
    repository.set_incident_status(incident.incident_id, IncidentStatus.RESOLVED)
    assert len(repository.list_incidents(status="RESOLVED")) == 1
    assert repository.list_incidents(status="OPEN") == []


# --------------------------------------------------------------------------- #
# the two-phase investigation write
# --------------------------------------------------------------------------- #
def test_an_investigation_is_visible_before_any_work_happens(repository) -> None:
    """Phase 1: intent is recorded before execution.

    This is what makes "the agent crashed" distinguishable from "the agent never
    ran" — the difference between a bug and an outage, at 2am.
    """
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")

    row = repository.get_investigation(investigation_id)
    assert row["status"] == IncidentStatus.INVESTIGATING.value
    assert row["planner"] == "stub"
    assert row["finished_at"] is None


def test_finishing_records_the_cost_of_getting_there(repository) -> None:
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    repository.finish_investigation(
        investigation_id,
        IncidentStatus.RESOLVED,
        rounds=2,
        llm_calls=7,
        tool_calls=5,
        tokens_in=1200,
        tokens_out=340,
    )
    row = repository.get_investigation(investigation_id)
    assert row["status"] == IncidentStatus.RESOLVED.value
    assert (row["rounds"], row["llm_calls"], row["tool_calls"]) == (2, 7, 5)
    assert (row["tokens_in"], row["tokens_out"]) == (1200, 340)
    assert row["finished_at"] is not None


def test_a_paused_investigation_is_not_finished(repository) -> None:
    """Waiting for a human is not the same as done, and `finished_at` is how the
    orphan sweeper tells them apart."""
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    repository.update_investigation(
        investigation_id, IncidentStatus.AWAITING_APPROVAL, llm_calls=4, tool_calls=3
    )

    row = repository.get_investigation(investigation_id)
    assert row["status"] == IncidentStatus.AWAITING_APPROVAL.value
    assert row["llm_calls"] == 4
    assert row["finished_at"] is None


def test_orphaned_investigations_are_swept(repository, conn) -> None:
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")

    # Backdate it to simulate a process that died long ago.
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE platform.investigations SET started_at = now() - interval '2 hours' "
            "WHERE investigation_id = %s",
            [investigation_id],
        )
    conn.commit()

    swept = repository.sweep_orphaned_investigations(max_age_minutes=30)
    assert swept == [investigation_id]
    row = repository.get_investigation(investigation_id)
    assert row["status"] == IncidentStatus.FAILED.value
    assert "orphaned" in row["error"]


def test_a_recent_investigation_is_not_swept(repository) -> None:
    """Sweeping a run that is merely slow would be a self-inflicted outage."""
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    assert repository.sweep_orphaned_investigations(max_age_minutes=30) == []
    assert repository.get_investigation(investigation_id)["finished_at"] is None


def test_a_paused_investigation_is_never_swept(repository, conn) -> None:
    """It is waiting for a human, not dead."""
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    repository.update_investigation(investigation_id, IncidentStatus.AWAITING_APPROVAL)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE platform.investigations SET started_at = now() - interval '2 hours' "
            "WHERE investigation_id = %s",
            [investigation_id],
        )
    conn.commit()
    assert repository.sweep_orphaned_investigations(max_age_minutes=30) == []


# --------------------------------------------------------------------------- #
# evidence, tool calls, steps
# --------------------------------------------------------------------------- #
def test_evidence_is_idempotent(repository) -> None:
    """Re-running an investigation re-observes the same facts. Without the
    constraint the table doubles and the evidence count becomes a lie."""
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    evidence = Evidence(
        source=EvidenceSource.SQL,
        tool="check_row_count",
        summary="claims partition 2026-09-20: 0 rows",
        detail={"row_count": 0},
    )
    for _ in range(3):
        repository.record_evidence(investigation_id, evidence)
    assert len(repository.get_evidence(investigation_id)) == 1


def test_distinct_evidence_is_kept(repository) -> None:
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    for row_count in (0, 200):
        repository.record_evidence(
            investigation_id,
            Evidence(
                source=EvidenceSource.SQL,
                tool="check_row_count",
                summary=f"rows={row_count}",
                detail={"row_count": row_count},
            ),
        )
    assert len(repository.get_evidence(investigation_id)) == 2


def test_tool_calls_record_failures_too(repository) -> None:
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    repository.record_tool_call(
        investigation_id, tool="run_sql", arguments={"query": "DROP TABLE x"}, ok=False, error="rejected"
    )
    repository.record_tool_call(
        investigation_id, tool="check_row_count", arguments={"table": "claims"}, ok=True
    )
    calls = repository.get_tool_calls(investigation_id)
    assert len(calls) == 2
    assert calls[0]["ok"] is False


def test_steps_form_an_ordered_trace(repository) -> None:
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    for seq, node in enumerate(("triage", "investigate", "assess_root_cause")):
        repository.record_step(
            investigation_id, seq=seq, node=node, ok=True, summary=f"{node} done", duration_ms=1.5
        )
    steps = repository.get_steps(investigation_id)
    assert [s["node"] for s in steps] == ["triage", "investigate", "assess_root_cause"]
    assert steps[0]["duration_ms"] == 1.5


# --------------------------------------------------------------------------- #
# approvals and reports
# --------------------------------------------------------------------------- #
def test_approvals_are_append_only(repository) -> None:
    """"Who approved this, and when" is exactly what a mutable row loses."""
    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    repository.record_approval(
        investigation_id, ApprovalDecision(approved=False, approver="a@x", note="no")
    )
    repository.record_approval(
        investigation_id, ApprovalDecision(approved=True, approver="b@x", note="yes")
    )
    assert repository.latest_approval(investigation_id)["approved"] is True

    with repository._conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM platform.approvals WHERE investigation_id = %s",
            [investigation_id],
        )
        assert cur.fetchone()["n"] == 2


def test_a_report_is_upserted_not_duplicated(repository) -> None:
    from app.agents.state import build_report, initial_state

    incident = repository.create_incident(alert())
    investigation_id = repository.start_investigation(incident.incident_id, "stub")
    state = initial_state(
        alert().model_dump(mode="json"),
        incident_id=incident.incident_id,
        investigation_id=investigation_id,
    )
    state["root_cause"] = RootCause(
        root_cause="upstream failed", category=FailureCategory.UPSTREAM_DEPENDENCY_FAILURE
    ).model_dump(mode="json")

    for _ in range(2):
        repository.save_report(investigation_id, build_report(state, model="stub", duration_ms=1.0))

    assert repository.get_report(investigation_id) is not None
    assert repository.stats()["reports"] == 1


def test_stats_count_what_the_platform_has_accumulated(repository) -> None:
    incident = repository.create_incident(alert())
    repository.start_investigation(incident.incident_id, "stub")
    stats = repository.stats()
    assert stats["incidents"] == 1
    assert stats["investigations"] == 1
    assert stats["reports"] == 0


def test_reset_clears_platform_state_only(repository) -> None:
    """Warehouse tables must survive a reset — they are the estate, not platform state."""
    repository.create_incident(alert())
    repository.reset()
    assert repository.list_incidents() == []
    assert repository._conn.execute("SELECT count(*) AS n FROM warehouse.claims").fetchone()["n"] > 0


def test_new_id_is_unique_and_prefixed() -> None:
    ids = {new_id("INV") for _ in range(200)}
    assert len(ids) == 200
    assert all(i.startswith("INV-") for i in ids)
