"""The graph's state: one typed dict that every node reads and writes.

LangGraph merges each node's return value into this state, so the shape *is* the
workflow's data contract. Two conventions keep it honest:

**Persisted objects are stored as plain dicts, not pydantic models.** A
checkpointer serialises state between the interrupt and the resume, and a
partially-typed object graph round-tripping through JSON is a class of bug
nobody needs. Nodes convert to a model at the boundary, validate, and store the
dump.

**Telemetry lives in the state, not in a module global.** ``llm_calls`` and
``tool_calls`` are carried alongside the findings, so a per-incident budget is a
state comparison rather than a guess, and the final report can state its own
cost.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict

from app.schemas import IncidentReport, IncidentStatus


class IncidentState(TypedDict, total=False):
    """Everything the workflow knows about one incident."""

    # --- identity ----------------------------------------------------------
    incident: dict[str, Any]  # IncidentCreate, dumped
    incident_id: str
    investigation_id: str

    # --- findings ----------------------------------------------------------
    triage: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    root_cause: dict[str, Any] | None
    remediation: dict[str, Any] | None
    approval: dict[str, Any] | None
    report: dict[str, Any] | None

    # --- control -----------------------------------------------------------
    rounds: int
    max_rounds: int
    needs_more_evidence: bool
    remediation_executed: bool

    # --- telemetry ---------------------------------------------------------
    planner: str  # "heuristic" or the model id — recorded, never inferred
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int
    errors: list[str]
    timeline: list[dict[str, Any]]
    status: str


def initial_state(
    incident: dict[str, Any],
    *,
    incident_id: str = "",
    investigation_id: str = "",
    planner: str = "",
    max_rounds: int = 3,
) -> IncidentState:
    """A fully-populated starting state.

    Every key is set explicitly even when empty. LangGraph tolerates missing
    keys, but a node that reads ``state["evidence"]`` and finds no key at all
    fails differently from one that finds an empty list — and only the second
    behaviour is worth debugging.
    """
    return IncidentState(
        incident=incident,
        incident_id=incident_id,
        investigation_id=investigation_id,
        triage=None,
        evidence=[],
        root_cause=None,
        remediation=None,
        approval=None,
        report=None,
        rounds=0,
        max_rounds=max_rounds,
        needs_more_evidence=False,
        remediation_executed=False,
        planner=planner,
        llm_calls=0,
        tool_calls=0,
        tokens_in=0,
        tokens_out=0,
        errors=[],
        timeline=[],
        status=IncidentStatus.OPEN.value,
    )


def append_timeline(
    state: IncidentState, node: str, summary: str, **detail: Any
) -> list[dict[str, Any]]:
    """Return a new timeline with one entry appended.

    Returns a new list rather than mutating: LangGraph decides what changed by
    comparing values, and an in-place append is invisible to that comparison.
    """
    from app.schemas import utcnow

    entry = {
        "at": utcnow().isoformat(),
        "node": node,
        "summary": summary,
        "detail": detail,
    }
    return [*state.get("timeline", []), entry]


def evidence_summaries(state: IncidentState, limit: int = 30) -> list[str]:
    """Evidence as short strings, for prompts and for the report."""
    return [e.get("summary", "") for e in state.get("evidence", [])][:limit]


def build_report(state: IncidentState, *, model: str, duration_ms: float) -> IncidentReport:
    """Assemble the final artefact from the accumulated state.

    Kept in the state module rather than in a node because it is a pure
    transformation of state → report, and pure things are worth being able to
    test without a graph.
    """
    from app.schemas import (
        ApprovalDecision,
        Evidence,
        FailureCategory,
        Remediation,
        ReportStep,
        RootCause,
        Triage,
    )

    raw_incident = state.get("incident") or {}
    triage_raw = state.get("triage")
    root_raw = state.get("root_cause")
    remediation_raw = state.get("remediation")
    approval_raw = state.get("approval")

    triage = Triage.model_validate(triage_raw) if triage_raw else None
    root_cause = RootCause.model_validate(root_raw) if root_raw else None
    remediation = Remediation.model_validate(remediation_raw) if remediation_raw else None
    approval = ApprovalDecision.model_validate(approval_raw) if approval_raw else None

    evidence: list[Evidence] = []
    for item in state.get("evidence", []):
        try:
            evidence.append(Evidence.model_validate(item))
        except Exception:  # a malformed evidence row must not break the report
            continue

    timeline = [
        ReportStep(
            at=_as_datetime(entry.get("at")),
            node=entry.get("node", ""),
            summary=entry.get("summary", ""),
            detail=entry.get("detail", {}),
        )
        for entry in state.get("timeline", [])
    ]

    status = IncidentStatus(state.get("status", IncidentStatus.OPEN.value))

    return IncidentReport(
        incident_id=state.get("incident_id", ""),
        pipeline=str(raw_incident.get("pipeline", "")),
        run_id=str(raw_incident.get("run_id", "")),
        status=status,
        severity=triage.severity if triage else None,
        category=root_cause.category
        if root_cause
        else (triage.category if triage else FailureCategory.UNKNOWN),
        summary=_summarise(status, root_cause, remediation, approval),
        root_cause=root_cause,
        remediation=remediation,
        approval=approval,
        evidence=evidence,
        timeline=timeline,
        investigation_rounds=int(state.get("rounds", 0)),
        llm_calls=int(state.get("llm_calls", 0)),
        tool_calls=int(state.get("tool_calls", 0)),
        tokens_in=int(state.get("tokens_in", 0)),
        tokens_out=int(state.get("tokens_out", 0)),
        duration_ms=round(duration_ms, 2),
        model=model,
    )


def _summarise(
    status: IncidentStatus,
    root_cause: Any,
    remediation: Any,
    approval: Any,
) -> str:
    """One paragraph a human can act on. Deliberately blunt about uncertainty."""
    if status is IncidentStatus.RESOLVED:
        parts = [f"Root cause identified: {root_cause.root_cause}."]
        if remediation:
            parts.append(f"Remediation {remediation.action.value.lower()} was approved and applied.")
        return " ".join(parts)

    if status is IncidentStatus.AWAITING_APPROVAL:
        if remediation:
            return (
                f"Root cause identified: {root_cause.root_cause}. "
                f"Proposed remediation {remediation.action.value} on {remediation.target} "
                "is awaiting human approval."
                if root_cause
                else "Awaiting approval."
            )
        return "Awaiting human approval."

    if status is IncidentStatus.UNRESOLVED:
        return (
            "Investigation completed without establishing a single confident root cause. "
            "The collected evidence is recorded below; escalate to a human investigator."
        )

    if status is IncidentStatus.FAILED:
        return "The investigation itself failed before reaching a conclusion."

    return f"Investigation is in state {status.value}."


def _as_datetime(value: object) -> datetime:
    """Coerce a timeline entry's timestamp back into a datetime.

    Timeline entries are serialised into the checkpointed state as ISO
    strings, so by the time a report is built they are strings again. A
    missing or unparseable value falls back to *now*, because a report with a
    slightly wrong timestamp is better than no report.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
        except ValueError:
            pass
    return datetime.now(tz=UTC)


__all__ = [
    "IncidentState",
    "append_timeline",
    "build_report",
    "evidence_summaries",
    "initial_state",
]
