"""HTTP routes.

Three conventions worth noting:

**Handlers are plain ``def``, not ``async def``.** All the work is blocking —
database queries, file reads, an HTTP call to a model. Declared ``async``, that
blocking work would run *on the event loop* and stall every other request;
declared plain, FastAPI runs each in a threadpool and the server stays
responsive. Using ``async def`` here would be a performance bug dressed as a
style choice.

**The platform comes from ``app.state``, never a module global.** A module-level
``app = FastAPI()`` that opens a database at import time breaks tests, breaks the
CLI, and makes it impossible to run two configurations in one process.

**Approval is addressed by incident, not by investigation.** An operator looking
at an incident page should not have to know which investigation attempt is
suspended; the API finds it, and returns ``409`` when there is nothing to decide.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.agents import InvestigationOutcome
from app.api.schemas import (
    ApprovalRequest,
    IncidentDetail,
    IncidentSummary,
    MetricsResponse,
    OutcomeResponse,
    RejectionRequest,
)
from app.observability import get_logger, log
from app.platform import Platform
from app.schemas import ApprovalDecision, IncidentCreate, IncidentStatus

_log = get_logger("app.api")

router = APIRouter()


def get_platform(request: Request) -> Platform:
    """The platform built by the app factory and stored on ``app.state``."""
    platform = getattr(request.app.state, "platform", None)
    if platform is None:  # pragma: no cover - misconfiguration, not a user error
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="platform not initialised",
        )
    return platform


PlatformDep = Annotated[Platform, Depends(get_platform)]


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
@router.get("/health", tags=["ops"], summary="Liveness plus capability")
def health(platform: PlatformDep) -> dict[str, Any]:
    """What this process can actually do — not merely that it is running.

    The ``capabilities`` block is what answers "why is the agent behaving like a
    stub in staging?" in one request.
    """
    return platform.health()


@router.get("/metrics", response_model=MetricsResponse, tags=["ops"])
def metrics(platform: PlatformDep) -> dict[str, Any]:
    """Counters and latency percentiles for this process."""
    return platform.metrics.snapshot()


# --------------------------------------------------------------------------- #
# incidents
# --------------------------------------------------------------------------- #
@router.post(
    "/incidents",
    response_model=OutcomeResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["incidents"],
    summary="Report a failed pipeline run",
)
def create_incident(
    alert: IncidentCreate,
    platform: PlatformDep,
    investigate: bool = Query(
        default=True, description="Run the agent immediately, or just record the alert"
    ),
) -> OutcomeResponse:
    """Record an alert and investigate it.

    Idempotent on ``(pipeline, run_id)``: a retried webhook enriches the existing
    incident instead of opening a second one.
    """
    if not investigate:
        incident = platform.repo.create_incident(alert)
        return OutcomeResponse(
            incident_id=incident.incident_id,
            investigation_id="",
            status=incident.status.value,
        )

    outcome = platform.agent.investigate(alert)
    return _outcome(outcome)


@router.get(
    "/incidents",
    response_model=list[IncidentSummary],
    tags=["incidents"],
    summary="List incidents, newest first",
)
def list_incidents(
    platform: PlatformDep,
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[IncidentSummary]:
    rows = platform.repo.list_incidents(status=status_filter, limit=limit, offset=offset)
    return [
        IncidentSummary(
            incident_id=row["incident_id"],
            pipeline=row["pipeline"],
            run_id=row["run_id"],
            status=row["status"],
            severity=row.get("severity"),
            category=row.get("category"),
            created_at=row.get("created_at"),
        )
        for row in rows
    ]


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentDetail,
    tags=["incidents"],
    summary="One incident, with its latest investigation",
)
def get_incident(incident_id: str, platform: PlatformDep) -> IncidentDetail:
    incident = platform.repo.get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail=f"unknown incident {incident_id!r}")

    investigation = platform.repo.latest_investigation(incident_id)
    counts: dict[str, int] = {}
    approval = None
    if investigation:
        investigation_id = investigation["investigation_id"]
        counts = {
            "steps": len(platform.repo.get_steps(investigation_id)),
            "evidence": len(platform.repo.get_evidence(investigation_id)),
            "tool_calls": len(platform.repo.get_tool_calls(investigation_id)),
        }
        approval = platform.repo.latest_approval(investigation_id)

    return IncidentDetail(
        incident=dict(incident),
        investigation=dict(investigation) if investigation else None,
        approval=dict(approval) if approval else None,
        counts=counts,
    )


@router.get(
    "/incidents/{incident_id}/report",
    tags=["incidents"],
    summary="The incident report",
)
def get_report(incident_id: str, platform: PlatformDep) -> dict[str, Any]:
    report = platform.repo.report_for_incident(incident_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"no report for incident {incident_id!r} (it may still be investigating "
            "or awaiting approval)",
        )
    return report["report"]


@router.get(
    "/incidents/{incident_id}/evidence",
    tags=["incidents"],
    summary="Every observation the investigation recorded",
)
def get_evidence(incident_id: str, platform: PlatformDep) -> list[dict[str, Any]]:
    investigation = _require_investigation(platform, incident_id)
    return [dict(row) for row in platform.repo.get_evidence(investigation["investigation_id"])]


@router.get(
    "/incidents/{incident_id}/timeline",
    tags=["incidents"],
    summary="The node-by-node trace of the investigation",
)
def get_timeline(incident_id: str, platform: PlatformDep) -> list[dict[str, Any]]:
    investigation = _require_investigation(platform, incident_id)
    return [dict(row) for row in platform.repo.get_steps(investigation["investigation_id"])]


# --------------------------------------------------------------------------- #
# human-in-the-loop
# --------------------------------------------------------------------------- #
@router.post(
    "/incidents/{incident_id}/approve",
    response_model=OutcomeResponse,
    tags=["approvals"],
    summary="Approve the proposed remediation and resume the workflow",
)
def approve(
    incident_id: str, body: ApprovalRequest, platform: PlatformDep
) -> OutcomeResponse:
    return _decide(
        platform,
        incident_id,
        ApprovalDecision(approved=True, approver=body.approver, note=body.note),
        action="approved",
    )


@router.post(
    "/incidents/{incident_id}/reject",
    response_model=OutcomeResponse,
    tags=["approvals"],
    summary="Reject the proposed remediation",
)
def reject(
    incident_id: str, body: RejectionRequest, platform: PlatformDep
) -> OutcomeResponse:
    note = body.reason or body.note
    return _decide(
        platform,
        incident_id,
        ApprovalDecision(approved=False, approver=body.approver, note=note),
        action="rejected",
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_investigation(platform: Platform, incident_id: str) -> dict[str, Any]:
    investigation = platform.repo.latest_investigation(incident_id)
    if investigation is None:
        raise HTTPException(
            status_code=404, detail=f"incident {incident_id!r} has no investigation"
        )
    return investigation


def _decide(
    platform: Platform,
    incident_id: str,
    decision: ApprovalDecision,
    *,
    action: str,
) -> OutcomeResponse:
    """Resume a suspended investigation, or explain why it cannot be resumed."""
    investigation = _require_investigation(platform, incident_id)
    current = investigation["status"]

    if current != IncidentStatus.AWAITING_APPROVAL.value:
        # 409, not 400: the request is well-formed, the resource is in the wrong
        # state. Approving a run that already finished must not re-execute it.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"investigation {investigation['investigation_id']} is {current!r}, "
                "not awaiting approval; there is nothing to decide"
            ),
        )

    log(
        _log,
        20,
        "approval_requested",
        incident_id=incident_id,
        investigation_id=investigation["investigation_id"],
        action=action,
        approver=decision.approver,
    )
    outcome = platform.agent.resume(investigation["investigation_id"], decision)
    return _outcome(outcome)


def _outcome(outcome: InvestigationOutcome) -> OutcomeResponse:
    return OutcomeResponse(
        incident_id=outcome.incident_id,
        investigation_id=outcome.investigation_id,
        status=outcome.status.value,
        interrupted=outcome.interrupted,
        awaiting_approval_url=(
            f"/incidents/{outcome.incident_id}/approve" if outcome.awaiting_approval else None
        ),
        approval_request=outcome.approval_request,
        error=outcome.error,
        report=outcome.report.model_dump(mode="json") if outcome.report else None,
    )
