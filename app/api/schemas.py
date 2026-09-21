"""Request and response models for the HTTP API.

The nested report is returned as the ``IncidentReport`` JSON rather than
re-modelled here: it is already a validated pydantic object with its own schema,
and duplicating that shape would create a second definition to keep in sync.
Everything that is *input*, or that is a stable summary an operator filters on,
gets a real model.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApprovalRequest(BaseModel):
    """A human approving a proposed remediation."""

    approver: str = Field(min_length=1, description="Who is taking responsibility")
    note: str = Field(default="", description="Optional context for the audit trail")


class RejectionRequest(ApprovalRequest):
    """A human declining a proposed remediation."""

    reason: str = Field(default="", description="Why it was rejected")
    escalate: bool = Field(
        default=True, description="Leave the incident open for a human investigator"
    )


class OutcomeResponse(BaseModel):
    """The result of an investigation — complete, or paused for a human."""

    incident_id: str
    investigation_id: str
    status: str
    interrupted: bool = False
    awaiting_approval_url: str | None = Field(
        default=None,
        description="Where to POST a decision, when the graph has paused",
    )
    approval_request: dict[str, Any] | None = None
    error: str | None = None
    report: dict[str, Any] | None = Field(
        default=None, description="The full IncidentReport, once one exists"
    )


class IncidentSummary(BaseModel):
    incident_id: str
    pipeline: str
    run_id: str
    status: str
    severity: str | None = None
    category: str | None = None
    created_at: Any = None


class IncidentDetail(BaseModel):
    incident: dict[str, Any]
    investigation: dict[str, Any] | None = None
    approval: dict[str, Any] | None = None
    counts: dict[str, int] = Field(default_factory=dict)


class MetricsResponse(BaseModel):
    counters: dict[str, float]
    latency: dict[str, dict[str, float]]
