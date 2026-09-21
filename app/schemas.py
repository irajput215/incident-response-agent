"""The domain model: incidents, evidence, root causes, remediations, reports.

These types are the contract between every layer — the LLM is asked to produce
them, the graph passes them around, the database persists them, and the API
serves them. Keeping them in one module is what stops "the shape of an
incident" from being re-invented slightly differently in four places.

A note on ``confidence``
------------------------
Every ``confidence`` field here is an **LLM self-assessment**, not a calibrated
probability. It is deliberately *not* computed from evidence counts, because
turning "3 of 4 signals agree" into "0.91 confident" is numerology dressed as
statistics. The right treatment is to carry it as an untrusted signal and then
*measure* whether it correlates with being right — which is what the Phase 4
evaluation does. Treating it as ground truth would be the single easiest way to
make this system look smarter than it is.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class FailureCategory(StrEnum):
    """What kind of failure this is. Drives both triage and remediation choice."""

    DATA_SOURCE_FAILURE = "DATA_SOURCE_FAILURE"
    UPSTREAM_DEPENDENCY_FAILURE = "UPSTREAM_DEPENDENCY_FAILURE"
    SCHEMA_CHANGE = "SCHEMA_CHANGE"
    DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
    RESOURCE_EXHAUSTION = "RESOURCE_EXHAUSTION"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    TRANSIENT_INFRASTRUCTURE = "TRANSIENT_INFRASTRUCTURE"
    CODE_ERROR = "CODE_ERROR"
    UNKNOWN = "UNKNOWN"


class Severity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class IncidentStatus(StrEnum):
    OPEN = "OPEN"
    INVESTIGATING = "INVESTIGATING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    REMEDIATING = "REMEDIATING"
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"  # investigation finished but no confident cause
    FAILED = "FAILED"  # the agent itself broke


class RemediationAction(StrEnum):
    RERUN_PIPELINE = "RERUN_PIPELINE"
    RERUN_UPSTREAM = "RERUN_UPSTREAM"
    BACKFILL_PARTITION = "BACKFILL_PARTITION"
    ALERT_OWNER = "ALERT_OWNER"
    ESCALATE = "ESCALATE"
    NO_ACTION = "NO_ACTION"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EvidenceSource(StrEnum):
    LOGS = "logs"
    SQL = "sql"
    PIPELINE_HISTORY = "pipeline_history"
    SCHEMA = "schema"
    METRICS = "metrics"


# --------------------------------------------------------------------------- #
# Inbound: the alert that starts everything
# --------------------------------------------------------------------------- #
class IncidentCreate(BaseModel):
    """The alert payload. This is the system's public input contract.

    Matches the shape an orchestrator (Airflow, Dagster, a CloudWatch alarm)
    would POST when a task fails.
    """

    model_config = ConfigDict(extra="ignore")

    pipeline: str = Field(min_length=1, description="Logical pipeline name")
    run_id: str = Field(min_length=1)
    status: Literal["FAILED"] = "FAILED"
    timestamp: datetime
    error: str | None = Field(default=None, description="Exception text if captured")
    logs: str | None = Field(default=None, description="Inline log excerpt, if any")
    # The dataset the pipeline writes. Optional because not every orchestrator
    # sends it — but when present it saves the agent from guessing a table name,
    # and guessing wrong costs a tool call and produces a confusing failure.
    table: str | None = Field(default=None, description="Target dataset, if the alert names it")

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        # A naive timestamp compared against tz-aware ones raises at runtime, far
        # from the cause. Normalise at the boundary instead.
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v


class Incident(BaseModel):
    """A persisted incident.

    Deliberately **not** a subclass of :class:`IncidentCreate`. The alert's
    ``status`` is a fixed ``"FAILED"`` — it is the wire format an orchestrator
    sends — whereas an incident's ``status`` is its lifecycle (``OPEN``,
    ``INVESTIGATING``, ``AWAITING_APPROVAL``, …). Sharing the field name across
    an inheritance boundary silently conflates the two, so the models are kept
    separate and the collision cannot happen.
    """

    model_config = ConfigDict(extra="ignore")

    incident_id: str
    pipeline: str
    run_id: str
    status: IncidentStatus = IncidentStatus.OPEN
    timestamp: datetime
    error: str | None = None
    logs: str | None = None
    table: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @field_validator("timestamp")
    @classmethod
    def _ensure_tz(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v


# --------------------------------------------------------------------------- #
# Node 1 — triage
# --------------------------------------------------------------------------- #
class Triage(BaseModel):
    """Fast, cheap classification before any expensive investigation.

    Its job is to decide *what kind of thing went wrong* and *how urgent it is*,
    and to seed a hypothesis for the investigator to confirm or kill.
    """

    model_config = ConfigDict(extra="ignore")

    category: FailureCategory
    severity: Severity
    initial_hypothesis: str
    rationale: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


# --------------------------------------------------------------------------- #
# Node 2 — investigation
# --------------------------------------------------------------------------- #
class Evidence(BaseModel):
    """One observation from one tool.

    ``supports``/``refutes`` are what make the evidence list usable for
    reasoning rather than just being a pile of facts: they record what the
    observation means for the current hypothesis.
    """

    model_config = ConfigDict(extra="ignore")

    source: EvidenceSource
    tool: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    supports: str | None = None
    refutes: str | None = None
    observed_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Node 3 — root cause
# --------------------------------------------------------------------------- #
class RootCause(BaseModel):
    """The agent's conclusion, plus the reasoning behind it.

    ``is_conclusive`` is what the graph branches on: an inconclusive verdict
    sends the workflow back for another investigation round rather than
    producing a confident-sounding guess. Refusing to answer is a valid,
    first-class outcome.
    """

    model_config = ConfigDict(extra="ignore")

    root_cause: str
    category: FailureCategory
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_conclusive: bool = False
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str = ""
    reasoning: str = ""

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))


# --------------------------------------------------------------------------- #
# Node 4 — remediation
# --------------------------------------------------------------------------- #
class Remediation(BaseModel):
    """A proposed fix.

    ``requires_approval`` defaults to **True** and the graph is built so that a
    proposal reaching the approval node stops there. The default must be safe:
    a plan that fails to set the flag gets a human, not an unattended rerun.
    """

    model_config = ConfigDict(extra="ignore")

    action: RemediationAction
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    risk: RiskLevel = RiskLevel.MEDIUM
    requires_approval: bool = True
    rationale: str = ""
    expected_effect: str = ""
    rollback: str = ""


class ApprovalDecision(BaseModel):
    """A human's verdict on a proposed remediation."""

    approved: bool
    approver: str = Field(min_length=1)
    note: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


# --------------------------------------------------------------------------- #
# Node 5 — report
# --------------------------------------------------------------------------- #
class ReportStep(BaseModel):
    """One entry in the incident timeline."""

    at: datetime = Field(default_factory=utcnow)
    node: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)


class IncidentReport(BaseModel):
    """The deliverable a human actually reads.

    Carries its own telemetry (``llm_calls``, ``tool_calls``, ``duration_ms``,
    ``model``) so that a report is self-describing about how much it cost to
    produce. Latency and token usage are evaluation dimensions, and this is
    where they attach to the artifact.
    """

    incident_id: str
    pipeline: str
    run_id: str
    status: IncidentStatus
    severity: Severity | None = None
    category: FailureCategory | None = None
    summary: str = ""
    root_cause: RootCause | None = None
    remediation: Remediation | None = None
    approval: ApprovalDecision | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    timeline: list[ReportStep] = Field(default_factory=list)
    investigation_rounds: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    model: str = ""
    generated_at: datetime = Field(default_factory=utcnow)
