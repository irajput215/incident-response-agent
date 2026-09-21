"""The orchestrator: one class that owns a whole investigation end to end.

The graph knows the workflow; it does not know about Postgres, incident records
or budget failures. This is the layer that:

1. records the alert and opens an investigation **before** any work happens
   (the two-phase write),
2. invokes the graph and interprets an interrupt as "waiting for a human",
3. closes the investigation out with its real cost, whatever happened,
4. converts every failure into a recorded status rather than an exception.

That last point is the contract that makes this operable: an incident-response
system that dies when the thing it is investigating dies is not useful at 2am.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agents.graph import build_graph
from app.agents.nodes import AgentNodes, BudgetExceeded
from app.agents.state import IncidentState, initial_state
from app.config import Settings, get_settings
from app.db import IncidentRepository
from app.llm import LLMClient, build_llm
from app.observability import METRICS, Metrics, get_logger, log
from app.schemas import (
    ApprovalDecision,
    IncidentCreate,
    IncidentReport,
    IncidentStatus,
    Triage,
)
from app.tools import ToolRegistry, build_context, build_registry

_log = get_logger("app.agents")


@dataclass
class InvestigationOutcome:
    """What came out of one attempt at one incident."""

    incident_id: str
    investigation_id: str
    status: IncidentStatus
    report: IncidentReport | None = None
    interrupted: bool = False
    approval_request: dict[str, Any] | None = None
    error: str | None = None
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def awaiting_approval(self) -> bool:
        return self.interrupted and self.status is IncidentStatus.AWAITING_APPROVAL


class IncidentAgent:
    """Wires the LLM, the tools, the database and the graph into one entry point."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        repository: IncidentRepository | None = None,
        metrics: Metrics | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.metrics = metrics if metrics is not None else METRICS
        self.llm = llm or build_llm(self.settings, metrics=self.metrics)
        self.registry = registry or build_registry(
            build_context(self.settings, metrics=self.metrics)
        )
        self.repo = repository
        # MemorySaver keeps the approval flow testable with no database; the
        # server swaps in a Postgres checkpointer so a pause survives a restart.
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()

        self.nodes = AgentNodes(
            llm=self.llm,
            registry=self.registry,
            settings=self.settings,
            metrics=self.metrics,
            repository=self.repo,
        )
        self.graph = build_graph(self.nodes, self.checkpointer)

    # --- entry points ------------------------------------------------------
    def investigate(self, alert: IncidentCreate, *, resume_from: str | None = None) -> InvestigationOutcome:
        """Investigate an incident from the alert, or resume a paused one."""
        if self.repo is None:
            raise RuntimeError("IncidentAgent requires a repository to investigate")

        incident = self.repo.create_incident(alert)
        investigation_id = self.repo.start_investigation(incident.incident_id, self.llm.model)

        log(
            _log,
            20,
            "incident_received",
            incident_id=incident.incident_id,
            pipeline=alert.pipeline,
            run_id=alert.run_id,
        )

        state = initial_state(
            alert.model_dump(mode="json"),
            incident_id=incident.incident_id,
            investigation_id=investigation_id,
            planner=self.llm.model,
            max_rounds=self.settings.max_investigation_rounds,
        )
        return self._run(state, investigation_id, incident.incident_id)

    def resume(self, investigation_id: str, decision: ApprovalDecision | bool) -> InvestigationOutcome:
        """Resume a paused investigation with a human decision.

        The ``thread_id`` is the investigation id, so the checkpointer restores
        exactly the state that was suspended — including evidence gathered
        before the pause, which is what makes the approve/reject endpoint
        stateless and safe to call from anywhere.
        """
        if self.repo is None:
            raise RuntimeError("IncidentAgent requires a repository to resume")
        record = self.repo.get_investigation(investigation_id)
        if record is None:
            raise KeyError(f"unknown investigation {investigation_id!r}")

        payload: Any = decision.model_dump(mode="json") if isinstance(decision, ApprovalDecision) else decision
        log(_log, 20, "investigation_resumed", investigation_id=investigation_id)
        return self._run(
            None,
            investigation_id,
            str(record["incident_id"]),
            resume=payload,
        )

    # --- the run -----------------------------------------------------------
    def _run(
        self,
        state: IncidentState | None,
        investigation_id: str,
        incident_id: str,
        *,
        resume: Any | None = None,
    ) -> InvestigationOutcome:
        config = {"configurable": {"thread_id": investigation_id}}

        try:
            if resume is None:
                assert state is not None
                result = self.graph.invoke(state, config=config)
            else:
                result = self.graph.invoke(Command(resume=resume), config=config)
        except BudgetExceeded as exc:
            return self._fail(investigation_id, incident_id, f"budget exceeded: {exc}", state)
        except Exception as exc:
            log(_log, 40, "investigation_crashed", investigation_id=investigation_id, error=str(exc)[:300])
            return self._fail(investigation_id, incident_id, f"{type(exc).__name__}: {exc}", state)

        return self._interpret(result, investigation_id, incident_id)

    def _interpret(
        self, result: dict[str, Any], investigation_id: str, incident_id: str
    ) -> InvestigationOutcome:
        interrupts = result.get("__interrupt__") or []
        counters = _counters(result)

        if interrupts:
            # Suspended on the approval node. The investigation is *not* finished
            # — it is waiting, which is why this does not set finished_at.
            payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            assert self.repo is not None
            self.repo.update_investigation(
                investigation_id, IncidentStatus.AWAITING_APPROVAL, **counters
            )
            if result.get("incident_id"):
                self.repo.set_incident_status(
                    str(result["incident_id"]), IncidentStatus.AWAITING_APPROVAL
                )
            log(
                _log,
                20,
                "investigation_awaiting_approval",
                investigation_id=investigation_id,
                action=(result.get("remediation") or {}).get("action"),
            )
            return InvestigationOutcome(
                incident_id=incident_id,
                investigation_id=investigation_id,
                status=IncidentStatus.AWAITING_APPROVAL,
                interrupted=True,
                approval_request=payload if isinstance(payload, dict) else {"payload": payload},
                state=dict(result),
            )

        status = IncidentStatus(result.get("status", IncidentStatus.OPEN.value))
        report = _report_of(result)
        assert self.repo is not None
        self.repo.finish_investigation(investigation_id, status, **counters)

        return InvestigationOutcome(
            incident_id=incident_id,
            investigation_id=investigation_id,
            status=status,
            report=report,
            state=dict(result),
        )

    def _fail(
        self,
        investigation_id: str,
        incident_id: str,
        error: str,
        state: IncidentState | None,
    ) -> InvestigationOutcome:
        assert self.repo is not None
        counters = _counters(state or {})
        self.repo.finish_investigation(
            investigation_id, IncidentStatus.FAILED, error=error, **counters
        )
        self.repo.set_incident_status(incident_id, IncidentStatus.FAILED)
        self.metrics.incr("agent.investigation_failed")
        return InvestigationOutcome(
            incident_id=incident_id,
            investigation_id=investigation_id,
            status=IncidentStatus.FAILED,
            error=error,
        )

    # --- helpers -----------------------------------------------------------
    def triage_preview(self, alert: IncidentCreate) -> Triage:
        """Triage without touching the database — used by ``POST /triage``."""
        from app.agents import prompts

        return self.llm.structured(
            prompts.triage_messages(alert.model_dump(mode="json")), Triage
        )


class _Counters(TypedDict):
    """The numbers every investigation records, wherever it ends up.

    A TypedDict rather than ``dict[str, int]`` so ``**counters`` can be unpacked
    into the repository methods: ``**dict[str, int]`` could contain any key, so
    the type checker rejects it, and rightly so.
    """

    rounds: int
    llm_calls: int
    tool_calls: int
    tokens_in: int
    tokens_out: int


def _counters(state: Mapping[str, Any]) -> _Counters:
    return _Counters(
        rounds=int(state.get("rounds", 0) or 0),
        llm_calls=int(state.get("llm_calls", 0) or 0),
        tool_calls=int(state.get("tool_calls", 0) or 0),
        tokens_in=int(state.get("tokens_in", 0) or 0),
        tokens_out=int(state.get("tokens_out", 0) or 0),
    )


def _report_of(result: dict[str, Any]) -> IncidentReport | None:
    raw = result.get("report")
    if not raw:
        return None
    try:
        return IncidentReport.model_validate(raw)
    except Exception:  # a malformed report must not crash the caller
        return None


__all__ = ["IncidentAgent", "InvestigationOutcome"]
