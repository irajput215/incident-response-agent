"""The graph's nodes: triage, investigate, assess, remediate, approve, report.

Each node is a method on :class:`AgentNodes`, which holds the dependencies
(``llm``, tool registry, repository, settings). That shape is deliberate — a
node is then callable in a test with no graph, no checkpointer and no HTTP, and
the graph is a thin wiring layer rather than the place logic hides.

Three behaviours here are the ones worth defending:

**Every node returns a partial update, and ``_finish`` is the only place that
assembles one.** Timeline, telemetry and the persisted step record are appended
in a single helper, so it is impossible to add a node that forgets to record
itself.

**A failing LLM degrades; it does not abort.** Triage, root cause and
remediation each catch :class:`LLMError` and fall back to the deterministic
analyst, recording the degradation in ``errors`` and in the step detail. The
run continues with a weaker answer and a visible note saying so — which is
strictly better than a 2am incident with no analysis at all.

**The investigation loop is bounded three ways**: by LLM calls per incident, by
tool iterations per round, and by investigation rounds. An agent with an
unbounded loop is an agent with an unbounded bill.
"""
from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Protocol

from langgraph.types import interrupt

from app.agents import prompts
from app.agents.state import IncidentState, append_timeline
from app.config import Settings, get_settings
from app.llm import HeuristicLLM, LLMClient, LLMError
from app.llm.base import Message
from app.observability import METRICS, Metrics, get_logger, log
from app.schemas import (
    ApprovalDecision,
    Evidence,
    EvidenceSource,
    IncidentStatus,
    Remediation,
    RemediationAction,
    RootCause,
    Triage,
)
from app.tools import ToolRegistry
from app.tracing import traceable

_log = get_logger("app.agents.nodes")

# How many tool-calling turns one investigation round may take. A round has to
# fit the full evidence-gathering plan — read the log, check history, read the
# dependency metadata, inspect the partition, measure the data — plus the follow-up
# call that chases an upstream pipeline the metadata revealed. Ten leaves headroom
# without becoming an unbounded loop.
MAX_TOOL_ITERATIONS = 12


class BudgetExceeded(RuntimeError):
    """The per-incident LLM budget was exhausted.

    Raised rather than swallowed: the whole point of a ceiling is that hitting
    it is a visible failure, not a system that quietly keeps spending.
    """


class Remediator(Protocol):
    """Executes an approved remediation.

    A protocol so that the seam to a real orchestrator (Airflow, Dagster, a
    Step Functions call) is a class, not a rewrite.
    """

    def execute(self, remediation: Remediation, *, incident_id: str) -> dict[str, Any]: ...


class SimulatedRemediator:
    """Records the intended action without touching a real pipeline estate.

    ⚠️ **This does not remediate anything.** There is no orchestrator behind the
    simulated estate to rerun. It exists so the approval → execution seam is
    real and testable, and it is deliberately loud about being simulated so the
    gap cannot be mistaken for a working integration.
    """

    def execute(self, remediation: Remediation, *, incident_id: str) -> dict[str, Any]:
        log(
            _log,
            30,
            "remediation_simulated",
            incident_id=incident_id,
            action=remediation.action.value,
            target=remediation.target,
        )
        return {
            "simulated": True,
            "action": remediation.action.value,
            "target": remediation.target,
            "detail": (
                "No orchestrator is connected in this environment; the action was "
                "recorded, not performed."
            ),
        }


class AgentNodes:
    """The workflow's nodes, bound to their dependencies."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        registry: ToolRegistry,
        settings: Settings | None = None,
        metrics: Metrics | None = None,
        repository: Any | None = None,
        remediator: Remediator | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.settings = settings or get_settings()
        self.metrics = metrics if metrics is not None else METRICS
        self.repo = repository
        self.remediator = remediator or SimulatedRemediator()
        self._fallback = HeuristicLLM()

    # --- shared plumbing ---------------------------------------------------
    def _budget_left(self, state: IncidentState) -> int:
        return self.settings.llm_max_calls_per_incident - int(state.get("llm_calls", 0))

    def _spend(self, state: IncidentState) -> None:
        if self._budget_left(state) <= 0:
            raise BudgetExceeded(
                f"LLM call budget of {self.settings.llm_max_calls_per_incident} "
                "exhausted for this incident"
            )

    def _record_step(
        self,
        state: IncidentState,
        node: str,
        *,
        ok: bool,
        summary: str,
        detail: dict[str, Any] | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        investigation_id = state.get("investigation_id")
        if not investigation_id or self.repo is None:
            return
        try:
            self.repo.record_step(
                investigation_id,
                seq=len(state.get("timeline", [])),
                node=node,
                ok=ok,
                summary=summary,
                detail=detail,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # persistence must never break the investigation
            log(_log, 30, "step_record_failed", node=node, error=str(exc)[:200])

    def _finish(
        self,
        state: IncidentState,
        node: str,
        *,
        summary: str,
        updates: dict[str, Any],
        started: float,
        ok: bool = True,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble a node's return value, recording it as a step."""
        duration_ms = (perf_counter() - started) * 1000
        self.metrics.observe(f"agent.{node}.ms", duration_ms)
        self._record_step(
            state, node, ok=ok, summary=summary, detail=detail, duration_ms=duration_ms
        )
        return {
            "timeline": append_timeline(state, node, summary, **(detail or {})),
            **updates,
        }

    def _fallback_or_raise(
        self,
        node: str,
        schema: type,
        messages: list[Message],
        errors: list[str],
        exc: LLMError,
    ) -> Any:
        """Degrade to the deterministic analyst, and say so in the record."""
        degraded = f"{node}: model call failed, used the offline analyst ({exc})"
        log(_log, 30, "node_degraded_to_heuristic", node=node, error=str(exc)[:200])
        self.metrics.incr(f"agent.{node}.degraded")
        errors.append(degraded)
        return self._fallback.structured(messages, schema)

    # --- node 1: triage ----------------------------------------------------
    @traceable(run_type="chain", name="node.triage")
    def triage(self, state: IncidentState) -> dict[str, Any]:
        """Classify the failure and seed a hypothesis. Cheap, and runs first."""
        started = perf_counter()
        incident = state["incident"]
        messages = prompts.triage_messages(incident)
        errors = list(state.get("errors", []))
        self._spend(state)

        try:
            triage = self.llm.structured(messages, Triage)
        except LLMError as exc:
            triage = self._fallback_or_raise("triage", Triage, messages, errors, exc)

        assert isinstance(triage, Triage)
        log(
            _log,
            20,
            "incident_triaged",
            incident_id=state.get("incident_id"),
            category=triage.category.value,
            severity=triage.severity.value,
            confidence=triage.confidence,
        )

        if self.repo is not None and state.get("incident_id"):
            try:
                self.repo.apply_triage(state["incident_id"], triage)
            except Exception as exc:
                log(_log, 30, "triage_persist_failed", error=str(exc)[:200])

        return self._finish(
            state,
            "triage",
            summary=(
                f"Classified as {triage.category.value} ({triage.severity.value}); "
                f"hypothesis: {triage.initial_hypothesis}"
            ),
            updates={
                "triage": triage.model_dump(mode="json"),
                "llm_calls": int(state.get("llm_calls", 0)) + 1,
                "errors": errors,
                "status": IncidentStatus.INVESTIGATING.value,
            },
            started=started,
            detail={"category": triage.category.value, "severity": triage.severity.value},
        )

    # --- node 2: investigate ----------------------------------------------
    @traceable(run_type="chain", name="node.investigate")
    def investigate(self, state: IncidentState) -> dict[str, Any]:
        """Run one round of tool-assisted evidence gathering."""
        started = perf_counter()
        incident = state["incident"]
        triage = state.get("triage")
        collected = list(state.get("evidence", []))
        round_number = int(state.get("rounds", 0)) + 1

        messages = prompts.investigation_messages(
            incident, triage, collected, round_number=round_number
        )
        seen = {(e.get("tool"), e.get("summary")) for e in collected}
        new_evidence: list[dict[str, Any]] = []

        llm_calls = 0
        tool_calls = 0
        tokens_in = 0
        tokens_out = 0
        errors = list(state.get("errors", []))

        for _ in range(MAX_TOOL_ITERATIONS):
            if self._budget_left(state) - llm_calls <= 0:
                errors.append(
                    "investigation stopped early: LLM budget exhausted mid-round"
                )
                break
            try:
                response = self.llm.complete(messages, tools=self.registry.specs())
            except LLMError as exc:
                errors.append(f"investigation: model call failed ({exc})")
                log(_log, 30, "investigation_llm_failed", error=str(exc)[:200])
                break

            llm_calls += 1
            tokens_in += response.tokens_in
            tokens_out += response.tokens_out

            if not response.wants_tools:
                messages.append({"role": "assistant", "content": response.text})
                break

            # Echo the assistant's tool request back into the conversation.
            # Providers require the assistant turn to precede its tool results,
            # and our offline analyst counts `role: "tool"` messages to decide
            # where it is in its plan.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )

            for call in response.tool_calls:
                if call.parse_error:
                    result_payload = {
                        "ok": False,
                        "tool": call.name,
                        "summary": call.parse_error,
                        "error": call.parse_error,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps(result_payload),
                        }
                    )
                    errors.append(f"{call.name}: {call.parse_error}")
                    continue

                result = self.registry.call(call.name, call.arguments)
                tool_calls += 1
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result.to_dict(), default=str),
                    }
                )

                if self.repo is not None and state.get("investigation_id"):
                    try:
                        self.repo.record_tool_call(
                            state["investigation_id"],
                            tool=call.name,
                            arguments=call.arguments,
                            ok=result.ok,
                            error=result.error,
                            duration_ms=result.duration_ms,
                        )
                    except Exception as exc:
                        log(_log, 30, "tool_call_persist_failed", error=str(exc)[:200])

                if result.ok:
                    key = (call.name, result.summary)
                    if key not in seen:
                        seen.add(key)
                        evidence = self._to_evidence(call.name, result.summary, result.data)
                        new_evidence.append(evidence.model_dump(mode="json"))
                        collected.append(new_evidence[-1])
                        if self.repo is not None and state.get("investigation_id"):
                            try:
                                self.repo.record_evidence(state["investigation_id"], evidence)
                            except Exception as exc:
                                log(_log, 30, "evidence_persist_failed", error=str(exc)[:200])
                else:
                    errors.append(f"{call.name} failed: {result.error}")

        all_evidence = [*state.get("evidence", []), *new_evidence]
        return self._finish(
            state,
            "investigate",
            summary=(
                f"Round {round_number}: {tool_calls} tool call(s), "
                f"{len(new_evidence)} new observation(s) "
                f"({len(all_evidence)} total)"
            ),
            updates={
                "evidence": all_evidence,
                "rounds": round_number,
                "llm_calls": int(state.get("llm_calls", 0)) + llm_calls,
                "tool_calls": int(state.get("tool_calls", 0)) + tool_calls,
                "tokens_in": int(state.get("tokens_in", 0)) + tokens_in,
                "tokens_out": int(state.get("tokens_out", 0)) + tokens_out,
                "errors": errors,
            },
            started=started,
            detail={"round": round_number, "tool_calls": tool_calls, "new_evidence": len(new_evidence)},
        )

    def _to_evidence(self, tool: str, summary: str, data: dict[str, Any]) -> Evidence:
        source = (
            self.registry.get(tool).source if tool in self.registry else EvidenceSource.SQL
        )
        return Evidence(
            source=source,
            tool=tool,
            summary=summary,
            detail=data,
        )

    # --- node 3: assess ----------------------------------------------------
    @traceable(run_type="chain", name="node.assess_root_cause")
    def assess_root_cause(self, state: IncidentState) -> dict[str, Any]:
        """Decide whether the evidence establishes a cause. May say no."""
        started = perf_counter()
        messages = prompts.root_cause_messages(
            state["incident"], state.get("triage"), state.get("evidence", [])
        )
        errors = list(state.get("errors", []))
        self._spend(state)

        try:
            root_cause = self.llm.structured(messages, RootCause)
        except LLMError as exc:
            root_cause = self._fallback_or_raise(
                "root_cause", RootCause, messages, errors, exc
            )

        assert isinstance(root_cause, RootCause)

        conclusive = (
            root_cause.is_conclusive
            and root_cause.confidence >= self.settings.root_cause_confidence_threshold
        )
        # "Conclusive" is the graph's decision, not the model's: a model that
        # claims certainty below the configured threshold still gets sent back
        # for more evidence.
        rounds = int(state.get("rounds", 0))
        max_rounds = int(state.get("max_rounds", self.settings.max_investigation_rounds))
        exhausted = rounds >= max_rounds

        if conclusive:
            status = IncidentStatus.INVESTIGATING.value  # about to remediate
            summary = f"Root cause established: {root_cause.root_cause}"
        elif exhausted:
            status = IncidentStatus.UNRESOLVED.value
            summary = (
                f"No confident root cause after {rounds} round(s); "
                "recording the incident as unresolved"
            )
        else:
            status = IncidentStatus.INVESTIGATING.value
            summary = (
                f"Inconclusive (confidence {root_cause.confidence:.2f}); "
                f"gathering more evidence (round {rounds} of {max_rounds})"
            )

        log(
            _log,
            20,
            "root_cause_assessed",
            incident_id=state.get("incident_id"),
            conclusive=conclusive,
            confidence=root_cause.confidence,
            rounds=rounds,
        )

        return self._finish(
            state,
            "assess_root_cause",
            summary=summary,
            updates={
                "root_cause": root_cause.model_dump(mode="json"),
                "needs_more_evidence": not conclusive and not exhausted,
                "status": status,
                "llm_calls": int(state.get("llm_calls", 0)) + 1,
                "errors": errors,
            },
            started=started,
            detail={
                "conclusive": conclusive,
                "confidence": root_cause.confidence,
                "category": root_cause.category.value,
            },
        )

    # --- node 4: remediation ----------------------------------------------
    @traceable(run_type="chain", name="node.plan_remediation")
    def plan_remediation(self, state: IncidentState) -> dict[str, Any]:
        """Propose a fix for a confirmed root cause."""
        started = perf_counter()
        root_cause = state.get("root_cause") or {}
        messages = prompts.remediation_messages(
            state["incident"], root_cause, state.get("evidence", [])
        )
        errors = list(state.get("errors", []))
        self._spend(state)

        try:
            remediation = self.llm.structured(messages, Remediation)
        except LLMError as exc:
            remediation = self._fallback_or_raise(
                "plan_remediation", Remediation, messages, errors, exc
            )

        assert isinstance(remediation, Remediation)

        # A low-risk, non-mutating action does not need a human. Everything else
        # does — and the default in the schema is already True, so the safe
        # behaviour is what you get if this logic is ever wrong.
        status = (
            IncidentStatus.AWAITING_APPROVAL.value
            if remediation.requires_approval
            else IncidentStatus.REMEDIATING.value
        )

        return self._finish(
            state,
            "plan_remediation",
            summary=(
                f"Proposed {remediation.action.value} on {remediation.target} "
                f"(risk {remediation.risk.value}, approval {'required' if remediation.requires_approval else 'not required'})"
            ),
            updates={
                "remediation": remediation.model_dump(mode="json"),
                "status": status,
                "llm_calls": int(state.get("llm_calls", 0)) + 1,
                "errors": errors,
            },
            started=started,
            detail={
                "action": remediation.action.value,
                "target": remediation.target,
                "requires_approval": remediation.requires_approval,
            },
        )

    # --- node 5: human approval -------------------------------------------
    @traceable(run_type="chain", name="node.approval")
    def approval(self, state: IncidentState) -> dict[str, Any]:
        """Pause the graph until a human decides.

        ``interrupt`` suspends execution and persists the state through the
        checkpointer; the value passed to ``Command(resume=...)`` becomes this
        call's return value. Nothing dangerous has happened before this point —
        the graph is built so that a write-consuming action cannot be reached
        without passing through here.
        """
        started = perf_counter()
        remediation = state.get("remediation") or {}

        decision_raw = interrupt(
            {
                "type": "approval_required",
                "incident_id": state.get("incident_id"),
                "investigation_id": state.get("investigation_id"),
                "question": (
                    f"Approve {remediation.get('action')} on {remediation.get('target')}?"
                ),
                "remediation": remediation,
                "root_cause": (state.get("root_cause") or {}).get("root_cause", ""),
            }
        )

        # Accept either a full decision object or a bare boolean, because an
        # operator clicking a button should not have to construct a payload.
        if isinstance(decision_raw, bool):
            decision_raw = {"approved": decision_raw, "approver": "unknown"}
        if isinstance(decision_raw, dict) and "approver" not in decision_raw:
            decision_raw = {**decision_raw, "approver": "unknown"}

        decision = ApprovalDecision.model_validate(decision_raw)

        if self.repo is not None and state.get("investigation_id"):
            try:
                self.repo.record_approval(state["investigation_id"], decision)
            except Exception as exc:
                log(_log, 30, "approval_persist_failed", error=str(exc)[:200])

        status = (
            IncidentStatus.REMEDIATING.value
            if decision.approved
            else IncidentStatus.UNRESOLVED.value
        )
        log(
            _log,
            20,
            "remediation_decision",
            incident_id=state.get("incident_id"),
            approved=decision.approved,
            approver=decision.approver,
        )

        return self._finish(
            state,
            "approval",
            summary=(
                f"{'Approved' if decision.approved else 'Rejected'} by {decision.approver}"
                + (f": {decision.note}" if decision.note else "")
            ),
            updates={
                "approval": decision.model_dump(mode="json"),
                "status": status,
            },
            started=started,
            detail={"approved": decision.approved, "approver": decision.approver},
        )

    # --- node 6: execute ---------------------------------------------------
    @traceable(run_type="chain", name="node.execute_remediation")
    def execute_remediation(self, state: IncidentState) -> dict[str, Any]:
        """Run the approved action through the remediator seam."""
        started = perf_counter()
        remediation = Remediation.model_validate(state.get("remediation") or {})
        outcome = self.remediator.execute(remediation, incident_id=state.get("incident_id", ""))

        return self._finish(
            state,
            "execute_remediation",
            summary=(
                f"{remediation.action.value} on {remediation.target} "
                + ("recorded (no orchestrator connected)" if outcome.get("simulated") else "executed")
            ),
            updates={
                "remediation_executed": True,
                "status": IncidentStatus.RESOLVED.value,
            },
            started=started,
            detail=outcome,
        )

    # --- node 7: report ----------------------------------------------------
    @traceable(run_type="chain", name="node.report")
    def report(self, state: IncidentState) -> dict[str, Any]:
        """Write the incident report and persist it."""
        from app.agents.state import build_report

        started = perf_counter()
        status = IncidentStatus(state.get("status", IncidentStatus.OPEN.value))

        # A rejected remediation is not a resolved incident; and an inconclusive
        # investigation must not be dressed up as one.
        root = state.get("root_cause") or {}
        if status is IncidentStatus.UNRESOLVED.value or (root and not root.get("is_conclusive")):
            final = IncidentStatus.UNRESOLVED
        else:
            final = status

        report = build_report(
            {**state, "status": final.value},
            model=self.llm.model,
            duration_ms=(perf_counter() - started) * 1000,
        )

        if self.repo is not None and state.get("investigation_id"):
            try:
                self.repo.save_report(state["investigation_id"], report)
                if state.get("incident_id"):
                    self.repo.set_incident_status(
                        state["incident_id"],
                        final,
                        severity=report.severity.value if report.severity else None,
                        category=report.category.value if report.category else None,
                    )
            except Exception as exc:
                log(_log, 30, "report_persist_failed", error=str(exc)[:200])

        log(
            _log,
            20,
            "incident_reported",
            incident_id=state.get("incident_id"),
            status=final.value,
            rounds=report.investigation_rounds,
            tool_calls=report.tool_calls,
        )

        return self._finish(
            state,
            "report",
            summary=f"Report generated with status {final.value}",
            updates={
                "report": report.model_dump(mode="json"),
                "status": final.value,
            },
            started=started,
            detail={"status": final.value, "evidence_count": len(report.evidence)},
        )

    # --- routing -----------------------------------------------------------
    def route_after_assessment(self, state: IncidentState) -> str:
        """Where to go once the evidence has been judged."""
        root = state.get("root_cause") or {}
        conclusive = bool(root.get("is_conclusive")) and float(
            root.get("confidence", 0.0)
        ) >= self.settings.root_cause_confidence_threshold
        if conclusive:
            return "plan_remediation"
        if state.get("needs_more_evidence") and int(state.get("rounds", 0)) < int(
            state.get("max_rounds", self.settings.max_investigation_rounds)
        ):
            return "investigate"
        return "report"

    def route_after_remediation(self, state: IncidentState) -> str:
        """Approval first for anything that touches data."""
        remediation = state.get("remediation") or {}
        if remediation.get("requires_approval", True):
            return "approval"
        return "execute_remediation"

    def route_after_approval(self, state: IncidentState) -> str:
        approval = state.get("approval") or {}
        return "execute_remediation" if approval.get("approved") else "report"


__all__ = [
    "MAX_TOOL_ITERATIONS",
    "AgentNodes",
    "BudgetExceeded",
    "RemediationAction",
    "Remediator",
    "SimulatedRemediator",
]
