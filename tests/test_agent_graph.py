"""End-to-end tests for the workflow — and the evaluation gate.

The first test in this file is the one that matters most: every scenario in the
catalogue, run through the real graph against a real database, must reach the
right root cause and propose the right action. It is simultaneously the demo, the
evaluation suite and the CI gate, so a regression cannot pass unnoticed in one
place while the others stay green.

The rest of the file tests the behaviours that make the workflow an agent rather
than a script: it pauses for a human, it survives a broken model, it bounds its
own spending, it refuses to answer when the evidence is thin, and a failed run
leaves the platform usable.
"""
from __future__ import annotations

from datetime import UTC, datetime

from app.agents import IncidentAgent, initial_state
from app.llm import LLMError
from app.scenarios import SCENARIOS, SCENARIOS_BY_ID
from app.schemas import ApprovalDecision, IncidentCreate, IncidentStatus

FIXED_TS = datetime(2026, 9, 21, 2, 14, tzinfo=UTC)


class BrokenLLM:
    """A model that is always down — the degradation path's worst case."""

    model = "broken"

    def complete(self, *args: object, **kwargs: object) -> object:
        raise LLMError("model is down")

    def structured(self, *args: object, **kwargs: object) -> object:
        raise LLMError("model is down")


def alert_for(scenario_id: str) -> IncidentCreate:
    return IncidentCreate.model_validate(SCENARIOS_BY_ID[scenario_id].incident)


def run_to_completion(agent: IncidentAgent, alert: IncidentCreate, approve: bool = True):
    """Investigate, and auto-approve if the graph pauses."""
    outcome = agent.investigate(alert)
    if outcome.awaiting_approval and approve:
        outcome = agent.resume(
            outcome.investigation_id,
            ApprovalDecision(approved=True, approver="oncall@acme.example"),
        )
    return outcome


# --------------------------------------------------------------------------- #
# THE GATE
# --------------------------------------------------------------------------- #
def test_every_scenario_reaches_the_expected_root_cause(agent: IncidentAgent, repository) -> None:
    """The evaluation gate. 5/5, with the report's own category field."""
    results = []
    for scenario in SCENARIOS:
        outcome = run_to_completion(agent, alert_for(scenario.id))
        report = outcome.report
        assert report is not None, f"{scenario.id} produced no report"
        assert report.category is not None
        results.append((scenario.id, report.category.value, scenario.expected_category))

    wrong = [(sid, got, want) for sid, got, want in results if got != want]
    assert not wrong, f"root-cause regression: {wrong}"
    assert len(results) == 5


def test_every_scenario_proposes_the_expected_action(agent: IncidentAgent) -> None:
    wrong = []
    for scenario in SCENARIOS:
        outcome = run_to_completion(agent, alert_for(scenario.id))
        assert outcome.report is not None and outcome.report.remediation is not None
        action = outcome.report.remediation.action.value
        if action != scenario.expected_action:
            wrong.append((scenario.id, action, scenario.expected_action))
    assert not wrong, f"remediation regression: {wrong}"


def test_the_upstream_failure_is_attributed_to_the_upstream_pipeline(agent: IncidentAgent) -> None:
    """The scenario's punchline: the fix belongs to another team's pipeline."""
    outcome = run_to_completion(agent, alert_for("missing_partition"))
    assert outcome.report is not None
    assert outcome.report.category.value == "UPSTREAM_DEPENDENCY_FAILURE"
    assert outcome.report.remediation is not None
    assert outcome.report.remediation.action.value == "RERUN_UPSTREAM"
    assert outcome.report.remediation.target == "claims_ingestion"


def test_investigations_gather_a_real_amount_of_evidence(agent: IncidentAgent) -> None:
    outcome = run_to_completion(agent, alert_for("missing_partition"))
    assert outcome.report is not None
    assert len(outcome.report.evidence) >= 5
    assert outcome.report.tool_calls >= 5


# --------------------------------------------------------------------------- #
# the human in the loop
# --------------------------------------------------------------------------- #
def test_a_data_changing_action_pauses_for_approval(agent: IncidentAgent) -> None:
    outcome = agent.investigate(alert_for("missing_partition"))
    assert outcome.interrupted is True
    assert outcome.awaiting_approval is True
    assert outcome.status is IncidentStatus.AWAITING_APPROVAL
    assert outcome.approval_request is not None
    assert outcome.approval_request["remediation"]["action"] == "RERUN_UPSTREAM"
    assert outcome.report is None, "nothing is executed before a human decides"


def test_the_pause_survives_because_state_is_checkpointed(agent: IncidentAgent, repository) -> None:
    """The investigation row is paused, not finished, and its cost is recorded."""
    outcome = agent.investigate(alert_for("missing_partition"))
    row = repository.get_investigation(outcome.investigation_id)
    assert row["status"] == IncidentStatus.AWAITING_APPROVAL.value
    assert row["finished_at"] is None
    assert row["llm_calls"] > 0, "the evidence gathered before the pause is not thrown away"


def test_approving_resolves_the_incident(agent: IncidentAgent) -> None:
    outcome = agent.investigate(alert_for("missing_partition"))
    resumed = agent.resume(
        outcome.investigation_id,
        ApprovalDecision(approved=True, approver="oncall@acme.example", note="confirmed"),
    )
    assert resumed.status is IncidentStatus.RESOLVED
    assert resumed.report is not None
    assert resumed.report.approval is not None
    assert resumed.report.approval.approved is True
    assert resumed.report.approval.approver == "oncall@acme.example"


def test_rejecting_an_incident_leaves_it_unresolved(agent: IncidentAgent) -> None:
    """A rejected remediation is not a fix, and must not be reported as one."""
    outcome = agent.investigate(alert_for("missing_partition"))
    resumed = agent.resume(
        outcome.investigation_id,
        ApprovalDecision(approved=False, approver="oncall@acme.example", note="not safe"),
    )
    assert resumed.status is IncidentStatus.UNRESOLVED
    assert resumed.report is not None
    assert resumed.report.approval is not None
    assert resumed.report.approval.approved is False


def test_a_bare_boolean_decision_is_accepted(agent: IncidentAgent) -> None:
    """An operator clicking a button should not have to build a payload."""
    outcome = agent.investigate(alert_for("missing_partition"))
    resumed = agent.resume(outcome.investigation_id, True)
    assert resumed.status is IncidentStatus.RESOLVED
    assert resumed.report is not None
    assert resumed.report.approval is not None
    assert resumed.report.approval.approver == "unknown"


def test_a_low_risk_action_does_not_need_a_human(agent: IncidentAgent) -> None:
    """Approval is required for data changes, not for notifying an owner."""
    outcome = agent.investigate(alert_for("schema_change"))
    assert outcome.interrupted is False
    assert outcome.status is IncidentStatus.RESOLVED


def test_the_approval_is_audited(repository, agent: IncidentAgent) -> None:
    outcome = agent.investigate(alert_for("missing_partition"))
    agent.resume(
        outcome.investigation_id, ApprovalDecision(approved=True, approver="auditor@acme.example")
    )
    approval = repository.latest_approval(outcome.investigation_id)
    assert approval is not None
    assert approval["approver"] == "auditor@acme.example"


def test_hitting_the_budget_fails_the_run_loudly(seeded_estate, repository, metrics, llm) -> None:
    """A ceiling that does not fail is not a ceiling."""
    tight = seeded_estate.model_copy(update={"llm_max_calls_per_incident": 1})
    agent = IncidentAgent(settings=tight, repository=repository, metrics=metrics, llm=llm)

    outcome = agent.investigate(alert_for("missing_partition"))
    assert outcome.status is IncidentStatus.FAILED
    assert outcome.error is not None and "budget" in outcome.error.lower()

    row = repository.get_investigation(outcome.investigation_id)
    assert row["status"] == IncidentStatus.FAILED.value
    assert row["finished_at"] is not None


# --------------------------------------------------------------------------- #
# degradation and resilience
# --------------------------------------------------------------------------- #
def test_a_broken_model_degrades_instead_of_aborting(seeded_estate, repository, metrics) -> None:
    """An incident-response system that dies when its model dies is not useful."""
    agent = IncidentAgent(
        settings=seeded_estate, repository=repository, metrics=metrics, llm=BrokenLLM()
    )
    outcome = run_to_completion(agent, alert_for("missing_partition"))

    assert outcome.status in (IncidentStatus.RESOLVED, IncidentStatus.UNRESOLVED)
    assert outcome.report is not None
    # The degradation must be recorded, not silent.
    assert any("offline analyst" in e for e in outcome.state.get("errors", []))
    assert metrics.snapshot()["counters"].get("agent.triage.degraded", 0) >= 1


def test_an_inconclusive_investigation_is_not_dressed_up_as_an_answer(
    seeded_estate, repository, metrics, llm
) -> None:
    """A pipeline nobody has ever heard of must end UNRESOLVED, not guessed at."""
    agent = IncidentAgent(settings=seeded_estate, repository=repository, metrics=metrics, llm=llm)
    outcome = run_to_completion(
        agent,
        IncidentCreate(
            pipeline="never_seen_pipeline",
            run_id="run_00001",
            timestamp=FIXED_TS,
            error="xyzzy plugh",
        ),
    )
    assert outcome.status is IncidentStatus.UNRESOLVED
    assert outcome.report is not None
    assert outcome.report.category.value == "UNKNOWN"


def test_the_investigation_loop_terminates_at_max_rounds(
    seeded_estate, repository, metrics, llm
) -> None:
    capped = seeded_estate.model_copy(update={"max_investigation_rounds": 1})
    agent = IncidentAgent(settings=capped, repository=repository, metrics=metrics, llm=llm)
    outcome = run_to_completion(
        agent,
        IncidentCreate(pipeline="unknown_one", run_id="r1", timestamp=FIXED_TS, error="xyzzy"),
    )
    assert outcome.state.get("rounds", 0) <= 1


def test_a_failed_run_leaves_the_platform_usable(agent: IncidentAgent) -> None:
    """Resilience stated as an executable contract.

    The third assertion is the point: anyone who later gives the agent
    process-wide mutable state gets caught here rather than in production.
    """
    broken = IncidentCreate(
        pipeline="pipeline_that_does_not_exist",
        run_id="run_broken",
        timestamp=FIXED_TS,
        error="xyzzy",
    )
    first = run_to_completion(agent, broken)
    assert first.report is not None or first.status is IncidentStatus.FAILED

    # ...and the very next investigation still works.
    second = run_to_completion(agent, alert_for("db_connection"))
    assert second.status is not IncidentStatus.FAILED
    assert second.report is not None


def test_telemetry_is_recorded_on_the_report(agent: IncidentAgent) -> None:
    outcome = run_to_completion(agent, alert_for("spark_oom"))
    report = outcome.report
    assert report is not None
    assert report.llm_calls > 0
    assert report.tool_calls > 0
    assert report.tokens_in > 0
    assert report.model == "stub"
    assert report.investigation_rounds >= 1


def test_the_planner_is_recorded_not_inferred(agent: IncidentAgent, repository) -> None:
    """"Why did this plan look wrong?" deserves a first-class answer."""
    outcome = agent.investigate(alert_for("missing_partition"))
    row = repository.get_investigation(outcome.investigation_id)
    assert row["planner"] == "stub"


def test_the_timeline_covers_every_node_that_ran(agent: IncidentAgent) -> None:
    outcome = run_to_completion(agent, alert_for("missing_partition"))
    nodes = [entry["node"] for entry in outcome.state.get("timeline", [])]
    for expected in ("triage", "investigate", "assess_root_cause", "plan_remediation", "approval", "report"):
        assert expected in nodes, f"{expected} missing from the timeline"


def test_the_report_carries_the_whole_timeline(agent: IncidentAgent) -> None:
    outcome = run_to_completion(agent, alert_for("missing_partition"))
    assert outcome.report is not None
    assert len(outcome.report.timeline) >= 6
    assert outcome.report.summary


# --------------------------------------------------------------------------- #
# initial state
# --------------------------------------------------------------------------- #
def test_initial_state_sets_every_key_explicitly() -> None:
    """A node reading `state["evidence"]` must find a list, not a missing key."""
    state = initial_state({"pipeline": "x", "run_id": "y"})
    assert state["evidence"] == []
    assert state["errors"] == []
    assert state["timeline"] == []
    assert state["llm_calls"] == 0
    assert state["triage"] is None


def test_a_task_with_no_category_still_produces_a_report(agent: IncidentAgent) -> None:
    outcome = run_to_completion(
        agent,
        IncidentCreate(pipeline="mystery", run_id="r9", timestamp=FIXED_TS, error="who knows"),
    )
    assert outcome.report is not None
    assert outcome.report.status is IncidentStatus.UNRESOLVED
