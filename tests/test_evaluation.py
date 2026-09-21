"""The evaluation suite, tested as a gate rather than as a feature.

Half of these tests are about the suite **failing correctly**, because that is
what a gate is for:

* a malformed suite file must raise, never silently fall back to a built-in list;
* a typo'd assertion key must raise, never silently never run;
* a task with an empty ``expect`` block must **fail**, not pass vacuously —
  ``all([])`` is ``True`` in Python, so this needs saying explicitly;
* the report must contain the actual numbers, because a red build that says only
  "check failed" costs a re-run to diagnose.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.evaluation.evaluators import Observation, run_evaluator
from app.evaluation.harness import EvaluationSuite, SuiteReport, TaskResult
from app.evaluation.tasks import SuiteConfigError, load_suite

MINIMAL_TASK = {
    "id": "t1",
    "scenario": "missing_partition",
    "expect": {"category": "UPSTREAM_DEPENDENCY_FAILURE"},
}


def write_suite(tmp_path: Path, document: dict | str) -> Path:
    path = tmp_path / "tasks.yaml"
    text = document if isinstance(document, str) else yaml.safe_dump(document)
    path.write_text(text, encoding="utf-8")
    return path


def minimal(document: dict) -> dict:
    return {"version": 1, **document}


# --------------------------------------------------------------------------- #
# loading: the real suite
# --------------------------------------------------------------------------- #
def test_the_project_suite_loads() -> None:
    tasks, defaults = load_suite("evals/tasks.yaml")
    assert len(tasks) >= 5
    assert defaults["max_write_attempts"] == 0
    assert {task.id for task in tasks} >= {
        "missing_partition",
        "schema_change",
        "db_connection",
        "data_quality",
        "spark_oom",
    }


def test_the_suite_includes_a_task_that_asserts_failure() -> None:
    """A suite that only tests success makes it easy to "fix" a system by making
    it never fail."""
    tasks, _ = load_suite("evals/tasks.yaml")
    assert any(task.expect.get("status") == "UNRESOLVED" for task in tasks)


# --------------------------------------------------------------------------- #
# loading: refusing to run a broken suite
# --------------------------------------------------------------------------- #
def test_a_missing_suite_raises_rather_than_falling_back(tmp_path: Path) -> None:
    with pytest.raises(SuiteConfigError) as excinfo:
        load_suite(tmp_path / "nope.yaml")
    assert "Refusing to substitute" in str(excinfo.value)


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, "tasks: [this is: not valid: yaml")
    with pytest.raises(SuiteConfigError):
        load_suite(path)


def test_a_suite_that_is_not_a_mapping_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(SuiteConfigError):
        load_suite(path)


def test_an_empty_suite_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, minimal({"tasks": []}))
    with pytest.raises(SuiteConfigError, match="no tasks"):
        load_suite(path)


def test_a_typo_in_an_assertion_key_raises(tmp_path: Path) -> None:
    """The whole point: a misspelled assertion must not silently never run."""
    path = write_suite(
        tmp_path,
        minimal({"tasks": [{**MINIMAL_TASK, "expect": {"categry": "SCHEMA_CHANGE"}}]}),
    )
    with pytest.raises(SuiteConfigError) as excinfo:
        load_suite(path)
    assert "categry" in str(excinfo.value)
    assert "silently never run" in str(excinfo.value)


def test_a_typo_in_defaults_raises(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path, minimal({"defaults": {"max_write_attemps": 0}, "tasks": [MINIMAL_TASK]})
    )
    with pytest.raises(SuiteConfigError):
        load_suite(path)


def test_a_duplicate_task_id_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, minimal({"tasks": [MINIMAL_TASK, MINIMAL_TASK]}))
    with pytest.raises(SuiteConfigError, match="duplicate"):
        load_suite(path)


def test_a_task_with_no_id_raises(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path, minimal({"tasks": [{"scenario": "missing_partition", "expect": {"status": "x"}}]})
    )
    with pytest.raises(SuiteConfigError, match="no id"):
        load_suite(path)


def test_a_task_with_no_incident_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, minimal({"tasks": [{"id": "t", "expect": {"status": "x"}}]}))
    with pytest.raises(SuiteConfigError, match="neither"):
        load_suite(path)


def test_an_unknown_scenario_raises(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path,
        minimal({"tasks": [{"id": "t", "scenario": "no_such", "expect": {"status": "x"}}]}),
    )
    with pytest.raises(SuiteConfigError, match="unknown scenario"):
        load_suite(path)


def test_an_unknown_task_key_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, minimal({"tasks": [{**MINIMAL_TASK, "aprove": True}]}))
    with pytest.raises(SuiteConfigError, match="unknown key"):
        load_suite(path)


def test_an_unknown_top_level_key_raises(tmp_path: Path) -> None:
    path = write_suite(tmp_path, minimal({"tasks": [MINIMAL_TASK], "taskz": []}))
    with pytest.raises(SuiteConfigError, match="top-level"):
        load_suite(path)


def test_min_checks_is_accepted_as_a_control_key(tmp_path: Path) -> None:
    """It configures the harness rather than asserting about a run."""
    path = write_suite(
        tmp_path,
        minimal({"defaults": {"min_checks": 2}, "tasks": [MINIMAL_TASK]}),
    )
    tasks, defaults = load_suite(path)
    assert defaults["min_checks"] == 2
    assert tasks[0].expected("min_checks") == 2


def test_incident_overrides_are_applied(tmp_path: Path) -> None:
    path = write_suite(
        tmp_path,
        minimal(
            {
                "tasks": [
                    {**MINIMAL_TASK, "incident_overrides": {"error": "stripped"}},
                ]
            }
        ),
    )
    tasks, _ = load_suite(path)
    assert tasks[0].incident["error"] == "stripped"
    assert tasks[0].incident["pipeline"] == "customer_claims_daily"  # from the scenario


# --------------------------------------------------------------------------- #
# evaluators: pure functions, tested without an agent
# --------------------------------------------------------------------------- #
def test_evaluator_details_carry_the_actual_numbers() -> None:
    """A red build that says "min_evidence failed" costs a re-run to diagnose."""
    obs = Observation(evidence_count=2)
    check = run_evaluator("min_evidence", obs, 5)
    assert check.ok is False
    assert check.detail == "2 >= 5"


def test_category_evaluator_shows_both_sides() -> None:
    check = run_evaluator("category", Observation(category="A"), "B")
    assert check.ok is False
    assert check.detail == "'A' == 'B'"


def test_required_tools_reports_what_is_missing() -> None:
    obs = Observation(tool_names=["get_pipeline_logs"])
    check = run_evaluator("required_tools", obs, ["get_pipeline_logs", "check_row_count"])
    assert check.ok is False
    assert "check_row_count" in check.detail


def test_forbidden_tools_reports_the_offender() -> None:
    obs = Observation(tool_names=["run_sql", "check_row_count"])
    check = run_evaluator("forbidden_tools", obs, ["run_sql"])
    assert check.ok is False
    assert "run_sql" in check.detail


def test_safety_evaluators_default_to_zero_tolerance() -> None:
    obs = Observation(write_attempts=1, unknown_tool_calls=2)
    assert run_evaluator("max_write_attempts", obs, 0).ok is False
    assert run_evaluator("max_unknown_tool_calls", obs, 0).ok is False


def test_token_budget_uses_the_total() -> None:
    obs = Observation(tokens_in=900, tokens_out=200)
    assert obs.total_tokens == 1100
    assert run_evaluator("max_tokens", obs, 1000).ok is False
    assert run_evaluator("max_tokens", obs, 2000).ok is True


def test_confidence_evaluators_bound_both_ends() -> None:
    obs = Observation(confidence=0.9)
    assert run_evaluator("min_confidence", obs, 0.5).ok is True
    assert run_evaluator("max_confidence", obs, 0.3).ok is False


def test_an_unknown_evaluator_name_is_a_programming_error() -> None:
    with pytest.raises(KeyError, match="unknown evaluator"):
        run_evaluator("nonsense", Observation(), 1)


# --------------------------------------------------------------------------- #
# the suite report
# --------------------------------------------------------------------------- #
def test_an_empty_suite_is_not_a_pass() -> None:
    """0/0 must be 0%, not 100% — a gate with no tasks is not a passing gate."""
    report = SuiteReport()
    assert report.accuracy == 0.0
    assert report.passed is False


def test_a_report_renders_per_check_detail() -> None:
    report = SuiteReport(
        results=[
            TaskResult(
                task_id="t1",
                description="d",
                passed=False,
                checks=[
                    run_evaluator("min_evidence", Observation(evidence_count=2), 5),
                ],
                status="UNRESOLVED",
            )
        ],
        model="stub",
    )
    rendered = report.render()
    assert "2 >= 5" in rendered
    assert "SUITE FAILED" in rendered
    assert "0/1 = 0%" in rendered


def test_a_report_is_json_serialisable() -> None:
    import json

    report = SuiteReport(
        results=[TaskResult(task_id="t", description="", passed=True, checks=[])],
        model="stub",
    )
    json.dumps(report.to_dict())


# --------------------------------------------------------------------------- #
# running the real suite
# --------------------------------------------------------------------------- #
def test_the_suite_passes_on_the_offline_analyst(seeded_estate, conn) -> None:
    """The CI gate itself: no API key, no network, every task green."""
    suite = EvaluationSuite(seeded_estate, connection=conn)
    try:
        report = suite.run()
    finally:
        suite.close()

    assert report.total >= 8
    failed = [r for r in report.results if not r.passed]
    detail = "\n".join(
        f"{r.task_id}: {[c.name for c in r.failed_checks]}" for r in failed
    )
    assert report.passed, f"evaluation regressed:\n{detail}\n{report.render()}"
    assert report.accuracy == 1.0


def test_a_subset_can_be_run_by_id(seeded_estate, conn) -> None:
    suite = EvaluationSuite(seeded_estate, connection=conn)
    try:
        report = suite.run(["spark_oom"])
    finally:
        suite.close()
    assert report.total == 1
    assert report.results[0].task_id == "spark_oom"


def test_an_unknown_task_id_is_an_error_not_an_empty_run(seeded_estate, conn) -> None:
    suite = EvaluationSuite(seeded_estate, connection=conn)
    try:
        with pytest.raises(SuiteConfigError, match="unknown task id"):
            suite.run(["no_such_task"])
    finally:
        suite.close()


def test_a_task_with_an_empty_expect_block_fails(tmp_path: Path, seeded_estate, conn) -> None:
    """`all([])` is True, so this must be rejected explicitly."""
    path = write_suite(
        tmp_path,
        minimal(
            {
                "defaults": {"min_checks": 1},
                "tasks": [
                    {"id": "asserts_nothing", "scenario": "missing_partition", "expect": {}}
                ],
            }
        ),
    )
    suite = EvaluationSuite(seeded_estate, tasks_path=path, connection=conn)
    try:
        report = suite.run()
    finally:
        suite.close()

    assert report.total == 1
    result = report.results[0]
    assert result.passed is False
    failing = {check.name for check in result.failed_checks}
    assert "has_expectations" in failing


def test_the_harness_resets_state_between_tasks(seeded_estate, conn) -> None:
    """Without this, a task could pass because the previous one left the right
    rows behind — the subtlest way for a suite to start lying."""
    from app.db import IncidentRepository

    suite = EvaluationSuite(seeded_estate, connection=conn)
    try:
        report = suite.run(["spark_oom", "missing_partition"])
    finally:
        suite.close()

    assert report.total == 2
    # Each task saw exactly its own incident, not two.
    assert IncidentRepository(conn).stats()["incidents"] == 1


def test_the_suite_records_cost_per_task(seeded_estate, conn) -> None:
    suite = EvaluationSuite(seeded_estate, connection=conn)
    try:
        report = suite.run(["missing_partition"])
    finally:
        suite.close()

    observation = report.results[0].observation
    assert observation is not None
    assert observation.llm_calls > 0
    assert observation.tool_calls > 0
    assert observation.total_tokens > 0
    assert observation.write_attempts == 0
    assert observation.unknown_tool_calls == 0
